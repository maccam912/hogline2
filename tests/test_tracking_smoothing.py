import copy

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from hogline2.tracking_calibration_fit import fit_geometry  # noqa: E402
from hogline2.tracking_smoothing import fit_glide, fused_measurement, rts_smooth, smooth_rows  # noqa: E402


def disappearing_stone_rows():
    rows = []
    for t in range(80):
        observed = t < 12 or 60 <= t < 65
        p = [0.08 * t, 35 - 1.5 * t] if t < 60 else [-1.0, 3 - 0.2 * (t - 60)]
        stone = dict(
            id="red_1", position=p, velocity=[0.08, -1.5],
            state="moving" if observed else ("occluded" if t < 18 else "lost"),
            estimate_kind="observed" if observed else "lost", visible=observed,
            position_covariance=[[0.04, 0], [0, 0.04]],
        )
        obs = dict(
            stone_id="red_1", xy=p, kind="overhead", view_id="house",
            quality=1, accepted=True, variance=[0.003, 0.003],
        )
        rows.append(dict(source_seconds=float(t), stones=[stone], observations=[obs] if observed else []))
    return rows


def test_missing_tail_cannot_pull_observed_path_to_opposite_side():
    rows = disappearing_stone_rows()[:60]
    short, _ = smooth_rows(rows[:12], {}, [])
    extended, summary = smooth_rows(rows, {}, [])
    assert len(summary["segments"]) == 1
    assert summary["segments"][0]["end_seconds"] == 11
    for i in range(12):
        assert extended[i]["stones"][0]["position"] == pytest.approx(short[i]["stones"][0]["position"])
    for i in range(12, 60):
        stone = extended[i]["stones"][0]
        assert not stone["smoothed"]
        assert stone["position"] == rows[i]["stones"][0]["position"]
        assert stone["estimate_kind"] == rows[i]["stones"][0]["estimate_kind"]


def test_reappearance_after_lost_gap_starts_independent_fit():
    rows = disappearing_stone_rows()
    out, summary = smooth_rows(rows, {}, [])
    assert [(s["start_seconds"], s["end_seconds"]) for s in summary["segments"]] == [(0, 11), (60, 64)]
    assert out[60]["stones"][0]["position"] == pytest.approx([-1, 3], abs=0.1)
    assert all(not out[i]["stones"][0]["smoothed"] for i in range(12, 60))


def test_rejected_measurement_cannot_anchor_smoothing():
    obs = disappearing_stone_rows()[0]["observations"][0]
    assert fused_measurement([{**obs, "accepted": False}])[0] is None
    assert fused_measurement([{**obs, "rejection": "unexplained_displacement"}])[0] is None


def test_converged_but_inaccurate_glide_falls_back_for_review():
    times = np.arange(20, dtype=float)
    points = np.column_stack((np.zeros(20), np.r_[np.linspace(30, 28, 10), np.linspace(27, 0, 10)]))
    seed = np.column_stack((points, np.zeros((20, 2))))
    fit = fit_glide(times, list(points), [np.array([0.01, 0.01])] * 20, seed)
    assert not fit["success"]
    assert fit["message"] == "Free-glide curve disagrees with observations"
    assert fit["positions"] == pytest.approx(points)


def test_future_observations_reduce_earlier_gap_error_and_covariance():
    times = np.arange(8, dtype=float)
    z = [
        np.array([0.0, 0.0]),
        np.array([0.0, 1.5]),
        None,
        None,
        None,
        None,
        np.array([0.0, 6.0]),
        np.array([0.0, 7.0]),
    ]
    variance = [np.array([0.01, 0.01]) if i != 1 else np.array([0.2, 0.2]) for i in range(8)]
    smoothed, cov, forward, forward_cov = rts_smooth(times, z, variance, acceleration=0.05)
    assert abs(smoothed[4, 1] - 4) < abs(forward[4, 1] - 4)
    assert cov[4, 1, 1] < forward_cov[4, 1, 1]


def test_camera_handoff_bias_cannot_create_speedup_or_sharp_curve():
    t = np.linspace(0, 20, 81)
    y = 30 - 3 * t + 0.075 * t * t
    x = 0.8 * (t / 20) ** 2
    truth = np.column_stack((x, y))
    raw = truth.copy()
    raw[30:45, 0] += 0.4
    raw[30:45, 1] += 1.0
    variance = [np.array([0.3**2, 0.6**2]) for _ in t]
    z = [p for p in raw]
    seed, _, _, _ = rts_smooth(t, z, variance)
    fit = fit_glide(t, z, variance, seed, start_anchor=truth[0], end_anchor=truth[-1], stopping=True)
    assert fit["success"]
    assert np.max(np.diff(fit["speeds"])) < 1e-7
    assert fit["max_curvature_per_m"] <= 0.25001
    assert fit["positions"][0] == pytest.approx(truth[0])
    assert fit["positions"][-1] == pytest.approx(truth[-1])
    headings = np.unwrap(np.arctan2(fit["velocities"][:-1, 0], fit["velocities"][:-1, 1]))
    assert np.max(np.abs(np.diff(headings))) < 0.05


def test_later_overhead_resting_positions_anchor_the_house_and_contact_does_not_move_past_rest():
    rows = []
    for t in range(20):
        if t < 8:
            p = [0.0, 9 - t]
            state = "moving"
        elif t < 13:
            p = [0.0, 1.0]
            state = "stationary"
        elif t < 17:
            p = [0.3 * (t - 12), 1.0]
            state = "moving"
        else:
            p = [1.2, 1.0]
            state = "stationary"
        stone = dict(id="red_1", position=p, state=state, visible=True, velocity=[0, 0])
        observation = dict(
            stone_id="red_1",
            xy=p,
            kind="overhead",
            view_id="house",
            quality=2,
            accepted=True,
            variance=[0.003, 0.003],
        )
        rows.append(dict(source_seconds=float(t), stones=[stone], observations=[observation]))
    original = copy.deepcopy(rows)
    out, summary = smooth_rows(
        rows, {}, [dict(type="possible_contact", stones=["red_1", "yellow_1"], seconds=12.0)]
    )
    assert rows == original
    assert summary["house_anchors"]
    for i in [9, 10, 11, 12]:
        assert out[i]["stones"][0]["position"] == pytest.approx([0, 1])
    assert out[-1]["stones"][0]["position"] == pytest.approx([1.2, 1])
    assert all(s["success"] for s in summary["segments"])
    assert all(s["max_speed_increase_mps"] < 1e-6 for s in summary["segments"])


def test_geometry_cannot_claim_measured_length_without_evidence():
    report = fit_geometry([], {"views": []})
    assert report["status"] == "insufficient_evidence"
    assert report["tee_distance_m"] == 34.747


def test_fixed_venue_length_cannot_be_shortened_to_explain_camera_disagreement():
    from hogline2.tracking_geometry import prepare_view, project

    landmarks = dict(button=[250, 250], left=[50, 250], right=[450, 250], back=[250, 50], front=[250, 450])
    views = [
        dict(id="a", kind="perspective", end="A", crop=[0, 0, 500, 500], landmarks=landmarks),
        dict(id="b", kind="perspective", end="B", crop=[0, 0, 500, 500], landmarks=landmarks),
    ]
    prepared = [prepare_view(v) for v in views]
    rows = []
    for i in range(15):
        observations = []
        for j, v in enumerate(prepared):
            point = [200 + i * 5, 200 + i * 2 + j * 10]
            observations.append(
                dict(
                    stone_id=f"red_{i % 3 + 1}",
                    quality=1,
                    view_id=v["id"],
                    point=point,
                    xy=project(v["image_to_ice"], point),
                )
            )
        rows.append(dict(observations=observations))
    report = fit_geometry(
        rows,
        dict(
            views=views,
            tee_distance_m=34.747,
            tee_distance_fixed=True,
            tee_distance_source="standard_inferred",
        ),
    )
    assert report["length_mode"] == "fixed_baseline"
    assert report["tee_distance_m"] == 34.747
    assert report["length_sensitivity_m"] is None
    assert all(r["tee_distance_m"] == 34.747 for r in report["leave_one_track_out"])
    assert report["status"] == "needs_review"
    assert len(report["leave_one_track_out"]) == 3
    assert report["held_out_pair_rms_m"] > 0.30
    assert all(r["held_out_pairs"] > 0 and r["held_out_rms_m"] > 0 for r in report["leave_one_track_out"])
    assert not report["correction_recommended"]
    rejected = copy.deepcopy(rows)
    for row in rejected:
        for observation in row["observations"]:
            observation["accepted"] = False
    assert fit_geometry(rejected, {"views": views})["status"] == "insufficient_evidence"

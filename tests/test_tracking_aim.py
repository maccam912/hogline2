import copy
import json

import pytest
from fastapi.testclient import TestClient

from hogline2.app import create_app
from hogline2.config import Config
from hogline2.store import Store
from hogline2.tracking_aim import build_report, compare_target, fit_heading, pair_headings, ray_band, ray_x
from hogline2.tracking_dimensions import sheet_geometry
from hogline2.tracking_geometry import prepare_view, project


def camera(name, end, kind):
    return dict(
        id=name,
        end=end,
        kind=kind,
        crop=[0, 0, 1000, 1000],
        landmarks=dict(
            button=[500, 500],
            left=[317.1, 500],
            right=[682.9, 500],
            back=[500, 317.1],
            front=[500, 682.9],
            hack=[500, 134.2],
        ),
    )


def evidence(end="A"):
    cal = dict(
        views=[camera("delivery", end, "perspective"), camera("far", "B" if end == "A" else "A", "overhead")],
        tee_distance_m=34.747,
        revision=5,
    )
    manifest = dict(
        calibration=cal,
        delivery_view="delivery",
        stones=[dict(id="red_1")],
        source_sha256="source",
        source_start=0,
        duration=30,
    )
    v = prepare_view(cal["views"][0], 34.747, sheet_geometry(cal))
    direction = 1 if end == "A" else -1
    hog = 6.401 if end == "A" else 34.747 - 6.401
    samples = []
    for i in range(13):
        d = direction * (i - 6) / 4
        xy = [0.2 + 0.02 * d + 0.008 * d * d, hog + d]
        samples.append(dict(seconds=10 + i / 4, xy=xy, frame=i, point=project(v["ice_to_image"], xy)))
    rows = [
        dict(
            source_seconds=s["seconds"],
            frame=s["frame"],
            stones=[],
            observations=[
                dict(
                    xy=s["xy"],
                    point=s["point"],
                    stone_id="red_1",
                    view_id="delivery",
                    accepted=True,
                    measurement_kind="visible_granite_bottom",
                    ground_uncertain=False,
                )
            ],
        )
        for s in samples
    ]
    return manifest, v, hog, direction, samples, rows


@pytest.mark.parametrize("end", ["A", "B"])
def test_local_tangent_and_target_sign_both_directions(end):
    m, v, hog, direction, samples, rows = evidence(end)
    ray = fit_heading(samples, hog, direction, v)
    assert ray["slope_dx_dy"] == pytest.approx(0.02)
    assert ray["origin"] == pytest.approx([0.2, hog])
    assert ray["crossing_seconds"] == pytest.approx(11.5)
    assert ray_band(ray, hog + 25 * direction) > ray_band(ray, hog)
    report = build_report(m, rows, [], "abc")
    far = next(v for v in report["views"] if v["id"] == "far")
    y = 34.747 if direction == 1 else 0
    # World +x is left when looking from A, right when looking from B.
    target = [ray_x(ray, y) + direction * 0.25, y]
    comparison = compare_target(ray, far, project(far["ice_to_image"], target))
    assert comparison["offset_right_m"] == pytest.approx(0.25)


def test_future_smoothed_positions_and_other_camera_cannot_change_heading():
    m, _, _, _, _, rows = evidence()
    first = build_report(m, rows, [], "abc")["rays"][0]
    for r in rows:
        r["stones"] = [dict(id="red_1", velocity=[900, 20], position=[100, 200])]
        r["observations"].append({**r["observations"][0], "view_id": "far", "xy": [999, 999]})
    rows.append(
        dict(
            frame=100,
            source_seconds=100,
            observations=[
                {
                    **rows[0]["observations"][0],
                    "xy": [-100, 30],
                    "point": [20, 20],
                }
            ],
        )
    )
    second = build_report(m, rows, [], "abc")["rays"][0]
    assert second["origin"] == first["origin"]
    assert second["slope_dx_dy"] == first["slope_dx_dy"]
    assert second["samples"] == first["samples"]


def test_missing_rejected_and_contact_observations_withhold_fixed_ray():
    m, v, hog, direction, samples, rows = evidence()
    with pytest.raises(ValueError, match="five local observations"):
        fit_heading(samples[4:8], hog, direction, v)
    with pytest.raises(ValueError, match="gap"):
        fit_heading(samples[:4] + samples[8:], hog, direction, v)
    for r in rows[3:10]:
        r["observations"][0]["accepted"] = False
    assert build_report(m, rows, [], "abc")["rays"][0]["status"] == "unavailable"
    _, _, _, _, _, rows = evidence()
    contacts = [dict(type="possible_contact", seconds=11.25, stones=["red_1", "yellow_1"])]
    assert build_report(m, rows, contacts, "abc")["rays"][0]["status"] == "unavailable"


def test_pair_rays_keep_jitter_but_do_not_bridge_missing_frames():
    _, _, _, direction, samples, _ = evidence()
    samples[6]["xy"][0] += 0.1
    pairs = pair_headings(samples, direction)
    assert len(pairs) == 12
    assert abs(pairs[5]["slope_dx_dy"] - pairs[6]["slope_dx_dy"]) > 0.7
    pairs = pair_headings(samples[:6] + samples[7:], direction)
    assert not any(p["previous_seconds"] == samples[5]["seconds"] for p in pairs)
    samples[6]["xy"][1] = samples[5]["xy"][1] - 0.1
    assert not any(p["frame"] == 6 for p in pair_headings(samples, direction))


def test_api_broom_persistence_staleness_bounds_and_conflicts(tmp_path):
    m, _, _, _, _, rows = evidence()
    Store(tmp_path).create("demo", source_sha256="source", media=dict(duration=100))
    folder = tmp_path / "recordings/demo/tracking" / ("a" * 32)
    folder.mkdir(parents=True)
    for name, data in [("manifest.json", m), ("events.json", []), ("status.json", {"state": "complete"})]:
        (folder / name).write_text(json.dumps(data))
    raw = folder / "observations.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in rows))
    client = TestClient(create_app(Config(tmp_path), start_worker=False))
    url = "/api/recordings/demo/tracking/runs/" + "a" * 32 + "/aim"
    report = client.get(url).json()
    ray = report["rays"][0]
    assert ray["status"] == "estimated"
    body = dict(
        evidence_sha256=ray["evidence_sha256"],
        revision=0,
        target=dict(view_id="far", source_seconds=11, point=[500, 500]),
    )
    saved = client.put(url + "/red_1", json=body)
    assert saved.status_code == 200, saved.text
    assert saved.json()["revision"] == 1
    assert client.get(url).json()["rays"][0]["review"]["target"] == body["target"]
    assert client.put(url + "/red_1", json=body).status_code == 409
    body["revision"] = 1
    bad = copy.deepcopy(body)
    bad["target"]["view_id"] = "delivery"
    assert client.put(url + "/red_1", json=bad).status_code == 400
    bad["target"] = dict(view_id="far", source_seconds=11, point=[-1, 500])
    assert client.put(url + "/red_1", json=bad).status_code == 400
    bad["target"] = dict(view_id="far", source_seconds=80, point=[500, 500])
    assert client.put(url + "/red_1", json=bad).status_code == 400
    bad["target"] = dict(view_id="far", source_seconds=11, point=["NaN", 500])
    assert client.put(url + "/red_1", json=bad).status_code == 422
    assert client.put(url + "/unknown", json=body).status_code == 404
    # A changed run calibration invalidates saved evidence even if the stone ID matches.
    m["calibration"]["revision"] += 1
    (folder / "manifest.json").write_text(json.dumps(m))
    changed = client.get(url).json()["rays"][0]
    assert changed["review"] is None and changed["review_stale"]
    assert client.put(url + "/red_1", json=body).status_code == 409
    body.update(evidence_sha256=changed["evidence_sha256"], target=None)
    assert client.put(url + "/red_1", json=body).json()["revision"] == 2
    # Loading a fresh app reads the cleared review from disk.
    fresh = TestClient(create_app(Config(tmp_path), start_worker=False))
    assert fresh.get(url).json()["rays"][0]["review"]["target"] is None
    (folder / "status.json").write_text('{"state":"running"}')
    assert client.get(url).status_code == 409


def test_pair_only_review_and_cached_source_replacement(tmp_path):
    m, _, _, _, _, rows = evidence()
    # Enough consecutive positions to inspect pairs, but no hog-line coverage.
    rows = rows[:4]
    store = Store(tmp_path)
    store.create("demo", source_sha256="source", media=dict(duration=100))
    folder = tmp_path / "recordings/demo/tracking" / ("b" * 32)
    folder.mkdir(parents=True)
    for name, data in [("manifest.json", m), ("events.json", []), ("status.json", {"state": "complete"})]:
        (folder / name).write_text(json.dumps(data))
    (folder / "observations.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    client = TestClient(create_app(Config(tmp_path), start_worker=False))
    url = "/api/recordings/demo/tracking/runs/" + "b" * 32 + "/aim"
    ray = client.get(url).json()["rays"][0]
    assert ray["status"] == "unavailable" and len(ray["pair_rays"]) == 3
    response = client.put(
        url + "/red_1",
        json=dict(
            evidence_sha256=ray["evidence_sha256"],
            revision=0,
            target=dict(view_id="far", source_seconds=10, point=[500, 500]),
        ),
    )
    assert response.status_code == 200
    assert "comparison" not in response.json()  # No invented fixed-ray measurement.
    store.update("demo", source_sha256="replaced")
    assert client.get(url).status_code == 409


def test_noisy_heading_is_withheld():
    _, v, hog, direction, samples, _ = evidence()
    for i, sample in enumerate(samples):
        sample["xy"][0] += 0.4 if i % 2 else -0.4
    with pytest.raises(ValueError, match="inconsistent"):
        fit_heading(samples, hog, direction, v)

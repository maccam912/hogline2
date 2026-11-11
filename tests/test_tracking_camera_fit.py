import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from hogline2.tracking_calibration_fit import fit_geometry  # noqa: E402
from hogline2.tracking_geometry import prepare_view, project, world_points  # noqa: E402


def test_independent_camera_zoom_roll_and_perspective_recover_shared_sheet():
    length = 34.747
    # Each perspective has its own scale, roll, skew and projective denominator.
    cameras = {
        "a": np.array([[123.0, 13, 360], [5, 20, 500], [0.002, 0.07, 1]]),
        "b": np.array([[91.0, 22, 350], [-4, 35, 520], [-0.003, 0.09, 1]])
        @ np.array([[-1, 0, 0], [0, -1, length], [0, 0, 1]]),
        "overhead": np.array([[100.0, 0, 250], [0, 100, 250], [0, 0, 1]]),
    }
    views = []
    for name, camera in cameras.items():
        end = "B" if name == "b" else "A"
        landmarks = {k: project(camera.ravel(), p) for k, p in world_points(end).items()}
        if name != "overhead":
            # A two-pixel ring-marking error can induce a large distant scale error.
            landmarks["front"][1] += 2 if name == "a" else -2
            landmarks["back"][1] -= 2 if name == "a" else -2
        views.append(
            dict(
                id=name,
                kind="overhead" if name == "overhead" else "perspective",
                end=end,
                crop=[0, 0, 800, 1000],
                landmarks=landmarks,
            )
        )
    prepared = {v["id"]: prepare_view(v) for v in views}
    rows = []
    truth = []
    for y in np.linspace(30, 0, 45):
        observations = []
        for stone, x in enumerate([-0.8, 0.1, 0.7]):
            xy = [x + 0.015 * y, y]
            for name, camera in cameras.items():
                if name == "overhead" and y > 2:
                    continue
                point = project(camera.ravel(), xy)
                observations.append(
                    dict(
                        stone_id=f"red_{stone + 1}",
                        view_id=name,
                        point=point,
                        xy=project(prepared[name]["image_to_ice"], point),
                        quality=0.9,
                        accepted=True,
                        measurement_kind="visible_granite_bottom",
                    )
                )
                truth.append((name, point, xy))
        rows.append(dict(observations=observations))
    report = fit_geometry(rows, dict(views=views, tee_distance_m=length, tee_distance_fixed=True))
    assert report["status"] == "camera_consistent"
    assert report["held_out_pair_rms_m"] < 0.10
    for name in ["a", "b"]:
        samples = [(point, xy) for view, point, xy in truth if view == name]
        error = np.mean(
            [np.linalg.norm(np.array(project(report["image_to_ice"][name], p)) - xy) for p, xy in samples]
        )
        before = np.mean(
            [np.linalg.norm(np.array(project(prepared[name]["image_to_ice"], p)) - xy) for p, xy in samples]
        )
        assert error < 0.20
        assert error < before * 0.3

import copy

import pytest
from fastapi.testclient import TestClient
from test_tracking import view

from hogline2.app import create_app
from hogline2.config import Config
from hogline2.store import Store
from hogline2.tracking_geometry import camera_reference, prepare_view, project, validate_calibration


def calibrated_view():
    v = view()
    # Introduce a scale/shear/perspective change about the fixed button.
    h = prepare_view(v)["image_to_ice"]
    h[1] = 0.0002
    h[2] -= 0.0002 * 250
    h[6] = 0.0001
    v["camera_calibration"] = dict(
        image_to_ice=h, reference_sha256=camera_reference(v), source_run="a" * 32, held_out_rms_m=0.09
    )
    return v


def test_saved_independent_transform_is_used_for_forward_and_inverse_projection():
    v = calibrated_view()
    prepared = prepare_view(v)
    assert prepared["image_to_ice"] == v["camera_calibration"]["image_to_ice"]
    legacy = copy.deepcopy(v)
    legacy.pop("camera_calibration")
    point = [110, 390]
    assert project(prepared["image_to_ice"], point) != pytest.approx(
        project(prepare_view(legacy)["image_to_ice"], point)
    )
    assert project(prepared["ice_to_image"], project(prepared["image_to_ice"], point)) == pytest.approx(point)


def test_stale_camera_fit_is_invalidated_when_landmarks_or_dimensions_change():
    record = dict(source_sha256="abc", media=dict(width=500, height=500))
    for mutation in ["landmark", "length", "crop"]:
        v = calibrated_view()
        cal = dict(source_sha256="abc", views=[v])
        if mutation == "landmark":
            v["landmarks"]["left"][0] += 1
        elif mutation == "length":
            cal["tee_distance_m"] = 35
        else:
            v["crop"][0] = 1
        validate_calibration(cal, record)
        assert "camera_calibration" not in v


def test_camera_transform_survives_api_roundtrip(tmp_path):
    store = Store(tmp_path)
    store.create("demo", source_sha256="abc", media=dict(width=500, height=500), ends=[])
    client = TestClient(create_app(Config(tmp_path), start_worker=False))
    v = calibrated_view()
    response = client.put(
        "/api/recordings/demo/tracking/calibration",
        json=dict(source_sha256="abc", source_seconds=0, views=[v], revision=0),
    )
    assert response.status_code == 200, response.text
    saved = response.json()["views"][0]
    assert saved["camera_calibration"]["image_to_ice"] == v["camera_calibration"]["image_to_ice"]
    assert prepare_view(saved)["image_to_ice"] == v["camera_calibration"]["image_to_ice"]


def test_singular_camera_transform_is_rejected():
    v = calibrated_view()
    v["camera_calibration"]["image_to_ice"] = [0] * 9
    with pytest.raises(ValueError, match="degenerate"):
        prepare_view(v)

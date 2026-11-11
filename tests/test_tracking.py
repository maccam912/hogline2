import copy
import math

import pytest
from fastapi.testclient import TestClient

from hogline2.app import create_app
from hogline2.config import Config
from hogline2.store import Store
from hogline2.tracking_geometry import (
    homography,
    prepare_view,
    project,
    validate_calibration,
)
from hogline2.tracking_state import EndState


def view():
    return dict(
        id="house_a",
        kind="overhead",
        end="A",
        crop=[0, 0, 500, 500],
        landmarks={
            "button": [250, 250],
            "left": [50, 250],
            "right": [450, 250],
            "back": [250, 50],
            "front": [250, 450],
        },
    )


def observation(name, xy, kind="overhead"):
    return dict(stone_id=name, xy=xy, kind=kind, quality=0.8)


def test_shared_plane_roundtrip_and_opposite_house():
    a = prepare_view(view())
    bview = view()
    bview["end"] = "B"
    b = prepare_view(bview)
    assert project(a["image_to_ice"], [250, 250]) == pytest.approx([0, 0])
    assert project(b["image_to_ice"], [250, 250]) == pytest.approx([0, 34.747])
    assert project(b["image_to_ice"], [450, 250]) == pytest.approx([-1.829, 34.747])
    for pixel in [[100, 200], [350, 400]]:
        assert project(a["ice_to_image"], project(a["image_to_ice"], pixel)) == pytest.approx(pixel)
    with pytest.raises(ValueError, match="degenerate"):
        homography([([0, 0], [0, 0])] * 4)


def test_calibration_rejects_wrong_source_and_button():
    record = dict(source_sha256="abc", media=dict(width=500, height=500))
    cal = dict(source_sha256="abc", views=[view()])
    assert validate_calibration(cal, record) is cal
    bad = copy.deepcopy(cal)
    bad["views"][0]["landmarks"]["button"] = [400, 400]
    with pytest.raises(ValueError, match="button disagrees"):
        validate_calibration(bad, record)
    with pytest.raises(ValueError, match="different source"):
        validate_calibration({**cal, "source_sha256": "other"}, record)


def test_ids_per_color_per_end_and_no_ninth():
    a, b = EndState(1), EndState(2)
    assert a.birth("red", 0, [0, 0], "v").id == "red_1"
    assert a.birth("yellow", 1, [0, 0], "v").id == "yellow_1"
    for i in range(2, 9):
        assert a.birth("red", i, [0, 0], "v").id == f"red_{i}"
    with pytest.raises(ValueError, match="eight"):
        a.birth("red", 10, [0, 0], "v")
    assert b.birth("red", 10, [0, 0], "v").id == "red_1"


def test_stationary_identity_survives_long_occlusion_and_rejects_teleport():
    state = EndState(1)
    stone = state.birth("red", 0, [0, 0], "a")
    for t in [0, 1, 2, 3]:
        state.update(t, [observation(stone.id, [0, 0])])
    assert stone.state == "stationary"
    state.update(100, [])
    assert stone.state == "stationary" and not stone.visible and stone.position == [0, 0]
    state.update(101, [observation(stone.id, [1, 1])])
    assert not stone.visible and stone.position == [0, 0]
    state.update(102, [observation(stone.id, [0.01, 0])])
    assert stone.visible and stone.id == "red_1"


def test_swept_contact_allows_bumped_stone_to_move_without_new_id():
    state = EndState(1)
    resting = state.birth("red", 0, [0, 0], "a")
    for t in range(4):
        state.update(t, [observation(resting.id, [0, 0])])
    delivered = state.birth("yellow", 4, [-1, 0], "a")
    state.update(4.5, [observation(delivered.id, [0.1, 0]), observation(resting.id, [0.35, 0])])
    assert resting.visible and resting.state == "moving" and resting.id == "red_1"
    assert len(state.stones) == 2
    assert any(e["type"] == "possible_contact" for e in state.events)


def test_same_color_handoff_ambiguity_is_not_forced():
    state = EndState(1)
    state.birth("red", 0, [0, 0], "side")
    state.birth("red", 1, [0.25, 0], "side")
    assert state.match("red", [0.12, 0], 2, "overhead") is None
    assert state.match("yellow", [0, 0], 2, "overhead") is None
    assert state.match("red", [-0.2, 0], 2, "overhead").id == "red_1"


def test_overhead_preferred_and_stationary_contact_is_not_required_for_occlusion():
    state = EndState(1)
    stone = state.birth("yellow", 0, [0, 0], "side")
    state.update(
        1,
        [
            observation(stone.id, [0.1, 0], "overhead"),
            observation(stone.id, [2, 1], "perspective"),
        ],
    )
    assert math.dist(stone.position, [0.1, 0]) < 0.001
    assert any(e["type"] == "view_disagreement" for e in state.events)


def test_calibration_api_revision_and_no_gpu_on_web_import(tmp_path):
    store = Store(tmp_path)
    store.create("demo", source_sha256="abc", media=dict(width=500, height=500), ends=[])
    client = TestClient(create_app(Config(tmp_path), start_worker=False))
    assert client.get("/tracking").status_code == 200
    cal = dict(
        source_sha256="abc",
        source_seconds=0,
        first_delivery_end="A",
        revision=0,
        views=[view()],
    )
    response = client.put("/api/recordings/demo/tracking/calibration", json=cal)
    assert response.status_code == 200 and response.json()["revision"] == 1
    assert client.put("/api/recordings/demo/tracking/calibration", json=cal).status_code == 409
    assert client.get("/api/recordings/demo/tracking/runs").json() == []
    assert client.get("/api/recordings/demo/tracking/runs/not-a-run/manifest.json").status_code == 404


def test_hidden_contact_does_not_claim_a_stone_is_still_stationary():
    state = EndState(1)
    hidden = state.birth("red", 0, [0, 0], "a")
    for t in range(4):
        state.update(t, [observation(hidden.id, [0, 0])])
    moving = state.birth("yellow", 4, [-1, 0], "a")
    state.update(4.5, [observation(moving.id, [0.1, 0])])
    assert not hidden.visible and hidden.state == "contact_occluded"
    assert hidden.position == [0, 0]


def test_stationary_mask_jitter_does_not_reset_identity_or_motion_state():
    state = EndState(1)
    stone = state.birth("red", 0, [0, 0], "a")
    for t in range(4):
        state.update(t, [observation(stone.id, [0, 0])])
    state.update(4, [observation(stone.id, [0.08, 0])])
    state.update(5, [])
    assert stone.state == "stationary" and stone.position == [0, 0]


def test_rotated_overhead_agrees_with_same_physical_house():
    v = view()
    v["rotation_degrees"] = 180
    calibrated = prepare_view(v)
    assert project(calibrated["image_to_ice"], [50, 250]) == pytest.approx([1.829, 0])
    assert project(calibrated["image_to_ice"], [250, 50]) == pytest.approx([0, 1.829])


def test_long_lost_moving_track_is_not_reassigned_to_an_unrelated_stone():
    state = EndState(1)
    stone = state.birth("red", 0, [0, 0], "a")
    stone.state = "occluded"
    assert state.match("red", [0, 0], 30, "b") is None


def test_departed_stone_cannot_reassociate_with_waiting_stone_at_hack():
    state = EndState(1)
    stone = state.birth("yellow", 428.667, [-0.2, 36.5], "perspective_b")
    for i in range(1, 17):
        t = 428.667 + i / 3
        state.update(t, [observation(stone.id, [-0.2 - i * 0.026, 36.5 - i * 0.87])])
    state.update(434.1, [])
    assert state.match("yellow", [0.96, 36.27], 437.0, "overhead_b", radius=0.85) is None
    assert state.match("yellow", [-0.85, 14.75], 437.0, "perspective_a", radius=0.85) is stone


def test_venue_button_spacing_is_used_in_projection_and_saved(tmp_path):
    v = view()
    v["end"] = "B"
    assert project(prepare_view(v, 36.1)["image_to_ice"], [250, 250]) == pytest.approx([0, 36.1])
    store = Store(tmp_path)
    store.create("demo", source_sha256="abc", media=dict(width=500, height=500), ends=[])
    client = TestClient(create_app(Config(tmp_path), start_worker=False))
    calibration = dict(
        source_sha256="abc",
        source_seconds=0,
        views=[v],
        revision=0,
        venue_id="test_club",
        tee_distance_m=36.1,
        tee_distance_source="measured",
    )
    assert client.put("/api/recordings/demo/tracking/calibration", json=calibration).status_code == 200
    import json

    profile = json.loads((tmp_path / "venues/test_club.json").read_text())
    assert profile["tee_distance_m"] == 36.1 and profile["tee_distance_source"] == "measured"


def test_standard_venue_profile_preserves_geometry_and_does_not_replace_camera_views(tmp_path):
    from hogline2.tracking_geometry import STANDARD_SHEET, world_points

    store = Store(tmp_path)
    store.create("demo", source_sha256="abc", media=dict(width=500, height=500), ends=[])
    client = TestClient(create_app(Config(tmp_path), start_worker=False))
    calibration = dict(
        source_sha256="abc",
        source_seconds=0,
        views=[view()],
        revision=0,
        venue_id="wausau",
        tee_distance_source="standard_inferred",
        tee_distance_fixed=True,
        sheet_geometry=STANDARD_SHEET,
        geometry_evidence="User research supports standard geometry; not surveyed",
    )
    response = client.put("/api/recordings/demo/tracking/calibration", json=calibration)
    assert response.status_code == 200
    profile = client.get("/api/tracking/venues/wausau").json()
    assert profile["sheet_geometry"] == STANDARD_SHEET
    assert profile["tee_distance_fixed"] is True
    assert profile["tee_distance_source"] == "standard_inferred"
    assert "views" not in profile and "revision" not in profile
    assert world_points("A", geometry=profile["sheet_geometry"])["hack"] == [0, -3.658]
    assert client.get("/api/tracking/venues/unknown").status_code == 404


def test_sheet_geometry_rejects_inconsistent_dimensions():
    record = dict(source_sha256="abc", media=dict(width=500, height=500))
    cal = dict(source_sha256="abc", views=[view()], sheet_geometry={"length_m": 36})
    with pytest.raises(ValueError, match="Hacks must fit"):
        validate_calibration(cal, record)
    cal["sheet_geometry"] = {"width_m": 3}
    with pytest.raises(ValueError, match="house must fit"):
        validate_calibration(cal, record)

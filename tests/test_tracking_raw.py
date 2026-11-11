import pytest

from hogline2.tracking_raw import add_raw_velocity


def measurement(xy=(0, 0), point=(10, 10), view="a", name="red_1"):
    return dict(xy=xy, point=point, view_id=view, stone_id=name)


def test_raw_velocity_preserves_large_jumps_and_does_not_smooth():
    first = add_raw_velocity(None, measurement(), 0, 100)
    second = add_raw_velocity(first, measurement((3, 4), (13, 14)), 1, 100.5)
    assert second["velocity"] == [6, 8]
    assert second["speed_mps"] == 10
    assert second["pixel_speed"] == 10
    third = add_raw_velocity(second, measurement((3, 4), (13, 14)), 2, 101)
    assert third["speed_mps"] == 0


@pytest.mark.parametrize(
    "frame,view,name,seconds",
    [
        (2, "a", "red_1", 101),
        (1, "b", "red_1", 100.5),
        (1, "a", "yellow_1", 100.5),
        (1, "a", "red_1", 100),
    ],
)
def test_raw_velocity_never_bridges_gap_camera_identity_or_zero_time(frame, view, name, seconds):
    first = add_raw_velocity(None, measurement(), 0, 100)
    second = add_raw_velocity(first, measurement((3, 4), view=view, name=name), frame, seconds)
    assert second["velocity"] is None
    assert second["speed_mps"] is None


def test_body_prompt_requires_granite_and_raw_measurement_does_not_infer_it():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from hogline2.tracking_raw import raw_mask_measurement
    from hogline2.tracking_reinitialize import body_prompt

    rgb = np.full((60, 60, 3), 255, np.uint8)
    rgb[20:25, 20:40] = [255, 220, 0]
    mask = np.zeros((60, 60), bool)
    mask[20:25, 20:40] = True
    assert body_prompt(rgb, mask, "yellow") is None
    rgb[25:35, 18:42] = 120
    hint = body_prompt(rgb, mask, "yellow")
    assert hint["outside_mask"] and hint["body"][1] >= 25
    view = dict(kind="perspective", crop=[0, 0, 60, 60], image_to_ice=[1, 0, 0, 0, 1, 0, 0, 0, 1])
    raw = raw_mask_measurement(mask, view)
    assert raw["point"][1] == 24  # The raw metric retains the bad mask's bottom edge.

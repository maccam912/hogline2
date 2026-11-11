import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from hogline2.tracking_proposals import DeliveryGate, candidates, mask_observation  # noqa: E402


def test_long_setup_at_hack_does_not_discard_delivery():
    gate = DeliveryGate()
    view = dict(kind="perspective", landmarks={"hack": [100, 100]})
    for t in range(31):
        assert not gate.update(t, view, [dict(color="red", point=[100, 100], xy=[0, -3])])
    results = []
    for t, y in [(31, 105), (32, 110), (33, 116)]:
        results.extend(gate.update(t, view, [dict(color="red", point=[100, y], xy=[0, -2])]))
    assert len(results) == 1 and results[0]["color"] == "red"


def synthetic_stone():
    background = np.full((300, 300, 3), 240, np.uint8)
    rgb = background.copy()
    cv2.rectangle(rgb, (139, 151), (161, 164), (120, 120, 120), -1)
    cv2.ellipse(rgb, (150, 149), (9, 2), 0, 0, 360, (240, 220, 20), -1)
    mask = np.zeros((300, 300), np.uint8)
    cv2.ellipse(mask, (150, 149), (9, 2), 0, 0, 360, 1, -1)
    view = dict(
        id="a",
        kind="perspective",
        end="A",
        crop=[0, 0, 300, 300],
        image_to_ice=[0.01, 0, -1.5, 0, 0.01, -1.5, 0, 0, 1],
    )
    return rgb, background, mask.astype(bool), view


def test_static_colored_ice_is_not_a_stone_proposal():
    rgb, background, _, view = synthetic_stone()
    assert not candidates(rgb, view, rgb.copy())
    assert any(p["color"] == "yellow" for p in candidates(rgb, view, background))


def test_handle_only_mask_uses_granite_for_ground_contact_without_expanding_mask():
    rgb, background, mask, view = synthetic_stone()
    before = mask.copy()
    obs = mask_observation(mask, 0.6, rgb, view, "yellow", background)
    assert obs is not None and obs["point"][1] >= 160
    assert np.array_equal(mask, before)


def test_distant_handle_does_not_use_nearby_person_as_granite_foot():
    rgb, background, mask, view = synthetic_stone()
    view["image_to_ice"][5] += 8
    obs = mask_observation(mask, 0.6, rgb, view, "yellow", background)
    assert obs is not None and obs["point"][1] == np.where(mask)[0].max()


def test_perspective_mask_and_proposal_continue_beyond_thirteen_metres():
    rgb, background, mask, view = synthetic_stone()
    view["image_to_ice"][5] += 22
    result = mask_observation(mask, 0.6, rgb, view, "yellow", background)
    assert result is not None and result["xy"][1] > 20
    assert any(p["xy"][1] > 20 for p in candidates(rgb, view, background))
    assert result["variance"][1] > 0


def test_delivery_gate_follows_hack_to_button_direction_in_inverted_view():
    gate = DeliveryGate()
    view = dict(kind="perspective", landmarks={"hack": [100, 150], "button": [100, 100]})
    results = []
    for t, y in enumerate([150, 146, 141, 136]):
        results.extend(gate.update(t, view, [dict(color="red", point=[100, y], xy=[0, -2])]))
    assert len(results) == 1


def test_handle_without_reliable_granite_reports_depth_uncertainty():
    rgb, background, mask, view = synthetic_stone()
    # Remove the body while retaining the colored handle.
    rgb[152:170] = background[152:170]
    result = mask_observation(mask, 0.6, rgb, view, "yellow", background)
    assert result is not None and result["ground_uncertain"]
    assert result["variance"][1] >= 0.8**2


def test_bottom_cropped_mask_is_not_mistaken_for_ice_contact():
    rgb, background, mask, view = synthetic_stone()
    mask[299, 150] = True
    assert mask_observation(mask, 1, rgb, view, "yellow", background) is None


def test_granite_mode_requires_body_in_saved_mask_not_an_inferred_foot():
    rgb, background, handle, view = synthetic_stone()
    assert mask_observation(handle, 0.7, rgb, view, "yellow", background, require_granite=True) is None
    full = handle.copy()
    full[151:165, 139:162] = True
    obs = mask_observation(full, 0.7, rgb, view, "yellow", background, require_granite=True)
    assert obs is not None
    assert obs["measurement_kind"] == "visible_granite_bottom"
    assert obs["point"][1] >= 160
    assert not obs["ground_uncertain"]

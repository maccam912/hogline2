import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from hogline2.tracking_stone import (  # noqa: E402
    body_clicks,
    granite_anchor,
    granite_evidence,
    seed_handle,
)


def stone(scale=1):
    rgb = np.full((240, 300, 3), 240, np.uint8)
    body = np.zeros(rgb.shape[:2], np.uint8)
    handle = body.copy()
    cv2.ellipse(body, (150, 157), (18, 11), 0, 0, 360, 1, -1)
    cv2.ellipse(handle, (150, 144), (10, 4), 0, 0, 360, 1, -1)
    rgb[body.astype(bool)] = [105, 105, 105]
    rgb[handle.astype(bool)] = [240, 220, 20]
    full = (body | handle).astype(bool)
    if scale != 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        full = cv2.resize(full.astype("uint8"), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) > 0
    return rgb, full


@pytest.mark.parametrize("scale", [1, 3])
def test_body_prompts_follow_stone_size_and_do_not_accept_handle_only(scale):
    rgb, full = stone(scale)
    handle = seed_handle(rgb, "yellow", [150 * scale, 144 * scale])
    evidence = granite_evidence(rgb, handle, "yellow")
    assert evidence is not None
    points, labels = body_clicks(evidence, rgb.shape)
    assert points[1, 1] > 151 * scale
    assert list(labels[:2]) == [1, 1]
    assert granite_anchor(handle, evidence) is None
    anchor = granite_anchor(full, granite_evidence(rgb, full, "yellow"))
    assert anchor is not None and 164 * scale <= anchor[1] <= 169 * scale


def test_remote_mask_pixel_does_not_move_bottom_and_person_mask_is_rejected():
    rgb, mask = stone()
    expected = granite_anchor(mask, granite_evidence(rgb, mask, "yellow"))
    mask[230, 280] = True
    assert granite_anchor(mask, granite_evidence(rgb, mask, "yellow")) == expected
    mask[20:145, 145:156] = True
    assert granite_anchor(mask, granite_evidence(rgb, mask, "yellow")) is None


def test_painted_blue_below_handle_is_not_granite():
    rgb, mask = stone()
    rgb[(rgb[:, :, 0] == 105)] = [20, 90, 190]
    assert granite_anchor(mask, granite_evidence(rgb, mask, "yellow")) is None

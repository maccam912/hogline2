"""Experimental automatic full-body reseeding, separate from motion filtering."""

import cv2
import numpy as np

from .tracking_sam import Sam21MLX


def body_prompt(rgb, mask, color):
    """Find compact granite immediately below the selected colored handle.

    No model mask is expanded. Returned points must be segmented again by SAM.
    This is a conservative experimental cue, not a general stone detector.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if color == "red":
        colored = ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 100)
    else:
        colored = (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] > 95)
    colored &= mask & (hsv[:, :, 2] > 75)
    ys, xs = np.where(colored)
    if len(xs) < 4:
        return None
    width = max(4, int(xs.max() - xs.min() + 1))
    cx, cy = float(np.median(xs)), float(np.median(ys))
    x0, x1 = max(0, int(cx - width)), min(rgb.shape[1], int(cx + width + 1))
    y0, y1 = max(0, int(ys.max())), min(rgb.shape[0], int(ys.max() + width * 1.3 + 1))
    patch = hsv[y0:y1, x0:x1]
    granite = ((patch[:, :, 1] < 80) & (patch[:, :, 2] > 40) & (patch[:, :, 2] < 205)).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(granite)
    choices = []
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area < max(8, width) or not width * 0.55 < w <= width * 2.2 or h > width * 1.5:
            continue
        if not x0 + x <= cx <= x0 + x + w or y > width * 0.35:
            continue
        gy, gx = np.where(components == i)
        target = [cx, min(y0 + y + h - 2, float(ys.max()) + width * 0.45)]
        best = np.argmin((gx + x0 - target[0]) ** 2 + (gy + y0 - target[1]) ** 2)
        body = [float(gx[best] + x0), float(gy[best] + y0)]
        if body[1] - cy < 4:
            continue
        choices.append((area, [cx, cy], body))
    if not choices:
        return None
    _, handle, body = max(choices)
    return dict(handle=handle, body=body, outside_mask=not bool(mask[int(body[1]), int(body[0])]))


class Sam21Reinitialized(Sam21MLX):
    name = "sam2.1-mlx-reinitialized"

    def __init__(self, project=None):
        super().__init__(project)
        self.hint = None
        self.bad_frames = {}
        self.last_reset = {}
        self.events = []

    def prompt_points(self, rgb, point):
        if self.hint is None:
            return super().prompt_points(rgb, point)
        handle, body = self.hint["handle"], self.hint["body"]
        points = np.asarray([handle, body, [handle[0], max(0, handle[1] - 28)]], np.float32)
        return points, np.asarray([1, 1, 0], np.int32)

    def observe(self, view, name, color, rgb, point=None, fps=3):
        key = (view["id"], name)
        if point is not None:
            self.states.pop(key, None)
        mask, quality = super().track(*key, rgb, point)
        hint = body_prompt(rgb, mask, color) if view["kind"] == "perspective" else None
        needs_reset = hint is not None and hint["outside_mask"]
        self.bad_frames[key] = self.bad_frames.get(key, 0) + 1 if needs_reset else 0
        if (
            needs_reset
            and (point is not None or self.bad_frames[key] >= 2)
            and (self.frame_index - self.last_reset.get(key, -10000) >= 2 * fps)
        ):
            old_state = self.states.pop(key)
            self.hint = hint
            try:
                candidate, candidate_quality = super().track(*key, rgb, hint["handle"])
            finally:
                self.hint = None
            overlap = (candidate & mask).sum() / max(1, mask.sum())
            bx, by = map(int, hint["body"])
            accepted = bool(candidate[by, bx] and overlap > 0.5 and candidate.sum() < mask.sum() * 5)
            if accepted:
                mask, quality = candidate, candidate_quality
            else:
                self.states[key] = old_state
            self.last_reset[key] = self.frame_index
            self.bad_frames[key] = 0
            self.events.append(
                dict(
                    type="full_body_reinitialize",
                    frame=self.frame_index,
                    stone=name,
                    view=view["id"],
                    accepted=accepted,
                    **hint,
                )
            )
        return mask, quality

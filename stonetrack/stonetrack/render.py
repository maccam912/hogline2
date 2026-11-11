"""Review pictures: the whole sheet from above with every stone's path and final position."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import BACK, FT, HOG, RINGS, STONE_RADIUS, TEE_DISTANCE

SCALE = 40  # pixels per metre
MARGIN = 20
COL = {"red": (40, 40, 220), "yellow": (0, 200, 240)}


def _pt(x, y):
    # sheet drawn left to right: tee A on the left, tee B on the right
    return int(round(MARGIN + (y + BACK + 1.9) * SCALE)), int(round(MARGIN + (2.6 - x) * SCALE))


def sheet_canvas():
    w = int((TEE_DISTANCE + 2 * (BACK + 1.9)) * SCALE) + 2 * MARGIN
    h = int(5.2 * SCALE) + 2 * MARGIN
    img = np.full((h, w, 3), 245, np.uint8)
    for ty in (0.0, TEE_DISTANCE):
        for r, c in ((RINGS["r6"], (200, 140, 60)), (RINGS["r4"], (245, 245, 245)), (RINGS["r2"], (120, 200, 120)), (0.5 * FT, (245, 245, 245))):
            cv2.circle(img, _pt(0, ty), int(r * SCALE), c, -1, cv2.LINE_AA)
        d = 1 if ty == 0 else -1
        for Y, thick in ((ty, 1), (ty - BACK * d, 1), (ty + HOG * d, 2)):
            cv2.line(img, _pt(-2.375, Y), _pt(2.375, Y), (90, 90, 90), thick, cv2.LINE_AA)
    cv2.line(img, _pt(0, -BACK), _pt(0, TEE_DISTANCE + BACK), (150, 150, 150), 1, cv2.LINE_AA)
    cv2.rectangle(img, _pt(2.375, -BACK - 1.83), _pt(-2.375, TEE_DISTANCE + BACK + 1.83), (60, 60, 60), 1)
    return img


def top_down(result, out_path, title=None):
    img = sheet_canvas()
    for s in result["stones"]:
        pts = np.array([_pt(p[1], p[2]) for p in s["path"]], np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, COL[s["colour"]], 1, cv2.LINE_AA)
    for s in result["stones"]:
        if not s["final"]:
            continue
        c = _pt(*s["final"])
        cv2.circle(img, c, int(STONE_RADIUS * SCALE), (90, 90, 90), -1, cv2.LINE_AA)
        cv2.circle(img, c, int(STONE_RADIUS * SCALE * 0.7), COL[s["colour"]], -1, cv2.LINE_AA)
        cv2.putText(img, s["id"], (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (MARGIN, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)
    return img

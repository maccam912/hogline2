"""Visible granite evidence and spatial anchors; no temporal position filtering."""

import cv2
import numpy as np


def handle_pixels(rgb, color):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = (
        ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165))
        if color == "red"
        else ((hsv[:, :, 0] >= 17) & (hsv[:, :, 0] <= 42))
    )
    return hue & (hsv[:, :, 1] > 85) & (hsv[:, :, 2] > 65)


def seed_handle(rgb, color, point):
    """The colored component containing/nearest the supplied handle point."""
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        handle_pixels(rgb, color).astype("uint8")
    )
    choices = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        distance = float(np.linalg.norm(centers[i] - point))
        if area >= 3 and distance <= max(8, w * 0.65) and w < rgb.shape[1] * 0.2 and h <= w * 1.5:
            choices.append((distance, i))
    return labels == min(choices)[1] if choices else np.zeros(rgb.shape[:2], bool)


def masked_granite(rgb, mask, handle, handle_top, handle_bottom, handle_width):
    """Accept an already complete model mask even when ice/base contrast is weak."""
    part = stone_component(mask, handle)
    yy, xx = np.where(part)
    if not len(xx) or yy.min() < handle_top - max(4, handle_width * 0.4):
        return None
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    body = part & (hsv[:, :, 1] < 95) & (hsv[:, :, 2] > 20) & (hsv[:, :, 2] < 220)
    body[: int(handle_bottom) + 1] = False
    by, bx = np.where(body)
    if len(bx) < max(12, handle_width * 2):
        return None
    depth = np.quantile(by, 0.95) - handle_bottom
    width = bx.max() - bx.min() + 1
    if depth < max(3, handle_width * 0.2) or width < handle_width * 0.8 or depth > width:
        return None
    if xx.max() - xx.min() + 1 > width * 1.5 or yy.max() - handle_top > width * 1.5:
        return None
    target = [float(np.median(bx)), float(np.quantile(by, 0.65))]
    i = np.argmin((bx - target[0]) ** 2 + (by - target[1]) ** 2)
    point = [float(bx[i]), float(by[i])]
    return dict(
        handle=handle,
        body_points=[point] * 3,
        pixels=(by, bx),
        bbox=[int(bx.min()), int(by.min()), int(bx.max() + 1), int(by.max() + 1)],
        handle_top=handle_top,
        handle_width=handle_width,
        edge_strength=None,
    )


def granite_evidence(rgb, mask, color):
    """Find a compact visible neutral body below a mask's colored handle.

    Evidence proposes SAM clicks; it is never painted into the returned model mask.
    Ambiguous/occluded bodies may deliberately produce no evidence.
    """
    colored = handle_pixels(rgb, color) & mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(colored.astype("uint8"))
    if count <= 1:
        return None
    selected = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cy, cx = np.where(labels == selected)
    if len(cx) < 3:
        return None
    width = max(4, int(cx.max() - cx.min() + 1))
    center = float(np.median(cx))
    handle = [float(np.median(cx)), float(np.quantile(cy, 0.8))]
    nearest = np.argmin((cx - handle[0]) ** 2 + (cy - handle[1]) ** 2)
    handle = [float(cx[nearest]), float(cy[nearest])]
    existing = masked_granite(rgb, mask, handle, int(cy.min()), int(cy.max()), width)
    if existing:
        return existing
    x0, x1 = max(0, int(center - width * 2)), min(rgb.shape[1], int(center + width * 2 + 1))
    y0, y1 = max(0, int(cy.max()) - 1), min(rgb.shape[0], int(cy.max() + max(24, width * 3) + 1))
    hsv = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
    # Find the lower body/ice transition before selecting gray pixels. This
    # prevents connected shadows and ice seams from defining the body's extent.
    left = max(0, int(center - x0 - width * 0.22))
    right = min(hsv.shape[1], int(center - x0 + width * 0.22) + 1)
    profile = np.median(hsv[:, left:right, 2], axis=1)
    if len(profile) < 8:
        return None
    gradient = np.diff(np.convolve(profile, np.ones(3) / 3, mode="valid"))
    gradient[:2] = 0
    gradient[-2:] = 0
    peak = float(gradient.max())
    if peak < 8:
        return None
    edges = np.where(gradient >= max(8, peak * 0.6))[0]
    for edge in edges:
        bottom = int(edge + 2)
        middle = max(1, int(bottom * 0.55))
        # Granite should be neutral at the core; a blue painted line is not a body.
        if np.median(hsv[middle, left:right, 1]) > 95:
            continue
        ceiling = min(
            215,
            float(np.median(profile[min(bottom + 2, len(profile) - 1) : min(bottom + 6, len(profile))])) - 12,
        )
        ceiling = min(
            ceiling, float(np.median(profile[max(1, int(bottom * 0.35)) : max(2, int(bottom * 0.75))])) + 30
        )
        gray = (
            (hsv[:bottom, :, 1] < 95) & (hsv[:bottom, :, 2] > 25) & (hsv[:bottom, :, 2] < ceiling)
        ).astype("uint8")
        n, components, boxes, _ = cv2.connectedComponentsWithStats(gray)
        choices = []
        for i in range(1, n):
            x, y, w, h, area = map(int, boxes[i])
            if area < max(8, width) or w < 0.65 * width or w > 4 * width + 1 or h < 3 or h > w:
                continue
            if not x0 + x <= center < x0 + x + w or y > 0.5 * width:
                continue
            yy, xx = np.where(components == i)
            xx, yy = xx + x0, yy + y0
            points = []
            for tx, ty in [
                (center, y0 + bottom * 0.6),
                (center, y0 + bottom * 0.6),
                (center, y0 + bottom * 0.7),
            ]:
                k = np.argmin((xx - tx) ** 2 + (yy - ty) ** 2)
                points.append([float(xx[k]), float(yy[k])])
            if points[-1][1] - handle[1] < 3:
                continue
            choices.append(
                (
                    area,
                    dict(
                        handle=handle,
                        body_points=points,
                        pixels=(yy, xx),
                        bbox=[x0 + x, y0 + y, x0 + x + w, y0 + bottom],
                        handle_top=int(cy.min()),
                        handle_width=width,
                        edge_strength=peak,
                    ),
                )
            )
        if choices:
            return max(choices, key=lambda item: item[0])[1]
    return None


def stone_component(mask, handle):
    """Retain the model's component containing the selected handle."""
    _, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"))
    x, y = map(lambda p: int(round(p)), handle)
    label = labels[min(y, mask.shape[0] - 1), min(x, mask.shape[1] - 1)]
    return labels == label if label else np.zeros_like(mask, dtype=bool)


def granite_anchor(mask, evidence):
    """Bottom envelope of a verified full-body mask, not the handle or a remote pixel."""
    if evidence is None:
        return None
    part = stone_component(mask, evidence["handle"])
    yy, xx = np.where(part)
    if not len(xx) or yy.max() >= mask.shape[0] - 1:
        return None
    by, bx = evidence["pixels"]
    if part[by, bx].mean() < 0.65:
        return None
    if not all(part[int(y), int(x)] for x, y in evidence["body_points"]):
        return None
    x0, y0, x1, y1 = evidence["bbox"]
    width = x1 - x0
    if yy.min() < evidence["handle_top"] - max(4, width * 0.25):
        return None
    if xx.max() - xx.min() + 1 > width * 1.8 or yy.max() - evidence["handle_top"] > 2 * (
        y1 - evidence["handle_top"]
    ):
        return None
    core_x = xx[yy >= max(p[1] for p in evidence["body_points"])]
    if not len(core_x):
        return None
    body_left, body_right = int(core_x.min()), int(core_x.max())
    body_width = body_right - body_left + 1
    columns = range(int(body_left + body_width * 0.2), int(body_right - body_width * 0.2) + 1)
    bottom = [(x, np.where(part[:, x])[0][-1]) for x in columns if part[:, x].any()]
    if len(bottom) < max(3, body_width * 0.35):
        return None
    # Spatial median of column bottoms resists thin protrusions; no time averaging.
    point = [float(np.median([p[0] for p in bottom])), float(np.median([p[1] for p in bottom]))]
    if point[1] < max(p[1] for p in evidence["body_points"]) + 1 or point[1] > y1 + max(3, (y1 - y0) * 0.5):
        return None
    return point


def body_clicks(evidence, shape):
    x0, y0, x1, y1 = evidence["bbox"]
    w = x1 - x0
    positives = [evidence["handle"], evidence["body_points"][-1]]
    negatives = [
        [evidence["handle"][0], max(0, evidence["handle_top"] - max(3, w * 0.12))],
        [max(0, x0 - max(3, w * 0.2)), (y0 + y1) / 2],
        [min(shape[1] - 1, x1 + max(3, w * 0.2)), (y0 + y1) / 2],
        [(x0 + x1) / 2, min(shape[0] - 1, y1 + max(3, w * 0.2))],
    ]
    return np.asarray(positives + negatives, np.float32), np.asarray([1] * 2 + [0] * 4, np.int32)

"""Stone detection in one mosaic frame.

At this footage's quality a stone is 35-60 px wide with a saturated red or yellow handle, so a
learned detector is not needed: colour finds candidates, the granite body under (or around) the
handle confirms them, and the camera model turns the handle's pixel into an ice position.

Each detection carries a 2x2 covariance in ice metres. That covariance is what lets the tracker
trust an overhead camera to a centimetre and a far perspective camera to ~10 cm along the sheet.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import HACK, SHEET_HALF_WIDTH, STONE_CAP_Z, STONE_HANDLE_Z, STONE_RADIUS, tee, toward_centre

CAP_RADIUS = 0.100  # coloured cap as seen from above, metres
COLOURS = ("red", "yellow")


@dataclass
class Detection:
    view: str
    kind: str
    colour: str
    px: tuple  # anchor pixel (cap or handle centroid)
    ice: tuple  # stone centre on the ice, metres
    cov: tuple  # (xx, xy, yy) in m^2
    quality: float  # 0..1: how complete / well confirmed
    bottom_px: tuple | None = None  # perspective only: lowest visible granite under the handle

    def as_dict(self):
        d = dict(view=self.view, colour=self.colour, px=[round(v, 2) for v in self.px], ice=[round(v, 4) for v in self.ice],
                 cov=[float(f"{v:.3g}") for v in self.cov], q=round(self.quality, 2))
        if self.bottom_px:
            d["bottom_px"] = [round(v, 2) for v in self.bottom_px]
        return d


def colour_masks(hsv, kind):
    """Handle colours as they actually appear here: the yellow is pale from the side cameras,
    and granite itself has a faint red hue, so saturation carries most of the decision."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h <= 8) | (h >= 163)) & (s >= 75) & (v >= 80)
    yellow = (h >= 16) & (h <= 36) & (s >= (70 if kind == "overhead" else 45)) & (v >= 105)
    return {"red": red, "yellow": yellow}


def _measurement_cov(cam, px, z, sigma_px, sigma_z):
    J = cam.jacobian(px, z)
    cov = J @ J.T * sigma_px**2
    if cam.height is not None and sigma_z:
        ice = cam.to_ice(px, z)[0]
        ray = ice - cam.ground  # an error in the assumed height slides the stone along this ray
        cov += np.outer(ray, ray) * (sigma_z / (cam.height - z)) ** 2
    # floors measured against the overhead cameras: a gripped or turned handle moves its centroid
    return cov + np.eye(2) * (0.012 if cam.kind == "overhead" else 0.04) ** 2


class Detector:
    """Per-recording detector: cameras plus the static background of the same recording."""

    def __init__(self, cameras, background):
        self.cams = cameras
        self.handle_z = {c.id: c.extra.get("handle_height", STONE_HANDLE_Z) for c in cameras.values()}
        self.bg = background
        self.bg_v = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)[..., 2].astype(np.int16)
        self.region = {cid: self._ice_region(cam) for cid, cam in cameras.items()}

    def _ice_region(self, cam):
        """Mask of the panel where a stone in play could show its handle."""
        x0, y0, x1, y1 = cam.crop
        mask = np.zeros(self.bg.shape[:2], np.uint8)
        if cam.kind == "overhead":
            mask[y0 + 6 : y1 - 6, x0 + 6 : x1 - 6] = 1
            return mask.astype(bool)
        ty, d = tee(cam.end)[1], toward_centre(cam.end)
        w = SHEET_HALF_WIDTH + 0.15
        corners = [[-w, ty - HACK * d], [w, ty - HACK * d], [w, ty + 24 * d], [-w, ty + 24 * d]]
        poly = cam.to_image(corners, STONE_HANDLE_Z)
        cv2.fillPoly(mask, [np.int32(np.round(poly))], 1)
        mask[:y0] = 0
        mask[y1:] = 0
        mask[:, :x0] = 0
        mask[:, x1:] = 0
        return mask.astype(bool)

    def detect(self, frame, foreground=None):
        """Detections in one frame. Pass a dict as `foreground` to also receive, per view, the
        boxes of everything that differs from the empty sheet: the tracker uses them to tell
        "that spot is visibly bare ice" from "something is standing in the way"."""
        out = []
        for cam in self.cams.values():
            x0, y0, x1, y1 = cam.crop
            sub = (slice(y0, y1), slice(x0, x1))
            hsv = cv2.cvtColor(frame[sub], cv2.COLOR_BGR2HSV)
            d = cv2.absdiff(frame[sub], self.bg[sub])
            changed = cv2.max(cv2.max(d[..., 0], d[..., 1]), d[..., 2]) > 30  # static paint and logos never fire
            masks = colour_masks(hsv, cam.kind)
            self._changed = changed
            if foreground is not None:
                small = cv2.resize(changed.astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                small = cv2.dilate((small > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
                n, _, st, _ = cv2.connectedComponentsWithStats(small, connectivity=8)
                foreground[cam.id] = [[int(x0 + 4 * bx), int(y0 + 4 * by), int(x0 + 4 * (bx + bw)), int(y0 + 4 * (by + bh))]
                                      for bx, by, bw, bh, area in st[1:] if area >= 6]
            keep = changed & self.region[cam.id][sub]
            for colour in COLOURS:
                m = (masks[colour] & keep).astype(np.uint8)
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                n, _, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
                for i in range(1, n):
                    bx, by, bw, bh, area = stats[i]
                    if area < 12:
                        continue
                    c = cents[i] + [x0, y0]
                    find = self._overhead if cam.kind == "overhead" else self._perspective
                    det = find(cam, colour, c, (bx + x0, by + y0, bw, bh, area), hsv, (x0, y0))
                    if det:
                        out.append(det)
        return out

    # ---- overhead: the coloured cap is a filled disc with a granite annulus around it
    def _overhead(self, cam, colour, c, box, hsv, origin):
        bx, by, bw, bh, area = box
        ice = cam.to_ice(c, STONE_CAP_Z)[0]
        r = CAP_RADIUS * cam.pixels_per_metre(ice)
        full = np.pi * r * r
        if not 0.25 * full <= area <= 1.7 * full or max(bw, bh) > 2.9 * r or min(bw, bh) < 0.6 * r:
            return None
        R = STONE_RADIUS * cam.pixels_per_metre(ice)
        yy, xx = np.ogrid[int(c[1] - R - 2) : int(c[1] + R + 3), int(c[0] - R - 2) : int(c[0] + R + 3)]
        x0, y0, x1, y1 = cam.crop
        inside = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        rr = np.hypot(xx - c[0], yy - c[1])
        ring = (rr > 1.2 * r) & (rr < 0.95 * R) & inside
        if ring.sum() < 20:
            return None
        ys, xs = np.nonzero(ring)
        ys += int(c[1] - R - 2)
        xs += int(c[0] - R - 2)
        s, v = hsv[ys - origin[1], xs - origin[0], 1], hsv[ys - origin[1], xs - origin[0], 2].astype(np.int16)
        granite = ((s < 90) & (self.bg_v[ys, xs] - v > 18)).mean()
        if granite < 0.3:
            return None
        quality = float(min(1.0, area / (0.85 * full)) * min(1.0, granite / 0.6))
        cov = _measurement_cov(cam, c, STONE_CAP_Z, 0.8 + 2.0 * (1 - quality), 0.01)
        return Detection(cam.id, cam.kind, colour, tuple(c), tuple(ice), (cov[0, 0], cov[0, 1], cov[1, 1]), quality)

    # ---- perspective: a coloured handle sitting on a grey body
    def _perspective(self, cam, colour, c, box, hsv, origin):
        bx, by, bw, bh, area = box
        hz = self.handle_z[cam.id]
        ice = cam.to_ice(c, hz)[0]
        W = 2 * STONE_RADIUS * cam.pixels_per_metre(ice)  # stone width in pixels here
        if not (0.22 * W <= bw <= 1.0 * W and 0.1 * W <= bh <= 0.65 * W and area >= 0.02 * W * W):
            return None
        x0, y0, x1, y1 = cam.crop
        top, bot = by + bh, min(int(by + bh + 0.45 * W) + 1, y1)
        lft, rgt = max(int(c[0] - 0.4 * W), x0), min(int(c[0] + 0.4 * W) + 1, x1)
        if bot - top < 3 or rgt - lft < 4:
            return None
        ox, oy = origin
        s = hsv[top - oy : bot - oy, lft - ox : rgt - ox, 1]
        v = hsv[top - oy : bot - oy, lft - ox : rgt - ox, 2].astype(np.int16)
        body = (s < 70) & (self.bg_v[top:bot, lft:rgt] - v > 45) & (v > 70)
        granite = float(body.mean())
        if granite < 0.3:
            return None
        # ...and the grey must stop: a face above a grey hoodie has "granite" that keeps going
        b0, b1 = min(int(top + 0.65 * W), y1), min(int(top + 0.95 * W) + 1, y1)
        if b1 - b0 >= 3:
            s2 = hsv[b0 - oy : b1 - oy, lft - ox : rgt - ox, 1]
            v2 = hsv[b0 - oy : b1 - oy, lft - ox : rgt - ox, 2].astype(np.int16)
            if ((s2 < 70) & (self.bg_v[b0:b1, lft:rgt] - v2 > 45) & (v2 > 70)).mean() > 0.6:
                return None
        # A stone is an island: bare ice beside it and in front of it. Faces and hands above grey
        # clothing pass the colour and body tests but are surrounded by more person.
        ch = self._changed

        def clear(xa, xb, ya, yb):
            xa, xb = max(int(xa), x0) - ox, min(int(xb) + 1, x1) - ox
            ya, yb = max(int(ya), y0) - oy, min(int(yb) + 1, y1) - oy
            return 1.0 if xb <= xa or yb <= ya else 1.0 - float(ch[ya:yb, xa:xb].mean())

        flank_rows = (top, top + 0.35 * W)
        islands = [clear(c[0] - 0.9 * W, c[0] - 0.65 * W, *flank_rows), clear(c[0] + 0.65 * W, c[0] + 0.9 * W, *flank_rows),
                   clear(c[0] - 0.3 * W, c[0] + 0.3 * W, top + 0.6 * W, top + 0.8 * W)]
        if max(islands) < 0.6:
            return None
        # lowest contiguous granite row under the handle: the near edge of the running band
        cols = body[:, int(0.25 * body.shape[1]) : int(0.75 * body.shape[1]) + 1]
        rows = cols.mean(axis=1) > 0.5
        bottom = None
        if rows.any():
            last = len(rows) - 1 - int(np.argmax(rows[::-1]))
            if last < len(rows) - 1:  # the body ends inside the window, so the edge is real
                bottom = (float(c[0]), float(top + last + 1))
        quality = float(min(1.0, granite / 0.6))
        cov = _measurement_cov(cam, c, hz, 1.5, 0.012)
        return Detection(cam.id, cam.kind, colour, tuple(c), tuple(ice), (cov[0, 0], cov[0, 1], cov[1, 1]), quality, bottom)


def draw(frame, detections, labels=None):
    """Review overlay: a marker on every detection's anchor pixel."""
    out = frame.copy()
    for i, d in enumerate(detections):
        col = (0, 0, 255) if d.colour == "red" else (0, 220, 255)
        p = (int(round(d.px[0])), int(round(d.px[1])))
        cv2.circle(out, p, 9, (0, 0, 0), 3)
        cv2.circle(out, p, 9, col, 2)
        if d.bottom_px:
            q = (int(round(d.bottom_px[0])), int(round(d.bottom_px[1])))
            cv2.line(out, (q[0] - 6, q[1]), (q[0] + 6, q[1]), (0, 255, 0), 1)
        if labels:
            cv2.putText(out, labels[i], (p[0] + 11, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, labels[i], (p[0] + 11, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out

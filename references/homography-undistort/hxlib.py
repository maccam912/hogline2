"""Shared helpers for the undistort + ice-plane homography experiment.

Conventions follow hogline2.tracking_geometry: pixel coordinates are in the full 1920x1080
mosaic; ice coordinates are metres with tee A at (0, 0), tee B at (0, TEE_DISTANCE), X lateral.
A view model is an ice->image homography G (3x3, G[2,2] = 1) followed by one-coefficient radial
distortion about a centre; `k1` is expressed at the panel half-diagonal so its value reads as the
fractional displacement of the panel corner.
"""
import json

import cv2
import numpy as np

TEE_DISTANCE = 34.747
FT = 0.3048
RINGS = {"r2": 2 * FT, "r4": 4 * FT, "r6": 6 * FT}
HOG = 6.401

PANELS = {  # id: (kind, end, crop x0,y0,x1,y1) from data/camera-layout.json
    "perspective_a": ("perspective", "A", (0, 0, 724, 990)),
    "perspective_b": ("perspective", "B", (1200, 0, 1920, 990)),
    "overhead_a": ("overhead", "A", (730, 7, 1188, 463)),
    "overhead_b": ("overhead", "B", (730, 611, 1188, 1063)),
}


def tee(end):
    return np.array([0.0, 0.0 if end == "A" else TEE_DISTANCE])


def apply_h(h, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    q = np.c_[pts, np.ones(len(pts))] @ np.asarray(h, float).reshape(3, 3).T
    return q[:, :2] / q[:, 2:3]


class View:
    """ice -> image: homography G then radial distortion (centre c, k1 at radius R)."""

    def __init__(self, G, centre, R, k1=0.0):
        self.G = np.asarray(G, float).reshape(3, 3)
        self.Ginv = np.linalg.inv(self.G)
        self.c = np.asarray(centre, float)
        self.R = float(R)
        self.k1 = float(k1)

    def distort(self, pu):
        d = np.atleast_2d(pu) - self.c
        r2 = (d**2).sum(1, keepdims=True) / self.R**2
        return self.c + d * (1 + self.k1 * r2)

    def undistort(self, pd):
        pd = np.atleast_2d(np.asarray(pd, float))
        if self.k1 == 0:
            return pd
        d = pd - self.c
        u = d.copy()
        for _ in range(8):  # fixed point; converges fast for small k1
            r2 = (u**2).sum(1, keepdims=True) / self.R**2
            u = d / (1 + self.k1 * r2)
        return self.c + u

    def to_image(self, ice):
        return self.distort(apply_h(self.G, ice))

    def to_ice(self, px):
        return apply_h(self.Ginv, self.undistort(px))

    def as_dict(self):
        return dict(ice_to_image=self.G.ravel().tolist(), image_to_ice=(self.Ginv / self.Ginv[2, 2]).ravel().tolist(),
                    distortion_centre=self.c.tolist(), distortion_radius=self.R, k1=self.k1)

    @classmethod
    def from_dict(cls, d):
        return cls(d["ice_to_image"], d["distortion_centre"], d["distortion_radius"], d["k1"])


def load_views(path):
    data = json.load(open(path))
    return {k: View.from_dict(v) for k, v in data["views"].items()}, data


def subpixel_edge(img, pts, normals, half=3.0, step=0.25, plateau=1.0):
    """Refine edge points along their normals: the 50% crossing of the colour step.

    img: float32 BGR. Returns (refined points, ok mask, contrast). Colour is projected onto the
    axis between the two side plateaus, so any pair of paints works and a thin dark outline on
    the boundary only shifts the crossing by a fraction of its own width.
    """
    ts = np.arange(-half, half + 1e-6, step)
    out = np.full_like(pts, np.nan, dtype=float)
    ok = np.zeros(len(pts), bool)
    contrast = np.zeros(len(pts))
    for i, (p, n) in enumerate(zip(pts, normals)):
        xy = p[None] + ts[:, None] * n[None]
        prof = cv2.remap(img, xy[:, 0].astype(np.float32)[None], xy[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
        a = np.median(prof[ts <= -half + plateau], axis=0)
        b = np.median(prof[ts >= half - plateau], axis=0)
        axis = b - a
        norm = float(axis @ axis)
        if norm < 20**2:  # weak step: not a paint edge
            continue
        s = (prof - a) @ axis / norm
        s = np.convolve(s, np.ones(3) / 3, mode="same")
        cross = np.where((s[:-1] < 0.5) & (s[1:] >= 0.5))[0]
        if len(cross) == 0:
            continue
        j = cross[np.argmin(np.abs(ts[cross]))]
        t = ts[j] + step * (0.5 - s[j]) / (s[j + 1] - s[j])
        out[i] = p + t * n
        ok[i] = True
        contrast[i] = norm**0.5
    return out, ok, contrast


def contour_points(mask, min_len=40):
    """Outer and hole contours of the largest blob with unit normals from the blurred mask."""
    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, []
    hier = hier[0]
    outers = [i for i in range(len(contours)) if hier[i][3] < 0]
    big = max(outers, key=lambda i: cv2.contourArea(contours[i]))
    holes = [i for i in range(len(contours)) if hier[i][3] == big and len(contours[i]) >= min_len]
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)
    gx = cv2.Sobel(soft, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(soft, cv2.CV_32F, 0, 1, ksize=3)

    def with_normals(c):
        p = c[:, 0, :].astype(float)
        n = np.c_[gx[c[:, 0, 1], c[:, 0, 0]], gy[c[:, 0, 1], c[:, 0, 0]]]
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        return p, -n  # outward from the blob

    return with_normals(contours[big]), [with_normals(contours[i]) for i in holes]


def ridge_minimum(profile):
    """Sub-sample position of the darkest point of a 1D profile (parabola through the minimum)."""
    i = int(np.argmin(profile))
    if i == 0 or i == len(profile) - 1:
        return None
    a, b, c = profile[i - 1], profile[i], profile[i + 1]
    den = a - 2 * b + c
    return i + (0.5 * (a - c) / den if den > 1e-9 else 0.0)

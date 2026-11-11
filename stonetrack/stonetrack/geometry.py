"""Sheet geometry and camera models.

Coordinates
  ice    metres. Tee A is (0, 0), tee B is (0, TEE_DISTANCE); X is lateral.
  image  pixels in the full broadcast mosaic (not panel-relative).

A camera is an ice->image homography `G` for the ice plane itself, plus where the camera stands:
the ice point beneath it (`ground`) and its `height`. Those two give the image of any point
*above* the ice (a stone's handle, its widest band) without a separate 3D calibration: a point at
height z over ice position P is seen where the ice point  ground + (P - ground) * H / (H - z)
would be. Measured lens distortion at this venue is below a pixel over the ice, so none is modelled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

FT = 0.3048
TEE_DISTANCE = 114 * FT  # 34.747 m
HOG = 21 * FT  # tee line to hog line
BACK = 6 * FT  # tee line to back line
HACK = 12 * FT
RINGS = {"r2": 2 * FT, "r4": 4 * FT, "r6": 6 * FT}
SHEET_HALF_WIDTH = 2.05  # beyond this a stone is against the side board: parked or out of play
FOUR_FOOT = 2 * FT  # the two faint lines parallel to the centre line

STONE_RADIUS = 0.1455  # widest band (914 mm circumference)
STONE_BAND_Z = 0.057  # height of that widest band
STONE_CAP_Z = 0.115  # coloured cap as seen from above
STONE_HANDLE_Z = 0.135  # centroid of the coloured handle as seen from the side
RUNNING_BAND_RADIUS = 0.0635


def tee(end: str) -> np.ndarray:
    return np.array([0.0, 0.0 if end == "A" else TEE_DISTANCE])


def toward_centre(end: str) -> int:
    """+1 / -1: the Y direction from this end's tee toward mid-sheet."""
    return 1 if end == "A" else -1


def solve_h(src, dst) -> np.ndarray:
    """Exact 4-point homography in float64."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(np.asarray(src, float), np.asarray(dst, float)):
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        rhs += [u, v]
    return np.append(np.linalg.solve(np.array(rows), np.array(rhs)), 1.0).reshape(3, 3)


def apply_h(h, pts) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, float))
    q = np.c_[pts, np.ones(len(pts))] @ np.asarray(h, float).reshape(3, 3).T
    return q[:, :2] / q[:, 2:3]


@dataclass
class Camera:
    id: str
    kind: str  # "overhead" | "perspective"
    end: str  # which house it looks at
    crop: tuple  # x0, y0, x1, y1 of its panel in the mosaic
    G: np.ndarray  # ice -> image
    ground: np.ndarray | None = None  # ice point under the camera
    height: float | None = None  # metres above the ice
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.G = np.asarray(self.G, float).reshape(3, 3)
        self.G = self.G / self.G[2, 2]
        self.Ginv = np.linalg.inv(self.G)
        if self.ground is not None:
            self.ground = np.asarray(self.ground, float)

    # -- elevated points ---------------------------------------------------------------
    def lift(self, ice, z):
        """Ice point whose image coincides with the point at height z above `ice`."""
        ice = np.atleast_2d(np.asarray(ice, float))
        if not z or self.height is None:
            return ice
        return self.ground + (ice - self.ground) * self.height / (self.height - z)

    def drop(self, ice_apparent, z):
        """Inverse of lift: true ice position of something at height z seen at `ice_apparent`."""
        ice_apparent = np.atleast_2d(np.asarray(ice_apparent, float))
        if not z or self.height is None:
            return ice_apparent
        return self.ground + (ice_apparent - self.ground) * (self.height - z) / self.height

    def to_image(self, ice, z=0.0):
        return apply_h(self.G, self.lift(ice, z))

    def to_ice(self, px, z=0.0):
        return self.drop(apply_h(self.Ginv, px), z)

    # -- local scale -------------------------------------------------------------------
    def jacobian(self, px, z=0.0, h=0.5):
        """d(ice)/d(pixel) at one image point: columns are metres per pixel in x and y."""
        px = np.asarray(px, float)
        a = self.to_ice([px + [h, 0], px - [h, 0], px + [0, h], px - [0, h]], z)
        return np.c_[(a[0] - a[1]) / (2 * h), (a[2] - a[3]) / (2 * h)]

    def pixels_per_metre(self, ice):
        """Image length of one lateral metre at an ice point (stone size in pixels)."""
        p = self.to_image([np.asarray(ice, float) - [0.5, 0], np.asarray(ice, float) + [0.5, 0]])
        return float(np.linalg.norm(p[1] - p[0]))

    def contains(self, px, margin=0):
        x0, y0, x1, y1 = self.crop
        px = np.atleast_2d(px)
        return (px[:, 0] >= x0 + margin) & (px[:, 0] < x1 - margin) & (px[:, 1] >= y0 + margin) & (px[:, 1] < y1 - margin)

    # -- persistence -------------------------------------------------------------------
    def as_dict(self):
        return dict(id=self.id, kind=self.kind, end=self.end, crop=list(self.crop), ice_to_image=self.G.ravel().tolist(),
                    image_to_ice=(self.Ginv / self.Ginv[2, 2]).ravel().tolist(),
                    ground=None if self.ground is None else self.ground.tolist(), height=self.height, **self.extra)

    @classmethod
    def from_dict(cls, d):
        known = {"id", "kind", "end", "crop", "ice_to_image", "image_to_ice", "ground", "height"}
        return cls(d["id"], d["kind"], d["end"], tuple(d["crop"]), d["ice_to_image"], d.get("ground"), d.get("height"),
                   {k: v for k, v in d.items() if k not in known})


def pinhole_pose(G, principal_point):
    """Camera position from an ice-plane homography, assuming square pixels and a known principal
    point. Returns (focal_px, [x, y, height]) or None when the view is too fronto-parallel for
    the plane to reveal the focal length (the overhead cameras)."""
    T = np.array([[1, 0, -principal_point[0]], [0, 1, -principal_point[1]], [0, 0, 1.0]])
    H = T @ np.asarray(G, float).reshape(3, 3)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    # two estimates of f^2 from r1.r2 = 0 and |r1| = |r2|; use the better conditioned one
    den_a, den_b = h1[2] * h2[2], h2[2] ** 2 - h1[2] ** 2
    cands = []
    if abs(den_b) > 1e-12:
        cands.append((abs(den_b), (h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / den_b))
    if abs(den_a) > 1e-12:
        cands.append((abs(den_a), -(h1[0] * h2[0] + h1[1] * h2[1]) / den_a))
    cands = [c for c in cands if c[1] > 0]
    if not cands:
        return None
    f = float(np.sqrt(max(cands)[1]))
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    a, b, t = Kinv @ h1, Kinv @ h2, Kinv @ h3
    lam = 2.0 / (np.linalg.norm(a) + np.linalg.norm(b))
    r1, r2, t = a * lam, b * lam, t * lam
    r3 = np.cross(r1, r2)
    R = np.c_[r1, r2, r3]
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    centre = -R.T @ t
    centre[2] = abs(centre[2])  # a mirrored ice frame only flips the sign of "up"
    tilt_ok = abs(h2[2]) * f > 0.05 * np.linalg.norm(h2[:2])  # plane clearly foreshortened
    return (f, centre) if tilt_ok else None


def save_calibration(path, cameras, meta):
    json.dump(dict(meta, cameras={c.id: c.as_dict() for c in cameras.values()}), open(path, "w"), indent=1)


def load_calibration(path):
    data = json.load(open(path))
    return {k: Camera.from_dict(v) for k, v in data["cameras"].items()}, data

"""Camera calibration from the paint on the ice.

Works on a temporal-median background image (people and stones removed). For each panel it
finds, to sub-pixel precision, the edges of the 2/4/6 ft rings and the thin painted lines, then
fits the ice->image homography by least squares on pixel (Sampson) distances.

What defines what
  rings        the tee position and the scale near the house
  hog line     the one fixed distance along the sheet (21 ft from the tee)
  thin lines   direction and straightness only: each gets a free lateral offset, because club
               paint wanders by about a centimetre and must not drag the tee around
  delta        one free "paint dilation" in pixels: saturated colour reads slightly fat in video
No manual landmarks: the initial guess comes from the ring's own bounding box.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

from .geometry import FOUR_FOOT, HOG, RINGS, Camera, apply_h, pinhole_pose, solve_h, tee, toward_centre

SIGN = {"r6": 1.0, "r4": -1.0, "r2": 1.0}  # which way a dilated colour moves each ring edge


# ---------------------------------------------------------------------------- edges
def subpixel_edge(img, pts, normals, half=3.0, step=0.25, plateau=0.75):
    """Refine edge points along their normals to the 50% crossing of the colour step."""
    ts = np.arange(-half, half + 1e-6, step)
    out = np.full(pts.shape, np.nan)
    ok = np.zeros(len(pts), bool)
    for i, (p, n) in enumerate(zip(pts, normals)):
        xy = (p[None] + ts[:, None] * n[None]).astype(np.float32)
        prof = cv2.remap(img, xy[None, :, 0], xy[None, :, 1], cv2.INTER_LINEAR)[0]
        a = np.median(prof[ts <= -half + plateau], axis=0)
        b = np.median(prof[ts >= half - plateau], axis=0)
        axis = b - a
        norm = float(axis @ axis)
        if norm < 20**2:  # too weak to be a paint edge
            continue
        s = np.convolve((prof - a) @ axis / norm, np.ones(3) / 3, mode="same")
        cross = np.where((s[:-1] < 0.5) & (s[1:] >= 0.5))[0]
        if len(cross) == 0:
            continue
        j = cross[np.argmin(np.abs(ts[cross]))]
        out[i] = p + (ts[j] + step * (0.5 - s[j]) / (s[j + 1] - s[j])) * n
        ok[i] = True
    return out, ok


def _contours_with_normals(mask):
    """(outer, holes) of the largest blob that has a hole; each is (points, outward normals)."""
    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, []
    hier = hier[0]
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)
    gx = cv2.Sobel(soft, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(soft, cv2.CV_32F, 0, 1, ksize=3)

    def pack(c):
        p = c[:, 0, :].astype(float)
        n = np.c_[gx[c[:, 0, 1], c[:, 0, 0]], gy[c[:, 0, 1], c[:, 0, 0]]]
        return p, -n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)

    outers = sorted((i for i in range(len(contours)) if hier[i][3] < 0), key=lambda i: -cv2.contourArea(contours[i]))
    for big in outers:
        holes = [i for i in range(len(contours)) if hier[i][3] == big and len(contours[i]) >= 40]
        if holes:
            return pack(contours[big]), [pack(contours[i]) for i in holes]
    return (pack(contours[outers[0]]) if outers else None), []


def ring_edges(bg, view):
    x0, y0, x1, y1 = view["crop"]
    overhead = view["kind"] == "overhead"
    inset = 9 if overhead else 0  # stay off the drawn panel frame
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    region = np.zeros(h.shape, bool)
    region[y0 + inset : y1 - inset, x0 + inset : x1 - inset] = True
    blue = region & (h > 95) & (h < 125) & (s > 40) & (v > 110)
    outer, holes = _contours_with_normals(blue)
    if outer is None or not holes:
        raise ValueError(f"{view['id']}: could not find the blue 12-foot ring")
    bx0, by0 = outer[0].min(0)
    bx1, by1 = outer[0].max(0)
    near = np.zeros(h.shape, bool)
    near[int(by0) : int(by1) + 1, int(bx0) : int(bx1) + 1] = True
    green = near & (h > 35) & (h < 85) & (s > (40 if overhead else 25)) & (v > 110)
    g_outer, _ = _contours_with_normals(green)
    img = bg.astype(np.float32)
    feats = {}
    for name, (pts, nrm) in {"r6": outer, "r4": max(holes, key=lambda c: len(c[0])), "r2": g_outer}.items():
        ref, ok = subpixel_edge(img, pts, nrm, half=3.0 if overhead else 2.5)
        m = 3 + inset
        ok &= (ref[:, 0] > x0 + m) & (ref[:, 0] < x1 - m) & (ref[:, 1] > y0 + m) & (ref[:, 1] < y1 - m)
        feats[name] = ref[ok]
    return feats, (bx0, by0, bx1, by1)


def _ridge_minimum(profile):
    i = int(np.argmin(profile))
    if i == 0 or i == len(profile) - 1:
        return None
    a, b, c = profile[i - 1], profile[i], profile[i + 1]
    den = a - 2 * b + c
    return i + (0.5 * (a - c) / den if den > 1e-9 else 0.0)


def track_line(gray, seed, positions, along, window=4, blur=11, min_depth=1.0, slope=0.0, first_window=10):
    """Follow a thin dark line from a seed, predicting each step from the points found so far.

    along='y' walks image rows (a roughly vertical line), 'x' walks columns. Walking on its own
    evidence keeps the tracker on the paint however rough the initial camera model is.
    """
    smooth = cv2.GaussianBlur(gray, (1, blur) if along == "y" else (blur, 1), 0)
    found, out, missed = [], [], 0
    for t in positions:
        if len(found) >= 12:
            recent = np.array(found[-60:])
            c = float(np.polyval(np.polyfit(recent[:, 0], recent[:, 1], 1), t))
        elif found:
            c = found[-1][1] + slope * (t - found[-1][0])
        else:
            c = seed
        w = window if found else first_window
        lo = int(round(c)) - w
        if lo - 12 < 0:
            continue
        prof = smooth[int(t), lo - 12 : lo + 2 * w + 13] if along == "y" else smooth[lo - 12 : lo + 2 * w + 13, int(t)]
        if len(prof) < 2 * w + 25:
            continue
        core = prof[12 : 12 + 2 * w + 1] - np.median(np.r_[prof[:10], prof[-10:]])
        m = _ridge_minimum(core)
        if m is None or core.min() > -min_depth:
            missed += 1
            if missed > 150:
                break
            continue
        missed = 0
        found.append([t, lo + m])
        out.append([lo + m, t] if along == "y" else [t, lo + m])
    return np.array(out, float).reshape(-1, 2)


def straightness(pts, along):
    """How far tracked paint departs from a straight line: the direct test for lens distortion."""
    if len(pts) < 20:
        return None
    t, v = (pts[:, 1], pts[:, 0]) if along == "y" else (pts[:, 0], pts[:, 1])
    res = v - np.polyval(np.polyfit(t, v, 1), t)
    span = float(t.max() - t.min())
    bow = float(np.polyfit(t - t.mean(), v, 2)[0] * (span / 2) ** 2)
    return dict(n=int(len(pts)), span_px=span, rms_px=float(res.std()), bow_px=bow)


# ---------------------------------------------------------------------------- model
def initial_model(view, box):
    """Rough ice->image from the ring's bounding box; orientation comes from the layout."""
    bx0, by0, bx1, by1 = box
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    ty, d, r = tee(view["end"])[1], toward_centre(view["end"]), RINGS["r6"]
    # perspective cameras look from the far end: image top is behind the house, image left is -X*d
    world = {"left": [-r * d, ty], "right": [r * d, ty], "top": [0, ty - r * d], "bottom": [0, ty + r * d]}
    if view.get("rotation_degrees", 0) == 180:
        world = {k: [-p[0], 2 * ty - p[1]] for k, p in world.items()}
    image = {"left": [bx0, cy], "right": [bx1, cy], "top": [cx, by0], "bottom": [cx, by1]}
    names = ["left", "right", "top", "bottom"]
    return solve_h([world[n] for n in names], [image[n] for n in names])


def collect(bg, view, G0, ring_feats):
    """Thin-line evidence, seeded from the current model `G0` and then followed on the image."""
    x0, y0, x1, y1 = view["crop"]
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ty, d = tee(view["end"])[1], toward_centre(view["end"])
    feats = dict(ring_feats)

    def x_at_row(X, row):
        a = apply_h(G0, [[X, ty - 2 * d], [X, ty + 12 * d]])
        return a[0, 0] + (row - a[0, 1]) * (a[1, 0] - a[0, 0]) / (a[1, 1] - a[0, 1])

    def y_at_col(Y, col):
        a = apply_h(G0, [[-2, Y], [2, Y]])
        return a[0, 1] + (col - a[0, 0]) * (a[1, 1] - a[0, 1]) / (a[1, 0] - a[0, 0])

    b = apply_h(G0, [[0, ty]])[0]
    if view["kind"] == "overhead":
        scale = np.linalg.norm(apply_h(G0, [[RINGS["r6"], ty]])[0] - b) / RINGS["r6"]
        kw = dict(window=4, blur=7, min_depth=2.0, first_window=5)
        feats["centre"] = np.vstack([track_line(gray, x_at_row(0.0, rows[0]), rows, "y", **kw)
                                     for rows in (range(int(b[1]) + 12, y1 - 14), range(int(b[1]) - 12, y0 + 14, -1))])
        feats["tee"] = np.vstack([track_line(gray, y_at_col(ty, cols[0]), cols, "x", **kw)
                                  for cols in (range(int(b[0]) + 12, x1 - 14), range(int(b[0]) - 12, x0 + 14, -1))])
        for name in ("centre", "tee"):  # stay clear of the button logo and the ring outlines
            p = feats[name]
            r = np.linalg.norm(p - b, axis=1) / scale
            ok = r > 0.2
            for ring in RINGS.values():
                ok &= np.abs(r - ring) > 0.06
            feats[name] = p[ok]
    else:
        front = int(apply_h(G0, [[0, ty + (RINGS["r6"] + 0.2) * d]])[0, 1]) + 2
        # the centre line leaves the house under the ring centre whatever the camera's skew, so
        # acquire it widely there (the rings alone do not pin down the sheet's direction)
        tries = [track_line(gray, x_at_row(0.0, front), range(front, y1 - 6), "y", first_window=w) for w in (30, 10)]
        feats["centre"] = max(tries, key=len)
        hog_y = y_at_col(ty + HOG * d, b[0])
        feats["hog"] = np.vstack([track_line(gray, hog_y, cols, "x", window=5, blur=9, min_depth=2.5)
                                  for cols in (range(int(b[0]) + 8, x1 - 30), range(int(b[0]) - 8, x0 + 6, -1))])
        start = int(np.median(feats["hog"][:, 1]) if len(feats["hog"]) else hog_y) + 8
        for name, X in (("par_neg", -FOUR_FOOT), ("par_pos", FOUR_FOOT)):
            seed = x_at_row(X, start)
            slope = (x_at_row(X, start + 50) - seed) / 50
            feats[name] = track_line(gray, seed, range(start, y1 - 6), "y", window=7, slope=slope)
    return feats


def fit(feats, G0, view):
    """Least-squares homography; residuals are Sampson distances in pixels."""
    ty, d = tee(view["end"])[1], toward_centre(view["end"])
    far = 2.0 if view["kind"] == "overhead" else 14.0  # anchors span the evidence
    anchors = np.array([[-2, ty - 2 * d], [2, ty - 2 * d], [2, ty + far * d], [-2, ty + far * d]], float)
    lines = [n for n in ("centre", "tee", "par_neg", "par_pos") if len(feats.get(n, ())) > 30]
    guess = {"centre": 0.0, "tee": 0.0, "par_neg": -FOUR_FOOT, "par_pos": FOUR_FOOT}
    used = [n for n in feats if (n in RINGS or n == "hog" or n in lines) and len(feats[n])]
    p0 = np.array(list(apply_h(G0, anchors).ravel()) + [0.5] + [guess[n] for n in lines])

    def errors(Ginv, p, name, px):
        q = apply_h(Ginv, px)
        if name in RINGS:
            return np.linalg.norm(q - [0, ty], axis=1) - RINGS[name]
        if name == "hog":
            return q[:, 1] - (ty + HOG * d)
        return q[:, 1 if name == "tee" else 0] - (ty if name == "tee" else 0.0) - p[9 + lines.index(name)]

    def pixel_residual(p, name):
        Ginv = np.linalg.inv(solve_h(anchors, p[:8].reshape(4, 2)))
        px, h = feats[name], 0.5
        e = errors(Ginv, p, name, px)
        gx = (errors(Ginv, p, name, px + [h, 0]) - errors(Ginv, p, name, px - [h, 0])) / (2 * h)
        gy = (errors(Ginv, p, name, px + [0, h]) - errors(Ginv, p, name, px - [0, h])) / (2 * h)
        return e / np.hypot(gx, gy) - SIGN.get(name, 0.0) * p[8]

    def residuals(p):  # every feature carries the same total weight, however many samples it has
        return np.concatenate([pixel_residual(p, n) * (200.0 / len(feats[n])) ** 0.5 for n in used])

    sol = least_squares(residuals, p0, loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=600)
    G = solve_h(anchors, sol.x[:8].reshape(4, 2))
    report = dict(
        residual_rms_px={n: float(np.sqrt(np.mean(pixel_residual(sol.x, n) ** 2))) for n in used},
        samples={n: int(len(feats[n])) for n in used},
        paint_dilation_px=float(sol.x[8]),
        line_offsets_m={n: float(sol.x[9 + i]) for i, n in enumerate(lines)},
        line_straightness={n: straightness(feats[n], "x" if n in ("tee", "hog") else "y") for n in lines + ["hog"] if n in feats},
    )
    return G / G[2, 2], report


def calibrate_view(bg, view):
    """Full paint calibration of one panel. Returns (Camera, report, features)."""
    rings, box = ring_edges(bg, view)
    G = initial_model(view, box)
    G, _ = fit(rings, G, view)  # rings alone first: a good seed for following the lines
    for _ in range(3):
        feats = collect(bg, view, G, rings)
        G, report = fit(feats, G, view)
    x0, y0, x1, y1 = view["crop"]
    cam = Camera(view["id"], view["kind"], view["end"], tuple(view["crop"]), G)
    pose = pinhole_pose(G, ((x0 + x1) / 2, (y0 + y1) / 2)) if view["kind"] == "perspective" else None
    if pose:
        cam.ground, cam.height = pose[1][:2], float(pose[1][2])
        report["focal_px"] = pose[0]
    return cam, report, feats


def draw_model(img, cam, colour=(0, 0, 255)):
    """Overlay the fitted sheet model on a mosaic image for visual review."""
    ty, d = tee(cam.end)[1], toward_centre(cam.end)
    th = np.linspace(0, 2 * np.pi, 720)
    curves = [np.c_[r * np.cos(th), ty + r * np.sin(th)] for r in list(RINGS.values()) + [0.5 * 0.3048]]
    reach = 2.2 if cam.kind == "overhead" else 21
    curves.append(np.c_[np.zeros(300), np.linspace(ty - 2.2 * d, ty + reach * d, 300)])
    curves.append(np.c_[np.linspace(-2.3, 2.3, 100), np.full(100, ty)])
    if cam.kind == "perspective":
        for Y in (HOG, -6 * 0.3048, -12 * 0.3048):
            curves.append(np.c_[np.linspace(-2.3, 2.3, 100), np.full(100, ty + Y * d)])
        for X in (-FOUR_FOOT, FOUR_FOOT):
            curves.append(np.c_[np.full(300, X), np.linspace(ty + HOG * d, ty + reach * d, 300)])
    for c in curves:
        p = cam.to_image(c)
        ok = cam.contains(p)
        for a, b, oa, ob in zip(p[:-1], p[1:], ok[:-1], ok[1:]):
            if oa and ob:
                cv2.line(img, tuple(np.int32(np.round(a * 8))), tuple(np.int32(np.round(b * 8))), colour, 1, cv2.LINE_AA, 3)
    return img

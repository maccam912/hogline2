"""Step 1+2: measure lens distortion and fit each camera's ice-plane homography from paint alone.

Input: the temporal-median background mosaic and data/camera-layout.json (initial guess only).
Evidence: sub-pixel edges of the 2/4/6 ft rings, the centre line, tee line (overhead), hog line
and the two faint lines parallel to the centre line (perspective; their lateral offsets are
free parameters, so they constrain direction only).
Output: markings.json with per-view models fitted with k1 = 0 and with k1 free, residuals, and
straight-line / ring-ratio distortion diagnostics; markings-overlay.png for visual review.

Usage: fit_markings.py BACKGROUND.png CAMERA_LAYOUT.json OUTDIR
"""
import json
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

from hxlib import HOG, PANELS, RINGS, TEE_DISTANCE, View, apply_h, contour_points, ridge_minimum, subpixel_edge, tee


def solve_h(src, dst):
    """Exact 4-point homography in float64 (cv2.getPerspectiveTransform works in float32)."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(np.asarray(src, float), np.asarray(dst, float)):
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        rhs += [u, v]
    return np.append(np.linalg.solve(np.array(rows), np.array(rhs)), 1.0).reshape(3, 3)


def initial_model(view, tee_distance):
    """ice->image from the saved layout (fitted perspective mapping or 4 ring landmarks)."""
    cam = view.get("camera_calibration")
    if cam:
        return np.linalg.inv(np.array(cam["image_to_ice"]).reshape(3, 3))
    r = RINGS["r6"]
    y, d = (0.0, 1) if view["end"] == "A" else (tee_distance, -1)
    world = {"left": [-r * d, y], "right": [r * d, y], "back": [0, y - r * d], "front": [0, y + r * d]}
    if view.get("rotation_degrees", 0) == 180:
        world = {k: [-v[0], 2 * y - v[1]] for k, v in world.items()}
    names = ["left", "right", "back", "front"]
    return solve_h([world[n] for n in names], [view["landmarks"][n] for n in names])


def ring_edges(bg, hsv, crop, kind):
    x0, y0, x1, y1 = crop
    inset = 9 if kind == "overhead" else 0
    region = np.zeros(hsv.shape[:2], bool)
    if kind == "overhead":
        region[y0 + inset : y1 - inset, x0 + inset : x1 - inset] = True
    else:
        region[y0 + 430 : y0 + 575, x0:x1] = True  # ice around the far house
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = region & (h > 95) & (h < 125) & (s > 40) & (v > 110)
    green = region & (h > 35) & (h < 85) & (s > (25 if kind == "perspective" else 40)) & (v > 110)
    img = bg.astype(np.float32)
    half = 3.0 if kind == "overhead" else 2.5
    feats = {}
    (outer, holes) = contour_points(blue)
    hole = max(holes, key=lambda c: len(c[0]))
    (g_outer, _) = contour_points(green)
    for name, (pts, nrm) in {"r6": outer, "r4": hole, "r2": g_outer}.items():
        if name == "r4":
            nrm = nrm  # contour normals already point out of the blob (into the hole)
        ref, ok, _ = subpixel_edge(img, pts, nrm, half=half, plateau=0.75)
        keep = ok & (ref[:, 0] > x0 + inset + 3) & (ref[:, 0] < x1 - inset - 3)
        keep &= (ref[:, 1] > y0 + inset + 3) & (ref[:, 1] < y1 - inset - 3)
        feats[name] = ref[keep]
    return feats


def track_line(gray, seed, positions, along, window=4, blur=11, min_depth=1.0, slope=0.0, first_window=10):
    """Follow a thin dark line from a seed, predicting each step from the points found so far.

    `along`='y' walks image rows (line roughly vertical), 'x' walks columns. `positions` must be
    ordered outward from the seed. Walking on its own evidence keeps the tracker on the paint even
    where the initial camera model is centimetres off.
    """
    k = (1, blur) if along == "y" else (blur, 1)
    smooth = cv2.GaussianBlur(gray, k, 0)
    found, out, missed = [], [], 0
    for t in positions:
        if len(found) >= 12:
            recent = np.array(found[-60:])
            lin = np.polyfit(recent[:, 0], recent[:, 1], 1)
            c = float(np.polyval(lin, t))
        elif found:
            c = found[-1][1] + slope * (t - found[-1][0])
        else:
            c = seed
        w = window if found else first_window  # acquire widely, then follow tightly
        lo = int(round(c)) - w
        if along == "y":
            prof = smooth[int(t), max(lo - 12, 0) : lo + 2 * w + 13]
        else:
            prof = smooth[max(lo - 12, 0) : lo + 2 * w + 13, int(t)]
        if len(prof) < 2 * w + 25:
            continue
        base = np.median(np.r_[prof[:10], prof[-10:]])
        core = prof[12 : 12 + 2 * w + 1] - base
        m = ridge_minimum(core)
        if m is None or core.min() > -min_depth:
            missed += 1
            if missed > 150:
                break
            continue
        missed = 0
        pos = lo + m
        found.append([t, pos])
        out.append([pos, t] if along == "y" else [t, pos])
    return np.array(out, float).reshape(-1, 2)


def straightness(pts, along):
    """Max and RMS deviation (px) of tracked points from their own best straight line."""
    if len(pts) < 20:
        return None
    t, v = (pts[:, 1], pts[:, 0]) if along == "y" else (pts[:, 0], pts[:, 1])
    lin = np.polyfit(t, v, 1)
    res = v - np.polyval(lin, t)
    quad = np.polyfit(t - t.mean(), v, 2)
    span = t.max() - t.min()
    return dict(n=len(pts), span_px=float(span), rms_px=float(res.std()), max_px=float(np.abs(res).max()),
                bow_px=float(quad[0] * (span / 2) ** 2))  # sagitta implied by a quadratic fit


def collect(bg, view, G0):
    kind, end, crop = PANELS[view["id"]]
    x0, y0, x1, y1 = crop
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    feats = ring_edges(bg, hsv, crop, kind)
    ty = tee(end)[1]
    d = 1 if end == "A" else -1

    def img_x_at_row(X, row):  # image x where the ice line X=const crosses this image row (initial model)
        a = apply_h(G0, [[X, ty - 6 * d], [X, ty + 25 * d]])
        return a[0, 0] + (row - a[0, 1]) * (a[1, 0] - a[0, 0]) / (a[1, 1] - a[0, 1])

    def img_y_at_col(Y, col):
        a = apply_h(G0, [[-3, Y], [3, Y]])
        return a[0, 1] + (col - a[0, 0]) * (a[1, 1] - a[0, 1]) / (a[1, 0] - a[0, 0])

    b = apply_h(G0, [[0, ty]])[0]
    if kind == "overhead":
        scale = np.linalg.norm(apply_h(G0, [[RINGS["r6"], ty]])[0] - b) / RINGS["r6"]
        halves = []
        for rows in (range(int(b[1]) + 12, y1 - 14), range(int(b[1]) - 12, y0 + 14, -1)):
            halves.append(track_line(gray, img_x_at_row(0.0, rows[0]), rows, "y", window=4, blur=7, min_depth=2.0, first_window=5))
        feats["centre"] = np.vstack(halves)
        halves = []
        for cols in (range(int(b[0]) + 12, x1 - 14), range(int(b[0]) - 12, x0 + 14, -1)):
            halves.append(track_line(gray, img_y_at_col(ty, cols[0]), cols, "x", window=4, blur=7, min_depth=2.0, first_window=5))
        feats["tee"] = np.vstack(halves)
        for name in ("centre", "tee"):  # keep clear of the button logo and the ring outlines
            p = feats[name]
            r = np.linalg.norm(p - b, axis=1) / scale
            ok = r > 0.2
            for ring in RINGS.values():
                ok &= np.abs(r - ring) > 0.06
            feats[name] = p[ok]
    else:
        front = int(apply_h(G0, [[0, ty + (RINGS["r6"] + 0.2) * d]])[0, 1]) + 2
        rows = range(front, y1 - 6)
        feats["centre"] = track_line(gray, img_x_at_row(0.0, front), rows, "y", window=4)
        hog_y = img_y_at_col(ty + HOG * d, b[0])
        halves = []
        for cols in (range(int(b[0]) + 8, x1 - 30), range(int(b[0]) - 8, x0 + 6, -1)):
            halves.append(track_line(gray, hog_y, cols, "x", window=5, blur=9, min_depth=2.5))
        feats["hog"] = np.vstack(halves)
        start = int(hog_y) + 8
        for name, X in (("par_neg", -0.62), ("par_pos", 0.62)):
            seed = img_x_at_row(X, start)
            centre_slope = (img_x_at_row(X, start + 50) - seed) / 50
            feats[name] = track_line(gray, seed, range(start, y1 - 6), "y", window=7, slope=centre_slope)
    return feats


def make_view(params, anchors_ice, centre, R, k1):
    G = solve_h(anchors_ice, params[:8].reshape(4, 2))
    return View(G / G[2, 2], centre, R, k1)


SIGN = {"r6": 1.0, "r4": -1.0, "r2": 1.0}  # coloured paint reads slightly dilated in the video


def fit(feats, G0, kind, end, crop, free_k1):
    """Least-squares camera model. Residuals are Sampson distances in pixels.

    Free: the homography (as the pixel positions of four ice anchors), optionally k1, one paint
    dilation `delta` (px), and a lateral offset per thin line -- so lines constrain direction and
    straightness while the ring centres alone define the tee. The hog line stays at 21 ft.
    """
    x0, y0, x1, y1 = crop
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    R = 0.5 * np.hypot(x1 - x0, y1 - y0)
    ty = tee(end)[1]
    d = 1 if end == "A" else -1
    far = 2.0 if kind == "overhead" else 14.0  # anchors span the evidence
    anchors = np.array([[-2, ty - 2 * d], [2, ty - 2 * d], [2, ty + far * d], [-2, ty + far * d]], float)
    lines = [n for n in ("centre", "tee", "par_neg", "par_pos") if n in feats and len(feats[n]) > 30]
    guess = {"centre": 0.0, "tee": 0.0, "par_neg": -0.62, "par_pos": 0.62}
    p0 = np.array(list(apply_h(G0, anchors).ravel()) + [0.0, 0.5] + [guess[n] for n in lines])
    used = [n for n in feats if (n in RINGS or n == "hog" or n in lines) and len(feats[n])]

    def errors(view, p, name, px):
        q = view.to_ice(px)
        if name in RINGS:
            return np.linalg.norm(q - [0, ty], axis=1) - RINGS[name]
        if name == "hog":
            return q[:, 1] - (ty + HOG * d)
        return q[:, 1 if name == "tee" else 0] - (ty if name == "tee" else 0.0) - p[10 + lines.index(name)]

    def pixel_residual(view, p, name, px):
        e = errors(view, p, name, px)
        h = 0.5  # Sampson distance: metres -> pixels through the local gradient
        gx = (errors(view, p, name, px + [h, 0]) - errors(view, p, name, px - [h, 0])) / (2 * h)
        gy = (errors(view, p, name, px + [0, h]) - errors(view, p, name, px - [0, h])) / (2 * h)
        return e / np.hypot(gx, gy) - SIGN.get(name, 0.0) * p[9], e

    def residuals(p):
        view = make_view(p, anchors, centre, R, p[8] if free_k1 else 0.0)
        # every feature carries equal total weight, however many edge samples it has
        return np.concatenate([pixel_residual(view, p, n, feats[n])[0] * (200.0 / len(feats[n])) ** 0.5 for n in used])

    sol = least_squares(residuals, p0, loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=600)
    view = make_view(sol.x, anchors, centre, R, sol.x[8] if free_k1 else 0.0)
    stats = {}
    for name in used:
        r, e = pixel_residual(view, sol.x, name, feats[name])
        stats[name] = dict(n=int(len(r)), rms_px=float(np.sqrt(np.mean(r**2))), median_abs_px=float(np.median(np.abs(r))),
                           mean_px=float(r.mean()))
    allr = residuals(sol.x)
    return view, dict(features=stats, k1=view.k1, paint_dilation_px=float(sol.x[9]),
                      line_offsets_m={n: float(sol.x[10 + i]) for i, n in enumerate(lines)},
                      weighted_rms=float(np.sqrt(np.mean(allr**2))))


def ring_ratio(feats):
    """Pixel radii of the three rings about their common centre: undistorted => 1 : 2 : 3."""
    out = {}
    for name in RINGS:
        (cx, cy), (a, b), ang = cv2.fitEllipse(np.float32(feats[name]))
        out[name] = dict(centre=[float(cx), float(cy)], semi_axes=[float(a) / 2, float(b) / 2], angle=float(ang))
    m = {n: float(np.mean(out[n]["semi_axes"])) for n in RINGS}
    out["mean_radius_ratio_to_r2"] = {n: m[n] / m["r2"] for n in RINGS}
    return out


def main():
    bg = cv2.imread(sys.argv[1])
    layout = json.load(open(sys.argv[2]))["calibration"]
    outdir = sys.argv[3]
    result = {"tee_distance_m": TEE_DISTANCE, "views": {}, "views_k1_free": {}, "report": {}}
    overlay = bg.copy()
    for view in layout["views"]:
        kind, end, crop = PANELS[view["id"]]
        G0 = initial_model(view, layout.get("tee_distance_m", TEE_DISTANCE))
        feats = collect(bg, view, G0)
        v0, s0 = fit(feats, G0, kind, end, crop, free_k1=False)
        v1, s1 = fit(feats, G0, kind, end, crop, free_k1=True)
        rep = dict(k1_fixed_zero=s0, k1_free=s1, counts={k: int(len(v)) for k, v in feats.items()})
        rep["line_straightness"] = {n: straightness(feats[n], "x" if n in ("tee", "hog") else "y")
                                    for n in feats if n not in RINGS}
        if kind == "overhead":
            rep["ring_ellipses"] = ring_ratio(feats)
        corner = 0.5 * np.hypot(crop[2] - crop[0], crop[3] - crop[1])
        rep["k1_corner_shift_px"] = float(v1.k1 * corner)
        result["views"][view["id"]] = v0.as_dict()
        result["views_k1_free"][view["id"]] = v1.as_dict()
        result["report"][view["id"]] = rep
        np.save(f"{outdir}/features-{view['id']}.npy", {k: v for k, v in feats.items()}, allow_pickle=True)
        # overlay: detected evidence (yellow) and the fitted model (red)
        for px in feats.values():
            for x, y in px[::3]:
                cv2.circle(overlay, (int(round(x)), int(round(y))), 1, (0, 255, 255), -1)
        ty = tee(end)[1]
        d = 1 if end == "A" else -1
        th = np.linspace(0, 2 * np.pi, 720)
        curves = [np.c_[r * np.cos(th), ty + r * np.sin(th)] for r in list(RINGS.values()) + [0.5 * 0.3048]]
        ylo, yhi = (ty - 2.2 * d, ty + (2.2 if kind == "overhead" else 21) * d)
        curves.append(np.c_[np.zeros(200), np.linspace(ylo, yhi, 200)])
        curves.append(np.c_[np.linspace(-2.4, 2.4, 100), np.full(100, ty)])
        if kind == "perspective":
            for Y in (HOG, -1.829, -3.658):
                curves.append(np.c_[np.linspace(-2.4, 2.4, 100), np.full(100, ty + Y * d)])
        x0, y0, x1, y1 = crop
        for c in curves:
            p = v0.to_image(c)
            ok = (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)
            for a, b, oa, ob in zip(p[:-1], p[1:], ok[:-1], ok[1:]):
                if oa and ob:
                    cv2.line(overlay, tuple(np.int32(np.round(a * 8))), tuple(np.int32(np.round(b * 8))), (0, 0, 255), 1,
                             cv2.LINE_AA, 3)
    json.dump(result, open(f"{outdir}/markings.json", "w"), indent=1)
    cv2.imwrite(f"{outdir}/markings-overlay.png", overlay)
    for vid, rep in result["report"].items():
        print("==", vid, rep["counts"])
        for tag in ("k1_fixed_zero", "k1_free"):
            s = rep[tag]
            print("  ", tag, "k1=%.5f" % s["k1"], "delta=%.2f" % s["paint_dilation_px"], "wrms=%.3f" % s["weighted_rms"],
                  {k: round(v["rms_px"], 2) for k, v in s["features"].items()},
                  {k: round(v, 4) for k, v in s["line_offsets_m"].items()})
        print("   k1 corner shift px: %.2f" % rep["k1_corner_shift_px"])
        for n, st in rep["line_straightness"].items():
            if st:
                print("   line", n, {k: round(v, 3) for k, v in st.items()})
        if "ring_ellipses" in rep:
            print("   rings", {n: [round(a, 2) for a in rep["ring_ellipses"][n]["semi_axes"]] for n in RINGS},
                  rep["ring_ellipses"]["mean_radius_ratio_to_r2"])


if __name__ == "__main__":
    main()

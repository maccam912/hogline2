"""Conservative multi-view geometry audit using simultaneous tracked stones.

Never promotes a venue length automatically unless independent tracks agree.
Overhead ring calibration supplies the metric scale; perspective landmarks and
stone correspondences constrain the two perspective homographies and tee spacing.
"""

import itertools
import math

import numpy as np
from scipy.optimize import least_squares

from .tracking_dimensions import TEE_DISTANCE, sheet_geometry
from .tracking_geometry import prepare_view, world_points


def project_many(h, points):
    p = np.column_stack((points, np.ones(len(points)))) @ h.T
    denominator = p[:, 2:3]
    denominator = np.where(abs(denominator) < 1e-5, np.copysign(1e-5, denominator), denominator)
    return p[:, :2] / denominator


def fit_geometry(rows, calibration):
    nominal = calibration.get("tee_distance_m", TEE_DISTANCE)
    fixed = calibration.get("tee_distance_fixed", False)
    geometry = sheet_geometry(calibration)
    # Old in-flight SAM processes retain their imported geometry module and snapshot.
    # Only new snapshots carry configurable dimensions and need the expanded API.
    configured = "sheet_geometry" in calibration
    views = {
        v["id"]: prepare_view(v, nominal, geometry) if configured else prepare_view(v, nominal)
        for v in calibration["views"]
    }
    transforms, matrices = {}, {}
    for vid, v in views.items():
        bx, by = v["landmarks"]["button"]
        scale = math.dist(v["landmarks"]["left"], v["landmarks"]["right"])
        transform = np.array([[scale, 0, bx], [0, scale, by], [0, 0, 1.0]])
        h = np.array(v["image_to_ice"]).reshape(3, 3)
        if v["end"] == "B":
            h = np.array([[-1, 0, 0], [0, -1, nominal], [0, 0, 1]]) @ h
        h = h @ transform
        h /= h[2, 2]
        transforms[vid], matrices[vid] = transform, h
    perspective = [vid for vid, v in views.items() if v["kind"] == "perspective"]
    pair_data = []
    last = {}
    verified_bodies = any(
        o.get("measurement_kind") == "visible_granite_bottom" for row in rows for o in row["observations"]
    )
    for row in rows:
        grouped = {}
        for o in row["observations"]:
            if (
                o.get("accepted") is False
                or o.get("quality", 0) < 0.5
                or o.get("ground_uncertain")
                or "point" not in o
            ):
                continue
            v = views[o["view_id"]]
            if (
                verified_bodies
                and v["kind"] == "perspective"
                and o.get("measurement_kind") != "visible_granite_bottom"
            ):
                continue
            box = o.get("bbox")
            w = v["crop"][2] - v["crop"][0]
            h = v["crop"][3] - v["crop"][1]
            if box and (box[0] <= 1 or box[1] <= 1 or box[2] >= w - 1 or box[3] >= h - 1):
                continue
            grouped.setdefault(o["stone_id"], []).append(o)
        for name, observations in grouped.items():
            for a, b in itertools.combinations(observations, 2):
                if a["view_id"] == b["view_id"]:
                    continue
                key = (name, a["view_id"], b["view_id"])
                if key in last and math.dist(a["xy"], last[key]) < 0.12:
                    continue
                last[key] = a["xy"]
                pair_data.append((name, a["view_id"], b["view_id"], a["point"], b["point"]))
    names = sorted({p[0] for p in pair_data})
    if not perspective or len(names) < 2 or len(pair_data) < 12:
        return dict(
            status="insufficient_evidence",
            tee_distance_m=nominal,
            pairs=len(pair_data),
            tracks=names,
            length_mode="fixed_baseline" if fixed else "estimated",
            dimension_source=calibration.get("tee_distance_source", "nominal"),
        )
    grouped_pairs = {}
    for name, a, b, pa, pb in pair_data:
        grouped_pairs.setdefault((a, b), []).append((name, pa, pb))
    # Normalize image coordinates around each button for numerical conditioning.
    pair_arrays = []
    for (a, b), points in grouped_pairs.items():
        pa = project_many(np.linalg.inv(transforms[a]), np.array([p[1] for p in points]))
        pb = project_many(np.linalg.inv(transforms[b]), np.array([p[2] for p in points]))
        pair_arrays.append((a, b, pa, pb, np.array([p[0] for p in points])))
    initial = np.concatenate([matrices[v].ravel()[:8] for v in perspective] + ([] if fixed else [[nominal]]))
    low = np.full(len(initial), -100.0)
    high = np.full(len(initial), 100.0)
    if not fixed:
        low[-1], high[-1] = 25.0, 45.0
    for i in range(len(perspective)):
        low[i * 8 + 6 : i * 8 + 8] = -4
        high[i * 8 + 6 : i * 8 + 8] = 4

    def decode(values):
        result = matrices.copy()
        for i, vid in enumerate(perspective):
            result[vid] = np.r_[values[i * 8 : i * 8 + 8], 1].reshape(3, 3)
        return result, nominal if fixed else float(values[-1])

    def to_world(vid, xy, length):
        return np.column_stack((-xy[:, 0], length - xy[:, 1])) if views[vid]["end"] == "B" else xy

    def residual(values, omit=None, details=False):
        hs, length = decode(values)
        parts = []
        landmark_errors = []
        pair_errors = []
        for vid in perspective:
            v = views[vid]
            expected = world_points("A", geometry=geometry) if configured else world_points("A")
            rotation = v.get("rotation_degrees", 0)
            labels = [k for k in ["left", "right", "back", "front", "button", "hack"] if k in v["landmarks"]]
            target = np.array([expected[k] for k in labels])
            if rotation == 180:
                target = -target
            try:
                pred = project_many(transforms[vid] @ np.linalg.inv(hs[vid]), target)
            except np.linalg.LinAlgError:
                return np.full(residual_size, 1e6)
            actual = np.array([v["landmarks"][k] for k in labels])
            error = pred - actual
            landmark_errors.extend(np.linalg.norm(error, axis=1).tolist())
            sigma = np.array([4.0 if k == "hack" else 2.5 for k in labels])[:, None]
            parts.extend((error / sigma).ravel())
        for a, b, pa, pb, ns in pair_arrays:
            use = ns != omit if omit else np.ones(len(ns), bool)
            aa = project_many(hs[a], pa[use])
            bb = project_many(hs[b], pb[use])
            error = to_world(a, aa, length) - to_world(b, bb, length)
            pair_errors.extend(np.linalg.norm(error, axis=1).tolist())
            parts.extend((error / 0.20).ravel())
        # Weak nominal prior: this regularizes an underdetermined fit, not a measurement.
        if not fixed:
            parts.append((length - nominal) / 5.0)
        if details:
            return landmark_errors, pair_errors
        return np.array(parts)

    residual_size = (
        2
        * sum(
            len(
                [
                    k
                    for k in ["left", "right", "back", "front", "button", "hack"]
                    if k in views[v]["landmarks"]
                ]
            )
            for v in perspective
        )
        + 2 * len(pair_data)
        + (0 if fixed else 1)
    )
    result = least_squares(residual, initial, bounds=(low, high), loss="soft_l1", max_nfev=500, x_scale="jac")
    hs, length = decode(result.x)
    landmarks, pairs = residual(result.x, details=True)
    leave_out = []

    def pair_distances(values, name=None):
        fitted, distance = decode(values)
        errors = []
        for a, b, pa, pb, ns in pair_arrays:
            use = ns == name if name else np.ones(len(ns), bool)
            aa = to_world(a, project_many(fitted[a], pa[use]), distance)
            bb = to_world(b, project_many(fitted[b], pb[use]), distance)
            errors.extend(np.linalg.norm(aa - bb, axis=1).tolist())
        return errors

    baseline_pairs = pair_distances(initial)
    if len(names) >= 3:
        for name in names:
            fit = least_squares(
                lambda x, omitted=name: residual(x, omitted),
                initial,
                bounds=(low, high),
                loss="soft_l1",
                max_nfev=250,
                x_scale="jac",
            )
            held = pair_distances(fit.x, name)
            baseline = pair_distances(initial, name)
            leave_out.append(
                dict(
                    omitted=name,
                    tee_distance_m=decode(fit.x)[1],
                    success=bool(fit.success),
                    held_out_pairs=len(held),
                    held_out_rms_m=float(np.sqrt(np.mean(np.square(held)))),
                    baseline_rms_m=float(np.sqrt(np.mean(np.square(baseline)))),
                )
            )
    spread = max((abs(e["tee_distance_m"] - length) for e in leave_out), default=float("inf"))
    rms_land = float(np.sqrt(np.mean(np.square(landmarks))))
    rms_pair = float(np.sqrt(np.mean(np.square(pairs))))
    held_out_rms = (
        float(np.sqrt(np.mean([f["held_out_rms_m"] ** 2 for f in leave_out]))) if leave_out else None
    )
    baseline_rms = float(np.sqrt(np.mean(np.square(baseline_pairs))))
    approved = bool(
        result.success
        and len(names) >= 3
        and leave_out
        and spread < 0.30
        and rms_land < 3
        and rms_pair < 0.30
        and held_out_rms is not None
        and held_out_rms < 0.30
        and all(
            f["success"] and f["held_out_rms_m"] <= max(0.10, f["baseline_rms_m"] * 1.1) for f in leave_out
        )
        and 25.5 < length < 44.5
    )
    world_h = {}
    for vid, h in hs.items():
        h = h @ np.linalg.inv(transforms[vid])
        if views[vid]["end"] == "B":
            h = np.array([[-1, 0, 0], [0, -1, length], [0, 0, 1]]) @ h
        h /= h[2, 2]
        world_h[vid] = h.ravel().tolist()
    return dict(
        status=("camera_consistent" if fixed else "candidate_consistent") if approved else "needs_review",
        length_mode="fixed_baseline" if fixed else "estimated",
        dimension_source=calibration.get("tee_distance_source", "nominal"),
        tee_distance_m=length,
        nominal_tee_distance_m=nominal,
        pairs=len(pair_data),
        tracks=names,
        landmark_rms_px=rms_land,
        pair_rms_m=rms_pair,
        baseline_pair_rms_m=baseline_rms,
        held_out_pair_rms_m=held_out_rms,
        correction_recommended=bool(approved and held_out_rms < baseline_rms * 0.8),
        measurement_basis="verified_granite_bottom" if verified_bodies else "legacy_mask_ground_estimates",
        leave_one_track_out=leave_out,
        length_sensitivity_m=spread if not fixed and math.isfinite(spread) else None,
        solver_success=bool(result.success),
        image_to_ice=world_h,
        notes=[
            "Estimate uses overhead ring scale and synchronized observations, not a surveyed measurement.",
            "Camera time offsets and lens distortion are not independently estimated.",
            "Visible base rims in opposing views are not the same physical point; "
            "residuals can include stone-radius bias.",
            "Length is fixed by the selected venue baseline; residuals assess camera geometry."
            if fixed
            else "Do not freeze an estimated venue constant while status is needs_review.",
        ],
    )

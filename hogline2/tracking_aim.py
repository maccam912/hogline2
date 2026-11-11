"""Local hog-line heading estimates. Never use filtered/future trajectory states."""

import hashlib
import json
import math
from statistics import median

from .tracking_dimensions import sheet_geometry
from .tracking_geometry import prepare_view, project, solve

METHOD = "local_hog_quadratic_v1"


def ray_x(ray, y):
    return ray["origin"][0] + ray["slope_dx_dy"] * (y - ray["origin"][1])


def ray_band(ray, y):
    d = y - ray["origin"][1]
    c = ray["coefficient_covariance"]
    return 2 * math.sqrt(max(0, c[0][0] + 2 * d * c[0][1] + d * d * c[1][1]))


def fit_heading(samples, hog_y, direction, view):
    """Quadratic x(y); its derivative at the hog is the frozen tangent.

    A two-pixel image-localization model and observed residual scatter set the
    sensitivity band. Systematic calibration/stone-height error is NOT covered.
    """
    samples = sorted(samples, key=lambda s: s["seconds"])
    pairs = list(zip(samples, samples[1:], strict=False))
    crossing = next(
        (
            (a, b)
            for a, b in pairs
            if direction * (a["xy"][1] - hog_y) <= 0 < direction * (b["xy"][1] - hog_y)
        ),
        None,
    )
    if not crossing:
        raise ValueError("No observed forward hog-line crossing")
    a, b = crossing
    t = a["seconds"] + (b["seconds"] - a["seconds"]) * ((hog_y - a["xy"][1]) / (b["xy"][1] - a["xy"][1]))
    local = [s for s in samples if abs(s["seconds"] - t) <= 1.6 and abs(s["xy"][1] - hog_y) <= 2]
    before = [s for s in local if direction * (s["xy"][1] - hog_y) < 0]
    after = [s for s in local if direction * (s["xy"][1] - hog_y) >= 0]
    if len(local) < 5 or len(before) < 2 or len(after) < 2:
        raise ValueError("Need at least five local observations, including two on each side of the hog line")
    gaps = [b["seconds"] - a["seconds"] for a, b in zip(local, local[1:], strict=False)]
    if max(gaps) > 0.75:
        raise ValueError("Observation gap near the hog line exceeds 0.75 seconds")
    if any(direction * (b["xy"][1] - a["xy"][1]) <= 0 for a, b in zip(local, local[1:], strict=False)):
        raise ValueError("Local observations reverse direction; check the stone identity")
    span = abs(local[-1]["xy"][1] - local[0]["xy"][1])
    if span < 1.5:
        raise ValueError("Less than 1.5 metres of observed travel near the hog line")
    features = [[1.0, s["xy"][1] - hog_y, (s["xy"][1] - hog_y) ** 2] for s in local]
    xs = [s["xy"][0] for s in local]
    # Propagate pixel jitter into lateral residuals, including longitudinal jitter.
    slope = (xs[-1] - xs[0]) / (local[-1]["xy"][1] - local[0]["xy"][1])
    variances = []
    for s in local:
        p = s["point"]
        xy = project(view["image_to_ice"], p)
        shifts = [
            project(view["image_to_ice"], [p[0] + 1, p[1]]),
            project(view["image_to_ice"], [p[0], p[1] + 1]),
        ]
        variances.append(
            max(0.003**2, 4 * sum(((q[0] - xy[0]) - slope * (q[1] - xy[1])) ** 2 for q in shifts))
        )
    normal = [
        [sum(f[i] * f[j] / v for f, v in zip(features, variances, strict=False)) for j in range(3)]
        for i in range(3)
    ]
    rhs = [sum(f[i] * x / v for f, x, v in zip(features, xs, variances, strict=False)) for i in range(3)]
    coef = solve(normal, rhs)
    residuals = [
        x - sum(a * b for a, b in zip(f, coef, strict=False)) for f, x in zip(features, xs, strict=False)
    ]
    rms = math.sqrt(sum(r * r for r in residuals) / len(local))
    if max(map(abs, residuals)) > 0.12 or rms > 0.06:
        raise ValueError("Local mask positions are too inconsistent to estimate aim")
    inflation = max(
        1.0, sum(r * r / v for r, v in zip(residuals, variances, strict=False)) / (len(local) - 3)
    )
    columns = [solve(normal, [float(i == j) for i in range(3)]) for j in range(3)]
    cov = [[columns[j][i] * inflation for j in range(2)] for i in range(2)]
    return dict(
        origin=[coef[0], hog_y],
        slope_dx_dy=coef[1],
        direction=direction,
        crossing_seconds=t,
        crossing_bracket=[a["seconds"], b["seconds"]],
        coefficient_covariance=cov,
        residual_rms_m=rms,
        span_m=span,
        sample_count=len(local),
        sample_rate_hz=1 / median(gaps),
        samples=local,
        angle_degrees=math.degrees(math.atan(coef[1] * direction)),
    )


def pair_headings(samples, direction):
    """Keep unsmoothed same-camera consecutive-frame differences, including jitter."""
    result = []
    for a, b in zip(samples, samples[1:], strict=False):
        dt = b["seconds"] - a["seconds"]
        dy = b["xy"][1] - a["xy"][1]
        if b["frame"] != a["frame"] + 1 or not 0 < dt <= 0.75 or direction * dy <= 0.01:
            continue
        result.append(
            dict(
                origin=b["xy"],
                slope_dx_dy=(b["xy"][0] - a["xy"][0]) / dy,
                direction=direction,
                seconds=b["seconds"],
                previous_seconds=a["seconds"],
                frame=b["frame"],
                point=b["point"],
                previous_point=a["point"],
                dt=dt,
            )
        )
    return result


def build_report(manifest, rows, events, fingerprint):
    calibration = manifest["calibration"]
    g = sheet_geometry(calibration)
    tee = calibration.get("tee_distance_m", 34.747)
    views = [prepare_view(v, tee, g) for v in calibration["views"]]
    delivery = next(v for v in views if v["id"] == manifest["delivery_view"])
    direction = 1 if delivery["end"] == "A" else -1
    far = "B" if direction == 1 else "A"
    hog = g["hog_from_tee_m"] if direction == 1 else tee - g["hog_from_tee_m"]
    groups = {}
    for row in rows:
        for o in row.get("observations", []):
            if (
                o.get("view_id") != delivery["id"]
                or o.get("accepted") is False
                or o.get("clipped")
                or o.get("ground_uncertain")
                or not o.get("point")
            ):
                continue
            if o.get("measurement_kind") not in {"visible_granite_bottom", "overhead_center"}:
                continue
            if not all(math.isfinite(v) for v in [*o["xy"], *o["point"], row["source_seconds"]]):
                continue
            groups.setdefault(o["stone_id"], []).append(
                dict(seconds=row["source_seconds"], xy=o["xy"], point=o["point"], frame=row["frame"])
            )
    rays = []
    for stone in manifest.get("stones", []):
        name = stone["id"]
        result = dict(stone_id=name, status="unavailable")
        try:
            samples = groups.get(name, [])
            contact = [
                e["seconds"]
                for e in events
                if "contact" in e["type"] and (e.get("stone") == name or name in e.get("stones", []))
            ]
            if contact:
                samples = [s for s in samples if s["seconds"] < min(contact)]
            result["pair_rays"] = pair_headings(samples, direction)
            ray = fit_heading(samples, hog, direction, delivery)
            ray["far_tee_x"] = ray_x(ray, tee if direction == 1 else 0)
            ray["far_tee_band_m"] = ray_band(ray, tee if direction == 1 else 0)
            result.update(ray, status="estimated")
        except ValueError as exc:
            result["reason"] = str(exc)
        result["reference_seconds"] = result.get(
            "crossing_seconds",
            min(
                groups.get(name, []),
                key=lambda s: abs(s["xy"][1] - hog),
                default={"seconds": manifest["source_start"]},
            )["seconds"],
        )
        if result.get("pair_rays") or result["status"] == "estimated":
            result["evidence_sha256"] = hashlib.sha256(
                json.dumps([fingerprint, name, result], sort_keys=True).encode()
            ).hexdigest()
        rays.append(result)
    return dict(
        method=METHOD,
        source_sha256=manifest["source_sha256"],
        input_sha256=fingerprint,
        delivery_view=delivery["id"],
        delivery_end=delivery["end"],
        receiving_end=far,
        receiving_views=[v["id"] for v in views if v["kind"] == "overhead" and v["end"] == far],
        views=views,
        rays=rays,
        tee_distance_m=tee,
    )


def compare_target(ray, view, point):
    target = project(view["image_to_ice"], point)
    if ray["direction"] * (target[1] - ray["origin"][1]) <= 0:
        raise ValueError("Broom must be ahead of the delivery hog line")
    # In this project's coordinates, screen-right looking A->B is world -x.
    offset = -ray["direction"] * (ray_x(ray, target[1]) - target[0])
    return dict(ice_point=target, offset_right_m=offset, band_m=ray_band(ray, target[1]))

"""Whole-clip RTS smoothing and constrained free-glide trajectories.

Raw masks/forward estimates are immutable. Contacts split motion episodes;
future overhead observations anchor placement without smoothing through impacts.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .tracking_calibration_fit import fit_geometry
from .tracking_state import measurement_variance

VERSION = "rts_constrained_glide_v2"


def rts_smooth(times, measurements, variances, acceleration=0.25):
    """Linear-Gaussian forward filter followed by Rauch–Tung–Striebel recursion."""
    n = len(times)
    h = np.array([[1, 0, 0, 0], [0, 1, 0, 0.0]])
    first = next((i for i, z in enumerate(measurements) if z is not None), None)
    if first is None:
        raise ValueError("No observations to smooth")
    x = np.r_[measurements[first], 0.0, 0.0]
    p = np.diag([1.0, 1.0, 9.0, 9.0])
    posterior = []
    covariance = []
    predicted = []
    prior = []
    transitions = []
    for i, t in enumerate(times):
        dt = 0 if i == 0 else t - times[i - 1]
        f = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        x = f @ x
        p = f @ p @ f.T + acceleration**2 * g @ g.T + np.eye(4) * 1e-9
        predicted.append(x.copy())
        prior.append(p.copy())
        transitions.append(f)
        if measurements[i] is not None:
            r = np.diag(variances[i])
            innovation = np.array(measurements[i]) - h @ x
            k = np.linalg.solve(h @ p @ h.T + r, h @ p).T
            x = x + k @ innovation
            a = np.eye(4) - k @ h
            p = a @ p @ a.T + k @ r @ k.T
        posterior.append(x.copy())
        covariance.append(p.copy())
    smooth = np.array(posterior)
    cov = np.array(covariance)
    for i in range(n - 2, -1, -1):
        gain = np.linalg.solve(prior[i + 1], transitions[i + 1] @ covariance[i]).T
        smooth[i] = posterior[i] + gain @ (smooth[i + 1] - predicted[i + 1])
        cov[i] = covariance[i] + gain @ (cov[i + 1] - prior[i + 1]) @ gain.T
        cov[i] = (cov[i] + cov[i].T) / 2
    return smooth, cov, np.array(posterior), np.array(covariance)


def fused_measurement(observations):
    usable = [
        o
        for o in observations
        if o.get("quality", 1) >= 0
        and o.get("accepted", True)
        and not o.get("rejection")
    ]
    overhead = [o for o in usable if o["kind"] == "overhead" and o.get("quality", 1) >= 0.5]
    if overhead:
        usable = overhead
    if not usable:
        return None, None, []
    xy = np.array([o["xy"] for o in usable])
    variance = np.array([measurement_variance(o) for o in usable])
    # Perspective calibration errors are correlated/systematic, not independent pixel noise.
    for i, o in enumerate(usable):
        if o["kind"] == "perspective":
            variance[i] = np.maximum(variance[i], [0.20**2, 0.45**2])
    weights = 1 / variance
    mean = (weights * xy).sum(0) / weights.sum(0)
    scatter = (weights * (xy - mean) ** 2).sum(0) / weights.sum(0)
    return mean, 1 / weights.sum(0) + scatter, [o["view_id"] for o in usable]


def arc_curve(parameters, times, knots):
    """Cubic spatial curve traversed by a non-increasing, continuous speed profile."""
    start = parameters[:2]
    end = parameters[2:4]
    a, b = parameters[4:6]
    delta = end - start
    length = max(np.linalg.norm(delta), 1e-5)
    normal = np.array([-delta[1], delta[0]]) / length
    u = np.linspace(0, 1, 801)
    lateral = a * u * (1 - u) + b * u * (1 - u) * (2 * u - 1)
    path = start + u[:, None] * delta + lateral[:, None] * normal
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    distance = np.r_[0, np.cumsum(segments)]
    total = max(distance[-1], 1e-6)
    profile = np.r_[1.0, parameters[6:]]
    # Analytic integral of piecewise-linear speed; inverse arclength supplies position.
    widths = np.diff(knots)
    cumulative = np.r_[0, np.cumsum((profile[:-1] + profile[1:]) * widths / 2)]
    t = np.clip(times - times[0], 0, knots[-1])
    idx = np.minimum(np.searchsorted(knots, t, side="right") - 1, len(knots) - 2)
    idx = np.maximum(idx, 0)
    dt = t - knots[idx]
    slope = (profile[idx + 1] - profile[idx]) / widths[idx]
    integral = cumulative[idx] + profile[idx] * dt + slope * dt * dt / 2
    normalizer = max(cumulative[-1], 1e-8)
    progress = np.interp(integral / normalizer * total, distance, u)
    lateral = a * progress * (1 - progress) + b * progress * (1 - progress) * (2 * progress - 1)
    position = start + progress[:, None] * delta + lateral[:, None] * normal
    derivative = (
        delta + (a * (1 - 2 * progress) + b * (-6 * progress**2 + 6 * progress - 1))[:, None] * normal
    )
    tangent = derivative / np.maximum(np.linalg.norm(derivative, axis=1)[:, None], 1e-8)
    speed = (profile[idx] + slope * dt) * total / normalizer
    velocity = tangent * speed[:, None]
    curvature = (
        np.abs(-2 * a + b * (6 - 12 * u))
        * length
        / (length**2 + (a * (1 - 2 * u) + b * (-6 * u * u + 6 * u - 1)) ** 2) ** 1.5
    )
    return position, velocity, speed, total, float(curvature.max())


def fit_glide(times, measurements, variances, seed, start_anchor=None, end_anchor=None, stopping=False):
    times = np.array(times)
    duration = times[-1] - times[0]
    if duration <= 0:
        raise ValueError("Motion interval has no duration")
    count = min(12, max(4, int(duration / 2.5) + 2))
    knots = np.linspace(0, duration, count)
    first = np.array(start_anchor if start_anchor is not None else seed[0, :2])
    last = np.array(end_anchor if end_anchor is not None else seed[-1, :2])
    distance = max(np.linalg.norm(last - first), 0.01)
    profile = np.linspace(1, 0 if stopping else 0.35, count)
    initial = np.r_[first, last, 0.0, 0.0, profile[1:]]
    bounds = []
    for point, anchor in [(first, start_anchor), (last, end_anchor)]:
        bounds.extend(
            [(float(v), float(v)) if anchor is not None else (float(v - 0.5), float(v + 0.5)) for v in point]
        )
    bounds.extend(
        [
            (-min(8, distance * 0.4), min(8, distance * 0.4)),
            (-min(3, distance * 0.12), min(3, distance * 0.12)),
        ]
    )
    bounds.extend([(0, 1)] * (count - 1))
    if stopping:
        bounds[-1] = (0, 0)
    indices = np.array([i for i, z in enumerate(measurements) if z is not None])
    xy = np.array([measurements[i] for i in indices])
    variance = np.array([variances[i] for i in indices])

    def objective(p):
        pos, _, _, _, _ = arc_curve(p, times, knots)
        residual = (pos[indices] - xy) / np.sqrt(variance)
        # Robust errors keep a disagreeing camera from forcing a kink into the path.
        data = np.sum(2 * (np.sqrt(1 + residual**2) - 1))
        v = np.r_[1.0, p[6:]]
        smooth_speed = 100 * np.sum(np.diff(v, n=2) ** 2)
        gentle_curve = 5 * (p[4] / distance) ** 2 + 60 * (p[5] / distance) ** 2
        return float(data + smooth_speed + gentle_curve)

    def constraints(p):
        _, _, speed, _, curvature = arc_curve(p, times, knots)
        return np.r_[-np.diff(np.r_[1.0, p[6:]]), 6 - speed[0], 0.25 - curvature]

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": constraints}],
        options={"maxiter": 400, "ftol": 1e-7},
    )
    feasible = bool(np.min(constraints(result.x)) >= -1e-5)
    if not result.success or not feasible:
        return dict(
            success=False,
            message=str(result.message),
            positions=seed[:, :2],
            velocities=seed[:, 2:],
            speeds=np.linalg.norm(seed[:, 2:], axis=1),
        )
    parameters = result.x.copy()
    parameters[6:] = np.minimum.accumulate(np.clip(parameters[6:], 0, 1))
    pos, vel, speed, length, curvature = arc_curve(parameters, times, knots)
    residual = np.linalg.norm(pos[indices] - xy, axis=1)
    rms = float(np.sqrt(np.mean(residual**2)))
    # Numerical convergence does not establish that a free-glide model fits the
    # evidence (for example, a player may push a short stone out of play).
    if rms > 0.75 or float(residual.max()) > 2.0:
        return dict(
            success=False,
            message="Free-glide curve disagrees with observations",
            rms_residual_m=rms,
            positions=seed[:, :2],
            velocities=seed[:, 2:],
            speeds=np.linalg.norm(seed[:, 2:], axis=1),
        )
    return dict(
        success=True,
        positions=pos,
        velocities=vel,
        speeds=speed,
        length_m=length,
        max_curvature_per_m=curvature,
        rms_residual_m=rms,
        max_speed_increase_mps=float(max(0, np.diff(speed).max(initial=0))),
        parameters=parameters.tolist(),
        knots_seconds=knots.tolist(),
    )


def refine_contacts(rows, events):
    times = np.array([r["source_seconds"] for r in rows])
    result = []
    for event in events:
        if event["type"] != "possible_contact":
            continue
        t = event["seconds"]
        i = max(0, int(np.searchsorted(times, t) - 1))
        refined = t
        evidence = "proximity_only"
        for name in event["stones"]:
            state = next((s for s in rows[i]["stones"] if s["id"] == name), None)
            if not state or state["state"] != "stationary":
                continue
            base = np.array(state["position"])
            for row in rows[i + 1 :]:
                if row["source_seconds"] > t + 2:
                    break
                observations = [
                    o
                    for o in row["observations"]
                    if o["stone_id"] == name and o["kind"] == "overhead" and o.get("quality", 0) > 0.5
                ]
                if (
                    observations
                    and min(np.linalg.norm(np.array(o["xy"]) - base) for o in observations) > 0.12
                ):
                    refined = row["source_seconds"]
                    evidence = "resting_stone_displacement_in_overhead"
                    break
            if evidence != "proximity_only":
                break
        result.append({**event, "original_seconds": t, "seconds": refined, "timing_evidence": evidence})
    return result


def smooth_rows(rows, manifest, events):
    rows = copy.deepcopy(rows)
    views = {v["id"]: v for v in manifest.get("calibration", {}).get("views", [])}
    for row in rows:
        for observation in row["observations"]:
            view = views.get(observation.get("view_id"))
            box = observation.get("bbox")
            if view and box:
                width, height = view["crop"][2] - view["crop"][0], view["crop"][3] - view["crop"][1]
                if box[3] >= height or (
                    view["kind"] == "overhead" and (box[0] <= 0 or box[1] <= 0 or box[2] >= width)
                ):
                    observation["rejection"] = "clipped_contact_point"
    output = copy.deepcopy(rows)
    summary = {
        "model": VERSION,
        "segments": [],
        "house_anchors": [],
        "contact_anchors": [],
        "contacts": refine_contacts(rows, events),
        "warnings": [],
    }
    times = np.array([r["source_seconds"] for r in rows])
    names = sorted({s["id"] for r in rows for s in r["stones"]})
    for row in output:
        row.pop("observations", None)
        row.pop("masks", None)
        for s in row["stones"]:
            s["forward_position"] = s["position"]
            s["trajectory_phase"] = "unresolved"
            s["smoothed"] = False
    for name in names:
        lookup = {
            i: next(s for s in row["stones"] if s["id"] == name)
            for i, row in enumerate(rows)
            if any(s["id"] == name for s in row["stones"])
        }
        begin = min(lookup)
        boundaries = [begin]
        boundaries.extend(
            int(np.searchsorted(times, e["seconds"]))
            for e in summary["contacts"]
            if name in e["stones"] and begin < np.searchsorted(times, e["seconds"]) < len(rows) - 1
        )
        boundaries = sorted(set(boundaries + [len(rows) - 1]))
        contact_boundaries = set(boundaries[1:-1])
        boundary_anchor = None
        boundary_variance = None
        observations = {i: [o for o in rows[i]["observations"] if o["stone_id"] == name] for i in lookup}
        measurements = {i: fused_measurement(observations[i]) for i in lookup}
        intervals = []
        for left, right in zip(boundaries, boundaries[1:], strict=False):
            # Lost tracks are not a single free-flight episode through the rest
            # of the end. Keep unobserved tails as forward predicted/lost state;
            # they must never supply the endpoint of a backwards-smoothed fit.
            cursor = left
            while cursor <= right:
                if lookup[cursor]["state"] == "lost":
                    cursor += 1
                    continue
                start = cursor
                while cursor < right and lookup[cursor + 1]["state"] != "lost":
                    cursor += 1
                supported = [i for i in range(start, cursor + 1) if measurements[i][0] is not None]
                if supported:
                    end = cursor if lookup[cursor]["state"] == "stationary" else supported[-1]
                    intervals.append((supported[0], end))
                cursor += 1
        previous_right = None
        for left, right in intervals:
            if left != previous_right:
                boundary_anchor = boundary_variance = None
            previous_right = right
            indices = list(range(left, right + 1))
            stop = next((i for i in indices if i > left and lookup[i]["state"] == "stationary"), right)
            # Don't infer a new stop immediately after contact from the previous stationary flag.
            if left != begin:
                moving = next((i for i in indices if lookup[i]["state"] == "moving"), None)
                stop = next(
                    (
                        i
                        for i in indices
                        if moving is not None and i > moving and lookup[i]["state"] == "stationary"
                    ),
                    right,
                )
            stationary = stop < right or lookup[right]["state"] == "stationary"
            anchor = None
            anchor_variance = None
            if stationary:
                candidates = [
                    o
                    for i in range(stop, right + 1)
                    for o in observations[i]
                    if o["kind"] == "overhead" and o.get("quality", 0) > 0.5 and not o.get("rejection")
                ]
                kind = "overhead"
                if not candidates:
                    candidates = [
                        o for i in range(stop, right + 1) for o in observations[i] if o.get("accepted")
                    ]
                    kind = "perspective"
                if candidates:
                    points = np.array([o["xy"] for o in candidates])
                    anchor = np.median(points, axis=0)
                    scatter = np.median((points - anchor) ** 2, axis=0)
                    anchor_variance = np.maximum(scatter, 0.05**2 if kind == "overhead" else 0.25**2)
                    summary["house_anchors"].append(
                        dict(
                            stone=name,
                            from_seconds=float(times[stop]),
                            to_seconds=float(times[right]),
                            position=anchor.tolist(),
                            variance=anchor_variance.tolist(),
                            kind=kind,
                            samples=len(candidates),
                        )
                    )
            motion_indices = list(range(left, stop + 1))
            z = []
            r = []
            for i in motion_indices:
                value, variance, _ = measurements[i]
                z.append(value)
                r.append(variance)
            if stop == right and right in contact_boundaries:
                nearby = []
                for k in range(max(begin, right - 2), min(len(rows), right + 3)):
                    if abs(times[k] - times[right]) > 0.7:
                        continue
                    for o in observations[k]:
                        if o["kind"] == "overhead" and o.get("quality", 0) > 0.5 and not o.get("rejection"):
                            nearby.append((abs(times[k] - times[right]), k, o))
                if nearby:
                    gap, k, o = min(nearby, key=lambda item: item[0])
                    z[-1] = np.array(o["xy"])
                    r[-1] = np.array(measurement_variance(o)) + gap**2 * 0.5**2
                    summary["contact_anchors"].append(
                        dict(
                            stone=name,
                            contact_seconds=float(times[right]),
                            observation_seconds=float(times[k]),
                            position=z[-1].tolist(),
                            variance=r[-1].tolist(),
                            view=o["view_id"],
                        )
                    )
            if boundary_anchor is not None:
                z[0] = boundary_anchor
                r[0] = boundary_variance if boundary_variance is not None else np.array([0.05**2] * 2)
            if anchor is not None:
                z[-1] = anchor
                r[-1] = anchor_variance
            if len(motion_indices) < 3 or sum(v is not None for v in z) < 3:
                continue
            smoothed, cov, _, _ = rts_smooth(times[motion_indices], z, r)
            end_anchor = anchor
            # A shared impact boundary retains a single position on both sides.
            if stop == right and right in contact_boundaries and z[-1] is not None:
                end_anchor = z[-1]
            fit = fit_glide(
                times[motion_indices],
                z,
                r,
                smoothed,
                start_anchor=boundary_anchor,
                end_anchor=end_anchor,
                stopping=stationary,
            )
            record = {k: v for k, v in fit.items() if k not in {"positions", "velocities", "speeds"}}
            record.update(
                stone=name,
                start_seconds=float(times[left]),
                end_seconds=float(times[stop]),
                stopping=stationary,
            )
            summary["segments"].append(record)
            if not fit["success"]:
                summary["warnings"].append(
                    f"{name} at {times[left]:.2f}s: constrained fit failed; "
                    "RTS-only result is flagged for review"
                )
            for j, i in enumerate(motion_indices):
                stone = next(s for s in output[i]["stones"] if s["id"] == name)
                position = fit["positions"][j]
                velocity = fit["velocities"][j]
                # Include disagreement with raw observations in uncertainty, not just the
                # optimistic covariance from a model that assumes perfect camera geometry.
                mismatch = np.zeros(2) if z[j] is None else (position - z[j]) ** 2
                variance = np.maximum(np.diag(cov[j])[:2], 0.05**2) + mismatch
                if j == 0 and boundary_anchor is not None:
                    variance = np.maximum(variance, r[0])
                if j == len(motion_indices) - 1 and end_anchor is not None:
                    variance = np.maximum(variance, r[-1])
                stone.update(
                    position=position.tolist(),
                    velocity=velocity.tolist(),
                    speed_mps=float(fit["speeds"][j]),
                    position_covariance=np.diag(variance).tolist(),
                    uncertainty95_m=float(np.sqrt(5.991 * max(variance))),
                    estimate_kind="smoothed" if z[j] is not None else "smoothed_gap",
                    trajectory_phase="free_glide" if fit["success"] else "needs_review",
                    smoothed=True,
                    uses_future_observations=True,
                )
            if anchor is not None:
                for i in range(stop, right + 1):
                    stone = next(s for s in output[i]["stones"] if s["id"] == name)
                    stone.update(
                        position=anchor.tolist(),
                        velocity=[0, 0],
                        speed_mps=0.0,
                        position_covariance=np.diag(anchor_variance).tolist(),
                        uncertainty95_m=float(np.sqrt(5.991 * max(anchor_variance))),
                        estimate_kind="house_anchored" if kind == "overhead" else "stationary_anchored",
                        trajectory_phase="stationary",
                        smoothed=True,
                        uses_future_observations=True,
                    )
            boundary_variance = np.diag(
                next(s for s in output[right]["stones"] if s["id"] == name)["position_covariance"]
            )
            boundary_anchor = np.array(
                next(s for s in output[right]["stones"] if s["id"] == name)["position"]
            )
    return output, summary


def process_run(root):
    root = Path(root)
    input_bytes = (root / "observations.jsonl").read_bytes()
    rows = [json.loads(line) for line in input_bytes.splitlines()]
    manifest = json.loads(
        (root / "manifest.json" if (root / "manifest.json").exists() else root / "job.json").read_text()
    )
    events = json.loads((root / "events.json").read_text())
    result, summary = smooth_rows(rows, manifest, events)
    geometry = fit_geometry(rows, manifest["calibration"])
    summary["geometry_status"] = geometry["status"]
    summary["tee_distance_m"] = manifest["calibration"].get("tee_distance_m", 34.747)
    summary["dimension_source"] = manifest["calibration"].get("tee_distance_source", "nominal")
    summary["length_mode"] = geometry.get("length_mode", "estimated")
    summary["warnings"].append(
        "Dimensions use the selected venue baseline; camera audit: " + geometry["status"]
        if manifest["calibration"].get("tee_distance_fixed")
        else "Sheet length and camera geometry remain unverified; audit: " + geometry["status"]
    )
    summary.update(
        input_sha256=hashlib.sha256(input_bytes).hexdigest(),
        calibration_revision=manifest["calibration"]["revision"],
    )
    for name, value in [
        ("trajectory.json", dict(model=VERSION, frames=result)),
        ("trajectory-summary.json", summary),
        ("geometry-report.json", geometry),
    ]:
        temporary = root / (name + ".tmp")
        temporary.write_text(json.dumps(value, separators=(",", ":")))
        temporary.replace(root / name)
    manifest["trajectory_model"] = VERSION
    manifest["trajectory_input_sha256"] = summary["input_sha256"]
    temporary = root / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(root / "manifest.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    summary = process_run(args.run)
    print(json.dumps({k: v for k, v in summary.items() if k != "house_anchors"}, indent=2))

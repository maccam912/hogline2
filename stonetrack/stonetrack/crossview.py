"""Align the cameras on stones that two of them see at once.

Paint fixes each camera's mapping of the ice plane. What paint cannot give is how a stone, which
stands 11 cm tall, relates to that plane in each view. Stones at rest in a house are seen by the
overhead camera and by the far perspective camera at the same moment, so they calibrate exactly
that: the overhead camera's height (its parallax grows linearly away from the point beneath it)
and the effective height of the handle as each perspective camera sees it.

Sampling keeps only stones that are not moving, so no time alignment between panels is needed.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from . import frames
from .geometry import STONE_CAP_Z, STONE_HANDLE_Z, Camera, apply_h, tee


def sample_stationary(run, cams, detector, times, out_path, budget=1e9):
    """Append stationary detections for each sample time to a JSONL cache (resumable)."""
    import time as _time

    size = (run["video"]["width"], run["video"]["height"])
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(line)["t"] for line in open(out_path) if line.strip()}
    began = _time.time()
    with open(out_path, "a") as fh:
        for t in times:
            t = round(float(t), 3)
            if t in done:
                continue
            if _time.time() - began > budget:
                return False
            pair = [f.copy() for _, f in frames.read(run["source"], t, 1.0, 2, size)][:2]
            if len(pair) < 2:
                continue
            moved = cv2.absdiff(pair[0], pair[1]).max(axis=2) > 25
            keep = []
            for d in detector.detect(pair[0]):
                x, y = int(d.px[0]), int(d.px[1])
                if moved[max(y - 14, 0) : y + 15, max(x - 14, 0) : x + 15].mean() < 0.03:
                    keep.append(d.as_dict())
            fh.write(json.dumps(dict(t=t, d=keep)) + "\n")
            fh.flush()
    return True


def turned(cam):
    """The same overhead camera read the other way up. Rings and lines look identical after a
    half turn about the tee, so paint cannot tell which way an overhead camera is mounted."""
    t = tee(cam.end)
    T = np.array([[-1, 0, 2 * t[0]], [0, -1, 2 * t[1]], [0, 0, 1.0]])
    ground = None if cam.ground is None else 2 * t - cam.ground
    return Camera(cam.id, cam.kind, cam.end, cam.crop, cam.G @ T, ground, cam.height, dict(cam.extra, half_turn=not cam.extra.get("half_turn", False)))


def pairs_in_house(samples, cams, max_dist=0.45, handle_z=None):
    """Match each overhead detection with the same-colour perspective detection of that house.
    Positions are recomputed from pixels, so the cameras passed in decide the geometry."""
    handle_z = handle_z or {}
    out = []
    for s in samples:
        for d in s["d"]:
            cam = cams[d["view"]]
            z = STONE_CAP_Z if cam.kind == "overhead" else handle_z.get(cam.id, cam.extra.get("handle_height", STONE_HANDLE_Z))
            d["_ice"] = cam.to_ice(d["px"], z)[0]
        for end in ("A", "B"):
            over = [d for d in s["d"] if cams[d["view"]].kind == "overhead" and cams[d["view"]].end == end and d["q"] >= 0.8]
            pers = [d for d in s["d"] if cams[d["view"]].kind == "perspective" and cams[d["view"]].end == end]
            for o in over:
                same = sorted((p for p in pers if p["colour"] == o["colour"]), key=lambda p: np.linalg.norm(p["_ice"] - o["_ice"]))
                if not same or np.linalg.norm(same[0]["_ice"] - o["_ice"]) > max_dist:
                    continue
                if len(same) > 1 and np.linalg.norm(same[1]["_ice"] - o["_ice"]) < 2 * max_dist:
                    continue  # two candidates: skip rather than guess
                out.append(dict(t=s["t"], end=end, colour=o["colour"], o_view=o["view"], p_view=same[0]["view"],
                                o_px=o["px"], p_px=same[0]["px"]))
    return out


def orient_overheads(samples, cams):
    """Pick each overhead camera's orientation by which one makes its stones line up with the
    perspective camera's. Returns {camera id: (pairs as given, pairs turned)}."""
    votes = {}
    for cid in [c for c in cams if cams[c].kind == "overhead"]:
        end = cams[cid].end
        as_given = sum(p["end"] == end for p in pairs_in_house(samples, cams))
        trial = dict(cams)
        trial[cid] = turned(cams[cid])
        as_turned = sum(p["end"] == end for p in pairs_in_house(samples, trial))
        votes[cid] = (as_given, as_turned)
        if as_turned > 1.5 * as_given + 5:
            cams[cid] = trial[cid]
    return votes


def _positions(pairs, cams, handle_z):
    o = np.array([cams[p["o_view"]].to_ice(p["o_px"], STONE_CAP_Z)[0] for p in pairs])
    q = np.array([cams[p["p_view"]].to_ice(p["p_px"], handle_z[p["p_view"]])[0] for p in pairs])
    return o, q


def residual_stats(pairs, cams, handle_z):
    """Perspective minus overhead, per house: metres across (x) and along (y) the sheet."""
    out = {}
    o, q = _positions(pairs, cams, handle_z)
    for end in ("A", "B"):
        m = np.array([p["end"] == end for p in pairs])
        if m.sum() < 5:
            continue
        d = (q - o)[m]
        out[end] = dict(n=int(m.sum()), mean_x=float(d[:, 0].mean()), mean_y=float(d[:, 1].mean()),
                        sd_x=float(d[:, 0].std()), sd_y=float(d[:, 1].std()),
                        rms=float(np.sqrt((d**2).sum(1).mean())), median_abs=float(np.median(np.hypot(d[:, 0], d[:, 1]))))
    return out


def fit(pairs, cams, handle_z=None):
    """Solve the overhead camera heights and the per-camera handle height. Mutates `cams`.

    Overhead height: across the sheet a perspective camera measures a stone's lateral position
    without parallax to speak of, so the overhead cap's apparent lateral offset from it, regressed
    on distance from the point beneath the overhead camera, has slope cap_height / camera_height.
    Handle height: the value that removes the mean along-sheet disagreement in that house.
    """
    handle_z = dict(handle_z or {c.id: STONE_HANDLE_Z for c in cams.values() if c.kind == "perspective"})
    report = {}
    for end in ("A", "B"):
        sel = [p for p in pairs if p["end"] == end]
        if len(sel) < 20:
            continue
        oc, pc = cams[sel[0]["o_view"]], cams[sel[0]["p_view"]]
        for _ in range(3):  # the two unknowns are nearly independent; a few rounds settle them
            app = np.array([apply_h(oc.Ginv, [p["o_px"]])[0] for p in sel])  # cap as if painted on the ice
            q = np.array([pc.to_ice(p["p_px"], handle_z[pc.id])[0] for p in sel])
            u = np.r_[app[:, 0] - oc.ground[0]]
            v = np.r_[app[:, 0] - q[:, 0]]
            keep = np.abs(v - np.median(v)) < 0.08
            A = np.c_[u[keep], np.ones(keep.sum())]
            (kappa, bias), *_ = np.linalg.lstsq(A, v[keep], rcond=None)
            resid = v[keep] - A @ [kappa, bias]
            kappa_se = float(resid.std() / (u[keep].std() * np.sqrt(keep.sum())))
            if kappa > 0.005:
                oc.height = float(STONE_CAP_Z / kappa)
            # handle height: shift along the viewing ray until the houses agree along the sheet
            o = np.array([oc.to_ice(p["o_px"], STONE_CAP_Z)[0] for p in sel])
            zs = np.linspace(0.08, 0.19, 111)
            means = [np.median(np.array([pc.to_ice(p["p_px"], z)[0] for p in sel])[:, 1] - o[:, 1]) for z in zs]
            handle_z[pc.id] = float(zs[int(np.argmin(np.abs(means)))])
        report[end] = dict(pairs=len(sel), overhead=oc.id, overhead_height_m=oc.height, parallax_slope=float(kappa),
                           parallax_slope_se=kappa_se, lateral_bias_m=float(bias), perspective=pc.id,
                           handle_height_m=handle_z[pc.id])
    return handle_z, report

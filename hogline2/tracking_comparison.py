"""Matched end-preview experiment; masks and velocities have no motion filtering."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from .tracking_geometry import prepare_view, sheet_geometry
from .tracking_raw import add_raw_velocity, raw_mask_measurement
from .tracking_runner import atomic


def run(path):
    job = json.loads(path.read_text())
    root = path.parent
    started = time.monotonic()
    status = dict(
        state="running",
        stage="Loading " + job["backend"],
        progress=0,
        pid=os.getpid(),
        started_at=time.time(),
        run_id=root.name,
        backend=job["backend"],
        raw_comparison=True,
    )
    atomic(root / "status.json", status)
    with Path(job["source_path"]).open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != job["source_sha256"]:
            raise ValueError("Source changed")
    if job["backend"] == "sam2.1-mlx-reinitialized":
        from .tracking_reinitialize import Sam21Reinitialized

        backend = Sam21Reinitialized()
    elif job["backend"] == "sam2.1-mlx-granite":
        from .tracking_sam2_granite import Sam21Granite

        backend = Sam21Granite()
    elif job["backend"] == "sam3.1-mlx-multiplex":
        from .tracking_sam31 import Sam31Multiplex

        backend = Sam31Multiplex()
    else:
        raise ValueError("Unknown comparison backend")
    views = [
        prepare_view(v, job["calibration"].get("tee_distance_m"), sheet_geometry(job["calibration"]))
        for v in job["calibration"]["views"]
    ]
    fps, start = job["fps"], job["source_start"]
    count = round(job["duration"] * fps)
    cap = cv2.VideoCapture(job["source_path"])
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    step = round(cap.get(cv2.CAP_PROP_FPS) / fps)
    w, h = job["media"]["width"], job["media"]["height"]
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(root / "overlay.partial.mp4"),
        ],
        stdin=subprocess.PIPE,
        stderr=(root / "encode.log").open("wb"),
    )
    (root / "masks").mkdir(exist_ok=True)
    previous, all_raw, rows, events = {}, [], [], []
    windows = job["track_windows"]
    names = sorted({window["stone_id"] for window in windows})
    with (root / "observations.jsonl").open("w") as output:
        try:
            for frame in range(count):
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError("Unexpected end of source")
                for _ in range(step - 1):
                    cap.grab()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                overlay = rgb.copy()
                seconds = start + frame / fps
                observations, packed = [], {}
                if hasattr(backend, "begin_frame"):
                    backend.begin_frame(frame)
                for view in views:
                    vid = view["id"]
                    active = [
                        a
                        for a in windows
                        if a["view_id"] == vid and a["first_frame"] <= frame <= a["last_frame"]
                    ]
                    if not active:
                        continue
                    x0, y0, x1, y1 = view["crop"]
                    crop = rgb[y0:y1, x0:x1]
                    seeds = [a for a in active if a["first_frame"] == frame]
                    if job["backend"] == "sam3.1-mlx-multiplex":
                        predictions = backend.step(view, crop, frame, seeds)
                    else:
                        predictions = {
                            a["stone_id"]: backend.observe(
                                view,
                                a["stone_id"],
                                a["color"],
                                crop,
                                a["point"] if a["first_frame"] == frame else None,
                                fps,
                            )
                            for a in active
                        }
                    for window in active:
                        name = window["stone_id"]
                        key = (vid, name)
                        mask, quality = predictions[name]
                        packed[f"{vid}__{name}"] = np.packbits(mask)
                        measured = raw_mask_measurement(mask, view)
                        reason = "empty_or_unprojectable_mask"
                        if job["backend"] == "sam2.1-mlx-granite" and view["kind"] == "perspective":
                            from .tracking_stone import granite_anchor, granite_evidence

                            anchor = granite_anchor(mask, granite_evidence(crop, mask, window["color"]))
                            measured = (
                                raw_mask_measurement(mask, view, anchor) if anchor is not None else None
                            )
                            reason = "granite_bottom_not_verified"
                            if measured is not None:
                                measured["measurement_kind"] = "visible_granite_bottom"
                        if measured is None:
                            previous.pop(key, None)
                            all_raw.append(
                                dict(
                                    frame=frame,
                                    source_seconds=seconds,
                                    stone_id=name,
                                    view_id=vid,
                                    valid=False,
                                    reason=reason,
                                )
                            )
                            continue
                        measured.update(
                            stone_id=name,
                            view_id=vid,
                            kind=view["kind"],
                            quality=quality,
                            valid=True,
                            accepted=True,
                            color=window["color"],
                        )
                        measured = add_raw_velocity(previous.get(key), measured, frame, seconds)
                        previous[key] = measured
                        observations.append(measured)
                        all_raw.append(measured)
                        color = np.array([255, 75, 90] if window["color"] == "red" else [255, 220, 30])
                        target = overlay[y0:y1, x0:x1]
                        target[mask] = (target[mask] * 0.5 + color * 0.5).astype(np.uint8)
                        yy, xx = np.where(mask)
                        cv2.putText(
                            overlay,
                            name,
                            (x0 + int(xx.mean()), y0 + int(yy.min()) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            tuple(map(int, color)),
                            1,
                        )
                stones = []
                for name in names:
                    choices = [o for o in observations if o["stone_id"] == name]
                    if not choices:
                        continue
                    selected = sorted(choices, key=lambda o: (o["kind"] != "overhead", o["view_id"]))[0]
                    stones.append(
                        dict(
                            id=name,
                            color=selected["color"],
                            position=selected["xy"],
                            velocity=selected["velocity"] or [0, 0],
                            visible=True,
                            state="raw",
                            estimate_kind="raw",
                            raw_view=selected["view_id"],
                        )
                    )
                mask_file = f"masks/{frame:06}.npz" if packed else None
                if packed:
                    np.savez_compressed(root / mask_file, **packed)
                row = dict(
                    frame=frame,
                    source_seconds=seconds,
                    stones=stones,
                    observations=observations,
                    raw_observations=observations,
                    masks=mask_file,
                )
                rows.append(row)
                output.write(json.dumps(row) + "\n")
                cv2.rectangle(overlay, (0, h - 34), (w, h), (15, 25, 25), -1)
                cv2.putText(
                    overlay,
                    f"{backend.name} | source {seconds:.2f}s | RAW masks, no motion filter",
                    (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (235, 245, 240),
                    1,
                )
                encoder.stdin.write(overlay.tobytes())
                if frame % 3 == 0 or frame == count - 1:
                    output.flush()
                    elapsed = time.monotonic() - started
                    status.update(
                        stage=f"Raw comparison {frame + 1}/{count} frames",
                        progress=(frame + 1) / count,
                        stones=len(stones),
                        elapsed_seconds=elapsed,
                        peak_metal_bytes=backend.peak_bytes,
                    )
                    atomic(root / "status.json", status)
                    print(status["stage"], f"{elapsed:.1f}s elapsed", flush=True)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("Encoding failed")
            (root / "overlay.partial.mp4").replace(root / "overlay.mp4")
        finally:
            cap.release()
            if encoder.poll() is None:
                encoder.terminate()
                encoder.wait()
    for event in backend.events:
        events.append({**event, "seconds": start + event["frame"] / fps})
    atomic(root / "events.json", events)
    atomic(
        root / "raw-velocity.json",
        dict(method="consecutive per-camera raw mask position differences", fps=fps, samples=all_raw),
    )
    atomic(
        root / "manifest.json",
        dict(
            **job,
            frames=len(rows),
            elapsed_seconds=time.monotonic() - started,
            peak_metal_bytes=backend.peak_bytes,
            weights_sha256=backend.weight_sha256,
            vendor_revision=backend.revision,
            motion_model=None,
            stones=[dict(id=name, color=name.split("_")[0]) for name in names],
            limitations=[
                "Matched identity and camera windows from archived preview; not autonomous discovery",
                "No motion gate, Kalman filter, inferred granite foot, smoothing, or interpolated velocity",
                "Camera calibration and mask height can distort projected speed; inspect pixel speed too",
                "Raw map selects a camera; velocity comparison always remains within one camera",
            ],
        ),
    )
    status.update(
        state="complete",
        stage="Raw velocity comparison ready",
        progress=1,
        elapsed_seconds=time.monotonic() - started,
        stones=len(names),
    )
    atomic(root / "status.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    try:
        with (args.job.parent.parent.parent.parent / "sam-worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args.job)
    except Exception as error:
        (args.job.parent / "error.log").write_text(traceback.format_exc())
        atomic(
            args.job.parent / "status.json", dict(state="failed", stage="Comparison failed", error=str(error))
        )
        raise


if __name__ == "__main__":
    main()

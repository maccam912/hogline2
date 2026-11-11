"""Isolated SAM process. Run an end (or an explicit preview interval) with bounded memory."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from .media import frame as source_frame
from .segmentation import load_backend
from .tracking_geometry import prepare_view, project, sheet_geometry
from .tracking_jobs import gpu_lease
from .tracking_proposals import DeliveryGate, candidates
from .tracking_proposals import mask_observation as measure_mask
from .tracking_state import EndState


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def run(job_path):
    job = json.loads(job_path.read_text())
    root = job_path.parent
    started = time.monotonic()
    status = dict(
        state="running",
        motion_model="kalman_cv_ice_v1",
        stage="Loading " + job.get("backend", "sam2.1-mlx"),
        progress=0,
        pid=os.getpid(),
        started_at=time.time(),
        run_id=root.name,
    )
    atomic(root / "status.json", status)
    with Path(job["source_path"]).open("rb") as source:
        expected_sha = job.get("source_media_sha256", job["source_sha256"])
        if hashlib.file_digest(source, "sha256").hexdigest() != expected_sha:
            raise ValueError("Source recording changed after calibration")
    backend = load_backend(job.get("backend", "sam2.1-mlx"))
    strict_body = getattr(backend, "granite_body", False)

    def segment(view, name, color, crop, point=None):
        if strict_body:
            return backend.observe(view, name, color, crop, point, job["fps"])
        return backend.track(view["id"], name, crop, point)

    def mask_observation(*args):
        return measure_mask(*args, require_granite=strict_body)

    views = [
        prepare_view(v, job["calibration"].get("tee_distance_m"), sheet_geometry(job["calibration"]))
        for v in job["calibration"]["views"]
    ]
    background = cv2.cvtColor(
        cv2.imdecode(
            np.frombuffer(
                Path(job["background_path"]).read_bytes()
                if job.get("background_path")
                else source_frame(
                    job["source_path"],
                    job["calibration"]["source_seconds"],
                    job["media"]["width"],
                ),
                np.uint8,
            ),
            cv2.IMREAD_COLOR,
        ),
        cv2.COLOR_BGR2RGB,
    )
    state = EndState(job["end"]["number"])
    gates = {v["id"]: DeliveryGate() for v in views}
    fps, start, duration = job["fps"], job["source_start"], job["duration"]
    width, height = job["media"]["width"], job["media"]["height"]
    count = job.get("expected_frames", int(math.ceil(duration * fps)))
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "2",
            "-ss",
            str(start - job.get("source_time_origin", 0)),
            "-i",
            job["source_path"],
            "-t",
            str(duration),
            "-vf",
            f"fps={fps}",
            "-threads",
            "2",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=(root / "decode.log").open("wb"),
    )
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
            f"{width}x{height}",
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
    masks_dir = root / "masks"
    masks_dir.mkdir(exist_ok=True)
    observations_file = (root / "observations.jsonl").open("w")
    seeds = list(job.get("seeds", []))
    lost = {}
    previous_proposals = {}
    pending_deliveries = {}
    frame_index = 0
    frames_processed = 0
    try:
        for frame_index in range(count):
            raw = decoder.stdout.read(width * height * 3)
            if not raw:
                break
            if len(raw) != width * height * 3:
                raise ValueError("Incomplete decoded frame")
            rgb = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
            frames_processed += 1
            overlay = rgb.copy()
            seconds = start + frame_index / fps
            backend.begin_frame(frame_index)
            observations, masks_out = [], {}
            crops, proposals = {}, {}
            for view in views:
                vid = view["id"]
                x0, y0, x1, y1 = view["crop"]
                crop = rgb[y0:y1, x0:x1]
                crops[vid] = crop
                proposals[vid] = candidates(crop, view, background[y0:y1, x0:x1], require_granite=strict_body)
                for candidate in proposals[vid]:
                    old = [p for p in previous_proposals.get(vid, []) if p["color"] == candidate["color"]]
                    nearest = min(
                        (math.dist(p["point"], candidate["point"]) for p in old),
                        default=1e9,
                    )
                    candidate["moving"] = 1.0 <= nearest <= 90 / fps
                previous_proposals[vid] = proposals[vid]
                automatic = gates[vid].update(seconds, view, proposals[vid])
                if automatic:
                    pending_deliveries[vid] = {**automatic[0], "expires": seconds + 3}
                pending_candidate = pending_deliveries.get(vid)
                automatic = []
                if pending_candidate and seconds <= pending_candidate["expires"]:
                    nearby = [
                        p
                        for p in proposals[vid]
                        if p["color"] == pending_candidate["color"]
                        and math.dist(p["point"], pending_candidate["point"]) < 45
                    ]
                    if nearby:
                        candidate = min(
                            nearby,
                            key=lambda p: math.dist(p["point"], pending_candidate["point"]),
                        )
                        pending_candidate.update(candidate)
                        automatic = [candidate]
                manual = [s for s in seeds if s["view_id"] == vid and seconds >= s["source_seconds"]]
                for seed in manual:
                    seeds.remove(seed)
                for candidate in automatic + manual:
                    color = candidate["color"]
                    # Only the hack at the delivery end may create a new physical identity.
                    if vid != job["delivery_view"]:
                        continue
                    if seconds > job["end"]["end_seconds"] - 3 or state.counts.get(color, 8) >= 8:
                        continue
                    if any(seconds - s.delivered_at < 15 for s in state.stones.values()):
                        continue
                    point = candidate["point"]
                    pending = "pending_delivery"
                    mask, quality = segment(view, pending, color, crop, [point[0] - x0, point[1] - y0])
                    obs = mask_observation(mask, quality, crop, view, color, background[y0:y1, x0:x1])
                    if obs is None:
                        backend.retire(vid, pending)
                        continue
                    pending_deliveries.pop(vid, None)
                    stone = state.birth(color, seconds, obs["xy"], vid)
                    backend.rename_track(vid, pending, stone.id)
                    observations.append({**obs, "stone_id": stone.id})
                    masks_out[(vid, stone.id)] = mask
            # Continue all existing view-local tracks, maintaining separate SAM memories.
            for view in views:
                vid = view["id"]
                crop = crops[vid]
                x0, y0, x1, y1 = view["crop"]
                for stone in list(state.stones.values()):
                    key = (vid, stone.id)
                    if key in masks_out:
                        continue
                    prompt = None
                    if not backend.has_track(vid, stone.id):
                        # Learn initial velocity from two measurements before a cross-view handoff.
                        if stone.accepted_observations < 2 and vid not in stone.views:
                            continue
                        choices = []
                        for candidate in proposals[vid]:
                            if (
                                seconds - stone.last_seen > 1.5
                                and stone.state != "stationary"
                                and not candidate["moving"]
                            ):
                                continue
                            matched = state.match(
                                candidate["color"],
                                candidate["xy"],
                                seconds,
                                vid,
                                variance=candidate.get("variance"),
                            )
                            if matched is stone:
                                choices.append(candidate)
                        if len(choices) != 1:
                            continue
                        prompt = [
                            choices[0]["point"][0] - x0,
                            choices[0]["point"][1] - y0,
                        ]
                    mask, quality = segment(view, stone.id, stone.color, crop, prompt)
                    obs = mask_observation(mask, quality, crop, view, stone.color, background[y0:y1, x0:x1])
                    # A continuing SAM mask can drift onto another same-color stone too.
                    # Apply the same motion gate used for a new camera association.
                    if (
                        obs is not None
                        and state.match(
                            stone.color,
                            obs["xy"],
                            seconds,
                            vid,
                            radius=0.85,
                            variance=obs.get("variance"),
                        )
                        is not stone
                    ):
                        obs = None
                    if obs is None and lost.get(key, 0) % 3 == 2:
                        recovery = [
                            p
                            for p in proposals[vid]
                            if state.match(
                                p["color"],
                                p["xy"],
                                seconds,
                                vid,
                                variance=p.get("variance"),
                            )
                            is stone
                        ]
                        if len(recovery) == 1:
                            p = recovery[0]["point"]
                            mask, quality = segment(view, stone.id, stone.color, crop, [p[0] - x0, p[1] - y0])
                            obs = mask_observation(
                                mask,
                                quality,
                                crop,
                                view,
                                stone.color,
                                background[y0:y1, x0:x1],
                            )
                            if (
                                obs is not None
                                and state.match(
                                    stone.color,
                                    obs["xy"],
                                    seconds,
                                    vid,
                                    radius=0.85,
                                    variance=obs.get("variance"),
                                )
                                is not stone
                            ):
                                obs = None
                            if obs is not None:
                                state.events.append(
                                    dict(
                                        type="mask_reprompt",
                                        stone=stone.id,
                                        view=vid,
                                        seconds=seconds,
                                    )
                                )
                    if obs is None:
                        backend.reject_current_memory(vid, stone.id)
                        lost[key] = lost.get(key, 0) + 1
                        if lost[key] > fps * 8:
                            backend.retire(vid, stone.id)
                        continue
                    if vid not in stone.views:
                        # Validate the actual SAM ground estimate, not just the color proposal.
                        if (
                            state.match(
                                stone.color,
                                obs["xy"],
                                seconds,
                                vid,
                                radius=0.85,
                                variance=obs.get("variance"),
                            )
                            is not stone
                        ):
                            backend.retire(vid, stone.id)
                            continue
                        stone.views[vid] = dict(joined_at=seconds, method="calibrated_color_motion_match")
                        state.events.append(
                            dict(
                                type="view_join",
                                stone=stone.id,
                                view=vid,
                                seconds=seconds,
                            )
                        )
                    lost[key] = 0
                    observations.append({**obs, "stone_id": stone.id})
                    masks_out[key] = mask
            stones = state.update(seconds, observations)
            packed = {}
            for (vid, name), mask in masks_out.items():
                if (vid, name) not in state.accepted:
                    backend.reject_current_memory(vid, name)
                    continue
                view = next(v for v in views if v["id"] == vid)
                x0, y0, x1, y1 = view["crop"]
                color = (255, 75, 90) if state.stones[name].color == "red" else (255, 220, 30)
                region = overlay[y0:y1, x0:x1]
                region[mask] = (region[mask] * 0.50 + np.asarray(color) * 0.50).astype(np.uint8)
                yy, xx = np.where(mask)
                cv2.putText(
                    overlay,
                    name,
                    (x0 + int(xx.mean()), y0 + int(yy.min()) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
                packed[f"{vid}__{name}"] = np.packbits(mask)
            mask_file = None
            if packed:
                mask_file = f"masks/{frame_index:06}.npz"
                np.savez_compressed(root / mask_file, **packed)
            # Hidden stationary stones have a dashed-style marker, never a fabricated mask.
            for stone in state.stones.values():
                if not stone.visible and stone.state == "stationary":
                    for view in views:
                        if view["kind"] != "overhead" or view["id"] not in stone.views:
                            continue
                        px, py = project(view["ice_to_image"], stone.position)
                        x0, y0, x1, y1 = view["crop"]
                        if x0 <= px < x1 and y0 <= py < y1:
                            cv2.drawMarker(
                                overlay,
                                (int(px), int(py)),
                                (160, 160, 160),
                                cv2.MARKER_CROSS,
                                20,
                            )
                            cv2.putText(
                                overlay,
                                stone.id + " hidden",
                                (int(px) + 12, int(py)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (180, 180, 180),
                                1,
                            )
            cv2.rectangle(overlay, (0, height - 34), (width, height), (15, 25, 25), -1)
            cv2.putText(
                overlay,
                f"SAM 2.1 | end {state.end_number} | source {seconds:.2f}s | "
                f"{len(stones)} stones | provisional identities",
                (12, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                1,
            )
            encoder.stdin.write(overlay.tobytes())
            observations_file.write(
                json.dumps(
                    dict(
                        frame=frame_index,
                        source_seconds=seconds,
                        stones=stones,
                        observations=observations,
                        masks=mask_file,
                    )
                )
                + "\n"
            )
            if frame_index % max(1, int(fps * 2)) == 0:
                observations_file.flush()
                status.update(
                    stage=f"Tracking {seconds - start:.1f} / {duration:.1f} seconds",
                    progress=(frame_index + 1) / count,
                    stones=len(stones),
                    elapsed_seconds=time.monotonic() - started,
                    peak_device_bytes=backend.peak_bytes,
                )
                atomic(root / "status.json", status)
                print(status["stage"], len(stones), flush=True)
        observations_file.flush()  # The offline pass must see the final buffered frames.
        decoder.stdout.close()
        if decoder.wait() != 0:
            raise ValueError("Video decoding failed")
        if not frames_processed or (job.get("expected_frames") and frames_processed != count):
            raise ValueError("Decoded frame count does not match the tracking job")
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise ValueError("Overlay encoding failed")
        (root / "overlay.partial.mp4").replace(root / "overlay.mp4")
        for event in getattr(backend, "events", []):
            state.events.append({**event, "seconds": start + event["frame"] / fps})
        atomic(root / "events.json", state.events)
        manifest = {
            **job,
            "backend": backend.name,
            "motion_model": "kalman_cv_ice_v1",
            "coordinate_system": {
                "units": "estimated_metres",
                "origin": "house_A_button",
                "x_axis": "across_sheet",
                "y_axis": "toward_house_B",
                "ice_plane_z": 0,
                "tee_spacing_m": job["calibration"].get("tee_distance_m", 34.747),
                "sheet_geometry": sheet_geometry(job["calibration"]),
                "full_3d_reconstruction": False,
            },
            "weights_sha256": backend.weight_sha256,
            "vendor_revision": backend.revision,
            "frames": frames_processed,
            "elapsed_seconds": time.monotonic() - started,
            "peak_device_bytes": backend.peak_bytes,
            "memory_device": getattr(backend, "memory_device", "metal"),
            "runtime_info": getattr(backend, "runtime_info", {}),
            "stones": stones,
            "limitations": [
                "Automatic delivery and identity proposals need review",
                "Calibration is an ice-plane approximation; stone bodies have height",
                "A hidden position is retained state, not a visible SAM observation",
                "Contact events indicate possible contact, not a proven collision",
                "Sampled frames can miss fast contacts; no cross-view accuracy guarantee",
                "Manual clearing and out-of-play classification are not yet modeled",
            ],
        }
        atomic(root / "manifest.json", manifest)
        status.update(stage="Refining whole-clip trajectories", progress=0.98)
        atomic(root / "status.json", status)
        try:
            from .tracking_smoothing import process_run

            process_run(root)
        except Exception as exc:
            manifest["trajectory_error"] = str(exc)
            atomic(root / "manifest.json", manifest)
            (root / "trajectory-error.log").write_text(traceback.format_exc())
        status.update(
            state="complete",
            stage="Tracking ready — review identities",
            progress=1,
            elapsed_seconds=time.monotonic() - started,
        )
        atomic(root / "status.json", status)
    finally:
        observations_file.close()
        for process in (decoder, encoder):
            if process.poll() is None:
                process.terminate()
                process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    root = args.job.parent
    try:
        with gpu_lease(root.parent.parent.parent):
            run(args.job)
    except Exception as exc:
        (root / "error.log").write_text(traceback.format_exc())
        atomic(
            root / "status.json",
            dict(
                state="failed",
                stage="Tracking needs attention",
                progress=0,
                error=f"{type(exc).__name__}: {exc}",
                run_id=root.name,
            ),
        )
        raise


if __name__ == "__main__":
    main()

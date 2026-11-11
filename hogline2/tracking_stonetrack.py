"""Run the stonetrack tracker as a Hogline tracking backend ("Run tracking on: stonetrack").

Same contract as hogline2.tracking_runner: it is started with a job.json, keeps status.json up to
date, and leaves manifest.json, observations.jsonl, events.json and overlay.mp4 in the run folder
so the existing review page can open the run. No SAM, no masks: stones are found by colour and
shape and tracked on the ice plane (see stonetrack/README.md).

stonetrack keeps its own per-recording work (background, paint calibration, detections) under
data/stonetrack/<recording>/ and reuses it across runs. It takes only the panel crops, camera
kinds and orientations from Hogline's calibration; landmarks are not used.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "stonetrack"))

FPS = 20.0  # the tracker's rules are written for this rate
ROW_EVERY = 4  # observations.jsonl is written at 5 rows per second to keep the review page light
OVERLAY_FPS = 10
OVERLAY_SIZE = (960, 540)
NAMES = {"R": "red", "Y": "yellow"}


def _write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


def _layout(job):
    views = [
        dict(
            id=v["id"],
            kind=v["kind"],
            end=v["end"],
            crop=[int(c) for c in v["crop"]],
            rotation_degrees=v.get("rotation_degrees", 0),
        )
        for v in job["calibration"]["views"]
    ]
    return dict(
        name=f"hogline2 layout for {job['recording_id']}",
        width=job["media"]["width"],
        height=job["media"]["height"],
        views=views,
    )


def prepare(job, work, say):
    """Background, paint calibration and stone alignment for this recording (cached)."""
    from stonetrack import cli, frames

    work.mkdir(parents=True, exist_ok=True)
    layout = _layout(job)
    run_path = work / "run.json"
    stale = True
    if run_path.exists():
        old = json.loads(run_path.read_text())
        stale = (
            old.get("layout", {}).get("views") != layout["views"] or old.get("source") != job["source_path"]
        )
    if stale:
        _write(
            run_path, dict(source=job["source_path"], layout=layout, video=frames.probe(job["source_path"]))
        )
        for name in ("calibration.json", "stationary.jsonl"):
            if (work / name).exists():
                (work / name).unlink()
    if not (work / "calibration.json").exists():
        say("Calibrating cameras from the paint", 0.03)
        cli.cmd_calibrate(argparse.Namespace(run=str(work), overhead_height=4.0))
    if "stone_alignment" not in json.loads((work / "calibration.json").read_text()):
        say("Aligning cameras on stones at rest", 0.08)
        cli.cmd_refine(
            argparse.Namespace(run=str(work), frames=None, start=60, end=None, every=20, budget=1e9)
        )


def detect(job, work, say):
    from stonetrack import cli

    start, end = job["source_start"], job["source_start"] + job["duration"]
    workers = max(2, min(8, (os.cpu_count() or 4) // 2))
    t = start
    while t < end:  # in slices, so the progress bar moves
        upto = min(end, t + 120)
        cli.cmd_detect(
            argparse.Namespace(
                run=str(work), start=t, end=upto, ends=None, fps=FPS, workers=workers, budget=1e9, redo=False
            )
        )
        t = upto
        say("Finding stones", 0.15 + 0.55 * (t - start) / (end - start))


def _stone_id(label):
    return f"{NAMES[label[0]]}_{label[1:]}"


def convert(job, res, folder):
    """Hogline's observation rows, events and manifest from a stonetrack result."""
    start, n = job["source_start"], int(round(job["duration"] * FPS))
    index = {}
    for s in res["stones"]:
        index[s["id"]] = {int(round((p[0] - start) * FPS)): p for p in s["path"]}
    last, rows = {}, []
    with (folder / "observations.jsonl").open("w") as out:
        for k in range(0, n, ROW_EVERY):
            t = start + k / FPS
            stones = []
            for s in res["stones"]:
                if s["delivered_at"] > t + 1e-6:
                    continue
                p = index[s["id"]].get(k)
                if p:
                    last[s["id"]] = p
                q = p or last.get(s["id"])
                if q is None:
                    continue
                sd = q[7] if len(q) > 7 else 0.05
                seen = bool(p and p[6])
                kind = "lost" if p is None else "observed" if seen else "predicted"
                stones.append(
                    dict(
                        id=_stone_id(s["id"]),
                        color=s["colour"],
                        delivered_at=s["delivered_at"],
                        position=[q[1], q[2]],
                        velocity=[q[3], q[4]] if p else [0.0, 0.0],
                        speed_mps=round((q[3] ** 2 + q[4] ** 2) ** 0.5, 3) if p else 0.0,
                        state="lost" if p is None else "moving" if q[5] == "moving" else "still",
                        visible=seen,
                        last_seen=q[0],
                        last_observed_position=[q[1], q[2]],
                        estimate_time=t,
                        estimate_kind=kind,
                        position_covariance=[[sd * sd, 0], [0, sd * sd]],
                        uncertainty95_m=round(2.45 * sd, 3),
                        still_since=None,
                        contact_until=-1,
                        views={},
                        warnings=[],
                        accepted_observations=len(index[s["id"]]),
                    )
                )
            row = dict(frame=k // ROW_EVERY, source_seconds=t, stones=stones)
            out.write(json.dumps(row) + "\n")
            rows.append(row)
    events = [
        dict(type="delivery", stone=_stone_id(s["id"]), seconds=s["delivered_at"], view=job["delivery_view"])
        for s in res["stones"]
    ]
    for e in res["events"]:
        if (
            e["type"] in ("knocked", "found again", "out")
            and e.get("stone")
            and not str(e["stone"]).startswith("#")
        ):
            events.append(
                dict(
                    type=e["type"].replace(" ", "_"),
                    stone=_stone_id(e["stone"]),
                    seconds=e["t"],
                    detail=e.get("why"),
                )
            )
    events.sort(key=lambda e: e["seconds"])
    _write(folder / "events.json", events)
    return rows


def overlay(job, res, cams, folder, say):
    """The source clip, half size, with every named stone circled and labelled in each camera."""
    import cv2

    from stonetrack import frames

    start, size = job["source_start"], (job["media"]["width"], job["media"]["height"])
    index = {
        s["id"]: ({int(round((p[0] - start) * FPS)): p for p in s["path"]}, s["colour"])
        for s in res["stones"]
    }
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{OVERLAY_SIZE[0]}x{OVERLAY_SIZE[1]}",
        "-r",
        str(OVERLAY_FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "26",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(folder / "overlay.mp4"),
    ]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = max(1, int(job["duration"] * OVERLAY_FPS))
    for i, (t, frame) in enumerate(
        frames.read(job["source_path"], start, job["duration"], OVERLAY_FPS, size)
    ):
        frame = frame.copy()
        k = int(round((t - start) * FPS))
        for label, (path, colour) in index.items():
            p = path.get(k) or path.get(k + 1) or path.get(k - 1)
            if not p:
                continue
            bgr = (40, 40, 230) if colour == "red" else (0, 210, 250)
            for cam in cams.values():
                px = cam.to_image([[p[1], p[2]]], 0.115)[0]
                if not cam.contains([px], margin=4)[0]:
                    continue
                c = (int(px[0]), int(px[1]))
                r = max(10, int(0.1455 * cam.pixels_per_metre([p[1], p[2]])))
                cv2.circle(frame, c, r + 3, (0, 0, 0), 4)
                cv2.circle(frame, c, r + 3, bgr, 2)
                cv2.putText(
                    frame,
                    label,
                    (c[0] + r + 4, c[1] - r),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    5,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    label,
                    (c[0] + r + 4, c[1] - r),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        enc.stdin.write(cv2.resize(frame, OVERLAY_SIZE, interpolation=cv2.INTER_AREA).tobytes())
        if i % (OVERLAY_FPS * 20) == 0:
            say("Drawing the review video", 0.78 + 0.2 * i / total)
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg could not encode the review video")


def run(job, folder, say):
    from stonetrack.geometry import load_calibration
    from stonetrack.track import check_settled, track_range

    data = Path(job["source_path"]).resolve().parents[2]
    work = data / "stonetrack" / job["recording_id"]
    prepare(job, work, say)
    detect(job, work, say)
    say("Tracking stones", 0.72)
    cams, _ = load_calibration(str(work / "calibration.json"))
    first = job["calibration"].get("first_delivery_end", "B")
    side = first if job["end"]["number"] % 2 == 1 else ("A" if first == "B" else "B")
    res, rows = track_range(
        str(work), cams, job["source_start"], job["source_start"] + job["duration"], FPS, side
    )
    if not rows:
        raise RuntimeError("stonetrack found no frames in this interval")
    res["check"] = check_settled(res, rows, cams)
    _write(folder / "stonetrack-result.json", res)
    obs = convert(job, res, folder)
    overlay(job, res, cams, folder, say)
    shots = {
        _stone_id(s["id"]): dict(
            tee_to_hog_s=s["tee_to_hog_s"],
            hog_to_hog_s=s["hog_to_hog_s"],
            speed_at_near_hog=s["speed_at_near_hog"],
        )
        for s in res["stones"]
    }
    manifest = dict(
        job,
        execution="stonetrack",
        motion_model="stonetrack_kalman_v1",
        frames=len(obs),
        coordinate_system=dict(
            units="metres",
            origin="tee A",
            x_axis="lateral",
            y_axis="tee A toward tee B",
            ice_plane_z=0,
            tee_spacing_m=job["calibration"].get("tee_distance_m", 34.747),
            full_3d_reconstruction=False,
        ),
        stones=obs[-1]["stones"] if obs else [],
        stonetrack=dict(
            delivery_end=side,
            settled_at=res["settled_at"],
            final_positions=res["final_positions"],
            shots=shots,
            check=res["check"],
            calibration=str(work / "calibration.json"),
        ),
        limitations=[
            "Stone identities are delivery order per colour and need review",
            "Stones hit while hidden can be dropped or left behind as ghosts; see stonetrack/README.md",
            "Camera mapping comes from stonetrack's own paint calibration, not the landmarks on this page",
            "No masks are produced; the overlay shows circles at tracked positions",
        ],
    )
    return manifest


def main():
    job_path = Path(sys.argv[1])
    folder = job_path.parent
    job = json.loads(job_path.read_text())
    started = time.time()

    def say(stage, progress, state="running", **extra):
        _write(
            folder / "status.json",
            dict(
                state=state,
                stage=stage,
                progress=round(float(progress), 3),
                pid=os.getpid(),
                started_at=started,
                run_id=folder.name,
                **extra,
            ),
        )

    try:
        say("Starting stonetrack", 0.01)
        manifest = run(job, folder, say)
        manifest["elapsed_seconds"] = time.time() - started
        _write(folder / "manifest.json", manifest)
        say(
            "Tracking ready — review identities",
            1,
            state="complete",
            stones=len(manifest["stones"]),
            elapsed_seconds=manifest["elapsed_seconds"],
        )
    except Exception as exc:  # the page shows this text next to the run
        traceback.print_exc()
        say("stonetrack failed", 0, state="failed", error=str(exc)[:300])
        sys.exit(1)


if __name__ == "__main__":
    main()

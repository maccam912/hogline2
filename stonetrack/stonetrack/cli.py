"""Command line: one run directory per recording.

  stonetrack init RUN --source video.mp4 [--layout layouts/wausau.json]
  stonetrack fetch RUN --url https://youtube.com/watch?v=...     (needs yt-dlp)
  stonetrack calibrate RUN       background + paint calibration + review overlay
  stonetrack refine RUN          align cameras on stones at rest seen by two cameras at once
  stonetrack detect RUN --start S --end E [--fps 20] [--budget 150]   resumable, chunked
  stonetrack track RUN --start S --end E      16-stone tracks for one end
  stonetrack render RUN ...       review images / video
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

from . import frames, markings
from .geometry import load_calibration, save_calibration, tee

CHUNK = 30.0  # seconds of video per detection file


def _run(path):
    meta = json.load(open(os.path.join(path, "run.json")))
    meta["dir"] = path
    return meta


def cmd_init(a):
    os.makedirs(a.run, exist_ok=True)
    info = frames.probe(a.source)
    layout = json.load(open(a.layout))
    if (info["width"], info["height"]) != (layout["width"], layout["height"]):
        sys.exit(f"layout is {layout['width']}x{layout['height']} but the video is {info['width']}x{info['height']}")
    json.dump(dict(source=os.path.abspath(a.source), layout=layout, video=info), open(os.path.join(a.run, "run.json"), "w"), indent=1)
    print("run ready:", a.run, info)


def cmd_fetch(a):
    os.makedirs(a.run, exist_ok=True)
    out = os.path.join(a.run, "source.mp4")
    subprocess.run(["yt-dlp", "-f", "bv*[height<=1080][ext=mp4]/bv*[height<=1080]", "--no-playlist", "-o", out, a.url], check=True)
    a.source = out
    cmd_init(a)


def cmd_calibrate(a):
    run = _run(a.run)
    size = (run["video"]["width"], run["video"]["height"])
    bg_path = os.path.join(a.run, "background.png")
    if not os.path.exists(bg_path):
        dur = run["video"]["duration"]
        times = np.arange(20, dur - 10, max(30.0, dur / 125))
        bg, n = frames.median_background(run["source"], times, size)
        cv2.imwrite(bg_path, bg)
        print("background from", n, "frames")
    bg = cv2.imread(bg_path)
    cams, report = {}, {}
    overlay = bg.copy()
    for view in run["layout"]["views"]:
        cam, rep, _ = markings.calibrate_view(bg, view)
        if cam.kind == "overhead":  # looks straight down: it stands over the middle of its own picture
            x0, y0, x1, y1 = cam.crop
            cam.ground = cam.to_ice([[(x0 + x1) / 2, (y0 + y1) / 2]])[0]
            cam.height = a.overhead_height
        cams[cam.id], report[cam.id] = cam, rep
        markings.draw_model(overlay, cam)
        print(cam.id, "rms px", {k: round(v, 2) for k, v in rep["residual_rms_px"].items()},
              "| stands at", np.round(cam.ground, 2).tolist(), "height %.2f m" % cam.height)
    save_calibration(os.path.join(a.run, "calibration.json"), cams, dict(layout=run["layout"]["name"], report=report))
    cv2.imwrite(os.path.join(a.run, "calibration-overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])


def cmd_refine(a):
    """Stone-based alignment: overhead camera heights and handle heights from stones at rest."""
    from . import crossview
    from .detect import Detector

    run = _run(a.run)
    cal_path = os.path.join(a.run, "calibration.json")
    cams, meta = load_calibration(cal_path)
    det = Detector(cams, cv2.imread(os.path.join(a.run, "background.png")))
    cache = os.path.join(a.run, "stationary.jsonl")
    if a.frames:  # pre-sampled "t0001230.0.jpg" + ".diff.png" pairs
        import glob

        with open(cache, "w") as fh:
            for f in sorted(glob.glob(os.path.join(a.frames, "t*.jpg"))):
                frame, moved = cv2.imread(f), cv2.imread(f.replace(".jpg", ".diff.png"), 0)
                if moved is None:
                    continue
                keep = [d.as_dict() for d in det.detect(frame)
                        if (moved[max(int(d.px[1]) - 14, 0) : int(d.px[1]) + 15, max(int(d.px[0]) - 14, 0) : int(d.px[0]) + 15] > 25).mean() < 0.03]
                fh.write(json.dumps(dict(t=float(os.path.basename(f)[1:-4]), d=keep)) + "\n")
    else:
        end = a.end or run["video"]["duration"] - 30
        if not crossview.sample_stationary(run, cams, det, np.arange(a.start, end, a.every), cache, a.budget):
            print("budget reached; rerun to continue")
            return 3
    samples = [json.loads(line) for line in open(cache) if line.strip()]
    votes = crossview.orient_overheads(samples, cams)  # sheets differ in how the overheads are mounted
    pairs = crossview.pairs_in_house(samples, cams)
    start_z = {c.id: c.extra.get("handle_height", 0.135) for c in cams.values() if c.kind == "perspective"}
    before = crossview.residual_stats(pairs, cams, start_z)
    # fit on the first half of the game, judge on the second half
    cut = np.median([p["t"] for p in pairs]) if pairs else 0
    train = [p for p in pairs if p["t"] <= cut]
    test = [p for p in pairs if p["t"] > cut]
    handle_z, rep = crossview.fit(train, cams, start_z)
    held_out = crossview.residual_stats(test, cams, handle_z)
    handle_z, rep = crossview.fit(pairs, cams, handle_z)  # final: all pairs
    for cid, z in handle_z.items():
        cams[cid].extra["handle_height"] = z
    meta = {k: v for k, v in meta.items() if k != "cameras"}
    meta["stone_alignment"] = dict(pairs=len(pairs), overhead_orientation_votes=votes, before=before, held_out_after=held_out, fit=rep,
                                   after_all=crossview.residual_stats(pairs, cams, handle_z))
    save_calibration(cal_path, cams, meta)
    print(json.dumps(meta["stone_alignment"], indent=1))


_DET = {}


def _detect_chunk(job):
    """Worker: detections for one CHUNK of video. Returns the chunk start, or None if out of time."""
    run_dir, source, size, t0, fps, deadline = job
    if time.time() > deadline:
        return None
    if "det" not in _DET:
        from .detect import Detector

        cams, _ = load_calibration(os.path.join(run_dir, "calibration.json"))
        _DET["det"] = Detector(cams, cv2.imread(os.path.join(run_dir, "background.png")))
    det = _DET["det"]
    lines = []
    for t, frame in frames.read(source, t0, CHUNK, fps, size):
        fg = {}
        found = det.detect(frame, fg)
        lines.append(json.dumps(dict(t=round(t, 4), d=[d.as_dict() for d in found], fg=fg)))
    path = os.path.join(run_dir, "detections", f"{t0:08.1f}-{fps:g}fps.jsonl")
    open(path + ".tmp", "w").write("\n".join(lines) + "\n")
    os.replace(path + ".tmp", path)
    return t0


def _ranges(a):
    if getattr(a, "ends", None):
        return [(e["start_seconds"], e["end_seconds"]) for e in json.load(open(a.ends))]
    return [(a.start, a.end)]


def cmd_detect(a):
    import multiprocessing as mp

    run = _run(a.run)
    size = (run["video"]["width"], run["video"]["height"])
    os.makedirs(os.path.join(a.run, "detections"), exist_ok=True)
    starts = set()
    for lo, hi in _ranges(a):
        t0 = np.floor(lo / CHUNK) * CHUNK
        while t0 < hi:
            starts.add(float(t0))
            t0 += CHUNK
    todo = [t for t in sorted(starts)
            if a.redo or not os.path.exists(os.path.join(a.run, "detections", f"{t:08.1f}-{a.fps:g}fps.jsonl"))]
    deadline = time.time() + a.budget
    jobs = [(a.run, run["source"], size, t, a.fps, deadline) for t in todo]
    if a.workers > 1 and len(jobs) > 1:
        # fork is cheap on Linux; on macOS forking after OpenCV has been used is unsafe
        with mp.get_context("spawn" if sys.platform == "darwin" else "fork").Pool(a.workers) as pool:
            done = [r for r in pool.imap_unordered(_detect_chunk, jobs) if r is not None]
    else:
        done = [r for r in map(_detect_chunk, jobs) if r is not None]
    left = len(todo) - len(done)
    print(f"{len(done)} chunks done, {left} to go" + ("; rerun to continue" if left else ""))
    return 3 if left else 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="stonetrack")
    sub = p.add_subparsers(dest="cmd", required=True)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("init", "fetch"):
        s = sub.add_parser(name)
        s.add_argument("run")
        s.add_argument("--source" if name == "init" else "--url", required=True)
        s.add_argument("--layout", default=os.path.join(here, "layouts", "wausau.json"))
    s = sub.add_parser("calibrate")
    s.add_argument("run")
    s.add_argument("--overhead-height", type=float, default=4.0)
    s = sub.add_parser("refine")
    s.add_argument("run")
    s.add_argument("--frames", default=None, help="directory of pre-sampled frame/diff pairs")
    s.add_argument("--start", type=float, default=60)
    s.add_argument("--end", type=float, default=None)
    s.add_argument("--every", type=float, default=10)
    s.add_argument("--budget", type=float, default=1e9)
    s = sub.add_parser("detect")
    s.add_argument("run")
    s.add_argument("--start", type=float)
    s.add_argument("--end", type=float)
    s.add_argument("--ends", help="JSON list of ends with start_seconds / end_seconds (hogline2's proposed-ends.json)")
    s.add_argument("--fps", type=float, default=20)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--budget", type=float, default=1e9, help="stop cleanly after this many seconds; rerun to resume")
    s.add_argument("--redo", action="store_true", help="recompute chunks that already exist")
    s = sub.add_parser("ends", help="track every end of a game and check each against fresh overhead detections")
    s.add_argument("run")
    s.add_argument("--ends", required=True)
    s.add_argument("--fps", type=float, default=20)
    s.add_argument("--first-delivery-end", default="B", choices=["A", "B"])
    s.add_argument("--only", type=int, nargs="*", help="end numbers to run (default all)")
    for name in ("track", "render"):
        s = sub.add_parser(name)
        s.add_argument("run")
        s.add_argument("--start", type=float, required=True)
        s.add_argument("--end", type=float, required=True)
        s.add_argument("--fps", type=float, default=20)
        s.add_argument("--first-delivery-end", default="B", choices=["A", "B"])
        s.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.cmd in ("track", "render", "ends"):
        from . import track as track_mod

        return getattr(track_mod, "cmd_" + a.cmd)(a)
    return globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

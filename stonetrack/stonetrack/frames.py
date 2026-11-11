"""Video access through ffmpeg: single frames, timed frame streams, and the static background."""
from __future__ import annotations

import json
import subprocess

import numpy as np


def probe(src):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", src],
                         capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    st = data["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return dict(width=int(st["width"]), height=int(st["height"]), fps=float(num) / float(den),
                duration=float(data["format"]["duration"]))


def read(src, start, duration, fps, size):
    """Yield (seconds, BGR frame) at a fixed rate from `start`. Frames are fresh arrays."""
    w, h = size
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", src,
           "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    try:
        i = 0
        while True:
            raw = proc.stdout.read(w * h * 3)
            if len(raw) < w * h * 3:
                break
            yield start + i / fps, np.frombuffer(raw, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        proc.kill()
        proc.wait()


def grab(src, t, size):
    for _, frame in read(src, t, 0.5, 2, size):
        return frame.copy()
    return None


def median_background(src, times, size):
    """Per-pixel temporal median over frames spread through the game: people and stones vanish,
    paint stays. Computed in strips to keep memory modest."""
    frames = [f for f in (grab(src, t, size) for t in times) if f is not None]
    stack = np.stack(frames)
    med = np.empty(stack.shape[1:], np.uint8)
    for y in range(0, stack.shape[1], 120):
        med[y : y + 120] = np.median(stack[:, y : y + 120], axis=0).astype(np.uint8)
    return med, len(frames)

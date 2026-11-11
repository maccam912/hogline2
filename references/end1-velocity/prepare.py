"""Freeze one common initialization schedule before either comparison runs."""

import json
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data/backups/end1-tracking-before-velocity-20261108"
PREVIOUS = ARCHIVE / "2ae825c04e3547be9b9a892fdc622282"
TRACKING = ROOT / "data/recordings/eDJdwj0JqBY/tracking"


def main():
    old = json.loads((PREVIOUS / "job.json").read_text())
    job = {
        **old,
        "calibration": json.loads((TRACKING / "calibration.json").read_text()),
        "duration": 120.0,
        "fps": 3.0,
        "comparison_id": "end1-raw-velocity-20261108",
        "initialization_source": str(PREVIOUS),
        "raw_comparison": True,
    }
    grouped = defaultdict(list)
    for line in (PREVIOUS / "observations.jsonl").read_text().splitlines():
        row = json.loads(line)
        for observation in row["observations"]:
            grouped[observation["view_id"], observation["stone_id"]].append((row, observation))
    capture = cv2.VideoCapture(job["source_path"])
    windows = []
    for (vid, name), samples in sorted(grouped.items()):
        row, obs = samples[0]
        view = next(v for v in job["calibration"]["views"] if v["id"] == vid)
        x0, y0, x1, y1 = view["crop"]
        capture.set(cv2.CAP_PROP_POS_MSEC, row["source_seconds"] * 1000)
        ok, bgr = capture.read()
        assert ok
        hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        packed = np.load(PREVIOUS / row["masks"])[f"{vid}__{name}"]
        mask = np.unpackbits(packed)[: (y1 - y0) * (x1 - x0)].reshape(y1 - y0, x1 - x0).astype(bool)
        color = name.split("_")[0]
        colored = (
            (
                ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
                if color == "red"
                else ((hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 40))
            )
            & (hsv[:, :, 1] > 95)
            & mask
        )
        yy, xx = np.where(colored)
        if not len(xx):
            yy, xx = np.where(mask)
        point = [float(np.median(xx)), float(np.median(yy))]
        windows.append(
            dict(
                stone_id=name,
                color=color,
                view_id=vid,
                kind=view["kind"],
                point=point,
                first_frame=row["frame"],
                last_frame=min(359, samples[-1][0]["frame"] + 2),
                seed_source_seconds=row["source_seconds"],
                window_evidence="Archived first-to-last observations plus two frames; internal gaps retained",
            )
        )
    capture.release()
    job["track_windows"] = windows
    output = ROOT / "data/experiments/end1-velocity-20261108"
    output.mkdir(exist_ok=True, parents=True)
    (output / "common-job.json").write_text(json.dumps(job, indent=2))
    runs = {}
    for backend in ["sam2.1-mlx-reinitialized", "sam3.1-mlx-multiplex"]:
        folder = TRACKING / uuid.uuid4().hex
        folder.mkdir()
        (folder / "job.json").write_text(json.dumps({**job, "backend": backend}, indent=2))
        (folder / "status.json").write_text(
            json.dumps(
                dict(
                    state="queued",
                    stage="Waiting for comparison worker",
                    raw_comparison=True,
                    backend=backend,
                )
            )
        )
        runs[backend] = str(folder)
    (output / "runs.json").write_text(json.dumps(runs, indent=2))
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()

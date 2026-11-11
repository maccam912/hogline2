"""Audit the normal tracker independently of the ungated mask experiment."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/sam2-granite-20261108"
parser = argparse.ArgumentParser()
parser.add_argument("--run-id")
parser.add_argument("--output", type=Path, default=OUT)
args = parser.parse_args()
run_id = args.run_id or json.loads((OUT / "production-run.json").read_text())["run_id"]
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
folder = ROOT / "data/recordings/eDJdwj0JqBY/tracking" / run_id
status = json.loads((folder / "status.json").read_text())
assert status["state"] == "complete", status
job = json.loads((folder / "job.json").read_text())
rows = [json.loads(s) for s in (folder / "observations.jsonl").read_text().splitlines()]
counts = Counter()
duplicates = []
for row in rows:
    for obs in row["observations"]:
        if obs.get("accepted"):
            counts[obs["stone_id"], obs["view_id"]] += 1
    mask_path = folder / row["masks"] if row.get("masks") else None
    if not mask_path or not mask_path.exists():
        continue
    with np.load(mask_path) as masks:
        for view in job["calibration"]["views"]:
            keys = [k for k in masks if k.startswith(view["id"] + "__red_")]
            if len(keys) != 2:
                continue
            a, b = [np.unpackbits(masks[k]).astype(bool) for k in keys]
            intersection = int(np.logical_and(a, b).sum())
            smaller = int(min(a.sum(), b.sum()))
            if smaller and intersection / smaller > 0.5:
                duplicates.append(dict(frame=row["frame"], view=view["id"], overlap=intersection / smaller))
summary = dict(
    run_id=run_id,
    frames=len(rows),
    elapsed_seconds=status.get("elapsed_seconds"),
    ids=sorted({s["id"] for r in rows for s in r["stones"]}),
    accepted_by_camera={f"{s}/{v}": n for (s, v), n in counts.items()},
    same_view_red_masks_over_50_percent_overlap=duplicates,
    final_stones=rows[-1]["stones"],
)
(OUT / "production-summary.json").write_text(json.dumps(summary, indent=2))
frames = [45, 180, 230, 320, 350, 359]
cap = cv2.VideoCapture(job["source_path"])
sheet = Image.new("RGB", (1000, len(frames) * 330), "white")
draw = ImageDraw.Draw(sheet)
for i, f in enumerate(frames):
    seconds = job["source_start"] + f / job["fps"]
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, bgr = cap.read()
    assert ok
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = rgb.copy()
    mask_path = folder / f"masks/{f:06}.npz"
    if mask_path.exists():
        with np.load(mask_path) as masks:
            for v in job["calibration"]["views"]:
                x0, y0, x1, y1 = v["crop"]
                crop = im[y0:y1, x0:x1]
                for stone, c in [
                    ("red_1", [255, 20, 190]),
                    ("red_2", [0, 240, 255]),
                    ("yellow_1", [255, 230, 0]),
                ]:
                    key = v["id"] + "__" + stone
                    if key not in masks:
                        continue
                    mask = (
                        np.unpackbits(masks[key], count=(y1 - y0) * (x1 - x0))
                        .reshape(y1 - y0, x1 - x0)
                        .astype(bool)
                    )
                    crop[mask] = (crop[mask] * 0.4 + np.array(c) * 0.6).astype("uint8")
                    yy, xx = np.where(mask)
                    if len(xx):
                        cv2.putText(
                            crop,
                            stone,
                            (int(xx.mean()), int(yy.min()) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            c,
                            1,
                        )
    # Perspective A, including all three stones at the final delivery.
    for j, source in enumerate([rgb, im]):
        tile = Image.fromarray(source).crop((0, 420, 724, 950)).resize((450, 300))
        sheet.paste(tile, (j * 500, i * 330 + 25))
        draw.text(
            (j * 500 + 5, i * 330 + 5),
            f"{seconds:.2f}s — " + ("source" if j == 0 else "normal tracker"),
            fill="black",
        )
sheet.save(OUT / "production-review.jpg")
cap.release()
print(json.dumps(summary, indent=2))

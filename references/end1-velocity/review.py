"""Render the same source pixels and unmodified saved masks for visual review."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/end1-velocity-20261108"


def main():
    job = json.loads((OUT / "common-job.json").read_text())
    runs = json.loads((OUT / "runs.json").read_text())
    views = {v["id"]: v for v in job["calibration"]["views"]}
    cap = cv2.VideoCapture(job["source_path"])
    cases = [
        (389 + 1 / 3, "perspective_b", [1490, 440, 1800, 750], "First delivery: occlusion"),
        (436, "perspective_a", [150, 460, 500, 990], "Stationary red: stray mask pixels"),
        (420, "perspective_a", [325, 465, 475, 550], "First red stone at rest"),
        (440, "perspective_a", [200, 635, 430, 830], "Yellow approach: SAM2 reset"),
        (470, "perspective_a", [350, 470, 510, 545], "Yellow at rest, perspective"),
        (470, "overhead_a", [810, 7, 1010, 135], "Yellow at rest, overhead"),
        (497 + 2 / 3, "overhead_a", [790, 7, 1030, 170], "Final contact"),
    ]
    sheet = Image.new("RGB", (1260, len(cases) * 310 + 40), "#f4f6f5")
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(["Source", "SAM2 reinitialized", "SAM3.1 temporal"]):
        draw.text((column * 420 + 12, 12), title, fill="black")
    for row, (seconds, vid, box, title) in enumerate(cases):
        frame = round((seconds - job["source_start"]) * job["fps"])
        if not all((Path(path) / f"masks/{frame:06}.npz").exists() for path in runs.values()):
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, bgr = cap.read()
        assert ok
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        variants = [rgb]
        x0, y0, x1, y1 = views[vid]["crop"]
        for path in runs.values():
            overlay = rgb.copy()
            with np.load(Path(path) / f"masks/{frame:06}.npz") as packed:
                for key in packed.files:
                    if not key.startswith(vid + "__"):
                        continue
                    mask = (
                        np.unpackbits(packed[key])[: (y1 - y0) * (x1 - x0)]
                        .reshape(y1 - y0, x1 - x0)
                        .astype(bool)
                    )
                    color = np.array([255, 50, 80] if "__red" in key else [255, 210, 10])
                    crop = overlay[y0:y1, x0:x1]
                    crop[mask] = (crop[mask] * 0.45 + color * 0.55).astype(np.uint8)
                    yy, xx = np.where(mask)
                    if len(xx) and views[vid]["kind"] == "perspective":
                        anchor = (int(np.median(xx[yy >= np.quantile(yy, 0.94)])), int(yy.max()))
                        cv2.drawMarker(crop, anchor, (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
            variants.append(overlay)
        for column, variant in enumerate(variants):
            tile = Image.fromarray(variant).crop(box)
            scale = min(404 / tile.width, 265 / tile.height)
            tile = tile.resize((round(tile.width * scale), round(tile.height * scale)))
            sheet.paste(tile, (column * 420 + 8, row * 310 + 65))
            draw.text((column * 420 + 8, row * 310 + 43), f"{seconds:.2f}s: {title}", fill="black")
    cap.release()
    sheet.save(OUT / "mask-comparison.jpg")


if __name__ == "__main__":
    main()

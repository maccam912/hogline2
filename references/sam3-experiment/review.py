"""Reproducible spot checks and visual comparisons from saved benchmark masks."""

import json
from pathlib import Path

import cv2
import numpy as np
from benchmark import CASES, EXTRA_CASES, OUT, ROOT, draw, frames
from PIL import Image, ImageDraw, ImageFont

BACKENDS = ["sam2", "sam3-detect", "sam31-detect", "sam31-memory"]
LABELS = ["Original", "SAM 2.1 points", "SAM3 text + IoU IDs", "SAM3.1 text + IoU IDs", "SAM3.1 memory"]


def masks_for(backend, case, frame):
    path = OUT / f"{backend}-{case}" / f"masks-{frame:04}.npz"
    with np.load(path) as packed:
        h, w = packed["shape"]
        ids = [int(k[4:]) for k in packed.files if k.startswith("obj_")]
        masks = [np.unpackbits(packed[f"obj_{i}"])[: h * w].reshape(h, w).astype(bool) for i in ids]
    return masks, ids


def bbox(mask):
    ys, xs = np.where(mask)
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] if len(xs) else [0, 0, 0, 0]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return float(intersection / union) if union else 0.0


def main():
    annotations = json.loads((Path(__file__).parent / "review-points.json").read_text())
    # Decode review images again so this script also works in a clean output folder.
    for case in CASES + EXTRA_CASES:
        wanted = {a["frame"] for a in annotations["samples"] if a["case"] == case["name"]}
        duration = 2 if case["name"] == "small-stone" else 8
        for i, rgb in frames(case, 6, duration):
            if i in wanted:
                Image.fromarray(rgb).save(OUT / f"{case['name']}-raw-{i:04}.png")
    summary = {"method": annotations["method"], "backends": {}}
    for backend in BACKENDS:
        rows = []
        cases = []
        for case in CASES + EXTRA_CASES:
            result = json.loads((OUT / f"{backend}-{case['name']}" / "results.json").read_text())
            cases.append(
                {
                    "name": case["name"],
                    "frames": len(result["frames"]),
                    "elapsed_seconds": result["elapsed_seconds"],
                    "peak_metal_bytes": result["peak_metal_bytes"],
                    "nonempty_frames": sum(any(area > 0 for area in f["areas"]) for f in result["frames"]),
                    "median_inference_seconds": float(np.median([f["seconds"] for f in result["frames"]])),
                    "initialization": result.get("initialization", "text"),
                }
            )
        for a in annotations["samples"]:
            masks, ids = masks_for(backend, a["case"], a["frame"])
            overlaps = [iou(bbox(mask), a["box"]) for mask in masks]
            best = int(np.argmax(overlaps)) if overlaps else None
            if best is not None and overlaps[best] > 0.1:
                probes = [bool(masks[best][y, x]) for x, y in a["body_points"]]
                object_id, overlap = ids[best], overlaps[best]
            else:
                probes, object_id, overlap = [False] * len(a["body_points"]), None, 0.0
            rows.append(
                {
                    "case": a["case"],
                    "frame": a["frame"],
                    "matched_id": object_id,
                    "approx_box_iou": overlap,
                    "body_probe_hits": sum(probes),
                    "body_probe_count": len(probes),
                }
            )
        summary["backends"][backend] = {
            "cases": cases,
            "samples": rows,
            "body_probe_hits": sum(r["body_probe_hits"] for r in rows),
            "body_probe_count": sum(r["body_probe_count"] for r in rows),
            "frames_with_all_body_probes": sum(r["body_probe_hits"] == r["body_probe_count"] for r in rows),
            "sample_count": len(rows),
        }
    (OUT / "review-results.json").write_text(json.dumps(summary, indent=2))

    # Four representative snapshots, with identical source pixels and crops in every column.
    chosen = [("small-stone", 0), ("sweepers", 24), ("opposite", 24), ("contact", 36)]
    tile_w, tile_h = 240, 190
    sheet = Image.new("RGB", (tile_w * 5, tile_h * len(chosen) + 30), "#eeeeee")
    painter = ImageDraw.Draw(sheet)
    for j, label in enumerate(LABELS):
        painter.text((j * tile_w + 8, 8), label, fill="black")
    for row, (case, frame) in enumerate(chosen):
        a = next(a for a in annotations["samples"] if a["case"] == case and a["frame"] == frame)
        raw = np.array(Image.open(OUT / f"{case}-raw-{frame:04}.png"))
        x1, y1, x2, y2 = a["box"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        crop = (cx - 60, max(0, cy - 40), cx + 60, max(0, cy - 40) + 80)
        for col, backend in enumerate([None, *BACKENDS]):
            if backend:
                masks, ids = masks_for(backend, case, frame)
                rendered = draw(raw, masks, ids, "")
            else:
                rendered = raw
            tile = Image.fromarray(rendered).crop(crop).resize((240, 160))
            sheet.paste(tile, (col * tile_w, 30 + row * tile_h))
            painter.text((col * tile_w + 8, 192 + row * tile_h), f"{case}, frame {frame}", fill="black")
    sheet.save(OUT / "comparison.jpg")

    # Reproduce the exact saved failure, rather than simulating a handle-only mask.
    production = ROOT / "data/recordings/eDJdwj0JqBY/tracking/2ae825c04e3547be9b9a892fdc622282"
    if not production.exists():
        production = ROOT / "data/backups/end1-tracking-before-velocity-20261108" / production.name
    observations = [json.loads(line) for line in (production / "observations.jsonl").read_text().splitlines()]
    observation = min(observations, key=lambda r: abs(r["source_seconds"] - 471))
    raw = np.array(Image.open(OUT / "small-stone-raw-0000.png"))
    with np.load(production / observation["masks"]) as packed:
        mask = np.unpackbits(packed["perspective_a__yellow_1"])[: 990 * 724].reshape(990, 724).astype(bool)
    images = [raw, draw(raw, [mask], [1], "")]
    for backend in BACKENDS:
        masks, ids = masks_for(backend, "small-stone", 0)
        images.append(draw(raw, masks, ids, ""))
    labels = [
        "Original",
        "Saved SAM2 run",
        "Fresh SAM2 prompt",
        "SAM3 detection",
        "SAM3.1 detection",
        "SAM3.1 tracking",
    ]
    sheet = Image.new("RGB", (1440, 210), "white")
    painter = ImageDraw.Draw(sheet)
    for i, (rgb, label) in enumerate(zip(images, labels, strict=True)):
        sheet.paste(Image.fromarray(rgb).crop((355, 460, 475, 550)).resize((240, 180)), (240 * i, 30))
        painter.text((240 * i + 8, 8), label, fill="black", font=ImageFont.load_default(size=15))
    sheet.save(OUT / "mask-failure-comparison.jpg")

    # Full 32-second comparison at the actual six sampled frames per second.
    writer = cv2.VideoWriter(str(OUT / "comparison-raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 6, (1440, 540))
    try:
        for case in CASES:
            for i, raw in frames(case, 6, 8):
                canvas = Image.new("RGB", (1440, 540), "#20242b")
                painter = ImageDraw.Draw(canvas)
                for col, backend in enumerate([None, "sam2", "sam3-detect", "sam31-memory"]):
                    rendered = raw
                    if backend:
                        masks, ids = masks_for(backend, case["name"], i)
                        rendered = draw(raw, masks, ids, "")
                    tile = Image.fromarray(rendered)
                    tile.thumbnail((360, 495))
                    canvas.paste(tile, (col * 360 + (360 - tile.width) // 2, 40))
                    painter.text(
                        (col * 360 + 8, 8), f"{backend or 'original'} | {case['name']}", fill="white"
                    )
                    painter.text((col * 360 + 8, 23), f"Source {case['start'] + i / 6:.2f}s", fill="white")
                writer.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    print(
        json.dumps(
            {b: {k: v for k, v in d.items() if k != "samples"} for b, d in summary["backends"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

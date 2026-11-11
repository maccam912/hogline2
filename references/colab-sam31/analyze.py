"""Review downloaded CUDA masks against the existing Mac experiment artifacts."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "data/experiments/colab-sam31-20261110"
INPUT = EXPERIMENT / "input"
RESULTS = EXPERIMENT / "results"
MAC = ROOT / "data/experiments/sam3-20261108"
ANNOTATIONS = json.loads((INPUT / "review-points.json").read_text())["samples"]
CASES = json.loads((INPUT / "input-manifest.json").read_text())["cases"]


def mask_at(directory, frame):
    with np.load(directory / f"masks-{frame:04}.npz") as packed:
        h, w = map(int, packed["shape"])
        return (
            np.unpackbits(packed["obj_1"])[: h * w].reshape(h, w).astype(bool)
            if "obj_1" in packed
            else np.zeros((h, w), bool)
        )


def overlay(rgb, mask):
    rgb = rgb.copy()
    rgb[mask] = (rgb[mask] * 0.55 + np.array([0, 230, 130]) * 0.45).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (0, 230, 130), 1)
    return rgb


def main():
    variants = {
        "Mac SAM2.1": lambda name: MAC / f"sam2-{name}",
        "Mac SAM3.1": lambda name: MAC / f"sam31-memory-{name}",
        "L4 points (failed)": lambda name: RESULTS / "warm" / name,
        "L4 text batch 1": lambda name: RESULTS / "text-batch1" / name,
        "L4 text batch 4": lambda name: RESULTS / "text" / name,
    }
    scores = {}
    for label, directory in variants.items():
        rows = []
        for annotation in ANNOTATIONS:
            mask = mask_at(directory(annotation["case"]), annotation["frame"])
            probes = [bool(mask[y, x]) for x, y in annotation["body_points"]]
            rows.append(
                dict(
                    case=annotation["case"],
                    frame=annotation["frame"],
                    body_probe_hits=sum(probes),
                    body_probe_count=len(probes),
                    area=int(mask.sum()),
                )
            )
        scores[label] = dict(
            body_probe_hits=sum(r["body_probe_hits"] for r in rows),
            body_probe_count=sum(r["body_probe_count"] for r in rows),
            samples=rows,
        )
    timings = {}
    for backend in ["sam2", "sam31-memory"]:
        times = [
            frame["seconds"]
            for case in CASES
            for frame in json.loads((MAC / f"{backend}-{case['name']}/results.json").read_text())["frames"]
        ]
        timings[backend] = dict(
            frames=len(times),
            inference_seconds=sum(times),
            mean_seconds_per_frame=float(np.mean(times)),
            median_seconds_per_frame=float(np.median(times)),
        )
    for name in ["text-summary-batch1.json", "text-summary-batch4.json"]:
        path = RESULTS / name
        if not path.exists():
            continue
        cases = json.loads(path.read_text())
        frames = sum(c["frames"] for c in cases)
        seconds = sum(c["seed_seconds"] + c["propagation_seconds"] for c in cases)
        timings[name] = dict(
            frames=frames,
            inference_seconds=seconds,
            mean_seconds_per_frame=seconds / frames,
            frames_per_second=frames / seconds,
            cases=cases,
        )
    report = dict(
        timings=timings,
        quality=scores,
        limitation="Body-probe coverage is a spot check, not mask IoU or a segmentation accuracy score.",
    )
    (EXPERIMENT / "analysis.json").write_text(json.dumps(report, indent=2))

    # Selected source crops, with the target mask and exact same pixels in all columns.
    selected = [("small-stone", 0), ("clear", 24), ("sweepers", 24), ("opposite", 24), ("contact", 36)]
    labels = ["Original", "Mac SAM2.1 (fresh)", "Colab L4 SAM3.1 (text)"]
    sheet = Image.new("RGB", (1080, len(selected) * 240 + 40), "white")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(labels):
        draw.text((col * 360 + 10, 10), label, fill="black")
    for row, (name, frame) in enumerate(selected):
        cap = cv2.VideoCapture(str(INPUT / f"{name}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        cap.release()
        assert ok
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        a = next(a for a in ANNOTATIONS if a["case"] == name and a["frame"] == frame)
        x1, y1, x2, y2 = a["box"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        crop = (cx - 60, max(0, cy - 36), cx + 60, max(0, cy - 36) + 72)
        masks = [None, mask_at(MAC / f"sam2-{name}", frame), mask_at(RESULTS / "text-batch1" / name, frame)]
        for col, mask in enumerate(masks):
            rendered = rgb if mask is None else overlay(rgb, mask)
            sheet.paste(Image.fromarray(rendered).crop(crop).resize((360, 216)), (col * 360, row * 240 + 40))
            hits = sum(bool(mask[y, x]) for x, y in a["body_points"]) if mask is not None else None
            label = f"{name}, frame {frame}" + (f" | body probes {hits}/3" if hits is not None else "")
            draw.text((col * 360 + 10, row * 240 + 259), label, fill="black")
    sheet.save(EXPERIMENT / "comparison.jpg", quality=95)
    compact = Image.new("RGB", (1080, 520), "white")
    compact.paste(sheet.crop((0, 0, 1080, 280)), (0, 0))
    compact.paste(sheet.crop((0, 1000, 1080, 1240)), (0, 280))
    compact.save(EXPERIMENT / "comparison-summary.jpg", quality=95)
    print(
        json.dumps(
            dict(
                timings={
                    key: {k: v for k, v in value.items() if k != "cases"} for key, value in timings.items()
                },
                coverage={k: [v["body_probe_hits"], v["body_probe_count"]] for k, v in scores.items()},
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

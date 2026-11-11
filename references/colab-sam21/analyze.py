"""Review downloaded SAM2.1 CUDA results against the prior Mac/CUDA runs."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "data/experiments/colab-sam21-20261110"
RESULTS = EXPERIMENT / "results"
PREVIOUS = ROOT / "data/experiments/colab-sam31-20261110"
INPUT = PREVIOUS / "input"
MAC = ROOT / "data/experiments/sam3-20261108"
CASES = json.loads((INPUT / "input-manifest.json").read_text())["cases"]
ANNOTATIONS = json.loads((INPUT / "review-points.json").read_text())["samples"]


def mask_at(directory, index):
    with np.load(directory / f"masks-{index:04}.npz") as packed:
        h, w = map(int, packed["shape"])
        assert packed["obj_1"].size == (h * w + 7) // 8
        return np.unpackbits(packed["obj_1"])[: h * w].reshape(h, w).astype(bool)


def render(rgb, mask):
    rgb = rgb.copy()
    rgb[mask] = (rgb[mask] * 0.55 + np.array([0, 230, 130]) * 0.45).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (0, 230, 130), 1)
    return rgb


def main():
    summary = json.loads((RESULTS / "summary.json").read_text())
    timings = {}
    counts = {}
    for model in ["base_plus", "large"]:
        for trial in ["cold", "warm"]:
            reports = [r for r in summary["reports"] if r["model"] == model and r["trial"] == trial]
            assert sum(r["frames"] for r in reports) == 204
            count = 0
            for case in CASES:
                directory = RESULTS / model / trial / case["name"]
                assert len(list(directory.glob("masks-*.npz"))) == case["frames"]
                rows = json.loads((directory / "results.json").read_text())["observations"]
                for i, row in enumerate(rows):
                    mask = mask_at(directory, i)
                    assert mask.shape == (case["height"], case["width"])
                    assert int(mask.sum()) == row["area"]
                    assert abs(row["source_seconds"] - case["start"] - i / case["fps"]) < 1e-8
                    count += 1
            total = sum(r["inference_seconds"] for r in reports)
            timings[f"{model}-{trial}"] = dict(
                frames=count,
                seconds=total,
                seconds_per_frame=total / count,
                fps=count / total,
                peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in reports),
            )
            counts[f"{model}-{trial}"] = count
    hashes = json.loads((RESULTS / "source-hashes.json").read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((RESULTS / "runtime-source" / name).read_bytes()).hexdigest() == expected
    variants = {
        "Mac SAM2.1": lambda name: MAC / f"sam2-{name}",
        "L4 SAM3.1 text": lambda name: PREVIOUS / "results/text-batch1" / name,
        "L4 SAM2.1 base+": lambda name: RESULTS / "base_plus/warm" / name,
        "L4 SAM2.1 large": lambda name: RESULTS / "large/warm" / name,
    }
    review = {}
    for label, directory in variants.items():
        rows = []
        for a in ANNOTATIONS:
            mask = mask_at(directory(a["case"]), a["frame"])
            rows.append(
                dict(
                    case=a["case"],
                    frame=a["frame"],
                    hits=sum(bool(mask[y, x]) for x, y in a["body_points"]),
                    count=len(a["body_points"]),
                )
            )
        review[label] = dict(
            hits=sum(r["hits"] for r in rows), count=sum(r["count"] for r in rows), samples=rows
        )
    analysis = dict(
        timings=timings,
        quality=review,
        validated_frame_counts=counts,
        limitation="Body probes are spot checks, not pixel-level mask accuracy.",
    )
    (EXPERIMENT / "analysis.json").write_text(json.dumps(analysis, indent=2))
    selected = [("small-stone", 0), ("clear", 24), ("sweepers", 24), ("opposite", 24), ("contact", 36)]
    labels = ["Original", *variants]
    sheet = Image.new("RGB", (1500, 1040), "white")
    draw = ImageDraw.Draw(sheet)
    for j, label in enumerate(labels):
        draw.text((j * 300 + 8, 10), label, fill="black")
    for row, (name, index) in enumerate(selected):
        cap = cv2.VideoCapture(str(INPUT / f"{name}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = cap.read()
        cap.release()
        assert ok
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        a = next(a for a in ANNOTATIONS if a["case"] == name and a["frame"] == index)
        x1, y1, x2, y2 = a["box"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        crop = (cx - 60, max(0, cy - 36), cx + 60, max(0, cy - 36) + 72)
        for col, directory in enumerate([None, *variants.values()]):
            mask = mask_at(directory(name), index) if directory else None
            tile = rgb if mask is None else render(rgb, mask)
            sheet.paste(Image.fromarray(tile).crop(crop).resize((300, 180)), (col * 300, 40 + row * 200))
            hits = sum(bool(mask[y, x]) for x, y in a["body_points"]) if mask is not None else None
            label = f"{name} f{index}" + (f" | probes {hits}/3" if hits is not None else "")
            draw.text((col * 300 + 8, 222 + row * 200), label, fill="black")
    sheet.save(EXPERIMENT / "comparison.jpg", quality=95)
    compact = Image.new("RGB", (900, 440), "white")
    compact_draw = ImageDraw.Draw(compact)
    for col, source_col in enumerate([0, 3, 4]):
        compact_draw.text((col * 300 + 8, 10), labels[source_col], fill="black")
        for row, source_row in enumerate([0, 4]):
            compact.paste(
                sheet.crop(
                    (source_col * 300, 40 + source_row * 200, (source_col + 1) * 300, 240 + source_row * 200)
                ),
                (col * 300, 40 + row * 200),
            )
    compact.save(EXPERIMENT / "comparison-summary.jpg", quality=95)
    print(
        json.dumps(
            dict(
                timings=timings,
                quality={k: [v["hits"], v["count"]] for k, v in review.items()},
                validated=counts,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

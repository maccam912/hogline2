"""Source / SAM2 / SAM3.1 highlight replay from saved, unmodified model masks."""

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/end1-velocity-20261108"


def main():
    job = json.loads((OUT / "common-job.json").read_text())
    runs = json.loads((OUT / "runs.json").read_text())
    views = {v["id"]: v for v in job["calibration"]["views"]}
    cap = cv2.VideoCapture(job["source_path"])
    cases = [
        (17, 50, "perspective_b", None, "First delivery: occlusion and wrong-object masks"),
        (171, 177, "perspective_a", [150, 460, 500, 990], "Stationary red: SAM3.1 stray mask pixels"),
        (180, 209, "perspective_a", None, "Yellow approach: full-body reinitialization"),
        (339, 359, "overhead_a", [790, 7, 1030, 190], "Overhead contact"),
    ]
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
            "1536x800",
            "-r",
            str(job["fps"]),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(OUT / "comparison.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 19)
    count = 0
    try:
        for first, last, vid, zoom, title in cases:
            x0, y0, x1, y1 = views[vid]["crop"]
            box = zoom or [x0, y0, x1, y1]
            cap.set(cv2.CAP_PROP_POS_MSEC, (job["source_start"] + first / job["fps"]) * 1000)
            step = round(cap.get(cv2.CAP_PROP_FPS) / job["fps"])
            for frame in range(first, last + 1):
                ok, bgr = cap.read()
                assert ok
                for _ in range(step - 1):
                    cap.grab()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                variants = [rgb]
                for path in runs.values():
                    overlay = rgb.copy()
                    with np.load(Path(path) / f"masks/{frame:06}.npz") as packed:
                        for key in packed.files:
                            if not key.startswith(vid + "__"):
                                continue
                            mask = np.unpackbits(packed[key])[: (y1 - y0) * (x1 - x0)]
                            mask = mask.reshape(y1 - y0, x1 - x0).astype(bool)
                            color = np.array([255, 50, 80] if "__red" in key else [255, 210, 10])
                            crop = overlay[y0:y1, x0:x1]
                            crop[mask] = (crop[mask] * 0.45 + color * 0.55).astype(np.uint8)
                    variants.append(overlay)
                sheet = Image.new("RGB", (1536, 800), "#172a29")
                draw = ImageDraw.Draw(sheet)
                for column, (name, variant) in enumerate(
                    zip(["Source", "SAM2 reinitialized", "SAM3.1 temporal"], variants, strict=True)
                ):
                    tile = Image.fromarray(variant).crop(box)
                    scale = min(500 / tile.width, 690 / tile.height)
                    tile = tile.resize((round(tile.width * scale), round(tile.height * scale)))
                    sheet.paste(tile, (column * 512 + 6, 52))
                    draw.text((column * 512 + 12, 14), name, font=font, fill="white")
                seconds = job["source_start"] + frame / job["fps"]
                draw.text((12, 750), f"Source {seconds:.2f}s | {title}", font=font, fill="white")
                encoder.stdin.write(sheet.tobytes())
                count += 1
        encoder.stdin.close()
        assert encoder.wait() == 0
    finally:
        cap.release()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    print(f"Rendered {count} frames, {count / job['fps']:.2f}s")
    # Serve the self-contained reports through the app's existing run-artifact endpoint.
    for path in runs.values():
        folder = Path(path)
        for name in ("comparison.html", "component-comparison.html", "comparison.mp4"):
            shutil.copy2(OUT / name, folder / name)
        manifest = json.loads((folder / "manifest.json").read_text())
        manifest["comparison_report"] = True
        temporary = folder / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(folder / "manifest.json")


if __name__ == "__main__":
    main()

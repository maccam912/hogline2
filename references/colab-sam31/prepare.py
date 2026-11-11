"""Package lossless, pixel-verified clips and existing baselines for Colab."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data/experiments/colab-sam31-20261110"
spec = importlib.util.spec_from_file_location(
    "mac_benchmark", ROOT / "references/sam3-experiment/benchmark.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def main():
    bundle = DEST / "input"
    bundle.mkdir(parents=True, exist_ok=True)
    cases = []
    for original in benchmark.CASES + benchmark.EXTRA_CASES:
        case = dict(original)
        duration = 2 if case["name"] == "small-stone" else 8
        samples = list(benchmark.frames(case, 6, duration))
        height, width = samples[0][1].shape[:2]
        video = bundle / f"{case['name']}.mp4"
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                "6",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264rgb",
                "-crf",
                "0",
                "-preset",
                "fast",
                str(video),
            ],
            stdin=subprocess.PIPE,
        )
        for _, rgb in samples:
            process.stdin.write(rgb.tobytes())
        process.stdin.close()
        if process.wait():
            raise RuntimeError("Lossless encoding failed")
        cap = cv2.VideoCapture(str(video))
        for _, rgb in samples:
            ok, decoded = cap.read()
            if not ok or not np.array_equal(rgb, cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)):
                raise RuntimeError(f"Lossless pixel verification failed: {video}")
        assert not cap.read()[0]
        cap.release()
        case.update(
            frames=len(samples),
            fps=6,
            width=width,
            height=height,
            sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
            pixel_verified=True,
        )
        cases.append(case)
        for backend in ["sam2", "sam31-memory"]:
            source = benchmark.OUT / f"{backend}-{case['name']}"
            target = bundle / "baselines" / source.name
            target.mkdir(parents=True, exist_ok=True)
            for path in [source / "results.json", *source.glob("masks-*.npz")]:
                shutil.copy2(path, target / path.name)
        print(f"Prepared {case['name']}: {len(samples)} exact frames", flush=True)
    annotations = ROOT / "references/sam3-experiment/review-points.json"
    shutil.copy2(annotations, bundle / annotations.name)
    with benchmark.SOURCE.open("rb") as source_file:
        source_sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()
    manifest = dict(
        cases=cases,
        source_recording="eDJdwj0JqBY",
        source_sha256=source_sha256,
        description="Lossless RGB clips; all 204 decoded frames equal original source crops.",
    )
    (bundle / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
    with zipfile.ZipFile(DEST / "hogline-sam31-input.zip", "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(bundle))
    print(DEST / "hogline-sam31-input.zip")


if __name__ == "__main__":
    main()

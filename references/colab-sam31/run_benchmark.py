"""Execute inside Colab after loading predictor and uploading the prepared bundle.

%run -i /content/run_benchmark.py
Requires `predictor` and `model_load_seconds` from the setup cells.
"""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image, ImageDraw

predictor = globals()["predictor"]
model_load_seconds = globals()["model_load_seconds"]

INPUT = Path("/content/hogline-benchmark")
OUTPUT = Path("/content/hogline-sam31-results")
OUTPUT.mkdir(exist_ok=True)
INPUT_MANIFEST = json.loads((INPUT / "input-manifest.json").read_text())
ANNOTATIONS = json.loads((INPUT / "review-points.json").read_text())


def masks_from_output(output):
    return {
        int(oid): np.asarray(mask, dtype=bool).copy()
        for oid, mask in zip(output["out_obj_ids"], output["out_binary_masks"], strict=True)
    }


def run_case(case, trial):
    name = case["name"]
    path = INPUT / f"{name}.mp4"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
    torch.cuda.synchronize()
    start = time.perf_counter()
    session = predictor.handle_request(dict(type="start_session", resource_path=str(path)))["session_id"]
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    x, y = case["point"]
    w, h = case["width"], case["height"]
    points = [[x / w, (y + 4) / h], [x / w, y / h], [x / w, max(0, y - 28) / h]]
    try:
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        seed = predictor.handle_request(
            dict(
                type="add_prompt",
                session_id=session,
                frame_index=0,
                obj_id=1,
                points=points,
                point_labels=[1, 1, 0],
            )
        )
        seed_masks = masks_from_output(seed["outputs"])
        torch.cuda.synchronize()
        seed_seconds = time.perf_counter() - start
        frames = {}
        intervals = []
        stream = iter(
            predictor.handle_stream_request(
                dict(
                    type="propagate_in_video",
                    session_id=session,
                    propagation_direction="forward",
                    start_frame_index=0,
                    max_frame_num_to_track=case["frames"],
                )
            )
        )
        start = time.perf_counter()
        while True:
            step_start = time.perf_counter()
            try:
                result = next(stream)
            except StopIteration:
                break
            index = int(result["frame_index"])
            assert index not in frames, f"Duplicate frame {index}"
            frames[index] = masks_from_output(result["outputs"])
            torch.cuda.synchronize()
            intervals.append(dict(frame=index, seconds=time.perf_counter() - step_start))
        torch.cuda.synchronize()
        propagation_seconds = time.perf_counter() - start
        peak_bytes = torch.cuda.max_memory_allocated()
        assert set(frames) == set(range(case["frames"])), f"Incomplete output: {name}"
        dest = OUTPUT / trial / name
        dest.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, masks in sorted(frames.items()):
            np.savez_compressed(
                dest / f"masks-{index:04}.npz",
                shape=np.array([h, w]),
                **{f"obj_{oid}": np.packbits(mask) for oid, mask in masks.items()},
            )
            target = masks.get(1, np.zeros((h, w), bool))
            assert target.shape == (h, w)
            ys, xs = np.where(target)
            rows.append(
                dict(
                    frame=index,
                    source_seconds=case["start"] + index / case["fps"],
                    area=int(target.sum()),
                    box=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                    if len(xs)
                    else None,
                )
            )
        report = dict(
            case=name,
            trial=trial,
            frames=len(frames),
            load_seconds=load_seconds,
            seed_seconds=seed_seconds,
            propagation_seconds=propagation_seconds,
            propagation_seconds_per_frame=propagation_seconds / len(frames),
            inference_seconds_per_frame=(seed_seconds + propagation_seconds) / len(frames),
            session_seconds=load_seconds + seed_seconds + propagation_seconds,
            peak_allocated_bytes=peak_bytes,
            nonempty_frames=sum(r["area"] > 0 for r in rows),
            seed_area=int(seed_masks.get(1, np.zeros((h, w), bool)).sum()),
            yield_intervals=intervals,
            observations=rows,
            prompt_points=points,
            initialization="two positive points and one negative, same as Mac SAM2 baseline",
        )
        (dest / "results.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ["yield_intervals", "observations", "prompt_points"]
                }
            ),
            flush=True,
        )
        return report
    finally:
        predictor.handle_request(dict(type="close_session", session_id=session))


def load_mask(path):
    with np.load(path) as packed:
        h, w = packed["shape"]
        key = "obj_1"
        return (
            np.unpackbits(packed[key])[: h * w].reshape(h, w).astype(bool)
            if key in packed
            else np.zeros((h, w), bool)
        )


def review():
    columns = ["Original", "Mac SAM2.1", "Mac SAM3.1", "L4 SAM3.1"]
    sheet = Image.new("RGB", (1200, 200 * len(ANNOTATIONS["samples"]) + 40), "white")
    draw = ImageDraw.Draw(sheet)
    for col, title in enumerate(columns):
        draw.text((col * 300 + 8, 10), title, fill="black")
    rows = []
    caps = {}
    for row, annotation in enumerate(ANNOTATIONS["samples"]):
        name, index = annotation["case"], annotation["frame"]
        if name not in caps:
            caps[name] = cv2.VideoCapture(str(INPUT / f"{name}.mp4"))
        cap = caps[name]
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = cap.read()
        assert ok
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        masks = [
            None,
            load_mask(INPUT / f"baselines/sam2-{name}/masks-{index:04}.npz"),
            load_mask(INPUT / f"baselines/sam31-memory-{name}/masks-{index:04}.npz"),
            load_mask(OUTPUT / f"warm/{name}/masks-{index:04}.npz"),
        ]
        x1, y1, x2, y2 = annotation["box"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        crop = (cx - 60, max(0, cy - 36), cx + 60, max(0, cy - 36) + 72)
        result = dict(case=name, frame=index, body_probe_count=len(annotation["body_points"]))
        for col, mask in enumerate(masks):
            rendered = rgb.copy()
            hits = None
            if mask is not None:
                hits = sum(bool(mask[y, x]) for x, y in annotation["body_points"])
                rendered[mask] = (rendered[mask] * 0.55 + np.array([0, 230, 130]) * 0.45).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(rendered, contours, -1, (0, 230, 130), 1)
                result[columns[col]] = dict(body_probe_hits=hits, area=int(mask.sum()))
            sheet.paste(Image.fromarray(rendered).crop(crop).resize((300, 180)), (col * 300, row * 200 + 40))
            draw.text(
                (col * 300 + 8, row * 200 + 222),
                f"{name} f{index}" + (f" | body {hits}/3" if hits is not None else ""),
                fill="black",
            )
        rows.append(result)
    for cap in caps.values():
        cap.release()
    sheet.save(OUTPUT / "review.jpg", quality=95)
    return rows


reports = [run_case(case, trial) for trial in ["cold", "warm"] for case in INPUT_MANIFEST["cases"]]
review_rows = review()
provenance = dict(
    python=platform.python_version(),
    torch=torch.__version__,
    numpy=np.__version__,
    opencv=cv2.__version__,
    gpu=torch.cuda.get_device_name(0),
    model_load_seconds=model_load_seconds,
    compile=False,
    use_fa3=False,
    precision="bfloat16 autocast",
    sam_source=importlib.metadata.distribution("sam3").read_text("direct_url.json"),
    input=INPUT_MANIFEST,
    numpy_constraint_exception="SAM declares numpy<2; tested with Colab NumPy 2",
    limitations="17 body-probe spot checks; no pixel-level ground truth or whole-game identity validation.",
)
(OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
(OUTPUT / "requirements-frozen.txt").write_text(subprocess.check_output(["pip", "freeze"], text=True))
(OUTPUT / "review-results.json").write_text(json.dumps(review_rows, indent=2))
summary = dict(
    reports=[{k: v for k, v in r.items() if k not in ["yield_intervals", "observations"]} for r in reports],
    review=review_rows,
    provenance=provenance,
)
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
archive_path = shutil.make_archive("/content/hogline-sam31-results", "zip", OUTPUT)
print("BENCHMARK COMPLETE", archive_path, flush=True)
display(Image.open(OUTPUT / "review.jpg"))

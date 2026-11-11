"""Isolated SAM comparison on local footage; never updates production tracking runs.

Run with .venv-sam3 for SAM3, .venv-sam for the existing SAM2.1 adapter.
The SAM3 detector+association and neural memory paths are reported separately.
"""

import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/sam3-20261108"
SOURCE = ROOT / "data/recordings/eDJdwj0JqBY/source.mp4"
CASES = [
    dict(name="clear", start=3603.0, crop=[0, 0, 720, 990], point=[405, 551]),
    dict(name="sweepers", start=1849.0, crop=[0, 0, 720, 990], point=[464, 657]),
    dict(name="opposite", start=5031.0, crop=[1200, 0, 1920, 990], point=[239, 611]),
    dict(name="contact", start=491.0, crop=[730, 7, 1188, 463], point=[162, 21]),
]
EXTRA_CASES = [
    dict(name="small-stone", start=471.0, crop=[0, 0, 724, 990], point=[416, 502]),
]


def frames(case, fps, duration):
    cap = cv2.VideoCapture(str(SOURCE))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, case["start"] * 1000)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        step = round(source_fps / fps)
        x1, y1, x2, y2 = case["crop"]
        for i in range(round(duration * fps)):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Unexpected end of source")
            yield i, cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            for _ in range(step - 1):
                cap.grab()
    finally:
        cap.release()


def load_sam3(model_path):
    import mlx.core as mx
    from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
    from mlx_vlm.utils import load_model
    from transformers import PreTrainedTokenizerFast

    mx.set_memory_limit(16 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    model = load_model(model_path)
    mx.eval(model.parameters())
    processor = Sam3Processor.from_pretrained(str(model_path))
    processor._tokenizer = PreTrainedTokenizerFast.from_pretrained(str(ROOT / "models/sam3-8bit"))
    return model, processor


def draw(rgb, masks, ids, title):
    out = rgb.copy()
    colors = [(0, 230, 130), (255, 80, 190), (50, 150, 255), (255, 170, 0)]
    for mask, object_id in zip(masks, ids, strict=True):
        mask = mask.astype(bool)
        color = colors[(int(object_id) - 1) % len(colors)]
        out[mask] = (out[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 1)
        ys, xs = np.where(mask)
        if len(xs):
            cv2.putText(
                out,
                str(object_id),
                (int(xs.mean()), int(ys.min()) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["sam2", "sam3-detect", "sam3-memory", "sam31-detect", "sam31-memory"],
        required=True,
    )
    parser.add_argument("--case", choices=[c["name"] for c in CASES + EXTRA_CASES])
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--duration", type=float, default=8)
    parser.add_argument("--prompt", default="curling stone")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import mlx.core as mx

    load_start = time.perf_counter()
    if args.backend == "sam2":
        sys.path.insert(0, str(ROOT))
        from hogline2.tracking_sam import Sam21MLX

        backend = Sam21MLX(ROOT)
    else:
        model_dir = "sam3.1-bf16" if args.backend.startswith("sam31") else "sam3-8bit"
        model, processor = load_sam3(ROOT / "models" / model_dir)
        from mlx_vlm.models.sam3.generate import Sam3Predictor, Sam3VideoPredictor, SimpleTracker, nms

        if args.backend == "sam31-memory":
            from mlx_vlm.models.sam3_1.generate import (
                _get_backbone_features,
                _init_tracker_state,
                _masks_to_detection,
                _propagate_tracker,
            )
    load_seconds = time.perf_counter() - load_start
    print(f"Loaded {args.backend} in {load_seconds:.2f}s", flush=True)
    for case in CASES + EXTRA_CASES if args.case else CASES:
        if args.case and args.case != case["name"]:
            continue
        dest = OUT / f"{args.backend}-{case['name']}"
        dest.mkdir(exist_ok=True)
        samples = list(frames(case, args.fps, args.duration))
        h, w = samples[0][1].shape[:2]
        writer = cv2.VideoWriter(str(dest / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        if args.backend.endswith("-detect") or args.backend == "sam31-memory":
            predictor = Sam3Predictor(model, processor, score_threshold=0.15)
            associator = SimpleTracker()
        elif args.backend == "sam3-memory":
            predictor = Sam3VideoPredictor(model, processor, score_threshold=0.15)
            predictor.set_video([Image.fromarray(rgb) for _, rgb in samples])
            # This SAM3 port's prompt encoder normalizes by its embedding grid,
            # unlike SAM3.1's wrapper, which accepts resized image pixels.
            x, y = case["point"]
            points = np.array([[x, y + 4], [x, y], [x, max(0, y - 28)]], np.float32)
            grid_h, grid_w = model.tracker_model.prompt_encoder.image_embedding_size
            points *= [grid_w / w, grid_h / h]
            predictor.add_point_prompt(points, np.array([1, 1, 0], np.int32), 0, 1)
        rows = []
        initialization = "points" if args.backend == "sam2" else "text"
        start = time.perf_counter()
        try:
            for i, rgb in samples:
                frame_start = time.perf_counter()
                if args.backend == "sam2":
                    backend.begin_frame(i)
                    mask, score = backend.track(
                        case["name"], "target", rgb, case["point"] if i == 0 else None
                    )
                    masks, scores, ids = [mask], [score], [1]
                elif args.backend.endswith("-detect"):
                    result = nms(predictor.predict(Image.fromarray(rgb), text_prompt=args.prompt))
                    result = associator.update(result)
                    masks, scores, ids = result.masks, result.scores, result.track_ids
                    if ids is None:
                        ids = []
                elif args.backend == "sam31-memory":
                    pixels = mx.array(processor.preprocess_image(Image.fromarray(rgb))["pixel_values"])
                    features = _get_backbone_features(model, pixels)
                    if i == 0:
                        result = nms(predictor.predict(Image.fromarray(rgb), text_prompt=args.prompt))
                        x, y = map(int, case["point"])
                        candidates = [j for j, m in enumerate(result.masks) if m[y, x]]
                        if not candidates:
                            annotations = json.loads(
                                (Path(__file__).parent / "review-points.json").read_text()
                            )
                            box = next(
                                a["box"]
                                for a in annotations["samples"]
                                if a["case"] == case["name"] and a["frame"] == 0
                            )
                            result = nms(
                                predictor.predict(
                                    Image.fromarray(rgb),
                                    text_prompt=args.prompt,
                                    boxes=np.array([box], np.float32),
                                )
                            )
                            candidates = [j for j, m in enumerate(result.masks) if m[y, x]]
                            initialization = "text+manual_box_fallback"
                        if candidates:
                            best = max(candidates, key=lambda j: result.scores[j])
                            masks, scores, ids = [result.masks[best]], [result.scores[best]], [1]
                            state = _init_tracker_state(model, features, masks)
                        else:
                            initialization = "direct_points_after_text_and_box_failed"
                            points = np.array([[x, y + 4], [x, y], [x, max(0, y - 28)]], np.float32)
                            points *= [processor.image_size / w, processor.image_size / h]
                            state = model.tracker_init_state(1)
                            output = model.tracker_model.track_step(
                                state,
                                frame_idx=0,
                                is_init_cond_frame=True,
                                frame_features=model.tracker_frame_features(features),
                                point_inputs={
                                    "point_coords": mx.array(points)[None],
                                    "point_labels": mx.array([[1, 1, 0]], mx.int32),
                                },
                                num_frames=len(samples),
                            )
                            result = _masks_to_detection(output, 1, (w, h))
                            masks, scores, ids = result.masks, result.scores, [1]
                    else:
                        result = _propagate_tracker(model, state, features, i, 1, (w, h))
                        masks, scores, ids = result.masks, result.scores, [1]
                else:
                    result = predictor.propagate(i, i + 1)[0]
                    masks, scores, ids = result.masks, result.scores, result.object_ids
                    predictor._frame_features[i] = None
                masks = [
                    cv2.resize(np.asarray(m, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    for m in masks
                ]
                elapsed = time.perf_counter() - frame_start
                boxes = []
                for mask in masks:
                    ys, xs = np.where(mask)
                    boxes.append(
                        [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                        if len(xs)
                        else None
                    )
                rows.append(
                    dict(
                        frame=i,
                        source_seconds=case["start"] + i / args.fps,
                        seconds=elapsed,
                        ids=[int(x) for x in ids],
                        scores=[float(x) for x in scores],
                        areas=[int(m.sum()) for m in masks],
                        boxes=boxes,
                    )
                )
                packed = {f"obj_{int(oid)}": np.packbits(mask) for oid, mask in zip(ids, masks, strict=True)}
                np.savez_compressed(dest / f"masks-{i:04}.npz", shape=np.array([h, w]), **packed)
                overlay = draw(rgb, masks, ids, f"{args.backend} {case['start'] + i / args.fps:.2f}s")
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                if i % args.fps == 0 or i == len(samples) - 1:
                    Image.fromarray(overlay).save(dest / f"frame-{i:04}.png")
                print(
                    f"{args.backend} {case['name']} {i + 1}/{len(samples)} "
                    f"{elapsed:.2f}s areas={rows[-1]['areas']}",
                    flush=True,
                )
        finally:
            writer.release()
            report = dict(
                backend=args.backend,
                initialization=initialization,
                case=case,
                fps=args.fps,
                duration=args.duration,
                prompt=args.prompt
                if args.backend.endswith("-detect") or args.backend == "sam31-memory"
                else "two positives + one negative",
                load_seconds=load_seconds,
                elapsed_seconds=time.perf_counter() - start,
                peak_metal_bytes=mx.get_peak_memory(),
                completed=len(rows) == len(samples),
                frames=rows,
                limitation="Nonempty masks and confidence are not ground-truth accuracy.",
            )
            (dest / "results.json").write_text(json.dumps(report, indent=2))
        mx.clear_cache()


if __name__ == "__main__":
    with (ROOT / "data/recordings/sam-worker.lock").open("a") as worker_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()

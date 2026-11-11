"""Official SAM2.1 CUDA benchmark, run inside the fresh Colab notebook.

The input ZIP and checkpoints are prepared in earlier notebook cells.
Run: %run -i /content/benchmark.py
"""

import gc
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import decord
import numpy as np
import sam2
import torch
from sam2.build_sam import build_sam2_video_predictor

INPUT = Path("/content/hogline-benchmark")
OUTPUT = Path("/content/hogline-sam21-results")
CHECKPOINTS = Path("/content/sam21-checkpoints")
OUTPUT.mkdir(exist_ok=True)
MANIFEST = json.loads((INPUT / "input-manifest.json").read_text())
ANNOTATIONS = json.loads((INPUT / "review-points.json").read_text())
MODELS = {
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def validate_decoder():
    """Verify Decord delivers the exact same 204 RGB frames as OpenCV."""
    decord.bridge.set_bridge("native")
    for case in MANIFEST["cases"]:
        path = INPUT / f"{case['name']}.mp4"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
        reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
        assert len(reader) == case["frames"]
        cap = cv2.VideoCapture(str(path))
        for i in range(len(reader)):
            ok, bgr = cap.read()
            assert ok
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            assert np.array_equal(rgb, reader[i].asnumpy()), (case["name"], i)
        assert not cap.read()[0]
        cap.release()
        print(f"DECODER VERIFIED {case['name']}: {len(reader)} exact RGB frames", flush=True)
    decord.bridge.set_bridge("torch")


def run_case(predictor, case, model_name, trial):
    """Fresh state per clip; account for the backbone work inside init_state."""
    dest = OUTPUT / model_name / trial / case["name"]
    dest.mkdir(parents=True, exist_ok=True)
    initial_feature_seconds = 0.0
    original_get_features = predictor._get_image_feature

    def timed_initial_features(*args, **kwargs):
        nonlocal initial_feature_seconds
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_get_features(*args, **kwargs)
        torch.cuda.synchronize()
        initial_feature_seconds += time.perf_counter() - started
        return result

    torch.cuda.reset_peak_memory_stats()
    predictor._get_image_feature = timed_initial_features
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        state = predictor.init_state(
            video_path=str(INPUT / f"{case['name']}.mp4"),
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
    finally:
        predictor._get_image_feature = original_get_features
    torch.cuda.synchronize()
    state_init_seconds = time.perf_counter() - started
    x, y = case["point"]
    points = np.array([[x, min(y + 4, case["height"] - 1)], [x, y], [x, max(0, y - 28)]], np.float32)
    labels = np.array([1, 1, 0], np.int32)
    torch.cuda.synchronize()
    started = time.perf_counter()
    predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=points, labels=labels)
    torch.cuda.synchronize()
    prompt_seconds = time.perf_counter() - started
    frames = {}
    intervals = []
    started = time.perf_counter()
    stream = iter(predictor.propagate_in_video(state, start_frame_idx=0))
    while True:
        step_started = time.perf_counter()
        try:
            index, ids, logits = next(stream)
        except StopIteration:
            break
        assert index not in frames
        masks = (logits > 0).squeeze(1).cpu().numpy().copy()
        frames[int(index)] = {int(oid): mask for oid, mask in zip(ids, masks, strict=True)}
        torch.cuda.synchronize()
        intervals.append(dict(frame=int(index), seconds=time.perf_counter() - step_started))
    torch.cuda.synchronize()
    propagation_seconds = time.perf_counter() - started
    peak_bytes = torch.cuda.max_memory_allocated()
    assert set(frames) == set(range(case["frames"])), case["name"]
    inference_seconds = initial_feature_seconds + prompt_seconds + propagation_seconds
    rows = []
    for index, masks in sorted(frames.items()):
        mask = masks.get(1, np.zeros((case["height"], case["width"]), bool))
        assert mask.shape == (case["height"], case["width"])
        ys, xs = np.where(mask)
        rows.append(
            dict(
                frame=index,
                source_seconds=case["start"] + index / case["fps"],
                area=int(mask.sum()),
                box=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None,
            )
        )
        np.savez_compressed(
            dest / f"masks-{index:04}.npz",
            shape=np.array(mask.shape),
            **{f"obj_{oid}": np.packbits(m) for oid, m in masks.items()},
        )
    report = dict(
        model=model_name,
        trial=trial,
        case=case["name"],
        frames=len(frames),
        state_init_seconds=state_init_seconds,
        initial_feature_seconds=initial_feature_seconds,
        prompt_seconds=prompt_seconds,
        propagation_seconds=propagation_seconds,
        inference_seconds=inference_seconds,
        seconds_per_frame=inference_seconds / len(frames),
        nonempty_frames=sum(row["area"] > 0 for row in rows),
        peak_allocated_bytes=peak_bytes,
        points=points.tolist(),
        labels=labels.tolist(),
        observations=rows,
        yield_intervals=intervals,
    )
    (dest / "results.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ["points", "labels", "observations", "yield_intervals"]
            }
        ),
        flush=True,
    )
    predictor.reset_state(state)
    return report


def review(model_name, trial="warm"):
    rows = []
    for a in ANNOTATIONS["samples"]:
        path = OUTPUT / model_name / trial / a["case"] / f"masks-{a['frame']:04}.npz"
        with np.load(path) as packed:
            h, w = map(int, packed["shape"])
            mask = np.unpackbits(packed["obj_1"])[: h * w].reshape(h, w).astype(bool)
        rows.append(
            dict(
                case=a["case"],
                frame=a["frame"],
                body_probe_hits=sum(bool(mask[y, x]) for x, y in a["body_points"]),
                body_probe_count=len(a["body_points"]),
                area=int(mask.sum()),
            )
        )
    return dict(
        body_probe_hits=sum(r["body_probe_hits"] for r in rows),
        body_probe_count=sum(r["body_probe_count"] for r in rows),
        samples=rows,
    )


def main():
    validate_decoder()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    all_reports = []
    model_provenance = {}
    components_available = bool(globals().get("cuda_components_available", False))
    for name, config in MODELS.items():
        checkpoint = CHECKPOINTS / f"sam2.1_hiera_{name}.pt"
        with checkpoint.open("rb") as file:
            checkpoint_sha = hashlib.file_digest(file, "sha256").hexdigest()
        started = time.perf_counter()
        predictor = build_sam2_video_predictor(
            config, str(checkpoint), device="cuda", apply_postprocessing=True, vos_optimized=False
        )
        if not components_available:
            predictor.fill_hole_area = 0
        model_load_seconds = time.perf_counter() - started
        model_provenance[name] = dict(
            config=config,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_bytes=checkpoint.stat().st_size,
            model_load_seconds=model_load_seconds,
            fill_hole_area=predictor.fill_hole_area,
            parameter_count=sum(p.numel() for p in predictor.parameters()),
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for trial in ["cold", "warm"]:
                for case in MANIFEST["cases"]:
                    all_reports.append(run_case(predictor, case, name, trial))
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
    quality = {name: review(name) for name in MODELS}
    provenance = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        opencv=cv2.__version__,
        decord=decord.__version__,
        gpu=torch.cuda.get_device_name(0),
        source=importlib.metadata.distribution("SAM-2").read_text("direct_url.json"),
        models=model_provenance,
        input=MANIFEST,
        decoder_pixel_equality=True,
        vos_optimized=False,
        precision="BF16 autocast, TF32 allowed",
        custom_cuda_components=components_available,
        timing=(
            "Initial backbone in init_state + prompting + propagation + masks to CPU; "
            "excludes decode/load/serialization"
        ),
        limitations="17-frame body-probe spot checks, not mask accuracy or whole-game identity validation",
    )
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (OUTPUT / "summary.json").write_text(json.dumps(dict(reports=all_reports, quality=quality), indent=2))
    (OUTPUT / "requirements-frozen.txt").write_text(subprocess.check_output(["pip", "freeze"], text=True))
    runtime_sources = {}
    for name in ["build_sam.py", "sam2_video_predictor.py", "utils/misc.py", "modeling/sam2_base.py"]:
        path = Path(sam2.__file__).parent / name
        runtime_sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        dest = OUTPUT / "runtime-source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    (OUTPUT / "source-hashes.json").write_text(json.dumps(runtime_sources, indent=2))
    shutil.copy2(__file__, OUTPUT / "benchmark.py")
    shutil.make_archive("/content/hogline-sam21-results", "zip", OUTPUT)
    print("SAM2.1 BENCHMARK COMPLETE", flush=True)
    for name in MODELS:
        for trial in ["cold", "warm"]:
            reports = [r for r in all_reports if r["model"] == name and r["trial"] == trial]
            seconds = sum(r["inference_seconds"] for r in reports)
            count = sum(r["frames"] for r in reports)
            print(
                name,
                trial,
                count,
                "frames;",
                round(seconds, 3),
                "seconds;",
                round(count / seconds, 2),
                "fps; body probes",
                quality[name]["body_probe_hits"],
                "/51",
            )


if __name__ == "__main__":
    main()

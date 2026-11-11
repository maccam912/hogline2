"""Load the tested official checkpoint in a Colab GPU runtime.

Install the official source at f1647e8d0c1d280991f4d1b7d974c2704537fe66 first.
Use --no-deps to preserve Colab's NumPy/OpenCV stack; this experiment used NumPy 2,
outside SAM's declared numpy<2 constraint. See docs/COLAB_SAM31_EVALUATION.md.
Upload the prepared input ZIP and run this script with `%run -i`.
"""

import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from sam3.model_builder import build_sam3_multiplex_video_predictor

INPUT = Path("/content/hogline-benchmark")
OUTPUT = Path("/content/hogline-sam31-results")
OUTPUT.mkdir(exist_ok=True)
archive = Path("/content/hogline-sam31-input.zip")
if not zipfile.is_zipfile(archive):
    raise ValueError("Wait for the input archive upload to finish")
with zipfile.ZipFile(archive) as zipped:
    if zipped.testzip() is not None:
        raise ValueError("Input archive CRC check failed")
    zipped.extractall(INPUT)
manifest = json.loads((INPUT / "input-manifest.json").read_text())
for case in manifest["cases"]:
    if hashlib.sha256((INPUT / f"{case['name']}.mp4").read_bytes()).hexdigest() != case["sha256"]:
        raise ValueError(f"Clip checksum mismatch: {case['name']}")

started = time.perf_counter()
predictor = build_sam3_multiplex_video_predictor(
    use_fa3=False, use_rope_real=False, compile=False, warm_up=False, async_loading_frames=False
)
model_load_seconds = time.perf_counter() - started
predictor.model.batched_grounding_batch_size = 1
predictor.model.postprocess_batch_size = 1
canonical_init = type(predictor.model).init_state.__get__(predictor.model)


def compatible_init_state(
    resource_path, offload_video_to_cpu=False, offload_state_to_cpu=False, _init=canonical_init, **kwargs
):
    """Keep official session bookkeeping and reject unsupported state offloading."""
    if offload_state_to_cpu:
        raise NotImplementedError("The installed multiplex model does not support state offloading")
    kwargs.update(use_cv2=True, input_is_mp4=True)
    return _init(resource_path, offload_video_to_cpu=offload_video_to_cpu, **kwargs)


predictor.model.init_state = compatible_init_state


def masks_from_output(output):
    return {
        int(oid): np.asarray(mask, dtype=bool).copy()
        for oid, mask in zip(output["out_obj_ids"], output["out_binary_masks"], strict=True)
    }


print(f"SAM3.1 ready on {torch.cuda.get_device_name(0)}; load {model_load_seconds:.2f}s")

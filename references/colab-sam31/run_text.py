"""Colab full-sequence text prompting; avoids upstream partial-bound bug."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

predictor = globals()["predictor"]
masks_from_output = globals()["masks_from_output"]
root = Path("/content/hogline-benchmark")
output = Path("/content/hogline-sam31-results")
manifest = json.loads((root / "input-manifest.json").read_text())
reports = []
for case in manifest["cases"]:
    name = case["name"]
    sid = predictor.handle_request(dict(type="start_session", resource_path=str(root / f"{name}.mp4")))[
        "session_id"
    ]
    dest = output / "text" / name
    dest.mkdir(parents=True, exist_ok=True)
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        seed = predictor.handle_request(
            dict(type="add_prompt", session_id=sid, frame_index=0, text="curling stone")
        )
        seed_masks = masks_from_output(seed["outputs"])
        x, y = case["point"]
        candidates = [oid for oid, mask in seed_masks.items() if mask[y, x]]
        target_id = min(candidates) if candidates else None
        torch.cuda.synchronize()
        seed_seconds = time.perf_counter() - start
        start = time.perf_counter()
        frames = {}
        # Omit max_frame_num_to_track: upstream mixes inclusive/exclusive bounds.
        for response in predictor.handle_stream_request(
            dict(type="propagate_in_video", session_id=sid, propagation_direction="forward")
        ):
            frames[int(response["frame_index"])] = masks_from_output(response["outputs"])
        torch.cuda.synchronize()
        propagation_seconds = time.perf_counter() - start
        assert set(frames) == set(range(case["frames"]))
        target_areas = []
        object_counts = []
        for index, masks in sorted(frames.items()):
            h, w = case["height"], case["width"]
            target = masks.get(target_id, np.zeros((h, w), bool))
            target_areas.append(int(target.sum()))
            object_counts.append(len(masks))
            np.savez_compressed(
                dest / f"masks-{index:04}.npz", shape=np.array([h, w]), obj_1=np.packbits(target)
            )
            np.savez_compressed(
                dest / f"all-masks-{index:04}.npz",
                shape=np.array([h, w]),
                **{f"obj_{oid}": np.packbits(mask) for oid, mask in masks.items()},
            )
        report = dict(
            case=name,
            frames=len(frames),
            target_id=target_id,
            seed_seconds=seed_seconds,
            propagation_seconds=propagation_seconds,
            inference_seconds_per_frame=(seed_seconds + propagation_seconds) / len(frames),
            nonempty_target_frames=sum(a > 0 for a in target_areas),
            target_areas=target_areas,
            object_counts=object_counts,
            selection="Fixed initial ID containing the same manual target point; no reassociation",
        )
        (dest / "results.json").write_text(json.dumps(report, indent=2))
        reports.append(report)
        print(json.dumps(report), flush=True)
    except Exception as error:
        (dest / "error.json").write_text(json.dumps(dict(error=repr(error)), indent=2))
        raise
    finally:
        predictor.handle_request(dict(type="close_session", session_id=sid))
(output / "text-summary.json").write_text(json.dumps(reports, indent=2))
print("TEXT BENCHMARK COMPLETE", flush=True)
shutil.make_archive("/content/hogline-sam31-results", "zip", output)

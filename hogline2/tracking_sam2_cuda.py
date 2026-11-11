"""Bounded, forward-only adapter for the pinned official SAM2 video predictor.

One backbone pass per camera/frame is shared by independently prompted objects.
The private predictor calls are pinned by tracking_colab.SAM_REVISION; this avoids
loading an entire end into GPU RAM through init_state(video_path).
"""

import hashlib
import importlib.metadata
import json
import os
import platform
from collections import OrderedDict
from pathlib import Path

from .tracking_sam2_granite import GraniteBodyMixin
from .tracking_sam2_specs import CHECKPOINTS, SAM_REVISION


class Sam21Cuda:
    model_size = "base_plus"
    memory_device = "cuda"
    revision = SAM_REVISION

    def __init__(self, project=None):
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        installed = json.loads(importlib.metadata.distribution("SAM-2").read_text("direct_url.json") or "{}")
        if installed.get("vcs_info", {}).get("commit_id") != self.revision:
            raise ValueError("SAM2 source revision changed; rerun the notebook setup cell")
        if not torch.cuda.is_available():
            raise ValueError("Choose a GPU runtime in Colab before running tracking")
        filename, config, self.weight_sha256 = CHECKPOINTS[self.model_size]
        checkpoint = Path(os.environ.get("HOGLINE_SAM2_CHECKPOINTS", "/content/sam21-checkpoints")) / filename
        with checkpoint.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != self.weight_sha256:
                raise ValueError("SAM2.1 checkpoint checksum mismatch")
        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.model = build_sam2_video_predictor(config, str(checkpoint), device="cuda", vos_optimized=False)
        self.runtime_info = dict(
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            sam2_revision=self.revision,
            model_size=self.model_size,
        )
        self.states, self.features = {}, {}
        self.frame_index = 0
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]
        torch.cuda.reset_peak_memory_stats()

    def begin_frame(self, frame_index):
        self.frame_index = frame_index
        self.features.clear()

    def _features(self, view_id, rgb):
        import cv2

        if view_id not in self.features:
            image = cv2.resize(rgb, (self.model.image_size, self.model.image_size))
            tensor = self.torch.as_tensor(image, device="cuda").permute(2, 0, 1).float() / 255
            tensor = ((tensor - self.mean) / self.std)[None]
            self.features[view_id] = (tensor, self.model.forward_image(tensor))
        return self.features[view_id]

    def _state(self, rgb):
        device = self.model.device
        return dict(
            images={},
            num_frames=0,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            video_height=rgb.shape[0],
            video_width=rgb.shape[1],
            device=device,
            storage_device=device,
            point_inputs_per_obj={},
            mask_inputs_per_obj={},
            cached_features={},
            constants={},
            obj_id_to_idx=OrderedDict(),
            obj_idx_to_id=OrderedDict(),
            obj_ids=[],
            output_dict_per_obj={},
            temp_output_dict_per_obj={},
            frames_tracked_per_obj={},
        )

    def track(self, view_id, stone_id, rgb, point=None):
        torch = self.torch
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            key, f = (view_id, stone_id), self.frame_index
            if key not in self.states:
                if point is None:
                    raise ValueError("New tracks need an explicit prompt")
                self.states[key] = self._state(rgb)
            state = self.states[key]
            state["num_frames"] = f + 1
            state["cached_features"] = {f: self._features(view_id, rgb)}
            try:
                if point is not None:
                    points, labels = self.prompt_points(rgb, point)
                    self.model.add_new_points_or_box(state, f, 1, points=points, labels=labels)
                    self.model.propagate_in_video_preflight(state)
                    output = state["output_dict_per_obj"][0]["cond_frame_outputs"][f]
                    masks = output["pred_masks"]
                else:
                    output, masks = self.model._run_single_frame_inference(
                        state, state["output_dict_per_obj"][0], f, 1, False, None, None, False, True
                    )
                    state["output_dict_per_obj"][0]["non_cond_frame_outputs"][f] = output
                state["frames_tracked_per_obj"][0][f] = {"reverse": False}
                _, original = self.model._get_orig_video_res_output(state, masks)
                mask = (original[0, 0] > 0).cpu().numpy()
                quality = float(output["object_score_logits"].sigmoid().item())
                for old in list(state["output_dict_per_obj"][0]["non_cond_frame_outputs"]):
                    if old < f - 16:
                        del state["output_dict_per_obj"][0]["non_cond_frame_outputs"][old]
                        state["frames_tracked_per_obj"][0].pop(old, None)
                return mask, quality
            finally:
                state["cached_features"].clear()

    def prompt_points(self, rgb, point):
        import numpy as np

        x, y = point
        return (
            np.asarray([[x, min(y + 4, rgb.shape[0] - 1)], [x, y], [x, max(0, y - 28)]], np.float32),
            np.asarray([1, 1, 0], np.int32),
        )

    def reject_current_memory(self, view_id, stone_id):
        state = self.states[(view_id, stone_id)]
        state["output_dict_per_obj"][0]["non_cond_frame_outputs"].pop(self.frame_index, None)

    def has_track(self, view_id, stone_id):
        return (view_id, stone_id) in self.states

    def rename_track(self, view_id, old_id, new_id):
        self.states[(view_id, new_id)] = self.states.pop((view_id, old_id))

    def retire(self, view_id, stone_id):
        self.states.pop((view_id, stone_id), None)

    @property
    def peak_bytes(self):
        return self.torch.cuda.max_memory_allocated()


class Sam21CudaGranite(GraniteBodyMixin, Sam21Cuda):
    name = "sam2.1-cuda-granite"


class Sam21CudaLargeGranite(Sam21CudaGranite):
    name = "sam2.1-cuda-large-granite"
    model_size = "large"

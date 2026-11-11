"""Pinned MLX adapter. Streaming frames, independent object memories, shared view features."""

import hashlib
import json
import sys
from pathlib import Path


class Sam21MLX:
    name = "sam2.1-mlx"
    revision = "036f2d02a9069871fd2bf47109ff3b522cd222c4"

    def __init__(self, project=None):
        project = Path(project or Path(__file__).resolve().parents[1])
        sys.path.insert(0, str(project / "vendor/sam21-mlx/src"))
        import mlx.core as mx
        from sam2.build import build_model
        from sam2.predictor import SAM2VideoPredictor

        self.mx, self.predictor_class = mx, SAM2VideoPredictor
        weights = project / "models/sam2.1/sam2.1_hiera_base_plus.safetensors"
        self.weight_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
        manifest = json.loads((weights.parent / "manifest.json").read_text())
        expected = manifest.get("sha256") or manifest.get("model_sha256")
        if expected and self.weight_sha256 != expected:
            raise ValueError("SAM checkpoint checksum mismatch")
        mx.set_cache_limit(512 * 1024**2)
        mx.set_memory_limit(12 * 1024**3)
        self.model = build_model("base_plus", str(weights))
        self.model.load_weights(str(weights), strict=True)
        mx.eval(self.model.parameters())
        self.states = {}
        self.features = {}

    def begin_frame(self, frame_index):
        self.frame_index = frame_index
        self.features.clear()

    def _features(self, view_id, rgb):
        import cv2
        import numpy as np
        from sam2.predictor import _imagenet_normalize_chw

        if view_id not in self.features:
            image = cv2.resize(rgb, (1024, 1024)).astype(np.float32) / 255
            normalized = _imagenet_normalize_chw(image.transpose(2, 0, 1))
            features = self.model.forward_image(self.mx.array(normalized)[None])
            self.mx.eval(*features.values())
            self.features[view_id] = features
        return self.features[view_id]

    def track(self, view_id, stone_id, rgb, point=None):
        import numpy as np

        key = (view_id, stone_id)
        f = self.frame_index
        if key not in self.states:
            if point is None:
                raise ValueError("New tracks need an explicit prompt")
            self.states[key] = self.predictor_class(self.model)
        pred = self.states[key]
        pred._original_sizes = {f: rgb.shape[:2]}
        pred._features = {f: self._features(view_id, rgb)}
        if point is not None:
            # Include the granite just beneath the handle and exclude the deliverer's hand/body.
            points, labels = self.prompt_points(rgb, point)
            pred.add_new_points(f, 1, points, labels)
            pred.refine_init_with_click_cc(f, 1, points, labels)
            conditioning = sorted(pred._cond_frame_outputs)
            for old in conditioning[1:-2]:
                del pred._cond_frame_outputs[old]
        else:
            pred._run_single_frame(f, 1, run_mem_encoder=True)
        masks, scores = pred._get_video_res_masks(f)
        best = int(np.argmax(scores[0]))
        mask = masks[0][best] > 0
        quality = float(scores[0][best])
        pred._output_dict.clear()
        pred._features.clear()
        for old in list(pred._non_cond_frame_outputs):
            if old < f - 16:
                del pred._non_cond_frame_outputs[old]
        return mask, quality

    def prompt_points(self, rgb, point):
        import numpy as np

        x, y = point
        points = np.asarray([[x, min(y + 4, rgb.shape[0] - 1)], [x, y], [x, max(0, y - 28)]], np.float32)
        return points, np.asarray([1, 1, 0], np.int32)

    def reject_current_memory(self, view_id, stone_id):
        self.states[(view_id, stone_id)]._non_cond_frame_outputs.pop(self.frame_index, None)

    def has_track(self, view_id, stone_id):
        return (view_id, stone_id) in self.states

    def rename_track(self, view_id, old_id, new_id):
        self.states[(view_id, new_id)] = self.states.pop((view_id, old_id))

    def retire(self, view_id, stone_id):
        self.states.pop((view_id, stone_id), None)

    @property
    def peak_bytes(self):
        return self.mx.get_peak_memory()

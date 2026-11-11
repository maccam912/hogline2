"""Experimental MLX SAM3.1 per-camera multiplex tracker for matched comparisons."""

import hashlib
import json
from pathlib import Path


class Sam31Multiplex:
    name = "sam3.1-mlx-multiplex"
    revision = "a992e302ea9b0f03f41dfd93414a4fd0e818f65b"

    def __init__(self, project=None):
        import mlx.core as mx
        from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
        from mlx_vlm.models.sam3_1.generate import _masks_to_detection
        from mlx_vlm.utils import load_model

        root = Path(project or Path(__file__).resolve().parents[1])
        model_dir = root / "models/sam3.1-bf16"
        with (model_dir / "model.safetensors").open("rb") as file:
            self.weight_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
        manifest = json.loads((model_dir / "hogline-manifest.json").read_text())
        if self.weight_sha256 != manifest["sha256"]["model.safetensors"]:
            raise ValueError("SAM3.1 checkpoint checksum mismatch")
        mx.set_memory_limit(28 * 1024**3)
        mx.set_cache_limit(256 * 1024**2)
        self.mx = mx
        self.model = load_model(model_dir)
        mx.eval(self.model.parameters())
        self.processor = Sam3Processor.from_pretrained(str(model_dir))
        self.convert = _masks_to_detection
        self.views = {}
        self.events = []

    def _seed(self, features, rgb, seeds, frame):
        import numpy as np

        from .tracking_reinitialize import body_prompt

        mx = self.mx
        h, w = rgb.shape[:2]
        points = []
        for seed in seeds:
            x, y = seed["point"]
            points.append([[x, min(h - 1, y + 4)], [x, y], [x, max(0, y - 28)]])

        def predict(points):
            scaled = np.asarray(points, np.float32) * [
                self.processor.image_size / w,
                self.processor.image_size / h,
            ]
            state = self.model.tracker_init_state(len(seeds))
            output = self.model.tracker_model.track_step(
                state,
                frame_idx=frame,
                is_init_cond_frame=True,
                frame_features=features,
                point_inputs={
                    "point_coords": mx.array(scaled),
                    "point_labels": mx.array([[1, 1, 0]] * len(seeds), mx.int32),
                },
                num_frames=frame + 1,
            )
            mx.eval(output)
            return state, output

        state, output = predict(points)
        result = self.convert(output, len(seeds), (w, h))
        changed = False
        for i, seed in enumerate(seeds):
            if seed["kind"] != "perspective":
                continue
            hint = body_prompt(rgb, result.masks[i].astype(bool), seed["color"])
            if hint and hint["outside_mask"]:
                points[i] = [
                    hint["handle"],
                    hint["body"],
                    [hint["handle"][0], max(0, hint["handle"][1] - 28)],
                ]
                changed = True
        if changed:
            state, output = predict(points)
        return state, output

    def step(self, view, rgb, frame, seeds):
        """Propagate every object in one camera jointly, then add newly seeded objects."""
        from PIL import Image

        mx = self.mx
        vid = view["id"]
        pixels = mx.array(self.processor.preprocess_image(Image.fromarray(rgb))["pixel_values"])
        backbone = self.model.detector_model.vision_encoder.backbone(pixels)
        features = self.model.tracker_frame_features(backbone, need_interactive=bool(seeds))
        mx.eval(features)
        if vid not in self.views:
            if not seeds:
                return {}
            state, output = self._seed(features, rgb, seeds, frame)
            names = [seed["stone_id"] for seed in seeds]
            self.views[vid] = dict(state=state, names=names)
        else:
            session = self.views[vid]
            state, names = session["state"], session["names"]
            output = self.model.tracker_model.propagate(state, frame, features, num_frames=frame + 1)
            if seeds:
                _, initial = self._seed(features, rgb, seeds, frame)
                masks = (initial["pred_masks_high_res"] > 0).astype(mx.float32)
                output = self.model.tracker_model.add_mask_prompt(
                    state,
                    frame,
                    features,
                    masks,
                )
                names.extend(seed["stone_id"] for seed in seeds)
            mx.eval(output)
        result = self.convert(output, len(names), (rgb.shape[1], rgb.shape[0]))
        return {name: (result.masks[i].astype(bool), float(result.scores[i])) for i, name in enumerate(names)}

    @property
    def peak_bytes(self):
        return self.mx.get_peak_memory()

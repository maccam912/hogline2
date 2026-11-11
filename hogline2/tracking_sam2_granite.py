"""SAM2 with size-aware body prompts and explicit visible-bottom validation."""

from .tracking_sam import Sam21MLX
from .tracking_stone import body_clicks, granite_anchor, granite_evidence, seed_handle, stone_component


class GraniteBodyMixin:
    granite_body = True

    def __init__(self, project=None):
        super().__init__(project)
        self.hint = None
        self.last_reset = {}
        self.events = []

    def prompt_points(self, rgb, point):
        return body_clicks(self.hint, rgb.shape) if self.hint else super().prompt_points(rgb, point)

    def observe(self, view, name, color, rgb, point=None, fps=3):
        key = (view["id"], name)
        perspective = view["kind"] == "perspective"
        if point is not None:
            self.states.pop(key, None)
            self.hint = granite_evidence(rgb, seed_handle(rgb, color, point), color) if perspective else None
        try:
            mask, quality = super().track(*key, rgb, point)
        finally:
            self.hint = None
        if not perspective:
            return mask, quality
        evidence = granite_evidence(rgb, mask, color)
        valid = granite_anchor(mask, evidence) is not None
        if evidence and not valid and self.frame_index - self.last_reset.get(key, -10000) >= fps:
            old = self.states.pop(key)
            self.hint = evidence
            try:
                candidate, score = super().track(*key, rgb, evidence["handle"])
            finally:
                self.hint = None
            accepted = granite_anchor(candidate, evidence) is not None
            if accepted:
                mask, quality = candidate, score
            else:
                self.states[key] = old
            self.last_reset[key] = self.frame_index
            self.events.append(
                dict(
                    type="granite_body_reprompt",
                    frame=self.frame_index,
                    stone=name,
                    view=view["id"],
                    accepted=accepted,
                    handle=evidence["handle"],
                    body_points=evidence["body_points"],
                    body_bbox=evidence["bbox"],
                )
            )
        if evidence:
            mask = stone_component(mask, evidence["handle"])
        return mask, quality

    def rename_track(self, view_id, old_id, new_id):
        super().rename_track(view_id, old_id, new_id)
        old, new = (view_id, old_id), (view_id, new_id)
        if old in self.last_reset:
            self.last_reset[new] = self.last_reset.pop(old)

    def retire(self, view_id, stone_id):
        super().retire(view_id, stone_id)
        self.last_reset.pop((view_id, stone_id), None)


class Sam21Granite(GraniteBodyMixin, Sam21MLX):
    name = "sam2.1-mlx-granite"

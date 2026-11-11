"""Provider-neutral contracts. SAM loads only in the isolated tracking process."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ViewClip:
    path: Path
    end_number: int
    source_start_seconds: float
    clip_video_start_seconds: float
    view_id: str
    # Camera crops and physical identities are managed outside the provider.
    source_crop_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class ObjectPrompt:
    object_id: str
    frame_index: int
    points_xy: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class MaskObservation:
    backend: str
    model_revision: str
    view_id: str
    object_id: str
    source_seconds: float
    mask_path: Path
    visible: bool


class Segmenter(Protocol):
    """Streaming provider boundary; identity and camera calibration stay outside the model."""

    name: str
    revision: str
    weight_sha256: str

    def begin_frame(self, frame_index: int): ...
    def track(self, view_id: str, stone_id: str, rgb, point=None): ...
    def has_track(self, view_id: str, stone_id: str) -> bool: ...
    def rename_track(self, view_id: str, old_id: str, new_id: str): ...
    def reject_current_memory(self, view_id: str, stone_id: str): ...
    def retire(self, view_id: str, stone_id: str): ...
    @property
    def peak_bytes(self) -> int: ...


def load_backend(name: str):
    if name in {"sam2.1-cuda-granite", "sam2.1-cuda-large-granite"}:
        from .tracking_sam2_cuda import Sam21CudaGranite, Sam21CudaLargeGranite

        return (Sam21CudaLargeGranite if "large" in name else Sam21CudaGranite)()
    if name == "sam2.1-mlx-granite":
        from .tracking_sam2_granite import Sam21Granite

        return Sam21Granite()
    if name != "sam2.1-mlx":
        raise ValueError(f"Unknown segmentation backend: {name}")
    from .tracking_sam import Sam21MLX

    return Sam21MLX()

"""Reuse the broadcast layout; perspective camera alignment stays recording-specific."""

import copy
import json

from .tracking_geometry import camera_reference, validate_calibration
from .tracking_jobs import atomic


def save_layout(data, record, calibration):
    """Establish a shared default once, without letting later games overwrite it."""
    path = data / "camera-layout.json"
    expected = {(kind, end) for kind in ("overhead", "perspective") for end in ("A", "B")}
    views = calibration.get("views", [])
    if path.exists() or len(views) != 4 or {(v["kind"], v["end"]) for v in views} != expected:
        return
    atomic(
        path,
        dict(
            source_recording=record["id"],
            width=record["media"]["width"],
            height=record["media"]["height"],
            calibration=calibration,
        ),
    )


def read_calibration(data, store, identifier):
    record = store.get(identifier)
    path = data / "recordings" / identifier / "tracking/calibration.json"
    existing = json.loads(path.read_text()) if path.exists() else None
    if existing and existing.get("views"):
        return existing
    empty = existing or dict(
        source_sha256=record.get("source_sha256", ""),
        source_seconds=record["ends"][0]["start_seconds"] if record["ends"] else 0,
        views=[],
        first_delivery_end="B",
        revision=0,
    )
    if not record.get("media") or not record.get("source_sha256"):
        return empty
    profile_path = data / "camera-layout.json"
    if not profile_path.exists():
        # Upgrade catalogs that already have one fully marked broadcast recording.
        candidates = []
        for other in store.all():
            other_path = data / "recordings" / other["id"] / "tracking/calibration.json"
            if other["id"] == identifier or not other_path.exists():
                continue
            try:
                value = validate_calibration(json.loads(other_path.read_text()), other)
            except (ValueError, KeyError, TypeError):
                continue
            if len(value["views"]) == 4 and {(v["kind"], v["end"]) for v in value["views"]} == {
                (kind, end) for kind in ("overhead", "perspective") for end in ("A", "B")
            }:
                candidates.append((other, value))
        if len(candidates) == 1:
            save_layout(data, *candidates[0])
    if not profile_path.exists():
        return empty
    profile = json.loads(profile_path.read_text())
    sx = record["media"]["width"] / profile["width"]
    sy = record["media"]["height"] / profile["height"]
    # A differently shaped composite cannot use these panel boundaries reliably.
    if abs(sx - sy) > 0.001:
        return empty
    value = copy.deepcopy(profile["calibration"])
    value.update(
        source_sha256=record["source_sha256"],
        source_seconds=empty["source_seconds"],
        revision=empty["revision"] + 1,
        first_delivery_end=empty["first_delivery_end"],
        layout_source_recording=profile["source_recording"],
        layout_source_revision=profile["calibration"]["revision"],
    )
    for view in value["views"]:
        view["crop"] = [round(p * (sx if i % 2 == 0 else sy)) for i, p in enumerate(view["crop"])]
        view["landmarks"] = {name: [p[0] * sx, p[1] * sy] for name, p in view["landmarks"].items()}
        if view["kind"] == "perspective":
            view.pop("camera_calibration", None)
            view["mapping_status"] = "inherited_estimate"
        else:
            view["mapping_status"] = "inherited"
            camera = view.get("camera_calibration")
            if camera:
                camera["image_to_ice"] = [
                    p / (sx if i % 3 == 0 else sy if i % 3 == 1 else 1)
                    for i, p in enumerate(camera["image_to_ice"])
                ]
                camera["reference_sha256"] = camera_reference(
                    view, value.get("tee_distance_m", 34.747), value.get("sheet_geometry")
                )
    validate_calibration(value, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic(path, value)
    return value

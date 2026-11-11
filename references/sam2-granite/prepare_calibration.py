"""Prepare the validated independent camera mappings for the calibration API."""

import copy
import json
from pathlib import Path

from hogline2.tracking_geometry import camera_reference, prepare_view, project, sheet_geometry

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/recordings/eDJdwj0JqBY/tracking/392ea127db464389a1b970e627289537"
OUT = ROOT / "data/experiments/camera-calibration-20261108"
job = json.loads((SOURCE / "job.json").read_text())
report = json.loads((SOURCE / "geometry-report.json").read_text())
current = json.loads((SOURCE.parent / "calibration.json").read_text())
assert current["revision"] == job["calibration"]["revision"], "Calibration changed since the audited run"
assert current == job["calibration"], "Calibrate the current camera setup before applying a fit"
assert report["correction_recommended"] and report["status"] == "camera_consistent"
assert report["tee_distance_m"] == current["tee_distance_m"]
updated = copy.deepcopy(current)
geometry = sheet_geometry(updated)
for view in updated["views"]:
    if view["kind"] != "perspective":
        continue
    view["camera_calibration"] = dict(
        method="independent_planar_homography",
        image_to_ice=report["image_to_ice"][view["id"]],
        reference_sha256=camera_reference(view, updated["tee_distance_m"], geometry),
        source_run=SOURCE.name,
        held_out_rms_m=report["held_out_pair_rms_m"],
    )
    prepared = prepare_view(view, updated["tee_distance_m"], geometry)
    # The playable sheet must not cross a projection horizon or fold over itself.
    inverse = prepared["ice_to_image"]
    denominators = [
        inverse[6] * x + inverse[7] * y + inverse[8]
        for x in [-geometry["width_m"] / 2, geometry["width_m"] / 2]
        for y in [-geometry["hack_from_tee_m"], updated["tee_distance_m"] + geometry["hack_from_tee_m"]]
    ]
    assert min(denominators) * max(denominators) > 0, "Camera horizon crosses the playing sheet"
    for x in [-2, 0, 2]:
        for y in [0, 10, 20, updated["tee_distance_m"]]:
            pixel = project(inverse, [x, y])
            restored = project(prepared["image_to_ice"], pixel)
            assert max(abs(restored[0] - x), abs(restored[1] - y)) < 1e-6
(OUT / "calibration-before.json").write_text(json.dumps(current, indent=2))
(OUT / "calibration-candidate.json").write_text(json.dumps(updated, indent=2))
(OUT / "fit-evidence.json").write_text(json.dumps(report, indent=2))
print(
    "Prepared independent perspective transforms; "
    "overhead camera mappings retain their individual ring calibrations."
)

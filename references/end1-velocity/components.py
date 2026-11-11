"""Secondary diagnostic: largest mask component, no temporal/velocity filtering.

Never overwrites the primary model masks, raw velocities, or tracking runs.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from hogline2.tracking_geometry import prepare_view, sheet_geometry
from hogline2.tracking_raw import add_raw_velocity, raw_mask_measurement

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/end1-velocity-20261108"


def main():
    job = json.loads((OUT / "common-job.json").read_text())
    runs = json.loads((OUT / "runs.json").read_text())
    views = {
        v["id"]: prepare_view(v, job["calibration"]["tee_distance_m"], sheet_geometry(job["calibration"]))
        for v in job["calibration"]["views"]
    }
    result, audit = {}, {}
    for backend, path in runs.items():
        folder = Path(path)
        assert json.loads((folder / "status.json").read_text())["state"] == "complete"
        previous, samples, changes = {}, [], []
        for frame in range(round(job["duration"] * job["fps"])):
            packed_path = folder / f"masks/{frame:06}.npz"
            if not packed_path.exists():
                continue
            seconds = job["source_start"] + frame / job["fps"]
            with np.load(packed_path) as packed:
                for key in packed.files:
                    vid, name = key.split("__")
                    view = views[vid]
                    x0, y0, x1, y1 = view["crop"]
                    mask = np.unpackbits(packed[key])[: (x1 - x0) * (y1 - y0)].reshape(y1 - y0, x1 - x0)
                    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
                    if count > 1:
                        component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                        largest = labels == component
                        removed = int(mask.sum()) - int(largest.sum())
                        if removed:
                            changes.append(dict(frame=frame, view=vid, stone=name, removed_pixels=removed))
                        mask = largest
                    measured = raw_mask_measurement(mask, view)
                    if measured is None:
                        previous.pop(key, None)
                        samples.append(
                            dict(
                                frame=frame,
                                source_seconds=seconds,
                                view_id=vid,
                                stone_id=name,
                                valid=False,
                            )
                        )
                    else:
                        measured.update(view_id=vid, stone_id=name, valid=True)
                        measured = add_raw_velocity(previous.get(key), measured, frame, seconds)
                        previous[key] = measured
                        samples.append(measured)
        result[backend] = samples
        audit[backend] = changes
    (OUT / "component-velocity.json").write_text(json.dumps(result))
    (OUT / "component-changes.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

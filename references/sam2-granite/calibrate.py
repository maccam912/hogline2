"""Evaluate camera fits and feed offsets on independently held-out stone tracks."""

import copy
import json
from pathlib import Path

from hogline2.tracking_calibration_fit import fit_geometry

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/sam2-granite-20261108"


def main():
    folder = Path(json.loads((OUT / "run.json").read_text())["path"])
    rows = [json.loads(s) for s in (folder / "observations.jsonl").read_text().splitlines()]
    job = json.loads((folder / "job.json").read_text())
    assert len(rows) == round(job["fps"] * job["duration"])
    calibration = copy.deepcopy(job["calibration"])
    calibration["tee_distance_fixed"] = True
    reports = []
    by_frame = {row["frame"]: row for row in rows}
    for shift in [0, -1, 1, -2, 2, -3, 3]:
        aligned = []
        for row in rows:
            current = [o for o in row["observations"] if o["view_id"] != "perspective_b"]
            other = by_frame.get(row["frame"] + shift, {}).get("observations", [])
            current += [o for o in other if o["view_id"] == "perspective_b"]
            aligned.append({**row, "observations": current})
        fit = fit_geometry(aligned, calibration)
        fit["perspective_b_offset_seconds"] = shift / job["fps"]
        reports.append(fit)
        print(
            shift,
            fit["status"],
            fit.get("baseline_pair_rms_m"),
            fit.get("pair_rms_m"),
            fit.get("held_out_pair_rms_m"),
            flush=True,
        )
    baseline = reports[0]
    eligible = [r for r in reports if r.get("held_out_pair_rms_m") is not None]
    best = min(eligible, key=lambda r: r["held_out_pair_rms_m"]) if eligible else None
    output = dict(
        source_run=folder.name,
        tee_distance_m=calibration["tee_distance_m"],
        length_fixed=True,
        trials=reports,
        best_offset_seconds=best["perspective_b_offset_seconds"] if best else None,
        applied=False,
        baseline=baseline,
        limitations=[
            "Offsets pair A at time t with B at time t+offset; 3fps only resolves one-third-second steps",
            "Motion, spatial calibration, visible rim offsets, and feed latency can be confounded",
            "Same-stone bottom silhouettes in opposite cameras need not be the same physical base point",
            "No independently identified lens distortion or full 3D camera intrinsics/extrinsics",
            "No calibration change is applied by this evaluation script",
        ],
    )
    (OUT / "camera-alignment.json").write_text(json.dumps(output, indent=2))
    (folder / "geometry-report.json").write_text(json.dumps(baseline, indent=2))


if __name__ == "__main__":
    main()

"""Recover the two completed inference runs affected by duplicate manifest kwargs.

Only accepts that exact finalization error and complete 360-frame artifacts.
Retains the error log and marks recovered metadata explicitly.
"""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def recover(folder, snapshot=None):
    status = json.loads((folder / "status.json").read_text())
    assert status["state"] == "failed"
    assert status["error"] == "dict() got multiple values for keyword argument 'raw_comparison'"
    job = json.loads((folder / "job.json").read_text())
    rows = [json.loads(line) for line in (folder / "observations.jsonl").read_text().splitlines()]
    count = round(job["fps"] * job["duration"])
    assert len(rows) == count and [r["frame"] for r in rows] == list(range(count))
    raw = json.loads((folder / "raw-velocity.json").read_text())["samples"]
    assert len(raw) == sum(w["last_frame"] - w["first_frame"] + 1 for w in job["track_windows"])
    video = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(folder / "overlay.mp4")]
        )
    )["streams"][0]
    assert int(video["nb_frames"]) == count
    elapsed = float(re.findall(r"360/360 frames ([\d.]+)s elapsed", (folder / "run.log").read_text())[-1])
    if job["backend"].startswith("sam2"):
        model = json.loads((ROOT / "models/sam2.1/manifest.json").read_text())
        checksum, revision = model["sha256"], model["vendor_revision"]
    else:
        model = json.loads((ROOT / "models/sam3.1-bf16/hogline-manifest.json").read_text())
        checksum = model["sha256"]["model.safetensors"]
        revision = "a992e302ea9b0f03f41dfd93414a4fd0e818f65b"
    names = sorted({w["stone_id"] for w in job["track_windows"]})
    manifest = dict(job)
    manifest.update(
        frames=count,
        elapsed_seconds=elapsed,
        peak_metal_bytes=(snapshot or {}).get("peak_metal_bytes"),
        weights_sha256=checksum,
        vendor_revision=revision,
        motion_model=None,
        stones=[dict(id=n, color=n.split("_")[0]) for n in names],
        finalization_recovery="Complete artifacts verified after duplicate raw_comparison kwarg error",
        timing_note="Inference wall time from final progress log; excludes final serialization",
        memory_note="Last captured progress peak; null when original progress status was lost",
        limitations=[
            "Matched identity and camera windows; not autonomous discovery",
            "No motion gate, Kalman filter, inferred granite foot, smoothing, or interpolated velocity",
            "Camera calibration and mask height can distort projected speed; inspect pixel speed too",
        ],
    )
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    status = dict(
        state="complete",
        stage="Raw velocity comparison ready",
        progress=1,
        backend=job["backend"],
        raw_comparison=True,
        run_id=folder.name,
        elapsed_seconds=elapsed,
        stones=len(names),
        recovered_finalization=True,
    )
    (folder / "status.json").write_text(json.dumps(status, indent=2))
    print(folder.name, "verified and recovered")


if __name__ == "__main__":
    import sys
    import time

    folder = Path(sys.argv[1])
    snapshot = None
    if "--watch" in sys.argv:
        while True:
            latest = json.loads((folder / "status.json").read_text())
            if latest["state"] != "running":
                break
            snapshot = latest
            (folder / "last-progress.json").write_text(json.dumps(snapshot, indent=2))
            time.sleep(3)
    recover(folder, snapshot)

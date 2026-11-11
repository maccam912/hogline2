"""Process ownership, bounded GPU leases and immutable tracking-job construction."""

import contextlib
import fcntl
import json
import subprocess
import time
import uuid
from pathlib import Path


def atomic(path, value):
    temp = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def alive(pid, module, job_path):
    if not pid:
        return False
    command = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
    return module in command and str(job_path) in command


def active_game(data, identifier=None):
    pattern = f"{identifier}/game-jobs/*/status.json" if identifier else "*/game-jobs/*/status.json"
    for path in (Path(data) / "recordings").glob(pattern):
        status = json.loads(path.read_text())
        if status.get("state") not in {"queued", "running", "stopping"}:
            continue
        if alive(status.get("pid"), "hogline2.game_runner", path.parent / "job.json"):
            return path.parent
        if status.get("state") == "queued" and time.time() - path.stat().st_mtime < 20:
            return path.parent
    return None


def active_tracking(data):
    for path in (Path(data) / "recordings").glob("*/tracking/*/status.json"):
        status = json.loads(path.read_text())
        if status.get("state") in {"queued", "running"} and any(
            alive(status.get("pid"), module, path.parent / "job.json")
            for module in ["hogline2.tracking_runner", "hogline2.tracking_comparison"]
        ):
            return path.parent
    return None


@contextlib.contextmanager
def gpu_lease(recordings, exclusive=False):
    """Two normal workers globally; legacy/comparison experiments stay exclusive."""
    recordings = Path(recordings)
    with (recordings / "sam-worker.lock").open("a") as global_lock:
        fcntl.flock(global_lock, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        if exclusive:
            yield
            return
        slot = None
        for i in range(2):
            candidate = (recordings / f"sam-slot-{i}.lock").open("a")
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                slot = candidate
                break
            except BlockingIOError:
                candidate.close()
        if slot is None:
            raise ValueError("Both SAM workers are occupied")
        try:
            yield
        finally:
            slot.close()


def full_end_job(record, calibration, end, fps, game_id):
    delivery_end = calibration["first_delivery_end"]
    if (end["number"] - 1) % 2:
        delivery_end = "A" if delivery_end == "B" else "B"
    view = next(
        (v for v in calibration["views"] if v["kind"] == "perspective" and v["end"] == delivery_end), None
    )
    if not view or not view["landmarks"].get("hack"):
        raise ValueError(f"Mark the delivery hack for house {delivery_end} before tracking the whole game")
    return dict(
        backend="sam2.1-mlx-granite",
        recording_id=record["id"],
        source_path=record["source_path"],
        source_sha256=record["source_sha256"],
        media=record["media"],
        boundary_revision=record["revision"],
        calibration=calibration,
        end=end,
        source_start=end["start_seconds"],
        duration=end["end_seconds"] - end["start_seconds"],
        fps=fps,
        delivery_view=view["id"],
        seeds=[],
        game_job_id=game_id,
    )

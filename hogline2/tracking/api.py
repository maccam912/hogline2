"""Start, stop and report on tracking jobs, and find the tracked result for each end."""

import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import Field

from ..config import PROJECT
from ..files import read_json, write_json
from ..models import Model
from .layout import camera_layout, delivery_end

RUNNER = "hogline2.tracking.runner"
ACTIVE = {"queued", "running", "stopping"}


class TrackRequest(Model):
    ends: list[int] | None = Field(default=None, min_length=1, max_length=99)


class TrackingSettings(Model):
    first_delivery_end: Literal["A", "B"]


def process_alive(pid, job_path):
    """True when `pid` is still the runner for this job (pids get reused)."""
    if not pid:
        return False
    command = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, check=False
    ).stdout
    return RUNNER in command and str(job_path) in command


def job_status(folder):
    """A job's status, corrected to `interrupted` when its process has gone away."""
    status = read_json(folder / "status.json")
    if status is None:
        return None
    if status.get("state") in ACTIVE and not process_alive(status.get("pid"), folder / "job.json"):
        # a freshly queued job has no pid until its process has started
        if status.get("pid") or time.time() - status.get("created_at", 0) > 20:
            status.update(state="interrupted", stage="Interrupted")
            for entry in status.get("ends", []):
                if entry["state"] in {"queued", "running"}:
                    entry["state"] = "stopped"
            write_json(folder / "status.json", status)
    return dict(status, job_id=folder.name)


def active_job(data, identifier=None):
    """The folder of a tracking job that is still running, for one recording or for any."""
    pattern = f"{identifier or '*'}/tracking-jobs/*/status.json"
    for path in (Path(data) / "recordings").glob(pattern):
        status = job_status(path.parent)
        if status and status["state"] in ACTIVE:
            return path.parent
    return None


def latest_runs(tracking):
    """Per end number, the folder of the best complete run: longest interval, then newest."""
    best = {}
    for path in Path(tracking).glob("*/stonetrack-result.json"):
        folder = path.parent
        status, job = read_json(folder / "status.json"), read_json(folder / "job.json")
        if not status or not job or status.get("state") != "complete":
            continue
        key = (job["duration"], status.get("started_at", 0))
        number = job["end"]["number"]
        if number not in best or key > best[number][0]:
            best[number] = (key, folder)
    return {number: folder for number, (_, folder) in best.items()}


def mount_tracking(app, config, store):
    lock = app.state.jobs_lock

    def recording_root(identifier):
        return config.data / "recordings" / identifier

    def jobs(identifier):
        folders = (recording_root(identifier) / "tracking-jobs").glob("*/status.json")
        found = [job_status(path.parent) for path in folders]
        return sorted((j for j in found if j), key=lambda j: j.get("created_at", 0), reverse=True)

    @app.get("/api/recordings/{identifier}/tracking")
    def tracking(identifier: str):
        record = store.get(identifier)
        runs = latest_runs(recording_root(identifier) / "tracking")
        ends = []
        for end in record.get("ends", []):
            folder = runs.get(end["number"])
            run = None
            if folder:
                status, job = read_json(folder / "status.json"), read_json(folder / "job.json")
                result = read_json(folder / "stonetrack-result.json") or {}
                run = dict(
                    run_id=folder.name,
                    tracked_at=status.get("started_at"),
                    stones=len(result.get("stones", [])),
                    check=result.get("check"),
                    whole_end=abs(job["duration"] - (end["end_seconds"] - end["start_seconds"])) < 1
                    and abs(job["source_start"] - end["start_seconds"]) < 1,
                )
            ends.append(
                dict(
                    number=end["number"],
                    start_seconds=end["start_seconds"],
                    end_seconds=end["end_seconds"],
                    delivery_end=delivery_end(record, end["number"]),
                    run=run,
                )
            )
        latest = jobs(identifier)
        return dict(
            first_delivery_end=record.get("first_delivery_end", "B"),
            ends=ends,
            job=latest[0] if latest else None,
        )

    @app.put("/api/recordings/{identifier}/tracking/settings")
    def settings(identifier: str, request: TrackingSettings):
        with lock:
            if active_job(config.data, identifier):
                raise HTTPException(409, "Stop tracking before changing the delivery direction")
            store.update(identifier, first_delivery_end=request.first_delivery_end)
        return tracking(identifier)

    @app.post("/api/recordings/{identifier}/tracking", status_code=202)
    def start(identifier: str, request: TrackRequest):
        with lock:
            record = store.get(identifier)
            if active_job(config.data):
                raise HTTPException(409, "Tracking is already running. Stop it or let it finish first.")
            if record["state"] in {"queued", "running"}:
                raise HTTPException(409, "Wait for the recording to finish downloading and splitting")
            if not record.get("ends"):
                raise HTTPException(400, "Split the recording into ends before tracking")
            if not Path(record.get("source_path", "")).is_file():
                raise HTTPException(400, "The source video is missing")
            numbers = request.ends or [e["number"] for e in record["ends"]]
            chosen = [e for e in record["ends"] if e["number"] in numbers]
            if len(chosen) != len(set(numbers)):
                raise HTTPException(400, "End not found")
            job_id = uuid.uuid4().hex
            folder = recording_root(identifier) / "tracking-jobs" / job_id
            folder.mkdir(parents=True)
            write_json(
                folder / "job.json",
                dict(
                    recording_id=identifier,
                    data_root=str(config.data),
                    source_path=record["source_path"],
                    source_sha256=record["source_sha256"],
                    boundary_revision=record["revision"],
                    layout=camera_layout(config, record),
                ),
                indent=1,
            )
            write_json(
                folder / "status.json",
                dict(
                    state="queued",
                    stage="Starting",
                    progress=0,
                    created_at=time.time(),
                    ends=[
                        dict(
                            end_number=e["number"],
                            end=dict(
                                number=e["number"],
                                start_seconds=e["start_seconds"],
                                end_seconds=e["end_seconds"],
                            ),
                            delivery_end=delivery_end(record, e["number"]),
                            state="queued",
                            progress=0,
                        )
                        for e in chosen
                    ],
                ),
            )
            with (folder / "run.log").open("ab") as log:
                subprocess.Popen(
                    [sys.executable, "-m", RUNNER, str(folder / "job.json")],
                    cwd=PROJECT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            return dict(job_id=job_id, state="queued")

    @app.post("/api/recordings/{identifier}/tracking/stop")
    def stop(identifier: str):
        with lock:
            store.get(identifier)
            folder = active_job(config.data, identifier)
            if folder is None:
                raise HTTPException(409, "Nothing is being tracked for this recording")
            status = read_json(folder / "status.json")
            if process_alive(status.get("pid"), folder / "job.json"):
                try:
                    os.killpg(status["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
            return dict(state="stopping")

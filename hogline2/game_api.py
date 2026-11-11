"""Whole-game jobs, recoverable cleanup and manual player/shot review."""

import json
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from fractions import Fraction
from typing import Literal

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import Field

from .config import PROJECT
from .models import Model
from .tracking_geometry import validate_calibration
from .tracking_jobs import active_game, active_tracking, alive, atomic, full_end_job
from .tracking_layout import read_calibration


class GameRequest(Model):
    concurrency: int = Field(default=1, ge=1, le=2)
    fps: float = Field(default=3, ge=1, le=60)
    execution: Literal["stonetrack", "local"] = "stonetrack"


class TeamLineups(Model):
    red: list[str] = Field(default_factory=lambda: ["", "", "", ""], min_length=4, max_length=4)
    yellow: list[str] = Field(default_factory=lambda: ["", "", "", ""], min_length=4, max_length=4)


class ShotReview(Model):
    player: str = Field(default="", max_length=100)
    rating: int | None = Field(default=None, ge=1, le=5)
    notes: str = Field(default="", max_length=2000)
    identity_checked: bool = False


def mount_game(app, config, store):
    lock = app.state.jobs_lock

    def root(identifier):
        store.get(identifier)
        result = config.data / "recordings" / identifier
        return result

    def folder(identifier, job_id):
        if not re.fullmatch("[a-f0-9]{32}", job_id):
            raise HTTPException(404, "Job not found")
        result = root(identifier) / "game-jobs" / job_id
        if not result.is_dir():
            raise HTTPException(404, "Job not found")
        return result

    def ensure_idle():
        if active_game(config.data) or active_tracking(config.data):
            raise HTTPException(409, "A tracking job is active. Stop it or let it finish first.")

    def spawn(target):
        (target / "stop").unlink(missing_ok=True)
        status = json.loads((target / "status.json").read_text())
        atomic(
            target / "status.json",
            {**status, "state": "queued", "stage": "Queued", "pid": None, "error": None},
        )
        with (target / "run.log").open("ab") as log:
            subprocess.Popen(
                [str(PROJECT / ".venv/bin/python"), "-m", "hogline2.game_runner", str(target / "job.json")],
                cwd=PROJECT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    @app.get("/api/recordings/{identifier}/game-jobs")
    def game_jobs(identifier: str):
        result = []
        for path in (root(identifier) / "game-jobs").glob("*/status.json"):
            value = json.loads(path.read_text())
            stale_queue = value.get("state") == "queued" and time.time() - path.stat().st_mtime >= 20
            if (value.get("state") in {"running", "stopping"} or stale_queue) and not alive(
                value.get("pid"), "hogline2.game_runner", path.parent / "job.json"
            ):
                value.update(state="interrupted", stage="Interrupted — resume keeps completed ends")
                atomic(path, value)
            job = json.loads((path.parent / "job.json").read_text())
            paused = 0
            for entry in value.get("ends", []):
                tracking_status = root(identifier) / "tracking" / entry["run_id"] / "status.json"
                if entry["state"] == "running" and tracking_status.is_file():
                    worker = json.loads(tracking_status.read_text())
                    if worker.get("paused_for_end"):
                        entry.update(state="paused", stage=worker["stage"])
                        paused += 1
            if paused and value["state"] == "running":
                running = sum(e["state"] == "running" for e in value["ends"])
                complete = sum(e["state"] == "complete" for e in value["ends"])
                value["stage"] = (
                    f"Tracking full ends: {complete}/{len(value['ends'])} finished; "
                    f"{running} running; {paused} paused"
                )
            result.append(
                {**value, "job_id": path.parent.name, "concurrency": job["concurrency"], "fps": job["fps"]}
            )
        return sorted(result, key=lambda x: x.get("created_at", 0), reverse=True)

    @app.post("/api/recordings/{identifier}/game-jobs", status_code=202)
    def start_game(identifier: str, request: GameRequest):
        with lock:
            ensure_idle()
            record = store.get(identifier)
            if record["state"] in {"running", "queued"}:
                raise HTTPException(409, "Wait for recording preparation to finish")
            calibration = read_calibration(config.data, store, identifier)
            if not calibration.get("views"):
                raise HTTPException(
                    400,
                    "No shared camera layout is available for this frame size. Set it up in Stone tracking.",
                )
            calibration = validate_calibration(calibration, record)
            if request.fps > float(Fraction(record["media"]["fps"])):
                raise HTTPException(400, "Tracking rate cannot exceed the source frame rate")
            for end in record.get("ends", []) or [
                dict(number=1, start_seconds=0, end_seconds=1),
                dict(number=2, start_seconds=1, end_seconds=2),
            ]:
                full_end_job(record, calibration, end, request.fps, "validation", request.execution)
            if not (PROJECT / ".venv-sam/bin/python").is_file():
                raise HTTPException(400, "Install the SAM environment first")
            job_id = uuid.uuid4().hex
            target = root(identifier) / "game-jobs" / job_id
            target.mkdir(parents=True)
            atomic(
                target / "job.json",
                dict(
                    recording_id=identifier,
                    data_root=str(config.data),
                    concurrency=request.concurrency,
                    fps=request.fps,
                    execution=request.execution,
                    calibration=calibration,
                    boundary_revision=record["revision"] if record.get("ends") else None,
                ),
            )
            atomic(
                target / "status.json",
                dict(state="queued", stage="Queued", progress=0, ends=[], created_at=time.time()),
            )
            spawn(target)
            return dict(job_id=job_id, state="queued")

    @app.post("/api/recordings/{identifier}/game-jobs/{job_id}/stop")
    def stop_game(identifier: str, job_id: str):
        with lock:
            target = folder(identifier, job_id)
            status = json.loads((target / "status.json").read_text())
            (target / "stop").touch()
            if alive(status.get("pid"), "hogline2.game_runner", target / "job.json"):
                status.update(state="stopping", stage="Stopping workers; completed ends are kept")
                atomic(target / "status.json", status)
                try:
                    os.killpg(status["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                for entry in status.get("ends", []):
                    child = root(identifier) / "tracking" / entry["run_id"]
                    if any(
                        alive(entry.get("pid"), module, child / "job.json")
                        for module in ("hogline2.tracking_runner", "hogline2.tracking_stonetrack")
                    ):
                        try:
                            os.kill(entry["pid"], signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                status.update(state="stopped", stage="Stopped — resume keeps completed ends")
                atomic(target / "status.json", status)
            return status

    @app.post("/api/recordings/{identifier}/game-jobs/{job_id}/resume", status_code=202)
    def resume_game(identifier: str, job_id: str):
        with lock:
            ensure_idle()
            target = folder(identifier, job_id)
            job = json.loads((target / "job.json").read_text())
            record = store.get(identifier)
            if record["state"] in {"running", "queued"}:
                raise HTTPException(409, "Wait for recording preparation to finish")
            if job.get("boundary_revision") is not None and job["boundary_revision"] != record["revision"]:
                raise HTTPException(409, "Boundaries changed; start a new whole-game job")
            if job.get("source_sha256") and job["source_sha256"] != record["source_sha256"]:
                raise HTTPException(409, "Source changed; start a new whole-game job")
            spawn(target)
            return dict(job_id=job_id, state="queued")

    @app.post("/api/recordings/{identifier}/tracking/clear")
    def clear_runs(identifier: str):
        with lock:
            ensure_idle()
            recording = root(identifier)
            archive = config.data / "backups" / f"tracking-cleanup-{identifier}-{uuid.uuid4().hex[:12]}"
            archive.mkdir(parents=True)
            moved = []
            for directory in [recording / "tracking", recording / "game-jobs"]:
                for target in directory.glob("*"):
                    if target.is_dir() and re.fullmatch("[a-f0-9]{32}", target.name):
                        dest = archive / directory.name / target.name
                        dest.parent.mkdir(exist_ok=True)
                        shutil.move(str(target), str(dest))
                        moved.append(str(target.relative_to(recording)))
            atomic(
                archive / "manifest.json",
                dict(
                    recording=identifier,
                    moved=moved,
                    preserved="Camera calibration, source, clips and player review notes",
                ),
            )
            return dict(cleared=len(moved), archive=str(archive))

    @app.get("/api/recordings/{identifier}/game-video")
    def game_video(identifier: str):
        record = store.get(identifier)
        clip = record.get("game_clip")
        if not clip or clip["revision"] != record["revision"]:
            raise HTTPException(404, "Process the whole game to create its trimmed recording")
        path = root(identifier) / clip["filename"]
        if not path.is_file():
            raise HTTPException(404, "Trimmed recording is missing")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/recordings/{identifier}/team-lineups")
    def team_lineups(identifier: str):
        path = root(identifier) / "team-lineups.json"
        return json.loads(path.read_text()) if path.is_file() else dict(red=[""] * 4, yellow=[""] * 4)

    @app.put("/api/recordings/{identifier}/team-lineups")
    def save_lineups(identifier: str, request: TeamLineups):
        value = request.model_dump()
        if any(len(name) > 100 for names in value.values() for name in names):
            raise HTTPException(400, "Player names must be at most 100 characters")
        value = {color: [name.strip() for name in names] for color, names in value.items()}
        with lock:
            atomic(root(identifier) / "team-lineups.json", value)
        return value

    def reviews_path(identifier):
        return root(identifier) / "shot-reviews.json"

    @app.get("/api/recordings/{identifier}/shot-reviews")
    def reviews(identifier: str):
        path = reviews_path(identifier)
        return json.loads(path.read_text()) if path.is_file() else {}

    @app.put("/api/recordings/{identifier}/shot-reviews/{run_id}/{stone_id}")
    def save_review(identifier: str, run_id: str, stone_id: str, request: ShotReview):
        if not re.fullmatch("[a-f0-9]{32}", run_id) or not re.fullmatch("(red|yellow)_[1-8]", stone_id):
            raise HTTPException(400, "Invalid stone identity")
        with lock:
            manifest_path = root(identifier) / "tracking" / run_id / "manifest.json"
            if not manifest_path.is_file():
                raise HTTPException(404, "Complete this tracking run before reviewing it")
            manifest = json.loads(manifest_path.read_text())
            if stone_id not in {s["id"] for s in manifest["stones"]}:
                raise HTTPException(404, "Stone not found in this run")
            value = reviews(identifier)
            value.setdefault(run_id, {})[stone_id] = {
                **request.model_dump(),
                "end_number": manifest["end"]["number"],
                "updated_at": time.time(),
            }
            atomic(reviews_path(identifier), value)
            return value[run_id][stone_id]

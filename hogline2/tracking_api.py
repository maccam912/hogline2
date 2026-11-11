import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import uuid
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Literal

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import Field

from .config import PROJECT
from .media import frame
from .models import Model
from .tracking_aim_api import mount_aim
from .tracking_colab import MAX_UPLOAD, import_results, notebook
from .tracking_geometry import validate_calibration
from .tracking_jobs import active_game
from .tracking_layout import read_calibration as load_calibration
from .tracking_layout import save_layout


class CameraCalibration(Model):
    method: Literal["independent_planar_homography"] = "independent_planar_homography"
    image_to_ice: list[float] = Field(min_length=9, max_length=9)
    reference_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_run: str = Field(pattern=r"^[a-f0-9]{32}$")
    held_out_rms_m: float = Field(ge=0)


class View(Model):
    id: str = Field(min_length=1, max_length=40)
    kind: Literal["perspective", "overhead"]
    end: Literal["A", "B"]
    rotation_degrees: Literal[0, 180] = 0
    crop: tuple[int, int, int, int]
    landmarks: dict[str, tuple[float, float]]
    camera_calibration: CameraCalibration | None = None
    mapping_status: Literal["marked", "inherited", "inherited_estimate"] = "marked"


class SheetGeometry(Model):
    length_m: float = Field(default=45.720, gt=0, le=60)
    width_m: float = Field(default=4.750, gt=0, le=10)
    hog_from_tee_m: float = Field(default=6.401, gt=0, le=15)
    hack_from_tee_m: float = Field(default=3.658, gt=0, le=10)
    back_from_tee_m: float = Field(default=1.829, gt=0, le=5)
    house_radius_m: float = Field(default=1.829, gt=0, le=3)


class Calibration(Model):
    source_sha256: str
    source_seconds: float = Field(ge=0)
    first_delivery_end: Literal["A", "B"] = "B"
    views: list[View] = Field(min_length=1, max_length=8)
    tee_distance_m: float = Field(default=34.747, ge=25, le=45)
    tee_distance_source: Literal["nominal", "measured", "estimated", "standard_inferred"] = "nominal"
    venue_id: str | None = Field(default=None, pattern=r"^[a-z0-9_-]{1,60}$")
    sheet_geometry: SheetGeometry = Field(default_factory=SheetGeometry)
    geometry_evidence: str = Field(default="Standard dimensions; not site-surveyed", max_length=2000)
    tee_distance_fixed: bool = False
    revision: int = 0
    layout_source_recording: str | None = None
    layout_source_revision: int | None = None


class Seed(Model):
    view_id: str
    source_seconds: float = Field(ge=0)
    color: Literal["red", "yellow"]
    point: tuple[float, float]


class TrackingRequest(Model):
    end_number: int = Field(ge=1, le=99)
    fps: float = Field(default=3, ge=1, le=60)
    offset_seconds: float = Field(default=0, ge=0)
    duration_seconds: float | None = Field(default=60, gt=0)
    delivery_view: str | None = None
    seeds: list[Seed] = Field(default_factory=list, max_length=16)
    execution: Literal["local", "colab", "stonetrack"] = "stonetrack"
    colab_model: Literal["base_plus", "large"] = "base_plus"


def mount_tracking(app, config, store):
    lock = app.state.jobs_lock
    processes = {}

    def root(identifier):
        store.get(identifier)
        path = config.data / "recordings" / identifier / "tracking"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def read_calibration(identifier):
        with lock:
            return load_calibration(config.data, store, identifier)

    def run_root(identifier, run_id):
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise HTTPException(404, "Tracking run not found")
        result = root(identifier) / run_id
        if not result.is_dir():
            raise HTTPException(404, "Tracking run not found")
        return result

    @app.get("/tracking")
    def page():
        return FileResponse(Path(__file__).parent / "static/tracking.html")

    @app.get("/api/recordings/{identifier}/tracking/frame")
    def tracking_frame(identifier: str, seconds: float = 0):
        record = store.get(identifier)
        if not 0 <= seconds < record["media"]["duration"]:
            raise HTTPException(400, "Frame lies outside the source")
        return Response(
            frame(record["source_path"], seconds, record["media"]["width"]), media_type="image/jpeg"
        )

    @app.get("/api/tracking/venues/{venue_id}")
    def venue_dimensions(venue_id: str):
        if not re.fullmatch(r"[a-z0-9_-]{1,60}", venue_id):
            raise HTTPException(404, "Venue not found")
        path = config.data / "venues" / (venue_id + ".json")
        if not path.is_file():
            raise HTTPException(404, "Venue not found")
        profile = json.loads(path.read_text())
        keys = {
            "venue_id",
            "tee_distance_m",
            "tee_distance_source",
            "sheet_geometry",
            "geometry_evidence",
            "tee_distance_fixed",
        }
        return {k: v for k, v in profile.items() if k in keys}

    @app.get("/api/recordings/{identifier}/tracking/calibration")
    def calibration(identifier: str):
        return read_calibration(identifier)

    @app.put("/api/recordings/{identifier}/tracking/calibration")
    def save_calibration(identifier: str, request: Calibration):
        with lock:
            if active_game(config.data, identifier):
                raise HTTPException(409, "Stop the whole-game job before editing calibration")
            current = read_calibration(identifier)
            if current["revision"] != request.revision:
                raise HTTPException(409, "Calibration changed; reload before saving")
            value = validate_calibration(request.model_dump(mode="json"), store.get(identifier))
            value["revision"] += 1
            target = root(identifier) / "calibration.json"
            temp = target.with_suffix(".tmp")
            temp.write_text(json.dumps(value, indent=2))
            temp.replace(target)
            save_layout(config.data, store.get(identifier), value)
            if value.get("venue_id"):
                venues = config.data / "venues"
                venues.mkdir(exist_ok=True)
                profile_path = venues / (value["venue_id"] + ".json")
                profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
                profile.update(
                    venue_id=value["venue_id"],
                    tee_distance_m=value["tee_distance_m"],
                    tee_distance_source=value["tee_distance_source"],
                    source_recording=identifier,
                    sheet_geometry=value["sheet_geometry"],
                    geometry_evidence=value["geometry_evidence"],
                    tee_distance_fixed=value["tee_distance_fixed"],
                )
                temp = profile_path.with_suffix(".tmp")
                temp.write_text(json.dumps(profile, indent=2))
                temp.replace(profile_path)
            return value

    @app.get("/api/recordings/{identifier}/tracking/runs")
    def runs(identifier: str):
        result = []
        for path in root(identifier).glob("*/status.json"):
            value = json.loads(path.read_text())
            job = json.loads((path.parent / "job.json").read_text())
            if value.get("state") in {"running", "queued"}:
                process = processes.get(path.parent.name)
                alive = process is not None and process.poll() is None
                pid = value.get("pid")
                if not alive and pid:
                    command = subprocess.run(
                        ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, check=False
                    ).stdout
                    alive = (
                        any(
                            runner in command
                            for runner in (
                                "hogline2.tracking_runner",
                                "hogline2.tracking_comparison",
                                "hogline2.tracking_colab",
                                "hogline2.tracking_stonetrack",
                            )
                        )
                        and f"{path.parent.name}/job.json" in command
                    )
                if not alive and value.get("state") == "queued" and job.get("game_job_id"):
                    alive = bool(active_game(config.data, identifier))
                if not alive:
                    value.update(state="interrupted", stage="Interrupted — start a new tracking run")
                    temp = path.with_suffix(".tmp")
                    temp.write_text(json.dumps(value))
                    temp.replace(path)
            manifest_path = path.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            result.append(
                {
                    **value,
                    "run_id": path.parent.name,
                    "trajectory_model": manifest.get("trajectory_model", "rts_constrained_glide_v1")
                    if (path.parent / "trajectory.json").exists()
                    else None,
                    "backend": job.get("backend", "sam2.1-mlx"),
                    "execution": job.get("execution", "local"),
                    "raw_comparison": job.get("raw_comparison", False),
                    "comparison_id": job.get("comparison_id"),
                    "end_number": job["end"]["number"],
                    "source_start": job["source_start"],
                    "duration": job["duration"],
                    "boundary_revision": job["boundary_revision"],
                    "calibration_revision": job["calibration"]["revision"],
                }
            )
        return sorted(result, key=lambda r: r.get("started_at", 0), reverse=True)

    @app.post("/api/recordings/{identifier}/tracking/runs", status_code=202)
    def start(identifier: str, request: TrackingRequest):
        with lock:
            if active_game(config.data):
                raise HTTPException(409, "The whole-game queue owns tracking; use its Stop control")
            if any(process.poll() is None for process in processes.values()):
                raise HTTPException(409, "A SAM run is already active")
            # A worker survives a web-server reload; retain its ownership of the GPU.
            for status_path in (config.data / "recordings").glob("*/tracking/*/status.json"):
                status = json.loads(status_path.read_text())
                if status.get("state") not in {"running", "queued"} or not status.get("pid"):
                    continue
                command = subprocess.run(
                    ["ps", "-p", str(status["pid"]), "-o", "command="],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout
                if (
                    any(
                        runner in command
                        for runner in (
                            "hogline2.tracking_runner",
                            "hogline2.tracking_comparison",
                            "hogline2.tracking_colab",
                            "hogline2.tracking_stonetrack",
                        )
                    )
                    and f"{status_path.parent.name}/job.json" in command
                ):
                    raise HTTPException(409, "A SAM run is already active")
            record = store.get(identifier)
            if record["state"] in {"queued", "running"}:
                raise HTTPException(409, "Wait for end detection/export to finish")
            if request.fps > float(Fraction(record["media"]["fps"])):
                raise HTTPException(400, "Tracking rate cannot exceed the source frame rate")
            calibration = validate_calibration(read_calibration(identifier), record)
            end = next((e for e in record["ends"] if e["number"] == request.end_number), None)
            if end is None:
                raise HTTPException(400, "End not found")
            delivery_end = calibration["first_delivery_end"]
            if (request.end_number - 1) % 2:
                delivery_end = "A" if delivery_end == "B" else "B"
            delivery_view = next((v for v in calibration["views"] if v["id"] == request.delivery_view), None)
            if request.delivery_view is None:
                delivery_view = next(
                    (
                        v
                        for v in calibration["views"]
                        if v["end"] == delivery_end and v["kind"] == "perspective"
                    ),
                    None,
                )
            if not delivery_view or not delivery_view["landmarks"].get("hack"):
                raise HTTPException(400, "Mark the hack in the delivery perspective")
            start_seconds = end["start_seconds"] + request.offset_seconds
            remaining = end["end_seconds"] - start_seconds
            if remaining <= 0:
                raise HTTPException(400, "Preview starts after this end")
            duration = min(request.duration_seconds or remaining, remaining)
            for seed in request.seeds:
                view = next((v for v in calibration["views"] if v["id"] == seed.view_id), None)
                if (
                    not view
                    or not start_seconds <= seed.source_seconds < start_seconds + duration
                    or not view["crop"][0] <= seed.point[0] < view["crop"][2]
                    or not view["crop"][1] <= seed.point[1] < view["crop"][3]
                ):
                    raise HTTPException(400, "Seed lies outside the selected view or interval")
            python = (
                Path(sys.executable) if request.execution == "colab" else PROJECT / ".venv-sam/bin/python"
            )
            if not python.is_file():
                raise HTTPException(400, "Install the isolated SAM environment from docs/SAM_SETUP.md")
            run_id = uuid.uuid4().hex
            folder = root(identifier) / run_id
            folder.mkdir()
            job = dict(
                backend=(
                    "sam2.1-cuda-large-granite" if request.colab_model == "large" else "sam2.1-cuda-granite"
                )
                if request.execution == "colab"
                else "stonetrack-0.1"
                if request.execution == "stonetrack"
                else "sam2.1-mlx-granite",
                execution=request.execution,
                recording_id=identifier,
                source_path=record["source_path"],
                source_sha256=record["source_sha256"],
                media=record["media"],
                boundary_revision=record["revision"],
                calibration=calibration,
                end=end,
                source_start=start_seconds,
                duration=duration,
                # stonetrack's tracking rules are written for 20 frames a second
                fps=20.0 if request.execution == "stonetrack" else request.fps,
                delivery_view=delivery_view["id"],
                seeds=[s.model_dump() for s in request.seeds],
            )
            (folder / "job.json").write_text(json.dumps(job, indent=2))
            (folder / "status.json").write_text(
                json.dumps(
                    dict(
                        state="queued",
                        stage="Preparing Colab"
                        if request.execution == "colab"
                        else "Starting stonetrack"
                        if request.execution == "stonetrack"
                        else "Starting SAM",
                        progress=0,
                    )
                )
            )
            with (folder / "run.log").open("wb") as log:
                processes[run_id] = subprocess.Popen(
                    [
                        str(python),
                        "-m",
                        "hogline2.tracking_colab"
                        if request.execution == "colab"
                        else "hogline2.tracking_stonetrack"
                        if request.execution == "stonetrack"
                        else "hogline2.tracking_runner",
                        str(folder / "job.json"),
                    ],
                    cwd=PROJECT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            return dict(run_id=run_id, state="queued")

    @app.post("/api/recordings/{identifier}/tracking/runs/{run_id}/stop")
    def stop(identifier: str, run_id: str):
        folder = run_root(identifier, run_id)
        if active_game(config.data, identifier):
            raise HTTPException(409, "Use Stop whole game to stop its workers together")
        status = json.loads((folder / "status.json").read_text())
        if status.get("state") == "awaiting_colab":
            status.update(state="stopped", stage="Colab run cancelled in Hogline", progress=0)
            (folder / "status.json").write_text(json.dumps(status))
            return dict(state="stopped")
        process = processes.get(run_id)
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        else:
            status = json.loads((folder / "status.json").read_text())
            pid = status.get("pid")
            command = subprocess.run(
                ["ps", "-p", str(pid or 0), "-o", "command="], capture_output=True, text=True, check=False
            ).stdout
            if (
                not any(
                    runner in command
                    for runner in (
                        "hogline2.tracking_runner",
                        "hogline2.tracking_comparison",
                        "hogline2.tracking_colab",
                        "hogline2.tracking_stonetrack",
                    )
                )
                or f"{folder.name}/job.json" not in command
            ):
                raise HTTPException(409, "No active process belongs to this run")
            os.kill(pid, signal.SIGTERM)
        (folder / "status.json").write_text(
            json.dumps(dict(state="stopped", stage="Stopped by user", progress=0))
        )
        return dict(state="stopped")

    mount_aim(app, store, run_root)

    @app.get("/api/recordings/{identifier}/tracking/runs/{run_id}/colab/bundle")
    def colab_bundle(identifier: str, run_id: str):
        folder = run_root(identifier, run_id)
        if not (folder / "colab-preparation.json").is_file():
            raise HTTPException(409, "Colab bundle is still being prepared")
        return FileResponse(folder / "colab-input.zip", filename="colab-input.zip")

    @app.get("/api/recordings/{identifier}/tracking/runs/{run_id}/colab/notebook")
    def colab_notebook(identifier: str, run_id: str):
        folder = run_root(identifier, run_id)
        if not (folder / "colab-preparation.json").is_file():
            raise HTTPException(409, "Colab notebook is still being prepared")
        return Response(
            json.dumps(notebook(folder), indent=2),
            media_type="application/x-ipynb+json",
            headers={"Content-Disposition": 'attachment; filename="hogline-colab.ipynb"'},
        )

    @app.post("/api/recordings/{identifier}/tracking/runs/{run_id}/colab/results")
    async def colab_results(identifier: str, run_id: str, request: Request):
        folder = run_root(identifier, run_id)
        if json.loads((folder / "status.json").read_text())["state"] != "awaiting_colab":
            raise HTTPException(409, "This run is not awaiting Colab results")
        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", dir=folder) as upload:
                size = 0
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, "Colab result ZIP exceeds 2 GiB")
                    upload.write(chunk)
                upload.flush()
                with lock:
                    if active_game(config.data, identifier):
                        raise HTTPException(409, "Stop the whole-game queue before importing this run")
                    import_results(folder, Path(upload.name), store.get(identifier))
        except (ValueError, KeyError, TypeError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
            raise HTTPException(400, f"Could not import Colab results: {exc}") from exc
        return dict(run_id=run_id, state="complete")

    @app.get("/api/recordings/{identifier}/tracking/runs/{run_id}/{artifact}")
    def artifact(identifier: str, run_id: str, artifact: str):
        if artifact not in {
            "overlay.mp4",
            "manifest.json",
            "observations.jsonl",
            "events.json",
            "trajectory.json",
            "trajectory-summary.json",
            "stonetrack-result.json",
            "geometry-report.json",
            "raw-velocity.json",
            "comparison.html",
            "component-comparison.html",
            "comparison.mp4",
        }:
            raise HTTPException(404, "Unknown tracking artifact")
        folder = run_root(identifier, run_id)
        path = folder / artifact
        if not path.is_file():
            raise HTTPException(404, "Artifact is not ready")
        return FileResponse(path)

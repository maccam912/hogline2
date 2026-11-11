"""Recordings: add one, split it into ends, edit the cuts, export and serve the clips."""

from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import Field

from ..media import youtube_id
from ..models import End, Model, validate_ends
from ..tracking.api import active_job
from .worker import BUSY


class AddRecording(Model):
    url: str = Field(max_length=2000)
    expected_ends: int | None = Field(default=None, ge=1, le=99)


class EditEnds(Model):
    revision: int
    ends: list[End] = Field(min_length=1, max_length=99)


def mount_recordings(app, config, store, worker):
    @app.get("/api/status")
    def status():
        return dict(vision_configured=bool(config.api_key), vision_model=config.model)

    @app.get("/api/recordings")
    def recordings():
        return store.all()

    @app.get("/api/recordings/{identifier}")
    def recording(identifier: str):
        return store.get(identifier)

    @app.post("/api/recordings", status_code=202)
    def add(request: AddRecording):
        identifier = youtube_id(request.url)
        with worker.lock:
            try:
                return store.get(identifier)
            except KeyError:
                store.create(
                    identifier,
                    url=f"https://www.youtube.com/watch?v={identifier}",
                    expected_ends=request.expected_ends,
                )
            worker.submit(identifier)
        return store.get(identifier)

    @app.post("/api/recordings/{identifier}/retry", status_code=202)
    def retry(identifier: str):
        record = store.get(identifier)
        worker.submit(identifier, record.get("action", "analyze"))
        return store.get(identifier)

    @app.post("/api/recordings/{identifier}/analyze", status_code=202)
    def analyze(identifier: str):
        worker.submit(identifier, "analyze")
        return store.get(identifier)

    @app.put("/api/recordings/{identifier}/ends")
    def edit(identifier: str, request: EditEnds):
        with worker.lock:
            record = store.get(identifier)
            if active_job(config.data, identifier):
                raise HTTPException(409, "Stop tracking before editing the cuts")
            if record["state"] in BUSY:
                raise HTTPException(409, "Wait for the current job before editing the cuts")
            if request.revision != record["revision"]:
                raise HTTPException(409, "The cuts changed elsewhere. Reload before saving.")
            ends = validate_ends(
                [e.model_dump() for e in request.ends], record.get("media", {}).get("duration", 0)
            )
            return store.update(
                identifier,
                ends=ends,
                revision=record["revision"] + 1,
                clips=[],
                boundary_origin="user_edited",
                state="edited",
                stage="Cuts saved",
                warnings=[],
                error=None,
            )

    @app.post("/api/recordings/{identifier}/export", status_code=202)
    def export(identifier: str):
        record = store.get(identifier)
        validate_ends(record["ends"], record.get("media", {}).get("duration", 0))
        worker.submit(identifier, "export")
        return store.get(identifier)

    @app.get("/api/recordings/{identifier}/source")
    def source(identifier: str):
        path = Path(store.get(identifier).get("source_path", ""))
        if not path.is_file():
            raise HTTPException(404, "The video is not available yet")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/recordings/{identifier}/clips/{number}")
    def clip(identifier: str, number: int):
        record = store.get(identifier)
        match = next(
            (c for c in record["clips"] if c["number"] == number and c["revision"] == record["revision"]),
            None,
        )
        if not match:
            raise HTTPException(404, "Export the clips for the current cuts first")
        path = config.data / "recordings" / identifier / match["filename"]
        if not path.is_file():
            raise HTTPException(404, "The clip file is missing; export again")
        return FileResponse(
            path, media_type="video/mp4", filename=match["filename"], content_disposition_type="inline"
        )

    @app.get("/api/recordings/{identifier}/manifest")
    def manifest(identifier: str):
        record = store.get(identifier)
        keys = ["id", "title", "url", "source_sha256", "media", "revision", "ends", "clips", "warnings"]
        return {k: record.get(k) for k in [*keys, "boundary_origin"]}

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field

from .config import Config
from .media import youtube_id
from .models import End, Model, validate_ends
from .pipeline import BUSY, Worker
from .store import Store


class AddRecording(Model):
    url: str = Field(max_length=2000)
    expected_ends: int | None = Field(default=None, ge=1, le=99)


class EditEnds(Model):
    revision: int
    ends: list[End] = Field(min_length=1, max_length=99)


def create_app(config=None, start_worker=True):
    config = config or Config.load()
    store = Store(config.data)
    worker = Worker(config, store)

    @asynccontextmanager
    async def lifespan(app):
        if start_worker:
            worker.start()
        yield
        if start_worker:
            worker.stop()

    app = FastAPI(title="Hogline 2", lifespan=lifespan)
    app.state.store, app.state.worker = store, worker
    app.state.jobs_lock = worker.lock
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": "Recording not found"}, status_code=404)

    @app.get("/")
    def home():
        return FileResponse(static / "index.html")

    @app.get("/api/status")
    def status():
        return dict(
            vision_configured=bool(config.api_key),
            vision_model=config.model,
            segmentation="available_on_request",
            sam_installed=(Path(__file__).resolve().parents[1] / ".venv-sam/bin/python").is_file(),
        )

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
                existing = store.get(identifier)
                return existing
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
            from .tracking_jobs import active_game

            if active_game(config.data, identifier):
                raise HTTPException(409, "Stop the whole-game job before editing boundaries")
            if record["state"] in BUSY:
                raise HTTPException(409, "Wait for the current job before editing boundaries")
            if request.revision != record["revision"]:
                raise HTTPException(409, "Boundaries changed. Reload before saving.")
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
                stage="Cuts saved — export clips",
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
        record = store.get(identifier)
        path = Path(record.get("source_path", ""))
        if not path.is_file():
            raise HTTPException(404, "Source is not available yet")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/recordings/{identifier}/clips/{number}")
    def clip(identifier: str, number: int):
        record = store.get(identifier)
        match = next(
            (c for c in record["clips"] if c["number"] == number and c["revision"] == record["revision"]),
            None,
        )
        if not match:
            raise HTTPException(404, "Export this boundary revision first")
        path = config.data / "recordings" / identifier / match["filename"]
        if not path.is_file():
            raise HTTPException(404, "Clip file is missing; export again")
        return FileResponse(
            path, media_type="video/mp4", filename=match["filename"], content_disposition_type="inline"
        )

    @app.get("/api/recordings/{identifier}/manifest")
    def manifest(identifier: str):
        record = store.get(identifier)
        return {
            k: record.get(k)
            for k in [
                "id",
                "title",
                "url",
                "source_sha256",
                "media",
                "revision",
                "ends",
                "clips",
                "warnings",
                "boundary_origin",
                "segmentation",
            ]
        }

    from .tracking_api import mount_tracking

    mount_tracking(app, config, store)
    from .game_api import mount_game

    mount_game(app, config, store)
    return app

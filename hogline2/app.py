from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Config
from .ends.api import mount_recordings
from .ends.worker import Worker
from .shots.api import mount_shots
from .store import Store
from .tracking.api import mount_tracking

STATIC = Path(__file__).parent / "static"


class Fresh(StaticFiles):
    """Static files the browser must revalidate, so an updated UI is never mixed with a cached one."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


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
    app.mount("/static", Fresh(directory=STATIC), name="static")

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Recording not found"}, status_code=404)

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    mount_recordings(app, config, store, worker)
    mount_tracking(app, config, store)
    mount_shots(app, config, store)
    return app

"""Tracking jobs: layout, job bookkeeping and the runner, with stonetrack itself stubbed out."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from hogline2.app import create_app
from hogline2.config import Config
from hogline2.tracking import api, runner
from hogline2.tracking.layout import camera_layout, delivery_end

MEDIA = dict(width=1920, height=1080, duration=4000, fps="60/1")
ENDS = [
    dict(number=1, start_seconds=100, end_seconds=900),
    dict(number=2, start_seconds=950, end_seconds=1800),
]


@pytest.fixture
def game(tmp_path):
    app = create_app(Config(tmp_path), start_worker=False)
    source = tmp_path / "recordings" / "game" / "source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    app.state.store.create(
        "game", state="ready", revision=1, ends=ENDS, media=MEDIA, source_path=str(source), source_sha256="s"
    )
    return app, TestClient(app), tmp_path


def test_default_layout_scales_and_directions_alternate(tmp_path):
    record = dict(id="game", media=dict(width=960, height=540))
    layout = camera_layout(Config(tmp_path), record)
    assert [v["id"] for v in layout["views"]] == [
        "perspective_a",
        "perspective_b",
        "overhead_a",
        "overhead_b",
    ]
    assert layout["views"][1]["crop"] == [600, 0, 960, 495] and layout["views"][2]["rotation_degrees"] == 180
    with pytest.raises(ValueError, match="layout"):
        camera_layout(Config(tmp_path), dict(id="game", media=dict(width=1000, height=1000)))
    assert [delivery_end({}, n) for n in (1, 2, 3)] == ["B", "A", "B"]
    assert delivery_end(dict(first_delivery_end="A"), 2) == "B"


def test_start_writes_a_job_and_refuses_a_second(game, monkeypatch):
    app, client, root = game
    launched = []

    class Process:
        pid = os.getpid()

    monkeypatch.setattr(api.subprocess, "Popen", lambda command, **_: launched.append(command) or Process())
    assert client.post("/api/recordings/game/tracking", json=dict(ends=[3])).status_code == 400
    started = client.post("/api/recordings/game/tracking", json=dict(ends=[2]))
    assert started.status_code == 202 and launched[0][1:3] == ["-m", "hogline2.tracking.runner"]
    folder = root / "recordings" / "game" / "tracking-jobs" / started.json()["job_id"]
    status = json.loads((folder / "status.json").read_text())
    assert [(e["end_number"], e["delivery_end"], e["state"]) for e in status["ends"]] == [(2, "A", "queued")]
    assert len(json.loads((folder / "job.json").read_text())["layout"]["views"]) == 4
    # still queued and young: it counts as running, so nothing else may start or change
    assert client.post("/api/recordings/game/tracking", json={}).status_code == 409
    assert client.put("/api/recordings/game/ends", json=dict(revision=1, ends=ENDS)).status_code == 409
    settings = client.put("/api/recordings/game/tracking/settings", json=dict(first_delivery_end="A"))
    assert settings.status_code == 409


def test_a_job_whose_process_died_is_reported_interrupted(game):
    app, client, root = game
    folder = root / "recordings" / "game" / "tracking-jobs" / "old"
    folder.mkdir(parents=True)
    entry = dict(end_number=1, state="running", progress=0.4)
    (folder / "status.json").write_text(json.dumps(dict(state="running", pid=1, created_at=1, ends=[entry])))
    job = client.get("/api/recordings/game/tracking").json()["job"]
    assert job["state"] == "interrupted" and job["ends"][0]["state"] == "stopped"
    assert client.post("/api/recordings/game/tracking/stop").status_code == 409


def test_runner_tracks_each_end_and_the_app_finds_the_results(game, monkeypatch):
    app, client, root = game
    folder = root / "recordings" / "game" / "tracking-jobs" / "job"
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(
        json.dumps(
            dict(
                recording_id="game",
                data_root=str(root),
                source_path="source.mp4",
                source_sha256="s",
                boundary_revision=1,
                layout=dict(views=[]),
            )
        )
    )
    entries = [
        dict(
            end_number=e["number"],
            end=e,
            delivery_end=delivery_end({}, e["number"]),
            state="queued",
            progress=0,
        )
        for e in ENDS
    ]
    (folder / "status.json").write_text(json.dumps(dict(state="queued", ends=entries)))
    monkeypatch.setattr(runner, "prepare", lambda *args: None)

    def fake_track(work, start, end, side, say):
        if start == 950:
            raise ValueError("No video frames were found in this end")
        stone = dict(id="R1", colour="red", delivered_at=start + 5, status="rest", path=[])
        return dict(delivery_end=side, start=start, end=end, stones=[stone], check=None)

    monkeypatch.setattr(runner, "track_end", fake_track)
    runner.Runner(folder / "job.json").run()
    status = json.loads((folder / "status.json").read_text())
    assert status["state"] == "failed" and "end 2" in status["stage"]
    assert [e["state"] for e in status["ends"]] == ["complete", "failed"]
    assert "No video frames" in status["ends"][1]["error"]

    listing = client.get("/api/recordings/game/tracking").json()
    assert listing["ends"][0]["run"]["stones"] == 1 and listing["ends"][0]["run"]["whole_end"]
    assert listing["ends"][1]["run"] is None and listing["ends"][1]["delivery_end"] == "A"

    # running the same job again only redoes the end that failed
    monkeypatch.setattr(
        runner, "track_end", lambda work, start, end, side, say: fake_track(work, 0, end, side, say)
    )
    runner.Runner(folder / "job.json").run()
    assert json.loads((folder / "status.json").read_text())["state"] == "complete"
    assert len(api.latest_runs(root / "recordings" / "game" / "tracking")) == 2

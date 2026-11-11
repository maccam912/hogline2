import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_tracking import view

from hogline2.app import create_app
from hogline2.config import Config
from hogline2.game_runner import GameRunner
from hogline2.store import Store
from hogline2.tracking_jobs import atomic, full_end_job, gpu_lease


def setup_record(tmp_path, count=3):
    store = Store(tmp_path)
    root = tmp_path / "recordings/demo"
    (root / "tracking").mkdir(parents=True)
    source = root / "source.mp4"
    source.write_bytes(b"fixture source")
    ends = [
        dict(number=i + 1, start_seconds=i * 10, end_seconds=(i + 1) * 10, evidence="", warnings=[])
        for i in range(count)
    ]
    clips = []
    for end in ends:
        name = f"end-{end['number']}.mp4"
        (root / name).touch()
        clips.append(dict(number=end["number"], revision=0, filename=name))
    (root / "game.mp4").touch()
    record = store.create(
        "demo",
        state="ready",
        source_path=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        media=dict(width=500, height=500, duration=100, fps="30/1"),
        ends=ends,
        clips=clips,
        game_clip=dict(revision=0, filename="game.mp4"),
    )
    views = []
    for side in ["A", "B"]:
        v = view()
        v.update(id=side, kind="perspective", end=side)
        v["landmarks"]["hack"] = [250, 30]
        views.append(v)
    cal = dict(
        source_sha256=record["source_sha256"],
        source_seconds=0,
        views=views,
        first_delivery_end="B",
        revision=5,
    )
    atomic(root / "tracking/calibration.json", cal)
    return store, record, cal


def test_full_end_jobs_preserve_full_duration_and_alternate_direction(tmp_path):
    _, record, cal = setup_record(tmp_path)
    tasks = [full_end_job(record, cal, e, 3, "game") for e in record["ends"]]
    assert [t["delivery_view"] for t in tasks] == ["B", "A", "B"]
    assert all(t["duration"] == 10 and t["calibration"]["revision"] == 5 for t in tasks)
    assert [t["source_start"] for t in tasks] == [0, 10, 20]


def test_two_gpu_slots_and_exclusive_experiment(tmp_path):
    with gpu_lease(tmp_path):
        with gpu_lease(tmp_path):
            with pytest.raises(ValueError, match="occupied"):
                with gpu_lease(tmp_path):
                    pass
        with pytest.raises(BlockingIOError):
            with gpu_lease(tmp_path, exclusive=True):
                pass
    with gpu_lease(tmp_path, exclusive=True):
        with pytest.raises(BlockingIOError):
            with gpu_lease(tmp_path):
                pass


def test_coordinator_bounded_parallelism_and_resume_skips_completed_ends(tmp_path):
    _, record, cal = setup_record(tmp_path)
    folder = tmp_path / "recordings/demo/game-jobs" / ("f" * 32)
    folder.mkdir(parents=True)
    atomic(
        folder / "job.json",
        dict(
            data_root=str(tmp_path),
            recording_id="demo",
            fps=3,
            concurrency=2,
            calibration=cal,
            boundary_revision=0,
        ),
    )
    atomic(folder / "status.json", dict(state="queued", ends=[]))
    processes = []

    class FakeProcess:
        pid = 12345

        def __init__(self, args, **kwargs):
            self.job = Path(args[-1])
            self.returncode = None
            processes.append(self)

        def poll(self):
            return self.returncode

        def finish(self, success=True):
            atomic(
                self.job.parent / "status.json", dict(state="complete" if success else "failed", progress=1)
            )
            self.returncode = 0 if success else 1

    runner = GameRunner(folder / "job.json", popen=FakeProcess)
    runner.prepare()
    assert runner.tick()
    assert len(processes) == 2 and len(runner.children) == 2
    assert runner.status["ends"][2]["state"] == "queued"
    processes[0].finish()
    assert runner.tick()
    assert len(processes) == 3 and len(runner.children) == 2
    completed = runner.status["ends"][0]["run_id"]
    processes[1].finish(False)
    processes[2].finish()
    assert not runner.tick()
    resumed = GameRunner(folder / "job.json", popen=FakeProcess)
    resumed.prepare()
    assert resumed.status["ends"][0]["run_id"] == completed
    assert [e["state"] for e in resumed.status["ends"]] == ["complete", "queued", "complete"]
    assert resumed.tick() and len(resumed.children) == 1


def test_game_api_archive_preserves_calibration_and_lineups(tmp_path, monkeypatch):
    _, record, cal = setup_record(tmp_path)
    client = TestClient(create_app(Config(tmp_path), start_worker=False))
    base = "/api/recordings/demo"
    assert (
        client.put(
            base + "/team-lineups", json=dict(red=["Lead", "Second", "Third", "Skip"], yellow=[""] * 4)
        ).status_code
        == 200
    )
    run_id = "a" * 32
    run = tmp_path / "recordings/demo/tracking" / run_id
    run.mkdir()
    atomic(run / "manifest.json", dict(stones=[dict(id="red_1")], end=dict(number=1)))
    assert (
        client.put(
            base + "/shot-reviews/" + run_id + "/red_1",
            json=dict(player="Lead", rating=4, identity_checked=True),
        ).status_code
        == 200
    )
    before = (run.parent / "calibration.json").read_bytes()
    response = client.post(base + "/tracking/clear")
    assert response.status_code == 200
    assert response.json()["cleared"] == 1 and not run.exists()
    assert (run.parent / "calibration.json").read_bytes() == before
    assert client.get(base + "/team-lineups").json()["red"][0] == "Lead"
    assert client.get(base + "/shot-reviews").json()[run_id]["red_1"]["rating"] == 4
    launches = []
    monkeypatch.setattr("hogline2.game_api.subprocess.Popen", lambda *a, **kw: launches.append((a, kw)))
    response = client.post(base + "/game-jobs", json=dict(concurrency=2, fps=3))
    assert response.status_code == 202 and len(launches) == 1
    job_id = response.json()["job_id"]
    assert client.post(base + "/game-jobs", json={}).status_code == 409
    assert client.post(base + "/tracking/clear").status_code == 409
    assert client.put(base + "/ends", json=dict(revision=0, ends=record["ends"])).status_code == 409
    assert client.put(base + "/tracking/calibration", json=cal).status_code == 409
    assert client.post(base + "/game-jobs/" + job_id + "/stop").status_code == 200
    assert client.post(base + "/game-jobs/" + job_id + "/resume").status_code == 202
    assert len(launches) == 2


def test_game_api_limits_and_calibration_prerequisite(tmp_path):
    setup_record(tmp_path)
    client = TestClient(create_app(Config(tmp_path), start_worker=False))
    base = "/api/recordings/demo"
    assert client.post(base + "/game-jobs", json=dict(concurrency=3)).status_code == 422
    assert client.post(base + "/game-jobs", json=dict(fps=40)).status_code == 400
    (tmp_path / "recordings/demo/tracking/calibration.json").unlink()
    assert client.post(base + "/game-jobs", json={}).status_code == 400
    assert client.put(base + "/team-lineups", json=dict(red=["one"], yellow=[""] * 4)).status_code == 422

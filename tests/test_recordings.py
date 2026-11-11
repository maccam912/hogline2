import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from hogline2.app import create_app
from hogline2.config import Config
from hogline2.ends.worker import Worker
from hogline2.media import enclosing_cut, export_ends, fingerprint, first_video_packet, probe, youtube_id
from hogline2.models import validate_ends
from hogline2.store import Store


@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/eDJdwj0JqBY?t=60",
        "https://www.youtube.com/watch?v=eDJdwj0JqBY",
        "https://youtube.com/live/eDJdwj0JqBY",
        "https://m.youtube.com/embed/eDJdwj0JqBY",
    ],
)
def test_urls(url):
    assert youtube_id(url) == "eDJdwj0JqBY"


@pytest.mark.parametrize(
    "url",
    ["file:///tmp/video", "https://youtube.com.evil.test/watch?v=eDJdwj0JqBY", "https://youtu.be/short"],
)
def test_bad_urls(url):
    with pytest.raises(ValueError):
        youtube_id(url)


def test_end_invariants():
    ends = [dict(number=1, start_seconds=2, end_seconds=30), dict(number=2, start_seconds=35, end_seconds=70)]
    assert len(validate_ends(ends, 100)) == 2
    for bad in [
        [],
        [dict(number=1, start_seconds=3, end_seconds=2)],
        [dict(number=1, start_seconds=0, end_seconds=float("nan"))],
        [dict(number=2, start_seconds=0, end_seconds=10)],
        [*ends[:1], dict(number=2, start_seconds=25, end_seconds=60)],
        [dict(number=1, start_seconds=0, end_seconds=101)],
    ]:
        with pytest.raises(ValueError):
            validate_ends(bad, 100)


def test_enclosing_not_trimming():
    assert enclosing_cut(2.3, 5.8, [0, 2, 4, 6, 8], 10) == (2, 6)
    assert enclosing_cut(2, 6, [0, 2, 4, 6, 8], 10) == (2, 6)
    assert enclosing_cut(8.2, 9.9, [0, 2, 4, 6, 8], 10) == (8, 10)


@pytest.fixture
def media_record(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    config = Config(tmp_path)
    root = tmp_path / "recordings" / "test"
    root.mkdir(parents=True)
    source = root / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "8",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-g",
            "60",
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    store = Store(tmp_path)
    record = store.create(
        "test", source_path=str(source), source_sha256=fingerprint(source), media=probe(source)
    )
    return config, store, record


def test_original_stream_exports(media_record):
    config, store, record = media_record
    record = store.update(
        "test",
        revision=1,
        ends=validate_ends(
            [
                dict(number=1, start_seconds=0.6, end_seconds=3.3),
                dict(number=2, start_seconds=4.4, end_seconds=7.1),
            ],
            8,
        ),
    )
    clips = export_ends(config, record, lambda *args: None)
    assert len(clips) == 2
    assert [(c["source_start"], c["source_end"]) for c in clips] == [(0, 4), (4, 8)]
    for c in clips:
        path = config.data / "recordings/test" / c["filename"]
        assert probe(path)["fps"] == "30/1"
        assert c["requested_start"] >= c["source_start"]
        assert c["requested_end"] <= c["source_end"]
        assert c["sha256"] == fingerprint(path)
        source_packet = first_video_packet(record["source_path"], c["requested_keyframe_start"])
        assert c["initial_packet_sha256"] == source_packet["data_hash"]
        assert c["source_start"] == float(source_packet["pts_time"])
        assert first_video_packet(path)["data_hash"] == source_packet["data_hash"]


def test_api_revision_conflict_and_stale_clip(media_record):
    config, store, record = media_record
    client = TestClient(create_app(config, start_worker=False))
    assert "vision_configured" in client.get("/api/status").json()
    assert client.get("/api/recordings/test/source", headers={"Range": "bytes=0-99"}).status_code == 206
    ends = [dict(number=1, start_seconds=0, end_seconds=4)]
    result = client.put("/api/recordings/test/ends", json=dict(revision=0, ends=ends))
    assert result.status_code == 200
    assert result.json()["revision"] == 1
    assert client.put("/api/recordings/test/ends", json=dict(revision=0, ends=ends)).status_code == 409
    assert client.get("/api/recordings/test/clips/1").status_code == 404
    assert client.post("/api/recordings/test/export").status_code == 202
    assert client.put("/api/recordings/test/ends", json=dict(revision=1, ends=ends)).status_code == 409


def test_one_step_and_dedup(tmp_path):
    app = create_app(Config(tmp_path), start_worker=False)
    client = TestClient(app)
    body = {"url": "https://youtube.com/watch?v=eDJdwj0JqBY", "expected_ends": 6}
    assert client.post("/api/recordings", json=body).json()["state"] == "queued"
    assert client.post("/api/recordings", json=body).json()["id"] == "eDJdwj0JqBY"
    assert len(client.get("/api/recordings").json()) == 1
    assert app.state.worker.queue.qsize() == 1


def test_resume_requires_explicit_action(tmp_path):
    store = Store(tmp_path)
    store.create("test", state="running")
    worker = Worker(Config(tmp_path), store)
    worker.start()
    try:
        assert store.get("test")["state"] == "interrupted"
        assert worker.queue.empty()
    finally:
        worker.stop()


def test_pipeline_does_not_force_expected_count(media_record, monkeypatch):
    config, store, record = media_record
    store.update("test", expected_ends=6)
    monkeypatch.setattr(
        "hogline2.ends.worker.EndDetector.detect",
        lambda *args: (validate_ends([dict(number=1, start_seconds=0, end_seconds=7)], 8), []),
    )
    Worker(config, store).run("test", "analyze")
    result = store.get("test")
    assert result["state"] == "ready"
    assert len(result["ends"]) == len(result["clips"]) == 1
    assert "Expected 6 ends, detected 1" in result["warnings"][0]

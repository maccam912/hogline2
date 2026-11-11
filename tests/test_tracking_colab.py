import hashlib
import json
import subprocess
import zipfile

import pytest
from fastapi.testclient import TestClient
from test_game_pipeline import setup_record

from hogline2.app import create_app
from hogline2.config import Config
from hogline2.tracking_colab import digest, import_results, notebook, prepare
from hogline2.tracking_jobs import atomic
from hogline2.tracking_sam2_specs import CHECKPOINTS, SAM_REVISION


@pytest.fixture
def pending(tmp_path, monkeypatch):
    store, record, _ = setup_record(tmp_path)
    commands = []

    class Process:
        def __init__(self, args, **kwargs):
            commands.append(args)

        def poll(self):
            return None

    monkeypatch.setattr("hogline2.tracking_api.subprocess.Popen", Process)
    app = create_app(Config(tmp_path), start_worker=False)
    client = TestClient(app)
    response = client.post(
        "/api/recordings/demo/tracking/runs",
        json={
            "end_number": 2,
            "duration_seconds": 2,
            "fps": 3,
            "execution": "colab",
        },
    )
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    folder = tmp_path / "recordings/demo/tracking" / run_id
    job = json.loads((folder / "job.json").read_text())
    assert commands[0][2] == "hogline2.tracking_colab"
    assert job["backend"] == "sam2.1-cuda-granite"
    assert job["source_start"] == 10 and job["delivery_view"] == "A"
    atomic(folder / "status.json", dict(state="awaiting_colab", stage="Ready"))
    atomic(
        folder / "colab-preparation.json",
        dict(
            frames=6,
            job_sha256=digest(folder / "job.json"),
            bundle_sha256="b" * 64,
            source_media_sha256="c" * 64,
        ),
    )
    return client, store, folder, job, record


def result_zip(folder, job, *, run_id=None, source_start=None, extra=None, corrupt=False):
    info = json.loads((folder / "colab-preparation.json").read_text())
    manifest = dict(
        **job,
        frames=6,
        weights_sha256=CHECKPOINTS["base_plus"][2],
        vendor_revision=SAM_REVISION,
        source_media_sha256=info["source_media_sha256"],
    )
    manifest["source_path"] = "/content/media.mp4"
    contents = {
        "manifest.json": json.dumps(manifest).encode(),
        "status.json": json.dumps(dict(state="complete", pid=123, progress=1)).encode(),
        "events.json": b"[]",
        "overlay.mp4": b"fixture overlay",
        "observations.jsonl": "".join(
            json.dumps(
                dict(
                    frame=i,
                    source_seconds=(source_start if source_start is not None else job["source_start"])
                    + i / job["fps"],
                    stones=[],
                    observations=[],
                    masks=None,
                )
            )
            + "\n"
            for i in range(6)
        ).encode(),
    }
    if extra:
        contents[extra] = b"unexpected"
    envelope = dict(
        schema=1,
        run_id=run_id or folder.name,
        job_sha256=info["job_sha256"],
        files={name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
    )
    if corrupt:
        contents["events.json"] = b"[0]"
    archive = folder / "test-results.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("result.json", json.dumps(envelope))
        for name, value in contents.items():
            output.writestr(name, value)
    return archive


def test_colab_import_round_trip_and_cannot_overwrite_completed_run(pending):
    client, _, folder, job, _ = pending
    archive = result_zip(folder, job)
    url = f"/api/recordings/demo/tracking/runs/{folder.name}/colab/results"
    response = client.post(url, content=archive.read_bytes(), headers={"Content-Type": "application/zip"})
    assert response.status_code == 200, response.text
    status = json.loads((folder / "status.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    assert status["state"] == "complete" and "pid" not in status
    assert manifest["source_path"] == job["source_path"]
    assert manifest["source_start"] == 10
    assert client.get(f"/api/recordings/demo/tracking/runs/{folder.name}/manifest.json").status_code == 200
    assert client.post(url, content=archive.read_bytes()).status_code == 409


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_id": "f" * 32},
        {"source_start": 0},
        {"corrupt": True},
        {"extra": "../escape.py"},
        {"extra": "comparison.html"},
        {"extra": "job.json"},
    ],
)
def test_rejects_wrong_run_timestamps_corruption_and_unsafe_files(pending, kwargs):
    _, _, folder, job, record = pending
    archive = result_zip(folder, job, **kwargs)
    with pytest.raises(ValueError):
        import_results(folder, archive, record)
    assert not (folder / "manifest.json").exists()
    assert json.loads((folder / "status.json").read_text())["state"] == "awaiting_colab"


def test_rejects_stale_boundaries_and_pending_run_survives_reload(pending):
    _, store, folder, job, record = pending
    archive = result_zip(folder, job)
    with pytest.raises(ValueError, match="boundaries"):
        import_results(folder, archive, {**record, "revision": record["revision"] + 1})
    client = TestClient(create_app(Config(store.root), start_worker=False))
    runs = client.get("/api/recordings/demo/tracking/runs").json()
    assert runs[0]["state"] == "awaiting_colab" and runs[0]["execution"] == "colab"
    assert client.post(f"/api/recordings/demo/tracking/runs/{folder.name}/stop").status_code == 200


def test_generated_notebook_has_gpu_and_job_specific_checksum(pending):
    _, _, folder, _, _ = pending
    value = notebook(folder)
    assert value["metadata"]["accelerator"] == "GPU"
    for cell in value["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook-cell", "exec")
    code = json.dumps(value)
    assert SAM_REVISION in code and "b" * 64 in code and "files.download" in code


def test_bundle_preserves_sampled_rgb_and_original_source_time(tmp_path):
    folder = tmp_path / ("a" * 32)
    folder.mkdir()
    source = folder / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=30:duration=2",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            str(source),
        ],
        check=True,
    )
    job = dict(
        source_path=str(source),
        source_sha256=digest(source),
        source_start=0.4,
        duration=0.8,
        fps=5,
        media=dict(width=64, height=48),
        calibration=dict(source_seconds=0),
    )
    atomic(folder / "job.json", job)
    prepare(folder / "job.json")
    with zipfile.ZipFile(folder / "colab-input.zip") as package:
        assert all(not name.endswith(".env") for name in package.namelist())
        portable = json.loads(package.read("job.json"))
        assert portable["source_time_origin"] == job["source_start"]
        assert portable["source_sha256"] == job["source_sha256"]
        assert portable["expected_frames"] == 4
        assert portable["source_path"] == "media.mp4"
        video = folder / "unpacked.mp4"
        video.write_bytes(package.read("media.mp4"))
    common = ["-t", "0.8", "-vf", "fps=5", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    original = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", "0.4", "-i", str(source), *common])
    exported = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(video), *common])
    assert original == exported

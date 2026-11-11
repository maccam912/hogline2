import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hogline2.app import create_app
from hogline2.config import Config
from hogline2.store import Store


@pytest.mark.parametrize("same_job", [True, False])
def test_external_comparison_job_liveness_and_raw_artifacts(tmp_path, monkeypatch, same_job):
    Store(tmp_path).create("demo", source_sha256="abc", media=dict(width=500, height=500), ends=[])
    run_id = "a" * 32
    folder = tmp_path / "recordings/demo/tracking" / run_id
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(
        json.dumps(
            dict(
                backend="sam3.1-mlx-multiplex",
                raw_comparison=True,
                comparison_id="paired",
                end=dict(number=1),
                source_start=378,
                duration=120,
                boundary_revision=2,
                calibration=dict(revision=4),
            )
        )
    )
    (folder / "status.json").write_text(json.dumps(dict(state="running", pid=123, stage="Tracking")))
    (folder / "raw-velocity.json").write_text('{"samples": []}')
    other_id = "b" * 32
    command = (
        "python -m hogline2.tracking_comparison data/recordings/demo/tracking/"
        + (run_id if same_job else other_id)
        + "/job.json"
    )
    monkeypatch.setattr(
        "hogline2.tracking_api.subprocess.run", lambda *a, **k: SimpleNamespace(stdout=command)
    )
    client = TestClient(create_app(Config(tmp_path), start_worker=False))
    result = client.get("/api/recordings/demo/tracking/runs").json()[0]
    assert result["state"] == ("running" if same_job else "interrupted")
    assert result["backend"] == "sam3.1-mlx-multiplex"
    assert result["comparison_id"] == "paired" and result["raw_comparison"]
    prefix = f"/api/recordings/demo/tracking/runs/{run_id}/"
    assert client.get(prefix + "raw-velocity.json").json() == {"samples": []}
    assert client.get(prefix + "job.json").status_code == 404

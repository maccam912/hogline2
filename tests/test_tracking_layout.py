import copy
import json

import pytest
from fastapi.testclient import TestClient
from test_game_pipeline import setup_record
from test_tracking import view

from hogline2.app import create_app
from hogline2.config import Config
from hogline2.tracking_geometry import camera_reference, prepare_view, project
from hogline2.tracking_jobs import atomic
from hogline2.tracking_layout import read_calibration


def setup_layout(tmp_path, scale=1):
    store, source, calibration = setup_record(tmp_path)
    for end in ("A", "B"):
        overhead = view()
        overhead.update(id="overhead_" + end, end=end, rotation_degrees=180)
        calibration["views"].append(overhead)
    for v in calibration["views"]:
        v["camera_calibration"] = dict(
            image_to_ice=prepare_view(v)["image_to_ice"],
            reference_sha256=camera_reference(v),
            source_run="a" * 32,
            held_out_rms_m=0.05,
        )
    calibration["first_delivery_end"] = "A"
    atomic(tmp_path / "recordings/demo/tracking/calibration.json", calibration)
    store.create(
        "next",
        source_sha256="new-source",
        source_path=source["source_path"],
        state="ready",
        media=dict(width=int(500 * scale), height=int(500 * scale), fps="30/1", duration=100),
        ends=source["ends"],
    )
    return store, calibration


def test_legacy_layout_reused_without_copying_perspective_fits(tmp_path):
    store, source = setup_layout(tmp_path)
    source_path = tmp_path / "recordings/demo/tracking/calibration.json"
    original = source_path.read_bytes()
    client = TestClient(create_app(Config(tmp_path), start_worker=False))
    url = "/api/recordings/next/tracking/calibration"
    response = client.get(url)
    assert response.status_code == 200
    cal = response.json()
    assert cal["source_sha256"] == "new-source"
    assert cal["layout_source_recording"] == "demo" and cal["layout_source_revision"] == 5
    assert cal["revision"] == 1 and cal["first_delivery_end"] == "B"
    for before, after in zip(source["views"], cal["views"], strict=True):
        for key in ("crop", "landmarks", "end", "kind"):
            assert after[key] == before[key]
        if after["kind"] == "perspective":
            assert "camera_calibration" not in after
            assert after["mapping_status"] == "inherited_estimate"
        else:
            assert after["rotation_degrees"] == 180
            assert after["camera_calibration"] == before["camera_calibration"]
    assert client.get(url).json() == cal
    # The editor can save inherited metadata without falsely confirming alignment.
    saved = client.put(url, json=cal)
    assert saved.status_code == 200, saved.text
    assert saved.json()["views"][0]["mapping_status"] == "inherited_estimate"
    assert saved.json()["revision"] == 2
    assert client.put(url, json=cal).status_code == 409
    assert source_path.read_bytes() == original
    assert read_calibration(tmp_path, store, "demo") == source


def test_scaled_layout_keeps_overhead_world_coordinates(tmp_path):
    store, source = setup_layout(tmp_path, scale=0.5)
    cal = read_calibration(tmp_path, store, "next")
    for before, after in zip(source["views"], cal["views"], strict=True):
        assert after["crop"] == [0, 0, 250, 250]
        assert after["landmarks"]["button"] == [125, 125]
        if after["kind"] == "overhead":
            original = project(prepare_view(before)["image_to_ice"], [160, 320])
            scaled = project(prepare_view(after)["image_to_ice"], [80, 160])
            assert scaled == pytest.approx(original)


def test_incompatible_aspect_ratio_does_not_inherit(tmp_path):
    store, _ = setup_layout(tmp_path)
    store.update("next", media=dict(width=600, height=500))
    assert read_calibration(tmp_path, store, "next")["views"] == []
    assert not (tmp_path / "recordings/next/tracking/calibration.json").exists()


def test_whole_game_inherits_without_opening_editor(tmp_path, monkeypatch):
    setup_layout(tmp_path)
    launches = []
    monkeypatch.setattr("hogline2.game_api.subprocess.Popen", lambda *a, **kw: launches.append(a))
    client = TestClient(create_app(Config(tmp_path), start_worker=False))
    response = client.post("/api/recordings/next/game-jobs", json={})
    assert response.status_code == 202, response.text
    assert len(launches) == 1
    job_path = tmp_path / "recordings/next/game-jobs" / response.json()["job_id"] / "job.json"
    job = json.loads(job_path.read_text())
    assert job["calibration"]["layout_source_recording"] == "demo"
    assert job["calibration"]["views"][0]["mapping_status"] == "inherited_estimate"


def test_saving_first_complete_setup_establishes_stable_default(tmp_path):
    store, source = setup_layout(tmp_path)
    client = TestClient(create_app(Config(tmp_path), start_worker=False))
    response = client.put("/api/recordings/demo/tracking/calibration", json=source)
    assert response.status_code == 200, response.text
    path = tmp_path / "camera-layout.json"
    default = path.read_bytes()
    next_cal = read_calibration(tmp_path, store, "next")
    adjusted = copy.deepcopy(next_cal)
    adjusted["views"][0]["mapping_status"] = "marked"
    response = client.put("/api/recordings/next/tracking/calibration", json=adjusted)
    assert response.status_code == 200, response.text
    assert path.read_bytes() == default

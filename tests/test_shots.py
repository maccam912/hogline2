"""Broom marks, aim numbers and the report on a synthetic stonetrack result (no video, no tracker)."""

import json

from fastapi.testclient import TestClient

from hogline2.app import create_app
from hogline2.config import Config
from hogline2.shots.aim import HOG, TEE, aim
from hogline2.shots.api import compact_path


def delivery(x0=0.10, curl=0.5, start=100.0):
    """A stone thrown from end B: dead straight for 6.5 m past the hog line, then curling to +X,
    which is the thrower's right when delivering from B. It then sits still for five seconds."""
    path = []
    for i in range(400):
        along = 2.3 * i / 20
        late = max(0.0, along - (HOG + 6.5)) / (TEE - 2 * HOG - 6.5)
        path.append([start + i / 20, x0 + curl * late, TEE - along, 0.0, -2.3, "moving", "pb", 0.02])
    x, y = path[-1][1], path[-1][2]
    path += [[start + 20 + i / 20, x, y, 0.0, 0.0, "rest", "oa", 0.01] for i in range(100)]
    return dict(id="R1", colour="red", delivered_at=start, hog_to_hog_s=9.5, tee_to_hog_s=2.8, path=path)


def game(tmp_path):
    """A catalog with one recording, one tracked end and a camera whose pixels are centimetres."""
    app = create_app(Config(tmp_path), start_worker=False)
    app.state.store.create("game", ends=[dict(number=1, start_seconds=90, end_seconds=990)])
    run = tmp_path / "recordings" / "game" / "tracking" / "run1"
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps(dict(state="complete", started_at=1)))
    (run / "job.json").write_text(json.dumps(dict(duration=900, source_start=90, end=dict(number=1))))
    result = dict(delivery_end="B", start=90, end=990, stones=[delivery()])
    (run / "stonetrack-result.json").write_text(json.dumps(result))
    (tmp_path / "recordings" / "game" / "team-lineups.json").write_text(
        json.dumps(dict(red=["Matt", "", "", ""]))
    )
    calibration = tmp_path / "stonetrack" / "game"
    calibration.mkdir(parents=True)
    camera = dict(
        id="overhead_a",
        kind="overhead",
        end="A",
        crop=[0, 0, 1000, 1000],
        image_to_ice=[0.01, 0, -5.0, 0, 0.01, 0.0, 0, 0, 1.0],
    )
    (calibration / "calibration.json").write_text(json.dumps(dict(cameras=dict(overhead_a=camera))))
    return TestClient(app)


def test_aim_signs_and_sizes():
    result = aim(delivery(), "B", [0.0, TEE - 30.0])
    assert result["turn"] == "clockwise"
    assert abs(result["miss_right_cm"] - 10.0) < 0.2  # 10 cm to the thrower's right of the broom
    assert abs(result["miss_narrow_cm"] - 10.0) < 0.2  # and it curls right, so that is narrow
    assert aim(delivery(x0=-0.10), "B", [0.0, TEE - 30.0])["miss_narrow_cm"] < -9  # left of it: wide
    assert abs(result["heading"][0][1] - (TEE - HOG)) < 0.01  # the drawn line starts at the hog line


def test_resting_stretches_are_compacted():
    rows = compact_path(delivery()["path"])
    assert len(rows) == 400 + 2  # every moving sample, then the first and last of the rest
    assert rows[-1][0] == 124.95 and rows[-1][3] == 0


def test_listing_names_players_and_frames_each_shot(tmp_path):
    listing = game(tmp_path).get("/api/recordings/game/shots").json()
    shot = listing["shots"][0]
    assert (shot["player"], shot["stone"], shot["throw"]) == ("Matt", "red_1", 1)
    assert shot["aim"]["turn"] == "clockwise" and shot["mark"] is None
    assert shot["stopped_at"] == 120.0 and shot["next_at"] == 990
    assert listing["ends"] == [
        dict(number=1, start_seconds=90, end_seconds=990, tracked=True, delivery_end="B")
    ]
    assert listing["lineups"]["yellow"] == ["", "", "", ""]


def test_mark_round_trip_and_report(tmp_path):
    client = game(tmp_path)
    url = "/api/recordings/game/shots/1/red_1"
    # broom on the centre line at 30 m of travel: pixel (500, 100 * (TEE - 30))
    saved = client.put(url, json=dict(seconds=97.0, px=[500, 100 * (TEE - 30.0)], call="draw"))
    assert saved.status_code == 200
    assert abs(saved.json()["shot"]["aim"]["miss_right_cm"] - 10.0) < 0.3
    row = saved.json()["report"][0]
    assert row["player"] == "Matt" and row["marked"] == 1
    assert row["weight"]["draw"]["mean_hog_to_hog_s"] == 9.5
    # the call alone can change, and clearing everything removes the mark
    assert client.put(url, json=dict(call="guard")).json()["shot"]["mark"]["call"] == "guard"
    assert client.put(url, json={}).json()["shot"]["mark"] is None
    assert client.put(url, json=dict(no_broom=True)).json()["shot"]["mark"]["no_broom"] is True


def test_mark_must_be_a_tracked_shot_in_the_house_it_is_thrown_at(tmp_path):
    client = game(tmp_path)
    assert client.put("/api/recordings/game/shots/1/red_9", json=dict(no_broom=True)).status_code == 404
    outside = client.put("/api/recordings/game/shots/1/red_1", json=dict(seconds=97.0, px=[1500, 10]))
    assert outside.status_code == 400 and "house A" in outside.json()["detail"]
    assert client.put("/api/recordings/game/shots/1/red_1", json=dict(px=[500, 500])).status_code == 400


def test_paths_and_lineups(tmp_path):
    client = game(tmp_path)
    paths = client.get("/api/recordings/game/ends/1/paths").json()
    assert paths["stones"][0]["stone"] == "red_1" and len(paths["stones"][0]["path"]) == 402
    assert client.get("/api/recordings/game/ends/2/paths").status_code == 404
    names = dict(red=["Matt", "Lindsey", "Ethan", "Angela"], yellow=[" Cal ", "", "", ""])
    assert client.put("/api/recordings/game/lineups", json=names).json()["yellow"][0] == "Cal"
    assert client.get("/api/recordings/game/shots").json()["shots"][0]["player"] == "Matt"

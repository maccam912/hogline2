"""Broom marks and aim numbers on a synthetic stonetrack run (no video, no tracker)."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hogline2.broom_api import HOG, TEE, aim, mount_broom


def delivery(x0=0.10, curl=0.5, start=100.0):
    """A stone thrown from end B: dead straight for 6.5 m past the hog line, then curling to +X,
    which is the thrower's right when delivering from B."""
    path = []
    for i in range(400):
        along = 2.3 * i / 20
        late = max(0.0, along - (HOG + 6.5)) / (TEE - 2 * HOG - 6.5)
        path.append([start + i / 20, x0 + curl * late, TEE - along, 0.0, -2.3, "moving", "pb", 0.02])
    return dict(id="R1", colour="red", delivered_at=start, hog_to_hog_s=9.5, tee_to_hog_s=2.8, path=path)


def test_aim_signs_and_sizes():
    result = aim(delivery(), "B", [0.0, TEE - 30.0])
    assert result["turn"] == "clockwise"
    assert abs(result["miss_right_cm"] - 10.0) < 0.2  # 10 cm to the thrower's right of the broom
    assert abs(result["miss_narrow_cm"] - 10.0) < 0.2  # and it curls right, so that is narrow
    assert aim(delivery(x0=-0.10), "B", [0.0, TEE - 30.0])["miss_narrow_cm"] < -9  # left of it: wide


def test_mark_round_trip_and_report(tmp_path):
    run = tmp_path / "recordings" / "game" / "tracking" / "run1"
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps(dict(state="complete", started_at=1)))
    (run / "job.json").write_text(json.dumps(dict(duration=900, end=dict(number=1))))
    (run / "stonetrack-result.json").write_text(json.dumps(dict(delivery_end="B", stones=[delivery()])))
    (tmp_path / "recordings" / "game" / "team-lineups.json").write_text(
        json.dumps(dict(red=["Matt", "", "", ""]))
    )
    cal = tmp_path / "stonetrack" / "game"
    cal.mkdir(parents=True)
    # one camera on house A whose pixels are simply centimetres on the ice
    camera = dict(
        id="overhead_a",
        kind="overhead",
        end="A",
        crop=[0, 0, 1000, 1000],
        image_to_ice=[0.01, 0, -5.0, 0, 0.01, 0.0, 0, 0, 1.0],
    )
    (cal / "calibration.json").write_text(json.dumps(dict(cameras=dict(overhead_a=camera))))

    class Config:
        data = tmp_path

    class Store:
        def get(self, identifier):
            return {}

    app = FastAPI()
    mount_broom(app, Config, Store())
    client = TestClient(app)
    listing = client.get("/api/recordings/game/broom").json()
    assert [s["player"] for s in listing["shots"]] == ["Matt"] and listing["shots"][0]["aim"][
        "turn"
    ] == "clockwise"
    # broom on the centre line at 30 m of travel: pixel (500, 100 * (TEE - 30))
    saved = client.put(
        "/api/recordings/game/broom/1/red_1",
        json=dict(seconds=97.0, px=[500, 100 * (TEE - 30.0)], call="draw"),
    )
    assert saved.status_code == 200
    assert abs(saved.json()["shot"]["aim"]["miss_right_cm"] - 10.0) < 0.3
    row = saved.json()["report"][0]
    assert (
        row["player"] == "Matt" and row["marked"] == 1 and row["weight"]["draw"]["mean_hog_to_hog_s"] == 9.5
    )
    assert (
        client.put("/api/recordings/game/broom/1/red_9", json=dict(seconds=1.0, no_broom=True)).status_code
        == 404
    )

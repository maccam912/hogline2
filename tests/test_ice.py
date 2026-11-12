"""The ice model on simulated games where the ice is known, and the ice report endpoint."""

import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hogline2.app import create_app
from hogline2.config import Config
from hogline2.ice import model
from hogline2.ice.model import HOG, TEE


def simulated_game(fall=None, curl=None, drag=None, ends=6, seed=1):
    """stonetrack results for a game on ice with the given fields.

    Each field is a function of ice position (x, y) in m/s^2: `fall` pushes along +X whatever
    the stone is doing, `curl` pushes toward the side the stone turns to, `drag` slows it.
    Shots vary in weight, line, turn, how hard they curl and how much they were swept, and the
    tracked positions carry noise that wanders rather than flickers, as the real ones do.
    """
    rng = np.random.default_rng(seed)
    fall = fall or (lambda x, y: 0.0)
    curl = curl or (lambda x, y: 0.010)
    drag = drag or (lambda x, y: 0.080)
    results, clock = {}, 100.0
    for number in range(1, ends + 1):
        delivery_end = "B" if number % 2 else "A"
        d = 1 if delivery_end == "A" else -1
        origin = 0.0 if d == 1 else TEE
        stones = []
        for i in range(16):
            turn = rng.choice([1, -1])
            own_curl, own_drag = rng.normal(1, 0.15), rng.normal(1, 0.06)
            speed = rng.uniform(2.05, 2.3) if rng.random() < 0.7 else rng.uniform(2.6, 3.4)
            # start on the side the stone will curl away from, so that it stays on the sheet
            x = rng.uniform(-0.9, 0.9) + d * turn * rng.uniform(0.5, 1.0)
            along, vx, t, dt = HOG - 1.0, 0.0, 0.0, 0.05
            vy = d * speed
            wander = np.zeros(2)
            path = []
            while True:
                y = origin + d * along
                v = math.hypot(vx, vy)
                if v < 0.06 or along > TEE + 1.7 or abs(x) > 2.1:
                    break
                wander = 0.9 * wander + rng.normal(0, [0.002, 0.012])
                path.append([round(clock + t, 3), x + wander[0], y + wander[1], vx, vy, "moving", "pa", 0.02])
                ux, uy = vx / v, vy / v
                slowing = (
                    drag(x, y) * own_drag * (1 + 0.25 * (1 / math.sqrt(max(v, 0.3)) - 1 / math.sqrt(1.6)))
                )
                turning = turn * curl(x, y) * own_curl
                ax = -slowing * ux + turning * -uy + fall(x, y)
                ay = -slowing * uy + turning * ux
                x, along = x + vx * dt + ax * dt * dt / 2, along + d * (vy * dt + ay * dt * dt / 2)
                vx, vy, t = vx + ax * dt, vy + ay * dt, t + dt
            colour = "red" if i % 2 == 0 else "yellow"
            stones.append(
                dict(
                    id=f"{colour[0].upper()}{i // 2 + 1}",
                    colour=colour,
                    delivered_at=clock,
                    status="out",
                    path=path,
                )
            )
            clock += t + 20
        results[number] = dict(
            delivery_end=delivery_end,
            start=stones[0]["delivered_at"] - 5,
            end=clock,
            stones=stones,
            events=[],
        )
    return results


def cell(report, field, along, across):
    """The grid cell nearest an ice position."""
    i = int(np.argmin(np.abs(model.ALONG_KNOTS - along)))
    j = int(np.argmin(np.abs(model.ACROSS_KNOTS - across)))
    return report[field]["cells"][i * len(model.ACROSS_KNOTS) + j]


@pytest.fixture(scope="module")
def uniform():
    return model.build(simulated_game())


def test_uniform_ice_is_reported_as_uniform(uniform):
    assert uniform["ready"] and uniform["shots"] > 80
    assert not uniform["fall"]["evidence"]["real"]
    assert not uniform["curl"]["evidence"]["real"]
    assert not uniform["speed"]["evidence"]["real"]
    assert abs(uniform["curl"]["mean"] - 0.010) < 0.0015
    assert abs(uniform["speed"]["mean"] - 0.080) < 0.006


def test_draw_weight_matches_the_simulated_ice(uniform):
    # a stone that stops on the tee line at 0.08 m/s^2 crosses the hogs in about 13.5 s
    for row in uniform["draws"]:
        assert 12.5 < row["hog_to_hog_s"] < 14.5
        assert 1.3 < row["curl_m"] < 2.6
        assert abs(row["fall_x_m"]) < 0.08


def test_a_slope_is_found_where_it_is_and_points_the_right_way():
    # the +X side of the middle third of the sheet falls toward +X; the rest is level
    def slope(x, y):
        return 0.0025 if x > 0.3 and 12 < y < 24 else 0.0

    report = model.build(simulated_game(fall=slope))
    assert report["fall"]["evidence"]["real"]
    on = cell(report, "fall", 18, 1.2)
    assert on["value"] > 1.0 and on["z"] > 2.5  # mm/s^2
    assert abs(cell(report, "fall", 18, -1.2)["value"]) < 0.5 * on["value"]
    assert abs(cell(report, "fall", 3, 1.2)["value"]) < 0.5 * on["value"]
    # it carries a draw down that side toward +X, thrown either way
    for row in report["draws"]:
        if row["x"] == 1.2:
            assert row["fall_x_m"] > 0.08


def test_a_straight_patch_is_found():
    # the -X side of the half nearer house B curls only half as much
    def patchy(x, y):
        return 0.005 if x < -0.3 and y > 20 else 0.010

    report = model.build(simulated_game(curl=patchy))
    assert report["curl"]["evidence"]["real"]
    straight = cell(report, "curl", 26, -1.2)
    # percent of the sheet's average; the map is shrunk toward "no difference", so it understates
    assert straight["value"] < -10 and straight["z"] < -2.5
    assert cell(report, "curl", 26, 1.2)["value"] > straight["value"] + 10


def test_a_slow_patch_is_found():
    def sticky(x, y):
        return 0.095 if 8 < y < 16 else 0.080

    report = model.build(simulated_game(drag=sticky))
    assert report["speed"]["evidence"]["real"]
    assert cell(report, "speed", 12, 0)["value"] > 6  # percent more drag than the sheet's average
    assert cell(report, "speed", 26, 0)["value"] < 0


def test_too_few_shots_is_not_a_report():
    game = simulated_game(ends=1)
    assert model.build(game)["ready"] is False


def test_the_endpoint_serves_the_report_and_the_glides(tmp_path):
    app = create_app(Config(tmp_path), start_worker=False)
    game = simulated_game()
    app.state.store.create(
        "game", ends=[dict(number=n, start_seconds=r["start"], end_seconds=r["end"]) for n, r in game.items()]
    )
    for number, result in game.items():
        if number == 6:
            continue  # one end left untracked
        run = tmp_path / "recordings" / "game" / "tracking" / f"run{number}"
        run.mkdir(parents=True)
        (run / "status.json").write_text(json.dumps(dict(state="complete", started_at=1)))
        (run / "job.json").write_text(json.dumps(dict(duration=900, end=dict(number=number))))
        (run / "stonetrack-result.json").write_text(json.dumps(result))
    client = TestClient(app)
    report = client.get("/api/recordings/game/ice").json()
    assert report["ready"] and report["untracked"] == [6]
    assert len(report["glides"]) == report["shots"] == 80
    assert {g["toward"] for g in report["glides"]} == {"A", "B"}
    assert len(report["curl"]["cells"]) == len(report["grid"]["along"]) * len(report["grid"]["across"])
    assert client.get("/api/recordings/nope/ice").status_code == 404

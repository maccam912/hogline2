import math
import random

import pytest

from hogline2.tracking_motion import Axis
from hogline2.tracking_state import EndState


def obs(stone, xy, variance=0.12**2, view="v"):
    return dict(stone_id=stone.id, xy=xy, variance=[variance, variance], kind="perspective", view_id=view)


def test_kalman_covariance_and_measurement_correction():
    a = Axis(0, 1, 1, 0, 1).forecast(1, acceleration=0)
    assert [a.p, a.pp, a.pv, a.vv] == pytest.approx([1, 2, 1, 1])
    a.correct(2, 1)
    assert [a.p, a.v, a.pp, a.pv, a.vv] == pytest.approx([5 / 3, 4 / 3, 2 / 3, 1 / 3, 2 / 3])


@pytest.mark.parametrize("direction", [-1, 1])
def test_noisy_motion_is_smoothed_without_spontaneous_reverse(direction):
    rng = random.Random(84)
    state = EndState(1)
    stone = state.birth("red", 0, [0, 10], "v")
    raw_error, filtered_error, track = [], [], []
    for i in range(1, 101):
        t = i / 10
        truth = [0.015 * t * t, 10 + direction * t]
        point = [v + rng.gauss(0, 0.12) for v in truth]
        state.update(t, [obs(stone, point)])
        if i > 15:
            raw_error.append(math.dist(point, truth) ** 2)
            filtered_error.append(math.dist(stone.position, truth) ** 2)
            track.append(list(stone.position))
    assert sum(filtered_error) < sum(raw_error) * 0.65
    assert all(direction * (b[1] - a[1]) >= -0.01 for a, b in zip(track, track[1:], strict=False))
    assert stone.state == "moving"


def moving_track():
    state = EndState(1)
    stone = state.birth("yellow", 0, [0, 25], "a")
    for i in range(1, 31):
        state.update(i / 10, [obs(stone, [0, 25 - i / 5])])
    return state, stone


def test_gap_predicts_with_growing_uncertainty_then_recovers_and_eventually_stops_extrapolating():
    state, stone = moving_track()
    initial = stone.uncertainty95_m
    state.update(4, [])
    assert stone.estimate_kind == "predicted" and not stone.visible
    assert stone.position[1] == pytest.approx(17, abs=0.15)
    assert stone.uncertainty95_m > initial
    uncertain = stone.uncertainty95_m
    state.update(4.1, [obs(stone, [0, 16.8])])
    assert stone.visible and stone.uncertainty95_m < uncertain
    state.update(11, [])
    assert stone.estimate_kind == "lost"
    position = stone.position.copy()
    state.update(30, [])
    assert stone.position == position


def test_backwards_outlier_is_rejected_and_cannot_manufacture_contact():
    state, stone = moving_track()
    parked = state.birth("red", 3, [0, 30], "b")
    parked.state = "stationary"
    state.update(3.1, [obs(stone, [0, 30])])
    assert not stone.visible and stone.position[1] < 19
    assert not any(e["type"] == "possible_contact" for e in state.events)


def test_bad_view_mask_is_not_accepted_because_another_view_is_good():
    state, stone = moving_track()
    good, bad = obs(stone, [0, 18.8], view="good"), obs(stone, [1, 30], view="bad")
    state.update(3.1, [good, bad])
    assert good["accepted"] and not bad["accepted"]
    assert ("good", stone.id) in state.accepted and ("bad", stone.id) not in state.accepted


def test_imprecise_measurement_has_less_influence():
    a, sa = moving_track()
    b, sb = moving_track()
    a.update(3.1, [obs(sa, [0.5, 18.8], variance=4)])
    b.update(3.1, [obs(sb, [0.5, 18.8], variance=0.04)])
    assert abs(sa.position[0]) < abs(sb.position[0])


def test_low_confidence_mask_fragment_does_not_pull_track_like_a_clear_stone():
    a, sa = moving_track()
    b, sb = moving_track()
    weak = {**obs(sa, [0.35, 18.5]), "quality": -1.5}
    clear = {**obs(sb, [0.35, 18.5]), "quality": 2}
    a.update(3.1, [weak])
    b.update(3.1, [clear])
    assert abs(sa.position[0]) < abs(sb.position[0])
    assert sa.uncertainty95_m > sb.uncertainty95_m


@pytest.mark.parametrize("fps", [2, 7.5, 15])
def test_initial_delivery_acceleration_and_camera_noise_change(fps):
    state = EndState(2)
    stone = state.birth("red", 0, [0, 0], "delivery")
    for i in range(1, int(8 * fps) + 1):
        t = i / fps
        y = t * t if t < 1.5 else 2.25 + 3 * (t - 1.5)
        variance = 0.3**2 if t < 4 else 0.06**2
        state.update(t, [obs(stone, [0.01 * t * t, y], variance=variance)])
    assert stone.visible and stone.state == "moving"
    assert stone.position[1] == pytest.approx(21.75, abs=0.2)
    assert stone.velocity[1] == pytest.approx(3, abs=0.2)

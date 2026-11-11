"""Physical identities, uncertainty-aware motion and collision hypotheses."""

import math
from dataclasses import asdict, dataclass, field

from .tracking_geometry import segment_distance
from .tracking_motion import MotionFilter

MAX_SPEED = 6.0
PREDICTION_SECONDS = 6.0


@dataclass
class Stone:
    id: str
    color: str
    delivered_at: float
    position: list[float]
    last_seen: float
    state: str = "moving"
    visible: bool = True
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0])
    still_since: float | None = None
    contact_until: float = -1
    views: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    last_observed_position: list = field(default_factory=list)
    estimate_time: float = 0
    estimate_kind: str = "observed"
    position_covariance: list = field(default_factory=lambda: [[0.0625, 0], [0, 0.0625]])
    uncertainty95_m: float = 0.612
    accepted_observations: int = 0


def measurement_variance(obs):
    default = 0.06**2 if obs.get("kind") == "overhead" else 0.25**2
    values = obs.get("variance", [default, default])
    # The MLX port exposes an uncalibrated quality score; low values must not
    # give a fragmentary mask the precision of a full visible stone.
    factor = max(1.0, (1.0 - obs.get("quality", 1.0)) * 12)
    return [max(0.0004, min(25.0, float(v) * factor)) for v in values]


class EndState:
    def __init__(self, end_number):
        self.end_number = end_number
        self.stones = {}
        self.counts = {"red": 0, "yellow": 0}
        self.events = []
        self.filters = {}
        self.history = {}
        self.accepted = set()

    def birth(self, color, seconds, position, view_id):
        if color not in self.counts or self.counts[color] >= 8:
            raise ValueError("An end supports at most eight delivered stones of each color")
        self.counts[color] += 1
        name = f"{color}_{self.counts[color]}"
        stone = Stone(
            name,
            color,
            seconds,
            list(position),
            seconds,
            last_observed_position=list(position),
            estimate_time=seconds,
        )
        stone.views[view_id] = {"joined_at": seconds, "method": "hack_delivery"}
        self.stones[name] = stone
        self.filters[name] = MotionFilter(position, seconds)
        self.history[name] = []
        self.events.append(dict(type="delivery", stone=name, seconds=seconds, view=view_id))
        return stone

    def compatibility(self, stone, xy, seconds, variance, contact=False):
        """Innovation gate plus speed/direction bounds. Lower is a better match."""
        if not all(math.isfinite(v) for v in xy + variance):
            return None
        gap = max(0, seconds - stone.last_seen)
        resting = stone.state == "stationary" and not contact
        if gap > 12 and not resting:
            return None
        axes = self.filters[stone.id].forecast(seconds, resting)
        delta = [p - a.p for p, a in zip(xy, axes, strict=True)]
        total = [a.pp + r for a, r in zip(axes, variance, strict=True)]
        nis = sum(d * d / max(s, 1e-6) for d, s in zip(delta, total, strict=True))
        if nis > 16 and not contact:
            return None
        distance = math.dist(xy, stone.last_observed_position)
        if distance > MAX_SPEED * max(gap, 0.1) + 3 * math.sqrt(max(variance)):
            return None
        speed = math.hypot(*stone.velocity)
        if speed > 0.4 and stone.accepted_observations >= 3 and not contact:
            forward = sum(
                (p - old) * v / speed
                for p, old, v in zip(xy, stone.last_observed_position, stone.velocity, strict=True)
            )
            # Measurement uncertainty permits noisy pixels; the output constraint below
            # prevents the posterior itself from jumping backward without contact evidence.
            if forward < -max(0.20, 2.5 * math.sqrt(max(variance))):
                return None
        if resting and math.dist(xy, stone.position) > 0.25:
            return None
        return nis

    def match(self, color, xy, seconds, view_id, radius=0.6, variance=None):
        variance = variance or [max(0.06, radius / 3) ** 2] * 2
        candidates = []
        for stone in self.stones.values():
            if stone.color != color:
                continue
            score = self.compatibility(stone, xy, seconds, variance, seconds <= stone.contact_until)
            if score is not None:
                estimate = self.filters[stone.id].forecast(seconds, stone.state == "stationary")
                distance = math.dist(xy, [a.p for a in estimate])
                # An uncertain prediction is not permission to associate anywhere on the sheet.
                if distance <= min(4.0, radius + 3 * math.sqrt(max(a.pp for a in estimate))):
                    candidates.append((distance, stone))
        candidates.sort(key=lambda row: row[0])
        if not candidates or (len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.20):
            return None
        return candidates[0][1]

    def update(self, seconds, observations):
        self.accepted = set()
        before = {name: list(s.position) for name, s in self.stones.items()}
        chosen, by_id = {}, {}
        for obs in observations:
            obs["accepted"] = False
            if obs["stone_id"] in self.stones:
                by_id.setdefault(obs["stone_id"], []).append(obs)
        for name, rows in by_id.items():
            stone = self.stones[name]
            valid = []
            for obs in rows:
                variance = measurement_variance(obs)
                score = self.compatibility(
                    stone, obs["xy"], seconds, variance, seconds <= stone.contact_until
                )
                # A resting stone's displacement may be justified by another stone's contact below.
                if score is not None or stone.state == "stationary":
                    valid.append(obs)
                else:
                    obs["rejection"] = "motion_gate"
            valid.sort(key=lambda o: (o["kind"] != "overhead", sum(measurement_variance(o))))
            if valid:
                chosen[name] = valid[0]
                if any(math.dist(valid[0]["xy"], o["xy"]) > 0.75 for o in valid[1:]):
                    self.events.append(dict(type="view_disagreement", stone=name, seconds=seconds))
        # Only motion-gated observations may suggest a collision. A bogus teleport cannot
        # create contact evidence that then excuses its own impossible motion.
        contacts = set()
        for name, obs in chosen.items():
            moving = self.stones[name]
            if moving.state == "stationary" or math.dist(moving.position, obs["xy"]) < 0.035:
                continue
            for other in self.stones.values():
                if other.id != name and segment_distance(other.position, moving.position, obs["xy"]) < 0.40:
                    if seconds > other.contact_until:
                        self.events.append(
                            dict(
                                type="possible_contact",
                                stones=[name, other.id],
                                seconds=seconds,
                            )
                        )
                        contacts.update((name, other.id))
                    moving.contact_until = other.contact_until = seconds + 2
                    if other.id not in chosen:
                        other.state = "contact_occluded"
        for name in contacts:
            self.filters[name].contact()
        for name, stone in self.stones.items():
            motion = self.filters[name]
            obs = chosen.get(name)
            contact = seconds <= stone.contact_until
            old_velocity = list(stone.velocity)
            dt = max(0, seconds - stone.estimate_time)
            stationary = stone.state == "stationary" and not contact
            predict_until = seconds if stationary else min(seconds, stone.last_seen + PREDICTION_SECONDS)
            motion.predict(predict_until, stationary)
            if obs is not None:
                variance = measurement_variance(obs)
                if self.compatibility(stone, obs["xy"], seconds, variance, contact) is None:
                    obs["rejection"] = "unexplained_displacement" if stationary else "motion_gate"
                    obs = None
                elif contact and any(
                    other.id != name
                    and other.color == stone.color
                    and math.dist(obs["xy"], other.position) < 0.20
                    for other in self.stones.values()
                ):
                    self.events.append(
                        dict(
                            type="collision_identity_ambiguous",
                            stone=name,
                            seconds=seconds,
                        )
                    )
                    obs["rejection"] = "ambiguous_contact"
                    obs = None
            stone.visible = obs is not None
            if obs is None:
                if by_id.get(name):
                    self.events.append(dict(type="measurement_rejected", stone=name, seconds=seconds))
                if stationary:
                    stone.estimate_kind = "held"
                elif seconds - stone.last_seen > PREDICTION_SECONDS:
                    stone.state, stone.estimate_kind = "lost", "lost"
                    for a in motion.axes:
                        a.pp = min(25, a.pp + dt * 0.25)
                else:
                    if stone.state != "contact_occluded":
                        stone.state = "occluded"
                    stone.estimate_kind = "predicted"
            else:
                if stationary:
                    # A small residual on a resting stone is mask jitter, not renewed motion.
                    for a, p in zip(motion.axes, before[name], strict=True):
                        a.p, a.v, a.pv, a.vv = p, 0, 0, 0.01
                else:
                    motion.correct(obs["xy"], variance)
                    old_speed = math.hypot(*old_velocity)
                    if not contact and stone.accepted_observations >= 3 and old_speed > 0.35:
                        heading = [v / old_speed for v in old_velocity]
                        displacement = [a.p - p for a, p in zip(motion.axes, before[name], strict=True)]
                        backward = sum(d * h for d, h in zip(displacement, heading, strict=True))
                        if backward < 0:
                            for a, h in zip(motion.axes, heading, strict=True):
                                a.p -= backward * h
                                a.pp += (backward * h) ** 2
                        backwards_velocity = sum(a.v * h for a, h in zip(motion.axes, heading, strict=True))
                        if backwards_velocity < 0:
                            for a, h in zip(motion.axes, heading, strict=True):
                                a.v -= backwards_velocity * h
                    speed = math.hypot(*motion.velocity)
                    if speed > MAX_SPEED:
                        for a in motion.axes:
                            a.v *= MAX_SPEED / speed
                    # One camera correction cannot imply arbitrarily large frame-to-frame speed.
                    distance = math.dist(motion.position, before[name])
                    limit = MAX_SPEED * dt + 0.02
                    if dt and distance > limit:
                        for a, p in zip(motion.axes, before[name], strict=True):
                            a.p = p + (a.p - p) * limit / distance
                            a.pp += (distance - limit) ** 2
                    history = self.history[name]
                    history.append((seconds, list(obs["xy"])))
                    history[:] = [(t, p) for t, p in history if seconds - t <= 2.5]
                    still = (
                        len(history) >= 3
                        and seconds - history[0][0] >= 1.8
                        and max(math.dist(p, history[0][1]) for _, p in history) < 0.16
                        and math.hypot(*motion.velocity) < 0.25
                    )
                    stone.state = "stationary" if still else "moving"
                    stone.still_since = history[0][0] if still else None
                    if still:
                        for a in motion.axes:
                            a.v, a.pv, a.vv = 0, 0, 0.01
                stone.last_seen = seconds
                stone.last_observed_position = list(obs["xy"])
                stone.accepted_observations += 1
                stone.estimate_kind = "observed"
                for row in by_id.get(name, []):
                    tolerance = max(
                        0.35,
                        3 * math.sqrt(max(measurement_variance(row)) + max(variance)),
                    )
                    if not row.get("rejection") and math.dist(row["xy"], obs["xy"]) <= tolerance:
                        row["accepted"] = True
                        self.accepted.add((row.get("view_id"), name))
            stone.position, stone.velocity = motion.position, motion.velocity
            stone.position_covariance = [
                [motion.variance[0], 0],
                [0, motion.variance[1]],
            ]
            stone.uncertainty95_m = motion.radius95
            stone.estimate_time = seconds
        return [asdict(stone) for stone in self.stones.values()]

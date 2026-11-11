"""Small constant-velocity Kalman filter; no GPU/numpy dependency in the web app.

Each axis has [position, velocity] and a symmetric 2x2 covariance. Measurement
variances come from the camera projection. Process noise permits acceleration;
contact hypotheses explicitly increase velocity uncertainty.
"""

import math
from dataclasses import dataclass


@dataclass
class Axis:
    p: float
    v: float = 0.0
    pp: float = 0.0625
    pv: float = 0.0
    vv: float = 9.0

    def forecast(self, dt, acceleration=0.4):
        q = acceleration**2
        return Axis(
            self.p + self.v * dt,
            self.v,
            self.pp + 2 * dt * self.pv + dt * dt * self.vv + q * dt**4 / 4,
            self.pv + dt * self.vv + q * dt**3 / 2,
            self.vv + q * dt * dt,
        )

    def correct(self, measurement, variance):
        variance = max(variance, 1e-6)
        innovation = measurement - self.p
        total = self.pp + variance
        kp, kv = self.pp / total, self.pv / total
        self.p += kp * innovation
        self.v += kv * innovation
        # Joseph covariance update, stable even for very precise measurements.
        pp, pv, vv = self.pp, self.pv, self.vv
        self.pp = (1 - kp) ** 2 * pp + kp**2 * variance
        self.pv = (1 - kp) * (pv - kv * pp) + kp * kv * variance
        self.vv = vv - 2 * kv * pv + kv * kv * (pp + variance)


class MotionFilter:
    def __init__(self, xy, seconds):
        self.axes = [Axis(float(p)) for p in xy]
        self.seconds = seconds
        self.started_at = seconds

    def forecast(self, seconds, stationary=False):
        dt = max(0.0, seconds - self.seconds)
        if stationary:
            return [Axis(a.p, 0, min(0.25, a.pp + dt * 0.0004), 0, 0.01) for a in self.axes]
        return [a.forecast(dt, acceleration=2.0 if seconds - self.started_at < 3 else 0.4) for a in self.axes]

    def predict(self, seconds, stationary=False):
        self.axes = self.forecast(seconds, stationary)
        self.seconds = max(self.seconds, seconds)

    def contact(self):
        for a in self.axes:
            a.vv = max(a.vv, 9.0)
            a.pp = max(a.pp, 0.04)
            a.pv = 0

    def correct(self, xy, variance):
        for a, p, r in zip(self.axes, xy, variance, strict=True):
            a.correct(p, r)

    @property
    def position(self):
        return [a.p for a in self.axes]

    @property
    def velocity(self):
        return [a.v for a in self.axes]

    @property
    def variance(self):
        return [max(1e-8, a.pp) for a in self.axes]

    @property
    def radius95(self):
        return math.sqrt(5.991 * max(self.variance))

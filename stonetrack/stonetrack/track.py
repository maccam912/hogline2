"""Multi-camera stone tracker for one end: 16 stable identities on the ice plane.

Every camera's detections arrive as ice positions with covariances, so the tracker never thinks
in pixels: an overhead camera simply counts for more (about 1 cm) than a far perspective camera
(about 10 cm along the sheet), and where two cameras overlap they fuse.

A stone is a small state machine
  moving   constant velocity with sliding friction; coasts on physics when nothing sees it
  rest     pinned; stays where it is while hidden, for as long as it is hidden
  out      left the playing area, or its spot was seen to be bare ice
Identity is the delivery order per colour (R1..R8, Y1..Y8). A stone gets its number when it is
first seen travelling from the delivering end toward the playing house, and keeps it for the end.
Hidden stones are never dropped for being hidden: a stone is removed only on positive evidence
(it left the sheet, or a camera can see that its spot is empty).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import BACK, HACK, HOG, SHEET_HALF_WIDTH, STONE_RADIUS, TEE_DISTANCE, load_calibration, tee, toward_centre

FRICTION = 0.10  # m/s^2 sliding deceleration used when coasting
ACCEL_SIGMA = 0.7  # m/s^2 process noise while moving (sweeping, curl, light contact)
PANEL_LAG = 0.08  # s: unknown timing offset between mosaic panels, as extra along-track noise
GATE = 16.0  # squared Mahalanobis distance
CONFIRM_HITS = 10
MIN_DELIVERY_GAP = 20.0  # s between two deliveries
SAME_COLOUR_GAP = 45.0  # teams alternate; the same colour twice running means one was missed
REST_SPEED = 0.05


class Stone:
    _next = 0

    def __init__(self, colour, z, R, t):
        Stone._next += 1
        self.uid = Stone._next
        self.colour = colour
        self.label = None
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([R[0, 0] + 0.01, R[1, 1] + 0.01, 9.0, 9.0])
        self.mode = "moving"
        self.status = "tentative"  # tentative | active | out
        self.hits, self.over_hits, self.side_hits, self.born, self.last_seen = 1, 0, 0, t, t
        self.born_along = None
        self.forward_since = None
        self.window = []  # (t, seen) for the last few seconds: how steadily it is being detected
        self.slow_since = None
        self.empty_since = None
        self.delivered_at = None
        self.path = []  # (t, x, y, vx, vy, mode, views)
        self.seen_this_frame = set()
        self.reason = None

    # -- motion -------------------------------------------------------------------------
    def predict(self, dt):
        if self.mode == "rest":
            self.P[:2, :2] += np.eye(2) * (0.002**2) * dt
            return
        v = self.x[2:]
        speed = np.linalg.norm(v)
        if speed > 1e-3:  # friction along the direction of travel, never reversing it
            dv = min(FRICTION * dt, speed)
            self.x[:2] += v * dt - 0.5 * (v / speed) * dv * dt
            self.x[2:] = v * (1 - dv / speed)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.P = F @ self.P @ F.T + G @ G.T * ACCEL_SIGMA**2

    def innovation(self, z, R):
        v = self.x[2:]
        speed = np.linalg.norm(v)
        if self.mode == "moving" and speed > 0.2:  # panels are not frame-synchronous
            u = v / speed
            R = R + np.outer(u, u) * (speed * PANEL_LAG) ** 2
        nu = z - self.x[:2]
        S = self.P[:2, :2] + R
        return nu, S, R

    def update(self, z, R, t, view):
        nu, S, _ = self.innovation(z, R)
        K = self.P[:, :2] @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.P = (np.eye(4) - K @ np.eye(2, 4)) @ self.P
        if self.mode == "rest":
            self.x[2:] = 0
        self.hits += 1
        self.over_hits += view.startswith("overhead")
        self.side_hits += view.startswith("perspective")
        self.last_seen = t
        self.seen_this_frame.add(view)

    def wake(self, z, R, t, view):
        """A pinned stone has visibly moved (it was hit): restart it as a moving stone."""
        self.mode = "moving"
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([R[0, 0] + 0.02, R[1, 1] + 0.02, 4.0, 4.0])
        self.slow_since = None
        self.last_seen = t
        self.seen_this_frame.add(view)


class Tracker:
    def __init__(self, cams, delivery_end):
        self.cams = cams
        self.delivery_end = delivery_end
        self.play_end = "A" if delivery_end == "B" else "B"
        self.dir = toward_centre(delivery_end)  # +Y or -Y: the way stones travel this end
        self.stones = []
        self.count = {"red": 0, "yellow": 0}
        self.last_delivery, self.last_colour = -1e9, None
        self.t = None
        self.events = []
        self.missing = []  # named stones whose spot went empty: (stone, time) awaiting a re-sighting

    # -- geometry helpers ---------------------------------------------------------------
    def in_play(self, p):
        """Inside the side boards, from the delivering hack to fully past the far back line."""
        where = self.along(p[1])
        return abs(p[0]) <= SHEET_HALF_WIDTH and -(HACK + 0.3) <= where <= TEE_DISTANCE + BACK + STONE_RADIUS

    def _overhead_sees(self, p):
        for cam in self.cams.values():
            if cam.kind == "overhead" and cam.contains(cam.to_image([p], 0.1), margin=24)[0]:
                return True
        return False

    def along(self, y):
        """Distance travelled from the delivering tee line, metres."""
        return (y - tee(self.delivery_end)[1]) * self.dir

    def _visible_empty(self, stone, fg):
        """Which kind of camera, if any, is looking at this stone's spot and seeing bare ice.

        A stone that is there changes the picture whether or not the detector recognises it, so
        "nothing differs from the empty sheet at that spot" is positive evidence that it is gone.
        A person standing in the way also changes the picture, and then this says nothing."""
        seen_by = None
        for cam in self.cams.values():
            if cam.kind == "perspective":
                rel = (stone.x[1] - tee(cam.end)[1]) * toward_centre(cam.end)
                if not -2.2 <= rel <= 19.5:
                    continue
            px = cam.to_image([stone.x[:2]], 0.06)[0]
            if not cam.contains([px], margin=22)[0]:
                continue
            r = STONE_RADIUS * cam.pixels_per_metre(stone.x[:2]) + 4
            if any(b[0] - r <= px[0] <= b[2] + r and b[1] - r <= px[1] <= b[3] + r for b in fg.get(cam.id, ())):
                return None  # something is there in this view: no verdict
            seen_by = "overhead" if cam.kind == "overhead" or seen_by == "overhead" else "perspective"
        return seen_by

    # -- one frame ----------------------------------------------------------------------
    def step(self, t, detections, fg):
        dt = 0.0 if self.t is None else t - self.t
        self.t = t
        live = [s for s in self.stones if s.status != "out"]
        for s in live:
            s.predict(dt)
            s.seen_this_frame = set()
        dets = [d for d in detections if self.in_play(d["ice"])]
        used = set()
        by_view = {}
        for i, d in enumerate(dets):
            by_view.setdefault(d["view"], []).append(i)
        # overhead first: the most precise measurement shapes the gate for the rest
        for view in sorted(by_view, key=lambda v: self.cams[v].kind != "overhead"):
            idx = by_view[view]
            cost = np.full((len(idx), len(live)), 1e6)
            for a, i in enumerate(idx):
                d = dets[i]
                z = np.array(d["ice"])
                R = np.array([[d["cov"][0], d["cov"][1]], [d["cov"][1], d["cov"][2]]])
                for b, s in enumerate(live):
                    if s.colour != d["colour"]:
                        continue
                    nu, S, _ = s.innovation(z, R)
                    m2 = float(nu @ np.linalg.solve(S, nu))
                    if m2 < GATE and np.linalg.norm(nu) < 1.2:
                        cost[a, b] = m2 + (0 if s.status == "active" else 2.0)  # prefer established stones
            if cost.size:
                for a, b in zip(*linear_sum_assignment(cost)):
                    if cost[a, b] < 1e5:
                        d = dets[idx[a]]
                        R = np.array([[d["cov"][0], d["cov"][1]], [d["cov"][1], d["cov"][2]]])
                        live[b].update(np.array(d["ice"]), R, t, view)
                        used.add(idx[a])
        # leftovers: a pinned stone that was knocked, or something new
        for i, d in enumerate(dets):
            if i in used:
                continue
            z = np.array(d["ice"])
            R = np.array([[d["cov"][0], d["cov"][1]], [d["cov"][1], d["cov"][2]]])
            if self.cams[d["view"]].kind == "overhead":
                cands = [s for s in live if s.colour == d["colour"] and s.mode == "rest" and s.status == "active"
                         and not s.seen_this_frame and np.linalg.norm(s.x[:2] - z) < 0.7]
                if cands:
                    s = min(cands, key=lambda s: np.linalg.norm(s.x[:2] - z))
                    s.wake(z, R, t, d["view"])
                    self.events.append(dict(t=t, type="knocked", stone=s.label or f"#{s.uid}"))
                    continue
            if any(s.colour == d["colour"] and np.linalg.norm(s.x[:2] - z) < 0.45 for s in live):
                continue  # too close to a stone we already have: not a new stone
            self.stones.append(Stone(d["colour"], z, R, t))
            self.stones[-1].seen_this_frame.add(d["view"])
            self.stones[-1].born_along = self.along(z[1])
            self.stones[-1].over_hits = int(d["view"].startswith("overhead"))
        self._manage(t, fg)

    def _manage(self, t, fg):
        for s in self.stones:
            if s.status == "out":
                continue
            speed = float(np.linalg.norm(s.x[2:]))
            if s.status == "tentative":
                # where an overhead camera can see, only it may vouch for a new stone: from the side a
                # broom head held on the ice looks enough like a handle
                vouched = s.over_hits >= 3 or not self._overhead_sees(s.x[:2])
                if s.hits >= CONFIRM_HITS and t - s.born >= 0.4 and vouched:
                    s.status = "active"
                elif t - s.last_seen > 0.5 or t - s.born > 3.0:
                    s.status, s.reason = "out", "never confirmed"
                    continue
            # settle
            if s.mode == "moving":
                # a side camera alone is noisy enough that the velocity estimate never quite reads zero
                limit = REST_SPEED if any(v.startswith("overhead") for v in s.seen_this_frame) or not s.seen_this_frame else 0.15
                if speed < limit:
                    s.slow_since = s.slow_since or t
                    if t - s.slow_since > 0.8:
                        s.mode = "rest"
                        s.x[2:] = 0
                        s.P[2:, 2:] = np.eye(2) * 1e-4
                        s.P[:2, 2:] = s.P[2:, :2] = 0
                        # still let later looks refine where exactly it stopped
                        s.P[:2, :2] = np.maximum(s.P[:2, :2], np.eye(2) * 0.05**2)
                else:
                    s.slow_since = None
            # a delivery: watched travelling down the sheet out of the delivering end
            forward = s.x[3] * self.dir
            if s.mode == "moving" and s.seen_this_frame and forward > 0.8:
                s.forward_since = s.forward_since or t
            elif s.seen_this_frame or s.mode == "rest":
                s.forward_since = None
            # A delivery is recognised a couple of metres past the near hog line, and only if both kinds
            # of camera agree it is a stone: an overhead camera saw a cap on granite as it left the
            # house, and a side camera has since followed a handle on granite down the ice. A broom
            # head can fool the first, a jacket the second; neither fools both.
            # Failing that (a hand over the cap hides it from above), a side camera alone is believed
            # only if it has held the stone almost every frame for three seconds: people who pass for
            # a handle do so in flickers.
            s.window = [w for w in s.window if t - w[0] <= 3.0] + [(t, bool(s.seen_this_frame))]
            steady = s.steady = t - s.window[0][0] >= 2.9 and np.mean([w[1] for w in s.window]) >= 0.8
            where = self.along(s.x[1])
            both = s.over_hits >= 3 and s.side_hits >= 15
            if s.status == "active" and s.label is None and s.forward_since and t - s.forward_since >= 0.5 \
                    and (both or (steady and where >= HOG + 4.0)):
                turn = s.colour != self.last_colour or t - self.last_delivery > SAME_COLOUR_GAP
                if HOG + 2.0 <= where <= TEE_DISTANCE - HOG and turn and t - self.last_delivery > MIN_DELIVERY_GAP \
                        and self.count[s.colour] < 8:
                    self.count[s.colour] += 1
                    s.label = f"{s.colour[0].upper()}{self.count[s.colour]}"
                    crossed = [p[0] for p in s.path if self.along(p[2]) >= 0]  # back-date to the tee line
                    s.delivered_at = crossed[0] if crossed else t
                    self.last_delivery, self.last_colour = t, s.colour
                    self.events.append(dict(t=s.delivered_at, type="delivered", stone=s.label, recognised_at=round(t, 2)))
            # leaving play
            if not self.in_play(s.x[:2]):
                s.status, s.reason = "out", "left the sheet"
                self.events.append(dict(t=t, type="out", stone=s.label or f"#{s.uid}", why=s.reason))
                continue
            # positive evidence of absence
            bare = None if s.seen_this_frame or s.status != "active" else self._visible_empty(s, fg)
            if bare:
                s.empty_since = s.empty_since or t
                if t - s.empty_since > (1.5 if bare == "overhead" else 2.5):
                    s.status, s.reason = "out", "spot seen empty"
                    self.events.append(dict(t=t, type="out", stone=s.label or f"#{s.uid}", why=s.reason))
                    continue
            elif s.seen_this_frame:
                s.empty_since = None
            # unlabelled movers that never became deliveries (a broom head, a stone being carried)
            if s.status == "active" and s.label is None and t - s.last_seen > 1.5:
                s.status, s.reason = "out", "unlabelled and lost"
                continue
            # the cameras cover the whole sheet: a stone cannot slide unseen for this long
            if s.status == "active" and s.mode == "moving" and t - s.last_seen > 8.0:
                s.status, s.reason = "out", "lost while moving"
                self.events.append(dict(t=t, type="out", stone=s.label, why=s.reason))
                continue
            if s.status == "active":
                s.path.append((round(t, 3), round(float(s.x[0]), 4), round(float(s.x[1]), 4), round(float(s.x[2]), 3),
                               round(float(s.x[3]), 3), s.mode, "".join(sorted(v[0] + v[-1] for v in s.seen_this_frame)),
                               round(float(np.sqrt(max(s.P[0, 0], s.P[1, 1]))), 3)))  # last: 1-sigma position, m
        # Reconcile with what the overhead camera plainly shows. Every stone in play was delivered and
        # so has a name; a stone lying at rest under an overhead camera with no name is therefore one
        # whose track went astray (hit while hidden, or dragged off by a neighbour). Give it the name
        # of the stone of its colour that most recently dropped out.
        for s in self.stones:
            if s.status == "out" and s.label and not getattr(s, "queued", False):
                s.queued = True
                self.missing.append((s, t))
        self.missing = [(m, when) for m, when in self.missing if t - when <= 60 and m.label]
        for c in self.stones:
            if c.status != "active" or c.label or c.mode != "rest":
                continue
            # believable as a stone: long under an overhead camera, or held steadily by a side camera
            if not (c.over_hits >= 30 or (getattr(c, "steady", False) and c.side_hits >= 60)):
                continue
            # ...and it turned up about when the named stone went astray (a skip's broom held still
            # in the guard zone turns up later, with the next shot)
            same = [(m, when) for m, when in self.missing if m.colour == c.colour and abs(c.born - when) <= 12]
            # a stone that slid out of play is set aside near the boards or behind the back line, where
            # it looks just like this; only a stone lying well inside the sheet can be a mistaken exit
            rel = self.along(c.x[1]) - TEE_DISTANCE
            inside = abs(c.x[0]) <= 1.65 and -HOG <= rel <= BACK - 0.3
            same = [(m, when) for m, when in same if m.reason != "left the sheet" or inside]
            if not same:
                continue
            lost, when = max(same, key=lambda mw: mw[1])
            c.label, c.delivered_at = lost.label, lost.delivered_at
            c.path = lost.path + [p for p in c.path if not lost.path or p[0] > lost.path[-1][0]]
            lost.label, lost.reason = None, "track continued elsewhere"
            self.missing.remove((lost, when))
            self.events = [e for e in self.events if not (e["type"] == "out" and e.get("stone") == c.label)]
            self.events.append(dict(t=t, type="found again", stone=c.label))
        # two tracks on one stone. Cameras can disagree by a few decimetres on a fast stone at long
        # range, enough to start a second track; the named track then coasts while the newcomer is
        # the one being seen. Hand the newcomer's fresher state to the named track.
        act = [s for s in self.stones if s.status in ("active", "tentative")]

        def rank(s):  # which of two tracks on one stone is the stone's real history
            return (bool(s.label), s.over_hits >= 3, -s.born)

        for a in act:
            for b in act:
                if a is b or a.colour != b.colour or b.label or "out" in (a.status, b.status) or rank(a) <= rank(b):
                    continue
                gap = np.linalg.norm(a.x[:2] - b.x[:2])
                moving_pair = a.mode == "moving" and b.mode == "moving" and gap < 1.5 and np.linalg.norm(a.x[2:] - b.x[2:]) < 1.2
                if gap < 0.2 or moving_pair:
                    if b.last_seen > a.last_seen and b.hits >= CONFIRM_HITS:
                        a.x, a.P, a.last_seen = b.x.copy(), b.P.copy(), b.last_seen
                        a.seen_this_frame |= b.seen_this_frame
                    if gap < 0.2 or b.hits >= CONFIRM_HITS:
                        b.status, b.reason = "out", "duplicate"

    def result(self):
        named = sorted((s for s in self.stones if s.label), key=lambda s: s.delivered_at)
        # The end as played: the first moment after the last delivery when nothing has moved for two
        # seconds (before anyone walks in to clear the house).
        settled, final = None, {}
        if named:
            times = sorted({p[0] for s in named for p in s.path})
            busy = {p[0] for s in named for p in s.path if p[5] == "moving"}
            alive = {}
            for s in named:
                for p in s.path:
                    alive[p[0]] = alive.get(p[0], 0) + 1
            quiet_moments, run_start = [], None
            for a in times:
                if a in busy:
                    run_start = None
                else:
                    run_start = a if run_start is None else run_start
                    if a - run_start >= 2.0:
                        quiet_moments.append(a)
            after_last = [a for a in quiet_moments if a > named[-1].delivered_at + 3]
            prev = named[-2].delivered_at + 3 if len(named) > 1 else -1e9
            before_last = [a for a in quiet_moments if prev < a < named[-1].delivered_at]
            # first calm after the last stone; or, if that stone was thrown through a decided end
            # while the house was already being cleared, the first calm after the stone before it
            settled = after_last[0] if after_last else (before_last[0] if before_last else None)
            if settled:
                for s in named:
                    at = [p for p in s.path if p[0] <= settled]
                    if at and settled - at[-1][0] < 0.5:
                        final[s.label] = [at[-1][1], at[-1][2]]
        return dict(delivery_end=self.delivery_end, settled_at=settled, final_positions=final, events=self.events,
                    stones=[dict(id=s.label, colour=s.colour, delivered_at=s.delivered_at, status=s.status, mode=s.mode,
                                 reason=s.reason, final=final.get(s.label), **self._shot(s), path=s.path) for s in named])

    def _shot(self, s):
        """Split times straight from the fused path: tee-to-hog, hog-to-hog, speed at the near hog."""
        def crossing(where):
            for (t0, _, y0, *_), (t1, _, y1, *_) in zip(s.path, s.path[1:]):
                a, b = self.along(y0), self.along(y1)
                if a < where <= b and t1 - t0 < 0.5 and t0 >= s.delivered_at - 0.2:
                    return t0 + (t1 - t0) * (where - a) / (b - a)
            return None

        tee_t, hog1, hog2 = crossing(0.0), crossing(HOG), crossing(TEE_DISTANCE - HOG)
        near = [abs(p[4]) for p in s.path if hog1 and abs(p[0] - hog1) < 0.25]
        return dict(tee_to_hog_s=round(hog1 - tee_t, 2) if tee_t and hog1 else None,
                    hog_to_hog_s=round(hog2 - hog1, 2) if hog1 and hog2 else None,
                    speed_at_near_hog=round(float(np.mean(near)), 2) if near else None)


def load_detections(run_dir, start, end, fps):
    rows = []
    for f in sorted(glob.glob(os.path.join(run_dir, "detections", f"*-{fps:g}fps.jsonl"))):
        t0 = float(os.path.basename(f).split("-")[0])
        if t0 + 60 < start or t0 > end:
            continue
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if start <= r["t"] <= end:
                    rows.append(r)
    rows.sort(key=lambda r: r["t"])
    return rows


def track_range(run_dir, cams, start, end, fps, delivery_end):
    rows = load_detections(run_dir, start, end, fps)
    Stone._next = 0
    tracker = Tracker(cams, delivery_end)
    for r in rows:
        tracker.step(r["t"], r["d"], r.get("fg", {}))
    res = tracker.result()
    res.update(start=start, end=end, fps=fps, frames=len(rows))
    return res, rows


def check_settled(res, rows, cams):
    """Independent look at the settled end. Detections from every camera in the two seconds around
    the settle time are clustered on their own (no tracker), then compared with the named stones:
    a stone the cameras hold steadily but the tracker has no name for, or a named stone no camera
    shows, is a bookkeeping error (or, for the latter, a stone someone is standing in front of)."""
    ts = res["settled_at"]
    if not ts:
        return None
    probe = Tracker(cams, res["delivery_end"])
    near = [r for r in rows if abs(r["t"] - ts) <= 1.0]
    clusters = []
    for r in near:
        for d in r["d"]:
            p = np.array(d["ice"])
            if not probe.in_play(p) or probe.along(p[1]) < HOG + 15:  # the playing half only
                continue
            for c in clusters:
                if c["colour"] == d["colour"] and np.linalg.norm(c["p"] - p) < 0.35:
                    c["frames"].add(r["t"])
                    break
            else:
                clusters.append(dict(colour=d["colour"], p=p, frames={r["t"]}))
    seen = [c for c in clusters if len(c["frames"]) >= 0.5 * len(near)]
    colour = {s["id"]: s["colour"] for s in res["stones"]}
    finals = res["final_positions"]

    def close(c, k):
        return colour[k] == c["colour"] and np.hypot(finals[k][0] - c["p"][0], finals[k][1] - c["p"][1]) < 0.4

    return dict(tracked=len(finals), detected=len(seen),
                detected_not_tracked=[(c["colour"][0], round(float(c["p"][0]), 2), round(float(c["p"][1]), 2))
                                      for c in seen if not any(close(c, k) for k in finals)],
                tracked_not_detected=[k for k in finals if not any(close(c, k) for c in seen)])


def cmd_ends(a):
    cams, _ = load_calibration(os.path.join(a.run, "calibration.json"))
    ends = json.load(open(a.ends))
    other = {"A": "B", "B": "A"}
    summary = []
    for e in ends:
        n = e["number"]
        if a.only and n not in a.only:
            continue
        side = a.first_delivery_end if n % 2 == 1 else other[a.first_delivery_end]
        res, rows = track_range(a.run, cams, e["start_seconds"], e["end_seconds"], a.fps, side)
        if not rows:
            print(f"end {n}: no detections yet")
            continue
        res["check"] = check_settled(res, rows, cams)
        json.dump(res, open(os.path.join(a.run, f"end-{n:02d}-tracks.json"), "w"))
        order = "".join(s["id"][0] for s in res["stones"])
        splits = [s["hog_to_hog_s"] for s in res["stones"] if s["hog_to_hog_s"]]
        c = res["check"] or {}
        line = dict(end=n, side=side, deliveries=len(res["stones"]), order=order,
                    alternating=all(x != y for x, y in zip(order, order[1:])), splits=len(splits),
                    split_range=[min(splits), max(splits)] if splits else None, settled=res["settled_at"],
                    tracked=c.get("tracked"), detected=c.get("detected"),
                    detected_not_tracked=c.get("detected_not_tracked"), tracked_not_detected=c.get("tracked_not_detected"),
                    found_again=sum(e["type"] == "found again" for e in res["events"]))
        summary.append(line)
        print(json.dumps(line))
    json.dump(summary, open(os.path.join(a.run, "ends-summary.json"), "w"), indent=1)
    return 0


def cmd_track(a):
    cams, _ = load_calibration(os.path.join(a.run, "calibration.json"))
    rows = load_detections(a.run, a.start, a.end, a.fps)
    if not rows:
        print("no detections in that range; run `stonetrack detect` first")
        return 2
    Stone._next = 0
    tracker = Tracker(cams, a.first_delivery_end)
    for r in rows:
        tracker.step(r["t"], r["d"], r.get("fg", {}))
    res = tracker.result()
    res.update(start=a.start, end=a.end, fps=a.fps, frames=len(rows))
    out = a.out or os.path.join(a.run, f"tracks-{int(a.start)}-{int(a.end)}.json")
    json.dump(res, open(out, "w"))
    print(f"{len(rows)} frames -> {out}")
    for s in res["stones"]:
        dur = s["path"][-1][0] - s["path"][0][0] if s["path"] else 0
        left = next((e for e in res["events"] if e["type"] == "out" and e.get("stone") == s["id"]), None)
        where = f"in play at x={s['final'][0]:+.2f} y={s['final'][1]:6.2f}" if s["final"] else \
            f"gone at {left['t']:.0f}s ({left['why']})" if left else "?"
        splits = f"hog-hog {s['hog_to_hog_s'] or '--':>5}s  {s['speed_at_near_hog'] or '--':>4} m/s"
        print(f"  {s['id']:>3} delivered {s['delivered_at']:7.1f}s  {splits}  tracked {dur:6.1f}s  {where}")
    print("settled at", res["settled_at"])
    return 0


def cmd_render(a):
    from .render import top_down

    cams, _ = load_calibration(os.path.join(a.run, "calibration.json"))
    src = a.out or os.path.join(a.run, f"tracks-{int(a.start)}-{int(a.end)}.json")
    res = json.load(open(src))
    out = src.replace(".json", ".png")
    top_down(res, out)
    print("wrote", out)
    return 0

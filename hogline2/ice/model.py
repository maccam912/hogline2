"""What the ice did to the stones in one game, estimated from every free glide in it.

A stone that has been released and has not yet touched anything is a probe of the ice it is
crossing. Three things move it, and a skip reads all three:

  speed   how hard the ice brakes the stone (keen or heavy ice)
  curl    the sideways pull toward the side the stone is turning to
  fall    a sideways pull that does not care how the stone turns: a slope, a run, a dished side

They can be told apart because a game sends stones both ways down the sheet with both turns.
In ice coordinates (tee A at (0, 0), tee B at (0, TEE), X across the sheet) a stone travelling
in direction d (+1 toward B) with turn r (+1 clockwise) is pulled sideways by

    fall(x, y)  -  d * r * curl(x, y)

so the part that flips with the turn is curl and the part that stays put on the ice is fall.
Along its travel it loses speed at  drag(x, y) + a term in 1/sqrt(speed) + d * tilt.

Nothing is differentiated. Each field is a sum of tent functions on a coarse grid, and a stone's
position is its starting position and velocity plus the double time-integral of those tents
along its own path, which makes the whole fit one linear least-squares problem. Two nuisances
are fitted alongside so that they are not mistaken for ice: a static error in where the cameras
put a stone, as a function of distance down the sheet (it displaces fast and slow stones alike,
where a real pull displaces a slow stone far more), and along the sheet a timing lag between
camera panels (a displacement proportional to speed).

Every field is shrunk toward "the same everywhere". How hard is chosen by predicting shots
that were left out of the fit, so a map only appears when it predicts stones it has not seen.
Uncertainty is a bootstrap over shots.

Sweeping is not observed. It lowers both drag and curl where it happens, so a zone that was
swept on most shots reads as faster and straighter than the bare ice there.
"""

import math

import numpy as np

FT = 0.3048
TEE = 114 * FT
HOG = 21 * FT
BACK = 6 * FT
HALF_WIDTH = 4.75 / 2
STONE_GAP = 0.42  # a stone this close to another's centre may already have touched it
GRAVITY = 9.81

ALONG_KNOTS = np.linspace(-BACK, TEE + BACK, 12)  # about 3.5 m apart
ACROSS_KNOTS = np.array([-1.2, 0.0, 1.2])
WARP_KNOTS = np.linspace(-BACK, TEE + BACK, 20)
LAG_KNOTS = np.linspace(-BACK, TEE + BACK, 8)
CELLS = len(ALONG_KNOTS) * len(ACROSS_KNOTS)

MIN_SAMPLES = 60  # three seconds of glide
BLIND = 1.0  # s without any camera on the stone before the rest of its glide is dropped
MIN_TURN = 0.002  # m/s^2 of whole-glide sideways acceleration needed to call the turn
MIN_SHOTS = 24
MIN_PASSES = 4  # stones through a cell before anything is said about it
BOOTSTRAPS = 300
# The smallest effects worth a skip's attention; anything smaller is not flagged however sure.
CURL_MATTERS = 8.0  # percent more or less curl than the sheet's average
SPEED_MATTERS = 4.0  # percent more or less drag
FALL_MATTERS = 0.3  # mm/s^2 of sideways pull: about 10 cm on a draw that spends 25 s in it
FOLDS = 5
# Effective noise of one sample, in metres. What a shot's own curl and drag leave unexplained is
# 1 to 2 cm sideways and about 5 cm along the sheet, correlated over a second or more, so each
# sample is worth far less than an independent reading; these are scaled up to allow for that.
SIDE_NOISE = 0.08
ALONG_NOISE = 0.25
SHOT_CURL = 0.003  # m/s^2: how much one shot's curl differs from the game's (rotation, sweeping)
SHOT_DRAG = 0.012  # m/s^2: the same for drag, which sweeping changes a good deal


def detrended(knots):
    """Combinations of the tents on `knots` that have no constant and no straight-line part.

    A camera error that is a shift or a stretch of the whole sheet is not something stones can
    reveal (a stretch of the along-sheet scale looks exactly like less drag), so the nuisance
    fields are not allowed one.
    """
    full = np.linalg.svd(np.column_stack([np.ones(len(knots)), knots]).T)[2]
    return full[2:].T


WARP = detrended(WARP_KNOTS)


def hats(u, knots):
    """Tent functions on `knots`, flat beyond the ends: an (n, len(knots)) matrix whose rows sum to 1."""
    u = np.clip(np.asarray(u, float), knots[0], knots[-1])
    i = np.clip(np.searchsorted(knots, u) - 1, 0, len(knots) - 2)
    f = (u - knots[i]) / (knots[i + 1] - knots[i])
    out = np.zeros((len(u), len(knots)))
    rows = np.arange(len(u))
    out[rows, i] = 1 - f
    out[rows, i + 1] = f
    return out


def cells(x, y):
    """The grid's tents at ice positions; the column of cell (i along, j across) is 3 * i + j."""
    return (hats(y, ALONG_KNOTS)[:, :, None] * hats(x, ACROSS_KNOTS)[:, None, :]).reshape(len(x), CELLS)


def twice_integrated(t, f):
    """The double time-integral from the first sample of each column of `f` (trapezoid rule)."""
    dt = np.diff(t)[:, None]
    once = np.vstack([np.zeros((1, f.shape[1])), np.cumsum((f[1:] + f[:-1]) / 2 * dt, axis=0)])
    return np.vstack([np.zeros((1, f.shape[1])), np.cumsum((once[1:] + once[:-1]) / 2 * dt, axis=0)])


# -- free glides ---------------------------------------------------------------------------------


def free_glides(results):
    """One record per delivery: its path from the near hog line until it stops, leaves, or touches a stone.

    `results` is {end number: stonetrack result}. Arrays are per video frame; `seen` is False where
    no camera had the stone and the tracker coasted it.
    """
    glides = []
    for number, result in sorted(results.items()):
        d = 1 if result["delivery_end"] == "A" else -1
        origin = 0.0 if d == 1 else TEE
        stones = result["stones"]
        knocks = [e["t"] for e in result.get("events", []) if e.get("type") == "knocked"]
        for i, stone in enumerate(stones):
            start = stone["delivered_at"]
            following = stones[i + 1]["delivered_at"] if i + 1 < len(stones) else result["end"]
            lying = []
            for other in stones[:i]:
                before = [p for p in other["path"] if p[0] <= start]
                gone = other.get("status") == "out" and other["path"] and other["path"][-1][0] < start
                if before and not gone:
                    lying.append(before[-1][1:3])
            lying = np.array(lying, float).reshape(-1, 2)
            contact = min((k for k in knocks if start < k < following), default=math.inf) - 0.3
            rows, last_seen = [], None
            for p in stone["path"]:
                if p[0] < start or p[5] != "moving":
                    continue
                along = (p[2] - origin) * d
                if along < HOG:
                    continue
                speed = math.hypot(p[3], p[4])
                if p[0] >= contact or p[0] > start + 45 or speed < 0.08:
                    break
                if rows:  # a track that jumps, or was blind for long, is not one stone's free glide
                    gap = p[0] - rows[-1][0]
                    if abs(p[1] - rows[-1][1]) > 0.03 + 0.4 * gap or gap > 0.5:
                        break
                    if abs(abs(p[2] - rows[-1][2]) - speed * gap) > 0.12 + 0.6 * gap:
                        break
                if p[6]:
                    last_seen = p[0]
                elif last_seen is not None and p[0] - last_seen > BLIND:
                    break
                if abs(p[1]) > HALF_WIDTH - 0.15 or along > TEE + BACK:
                    break
                if len(lying) and np.min(np.hypot(lying[:, 0] - p[1], lying[:, 1] - p[2])) < STONE_GAP:
                    break
                rows.append((p[0], p[1], p[2], speed, 1.0 if p[6] else 0.0))
            while rows and not rows[-1][4]:  # do not end on coasted samples
                rows.pop()
            if len(rows) < MIN_SAMPLES:
                continue
            a = np.array(rows)
            t = a[:, 0] - a[0, 0]
            seen = a[:, 4] > 0
            if seen.sum() < MIN_SAMPLES:
                continue
            sideways = 2 * np.polyfit(t[seen], a[seen, 1], 2)[0]
            right = -d * sideways  # the thrower's right is -d * X
            # Drag acts along the stone's travel, so once the stone is heading a little across the
            # sheet a share of it acts across the sheet too. That share is known from the shot
            # itself (its own slowing and heading, smoothed) and is taken out of the sideways
            # motion, leaving only what pushed the stone off its heading.
            go = np.polyfit(t[seen], (a[seen, 2] - origin) * d, 3)
            across = np.polyfit(t[seen], a[seen, 1], 4)
            pace = np.clip(np.polyval(np.polyder(go), t), 0.15, None)
            slowing = np.clip(-np.polyval(np.polyder(go, 2), t), 0.02, 0.3)
            heading = np.clip(np.polyval(np.polyder(across), t) / pace, -0.3, 0.3)
            steer = twice_integrated(t, (slowing * heading)[:, None])[:, 0]
            glides.append(
                dict(
                    end=number,
                    stone=stone["id"],
                    d=d,
                    turn=0 if abs(right) < MIN_TURN else (1 if right > 0 else -1),
                    t=t,
                    x=a[:, 1],
                    off_heading=a[:, 1] + steer,
                    y=a[:, 2],
                    along=(a[:, 2] - origin) * d,
                    speed=a[:, 3],
                    seen=seen,
                    stopped=bool(a[-1, 3] < 0.35),
                )
            )
    return glides


# -- the linear problem --------------------------------------------------------------------------


class Problem:
    """Per-shot normal equations of one direction of motion, with the shot's own start projected out."""

    def __init__(self, blocks, priors, noise):
        """`blocks` is [(name, size)] in column order; `priors` maps a name to (width, smooth)."""
        self.blocks, self.priors, self.noise = blocks, priors, noise
        self.slices, at = {}, 0
        for name, size in blocks:
            self.slices[name] = slice(at, at + size)
            at += size
        self.size = at
        self.A, self.b, self.yy, self.n, self.rows = [], [], [], [], []

    def add(self, t, columns, observed, seen, own, own_width):
        """One shot. `own` is the column of the shot's private effect (its own curl or drag), which
        is free within `own_width` of zero and is eliminated here along with the shot's start."""
        X = np.hstack(columns)[seen]
        y = observed[seen]
        z = own[seen]
        N = np.column_stack([np.ones(seen.sum()), t[seen]])
        proj = np.linalg.solve(N.T @ N, N.T)
        X = X - N @ (proj @ X)
        y = y - N @ (proj @ y)
        z = z - N @ (proj @ z)
        slack = z @ z + (self.noise / own_width) ** 2
        Xz, yz = X.T @ z, y @ z
        self.A.append(X.T @ X - np.outer(Xz, Xz) / slack)
        self.b.append(X.T @ y - Xz * yz / slack)
        self.yy.append(float(y @ y - yz * yz / slack))
        self.n.append(len(y))

    def finish(self):
        self.A, self.b = np.array(self.A), np.array(self.b)
        self.yy, self.n = np.array(self.yy), np.array(self.n)

    def penalty(self, widths):
        """Ridge toward zero plus a smoothness term for the gridded fields."""
        P = np.zeros((self.size, self.size))
        for name, size in self.blocks:
            width, shape = widths.get(name, self.priors[name])
            s = self.slices[name]
            if width is None:  # the field is switched off
                P[s, s] += np.eye(size) * 1e12
                continue
            weight = (self.noise / width) ** 2
            P[s, s] += np.eye(size) * weight
            if shape == "grid":
                P[s, s] += 2 * weight * _roughness()
            elif shape == "line":
                D = np.diff(np.eye(size), axis=0)
                P[s, s] += 4 * weight * D.T @ D
        return P

    def solve(self, weights, widths):
        A = np.tensordot(weights, self.A, axes=1) + self.penalty(widths)
        return np.linalg.solve(A, weights @ self.b)

    def misfit(self, theta):
        """Per shot, the root mean square of what the fit leaves, in metres."""
        left = self.yy - 2 * self.b @ theta + np.einsum("i,sij,j->s", theta, self.A, theta)
        return np.sqrt(np.maximum(left, 0.0) / self.n)

    def trust(self, widths):
        """Weights that stop a few badly tracked shots from steering the fit (Huber, per shot)."""
        self.weights = np.ones(len(self.A))
        for _ in range(3):
            rms = self.misfit(self.solve(self.weights, widths))
            self.weights = np.minimum(1.0, (2.0 * np.median(rms) / np.maximum(rms, 1e-9)) ** 2)
        return self.weights

    def left_out(self, widths, folds):
        """Per shot, the weighted squared error left when the shot was not in the fit."""
        loss = np.zeros(len(self.A))
        for k in range(FOLDS):
            theta = self.solve(self.weights * (folds != k), widths)
            test = folds == k
            left = (
                self.yy[test] - 2 * self.b[test] @ theta + np.einsum("i,sij,j->s", theta, self.A[test], theta)
            )
            loss[test] = self.weights[test] * np.maximum(left, 0.0)
        return loss

    def rms(self, loss):
        return math.sqrt(loss.sum() / (self.weights @ self.n))


_ROUGH = None


def _roughness():
    """Squared differences between neighbouring cells, along and across the sheet."""
    global _ROUGH
    if _ROUGH is None:
        na, nx = len(ALONG_KNOTS), len(ACROSS_KNOTS)
        rows = []
        for i in range(na):
            for j in range(nx):
                for i2, j2 in ((i + 1, j), (i, j + 1)):
                    if i2 < na and j2 < nx:
                        row = np.zeros(CELLS)
                        row[i * nx + j], row[i2 * nx + j2] = 1, -1
                        rows.append(row)
        D = np.array(rows)
        _ROUGH = D.T @ D
    return _ROUGH


def sideways_problem(glides, ends):
    """x(t) of every shot whose turn is known."""
    problem = Problem(
        [
            ("fall", CELLS),
            ("curl", CELLS),
            ("curl0", 1),
            ("curl_by_speed", 1),
            ("curl_trend", 1),
            ("warp", WARP.shape[1]),
        ],
        dict(
            fall=(0.0015, "grid"),
            curl=(0.003, "grid"),
            curl0=(1.0, None),
            curl_by_speed=(1.0, None),
            curl_trend=(0.01, None),
            warp=(0.03, None),
        ),
        SIDE_NOISE,
    )
    used = []
    for g in glides:
        if not g["turn"]:
            continue
        pull = twice_integrated(g["t"], cells(g["x"], g["y"]))
        whole = pull.sum(1, keepdims=True)  # the tents sum to one: this is t^2 / 2
        by_speed = twice_integrated(g["t"], speed_term(g["speed"])[:, None])
        sign = -g["d"] * g["turn"]
        problem.add(
            g["t"],
            [
                pull,
                sign * pull,
                sign * whole,
                sign * by_speed,
                sign * ends[g["end"]] * whole,
                hats(g["y"], WARP_KNOTS) @ WARP,
            ],
            g["off_heading"],
            g["seen"],
            whole[:, 0],
            SHOT_CURL,
        )
        used.append(g)
    problem.finish()
    return problem, used


def speed_term(speed):
    """Drag on ice rises as a stone slows, roughly as 1 / sqrt(speed). Zero near a draw's mid-sheet pace."""
    return 1 / np.sqrt(np.clip(speed, 0.3, None)) - 1 / math.sqrt(1.6)


def along_problem(glides, ends):
    """Distance travelled from the delivery tee, for every shot."""
    problem = Problem(
        [
            ("drag", CELLS),
            ("drag0", 1),
            ("by_speed", 1),
            ("tilt", 1),
            ("drag_trend", 1),
            ("warp", WARP.shape[1]),
            ("lag", len(LAG_KNOTS)),
        ],
        dict(
            drag=(0.008, "grid"),
            drag0=(1.0, None),
            by_speed=(1.0, None),
            tilt=(0.02, None),
            drag_trend=(0.02, None),
            warp=(0.1, None),
            lag=(0.05, "line"),
        ),
        ALONG_NOISE,
    )
    for g in glides:
        brake = -twice_integrated(g["t"], cells(g["x"], g["y"]))
        whole = brake.sum(1, keepdims=True)
        by_speed = -twice_integrated(g["t"], speed_term(g["speed"])[:, None])
        problem.add(
            g["t"],
            [
                brake,
                whole,
                by_speed,
                g["d"] * whole,
                ends[g["end"]] * whole,
                g["d"] * (hats(g["y"], WARP_KNOTS) @ WARP),
                g["speed"][:, None] * hats(g["y"], LAG_KNOTS),
            ],
            g["along"],
            g["seen"],
            whole[:, 0],
            SHOT_DRAG,
        )
    problem.finish()
    return problem, glides


def choose(problem, grids, folds):
    """Prior widths chosen by predicting left-out shots, and what each field is worth.

    Returns (widths, evidence). For each field in `grids`, evidence compares the best fit that
    has the field with the best fit that does not: the held-out error of each, the share of
    the error the field removes, and a paired z over shots of how much better each left-out
    shot was predicted with it.
    """
    names = list(grids)
    shape = {name: problem.priors[name][1] for name in names}
    options = {name: [None, *grids[name]] for name in names}
    tried = {}

    def walk(i, picked):
        if i == len(names):
            widths = {name: (width, shape[name]) for name, width in zip(names, picked, strict=True)}
            tried[tuple(picked)] = problem.left_out(widths, folds)
            return
        for width in options[names[i]]:
            walk(i + 1, [*picked, width])

    walk(0, [])
    total = {key: loss.sum() for key, loss in tried.items()}
    best = min(total, key=total.get)
    evidence = {}
    for i, name in enumerate(names):
        without = min((k for k in total if k[i] is None), key=total.get)
        with_it = min((k for k in total if k[i] is not None), key=total.get)
        gap = tried[without] - tried[with_it]
        spread = gap.std(ddof=1) / math.sqrt(len(gap))
        off, on = problem.rms(tried[without]), problem.rms(tried[with_it])
        gain = 1 - on / off if off > 0 else 0.0
        z = float(gap.mean() / spread) if spread > 0 else 0.0
        evidence[name] = dict(
            uniform_cm=round(100 * off, 2),
            mapped_cm=round(100 * on, 2),
            gain=round(gain, 3),
            z=round(z, 1),
            better_shots=int((gap > 0).sum()),
            shots=len(gap),
            real=bool(gain >= 0.01 and z >= 2.0),
        )
    widths = {name: (width, shape[name]) for name, width in zip(names, best, strict=True)}
    return widths, evidence


def bootstrap(problem, widths, rng):
    n = len(problem.A)
    theta = problem.solve(problem.weights, widths)
    draws = np.array(
        [
            problem.solve(problem.weights * np.bincount(rng.integers(0, n, n), minlength=n), widths)
            for _ in range(BOOTSTRAPS)
        ]
    )
    return theta, draws


# -- reading the fit -----------------------------------------------------------------------------


def simulate_draws(d, x_end, drag, curl, fall, by_speed, tilt, curl_by_speed, stop_along=TEE):
    """Slide stones from the near hog line so that each stops on the far tee line at `x_end`.

    Every argument after `x_end` has one row (or entry) per parameter set, so a fit and its
    bootstrap copies are simulated together. Returns three arrays: hog-to-hog seconds, curl in
    metres toward the turn's side, and fall in metres along +X. The last two split the sideways
    distance between where the stone stops and where its heading at the hog line pointed.
    """
    origin = 0.0 if d == 1 else TEE
    step = 0.25
    grid = np.arange(HOG, TEE + 3 * BACK + step, step)
    basis = cells(np.full(len(grid), x_end), origin + d * grid)
    drag_at, curl_at, fall_at = drag @ basis.T, curl @ basis.T, fall @ basis.T  # (sets, grid)
    sets = np.arange(len(drag_at))

    def look(table, along):
        f = np.clip((along - HOG) / step, 0, len(grid) - 1.001)
        i = f.astype(int)
        return table[sets, i] * (1 - (f - i)) + table[sets, i + 1] * (f - i)

    def run(v0):
        dt = 0.1
        along, v, t = np.full(len(sets), HOG), v0.copy(), np.zeros(len(sets))
        side_curl, side_fall, vc, vf = (np.zeros(len(sets)) for _ in range(4))
        far_hog = np.full(len(sets), np.nan)
        for _ in range(600):
            live = v > 0.02
            if not live.any():
                break
            decel = np.maximum(0.01, look(drag_at, along) + by_speed * speed_term(v) + d * tilt)
            h = np.where(live, np.minimum(dt, v / decel), 0.0)
            ac, af = look(curl_at, along) + curl_by_speed * speed_term(v), look(fall_at, along)
            brake = decel / np.maximum(v, 0.05)  # drag follows the heading, so it brakes the drift too
            ac, af = ac - brake * vc, af - brake * vf
            side_curl += vc * h + ac * h * h / 2
            side_fall += vf * h + af * h * h / 2
            vc += ac * h
            vf += af * h
            along = along + v * h - decel * h * h / 2
            v = v - decel * h
            t = t + h
            far_hog = np.where(np.isnan(far_hog) & (along >= TEE - HOG), t, far_hog)
        return along, far_hog, side_curl, side_fall

    lo, hi = np.full(len(sets), 1.0), np.full(len(sets), 4.5)
    for _ in range(22):
        mid = (lo + hi) / 2
        short = run(mid)[0] < stop_along
        lo, hi = np.where(short, mid, lo), np.where(short, hi, mid)
    _, far_hog, side_curl, side_fall = run((lo + hi) / 2)
    return far_hog, side_curl, side_fall


def _stat(value, draws):
    sd = float(np.std(draws))
    return dict(value=float(value), sd=sd, z=float(value / sd) if sd > 0 else 0.0)


def _grade(z):
    z = abs(z)
    return "clear" if z >= 2.5 else "likely" if z >= 1.8 else "faint"


def build(results):
    """The ice report for one game. `results` is {end number: stonetrack result}."""
    glides = free_glides(results)
    numbers = sorted(results)
    centre = (numbers[0] + numbers[-1]) / 2 if numbers else 0
    ends = {n: n - centre for n in numbers}
    turned = [g for g in glides if g["turn"]]
    counts = {
        f"{'B' if d == 1 else 'A'}:{'cw' if r == 1 else 'ccw'}": int(
            sum(g["d"] == d and g["turn"] == r for g in turned)
        )
        for d in (1, -1)
        for r in (1, -1)
    }
    base = dict(
        shots=len(glides),
        shots_with_turn=len(turned),
        counts=counts,
        ends=[dict(number=n, delivery_end=results[n]["delivery_end"]) for n in numbers],
        grid=dict(along=[round(float(v), 3) for v in ALONG_KNOTS], across=[float(v) for v in ACROSS_KNOTS]),
    )
    if len(glides) < MIN_SHOTS or min(counts.values()) < 3:
        return dict(base, ready=False)

    rng = np.random.default_rng(20261111)
    side, side_shots = sideways_problem(glides, ends)
    along, _ = along_problem(glides, ends)
    side.trust({})
    along.trust({})
    side_folds = rng.permutation(len(side_shots)) % FOLDS
    along_folds = rng.permutation(len(glides)) % FOLDS

    side_widths, side_evidence = choose(
        side, dict(fall=[0.0004, 0.0008, 0.0015, 0.003], curl=[0.001, 0.002, 0.004]), side_folds
    )
    along_widths, along_evidence = choose(along, dict(drag=[0.003, 0.006, 0.012]), along_folds)

    # the maps are drawn from a fit with every field switched on at a moderate width, so that the
    # picture does not vanish when the evidence is thin; the verdict says how far to trust it
    def shown(problem, chosen, defaults):
        return {
            name: chosen[name] if chosen.get(name, (None,))[0] is not None else (defaults[name], "grid")
            for name in defaults
        }

    side_shown = shown(side, side_widths, dict(fall=0.0008, curl=0.002))
    along_shown = shown(along, along_widths, dict(drag=0.006))
    s, s_draws = bootstrap(side, side_shown, rng)
    a, a_draws = bootstrap(along, along_shown, rng)

    sl, al = side.slices, along.slices
    passes = np.zeros(CELLS)
    for g in glides:
        passes += cells(g["x"], g["y"]).max(0) > 0.35
    covered = passes >= MIN_PASSES

    curl0, drag0 = float(s[sl["curl0"]][0]), float(a[al["drag0"]][0])

    # mean-centre the gridded parts over the covered cells so "0" is this sheet's own average
    def centred(theta, draws, block, level):
        value = theta[block] - theta[block][covered].mean()
        spread = draws[:, block] - draws[:, block][:, covered].mean(1, keepdims=True)
        return value, spread, level + theta[block][covered].mean()

    curl_dev, curl_spread, curl_mean = centred(s, s_draws, sl["curl"], curl0)
    drag_dev, drag_spread, drag_mean = centred(a, a_draws, al["drag"], drag0)
    fall, fall_spread = s[sl["fall"]], s_draws[:, sl["fall"]]

    def cell_list(value, spread, scale, matters):
        out = []
        for i in range(len(ALONG_KNOTS)):
            for j in range(len(ACROSS_KNOTS)):
                k = i * len(ACROSS_KNOTS) + j
                if not covered[k]:
                    out.append(None)
                    continue
                st = _stat(value[k], spread[:, k])
                size = abs(st["value"] * scale)
                out.append(
                    dict(
                        value=round(st["value"] * scale, 3),
                        sd=round(st["sd"] * scale, 3),
                        z=round(st["z"], 2),
                        grade=_grade(st["z"]) if size >= matters else "faint",
                        passes=int(passes[k]),
                    )
                )
        return out

    # a draw to the tee line, each way and each turn, down the centre and down each side
    by_speed, tilt = float(a[al["by_speed"]][0]), float(a[al["tilt"]][0])
    draws_table = []
    sample = rng.choice(BOOTSTRAPS, 100, replace=False)

    st = np.vstack([s[None], s_draws[sample]])
    at = np.vstack([a[None], a_draws[sample]])
    for d in (1, -1):
        for x in (-1.2, 0.0, 1.2):
            time, curl_m, fall_m = simulate_draws(
                d,
                x,
                at[:, al["drag"]] + at[:, al["drag0"]],
                st[:, sl["curl"]] + st[:, sl["curl0"]],
                st[:, sl["fall"]],
                at[:, al["by_speed"]][:, 0],
                at[:, al["tilt"]][:, 0],
                st[:, sl["curl_by_speed"]][:, 0],
            )
            draws_table.append(
                dict(
                    toward="B" if d == 1 else "A",
                    x=x,
                    hog_to_hog_s=round(float(time[0]), 2),
                    hog_to_hog_sd=round(float(np.nanstd(time[1:])), 2),
                    curl_m=round(float(curl_m[0]), 3),
                    curl_sd=round(float(np.std(curl_m[1:])), 3),
                    fall_x_m=round(float(fall_m[0]), 3),
                    fall_sd=round(float(np.std(fall_m[1:])), 3),
                )
            )

    return dict(
        base,
        ready=True,
        used=dict(
            sideways=int(round(float(side.weights.sum()))),
            along=int(round(float(along.weights.sum()))),
        ),
        curl=dict(
            mean=round(curl_mean, 5),
            trend=_rounded(_stat(s[sl["curl_trend"]][0], s_draws[:, sl["curl_trend"]][:, 0]), 5),
            cells=cell_list(curl_dev / curl_mean, curl_spread / curl_mean, 100, CURL_MATTERS),
            evidence=side_evidence["curl"],
        ),
        fall=dict(
            cells=cell_list(fall, fall_spread, 1000, FALL_MATTERS),
            evidence=side_evidence["fall"],
        ),
        speed=dict(
            mean=round(float(drag_mean), 4),
            by_speed=round(by_speed, 4),
            tilt=_rounded(_stat(tilt, a_draws[:, al["tilt"]][:, 0]), 5),
            trend=_rounded(_stat(a[al["drag_trend"]][0], a_draws[:, al["drag_trend"]][:, 0]), 5),
            cells=cell_list(drag_dev / drag_mean, drag_spread / drag_mean, 100, SPEED_MATTERS),
            evidence=along_evidence["drag"],
        ),
        draws=draws_table,
    )


def _rounded(stat, digits):
    return dict(value=round(stat["value"], digits), sd=round(stat["sd"], digits), z=round(stat["z"], 2))

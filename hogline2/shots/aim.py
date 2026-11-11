"""Aim and weight numbers for one delivery, from its tracked path and a hand-marked broom.

The skip's broom is marked once per shot, where it touches the ice. For every delivery:

  line   where the stone's heading over its first six metres past the hog line would cross the
         broom's depth, compared with the broom: a miss in centimetres, to the thrower's left or
         right, and "narrow" (toward the side the stone curls) or "wide"
  turn   which way the stone curled away from that initial line (clockwise curls right)
  weight the hog-to-hog and tee-to-hog split times stonetrack measures

Positions are ice metres with tee A at (0, 0) and tee B at (0, TEE); pixels are in the full mosaic.
"""

import math

FT = 0.3048
TEE = 114 * FT
HOG = 21 * FT
POSITIONS = ["lead", "second", "third", "fourth"]
CALLS = ("draw", "guard", "takeout", "other")


def project(h, p):
    z = h[6] * p[0] + h[7] * p[1] + h[8]
    return [(h[0] * p[0] + h[1] * p[1] + h[2]) / z, (h[3] * p[0] + h[4] * p[1] + h[5]) / z]


def camera_at(calibration, px):
    for cam in calibration["cameras"].values():
        x0, y0, x1, y1 = cam["crop"]
        if x0 <= px[0] < x1 and y0 <= px[1] < y1:
            return cam
    return None


def _line(points):
    """Least-squares x = a + b * along."""
    n = len(points)
    mu = sum(p[0] for p in points) / n
    mx = sum(p[1] for p in points) / n
    den = sum((p[0] - mu) ** 2 for p in points)
    b = sum((p[0] - mu) * (p[1] - mx) for p in points) / den
    return mx - b * mu, b


def shot_line(stone, delivery_end):
    """Initial heading and curl of one delivery. None if the path after release is too thin."""
    d = 1 if delivery_end == "A" else -1
    origin = 0.0 if delivery_end == "A" else TEE
    glide = [
        ((p[2] - origin) * d, p[1], p[0])
        for p in stone["path"]
        if p[5] == "moving" and p[0] >= stone["delivered_at"] and p[0] <= stone["delivered_at"] + 30
    ]
    first = [g for g in glide if HOG + 0.5 <= g[0] <= HOG + 6.5]
    if len(first) < 15 or first[-1][0] - first[0][0] < 3:
        return None
    a, b = _line(first)
    late = [g for g in glide if HOG + 6.5 < g[0] <= TEE - HOG]
    curl_right = None
    if late:
        along, x, _ = late[-1]
        curl_right = -d * (x - (a + b * along))  # the thrower's right is -d * X
    turn = None
    if curl_right is not None and abs(curl_right) >= 0.05:
        turn = "clockwise" if curl_right > 0 else "counter-clockwise"
    return dict(a=a, b=b, d=d, origin=origin, curl_right_m=curl_right, turn=turn, samples=len(first))


def aim(stone, delivery_end, broom_ice):
    line = shot_line(stone, delivery_end)
    if not line:
        return None

    def on_line(along):  # ice position of the initial heading at a distance from the delivery tee
        return [round(line["a"] + line["b"] * along, 3), round(line["origin"] + line["d"] * along, 3)]

    out = dict(
        turn=line["turn"],
        curl_right_m=None if line["curl_right_m"] is None else round(line["curl_right_m"], 3),
        heading=[on_line(HOG), on_line(TEE + 1.8)],
    )
    if broom_ice:
        depth = (broom_ice[1] - line["origin"]) * line["d"]
        miss = line["a"] + line["b"] * depth - broom_ice[0]
        right = -line["d"] * miss
        out.update(broom_depth_m=round(depth, 2), miss_right_cm=round(100 * right, 1))
        if line["turn"]:
            out["miss_narrow_cm"] = round(100 * right * (1 if line["turn"] == "clockwise" else -1), 1)
    return out


def _mean(values):
    return sum(values) / len(values) if values else None


def _sd(values):
    if len(values) < 2:
        return None
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def summarise(shots):
    """Per player: line by turn, and weight by called shot."""
    players = {}
    for s in shots:
        p = players.setdefault(
            s["player"], dict(player=s["player"], colour=s["colour"], shots=0, marked=0, turns={}, calls={})
        )
        p["shots"] += 1
        a = s.get("aim") or {}
        if a.get("miss_right_cm") is not None:
            p["marked"] += 1
            t = p["turns"].setdefault(a.get("turn") or "unclear", dict(narrow=[], absolute=[]))
            t["absolute"].append(abs(a["miss_right_cm"]))
            if a.get("miss_narrow_cm") is not None:
                t["narrow"].append(a["miss_narrow_cm"])
        if s.get("hog_to_hog_s"):
            p["calls"].setdefault((s.get("mark") or {}).get("call") or "unlabelled", []).append(
                s["hog_to_hog_s"]
            )
    out = []
    for p in players.values():
        turns = {
            name: dict(
                shots=len(t["absolute"]),
                mean_abs_cm=round(_mean(t["absolute"]), 1),
                mean_narrow_cm=None if not t["narrow"] else round(_mean(t["narrow"]), 1),
            )
            for name, t in p["turns"].items()
        }
        calls = {
            name: dict(
                shots=len(v),
                mean_hog_to_hog_s=round(_mean(v), 2),
                spread_s=None if _sd(v) is None else round(_sd(v), 2),
            )
            for name, v in p["calls"].items()
        }
        out.append(
            dict(
                player=p["player"],
                colour=p["colour"],
                shots=p["shots"],
                marked=p["marked"],
                line=turns,
                weight=calls,
            )
        )
    return sorted(out, key=lambda r: (r["colour"], r["player"]))

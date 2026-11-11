"""Shot review: every tracked delivery, its broom mark and call, team lineups and the report."""

import time
from typing import Literal

from fastapi import HTTPException
from pydantic import Field

from ..files import read_json, write_json
from ..models import Model
from ..tracking.api import latest_runs
from ..tracking.layout import other
from .aim import CALLS, POSITIONS, aim, camera_at, project, summarise

REST_TOLERANCE_M = 0.005


class Mark(Model):
    """What the reviewer recorded for one shot. `px` is the broom on the ice, in mosaic pixels."""

    seconds: float | None = Field(default=None, ge=0)
    px: tuple[float, float] | None = None
    call: Literal["draw", "guard", "takeout", "other"] | None = None
    no_broom: bool = False


class Lineups(Model):
    red: list[str] = Field(default_factory=lambda: [""] * 4, min_length=4, max_length=4)
    yellow: list[str] = Field(default_factory=lambda: [""] * 4, min_length=4, max_length=4)


def stone_key(label):
    """stonetrack's R3 is this app's red_3."""
    return f"{'red' if label[0] == 'R' else 'yellow'}_{label[1:]}"


def compact_path(path):
    """[seconds, x, y, moving] rows, without the repeats of a stone sitting still."""
    rows = []
    for p in path:
        row = [round(p[0], 3), round(p[1], 3), round(p[2], 3), 1 if p[5] == "moving" else 0]
        still = (
            len(rows) > 1
            and not (row[3] or rows[-1][3] or rows[-2][3])
            and row[0] - rows[-1][0] < 0.3
            and abs(row[1] - rows[-2][1]) < REST_TOLERANCE_M
            and abs(row[2] - rows[-2][2]) < REST_TOLERANCE_M
        )
        if still:
            rows[-1] = row  # slide the end of the resting stretch forward
        else:
            rows.append(row)
    return rows


def mount_shots(app, config, store):
    lock = app.state.jobs_lock

    def root(identifier):
        store.get(identifier)  # 404 for an unknown recording
        return config.data / "recordings" / identifier

    def marks_path(identifier):
        return root(identifier) / "tracking" / "broom-marks.json"

    def lineups(identifier):
        saved = read_json(root(identifier) / "team-lineups.json") or {}
        return {colour: (saved.get(colour) or [""] * 4)[:4] for colour in ("red", "yellow")}

    cache = {}  # result files are several megabytes; keep the parsed ones while they are unchanged

    def load_result(path):
        stamp = path.stat().st_mtime_ns
        if cache.get(path, (None,))[0] != stamp:
            cache[path] = (stamp, read_json(path))
        return cache[path][1]

    def results(identifier):
        runs = latest_runs(root(identifier) / "tracking")
        return {n: (f.name, load_result(f / "stonetrack-result.json")) for n, f in sorted(runs.items())}

    def cameras(identifier):
        calibration = read_json(config.data / "stonetrack" / identifier / "calibration.json")
        return calibration["cameras"] if calibration else {}

    def build(identifier, tracked=None):
        names, marks = lineups(identifier), read_json(marks_path(identifier)) or {}
        shots = []
        for number, (run_id, result) in (tracked or results(identifier)).items():
            stones = result["stones"]
            for i, stone in enumerate(stones):
                colour, n = stone["colour"], int(stone["id"][1:])
                position = POSITIONS[(n - 1) // 2]
                key = stone_key(stone["id"])
                mark = marks.get(f"{number}:{key}")
                rest = [p[0] for p in stone["path"] if p[0] > stone["delivered_at"] + 2 and p[5] != "moving"]
                following = stones[i + 1]["delivered_at"] if i + 1 < len(stones) else result["end"]
                shots.append(
                    dict(
                        end=number,
                        stone=key,
                        label=stone["id"],
                        colour=colour,
                        throw=i + 1,
                        position=position,
                        player=names[colour][(n - 1) // 2] or f"{colour.capitalize()} {position}",
                        run_id=run_id,
                        delivery_end=result["delivery_end"],
                        delivered_at=stone["delivered_at"],
                        stopped_at=rest[0] if rest else stone["path"][-1][0],
                        next_at=following,
                        hog_to_hog_s=stone.get("hog_to_hog_s"),
                        tee_to_hog_s=stone.get("tee_to_hog_s"),
                        outcome=stone.get("reason") if stone.get("status") == "out" else "in play",
                        mark=mark,
                        aim=aim(stone, result["delivery_end"], (mark or {}).get("ice")),
                    )
                )
        return shots

    @app.get("/api/recordings/{identifier}/shots")
    def shots(identifier: str):
        record = store.get(identifier)
        tracked = results(identifier)
        found = build(identifier, tracked)
        ends = [
            dict(
                number=e["number"],
                start_seconds=e["start_seconds"],
                end_seconds=e["end_seconds"],
                tracked=e["number"] in tracked,
                delivery_end=tracked[e["number"]][1]["delivery_end"] if e["number"] in tracked else None,
            )
            for e in record.get("ends", [])
        ]
        return dict(
            shots=found,
            ends=ends,
            cameras=list(cameras(identifier).values()),
            calls=CALLS,
            lineups=lineups(identifier),
            report=summarise(found),
        )

    @app.get("/api/recordings/{identifier}/ends/{number}/paths")
    def paths(identifier: str, number: int):
        tracked = results(identifier)
        if number not in tracked:
            raise HTTPException(404, "This end has not been tracked")
        result = tracked[number][1]
        return dict(
            end=number,
            start=result["start"],
            finish=result["end"],
            stones=[
                dict(
                    stone=stone_key(s["id"]),
                    label=s["id"],
                    colour=s["colour"],
                    out=s.get("status") == "out",
                    path=compact_path(s["path"]),
                )
                for s in result["stones"]
            ],
        )

    @app.put("/api/recordings/{identifier}/shots/{end_number}/{stone}")
    def save_mark(identifier: str, end_number: int, stone: str, request: Mark):
        with lock:
            shot = next(
                (s for s in build(identifier) if s["end"] == end_number and s["stone"] == stone), None
            )
            if shot is None:
                raise HTTPException(404, "No tracked delivery with that end and stone")
            mark = dict(call=request.call, no_broom=request.no_broom, saved_at=time.time())
            if request.px and not request.no_broom:
                if request.seconds is None:
                    raise HTTPException(400, "A broom mark needs the video time it was made at")
                playing = other(shot["delivery_end"])
                camera = camera_at(dict(cameras=cameras(identifier)), request.px)
                if camera is None or camera["end"] != playing:
                    raise HTTPException(
                        400, f"Click the broom in a view of house {playing}, where this stone is headed"
                    )
                mark.update(
                    seconds=request.seconds,
                    px=list(request.px),
                    view=camera["id"],
                    ice=[round(v, 4) for v in project(camera["image_to_ice"], request.px)],
                )
            marks = read_json(marks_path(identifier)) or {}
            key = f"{end_number}:{stone}"
            if mark.get("px") or mark["call"] or mark["no_broom"]:
                marks[key] = mark
            else:
                marks.pop(key, None)
            write_json(marks_path(identifier), marks, indent=1)
            found = build(identifier)
            return dict(
                shot=next(s for s in found if s["end"] == end_number and s["stone"] == stone),
                report=summarise(found),
            )

    @app.get("/api/recordings/{identifier}/lineups")
    def get_lineups(identifier: str):
        return lineups(identifier)

    @app.put("/api/recordings/{identifier}/lineups")
    def save_lineups(identifier: str, request: Lineups):
        value = {
            colour: [name.strip()[:100] for name in names] for colour, names in request.model_dump().items()
        }
        with lock:
            write_json(root(identifier) / "team-lineups.json", value, indent=1)
        return value

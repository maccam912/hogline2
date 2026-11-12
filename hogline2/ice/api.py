"""The ice report of a recording: the model's maps plus the glides they were read from."""

from ..files import read_json
from ..tracking.api import latest_runs
from .model import build, free_glides


def outline(glide, points=24):
    """A glide thinned to a few [x, y] points, for drawing where the stones went."""
    step = max(1, len(glide["x"]) // points)
    xs, ys = glide["x"][::step], glide["y"][::step]
    return dict(
        end=glide["end"],
        stone=glide["stone"],
        toward="B" if glide["d"] == 1 else "A",
        turn={1: "clockwise", -1: "counter-clockwise", 0: None}[glide["turn"]],
        points=[[round(float(x), 2), round(float(y), 2)] for x, y in zip(xs, ys, strict=True)],
    )


def mount_ice(app, config, store):
    cache = {}  # the fit takes a second or two; keep it until a tracked end changes

    @app.get("/api/recordings/{identifier}/ice")
    def ice(identifier: str):
        record = store.get(identifier)
        runs = latest_runs(config.data / "recordings" / identifier / "tracking")
        stamp = tuple(
            (n, f.name, (f / "stonetrack-result.json").stat().st_mtime_ns) for n, f in sorted(runs.items())
        )
        if cache.get(identifier, (None,))[0] != stamp:
            results = {n: read_json(f / "stonetrack-result.json") for n, f in sorted(runs.items())}
            report = build(results)
            report["glides"] = [outline(g) for g in free_glides(results)]
            cache[identifier] = (stamp, report)
        untracked = [e["number"] for e in record.get("ends", []) if e["number"] not in runs]
        return dict(cache[identifier][1], untracked=untracked)

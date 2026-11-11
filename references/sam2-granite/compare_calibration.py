"""Compare the original and independently calibrated normal tracking runs."""

# Literal report HTML and SVG are kept on long lines.
# ruff: noqa: E501
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/camera-calibration-20261108"
TRACKING = ROOT / "data/recordings/eDJdwj0JqBY/tracking"
NEW = TRACKING / json.loads((OUT / "run.json").read_text())["run_id"]
OLD = TRACKING / "392ea127db464389a1b970e627289537"


def load(folder):
    assert json.loads((folder / "status.json").read_text())["state"] == "complete"
    return dict(
        manifest=json.loads((folder / "manifest.json").read_text()),
        rows=[json.loads(s) for s in (folder / "observations.jsonl").read_text().splitlines()],
        fitted=json.loads((folder / "trajectory.json").read_text())["frames"],
        events=json.loads((folder / "events.json").read_text()),
    )


def velocities(rows):
    last = {}
    samples = []
    for row in rows:
        seconds = row["source_seconds"]
        for o in row["observations"]:
            if not o.get("accepted"):
                continue
            key = (o["stone_id"], o["view_id"])
            previous = last.get(key)
            if previous and abs(seconds - previous[0] - 1 / 3) < 0.001:
                dt = seconds - previous[0]
                samples.append(
                    dict(
                        stone=o["stone_id"],
                        view=o["view_id"],
                        seconds=seconds,
                        speed=math.dist(o["xy"], previous[1]) / dt,
                        xy=o["xy"],
                    )
                )
            last[key] = (seconds, o["xy"])
    return samples


def pairs(samples):
    grouped = {}
    for sample in samples:
        if sample["view"].startswith("perspective_"):
            grouped.setdefault((sample["stone"], sample["seconds"]), {})[sample["view"]] = sample
    result = []
    for (stone, seconds), views in grouped.items():
        if len(views) != 2:
            continue
        a, b = views["perspective_a"], views["perspective_b"]
        result.append(
            dict(
                stone=stone,
                seconds=seconds,
                speed_a=a["speed"],
                speed_b=b["speed"],
                absolute_speed_difference=abs(a["speed"] - b["speed"]),
                position_difference=math.dist(a["xy"], b["xy"]),
            )
        )
    return result


def mean(values):
    return sum(values) / len(values) if values else None


def svg_chart(data, stone):
    start = 378
    duration = 120
    width = 1000
    height = 220

    def x(t):
        return 50 + (t - start) / duration * 920

    def y(speed):
        return 180 - speed / 4 * 150

    svg = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{stone} before and after speed">',
        '<rect width="1000" height="220" rx="8" fill="#edf5f1"/>',
        f'<text x="50" y="20">{stone} · fitted speed, m/s</text>',
    ]
    for speed in [0, 1, 2, 3, 4]:
        svg.append(
            f'<path d="M50 {y(speed)} H970" stroke="#c9d7d0"/><text x="20" y="{y(speed) + 4}">{speed}</text>'
        )
    for seconds in [0, 30, 60, 90, 120]:
        svg.append(f'<text x="{x(start + seconds) - 8}" y="205">{seconds}s</text>')
    for name, color in [("before", "#919b9a"), ("after", "#15784b")]:
        points = []
        segments = []
        for row in data[name]["fitted"]:
            s = next((s for s in row["stones"] if s["id"] == stone), None)
            if not s or s.get("estimate_kind") == "lost":
                if points:
                    segments.append(points)
                    points = []
                continue
            points.append(
                f"{x(row['source_seconds']):.2f},{y(s.get('speed_mps', math.hypot(*s['velocity']))):.2f}"
            )
        if points:
            segments.append(points)
        for segment in segments:
            svg.append(
                f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2.5"/>'
            )
    svg.append("</svg>")
    return "".join(svg)


def main():
    data = {"before": load(OLD), "after": load(NEW)}
    all_pairs = {name: pairs(velocities(d["rows"])) for name, d in data.items()}
    metrics = {}
    for name, d in data.items():
        counts = Counter(
            (o["stone_id"], o["view_id"]) for r in d["rows"] for o in r["observations"] if o.get("accepted")
        )
        pp = all_pairs[name]
        metrics[name] = dict(
            run=d["manifest"]["run_id"]
            if "run_id" in d["manifest"]
            else (OLD if name == "before" else NEW).name,
            calibration_revision=d["manifest"]["calibration"]["revision"],
            ids=sorted({s["id"] for r in d["rows"] for s in r["stones"]}),
            accepted={f"{s}/{v}": n for (s, v), n in counts.items()},
            paired_speed_frames=len(pp),
            mean_absolute_speed_disagreement=mean([p["absolute_speed_difference"] for p in pp]),
            mean_position_disagreement=mean([p["position_difference"] for p in pp]),
            red_at_394=next((p for p in pp if p["stone"] == "red_1" and p["seconds"] == 394), None),
        )
    common = {(p["stone"], p["seconds"]) for p in all_pairs["before"]} & {
        (p["stone"], p["seconds"]) for p in all_pairs["after"]
    }
    matched = {
        name: dict(
            frames=len(common),
            mean_absolute_speed_disagreement=mean(
                [p["absolute_speed_difference"] for p in pp if (p["stone"], p["seconds"]) in common]
            ),
        )
        for name, pp in all_pairs.items()
    }
    result = dict(
        runs=metrics,
        matched_frames=matched,
        pairs=all_pairs,
        note="Per-camera velocities use consecutive accepted raw positions; no velocity smoothing. "
        "New SAM tracking is rerun, so masks can differ. Comparison curves show whole-clip estimates.",
    )
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2))
    evidence = json.loads((OUT / "fit-evidence.json").read_text())
    a, b = metrics["before"]["red_at_394"], metrics["after"]["red_at_394"]
    page = """<!doctype html><meta charset="utf-8"><title>Independent camera calibration</title>
<style>body{font:17px/1.55 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;background:#f7faf8;color:#18302b}svg{width:100%;font:13px system-ui;margin-bottom:16px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #cbd8d1}a{color:#146a54}</style>
<h1>Independent camera calibration — applied</h1>
<p>Calibration revision 5 gives each perspective its own eight-parameter image-to-ice mapping. The camera mappings do not share zoom, roll, direction, tilt or a single pixels-per-metre scale. Each overhead retains its independently marked ring calibration. The adopted 34.747 m button spacing is fixed.</p>
<p>The old calibration and runs are preserved. The updated normal tracker was rerun on source 378–498 seconds at 3 fps.</p>
<h2>The first red stone at source 394 s</h2>
<table><tr><th>Mapping</th><th>Delivery camera B</th><th>Receiving camera A</th></tr>"""
    page += f"<tr><td>Before</td><td>{a['speed_b']:.2f} m/s</td><td>{a['speed_a']:.2f} m/s</td></tr>"
    page += (
        f"<tr><td>After rerun</td><td>{b['speed_b']:.2f} m/s</td><td>{b['speed_a']:.2f} m/s</td></tr></table>"
    )
    page += f"<p>Across {len(common)} matched simultaneous samples, mean absolute camera-speed disagreement changes from {matched['before']['mean_absolute_speed_disagreement']:.3f} to {matched['after']['mean_absolute_speed_disagreement']:.3f} m/s. These are raw same-camera position differences, not fitted speeds.</p>"
    page += "<h2>Every stone: before (grey) / after (green)</h2><p>Same time and speed scales. The curves below are whole-clip trajectory estimates; they can still include segmentation noise, model assumptions and contact errors.</p>"
    page += "".join(svg_chart(data, s) for s in ["red_1", "yellow_1", "red_2"])
    page += f"<p><a href='/tracking?recording=eDJdwj0JqBY&amp;run={NEW.name}'>Open the recalibrated replay</a></p>"
    page += f"<h2>Validation and limits</h2><p>The fit used {evidence['pairs']} spaced shared-stone pairs and marked sheet features, with leave-one-stone-out RMS disagreement of {evidence['held_out_pair_rms_m']:.3f} m. Each camera has separate free projective coefficients. This is an ice-plane calibration, not a separately recovered 3D lens/camera-pose model. Radial lens distortion and visible-rim versus true stone-center bias remain unresolved. Agreement does not establish surveyed absolute accuracy.</p>"
    page += "<p>Camera fits are saved with their source run and a fingerprint of the crop, landmarks, orientation and sheet dimensions. Editing any of those invalidates the affected fit. Source masks and previous calibration snapshots remain available.</p>"
    page += "<p>Method background: <a href='https://docs.opencv.org/4.12.0/d9/dab/tutorial_homography.html'>OpenCV planar camera mappings</a>.</p>"
    (OUT / "comparison.html").write_text(page)
    (NEW / "comparison.html").write_text(page)
    manifest = data["after"]["manifest"]
    manifest["camera_report"] = True
    (NEW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(runs=metrics, matched=matched), indent=2))


if __name__ == "__main__":
    main()

"""Compare raw velocity at matched times, preserving spikes and reporting gaps."""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/end1-velocity-20261108"
WINDOWS = [
    ("rest_red_house", "red_1", "overhead_a", 414, 442, "stationary"),
    ("rest_red_perspective", "red_1", "perspective_a", 414, 442, "stationary"),
    ("rest_yellow_house", "yellow_1", "overhead_a", 454, 491, "stationary"),
    ("rest_yellow_perspective", "yellow_1", "perspective_a", 454, 491, "stationary"),
    ("red_delivery", "red_1", "perspective_b", 385, 391, "travel"),
    ("red_arrival", "red_1", "perspective_a", 394, 405, "travel"),
    ("yellow_delivery", "yellow_1", "perspective_b", 430, 435, "travel"),
    ("yellow_arrival", "yellow_1", "perspective_a", 439, 445, "travel"),
    ("red2_delivery", "red_2", "perspective_b", 473, 479, "travel"),
    ("red2_arrival", "red_2", "perspective_a", 483, 491, "travel"),
]


def stats(samples):
    measured = [p for p in samples if p.get("valid")]
    speeds = [p["speed_mps"] for p in measured if p.get("speed_mps") is not None]
    pixels = [p["pixel_speed"] for p in measured if p.get("pixel_speed") is not None]
    jumps = []
    previous = None
    for point in samples:
        if (
            previous
            and point.get("velocity") is not None
            and previous.get("velocity") is not None
            and point["frame"] == previous["frame"] + 1
        ):
            jumps.append(math.dist(point["velocity"], previous["velocity"]))
        previous = point
    return dict(
        expected_frames=len(samples),
        measured_frames=len(measured),
        velocity_samples=len(speeds),
        empty_frames=len(samples) - len(measured),
        clipped_frames=sum(p.get("clipped", False) for p in measured),
        median_speed=float(np.median(speeds)) if speeds else None,
        p95_speed=float(np.percentile(speeds, 95)) if speeds else None,
        max_speed=max(speeds) if speeds else None,
        p95_pixel_speed=float(np.percentile(pixels, 95)) if pixels else None,
        p95_velocity_step=float(np.percentile(jumps, 95)) if jumps else None,
    )


def main(components=False):
    runs = json.loads((OUT / "runs.json").read_text())
    all_data = {
        backend: json.loads((Path(path) / "raw-velocity.json").read_text())["samples"]
        for backend, path in runs.items()
    }
    if components:
        all_data = json.loads((OUT / "component-velocity.json").read_text())
    summary = {
        "method": "Raw finite differences; no smoothing, gates, speed clipping, or gap interpolation",
        "scope": "First 120 seconds of end 1, three identities, matched camera windows",
        "materiality_rule": (
            "Exploratory: >=25% lower stationary p95 speed on both cameras, "
            "without coverage loss or identity drift; no significance claim"
        ),
        "windows": [],
        "runs": {},
    }
    if components:
        summary["method"] = "Largest connected mask component only; no temporal or velocity filtering"
        summary["post_hoc"] = True
    for backend, path in runs.items():
        manifest = json.loads((Path(path) / "manifest.json").read_text())
        events = json.loads((Path(path) / "events.json").read_text())
        summary["runs"][backend] = dict(
            path=path,
            elapsed_seconds=manifest["elapsed_seconds"],
            peak_metal_bytes=manifest["peak_metal_bytes"],
            reseed_attempts=len(events),
            reseed_accepted=sum(e.get("accepted", False) for e in events),
        )
    for label, stone, view, start, end, kind in WINDOWS:
        selected = {
            b: [
                p
                for p in data
                if p["stone_id"] == stone and p["view_id"] == view and start <= p["source_seconds"] <= end
            ]
            for b, data in all_data.items()
        }
        common = set.intersection(
            *[
                {p["frame"] for p in samples if p.get("speed_mps") is not None}
                for samples in selected.values()
            ]
        )
        row = dict(
            name=label,
            stone=stone,
            view=view,
            start=start,
            end=end,
            kind=kind,
            per_backend={b: stats(v) for b, v in selected.items()},
            paired={b: stats([p for p in v if p["frame"] in common]) for b, v in selected.items()},
        )
        summary["windows"].append(row)
    prefix = "component-" if components else ""
    (OUT / (prefix + "summary.json")).write_text(json.dumps(summary, indent=2))
    # Embed the actual raw series so the standalone review does not depend on fetch/file permissions.
    payload = json.dumps(dict(data=all_data, summary=summary)).replace("</", "<\\/")
    html = Path(__file__).with_name("comparison-template.html").read_text().replace("PAYLOAD", payload)
    if components:
        html = html.replace(
            "<h1>End 1: raw velocity</h1>",
            (
                "<h1>End 1: largest-component mask diagnostic</h1>"
                "<p><strong>Secondary, post-hoc check:</strong> only the largest connected component of "
                "each mask supplies its position. This removes stray mask islands but applies no "
                "temporal or velocity filter. Original masks and the primary comparison are unchanged. "
                "<a href='comparison.html'>View fully raw comparison</a>.</p>"
            ),
        )
    else:
        html = html.replace(
            "</h1>",
            (
                "</h1><p><a href='component-comparison.html'>Separate largest-component diagnostic</a> "
                "· <a href='comparison.mp4'>Source / SAM2 / SAM3.1 highlight replay</a></p>"
            ),
            1,
        )
    (OUT / (prefix + "comparison.html")).write_text(html)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
    if (OUT / "component-velocity.json").exists():
        main(components=True)

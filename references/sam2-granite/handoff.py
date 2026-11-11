"""Compare same-camera speeds across the first red stone's perspective handoff."""

import json
import math
from pathlib import Path

from hogline2.tracking_geometry import project

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/recordings/eDJdwj0JqBY/tracking/392ea127db464389a1b970e627289537"
rows = [json.loads(line) for line in (RUN / "observations.jsonl").read_text().splitlines()]
whole = json.loads((RUN / "trajectory.json").read_text())["frames"]
homographies = json.loads((RUN / "geometry-report.json").read_text())["image_to_ice"]
last = {}
result = []
for row, fitted in zip(rows, whole, strict=True):
    seconds = row["source_seconds"]
    cameras = []
    for obs in row["observations"]:
        if obs["stone_id"] != "red_1" or not obs.get("accepted"):
            continue
        view = obs["view_id"]
        revised = project(homographies[view], obs["point"])
        previous = last.get(view)
        if previous and abs(seconds - previous[0] - 1 / 3) < 0.001:
            dt = seconds - previous[0]
            cameras.append(
                dict(
                    view=view,
                    point=obs["point"],
                    xy=obs["xy"],
                    current_calibration_speed_mps=math.dist(obs["xy"], previous[1]) / dt,
                    candidate_calibration_speed_mps=math.dist(revised, previous[2]) / dt,
                )
            )
        last[view] = (seconds, obs["xy"], revised)
    if 387 <= seconds <= 402:
        forward = next(s for s in row["stones"] if s["id"] == "red_1")
        fit = next(s for s in fitted["stones"] if s["id"] == "red_1")
        result.append(
            dict(
                source_seconds=seconds,
                replay_seconds=seconds - 378,
                forward_speed_mps=math.hypot(*forward["velocity"]),
                fitted_speed_mps=fit["speed_mps"],
                cameras=cameras,
            )
        )
output = dict(
    run_id=RUN.name,
    stone="red_1",
    method="Consecutive accepted same-camera position differences. No speed smoothing. "
    "Candidate transforms reproject identical saved pixel anchors; not applied to replay.",
    caveat="Candidate fitted to these tracks; this is a calibration diagnostic, not ground-truth speed.",
    samples=result,
)
(ROOT / "data/experiments/velocity-handoff-20261108/diagnostic.json").write_text(json.dumps(output, indent=2))
print(json.dumps(next(s for s in result if s["source_seconds"] == 394), indent=2))

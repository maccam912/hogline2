"""Re-measure both saved SAM2 runs with the same visible-body anchor."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from hogline2.tracking_geometry import prepare_view, sheet_geometry
from hogline2.tracking_raw import add_raw_velocity, raw_mask_measurement
from hogline2.tracking_stone import granite_anchor, granite_evidence

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/sam2-granite-20261108"
RUNS = {
    "previous": ROOT / "data/recordings/eDJdwj0JqBY/tracking/ff22e901162b4412878e6a3ad95ea57e",
    "granite": Path(json.loads((OUT / "run.json").read_text())["path"]),
}
spec = importlib.util.spec_from_file_location(
    "velocity_analysis", ROOT / "references/end1-velocity/analyze.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
job = json.loads((RUNS["granite"] / "job.json").read_text())
views = {
    v["id"]: prepare_view(v, job["calibration"]["tee_distance_m"], sheet_geometry(job["calibration"]))
    for v in job["calibration"]["views"]
}
cases = [
    (17, "perspective_b", "red_1"),
    (30, "perspective_b", "red_1"),
    (45, "perspective_a", "red_1"),
    (57, "perspective_a", "red_1"),
    (114, "perspective_a", "red_1"),
    (165, "perspective_b", "yellow_1"),
    (180, "perspective_a", "yellow_1"),
    (210, "perspective_a", "yellow_1"),
    (279, "perspective_a", "yellow_1"),
    (285, "perspective_b", "red_2"),
    (311, "perspective_a", "red_2"),
    (336, "perspective_a", "red_2"),
    (358, "perspective_a", "red_2"),
]
montage = Image.new("RGB", (1050, len(cases) * 215), "#ffffff")
draw = ImageDraw.Draw(montage)
all_data = {name: [] for name in RUNS}
previous = {name: {} for name in RUNS}
cap = cv2.VideoCapture(job["source_path"])
cap.set(cv2.CAP_PROP_POS_MSEC, job["source_start"] * 1000)
step = round(cap.get(cv2.CAP_PROP_FPS) / job["fps"])
for frame in range(round(job["duration"] * job["fps"])):
    ok, bgr = cap.read()
    assert ok
    for _ in range(step - 1):
        cap.grab()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    seconds = job["source_start"] + frame / job["fps"]
    masks = {}
    anchors = {}
    active = [
        w
        for w in job["track_windows"]
        if w["kind"] == "perspective" and w["first_frame"] <= frame <= w["last_frame"]
    ]
    for name, folder in RUNS.items():
        path = folder / f"masks/{frame:06}.npz"
        if not path.exists():
            continue
        with np.load(path) as packed:
            for w in active:
                vid, stone = w["view_id"], w["stone_id"]
                view = views[vid]
                x0, y0, x1, y1 = view["crop"]
                crop = rgb[y0:y1, x0:x1]
                key = f"{vid}__{stone}"
                if key not in packed:
                    continue
                mask = (
                    np.unpackbits(packed[key], count=(y1 - y0) * (x1 - x0))
                    .reshape(y1 - y0, x1 - x0)
                    .astype(bool)
                )
                anchor = granite_anchor(mask, granite_evidence(crop, mask, w["color"]))
                masks[name, vid, stone] = mask
                anchors[name, vid, stone] = anchor
                row = dict(
                    frame=frame, source_seconds=seconds, stone_id=stone, view_id=vid, valid=anchor is not None
                )
                if anchor is None:
                    previous[name].pop(key, None)
                else:
                    row = add_raw_velocity(
                        previous[name].get(key),
                        {**row, **raw_mask_measurement(mask, view, anchor)},
                        frame,
                        seconds,
                    )
                    previous[name][key] = row
                all_data[name].append(row)
    for i, (f, vid, stone) in enumerate(cases):
        if f != frame:
            continue
        x0, y0, x1, y1 = views[vid]["crop"]
        crop = rgb[y0:y1, x0:x1]
        bounds = []
        for name in RUNS:
            mask = masks.get((name, vid, stone))
            if mask is not None:
                yy, xx = np.where(mask)
                if len(xx):
                    bounds.append([xx.min(), yy.min(), xx.max(), yy.max()])
        if not bounds:
            continue
        # Center on the new model's stone, retaining context around it.
        bound = bounds[-1]
        cx = (bound[0] + bound[2]) / 2
        cy = (bound[1] + bound[3]) / 2
        box = (
            max(0, int(cx - 72)),
            max(0, int(cy - 45)),
            min(crop.shape[1], int(cx + 72)),
            min(crop.shape[0], int(cy + 65)),
        )
        for j, name in enumerate(["source", *RUNS]):
            im = crop.copy()
            anchor = anchors.get((name, vid, stone))
            mask = masks.get((name, vid, stone))
            if mask is not None:
                im[mask] = (im[mask] * 0.5 + np.array([255, 50, 110]) * 0.5).astype("uint8")
            if anchor is not None:
                cv2.drawMarker(im, tuple(map(round, anchor)), (0, 255, 90), cv2.MARKER_CROSS, 12, 1)
            tile = Image.fromarray(im).crop(box)
            tile.thumbnail((340, 180))
            tile = tile.resize((int(tile.width * 1.55), int(tile.height * 1.55)))
            tile.thumbnail((340, 180))
            montage.paste(tile, (j * 350, i * 215 + 30))
            draw.text((j * 350 + 4, i * 215 + 4), f"{seconds:.2f} {vid} {stone}", fill="black")
            draw.text(
                (j * 350 + 4, i * 215 + 17),
                name + (" / body unverified" if name != "source" and anchor is None else ""),
                fill="black",
            )
    if frame % 90 == 0:
        print(frame, flush=True)
cap.release()
montage.save(OUT / "mask-review.jpg")
summary = {
    "method": (
        "Both saved runs use identical visible-granite validation and spatial bottom-envelope anchors. "
        "No time smoothing, speed gates or interpolation."
    ),
    "runs": {},
    "windows": [],
}
for name, samples in all_data.items():
    grouped = {}
    for s in samples:
        grouped.setdefault((s["view_id"], s["stone_id"]), []).append(s)
    summary["runs"][name] = {v + " / " + s: analysis.stats(values) for (v, s), values in grouped.items()}
for label, stone, vid, start, end, kind in analysis.WINDOWS:
    if "perspective" not in vid:
        continue
    sets = {
        n: [
            s
            for s in data
            if s["stone_id"] == stone and s["view_id"] == vid and start <= s["source_seconds"] < end
        ]
        for n, data in all_data.items()
    }
    common = set.intersection(
        *[{s["frame"] for s in ss if s.get("speed_mps") is not None} for ss in sets.values()]
    )
    summary["windows"].append(
        dict(
            label=label,
            kind=kind,
            source_seconds=[start, end],
            all_available={n: analysis.stats(s) for n, s in sets.items()},
            common_velocity_frames=len(common),
            common={n: analysis.stats([s for s in ss if s["frame"] in common]) for n, ss in sets.items()},
        )
    )
(OUT / "matched-body-summary.json").write_text(json.dumps(summary, indent=2))
(OUT / "matched-body-velocity.json").write_text(json.dumps(all_data))
print(json.dumps(summary, indent=2))

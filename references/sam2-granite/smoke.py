import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from hogline2.tracking_sam2_granite import Sam21Granite
from hogline2.tracking_stone import granite_anchor, granite_evidence

root = Path(__file__).resolve().parents[2]
out = root / "data/experiments/sam2-granite-20261108"
job = json.loads((root / "data/experiments/end1-velocity-20261108/common-job.json").read_text())
cases = [w for w in job["track_windows"] if w["kind"] == "perspective"]
cases.append(
    dict(
        stone_id="yellow_1",
        color="yellow",
        view_id="perspective_a",
        point=[417, 501],
        seed_source_seconds=471,
    )
)
views = {v["id"]: v for v in job["calibration"]["views"]}
model = Sam21Granite()
cap = cv2.VideoCapture(job["source_path"])
sheet = Image.new("RGB", (900, len(cases) * 225), "white")
draw = ImageDraw.Draw(sheet)
results = []
for f, case in enumerate(cases):
    model.begin_frame(f)
    seconds = case["seed_source_seconds"]
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, bgr = cap.read()
    assert ok
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    view = views[case["view_id"]]
    x0, y0, x1, y1 = view["crop"]
    crop = rgb[y0:y1, x0:x1]
    mask, quality = model.observe(view, case["stone_id"], case["color"], crop, case["point"])
    e = granite_evidence(crop, mask, case["color"])
    point = granite_anchor(mask, e)
    results.append(
        dict(
            seconds=seconds,
            view=view["id"],
            stone=case["stone_id"],
            area=int(mask.sum()),
            anchor=point,
            body_bbox=e["bbox"] if e else None,
        )
    )
    print(results[-1], flush=True)
    overlay = crop.copy()
    overlay[mask] = (overlay[mask] * 0.5 + np.array([255, 60, 120]) * 0.5).astype("uint8")
    if point:
        cv2.drawMarker(overlay, tuple(map(round, point)), (0, 255, 100), cv2.MARKER_CROSS, 12, 1)
    px, py = case["point"]
    box = [
        max(0, int(px - 65)),
        max(0, int(py - 45)),
        min(crop.shape[1], int(px + 65)),
        min(crop.shape[0], int(py + 65)),
    ]
    for col, img in enumerate([crop, overlay]):
        tile = Image.fromarray(img).crop(box)
        tile.thumbnail((430, 190))
        tile = tile.resize((round(tile.width * 1.6), round(tile.height * 1.6)))
        tile.thumbnail((430, 190))
        sheet.paste(tile, (col * 450, f * 225 + 27))
        draw.text(
            (col * 450 + 5, f * 225 + 5),
            f"{seconds:.2f} {view['id']} {case['stone_id']} " + ("Source" if col == 0 else str(point)),
            fill="black",
        )
    np.savez_compressed(out / f"smoke-{f:02}.npz", mask=mask)
sheet.save(out / "smoke.jpg")
(out / "smoke.json").write_text(json.dumps(results, indent=2))

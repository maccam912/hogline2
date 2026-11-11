"""Build a portable review page for the saved experiment."""

# Long lines below are literal HTML content.
# ruff: noqa: E501
import base64
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/sam2-granite-20261108"
run = Path(json.loads((OUT / "run.json").read_text())["path"])
summary = json.loads((OUT / "matched-body-summary.json").read_text())
alignment = json.loads((OUT / "camera-alignment.json").read_text())


def image(name):
    return (
        '<img alt="Source, previous SAM2 and granite-body SAM2 masks" src="data:image/jpeg;base64,'
        + base64.b64encode((OUT / name).read_bytes()).decode()
        + '">'
    )


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


rows = []
for w in summary["windows"]:
    a, b = w["all_available"]["previous"], w["all_available"]["granite"]
    rows.append(
        f"<tr><td>{html.escape(w['label'])}</td><td>{a['measured_frames']}/{a['expected_frames']}</td><td>{b['measured_frames']}/{b['expected_frames']}</td><td>{fmt(a['p95_speed'])}</td><td>{fmt(b['p95_speed'])}</td></tr>"
    )
folds = []
for f in alignment["baseline"]["leave_one_track_out"]:
    folds.append(
        f"<tr><td>{f['omitted']}</td><td>{f['held_out_pairs']}</td><td>{fmt(f['baseline_rms_m'])}</td><td>{fmt(f['held_out_rms_m'])}</td></tr>"
    )
page = (
    """<!doctype html><meta charset="utf-8"><title>SAM2 granite masks and camera alignment</title>
<style>body{font:17px/1.55 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;background:#f5f7f5;color:#172c29}h1,h2{line-height:1.2}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:10px;text-align:left;border-bottom:1px solid #cbd5d0}.notice{border-left:5px solid #c88120;padding:14px;background:#fff7e8}video{width:100%}small{color:#465e56}</style>
<h1>SAM2: measure the granite base</h1>
<p>End 1, source 378–498 seconds, 3 fps, three deliveries. The body-prompt comparison took 289 seconds on this Mac. Both models' saved masks are re-measured with the same bottom-envelope method. No temporal smoothing, velocity clipping or interpolation.</p>
<p>Prompts now include visible granite below the handle and scale with its image size. A handle-only mask cannot produce a perspective position. Uncertain or occluded bodies produce gaps. The green cross marks the supported lower mask envelope; it is a visible rim estimate, not a solved 3D stone center.</p>
<div class="notice"><strong>Identity remains a separate problem.</strong> This controlled diagnostic uses fixed seed windows and bypasses the normal identity checks. Red_1 is lost after an occlusion and later follows red_2. Thus body-validation counts are not tracking accuracy. Use the normal production replay to review physical identities.</div>
<h2>Source / previous SAM2 / body-prompt SAM2</h2>"""
    + image("mask-review.jpg")
    + """
<h2>Raw speed: mixed results</h2><p>Stationary red p95 speed improves from 0.201 to 0.023 m/s with equal measured coverage. Stationary yellow worsens from 0.099 to 0.199 m/s with equal coverage. Movement speed is not a jitter metric: delivery speeds below are shown for context. Do not infer a general accuracy improvement from lower speed alone.</p>
<table><tr><th>Window</th><th>Previous body frames</th><th>New body frames</th><th>Previous p95 m/s</th><th>New p95 m/s</th></tr>"""
    + "".join(rows)
    + """
</table><h2>Can shared stone views align the cameras?</h2>
<p>Yes, they constrain a candidate ice-plane mapping. Using 27 spaced correspondences across three stones, the existing camera disagreement is 1.274 m RMS. A robust perspective-homography fit reduces training disagreement to 0.429 m, and leave-one-stone-out testing gives 0.436 m RMS averaged across the three held-out tracks. The sheet's 34.747 m button spacing stays fixed; overhead views retain their ring calibration.</p>
<table><tr><th>Stone held out of fit</th><th>Pairs</th><th>Existing disagreement, m</th><th>Held-out fit, m</th></tr>"""
    + "".join(folds)
    + """
</table><p><strong>Candidate only; calibration revision 4 is unchanged.</strong> The residual exceeds the 0.30 m acceptance threshold. Opposite cameras see opposite bottom rims, so the matched silhouette points need not be identical physical points. Shape bias, feed timing and projection error remain mixed together.</p>
<p>A diagnostic offset scan at 3 fps gives the lowest held-out discrepancy, 0.386 m, pairing camera A at t with camera B at t−1/3 second. This still fails acceptance and does not establish a real feed delay. Full camera pose, focal length and radial distortion have not been independently recovered.</p>
<h2>Occlusion and identity failure in the ungated diagnostic</h2><p>Magenta is red_1, cyan red_2, yellow yellow_1. This is why body quality and identity quality are evaluated separately.</p>"""
    + image("identity-review.jpg")
    + """
<p>Reproduction: references/sam2-granite/analyze.py and calibrate.py. Machine-readable results are in data/experiments/sam2-granite-20261108.</p>"""
)
production = json.loads((OUT / "production-summary.json").read_text())
production_folder = ROOT / "data/recordings/eDJdwj0JqBY/tracking" / production["run_id"]
geometry = json.loads((production_folder / "geometry-report.json").read_text())
production_section = f"""<h2>Normal tracker: three deliveries kept distinct</h2>
<p>The normal workflow completed all 360 frames in {production["elapsed_seconds"]:.1f} seconds.
It detected red_1, yellow_1 and red_2, recovered red_1 after occlusion, and retained its
separate resting position through the second red delivery. All four cameras contributed
accepted observations for each identity. No saved same-camera red mask pair overlapped
by more than 50% of the smaller mask. Six review frames are shown below; this is a short
preview, not an entire-end identity guarantee.</p>
<p>The geometry audit uses {geometry["pairs"]} accepted shared-stone pairs. Existing discrepancy
is {geometry["baseline_pair_rms_m"]:.3f} m RMS; candidate training discrepancy is
{geometry["pair_rms_m"]:.3f} m and held-out discrepancy is {geometry["held_out_pair_rms_m"]:.3f} m.
This passes the internal consistency check. These observations were selected by the normal
identity and motion gates; the result does not establish absolute accuracy on independently
surveyed points. The fitted transforms are saved in the production geometry report, but
have not replaced calibration revision 4. Full 3D and lens distortion remain unresolved.</p>
<p><a href="/tracking?recording=eDJdwj0JqBY&amp;run={production["run_id"]}">Open normal tracking replay</a></p>
""" + image("production-review.jpg")
page = page.replace(
    "<h2>Occlusion and identity failure", production_section + "<h2>Occlusion and identity failure"
)
(OUT / "review.html").write_text(page)
(run / "comparison.html").write_text(page)
manifest = json.loads((run / "manifest.json").read_text())
manifest["granite_report"] = True
warning = "Ungated diagnostic: red_1 loses its target after occlusion and later duplicates red_2; not an identity-validated trajectory"
if warning not in manifest["limitations"]:
    manifest["limitations"].append(warning)
(run / "manifest.json").write_text(json.dumps(manifest, indent=2))

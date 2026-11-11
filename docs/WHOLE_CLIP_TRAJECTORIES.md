# Whole-clip trajectories and venue geometry

A second pass now refines completed SAM runs automatically. It also works on existing
`observations.jsonl` files, so the saved two-minute preview did not need another GPU run.
Raw masks, source timestamps, forward states and IDs remain unchanged.

The standard Rauch–Tung–Striebel recursion runs forward filtering and then backward smoothing,
using future observations to refine earlier positions and covariance. See
[Technion's optimal-smoothing notes](https://webee.technion.ac.il/people/shimkin/Estimation09/ch7_smoothing.pdf).
A constrained robust fit then traverses a gentle cubic spatial curve with a continuous,
non-increasing speed profile during each free-travel episode. This prevents a camera handoff
from producing a speed increase or an abrupt lateral turn. A gradual S is possible; sharp
curvature is bounded. This is an approximate motion prior, not a full curling physics simulator.

Contacts split episodes: positions remain continuous, but velocity can change at an impact.
When a resting stone is visible overhead, its displacement helps refine the approximate
contact time. Later overhead observations supply resting-house anchors and, when available,
contact positions. No velocity smoothing crosses an impact. Ambiguous input identities are
not solved merely by making their paths smooth.

The fitted covariance display combines backward-smoothed uncertainty with observation
mismatch. It is not a calibrated posterior confidence guarantee for the constrained fit.
A solver failure is flagged for review; it does not silently certify a physical trajectory.

Version `rts_constrained_glide_v2` ends moving fits at the last usable observation
and splits them when the forward tracker becomes lost. Unobserved trailing frames
retain their forward predicted/lost estimates. Stationary stones can still retain
their supported resting anchors. Explicitly rejected observations are excluded.
A converged glide fit with RMS observation error above 0.75 m or any residual above
2 m falls back to RTS and is marked for review; these are heuristic review thresholds.

The end 5 first-stone report exposed the missing-tail bug: the previous fit extended
from source 4362 s to the end of the clip (5163.67 s), creating a 648.66 m path and
pulling the early path onto the opposite side. Reprocessing end 5 with v2 limits that
fit to its observation span, ending at 4397.67 s, with a 39.62 m fitted path. Raw
observations and the SAM overlay are unchanged. This fixes the smoothing distortion;
it does not establish that all masks, later stone handling, or trajectories are correct.

## Image points

Perspective measurements use the bottom ice-contact point, not mask centroid. A handle-only
mask may use a separately inferred granite contact point with increased uncertainty. A stone
cut off at the bottom of a camera crop has no visible contact point and is rejected. Overhead
measurements use the stone's center; a cropped overhead mask is rejected to avoid centroid bias.

## Wausau length audit

Wausau now uses a fixed **standard-inferred** venue baseline: 45.720 m full length,
34.747 m button spacing and 4.750 m playing width. This follows the user's facility
research and standard geometry, not a site survey. See [Wausau geometry](WAUSAU_GEOMETRY.md).

The earlier unconstrained audit found unstable, bound-limited lengths. Those candidates
are retained for diagnosis and are not applied. New Wausau audits hold the selected
length fixed and assess camera consistency. Source-specific camera calibration,
parallax and timing remain uncertain even with a fixed physical baseline.

Venue dimensions can be loaded into another recording without copying its camera
landmarks. Each saved recording calibration and each run retain their own snapshots.

## Artifacts and UI

- `trajectory.json`: refined per-frame positions, velocities, phase, covariance and future-data flag.
- `trajectory-summary.json`: travel fits, overhead anchors, contact timing evidence and warnings.
- `geometry-report.json`: candidate geometry, landmark/pair residuals and length sensitivity.
- `manifest.json`: trajectory model and checksum of the immutable observations used as input.

The replay defaults to **Whole clip — future observations** when available. Switch to
**Forward filter** to compare. The speed graph shows both estimates and contact boundaries;
raw image-derived measurements can still be displayed on the map.

Install the additional CPU dependency with:

```sh
uv pip install --python .venv-sam/bin/python -r requirements-tracking.txt
```

Manually refine a completed run with:

```sh
.venv-sam/bin/python -m hogline2.tracking_smoothing data/recordings/ID/tracking/RUN_ID
```

# Hog-line aim review

Choose a completed tracking run and open **Review hog-line aim**, above the replay.
The analysis reuses that run's accepted delivery-camera observations. No SAM job or
paid analysis is started. Existing 3 fps runs have 3 fps pair rays; stepping the broom
reference by a source frame does not add tracking observations.

- **Orange:** a fixed local tangent at the delivery hog line, projected into the
  receiving overhead camera. This is a straight heading, not a curl prediction.
- **Magenta:** each consecutive forward-moving pair's unsmoothed direction. Use
  Previous/Next pair, the slider, or Play pairs to inspect the wiggle. The far-house
  reference stays frozen so the skip's target remains visible. The optional Follow
  replay control updates the pair ray with the existing tracking video instead.
- **Cyan:** click the broom's contact point on the ice, then Save broom mark. Move the
  reference time earlier if the skip has already lifted it. The dashed cyan line
  connects the hog crossing to the marked target when a fixed ray is available.
- **Amber band:** approximate fit sensitivity, not a calibrated accuracy interval.

Offsets are evaluated at the marked broom's longitudinal depth. Left/right is from
behind the delivery looking toward the receiving house, independent of the overhead
camera's rotation. A pair that has already passed that depth is not extrapolated
backward to claim an aim error. Broom marks also work for shots with pair rays but
insufficient coverage for a fixed ray.

The delivery panel shows original source frames, a calibrated hog line, and observed
anchor points. The white cross belongs to the previous frame; magenta to the current
frame; green identifies a sample in the fixed fit. Original full-size frames can be
opened separately. Pair playback waits for each delivery frame to decode, so it may
run slower than real time. Missing/rejected observations are never bridged by pair
rays. The slider indexes available pairs; it can skip a gap, with source times shown.

## Estimation and limits

`tracking_aim.py` uses only accepted, non-clipped, supported granite-bottom observations
from the delivery camera. Filtered state velocities, whole-clip trajectories, receiving
camera positions and the resting position do not enter the fit. Observations at or after
this stone's first possible contact are excluded.

A forward hog crossing must be bracketed by observations. A local quadratic x(y) fit
uses observations within 2 metres and 1.6 seconds of the crossing. Its derivative at
the hog line supplies the heading; time is interpolated between the bracketing samples.
At least five samples, two on either side, 1.5 metres of travel and gaps no larger than
0.75 seconds are required. Reversals, RMS residuals over 6 cm, or individual residuals
over 12 cm withhold the fixed ray. These are conservative review gates, not an accuracy
certification. The crossing refers to the tracker's approximate ice-plane anchor,
not a separately reconstructed three-dimensional stone center.

The band propagates a two-pixel localization model through the delivery-camera mapping
and inflates it for local residual scatter. It excludes correlated mask bias,
calibration error, lens distortion, physical stone height, source timing offsets and
manual broom uncertainty. A small band therefore does not prove good aim. The paired
rays deliberately preserve jitter. They require consecutive analyzed frames, a positive
forward displacement greater than 1 cm, and a time interval no longer than 0.75 seconds.

## Persistence and validation

`GET .../tracking/runs/RUN/aim` returns estimates, supporting frames, pair rays and saved
reviews. `PUT .../aim/STONE` saves or clears a target in the run's `aim-reviews.json`.
Reviews pin a SHA-256 of the method, source-bound manifest, observations and events;
changed evidence hides stale targets. Revision conflicts prevent silent overwrites by
another tab. Current recording calibration edits do not rewrite an older run's snapshot.

Tests cover both directions, perspective/overhead projection, target sign, tangent
recovery on a curved synthetic path, independence from future smoothed positions,
noise/gap/contact rejection, unsmoothed pair jitter, missing-frame behavior, persistence,
stale evidence, source changes and concurrent review conflicts. Real saved-run checks
and browser review are consistency checks, not surveyed aim-accuracy validation.

## First-version check — 2026-11-08

- Web suite: 66 passed, 5 skipped. JavaScript aim/map checks: 3 passed.
- Four completed saved ends produced 37 fixed rays and 1,446 consecutive pair rays.
  Other fixed estimates were withheld for insufficient local coverage. This count
  describes available estimates, not correct identities or accurate deliveries.
- Browser review exercised both delivery directions, pair stepping/playback, a
  sparse-coverage shot, broom save/reload/clear, and the displayed overlays. The
  temporary test broom mark was removed afterward.
- The local server was reloaded on port 8772. The independent whole-game runner
  continued processing ends 5 and 6 across the reload.
- [Preview](hog-line-aim-preview.jpg). Saved-run counts are retained in
  `data/experiments/hog-line-aim-20261108/saved-run-check.json`.

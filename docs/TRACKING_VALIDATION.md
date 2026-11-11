# SAM tracking validation — 2026-11-08

The first integration is available in the Stone tracking page. The original Hogline service,
recording and 60 fps end clips were left unchanged. Tracking runs in Hogline 2's isolated
Python/MLX process and writes separate sampled overlays and observation files.

## Scope

- Source `eDJdwj0JqBY`, end 1, original-source seconds 378–498, sampled at 3 fps.
- Source-specific manual ring/button/hack calibration, revision 2, four simultaneous panels.
- Both overhead panels need the saved 180° orientation to agree with the perspective views.
- Physical IDs reset each end. Approximate x/y metres share an ice plane with z=0;
  these are not measured 3D trajectories.
- Model SAM 2.1 BasePlus, MLX port revision `036f2d02a9069871fd2bf47109ff3b522cd222c4`.
- Source/model hashes and full job/calibration snapshots are recorded in the run manifest.

## Automated checks

28 tests passed in `.venv`; four additional OpenCV proposal tests passed in `.venv-sam`.
Ruff and JavaScript syntax checks passed. Coverage includes opposite camera orientations,
source/calibration validation, per-end ID limits, held occluded positions, stationary jitter,
possible contact and same-color uncertainty, long setup pauses at the hack, static paint
rejection, handle-only ground estimates and rejecting a lost moving stone's match to a
waiting stone back at the hack. These tests do not measure real-video accuracy.

## Known limits

The full-end option is implemented but full-end SAM tracking has not been validated. A
separate color/motion-only audit found 18 raw delivery candidates in this first end,
including false/miscolored candidates. That audit does not establish sixteen correct IDs.

Mask drift, missed delivery starts, motion extrapolation across camera gaps and ambiguous
contacts remain possible. Review both the overlay and uncertainty events. Landmark detection
is manual. There is no calibrated 3D reconstruction, automatic out-of-play adjudication,
manual-clearing classification or resume-from-SAM-memory checkpoint yet.

Earlier diagnostic runs are preserved under `tracking/diagnostics/`; they include unsuccessful
identity associations and should not be treated as validated tracking results.

## Final preview result

Run `b57e186ca40640968df6b1e4a01c67c2` completed 360 frames in 354.5 seconds,
with 4.15 GB peak Metal allocation.

It found red_1 at 383.67s, yellow_1 at 428.67s and red_2 at 471.33s.
The first two stones joined all four views. The possible contact at 447s was
reviewed against the video: red_1 retained its ID as it moved outside the house,
and yellow_1 stopped near the front of the house. The overlay at source 455s
visually agrees with those two stones.

The third delivery was detected, but its subsequent camera handoff was not
reliably recovered; it ends marked occluded. This is a known tracking failure,
not a successful three-stone reconstruction. The first two identities and
contact example are useful for review; this run does not pass full automation
acceptance. See TRACKING_GENERALIZATION.md for the broader requirements.

## Motion-filter revision — 2026-11-08

Run `2ae825c04e3547be9b9a892fdc622282` supersedes the original preview above.
It completed 360 frames (source 378–498s) in 376.5 seconds at 3 fps, using
4.17 GB peak Metal allocation. All three delivered stones joined all four
views. Red_2 now remains tracked from delivery to the house and its contact
with yellow_1 at 495.33s, rather than disappearing near source 474s.

Measurement-supported frames in the same two-minute interval:

| Stone | Original preview | Kalman/coverage revision | Longest new measurement gap |
|---|---:|---:|---:|
| red_1 | 327 | 329 | 2.00s |
| yellow_1 | 194 | 207 | 0.67s |
| red_2 | 9 | 77 | 1.00s |

These counts measure tracker availability, not ground-truth accuracy. Gaps
are explicitly predicted with covariance. Source 497s was visually checked:
red_1 remains outside the house, while red_2 and yellow_1 are correctly labelled
in the overhead view following their contact. The clip ends during the aftermath
of that contact; it does not validate the remainder of the end.

Validation: 39 web/core tests plus seven OpenCV proposal tests passed, along
with JavaScript orientation/event-grouping checks and Ruff. Browser checks
cover selecting the completed replay, map viewpoints, raw measurements and
video/map synchronization. See [motion estimator details](MOTION_TRACKING.md).

The user's old full-end run was stopped with permission and its partial
artifacts remain preserved. Two intermediate motion diagnostics are retained
under `tracking/diagnostics/`. No original source video or end clip was changed.


## Whole-clip refinement — 2026-11-08

The same 360-frame preview was refined from its saved observations without another GPU run.
All seven fitted travel segments succeeded and have zero positive speed increments between
contacts. The first red stone's main delivery has maximum curvature 0.021/m and a 0.51 m
RMS observation residual; smoothness does not establish ground-truth accuracy. Two resting
positions are anchored by overhead views. Contact hypotheses moved from 447.0 to 448.33s
and from 495.33 to 496.33s using displacement of the resting stone in overhead views.

Raw observations and masks are preserved. The replay defaults to the whole-clip estimate;
the forward comparison, speed graph and raw measurements were checked in the browser.
Perspective positions use the bottom contact point; cropped masks are rejected on future
SAM runs and cropped saved observations are excluded from refinement.

The sheet-length audit remains inconclusive: the candidate reaches its search bound and
leave-one-track-out estimates vary by 7.7 m. Wausau retains 34.747 m as nominal/unverified;
no fitted candidate was applied. This preview does not establish surveyed geometry or
performance on other recordings. See [whole-clip details](WHOLE_CLIP_TRAJECTORIES.md).

Validation: 40 web/core tests and 12 proposal/smoothing tests, plus JavaScript checks and Ruff.

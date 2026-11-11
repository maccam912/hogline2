# Independent camera calibration

Calibration revision 5 for recording `eDJdwj0JqBY` applies the shared-stone camera fit. It is used by new tracking runs, rather than remaining only in a diagnostic report. Earlier runs retain their original calibration snapshots.

Each perspective camera has its own full 3×3 image-to-ice homography (eight independent coefficients after normalization). There is no shared zoom, roll, tilt, viewing direction, or global pixels-per-metre factor. Each overhead camera retains its separate ring-derived mapping. Camera A/B orientation is part of the sheet coordinate convention, not an assertion that camera poses match. This models an approximately planar ice surface; camera intrinsics, physical 3D pose and radial distortion are not separately identified. See [OpenCV's planar camera geometry](https://docs.opencv.org/4.12.0/d9/dab/tutorial_homography.html).

The fit combines each perspective's marked button/ring/hack coordinates with simultaneous observations of the same stone across views. It uses only accepted observations with verified granite-bottom anchors in perspective views, and excludes clipped masks. The adopted sheet dimensions, including 34.747 m button spacing, are held fixed. There is no speed-curve target in the calibration objective.

The source run is `392ea127db464389a1b970e627289537`. It supplied 34 spaced shared-stone correspondences from three stones. Position disagreement was 0.617 m RMS before fitting, 0.076 m on fitted pairs and 0.095 m across leave-one-stone-out validation. Landmark reprojection RMS was 2.001 px. These are consistency checks, not surveyed absolute accuracy. Opposite visible stone rims, time offsets, mask jitter and lens effects remain potential biases.

## Persistence and invalidation

A view's optional `camera_calibration` stores the mapping, method, source run, held-out error and a SHA-256 fingerprint of its crop, landmarks, orientation and sheet dimensions. Projection, inverse projection, proposal association and measurement uncertainty use the saved mapping. Web API round trips preserve it. Editing any fingerprint input clears the stale fit when saving calibration; mismatched unsaved snapshots are rejected by the projection helper. Singular and non-finite mappings are rejected.

Applying this fit checked that the playing sheet does not cross the mapping's horizon and that forward/inverse projection round trips agree throughout a sheet grid. The original revision 4 calibration, applied revision 5, and fit evidence are saved under `data/experiments/camera-calibration-20261108/`. `references/sam2-granite/prepare_calibration.py` reproduces the preparation against the exact source calibration; it refuses a changed revision or different source setup.

## Verification

The web suite covers saved mapping use, forward/inverse agreement, API persistence, stale-fit invalidation and degenerate matrices. A synthetic multi-view test uses deliberately different scales, roll, skew and projective denominators; the independent fit recovers the shared sheet despite perturbed ring marks. The real first-two-minute SAM2 preview is rerun after applying the calibration, including all three deliveries and contacts.

`references/sam2-granite/compare_calibration.py` produces the before/after speed report from saved observations and trajectories. Raw camera speeds use consecutive accepted same-camera positions without velocity smoothing. The report distinguishes these from the green whole-clip trajectory estimates.

Calibration is tied to a static camera/mosaic setup. A later physical zoom, pan, camera move or lens change requires recalibration even if the panel crop stays the same; a fingerprint of editor values cannot detect unannounced changes inside the video. The current two-minute validation does not certify later ends or another recording.

## Completed revision 5 preview

Run `ac867e8ab9d74aa9a668d66527e139e4` completed all 360 frames in 425.63 seconds. It retained red_1, yellow_1 and red_2, with accepted measurements from all four cameras for each stone. Six source/mask frames were reviewed through the last contact; no saved same-camera red-mask pair overlapped by more than 50% of the smaller mask.

At source 394 s, delivery/receiving camera speeds changed from 2.585/0.888 m/s to 1.623/1.564 m/s. Across 12 matched simultaneous samples, mean absolute speed disagreement decreased from 1.705 to 0.100 m/s. There are now 29 paired camera-speed samples versus 12 previously. These are raw consecutive per-camera measurements, with unchanged speed-smoothing rules. The new run may produce different SAM masks because calibrated association changes where tracks initialize.

The applied mapping's position RMS on 56 spaced pairs in the new run is 0.086 m. An *additional* proposed refit is not recommended: its held-out RMS is 0.098 m and one held-out stone worsens beyond the relative-improvement guard. Revision 5 remains applied; the new run's `needs_review` audit refers to that additional candidate, not a failure to apply the saved calibration. The review notice distinguishes the two.

The before/after report and machine-readable measurements are `data/experiments/camera-calibration-20261108/comparison.html` and `comparison.json`; the report is also linked from the new replay. Validation: 53 web tests pass (five optional ML skips), plus 24 focused ML tests and the independent-camera synthetic fit test. Python lint and JavaScript syntax checks pass.

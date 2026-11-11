# SAM2 granite-body tracking and camera alignment

New UI runs use `sam2.1-mlx-granite` in the existing isolated MLX environment. This remains SAM2.1 with the same pinned model weights. SAM3.1 is retained as an experiment, not selected for new runs.

## Measurement changes

- Initial prompts include the colored handle and a visible neutral granite region beneath it. Negative points exclude the hand, nearby ice and adjacent objects. Prompt spacing follows the observed stone size.
- A handle-only continuation can trigger a fresh body prompt at most once per second. A rejected corrective mask restores the previous model state.
- Perspective measurements require granite inside the actual SAM mask. Color/edge heuristics guide prompts and validation; they do not paint a fabricated body into the mask.
- The position anchor is the spatial median of the lower mask envelope across its central body columns. Detached mask islands cannot move it. Unsupported, clipped or occluded bodies leave measurement gaps.
- Overhead measurements retain their mask centroids. Production identity association, motion gates, occlusion handling and calibrated camera handoffs remain active. Raw diagnostic runs deliberately bypass those identity gates.

The perspective anchor is an estimate of the visible bottom rim, not the true center of the running band or a fully reconstructed 3D point. Pixel quantization, shadows, partial occlusion and perspective remain sources of error. There is no temporal smoothing in the raw comparison.

## Controlled first-two-minute experiment

Recording `eDJdwj0JqBY`, end 1, source 378–498 s, 3 fps, 360 frames, three deliveries. New run `fcf90d0406cb46e88c918ad16b86411c` uses the same initial points and view windows as the earlier reinitialized SAM2 run `ff22e901162b4412878e6a3ad95ea57e`. It completed in 288.95 seconds, with 4.26 GB peak Metal allocation.

Both saved runs are re-measured with the **same** new body validation and bottom-envelope anchor, so the comparison does not confuse a changed position formula with a changed mask.

| Window | Previous body measurements | New body measurements | Previous p95 raw speed | New p95 raw speed |
| --- | ---: | ---: | ---: | ---: |
| First red delivery, B, 385–391 s | 3/18 | 18/18 | No consecutive valid pair | 2.70 m/s (moving) |
| Resting red, A, 414–442 s | 73/84 | 73/84 | 0.201 m/s | 0.023 m/s |
| Resting yellow, A, 454–491 s | 102/111 | 102/111 | 0.099 m/s | 0.199 m/s |

Visual review shows whole granite where previous masks followed just the yellow handle or the first deliverer's hand. Improvement is not uniform: yellow resting jitter worsens. Body-valid counts are heuristic measurement availability, not manually labeled accuracy.

The ungated diagnostic loses red_1 after occlusion near 442 s and later follows red_2 under both IDs. That is an identity failure, not a successful reacquisition. The normal production run is evaluated separately; the diagnostic must not be used as an identity-validated trajectory.

Artifacts:

- `data/experiments/sam2-granite-20261108/review.html`: portable visual report, also served as the raw run's `comparison.html`.
- `mask-review.jpg`: source / previous / new masks, with supported bottom anchors.
- `identity-review.jpg`: explicit ungated identity failure.
- `matched-body-summary.json`, `matched-body-velocity.json`: same-anchor measurements and window statistics.
- `references/sam2-granite/analyze.py`, `smoke.py`, `report.py`: reproducible analysis and visual checks.

## Multi-view camera refinement

The geometry audit fits perspective image-to-ice homographies against the marked sheet landmarks and simultaneous same-stone observations. It retains the overhead ring scale and fixes the adopted 34.747 m tee spacing. It does not shorten the sheet to make cameras agree.

Validation now actually refits without each stone and evaluates that omitted stone. Previously, fixed-length sensitivity alone could not establish camera accuracy. Candidate approval additionally requires held-out RMS below 0.30 m and no materially degraded held-out track.

For the controlled body run, 27 spaced correspondences from three stones give:

| Metric | Result |
| --- | ---: |
| Existing cross-camera RMS discrepancy | 1.274 m |
| Candidate fit discrepancy, training pairs | 0.429 m |
| Held-out discrepancy, RMS across three stone folds | 0.436 m |
| Candidate landmark reprojection RMS | 1.838 px |
| Best diagnostic B offset | −1/3 s |
| Held-out discrepancy with that offset | 0.386 m |

All held-out stones improve, but this still fails the 0.30 m threshold. **No candidate transforms or time offset have been applied. Calibration revision 4 remains in use.** The discrepancy is agreement between cameras, not independently measured position accuracy.

Opposite cameras see opposite visible rims; these are not identical physical points. Their bias, segmentation, lens distortion, feed offsets and camera geometry can be confounded. The offset scan only resolves 1/3-second steps and does not prove an actual feed delay. Full 3D camera pose, focal length and radial distortion have not been independently solved. More well-distributed, reliable shared positions and fixed sheet features are needed before promoting a calibration correction.

Reproduce with `PYTHONPATH=. .venv-sam/bin/python references/sam2-granite/calibrate.py`. Results are in `camera-alignment.json`; the run's `geometry-report.json` contains the zero-offset candidate.

## Validation

The web-environment suite passes: 49 tests, with four ML dependency skips. The isolated tracking environment passes all 24 focused body, projection, raw velocity and smoothing/calibration tests. JavaScript syntax and repository Python lint pass. Tests cover handle-only rejection, scale-aware prompts, detached mask pixels, painted ice, player-shaped masks, and held-out camera validation.

The web and ML environments use different Python ABIs; the web API suite runs in `.venv`, not by mixing its binary dependencies into `.venv-sam`.

## Normal workflow verification

The UI/API-created run `392ea127db464389a1b970e627289537` completed 360 frames in 399.69 seconds, peaking at 4.16 GB Metal allocation. It detected red_1, yellow_1 and red_2; each identity has accepted observations from all four cameras. It recovered red_1 after occlusion and retained that separate stone through the later red_2 delivery. Across all saved masks, no same-view red pair overlapped by more than 50% of the smaller mask. Six source/overlay frames were visually reviewed, including both red stones and the final contact. This is not an entire-end identity guarantee.

The normal run's camera candidate uses 34 accepted shared-stone pairs: existing RMS disagreement 0.617 m, candidate training RMS 0.076 m, held-out RMS 0.095 m, landmark RMS 2.001 px. Rejected observations are excluded. This passes the internal consistency test (`camera_consistent`). Its stronger result is specific to observations selected by the normal identity/motion checks; it is not evidence of 9.5 cm absolute accuracy. The complete fitted transforms are saved in the run's `geometry-report.json`. Calibration revision 4 remains unchanged because this audit does not yet validate the transform against independent control points or recover the different physical rim offsets. No full 3D or lens-distortion estimate is claimed.

`production-summary.json`, `production-review.jpg`, and `references/sam2-granite/production_review.py` preserve the separate normal-workflow evaluation.

## Apparent braking at the camera handoff

Inspection of the normal replay's red_1 curve found a calibration discontinuity around source 392.7–396 s (14.7–18 s into the replay). The tracked feature remains the visible granite bottom. At source 394 s, consecutive accepted observations from perspective B imply 2.585 m/s while perspective A implies 0.888 m/s. Reprojecting the exact same pixel anchors using the saved candidate homographies gives 1.623 and 1.609 m/s respectively. This strongly implicates different image-to-metres scales, not a real sudden braking event. The forward filter reacts to the camera change, and the constrained whole-clip fit spreads the disagreement into a steep deceleration. The candidate remains unapplied and these numbers are not ground-truth speed.

Reproduce with `PYTHONPATH=. .venv/bin/python references/sam2-granite/handoff.py`; output is `data/experiments/velocity-handoff-20261108/diagnostic.json`. The review UI now shows every stone's curve by default, annotates camera joins in blue, and seeks the video when a chart time is clicked. Lost tracks break the estimated-speed line. The displayed trajectory data itself has not been altered by this UI change.

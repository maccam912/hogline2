# End 1: reinitialized SAM2 versus SAM3.1

## Assessment

**Do not switch the default on this evidence.** SAM3.1 works on this Mac and improves several moving-stone segments, but it is not a consistently more stable, less fiddly replacement for fully raw velocity. The stationary-jitter materiality rule was not met. A separate spatial cleanup diagnostic fixes its mask-island spikes, but leaves ordinary perspective jitter similar and cannot fix a wrong-object initialization.

| Measurement (m/s unless stated) | SAM2 reinitialized | SAM3.1 |
|---|---:|---:|
| First delivery, maximum raw speed, 385–391s | 31.781 | 2.731 |
| Yellow approach, p95 frame-to-frame velocity change, 439–445s | 1.663 | 0.236 |
| Resting red, overhead p95 speed | 0.00867 | 0.00559 |
| Resting red, perspective p95 speed | 0.20223 | 0.20287 |
| Resting yellow, overhead p95 speed | 0.00543 | 0.00463 |
| Resting yellow, perspective p95 speed | 0.20486 | 16.87978 |
| Resting yellow, perspective p95 after largest-component diagnostic | 0.20486 | 0.20070 |
| Runtime for the two-minute preview | 258.8s (4.3 min) | 1332.7s (22.2 min) |

These are full-window statistics, with missing samples retained as gaps. Raw data and paired-common-frame statistics are in `summary.json`. SAM2 has 928 nonempty/projectable masks from 956 scheduled observations; SAM3.1 has 933. These are measurement counts, **not correct-object accuracy**. The resting red's perspective window has 75/85 measured frames for SAM2 and 76/85 for SAM3.1; both overhead resting windows and both yellow resting windows have full measurement coverage.

The visual checks explain why a single ranking would be misleading:

- On the first delivery, SAM2 drifts onto clothing and other nearby objects, while SAM3.1 retains the stone. The 31.8 m/s spike is a tracking failure, not just metre calibration.
- On the yellow approach, SAM2's body reinitialization creates a position discontinuity. SAM3.1 has much smaller velocity steps in that interval. Across the six selected travel windows, SAM3.1 has lower p95 velocity-step magnitude in five, with a small regression on the first red's unobstructed arrival.
- SAM3.1's perspective masks sometimes contain tiny remote islands. At 436s the red stone mask has one 630-pixel main component plus one disconnected pixel at crop coordinate (217,968). That pixel alone creates a 36.4 m/s spike under the lowest-pixel anchor. The yellow resting interval has repeated island-driven spikes, so its fully raw p95 reaches 16.88 m/s. Keeping only the largest component removes these extreme spikes without temporal smoothing, but normal resting perspective p95 remains about 0.20 m/s for both models.
- Under this calibration, one vertical pixel near the resting red stone is approximately 0.06696 metres, or 0.20088 m/s at 3 fps. This sensitivity explains much of the residual perspective noise. It is a calibration/anchor sensitivity estimate, not a surveyed distance.
- At the final overhead contact, both follow the yellow stone. The partly cropped entering `red_2` point prompt initializes SAM3.1 on a broom head instead, and it follows that broom throughout the seven sampled frames from 495.67–497.67s. SAM2 follows the actual red stone. Largest-component cleanup cannot repair this error. This is a failed initialization, not evidence of a later identity swap.

SAM2 made five full-body reinitialization attempts and accepted four. SAM3.1 used no ongoing reinitializations after its initial prompts. Its last captured peak MLX allocation was 13.21 GB (decimal); total process/system memory was not measured. SAM2's peak allocation was lost when its original final status was overwritten by the bookkeeping error, so it is intentionally unreported. SAM3.1 took about 5.15 times as long. No whole-end extrapolation was tested.

The next useful change is a consistent stone-position anchor and reliable initialization/occlusion recovery. Increasing SAM2 reset heuristics alone, or substituting SAM3.1 alone, does not solve the raw-velocity problem in this clip. SAM3.1 remains a viable experimental backend for a more targeted initialization test.

[Interactive raw comparison](../data/experiments/end1-velocity-20261108/comparison.html) · [Spatial-cleanup diagnostic](../data/experiments/end1-velocity-20261108/component-comparison.html) · [30.67-second source/model replay](../data/experiments/end1-velocity-20261108/comparison.mp4)

![Mask comparisons; white crosses mark perspective measurement anchors](../data/experiments/end1-velocity-20261108/mask-comparison.jpg)

![All seven final-contact frames](../data/experiments/end1-velocity-20261108/contact-inspection.jpg)

## Scope and method

This is the requested first-two-minute assessment, not a whole-end run. Both backends process source seconds 378–498 at 3 sampled frames/second (360 composite frames), with the same source, calibration revision 4, three physical identities (`red_1`, `yellow_1`, `red_2`), and 12 matched stone/camera windows. The source recording is `eDJdwj0JqBY`, SHA-256 `44590b63cab5d49e1c253fbac1224e84fe072016b8cdaae860c85844db8aabe2`.

Camera windows and initial handle positions come from the archived preview. Each window starts at its first previous observation and ends two samples after its last observation, capped at the preview boundary. This controls discovery and identity assignment so the comparison tests within-camera mask propagation. It does not validate autonomous delivery discovery, cross-camera identity association, all sixteen stones, or a whole end.

**SAM3.1 tracking** here means propagating selected stone masks with the model's temporal memory. It uses the community MLX port of Meta's Object Multiplex tracker, with a shared image backbone and joint multi-object propagation within each camera. It is not independent text detection on every frame, and it does not estimate physical velocity itself. [Meta's release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md) explain Object Multiplex. This experiment tests the local MLX conversion, not Meta's CUDA implementation.

Both start each camera/stone with the same handle location, two positive points and one negative point. Initial perspective masks can receive a granite-body point when a compact gray component is found below the selected handle. SAM2 additionally checks for handle-only drift on each frame: two consecutive body-cue failures trigger a fresh memory and body prompt, at most once per two seconds. A replacement must contain the body cue, overlap more than half the old mask, and remain below five times its area. Rejected candidates restore the old state. SAM3.1 uses its temporal memory after initialization, with new objects added as their matched windows begin.

These are two practical tracking configurations, not an isolated architecture comparison with identical corrective prompts. The SAM2 reinitialization cue is an experimental color/shape heuristic; it can miss failures or confuse neighboring objects. Its acceptance criteria affect which model mask is retained, but no motion gate or velocity filter is used afterward.

For each saved mask, the measurement is its centroid in an overhead view, or the median horizontal coordinate of its bottom 6% of pixels and its lowest vertical pixel in perspective. The existing homography projects this point to the ice plane. There is no inferred granite foot, mask expansion, Kalman filter, smoothing, speed clipping, or gap interpolation. Velocity is the difference between consecutive same-camera positions divided by their actual source-time difference. Pixel speed is available alongside metres/second to expose mask instability independently of calibration. Empty/unprojectable masks are recorded explicitly and break the velocity series.

The map prefers an overhead camera when available. It displays discrete raw positions, with trails broken at camera changes. The comparison graph always stays within one camera. Low apparent speed can still come from a stable wrong-object mask, so coverage and the original overlays must be reviewed together with jitter.

Preselected resting intervals are red at 414–442s and yellow at 454–491s, each in overhead A and perspective A. Six traveling intervals cover the deliveries and approaches. The exploratory materiality rule is at least 25% lower stationary p95 speed in both cameras without reduced coverage or identity drift. Travel velocity-step measurements also retain real deceleration/curl and do not represent pure error. Paired metrics use only frames where both runs have a valid velocity; full-window coverage is reported separately.

## Artifacts and review

- SAM2: `data/recordings/eDJdwj0JqBY/tracking/ff22e901162b4412878e6a3ad95ea57e/`.
- SAM3.1: `data/recordings/eDJdwj0JqBY/tracking/4c0e529d03244e32b4b98ae254667765/`.
- Common jobs, metrics and standalone graph: `data/experiments/end1-velocity-20261108/`.
- Thirteen prior end-1 testing runs were removed from the active run list and preserved under `data/backups/end1-tracking-before-velocity-20261108/`, with an archive manifest. Calibration and source video were retained.

Each new run contains original boolean masks, raw velocity samples including missing masks, per-frame observations, overlay video, prompt events, source/model provenance and explicit limitations. In the app, select either completed raw comparison, then select the stone, camera, and speed units. Both completed models appear together: blue SAM2, orange SAM3.1.

The normal **Track stones** button still runs the existing production provider. These experiments are intentionally separate from automatic delivery discovery. No whole-end processing was launched.

## Reproduction

Metal access is required. Existing `job.json` files are immutable inputs; copying each to a new run folder preserves the original results. Do not rerun in an existing result folder unless intentionally replacing it.

```sh
.venv-sam/bin/python -m hogline2.tracking_comparison PATH_TO_SAM2_JOB
.venv-sam3/bin/python -m hogline2.tracking_comparison PATH_TO_SAM31_JOB
PYTHONPATH=. .venv-sam/bin/python references/end1-velocity/components.py
.venv-sam/bin/python references/end1-velocity/analyze.py
.venv-sam/bin/python references/end1-velocity/review.py
.venv-sam/bin/python references/end1-velocity/replay.py
```

The two model runs must be sequential; both take the app's SAM worker lock. `references/end1-velocity/prepare.py` creates new run folders from the archived preview. Model installation and pinning are documented in [the initial Mac evaluation](SAM3_EVALUATION.md). SAM2 uses the existing base-plus MLX checkpoint; SAM3.1 uses `mlx-community/sam3.1-bf16` revision `a992e302ea9b0f03f41dfd93414a4fd0e818f65b`, whose stored tensors are actually F32.

The first executions completed inference but encountered a duplicate `raw_comparison` keyword while constructing their final manifests. That code is fixed. The recovery utility accepts only that exact error and verifies all 360 observation/video frames and every expected raw sample before reconstructing metadata. The original error logs remain for audit. Recorded wall time comes from the final progress log, excluding final serialization; unavailable peak allocation is null rather than guessed.

A separate, post-hoc diagnostic keeps only the largest connected component of each saved mask before calculating the same unsmoothed velocity. It does not reject frames or apply a temporal filter. This was added after finding that one disconnected pixel at source 436s caused a 36.4 m/s spike in SAM3.1. Results are saved separately as `component-velocity.json`, `component-summary.json`, and `component-comparison.html`; all original masks and primary raw measurements remain intact. This checks sensitivity to spatial measurement extraction and is not a replacement for the requested fully raw comparison.

## Validation

Both run artifacts contain all 360 H.264 overlay frames (120 seconds) and 956 scheduled mask observations. All 909 SAM2 and 916 SAM3.1 saved velocity vectors were independently recomputed from consecutive same-camera coordinates; missing observations never bridge a gap. The replay contains 92 frames (30.67 seconds). Original masks remain unchanged by the component diagnostic. The active end-1 run list contains just the two completed comparisons.

The standard test suite passed 47 tests with three dependency-related skips; two additional API checks passed, and 14 computer-vision/raw-measurement checks passed in the tracking environment. Map checks, JavaScript syntax checks and Ruff passed. Browser review verified both completed runs, report links, window selection and the plotted raw data.

# SAM tracking in Hogline 2

SAM tracking is now enabled on request from a recording's **Stone tracking** page.
Downloading and splitting ends remains independent; it never silently launches GPU jobs.
New runs use SAM 2.1 MLX with granite-body prompts and visible-bottom validation (`sam2.1-mlx-granite`). Calibration, physical identities, contact hypotheses,
and the top-down replay are outside the provider, so a later SAM 3.1 adapter does not need
to redefine the app's stone IDs or source timestamps.

## Run and review

1. Open the recording, then **Stone tracking**.
2. The shared layout fills in the panels, overhead landmarks, physical houses and
   orientations automatically. Optional alignment adjustments cover perspective zoom
   and aim; copied perspective landmarks are explicitly labeled as estimates.
   Set which house delivers the first end when needed. Subsequent ends alternate direction;
   the API also accepts an explicit delivery-view override.
3. Start with a one- or two-minute preview. Choose **Whole end** to track the complete end.
   The default is 3 sampled frames/second; use a higher rate when evaluating fast contacts.
4. Select a completed run to play its mask overlay and synchronized top-down replay.
   Hollow map markers are predicted or held estimates; crosses are lost tracks. They are not detections.

The Nov 6 recording has a manually inspected initial calibration saved in its own
`data/recordings/eDJdwj0JqBY/tracking/calibration.json`. The two overhead panels are rotated
180° relative to their corresponding perspectives. Its stable broadcast layout is reused
across sheets through `data/camera-layout.json`. The first complete four-view setup
establishes that default; an older catalog with exactly one complete setup is migrated
automatically. Existing per-recording calibrations are never overwritten. Matching-aspect
resolutions scale the crops and landmarks; different aspect ratios need a separate layout.
Perspective fitted transforms are discarded on reuse, and perspective landmarks remain
explicit starting estimates because zoom and aim can change. Automatic perspective
alignment is not implemented. Each recording and run retains its own calibration snapshot.

## Geometry and physical identities

Coordinates are approximate metres on the ice plane: x across the sheet, y from house A's
button toward B's button. Four outer-ring correspondences determine each homography; the
button is an independent consistency check. Nominal outer radius 1.829 m and tee spacing
34.747 m come from [World Curling's sheet measurements](https://worldcurling.org/wp-content/uploads/2020/10/Sheet-measurements.pdf).
This is an initial calibration, not a surveyed rink or full camera/3D reconstruction.
The small, foreshortened perspective rings make distant projection particularly uncertain.
New runs require the granite in the actual SAM mask before emitting a perspective position.
Handle-only or unsupported body masks produce gaps and may trigger a fresh body prompt.
The anchor follows the visible lower granite rim; it is not a solved 3D center. Older
`sam2.1-mlx` runs retain their separate inferred-foot estimate for reproducibility.
The [granite tracking evaluation](SAM2_GRANITE_TRACKING.md) includes matched raw-velocity
checks and held-out multi-view calibration candidates. Camera intrinsics and distortion remain unresolved.

Color, motion, and differences from the calibration frame provide cheap stone **proposals**.
Use an early frame with clear, preferably empty houses for calibration; static painted ice
is rejected as background. Long setup pauses at the hack do not expire a delivery candidate. A candidate must originate near the
selected delivery hack and move toward the ice. SAM then segments it. Accepted deliveries
receive red_1…red_8 or yellow_1…yellow_8, reset per end, in detected delivery order. No missing
stones are invented to make the count reach eight, and overhead detections cannot create
new physical IDs. Missed or false delivery proposals can still make numbering incorrect.

Each view holds its own SAM track linked to that physical ID. Handoffs require compatible
color, calibrated position, time and motion, with a uniqueness margin. Long-lost moving
tracks are not attached arbitrarily. Existing SAM tracks also pass a motion-consistency gate when recovering after a gap, to reject a mask that jumps back onto a waiting stone. Overhead observations take precedence near the house;
incompatible cameras produce disagreement events instead of averaged coordinates.

Resting positions use hysteresis to resist mask jitter. Occlusion retains the same physical
ID. Resting positions are held; moving positions are predicted through short gaps with growing uncertainty. A moving stone's swept path near another stone creates a
**possible_contact** event. Hidden contacted stones become uncertain rather than being
asserted stationary. Visible motion can then update the original ID; ambiguous same-color
contacts are flagged. This is not a physics solver or proof of a collision. Manually clearing
stones and out-of-play adjudication are not yet modeled.

## Isolated Mac runtime

Already installed in this checkout:

- `.venv-sam`: Python 3.12, MLX/Metal, NumPy, OpenCV and Safetensors, pinned by the reference requirements.
- `vendor/sam21-mlx`: https://github.com/eisneim/sam2.1_mlx at
  `036f2d02a9069871fd2bf47109ff3b522cd222c4`.
- `models/sam2.1/sam2.1_hiera_base_plus.safetensors`: about 323 MB, verified against its manifest.

Recreate when necessary:

```sh
uv venv .venv-sam --python 3.12
uv pip install --python .venv-sam/bin/python -r references/sam21-experiment/requirements-frozen.txt
uv pip install --python .venv-sam/bin/python -r requirements-tracking.txt
git clone https://github.com/eisneim/sam2.1_mlx vendor/sam21-mlx
git -C vendor/sam21-mlx checkout 036f2d02a9069871fd2bf47109ff3b522cd222c4
```

The web environment never imports MLX. A separate process loads weights strictly and verifies
both source and model checksums. A process lock prevents simultaneous SAM workers in this
catalog. Decoding streams one composite frame at a time. Image features are shared across
objects in the same view, then released; each object retains its own initial conditioning
and 16 recent memory slots, plus at most two recent corrective conditioning frames. This avoids the port's stock frame-only multi-object memory
collision and avoids loading a whole end's tensors into unified memory. MLX allocation
is capped at 12 GiB with a 512 MiB cache. It still uses the shared Mac GPU while running.

`hogline2/segmentation.py` is the streaming provider contract. The pinned port's private
memory methods are contained in `tracking_sam.py`; identity logic does not depend on them.
New providers implement begin_frame, track, has/rename/retire_track and memory rejection,
returning a mask and a backend-specific quality estimate. Those scores are not calibrated
accuracy probabilities. The original short-clip experiment remains in `references/`.

## Artifacts and limits

Every run lives in `data/recordings/ID/tracking/RUN_ID/`, with:

- `job.json`: exact source, end revision, calibration snapshot, interval, sample rate and optional manual seeds.
- `status.json` and `run.log`: durable progress; **Stop run** terminates a UI-started job.
- `overlay.mp4`: original composite with SAM masks and IDs, sampled at the chosen tracking rate.
- `observations.jsonl`: original-source seconds, physical stone state and per-view observations.
- `masks/*.npz`: losslessly packed boolean masks; unpack bits and reshape to the view crop's height/width.
- `events.json`: deliveries, camera joins, possible contacts and uncertainty.
- `manifest.json`: provider/weight revision, runtime, memory use and explicit limitations.

The original 60 fps end clips are unchanged. Overlay time zero maps to `source_start` in the
run manifest. A stopped run retains partial diagnostics; resuming SAM memory mid-run is not
yet implemented, so start a new run after stopping. Short previews are validation samples,
not evidence that all sixteen identities or an entire end have been correctly reconstructed.


See [motion tracking](MOTION_TRACKING.md) for Kalman filtering, measurement uncertainty,
full-sheet camera coverage, and the corrected selectable map orientation.

Whole-clip backward smoothing, constrained travel fits and reusable venue geometry are documented in [whole-clip trajectories](WHOLE_CLIP_TRAJECTORIES.md).

The [SAM3 / SAM3.1 Mac evaluation](SAM3_EVALUATION.md) includes local model comparisons,
timings, reproducible scripts, and a side-by-side replay. SAM2.1 remains the production backend.

The [first-two-minute raw velocity comparison](END1_RAW_VELOCITY_COMPARISON.md) tests reinitialized SAM2 against SAM3.1 on matched camera windows. Both completed runs are available in the review page, with per-camera raw speed, a windowed comparison report, and a separate spatial-cleanup diagnostic. SAM3.1 was not a clear drop-in improvement; the normal tracking provider remains SAM2.1.

[Independent camera calibration](CAMERA_CALIBRATION.md) is now applied in revision 5 for this recording. New runs use separate fitted perspective mappings; older runs keep their calibration snapshots. The run list includes calibration revisions so before/after results can be distinguished.

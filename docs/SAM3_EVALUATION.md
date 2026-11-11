# SAM3 / SAM3.1 Mac evaluation — 2026-11-08

SAM3 and SAM3.1 both run locally on this Mac through MLX. SAM3.1's neural memory
tracker followed the selected stones, including the yellow stone displaced during
the overhead contact. This evaluation does **not** establish a quality improvement
over freshly prompted SAM2.1. SAM3.1 memory propagation was about 5.5 times slower.
The production backend remains SAM2.1; this task added isolated experiments only.

The most useful finding concerns initialization and history. At source 471s, the
saved production SAM2 run has a 173-pixel handle-only mask on yellow_1. A fresh
SAM2 prompt on the same image recovers the granite body. SAM3 and SAM3.1 also
segment the body. This supports testing full-body reinitialization and drift
detection before replacing the tracker throughout the application. It does not
prove that all bad masks have the same cause.

![Same source pixels, saved mask versus fresh inference](../data/experiments/sam3-20261108/mask-failure-comparison.jpg)

[32-second side-by-side replay](../data/experiments/sam3-20261108/comparison.mp4)
shows original / fresh SAM2 / SAM3 detection with IoU association / SAM3.1 memory.
[Additional comparison snapshots](../data/experiments/sam3-20261108/comparison.jpg)
include SAM3.1 detection. The separate two-second small-stone test is in the
snapshots and saved per-backend overlays, not the 32-second replay.

| Tested path | Median seconds / sampled frame | Peak MLX allocation | Observed result |
|---|---:|---:|---|
| Current SAM2.1 MLX adapter, fresh point prompts | 0.334 | 3.97 GB | Full body on all 17 reviewed target frames; follows the contact |
| SAM3 8-bit text detection + simple IoU IDs | 0.845 | 3.64 GB | Full body in perspective views; misses the initial overhead target; false positives |
| SAM3.1 text detection + simple IoU IDs | 0.755 | 6.03 GB | Very similar to SAM3 in these samples; same initial overhead miss |
| SAM3.1 neural memory, selected target | 1.831 | 11.70 GB | Tracks the selected targets; overhead initialization requires direct points |

Hardware: Apple M5 Pro, 20 GPU cores, 64 GB unified memory. Each of the four
paths above processed 204 frames. Timings include mask materialization and
resizing, exclude model loading and overlay encoding, and use a fresh image
backbone every frame. GPU jobs were sequential. GB denotes decimal gigabytes;
MLX allocation is not total system/process memory. These are offline inference
rates at full resolution, not real-time 60 fps video processing.

The separately tested SAM3 neural-memory implementation did not work reliably:
on the sweepers clip it produced a 501-pixel initial mask, followed by 47 empty
masks. This result is after correcting its unusual point-coordinate convention:
SAM3's port expects embedding-grid coordinates, whereas SAM3.1's tracker wrapper
accepts resized-image pixels. The earlier smoke test used the wrong convention
and is not evidence against the model. Even the corrected result concerns this
specific MLX port/API, not official SAM3's general tracking ability.

SAM3.1 text-only detection and text-plus-box detection both failed to initialize
the stationary overhead target at source 491s. Direct positive/negative point
prompts succeeded, and the same target remained selected after the contact at
497s and 498.83s. This was a single tracked yellow stone; it does not validate
simultaneous identity preservation for every stone in a collision.

In the clear-yellow clip, SAM3.1 memory also tended to preserve a body-only mask
that omitted some of the handle. A nonempty mask alone is not proof of a complete
object silhouette. Text detection at the port's 0.15 threshold produced false
positives on unrelated objects; production use would need selection and rejection
rules. The detector CLI's IoU IDs are not the app's physical stone IDs and are
not the neural memory tracker.

The spot check used 17 manually reviewed frames with three interior body probes
per frame. Fresh SAM2 and SAM3.1 memory covered 51/51 probes; both text detectors
covered 48/51, missing the initial overhead stone. These are deliberately simple
coverage checks, not pixel-level IoU ground truth or an accuracy percentage. The
annotations also have approximate hand-marked boxes and notes for cropped and
partly occluded stones. The initial 15 annotations were made on unmasked source
crops before comparing model outputs; the two small-stone checks were added
after locating a production failure. No held-out recording was evaluated.

| Case | Original source interval | View | Purpose |
|---|---|---|---|
| clear | 3603–3611s | perspective A | Delivery moving toward camera, then exiting |
| sweepers | 1849–1857s | perspective A | Moving red stone among sweepers and brooms |
| opposite | 5031–5039s | perspective B | Opposite camera and a cropped exit |
| contact | 491–499s | overhead A | Stationary yellow stone displaced during contact |
| small-stone | 471–473s | perspective A | Reproduce a production handle-only mask |

All clips come from local recording `eDJdwj0JqBY`, SHA-256
`44590b63cab5d49e1c253fbac1224e84fe072016b8cdaae860c85844db8aabe2`, sampled at 6 fps.
The comparison tests segmentation and within-camera propagation. It does not
exercise automatic delivery discovery, camera handoffs, the Kalman estimator,
whole-end reconstruction, or all sixteen physical identities. The SAM2 baseline
uses the current adapter's two positive points and one negative point with fresh
memory, not the old production run's accumulated memory. SAM3.1 perspective
tracks start from text masks selected using the target point, so initialization
is not identical across models.

Installed in `.venv-sam3`: `mlx-vlm==0.7.12`, `mlx==0.32.4`. Existing web and SAM2
environments were not modified. Public model conversions:

- [SAM3 8-bit](https://huggingface.co/mlx-community/sam3-8bit), revision
  `bcd63fddcd2082f3841c7f25dc67a1bcb9e95dcf`, approximately 1.04 GB.
- [SAM3.1 MLX](https://huggingface.co/mlx-community/sam3.1-bf16), revision
  `a992e302ea9b0f03f41dfd93414a4fd0e818f65b`, approximately 3.49 GB.
  Despite the repository name `bf16`, all 1,961 tensors in this checkpoint's
  safetensors header are F32. This evaluation preserved its stored precision;
  it did not measure an FP16/BF16 optimization.

[Meta's SAM3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
describe the newer multiplex tracker. Its published NVIDIA results are not Mac
measurements. [SAMURAI](https://github.com/yangchris11/samurai) was researched as
an alternative but was not installed or benchmarked in this evaluation.

The recommended next implementation experiment is to detect handle-only masks
and reinitialize with full-body prompts, with a SAM3/SAM3.1 detector as a possible
source of those prompts. Keep the existing physical identities and cross-camera
logic. SAM3.1 is a usable experimental backend, but the current evidence does not
justify making it the default or claiming it fixes automatic stone tracking.

Reproduction, from the project root (Metal GPU access is required):

```sh
uv venv .venv-sam3 --python .venv-sam/bin/python
uv pip install --python .venv-sam3/bin/python -r references/sam3-experiment/requirements-frozen.txt

# Weights are already downloaded under models/. Exact revisions and SHA-256s
# are recorded in each model's hogline-manifest.json.
.venv-sam/bin/python references/sam3-experiment/benchmark.py --backend sam2
.venv-sam3/bin/python references/sam3-experiment/benchmark.py --backend sam3-detect
.venv-sam3/bin/python references/sam3-experiment/benchmark.py --backend sam31-detect
.venv-sam3/bin/python references/sam3-experiment/benchmark.py --backend sam31-memory

# Run the extra two-second case for each of the same four backends:
.venv-sam/bin/python references/sam3-experiment/benchmark.py --backend sam2 --case small-stone --duration 2
.venv-sam3/bin/python references/sam3-experiment/benchmark.py --backend sam3-detect --case small-stone --duration 2
.venv-sam3/bin/python references/sam3-experiment/benchmark.py --backend sam31-detect --case small-stone --duration 2
.venv-sam3/bin/python references/sam3-experiment/benchmark.py --backend sam31-memory --case small-stone --duration 2

# Corrected SAM3 neural-memory diagnostic:
.venv-sam3/bin/python references/sam3-experiment/benchmark.py --backend sam3-memory --case sweepers
.venv-sam/bin/python references/sam3-experiment/review.py
ffmpeg -i data/experiments/sam3-20261108/comparison-raw.mp4 -c:v libx264 -crf 21 -pix_fmt yuv420p -movflags +faststart data/experiments/sam3-20261108/comparison.mp4
```

Runs overwrite their named experiment outputs. The harness takes the existing
SAM worker lock so future experiments cannot overlap a UI-started tracking job.
Model weights load strictly; model hashes, source hash, runtime source hashes,
frozen dependencies, raw packed masks, per-frame timing, initialization method,
scores, IDs, and overlays are retained. No source recording or production job
was edited. `review-results.json` contains the spot checks; `manifest.json`
contains hardware/model/source provenance. All artifacts are under
`data/experiments/sam3-20261108/`.

Validation: Ruff and Python compilation pass for the two experiment scripts.
The comparison video decodes as H.264, 1440×540, 192 frames, 32 seconds. Saved
mask dimensions/counts and all 20 completed main-case outputs were checked.
The comparison images were visually inspected. This is an exploratory test,
not a labeled segmentation benchmark or whole-game acceptance test.

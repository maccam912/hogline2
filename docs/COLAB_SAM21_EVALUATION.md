# SAM2.1 on Colab L4 — November 10, 2026

SAM2.1 ran successfully in a [new Chrome Colab notebook](https://colab.research.google.com/drive/1hwJfj0PDiFQDE5WozNZawnhUFifbyF3G)
and a fresh NVIDIA L4 runtime. **Base-plus is a promising faster backend:** it took
12.09 seconds for the same 204 test frames, versus 66.64 seconds for the retained
Mac SAM2.1 baseline. Both CUDA sizes covered all 51 existing body probes, including
the overhead target missed by the earlier SAM3.1 text experiment. Large was slower
with no clear mask-quality improvement in these samples.

| Path | Frames | Inference seconds | Mean seconds/frame | Frames/second | Body probes |
|---|---:|---:|---:|---:|---:|
| Mac SAM2.1 MLX base-plus, prior run | 204 | 66.64 | 0.327 | 3.06 | 51/51 |
| L4 SAM3.1 text, prior run | 204 | 60.22 | 0.295 | 3.39 | 42/51 |
| L4 SAM2.1 base-plus, first pass | 204 | 14.11 | 0.069 | 14.46 | — |
| **L4 SAM2.1 base-plus, warm model** | **204** | **12.09** | **0.059** | **16.87** | **51/51** |
| L4 SAM2.1 large, first pass | 204 | 21.39 | 0.105 | 9.54 | — |
| L4 SAM2.1 large, warm model | 204 | 21.51 | 0.105 | 9.48 | 51/51 |

Warm base-plus throughput is about **5.5 times** the Mac SAM2.1 baseline and
**5.0 times** the preceding Colab SAM3.1 text run. Each pass starts fresh per-clip
tracking states and recomputes image features; warm refers to the loaded model
and initialized CUDA kernels, not reused masks. The first base-plus pass includes
about two seconds of extra initial CUDA work. All four passes completed: 816 masks.

Timing includes initial backbone computation, prompting, full propagation, CUDA
synchronization and boolean masks copied to the CPU. Official `init_state` warms
the first image backbone while loading the video; a timing wrapper measures that
backbone call separately so it is counted in inference, rather than hidden in
video loading. Video decode/loading, installation, downloads/uploads and mask-file
serialization are excluded. Transfers still take minutes in this manual browser
workflow, so this is not an end-to-end whole-game throughput claim.

## Masks and selection

![Identical source pixels, same review crops](../data/experiments/colab-sam21-20261110/comparison.jpg)

Both official CUDA SAM2.1 models segment the body in all 17 reviewed target frames
and follow the selected overhead yellow stone after contact. This improves on
the SAM3.1 **text-only** result on that overhead case. It does not establish better
silhouettes than fresh Mac SAM2.1: the inspected masks are broadly comparable.
Some initial small-stone masks omit the handle, despite covering the granite.
The larger checkpoint showed no clear visual advantage sufficient to justify its
roughly 1.8-times higher inference cost here.

These 51 probes are interior points on previously reviewed images, **not**
pixel-level IoU, an accuracy percentage, or a held-out dataset. They do not measure
complete boundaries, false positives, all contacts or long-term identity drift.
Both CUDA models produced nonempty masks on 188/204 frames; clear-travel and
sweepers clips include target exits, so this count alone is not an accuracy score.

CUDA SAM2.1 uses one manually selected target with the same two positive and one
negative initial points as the Mac comparison. There are no later corrective
prompts or identity reassociations. Its local object ID is 1; no production physical
IDs were changed. The prior SAM3.1 text path detects and tracks multiple visible
stones, so that comparison also differs in prompting and workload. Mac timings
are the retained November 8 experiment, not a rerun of today's granite-body provider.

## Runtime and provenance

- Fresh NVIDIA L4 runtime, Python 3.13.16, PyTorch 2.14.1+cu130, NumPy 2.4.6,
  OpenCV 4.14.0 and Decord 0.6.0.
- Official `facebookresearch/sam2` at
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`, installed through the browser notebook.
- BF16 autocast with TF32 allowed; `vos_optimized=False`, with no `torch.compile`.
- Default official postprocessing enabled, including stability-based multimask
  fallback, binarized prompt-frame memory, and filling holes up to 8 low-resolution
  pixels. The compiled connected-component CUDA kernel was installed and tested.
- Base-plus: 80,850,178 parameters, checkpoint 323,606,802 bytes, SHA-256
  `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`.
- Large: 224,446,642 parameters, checkpoint 898,083,611 bytes, SHA-256
  `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`.
- Peak allocated PyTorch tensor memory during these short clips: about 1.40 GB
  for base-plus and 2.04 GB for large. These are not total process/system memory
  or a memory requirement for full ends.

The official [repository and checkpoints](https://github.com/facebookresearch/sam2)
and [video predictor builder](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/build_sam.py)
define the tested implementation. The Mac adapter uses another implementation and
converted weights. Matching model families does not isolate hardware performance
from precision, preprocessing and implementation differences.

## Inputs, artifacts and reproduction

The experiment reuses the [SAM3.1 Colab input bundle](COLAB_SAM31_EVALUATION.md):
clear, sweepers, opposite, contact and small-stone clips from `eDJdwj0JqBY`, at
6 sampled frames/second. All 204 original RGB source crops were preserved using
lossless RGB H.264. The new runtime verified upload byte counts, archive SHA-256,
archive CRC and every clip SHA-256 before inference. It additionally verified that
**all 204 unscaled Decord RGB frames equal OpenCV's decoded frames exactly**.
The official SAM2 loader then applies its normal resize and normalization.

The notebook contains installation, official checkpoint downloads, a tested CUDA
postprocessing check, verified input loading, both model runs, an artifact download
cell and a final comparison table. It remains separate from the SAM3.1 notebook.
The benchmark's executed source copy is retained in `results/benchmark.py`.

Local artifacts: `data/experiments/colab-sam21-20261110/` contains the downloaded
result ZIP, 816 packed boolean masks, per-frame source timestamps and mask areas,
initial points, timing breakdowns, checkpoint hashes, dependency freeze, installed
runtime-source snapshots/hashes, `analysis.json`, `comparison.jpg` and a compact
`comparison-summary.jpg`. The input ZIP remains under the preceding experiment.

To rerun, upload that same input ZIP and `references/colab-sam21/benchmark.py` to
a fresh GPU Colab notebook, install the official repository and download the two
official SAM2.1 checkpoints into `/content/sam21-checkpoints/`. Extract the verified
input bundle to `/content/hogline-benchmark/`, then run the harness with `%run -i`.
Set `cuda_components_available=True` only after the kernel check passes; otherwise
the harness explicitly disables the optional hole filling. The live notebook
records the exact original uploaded-file hashes and commands.

After downloading/extracting results locally:

```sh
.venv-sam/bin/python references/colab-sam21/analyze.py
```

Validation checked all 816 mask counts, packed dimensions, areas and source
timestamps; recomputed body probes; and verified installed-source snapshot hashes.
Input and result ZIP CRC checks passed. The full comparison image was visually
inspected. Local harness files pass Ruff and Python compilation.

No production tracking code, existing runs, source videos or default provider
changed. A CUDA service/provider, whole-end resource management, automatic seeding,
camera handoffs and full-game identity validation remain outside this experiment.

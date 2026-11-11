# SAM3.1 on Colab L4 — November 10, 2026

The official CUDA implementation ran through the existing Chrome Colab tab on an
NVIDIA L4. It is much faster than the local SAM3.1 MLX port, but this experiment
does **not** establish better masks or a compelling replacement for SAM2.1.

| Path | Frames | Inference seconds | Mean seconds/frame | Frames/second |
|---|---:|---:|---:|---:|
| Mac SAM2.1 MLX, fresh point initialization | 204 | 66.64 | 0.327 | 3.06 |
| Mac SAM3.1 MLX memory | 204 | 355.06 | 1.740 | 0.57 |
| L4 SAM3.1 text, grounding/postprocess batch 1 | 204 | 60.22 | 0.295 | 3.39 |
| L4 SAM3.1 text, grounding/postprocess batch 4 | 204 | 63.17 | 0.310 | 3.23 |

Colab was about 5.9 times faster than the local SAM3.1 port and about 11% faster
in throughput than the SAM2.1 baseline. This is a cross-implementation comparison:
CUDA text inference detects/tracks multiple visible stones, while the Mac
baselines follow one selected stone. Initialization and precision also differ.

Timings include prompting, propagation, synchronization and materializing boolean
masks on the CPU. They exclude installation, model loading, video loading,
upload/download, encoding and disk serialization. The 42 MB input upload took
several minutes through this browser session. These numbers do not establish an
end-to-end whole-game speedup. Mac timings are retained November 8 runs, not
simultaneous reruns against the current granite-body production provider.

## Quality

Perspective masks include the granite body and look similar to fresh SAM2.1 in
the inspected crops. The small-stone case recovers the body that a historic
production run reduced to a handle-only mask. Fresh SAM2.1 also fixes that case.
Some initial masks omit the handle, so body coverage does not imply a full silhouette.

Both CUDA text configurations miss the selected small overhead stone at source
491 seconds, leaving the contact case without a selected target. Other detections
later in that clip are not silently reattached to the intended target. CUDA text
masks cover 42/51 existing interior review probes on 17 frames, versus 51/51 for
fresh Mac SAM2.1 and Mac SAM3.1 memory. All nine misses are in the overhead case.
This is a spot check, **not** pixel-level IoU, an accuracy percentage, or a held-out
benchmark. All 42 perspective probes are covered by both text configurations.

![Same pixels and selected target masks](../data/experiments/colab-sam31-20261110/comparison.jpg)

Text tracking retains initial selected IDs for 44/48 clear-travel frames, 35/48
sweepers frames, 48/48 opposite-camera frames and 12/12 small-stone frames. The
first two stones exit the crop; nonempty counts alone do not measure correctness.
IDs are local masklet IDs, not Hogline's physical stone identities. False positives,
occlusion recovery and whole-game identity preservation were not comprehensively evaluated.

## Runtime findings

- The saved notebook had a broken session monkey-patch. A fresh predictor retains
  official `_all_inference_states` bookkeeping. A narrow model-init shim removes
  unsupported disabled `offload_state_to_cpu` and selects OpenCV MP4 loading.
  Enabled state offloading raises an error.
- Point-only initialization produced a plausible seed-frame mask, then no target
  mask on subsequent frames in all five clips. Fresh sessions, real/complex RoPE
  and smaller grounding/postprocessing batches did not fix it. Its cause remains
  unresolved; this concerns the tested runtime, not SAM3.1's general capability.
  Failed point outputs remain separate from working text runs.
- A short text test with an explicit partial frame limit hit an empty-feature
  error. Full-sequence propagation works. Observed boundary behavior is consistent
  with [upstream issue #618](https://github.com/facebookresearch/sam3/issues/618).
- Default grounding batch 16 exhausted L4 memory in this session. Batches 1 and 4
  completed all clips; batch 1 was slightly faster. Diagnostic model/exception
  references retained GPU tensors, so recorded process peak allocations are not
  clean standalone model memory requirements.
- Colab used Python 3.13.16, PyTorch 2.14.1+cu130 and NumPy 2.4.6. SAM declares
  NumPy `<2`; installation used `--no-deps` and preserved Colab's NumPy/OpenCV
  stack. Successful text runs do not establish support for this dependency
  combination. Deployment should use a supported pinned environment.
- CUDA used BF16 autocast, FlashAttention 3 disabled, complex RoPE, and no
  `torch.compile`. Compilation was not benchmarked. The Mac conversion stores
  F32 tensors, as documented in the [prior evaluation](SAM3_EVALUATION.md).

## Reproduction and artifacts

The [prior evaluation](SAM3_EVALUATION.md) defines four eight-second clips and one
two-second clip at 6 fps: clear, sweepers, opposite, contact and small-stone.
`references/colab-sam31/prepare.py` encodes identical source crops as lossless RGB
H.264 MP4 and verifies **all 204 decoded frames** equal the original RGB pixels.
The ZIP includes baseline masks, annotations and an input manifest. Clip SHA-256s
and archive CRC were verified after upload.

Official source: `facebookresearch/sam3` at
`f1647e8d0c1d280991f4d1b7d974c2704537fe66`.
Checkpoint: `facebook/sam3.1`, revision
`daa63191845a41281374e725f4c9e51c7a824460`, file `sam3.1_multiplex.pt`, SHA-256
`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.
The [official builder](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/sam3/model_builder.py)
and selected installed source files are retained with the experiment.

Prepare locally:

```sh
.venv-sam/bin/python references/colab-sam31/prepare.py
```

Upload `data/experiments/colab-sam31-20261110/hogline-sam31-input.zip`,
`references/colab-sam31/setup_colab.py` and `run_text.py` to Colab. Install in a cell:

```python
%pip -q install --no-deps 'git+https://github.com/facebookresearch/sam3.git@f1647e8d0c1d280991f4d1b7d974c2704537fe66'
%pip -q install 'timm>=1.0.17' 'ftfy==6.1.1' 'iopath>=0.1.10' huggingface_hub einops pycocotools 'setuptools<81'
%run -i /content/setup_colab.py
%run -i /content/run_text.py
```

Download `/content/hogline-sam31-results.zip`. This experiment also preserved a
second text pass with batch 4. `analyze.py` reviews both retained text passes and
the point passes; `run_benchmark.py` reproduces the failed cold/warm point comparison.

Artifacts are under `data/experiments/colab-sam31-20261110/`: input/result ZIPs,
204-frame text outputs for both batch sizes, two 204-frame point passes, one extra
48-frame point diagnostic, losslessly packed target/all-object masks, per-case
timings, original source timestamps, dependency freeze, checkpoint/source hashes,
selected installed source files, `analysis.json` and `comparison.jpg`.

The [Chrome Colab notebook](https://colab.research.google.com/drive/17yiD_SsrzbX0arfUcnqmunZWCNknMZb9)
contains added experiment cells and a final table above the original example.
The original cells and saved outputs were preserved. Production runs, source
recordings and the default provider were not changed.

Validation: input pixel equality and ZIP CRC checks passed; output counts/shapes,
source/model hashes and probe counts were checked; the comparison image was
visually inspected. Ruff and Python compilation pass for the local harness.

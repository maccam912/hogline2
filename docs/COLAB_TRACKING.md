# Run tracking on a Colab GPU

On a recording's **Stone tracking** page, under **2. Run SAM tracking**:

1. Choose an end, preview/full-end interval and frame rate.
2. Set **Run tracking on → Google Colab GPU**. Base-plus is the default; large is optional.
3. Click **Prepare Colab run**. Download the notebook and video bundle when their links appear.
4. Open [Colab](https://colab.research.google.com/), use **File → Upload notebook** and select
   `hogline-colab.ipynb`. Choose a GPU in **Runtime → Change runtime type**.
5. Run the notebook cells in order. When prompted, upload the matching `colab-input.zip`.
   Wait for the upload and tracking to finish; the notebook downloads `colab-results.zip`.
6. Import that ZIP into its pending run on the tracking page. The overlay, identities,
   trajectories, camera/aim review and player review use the existing replay tools.

Start with a minute preview. Video is sampled locally at the selected frame rate and encoded
losslessly without resizing; uploading a full end can take longer than GPU inference.
Only the selected interval, reference image, frozen calibration/job and selected Python
tracking modules are bundled. Environment files, credentials, cookies and other recordings
are excluded. These files go to your Colab runtime when you upload them.

The normal **Process whole game** queue continues to use this Mac. The Colab option currently
exports one selected preview or end at a time. Preparing a bundle uses FFmpeg on the Mac;
SAM inference and trajectory refinement run on Colab. GPU availability, compute quotas and
runtime lifetime are managed by Colab; the app does not purchase compute or mount Drive.

Pending Colab runs survive a Hogline restart. Cancelling a pending run in Hogline prevents
import; stop execution separately in Colab. An import cannot replace a completed run.
If you change the recording or end boundaries, prepare a fresh run. Historical calibration
is frozen per run, just as it is for local tracking. Keep the notebook paired with its bundle;
each notebook verifies its exact input ZIP checksum. Result uploads are limited to 2 GiB
(4 GiB expanded); for larger jobs, reduce the interval or frame rate.

## Implementation and validation

`tracking_colab.py` exports the bundle and validates imports. `tracking_colab_worker.py`
runs the existing `tracking_runner` with the official CUDA SAM2.1 adapter. The pinned SAM2
revision is `2b90b9f5ceec907a1c18123530e92e794ad901a4`; both official checkpoint hashes
are verified. The adapter shares each camera/frame backbone pass between objects and retains
bounded forward memory instead of loading the entire video into GPU memory. It uses the same
granite prompts, component selection, visible-body validation, identity/camera logic and
trajectory refinement as local SAM2.1 tracking. Mask scores differ between providers and are
not a calibrated measure of pixel accuracy.

The input video carries an explicit original-source time origin. Results retain source
timestamps, original recording SHA-256, boundary revision and calibration revision.
Imports verify job identity, model/video hashes, frame count, timestamps, required artifacts
and per-file checksums. Only supported data artifacts are accepted; returned Python, HTML,
job files and path traversal are rejected. Original local source paths remain local.

Unit/integration tests cover matched RGB frame sampling, source offsets, notebook generation,
pending-run persistence, result import and rejection of stale, corrupt and mismatched results.
A real Colab L4 smoke test processed 24 frames across an eight-second source interval with
one seeded yellow stone, a perspective-camera handoff, saved masks and trajectory refinement;
the downloaded ZIP was imported through Chrome into the normal replay. This is a short
workflow validation, not whole-end identity/accuracy validation or an end-to-end speed claim.
See [the earlier matched-frame model comparison](COLAB_SAM21_EVALUATION.md) for inference timings.

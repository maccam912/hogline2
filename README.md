# Hogline 2

A fresh, independent curling-video workbench: paste a YouTube recording, download it,
detect physical ends, and export one intact original-layout clip per end. The end splitter does not score or identify players. A separate SAM tracking page now
provides calibrated camera views, per-end stone identities and a top-down replay.

## Run

Requires Python 3.12+, `uv`, `ffmpeg`/`ffprobe`, and `yt-dlp` on PATH.

```sh
cd ~/dev/hogline2
uv sync --group dev
cp .env.example .env # only if you do not already have a configured .env
# Set OPENAI_API_KEY in .env for automatic boundary detection.
uv run hogline2 serve --port 8772
```

Open http://127.0.0.1:8772. Paste an ended YouTube stream and choose **Download & split
into ends**. Jobs, source video, boundary evidence, editable cut points and exported
clips are in one screen. The app binds to localhost. One worker processes recordings
sequentially; the visual survey has two concurrent API requests. Refreshing preserves
progress. After a restart, interrupted work waits for an explicit Retry; completed
vision responses and the downloaded source are reused.

Detection uses the existing Hogline vision configuration pattern, currently
`gpt-6.1-terra`, through OpenAI Responses. This is paid image analysis, separate from
SAM. No API key is stored in manifests, API responses or tracked files. Configure
`HOGLINE2_YOUTUBE_BROWSER` for yt-dlp authentication when needed. Metadata and video
are downloaded without retaining signed media URLs or browser cookies in the catalog.

## What counts as an end

Start before the first stone's delivery motion (including standing/stick deliveries).
Finish after score determination and the clearing of played stones. Keep all throws,
pauses, discussion and measurement within that end. Do not cut at the final stone's
rest or delayed scoreboard posting. Pre-game warm-up shots and post-game practice are excluded using the sustained sequence
of competitive ends; ambiguous practice/game transitions are flagged for review.
There is no fixed end count. An optional expected
count generates a discrepancy warning; it never changes detection.

The detector surveys the full source at 20-second intervals, creates a visual phase
timeline, proposes physical ends and inspects two-second samples within 80 seconds
of each proposed boundary. It retains conservative boundary brackets plus two seconds.
Missed events, occlusion, abnormal layouts and partial recordings can still produce
wrong proposals. All automatic cuts remain reviewable proposals. Unsupported overlaps
stop export rather than being silently trimmed. Review **Watch boundary** against the
original player and save/export revised cuts. Use **Add end** and **Remove** to correct
the detected count. No complete-game accuracy claim is made without real validation.

Exports copy the original video/audio bitstreams, with no resizing, crop or frame-rate
change. They expand outwards to safe keyframes, so they can include a few extra seconds
around the requested interval. The immutable original remains available. The manifest
records requested cuts, actual source spans, clip video start PTS, hashes and revision.
The initial copied packet is matched by SHA-256 to the source to verify the actual cut
origin; FFmpeg can include an earlier keyframe block than the requested seek.
For a decoded clip PTS, `source_seconds = source_start + clip_pts - clip_video_start`.
Do not infer source time from a rounded player display. Editing cuts invalidates the
published clip list; earlier files remain on disk until deliberately cleaned up.

## Existing local video

```sh
uv run hogline2 import-local /absolute/path/recording.mp4 \
  --url 'https://www.youtube.com/watch?v=eDJdwj0JqBY' \
  --title 'Nov 6 — Koski vs Tillisch (Sheet 1)' --expected-ends 6
```

This creates an independent media copy and catalog record. Use the UI to detect ends.
CLI `hogline2 analyze ID` / `hogline2 export ID` work when the web server is stopped;
they share a process lock so two workers cannot mutate the catalog concurrently.

## SAM stone tracking

New recordings reuse the saved broadcast panels, overhead alignment and house directions.
Perspective zoom and aim start as labeled estimates; alignment adjustments are optional.
Open a recording's **Stone tracking** page to review the shared layout, run a short
preview or an entire end, and play the SAM overlay beside a synchronized top-down map.
Red/yellow IDs reset per end in detected delivery order and are shared across view-local
tracks. Hidden stationary stones retain their identity; possible contacts and ambiguous
matches are recorded for review. This is an initial tracking implementation, not a claim
of complete reconstruction accuracy or full 3D camera calibration.

SAM 2.1 MLX runs in its own process and environment. Downloading/splitting an end does not
automatically start SAM. The model processes streamed frames with bounded memory. Read
[SAM setup, calibration and limitations](docs/SAM_SETUP.md) for the provider contract and
artifacts; a later SAM provider can replace MLX without replacing physical-stone identities.

To try a GPU, choose **Run tracking on → Google Colab GPU**, then **Prepare Colab run**.
Download the generated notebook and sampled video bundle, run them in Colab, and import
the result ZIP into the same page. Both SAM2.1 base-plus and large are available.
See [the Colab tracking workflow](docs/COLAB_TRACKING.md) for steps and transfer limits.

The [Colab L4 SAM3.1 experiment](docs/COLAB_SAM31_EVALUATION.md) compares CUDA text
tracking with the existing Mac runs, including timings, masks and reproducible scripts.
The subsequent [SAM2.1 Colab comparison](docs/COLAB_SAM21_EVALUATION.md) tests the
official base-plus and large checkpoints in a fresh L4 notebook.

**Process whole game** prepares the trimmed recording and end clips, then queues
full-end SAM2 tracking with one end at a time by default (optionally two). Stop/resume retains completed
ends. Team lineups assign players by delivery order, with per-stone overrides,
velocity curves, ratings and accuracy notes. See [whole-game workflow](docs/WHOLE_GAME.md)
for camera prerequisites, recovery and review limitations.

`data/` and `.env` belong only to Hogline 2. There is no runtime connection to Hogline 1's
catalog, queue, service or environment. See [migration notes](docs/MIGRATION.md).

Completed runs also include [hog-line aim review](docs/HOG_LINE_AIM.md): a fixed local
heading, unsmoothed frame-pair rays over the far house, and saved manual broom targets.

## Verification

`uv run pytest` covers URL normalization, end interval invariants, enclosing keyframe
cuts, original-video export, API revision conflicts, stale clip withholding, request
deduplication, restart handling plus calibration, physical identity, occlusion and possible-contact behavior. Tests use generated media
and fake vision results; they do not spend API credits or run SAM.

The first full-recording run and its six exported clips are documented in [validation notes](docs/VALIDATION.md).

# Hogline 2

A local workbench for curling game video. Paste the link to a finished YouTube stream and it will:

1. download the recording and split it into ends,
2. track every stone down the sheet in each end,
3. let you step through the shots, mark the skip's broom and label the call, and
4. report line and weight per player.

## Run

Requires Python 3.12+, [`uv`](https://docs.astral.sh/uv/), `ffmpeg`/`ffprobe` and `yt-dlp` on PATH.

```sh
uv sync --group dev
cp .env.example .env    # set OPENAI_API_KEY for automatic end detection
uv run hogline2 serve   # http://127.0.0.1:8772
```

The app binds to localhost. Everything it produces lives under `data/` (ignored by git).

## Using it

The four tabs of a recording are the four steps, in order.

**Ends.** The recording is surveyed with a vision model, which proposes where each end starts
(the first delivery motion) and finishes (stones cleared). Check the cuts against the video, fix
them with **Set** / **Go**, then **Save cuts**. **Export clips** writes one MP4 per end without
re-encoding, so clips begin and finish on the nearest keyframes outside the cuts. Without an API
key, add the ends by hand. There is no fixed number of ends; the optional count you give when
adding a recording only produces a warning when detection disagrees.

**Tracking.** **Track all ends** runs [stonetrack](stonetrack/README.md) over each end. The first
run on a recording calibrates the cameras from the painted rings and lines (a few minutes); after
that an end takes well under a minute, and detections are cached so tracking an end again is
quick. Ends alternate direction; set which house end 1 is thrown from if the default is wrong.
The status column flags ends where fewer than 16 deliveries were found or where the stones the
tracker named do not match what the cameras show at the end of play.

**Review shots.** One row per delivery. The video opens just before the stone is thrown, on the
camera that looks at the target house. Pause when the skip's broom is down and click where it
touches the ice: that is the whole marking step, and it saves immediately. Label the call with
`1`–`4`, press `Enter` for the next unmarked shot. Tracked stones are outlined and named on the
video so wrong identities are easy to spot, and the two top-down maps show the stone's path, its
heading at release (dashed) and the broom.

**Report.** Per player: how far the release heading was from the broom, split by turn, and
hog-to-hog time by call. Team lineups (who throws which stones) are edited here, and every shot
can be downloaded as CSV.

What the numbers mean:

- **Line** is the stone's heading over its first six metres past the hog line, carried to the
  broom's depth, compared with the broom: centimetres to the thrower's left or right.
  *Narrow* means it started toward the side it then curled to; *wide* is the other side.
- **Turn** is the direction the stone curled away from that initial line.
- **Weight** is the hog-to-hog split time; tee-to-hog is shown per shot as well.

Tracking is not perfect. Identities are delivery order per colour, and stones hit while hidden
can be dropped or left behind. [stonetrack/README.md](stonetrack/README.md) has the measurements
and the known limits.

## Command line

```sh
uv run hogline2 import-local /path/recording.mp4 --title 'Nov 6, sheet 1' --expected-ends 6
uv run hogline2 analyze ID    # detect ends and export clips (web server stopped)
uv run hogline2 export ID
```

## Layout of the repository

```
hogline2/
  app.py            the FastAPI app; mounts the three areas below
  cli.py config.py store.py models.py media.py files.py
  ends/             download, end detection (vision model), cuts and clip export
  tracking/         camera layout, tracking jobs, the tracking process (runner.py)
  shots/            aim maths, broom marks, lineups, report
  static/           the web UI: plain ES modules, no build step
stonetrack/         the stone tracker, a self-contained package with its own README and tests
tests/              pytest for the app, node --test for the UI geometry
docs/               notes on end detection
```

Data on disk, per recording `data/recordings/<id>/`:

| Path | Contents |
| --- | --- |
| `source.mp4`, `end-NN-rR.mp4` | the download and the exported clips |
| `tracking/<run>/` | one tracked end: `job.json`, `status.json`, `stonetrack-result.json` |
| `tracking/broom-marks.json` | broom marks and calls, keyed `end:stone` |
| `tracking-jobs/<job>/` | progress and log of a tracking job |
| `team-lineups.json` | player names |
| `layout.json` | optional: camera panel layout for a mosaic other than the default |

`data/stonetrack/<id>/` is stonetrack's own cache (background, calibration, detections).
The catalog of recordings is `data/catalog.sqlite3`.

The camera panel layout defaults to `stonetrack/layouts/wausau.json`. A different broadcast
mosaic needs a `layout.json` in the same format, in `data/` or in one recording's folder.

## Tests

```sh
uv run pytest
uv run ruff check
node --test tests/
```

Tests use generated video and stubbed vision and tracking; they spend no API credits.

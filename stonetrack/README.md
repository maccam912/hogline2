# stonetrack

Curling stone tracking from the club's four-camera YouTube broadcasts. A fresh, self-contained
core: it shares no code or data with `hogline2/` and can be moved out of this folder as is.

Dependencies: Python 3.10+, `numpy`, `opencv-python-headless`, `scipy`, and `ffmpeg` on PATH
(`yt-dlp` only for `fetch`).

```sh
cd ~/dev/hogline2/stonetrack
uv run stonetrack init runs/GAME --source /path/to/source.mp4     # or: fetch runs/GAME --url <youtube url>
uv run stonetrack calibrate runs/GAME        # background + paint calibration (about 40 s)
uv run stonetrack refine runs/GAME           # stone-based alignment of the cameras (a few minutes)
uv run stonetrack detect runs/GAME --start 378 --end 1304        # one end; about 8x real time on 4 cores
uv run stonetrack track  runs/GAME --start 378 --end 1304 --first-delivery-end B
uv run stonetrack render runs/GAME --start 378 --end 1304        # top-down picture of the end
```

`detect` and `refine` are resumable (`--budget SECONDS` stops cleanly; rerun to continue).
Odd ends deliver from end B (the right-hand perspective panel's house), even ends from A.

## How it works

1. **Background.** A per-pixel median over ~125 frames spread through the game removes people
   and stones and leaves the paint.
2. **Paint calibration** (`markings.py`). Sub-pixel edges of the 2/4/6 ft rings, the centre line,
   the hog line and the two four-foot lines give each panel an ice-plane homography. No clicks:
   the first guess is the ring's own bounding box. Each sheet has its own cameras, so every
   recording is calibrated on its own.
3. **Where a stone is relative to the ice plane** (`geometry.py`, `crossview.py`). A stone stands
   11 cm tall, so its handle is not where its footprint is. Each camera carries its ground
   position and height; the perspective cameras get theirs from the homography itself, the
   overhead cameras from the parallax of stones at rest that a perspective camera sees at the
   same moment.
4. **Detection** (`detect.py`). Colour finds handle candidates, granite under or around the handle
   confirms them, and each detection becomes an ice position with a covariance.
5. **Tracking** (`track.py`). A Kalman tracker on the ice plane fuses all cameras. Stones are
   `moving` (coast on friction when hidden), at `rest` (stay put while hidden), or `out` (left the
   sheet, or their spot was seen empty). IDs are delivery order per colour: R1..R8, Y1..Y8.

## What was measured (sheet 1, 6 Nov 2026 recording `eDJdwj0JqBY`)

- **Lens distortion: negligible.** Painted lines bow by at most 0.8 px over a full panel. A free
  radial term moves the panel corners by 1 to 3 px and changes the fit residual by under 0.01 px.
  The cameras are long lenses (about 5100 px focal length); nothing is undistorted.
- **Paint fit: 0.1 to 0.6 px RMS** per feature in all four panels. The four-foot lines land at
  0.605 to 0.612 m from the ring centres without being told where they are.
- **Cameras found by the fit:** perspective A stands 7.0 m behind tee B at 4.4 m height,
  perspective B 6.5 m behind tee A at 3.9 m; the overhead cameras are 4.1 m and 4.4 m up.
- **Stones at rest, overhead vs perspective, 587 pairs:** lateral agreement 3 to 4 cm; along the
  sheet 8 to 9 cm robust spread (one pixel is 8 cm there). Fitting on the first half of the game
  and checking on the second leaves a 10 cm along-sheet bias, so the perspective cameras are a
  ~10 cm instrument in the far house. In the house, trust the overhead camera (about 1 cm).
- **Mid-sheet hand-off, 966 simultaneous pairs over 15 deliveries:** the two perspective cameras
  agree to 1 cm laterally (1.8 cm spread) and 8 cm along the sheet (11 cm spread), with no
  cross-camera fitting there.
- **End 1 tracked end to end:** 16 deliveries found, alternating R1, Y1 ... R8, Y8, hog-to-hog
  times 8.8 to 14.7 s. The eight stones visible in the overhead frame at the end of play match
  the tracker's positions and colours.

## Second sheet (sheet 4, 30 Oct 2026 recording `4NHlUXPYEBs`)

Run to check that nothing is specific to sheet 1. The sheets are **not** set up alike:

- Same mosaic layout, different cameras: houses sit about 30 px elsewhere, zoom differs, and the
  overhead camera over house A is mounted a half turn the other way. Paint cannot show that
  (rings look the same either way up), so `refine` decides each overhead camera's orientation by
  which one makes its stones agree with the perspective camera (194 pairs against 36 here).
- Paint fit 0.1 to 0.9 px RMS; stones at rest agree 1.5 to 3.5 cm laterally between cameras.
- End 1: 16 deliveries, alternating R1 ... Y8, hog-to-hog 8.6 to 14.9 s. The six stones visible
  from above at the end of play match the tracker, including one that was knocked while hidden
  and picked up again under its own ID.

## All twelve ends (both games)

`stonetrack detect RUN --ends proposed-ends.json` then `stonetrack ends RUN --ends proposed-ends.json`
runs a whole game; each end gets `end-NN-tracks.json` and a line in `ends-summary.json`.

- **Deliveries:** 16 found in every one of the 12 ends, colours alternating, in both delivery
  directions. 190 of 192 have a hog-to-hog split (8.0 to 21.2 s). Not checked against a human
  log; alternation and the cap of eight per colour are enforced by rule, so a missed stone
  paired with a false one would not show here. One gap between deliveries is 21 s, just over
  the 20 s minimum, and deserves a look.
- **Paths:** 189 of 192 deliveries have a clean hog-to-hog path. Scatter about a smooth curve is
  0.7 cm laterally (2.4 cm at the 90th percentile) and 4.9 cm along the sheet (9.3 cm). Mean
  deceleration is 0.078 m/s^2 (0.069 to 0.092), which is what curling ice should give.
- **Stones at rest at the end of an end: the weak part.** Compared with detections clustered
  independently at the settle moment, 6 of 91 steadily detected stones had no named stone within
  0.4 m, and 23 of the tracker's 113 in-play stones were not detected. Some of the 23 are hidden
  behind people; some are ghosts, clearest in crowded ends (one end has all 16 "in play" against
  11 detected). Stones hit while hidden, and stones lifted away by hand outside the overhead
  view, are the causes found so far.
- The settle moment is wrong when the house is cleared within seconds of the last stone.

## Known limits

- Two games (sheets 1 and 4) are all that has been run; the rules were tuned on these same ends.
- IDs are checked for count, order and final positions, not shot by shot against a human log.
- A stone's label is given 8 to 10 m down the sheet, once an overhead camera and a side camera
  have both seen it as a stone, or a side camera alone has held it steadily for three seconds.
  A delivery that never gets that far is not counted.
- People fool the detector now and then (a maroon jacket or a red hat reads as a handle or cap).
  The delivery rules above, team alternation and a 20 s minimum gap keep those out of the IDs
  on the two ends tested; a learned detector would be the sturdier fix if they leak through.
- Panel timing is not synchronised; an 8 cm mean disagreement at mid-sheet may be a ~40 ms lag.
- Aim (on-broom) and curl metrics are not built yet; `tracks-*.json` has the fused path per stone
  plus tee-to-hog and hog-to-hog splits.
- `fetch` (yt-dlp) is written but untested here.
- The panel layout (`layouts/wausau.json`) is the same on both sheets checked; a different mosaic
  needs a new layout file.

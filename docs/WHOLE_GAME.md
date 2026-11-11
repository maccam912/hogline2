# Whole-game processing and player review

Open a recording or its Stone tracking page and choose **Process whole game**.
The default is **one end at a time**, at 3 sampled frames/second; two concurrent
ends remain an option for higher GPU load. Each worker tracks all stones and camera views within
one full end, maintaining a single set of physical IDs for that end.

New recordings automatically reuse the shared broadcast layout, including panel crops,
overhead alignment, house directions and delivery-hack starting positions. The whole-game
button loads this layout directly; no visit to the landmark editor is required.
Perspective zoom and aim remain estimates until aligned for that recording; fitted
perspective transforms from another recording are never copied. Existing calibration
revisions stay intact, and camera mappings are frozen for each job.

The button reuses saved boundaries and verified clips, or runs boundary detection
and export if missing. Boundary detection uses the configured vision service;
SAM tracking runs locally. The job creates a trimmed game spanning the first
end's start through the last end's finish, exports each end, and queues full-end
SAM2 granite-body tracking. Lossless video cuts expand to safe keyframes; tracking
uses the exact saved intervals in original-source time, so export padding does
not shift the coordinates or velocities.

Progress survives page refreshes and web-server restarts. **Stop whole game** stops
the coordinator and its workers. **Resume unfinished ends** retains completed ends
and retries unfinished or failed ends from their beginning. Changed source or end
boundaries require a new job. Running jobs block conflicting preparation, boundary,
calibration, cleanup and manual tracking actions.

Enter four names for each colour under **Team lineups**. Names may be entered before
or after tracking: they assign detected deliveries 1–2, 3–4, 5–6 and 7–8 to the four
players when opening an end's review. A per-stone player override accommodates
substitutions and delivery-detection mistakes. Previously saved overrides remain.
Each stone keeps its velocity plot, identity review, manual 1–5 rating and accuracy
notes. The whole-game summary averages your ratings from identity-checked shots;
it is not an automatic assessment of skill or intended-shot accuracy.

Processing completion means the interval was processed, not that all sixteen
stones were identified correctly. Review the detected counts, masks, camera joins
and identities. A missed delivery can shift lineup assignments; correct those
assignments before using the player summary.

**Clear previous tracking runs** moves run and game-job directories into a dated
recovery archive under `data/backups/tracking-cleanup-*`. Calibration, original
video, end clips, lineups and review notes are retained. To restore an archive,
stop active tracking and move its `tracking/<run-id>` and `game-jobs/<job-id>`
directories back under the corresponding recording without overwriting existing
directories. Its `manifest.json` lists the moved paths.

## Validation on this Mac

On 2026-11-08, an isolated catalog processed two 20-second source windows with two
real MLX SAM2 workers, including trim/export and trajectory refinement, in about
52 seconds. Both jobs completed; the first found one delivery and the second
contained no detected delivery. This verifies orchestration and concurrent GPU
execution, not full-game identity accuracy or sustained full-end performance.
Results are in `data/experiments/whole-game-smoke/verification.json`. Browser checks
verified lineup assignment and saved shot ratings. Automated tests cover the
two-worker bound, alternating direction, full intervals, failed-end retries,
completed-end reuse, input validation and recoverable cleanup.

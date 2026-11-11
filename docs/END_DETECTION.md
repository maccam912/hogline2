# End detection: first full run (November 8, 2026)

The end-detection pipeline ran against the local Nov 6 Koski–Tillisch recording.
The source was independently APFS-cloned and SHA-256 verified; this run did not redownload YouTube.
One-step URL import/queue behavior is covered by tests; a fresh network download was not exercised.

The detector found six ends without receiving six as a target. Practice filtering moved the first
requested cut from 56 seconds to 378 seconds; the observed first competitive delivery begins
around 380–382 seconds. The final requested cut is 6582 seconds, after clearing. No additional
end was inferred from the remaining idle footage. Automated boundaries remain reviewable proposals.

| End | Requested source interval (seconds) | Exported source interval (seconds) | File |
| --- | --- | --- | --- |
| 1 | 378.000–1304.000 | 370.000–1305.000 | `end-01-r2.mp4` |
| 2 | 1362.000–2200.000 | 1355.000–2200.000 | `end-02-r2.mp4` |
| 3 | 2222.000–3166.000 | 2215.000–3170.000 | `end-03-r2.mp4` |
| 4 | 3238.000–4320.000 | 3230.000–4320.067 | `end-04-r2.mp4` |
| 5 | 4358.000–5164.000 | 4350.067–5165.067 | `end-05-r2.mp4` |
| 6 | 5240.000–6582.000 | 5230.067–6585.067 | `end-06-r2.mp4` |

All six MP4s preserve the original 1920×1080, 60 fps composite without re-encoding. This
particular local source has no audio stream. Other sources retain their audio when present.
Each exported initial video packet was matched by SHA-256 to the source to verify its time origin.
This caught and corrected an FFmpeg demuxer seek behavior that included an earlier GOP.
The manifest uses the actual retained origin, and exports keep conservative keyframe padding.

Boundary contact sheets were visually inspected. Browser playback of end 1 loaded at 1920×1080,
advanced normally, and reported no video error. This is a single-recording validation, not a broad end-detection accuracy benchmark.

Current clips and `manifest-r2.json` are under `data/recordings/eDJdwj0JqBY/`.
Revision 1 files remain as superseded artifacts; the UI only exposes revision 2.

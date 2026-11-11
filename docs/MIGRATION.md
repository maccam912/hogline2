# Migration scope

Hogline 2 is a fresh application, not a copy of the old database and worker stack.

Reused/adapted from Hogline:
- `hogline/adapters/sources.py`: strict YouTube ID parsing and optional browser-cookie options.
- `hogline/media/post_live.py`: download-first acquisition, duration verification and SHA-256 provenance.
- `hogline/perception/vision_config.py`: literal environment-file parsing, no shell evaluation.
- `hogline/perception/vision_calls.py`: Responses API image input and strict structured proposals.
- Durable progress, explicit retry, source-time evidence, non-destructive exports and revision pins.
- The standalone SAM 2.1 experiment as reference material and its verified weights.

Intentionally not migrated: old catalog, jobs, scoring/reconciliation, team/player identities,
biometric assets, budget history, old UI pages, fitted trajectories and fixed 1280×720
four-view calibration. There is no dependency on the original app at runtime.

The local Nov 6 Koski–Tillisch recording can be copied or APFS-cloned into this app's
own media folder. A clone is a separate file whose blocks are shared copy-on-write,
not a hard link or a reference to a file that Hogline 1 could remove. Source checksums
verify equivalence. Only the required vision/browser configuration is copied into the
new ignored `.env`; secrets never go into source code or documentation.

End detection currently uses sparse vision proposals and refined local visual windows.
It does not depend on SAM. Six ends is only a user-supplied expectation for the example,
not a target supplied to the model or an algorithmic constant.

API implementation reference checked during setup:
https://developers.openai.com/api/docs/guides/images-vision

# Wausau geometry baseline — November 8, 2026

Use standard geometry as the fixed working baseline for Wausau. This is a
**standard-inferred** venue profile, not a site survey or image-derived measurement.
The user's research describes an eight-sheet Olympic-size facility and found no
published evidence of a nonstandard sheet. Specific source links and the satellite
measurement figure were not included, so those facility claims are retained as
user-supplied evidence, not independently verified measurements.

The numeric baseline follows [World Curling's sheet diagram](https://worldcurling.org/wp-content/uploads/2020/10/Sheet-measurements.pdf):

| Quantity | Working value |
|---|---:|
| Backboard to backboard | 45.720 m (150 ft) |
| Button to button | 34.747 m (114 ft) |
| Playing width | 4.750 m (15 ft 7 in) |
| Tee to hog line | 6.401 m (21 ft) |
| Tee to hack line | 3.658 m (12 ft) |
| Tee to back line | 1.829 m (6 ft) |
| Outer house radius | 1.829 m (6 ft) |

The report's approximate 16 ft width is not sufficiently precise to replace the
standard playing width. Its final “12 m tee/hack” phrase is a unit typo: the table
and standard specify 12 **ft**. The hack-line reference is used; an unspecified
physical hack edge is not a separate control point. Rounded metric values follow
the official diagram, so doubling a radius can differ by a millimetre from a
separately rounded diameter.

`data/venues/wausau.json` stores these dimensions, evidence, and fixed button spacing.
**Load venue dimensions** copies only geometry/evidence into a recording's editor;
it preserves source-specific camera landmarks. Saving creates a new calibration
revision, and future jobs snapshot that revision.

The camera audit holds Wausau's selected button spacing fixed. It can still flag
camera residuals, but cannot shorten the rink to explain inconsistent observations.
It does not automatically apply fitted camera homographies. An earlier unconstrained
length audit is retained as historical evidence of camera ambiguity, not a club
measurement. The old two-minute preview coordinates remain unchanged because their
button spacing and house scale already match this baseline.

The map now uses the full 45.720 m extent instead of arbitrary padding around the
buttons, and draws hog, tee, back, and hack lines in either viewing direction.
Historical runs use their own calibration snapshots; missing geometry fields use
standard defaults. The active whole-end run retains its original snapshot and GPU
process. No source clips or SAM masks are changed by this update.

Validation: 42 web/core and 13 proposal/smoothing tests passed, along with map tests
and Ruff. Checks cover profile persistence/loading, invalid geometry, both map
orientations, full sheet extent and line placement, and preventing an inconsistent
camera fit from changing fixed venue length. Compatibility with the active worker's
already-imported geometry module was checked using saved preview observations.

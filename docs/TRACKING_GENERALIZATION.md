# Generalization requirements

The product target is automatic tracking across recordings, not a reconstruction tailored
to one match. The current integration is an experimental first version, not evidence that
that target has been reached.

## Already separated

- Source-specific panel crops, landmarks, orientation and delivery-end setup live in a
  versioned calibration artifact bound to the recording hash, never in tracking code.
- Delivery times, stone paths, color order and expected sixteen-stone counts are not
  supplied by the validation match. Delivery proposals come from video motion.
- Physical stone IDs and ice coordinates are independent of the SAM provider. Each end
  gets a fresh registry; every run records its inputs and model revision.
- The API accepts varying camera counts and panel crops. That flexibility is an interface
  capability, not proof that every camera arrangement works accurately.

## Required before claiming general automatic operation

1. Replace image-scale assumptions in color/motion proposal gates and SAM prompts with
   calibrated stone/house scale. Validate at multiple resolutions and camera zoom levels.
2. Detect panel layout, houses, buttons and hacks automatically, with confidence checks.
   Reusable venue/camera profiles must be checked against each recording; a matching title
   or resolution alone must never silently validate a calibration.
3. Detect camera cuts, movement, crop changes and time offsets. Invalidate or recompute
   geometry rather than carrying stale calibration through a changed view.
4. Evaluate on held-out recordings and both delivery directions. Include different lighting,
   crowded houses, occlusions, raised/moved brooms, same-color collisions and clearing.
5. Measure delivery precision/recall, identity switches, handoff coverage, position error
   and uncertainty coverage. Keep development clips separate from evaluation clips.
6. Add targeted correction controls for uncertain birth, handoff and contact decisions.
   Automation should request review of uncertain portions instead of inventing continuity.

Current real-video validation uses one recording and a short interval. Generic unit tests
and a successful preview do not establish robustness across matches. SAM is not fine-tuned
on this match; heuristics have been developed using it and therefore need held-out evaluation.

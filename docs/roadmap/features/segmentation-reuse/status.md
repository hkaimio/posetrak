```toml
name = "Segmentation Reuse (Time-Range-Scoped Bbox Source)"
status = "in_progress"
progress_pct = 35
description = """
Makes a Cutie segmentation a time-range-scoped, reusable bbox source independent of any \
specific detection run — so a redo with a different pose model, or an added hand-refinement \
pass, can reuse an existing human-reviewed segmentation instead of requiring a fresh one or a \
one-off script to copy detections across runs.
"""
categories = ["detection-pipeline", "data-model"]
target_release = "TBD"
last_updated = 2026-08-16
```

# Segmentation Reuse — Implementation Status

See [segmentation-reuse-design.md](segmentation-reuse-design.md) for the full motivating
misunderstanding, current-state trace, target vision, and sketch of changes.

## Current state

**2026-08-16: gap 1 (the schema mismatch) implemented and live-tested.** Found again from a
different angle: a live end-to-end tracking run surfaced that there was no way to segment a
capture *before* running detection at all, even though segmentation's whole point is to give the
pose extractor a better starting point than plain bbox detection. Traced to the same root cause
this doc already diagnosed — `seg_quality_runs.detection_run_id` being a `NOT NULL` 1:1 FK meant
the segmentation UI needed a `detection_runs` row to attach to, which meant running YOLO first,
purely for bookkeeping.

Implemented:
- **Schema** (session v41→v42): `seg_quality_runs` gains `shot_id`/`trial_id`/`time_start_s`/
  `time_end_s` (mirroring `detection_runs`' own columns), drops `detection_run_id`. Table rebuild
  (SQLite can't drop a NOT NULL column via ALTER), existing rows backfilled from their owning
  detection run. `python/tools/add_seg_quality.py`, `run_cutie_pose.py`, `apply_seg_weighting.py`,
  and `posetrak/db/trial_export.py` updated to match.
- **`CutieInitPanel`** now takes a capture id (`shot_id`) directly instead of a `detection_run_id`
  — cameras come from `capture_videos` directly, an existing segmentation is resolved by
  `shot_id`, and a detection run is created *lazily*, only when pose extraction is actually queued
  (`_resolve_or_create_detection_run`, which already existed for the "queue pose" flow and needed
  only to stop assuming a parent detection run existed to source `shot_id`/`sync_config_id` from).
- **Person list**: switched from `detection_track_assignments` (which only exists after a
  detection run's tracks have been manually assigned to persons) to the capture-level
  `capture_persons` (`list_persons`, config-improvements feature) as the primary source, unioned
  with any names already assigned from a prior detection run so already-migrated captures keep
  working unchanged. This is what the real prerequisite for segmentation turned out to be — a
  capture needs at least one *person defined*, not a detection run.
- `TrialPanel`'s "Create segmentation" button (`content_panels.py`) now gates on the capture
  having at least one person defined instead of at least one detection run.

**Not done, deliberately deferred** (per the scoping discussion when this was picked back up):
gap 2 (pipeline convergence — giving `RunDetectionDialog`/the YOLO flow a "use an existing
segmentation" choice) and gap 3 (auto-assignment — generating `detection_track_assignments`
automatically from segmentation labels instead of requiring the manual stitcher). Both remain
real, tracked gaps; today, running detection from an early-created segmentation still means using
`CutieInitPanel`'s own "Queue Pose" button (which already works, unaffected by any of this) and
still means a manual stitcher pass before `finalise_to_db` can run. See "Known issues" below.

Original three-gap framing (unchanged, still accurate for what's left): segmentation-driven pose
extraction being a wholly separate code path (`PoseWorker`/`PoseExtractionJob`) from YOLO-driven
detection (`DetectionPipeline`/`DetectionJob`), not a bbox-source option within one flow; and
auto-assignment from segmentation labels not being built even though `persons_ordered` already
gives a stable track-id → person-name mapping.

## Known issues / open questions

- **Gap 2 (pipeline convergence) and gap 3 (auto-assignment) remain unstarted** — see "Current
  state" above for exactly what still requires the old detect-first-then-stitch flow.
- An interactively-created segmentation's own `time_start_s`/`time_end_s` are set to `0.0`/a large
  sentinel (effectively "covers the whole capture") rather than the actually-segmented range —
  masks are stored per-frame regardless of range today, so this doesn't affect correctness, but it
  means the future containment check (design doc's target vision) can't yet distinguish "a short,
  trial-specific segmentation" from "one that happens to cover everything." Narrowing this to the
  real segmented range is real future work, not done here.
- Whether `keypoint_obs_quality` scoring stays keyed to the segmentation alone once one
  segmentation can feed multiple detection runs, or needs rekeying per consuming run.
- UI for the case where more than one existing segmentation covers a trial's time range.
- Migration path correctness for existing sessions — believed straightforward given today's 1:1
  relationship (confirmed via a live migration smoke test against a synthetic old-shape DB, not
  yet against a real multi-segmentation session), but not confirmed no session already has more
  than one `seg_quality_runs` row per capture in a way that would change post-migration semantics.

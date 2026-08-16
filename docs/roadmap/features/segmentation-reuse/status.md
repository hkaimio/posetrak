```toml
name = "Segmentation Reuse (Time-Range-Scoped Bbox Source)"
status = "in_progress"
progress_pct = 85
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

**2026-08-16: all three gaps implemented and unit-tested; not yet live-tested end to end.**

**Gap 1 (schema mismatch).** Found again from a different angle: a live end-to-end tracking run
surfaced that there was no way to segment a capture *before* running detection at all, even though
segmentation's whole point is to give the pose extractor a better starting point than plain bbox
detection. Traced to the same root cause this doc already diagnosed —
`seg_quality_runs.detection_run_id` being a `NOT NULL` 1:1 FK meant the segmentation UI needed a
`detection_runs` row to attach to, which meant running YOLO first, purely for bookkeeping.

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

**Gap 3 (auto-assignment), implemented before gap 2 since gap 2 depends on it.** `finalise_to_db`
already took `assignments: list[TrackAssignment]` as a plain parameter — fully decoupled from
*how* they were derived, so nothing about it needed to change. New `auto_assign_and_finalise()`
(`app/pose/finalise.py`) builds that list directly from `person_tracks` (`track_id N ->
persons_ordered[N-1]`, the same mask-label convention `pose_worker._bboxes_from_mask` already
uses to *write* detections) instead of requiring the interactive stitcher. `CutieInitPanel` gained
a "✓ Finalise" button next to "Queue Pose", enabled once a pose job has completed — the manual
stitcher is still there for correcting mistakes, but stops being a mandatory gate for the common
case.

One correctness wrinkle surfaced while wiring gap 2: `persons_ordered` (the ordinal→name mapping
baked into a segmentation's own mask labels) only existed in `CutieInitPanel`'s in-memory
`self._persons` — fine for that same panel finalising its own work, but a *different* caller
reusing an existing segmentation later would have had to assume today's `capture_persons` order
still matched whatever order was in effect when the masks were made. Fixed with a small schema
addition (session v42→v43): `seg_quality_runs.persons_json`, a JSON-encoded snapshot written at
mask-creation time (`_ensure_seg_run`), read back via `manage_person.persons_ordered_for_seg_run`
(falls back to today's `capture_persons` order for a segmentation created before this column
existed). `CutieInitPanel._on_finalise` itself now also reads this persisted value rather than the
live `self._persons`, for the same single-source-of-truth reason.

**Gap 2 (pipeline convergence).** `RunDetectionDialog` gained a "Bbox source" combo — "YOLO
detection" (default, unchanged) or any existing segmentation covering the capture, only shown at
all when at least one exists (so the common no-segmentation case looks exactly as before). Chosen
per design doc option (a) ("much less work"): invokes the same `PoseWorker`/`PoseExtractionJob`/
`JobQueueRunner` machinery `CutieInitPanel`'s own "Queue Pose" already uses, rather than teaching
`DetectionPipeline` a second bbox source. Per-camera frame ranges are resolved from the requested
time range via the same `SyncTable`-based global-time→frame conversion
`DetectionPipeline._frame_range` uses, so both bbox sources cover identical frames for a given
range. On queue completion, auto-finalises via gap 3 and reuses the existing `_on_finished` trial-
linking logic unchanged (it was already bbox-source-agnostic).

25 new tests across `test_finalise.py`, `test_cutie_init_panel.py`, `test_manage_person.py`
(new), `test_posetrak_db.py`, and `test_run_detection_dialog.py` (new — this dialog had zero prior
coverage). Full regression sweep clean (1759 passed, 19 skipped, 6 known pre-existing
deselections). **Not yet exercised against a real video/session** — the actual
`PoseWorker`/`JobQueueRunner` execution needs a real video file and model, so gap 2's job-queue
wiring is tested with `JobQueueRunner.start()` stubbed out and the finalise step driven directly
against seeded DB state, the same pattern already used for gap 3's `CutieInitPanel` tests. A live
run through the actual dialog is the natural next validation step.

## Known issues / open questions

- **Not yet live-tested end to end** — see "Current state" above.
- An interactively-created segmentation's own `time_start_s`/`time_end_s` are set to `0.0`/a large
  sentinel (effectively "covers the whole capture") rather than the actually-segmented range —
  masks are stored per-frame regardless of range today, so this doesn't affect correctness, but it
  means the containment check design doc's target vision describes can't yet distinguish "a short,
  trial-specific segmentation" from "one that happens to cover everything." `RunDetectionDialog`'s
  segmentation picker currently lists *every* segmentation for the capture unconditionally rather
  than filtering by range containment against the trial being detected — narrowing both of these
  to the real segmented range is real future work, not done here.
- No UI for the case where more than one existing segmentation covers a trial's time range beyond
  "list them all in the combo, most recent first" — no smarter disambiguation.
- Whether `keypoint_obs_quality` scoring stays keyed to the segmentation alone once one
  segmentation can feed multiple detection runs, or needs rekeying per consuming run.
- Migration path correctness for existing sessions — believed straightforward given today's 1:1
  relationship (confirmed via a live migration smoke test against a synthetic old-shape DB, not
  yet against a real multi-segmentation session), but not confirmed no session already has more
  than one `seg_quality_runs` row per capture in a way that would change post-migration semantics.
- `persons_json` is NULL for every segmentation created before 2026-08-16 — `persons_ordered_for_
  seg_run`'s capture_persons-order fallback is best-effort for those, not guaranteed correct if
  the order has since changed.

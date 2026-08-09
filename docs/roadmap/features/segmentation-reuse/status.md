+++
name = "Segmentation Reuse (Time-Range-Scoped Bbox Source)"
status = "proposal"
description = """
Makes a Cutie segmentation a time-range-scoped, reusable bbox source independent of any \
specific detection run — so a redo with a different pose model, or an added hand-refinement \
pass, can reuse an existing human-reviewed segmentation instead of requiring a fresh one or a \
one-off script to copy detections across runs.
"""
categories = ["detection-pipeline", "data-model"]
target_release = "TBD"
last_updated = 2026-08-06
+++

# Segmentation Reuse — Implementation Status

See [segmentation-reuse-design.md](segmentation-reuse-design.md) for the full motivating
misunderstanding, current-state trace, target vision, and sketch of changes.

## Current state

Sketch only, not implemented. **Explicitly postponed** in favor of
[hand-detection-refinement](../hand-detection-refinement/status.md) Phase 2/3, which took
priority when this gap was found. Written up after a real-data test of that feature needed to
reuse an existing segmentation across two detection runs and found no supported path — traced
to a genuine schema/architecture mismatch (`seg_quality_runs.detection_run_id` is a `NOT NULL`
FK, 1:1 with one detection run, not time-range-scoped), not a small missing option.

Three separate gaps identified, none small: the schema mismatch above; segmentation-driven pose
extraction being a wholly separate code path (`PoseWorker`/`PoseExtractionJob`) from YOLO-driven
detection (`DetectionPipeline`/`DetectionJob`), not a bbox-source option within one flow; and
auto-assignment from segmentation labels not being built even though `persons_ordered` already
gives a stable track-id → person-name mapping.

## Known issues / open questions

- Whether to converge the two pose-extraction pipelines into one (more correct, bigger lift) or
  keep them separate with a dispatch point in `RunDetectionDialog` (much less work) — not
  evaluated against each other yet.
- Whether `keypoint_obs_quality` scoring stays keyed to the segmentation alone once one
  segmentation can feed multiple detection runs, or needs rekeying per consuming run.
- UI for the case where more than one existing segmentation covers a trial's time range.
- Migration path correctness for existing sessions — believed straightforward given today's 1:1
  relationship, but not confirmed no session already has more than one `seg_quality_runs` row
  per capture in a way that would change post-migration semantics.

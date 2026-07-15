# First-release backlog

Captured 2026-07-15 (Harri). Six items identified as needed before a first
release — one expected to meaningfully improve tracking quality, the rest
tech debt / UX. Not sequenced here beyond the note on each; see individual
design docs (linked below) for anything already scoped in more detail.

## 1. Cross-person relative observations

The one item with high expected impact on tracking quality — assisted
movements, handshakes, two-person contact (ukemi throws are exactly this
case). Design already exists: Phase 5 of
`docs/roadmap/features/error-improvements/implementation-plan.md`. Its
prerequisites (Phases 1, 3, 4 — split pose/calib noise, within-person
`PAIR_DIFF` relative measurements, spatial cross-pairs) are already
implemented and tested; Phase 5 itself (cross-*person* `ANCHORED_RELATIVE`
mode, the `MultiPersonTracker` Gauss-Seidel orchestrator, contact-window
detection) is not yet built. Estimated 3-5 days per the doc. Currently
being planned (2026-07-15).

## 2. Update the keypoint-editing user guide

`docs/roadmap/features/keypoint-editing/` predates hand-detection-refinement
Idea 3. Needs updating for: the "Auto-redetect hands" toggle and what
auto-detect vs. keep-existing-state means in practice, the new
`STATUS_ORANGE` timeline color and what it signals, the "Revert hand
redetection" context-menu action, and how interpolation now interacts with
automated hand redetection (interpolate wrist/elbow → unselected fingers
get redetected, not geometrically interpolated).

## 3. Verify hand-detection-refinement's actual tracking impact, tune if needed

Idea 3 is implemented and confirmed *working* (no crashes, redetection
fires and writes sensible data), but its effect on tracking *quality* at
scale hasn't been measured yet — the hand-detection-refinement design doc's
own validation criteria (before/after garbage-detection comparison at trial
scale, hand-editing completion-time comparison) are still open. Also: the
700ms debounce window is an untuned guess, worth adjusting against how it
actually feels over extended real use.

## 4. CLI tools and MCP server: verify compatibility with new features, extend as needed

The MCP diagnostic server (`python/app/mcp/`) and CLI tooling predate the
Phase 2 multi-source `pose_observations` schema and Idea 3's `.refined`
sources / auto-detect toggle. Check whether `describe_config`-style tools
and any observation-quality tooling account for the new `source` values
correctly, and whether the upcoming Phase 5 multi-person config
(`cross_person_max_world_mm`, iteration count, contact windows) needs its
own MCP/CLI surfacing (the error-improvements plan already calls this out
for `describe_config`).

## 5. Surface "crisis debugging" patterns directly in the app

`docs/roadmap/features/tracking-crisis-debugging-log.md` documents several
real failure patterns diagnosed by hand, ad hoc, over multiple sessions
(swapped shoulder/wrist keypoints, near-origin/garbage edited keypoints,
covariance condition-number blowups, frequent PSD-eigensolver repairs,
edits that get silently gate-rejected). Each of these required a one-off
script or manual cross-referencing to find. Worth adding: timeline
warnings that flag a time range as likely problematic, with a short,
specific explanation of the probable root cause (not just "something's
wrong here") — turning this session's diagnostic experience into a
standing feature rather than one-off investigations repeated per trial.

## 6. Fix segmentation reusability properly (schema change)

Draft design already written:
`docs/roadmap/features/segmentation-reuse/segmentation-reuse-design.md`.
Today, a Cutie segmentation is permanently tied to the one detection run
it was created for (`seg_quality_runs.detection_run_id`), so it can't be
reused as the bbox source for a second detection run (e.g. a redo with a
different pose model, or adding a hand-refinement pass to an older
segmentation). The draft resolves this with a time-range-scoped
segmentation (own `time_start_s`/`time_end_s`, reusable by any trial it
fully contains) instead of a capture- or detection-run-level link — a
non-additive schema migration (PK rebuild), plus either a new UI dispatch
point or a real convergence of the YOLO and segmentation-driven pose
pipelines. Explicitly postponed until after hand-detection-refinement;
now back on the table.

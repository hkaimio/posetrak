# Keypoint Editing — Implementation Status

See:
- [keypoint-editing-brief.md](keypoint-editing-brief.md) — original problem statement and design brief
- [keypoint-editing-design.md](keypoint-editing-design.md) — full technical design (phases 1-10 core
  feature; "Improvements" section, phases 11-14, for the timeline view)
- [keypoint-editing-improvements-brief.md](keypoint-editing-improvements-brief.md) — improvement ideas
  brief that phases 11-14 (and the follow-up UX rounds below) implement
- [keypoint-editing-user-guide.md](keypoint-editing-user-guide.md) — user-facing guide

## Phase summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data layer: `pose_observation_edits` table, merge logic, C++ reader | ✅ Done |
| 2 | Crop grid widget (`PersonCropGridWidget`, JPEG-cache frame loading) | ✅ Done |
| 3 | Keypoint trail overlay (past/future positions, ghost interpolation) | ✅ Done |
| 4 | Mouse interaction: click-select, drag-to-move, DB write | ✅ Done |
| 5 | Keyboard shortcuts: `a`/`d`, nudge, `Space`, `Shift+A/D` range | ✅ Done |
| 6 | Union-bbox crop backfill for undetected frames | ✅ Done |
| 7 | Ghost-frame keypoint placement | ✅ Done |
| 8 | Multi-keypoint selection (rubber-band, named groups) | ✅ Done |
| 9 | Copy / paste keypoints | ✅ Done |
| 10 | Frame range selection + two-anchor linear interpolation | ✅ Done |
| 11 | Timeline status data plumbing (`timeline_status.py`) | ✅ Done |
| 12 | `KeypointTimelineWidget` skeleton (tree rows, camera tabs, playhead) | ✅ Done |
| 13 | Timeline selection & keyboard parity (rubber-band, keyframe toggle) | ✅ Done |
| 14 | Multi-keyframe interpolation (N anchors, not just the range's two ends) | ✅ Done |
| — | Partial tracking (checkpoint + resume from mid-trial) | ⬜ Designed, not implemented |
| — | Per-keypoint/camera/frame measurement noise override | ⬜ Designed, not implemented |
| — | Background wide-crop frame cache (person-cluster merging) | ✅ Done (see below) |

Phases 1-10 predate this status document; phases 11-14 and the follow-up UX rounds below were
implemented together as one continuous effort.

## Follow-up UX work on the timeline (not separately numbered in the design doc)

Building and then actually using phases 12-14 surfaced several UX problems that were fixed in four
rounds, plus one additional feature, all in `python/app/ui/keypoint_timeline_widget.py`:

- **Round 1** — removed the standalone scrub slider; the timeline became the trial's only clock.
  Added zoom (Ctrl+wheel, `−`/`+`/`Fit` buttons) and a horizontal pan scrollbar; made the timeline
  collapsible to a single ruler strip.
- **Round 2** — seeking and selecting were conflicting: a click on the row tree both scrubbed *and*
  mutated the selection, so scrubbing could silently collapse a multi-keypoint selection. Split them:
  a new always-visible `_RulerWidget` owns scrubbing exclusively; the row tree became selection-only
  (click clears, drag selects). Zoom now anchors on the playhead instead of the cursor position.
  Frame cells get a small gap once zoomed in past ~6 px/frame.
- **Round 3** — ruler ticks now show the same capture-global timestamp as the overlay row's
  current-time label (previously time-from-trial-start — two different clocks looked like a bug).
  The range-selection overlay snaps to whole-frame pixel bounds instead of a fractional ms span.
  Row clicks clear the selection *and* move the playhead (round 2 made them selection-only, which
  felt unresponsive). The collapse/expand arrow moved from the tab row onto the ruler, since the
  ruler is the part that stays visible while collapsed. Fixed a real ruler/canvas pixel-misalignment
  bug along the way: the ruler was mapping time↔pixel using the *canvas's* width, which silently
  diverges once the row tree's `QScrollArea` grows a vertical scrollbar.
- **Round 4 (bugfix)** — interpolating (or any other edit) was resetting the timeline's zoom on every
  single call, because the post-edit status refresh path called `set_time_range` unconditionally.
  Added `set_svid`, which updates only the active camera without resetting the view window.
- **Keypoint visibility** — an eye icon on each timeline row (leaf or group) hides/shows that
  keypoint. Hidden keypoints are excluded from crop-grid drawing/hit-testing/rubber-band/group-select
  and from timeline drag-select/keyframe-toggle — not just visually dimmed, a real interaction
  exclusion. State is session-local (`PersonCropGridWidget._hidden_kp_indices`), not persisted.
- **COCO133 "Face" group split** — the default skeleton only attaches markers to nose + ears, not
  eyes or the 68 detailed face landmarks, so "Select Face" pulling in ~70 points was more noise than
  help. Split into "Face" (nose + ears) and "Face (detail)" (eyes + landmarks), a non-overlapping
  partition of the original group.

## Background wide-crop frame cache (`app/pose/wide_crop_cache.py`)

Implements the design in *keypoint-editing-design.md*: `WideCropExtractWorker` walks each camera's
video sequentially per fixed-length epoch, computes each tracked person's padded crop window (with
gap-search past the epoch boundary for detection gaps longer than one epoch), clusters overlapping
windows with a merge-area guard so nearby people share one cached crop, and exposes results through
an in-memory index (`get_cluster_result`). `FrameCropCacheManager` reference-counts one worker per
`detection_run_id` across every open `PersonPanel`, so a second person's panel in the same trial
reuses the first panel's cache instead of rebuilding it. `PersonCropGridWidget._load_frame` checks
this cache before the Phase 6 in-memory crop / `frame_cache_entries` DB blob, and re-derives a tight,
per-frame display window from the wider cached crop when a real bbox exists at that frame (falling
back to showing the full generous crop on ghost/gap frames, where that generosity is the point).

Status-bar progress messaging for this worker was not wired up — the same gap already exists for the
Phase 6 `CropBackfillWorker` (`status_message` is only used for copy/paste/interpolation feedback
today, not backfill progress), so this isn't a regression, just an existing gap inherited by the new
worker.

## Test coverage

239 tests across the feature's test files as of this writing:

| File | Covers |
|---|---|
| `test_phase4.py` | Legacy crop-editor mouse interaction (separate widget, `app/pose/crop_editor.py`) |
| `test_phase5.py` | Keyboard shortcuts (nudge, outlier toggle) |
| `test_phase9.py` | Copy / paste |
| `test_phase10.py` | Frame range selection, two-anchor interpolation |
| `test_phase11.py` | `timeline_status.py`, `PoseModel.tree_groups`, `clear_single_keypoint_edit` |
| `test_phase12.py` | `KeypointTimelineWidget`/`_TimelineCanvas` skeleton: rows, tabs, status cells |
| `test_phase13.py` | Timeline rubber-band select, click-to-clear, Ctrl+click keyframe toggle |
| `test_phase14.py` | Multi-keyframe interpolation (the three brief validation cases + edge cases) |
| `test_timeline_ux_fixes.py` | All four UX rounds: seek/ruler, zoom, alignment, zoom-reset bugfix |
| `test_keypoint_visibility.py` | Eye icon, hidden-keypoint exclusion, `Face`/`Face (detail)` split |
| `test_trail.py`, `test_crop_editor.py` | Trail overlay, crop editor (adjacent, exercised for regressions) |
| `test_wide_crop_cache.py` | Wide-crop cache geometry: padding, overlap clustering + merge guard, per-track gap search, JPEG encode/clip |

Run with `pytest python/tests/app/test_phase{4,5,9,10,11,12,13,14}.py python/tests/app/test_timeline_ux_fixes.py python/tests/app/test_keypoint_visibility.py python/tests/app/test_wide_crop_cache.py`.

## Known limitations

- **Ruler/canvas alignment** relies on querying the row tree's `QScrollArea` vertical scrollbar width
  at runtime (`KeypointTimelineWidget._sync_ruler_margin`). The fix is structurally correct (both
  widgets now map time↔pixel from their own width) but hasn't been visually verified across
  platforms/Qt styles where scrollbar width might differ.
- **Timeline zoom/pan/collapse state and keypoint visibility are session-local** — nothing here is
  persisted to the database. Reopening the editor resets all of it.
- **Axis 2 (last tracking-run outlier verdict)** from the original timeline design — an overlay
  showing whether the tracker's own outlier rejection still distrusts an edited keypoint — was never
  assigned a phase number and is not implemented. Only axis 1 (edit state: green/yellow/blue/grey)
  exists today.
- **Partial tracking and per-keypoint measurement noise** are fully designed (see
  keypoint-editing-design.md) but not started. Neither blocks the current feature; they were
  lower-priority ideas from the same improvements brief.
- **Background wide-crop frame cache** (see above) has not yet had a manual UI validation pass
  (real multi-person trial, scrubbing through a long detection gap) — the algorithms are unit
  tested but the worker's QThread mechanics and on-screen framing haven't been eyeballed live, the
  same gap the original `CropBackfillWorker` has (no dedicated test file for it either).

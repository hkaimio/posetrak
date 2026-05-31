# Cutie Interactive Init Widget — Implementation Plan

**Created:** 2026-05-31
**Last updated:** 2026-05-31 (after Phase 3 completion)
**Design background:** `docs/segmentation-keypoint-weighting-design.md`

---

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | DB schema, FrameCache, VideoCanvas, scrubber | ✅ Done |
| 2 | SAM2 click interaction, PersonSelector | ✅ Done |
| 3 | CutieWorker, Track/Stop, mask persistence | ✅ Done |
| 4 | Multi-person edit, tracking range, frame range UI | — |
| 5 | Job queue, pose extraction integration | — |
| 6 | Drawing tool, person management, correction workflow | — |

---

## Motivation

Automatic YOLO+SAM2 initialisation produced identity switches and incorrect mask boundaries on
4 of 5 cameras in the first all-camera run.  Segmentation quality is gated on init quality, not
Cutie propagation quality.  The solution is a PySide6 interactive widget that lets the user
manually click to assign persons, track, stop at any frame, correct, and re-track.

---

## Architecture overview

```
CutieInitPanel (PySide6 panel)
├── VideoCanvas         — scaled frame display + mask overlay + mouse events
├── FrameCache          — temp-file LRU (OrderedDict), lazy OpenCV decode
├── PersonSelector      — coloured toggle buttons 1..N + keyboard shortcuts 1-9
├── ClickController     — SAM2 single-image prompts → labeled mask per-person
├── CutieWorker(QThread)— Cutie InferenceCore; mask_ready signal per frame
├── TrackingJobQueue    — ordered list of pending Track jobs (Phase 5)
└── MaskRangeBar        — coloured scrubber overlay showing tracked frame ranges (Phase 4)
```

---

## Database

### `seg_masks` table (migration 018, schema v19)

```sql
CREATE TABLE seg_masks (
    seg_quality_run_id TEXT    NOT NULL REFERENCES seg_quality_runs(id),
    shot_video_id      TEXT    NOT NULL REFERENCES capture_videos(id),
    frame_idx          INTEGER NOT NULL,
    mask_blob          BLOB    NOT NULL,   -- indexed PNG, uint8 labels 0=bg 1..N=person
    PRIMARY KEY (seg_quality_run_id, shot_video_id, frame_idx)
);
```

One `seg_quality_run` per interactive session per detection run.  Masks are ground truth; all
derived data (keypoint_obs_quality, ROI bboxes, detection_keypoints) can be recomputed.

Correction invalidation: `DELETE FROM seg_masks WHERE seg_quality_run_id=? AND shot_video_id=?
AND frame_idx > M` before re-seeding Cutie at frame M.

---

## Completed phases

### Phase 1 — Foundation (done)
DB schema, `FrameCache` (temp-file LRU, `max_dim` scaling), `VideoCanvas` (letterbox display,
DAVIS palette mask overlay, click signals), `CutieInitPanel` scrubber + camera selector,
"Segmentation…" entry point in `TrialPanel`.

### Phase 2 — SAM2 click interaction (done)
`ClickController` wrapping `ultralytics.SAM`; `PersonSelector` coloured toggle buttons with
keyboard shortcuts 1–9; left/right click → positive/negative SAM2 prompt; lazy encoder warm-up
via debounce timer.

Key finding: `ultralytics.SAM.predict()` with flat `points`/`labels` lists is more reliable
than the internal `prompt_inference` fast path (which requires patching `predictor.batch` after
`set_image()`).  200 ms per click on GPU is acceptable; optimise later if needed.

### Phase 3 — Cutie tracking (done)
`CutieWorker(QThread)` with `mask_ready(frame_idx, ndarray)` + `finished` + `error` signals.
Track Forward / Track Backward / Stop buttons.  Masks persisted to `seg_masks` DB table in
50-frame flush batches; committed on `finished`.

Key fixes during implementation:
- `output_prob_to_mask` returns int64; PNG encoder requires explicit `.astype(np.uint8)`.
- Cutie input frames must be scaled to `max_dim` to match FrameCache resolution; otherwise
  stored masks are at 4K and the shape equality check in `VideoCanvas` drops them silently.
- Backward pass pre-reads frames into memory: raw 4K arrays cause OOM (~40 GB for 1700 frames).
  Fix: scale to `max_dim` + JPEG-compress (~150 KB/frame) before storing in list.
- Hydra re-init error on second+ tracking pass: `GlobalHydra.instance().clear()` before
  `get_default_model()`.
- `FrameCache._lru` must be `OrderedDict`, not `dict` — `dict.popitem(last=False)` is not
  valid in Python 3.13.

---

## Phase 4 — Multi-person edit, tracking range, frame range UI

### 4a — Preserve other persons on click
**Problem:** clicking for person X on a frame that already has stored masks for persons Y and Z
clears Y and Z from the display (ClickController has no state for them).

**Fix:** when the user starts editing a frame that has a stored DB mask, load it as the
"base mask" in the ClickController.  Live SAM2 result for person X replaces only person X's
pixels in the combined mask; other persons' regions come from the base.

Implementation: `ClickController.set_base_mask(labeled_mask)` — stores the base, used as
starting point in `_run_predictions`.  Called automatically in `_ensure_encoded` if there is
a stored DB mask for the current frame.

### 4b — Tracking range markers (start / end frame)
Instead of "track from init frame to range boundary", add two draggable markers on the scrubber
or two spin boxes: **Track from** / **Track to**.  Clicking Track Forward/Backward tracks only
within `[track_from, init_frame]` (backward) and `[init_frame, track_to]` (forward).  Existing
masks outside this window are left untouched.

Implementation: two `QSpinBox` widgets in the tracking controls group, defaulting to
`track_first` / `track_last`.  Pass `first_frame` / `last_frame` overrides to `CutieWorker`.

### 4c — Frame range indicator (MaskRangeBar)
A thin coloured bar beneath the scrubber showing which frames have stored masks in the current
`seg_quality_run`.  Per-camera, queried once after load and after each tracking pass.  Coloured
segments per person (same DAVIS palette).

Implementation: custom `QWidget` (paint with `QPainter`); re-queried via
`SELECT MIN(frame_idx), MAX(frame_idx) FROM seg_masks WHERE ... GROUP BY ...` after each flush.

---

## Phase 5 — Job queue + pose extraction

### 5a — Tracking job queue
**Motivation:** setting up reference frames for all 5 cameras and then leaving the app to track
while away takes tens of minutes.  Users should be able to queue jobs rather than babysitting
each run.

**Design:**

Each job entry:
```python
@dataclass
class TrackingJob:
    camera_label: str
    shot_video_id: str
    video_path: str
    seg_quality_run_id: str
    init_frame: int
    init_mask: np.ndarray        # stored as PNG blob; loaded when job runs
    persons_ordered: list[str]
    first_frame: int             # tracking range start
    last_frame: int              # tracking range end
    direction: str               # "forward" | "backward"
    run_pose_after: bool         # trigger RTMPose batch on completion
```

Forward and backward are separate queue entries so each can be stopped independently without
cancelling the other.

UI: right-hand panel in the segmentation window (QListWidget).  Each entry shows
`camera · direction · init_frame → range`.  Status icon: ⏳ pending / ▶ running / ✓ done / ✗ failed.
"Add to queue" button (or automatic on person selection + range confirmation).
"Run all" / "Stop current" / "Remove" controls.

Queue executor: `JobQueueRunner(QObject)` in the main thread — owns a single `CutieWorker` and
starts the next job automatically on each `finished` signal.

Init masks for queued jobs are saved as PNG blobs in memory (or in a lightweight temp DB table)
so the queue survives a panel close/reopen.

### 5b — Pose extraction as a queued job
After segmentation, users can queue RTMPose extraction per camera.  This replaces the manual
`run_cutie_pose.py` step.

When `run_pose_after=True` on a job, on completion:
1. Create / reuse a `detection_run` (model=`cutie-sam2`) for this detection run ID.
2. For each tracked frame:
   - Load mask from `seg_masks`.
   - Compute tight bbox per person from mask pixels (store as `person_detections`).
   - Run RTMPose on the padded crop.
   - Write `detection_keypoints` + `keypoint_obs_quality`.
3. Emit `pose_done` signal.

**Caching tight bboxes:** compute and store during the segmentation pass (not separately).
Add a `seg_bboxes` table (or embed in `seg_masks` as a JSON sidecar) so the RTMPose pass
can skip mask → bbox recomputation.  Alternatively, skip caching and recompute from mask on
the fly — for 3 400 frames this is fast (< 1 s total).

Recommended: recompute on the fly.  Only cache if profiling shows it matters.

---

## Phase 6 — Drawing tool, person management, correction

### 6a — Drawing tool (brush add/erase)
A paint mode alongside SAM2 click mode.  Brush cursor; left-drag adds pixels for current
person; right-drag erases.  Adjustable brush size (slider or `[` / `]` keys).

Implementation: `VideoCanvas` mouse-move events + a `MaskEditor` that maintains a
per-frame "override layer" on top of the stored or SAM2 mask.  The override layer is a uint8
array (same shape as the mask) with a "dirty bit" per pixel.  Saved to `seg_masks` via the
same PNG blob path.

### 6b — Person management UI
Currently persons are loaded from `detection_track_assignments` for an existing detection run.
For a Cutie-only workflow (no prior YOLO detection) users need to define persons from scratch.

A small `PersonManagerDialog` in the tracking controls:
- Add person (name, auto-assigned colour).
- Remove person (prompts if they have masks).
- Reorder (changes label assignments).

Persons stored in the `seg_quality_run` as a `persons_ordered` TEXT (JSON array) column
(migration needed — add this column to `seg_quality_runs`).

### 6c — Correction workflow (invalidate-forward)
When the user scrubs to frame M that has stored masks and starts editing:
1. Pop up: "This frame has tracked masks. Edit here and re-track?"
2. On confirm: `DELETE FROM seg_masks WHERE ... AND frame_idx > M`.
3. User corrects mask, clicks Track Forward from M.

The "track from" marker (Phase 4b) automatically adjusts to `M` when correction is confirmed.

---

## Known limitations / future improvements

- **Backward pass chunk loading**: currently pre-reads ALL backward frames (JPEG-compressed,
  ~150 KB each) into memory before reversing.  For very long backward ranges (e.g., 3000 frames
  at 120 fps = ~450 MB) this is acceptable but not ideal.  Future: chunk the read into windows
  of ~1000 frames, process each chunk in reverse, then move to the next chunk backward.

- **SAM2 encoder caching**: currently `SAM.predict()` re-encodes the image for every click
  (~200 ms).  The `ultralytics` predictor's `set_image()` + `prompt_inference()` fast path
  is theoretically available but requires patching `predictor.batch` after `set_image()`.
  Worth revisiting once the full UX is stable.

- **Cutie checkpoint resume**: correcting frame M late in a clip re-runs Cutie from M to end.
  Optional: save Cutie `InferenceCore` state snapshots every N frames (pickled tensors) so
  re-tracking can resume from the nearest checkpoint rather than re-running from scratch.

---

## Packaging

`omegaconf`, `hydra-core`, `einops` added to the posetrak venv (required by Cutie, installed
directly since Cutie itself is accessed via `sys.path` from `../tests/Cutie`).

Production packaging plan (deferred):
```toml
[project.optional-dependencies]
cutie = [
    "sam2",
    "omegaconf>=2.3",
    "hydra-core>=1.3",
    "einops>=0.6",
    "cutie @ git+https://github.com/hkchengrex/Cutie.git@v1.0",
]
```

Weight download helper: `python/tools/download_cutie_weights.py` (not yet written).

SAM2 weight license: Meta custom research license — verify before production/commercial use.
Cutie weight license: check repo (likely CC BY-NC).

---

## Key design decisions

- **Segmentation-first**: interactive loop only runs Cutie; RTMPose runs as a separate
  queued post-step.
- **Invalidate-forward on correction**: one contiguous valid mask range per camera per session;
  no branching.
- **ClickController abstraction**: thin SAM2 wrapper; swappable for RITM or any model.
- **max_dim=1920 throughout**: FrameCache, CutieWorker input, SAM2 init masks all at the same
  resolution to avoid shape mismatches between stored masks and display frames.
- **Masks are ground truth**: all other derived tables (keypoint_obs_quality, person_detections,
  detection_keypoints) can be recomputed from seg_masks; never duplicate data unnecessarily.

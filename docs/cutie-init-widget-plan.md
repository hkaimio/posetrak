# Cutie Interactive Init Widget — Implementation Plan

**Date:** 2026-05-31
**Status:** Phase 1 in progress
**Design background:** `docs/segmentation-keypoint-weighting-design.md`

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
├── FrameCache          — temp-file LRU, lazy OpenCV decode, bg pre-fetch (Phase 3)
├── PersonSelector      — coloured buttons 1..N, currently selected person
├── CutieWorker(QThread)— Cutie InferenceCore; emits mask_ready per frame (Phase 3)
└── ClickController     — SAM2 single-image prompts → labeled mask (Phase 2)
```

Data flow per phase:

```
Phase 1: DB + scrubber
  VideoCanvas ← FrameCache ← cv2.VideoCapture(file_path)
  mask overlay ← seg_masks table (read only)

Phase 2: Click → SAM2 → mask
  click → ClickController.push_point() → SAM2 → labeled mask → VideoCanvas overlay

Phase 3: Cutie tracking
  Track/Stop buttons → CutieWorker QThread → mask_ready signal
  → VideoCanvas overlay + write to seg_masks table

Phase 4: Correction + pose extraction
  scrub into tracked range → load mask from DB → click to correct
  → invalidate seg_masks rows > frame_idx → re-seed CutieWorker → re-track
  Run Pose button → RTMPose batch over seg_masks → new detection_run
```

---

## Database

### `seg_masks` table (added in migration 018, schema v19)

```sql
CREATE TABLE seg_masks (
    seg_quality_run_id TEXT    NOT NULL REFERENCES seg_quality_runs(id),
    shot_video_id      TEXT    NOT NULL REFERENCES capture_videos(id),
    frame_idx          INTEGER NOT NULL,
    -- Indexed PNG: uint8 per pixel, label 0=background, 1..N=person.
    -- Label→person mapping from seg_quality_runs.persons_ordered JSON array.
    -- Typical compressed size at 1080p: 5–15 KB.
    mask_blob          BLOB    NOT NULL,
    PRIMARY KEY (seg_quality_run_id, shot_video_id, frame_idx)
);
```

At 3 400 frames × 5 cameras × ~10 KB ≈ 170 MB per run.  Masks are ground truth; all
derived data (keypoint_obs_quality, ROI bboxes, detection_keypoints) can be recomputed.

Correction invalidation: delete all rows where `frame_idx > M` for the given
`(seg_quality_run_id, shot_video_id)` before re-seeding Cutie at frame M.

---

## Packaging

Optional `[cutie]` extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
cutie = [
    "sam2",
    "cutie @ git+https://github.com/hkchengrex/Cutie.git@v1.0",
]
```

Install with: `pip install -e ".[cutie]"`

Model weights: `python/tools/download_cutie_weights.py` script (Phase 4).

All Cutie/SAM2 imports are guarded with `try/except ImportError` so the rest of the app
loads without the optional dependencies.

---

## Reuse from Cutie interactive_demo.py

The Cutie demo (`/home/harri/projects/tests/Cutie/gui/`) is already PySide6:

| Demo file | Reuse plan |
|---|---|
| `interactive_utils.py` | Copy davis-overlay rendering; adapt to our palette |
| `reader.py` | `PropagationReader` dataset — copy as-is for Phase 3 |
| `gui.py` canvas coord transform | Port `map_np_input_to_image` / offset logic |
| `resource_manager.py` | LRU cache pattern; replace file IO with our DB layer |
| `click_controller.py` | Replace RITM with SAM2; keep same interface |

---

## Phase 1 — Foundation: DB, frame cache, video canvas

**Files:**
- `db/migrations/018_seg_masks.sql` — new table
- `python/posetrak/db/db.py` — bump SESSION_SCHEMA_VERSION 18→19, add migration
- `python/app/pose/frame_cache.py` — `FrameCache`: LRU, lazy cv2 decode
- `python/app/pose/video_canvas.py` — `VideoCanvas`: scaled display, mask overlay, click signals
- `python/app/pose/cutie_init_panel.py` — `CutieInitPanel`: camera selector + scrubber
- `python/app/ui/content_panels.py` — "Segmentation…" button in `TrialPanel` header

No Cutie/SAM2 imports.  Result: scrubber that shows video frames and existing stored masks.

## Phase 2 — Click interaction: SAM2 → mask

**Files:**
- `python/app/pose/cutie_click_controller.py` — `ClickController` wrapping SAM2 single-image
- `python/app/pose/cutie_init_panel.py` — wire clicks, add PersonSelector row
- `python/pyproject.toml` — add `[cutie]` optional extra

SAM2 imported lazily (`try/except`).  Each left/right click updates the mask overlay live.

## Phase 3 — Cutie worker: track/stop/persist

**Files:**
- `python/app/pose/cutie_worker.py` — `CutieWorker(QThread)` with `mask_ready` signal
- `python/app/pose/cutie_init_panel.py` — Track Fwd/Bwd/Stop buttons; write to `seg_masks`

Reuses `PropagationReader` from the Cutie demo.  Masks persisted to DB frame-by-frame.

## Phase 4 — Correction + RTMPose post-step

**Files:**
- `python/app/pose/cutie_init_panel.py` — correction mode (scrub → click → invalidate → re-track)
- `python/tools/run_seg_pose.py` — RTMPose batch over `seg_masks` (replaces run_cutie_pose.py
  auto-init path; reads masks from DB)
- `python/tools/download_cutie_weights.py` — weight download helper
- `python/pyproject.toml` — finalise `[cutie]` extra

Auto-init convenience: "Auto-init from YOLO" button seeds ClickController from YOLO bboxes
and prompts the user to verify before tracking.

---

## Init model: SAM2 vs RITM

SAM2 is the chosen approach (consistent with existing `run_cutie_pose.py`; better quality;
on PyPI; Apache 2.0 code license).  The `ClickController` abstraction keeps the model
swappable — a RITM backend can be added later if SAM2 proves too slow or the weight license
is problematic.

SAM2 weight license: Meta's custom research license — verify terms before production/commercial
distribution.  Cutie weight license: check repo (likely CC BY-NC).

---

## Key design decisions

- **Segmentation-first**: tracking and pose estimation are separate steps; the interactive loop
  only runs Cutie, not RTMPose.
- **Invalidate-forward on correction**: clicking in frame M deletes all `seg_masks` rows with
  `frame_idx > M` for that camera and re-runs Cutie from M.  No branching, one contiguous valid
  range per camera.
- **Cutie checkpoint resume** (Phase 4): optional 300-frame memory snapshots (pickled tensors in
  temp dir) limit re-run replay to the last 300 frames when correcting late in a clip.
- **ClickController abstraction**: thin interface (`push_point`, `clear`, `get_mask`) wrapping
  SAM2; swappable for RITM or any other model without touching the panel code.

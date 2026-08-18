# Pose Extraction Application — Design Proposal

This document supersedes the Stage 1 section of `pipeline-ui-requirements.md` for the
DB-integrated pipeline.  The earlier doc describes a Marimo-based approach; this one
describes the Qt desktop application that replaces the three-script workflow:

```
OLD: undistorted video files → pose_extraction.py (marimo) → OpenPose JSON → pose import CLI
NEW: original videos + DB  → PoseExtractionWindow  → DB (detections + observations)
```

---

## Current Status (2026-04-10)

### Phase A — complete
Detection pipeline (`DetectionPipeline` QThread worker), DB schema (`detection_runs`,
`person_detections`, `detection_keypoints`), YOLO and RTMPose backends, and the CLI entry
point are all implemented and in use.

### Phase B — partially complete

**Done:**
- `PoseExtractionWindow` with shot/sync/run selectors, Mark Start/End range, Run Detection
  button, and progress display.
- `StitcherWidget`: timeline with dynamic scale (fitted to viewport width, rebuilt on
  resize), per-camera track bars, playhead, person assignment via context menu.  Bar
  colors use matplotlib tab10 palette (consistent with overlay).
- `FrameViewWidget`: frame display with COCO skeleton overlay; camera dropdown preserving
  global time on switch; seek-to-clicked-position from stitcher (`time_clicked` signal).
- Overlay bbox colors match stitcher bar colors; selected track shown with 4 px border.
- `finalise.py`: writes `pose_observation_sequences` + `pose_observations` from
  assignments.

**Not yet done (Phase B remaining):**
- `PersonPreviewWidget` — live bbox crop panel (US-3; replaces hover tooltip).
- "From here onwards" assignment scope toggle (US-4).
- Assignment conflict detection and resolution dialog (US-4a).
- Assignment state persisted to `person_tracks` DB table; currently in-memory only.
- `FrameCache` integration into `FrameViewWidget` (currently uses raw cv2 seek per frame).

### Phase C — not started
Manual bbox correction, confidence sparkline (postponed), multi-camera suggestions.

---

## 1. Context and Goals

### What the current workflow costs

A single multi-camera shot currently requires:

1. **Undistort + clip extraction** — run a script to decode each camera's video, apply the
   undistortion maps, and write new MP4 files.  Disk cost: full copy of every video.
2. **pose_extraction.py (Marimo)** — open a browser notebook, configure video paths,
   wait for YOLO to finish (minutes), do interactive stitching, wait for RTMPose
   (minutes), export JSON.  All camera state is in-memory only; a crash loses everything.
3. **`posetrak-db pose import`** — run a CLI command with a pile of flags to read the JSON
   directory tree and populate the database.

Steps 1 and 3 are pure plumbing.  Step 2 has one genuinely interactive moment (stitching)
surrounded by two waiting periods.  The proposal collapses all three into one application
that eliminates the intermediate files.

### Goals

- G1: Single application covers the full range from shot selection to DB write.
- G2: No intermediate files on disk (no rectified video clips, no OpenPose JSON).
- G3: One wait before the interactive step: detection + pose estimation run together
  in the background first.
- G4: Detection results cached in DB so re-running stitching or changing thresholds does
  not re-run the GPU models.
- G5: Detector and pose estimator are swappable backends, not hard-wired imports.
- G6: Architecture supports future multi-camera assisted stitching (coarse triangulation
  to auto-suggest cross-camera person identity).

---

## 2. User Stories

### US-1 — Select processing range
*As a practitioner I often leave cameras running before and after the technique, so I
want to define a start/end time for the portion I actually want to track, rather than
processing the full video.*

Acceptance criteria:
- The UI shows a timeline of the shot (global time, derived from sync).
- I can drag handles to set start/end, or type times directly.
- The range is stored with the detection run so I can reproduce it.

### US-2 — Background detection
*I want to click "Run" and be able to do other work while YOLO and RTMPose process the
videos.  I do not want the UI to freeze.*

Acceptance criteria:
- A progress bar shows per-camera and overall progress.
- The UI remains responsive (I can inspect earlier results or scroll while processing).
- If the application is closed and reopened, a completed detection run is still available
  (cached in DB).

### US-3 — Person preview panel
*I want to see the current person's bbox crop and skeleton overlay while stitching, so I
can confirm which person a track represents without losing context on the full frame view.*

Acceptance criteria:
- A `PersonPreviewWidget` panel is always visible alongside the frame view.
- When a track segment is selected, the panel shows a cropped view of that track's bbox
  in the current frame with a skeleton overlay.
- The crop updates as the playhead moves through the selected track.
- No hover tooltip on the stitcher (removed); the preview panel replaces this need.

### US-4 — Person stitching
*YOLO re-assigns track IDs after occlusions or camera cuts.  I need to merge these
fragments and assign them to named persons (e.g. "harri", "timo").*

Acceptance criteria:
- Multi-camera timeline heatmap: rows = tracks, columns = time, one panel per camera.
- Click a segment to select it; assign it to a named person with a dropdown or keyboard shortcut.
- Reassignment is instant; no recomputation.
- Stitching state is saved to DB so I can close and resume.
- Assignment scope: "this segment only" (default) or "from here onwards" (assigns all
  unassigned segments for this camera that start at or after the selected segment's start
  time and that do not already belong to a different person).
- Conflict prevention: assigning a person to a segment that time-overlaps an existing
  segment already assigned to that same person shows a confirmation dialog (see §5.5).

### US-4a — Assignment conflict resolution
*If I accidentally try to put two tracks for the same person in the same time window I
want to be warned and given a clear way to resolve it.*

Acceptance criteria:
- The application detects when a new assignment would give the same person two
  simultaneous detections within the same camera (time-overlapping track segments).
- A confirmation dialog lists the conflicting segments and offers two choices:
  - **Detach conflicting bars** — remove the person assignment from every segment that
    overlaps in time with the newly assigned segment (the new assignment proceeds).
  - **Cancel** — leave all assignments unchanged.
- Partial overlap (conflicting bar extends beyond the new bar's time range) is treated
  the same as full overlap: the entire conflicting bar is detached, since tracks are
  atomic and cannot be split.
- The conflict check and resolution logic has unit tests covering all corner cases:
  adjacent (no overlap), partial overlap, full containment, multiple conflicts at once.

### US-5 — Inspect and correct
*When pose confidence is low in a region I want to see what went wrong and optionally
correct bounding boxes.*

Acceptance criteria:
- Click a low-confidence region in the sparkline → jump to that frame in the video view.
- Video view shows the undistorted frame with YOLO bbox and skeleton overlay.
- I can drag the bbox to correct it; re-running RTMPose on the corrected region is a
  single button press.

### US-6 — Finalise to DB
*After stitching I want to write the observations for the named persons into the session
DB so the tracker can be run immediately.*

Acceptance criteria:
- "Finalise" creates a `pose_observation_sequence` row and populates `pose_observations`
  for the assigned persons only.
- RMS keypoint confidence and frame count are shown as a summary before I confirm.
- The sequence is immediately visible to the tracker config page / CLI.

### US-7 — Re-run without losing earlier work
*If I change the confidence threshold or want to try a different RTMPose model, I should
be able to re-run without re-doing the stitching.*

Acceptance criteria:
- Detection runs are versioned in the DB.  A new run stores its own detections.
- An existing stitch can be applied to a new detection run if the frame range and cameras
  match (with a warning if track IDs differ significantly).
- Old detection runs can be deleted to recover disk space (thumbnails are the main cost).

### US-8 — Swappable backends (developer story)
*I want to try a different detector (e.g. RT-DETR) or pose estimator (e.g. ViTPose)
without changing the application logic.*

Acceptance criteria:
- Detector and pose estimator are selected from a dropdown populated by a registry.
- A new backend is registered by implementing two small protocol classes and adding an
  entry to a config file; no changes to the application core.

---

## 3. Requirements

### Functional

| ID   | Requirement |
|------|-------------|
| R-01 | Load shots from the session DB; display camera count, video duration, sync status. |
| R-02 | Show a global-time timeline derived from the shot's sync config; let user select a processing range. |
| R-03 | Run person detection (YOLO) and pose estimation (RTMPose) together in a background thread; stream per-frame progress to the UI. |
| R-04 | Apply undistortion maps (from DB intrinsics) in-memory during frame decode; no rectified video files are written. |
| R-05 | Cache detection results (bboxes, keypoints, thumbnails) in the session DB after a run; do not re-run if a cached run exists for the same video range and model. |
| R-06 | Provide a multi-camera timeline stitcher UI; track-to-person assignments are stored in DB and survive application restart. |
| R-07 | Show skeleton overlays and confidence sparklines in the stitcher without requiring additional computation. |
| R-08 | Allow manual bbox correction on any frame; re-run pose estimation on corrected frames only. |
| R-09 | Write finalised observations to `pose_observations` keyed to the chosen sync config; set `pixels_are_undistorted = 1`. |
| R-10 | Support multiple people per shot (≥ 4 simultaneous; typical case is 2). |
| R-11 | Support re-runs: multiple `detection_run` records per shot; let user choose which run to stitch. |
| R-12 | Record provenance: model names, versions, thresholds, and processing timestamp stored with every detection run. |
| R-13 | Detector and pose estimator selectable from a UI dropdown; new backends require no changes to application code. |

### Non-functional

| ID   | Requirement |
|------|-------------|
| N-01 | UI thread never blocks; all heavy computation runs in a `QThread` worker. |
| N-02 | Detection throughput target: ≥ 60 fps per camera on a mid-range NVIDIA GPU (e.g. RTX 3060). |
| N-03 | Thumbnail storage: JPEG at quality 80, max 160×120 px per crop; < 5 kB each. |
| N-04 | DB size overhead: for a 10-minute 120 fps 4-camera shot (~288 000 frame-person pairs at 2 persons/cam), total detection cache ≤ 500 MB including thumbnails. |
| N-05 | Application is self-contained: no browser, no separate notebook server. |
| N-06 | Works without a GPU (CPU fallback for detection/pose; slower but functional). |

---

## 4. Database Schema Changes

### Coordinate system: original (distorted) pixel space

All pixel coordinates stored in `person_detections` and intermediate keypoint tables use
**original video coordinates** — the pixel space of the unmodified camera output, before
any undistortion is applied.

Rationale:
- If camera intrinsics are later found to be wrong and re-calibrated, stored detections
  remain valid.  Only the undistortion step at load time changes.
- YOLO and RTMPose run on the original frames (or on frames undistorted in-memory for the
  inference call — see §5.3), but coordinates are converted back to distorted space before
  writing to DB.
- The `pixels_are_undistorted` column on `pose_observation_sequences` (already in the
  schema) will be `0` for sequences finalised from this pipeline.  The C++ tracker already
  handles both cases.
- Thumbnails (crops from the original frame) are inherently in distorted space; there is
  no need to rectify them for display in the stitcher.

The one exception is the undistortion maps themselves, which remain in `intrinsics_calibrations`
(already stored there) and are applied by the tracker and by the frame view when showing
overlays.

### Tables from `capture-pipeline-architecture.md` (already designed)

`capture-pipeline-architecture.md` §"Schema additions actually needed" already defines
the three tables this pipeline needs:

- **`person_detections`** — bbox per frame × camera × track × region_type
- **`person_tracks`** — named-person timeline assembled by the stitcher
- **`frame_cache_entries`** — JPEG thumbnail / crop cache (replaces inline thumbnail BLOBs)

Those definitions are authoritative.  This document adds only what is missing from them.

### One addition: `detection_runs` for provenance and versioning

The architecture doc's tables do not record which model + parameters produced a set of
detections, or support keeping multiple runs for the same shot.  A thin provenance table
fills this gap:

```sql
-- One row per processing run (model + range + parameters).
-- Provides versioning: multiple runs per shot are kept; user chooses which to stitch.
CREATE TABLE detection_runs (
    id                  TEXT PRIMARY KEY,
    shot_id             TEXT NOT NULL REFERENCES shots(id),
    sync_config_id      TEXT NOT NULL REFERENCES sync_configs(id),
    time_start_s        REAL NOT NULL,   -- global time (inclusive)
    time_end_s          REAL NOT NULL,   -- global time (exclusive)
    detector_model      TEXT NOT NULL,   -- e.g. "yolo11x"
    pose_model          TEXT NOT NULL,   -- e.g. "rtmpose-l-133kp"
    detector_version    TEXT,
    pose_version        TEXT,
    detector_conf       REAL NOT NULL DEFAULT 0.3,
    pose_conf_threshold REAL NOT NULL DEFAULT 0.3,
    status              TEXT NOT NULL DEFAULT 'running',  -- running | complete | failed
    created_at          TEXT NOT NULL,
    completed_at        TEXT
);
```

`person_detections` and `frame_cache_entries` gain a `detection_run_id` foreign key so
that results from different runs are kept separate and individually deletable:

```sql
ALTER TABLE person_detections    ADD COLUMN detection_run_id TEXT REFERENCES detection_runs(id);
ALTER TABLE frame_cache_entries  ADD COLUMN detection_run_id TEXT REFERENCES detection_runs(id);
```

When `detection_run_id IS NULL` the row was inserted by an earlier tool (e.g. the old
Marimo notebook) and is treated as belonging to an implicit legacy run.

### One addition: intermediate keypoints table

`person_detections` stores bounding boxes only.  Because RTMPose runs before stitching
(§2 US-3), keypoints need somewhere to live until the `pose_observations` table is
populated at finalise time.

```sql
-- Per-frame, per-track RTMPose output, before person assignment.
-- Coordinates in original (distorted) pixel space, matching person_detections.
-- Deleted (or kept as cache) once finalise writes to pose_observations.
CREATE TABLE detection_keypoints (
    detection_run_id    TEXT NOT NULL REFERENCES detection_runs(id),
    shot_video_id       TEXT NOT NULL REFERENCES shot_videos(id),
    video_frame         INTEGER NOT NULL,
    track_id            INTEGER NOT NULL,
    region_type         TEXT NOT NULL DEFAULT 'full_body',
    keypoints           BLOB NOT NULL,  -- float32[N, 3]: x, y, conf  (distorted px)
    PRIMARY KEY (detection_run_id, shot_video_id, video_frame, track_id, region_type)
);
```

### Thumbnails: use `frame_cache_entries`, not inline BLOBs

The architecture doc's `frame_cache_entries` table (with `cache_type = 'person_crop'`)
is the correct place for stitcher thumbnails.  The earlier draft of this document
proposed an inline `thumbnail BLOB` column on the detections table — that was wrong and
is not used.

Thumbnail generation policy:
- One crop per track per second (subsampled from frame rate; 600 s × 8 tracks = 4 800 entries).
- Stored inline (`data` column) when the crop is ≤ 160×120 px (≤ ~5 kB JPEG); as file otherwise.
- `detection_run_id` on `frame_cache_entries` links the crop to the run that generated it.
- Crops are derived data: safe to delete and regenerate from the original video at any time.

### Unchanged

`pose_observation_sequences` and `pose_observations` are written only at finalise time and
are otherwise untouched by this pipeline.

---

## 5. Software Design

### 5.1 Module layout

```
python/app/pose/
    __init__.py
    main.py                  # entry point: PoseExtractionWindow
    detection_pipeline.py    # QThread worker; orchestrates decode → detect → pose
    backends.py              # PersonDetector / PoseEstimator protocols + registry
    # detector implementation now lives in posetrak/detection/backends_rtmdet.py
    # (YOLOXDetector, rtmlib -- see docs/license-analysis.md)
    backends_rtmpose.py      # RTMPose implementation
    db_cache.py              # read/write detection_runs, person_detections,
                             #   detection_keypoints, person_tracks, frame_cache_entries
    stitcher.py              # StitcherWidget (timeline heatmap + assignment controls)
    frame_view.py            # FrameViewWidget (video frame + overlay)
    confidence_plot.py       # ConfidencePlotWidget (sparkline per track)
```

The application is launched via a new entry point `posetrak-pose` registered in
`pyproject.toml`.

### 5.2 Backend protocol

```python
# backends.py

from typing import Protocol
import numpy as np
from dataclasses import dataclass

@dataclass
class PersonDetection:
    track_id: int
    bbox: np.ndarray      # float32[4]: x, y, w, h  (pixel coords)
    confidence: float

@dataclass
class PoseResult:
    track_id: int
    keypoints: np.ndarray  # float32[N, 3]: x, y, conf

class PersonDetector(Protocol):
    name: str             # e.g. "yolo11x"
    version: str          # package version

    def detect_and_track(
        self,
        frame: np.ndarray,       # uint8 HxWx3 BGR, original distorted video frame
        frame_idx: int,
    ) -> list[PersonDetection]: ...
    # Returned bbox coordinates are in original (distorted) pixel space.

    def reset_tracker(self) -> None:
        """Call between shots to clear tracker state."""
        ...

class PoseEstimator(Protocol):
    name: str
    version: str
    n_keypoints: int       # 17 for COCO, 133 for Halpe/Wholebody

    def estimate(
        self,
        frame: np.ndarray,             # original distorted frame (full)
        detections: list[PersonDetection],  # bboxes in distorted px
    ) -> list[PoseResult]: ...
    # Returned keypoint coordinates are in original (distorted) pixel space.


# Registry — populated from config or direct registration
_detector_registry: dict[str, type[PersonDetector]] = {}
_estimator_registry: dict[str, type[PoseEstimator]] = {}

def register_detector(cls: type[PersonDetector]) -> type[PersonDetector]:
    _detector_registry[cls.name] = cls
    return cls

def available_detectors() -> list[str]:
    return list(_detector_registry.keys())
```

`YOLOXDetector` and `RTMPoseEstimator` are the built-in implementations; other backends
register themselves by decorating with `@register_detector` / `@register_estimator`.

### 5.3 Detection pipeline (background worker)

```python
# detection_pipeline.py

class DetectionPipeline(QThread):
    """Runs decode → undistort → detect → pose for all cameras in a detection run."""

    progress = Signal(int, int, str)    # (done_frames, total_frames, camera_id)
    camera_done = Signal(str)           # camera_instance_id
    finished = Signal(str)              # detection_run_id
    error = Signal(str)                 # error message

    def __init__(
        self,
        session: sqlite3.Connection,
        shot_id: str,
        sync_config_id: str,
        time_start_s: float,
        time_end_s: float,
        detector: PersonDetector,
        estimator: PoseEstimator,
        thumbnail_interval_s: float = 1.0,   # generate thumbnail every N seconds
    ): ...

    def run(self) -> None:
        run_id = self._create_run_record()
        for cam in self._cameras:
            self._process_camera(run_id, cam)
        self._mark_complete(run_id)
        self.finished.emit(run_id)

    def _process_camera(self, run_id: str, cam: CameraInfo) -> None:
        """
        Inner loop:
          1. Seek to time_start frame using sync anchor
          2. Decode frames with PyAV (original distorted video, no undistortion)
          3. detector.detect_and_track(frame) → bboxes in distorted px
          4. estimator.estimate(frame, detections) → keypoints in distorted px
          5. db_cache.write_detection_batch(run_id, cam_id, batch)
             writes to person_detections + detection_keypoints (distorted coords)
          6. Every thumbnail_interval_s: JPEG-encode the bbox crop from the
             original frame; store in frame_cache_entries (person_crop, distorted)
        """
        ...

    def stop(self) -> None:
        self._stop_flag.set()
```

Frame decode uses PyAV (already used in `frame_cache.py`).  No undistortion is applied
during the detection pass — the original distorted frames are fed directly to YOLO and
RTMPose.  For strongly distorted fisheye cameras this may reduce detection quality; a
future option is to undistort frames in-memory for the inference call while still
converting the output keypoints back to distorted coordinates before writing to DB.
Undistortion maps remain in `intrinsics_calibrations` and are used only by the frame
view widget (for display overlays) and by the C++ tracker at load time.

The pipeline processes cameras sequentially by default.  A future optimisation is to
process cameras in parallel (each on its own QThread) since GPU memory allows it.

### 5.4 Main window layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  Shot: 2026-03-10 / Harri_aihanmi  ▼     Sync: LED auto (4 cams)    │
├──────────────────────────────────────────────────────────────────────┤
│  [◄──────────────●══════════════════●──────────────►]               │
│   00:00                range: 00:12 – 01:45              02:30       │
│                                                                      │
│  Detector: yolo11x ▼   Pose: rtmpose-l-133kp ▼   Conf: 0.30        │
│                                          [▶ Run Detection]           │
├──────────────────────────────────────────────────────────────────────┤
│  ████████████████████████████░░░░░░ 72%  cam3/4  frame 5412/7500    │
├────────────────────┬─────────────────────────────────────────────────┤
│                    │  CAM1     cam1_track0 ══════════════════─────── │
│  [frame view]      │           cam1_track1 ──────═══──────══════════ │
│                    │  CAM2     cam2_track0 ══════════════════─────── │
│  [skeleton         │           cam2_track1 ──────═══──────══════════ │
│   overlay]         │  CAM3     ...                                   │
│                    ├─────────────────────────────────────────────────┤
│  frame: 5412       │  Selected: cam1_track0  [00:12 – 01:32]        │
│  t: 00:45.10       │  Assign to: [harri ▼]  [+ Add person]         │
│                    │                          [✓ Finalise]           │
└────────────────────┴─────────────────────────────────────────────────┘
```

**Time range widget** (top): dual-handle slider over the shot duration; ticks at 5 s
intervals; displays global timestamps.

**Detection config** (below range): model dropdowns + confidence slider; "Run Detection"
button; greyed out while a run is in progress.

**Progress bar**: shown only during a run; includes per-camera status.

**Frame view** (bottom left): the multi-camera video widget from the sync page, reused.
Shows undistorted frame + YOLO bbox + RTMPose skeleton overlay for the selected frame/track.

**Person preview panel** (below frame view or in a separate splitter pane): a cropped
and zoomed view of the selected track's bounding box with skeleton overlay, updated as
the playhead moves.  Replaces the hover-tooltip thumbnail from the earlier design.

**Stitcher timeline** (bottom right): one row per track per camera; columns = time.
Track segments coloured by assigned person (or grey if unassigned).
Click a segment → load that frame/track in the frame view and person preview; enable
assignment controls.

**Assignment controls**: person name dropdown (populated from DB or typed freeform),
scope toggle ("this segment only" / "from here onwards"), "Finalise" button.
Assignment triggers conflict check (see US-4a / §5.5).

### 5.5 StitcherWidget internals

The timeline heatmap is rendered as a `QGraphicsScene` for efficient partial updates.
Each track segment is a `QGraphicsRectItem`; colour is set when the detection run loads
or when an assignment changes.  No hover-tooltip thumbnails — the person preview panel
(§5.6) handles this need more reliably.

Time axis: scale is computed dynamically from `viewport_width / total_duration_s`,
clamped to a `[5, 500]` px/s range; rebuilt on resize.  A horizontal scrollbar covers
the full range.

**Assignment scope — "from here onwards"**: when this mode is active, assigning a
segment also assigns every other unassigned segment for the same camera whose start time
is ≥ the selected segment's start time.  Segments already assigned to a different person
are skipped (not overwritten).

**Conflict detection and resolution**: before writing any assignment, the widget checks
whether the person already has a segment assigned in the same camera that time-overlaps
the new segment.  Two segments overlap when `max(start_a, start_b) < min(end_a, end_b)`.
Adjacent segments (one ends exactly where the other begins) are not considered a conflict.
If a conflict is found, a `QMessageBox` lists the offending segments and offers:

1. **Detach conflicting bars** — removes the person assignment from every overlapping
   segment, then proceeds with the new assignment.
2. **Cancel** — aborts the new assignment; all existing assignments unchanged.

Partial overlaps are handled identically to full overlaps: the entire conflicting bar is
detached (tracks are atomic; frame-level splitting is not supported).

The conflict logic must be unit-tested for: no overlap, adjacent (touching) segments,
partial overlap from the left, partial overlap from the right, full containment of new
bar inside existing, full containment of existing inside new, multiple conflicts at once,
and "from here onwards" combined with conflicts.

### 5.6 PersonPreviewWidget

A persistent panel (not a tooltip) that shows the selected track's person crop updated
in real time as the playhead moves.

- Crops the bounding box of the selected track from the current frame, with a small
  margin (~10 % on each side).
- Draws the skeleton overlay on the crop using the same `SkeletonDetectionOverlay` logic
  as the full frame view, scaled to the crop dimensions.
- Updates whenever `frame_changed` fires on the frame view.
- If the selected track has no detection in the current frame the panel shows the most
  recent crop for that track (last-seen behaviour).
- Crops are sourced from `frame_cache_entries` (type `PERSON_CROP`) when available,
  falling back to a live cv2 decode.

### 5.7 Confidence sparkline *(postponed)*

A `QChartView` (QtCharts) showing per-keypoint mean confidence over time for the selected
track.  Low-confidence regions shaded red; clicking seeks the frame view.

**Status**: postponed.  The person preview panel (§5.6) provides sufficient visual
feedback for now.  Revisit if confidence-based correction (US-5) is implemented.

### 5.8 Finalise → DB write

```python
def finalise_to_db(
    session: sqlite3.Connection,
    detection_run_id: str,
    shot_id: str,
    sync_config_id: str,
    assignments: list[TrackAssignment],
    pose_model: str,
    conf_threshold: float,
) -> str:
    """
    Create pose_observation_sequence + pose_observations from the
    detections + track_assignments tables.

    Returns the new pose_observation_sequence_id.
    """
    # 1. Create sequence row
    seq_id = generate_id()
    session.execute(
        "INSERT INTO pose_observation_sequences ... VALUES (?,...)",
        (seq_id, shot_id, sync_config_id, ..., pixels_are_undistorted=1),
    )
    # 2. For each segment in person_tracks.track_segments
    #    (person_name, track_id, start_frame, end_frame):
    #    JOIN detection_keypoints ON (run_id, shot_video_id, track_id, frame range)
    #    JOIN sync to convert video_frame → timestamp_s
    #    INSERT INTO pose_observations (seq_id, camera_instance_id, timestamp_s,
    #         person_name, keypoints_blob)
    #    keypoints are stored as-is (distorted px); pixels_are_undistorted=0
    # 3. Done — no JSON files, no import step
```

### 5.9 Future: multi-camera stitching assistance

The `detections` table is structured to support this without schema changes.  The
algorithm outline:

1. For each pair of cameras, find timesteps where both have ≥ 1 detection.
2. For each pair of tracks (one from each camera), extract the bbox centre in undistorted
   pixel space and coarsely triangulate using the known camera extrinsics.
3. Cluster the 3D trajectories over the time intersection to find candidate "same person"
   pairs.
4. Present suggestions in the stitcher UI as pre-filled assignments with a confidence
   score; user confirms or overrides.

This requires extrinsics to be in the DB (which is already the intended workflow) and is
feasible once the wand calibration is working.

---

## 6. Phasing

### Phase A — Core pipeline (no UI)

- `detection_pipeline.py`: decode + undistort + detect + pose → DB, no stitcher yet.
- `db_cache.py`: write/read `detection_runs`, `detections`, `track_assignments`.
- `backends_rtmdet.py` + `backends_rtmpose.py`: thin wrappers around existing rtmlib code.
- DB migration for the three new tables.
- CLI entry point: `posetrak-pose run --shot <id> --sync <id> --start 12 --end 105`
  for scriptable use without the GUI.

Deliverable: can replace steps 1 and 3 of the old pipeline (undistorted video extract
and JSON import).  Pose Marimo notebook still used for stitching in this phase.

### Phase B — Stitcher window

- `PoseExtractionWindow` with time range selector, progress bar, and stitcher timeline.
- `StitcherWidget` with track-to-person assignment, "this segment only" / "from here
  onwards" scope toggle, and conflict detection/resolution dialog (US-4a).
- `PersonPreviewWidget` — live bbox crop with skeleton overlay (replaces hover tooltip).
- Frame view with skeleton overlay (reuse existing `CameraCell` widget from sync page).
- Finalise → DB write.
- Assignment state persisted to `person_tracks` in DB; survives application restart.

Deliverable: Marimo notebook fully replaced.  Full old pipeline (steps 1–3) replaced.

### Phase C — Refinement

- Manual bbox correction + partial RTMPose re-run (US-5).
- Confidence sparkline (postponed from Phase B; see §5.7).
- Multi-camera suggested assignments (future, needs extrinsics in DB).
- Backend registry UI (swap models from dropdown).

---

## 7. Reuse from existing code

| Existing component | Reuse |
|---|---|
| `poseanalysis.py` — `analyze_video_with_yolo_tracker` | Refactor into `YOLOXDetector`; strip ipywidgets dependency |
| `poseanalysis.py` — `VideoData` / `MultiVideoPoseDataset` | Replace with `DetectionPipeline`; DB replaces in-memory cache |
| `poseanalysis.py` — `NamedPersonStitcher` | Replace with `StitcherWidget`; stitching state goes to DB |
| `frame_cache.py` — PyAV decode + undistort | Reuse directly in `DetectionPipeline._process_camera` |
| `page_sync.py` — `CameraCell` / multi-camera video widget | Reuse as `FrameViewWidget` |
| `overlay.py` — `AnnotationPointOverlay`, `ReprojectionOverlay` | Reuse for manual correction overlay |
| `import_pose_json.py` — DB write logic | Replace with `finalise_to_db()`; JSON format no longer needed |

The key change in `poseanalysis.py` is removing the assumption that results are cached
in-memory for the duration of the process.  The GPU inference loop (`_process_camera`)
becomes a generator that yields batches to the DB writer; memory for a single batch
(e.g. 32 frames) is all that is held at once.

# Capture Pipeline: Current Process & Architecture Proposal

## Current Process

### Step 1 — Rough video synchronization: `sync_videos.py`

**Tool type**: PyQt6 desktop GUI
**Input**: Video files (MP4/AVI/MOV) or a project YAML file
**Output**: Project YAML with `cameras[].{name, path, sync_frame, fps}` + `ref_camera`

**What it does**: Shows all camera videos in a grid. User steps frame-by-frame through each video to find the same physical moment across cameras (e.g., a clap, a jump). "Set sync reference" records the current frame in each video as `sync_frame`. The sync frame is used as a time origin; relative offsets allow scrubbing synchronized across cameras ("Synchronize to ref_camera" jumps all other videos to the equivalent moment).

**Key limitation**: Manual, coarse — frame-level accuracy only. Fine-grained sync (sub-frame) requires the LED-based step later.

---

### Step 2 — Intrinsics calibration: `calibrate_intrinsics.py`

**Tool type**: CLI (Click)
**Input**: Checkerboard calibration video or image directory; `--camera-name`, `--camera-mode`, `--rows`, `--cols`, `--fisheye`
**Output**: HDF5 file with:
- `intrinsics/matrix` — original K from calibration
- `intrinsics/matrix_undistorted` — new K for undistorted images
- `intrinsics/distortions` — distortion coefficients
- `undistortion_maps/mapx`, `mapy` — dense remap arrays (compressed)
- `calibration_undistorted/` — K from re-calibrating on undistorted frames
- `video_properties` attrs (fps, size, etc.)

**Algorithm**: Select sharp frames via Laplacian variance local maxima → OpenCV checkerboard corner detection → `calibrateCamera` (pinhole + rational model) or `fisheye.calibrate`. Then undistorts frames and re-calibrates to validate.

**Usage pattern**: Done once per (camera model × capture mode), reused across many sessions.

---

### Step 3 — Project setup & undistortion: `setup_pose2sim_project.py`

**Tool type**: CLI (Click)
**Input**: Edited project YAML (cameras with `path`, `sync_frame`, `calib.intrinsics` HDF5 path, `calib.extrinsics.frame`; scenes with `start_frame`/`end_frame`)
**Output** (written to `project.path/`):
- Per scene × camera: undistorted video clip extracted between scene frame range, synchronized via `sync_frame`
- Per camera: undistorted calibration JPEG (the `extrinsics.frame` from that camera's video)
- Pose2Sim TOML with undistorted intrinsics (K_new, no distortion, size)

**Key design issue**: All downstream steps work with undistorted images/coordinates. Changing intrinsics invalidates all subsequent outputs.

---

### Step 4 — Extrinsics annotation: VIA (external tool)

**Tool type**: Web-based image annotator (VGG Image Annotator)
**Input**: Calibration JPEGs from step 3
**Output**: VIA JSON file — per image, per region: `{shape: point, cx, cy}` + attributes `{row, col}` (for known 3D grid points) or `{feature_name}` (for named common points without known 3D coords)

**Workflow**: User opens all calibration JPEGs in VIA, clicks each visible corner of the calibration target (or other known 3D landmarks), assigns coordinates. Requires knowing the physical 3D world coordinates of each annotated point.

---

### Step 5 — Extrinsics calibration: `calibrate_extrinsics.py`

**Tool type**: CLI
**Input**: VIA JSON annotations, intrinsics HDF5s, project YAML (for camera list)
**Output**: Pose2Sim TOML updated with `rotation` (Rodrigues vector) and `translation` per camera

**Algorithm**: Optionally converts annotation coordinates from distorted to undistorted (using stored undistortion maps). Per-camera PnP solve (`cv2.solvePnP`) using known 3D control points. Optionally followed by bundle adjustment (`scipy.optimize.least_squares`) across all cameras jointly.

---

### Step 6 — Fine synchronization: `video_sync.py`

**Tool type**: Marimo notebook
**Input**: Videos, manually identified LED ROI bounding boxes per camera (currently hardcoded in notebook)
**Output**: Sync JSON with camera frame → global timeline mapping

**Algorithm**: Extract per-frame max brightness change within each LED ROI → find peaks → cross-correlate peaks across cameras. Produces `{cam: {video_frame: global_time}}` mapping for one or more anchor events.

**Key limitation**: LED bounding boxes currently hardcoded. No interactive ROI selection in the notebook.

---

### Step 7 — Person detection & pose extraction: `pose_extraction.py`

**Tool type**: Marimo notebook (full-width)
**Input**: Undistorted video clips (one directory of `cam{N}.mp4`), YOLO11 model, person names
**Output**: OpenPose JSON files per camera × frame (directory `pose/`)

**Phase 1 — YOLO tracking**: Runs YOLO11 tracker on each camera video. Results pickled per video (cache keyed by path + mtime). Timeline stitcher UI: interactive heatmap (frame × person/track) where user merges or splits YOLO track IDs to form named-person continuous timelines. Problem: YOLO frequently loses and re-acquires tracks, so a single person may span many track IDs; rarely, one ID covers multiple people.

**Phase 2 — RTMpose**: For each named person's timeline, crops bounding boxes and runs RTMPose. Exports per-frame keypoints in OpenPose JSON format.

**Cached**: YOLO results cached as pickle to avoid re-running (~30 min per camera on GPU). RTMpose results exported only when user clicks Export.

---

### Current Tool & Data Flow

```
sync_videos.py  →  project.yaml
     +
calibrate_intrinsics.py  →  calibration.h5

setup_pose2sim_project.py  →  undistorted videos + calib JPEGs + Pose2Sim TOML (intrinsics)

VIA (external)  →  annotations.json

calibrate_extrinsics.py  →  Pose2Sim TOML (intrinsics + extrinsics)

video_sync.py  →  sync.json

pose_extraction.py  →  pose/ (OpenPose JSON per camera × frame)

posetrak-db (import)  →  session.db

posetrak (tracker)  →  tracking_results in session.db
```

---

## Architecture Proposal

### Design Goals

1. **Unified UI** covering the full pipeline (except intrinsics calibration — one-time per camera model)
2. **DB-backed** — all results in posetrak SQLite, no intermediate YAML/JSON/HDF5 hand-offs
3. **Background processing** — long GPU jobs (YOLO, RTMpose) run without blocking the UI; user can stitch camera 1 while camera 2 YOLO is running
4. **Responsive video scrubbing** — JPEG frame cache for multi-video playback with annotations
5. **Work with original (distorted) videos** — undistortion applied per-pixel at render time, not baked into stored videos
6. **Integrated results visualization** — reprojection error, 3D pose, video with overlay

---

### Framework: Marimo + in-process frame server

The existing Marimo investment is the right foundation. The missing piece for responsive video is a lightweight local HTTP frame server running as a background thread in the same Python process. This gives the UI access to `<img src="http://localhost:PORT/frame?cam=cam1&frame=42&size=480">` endpoints, which browsers cache efficiently and JavaScript can swap without Python round-trips.

For background compute jobs (YOLO, RTMpose, posetrak), use `concurrent.futures.ProcessPoolExecutor` with `mo.state()` for progress reporting.

**Why Marimo over a full web app**:
- No new language/stack separation
- The pipeline is researcher-centric: power matters more than polish
- Marimo's reactive cell model maps naturally onto pipeline steps
- Custom `mo.Html()` gives access to raw JS when needed (video scrubber, 3D viewer)

---

### SQLite Schema Extensions

```
intrinsic_calibrations           NEW — replaces HDF5 files
  id TEXT PK
  camera_model_id FK
  camera_mode_id FK
  matrix_original BLOB            3×3 float64 as binary
  matrix_undistorted BLOB
  distortion BLOB
  mapx BLOB (compressed)          dense undistortion map
  mapy BLOB (compressed)
  image_size TEXT                 JSON [w, h]
  model_type TEXT                 'standard' | 'fisheye'
  calibration_error REAL
  imported_at TEXT

videos                           NEW — one row per camera per shot
  id TEXT PK
  shot_id FK
  camera_instance_id FK
  file_path TEXT
  frame_count INT
  fps REAL

shot_calibrations                NEW — extrinsics per shot (not global)
  id TEXT PK
  shot_id FK
  camera_instance_id FK
  rotation BLOB                   Rodrigues vector (3×1 float64)
  translation BLOB                (3×1 float64)
  calibration_frame INT           video frame used for annotation
  calibration_method TEXT         'pnp' | 'bundle_adjustment' | 'charuco'

person_tracks                    NEW — named persons, stitched YOLO IDs
  id TEXT PK
  pose_observation_sequence_id FK
  camera_id INT
  person_name TEXT
  yolo_track_segments TEXT        JSON [[track_id, start_frame, end_frame], ...]

yolo_detections                  NEW — raw per-frame YOLO output
  sequence_id FK
  camera_id INT
  video_frame INT
  track_id INT
  bbox_x1 REAL
  bbox_y1 REAL
  bbox_x2 REAL
  bbox_y2 REAL
  confidence REAL
  PRIMARY KEY (sequence_id, camera_id, video_frame, track_id)

frame_cache_entries              NEW — paths to cached JPEG thumbnails
  video_id FK
  frame_idx INT
  size INT                        thumbnail width in pixels
  file_path TEXT
  PRIMARY KEY (video_id, frame_idx, size)
```

---

### UI Flow

```
App
├── Sessions tab
│   └── Session list → Create / Open
│       ├── Session overview (cameras, shots, runs)
│       ├── Shot setup wizard
│       │   ├── Step 1: Add videos → assign camera mode + intrinsics calib
│       │   ├── Step 2: Rough sync  (multi-video scrubber)
│       │   ├── Step 3: Fine sync — draw LED ROI per camera → run detection (background)
│       │   ├── Step 4: Extrinsics — extract calib frame, annotate inline, compute
│       │   ├── Step 5: People detection — run YOLO (background), stitch timelines
│       │   ├── Step 6: Pose extraction — run RTMpose per person (background)
│       │   └── Step 7: Run tracker → view results
│       └── Results view
│           ├── Multi-camera video with overlay (reprojection, 3D skeleton)
│           └── Analytics (joint angles, reprojection error, tracking stats)
└── Calibration tab
    └── Intrinsics management (import .h5, run new calibration, register model/mode)
```

---

### Core Components

#### 1. Video Frame Server (background thread, in-process)

```python
class FrameServer:
    """HTTP server on localhost serving JPEG frames with optional undistortion."""

    def get_frame(self, video_path: str, frame_idx: int, width: int,
                  undistort_maps: tuple | None = None) -> bytes:
        # 1. Check LRU cache (key: path + frame + width + maps_hash)
        # 2. Decode with PyAV (preferred) or cv2.VideoCapture
        # 3. Resize to requested width
        # 4. Apply undistortion if maps provided (cv2.remap)
        # 5. Encode JPEG and cache
        # 6. Return bytes
        ...
```

Cache: LRU in-memory for ~200 full-frame thumbnails + ~1000 person-crop thumbnails. Makes scrubbing feel instant after the first pass.

#### 2. Multi-Video Scrubber Component

Custom `mo.Html()` component using a small JS snippet that fetches frames from the frame server:

```
┌─────────┬─────────┬─────────┬─────────┐
│  cam1   │  cam2   │  cam3   │  cam4   │
│ [img]   │ [img]   │ [img]   │ [img]   │
│ canvas overlay for annotations        │
└───────────────────────────────────────┘
════════════════════════════╗
  frame slider              ║  ← person-track color bands
  frame: 1234 / 4500        ║     show YOLO track assignments
                            ║
```

JS fires parallel `fetch()` calls to the frame server for all cameras when slider moves. Python stays out of the rendering hot path entirely.

#### 3. Background Job Manager

```python
@dataclass
class Job:
    job_id: str
    kind: Literal["yolo", "rtmpose", "tracker", "led_sync", "extrinsics"]
    camera: str | None
    shot_id: str
    status: Literal["queued", "running", "done", "failed"]
    progress: float      # 0.0 .. 1.0
    error: str | None

# marimo state drives the job table UI
jobs, set_jobs = mo.state({})
```

Each job runs in a `ProcessPoolExecutor` worker, reporting progress back through a `multiprocessing.Queue`. A background thread polls the queue and calls `set_jobs(...)` to trigger UI updates. Completing a job writes results to the DB.

#### 4. Inline Extrinsics Annotator

Replaces the VIA round-trip:

- Extract calibration frame from each camera (undistorted, via frame server)
- Show all frames in a grid with a canvas overlay
- User clicks corresponding points across views — each click adds a labeled dot
- Known 3D world coordinates entered as a text field (JSON) or detected from a ChArUco board
- "Compute" runs PnP + optional bundle adjustment, shows reprojection overlay immediately
- Results written directly to `shot_calibrations` table

#### 5. Timeline Stitcher (refined from `pose_extraction.py`)

Keep the existing heatmap concept and add:
- Per-track thumbnail strip (crops from frame server on hover)
- Click-to-merge and click-to-split with visual confirmation overlay
- Cross-camera consistency view: show which YOLO track ID in cam2 best matches a named person's bounding box trajectory from cam1

---

### Undistortion Strategy

The user's instinct to work with original videos is correct. The better approach:

- **Store original videos** by path in the DB; never write undistorted videos to disk
- **Store undistortion maps** in `intrinsic_calibrations` table as compressed BLOBs
- **Apply undistortion at render time** in the frame server (fast with pre-loaded maps)
- **For the tracker**: project 3D markers to distorted pixel space using the full camera model (`K_original + distortion`) instead of undistorting observations. This requires adding distortion parameters to the tracker's camera struct — significant but architecturally cleaner.
- **For extrinsics calibration and pose extraction**: undistort annotation/detection points mathematically (as `calibrate_extrinsics.py` already does with `undistort_points`) rather than undistorting images.

**Short-term pragmatic path**: Keep the current undistorted video workflow for the tracker, but stop writing undistorted videos to disk permanently. Generate them on-the-fly into a `derived/` cache directory that can be safely deleted and regenerated from the DB.

---

### Implementation Phases

**Phase 1 — DB integration for existing tools** (glue, minimal new code)
- Import intrinsics HDF5 → `intrinsic_calibrations` table
- Import project YAML → session + shot + video records
- Import extrinsics TOML → `extrinsic_calibrations` / `shot_calibrations` table
- Import sync JSON → `sync_configs` table (already partially done)
- OpenPose JSON → `pose_observations` (already done via `import_pose_json.py`)

**Phase 2 — Unified setup app**
- Frame server thread
- Multi-video scrubber component
- Shot setup wizard: rough sync, LED fine sync, inline extrinsics annotator

**Phase 3 — Background job integration**
- YOLO + RTMpose wrapped as job workers with DB output
- Timeline stitcher rewritten against `person_tracks` / `yolo_detections` tables
- Progress reporting in UI

**Phase 4 — Results visualization**
- Integrate tracker-debug and visualize_tracking capabilities into same app
- 3D pose viewer using Three.js via `mo.Html()`
- Video overlay with reprojection circles (reusing `visualize_tracking.py` logic)

---

### Technology Choices

| Need | Choice | Rationale |
|------|--------|-----------|
| UI framework | Marimo | Existing investment; reactive model suits pipeline steps |
| Video frame serving | `waitress` / `http.server` in background thread | Browser caches JPEGs; JS fetches all cameras in parallel — no Python in hot path |
| Video decoding | PyAV (preferred) or OpenCV | PyAV ~3× faster than cv2.VideoCapture for random seek |
| Background jobs | `multiprocessing.ProcessPoolExecutor` + queue | True parallelism for GPU-bound YOLO/RTMpose |
| DB | Extend existing posetrak SQLite | Already designed for this domain |
| Extrinsics annotation | Custom `mo.Html()` canvas component | Eliminates VIA dependency |
| Sync detection | Reuse existing LED algorithm | Already works well |
| 3D visualization | Three.js via `mo.Html()` | Native WebGL, smooth 3D; no Python rendering overhead |

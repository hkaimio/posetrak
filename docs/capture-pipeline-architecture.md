# Capture Pipeline: Current Process & Architecture Proposal

## Tool locations

All paths are relative to their respective repository roots. Four repositories are involved:

| Repository (root) | WSL path |
|---|---|
| **posetrak** (this repo) | `~/projects/posetrak/` |
| **pose2sim-preprocess** | `/mnt/c/Users/HarriKaimio/projects/pose2sim-preprocess/` |
| **rtmlib** | `/mnt/c/Users/HarriKaimio/projects/rtmlib/` |

| Tool | Repo | Path from repo root |
|---|---|---|
| `sync_videos.py` | pose2sim-preprocess | `sync_videos.py` |
| `calibrate_intrinsics.py` | pose2sim-preprocess | `calibrate_intrinsics.py` |
| `setup_pose2sim_project.py` | pose2sim-preprocess | `setup_pose2sim_project.py` |
| `calibrate_extrinsics.py` | pose2sim-preprocess | `calibrate_extrinsics.py` |
| `video_sync.py` | rtmlib | `harritests/video_sync.py` |
| `pose_extraction.py` | rtmlib | `harritests/pose_extraction.py` |
| `poseanalysis.py` (YOLO/RTMpose library) | rtmlib | `harritests/poseanalysis.py` |
| `posetrak-db` CLI | posetrak | `scripts/db/posetrak_db_cli.py` |
| `import_pose_json.py` | posetrak | `scripts/db/import_pose_json.py` |
| `visualize_tracking.py` | posetrak | `scripts/visualize_tracking.py` |
| `tracker-debug.py` | posetrak | `notebooks/tracker-debug.py` |
| `body-measurements.py` | posetrak | `notebooks/body-measurements.py` |
| `export_bvh.py` | posetrak | `scripts/export_bvh.py` |

---

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
**Input**: Undistorted video clips in a directory
**Output**: Sync JSON with camera frame → global timeline mapping

**Workflow**:
1. Video file selector dropdown for each camera
2. Frame display cell — user browses to the first frame of a video and visually locates the sync LED; clicking the displayed image reports pixel coordinates
3. `led_locs` dict cell — user fills in bounding box `(y1, y2, x1, x2)` for each camera based on the coordinates found above (this step requires manually editing the cell)
4. LED intensity change extraction — scans each video's LED ROI, computes per-frame brightness change
5. `synchronize_cameras()` call — finds peaks, cross-correlates between cameras
6. Result inspection cells — plots aligned intensity signals on a global timeline
7. Sync JSON export cell — writes `{cam: {fps, syncpoints: [{frame, timestamp}]}}` per camera

The LED location step is semi-interactive: the notebook provides tools to view a frame and click to get coordinates, but the coordinates must be manually copied into the `led_locs` cell. The algorithm configuration (fps per camera, cross-correlation parameters) is also set per-session by editing cells.

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
sync_videos.py  →  project.yaml  →  (manually edited to add scenes, intrinsics paths)
calibrate_intrinsics.py  →  calibration.h5

setup_pose2sim_project.py  →  undistorted videos + calib JPEGs + Pose2Sim TOML (intrinsics)

VIA (external)  →  annotations.json

calibrate_extrinsics.py  →  Pose2Sim TOML (intrinsics + extrinsics)

video_sync.py  →  sync.json

pose_extraction.py  →  pose/ (OpenPose JSON per camera × frame)

posetrak-db import commands  →  session.db
  (scripts/db/posetrak_db_cli.py: session, shot, extrinsics, sync, pose imports)

posetrak track config.toml  →  tracking_results in session.db

scripts/visualize_tracking.py        # video overlay with reprojection + outliers
notebooks/tracker-debug.py           # joint angles, stats, 3D marker positions
notebooks/body-measurements.py       # bone length calibration analysis
scripts/export_bvh.py                # BVH export for external tools
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

### Database Schema: New vs. Existing

The existing schema (documented in `docs/data-model-and-storage.md`) already covers most of the data model. The table below maps proposed additions against what already exists.

#### Tables that already exist (no new table needed)

| Proposed "new" table | Already exists as | Notes |
|---|---|---|
| `videos` | `ShotVideo` | Has `shot_id`, `camera_instance_id`, `file_path`, `first_video_frame`, `last_video_frame`, `actual_fps` |
| `shot_calibrations` | `ExtrinsicCalibration` + `ExtrinsicEntry` | `Shot.extrinsic_calibration_id` FK already supports per-shot extrinsics; `ExtrinsicEntry` holds R/t per camera |
| `intrinsic_calibrations` | `IntrinsicsCalibration` | Already exists with `camera_mode_id FK`, `distortion_model`, `fx/fy/cx/cy`, `dist_coeffs`, `rms_error` |

#### Schema additions actually needed

**`IntrinsicsCalibration` — new columns** (not a new table):

```sql
ALTER TABLE intrinsic_calibrations ADD COLUMN undistort_mapx BLOB;   -- compressed float32 remap array
ALTER TABLE intrinsic_calibrations ADD COLUMN undistort_mapy BLOB;   -- compressed float32 remap array
ALTER TABLE intrinsic_calibrations ADD COLUMN matrix_original BLOB;  -- 3×3 float64, K before undistortion
```

The current schema stores scalar fx/fy/cx/cy columns, which is correct for the undistorted K. Adding `matrix_original` (the K returned directly by `calibrateCamera`) and the undistortion maps enables on-the-fly frame undistortion without HDF5 files. The `dist_coeffs` blob is already present.

**Three genuinely new tables** for the YOLO/pose-extraction pipeline (currently no equivalent in the schema):

```sql
-- Raw YOLO tracker output per camera per frame.
-- Source: pose_extraction.py Phase 1. Currently only exported as pickle cache files.
CREATE TABLE yolo_detections (
    sequence_id  TEXT NOT NULL REFERENCES pose_observation_sequences(id),
    camera_id    INTEGER NOT NULL,    -- matches camera order in active_camera_ids
    video_frame  INTEGER NOT NULL,
    track_id     INTEGER NOT NULL,    -- YOLO tracking ID (not stable across runs)
    bbox_x1      REAL NOT NULL,
    bbox_y1      REAL NOT NULL,
    bbox_x2      REAL NOT NULL,
    bbox_y2      REAL NOT NULL,
    confidence   REAL NOT NULL,
    PRIMARY KEY (sequence_id, camera_id, video_frame, track_id)
);

-- Named-person timelines assembled by the timeline stitcher.
-- Maps a person name to one or more contiguous YOLO track segments per camera.
-- Source: pose_extraction.py stitcher UI. Currently stored only in notebook state.
CREATE TABLE person_tracks (
    id              TEXT PRIMARY KEY,
    sequence_id     TEXT NOT NULL REFERENCES pose_observation_sequences(id),
    camera_id       INTEGER NOT NULL,
    person_name     TEXT NOT NULL,
    track_segments  TEXT NOT NULL    -- JSON: [[track_id, start_frame, end_frame], ...]
);

-- Thumbnail JPEG cache entries for the video frame server.
-- Allows the UI frame server to locate cached thumbnails without re-decoding.
-- Entries are derived data; can be deleted and regenerated freely.
CREATE TABLE frame_cache_entries (
    shot_video_id  TEXT NOT NULL REFERENCES shot_videos(id),
    frame_idx      INTEGER NOT NULL,
    size_px        INTEGER NOT NULL,   -- thumbnail width in pixels
    file_path      TEXT NOT NULL,
    PRIMARY KEY (shot_video_id, frame_idx, size_px)
);
```

#### No change needed

- `SyncConfig` / `SyncPoint` — already supports multiple anchor points per camera (primary key includes `video_frame` as of recent work)
- `PoseObservation` — already stores RTMpose keypoints as blobs
- `TrackingRun` / `TrackingResult` / `TrackingObsResult` — complete as-is

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

### Framework Comparison

The three realistic options are Marimo, a native web stack (FastAPI + JS frontend), and PySide6. Each is evaluated against the specific requirements of this pipeline.

---

#### Option A: Marimo

**What it is**: A Python reactive notebook framework where cells automatically re-execute when their dependencies change. Ships built-in UI widgets (`mo.ui.*`) and allows arbitrary HTML/JS via `mo.Html()`.

| Requirement | Assessment |
|---|---|
| Pipeline step UI | ✅ Reactive cells map naturally onto sequential setup steps |
| Background jobs | ⚠️ Requires explicit threading; `mo.state()` works but is not designed for it |
| Video scrubbing | ⚠️ Native rendering is matplotlib (slow). Workable via `mo.Html()` + JS calling a local frame server, but this means writing JS anyway |
| Custom annotations | ⚠️ Canvas-based annotator requires `mo.Html()` + JS. Achievable but not idiomatic |
| 3D visualization | ⚠️ Must embed Three.js via `mo.Html()` |
| Distribution | ✅ Single Python process, `marimo run` to launch |
| Python interop | ✅ Direct — no serialization layer between UI and Python data |
| Learning curve | ✅ Low; team already uses it |

**"Growing out of it" risk**: The concern is real. Marimo's reactive model is excellent for tabular/analytical UIs but video scrubbing, annotation canvases, and real-time job progress all require dropping into `mo.Html()` with raw JavaScript. At some point the app becomes a JS frontend that happens to be hosted inside Marimo, which loses most of the framework's benefit. The tipping point is roughly when more than ~20% of the UI logic lives in `mo.Html()` strings.

**Mitigation**: Keep Marimo for the analytical/configuration screens (session setup, tracker-debug, body-measurements) and use `mo.Html()` only for the video scrubber widget. This is a pragmatic split that avoids a full rewrite.

---

#### Option B: FastAPI + Web Frontend (React or Svelte)

**What it is**: A Python FastAPI backend serving a REST/WebSocket API, with a JS frontend (React or SvelteKit) running in the browser.

| Requirement | Assessment |
|---|---|
| Pipeline step UI | ✅ Full control over step-by-step wizard UI |
| Background jobs | ✅ Native async/await + `BackgroundTasks`; WebSocket for progress streaming |
| Video scrubbing | ✅ `/frame?cam=X&n=N` endpoint returns JPEG; JS `<img>` swap is trivially fast |
| Custom annotations | ✅ Canvas API in JS is first-class |
| 3D visualization | ✅ Three.js or Babylon.js natively in the browser |
| Distribution | ⚠️ Requires running two processes (FastAPI + vite dev server, or bundled); packaging for non-dev use needs extra work |
| Python interop | ⚠️ All data must be serialized to JSON/binary over HTTP; adds boilerplate |
| Learning curve | ❌ Requires JS/TS + React or Svelte + async Python; significant investment if not already familiar |

**Honest assessment**: This is the right architecture for a team product or something that needs to run in a browser for multiple users. For a single-researcher tool, the two-codebase overhead is a real cost. The main concrete advantage over Marimo is that the video/annotation/3D parts become straightforward instead of workarounds — but those parts are a minority of the total UI surface area.

**When to choose this**: If the tool will eventually be shared with collaborators who aren't running Python locally, or if the annotation/visualization components grow complex enough that maintaining them in `mo.Html()` strings becomes painful.

---

#### Option C: PySide6 (Qt)

**What it is**: Python bindings for Qt — a mature cross-platform native GUI framework. The existing `sync_videos.py` is already PyQt6, which is the same underlying library.

| Requirement | Assessment |
|---|---|
| Pipeline step UI | ✅ `QWizard` or `QStackedWidget` for multi-step setup; full widget library |
| Background jobs | ✅ `QThread` + signals/slots is Qt's native pattern; well-understood |
| Video scrubbing | ✅ `QOpenGLWidget` or `QVideoWidget` for hardware-accelerated playback; can overlay with `QPainter`; fastest of the three options |
| Custom annotations | ✅ `QPainter` on `QLabel` or `QOpenGLWidget`; well-established pattern |
| 3D visualization | ✅ OpenGL via `QOpenGLWidget`; or embed a WebView for Three.js |
| Distribution | ✅ Single Python process; `PyInstaller` for standalone exe; existing `sync_videos.py` proves it works |
| Python interop | ✅ Direct — no serialization, direct access to numpy arrays, cv2 frames, etc. |
| Learning curve | ⚠️ Qt's signal/slot model and widget layout system have a learning curve; however, `sync_videos.py` demonstrates existing familiarity |
| Data visualization | ⚠️ Matplotlib can be embedded (`FigureCanvasQtAgg`) but is not as fluid as browser-based Plotly; `pyqtgraph` is faster but less feature-rich |

**Honest assessment**: Qt is the strongest choice for the video/annotation components specifically. Frame rendering at full resolution without browser overhead, hardware-accelerated OpenGL, proper multi-threaded job management, and the existing `sync_videos.py` codebase all point toward PySide6. The weakness is data visualization — the analytical screens (joint angle plots, tracking stats, 3D scatter) are more cumbersome than in Marimo/Plotly.

**When to choose this**: If video rendering performance and annotation precision are the top priorities, and analytical visualization is secondary.

---

#### Comparison Summary

| | Marimo | FastAPI+JS | PySide6 |
|---|---|---|---|
| Video scrubbing performance | ⚠️ Workaround needed | ✅ Good | ✅ Best |
| Annotation canvas | ⚠️ JS workaround | ✅ Good | ✅ Native |
| Background GPU jobs | ⚠️ Threading workaround | ✅ Native async | ✅ QThread native |
| Analytical plots | ✅ Plotly/matplotlib | ✅ Plotly | ⚠️ Matplotlib or pyqtgraph |
| Python data access | ✅ Direct | ⚠️ Serialized | ✅ Direct |
| Existing code reuse | ✅ tracker-debug etc. | ⚠️ Rewrite | ✅ sync_videos.py |
| Two codebases? | ✅ No | ❌ Yes (Python + JS) | ✅ No |
| "Growing out of it" risk | ❌ Real, if video UI grows | ✅ Low | ✅ Low |
| Packaging/distribution | ✅ Easy | ⚠️ Moderate | ✅ PyInstaller |

---

#### Recommended hybrid

Given the requirements, a **two-app split** avoids the weaknesses of any single choice:

- **Setup & annotation app**: PySide6 — builds directly on `sync_videos.py`, handles video rendering and annotation natively, background jobs via QThread, no JS required
- **Analysis & results app**: Marimo — keeps existing `tracker-debug.py`, `body-measurements.py` cells as-is; Plotly for analytical charts; `mo.Html()` for 3D viewer

Both apps share the same SQLite DB as the data layer. This gives the best video/annotation UX without discarding the existing Marimo analytical work.

If a single app is preferred over two, **PySide6 is the stronger single-app choice** because video performance and background jobs are harder to retrofit into Marimo than embedding Matplotlib into Qt is.

---

### Core Components (framework-agnostic)

#### Video Frame Server

Needed by Marimo (as HTTP endpoint) or PySide6 (as in-process cache). Logic is the same:

```python
class FrameCache:
    """LRU cache of decoded JPEG frames, with optional on-the-fly undistortion."""

    def get_frame(self, shot_video_id: str, frame_idx: int, width_px: int,
                  undistort_maps: tuple | None = None) -> bytes:
        # 1. Check LRU (key: shot_video_id + frame_idx + width_px)
        # 2. Decode with PyAV (preferred) or cv2.VideoCapture
        # 3. Resize to width_px maintaining aspect ratio
        # 4. Apply cv2.remap(undistort_maps) if provided
        # 5. Encode as JPEG and cache
        # 6. Return bytes
```

Cache sizing: ~200 full thumbnails + ~1000 person-crop thumbnails fits comfortably in RAM.

#### Background Job Manager

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
```

Implementation: `multiprocessing.ProcessPoolExecutor` for GPU jobs, `threading.Thread` for IO-bound tasks. Progress reported via `multiprocessing.Queue`; a polling thread writes results to the DB and updates job state.

#### Inline Extrinsics Annotator

Replaces the VIA round-trip. Shows all calibration frames in a grid with a canvas overlay. User clicks corresponding points across views; each click adds a labeled dot. "Compute" runs PnP + optional bundle adjustment inline and shows reprojection overlay. Results written directly to `ExtrinsicCalibration` / `ExtrinsicEntry` tables.

---

### Undistortion Strategy

The user's instinct to work with original videos is correct. Recommended approach:

- **Store original videos** by path in `ShotVideo.file_path`; never write undistorted videos to disk permanently
- **Store undistortion maps** in new `undistort_mapx`/`undistort_mapy` columns on `IntrinsicsCalibration`
- **Apply undistortion at render time** in the frame cache (fast with pre-loaded maps)
- **For the tracker**: project 3D markers to distorted pixel space using the full camera model (`K_original + distortion`) instead of undistorting observations. This requires adding `K_original` and distortion coefficients to the tracker's `Camera` struct — significant but architecturally correct.
- **For extrinsics calibration and pose extraction**: undistort annotation/detection points mathematically (as `calibrate_extrinsics.py` already does) rather than undistorting images.

**Short-term path**: Keep the current undistorted video workflow for the tracker. Stop writing undistorted videos to a permanent location — generate them on demand into a `derived/` cache directory that can be safely deleted and regenerated from `IntrinsicsCalibration` data.

---

### Implementation Phases

**Phase 1 — DB integration for existing tools** (glue, minimal new code)
- Import intrinsics HDF5 → `IntrinsicsCalibration` table (adding undistortion map columns)
- Import project YAML → session + shot + `ShotVideo` records
- Import extrinsics TOML → `ExtrinsicCalibration` / `ExtrinsicEntry` tables (partially done via CLI)
- Import sync JSON → `SyncConfig` / `SyncPoint` (already done)
- OpenPose JSON → `PoseObservation` (already done via `import_pose_json.py`)

**Phase 2 — Unified setup app**
- Frame cache component
- Multi-video scrubber (PySide6: `QOpenGLWidget`; Marimo: `mo.Html()` + JS)
- Shot setup wizard: rough sync, LED fine sync with interactive ROI selection, inline extrinsics annotator

**Phase 3 — Background job integration**
- YOLO + RTMpose wrapped as job workers writing to `yolo_detections` and `PoseObservation`
- Timeline stitcher rewritten against `person_tracks` / `yolo_detections` tables
- Progress reporting in UI

**Phase 4 — Results visualization**
- Integrate tracker-debug and visualize_tracking capabilities
- 3D pose viewer
- Video overlay with reprojection circles

# Python application architecture

The `python/` directory contains GUI applications that share a common SQLite data layer, installed as a single `posetrak` package.  Logically there are three distinct functional areas; before publication these will be consolidated into a single entry point (`posetrak-app gui` / `posetrak-app skeleton list` / etc.).  Currently they are run as separate Python scripts.

---

## The three applications

### 1. Setup wizard — `python/app/setup/`

Manages session, camera, and calibration metadata.  A wizard-style flow guides the user through:

- Registering camera hardware (models, modes, instances) in the registry DB
- Importing intrinsics calibrations (from OpenCV ChArUco or Kalibr)
- Creating a new session and associating cameras
- Running extrinsics calibration (ArUco board, multi-camera)
- Configuring sync (clap / flash events)

**Entry point:** `python/app/setup/main.py`

### 2. Pose extraction — `python/app/pose/`

The detection and stitching pipeline.  Invoked from within the main app UI, not as a separate top-level entry point.

**What it does:**
1. Runs YOLOv11 + ByteTrack person detection on each camera's video, or SAM2 segmentation as an alternative two-pass approach
2. Runs RTMPose or VITpose++ keypoint estimation on each person crop
3. Writes detection keypoints and JPEG crop blobs to the session DB
4. Presents a stitching UI where the user assigns anonymous tracks to named persons
5. Finalises the assignment into a `pose_observation_sequence` ready for the tracker

**Primary class:** `PoseExtractionWindow` (`python/app/pose/main.py`)

Key modules:

| Module | Purpose |
|---|---|
| `detection_pipeline.py` | `DetectionPipeline` orchestrator |
| `backends_yolo.py` | `YOLOv11Detector` |
| `backends_rtmpose.py` | `RTMPoseEstimator` |
| `db_cache.py` | `DetectionBatchWriter`, `read_*/write_*` helpers, `read_observations_with_edits()` |
| `finalise.py` | `finalise_to_db()` — produces `pose_observation_sequences` |
| `filmstrip_stitcher.py` | Stitching timeline UI |
| `run_tracker.py` | `RunTrackerDialog` — launches the C++ tracker from within this app |
| `cutie_*.py` | Cutie segmentation integration for keypoint quality scoring |

### 3. Main viewer / editor — `python/app/ui/`

The primary data inspection and editing tool.  Shows a tree of sessions → captures → persons → tracking runs.  Selecting a person opens a multi-camera crop grid with keypoint overlays, time scrubber, and keypoint editing mode.

**Entry point:** `python/app/ui/main.py`
**Primary window class:** `MainWindow`

Key classes:

| Class | File | Purpose |
|---|---|---|
| `MainWindow` | `main_window.py` | QMainWindow; session tree on left, content panel on right |
| `SessionTreeWidget` | `session_tree.py` | Tree: sessions → captures → persons |
| `PersonPanel` | `content_panels.py` | Panel shown when a person node is selected; contains the crop grid and info pane |
| `PersonCropGridWidget` | `content_panels.py` | Multi-camera crop grid; keypoint overlays; edit mode |
| `CropBackfillWorker` | `content_panels.py` | QThread that generates crops for ghost frames (no detection) |

---

## Shared data layer — `python/posetrak/db/`

All three apps read and write the same two SQLite files through a common DB layer:

| Module | Purpose |
|---|---|
| `db.py` | `open_registry()`, `open_session()`, schema migrations |
| `session_reader.py` | Convenience reads for session structure |
| `load_session.py` | Loads pose observations for analysis scripts |
| `import_*.py` | Importers for YAML, JSON, TOML, H5 calibration files |
| `manage_config.py` | CRUD for tracker configs |
| `manage_skeleton.py` | Import and version skeleton YAML files |

Install with `pip install -e python/` (or `uv pip install -e python/`).

---

## Code layout

```
python/
├── posetrak/               installable package
│   └── db/                 DB layer (above)
│
├── app/
│   ├── ui/                 Main viewer / editor
│   │   ├── main.py
│   │   ├── main_window.py
│   │   ├── content_panels.py   PersonPanel, PersonCropGridWidget, …
│   │   └── session_tree.py
│   │
│   ├── pose/               Pose extraction app
│   │   ├── main.py
│   │   ├── detection_pipeline.py
│   │   ├── backends_yolo.py
│   │   ├── backends_rtmpose.py
│   │   ├── db_cache.py
│   │   ├── finalise.py
│   │   ├── filmstrip_stitcher.py
│   │   ├── run_tracker.py
│   │   └── cutie_*.py
│   │
│   └── setup/              Setup wizard
│       ├── main.py
│       └── page_*.py
│
├── analysis/               Marimo analysis notebooks
├── tools/                  Standalone utility scripts (param_sweep.py, etc.)
└── tests/                  pytest suite; run with: uv run pytest python/tests/
```

---

## Running the apps

```bash
# Install dependencies and package in editable mode
uv pip install -e python/

# Main app (includes pose extraction)
uv run python python/app/ui/main.py

# Setup wizard
uv run python python/app/setup/main.py

# Tests
uv run pytest python/tests/
```

---

## Ghost frame crops

Frames where the person was not detected by YOLO have no bbox in `person_detections`.  The `CropBackfillWorker` synthesises a crop for these *ghost frames* by using the union of the nearest bboxes before and after.  These synthetic crops are cached in `frame_cache_entries` with `cache_type='ghost_crop'`, `track_id=-1`, `detection_run_id=''` so they are loaded from DB on subsequent views rather than regenerated.

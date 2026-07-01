# Python application architecture

The `python/` directory contains GUI applications that share a common SQLite data layer, installed as a single `posetrak` package via `uv sync`.

---

## The three applications

### 1. Setup wizard — `python/app/setup/`

Manages session, camera, and calibration metadata.  A wizard-style flow guides the user through:

- Registering camera hardware (models, modes, instances) in the registry DB
- Importing intrinsics calibrations (from OpenCV ChArUco or Kalibr)
- Creating a new session and associating cameras
- Running extrinsics calibration (ArUco board, multi-camera)
- Configuring sync (clap / flash events)

**Entry point:** `uv run posetrak-setup` (`python/app/setup/main.py`)

### 2. Pose extraction — `python/app/pose/`

The detection and stitching pipeline.  Can be launched standalone or opened from within the main app.

**What it does:**
1. Runs YOLOv11 + ByteTrack person detection on each camera's video, or SAM2 / Cutie segmentation as an alternative two-pass approach
2. Runs RTMPose or VITpose++ keypoint estimation on each person crop
3. Writes detection keypoints and JPEG crop blobs to the session DB
4. Presents a stitching UI (`StandaloneRunPanel`) where the user assigns anonymous tracks to named persons
5. Finalises the assignment into a `pose_observation_sequence` ready for the tracker

**Entry point:** `uv run posetrak-pose` (`python/app/pose/main.py` → `PoseExtractionWindow`)

Key modules:

| Module | Purpose |
|---|---|
| `detection_pipeline.py` | `DetectionPipeline` orchestrator |
| `backends_yolo.py` | `YOLOv11Detector` |
| `backends_rtmpose.py` | `RTMPoseEstimator` |
| `db_cache.py` | `DetectionBatchWriter`, `read_*/write_*` helpers, `read_observations_with_edits()` |
| `finalise.py` | `finalise_to_db()` — produces `pose_observation_sequences` |
| `stitcher_panel.py` | `StitcherPanel` — the track-to-person assignment grid |
| `run_tracker.py` | `RunTrackerDialog` — launches the C++ tracker subprocess |
| `cutie_*.py` | Cutie video segmentation integration |

### 3. Main viewer / editor — `python/app/ui/`

The primary data inspection and editing tool.  Shows a tree of sessions → captures → trials → detection runs / persons on the left; the right panel changes based on what is selected.

**Entry point:** `uv run posetrak-ui` (`python/app/ui/main.py` → `MainWindow`)

Key classes:

| Class | File | Purpose |
|---|---|---|
| `MainWindow` | `main_window.py` | QMainWindow; session tree on left, content panel on right |
| `SessionTreeWidget` | `session_tree.py` | Tree: sessions → captures → trials → detection runs / persons |
| `CapturePanel` | `content_panels.py` | Shown when a capture is selected; "New trial…" button |
| `TrialPanel` | `content_panels.py` | Shown when a trial is selected; lists detection runs and tracking runs, segmentation button |
| `StandaloneRunPanel` | `content_panels.py` | Shown when a detection run is selected; track-to-person assignment editor |
| `PersonPanel` | `content_panels.py` | Shown when a person node is selected; multi-camera crop grid with keypoint overlays and edit mode |
| `PersonCropGridWidget` | `content_panels.py` | The crop grid widget inside `PersonPanel` |
| `CropBackfillWorker` | `content_panels.py` | QThread that generates crops for ghost frames (no detection) |

#### Navigation flow

```
Session tree click
    capture  →  CapturePanel   (New trial… button opens _NewTrialDialog)
    trial    →  TrialPanel     (Run detection… → RunDetectionDialog; double-click lists navigate)
    det run  →  StandaloneRunPanel   (track assignment, Save assignments)
    person   →  PersonPanel    (crop grid, keypoint editing, tracking run selector)
```

The main window wires `TrialPanel.navigate_detection` and `TrialPanel.navigate_tracking` signals to swap to `StandaloneRunPanel` or the tracking run view respectively.

---

## Shared data layer — `python/posetrak/`

All three apps read and write the same two SQLite files through a common DB layer:

| Module | Purpose |
|---|---|
| `db/db.py` | `open_registry()`, `open_session()`, schema migrations |
| `db/session_reader.py` | Convenience reads for session structure |
| `db/load_session.py` | Loads pose observations for analysis scripts |
| `db/import_*.py` | Importers for YAML, JSON, TOML, H5 calibration files |
| `db/manage_config.py` | CRUD for tracker configs |
| `db/manage_skeleton.py` | Import and version skeleton YAML files |
| `cli/` | `posetrak-db` CLI: `trial`, `video`, `capture`, `export`, `import`, … |

---

## Code layout

```
python/
├── posetrak/               installable package
│   ├── db/                 DB layer (above)
│   ├── cli/                CLI commands (posetrak-db entry point)
│   └── data/skeletons/     bundled default skeleton YAMLs
│
├── app/
│   ├── ui/                 Main viewer / editor
│   │   ├── main.py
│   │   ├── main_window.py
│   │   ├── content_panels.py   CapturePanel, TrialPanel, StandaloneRunPanel,
│   │   │                       PersonPanel, PersonCropGridWidget, …
│   │   └── session_tree.py
│   │
│   ├── pose/               Pose extraction app
│   │   ├── main.py         PoseExtractionWindow
│   │   ├── detection_pipeline.py
│   │   ├── backends_yolo.py
│   │   ├── backends_rtmpose.py
│   │   ├── db_cache.py
│   │   ├── finalise.py
│   │   ├── stitcher_panel.py
│   │   ├── run_tracker.py
│   │   └── cutie_*.py
│   │
│   ├── setup/              Setup wizard
│   │   ├── main.py
│   │   └── page_*.py
│   │
│   ├── analysis/           Marimo analysis scripts
│   └── mcp/                Read-only MCP diagnostic server (posetrak-mcp)
│
├── pipeline/               Standalone pipeline tools (calibration, pose extraction helpers)
├── tools/                  Utility scripts (param_sweep.py, etc.)
└── tests/                  pytest suite
```

---

## Running the apps

```bash
uv sync                       # install / update dependencies
uv run posetrak-ui            # main viewer / editor
uv run posetrak-pose          # pose extraction pipeline
uv run posetrak-setup         # setup wizard
uv run posetrak-db --help     # database CLI

uv run pytest python/tests/   # run Python tests
```

---

## Ghost frame crops

Frames where the person was not detected by YOLO have no bbox in `person_detections`.  The `CropBackfillWorker` synthesises a crop for these *ghost frames* by using the union of the nearest bboxes before and after.  These synthetic crops are cached in `frame_cache_entries` with `cache_type='ghost_crop'`, `track_id=-1`, `detection_run_id=''` so they are loaded from DB on subsequent views rather than regenerated.

# Python application architecture

The `python/` directory contains GUI and CLI applications that share a common SQLite data layer, installed as a single `posetrak` package via `uv sync`.

There are two primary entry points:

| Command | Entry point | Purpose |
|---|---|---|
| `uv run posetrak-ui` | `app.ui.main:main` | GUI — main viewer, editor, and pose extraction |
| `uv run posetrak` | `posetrak.cli.main:main` | CLI — session, trial, capture, video, export/import management |

---

## `posetrak-ui` — GUI application

The single GUI entry point covers everything a user does interactively:

- **Setup** — camera registration, intrinsics, extrinsics, sync configuration
- **Pose extraction** — YOLO/SAM2 detection, Cutie segmentation, RTMPose keypoint estimation, track-to-person stitching
- **Inspection and editing** — multi-camera crop grid, keypoint editing, segmentation correction
- **Tracker invocation** — launching `posetrak-tracker` as a subprocess and viewing results

**Entry point:** `python/app/ui/main.py` → `MainWindow`

### Application structure

The main window shows a session tree on the left (sessions → captures → trials → detection runs / persons); the right panel changes based on selection:

| Class | File | Purpose |
|---|---|---|
| `MainWindow` | `main_window.py` | QMainWindow; session tree + content panel |
| `SessionTreeWidget` | `session_tree.py` | Tree: sessions → captures → trials → detection runs / persons |
| `CapturePanel` | `content_panels.py` | Selected capture; "New trial…" button |
| `TrialPanel` | `content_panels.py` | Selected trial; detection run list, tracking run list, segmentation button |
| `StandaloneRunPanel` | `content_panels.py` | Selected detection run; track-to-person assignment editor |
| `PersonPanel` | `content_panels.py` | Selected person; multi-camera crop grid, keypoint overlays, edit mode |
| `PersonCropGridWidget` | `content_panels.py` | Crop grid inside `PersonPanel` |
| `CropBackfillWorker` | `content_panels.py` | QThread: synthesises crops for ghost frames (no detection) |

#### Navigation flow

```
Session tree click
    capture  →  CapturePanel       (New trial… → _NewTrialDialog)
    trial    →  TrialPanel         (Run detection… → RunDetectionDialog;
                                    double-click rows → navigate)
    det run  →  StandaloneRunPanel (track assignment, Save assignments)
    person   →  PersonPanel        (crop grid, keypoint editing, tracking run selector)
```

### Pose extraction modules — `python/app/pose/`

| Module | Purpose |
|---|---|
| `detection_pipeline.py` | `DetectionPipeline` orchestrator |
| `backends_rtmpose.py` | `RTMPoseEstimator` (re-exports `posetrak.detection.backends_rtmpose`) |
| `db_cache.py` | `DetectionBatchWriter`, read/write helpers, `read_observations_with_edits()` |
| `finalise.py` | `finalise_to_db()` — produces `pose_observation_sequences` |
| `stitcher_panel.py` | `StitcherPanel` — the track-to-person assignment grid |
| `run_tracker.py` | `RunTrackerDialog` — launches `posetrak-tracker` subprocess |
| `cutie_*.py` | Cutie video segmentation integration |

The person-detector implementation itself (`YOLOXDetector`, rtmlib's YOLOX +
a lightweight IoU tracker) lives in `python/posetrak/detection/backends_rtmdet.py`,
shared with the `posetrak-pose`/`posetrak` CLIs rather than being GUI-specific.

### Setup wizard modules — `python/app/setup/`

Wizard pages for camera hardware, intrinsics, extrinsics, and sync.  Entry via `page_*.py` modules orchestrated from `setup/main.py`.

---

## `posetrak` — CLI

The CLI (`posetrak.cli.main:main`, also aliased as `posetrak-db` for backwards compatibility) manages session data without the GUI.

Key command groups:

| Command | Purpose |
|---|---|
| `posetrak session` | Create sessions, import a project YAML, clone camera intrinsics from another session/registry (`add-camera`) |
| `posetrak trial` | Create, list, show trials; extract per-camera video clips for a time range (`export-video`) |
| `posetrak capture` | List and show captures |
| `posetrak video` | List, locate, and relocate video files |
| `posetrak export` | Export a trial to a portable archive |
| `posetrak import` | Import a trial archive into a session |

---

## Shared data layer — `python/posetrak/`

| Module | Purpose |
|---|---|
| `db/db.py` | `open_registry()`, `open_session()`, schema migrations |
| `db/session_reader.py` | Convenience reads for session structure |
| `db/load_session.py` | Loads pose observations for analysis scripts |
| `db/import_*.py` | Importers for YAML, JSON, TOML, H5 calibration files |
| `db/manage_config.py` | CRUD for tracker configs |
| `db/manage_skeleton.py` | Import and version skeleton YAML files |

---

## Code layout

```
python/
├── posetrak/               installable package
│   ├── db/                 DB layer (above)
│   ├── cli/                CLI commands (posetrak entry point)
│   └── data/skeletons/     bundled default skeleton YAMLs
│
├── app/
│   ├── ui/                 Main viewer / editor (posetrak-ui)
│   │   ├── main.py
│   │   ├── main_window.py
│   │   ├── content_panels.py
│   │   └── session_tree.py
│   │
│   ├── pose/               Pose extraction (embedded in posetrak-ui)
│   │   ├── detection_pipeline.py
│   │   ├── backends_rtmpose.py
│   │   ├── db_cache.py
│   │   ├── finalise.py
│   │   ├── stitcher_panel.py
│   │   ├── run_tracker.py
│   │   └── cutie_*.py
│   │
│   ├── setup/              Setup wizard (embedded in posetrak-ui)
│   │   └── page_*.py
│   │
│   ├── analysis/           Marimo analysis scripts
│   └── mcp/                Read-only MCP diagnostic server (posetrak-mcp)
│
├── pipeline/               Standalone pipeline tools
├── tools/                  Utility scripts (param_sweep.py, etc.)
└── tests/                  pytest suite
```

---

## Running

```bash
uv sync                        # install / update dependencies
uv run posetrak-ui             # GUI (viewer, editor, pose extraction, setup)
uv run posetrak --help         # CLI

uv run pytest python/tests/    # Python tests
```

---

## Ghost frame crops

Frames where the person was not detected have no bbox in `person_detections`.  `CropBackfillWorker` synthesises a crop for these *ghost frames* using the union of the nearest bboxes before and after.  Synthetic crops are cached in `frame_cache_entries` with `cache_type='ghost_crop'`, `track_id=-1`, `detection_run_id=''`.

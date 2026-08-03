# Capture Pipeline: Current Process & Architecture Proposal

---

## Phase 0: Repository consolidation and layout restructuring

### Goals

- Consolidate all project code into this repository (posetrak); eliminate the dependency on scripts scattered across pose2sim-preprocess and rtmlib repos for day-to-day use
- Establish a directory layout that clearly separates C++ tracker code, Python package code, user-facing applications, and pipeline tools
- Make the Python package installable as `posetrak` via `pip install -e .` so all internal imports are stable regardless of working directory

### Current layout problems

| Problem | Current state |
|---|---|
| Python code has no consistent package root | `scripts/db/` imported as `scripts.db`, which only works when CWD is repo root and path is manually added to `sys.path` |
| No clear home for pipeline tools | `calibrate_intrinsics.py`, `sync_videos.py`, `pose_extraction.py` live in other repos |
| `scripts/` is a flat dump | DB layer, analysis modules, ad-hoc utilities, and shell scripts all mixed together |
| Analysis notebooks unstructured | `notebooks/` holds a mix of work-in-progress, debug, and production notebooks with no organisation |
| PySide6 app has no home | No directory exists yet |
| Python tests buried under C++ tests | `tests/python/` sits under `tests/` alongside C++ `.cpp` files |

### Proposed directory layout

```
posetrak/                    ← repo root
├── src/                     ← C++ library source        (UNCHANGED)
├── include/                 ← C++ public headers        (UNCHANGED)
├── cli/                     ← C++ posetrak CLI          (UNCHANGED)
├── db/                      ← SQL schema files          (UNCHANGED)
├── docs/                    ← Documentation             (UNCHANGED)
├── tests/                   ← C++ tests                 (UNCHANGED — .cpp files + data/)
├── meson.build              ← Root Meson build          (UNCHANGED)
│
├── python/                  ← NEW: all Python code lives here
│   ├── posetrak/            ← Installable Python package (import as `posetrak`)
│   │   ├── __init__.py
│   │   ├── db/              ← DB layer  (MOVED from scripts/db/)
│   │   │   ├── __init__.py
│   │   │   ├── db.py            (renamed from posetrak_db.py)
│   │   │   ├── cli.py           (renamed from posetrak_db_cli.py)
│   │   │   ├── import_pose_json.py
│   │   │   ├── import_sync_json.py
│   │   │   ├── import_calib_toml.py
│   │   │   ├── import_extrinsics.py
│   │   │   ├── manage_config.py
│   │   │   ├── manage_skeleton.py
│   │   │   ├── load_session.py
│   │   │   └── skeleton_layout.py
│   │   └── analysis/        ← Shared analysis helpers (future: FK, visualisation utilities)
│   │       └── __init__.py
│   │
│   ├── app/                 ← User-facing applications
│   │   ├── setup/           ← NEW: PySide6 capture-setup application
│   │   │   └── __init__.py
│   │   └── analysis/        ← Marimo analysis notebooks  (MOVED from notebooks/)
│   │       ├── tracker_debug.py       (renamed from tracker-debug.py)
│   │       ├── body_measurements.py   (renamed from body-measurements.py)
│   │       └── ...
│   │
│   ├── pipeline/            ← Capture pipeline tools  (COPIED from other repos)
│   │   ├── calibration/     ← Intrinsics + extrinsics + project setup
│   │   │   ├── __init__.py
│   │   │   ├── calibrate_intrinsics.py    (from pose2sim-preprocess)
│   │   │   ├── calibrate_extrinsics.py    (from pose2sim-preprocess)
│   │   │   ├── setup_project.py           (from setup_pose2sim_project.py)
│   │   │   ├── sync_videos.py             (from pose2sim-preprocess)
│   │   │   └── inspect_calibration.py     (from pose2sim-preprocess)
│   │   └── pose/            ← Person detection + pose extraction
│   │       ├── __init__.py
│   │       ├── poseanalysis.py            (from rtmlib/harritests — YOLO/RTMpose library)
│   │       ├── pose_extraction.py         (from rtmlib/harritests — Marimo app)
│   │       ├── video_sync.py              (from rtmlib/harritests — Marimo app)
│   │       └── export_to_openpose.py      (from rtmlib/harritests)
│   │
│   ├── tools/               ← Standalone utility scripts  (MOVED from scripts/)
│   │   ├── visualize_tracking.py
│   │   ├── export_bvh.py
│   │   ├── process_session.py
│   │   ├── calibrate_scale.py
│   │   └── ...              (other scripts from scripts/)
│   │
│   └── tests/               ← Python tests  (MOVED from tests/python/)
│       ├── __init__.py
│       └── db/              (moved from tests/python/db/)
│           ├── test_load_session.py
│           ├── test_skeleton_layout.py
│           └── test_import_pose_json.py
│
├── pyproject.toml           ← Updated for new package layout
└── CLAUDE.md                ← Updated paths and instructions
```

### What is copied vs. what stays in the original repos

**Copied into posetrak** (original repo may continue to exist for other purposes):

| Source file | Destination | Notes |
|---|---|---|
| `pose2sim-preprocess/calibrate_intrinsics.py` | `python/pipeline/calibration/` | No changes needed initially |
| `pose2sim-preprocess/calibrate_extrinsics.py` | `python/pipeline/calibration/` | No changes needed initially |
| `pose2sim-preprocess/setup_pose2sim_project.py` | `python/pipeline/calibration/setup_project.py` | Will be superseded in Phase 2 |
| `pose2sim-preprocess/sync_videos.py` | `python/pipeline/calibration/` | Will grow into Phase 2 setup app |
| `pose2sim-preprocess/inspect_calibration.py` | `python/pipeline/calibration/` | Utility, copy as-is |
| `rtmlib/harritests/poseanalysis.py` | `python/pipeline/pose/` | Core YOLO/RTMpose library code |
| `rtmlib/harritests/pose_extraction.py` | `python/pipeline/pose/` | Marimo app, copy as-is |
| `rtmlib/harritests/video_sync.py` | `python/pipeline/pose/` | Marimo app, copy as-is |
| `rtmlib/harritests/export_to_openpose.py` | `python/pipeline/pose/` | Utility, copy as-is |

**Not copied** (out of scope or obsolete):
- `pose2sim-preprocess/calibrate.py` — older version; superseded by `calibrate_intrinsics.py`
- `pose2sim-preprocess/test_*.py` — not relevant to posetrak
- `rtmlib/harritests/skeletons.py` — rtmlib-specific skeleton definitions, not used here
- Old `.ipynb` notebooks in rtmlib — replaced by the Marimo `pose_extraction.py`

### File moves within the repo

| From | To |
|---|---|
| `scripts/db/` | `python/posetrak/db/` |
| `scripts/db/posetrak_db.py` | `python/posetrak/db/db.py` |
| `scripts/db/posetrak_db_cli.py` | `python/posetrak/db/cli.py` |
| `notebooks/tracker-debug.py` | `python/app/analysis/tracker_debug.py` |
| `notebooks/body-measurements.py` | `python/app/analysis/body_measurements.py` |
| `notebooks/calibrate-scale-debug.py` | `python/app/analysis/calibrate_scale_debug.py` |
| `notebooks/key-measurements.py` | `python/app/analysis/key_measurements.py` |
| `scripts/visualize_tracking.py` | `python/tools/visualize_tracking.py` |
| `scripts/export_bvh.py` | `python/tools/export_bvh.py` |
| `scripts/process_session.py` | `python/tools/process_session.py` |
| `scripts/calibrate_scale.py` | `python/tools/calibrate_scale.py` |
| `scripts/bisect_compare.py` | `python/tools/bisect_compare.py` |
| `tests/python/` | `python/tests/` |

Files to **delete** (artefacts that should not be in version control):
- `tracking_output/` — tracker output directory, should be in `.gitignore`
- `tracking_tests/` — test run outputs, should be in `.gitignore`
- `tmp.txt`
- Root-level `.csv`, `.log`, `.rrd` files (artefacts from old tracker runs)
- `check_rerun_api.py`, `test_rerun_api.py` — old rerun experiments (check first)

### pyproject.toml changes

```toml
[project]
name = "posetrak"
version = "0.1.0"
requires-python = ">=3.12"

dependencies = [
    # Existing
    "opencv-python>=4.13",
    "pandas>=2.0",
    "toml>=0.10",
    "numpy>=1.26",
    "scipy>=1.12",
    # DB layer
    "h5py>=3.10",          # intrinsics HDF5 import
    # Pipeline tools
    "click>=8.1",
    "pyyaml>=6.0",
    # UI - installed separately to avoid forcing GPU dependencies on CI
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]
analysis = [
    "marimo>=0.19",
    "plotly>=5.0",
]
setup-app = [
    "PySide6>=6.7",
]
pipeline = [
    "h5py>=3.10",
    "rtmlib",           # GPU-heavy: pulls in ultralytics, onnxruntime-gpu, etc.
    "ultralytics",      # YOLO11 (transitive via rtmlib but pinned separately for clarity)
    "ipycanvas",        # required by poseanalysis.py (Jupyter-era widget)
    "ipywidgets",       # required by poseanalysis.py
]
# NOTE: the pipeline group is never installed in CI. Developers running pose
# extraction install it explicitly: `uv sync --group pipeline`.
# rtmlib must be installed from a local clone or git URL since it is not on PyPI.

[tool.setuptools.packages.find]
where = ["python"]

[tool.pytest.ini_options]
testpaths = ["python/tests"]
```

### Import path changes

All internal `from scripts.db.X import Y` becomes `from posetrak.db.X import Y`.
All internal `sys.path.insert(0, project_root)` hacks in notebooks are removed — the package is installed (`pip install -e .`) so imports work from anywhere.

The `posetrak_db` CLI entry point changes from `python scripts/db/posetrak_db_cli.py` to just `posetrak-db` (registered as a console script):

```toml
[project.scripts]
posetrak-db = "posetrak.db.cli:main"
```

### CLAUDE.md changes

- Update all script paths to `python/` subtree
- Update test command: `pytest python/tests/`
- Document `pip install -e .` as required setup step
- Update Python import conventions

### Acceptance criteria for Phase 0

1. All listed files exist at their new locations
2. `pip install -e .` succeeds
3. `from posetrak.db.db import create_session` works in a fresh Python shell
4. `pytest python/tests/` passes (all tests that passed before still pass)
5. `posetrak-db --help` works from any directory
6. Marimo notebooks (`python/app/analysis/*.py`) open without import errors
7. No stale `scripts/` or `notebooks/` directories remain
8. `.gitignore` updated to exclude tracker output files

---

## Phase 1: DB integration for existing pipeline tools

### Goals

- Bring all pipeline data (intrinsics, project structure, sync, extrinsics) into the posetrak SQLite DB so downstream steps (pose import, tracking, analysis) can access everything through one interface
- Extend the `IntrinsicsCalibration` schema to store undistortion maps, eliminating the HDF5 files as a required runtime dependency
- Create import commands in the `posetrak-db` CLI for each data source

### 1a. Schema extension: `IntrinsicsCalibration`

The current `IntrinsicsCalibration` schema stores scalar `fx`, `fy`, `cx`, `cy` (the undistorted K matrix components) plus a `dist_coeffs` blob. Three additions are needed to make it self-sufficient:

```sql
-- Registry DB: extend intrinsic_calibrations table
ALTER TABLE intrinsic_calibrations ADD COLUMN matrix_original  BLOB;
    -- 3×3 float64, row-major: K directly from calibrateCamera()
    -- Needed to project distorted observations back to camera space

ALTER TABLE intrinsic_calibrations ADD COLUMN undistort_mapx   BLOB;
    -- float32 array, shape (height, width), zlib-compressed
    -- cv2.remap mapx: for each undistorted pixel, the source x in distorted space

ALTER TABLE intrinsic_calibrations ADD COLUMN undistort_mapy   BLOB;
    -- float32 array, shape (height, width), zlib-compressed
```

These columns are nullable — existing rows without maps are still valid for use cases that only need K and distortion coefficients. The maps are only required for on-the-fly frame undistortion in the frame server.

**Migration**: Since the registry is not yet in production deployment, `ALTER TABLE` in a migration script is sufficient. A `db/migrations/001_intrinsics_maps.sql` file documents the change.

### 1b. CLI command: `posetrak-db calib import-h5`

Reads a calibration HDF5 file produced by `calibrate_intrinsics.py` and inserts or updates an `IntrinsicsCalibration` row.

**Command signature**:
```
posetrak-db calib import-h5 <h5_file>
    --registry <path>
    --camera-model <id-or-prefix>
    --camera-mode <id-or-prefix>
    [--notes <text>]
    [--no-maps]          # skip storing undistortion maps (saves ~10 MB per camera)
```

**HDF5 → DB field mapping**:

| HDF5 field | DB column |
|---|---|
| `intrinsics/matrix` (3×3) | `matrix_original` (blob) + scalar `fx, fy, cx, cy` from `matrix_undistorted` |
| `intrinsics/matrix_undistorted` (3×3) | `fx, fy, cx, cy` |
| `intrinsics/distortions` (1D) | `dist_coeffs` |
| `intrinsics/size` attr | `image_width`, `image_height` (new cols needed, or fold into existing schema) |
| `intrinsics/model_type` attr | `distortion_model` (`"standard"` → `"radtan"`, `"fisheye"` → `"fisheye"`) |
| `intrinsics/error` attr | `rms_error` |
| `undistortion_maps/mapx` | `undistort_mapx` (compressed) |
| `undistortion_maps/mapy` | `undistort_mapy` (compressed) |
| `camera_name` attr | used to look up `camera_model_id` |
| `camera_mode` attr | used to look up `camera_mode_id` |

Map compression: `zlib.compress(mapx.astype(np.float32).tobytes(), level=6)` — maps are ~8 MB uncompressed per 4K camera, compress to ~3 MB.

**Behaviour on re-import**: if a row already exists for the `(camera_mode_id)` and the content is identical (checked via `rms_error` + matrix values), skip silently. If different, insert a new row (calibration history is preserved; the new one becomes the "latest").

### 1c. CLI command: `posetrak-db session import-yaml`

Reads a project YAML file (output of `sync_videos.py`, manually extended with scene and intrinsics info) and creates the corresponding DB records.

**Command signature**:
```
posetrak-db session import-yaml <project_yaml>
    --session-db <path>    # session DB to write into (creates if not exists)
    --registry <path>
    [--session-label <text>]
    [--dry-run]            # print what would be created without writing
```

**YAML → DB mapping**:

```yaml
# project.yaml structure
name: "mocap-test-setup"
path: "c:/temp/mocap-test-setup"
ref_camera: "cam1"
cameras:
  - name: "cam1"
    path: "M:/videos/cam1.mp4"
    sync_frame: "00:01:07.25"       # rough sync only (becomes SyncPoint with LED sync later)
    fps: 120.0
    calib:
      intrinsics:
        h5: "calibration.h5"        # path to HDF5 (extended YAML field)
      extrinsics:
        frame: "00:00:33.15"        # video frame used for extrinsics annotation
scenes:
  - name: "scene1"
    start_frame: "00:01:11.21"
    end_frame: "00:01:19.21"
```

| YAML field | DB record |
|---|---|
| `name` | `MocapSession.notes` (or session label) |
| `cameras[].name` | `CameraInstance.label` |
| `cameras[].path` | `ShotVideo.file_path` |
| `cameras[].fps` | `ShotVideo.actual_fps` |
| `cameras[].sync_frame` | `SyncPoint.video_frame` + `SyncPoint.timestamp_s = 0` (rough sync; one point per camera in a new `SyncConfig`) |
| `cameras[].calib.intrinsics.h5` | looked up in `IntrinsicsCalibration` by camera model/mode; `SessionCamera.intrinsics_calibration_id` set |
| `cameras[].calib.extrinsics.frame` | stored as `ShotVideo`-level annotation frame; used in Phase 2 extrinsics annotator |
| `scenes[].name` | `Shot.label` |
| `scenes[].start_frame` (per-camera timecode) | `ShotVideo.first_video_frame` (converted from timecode using camera fps + sync_frame offset) |
| `scenes[].end_frame` | `ShotVideo.last_video_frame` |

**Resolution of camera instances**: a camera named `"cam1"` in the YAML is looked up in the registry by label. If no match, the command prompts (or `--dry-run` reports) which `CameraInstance` to use, or creates a stub if `--create-instances` is passed.

**Sync config creation**: the rough sync frames from the YAML are inserted as a `SyncConfig` with `created_by = "yaml-import-rough"` and `notes = "coarse sync from project.yaml sync_frame fields"`. This is later replaced or supplemented by the LED fine sync output.

**Idempotency**: running the command twice on the same YAML does not create duplicate sessions. A session is identified by `(yaml_path hash, recorded_at)`; if it already exists, the command reports that and exits unless `--force` is given.

### 1d. CLI command: `posetrak-db session import-extrinsics-toml`

Already partially implemented as `extrinsics import` in the current CLI. Needs:
- Accept the extended Pose2Sim TOML that `calibrate_extrinsics.py` produces
- Accept `--shot <id-or-prefix>` to associate the calibration with a specific shot (rather than floating at session level)
- Create an `ExtrinsicCalibration` row and one `ExtrinsicEntry` per camera

This command already works for the basic case; the main gap is the `--shot` association.

### 1e. CLI command: `posetrak-db sync import` (rename + verify)

The command already exists as `posetrak-db sync import --sync-json <file>`. The JSON format from `video_sync.py` is the canonical format; no separate "led" variant is needed since the sync data format is the same regardless of how the sync was determined (LED detection, audio click, manual frame matching, etc.).

The JSON format `video_sync.py` produces:
```json
{
  "cam1": { "fps": 120, "syncpoints": [{"frame": 1234, "timestamp": 10.28}, ...] },
  "cam2": { "fps": 60,  "syncpoints": [{"frame": 617,  "timestamp": 10.28}, ...] }
}
```

**Action for Phase 1**: verify the existing `sync import` command handles multiple sync points per camera correctly (the `SyncPoint` primary key was extended to include `video_frame` specifically for this). Add a `--notes` flag to record the sync method (e.g. `"LED detection"`, `"manual"`) on the `SyncConfig` row.

### 1f. Existing commands: verify and document gaps

| Command | Status | Gap |
|---|---|---|
| `posetrak-db skeleton import` | ✅ Done | — |
| `posetrak-db extrinsics import` | ✅ Done | Needs `--shot` association |
| `posetrak-db sync import` | ✅ Done | Verify multi-point support |
| `posetrak-db pose import` | ✅ Done | — |
| `posetrak-db session create` | ✅ Done | — |
| `posetrak-db shot create` | ✅ Done | — |
| `posetrak-db calib import-h5` | ❌ New | Phase 1b above |
| `posetrak-db session import-yaml` | ❌ New | Phase 1c above |

### 1g. Refactor `poseanalysis.py` to remove Jupyter dependencies

`poseanalysis.py` currently imports `ipycanvas`, `ipywidgets`, and `IPython` at module level. These are Jupyter notebook widgets that are unused by `pose_extraction.py` (which is a Marimo app with its own UI). They exist only because `poseanalysis.py` was originally written as part of a Jupyter workflow.

**Task**: audit which symbols `pose_extraction.py` imports from `poseanalysis.py`, then split the file:

- `python/pipeline/pose/poseanalysis.py` — keep only the algorithm functions used by the Marimo app (YOLO tracking, RTMpose inference, `NamedPersonTimeline`, `VideoData`, `MultiVideoPoseDataset`, frame-reading helpers). No Jupyter imports.
- The Jupyter-facing widgets code can be deleted entirely — it is not used in the current workflow.

This allows removing `ipycanvas` and `ipywidgets` from `[dependency-groups.pipeline]`.

**Acceptance**: `from posetrak.pipeline.pose.poseanalysis import analyze_video_with_yolo_tracker` succeeds in an environment that has `rtmlib` and `ultralytics` but not `ipycanvas`.

### 1h. Schema: `image_width` / `image_height` on `IntrinsicsCalibration`

The current schema has no explicit image size on `IntrinsicsCalibration`. Size is implied by the undistortion maps (if present) or stored in the HDF5. During Phase 1 import, add:

```sql
ALTER TABLE intrinsic_calibrations ADD COLUMN image_width  INTEGER;
ALTER TABLE intrinsic_calibrations ADD COLUMN image_height INTEGER;
```

### Acceptance criteria for Phase 1

1. `posetrak-db calib import-h5 calibration.h5 --registry r.db --camera-model ... --camera-mode ...` creates a valid `IntrinsicsCalibration` row with maps
2. `posetrak-db session import-yaml project.yaml --session-db s.db --registry r.db` creates: one `MocapSession`, `SessionCamera` rows, `ShotVideo` rows per scene × camera, one rough `SyncConfig`, and `Shot` rows
3. `python -c "from posetrak.db.db import open_registry; r = open_registry('r.db')"` works after `pip install -e .` (verifies Phase 0 package layout)
4. Existing tests still pass
5. A real session from the current `mocap/` archive can be round-tripped: import YAML + HDF5 + LED sync JSON + extrinsics TOML → session DB → `load_tracking_run_with_markers()` returns data equivalent to what the CSV-based path returned
6. `from posetrak.pipeline.pose.poseanalysis import analyze_video_with_yolo_tracker` succeeds without `ipycanvas` or `ipywidgets` installed

---

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
-- Raw person detector output per camera per frame.
-- Named generically (not "yolo_detections") because the detection model may change;
-- the schema is model-agnostic — any bbox-producing tracker can write here.
-- Keyed on shot_video_id (one video = one camera for one shot), not sequence_id,
-- because detection runs before a pose_observation_sequence is created.
-- region_type supports multiple crop types per track (full body, left hand, face, etc.)
-- so that specialised downstream trackers (hand tracker, face tracker) can each get
-- their own tight crop without requiring a separate detection pass.
CREATE TABLE person_detections (
    shot_video_id  TEXT NOT NULL REFERENCES shot_videos(id),
    video_frame    INTEGER NOT NULL,
    track_id       INTEGER NOT NULL,    -- detector tracking ID (not stable across runs)
    region_type    TEXT NOT NULL DEFAULT 'full_body',
                                        -- 'full_body' | 'face' | 'hand_l' | 'hand_r' | ...
    bbox_x1        REAL NOT NULL,
    bbox_y1        REAL NOT NULL,
    bbox_x2        REAL NOT NULL,
    bbox_y2        REAL NOT NULL,
    confidence     REAL NOT NULL,
    model_name     TEXT,                -- e.g. 'yolov8n', 'rtmdet' (informational)
    PRIMARY KEY (shot_video_id, video_frame, track_id, region_type)
);

-- Named-person timelines assembled by the timeline stitcher.
-- Maps a person name to one or more contiguous detector track segments for one video.
-- Also keyed on shot_video_id for the same reason as person_detections.
-- Source: pose_extraction.py stitcher UI. Currently stored only in notebook state.
CREATE TABLE person_tracks (
    id              TEXT PRIMARY KEY,
    shot_video_id   TEXT NOT NULL REFERENCES shot_videos(id),
    person_name     TEXT NOT NULL,
    -- JSON: [[track_id, start_frame, end_frame], ...]
    -- Ordered list; JSON is appropriate here because segments are always read
    -- atomically and individual entries are never queried independently.
    track_segments  TEXT NOT NULL
);

-- Thumbnail/crop JPEG cache for the video frame server.
-- Covers multiple cache types (full-frame thumbnails, person crops, etc.).
-- cache_type values mirror CacheType enum in frame_cache.py.
-- For PERSON_CROP entries, track_id identifies which detection bbox was used
-- (not person_name, because crops are needed before timeline stitching assigns names).
-- region_type further narrows which bbox within a track (full_body / face / hand_l / ...).
-- src_* columns describe the region of the original (pre-crop) frame used.
-- For full-frame thumbnails src_* covers the entire frame (0, 0, full_width, full_height).
-- file_path and data are mutually exclusive storage backends:
--   small thumbnails (≤256px) may be stored inline as data BLOBs;
--   larger crops are stored as files.
-- Entries are derived data and can be deleted and regenerated freely.
CREATE TABLE frame_cache_entries (
    shot_video_id  TEXT NOT NULL REFERENCES shot_videos(id),
    frame_idx      INTEGER NOT NULL,
    cache_type     TEXT NOT NULL,       -- 'full_frame' | 'thumb' | 'person_crop'
    track_id       INTEGER,             -- non-NULL for person_crop entries
    region_type    TEXT,                -- non-NULL for person_crop entries
    width_px       INTEGER NOT NULL,    -- width of the stored image
    height_px      INTEGER NOT NULL,    -- height of the stored image
    src_x          INTEGER NOT NULL,    -- source rect in the original frame (pixels)
    src_y          INTEGER NOT NULL,
    src_w          INTEGER NOT NULL,
    src_h          INTEGER NOT NULL,
    file_path      TEXT,                -- path to JPEG file (NULL if stored inline)
    data           BLOB,                -- inline JPEG bytes (NULL if stored as file)
    PRIMARY KEY (shot_video_id, frame_idx, cache_type, track_id, region_type, width_px)
);
```

**Design notes on these tables:**

**`camera_id` vs `shot_video_id`**: The original draft used `camera_id INTEGER` (an index into `active_camera_ids`) for both `person_detections` and `person_tracks`. This is inconsistent with the rest of the schema (which uses UUID foreign keys) and fragile (breaks if camera ordering changes). Both tables now reference `shot_video_id` instead, which uniquely identifies a (shot, camera) pair and is the natural anchor for per-video processing results.

**`person_detections` not `yolo_detections`**: The table is named generically because the detection model may change (YOLO → RTMDet → any future model). The schema is model-agnostic; `model_name` captures provenance informally without constraining the structure.

**`region_type` in `person_detections`**: A single detection pass can yield multiple bounding boxes per person per frame — for example a full-body box at 384×256 for the main pose tracker, a hand-crop box for a hand tracker, and a face-crop box for a face tracker. Rather than separate tables per region (which would duplicate the tracking ID and frame FK machinery), `region_type` is a discriminator column in the same table. Downstream trackers filter by `region_type` to get the specific crop they need. New region types can be added without schema migration.

**Why not a generic "observation from video" table?** Three separate tables (`person_detections`, `PoseObservation`, and future `marker_detections`) are justified here because:
- The data shapes differ significantly: person detection produces bboxes + track IDs; RTMpose produces per-keypoint coordinates + confidences; marker detection would produce labelled 2D point sets with different semantics.
- Raw detection tables feed different downstream pipelines; they are never usefully queried together.
- `PoseObservation` (RTMpose/marker keypoints) is the unified abstraction at the *output* level — the inputs to the tracker regardless of capture method. The raw detection tables are intermediate processing artefacts below that level.
- If a future capture method produces data of genuinely the same shape as an existing table, it should reuse that table with a `source_type` discriminator column rather than adding a new table.

**JSON in `person_tracks.track_segments`**: Justified. The segments are always read as a complete ordered list — there is no use case for querying individual segment entries via SQL. A normalised `person_track_segments` child table would add joins without any query benefit. JSON is used elsewhere in the schema for similarly atomic ordered lists.

**JPEG storage in `frame_cache_entries`**: Supporting both inline blobs (`data`) and file references (`file_path`) gives flexibility without committing to one approach. Small thumbnails for the timeline scrubber (≤256px, ~5–15 kB each) are good candidates for inline storage (keeps the cache self-contained, fast random access by primary key). Full-resolution person crops (several hundred kB) should stay as files to avoid bloating the DB and to enable efficient bulk deletion. A `CHECK (file_path IS NOT NULL OR data IS NOT NULL)` constraint enforces that at least one storage backend is set.

**`track_id` not `person_name` in `frame_cache_entries`**: Person crops are needed during the stitcher UI, before `person_tracks` has been populated (and therefore before a `person_name` exists). Using `track_id` as the cache key for `PERSON_CROP` entries means crops are available immediately after detection, and the stitcher UI can display them without waiting for the user to assign names. Post-stitching, looking up a crop by person name requires one extra step: resolve `(person_name, frame)` → `track_id` via `person_tracks.track_segments`, then use `track_id` as the cache key. This indirection is small and keeps the cache key stable regardless of when stitching happens.

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

Needed by Marimo (as HTTP endpoint) or PySide6 (as in-process cache). Logic is the same. See the detailed design in section 2a for the full `CacheKey` / `CacheType` API; the summary interface is:

```python
class CacheType(enum.Enum):
    FULL_FRAME   = "full_frame"   # full-resolution decoded frame
    THUMB        = "thumb"        # small thumbnail for timeline strip
    PERSON_CROP  = "person_crop"  # crop for a specific detector track + region_type

@dataclass(frozen=True)
class CacheKey:
    shot_video_id: str
    frame_idx:     int
    cache_type:    CacheType
    track_id:      int  | None = None   # required for PERSON_CROP
    region_type:   str  | None = None   # required for PERSON_CROP ('full_body', 'face', ...)
    width_px:      int  | None = None   # required for THUMB

class FrameCache:
    """LRU cache of decoded frames with optional on-the-fly undistortion."""

    def get(self, key: CacheKey, *, undistort: bool = False) -> np.ndarray:
        # 1. Check in-memory LRU
        # 2. Check frame_cache_entries in DB
        # 3. Decode from video (sequential read preferred; seek only when necessary)
        # 4. Crop (PERSON_CROP) or resize (THUMB) as needed
        # 5. Optionally undistort using stored K/dist maps
        # 6. Store in LRU + write to DB cache asynchronously
```

`track_id` (not `person_name`) is used for `PERSON_CROP` keys so that crops are accessible during the stitcher UI before person names have been assigned. Post-stitching, callers resolve `person_name` → `track_id` via `person_tracks` before calling `get()`. The `region_type` field supports multiple crop windows per track per frame (full body, face, hand, etc.) feeding specialised downstream trackers.

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

### Open Issues / Design Decisions

**1. `frame_cache_entries` belongs in a separate DB or a `cache.db` sidecar**
The session DB stores durable pipeline data; `frame_cache_entries` is derived, disposable data that can be several GB for a full session. Mixing them means `VACUUM`/backup operations handle large transient BLOBs unnecessarily. Options:
- Store cache entries in a separate `cache.db` file alongside the session DB (simple, no cross-DB FKs)
- Keep in session DB but in a `cache_*` table that is always excluded from backups/exports

**2. `pose_observation_sequences` ↔ `shot_video_id` relationship is not explicit**
The sequence groups per-camera observations together, but there is currently no table that lists which `shot_video_id` rows feed a given `sequence_id`. This makes it hard to navigate from a sequence back to its source videos. Consider:
- A `sequence_videos` join table: `(sequence_id, shot_video_id)`, one row per camera
- Or adding `shot_id` as a direct FK on `pose_observation_sequences`

**3. Undistortion pipeline for the tracker (long-term)**
The current tracker works with undistorted pixel coordinates (observation → undistorted camera plane). To work with original distorted videos end-to-end, the tracker's `Camera` struct needs `K_original` and distortion coefficients so it can project markers directly to distorted space. This is a significant change deferred post-Phase 2.

**4. `person_tracks` unique constraint**
There should be a `UNIQUE (shot_video_id, person_name)` constraint to prevent duplicate timelines. Adding this prevents silent data corruption if the stitcher UI is run twice.

**5. `person_detections` — re-run semantics**
Person detection is non-deterministic (track IDs change across runs). If detection is re-run on a video, existing rows for that `(shot_video_id, region_type)` pair must be deleted before inserting new results — a re-run of the full-body detector should not invalidate separately-stored hand detections. The CLI / job worker must enforce this explicitly; it is not a cascade-delete scenario.

---

### Implementation Phases

**Phase 1 — DB integration for existing tools** (glue, minimal new code)
- Import intrinsics HDF5 → `IntrinsicsCalibration` table (adding undistortion map columns)
- Import project YAML → session + shot + `ShotVideo` records
- Import extrinsics TOML → `ExtrinsicCalibration` / `ExtrinsicEntry` tables (partially done via CLI)
- Import sync JSON → `SyncConfig` / `SyncPoint` (already done)
- OpenPose JSON → `PoseObservation` (already done via `import_pose_json.py`)

**Phase 2 — Unified setup app**

The goal is a PySide6 application that replaces the current sequence of manual hand-off steps (YAML editing, VIA annotator, separate Marimo notebooks) with a wizard that drives the full session setup flow. It builds on `sync_videos.py` as its video scrubber foundation.

---

### Phase 2 detailed design

#### Application structure

```
python/app/setup/
├── __init__.py
├── main.py                  ← QApplication entry point; opens SetupWizard
├── wizard.py                ← SetupWizard (QWizard subclass); owns DB connection
├── db_context.py            ← thin wrapper: open_session + cached lookups
├── components/
│   ├── frame_cache.py       ← FrameCache (LRU + DB persistence)
│   ├── video_scrubber.py    ← MultiVideoScrubber widget
│   ├── overlay.py           ← Overlay Protocol + concrete overlay classes
│   └── job_runner.py        ← BackgroundJob + QThread wrapper with progress signal
└── pages/
    ├── page_open_session.py     ← Step 0: open or create session DB
    ├── page_add_videos.py       ← Step 1: add ShotVideo rows + shot boundaries
    ├── page_sync.py             ← Step 2: rough sync + LED fine sync (single page)
    └── page_extrinsics.py       ← Step 3: extrinsics annotation + PnP
```

The wizard is launched with a session DB path (either passed on the command line or chosen via the step 0 file dialog). All subsequent wizard pages read and write directly to that DB. There is no intermediate state file; every committed action is immediately durable.

---

#### 2a. `FrameCache`

Central frame provider used by every UI widget that needs to display video pixels.

```python
class CacheType(enum.Enum):
    FULL_FRAME   = "full_frame"   # full-resolution decoded frame
    THUMB        = "thumb"        # small thumbnail for timeline strip (e.g. 320×180)
    PERSON_CROP  = "person_crop"  # tight crop for one detector track + region

@dataclass(frozen=True)
class CacheKey:
    shot_video_id: str
    frame_idx:     int
    cache_type:    CacheType
    track_id:      int | None = None   # required for PERSON_CROP
    region_type:   str | None = None   # required for PERSON_CROP; e.g. 'full_body', 'face', 'hand_l'
    width_px:      int | None = None   # required for THUMB
    height_px:     int | None = None   # required for THUMB
```

`CacheKey` is the identity for every cache entry. `track_id` + `region_type` together identify which bounding box from `person_detections` to crop — using the raw detector track ID rather than `person_name` so that crops are accessible during the stitcher UI before the user has assigned person names. `region_type` supports multiple crop windows per track per frame, allowing different downstream trackers (full-body pose, face, hand) to each get their own appropriately-sized crop without an extra detection pass. `width_px`/`height_px` allow multiple thumbnail resolutions to coexist.

```
FrameCache
├── _lru: dict[CacheKey → np.ndarray]   (in-memory, ~200 entry cap)
├── _caps: dict[shot_video_id → cv2.VideoCapture]   (open capture pool)
├── _last_frame: dict[shot_video_id → int]           (last decoded frame index for sequential detection)
└── get(key: CacheKey, *, undistort: bool = False) → np.ndarray
```

**`get()` logic**:
1. Check in-memory LRU → return if found.
2. Query `frame_cache_entries` in DB matching the key fields → decompress and return if found.
3. Cache miss: decode from video.
   - If `key.frame_idx == _last_frame[id] + 1`: read next frame sequentially (no seek needed).
   - Otherwise: `cap.set(CAP_PROP_POS_FRAMES, frame_idx)` then read.
4. Apply crop for `PERSON_CROP` (bounding box from `person_tracks`), or resize for `THUMB`.
5. If `undistort=True`: apply `cv2.undistort` using stored K/dist.
6. Store in LRU; write compressed JPEG to `frame_cache_entries` asynchronously (off main thread via a write queue) to avoid blocking the UI.

**Performance note — initial scrubbing**: When the user first scrubs a video in the UI, frames are decoded on demand via random seek, which is slow for compressed codecs (H.264 requires decoding from the previous keyframe). This is acceptable for interactive use (a few frames at a time) but would be unacceptable for pre-populating the cache across a full shot. Pre-population should only be triggered explicitly (e.g. a "Generate thumbnails" background job), not implicitly on every `get()` call. If scrubbing performance is still unsatisfactory, the fallback is to use `ffmpeg -vf fps=2` to extract thumbnails into a sidecar directory, bypassing OpenCV entirely for the timeline strip.

**Thread safety**: `get()` is called from the Qt main thread only. Background jobs that decode frames (e.g. LED sync ROI scan) use their own `cv2.VideoCapture` instances, not the pool.

---

#### 2b. `Overlay` protocol

Overlays are typed via a `Protocol` rather than duck typing, giving static-analysis safety without requiring a common base class:

```python
from typing import Protocol

class Overlay(Protocol):
    def paint(
        self,
        painter:  QPainter,
        frame_w:  int,    # source video frame width in pixels
        frame_h:  int,    # source video frame height in pixels
        cell_w:   int,    # display cell width in pixels
        cell_h:   int,    # display cell height in pixels
    ) -> None: ...

    def mouse_press(self, x_px: int, y_px: int) -> None: ...
    def mouse_move(self,  x_px: int, y_px: int) -> None: ...
    def mouse_release(self, x_px: int, y_px: int) -> None: ...
```

Pixel coordinates passed to mouse events are in video-frame space (already mapped from display space by `CameraCell`). Concrete overlay classes:

| Class | Used by | Behaviour |
|---|---|---|
| `SyncAnchorOverlay` | Sync page | draws a vertical tick on timeline at user-set anchor frame |
| `ROIDrawOverlay` | Sync page | rubber-band rectangle for LED ROI selection |
| `AnnotationPointOverlay` | Extrinsics page | labelled dots at control point clicks; handles zoom-refine interaction |
| `ReprojectionOverlay` | Extrinsics page | circles + residual lines after PnP compute |

---

#### 2c. `MultiVideoScrubber`

`QWidget` displaying all cameras for a shot in a grid. Each cell is a `CameraCell` (`QLabel` or `QOpenGLWidget`).

```
MultiVideoScrubber
├── cells: list[CameraCell]
├── sync_table: SyncTable | None        ← None = no sync set yet (independent mode)
├── focused_cell: int | None            ← index of cell receiving keyboard input
├── _timestamps: dict[int, float]       ← shot_video_id → current position in seconds
└── methods:
    seek_synced(timestamp_s)    ← move all cameras together via sync_table
    seek_camera(cell_idx, frame_idx)  ← move one camera independently
```

**Navigation modes**:

- **Synced mode** (sync_table present): `←`/`→` advance the reference camera by 1 frame; all other cameras follow via `sync_table.lookup()`. `Shift+←`/`Shift+→` = ±10 frames.
- **Independent mode** (no sync or user presses `Tab` to focus a cell): keyboard controls move only the focused camera. The focused cell is highlighted with a border. This is essential for rough sync: the user must be able to scrub each camera to the same physical moment independently before setting anchors.

`Space` toggles play/pause (real-time playback at reference fps via `QTimer`). `Home`/`End` go to first/last frame. Clicking a cell focuses it for independent navigation.

**Overlay protocol**: Each `CameraCell` holds a list of `Overlay` instances (typed as `list[Overlay]`). After rendering the video frame it calls `paint()` on each overlay in order. Mouse events on the cell are forwarded to all overlays in reverse order (top-most overlay gets first chance).

**Sync source**: loads the best available `SyncConfig` from the DB (LED preferred, manual-rough as fallback). Exposes a `reload_sync()` slot connected to the DB write signal.

---

#### 2d. Wizard pages

##### Page 0 — Open / create session

File-open dialog with two options:
- Open existing session DB (`.db` file picker).
- Create new session DB (name + directory → calls `create_session()`).

Displays a summary of existing DB contents (shots, cameras, sync state). On "Next" the wizard receives the DB path and all subsequent pages share it via `DBContext`.

##### Page 1 — Add videos

Lists cameras from `session_cameras`. For each:
- File path field + "Browse" button → sets `shot_videos.file_path`.
- Auto-probes with `cv2.VideoCapture` to fill `actual_fps`, `frame_count`, `width`, `height`.
- Camera model/mode dropdown from registry.

**Shot boundary definition**: A thumbnail strip for the reference camera (populated lazily by a background `THUMB` cache warm job). The user drags start/end handles to define shot boundaries. Multiple shots can be defined in one pass; each creates a `Shot` row and `ShotVideo` rows for all cameras.

On "Next", writes all rows to DB.

##### Page 2 — Sync (rough + LED, single page)

Rough sync and LED sync are combined on one page because they are iterative: the user may refine the rough sync after seeing LED results, or re-draw an ROI and re-run LED sync. Separating them into two pages would force unnecessary wizard navigation.

The page layout:

```
┌─────────────────────────────────────────┐
│  MultiVideoScrubber  (most of the page) │
│  (cells show current camera frames)     │
├──────────────┬──────────────────────────┤
│ Sync panel   │  LED sync panel          │
│              │                          │
│ [Set anchor] │  (ROI draw mode toggle)  │
│ cam1: frame? │  [Run LED sync]          │
│ cam2: frame? │  ████████░░ Analyzing    │
│ ...          │  camera 3/6 · frame 870  │
│ [Apply rough]│  [Accept LED result]     │
└──────────────┴──────────────────────────┘
```

**Rough sync workflow**:
1. Initially no sync_table → scrubber is in independent mode.
2. User clicks on a camera cell to focus it (`Tab` or mouse click), then steps to the sync moment.
3. Clicks "Set anchor" (or `S`). The anchor frame is recorded for that camera and shown in the panel. `SyncAnchorOverlay` marks it on each cell's timeline strip.
4. Repeat for all cameras.
5. "Apply rough sync": computes frame offsets relative to reference camera, writes `SyncConfig(method="manual-rough")`. Scrubber reloads in synced mode; user verifies alignment.

**LED sync workflow** (continues on the same page after rough sync):
1. User clicks "Draw LED ROI" toggle. Scrubber cells enter ROI draw mode (`ROIDrawOverlay` active). User drags a rectangle on each camera cell over the LED area. ROIs stored in page state (not in DB).
2. "Run LED sync" launches a `LedSyncJob` (see below). A single progress bar appears in the LED panel with text like `Analyzing camera 3/6 · frame 870 of 1 500`. No per-camera dialogs.
3. On completion: results shown as anchor dots on the scrubber timeline strip. The panel shows per-camera quality metrics (peak signal-to-noise, correlation score).
4. If quality is poor: user can adjust the ROI and re-run without leaving the page.
5. "Accept LED sync": writes `SyncConfig(method="led-auto")` with all sync points. Scrubber reloads.

**`LedSyncJob`**:
```python
class LedSyncJob(BackgroundJob):
    """Scans ROI patches for brightness peaks; no full-frame decode needed."""

    def run(self):
        for cam_idx, (shot_video_id, roi) in enumerate(self._cameras):
            cap = cv2.VideoCapture(file_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            brightness = []
            for f in range(n_frames):
                ret, frame = cap.read()   # sequential — no seek
                if not ret:
                    break
                # Crop to ROI and compute mean brightness of the patch only
                patch = frame[roi.y1:roi.y2, roi.x1:roi.x2]
                brightness.append(patch.mean())
                if f % 50 == 0:
                    pct = int(100 * (cam_idx * n_frames + f) / (n_cams * n_frames))
                    self.progress.emit(pct, f"Analyzing camera {cam_idx+1}/{n_cams} · frame {f:,} of {n_frames:,}")
            # find peaks, cross-correlate with reference
            ...
        self.finished.emit(result)
```

Key performance point: the job reads frames **sequentially** and only computes the mean of the small ROI patch (not the full frame). Even though every compressed frame is decoded, the CPU cost of the decode itself dominates; the ROI crop does not add meaningful overhead. For a 4K H.264 video at 120 fps, full decode is roughly 8–15 ms/frame; for a 1080p video it is 2–5 ms/frame. A 90-second shot at 120 fps (10 800 frames) would take ~1.5 minutes per camera at 4K or ~25 s at 1080p. For 6 cameras sequentially, this is 9–15 minutes total. If that proves too slow, the optimization path is to use an `ffmpeg` subprocess with `-vf crop=W:H:X:Y` to decode only the ROI region — many decoders can skip chroma planes and full luma decode when the output region is tiny.

**`BackgroundJob` base class**:
```python
class BackgroundJob(QThread):
    progress = Signal(int, str)   # (percent 0–100, human-readable message)
    finished = Signal(object)     # result payload (job-specific type)
    error    = Signal(str)        # error message if run() raises

    def run(self) -> None: ...    # override in subclass
```

##### Page 3 — Extrinsics annotation

Shows one fixed calibration frame per camera (not a synchronized scrub). `MultiVideoScrubber` is reused but each cell is locked to its designated calibration frame.

**Control point set**: loaded from a JSON file (list of `{name, x_m, y_m, z_m}` entries) or typed in via a small table editor on the page. The set is not persisted to the session DB (it is a property of the physical calibration rig, not the capture session).

**`AnnotationPointOverlay` interaction — zoom-to-refine**:

Each camera cell has a mode toggle: overview (video scaled to fit cell) vs. zoom-refine (1:1 pixel scale, panned so the clicked point stays centred). The interaction:

1. In overview mode: user clicks on a control point location → a new dot is placed, labelled with the point name. The cell immediately switches to zoom-refine mode: the image is displayed at 1:1 scale, panned so that the clicked pixel stays at the same screen position under the cursor. (The cell renders a `QTransform` that maps video pixels to display pixels with the clicked point as the fixed point.)
2. In zoom-refine mode with mouse button held: moving the mouse adjusts the dot position at full resolution. The label and a crosshair follow the cursor. This allows sub-pixel precision that would be impossible in the scaled-down overview.
3. On mouse release: the final pixel coordinate is committed to the overlay's point list. The cell returns to overview mode.
4. Clicking an existing dot selects it and re-enters zoom-refine mode for that point.
5. Right-click on a dot: delete it.

The coordinate stored is always in original video pixel space (before any display scaling), so there is no precision loss from the overview rendering.

**Compute and review**:
1. "Compute extrinsics" runs per-camera `cv2.solvePnPRansac` → R, t per camera.
2. `ReprojectionOverlay` activates: draws circles at reprojected control point positions, residual lines from annotation clicks to reprojections, per-camera RMS reprojection error label.
3. Optional "Bundle adjustment" refines all cameras jointly using SciPy `least_squares`.
4. "Accept" writes `ExtrinsicCalibration` + `ExtrinsicEntry` rows.

**Annotation persistence**: annotation clicks are stored in the page's own state dict (keyed by `shot_video_id → list[(point_name, x_px, y_px)]`). A future `calibration_annotations` table can be added to persist them across app restarts.

---

#### 2e. DB writes from the wizard

All writes go through a `DBContext` object owned by the wizard. Pages call methods on it; they do not open their own connections. A transaction is held open within each page; "Back" rolls it back.

```python
class DBContext:
    conn: sqlite3.Connection
    def create_shot(self, label: str) -> str
    def create_shot_video(self, shot_id: str, cam_instance_id: str,
                          path: str, fps: float, frame_count: int,
                          width: int, height: int) -> str
    def write_sync_config(self, shot_id: str, method: str,
                          points: dict[str, list[SyncPoint]]) -> str
    def write_extrinsics(self, shot_id: str,
                         entries: list[ExtrinsicEntry]) -> str
    def get_shot_videos(self, shot_id: str) -> list[ShotVideoInfo]
    def get_active_sync(self, shot_id: str) -> SyncTable | None
```

`SyncPoint`, `ExtrinsicEntry`, `ShotVideoInfo` are typed `dataclass` or `NamedTuple` objects, not plain dicts.

---

#### 2f. Dependency and launch

The setup app lives in `python/app/setup/` and is launched via:

```bash
uv run python -m posetrak.app.setup.main [session.db]
```

or as a named script in `pyproject.toml`:

```toml
[project.scripts]
posetrak-setup = "posetrak.app.setup.main:main"
```

Dependencies (added to `[dependency-groups.app]`):
- `PySide6` (already listed)
- `opencv-python` (already used in pipeline)
- `numpy` (already used)
- `scipy` (for bundle adjustment, optional — guarded by `try/import`)

**Phase 3 — Background job integration**
- Person detector wrapped as a job worker writing to `person_detections` (keyed on `shot_video_id` + `region_type`)
- RTMpose wrapped as a job worker writing to `PoseObservation` (existing table)
- Timeline stitcher rewritten against `person_tracks` / `person_detections` tables
- Progress reporting in UI

**Phase 4 — Results visualization**
- Integrate tracker-debug and visualize_tracking capabilities
- 3D pose viewer
- Video overlay with reprojection circles

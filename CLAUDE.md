# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Posetrak is an application suite for video based multi-camera motion capture. Its main applications include

* *posetrak* is the actual motion tacking application. It uses Unscented Kalman Filter based algorithm to predict skeleton pose and joint angles from body keypoints detected from videos. Command line applicatio written in C++, see docs/cpp-architecture-overview.md for overview of architecture

* Posetrak setup provides GUI for setting up and managing motion capture projects. Python, PySide 6. See docs/pipeline-ui-requirements.md and docs/pose-extraction-app-design.md

## Build System

This project uses **Meson** with wrap-based dependency management. All dependencies download automatically on first setup.

```bash
# Initial setup (downloads all dependencies via wraps)
meson setup builddir

# Build
meson compile -C builddir

# Reconfigure (e.g. after meson.build changes)
meson setup --wipe builddir
```

**Pinocchio** is installed system-wide at `/opt/openrobots/` and used header-only (no linking required). See `docs/pinocchio-header-only-analysis.md` for rationale.

## Running Tests

**Important**: If conda is active, use `run_tests.sh` to avoid `GLIBCXX` version conflicts:

```bash
# Preferred: avoids conda library conflicts
./run_tests.sh

# With arguments passed through to meson test
./run_tests.sh -v
./run_tests.sh --test-args="[skeleton]"   # Run tests matching a tag

# Direct (only if conda is not active)
meson test -C builddir
meson test -C builddir -v
meson test -C builddir --test-args="[skeleton]"
```

All tests are compiled into a single executable `builddir/tests/test_posetrak` using **Catch2**. To run a single test by name:

```bash
./run_tests.sh --test-args="test_name_here"
```

## Test Coverage

```bash
./scripts/run_coverage.sh
# Report at: builddir/coverage_html/index.html
```

## Architecture Overview

PoseTrack is a C++20 motion capture tracker that estimates skeletal pose from multi-camera 2D marker observations using an **Unscented Kalman Filter (UKF) in joint space**.

### Data Flow

```
YAML skeleton + TOML cameras + OpenPose JSON observations
    → SkeletonLoader / CameraLoader / ObservationLoader
    → Triangulator (DLT multi-view) + InverseKinematics (damped least-squares)
    → Tracker::initialize()  [initial pose]
    → per-frame: Tracker::track_frame()
        → UKF::predict() [constant-velocity process model]
        → UKF::update() [project sigma points through FK → camera, Mahalanobis outlier rejection]
    → TrackingExporter → CSV files (state_vectors, joint_angles, root_pose, etc.)
```

### Core Abstractions

**`Skeleton`** (`include/posetrak/core/skeleton.hpp`) — kinematic tree of `Joint` objects (REVOLUTE, SPHERICAL, FIXED, PRISMATIC) with attached `Marker` points. Loaded from YAML via `SkeletonLoader`.

**`SkeletonLayout`** (`include/posetrak/core/skeleton_layout.hpp`) — immutable, precomputed DOF index table derived from a `Skeleton`. Shared via `shared_ptr<const SkeletonLayout>` between UKF, process model, and sigma point generator. **All DOF index arithmetic must go through this class.** Two factory functions: `from_full_skeleton()` and `from_groups()` (for child filters).

**`State`** (`include/posetrak/core/state.hpp`) — root pose (position + quaternion) + joint angles + velocities. Uses **error-state formulation**: orientation updates happen on the quaternion manifold via axis-angle perturbations.

**`UnscentedKalmanFilter`** (`include/posetrak/filters/ukf.hpp`) — error-state UKF. Sigma points are generated in tangent space, propagated through the constant-velocity process model, and measurement-updated via FK + camera projection. Uses **Joseph form** covariance update for numerical stability.

**`ForwardKinematics`** (`include/posetrak/kinematics/forward_kinematics.hpp`) — wraps Pinocchio. Converts `State` → Pinocchio config vector → computes joint transforms → extracts marker world positions. Critical: must call both `forwardKinematics()` and `updateFramePlacements()`. Pinocchio quaternion convention is `[x,y,z,w]` (not `[w,x,y,z]`).

**`Tracker`** (`include/posetrak/tracking/tracker.hpp`) — orchestrates initialization (triangulation → IK) and the per-frame predict/update cycle. Owns the UKF, FK, triangulator, IK solver, and Pinocchio model/data. Supports optional **RTS smoothing** (enable before tracking, call `smooth()` after).

### Key Design Decisions

- **Error-state UKF on manifold**: Sigma points live in tangent space (axis-angle for orientations), applied via retraction `State::apply_error_update()`.
- **UKF alpha must be ~0.5**: For a ~58-dimensional state, alpha=0.001 causes negative covariance weights. Alpha ≥ 0.5 is required for positive-definite weights.
- **SPHERICAL joints always occupy 3 state slots**, even if some DOFs are locked by equal limits. Active/locked distinction is handled at the layout level.
- **`SkeletonLayout` is the single source of truth** for all joint-to-state-vector index mapping. Do not recompute indices ad hoc.
- **Pinocchio header-only mode**: `PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION` is not defined, so no `libpinocchio_default.so` is linked.

### Config File Format

The CLI accepts a **TOML** config file. Example at `tests/regress.toml`. Key sections: `[data]`, `[tracking]`, `[tracking.initialization]`, `[tracking.ukf]`, `[output]`, `[processing]`, and optionally `[hierarchical]`.

Skeleton files use **YAML** format (see `docs/skeleton-format.md`). Camera calibration uses **Pose2Sim TOML** format.

### CLI

```bash
# Run tracker
builddir/cli/posetrak track config.toml

# Post-process bone-length calibration run
builddir/cli/posetrak scale config.toml
```

Performacne between debug & optimized builds is big; therefore before executing actual tracking tests with posetrak CLI create another meson build environmnet to `optbuild/` directory and use it. For unit testing and debugging it is ok to use the debug build in `builddir/`.

### Output Files

The tracker writes CSV files to `output_dir/`: `state_vectors.csv`, `joint_angles.csv`, `root_pose.csv`, `marker_projections.csv`, `observations.csv`, `tracking_stats.csv`, `predicted_observations.csv`, `tracking_results.csv`.

### MCP Diagnostic Server

`python/app/mcp/` contains a read-only MCP server for diagnosing tracking runs. It is intended to be used with Claude Code (or any MCP-compatible client) when investigating tracking anomalies — the tools give structured access to filter statistics, camera coverage, and observation data without requiring manual SQL queries.

**Running the server:**

```bash
# Install the mcp dependency group first (one-time)
uv sync --group mcp-server

# Start the server against a session database
uv run posetrak-mcp --db-path /path/to/session.db
```

**Wiring it into Claude Code** — add to `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "posetrak": {
      "command": "uv",
      "args": ["run", "posetrak-mcp", "--db-path", "/path/to/session.db"]
    }
  }
}
```

**Available tools and recommended workflow:**

1. `list_tracking_runs` — see all runs in the DB with time ranges
2. `get_run_info` — tracker config (flags large `measurement_noise_std`), camera list, marker list
3. `get_filter_stats(run_id, start_s, end_s)` — per-step NIS/DOF and covariance condition number; anomaly summary at top
4. `get_camera_coverage(run_id, start_s, end_s, markers)` — inlier/outlier/absent grid showing which cameras see which markers
5. `get_observation_gaps(run_id, start_s, end_s, markers)` — actual vs predicted pixel positions; gaps ≥ 30 px flagged
6. `get_camera_geometry(run_id)` — 3-D camera positions and pairwise parallax with GOOD/MODERATE/POOR depth quality labels
7. `get_edit_coverage(run_id)` — which HALPE keypoints are edited per camera; flags key body landmarks left unedited

**Key schema notes for server development:**

- `tracking_runs.active_camera_ids` — JSON array of camera **labels** (e.g. `"gopro-11_mini_01"`), not UUIDs. `get_run_cameras()` in `db.py` resolves these to `camera_instances.id` UUIDs so they match `extrinsic_entries` and `pose_observation_edits`.
- `tracking_obs_results.obs_blob` — `float32[n_cam, n_mrk, 8]`: fields are `[actual_x, actual_y, pred_x, pred_y, mahal_dist, used (1=inlier), is_outlier, pad]`. NaN actual_x means no observation for that camera/marker slot.
- Always open session DBs read-only: `sqlite3.connect(f"file:{path}?mode=ro", uri=True)`.

### Python Package

The `python/` directory contains the installable `posetrak` Python package:

- `python/posetrak/db/` — SQLite DB layer (install with `pip install -e .`)
- `python/app/analysis/` — Marimo analysis scripts (formerly `notebooks/`)
- `python/tools/` — standalone utility scripts
- `python/tests/` — pytest suite; run with `pytest python/tests/`
- `python/pipeline/` — capture pipeline tools (calibration, pose extraction)

### Python GUI applications — two separate entry points

There are **two distinct GUI applications** in `python/app/`. Do not confuse them:

**`python/app/pose/main.py`** — *Detection pipeline tool* (`PoseExtractionWindow`).
Used to run YOLO+RTMPose on captured video, assign tracks to persons, and finalise
pose observation sequences.  This window does **not** show tracking results and is
not the place to add editing or analysis UI.

**`python/app/ui/main_window.py`** — *Main posetrak viewer/editor* (`MainWindow`).
Shows a tree view of sessions, captures, persons, and tracking runs on the right.
Selecting a person opens `PersonPanel` (in `content_panels.py`), which contains
`PersonCropGridWidget` — the multi-camera crop grid with keypoint overlays,
time scrubber, and (as of the keypoint-editing feature) edit mode.  Any UI that
lets a user inspect or correct motion capture data belongs here.

When adding new UI features, always check which application context the feature
belongs to before creating new widgets.

Data lives on `/mnt/d/mocap/` (Windows drive mount).

## Code Style

C++20, enforced via clang-format (pre-commit hook). All linear algebra uses Eigen. All string formatting uses `fmt`. Use `std::optional`, concepts, and ranges where natural.

## Git conventions

The project uses pre-commit hooks to ensure consistency of Git commit. Always make sure that pre-commit is isntalled before creating commit in new workarea.

Commit changes in sel-contained, cohesive commits. If there are unrelated changes in the workarea, create separate commits for those.

Do not blindly use `git commit -a` or other commands that might add unintended files to the commit.

Before commit, make sure that unit tests pass.

Commit messages:

* Start commit message with title in format "comp: short description". Comp indicates area affected by change, it canb e make of the dource file directory or doc, test, ci.
* Follow with description of the change
* Do not use ephermal planning step names like step 1, phase 3a etc. often used in plan documents created for individual tasks. Commit messages record project history and terms used in them must be understandable without additional context after many years. DO refer to Github issue IDs if the change is related to a open issue.
* Do not icnlude Co-authored-by: tag to commit messages.

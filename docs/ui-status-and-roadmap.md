# Posetrak UI — Status and Roadmap

**Last updated:** 2026-04-25

---

## 1. Current application structure

Three separate entry points today:

| Entry point | Description |
|---|---|
| `posetrak-setup` | Qt wizard: session DB, shots, camera sync |
| `posetrak-pose` | Qt window: pose detection, stitching, finalise |
| `posetrak-db` | CLI: all remaining steps (extrinsics, skeleton, config, run tracker, export) |

Everything downstream of pose finalisation — extrinsics import, skeleton setup, running the
tracker, exporting BVH — is CLI-only.

---

## 2. Status: what is implemented

### posetrak-setup wizard (3 pages)

| Feature | Status |
|---|---|
| Session DB open / create | Done |
| Camera registry (models, modes, instances) | Done |
| Intrinsics import — HDF5 (Pose2Sim output) | Done |
| Intrinsics entry — manual (pinhole + Brown-Conrady only) | Done; fisheye model not selectable |
| Shots page — add shot, attach video files | Done |
| Camera instance and mode assignment per video | Done |
| Sync page — LED auto-sync with ROI selection | Done |
| Sync page — manual frame-stepping fallback | Done |

### posetrak-pose window

| Feature | Status |
|---|---|
| Shot / sync / detection-run selectors | Done |
| Time-range marking (Mark Start / Mark End) | Done |
| Background detection pipeline (YOLO + RTMPose → DB) | Done |
| Stitcher timeline with person assignment | Done |
| Assignment conflict detection + resolution dialog | Done |
| Frame view with skeleton overlay | Done |
| Person preview panel (live bbox crop) | Done |
| Finalise → `pose_observation_sequences` + `pose_observations` | Done |
| Assignment state persisted to DB (survives restart) | Done |
| PersonPreviewWidget | Done |
| Confidence sparkline | Postponed (Phase C) |
| Manual bbox correction + partial re-run | Postponed (Phase C) |

### Backend readiness for distorted-pixel pipeline

`finalise.py` writes `pixels_are_undistorted = 0`; the C++ tracker reads this flag and
calls `Camera::undistort()` on every observation.  The tracker implements both
Brown-Conrady and fisheye (kb4) iterative inversion.

For cameras calibrated via HDF5 (Pose2Sim), `K_original` (distorted) and `K_new`
(optimal undistorted) are both stored and the tracker's undistortion is fully correct.

**Gap:** The manual calibration UI entry hardcodes `radtan` and does not compute or store
`matrix_original`.  For cameras with mild distortion this is acceptable (the tracker
remains self-consistent, just using a sub-optimal `K_new`).  For true fisheye lenses
the distortion model must be set correctly; intrinsics for such cameras must currently
come from the HDF5 import path, not the manual entry form.

---

## 3. What is still needed for a full end-to-end GUI run

The steps below are today CLI-only.  Each is a self-contained feature that can be added
independently.

### 3.1 Extrinsics import

**What:** A UI action to import a Pose2Sim `cameras.toml` (or HDF5 equivalent) and
create an `extrinsic_calibrations` row linked to the current session.

**Backend:** `import_extrinsics.py` / `posetrak-db extrinsics import` — fully
implemented.  The UI is a single file-picker dialog plus a camera-matching step (same
pattern as the existing intrinsics import).

**Note:** This is a placeholder.  A new extrinsics calibration method (wand-based or
similar) is planned; when that lands the import dialog can remain as a fallback.

### 3.2 Skeleton setup

**What:** A wizard page (or sidebar panel) for choosing the skeleton a tracking run
will use.

**Two entry paths:**

1. **Pick from registry** — select an existing skeleton stored in the DB (previously
   imported or scaled).
2. **Import YAML** — browse for a skeleton YAML file; wraps
   `posetrak-db skeleton import`.

**Rough scaling before the first run:**
The UKF works best when bone lengths are at least roughly right from the start.
Before the first tracking run the user should be able to set a simple scale factor
(or per-segment overrides: shin, femur, upper arm, lower arm, torso height, shoulder
width, head) applied to the template skeleton.  The UI pre-fills these from rough
physical measurements or body-height estimate.  The scaled skeleton is saved to the
DB and used for the run.

After the short calibration tracking run, if the template was "roughly right",
outlier rejection will have produced a clean set of inlier observations.  Those
inliers can be triangulated to measure actual joint-to-joint distances, and the
skeleton can be re-scaled to match — this is what `body_measurements.py` (Marimo) and
`posetrak-db skeleton scale` do today.  Integrating this into the Qt UI removes the
need to open the notebook.

**Final scaling workflow (integrated):**
1. Pick a tracking run from the current session.
2. The UI triangulates inlier observations, computes per-segment medians, and shows
   them against the template values (the table from `body_measurements.py`).
3. User reviews and optionally overrides individual measurements.
4. "Save scaled skeleton" writes a new skeleton row to the DB and optionally sets it
   as the default for future runs in this session.

**Backend:** `scale_skeleton.py`, `manage_skeleton.py` — fully implemented.

### 3.3 Tracker config (UKF parameters)

**What:** A form for creating or editing a `tracker_configs` row: UKF noise parameters
(process noise, measurement noise, initial covariance), outlier rejection threshold,
joint limit enforcement, and which skeleton + pose sequence to use.

**Backend:** `manage_config.py` / `posetrak-db config create|edit` — fully
implemented.

### 3.4 Run tracker

**What:** A "Run Tracker" button that:
1. Resolves the `posetrak` binary path (configurable, defaults to
   `optbuild/cli/posetrak`).
2. Invokes `posetrak track --session-db … --config-id …` in a `QThread`.
3. Streams stdout progress to a progress bar / log panel.
4. On completion writes the `tracking_runs` row ID and offers to proceed to results.

**Backend:** The C++ CLI, `tracking_runs` schema, and `run_project.py` (batch script)
all exist.  This is purely a Qt wrapper.

### 3.5 Results viewer and BVH export

**What:** A read-only panel showing the selected tracking run:
- Summary stats (frames tracked, inlier %, per-camera breakdown).
- A playback view overlaying the tracked skeleton onto the original video (reuses
  `FrameViewWidget` from the pose window with a different data source).
- "Export BVH" button wrapping `export_bvh.py`.

**Backend:** `export_bvh.py` (standalone script), tracking results in DB — both exist.
The overlay rendering for tracked results is currently only in `visualize_tracking.py`
(Rerun-based); a Qt equivalent needs to be written or the Rerun viewer launched as a
subprocess.

---

## 4. Future enhancements (after e2e is complete)

### 4.1 Merge setup wizard and pose window into one application

Today the user runs `posetrak-setup` to prepare the session, then separately launches
`posetrak-pose` for detection and stitching.  The goal is a single application with
a persistent navigation sidebar (or tab strip) covering the full pipeline:

```
[1. Session]  [2. Cameras]  [3. Shots & Sync]  [4. Pose Detection]
[5. Skeleton]  [6. Run Tracker]  [7. Results]
```

Each stage stays accessible after completion so the user can go back, re-run a step,
or open a session that was partially processed.

The two existing windows share no global state (each opens its own DB connection), so
the merge is mainly a navigation and layout change.  The `SyncPage` and
`PoseExtractionWindow` widget trees can be embedded as-is into the new shell.

### 4.2 Intrinsics calibration UI

**What:** An in-app calibration wizard:
1. Select camera mode.
2. Record or load a set of calibration frames (checkerboard or ChArUco).
3. Run OpenCV calibration; display reprojection error and per-frame residuals.
4. Accept → writes `intrinsics_calibrations` row and generates undistortion maps.

This removes the dependency on Pose2Sim / external HDF5 files for intrinsics.

The UI should also fix the current gap in the manual entry form:
- Add a distortion model selector (Brown-Conrady / Fisheye).
- Compute and store `matrix_original` and the optimal `K_new` when distortion
  coefficients are provided, so the C++ tracker's undistortion is fully correct.

### 4.3 Extrinsics calibration UI (placeholder)

A wand-based or structured-light extrinsics calibration method is planned; the
specific algorithm is TBD.  When implemented it replaces the import-from-TOML step
(3.1) with an in-app capture + solve workflow.  For now, import from Pose2Sim TOML
(section 3.1) is the production path.

---

## 5. Implementation phasing

### Phase 1 — Complete the e2e pipeline in the current two-app structure

Priority: unblock full end-to-end runs without touching the CLI.

| # | Task | Effort |
|---|---|---|
| T1.1 | Extrinsics import dialog (3.1) | S |
| T1.2 | Skeleton pick / import page (3.2 — pick from registry + import YAML) | S |
| T1.3 | Rough scaling UI before first run (3.2 — scale sliders + save) | M |
| T1.4 | Tracker config form (3.3) | M |
| T1.5 | Run tracker + progress panel (3.4) | M |
| T1.6 | Tracking run summary + BVH export (3.5 — stats + export button) | M |
| T1.7 | Fix manual calibration: distortion model selector + `matrix_original` (4.2 partial) | S |

`S` = 1–2 days, `M` = 2–4 days.

**Deliverable:** A practitioner can go from raw videos to BVH using only the Qt apps;
the CLI remains available but is no longer required for any step.

### Phase 2 — Skeleton scaling from tracking run (integrated)

Integrates the `body_measurements.py` notebook workflow into the Qt UI.  Depends on
Phase 1 (needs at least one tracking run in the DB).

| # | Task | Effort |
|---|---|---|
| T2.1 | Post-run body measurements panel (triangulate inliers, show per-segment medians) | M |
| T2.2 | Override fields + "Save scaled skeleton" action | S |
| T2.3 | Wire scaled skeleton as default for next run in session | S |

**Deliverable:** Full skeleton sizing workflow without opening a browser or CLI.

### Phase 3 — Merge into single application

Combines the two Qt entry points into one unified shell with sidebar navigation.

| # | Task | Effort |
|---|---|---|
| T3.1 | New shell window with sidebar / tab navigation | M |
| T3.2 | Embed SyncPage and PoseExtractionWindow as sidebar panels | S |
| T3.3 | Embed Phase 1 panels (extrinsics, skeleton, tracker, results) | S |
| T3.4 | Session persistence — remember last-opened DB, last active panel | S |

**Deliverable:** `posetrak-setup` and `posetrak-pose` replaced by a single `posetrak-ui`
entry point; old entry points kept as aliases.

### Phase 4 — Calibration UIs

| # | Task | Effort |
|---|---|---|
| T4.1 | Intrinsics calibration wizard (checkerboard / ChArUco, OpenCV) | L |
| T4.2 | Extrinsics calibration UI (TBD algorithm; placeholder) | L |

`L` = 1–2 weeks.

**Deliverable:** Self-contained calibration without Pose2Sim or external files.

### Phase 5 — Results viewer with video overlay

| # | Task | Effort |
|---|---|---|
| T5.1 | Qt-native skeleton overlay on original video (frame-by-frame, reusing FrameViewWidget) | L |
| T5.2 | Replace / supplement Rerun-based `visualize_tracking.py` | M |
| T5.3 | Joint angle plots alongside video | M |

**Deliverable:** In-app visual validation of tracking results.

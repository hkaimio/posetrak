# Posetrak architecture notes

Source material for a future architecture overview document.  Written from
code reading during the keypoint editing design work (2026-06).

---

## Two-database design

Posetrak uses two separate SQLite files:

| DB | Default location | Purpose |
|---|---|---|
| **Registry DB** | shared, user-global | Camera hardware definitions, intrinsics calibrations, skeleton YAML files, tracker configs |
| **Session DB** | one per recording session | All session-specific data: captures, videos, detections, observations, tracking results |

The session DB is **self-contained** — it mirrors the relevant registry rows
at creation time so it can be moved to another machine without the registry.
Foreign keys across the two files are stored as TEXT IDs but cannot be
enforced by SQLite (cross-file FKs are noted in comments in the schema).

The `SessionReader` C++ class opens the session DB read-only; all writes go
through Python helpers.

---

## Session DB schema layers

The session schema can be understood as four layers:

### Layer 0 — session structure

| Table | Key | Notes |
|---|---|---|
| `mocap_sessions` | `id` | Top-level session record |
| `captures` | `id` | One continuous camera recording (cameras on → off).  Historical alias: *shot* |
| `trials` | `id` | Named, bounded time window within a capture (one technique, one attempt) |
| `capture_videos` | `id` | One video file per camera per capture.  Historical alias: *shot_video* |
| `session_cameras` | `(session_id, camera_instance_id)` | Cameras that participated in the session |

### Layer 1 — calibration and sync

| Table | Key | Notes |
|---|---|---|
| `extrinsic_calibrations` | `id` | One calibration event per session |
| `extrinsic_entries` | `(calib_id, camera_instance_id)` | R, t blobs (float64) per camera |
| `intrinsics_calibrations` | `id` | Mirrored from registry; fx, fy, cx, cy, dist_coeffs |
| `sync_configs` | `id` | One sync solution per capture |
| `sync_points` | `(sync_config_id, camera_instance_id, video_frame)` | Time anchor per camera; multiple per camera for piecewise-linear interpolation |
| `sync_anchors` / `sync_anchor_observations` | — | Input events for the graph-based sync solver |

### Layer 2 — detection pipeline (raw, anonymous tracks)

Run by the Python detection pipeline (`DetectionPipeline`, `backends_yolo.py`,
`backends_rtmpose.py`).

| Table | Key | Notes |
|---|---|---|
| `detection_runs` | `id` | One per pipeline execution; records detector/pose model names, time range, status |
| `detection_keypoints` | `(run_id, svid, frame, track_id, region_type)` | `float32[N,3]` (x, y, confidence) blob per frame per anonymous track |
| `person_detections` | same PK | Bounding box + confidence per frame per anonymous track |
| `person_tracks` | `(id)`, unique on `(run_id, svid, track_id)` | Span (first_frame, last_frame) per track |
| `frame_cache_entries` | `(svid, frame_idx, cache_type, track_id, region_type, width_px, run_id)` | JPEG crop blobs; `src_x/y/w/h` record the crop region in the original full-resolution frame |
| `detection_track_assignments` | `(run_id, svid, track_id, first_frame)` | User's mapping of anonymous `track_id` → named person; persisted so the stitching UI can be restored |
| `keypoint_obs_quality` | `(seg_run_id, svid, frame, track_id)` | Per-keypoint segmentation quality scores from Cutie (float32[133]) |
| `seg_quality_runs` / `seg_masks` | — | Segmentation quality run metadata and raw masks |

`track_id` in this layer is the ID assigned by the person-tracking algorithm
(YOLO ByteTrack or similar) and carries **no person identity**.

### Layer 3 — named person observations (post-stitch)

Produced by `finalise_to_db()` in `finalise.py` after the user assigns
anonymous tracks to named persons in the stitching UI.

| Table | Key | Notes |
|---|---|---|
| `pose_observation_sequences` | `id` | One per named person per detection run.  Links back to `detection_run_id` |
| `sequence_persons` | `(sequence_id, person_id)` | Maps `person_id` (always 0 in current pipeline) → human name |
| `pose_observations` | `(sequence_id, camera_instance_id, video_frame, person_id)` | `float32[N,3]` (x, y, confidence) blob; post-undistortion if `pixels_are_undistorted=1` |
| `pose_observation_edits` *(proposed)* | `(sequence_id, camera_instance_id, video_frame)` | User corrections: blob+mask overlay applied by C++ tracker at load time |

The `kp_blob` in `pose_observations` uses **undistorted pixel coordinates**
(K_new space) when `pixels_are_undistorted = 1` (the current default).  The
C++ tracker skips its undistortion step for these rows.

### Layer 4 — tracking results

Written by the C++ tracker via `ResultWriter`.

| Table | Key | Notes |
|---|---|---|
| `tracking_runs` | `id` | Links to `observation_sequence_id`, `tracker_config_id`, `skeleton_id`, `extrinsic_calibration_id` |
| `tracking_run_persons` | `(run_id, person_id)` | Per-person skeleton override (multi-person) |
| `tracking_results` | `(run_id, person_id, tracker_step, is_smoothed)` | Per-frame UKF state vector + covariance diagonal + NIS value |
| `tracking_obs_results` | `(run_id, person_id, tracker_step)` | Per-frame 2-D projected marker observations (float32 blob); used for residual analysis |

---

## Key data flows

### Flow 1 — detection pipeline

```
capture_videos (file paths)
    ↓  VideoCapture (OpenCV)
Raw frames
    ↓  YOLOv11Detector  →  person bboxes + track_ids
    ↓  RTMPoseEstimator →  keypoint blobs per track
    ↓  _encode_crop()   →  JPEG crop at bbox
    ↓  DetectionBatchWriter (db_cache.py)
        → detection_keypoints (keypoint blobs)
        → person_detections   (bbox rows)
        → person_tracks       (track spans)
        → frame_cache_entries (JPEG crops with src_x/y/w/h)
```

Crop coordinates (`src_x, src_y, src_w, src_h`) are stored at original video
resolution before any JPEG downscale, enabling exact inversion of the
display-crop ↔ full-frame coordinate transform in the editing UI.

### Flow 2 — stitching (finalization)

```
User selects detection run, assigns anonymous tracks to named persons
    ↓  finalise_to_db() (finalise.py)
        → pose_observation_sequences (one per person)
        → sequence_persons           (person_id → name)
        → pose_observations          (merged keypoint blobs, undistorted)
        → detection_track_assignments (assignment record for UI restore)
```

`finalise_to_db` can be run again (re-stitch), which deletes and recreates the
sequences including any downstream `tracking_runs` and `tracking_results`.
**Existing `pose_observation_edits` would be orphaned by a re-stitch** because
they are keyed by `sequence_id`.  Users should complete stitching before editing.

### Flow 3 — C++ tracking

```
session DB  ←  SessionReader::load_cameras()     (intrinsics, extrinsics, sync)
            ←  SessionReader::load_skeleton_yaml()
            ←  SessionReader::load_tracker_config()
            ←  SessionReader::load_observations(sequence_id)
                    pose_observations  (kp_blob per frame per camera)
                  + pose_observation_edits  (overlay, applied per frame)
                  → ObservationSet

Tracker (UKF predict + update per frame)
    → ResultWriter
        → tracking_results        (state vectors)
        → tracking_obs_results    (2-D residuals)
```

### Flow 4 — keypoint editing (proposed)

```
User selects a named person under a detection run
    → PersonCropGridWidget loads frame_cache_entries (JPEG blobs) for display
    → read_observations_with_edits() loads pose_observations + pose_observation_edits
    → User drags keypoint / toggles outlier
    → write_observation_edit() upserts pose_observation_edits row
User re-runs tracker → Flow 3 picks up edits automatically
```

---

## Python application structure

```
python/
├── posetrak/db/          # installable DB layer (pip install -e .)
│   ├── db.py             # open/create registry + session DBs, migrations
│   ├── session_reader.py # (not to confuse with C++ SessionReader)
│   ├── load_session.py   # loads pose_observations for analysis scripts
│   ├── import_*.py       # importers for YAML, JSON, TOML, H5 calibration files
│   ├── manage_config.py  # CRUD for tracker_configs
│   └── manage_skeleton.py
│
└── app/
    ├── ui/               # top-level Qt shell
    │   ├── main.py       # application entry point
    │   ├── main_window.py  # QMainWindow, hosts session tree + content area
    │   ├── content_panels.py  # per-view content panels
    │   └── session_tree.py    # left-panel tree: sessions → captures → persons
    │
    ├── pose/             # pose extraction UI
    │   ├── main.py       # PoseExtractionWindow (detection + stitching)
    │   ├── detection_pipeline.py  # DetectionPipeline orchestrator
    │   ├── backends_yolo.py       # YOLOv11Detector
    │   ├── backends_rtmpose.py    # RTMPoseEstimator
    │   ├── db_cache.py            # DetectionBatchWriter, read_*/write_* helpers
    │   ├── finalise.py            # finalise_to_db()
    │   ├── frame_view.py          # FrameViewWidget (single camera, raw video seek)
    │   ├── video_canvas.py        # VideoCanvas (letterboxed display + overlays)
    │   ├── filmstrip_stitcher.py  # StitcherWidget (timeline + filmstrip bars)
    │   ├── filmstrip_bar.py       # FilmstripBarItem (one track segment bar)
    │   ├── person_preview.py      # PersonPreviewWidget (zoomed bbox crop)
    │   ├── frame_cache.py         # FrameCache (LRU cache of decoded video frames)
    │   ├── run_tracker.py         # RunTrackerDialog (UI to invoke C++ tracker)
    │   └── cutie_*.py             # Cutie segmentation integration
    │
    └── setup/            # session setup UI (cameras, sync, extrinsics)
        ├── main.py
        ├── page_*.py     # wizard-style setup pages
        └── ...
```

### `FrameViewWidget` vs planned `PersonCropGridWidget`

`FrameViewWidget` seeks raw video files via OpenCV — adequate for
browse/scrub but too slow for frame-by-frame editing.  The proposed
`PersonCropGridWidget` reads JPEG blobs from `frame_cache_entries` (one DB
read per frame, all cameras at once), making frame navigation instantaneous.

---

## C++ tracker structure

```
src/
├── cli/           # posetrak track / scale entry points
├── core/
│   ├── skeleton.hpp / skeleton.cpp     # kinematic tree, joints, markers
│   ├── skeleton_layout.hpp/.cpp        # DOF index table (single source of truth)
│   ├── state.hpp / state.cpp           # error-state representation, quaternion manifold ops
│   ├── observation.hpp                 # Observation, ObservationSet, ObservationSequence
│   └── config.cpp                      # TOML config loading
├── db/
│   ├── session_reader.hpp/.cpp         # SQLite read-only access to session DB
│   ├── result_writer.hpp/.cpp          # writes tracking_results + tracking_obs_results
│   └── blob_codec.hpp/.cpp             # encode/decode float32/float64 blobs; kp deserialization
├── filters/
│   ├── ukf.hpp/.cpp                    # UnscentedKalmanFilter (error-state, Joseph form cov update)
│   └── subset_ukf.hpp/.cpp             # child filter for hierarchical tracking
├── io/
│   ├── observation_loader.hpp/.cpp     # loads from JSON (legacy) or TOML
│   ├── tracking_export.hpp/.cpp        # exports CSV results
│   └── statistics_tracker.hpp/.cpp     # per-frame outlier / NIS statistics
├── kinematics/
│   ├── forward_kinematics.hpp/.cpp     # Pinocchio wrapper: State → marker world positions
│   ├── inverse_kinematics.hpp/.cpp     # damped least-squares IK initializer
│   └── triangulation.hpp/.cpp          # DLT multi-view triangulation
└── tracking/
    └── tracker.hpp/.cpp                # Tracker: orchestrates initialize() + track_frame()
```

### Key C++ design invariants (from CLAUDE.md)

- `SkeletonLayout` is the single source of truth for all DOF-to-state-vector
  index mapping.  Never recompute indices ad-hoc.
- UKF alpha must be ≥ 0.5 for a ~58-DOF state (smaller alpha → negative
  covariance weights).
- SPHERICAL joints always occupy 3 state slots regardless of locked DOFs.
- Error-state formulation: orientations update on the quaternion manifold via
  axis-angle perturbations (`State::apply_error_update`).
- Pinocchio quaternion convention is `[x, y, z, w]` (not `[w, x, y, z]`).
- `ForwardKinematics` must call both `forwardKinematics()` and
  `updateFramePlacements()` to get correct marker world positions.
- Pinocchio is used header-only (`PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION` not
  defined) — no linking against `libpinocchio_default.so`.

---

## Coordinate spaces

| Space | Units | Where used |
|---|---|---|
| Full-frame distorted (K_original) | pixels | Raw video frame.  **All keypoint coordinates in `detection_keypoints` and `pose_observations` (current pipeline) are stored here.** |
| Full-frame undistorted (K_new) | pixels | The C++ tracker's working space after `camera.undistort()` is applied.  Projected-back observations in `tracking_obs_results` live here. |
| Crop display space | pixels | Widget pixels after letterbox scaling; `src_x/y/w/h` from `frame_cache_entries` define the inverse transform back to K_original. |
| Camera space | metres | After full projection: `x_cam = R * x_world + t` |
| World space | metres | The common 3-D reference frame; camera positions stored as `−R^T t` |

### Distorted vs undistorted — the full story

The `DetectionPipeline` docstring states explicitly: *"Coordinates are in
original (distorted) pixel space throughout."*  `finalise_to_db` writes
`pixels_are_undistorted = 0` into every new `pose_observation_sequences` row,
so the C++ tracker always calls `camera->undistort()` on those values before
computing FK / reprojection.

`pose_observation_sequences.pixels_are_undistorted` exists because an earlier
version of the pipeline ran pose estimation on pre-undistorted video frames,
producing K_new coordinates.  The column defaults to `1` in the schema (and
the v4 migration comment says "existing rows default to 1 because all prior
captures used undistorted video").  Those older sequences and any imported
OpenPose JSON data may have `pixels_are_undistorted = 1`; new sequences
produced by the integrated detection pipeline always have it set to `0`.

**What this means for keypoint editing:**
- `pose_observation_edits` must store coordinates in the same distorted pixel
  space as `pose_observations` (K_original).
- Clicks in the `PersonCropGridWidget` are converted to full-frame distorted
  coordinates using `src_x/y/w/h` from `frame_cache_entries`.
- No undistortion is applied in the editing UI; the C++ tracker handles it.

**A comment to be aware of:** `load_session.py:469` says *"Pixels are in
undistorted pixel space (K_new)"* — this refers to `tracking_obs_results`
(the tracker's *output*, projected back from its 3-D state into camera space),
not to `pose_observations`.  It is correct for that table but easy to
misread as describing the input observations.

The `Intrinsics::undistort()` and `camera.project()` methods handle Brown-
Conrady (radial/tangential) and fisheye (OpenCV) distortion models.

---

## Schema versioning

Both DBs use `PRAGMA user_version` as a schema version counter.  Each
migration function in `db.py` bumps the version by 1.  The Python
`open_session()` call checks the version and applies any outstanding
migrations automatically.  The C++ `SessionReader` does **not** check the
version — it reads whatever tables are present, which means new optional tables
(like `pose_observation_edits`) must be handled gracefully with `SQLITE_ERROR`
catch-on-prepare.

Current versions (as of 2026-06):
- Registry DB: version tracked separately in `REGISTRY_SCHEMA_VERSION`
- Session DB: version 20+ (see migration functions in `db.py`)

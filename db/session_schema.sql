-- session_schema.sql
-- Schema for per-session posetrak databases.
-- user_version is set programmatically by posetrak_db.py, not here.
--
-- Session databases are separate SQLite files from the registry.
-- Foreign keys into registry tables (camera_instances, intrinsics_calibrations, etc.)
-- are stored as TEXT IDs but CANNOT use SQLite REFERENCES across separate DB files.
-- Such cross-DB constraints are noted in comments only.
--
-- SELF-CONTAINMENT REQUIREMENT:
--   A session DB file must be fully portable — it must not depend on the registry DB
--   being present on the destination machine.  Concretely this means:
--     • camera_models, camera_modes, camera_instances rows that are referenced by
--       this session are duplicated into the session DB at creation time.
--     • intrinsics_calibrations rows are mirrored into the session DB whenever a
--       calibration is saved (even if the primary write goes to the registry DB).
--   The UI enforces this by writing to both connections whenever a separate registry
--   DB is in use (see IntrinsicsCalibDialog._mirror_to_session and the dual-write
--   pattern in InlineCreateModelDialog / InlineCreateCameraDialog).
--
-- Terminology:
--   capture     — one continuous camera recording (cameras on → off); owns video files.
--   trial       — a named, bounded time window within a capture (one technique/attempt).
--   detection_run — one pose-detection execution over a trial's time window.
-- FK columns referencing captures(id) and capture_videos(id) retain the historical
-- names shot_id / shot_video_id to avoid table-recreation migrations.

-- Top-level session record
CREATE TABLE IF NOT EXISTS mocap_sessions (
    id           TEXT PRIMARY KEY,
    recorded_at  TEXT NOT NULL,
    location     TEXT,
    notes        TEXT
);

-- Cameras that participated in this session.
-- Mode and intrinsics are per-video (on capture_videos), not per-session-camera,
-- so the same physical camera can be used in different modes across captures.
-- camera_instance_id  -- references registry: camera_instances(id)
CREATE TABLE IF NOT EXISTS session_cameras (
    session_id         TEXT NOT NULL REFERENCES mocap_sessions(id),
    camera_instance_id TEXT NOT NULL, -- references registry: camera_instances(id)
    label              TEXT,
    PRIMARY KEY (session_id, camera_instance_id)
);

-- Extrinsic calibration sets (one per calibration event in the session)
CREATE TABLE IF NOT EXISTS extrinsic_calibrations (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES mocap_sessions(id),
    calibrated_at TEXT NOT NULL,
    method        TEXT,
    rms_error     REAL
);

-- Per-camera extrinsic entries belonging to an extrinsic_calibration
-- R and t are little-endian float64 blobs (R: 9 elements row-major, t: 3 elements)
-- camera_instance_id -- references registry: camera_instances(id)
CREATE TABLE IF NOT EXISTS extrinsic_entries (
    extrinsic_calibration_id  TEXT NOT NULL REFERENCES extrinsic_calibrations(id),
    camera_instance_id        TEXT NOT NULL, -- references registry: camera_instances(id)
    R                         BLOB NOT NULL,
    t                         BLOB NOT NULL,
    PRIMARY KEY (extrinsic_calibration_id, camera_instance_id)
);

-- A capture is a single continuous camera recording within a session.
-- extrinsic_calibration_id is nullable: captures may be created before extrinsics are imported.
-- shot_id FK columns in other tables reference captures(id) (historical naming).
CREATE TABLE IF NOT EXISTS captures (
    id                       TEXT PRIMARY KEY,
    session_id               TEXT NOT NULL REFERENCES mocap_sessions(id),
    extrinsic_calibration_id TEXT REFERENCES extrinsic_calibrations(id),
    capture_number           INTEGER NOT NULL,
    label                    TEXT,
    notes                    TEXT
);

-- A trial is a named, bounded time window within a capture: one technique, one attempt.
-- The user-facing unit of analysis (e.g. "shomenuchi shihonage take 1").
CREATE TABLE IF NOT EXISTS trials (
    id           TEXT PRIMARY KEY,
    capture_id   TEXT NOT NULL REFERENCES captures(id),
    name         TEXT,
    time_start_s REAL,
    time_end_s   REAL,
    notes        TEXT
);

-- Video files associated with a capture, one per camera.
-- camera_instance_id     -- references registry: camera_instances(id)
-- camera_mode_id         -- references registry: camera_modes(id); nullable until wizard sets it
-- intrinsics_calibration_id -- references registry: intrinsics_calibrations(id); nullable
-- shot_video_id FK columns in other tables reference capture_videos(id) (historical naming).
CREATE TABLE IF NOT EXISTS capture_videos (
    id                        TEXT PRIMARY KEY,
    shot_id                   TEXT NOT NULL REFERENCES captures(id),
    camera_instance_id        TEXT NOT NULL, -- references registry: camera_instances(id)
    file_path                 TEXT NOT NULL,
    first_video_frame         INTEGER NOT NULL,
    last_video_frame          INTEGER NOT NULL,
    actual_fps                REAL NOT NULL,
    camera_mode_id            TEXT,          -- references registry: camera_modes(id)
    intrinsics_calibration_id TEXT           -- references registry: intrinsics_calibrations(id)
);

-- Synchronisation configuration: maps each camera to a common time axis
CREATE TABLE IF NOT EXISTS sync_configs (
    id         TEXT PRIMARY KEY,
    shot_id    TEXT NOT NULL REFERENCES captures(id),
    created_by TEXT,
    notes      TEXT
);

-- Per-camera sync points within a sync configuration.
-- Multiple rows per camera are allowed; the tracker uses all of them for
-- piecewise-linear timestamp interpolation between anchor frames.
-- camera_instance_id -- references registry: camera_instances(id)
CREATE TABLE IF NOT EXISTS sync_points (
    sync_config_id     TEXT    NOT NULL REFERENCES sync_configs(id),
    camera_instance_id TEXT    NOT NULL, -- references registry: camera_instances(id)
    shot_video_id      TEXT    NOT NULL REFERENCES capture_videos(id),
    video_frame        INTEGER NOT NULL,
    timestamp_s        REAL    NOT NULL,
    PRIMARY KEY (sync_config_id, camera_instance_id, video_frame)
);

-- Input layer for the graph-based sync solver (see sync_solver.py).
-- Each sync_anchor records one real-world event (e.g. LED flash, clap)
-- that was simultaneously visible in two or more camera videos.
CREATE TABLE IF NOT EXISTS sync_anchors (
    id         TEXT PRIMARY KEY,
    shot_id    TEXT NOT NULL REFERENCES captures(id),
    notes      TEXT
);

-- Per-video observation of a sync_anchor event.
-- video_frame: integer frame number (0-based) where the event is visible.
-- subframe: fractional frame from LED brightness peak fit; 0.0 for manual.
CREATE TABLE IF NOT EXISTS sync_anchor_observations (
    id             TEXT    PRIMARY KEY,
    sync_anchor_id TEXT    NOT NULL REFERENCES sync_anchors(id),
    shot_video_id  TEXT    NOT NULL REFERENCES capture_videos(id),
    video_frame    INTEGER NOT NULL,
    subframe       REAL    NOT NULL DEFAULT 0.0,
    UNIQUE (sync_anchor_id, shot_video_id)
);

-- A sequence of 2-D pose observations covering a time window of a capture.
-- name: user-assigned label (e.g. performer name or role).
-- pixels_are_undistorted: 1 if keypoint coordinates are already in undistorted
--   pixel space (K_new) and must NOT be undistorted by the tracker again;
--   0 if coordinates are in distorted pixel space (K_original) and the tracker
--   must apply undistortion.  Default 1 matches the current pipeline where pose
--   estimation runs on pre-undistorted video frames.
-- detection_run_id: optional link to the detection run that produced the observations.
CREATE TABLE IF NOT EXISTS pose_observation_sequences (
    id                      TEXT PRIMARY KEY,
    shot_id                 TEXT NOT NULL REFERENCES captures(id),
    sync_config_id          TEXT NOT NULL REFERENCES sync_configs(id),
    time_start_s            REAL NOT NULL,
    time_end_s              REAL NOT NULL,
    name                    TEXT,
    pose_model              TEXT,
    notes                   TEXT,
    pixels_are_undistorted  INTEGER NOT NULL DEFAULT 1,
    detection_run_id        TEXT REFERENCES detection_runs(id)
);

-- Maps integer person_id → human-readable person name within a sequence.
-- Written by finalise_to_db so assignment colours can be restored on reopen.
CREATE TABLE IF NOT EXISTS sequence_persons (
    sequence_id TEXT    NOT NULL REFERENCES pose_observation_sequences(id),
    person_id   INTEGER NOT NULL,
    person_name TEXT    NOT NULL,
    PRIMARY KEY (sequence_id, person_id)
);

-- Individual 2-D pose observations: one row per (sequence, camera, frame, person, source)
-- kp_blob: little-endian float32 array shaped [n_keypoints, 3] (x, y, confidence)
-- camera_instance_id -- references registry: camera_instances(id)
-- noise_scale: measurement noise scale factor (bbox_w / pose_input_width)
-- source: 'body' | 'hand_l' | 'hand_r' — matches detection_keypoints.region_type's
--   spelling ('full_body' rows become source='body' here). Multiple sources can
--   coexist for the same (sequence, camera, frame, person); the C++ loader merges
--   them into one dense per-marker array, each source keeping its own noise_scale.
-- detection_run_id: the detection run that produced this row. Usually matches the
--   sequence's own detection_run_id, but tracked per-row so a source's provenance
--   is unambiguous even if that changes in the future.
CREATE TABLE IF NOT EXISTS pose_observations (
    sequence_id        TEXT    NOT NULL REFERENCES pose_observation_sequences(id),
    camera_instance_id TEXT    NOT NULL, -- references registry: camera_instances(id)
    video_frame        INTEGER NOT NULL,
    timestamp_s        REAL    NOT NULL,
    person_id          INTEGER NOT NULL,
    source             TEXT    NOT NULL DEFAULT 'body',
    detection_run_id   TEXT    REFERENCES detection_runs(id),
    kp_blob            BLOB    NOT NULL,
    noise_scale        REAL,
    PRIMARY KEY (sequence_id, camera_instance_id, video_frame, person_id, source)
);

-- Non-destructive keypoint edits applied on top of pose_observations.
-- kp_blob: float32[N,3] (x, y, is_outlier) — same N as pose_observations for the sequence.
-- kp_mask: uint8[ceil(N/8)] bitmask; bit i set means slot i is overridden.
CREATE TABLE IF NOT EXISTS pose_observation_edits (
    id                 TEXT PRIMARY KEY,
    sequence_id        TEXT NOT NULL REFERENCES pose_observation_sequences(id),
    camera_instance_id TEXT NOT NULL, -- references registry: camera_instances(id)
    video_frame        INTEGER NOT NULL,
    kp_blob            BLOB NOT NULL,
    kp_mask            BLOB NOT NULL,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS pose_observation_edits_unique
    ON pose_observation_edits (sequence_id, camera_instance_id, video_frame);

-- A single tracker execution record
-- tracker_config_id  -- references registry: tracker_configs(id)
-- skeleton_id        -- references registry: skeletons(id)
-- active_camera_ids  -- JSON array of camera instance ID strings
-- marker_names       -- JSON array of marker name strings (column index in state blobs)
CREATE TABLE IF NOT EXISTS tracking_runs (
    id                       TEXT PRIMARY KEY,
    observation_sequence_id  TEXT NOT NULL REFERENCES pose_observation_sequences(id),
    tracker_config_id        TEXT NOT NULL, -- references registry: tracker_configs(id)
    skeleton_id              TEXT NOT NULL, -- references registry: skeletons(id)
    extrinsic_calibration_id TEXT NOT NULL REFERENCES extrinsic_calibrations(id),
    sync_config_id           TEXT NOT NULL REFERENCES sync_configs(id),
    ran_at                   TEXT NOT NULL,
    posetrak_version         TEXT NOT NULL,
    active_camera_ids        TEXT NOT NULL,
    marker_names             TEXT NOT NULL,
    notes                    TEXT,
    trial_id                 TEXT REFERENCES trials(id)
);

-- Per-person skeleton override within a run (supports multi-person tracking)
-- skeleton_id -- references registry: skeletons(id)
CREATE TABLE IF NOT EXISTS tracking_run_persons (
    run_id     TEXT    NOT NULL REFERENCES tracking_runs(id),
    person_id  INTEGER NOT NULL,
    skeleton_id TEXT   NOT NULL, -- references registry: skeletons(id)
    PRIMARY KEY (run_id, person_id)
);

-- Per-frame tracking results
-- state    : little-endian float64 blob — full UKF state vector
-- cov_diag : little-endian float64 blob — diagonal of covariance matrix
CREATE TABLE IF NOT EXISTS tracking_results (
    run_id                TEXT    NOT NULL REFERENCES tracking_runs(id),
    person_id             INTEGER NOT NULL,
    tracker_step          INTEGER NOT NULL,
    is_smoothed           INTEGER NOT NULL DEFAULT 0,
    timestamp_s           REAL    NOT NULL,
    tracking_lost         INTEGER NOT NULL DEFAULT 0,
    n_inlier_observations INTEGER,
    cov_condition_number  REAL,
    nis_value             REAL,
    nis_dof               INTEGER,
    state                 BLOB    NOT NULL,
    cov_diag              BLOB    NOT NULL,
    PRIMARY KEY (run_id, person_id, tracker_step, is_smoothed)
);

-- Per-frame observation residuals / projected observations
-- obs_blob: little-endian float32 blob — projected 2-D marker observations
CREATE TABLE IF NOT EXISTS tracking_obs_results (
    run_id       TEXT    NOT NULL REFERENCES tracking_runs(id),
    person_id    INTEGER NOT NULL,
    tracker_step INTEGER NOT NULL,
    obs_blob     BLOB    NOT NULL,
    PRIMARY KEY (run_id, person_id, tracker_step)
);

-- Detection runs: one row per execution of the pose extraction pipeline.
-- Tracks which detector/pose model was used, over which time range.
-- trial_id: optional link to the trial this run belongs to.
CREATE TABLE IF NOT EXISTS detection_runs (
    id                  TEXT PRIMARY KEY,
    shot_id             TEXT NOT NULL REFERENCES captures(id),
    sync_config_id      TEXT NOT NULL REFERENCES sync_configs(id),
    trial_id            TEXT REFERENCES trials(id),
    time_start_s        REAL NOT NULL,
    time_end_s          REAL NOT NULL,
    detector_model      TEXT NOT NULL,
    pose_model          TEXT NOT NULL,
    detector_version    TEXT,
    pose_version        TEXT,
    detector_conf       REAL NOT NULL DEFAULT 0.3,
    pose_conf_threshold REAL NOT NULL DEFAULT 0.3,
    pose_input_width    INTEGER,
    pose_input_height   INTEGER,
    status              TEXT NOT NULL DEFAULT 'running',
    created_at          TEXT NOT NULL,
    completed_at        TEXT
);

-- Raw keypoints produced by pose estimation, keyed by detection run.
-- keypoints: float32 blob shaped [n_kp, 3] (x, y, confidence) in distorted px.
-- noise_scale: bbox_w / pose_input_width, used when converting to pose_observations.
CREATE TABLE IF NOT EXISTS detection_keypoints (
    detection_run_id    TEXT NOT NULL REFERENCES detection_runs(id),
    shot_video_id       TEXT NOT NULL REFERENCES capture_videos(id),
    video_frame         INTEGER NOT NULL,
    track_id            INTEGER NOT NULL,
    region_type         TEXT NOT NULL DEFAULT 'full_body',
    keypoints           BLOB NOT NULL,
    noise_scale         REAL,
    PRIMARY KEY (detection_run_id, shot_video_id, video_frame, track_id, region_type)
);

-- Person detections: one row per (detection run, video, frame, track, region type).
-- Model-agnostic; supports full-body, face, and hand detection models.
-- region_type: 'full_body' | 'face' | 'hand_l' | 'hand_r'
-- track_id: detection tracker ID, assigned before person identity is known.
CREATE TABLE IF NOT EXISTS person_detections (
    detection_run_id    TEXT NOT NULL REFERENCES detection_runs(id),
    shot_video_id       TEXT NOT NULL REFERENCES capture_videos(id),
    video_frame         INTEGER NOT NULL,
    track_id            INTEGER NOT NULL,
    region_type         TEXT    NOT NULL DEFAULT 'full_body',
    model_name          TEXT,
    bbox_x              REAL,
    bbox_y              REAL,
    bbox_w              REAL,
    bbox_h              REAL,
    confidence          REAL,
    PRIMARY KEY (detection_run_id, shot_video_id, video_frame, track_id, region_type)
);

CREATE INDEX IF NOT EXISTS idx_person_detections_run_video
    ON person_detections(detection_run_id, shot_video_id, video_frame);

-- Segmentation quality run: parameters and provenance for one add_seg_quality execution.
-- mask_dir: optional path to a directory containing per-video NPZ debug mask files.
CREATE TABLE IF NOT EXISTS seg_quality_runs (
    id               TEXT PRIMARY KEY,
    detection_run_id TEXT NOT NULL,  -- references detection_runs(id)
    created_at       TEXT NOT NULL,
    quality_source   TEXT NOT NULL DEFAULT 'cutie',
    erosion_px       INTEGER NOT NULL DEFAULT 5,
    mask_dir         TEXT,
    notes            TEXT
);

-- Per-keypoint segmentation quality scores, aligned with detection_keypoints.
-- quality_blob: little-endian float32 array of length N_KEYPOINTS (133).
-- Values: 1.0=inside, 0.5=boundary, 0.0=outside, -1.0=unavailable.
-- One row per (seg_run, video, frame, track_id); mirrors detection_keypoints PK.
CREATE TABLE IF NOT EXISTS keypoint_obs_quality (
    seg_run_id    TEXT    NOT NULL,  -- references seg_quality_runs(id)
    shot_video_id TEXT    NOT NULL,  -- references capture_videos(id)
    video_frame   INTEGER NOT NULL,
    track_id      INTEGER NOT NULL,
    quality_blob  BLOB    NOT NULL,
    PRIMARY KEY (seg_run_id, shot_video_id, video_frame, track_id)
);

CREATE INDEX IF NOT EXISTS idx_keypoint_obs_quality_video_frame
    ON keypoint_obs_quality (shot_video_id, video_frame);

-- Person tracks: one row per continuous track span within a detection run.
CREATE TABLE IF NOT EXISTS person_tracks (
    id                  TEXT PRIMARY KEY,
    detection_run_id    TEXT NOT NULL REFERENCES detection_runs(id),
    shot_video_id       TEXT NOT NULL REFERENCES capture_videos(id),
    track_id            INTEGER NOT NULL,
    first_frame         INTEGER NOT NULL,
    last_frame          INTEGER NOT NULL,
    UNIQUE (detection_run_id, shot_video_id, track_id)
);

-- Frame cache: stores decoded/cropped image data for fast UI access.
-- cache_type: 'full_frame' | 'thumb' | 'person_crop'  (CacheType.value)
-- track_id:   -1 for cache types that do not require a person track (e.g. thumb)
-- region_type: '' for thumb/full_frame; 'full_body'/'face'/'hand_l'/'hand_r' for person_crop
-- width_px / height_px: output image dimensions (0 if not applicable).
-- detection_run_id: '' for non-run-linked entries; run ID for person_crop entries.
--   Included in the PK so crops from multiple detection runs on the same shot
--   coexist without conflict.
CREATE TABLE IF NOT EXISTS frame_cache_entries (
    shot_video_id       TEXT    NOT NULL REFERENCES capture_videos(id),
    frame_idx           INTEGER NOT NULL,
    cache_type          TEXT    NOT NULL,
    track_id            INTEGER NOT NULL DEFAULT -1,
    region_type         TEXT    NOT NULL DEFAULT '',
    width_px            INTEGER NOT NULL DEFAULT 0,
    height_px           INTEGER NOT NULL DEFAULT 0,
    image_data          BLOB    NOT NULL,
    detection_run_id    TEXT    NOT NULL DEFAULT '',
    -- Source rectangle in original full-resolution frame (pixels).
    -- Set for PERSON_CROP; NULL for FULL_FRAME and THUMB.
    -- src_w/src_h are the crop dimensions BEFORE any JPEG downscale.
    src_x               INTEGER,
    src_y               INTEGER,
    src_w               INTEGER,
    src_h               INTEGER,
    PRIMARY KEY (shot_video_id, frame_idx, cache_type, track_id, region_type, width_px, detection_run_id)
);

-- Explicit track_id → person_name assignments from the pose extraction UI.
-- Allows restoring assignments when a detection run is reopened.
CREATE TABLE IF NOT EXISTS detection_track_assignments (
    detection_run_id TEXT    NOT NULL REFERENCES detection_runs(id),
    shot_video_id    TEXT    NOT NULL,
    track_id         INTEGER NOT NULL,
    person_name      TEXT    NOT NULL,
    first_frame      INTEGER NOT NULL,
    last_frame       INTEGER NOT NULL,
    PRIMARY KEY (detection_run_id, shot_video_id, track_id, first_frame)
);
-- Segmentation masks for interactive Cutie init widget.
-- One row per (seg_quality_run, camera, frame).  Indexed PNG blob, label 0=bg, 1..N=person.
CREATE TABLE IF NOT EXISTS seg_masks (
    seg_quality_run_id TEXT    NOT NULL REFERENCES seg_quality_runs(id),
    shot_video_id      TEXT    NOT NULL REFERENCES capture_videos(id),
    frame_idx          INTEGER NOT NULL,
    mask_blob          BLOB    NOT NULL,
    PRIMARY KEY (seg_quality_run_id, shot_video_id, frame_idx)
);

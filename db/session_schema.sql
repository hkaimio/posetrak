-- session_schema.sql
-- Schema for per-session posetrak databases.
-- user_version is set programmatically by posetrak_db.py, not here.
--
-- Session databases are separate SQLite files from the registry.
-- Foreign keys into registry tables (camera_instances, intrinsics_calibrations, etc.)
-- are stored as TEXT IDs but CANNOT use SQLite REFERENCES across separate DB files.
-- Such cross-DB constraints are noted in comments only.

-- Top-level session record
CREATE TABLE IF NOT EXISTS mocap_sessions (
    id           TEXT PRIMARY KEY,
    recorded_at  TEXT NOT NULL,
    location     TEXT,
    notes        TEXT
);

-- Cameras that participated in this session.
-- Mode and intrinsics are per-video (on shot_videos), not per-session-camera,
-- so the same physical camera can be used in different modes across shots.
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

-- A shot is a single continuous capture take within a session
-- extrinsic_calibration_id is nullable: shots may be created before extrinsics are imported
CREATE TABLE IF NOT EXISTS shots (
    id                       TEXT PRIMARY KEY,
    session_id               TEXT NOT NULL REFERENCES mocap_sessions(id),
    extrinsic_calibration_id TEXT REFERENCES extrinsic_calibrations(id),
    shot_number              INTEGER NOT NULL,
    label                    TEXT,
    notes                    TEXT
);

-- Video files associated with a shot, one per camera.
-- camera_instance_id     -- references registry: camera_instances(id)
-- camera_mode_id         -- references registry: camera_modes(id); nullable until wizard sets it
-- intrinsics_calibration_id -- references registry: intrinsics_calibrations(id); nullable
CREATE TABLE IF NOT EXISTS shot_videos (
    id                        TEXT PRIMARY KEY,
    shot_id                   TEXT NOT NULL REFERENCES shots(id),
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
    shot_id    TEXT NOT NULL REFERENCES shots(id),
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
    shot_video_id      TEXT    NOT NULL REFERENCES shot_videos(id),
    video_frame        INTEGER NOT NULL,
    timestamp_s        REAL    NOT NULL,
    PRIMARY KEY (sync_config_id, camera_instance_id, video_frame)
);

-- A sequence of 2-D pose observations covering a time window of a shot
-- pixels_are_undistorted: 1 if keypoint coordinates are already in undistorted
--   pixel space (K_new) and must NOT be undistorted by the tracker again;
--   0 if coordinates are in distorted pixel space (K_original) and the tracker
--   must apply undistortion.  Default 1 matches the current pipeline where pose
--   estimation runs on pre-undistorted video frames.
-- detection_run_id: optional link to the detection run that produced the observations.
CREATE TABLE IF NOT EXISTS pose_observation_sequences (
    id                      TEXT PRIMARY KEY,
    shot_id                 TEXT NOT NULL REFERENCES shots(id),
    sync_config_id          TEXT NOT NULL REFERENCES sync_configs(id),
    time_start_s            REAL NOT NULL,
    time_end_s              REAL NOT NULL,
    pose_model              TEXT,
    notes                   TEXT,
    pixels_are_undistorted  INTEGER NOT NULL DEFAULT 1,
    detection_run_id        TEXT REFERENCES detection_runs(id)
);

-- Individual 2-D pose observations: one row per (sequence, camera, frame, person)
-- kp_blob: little-endian float32 array shaped [n_keypoints, 3] (x, y, confidence)
-- camera_instance_id -- references registry: camera_instances(id)
-- noise_scale: measurement noise scale factor (bbox_w / pose_input_width)
CREATE TABLE IF NOT EXISTS pose_observations (
    sequence_id        TEXT    NOT NULL REFERENCES pose_observation_sequences(id),
    camera_instance_id TEXT    NOT NULL, -- references registry: camera_instances(id)
    video_frame        INTEGER NOT NULL,
    timestamp_s        REAL    NOT NULL,
    person_id          INTEGER NOT NULL,
    kp_blob            BLOB    NOT NULL,
    noise_scale        REAL,
    PRIMARY KEY (sequence_id, camera_instance_id, video_frame, person_id)
);

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
    marker_names             TEXT NOT NULL
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
CREATE TABLE IF NOT EXISTS detection_runs (
    id                  TEXT PRIMARY KEY,
    shot_id             TEXT NOT NULL REFERENCES shots(id),
    sync_config_id      TEXT NOT NULL REFERENCES sync_configs(id),
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
    shot_video_id       TEXT NOT NULL REFERENCES shot_videos(id),
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
    shot_video_id       TEXT NOT NULL REFERENCES shot_videos(id),
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

-- Person tracks: one row per continuous track span within a detection run.
CREATE TABLE IF NOT EXISTS person_tracks (
    id                  TEXT PRIMARY KEY,
    detection_run_id    TEXT NOT NULL REFERENCES detection_runs(id),
    shot_video_id       TEXT NOT NULL REFERENCES shot_videos(id),
    track_id            INTEGER NOT NULL,
    first_frame         INTEGER NOT NULL,
    last_frame          INTEGER NOT NULL,
    UNIQUE (detection_run_id, shot_video_id, track_id)
);

-- Frame cache: stores decoded/cropped image data for fast UI access.
-- cache_type: 'THUMB' | 'PERSON_CROP'
-- track_id:   -1 for cache types that do not require a person track (e.g. THUMB)
-- region_type: '' (empty) for THUMB; 'full_body'/'face'/'hand_l'/'hand_r' for PERSON_CROP
-- width_px / height_px: output image dimensions.
-- detection_run_id: optional link to the detection run that produced this crop.
CREATE TABLE IF NOT EXISTS frame_cache_entries (
    shot_video_id       TEXT    NOT NULL REFERENCES shot_videos(id),
    frame_idx           INTEGER NOT NULL,
    cache_type          TEXT    NOT NULL,
    track_id            INTEGER NOT NULL DEFAULT -1,
    region_type         TEXT    NOT NULL DEFAULT '',
    width_px            INTEGER NOT NULL,
    height_px           INTEGER NOT NULL,
    image_data          BLOB    NOT NULL,
    detection_run_id    TEXT    REFERENCES detection_runs(id),
    PRIMARY KEY (shot_video_id, frame_idx, cache_type, track_id, region_type, width_px)
);

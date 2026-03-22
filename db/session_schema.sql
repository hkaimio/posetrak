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

-- Cameras that participated in this session, with their calibration references
-- camera_instance_id  -- references registry: camera_instances(id)
-- camera_mode_id      -- references registry: camera_modes(id)
-- intrinsics_calibration_id -- references registry: intrinsics_calibrations(id)
CREATE TABLE IF NOT EXISTS session_cameras (
    session_id                  TEXT NOT NULL REFERENCES mocap_sessions(id),
    camera_instance_id          TEXT NOT NULL, -- references registry: camera_instances(id)
    camera_mode_id              TEXT NOT NULL, -- references registry: camera_modes(id)
    intrinsics_calibration_id   TEXT NOT NULL, -- references registry: intrinsics_calibrations(id)
    label                       TEXT,
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

-- Video files associated with a shot, one per camera
-- camera_instance_id -- references registry: camera_instances(id)
CREATE TABLE IF NOT EXISTS shot_videos (
    id                 TEXT PRIMARY KEY,
    shot_id            TEXT NOT NULL REFERENCES shots(id),
    camera_instance_id TEXT NOT NULL, -- references registry: camera_instances(id)
    file_path          TEXT NOT NULL,
    first_video_frame  INTEGER NOT NULL,
    last_video_frame   INTEGER NOT NULL,
    actual_fps         REAL NOT NULL
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
CREATE TABLE IF NOT EXISTS pose_observation_sequences (
    id               TEXT PRIMARY KEY,
    shot_id          TEXT NOT NULL REFERENCES shots(id),
    sync_config_id   TEXT NOT NULL REFERENCES sync_configs(id),
    time_start_s     REAL NOT NULL,
    time_end_s       REAL NOT NULL,
    pose_model       TEXT,
    notes            TEXT
);

-- Individual 2-D pose observations: one row per (sequence, camera, frame, person)
-- kp_blob: little-endian float32 array shaped [n_keypoints, 3] (x, y, confidence)
-- camera_instance_id -- references registry: camera_instances(id)
CREATE TABLE IF NOT EXISTS pose_observations (
    sequence_id        TEXT    NOT NULL REFERENCES pose_observation_sequences(id),
    camera_instance_id TEXT    NOT NULL, -- references registry: camera_instances(id)
    video_frame        INTEGER NOT NULL,
    timestamp_s        REAL    NOT NULL,
    person_id          INTEGER NOT NULL,
    kp_blob            BLOB    NOT NULL,
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

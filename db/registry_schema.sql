-- registry_schema.sql
-- Schema for the posetrak registry database.
-- user_version is set programmatically by posetrak_db.py, not here.

-- key 'project_root' is used for relative path resolution of session files
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Physical camera hardware models (make/model, sensor)
CREATE TABLE IF NOT EXISTS camera_models (
    id            TEXT PRIMARY KEY,
    manufacturer  TEXT,
    model_name    TEXT,
    sensor_size   TEXT
);

-- Capture modes associated with a camera model (resolution, fps, codec)
CREATE TABLE IF NOT EXISTS camera_modes (
    id               TEXT PRIMARY KEY,
    camera_model_id  TEXT    NOT NULL REFERENCES camera_models(id),
    width_px         INTEGER NOT NULL DEFAULT 0,
    height_px        INTEGER NOT NULL DEFAULT 0,
    nominal_fps      REAL    NOT NULL DEFAULT 0.0,
    codec            TEXT,
    notes            TEXT
);

-- Individual physical camera units (serial number, user label)
CREATE TABLE IF NOT EXISTS camera_instances (
    id               TEXT PRIMARY KEY,
    camera_model_id  TEXT NOT NULL REFERENCES camera_models(id),
    serial_number    TEXT,
    label            TEXT NOT NULL
);

-- Intrinsic calibrations tied to a specific camera mode
-- dist_coeffs is a little-endian float64 blob (radtan: [k1,k2,p1,p2], fisheye: [k1,k2,k3,k4])
CREATE TABLE IF NOT EXISTS intrinsics_calibrations (
    id                TEXT PRIMARY KEY,
    camera_mode_id    TEXT NOT NULL REFERENCES camera_modes(id),
    calibrated_at     TEXT NOT NULL,
    calibration_tool  TEXT,
    distortion_model  TEXT NOT NULL DEFAULT 'radtan',
    fx                REAL NOT NULL,
    fy                REAL NOT NULL,
    cx                REAL NOT NULL,
    cy                REAL NOT NULL,
    dist_coeffs       BLOB,
    rms_error         REAL,
    notes             TEXT
);

-- Skeleton definitions; id is SHA-256 of yaml_content
-- parent_id allows tracking skeleton lineage (e.g. scaled versions)
CREATE TABLE IF NOT EXISTS skeletons (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    parent_id    TEXT REFERENCES skeletons(id),
    person_label TEXT,
    source       TEXT,
    yaml_content TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    notes        TEXT
);

-- UKF / tracker configuration snapshots
CREATE TABLE IF NOT EXISTS tracker_configs (
    id                     TEXT PRIMARY KEY,
    name                   TEXT NOT NULL,
    parent_id              TEXT REFERENCES tracker_configs(id),
    created_at             TEXT NOT NULL,
    alpha                  REAL,
    beta                   REAL,
    kappa                  REAL,
    process_noise_std      REAL,
    measurement_noise_std  REAL,
    outlier_threshold      REAL,
    tracker_fps            REAL,
    ik_max_iterations      INTEGER,
    ik_tolerance           REAL,
    init_position_std      REAL,
    init_orientation_std   REAL,
    init_joint_std         REAL,
    init_velocity_std      REAL,
    min_cameras_for_init   INTEGER,
    notes                  TEXT
);

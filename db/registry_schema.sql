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
-- default_intrinsics_calibration_id: the preferred calibration for this mode,
-- auto-offered by the shot wizard; NULL if no default has been set.
CREATE TABLE IF NOT EXISTS camera_modes (
    id                                TEXT PRIMARY KEY,
    camera_model_id                   TEXT    NOT NULL REFERENCES camera_models(id),
    width_px                          INTEGER NOT NULL DEFAULT 0,
    height_px                         INTEGER NOT NULL DEFAULT 0,
    nominal_fps                       REAL    NOT NULL DEFAULT 0.0,
    codec                             TEXT,
    notes                             TEXT,
    default_intrinsics_calibration_id TEXT    REFERENCES intrinsics_calibrations(id)
);

-- Individual physical camera units (serial number, user label)
CREATE TABLE IF NOT EXISTS camera_instances (
    id               TEXT PRIMARY KEY,
    camera_model_id  TEXT NOT NULL REFERENCES camera_models(id),
    serial_number    TEXT,
    label            TEXT NOT NULL
);

-- Intrinsic calibrations tied to a specific camera mode
-- dist_coeffs: little-endian float64 blob (radtan: [k1,k2,p1,p2], fisheye: [k1,k2,k3,k4])
-- matrix_original: little-endian float64 blob, 3×3 row-major — K directly from calibrateCamera()
-- undistort_mapx/mapy: zlib-compressed float32 arrays (cv2.remap maps), shape (height, width)
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
    notes             TEXT,
    image_width       INTEGER,
    image_height      INTEGER,
    matrix_original   BLOB,
    undistort_mapx    BLOB,
    undistort_mapy    BLOB
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
    process_noise_vel_std  REAL,
    velocity_half_life_s   REAL,
    measurement_noise_std  REAL,
    outlier_threshold      REAL,
    tracker_fps            REAL,
    ik_max_iterations      INTEGER,
    ik_tolerance           REAL,
    init_position_std      REAL,
    init_orientation_std   REAL,
    init_joint_std         REAL,
    init_velocity_std      REAL,
    min_cameras_for_init              INTEGER,
    velocity_mode_camera_ids          TEXT,  -- JSON array of integer camera indices; NULL = all cameras use position mode
    velocity_measurement_noise_std    REAL,  -- Measurement noise std for velocity cameras (pixels/frame); NULL = use measurement_noise_std
    notes                             TEXT,
    -- Added in schema migration v22 (pose_noise_std), v23 (relative obs), v24 (cross-pair obs):
    pose_noise_std                    REAL,  -- Pose estimation noise std in model-input pixels
    use_relative_observations         INTEGER, -- 0/1 flag for child-minus-parent pixel observations
    relative_min_confidence           REAL,  -- Min keypoint confidence for relative pairs
    cross_pair_max_px                 REAL,  -- Pixel radius for spatial cross-pair observations; NULL = disabled
    cross_pair_max_n                  INTEGER, -- Max cross-pairs per frame per camera
    -- Added in schema migration v26: adaptive process noise (Phase 1, velocity-driven per-DOF scaling)
    process_noise_vel_gain_joint      REAL,  -- Velocity gain for joint DOFs; NULL/0 = disabled
    process_noise_vel_ref_joint       REAL,  -- Reference velocity for joint DOFs (rad/s)
    process_noise_vel_gain_root       REAL,  -- Velocity gain for root DOFs; NULL/0 = disabled
    process_noise_vel_ref_root        REAL,  -- Reference velocity for root DOFs (m/s, rad/s)
    -- Added in schema migration v27: scope the joint gain to specific joints by literal
    -- name (e.g. exclude arms) instead of applying it body-wide. Name-based rather than
    -- skeleton-group-based since existing skeleton YAMLs don't define groups fine-grained
    -- enough for this (one "main" group spans the whole body).
    process_noise_vel_joint_names     TEXT,  -- JSON array of joint names; NULL/[] = all joints
    -- Added in schema migration v28: pose regularization for a kinematically redundant
    -- joint chain (e.g. spine1/spine2) -- see
    -- docs/roadmap/features/pose-regularization/pose-regularization-design.md.
    pose_reg_joint_names              TEXT,  -- JSON array of joint names; NULL/[] = disabled
    pose_reg_equal_split_noise_std    REAL,  -- Radians; NULL/0 = disabled
    pose_reg_rest_pose_noise_std      REAL,  -- Radians; NULL/0 = disabled
    -- Added in schema migration v29: NIS-feedback regional fading safety net
    -- (Mechanism B) -- see
    -- docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md.
    nis_feedback_scopes               TEXT,  -- JSON array of {name, joint_names}; NULL/[] = disabled
    nis_feedback_window               INTEGER,  -- Moving window size, in tracker steps
    nis_feedback_threshold            REAL,     -- Windowed NIS/DOF above this triggers fading
    nis_feedback_max_multiplier       REAL,     -- Cap on the variance-domain multiplier
    -- Added in schema migration v31 (replacing v30's single hardcoded "arms" scope
    -- once one split stopped being enough): an arbitrary list of additional,
    -- independent adaptive process noise gain scopes beyond
    -- process_noise_vel_gain_joint/process_noise_vel_joint_names above -- see
    -- UnscentedKalmanFilter::set_velocity_noise_gain_scopes().
    process_noise_vel_scopes TEXT,  -- JSON array of {name, joint_names, gain, vel_ref}; NULL/[] = none
    -- Added in schema migration v32: soft joint-limit repulsion -- see
    -- docs/roadmap/features/soft-joint-limits/soft-joint-limits-design.md.
    soft_limit_joint_names            TEXT,  -- JSON array of joint names; NULL/[] = disabled
    soft_limit_margin_rad             REAL,  -- Radians; width of the soft zone
    soft_limit_noise_std              REAL,  -- Radians; NULL/0 = disabled
    -- Added in schema migration v33: near-limit process-noise damping -- see
    -- docs/roadmap/features/tracking-crisis-debugging-log.md, "Proposals".
    near_limit_damping_joint_names    TEXT,  -- JSON array of joint names; NULL/[] = disabled
    near_limit_margin_rad             REAL,  -- Radians; detection-zone width
    near_limit_spread_sigma           REAL,  -- Multiplier on sqrt(covariance) for spread estimate
    near_limit_damping_factor         REAL,  -- Variance-domain multiplier; NULL/1.0 = disabled
    -- Added in schema migration v34: trusted keypoint edits (Phase 0) -- see
    -- docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md.
    edited_kp_noise_std               REAL,  -- Pixels; NULL/0 = disabled
    -- Added in schema migration v36: cross-person relative observations (Phase 5,
    -- error-improvements) -- see
    -- docs/roadmap/features/error-improvements/phase5-cross-person-plan.md.
    cross_person_max_world_mm         REAL,  -- 3D marker-pair distance gate (mm); NULL/0 = disabled
    cross_person_min_confidence       REAL,  -- Min keypoint confidence for both people's detections
    cross_person_max_n                INTEGER  -- Max cross-person anchors per pair per camera per frame
);

-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

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

-- Marker body definitions: a named, reusable rigid body carrying one or
-- more fiducial markers at fixed positions (portable calibration rigs,
-- future marker-cluster objects) -- id is SHA-256 of yaml_content, same
-- content-addressed convention as skeletons above. See
-- docs/roadmap/features/extrinsics-improvements/
-- extrinsics-improvements-design.md, section 10.
CREATE TABLE IF NOT EXISTS marker_body_definitions (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    yaml_content TEXT NOT NULL,
    source       TEXT,
    created_at   TEXT NOT NULL,
    notes        TEXT
);

-- UKF / tracker configuration snapshots
CREATE TABLE IF NOT EXISTS tracker_configs (
    id                     TEXT PRIMARY KEY,                    -- UUID of the config snapshot
    name                   TEXT NOT NULL,                       -- display name; auto-generated for per-run snapshots (is_named = 0)
    parent_id              TEXT REFERENCES tracker_configs(id), -- config this one was derived from by an edit; NULL for a root
    created_at             TEXT NOT NULL,                       -- ISO 8601 UTC
    alpha                  REAL,    -- UKF sigma-point spread; about 0.5 for a ~58-dimensional state (tiny values give negative covariance weights)
    beta                   REAL,    -- UKF prior-distribution parameter; 2 is optimal for a Gaussian
    kappa                  REAL,    -- UKF secondary scaling parameter; usually 0
    process_noise_std      REAL,    -- process noise std for angle/position DOFs
    process_noise_vel_std  REAL,    -- process noise std for velocity DOFs; NULL = same as process_noise_std
    velocity_half_life_s   REAL,    -- velocity decay half-life in seconds; NULL = no decay
    measurement_noise_std  REAL,    -- legacy name of the calibration noise: loaded as calib_noise_std, pixels in the original video
    outlier_threshold      REAL,    -- chi-squared threshold for rejecting an observation as an outlier
    tracker_fps            REAL,    -- tracker sample rate, Hz
    ik_max_iterations      INTEGER, -- inverse-kinematics iteration cap for the initial pose
    ik_tolerance           REAL,    -- inverse-kinematics convergence tolerance
    init_position_std      REAL,    -- initial covariance std of the root position, metres
    init_orientation_std   REAL,    -- initial covariance std of the root orientation, radians
    init_joint_std         REAL,    -- initial covariance std of joint angles, radians (metres for prismatic joints)
    init_velocity_std      REAL,    -- initial covariance std of the velocities
    min_cameras_for_init              INTEGER,  -- fewest cameras that must see a frame for it to initialise the tracker
    velocity_mode_camera_ids          TEXT,  -- JSON array of integer camera indices; NULL = all cameras use position mode
    velocity_measurement_noise_std    REAL,  -- Measurement noise std for velocity cameras (pixels/frame); NULL = use measurement_noise_std
    notes                             TEXT,     -- free-text notes; NULL if none
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
    cross_person_max_n                INTEGER, -- Max cross-person anchors per pair per camera per frame
    -- Added in schema migration v38 (session)/v7 (registry): explicit flag
    -- distinguishing a user-saved, browsable named config, set via an
    -- explicit "Save as..." action, from an auto-generated per-run
    -- snapshot. name alone can't reliably signal this -- nothing stops an
    -- auto-generated label from coincidentally matching a real name, or a
    -- user from picking a short, timestamp-shaped name on purpose. See
    -- docs/roadmap/features/configuration-improvements/config-improvements-design.md.
    is_named                          INTEGER NOT NULL DEFAULT 0,
    -- Added in schema migration v50 (session)/v9 (registry): streak-derived dot
    -- velocity -- see
    -- docs/roadmap/features/marker-based-mocap/streak-velocity-design.md §3/§4.
    dot_streak_velocity_enabled       INTEGER,  -- 0/1 flag; NULL/0 = disabled
    dot_streak_k_window               INTEGER,  -- Rolling sample window per camera
    dot_streak_k_min_samples          INTEGER,  -- Min samples before k is trusted
    dot_streak_min_displacement_px    REAL,     -- "Only dots with actual movement" gate
    dot_streak_min_elongation_px      REAL,     -- Matches the streak-noise-inflation gate
    dot_streak_velocity_noise_std     REAL,     -- noise_std_override for the emitted observation
    -- Added in schema migration v51 (session)/v10 (registry): cap on the adaptive
    -- process noise (Mechanism A) variance-domain multiplier, previously a hardcoded
    -- UKF constant -- see UnscentedKalmanFilter::set_velocity_noise_max_multiplier().
    process_noise_vel_max_multiplier REAL,     -- Variance-domain cap; NULL/10.0 = prior hardcoded default
    -- Added in schema migration v52 (session)/v11 (registry): dot-candidate
    -- assignment gate and its tracklet-aware relaxation.
    -- dot_assignment_gate_mahalanobis existed as a TOML-only tunable before
    -- this migration; it had no DB column at all, so a DB-driven tracker_config
    -- row could never actually override it -- fixed here alongside the new
    -- tracklet multiplier since both touch the same assignment gate.
    dot_assignment_gate_mahalanobis  REAL,     -- Chi-square gate on assignment cost; NULL/9.21 = prior default
    dot_tracklet_gate_multiplier     REAL,     -- Divides cost for a same-tracklet pairing; NULL/1.0 = disabled
    -- Added in schema migration v53 (session)/v12 (registry): experimental
    -- per-marker confidence-threshold override (markers such as nose/ear stay
    -- confident even when occluded by the back of the head).
    -- NULL/empty = disabled (every
    -- marker uses the existing global min_confidence path, unchanged).
    confidence_threshold_marker_names TEXT,    -- JSON string array, e.g. ["MRK-nose","MRK-ear.L"]
    confidence_threshold_override     REAL,    -- Confidence floor applied to the markers named above
    -- Added in schema migration v54 (session)/v13 (registry): reacquisition
    -- gate for the shared dot-assignment phase -- see ReacquisitionGateModifier
    -- (dot_assignment.hpp). Requires independent evidence (tracklet continuity,
    -- or a near-prediction candidate in at least two cameras) before a slot
    -- left unresolved for a run of steps may resolve again.
    dot_reacquire_gap_frames INTEGER,  -- Steps unresolved before the gate applies; NULL/0 = disabled
    dot_reacquire_max_px     REAL,     -- Cross-camera "near the prediction" radius, px; NULL/30.0 = default
    -- Added in schema migration v55 (session)/v14 (registry): cross-view
    -- corroboration and per-camera trust for the shared dot-assignment phase --
    -- see CrossViewCorroborationModifier and dot_camera_noise_scale's own doc
    -- comment (dot_assignment.hpp / config.hpp).
    dot_cross_view_corroboration_px  REAL,  -- Epipolar tolerance, px; NULL/0.0 = disabled
    dot_corroboration_gate_multiplier REAL, -- Divides cost for a corroborated pairing; NULL/1.0 = disabled
    dot_camera_noise_scale           TEXT,  -- JSON object {camera_id: scale, ...}; NULL/empty = no change
    -- Added in schema migration v57 (session)/v15 (registry): segmentation-
    -- mask-based trust for the shared dot-assignment phase. Shared tunables
    -- only -- per-subject identity (which seg_run, which label) is supplied
    -- separately per subject at track-invocation time, never persisted here,
    -- since it legitimately differs between subjects sharing one solve --
    -- see SegmentationMaskModifier and TrackerConfig's own dot_seg_mask_*
    -- field doc comments (dot_assignment.hpp / config.hpp).
    dot_seg_mask_enabled              INTEGER, -- 0/1 flag; NULL/0 = disabled
    dot_seg_mask_dilate_px            REAL,    -- Search radius, native px; NULL/20.0 = default
    dot_seg_mask_tracklet_window      INTEGER, -- Rolling per-tracklet window, frames; NULL/60 = default
    dot_seg_mask_min_tracklet_fraction REAL,   -- Hard-exclude threshold; NULL/0.3 = default
    dot_seg_mask_cost_scale_at_zero_fraction REAL, -- Soft-penalty ceiling; NULL/4.0 = default
    -- Added in schema migration v61 (session)/v16 (registry). A candidate
    -- farther than this many native pixels from its subject's mask label in
    -- the current frame is excluded outright, with no tracklet history needed
    -- (a flickering static light gets a new tracklet id every few frames);
    -- skipped for a camera whose mask lacks the label entirely.
    dot_seg_mask_hard_distance_px     REAL,    -- Native px; NULL/0 = disabled
    -- Added in schema migration v67 (session)/v17 (registry): the joint groups the tracker's
    -- main (parent) stage tracks. A hierarchical configuration (one with tracker_config_stages
    -- rows) restricted to its parent group lets the child stages carry the hands; without the
    -- restriction the parent stage tracks the whole skeleton and the child stages run in
    -- addition. An explicit --joint-groups on the tracker command line overrides this column.
    -- See docs/roadmap/features/tracker-performance/tracker-performance-design.md.
    active_joint_groups               TEXT,    -- JSON array of skeleton group names, e.g. ["main"]; NULL/[] = the whole skeleton
    -- Added in schema migration v69 (session)/v18 (registry): keypoint candidates. 1 = the tracker takes the
    -- sequence's keypoint candidates (pose_observations source 'kp_candidates') through the shared assignment
    -- instead of the fixed body observations, for the markers bound to that pose model (the hand stages keep
    -- their refined-hand observations). Gate, tracklet, corroboration and mask behaviour are the dot_* settings
    -- above. See docs/roadmap/features/keypoint-candidates/keypoint-candidates-design.md.
    kp_candidates_enabled             INTEGER, -- 0/1 flag; NULL/0 = the fixed observations are used even if the sequence holds candidates
    -- Weight of the peak term in the assignment cost of a keypoint candidate: cost = squared Mahalanobis distance
    -- + ln det S - weight * ln(peak), S the slot covariance with the candidate's detection noise (the gate tests
    -- the distance alone). NULL/0 = no peak term. Has no meaning for dots.
    kp_candidate_peak_weight          REAL,
    -- Divides the squared Mahalanobis distance of a keypoint candidate that continues the tracklet resolved into
    -- the same camera and marker on the previous step (dot_tracklet_gate_multiplier is the dot's). NULL/1.0 = off.
    kp_candidate_tracklet_multiplier  REAL,
    -- Added in schema migration v70 (session)/v19 (registry): the choice of keypoint candidates over a time
    -- window (see docs/roadmap/features/keypoint-candidates/keypoint-candidates-design.md, section 10). The
    -- choice runs as a phase of the tracking run before the first filter step and feeds its picks to the filter
    -- as the fixed observations of the markers bound to the keypoints.
    -- 'window' = the windowed choice, 'step' = the per-step assignment of the shared dot-assignment phase;
    -- NULL = 'step'. Only read when kp_candidates_enabled is 1.
    kp_candidate_choice               TEXT,
    -- Half width in seconds of the window the neighbour and cross-camera support of a candidate are averaged over
    -- along its tracklet; NULL = 1.0.
    kp_choice_support_window_s        REAL,
    -- Weight of the cross-camera (epipolar) support in the score of a candidate; NULL = 1.0, 0 = off.
    kp_choice_delta                   REAL,
    -- Weight of the multi-camera consensus flag in the score of a candidate; NULL = 0.3.
    kp_choice_beta                    REAL,
    -- Weight of the heatmap peak in the score of a candidate; NULL = 0.2.
    kp_choice_gamma                   REAL,
    -- Number of scoring rounds, each started from the picks of the previous one; NULL = 3.
    kp_choice_rounds                  INTEGER,
    -- JSON array of integer camera indices (as velocity_mode_camera_ids) that take no part in the windowed
    -- choice: their fixed observations are kept and their candidates give no support. NULL/[] = every camera takes part.
    kp_choice_skip_camera_ids         TEXT,
    -- Strict mask filter before the choice: 1 = a candidate is dropped unless enough of the detections of its
    -- tracklet around its frame lie on the tracked person's own mask (needs --seg-run); NULL/0 = off.
    kp_strict_filter                  INTEGER,
    -- Half width of the strict filter's window in video frames, looking backwards and forwards; NULL = 10.
    kp_strict_filter_window_frames    INTEGER,
    -- Smallest share of the detections in that window that must lie on the person's mask; NULL = 0.5.
    kp_strict_filter_min_fraction     REAL,
    -- Weight of the soft mask term in the score of a candidate (+1 on the person's mask, -1 on another person's,
    -- averaged along the tracklet); NULL/0 = off. Needs --seg-run.
    kp_mask_term_weight               REAL,
    -- Half width in seconds of the window the mask term is averaged over; NULL = 2.0.
    kp_mask_term_window_s             REAL,
    -- Comma-separated keypoint classes the mask term applies to ('body', 'hands', 'feet'; the classes of a pose
    -- model are in its choice model file); NULL = 'body'.
    kp_mask_term_classes              TEXT,
    -- Comma-separated keypoint classes the strict mask filter applies to ('body', 'hands', 'feet'); NULL = 'body'.
    kp_strict_filter_classes          TEXT,
    -- Score lost by a change of candidate between consecutive frames in the chain of picks the windowed choice
    -- takes (a change is another tracklet or a jump of the pick, see kp_choice_jump_box); NULL/0 = every frame takes
    -- its best score on its own.
    kp_choice_switch_penalty          REAL,
    -- Box heights between consecutive frames above which a move of the pick counts as a change; NULL = 0.04.
    kp_choice_jump_box                REAL
);

-- Added in schema migration v37: hierarchical body/hand solver -- per-stage
-- tuning, keyed by group name, NULL inherits from the parent tracker_configs
-- row above (mirrors tracker_configs' own parent_id inheritance). NOT
-- skeleton metadata: skeletons are shared, referenced-by-id registry
-- entities describing topology; tuning is iterated per run the same way
-- every other tracker_configs column already is (same config-side-scoping-
-- over-skeleton-groups precedent as process_noise_vel_joint_names above). A
-- tracker_config_id with any rows here is what selects hierarchical mode;
-- one without runs monolithic, unchanged. See
-- docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md.
CREATE TABLE IF NOT EXISTS tracker_config_stages (
    tracker_config_id     TEXT NOT NULL REFERENCES tracker_configs(id),
    group_name            TEXT NOT NULL,
    process_noise_std     REAL,
    process_noise_vel_std REAL,
    velocity_half_life_s  REAL,
    pose_noise_std        REAL,
    calib_noise_std       REAL,
    outlier_threshold     REAL,
    min_inliers_ratio     REAL,
    max_innovation_norm   REAL,
    init_joint_std        REAL,
    init_velocity_std     REAL,
    PRIMARY KEY (tracker_config_id, group_name)
);

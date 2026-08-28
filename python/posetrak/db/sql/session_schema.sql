-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

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

-- Solved marker-body poses for this session: the portable calibration
-- rig's own anchor pose, and/or ordinary scattered scene tags. One row per
-- solved body INSTANCE, not per marker -- a rigid multi-marker body only
-- ever needs one pose, since every marker's world position follows from
-- definition + this pose (see anchor_from_marker_rig in
-- app/setup/fiducial_markers.py). marker_body_definition_id is NULL for a
-- lone scattered tag -- its local geometry is always just a single
-- marker's own square, nothing worth a bespoke YAML definition for -- and
-- the marker_type/dictionary/marker_id/marker_size columns cover that case
-- inline instead. See docs/roadmap/features/extrinsics-improvements/
-- extrinsics-improvements-design.md, section 10 (supersedes that design
-- doc's earlier scene_fiducial_markers sketch, which was never
-- implemented).
-- marker_body_definition_id -- references marker_body_definitions(id),
--   embedded in this session DB (see the SELF-CONTAINMENT REQUIREMENT
--   header above) -- not a SQL REFERENCES constraint, matching this file's
--   existing convention for columns pointing at a registry-origin table
--   (e.g. extrinsic_entries.camera_instance_id above).
-- R and t are little-endian float64 blobs (R: 9 elements row-major, t: 3
--   elements), body-local -> world, same convention as extrinsic_entries.
CREATE TABLE IF NOT EXISTS scene_marker_bodies (
    id                               TEXT PRIMARY KEY,
    session_id                       TEXT NOT NULL REFERENCES mocap_sessions(id),
    label                            TEXT NOT NULL,
    -- User-chosen name grouping every marker anchored together in one
    -- physical space (e.g. "room7") -- '' (not NULL -- SQLite's unique
    -- index treats every NULL as distinct, which would silently defeat
    -- this column's role in the uniqueness constraint below) for
    -- legacy/ungrouped rows. Lets a later capture in the same room pick
    -- "room7" from a list instead of the session's markers from every
    -- room loading together indiscriminately. Part of (session_id,
    -- group_name, label)'s uniqueness so two rooms can reuse the same
    -- tag id without colliding.
    group_name                       TEXT NOT NULL DEFAULT '',
    marker_body_definition_id        TEXT,
    marker_type                      TEXT,
    dictionary                       TEXT,
    marker_id                        TEXT,
    marker_size                      REAL,
    R                                BLOB NOT NULL,
    t                                BLOB NOT NULL,
    is_primary_anchor                INTEGER NOT NULL DEFAULT 0,
    source_extrinsic_calibration_id  TEXT REFERENCES extrinsic_calibrations(id),
    updated_at                       TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS scene_marker_bodies_unique
    ON scene_marker_bodies (session_id, group_name, label);

-- A capture is a single continuous camera recording within a session.
-- extrinsic_calibration_id is nullable: captures may be created before extrinsics are imported.
-- shot_id FK columns in other tables reference captures(id) (historical naming).
-- default_tracker_config_id -- references tracker_configs(id) (registry table,
--   also embedded in this session DB -- see the SELF-CONTAINMENT REQUIREMENT
--   header). Added in schema migration v38. Resolved trial -> capture ->
--   a checked-in baseline config when starting a new tracking run; NULL
--   falls through to the next level. Never mutated in place once set --
--   "editing" the default always creates a new tracker_configs row via
--   manage_config.edit_config() and repoints this column to it. See
--   docs/roadmap/features/configuration-improvements/config-improvements-design.md.
CREATE TABLE IF NOT EXISTS captures (
    id                       TEXT PRIMARY KEY,
    session_id               TEXT NOT NULL REFERENCES mocap_sessions(id),
    extrinsic_calibration_id TEXT REFERENCES extrinsic_calibrations(id),
    capture_number           INTEGER NOT NULL,
    label                    TEXT,
    notes                    TEXT,
    default_tracker_config_id TEXT
);

-- A trial is a named, bounded time window within a capture: one technique, one attempt.
-- The user-facing unit of analysis (e.g. "shomenuchi shihonage take 1").
-- default_tracker_config_id -- see captures.default_tracker_config_id above;
--   resolved before the capture-level default when starting a new run.
CREATE TABLE IF NOT EXISTS trials (
    id           TEXT PRIMARY KEY,
    capture_id   TEXT NOT NULL REFERENCES captures(id),
    name         TEXT,
    time_start_s REAL,
    time_end_s   REAL,
    notes        TEXT,
    default_tracker_config_id TEXT
);

-- Named performers, defined once per capture rather than per detection run --
-- trials within one capture are near-certain to share the same physical
-- performers (see
-- docs/roadmap/features/configuration-improvements/config-improvements-design.md,
-- "Person model: promote identity to capture level"). A trial-level override
-- of an existing capture person's skeleton belongs to tracking_run_persons,
-- not here.
-- default_skeleton_id -- references registry: skeletons(id); nullable until assigned.
CREATE TABLE IF NOT EXISTS capture_persons (
    id                  TEXT PRIMARY KEY,
    capture_id          TEXT NOT NULL REFERENCES captures(id),
    name                TEXT NOT NULL,
    default_skeleton_id TEXT,
    notes               TEXT,
    created_at          TEXT NOT NULL
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
-- capture_person_id -- links to this capture's named performer (nullable:
--   NULL for rows written before the person model existed, or where the
--   free-text name wasn't matched to a capture_persons row); person_name
--   stays the display/CSV-export field regardless, mirroring the linked
--   row's name when set.
CREATE TABLE IF NOT EXISTS sequence_persons (
    sequence_id       TEXT    NOT NULL REFERENCES pose_observation_sequences(id),
    person_id         INTEGER NOT NULL,
    person_name       TEXT    NOT NULL,
    capture_person_id TEXT    REFERENCES capture_persons(id),
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
-- state    : little-endian float64 blob, STORAGE-indexed:
--              [root_pos(3), root_axis_angle(3), joint_angles(K_storage),
--               root_vel(3), root_angvel(3), joint_velocities(K_storage)]
--            where K_storage = SkeletonLayout::total_storage_dof_count() --
--            every SPHERICAL joint always occupies 3 slots here, even if one
--            of its axes is locked by equal limits.
-- cov_diag : little-endian float64 blob, ERROR-STATE-indexed (the diagonal
--            of the UKF's own covariance matrix):
--              [root_pos(3), root_ori(3), joint_pos(K_active),
--               root_vel(3), root_angvel(3), joint_vel(K_active)]
--            where K_active = SkeletonLayout::joint_active_dof_count() -- a
--            locked SPHERICAL axis contributes no slot here. K_active <
--            K_storage whenever any joint in the skeleton has a locked axis,
--            so state and cov_diag are NOT the same length in general and
--            must never be indexed with the same offset arithmetic; see
--            SkeletonLayout::build_index_map_from() (state, storage-indexed)
--            vs. build_error_index_map_from() (cov_diag, error-indexed).
--
-- Hierarchical solver runs (see tracking_run_stages below and
-- docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md):
-- each row still spans the person's FULL, unmodified skeleton -- there is no
-- separate run/row per stage -- even though every stage's own Tracker only
-- estimates a subset of joints (e.g. "main", "HandL", "HandR"). Each stage
-- read-modify-writes only the index range it owns:
--   * The first stage to write a row (the one holding the skeleton's true
--     floating root) expands both state and cov_diag to full width before
--     writing. State's not-yet-solved range is filled with rest-pose
--     defaults; cov_diag's is filled with a placeholder variance derived
--     from the run's own tracker_configs.init_joint_std/init_velocity_std
--     (NOT a real per-DOF uncertainty).
--   * Every later stage (e.g. a hand child) patches its own owned range in
--     both blobs with its real solved values -- state via
--     build_index_map_from(), cov_diag via build_error_index_map_from().
-- Whether a given DOF's cov_diag entry is a real value or still the
-- placeholder is NOT recoverable by sniffing the blob -- check
-- tracking_run_stages.status for that DOF's owning group instead.
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

-- Hierarchical body/hand solver: one row per (run, person, skeleton group)
-- the solver treats as its own filter pass (e.g. "main", "HandL", "HandR").
-- Every stage read-modify-writes the same tracking_results/
-- tracking_obs_results rows for its owned DOF/marker range rather than
-- getting its own run_id -- this table gives an atomic per-stage
-- completion boundary, the staleness flag a parent re-run needs to
-- invalidate every child stage that consumed its smoothed trajectory, and
-- a progress surface for the UI. See
-- docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md.
CREATE TABLE IF NOT EXISTS tracking_run_stages (
    run_id       TEXT    NOT NULL REFERENCES tracking_runs(id),
    person_id    INTEGER NOT NULL,
    group_name   TEXT    NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending', 'running', 'complete', 'stale')),
    started_at   TEXT,
    completed_at TEXT,
    PRIMARY KEY (run_id, person_id, group_name)
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

-- Segmentation quality run: parameters and provenance for one segmentation
-- (Cutie interactive init, or the offline add_seg_quality tool).
-- Time-range-scoped on the capture's own timeline, not tied to any one
-- detection run -- a segmentation is reusable by any detection run/trial
-- whose range it covers (containment: shot_id matches, and
-- time_start_s <= trial.time_start_s AND time_end_s >= trial.time_end_s).
-- See docs/roadmap/features/segmentation-reuse/segmentation-reuse-design.md.
-- trial_id is optional provenance (which trial this was created from), not
-- a scoping constraint. mask_dir: optional path to a directory containing
-- per-video NPZ debug mask files. persons_json: JSON array of person
-- names, index i = mask label i+1 (same convention tracking_runs.
-- marker_names/active_camera_ids already use for an ordered string list
-- in one column) -- the ordinal->name mapping baked into this
-- segmentation's own mask labels at creation time, so a *different*
-- caller reusing this segmentation later (gap 2, RunDetectionDialog)
-- doesn't have to assume today's capture_persons order still matches
-- whatever order was in effect when the masks were made. NULL for
-- segmentations created before this column existed, or via the offline
-- add_seg_quality.py tool (no interactive person labeling there).
CREATE TABLE IF NOT EXISTS seg_quality_runs (
    id             TEXT PRIMARY KEY,
    shot_id        TEXT NOT NULL REFERENCES captures(id),
    trial_id       TEXT REFERENCES trials(id),
    time_start_s   REAL NOT NULL,
    time_end_s     REAL NOT NULL,
    created_at     TEXT NOT NULL,
    quality_source TEXT NOT NULL DEFAULT 'cutie',
    erosion_px     INTEGER NOT NULL DEFAULT 5,
    mask_dir       TEXT,
    notes          TEXT,
    persons_json   TEXT,
    name           TEXT
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
-- capture_person_id -- see sequence_persons.capture_person_id above; same
--   nullable, additive-not-replacing relationship to person_name.
CREATE TABLE IF NOT EXISTS detection_track_assignments (
    detection_run_id  TEXT    NOT NULL REFERENCES detection_runs(id),
    shot_video_id     TEXT    NOT NULL,
    track_id          INTEGER NOT NULL,
    person_name       TEXT    NOT NULL,
    capture_person_id TEXT    REFERENCES capture_persons(id),
    first_frame       INTEGER NOT NULL,
    last_frame        INTEGER NOT NULL,
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

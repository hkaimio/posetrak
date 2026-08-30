-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- Migration: session schema v8 → v9
-- Adds detection_runs and detection_keypoints tables for the integrated
-- pose extraction pipeline, and extends person_detections / person_tracks /
-- frame_cache_entries / pose_observation_sequences / pose_observations
-- with detection_run_id and noise_scale columns.

BEGIN;

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

-- Recreate person_detections with detection_run_id in the primary key
-- (safe: table was added in v8 and has never been populated by any workflow)
DROP TABLE IF EXISTS person_detections;
CREATE TABLE person_detections (
    detection_run_id    TEXT NOT NULL REFERENCES detection_runs(id),
    shot_video_id       TEXT NOT NULL REFERENCES shot_videos(id),
    video_frame         INTEGER NOT NULL,
    track_id            INTEGER NOT NULL,
    region_type         TEXT NOT NULL DEFAULT 'full_body',
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

-- Recreate person_tracks with detection_run_id
-- (safe: never populated by prior workflow)
DROP TABLE IF EXISTS person_tracks;
CREATE TABLE person_tracks (
    id                  TEXT PRIMARY KEY,
    detection_run_id    TEXT NOT NULL REFERENCES detection_runs(id),
    shot_video_id       TEXT NOT NULL REFERENCES shot_videos(id),
    track_id            INTEGER NOT NULL,
    first_frame         INTEGER NOT NULL,
    last_frame          INTEGER NOT NULL,
    UNIQUE (detection_run_id, shot_video_id, track_id)
);

ALTER TABLE frame_cache_entries
    ADD COLUMN detection_run_id TEXT REFERENCES detection_runs(id);

ALTER TABLE pose_observation_sequences
    ADD COLUMN detection_run_id TEXT REFERENCES detection_runs(id);

ALTER TABLE pose_observations
    ADD COLUMN noise_scale REAL;

PRAGMA user_version = 9;
COMMIT;

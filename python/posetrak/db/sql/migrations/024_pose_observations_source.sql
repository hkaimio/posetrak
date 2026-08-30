-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- Migration v34 → v35: add `source` to the pose_observations primary key.
--
-- Lets multiple detection sources (whole-body pass, refined hand passes)
-- contribute rows for the same (sequence, camera, frame, person) instead of
-- being collapsed into one shared kp_blob/noise_scale — each source keeps
-- its own noise_scale, matching detection_keypoints.region_type's existing
-- 'full_body' | 'face' | 'hand_l' | 'hand_r' spelling.
--
-- Also adds detection_run_id so a pose_observations row can be traced back
-- to the detection run that produced it without joining through
-- pose_observation_sequences (which only tracks one "primary" run per
-- sequence and stays unchanged).
--
-- SQLite cannot ALTER a primary key in place, so the table is rebuilt.
-- Existing rows are preserved with source='body' and detection_run_id
-- backfilled from their parent sequence.

BEGIN;

CREATE TABLE pose_observations_new (
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

INSERT INTO pose_observations_new
    (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id,
     source, detection_run_id, kp_blob, noise_scale)
SELECT po.sequence_id, po.camera_instance_id, po.video_frame, po.timestamp_s, po.person_id,
       'body', seq.detection_run_id, po.kp_blob, po.noise_scale
FROM pose_observations po
JOIN pose_observation_sequences seq ON seq.id = po.sequence_id;

DROP TABLE pose_observations;
ALTER TABLE pose_observations_new RENAME TO pose_observations;

PRAGMA user_version = 35;
COMMIT;

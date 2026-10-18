-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- Session schema v64 -> v65: the keypoint manifest is kept per observation source.
-- Existing rows describe the body blob ('body'). Column documentation lives in session_schema.sql.
BEGIN;

CREATE TABLE pose_sequence_keypoints_new (
    sequence_id   TEXT NOT NULL REFERENCES pose_observation_sequences(id),
    obs_source    TEXT NOT NULL DEFAULT 'body',
    keypoint_idx  INTEGER NOT NULL,
    name          TEXT NOT NULL,
    source        TEXT NOT NULL,
    PRIMARY KEY (sequence_id, obs_source, keypoint_idx)
);
INSERT INTO pose_sequence_keypoints_new (sequence_id, obs_source, keypoint_idx, name, source)
    SELECT sequence_id, 'body', keypoint_idx, name, source FROM pose_sequence_keypoints;
DROP TABLE pose_sequence_keypoints;
ALTER TABLE pose_sequence_keypoints_new RENAME TO pose_sequence_keypoints;

PRAGMA user_version = 65;
COMMIT;

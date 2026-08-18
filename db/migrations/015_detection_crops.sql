-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

BEGIN;

CREATE TABLE IF NOT EXISTS detection_crops (
    detection_run_id TEXT NOT NULL,
    shot_video_id    TEXT NOT NULL,
    video_frame      INTEGER NOT NULL,
    track_id         INTEGER NOT NULL,
    jpeg_data        BLOB NOT NULL,
    PRIMARY KEY (detection_run_id, shot_video_id, video_frame, track_id)
);

CREATE INDEX IF NOT EXISTS idx_detection_crops_lookup
    ON detection_crops (detection_run_id, shot_video_id, track_id, video_frame);

PRAGMA user_version = 16;
COMMIT;

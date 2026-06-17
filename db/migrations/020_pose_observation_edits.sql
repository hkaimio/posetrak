-- 020_pose_observation_edits.sql
-- Non-destructive keypoint edit overlay for post-stitch pose observations.
--
-- Each row stores an edited keypoint blob (same float32[N,3] format as
-- pose_observations.kp_blob) together with a uint8 bitmask that marks which
-- keypoint slots are overridden.  Slots not in the mask keep their original
-- pose_observations values.  This is keyed identically to pose_observations
-- (sequence_id, camera_instance_id, video_frame) so the C++ tracker can look
-- edits up with a simple point query.

CREATE TABLE pose_observation_edits (
    id                 TEXT PRIMARY KEY,
    sequence_id        TEXT NOT NULL REFERENCES pose_observation_sequences(id),
    camera_instance_id TEXT NOT NULL,
    video_frame        INTEGER NOT NULL,
    -- float32[N,3]: x, y, is_outlier for every keypoint slot (N matches pose_observations).
    -- Only slots with the corresponding bit set in kp_mask are applied.
    kp_blob            BLOB NOT NULL,
    -- uint8[ceil(N/8)]: bitmask; bit i set means slot i is overridden.
    kp_mask            BLOB NOT NULL,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE UNIQUE INDEX pose_observation_edits_unique
    ON pose_observation_edits (sequence_id, camera_instance_id, video_frame);

PRAGMA user_version = 21;

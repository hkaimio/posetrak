-- Migration v14 → v15: add detection_track_assignments table.
--
-- Records the explicit (track_id → person_name) mapping made in the pose
-- extraction UI so that assignments can be restored exactly when a detection
-- run is reopened, without trying to reverse-engineer the mapping from
-- pose_observations (which lacks a track_id column and produces cross-product
-- ambiguity for multi-person frames).
CREATE TABLE IF NOT EXISTS detection_track_assignments (
    detection_run_id TEXT    NOT NULL REFERENCES detection_runs(id),
    shot_video_id    TEXT    NOT NULL,
    track_id         INTEGER NOT NULL,
    person_name      TEXT    NOT NULL,
    first_frame      INTEGER NOT NULL,
    last_frame       INTEGER NOT NULL,
    PRIMARY KEY (detection_run_id, shot_video_id, track_id, first_frame)
);

PRAGMA user_version = 15;

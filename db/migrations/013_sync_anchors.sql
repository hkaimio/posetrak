-- Migration 013: sync anchor observation tables.
--
-- Adds an input layer for the graph-based sync solver.  The existing
-- sync_configs / sync_points tables are the solver OUTPUT and are unchanged.
--
-- sync_anchors: one row per shared real-world event (e.g. "LED flash",
--   "clap"), visible simultaneously in two or more camera videos.
--
-- sync_anchor_observations: one row per video that captured the event,
--   recording the integer frame number and an optional sub-frame offset
--   from LED peak detection (0.0 for manually-marked frames).

CREATE TABLE IF NOT EXISTS sync_anchors (
    id         TEXT PRIMARY KEY,
    shot_id    TEXT NOT NULL REFERENCES captures(id),
    notes      TEXT
);

CREATE TABLE IF NOT EXISTS sync_anchor_observations (
    id             TEXT    PRIMARY KEY,
    sync_anchor_id TEXT    NOT NULL REFERENCES sync_anchors(id),
    shot_video_id  TEXT    NOT NULL REFERENCES capture_videos(id),
    video_frame    INTEGER NOT NULL,
    subframe       REAL    NOT NULL DEFAULT 0.0,
    UNIQUE (sync_anchor_id, shot_video_id)
);

PRAGMA user_version = 14;

-- Migration: session schema v12 → v13
-- Terminology update: "shot" → "capture" (a raw camera recording), plus a new
-- "trials" table that represents a named, bounded time window within a capture
-- (one technique, one attempt — the user-facing unit of analysis).
--
-- Table renames: shots → captures, shot_videos → capture_videos.
-- shot_number column renamed to capture_number within captures.
-- FK columns in referencing tables (shot_id, shot_video_id) are NOT renamed;
-- they remain valid UUID references and renaming them would require table recreation.
--
-- New columns:
--   detection_runs.trial_id            → links a detection run to a trial
--   pose_observation_sequences.name    → user-assigned name for a person track
--   tracking_runs.notes                → free-text notes on a tracking run

ALTER TABLE shots RENAME TO captures;
ALTER TABLE captures RENAME COLUMN shot_number TO capture_number;
ALTER TABLE shot_videos RENAME TO capture_videos;

CREATE TABLE IF NOT EXISTS trials (
    id           TEXT PRIMARY KEY,
    capture_id   TEXT NOT NULL REFERENCES captures(id),
    name         TEXT,
    time_start_s REAL,
    time_end_s   REAL,
    notes        TEXT
);

ALTER TABLE detection_runs ADD COLUMN trial_id TEXT REFERENCES trials(id);
ALTER TABLE pose_observation_sequences ADD COLUMN name TEXT;
ALTER TABLE tracking_runs ADD COLUMN notes TEXT;

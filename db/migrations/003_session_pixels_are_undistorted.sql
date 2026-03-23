-- Migration: session schema v3 → v4
-- Adds pixels_are_undistorted to pose_observation_sequences.
-- When 1 (default), stored keypoint coordinates are already in undistorted
-- pixel space (K_new) and must NOT be undistorted again by the tracker.
-- When 0, coordinates are in distorted pixel space (K_original) and the
-- tracker must apply undistortion before use.
-- Existing rows default to 1 because all prior captures used undistorted video.

BEGIN;

ALTER TABLE pose_observation_sequences
    ADD COLUMN pixels_are_undistorted INTEGER NOT NULL DEFAULT 1;

PRAGMA user_version = 4;

COMMIT;

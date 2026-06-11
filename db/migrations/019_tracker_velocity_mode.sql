-- Migration 019: velocity-mode camera support in tracker_configs.
--
-- Adds two nullable columns to tracker_configs (present in both registry and session DBs):
--   velocity_mode_camera_ids         TEXT   JSON array of camera instance ID strings.
--                                           NULL means all cameras use absolute-position measurements.
--   velocity_measurement_noise_std   REAL   Measurement noise std for velocity-mode cameras
--                                           (pixels/frame).  NULL = fall back to measurement_noise_std.
--
-- Both columns are backward-compatible: NULL in existing rows preserves the old behaviour.

BEGIN;
ALTER TABLE tracker_configs ADD COLUMN velocity_mode_camera_ids TEXT;
ALTER TABLE tracker_configs ADD COLUMN velocity_measurement_noise_std REAL;
PRAGMA user_version = 20;
COMMIT;

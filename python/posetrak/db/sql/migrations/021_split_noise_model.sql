-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- Migration 021: split measurement noise into pose and calibration components.
--
-- Adds pose_noise_std to tracker_configs (both registry and session DBs):
--   pose_noise_std   REAL   Pose estimation error in model-input pixels, scaled by
--                           noise_scale (bbox_width / pose_input_width) before being
--                           added to calib_noise_std.  NULL / 0.0 = use calibration-
--                           only formula (backward-compatible with old configs).
--
-- The existing measurement_noise_std column is kept as calib_noise_std equivalent
-- for backward compatibility with pre-existing tracker_configs rows.

BEGIN;
ALTER TABLE tracker_configs ADD COLUMN pose_noise_std REAL;
PRAGMA user_version = 22;
COMMIT;

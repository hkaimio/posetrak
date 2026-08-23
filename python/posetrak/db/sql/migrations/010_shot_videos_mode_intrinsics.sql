-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- Migration: session schema v10 → v11
-- Moves camera_mode_id and intrinsics_calibration_id from session_cameras to shot_videos.
-- Each shot_video now declares its own capture mode and intrinsics calibration, allowing
-- the same physical camera to be used in different modes across shots in one session.
--
-- Pre-migration shot_videos rows receive NULL for both new columns (no calibration linked).
-- Pre-migration session_cameras rows lose the two columns; they had rarely been populated
-- by the wizard anyway (the wizard did not link cameras to registry records before Phase 2).

ALTER TABLE shot_videos ADD COLUMN camera_mode_id TEXT;
ALTER TABLE shot_videos ADD COLUMN intrinsics_calibration_id TEXT;

ALTER TABLE session_cameras DROP COLUMN camera_mode_id;
ALTER TABLE session_cameras DROP COLUMN intrinsics_calibration_id;

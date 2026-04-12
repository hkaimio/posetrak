-- Migration: registry schema v4 → v5, session schema v9 → v10
-- Adds default_intrinsics_calibration_id to camera_modes so the wizard can
-- auto-select the preferred calibration when a mode is picked for a shot video.
--
-- This migration is identical for both registry and session databases because
-- session databases embed a full copy of the registry schema (camera_modes exists
-- in both). The PRAGMA user_version line is set by the calling migration function
-- in db.py, not in this file, so the same SQL can be reused for both DB types.

ALTER TABLE camera_modes
    ADD COLUMN default_intrinsics_calibration_id TEXT
    REFERENCES intrinsics_calibrations(id);

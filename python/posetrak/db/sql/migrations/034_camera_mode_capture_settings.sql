-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- 034_camera_mode_capture_settings.sql
-- A capture mode gets a name the operator picks it by and the settings the
-- camera's driver applies to record in it (docs/roadmap/features/camera-control).
-- The columns camera_modes.name and camera_modes.capture_settings (TEXT) and
-- capture_events.camera_mode_id and capture_events.settings_json (TEXT) are added by
-- the migration functions in db.py, which skip columns that already exist; this
-- file holds the rest.

-- A model's modes are told apart by name; unnamed (older) modes are not constrained.
CREATE UNIQUE INDEX IF NOT EXISTS idx_camera_modes_model_name
    ON camera_modes (camera_model_id, name) WHERE name IS NOT NULL;

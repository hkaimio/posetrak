-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- Migration 022: add relative observation mode columns to tracker_configs.
--
-- use_relative_observations  INTEGER  0/1 flag; when 1 the tracker emits an
--                                     additional RELATIVE (child-minus-parent)
--                                     observation for each marker pair where
--                                     both are visible in the same frame/camera
--                                     with confidence >= relative_min_confidence.
--                                     NULL treated as 0 (disabled, backward-compat).
--
-- relative_min_confidence    REAL     Minimum keypoint confidence for both the
--                                     child and parent to form a RELATIVE pair.
--                                     NULL treated as 0.5.

BEGIN;
ALTER TABLE tracker_configs ADD COLUMN use_relative_observations INTEGER;
ALTER TABLE tracker_configs ADD COLUMN relative_min_confidence REAL;
PRAGMA user_version = 23;
COMMIT;

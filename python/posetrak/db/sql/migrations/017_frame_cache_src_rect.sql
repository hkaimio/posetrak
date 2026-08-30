-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- Migration v17 → v18
-- Add source-rectangle columns to frame_cache_entries.
--
-- src_x, src_y: top-left corner of the crop in the original (full-resolution)
--   video frame, in pixels.
-- src_w, src_h: width and height of the crop region in the original frame
--   (before any JPEG resize).  The stored JPEG may be smaller if the crop
--   exceeded _CROP_TARGET_HEIGHT.
--
-- These four values are set for PERSON_CROP entries so that consumers can
-- correctly transform full-frame coordinates (e.g. skeleton keypoints) into
-- the JPEG coordinate space without having to re-derive the crop region from
-- the detection bounding box.  They are NULL for FULL_FRAME and THUMB entries.

PRAGMA user_version = 18;

ALTER TABLE frame_cache_entries ADD COLUMN src_x INTEGER;
ALTER TABLE frame_cache_entries ADD COLUMN src_y INTEGER;
ALTER TABLE frame_cache_entries ADD COLUMN src_w INTEGER;
ALTER TABLE frame_cache_entries ADD COLUMN src_h INTEGER;

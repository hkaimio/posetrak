-- Migration: registry schema v1 → v2
-- Adds image dimensions and undistortion map columns to intrinsics_calibrations.
-- All new columns are nullable so existing rows remain valid.

ALTER TABLE intrinsics_calibrations ADD COLUMN image_width     INTEGER;
ALTER TABLE intrinsics_calibrations ADD COLUMN image_height    INTEGER;
ALTER TABLE intrinsics_calibrations ADD COLUMN matrix_original BLOB;
ALTER TABLE intrinsics_calibrations ADD COLUMN undistort_mapx  BLOB;
ALTER TABLE intrinsics_calibrations ADD COLUMN undistort_mapy  BLOB;

PRAGMA user_version = 2;

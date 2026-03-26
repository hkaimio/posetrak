-- Migration: tracker_configs schema v7 (session) / v4 (registry)
-- Adds velocity_half_life_s to tracker_configs for exponential velocity damping.
-- NULL = no damping (backward compatible with all existing configurations).

BEGIN;
ALTER TABLE tracker_configs ADD COLUMN velocity_half_life_s REAL;
PRAGMA user_version = 7;
COMMIT;

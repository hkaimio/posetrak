-- Migration 023: add spatial cross-pair relative observation columns to tracker_configs.
--
-- cross_pair_max_px  REAL     Pixel-distance threshold for spatial RELATIVE pairs.
--                             When two visible markers in the same frame/camera are
--                             within this distance (pixels) AND their skeleton-tree
--                             distance is > 2 joint hops, a RELATIVE observation is
--                             emitted. 0 / NULL disables cross-pair generation.
--
-- cross_pair_max_n   INTEGER  Maximum cross-pairs per frame per camera (sorted by
--                             proximity; closest pairs kept). NULL treated as 10.

BEGIN;
ALTER TABLE tracker_configs ADD COLUMN cross_pair_max_px REAL;
ALTER TABLE tracker_configs ADD COLUMN cross_pair_max_n INTEGER;
PRAGMA user_version = 24;
COMMIT;

-- Migration 025: add cross-person relative observation columns to tracker_configs.
--
-- cross_person_max_world_mm   REAL     3D world-space marker-pair distance gate (mm)
--                                      for cross-person PAIR_DIFF anchoring (e.g. ukemi
--                                      throws, handshakes). NULL / 0 disables the feature.
--
-- cross_person_min_confidence REAL     Minimum keypoint confidence required of both
--                                      people's detections to form a cross-person anchor.
--                                      NULL treated as 0.5.
--
-- cross_person_max_n          INTEGER  Maximum cross-person anchor observations per
--                                      person pair per camera per frame, closest-first
--                                      (mirrors cross_pair_max_n). NULL treated as 10.

BEGIN;
ALTER TABLE tracker_configs ADD COLUMN cross_person_max_world_mm REAL;
ALTER TABLE tracker_configs ADD COLUMN cross_person_min_confidence REAL;
ALTER TABLE tracker_configs ADD COLUMN cross_person_max_n INTEGER;
PRAGMA user_version = 36;
COMMIT;

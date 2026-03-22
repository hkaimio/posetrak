-- Migration: session schema v2 → v3
-- Makes shots.extrinsic_calibration_id nullable so shots can be created before
-- extrinsics are imported (e.g. during YAML project import).
-- SQLite does not support DROP CONSTRAINT, so we recreate the table.

BEGIN;

CREATE TABLE shots_v3 (
    id                       TEXT PRIMARY KEY,
    session_id               TEXT NOT NULL REFERENCES mocap_sessions(id),
    extrinsic_calibration_id TEXT REFERENCES extrinsic_calibrations(id),
    shot_number              INTEGER NOT NULL,
    label                    TEXT,
    notes                    TEXT
);

INSERT INTO shots_v3 SELECT * FROM shots;
DROP TABLE shots;
ALTER TABLE shots_v3 RENAME TO shots;

PRAGMA user_version = 3;

COMMIT;

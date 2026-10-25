-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- Session schema v61 -> v62: automatic dot-to-slot proposals and human edits of them.
-- Column documentation lives in session_schema.sql.
BEGIN;

CREATE TABLE IF NOT EXISTS dot_assignment_proposals (
    id                 TEXT    PRIMARY KEY,
    detection_run_id   TEXT    NOT NULL REFERENCES detection_runs(id),
    subject_id         TEXT,
    camera_instance_id TEXT    NOT NULL,
    tracklet_id        INTEGER NOT NULL,
    first_frame        INTEGER NOT NULL,
    last_frame         INTEGER NOT NULL,
    slot_name          TEXT,
    cost_px            REAL    NOT NULL,
    margin_px          REAL    NOT NULL,
    support            TEXT    NOT NULL,
    confidence         TEXT    NOT NULL,
    pose_run_id        TEXT    REFERENCES tracking_runs(id),
    proposal_set       TEXT    NOT NULL,
    created_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_dot_assignment_proposals_lookup
    ON dot_assignment_proposals (detection_run_id, camera_instance_id, tracklet_id);

CREATE TABLE IF NOT EXISTS dot_assignment_edits (
    id                 TEXT    PRIMARY KEY,
    trial_id           TEXT    NOT NULL REFERENCES trials(id),
    detection_run_id   TEXT    NOT NULL REFERENCES detection_runs(id),
    subject_id         TEXT,
    camera_instance_id TEXT    NOT NULL,
    tracklet_id        INTEGER NOT NULL,
    first_frame        INTEGER,
    last_frame         INTEGER,
    kind               TEXT    NOT NULL CHECK (kind IN ('assign', 'forbid', 'reject', 'other')),
    slot_name          TEXT,
    created_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_dot_assignment_edits_lookup
    ON dot_assignment_edits (trial_id, detection_run_id, camera_instance_id, tracklet_id);

PRAGMA user_version = 62;
COMMIT;

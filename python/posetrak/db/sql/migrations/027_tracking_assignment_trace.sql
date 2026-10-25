-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- 027_tracking_assignment_trace.sql
-- Assignment-decision trace: every (candidate, marker slot) pair the shared
-- dot-assignment phase (dot_assignment.cpp, resolve_dot_assignment()) costed
-- for a frame, not just the winner tracking_obs_results records -- the "why
-- was detection N (not) assigned to slot X in frame F" instrumentation
-- (sword-swing-tracking plan, 2026-09-27). Opt-in and scoped to a requested
-- step range at tracking time; a run made without tracing requested writes
-- no rows here at all. camera_idx/marker_idx index into the owning
-- tracking_runs row's own active_camera_ids/marker_names JSON arrays, the
-- same convention tracking_obs_results.obs_blob's two axes already use.

BEGIN;

CREATE TABLE IF NOT EXISTS tracking_assignment_trace (
    run_id              TEXT    NOT NULL REFERENCES tracking_runs(id),
    person_id           INTEGER NOT NULL,
    tracker_step        INTEGER NOT NULL,
    camera_idx          INTEGER NOT NULL,
    candidate_row       INTEGER NOT NULL,
    tracklet_id         INTEGER NOT NULL,
    candidate_x         REAL    NOT NULL,
    candidate_y         REAL    NOT NULL,
    marker_idx          INTEGER NOT NULL,
    mahal_cost          REAL    NOT NULL,
    final_cost          REAL    NOT NULL,
    excluded_by         TEXT    NOT NULL DEFAULT '',
    in_final_assignment INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tracking_assignment_trace_lookup
    ON tracking_assignment_trace (run_id, person_id, tracker_step);

PRAGMA user_version = 56;
COMMIT;

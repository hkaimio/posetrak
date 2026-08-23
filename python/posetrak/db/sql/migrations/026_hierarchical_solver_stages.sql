-- SPDX-FileCopyrightText: 2026 Harri Kaimio
--
-- SPDX-License-Identifier: Apache-2.0

-- 026_hierarchical_solver_stages.sql
-- Hierarchical body/hand solver: per-stage run bookkeeping and per-stage
-- tracker tuning. See
-- docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md.
--
-- tracking_run_stages: one row per (run, person, skeleton group) the solver
-- treats as its own filter pass (e.g. "main", "HandL", "HandR"). Every
-- solver stage -- the parent included -- read-modify-writes the same
-- tracking_results/tracking_obs_results rows for its owned DOF/marker
-- range rather than getting its own run_id, so this table is what makes
-- that safe: it gives an atomic per-stage completion boundary (a crash
-- mid-child leaves rows silently half-patched otherwise, indistinguishable
-- from complete output), the staleness flag a parent re-run needs to
-- invalidate every child stage that consumed its smoothed trajectory, and
-- a progress surface for the UI.
--
-- tracker_config_stages: per-stage tuning, keyed by group name, NULL
-- inherits from the parent tracker_configs row -- mirrors the
-- parent_id inheritance tracker_configs itself already has. Deliberately
-- NOT skeleton metadata: skeletons are shared, referenced-by-id registry
-- entities describing topology; tuning is iterated per run the same way
-- every other tracker_configs column already is (see the v27 comment on
-- process_noise_vel_joint_names for the same config-side-scoping-over-
-- skeleton-groups precedent). A tracker_config_id with any
-- tracker_config_stages rows is what selects hierarchical mode; one
-- without runs monolithic, unchanged.

BEGIN;

CREATE TABLE IF NOT EXISTS tracking_run_stages (
    run_id       TEXT    NOT NULL REFERENCES tracking_runs(id),
    person_id    INTEGER NOT NULL,
    group_name   TEXT    NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending', 'running', 'complete', 'stale')),
    started_at   TEXT,
    completed_at TEXT,
    PRIMARY KEY (run_id, person_id, group_name)
);

-- tracker_config_id -- references registry: tracker_configs(id)
-- group_name         -- matches a skeleton groups: entry (e.g. "HandL"); the
--                        group's joint/marker/depends_on list and reference
--                        marker live in the skeleton, not here.
-- All tuning columns NULL = inherit the parent tracker_configs row's value.
CREATE TABLE IF NOT EXISTS tracker_config_stages (
    tracker_config_id     TEXT NOT NULL,
    group_name            TEXT NOT NULL,
    process_noise_std     REAL,
    process_noise_vel_std REAL,
    velocity_half_life_s  REAL,
    pose_noise_std        REAL,
    calib_noise_std       REAL,
    outlier_threshold     REAL,
    min_inliers_ratio     REAL,
    max_innovation_norm   REAL,
    init_joint_std        REAL,
    init_velocity_std     REAL,
    PRIMARY KEY (tracker_config_id, group_name)
);

PRAGMA user_version = 37;
COMMIT;

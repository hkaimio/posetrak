# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for PR8's hierarchical-solver awareness in the MCP server --
get_run_stages()/get_marker_groups() in db.py, plus their consumers in
get_run_info() and get_filter_stats(). See
docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from app.mcp.db import (
    OBS_ACTUAL_X, OBS_ACTUAL_Y, OBS_MAHAL, OBS_OUTLIER, OBS_PRED_X, OBS_PRED_Y, OBS_USED,
    get_marker_groups,
    get_run_stages,
)
from app.mcp.tools.diagnostics import get_filter_stats
from app.mcp.tools.runs import get_run_info

RUN_ID = "run1"
SKELETON_ID = "skel1"
CONFIG_ID = "cfg1"
CAMERA_LABELS = ["cam0", "cam1"]
MARKER_NAMES = ["MRK-shoulder.R", "MRK-wrist.R", "MRK-index.R"]

_SKELETON_YAML = """
name: test
groups:
  - name: main
    joints: [hips, shoulder.R, forearm.R]
    markers: [MRK-shoulder.R, MRK-wrist.R]
  - name: HandR
    joints: [hand.R]
    markers: [MRK-wrist.R, MRK-index.R]
    freeflyer_joint: forearm.R
    ref_marker: MRK-wrist.R
"""


def _make_obs_blob(slots: dict[tuple[int, int], tuple[float, ...]]) -> bytes:
    n_cam, n_mrk = len(CAMERA_LABELS), len(MARKER_NAMES)
    blob = np.full((n_cam, n_mrk, 8), np.nan, dtype=np.float32)
    for (ci, mi), fields in slots.items():
        blob[ci, mi, :] = fields
    return blob.tobytes()


@pytest.fixture
def conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("CREATE TABLE skeletons (id TEXT PRIMARY KEY, name TEXT, yaml_content TEXT);")
    conn.execute("""
        CREATE TABLE tracker_configs (
            id TEXT PRIMARY KEY, measurement_noise_std REAL, outlier_threshold REAL,
            process_noise_std REAL, process_noise_vel_std REAL, velocity_half_life_s REAL,
            tracker_fps REAL, cross_person_max_world_mm REAL, cross_person_min_confidence REAL,
            cross_person_max_n INTEGER
        );
    """)
    conn.execute("""
        CREATE TABLE tracking_runs (
            id TEXT PRIMARY KEY, ran_at TEXT, notes TEXT, skeleton_id TEXT,
            tracker_config_id TEXT, active_camera_ids TEXT, marker_names TEXT,
            observation_sequence_id TEXT
        );
    """)
    conn.execute("CREATE TABLE camera_instances (id TEXT PRIMARY KEY, label TEXT);")
    conn.execute("""
        CREATE TABLE tracking_results (
            run_id TEXT, person_id INTEGER, tracker_step INTEGER, is_smoothed INTEGER,
            timestamp_s REAL, n_inlier_observations INTEGER, cov_condition_number REAL,
            nis_value REAL, nis_dof INTEGER, tracking_lost INTEGER
        );
    """)
    conn.execute("""
        CREATE TABLE tracking_obs_results (
            run_id TEXT, person_id INTEGER, tracker_step INTEGER, obs_blob BLOB
        );
    """)
    conn.execute("""
        CREATE TABLE tracking_run_stages (
            run_id TEXT, person_id INTEGER, group_name TEXT, status TEXT,
            started_at TEXT, completed_at TEXT
        );
    """)
    conn.execute("CREATE TABLE tracking_run_persons (run_id TEXT, person_id INTEGER, skeleton_id TEXT);")
    conn.execute("CREATE TABLE sequence_persons (sequence_id TEXT, person_id INTEGER, person_name TEXT);")
    conn.execute(
        "INSERT INTO skeletons (id, name, yaml_content) VALUES (?, ?, ?)",
        (SKELETON_ID, "Test Skeleton", _SKELETON_YAML),
    )
    conn.execute(
        "INSERT INTO tracker_configs (id, measurement_noise_std, outlier_threshold, "
        "process_noise_std, process_noise_vel_std, velocity_half_life_s, tracker_fps) "
        "VALUES (?, 5.0, 5.99, 0.5, 0.1, 0.5, 120.0)",
        (CONFIG_ID,),
    )
    conn.execute(
        "INSERT INTO tracking_runs (id, ran_at, notes, skeleton_id, tracker_config_id, "
        "active_camera_ids, marker_names) VALUES (?, '2026-01-01', NULL, ?, ?, ?, ?)",
        (RUN_ID, SKELETON_ID, CONFIG_ID, '["cam0","cam1"]', str(MARKER_NAMES).replace("'", '"')),
    )
    for i, label in enumerate(CAMERA_LABELS):
        conn.execute(
            "INSERT INTO camera_instances (id, label) VALUES (?, ?)", (f"cam-uuid-{i}", label)
        )
    conn.execute(
        "INSERT INTO tracking_results (run_id, person_id, tracker_step, is_smoothed, "
        "timestamp_s, n_inlier_observations, cov_condition_number, nis_value, nis_dof, "
        "tracking_lost) VALUES (?, 0, 1, 0, 0.1, 3, 10.0, 2.0, 3, 0)",
        (RUN_ID,),
    )
    yield conn
    conn.close()


def test_get_run_stages_empty_for_monolithic_run(conn):
    assert get_run_stages(conn, RUN_ID) == []


def test_get_run_stages_returns_rows_for_hierarchical_run(conn):
    conn.execute(
        "INSERT INTO tracking_run_stages (run_id, person_id, group_name, status, "
        "started_at, completed_at) VALUES (?, 0, 'HandR', 'complete', 't0', 't1')",
        (RUN_ID,),
    )
    stages = get_run_stages(conn, RUN_ID)
    assert len(stages) == 1
    assert stages[0]["group_name"] == "HandR"
    assert stages[0]["status"] == "complete"


def test_get_marker_groups_handles_dual_membership(conn):
    groups = get_marker_groups(conn, SKELETON_ID)
    assert groups["MRK-shoulder.R"] == ["main"]
    assert set(groups["MRK-wrist.R"]) == {"main", "HandR"}
    assert groups["MRK-index.R"] == ["HandR"]


def test_get_marker_groups_empty_skeleton_returns_empty_dict(conn):
    conn.execute(
        "INSERT INTO skeletons (id, name, yaml_content) VALUES ('skel2', 'no groups', 'name: x')"
    )
    assert get_marker_groups(conn, "skel2") == {}


def test_get_run_info_reports_monolithic_when_no_stages(conn):
    text = get_run_info(conn, RUN_ID)
    assert "Hierarchical stages: none (monolithic run)" in text


def test_get_run_info_lists_stages_when_hierarchical(conn):
    conn.execute(
        "INSERT INTO tracking_run_stages (run_id, person_id, group_name, status, "
        "started_at, completed_at) VALUES (?, 0, 'HandR', 'complete', 't0', 't1')",
        (RUN_ID,),
    )
    text = get_run_info(conn, RUN_ID)
    assert "Hierarchical stages, person 0:" in text
    assert "HandR" in text
    assert "complete" in text
    assert "NIS/cov_condition_number/n_inlier_observations below reflect the PARENT" in text


def test_get_filter_stats_no_hierarchical_note_when_monolithic(conn):
    text = get_filter_stats(conn, RUN_ID, 0.0, 1.0)
    assert "hierarchical" not in text.lower()
    assert "Per-stage observation summary" not in text


def test_get_filter_stats_labels_parent_only_and_adds_per_stage_summary(conn):
    conn.execute(
        "INSERT INTO tracking_run_stages (run_id, person_id, group_name, status, "
        "started_at, completed_at) VALUES (?, 0, 'HandR', 'complete', 't0', 't1')",
        (RUN_ID,),
    )
    # HandR markers: MRK-wrist.R (also main -- parent-wins, no obs here) and MRK-index.R.
    blob = _make_obs_blob({
        # cam0: index inlier, mahal=1.5
        (0, 2): (10, 20, 11, 19, 1.5, 1, 0, 0.0),
        # cam1: index outlier
        (1, 2): (10, 20, 50, 50, 9.0, 0, 1, 0.0),
    })
    conn.execute(
        "INSERT INTO tracking_obs_results (run_id, person_id, tracker_step, obs_blob) "
        "VALUES (?, 0, 1, ?)",
        (RUN_ID, blob),
    )

    text = get_filter_stats(conn, RUN_ID, 0.0, 1.0)
    assert "this run is hierarchical" in text
    assert "reflect the PARENT (body-only) filter instance only" in text
    assert "Per-stage observation summary" in text
    assert "HandR" in text
    assert "inlier=     1" in text
    assert "outlier=    1" in text
    assert "mean_mahal(inliers)=1.50" in text

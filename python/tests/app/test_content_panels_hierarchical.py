# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for PR8's hierarchical-solver-aware helpers in content_panels.py --
_get_config_stage_groups(), _get_run_stage_rows(), _stages_text(), and
_cfg_text()'s hierarchical_groups annotation. See
docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md.

These are pure DB-query/formatting helpers, importable and testable headlessly
without a running Qt application -- the widget layout itself (_RunInfoPane,
TrackingRunPanel) still needs the live UI walkthrough CLAUDE.md requires
before PR8 can be called done end-to-end.
"""

from __future__ import annotations

import sqlite3

import pytest

from app.ui.content_panels import (
    _cfg_text,
    _get_config_stage_groups,
    _get_run_stage_rows,
    _stages_text,
)

RUN_ID = "run1"
CONFIG_ID = "cfg1"


@pytest.fixture
def conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE tracker_configs (
            id TEXT PRIMARY KEY, name TEXT, process_noise_std REAL,
            process_noise_vel_std REAL, velocity_half_life_s REAL,
            measurement_noise_std REAL, pose_noise_std REAL, outlier_threshold REAL,
            tracker_fps REAL, velocity_mode_camera_ids TEXT,
            velocity_measurement_noise_std REAL, use_relative_observations INTEGER,
            relative_min_confidence REAL, cross_pair_max_px REAL, cross_pair_max_n INTEGER,
            cross_person_max_world_mm REAL, cross_person_min_confidence REAL,
            cross_person_max_n INTEGER
        );
    """)
    conn.execute("""
        CREATE TABLE tracker_config_stages (tracker_config_id TEXT, group_name TEXT);
    """)
    conn.execute("""
        CREATE TABLE tracking_run_stages (
            run_id TEXT, person_id INTEGER, group_name TEXT, status TEXT,
            started_at TEXT, completed_at TEXT
        );
    """)
    conn.execute(
        "INSERT INTO tracker_configs (id, name, measurement_noise_std, outlier_threshold) "
        "VALUES (?, 'base', 25.0, 6.0)",
        (CONFIG_ID,),
    )
    yield conn
    conn.close()


def test_get_config_stage_groups_empty_for_monolithic_config(conn):
    assert _get_config_stage_groups(conn, CONFIG_ID) == []


def test_get_config_stage_groups_empty_for_none_config_id(conn):
    assert _get_config_stage_groups(conn, None) == []


def test_get_config_stage_groups_returns_sorted_names(conn):
    conn.execute(
        "INSERT INTO tracker_config_stages (tracker_config_id, group_name) VALUES (?, 'HandR')",
        (CONFIG_ID,),
    )
    conn.execute(
        "INSERT INTO tracker_config_stages (tracker_config_id, group_name) VALUES (?, 'HandL')",
        (CONFIG_ID,),
    )
    assert _get_config_stage_groups(conn, CONFIG_ID) == ["HandL", "HandR"]


def test_get_run_stage_rows_empty_for_missing_run_id(conn):
    assert _get_run_stage_rows(conn, None) == []


def test_get_run_stage_rows_scoped_to_person(conn):
    conn.execute(
        "INSERT INTO tracking_run_stages (run_id, person_id, group_name, status) "
        "VALUES (?, 0, 'HandR', 'complete')",
        (RUN_ID,),
    )
    conn.execute(
        "INSERT INTO tracking_run_stages (run_id, person_id, group_name, status) "
        "VALUES (?, 1, 'HandR', 'running')",
        (RUN_ID,),
    )
    rows = _get_run_stage_rows(conn, RUN_ID, person_id=0)
    assert len(rows) == 1
    assert rows[0]["status"] == "complete"


def test_stages_text_monolithic():
    assert _stages_text([]) == "none (monolithic run)"


def test_stages_text_lists_group_and_status(conn):
    conn.execute(
        "INSERT INTO tracking_run_stages (run_id, person_id, group_name, status) "
        "VALUES (?, 0, 'HandL', 'complete')",
        (RUN_ID,),
    )
    conn.execute(
        "INSERT INTO tracking_run_stages (run_id, person_id, group_name, status) "
        "VALUES (?, 0, 'HandR', 'running')",
        (RUN_ID,),
    )
    rows = _get_run_stage_rows(conn, RUN_ID)
    text = _stages_text(rows)
    assert "HandL: complete" in text
    assert "HandR: running" in text


def test_cfg_text_no_hier_suffix_when_monolithic(conn):
    cfg = conn.execute("SELECT * FROM tracker_configs WHERE id=?", (CONFIG_ID,)).fetchone()
    text = _cfg_text(cfg, CONFIG_ID, hierarchical_groups=[])
    assert "hier:" not in text


def test_cfg_text_appends_hier_suffix_when_hierarchical(conn):
    cfg = conn.execute("SELECT * FROM tracker_configs WHERE id=?", (CONFIG_ID,)).fetchone()
    text = _cfg_text(cfg, CONFIG_ID, hierarchical_groups=["HandL", "HandR"])
    assert "hier:HandL,HandR" in text

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for PR8's hierarchical-solver toggle support in run_tracker.py --
discover_stage_groups(). See
docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md.

Pure DB-query/YAML-parsing helper, importable and testable headlessly without
a running Qt application -- the widget behaviour itself (the stage table,
_create_config()'s tracker_config_stages inserts) still needs the live UI
walkthrough CLAUDE.md requires before PR8 can be called done end-to-end.
"""

from __future__ import annotations

import sqlite3

import pytest

from app.pose.run_tracker import discover_stage_groups

_SKELETON_WITH_HANDS = """
name: test
groups:
  - name: main
    joints: [hips]
    markers: []
  - name: HandL
    joints: [hand.L]
    markers: []
    freeflyer_joint: forearm.L
    ref_marker: MRK-wrist.L
  - name: HandR
    joints: [hand.R]
    markers: []
    freeflyer_joint: forearm.R
    ref_marker: MRK-wrist.R
"""

_SKELETON_NO_HANDS = """
name: plain
groups:
  - name: main
    joints: [hips]
    markers: []
"""


@pytest.fixture
def conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE skeletons (id TEXT PRIMARY KEY, yaml_content TEXT);")
    conn.execute(
        "INSERT INTO skeletons (id, yaml_content) VALUES ('hands', ?)", (_SKELETON_WITH_HANDS,)
    )
    conn.execute(
        "INSERT INTO skeletons (id, yaml_content) VALUES ('plain', ?)", (_SKELETON_NO_HANDS,)
    )
    yield conn
    conn.close()


def test_discover_stage_groups_finds_freeflyer_groups(conn):
    assert discover_stage_groups(conn, ["hands"]) == ["HandL", "HandR"]


def test_discover_stage_groups_excludes_main(conn):
    groups = discover_stage_groups(conn, ["hands"])
    assert "main" not in groups


def test_discover_stage_groups_empty_for_skeleton_without_freeflyer_groups(conn):
    assert discover_stage_groups(conn, ["plain"]) == []


def test_discover_stage_groups_unions_across_skeletons_without_duplicates(conn):
    # "plain" contributes nothing; "hands" contributes both -- verifies dedup
    # when the same group name would otherwise appear twice.
    groups = discover_stage_groups(conn, ["plain", "hands", "hands"])
    assert groups == ["HandL", "HandR"]


def test_discover_stage_groups_ignores_unknown_skeleton_id(conn):
    assert discover_stage_groups(conn, ["does-not-exist"]) == []


def test_discover_stage_groups_empty_input_list(conn):
    assert discover_stage_groups(conn, []) == []

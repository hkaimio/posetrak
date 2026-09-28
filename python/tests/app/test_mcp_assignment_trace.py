# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for get_assignment_trace -- the "why was detection N (not) assigned
to slot X in frame F" MCP diagnostic tool over tracking_assignment_trace
(dot_assignment.hpp's AssignmentTrace, sword-swing-tracking plan, 2026-09-27).
"""

from __future__ import annotations

import sqlite3

import pytest

from app.mcp.tools.diagnostics import get_assignment_trace

RUN_ID = "run1"
CAMERA_LABELS = ["cam0", "cam1"]
MARKER_NAMES = ["dot0", "dot1"]


@pytest.fixture
def conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("""
        CREATE TABLE tracking_runs (
            id TEXT PRIMARY KEY, active_camera_ids TEXT, marker_names TEXT
        );
    """)
    conn.execute("CREATE TABLE camera_instances (id TEXT PRIMARY KEY, label TEXT);")
    conn.execute("""
        CREATE TABLE tracking_results (
            run_id TEXT, person_id INTEGER, tracker_step INTEGER,
            is_smoothed INTEGER, timestamp_s REAL
        );
    """)
    conn.execute("""
        CREATE TABLE tracking_assignment_trace (
            run_id TEXT, person_id INTEGER, tracker_step INTEGER, camera_idx INTEGER,
            candidate_row INTEGER, tracklet_id INTEGER, candidate_x REAL, candidate_y REAL,
            marker_idx INTEGER, mahal_cost REAL, final_cost REAL, excluded_by TEXT,
            in_final_assignment INTEGER
        );
    """)
    conn.execute(
        "INSERT INTO tracking_runs (id, active_camera_ids, marker_names) VALUES (?, ?, ?)",
        (RUN_ID, '["cam0","cam1"]', '["dot0","dot1"]'),
    )
    for i, label in enumerate(CAMERA_LABELS):
        conn.execute(
            "INSERT INTO camera_instances (id, label) VALUES (?, ?)", (f"cam-uuid-{i}", label)
        )
    conn.execute(
        "INSERT INTO tracking_results (run_id, person_id, tracker_step, is_smoothed, timestamp_s) "
        "VALUES (?, 0, 100, 0, 1.0)",
        (RUN_ID,),
    )
    yield conn
    conn.close()


def _insert_trace_row(conn, **fields):
    defaults = dict(
        run_id=RUN_ID, person_id=0, tracker_step=100, camera_idx=0, candidate_row=0,
        tracklet_id=-1, candidate_x=10.0, candidate_y=20.0, marker_idx=0, mahal_cost=1.0,
        final_cost=1.0, excluded_by="", in_final_assignment=0,
    )
    defaults.update(fields)
    conn.execute(
        "INSERT INTO tracking_assignment_trace "
        "(run_id, person_id, tracker_step, camera_idx, candidate_row, tracklet_id, "
        " candidate_x, candidate_y, marker_idx, mahal_cost, final_cost, excluded_by, "
        " in_final_assignment) VALUES "
        "(:run_id, :person_id, :tracker_step, :camera_idx, :candidate_row, :tracklet_id, "
        " :candidate_x, :candidate_y, :marker_idx, :mahal_cost, :final_cost, :excluded_by, "
        " :in_final_assignment)",
        defaults,
    )


def test_no_rows_in_range_says_so_not_silently_empty(conn):
    text = get_assignment_trace(conn, RUN_ID, 0.0, 2.0)
    assert "No assignment-trace rows" in text
    assert "re-run with --trace-assignment-start" in text


def test_shows_winner_and_excluded_candidates(conn):
    _insert_trace_row(
        conn, candidate_row=0, marker_idx=0, mahal_cost=0.8, final_cost=0.8,
        in_final_assignment=1,
    )
    _insert_trace_row(
        conn, candidate_row=1, marker_idx=0, mahal_cost=2.0, final_cost=2.0,
        excluded_by="candidate_ownership", in_final_assignment=0,
    )

    text = get_assignment_trace(conn, RUN_ID, 0.0, 2.0)

    lines = [l for l in text.splitlines() if l.strip().startswith("100")]
    assert len(lines) == 2
    winner_line = next(l for l in lines if "WON" in l)
    assert "dot0" in winner_line
    excluded_line = next(l for l in lines if "candidate_ownership" in l)
    assert "WON" not in excluded_line


def test_marker_filter_narrows_to_one_slot(conn):
    _insert_trace_row(conn, candidate_row=0, marker_idx=0, in_final_assignment=1)
    _insert_trace_row(conn, candidate_row=0, marker_idx=1, in_final_assignment=1)

    text = get_assignment_trace(conn, RUN_ID, 0.0, 2.0, marker="dot1")

    assert "dot0" not in text.split("marker=dot1")[1].split("\n", 1)[1]
    assert "dot1" in text


def test_unknown_marker_raises_via_marker_indices(conn):
    _insert_trace_row(conn)
    with pytest.raises(ValueError, match="not found in this run"):
        get_assignment_trace(conn, RUN_ID, 0.0, 2.0, marker="nope")

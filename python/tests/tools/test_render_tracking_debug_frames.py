# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for render_tracking_debug_frames.py's _nearest_tracker_step() --
this module had no test coverage at all before (a standalone visualization
tool, same "no unit tests for the DB-loading parts" precedent as
calibrate_rigid_marker_body.py), but the one real bug found in it
(2026-09-05) was serious enough to cover directly: RTS-smoothed
tracking_results rows use a *different* tracker_step numbering than raw
ones (smoothing only ever covers steps that were actually tracked, so a
gap-heavy run's smoothed index runs ahead of the raw one by however many
steps were lost before it), but tracking_obs_results is only ever written
from the raw forward pass -- so a query that doesn't filter is_smoothed=0
can return a smoothed row whose tracker_step, reused to look up
tracking_obs_results, silently fetches a *different, real* instant's
observation data and overlays it on the wrong video frame. Confirmed on a
real run: raw tracker_step 1933 and smoothed tracker_step 1603 both carry
timestamp_s=53.766 -- reusing 1603 to key into tracking_obs_results
actually fetches data from timestamp_s=50.466 instead.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "render_tracking_debug_frames.py"
)
_spec = importlib.util.spec_from_file_location("render_tracking_debug_frames", _MODULE_PATH)
render_tracking_debug_frames = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = render_tracking_debug_frames
_spec.loader.exec_module(render_tracking_debug_frames)

_nearest_tracker_step = render_tracking_debug_frames._nearest_tracker_step


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE tracking_results ("
        " run_id TEXT, person_id INTEGER, tracker_step INTEGER,"
        " is_smoothed INTEGER, timestamp_s REAL, state BLOB)"
    )
    return conn


def test_nearest_tracker_step_ignores_a_smoothed_rows_own_step_numbering():
    """The real bug: a smoothed row's tracker_step (1603) is a different
    numbering than the raw row that actually shares its real timestamp
    (1933) -- a naive nearest-timestamp query with no is_smoothed filter
    can return the smoothed row and its (wrong-for-lookup-purposes) step
    number, which tracking_obs_results (raw-indexed only) cannot answer
    for correctly.
    """
    conn = _make_conn()
    rows = [
        ("run1", 0, 1603, 0, 50.466, b"raw-1603"),  # raw: step 1603 is an EARLIER instant
        ("run1", 0, 1603, 1, 53.766, b"smoothed-1603"),  # smoothed: step 1603 reused, later instant
        ("run1", 0, 1933, 0, 53.766, b"raw-1933"),  # raw: this is the real step for 53.766
        ("run1", 0, 1933, 1, 58.406, b"smoothed-1933"),
    ]
    conn.executemany(
        "INSERT INTO tracking_results VALUES (?,?,?,?,?,?)", rows
    )

    step, ts, state = _nearest_tracker_step(conn, "run1", 53.766)

    assert step == 1933  # the raw row's own step number, not the smoothed row's
    assert ts == 53.766
    assert state == b"raw-1933"


def test_nearest_tracker_step_picks_the_closest_raw_row():
    conn = _make_conn()
    rows = [
        ("run1", 0, 10, 0, 1.0, b"a"),
        ("run1", 0, 11, 0, 2.0, b"b"),
        ("run1", 0, 12, 0, 3.0, b"c"),
        ("run1", 0, 10, 1, 100.0, b"smoothed-far-away"),
    ]
    conn.executemany("INSERT INTO tracking_results VALUES (?,?,?,?,?,?)", rows)

    step, ts, state = _nearest_tracker_step(conn, "run1", 2.1)

    assert step == 11
    assert state == b"b"


def test_nearest_tracker_step_returns_none_with_no_matching_run():
    conn = _make_conn()
    assert _nearest_tracker_step(conn, "no-such-run", 1.0) is None

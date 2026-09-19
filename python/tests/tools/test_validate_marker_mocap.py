# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the helpers of scripts/validate_marker_mocap.py that need no
real capture: parsing the tracker's output and detecting a physical candidate
used by two subjects at once."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "validate_marker_mocap.py"
_spec = importlib.util.spec_from_file_location("validate_marker_mocap", _MODULE_PATH)
validate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = validate
_spec.loader.exec_module(validate)

CAMERAS = ["camA", "camB"]


def _blob(*observations: tuple[int, float, float, int]) -> bytes:
    """obs_blob for 2 cameras x 1 marker: each observation is (camera index,
    actual_x, actual_y, used); mode 0 (position)."""
    arr = np.full((len(CAMERAS), 1, 8), np.nan, dtype=np.float32)
    for cam, x, y, used in observations:
        arr[cam, 0] = [x, y, x, y, 0.0, used, 0.0, 0.0]
    return arr.tobytes()


@pytest.fixture()
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(
        """
        CREATE TABLE tracking_runs (id TEXT PRIMARY KEY, active_camera_ids TEXT);
        CREATE TABLE tracking_results (run_id TEXT, tracker_step INTEGER, timestamp_s REAL, is_smoothed INTEGER);
        CREATE TABLE tracking_obs_results (run_id TEXT, tracker_step INTEGER, obs_blob BLOB);
        """
    )
    return c


def _add_run(c: sqlite3.Connection, run_id: str, steps: dict[int, bytes]) -> None:
    c.execute("INSERT INTO tracking_runs VALUES (?, ?)", (run_id, json.dumps(CAMERAS)))
    for step, blob in steps.items():
        c.execute("INSERT INTO tracking_results VALUES (?, ?, ?, 0)", (run_id, step, 10.0 + step / 120.0))
        c.execute("INSERT INTO tracking_obs_results VALUES (?, ?, ?)", (run_id, step, blob))


def test_double_claims_counts_a_pixel_used_by_two_runs_at_the_same_time_and_camera(conn):
    _add_run(conn, "a", {1: _blob((0, 100.0, 200.0, 1))})
    _add_run(conn, "b", {1: _blob((0, 100.0, 200.0, 1))})
    assert validate.double_claims(conn, ["a", "b"]) == 1


def test_double_claims_ignores_different_pixels_cameras_times_and_unused_observations(conn):
    _add_run(conn, "a", {1: _blob((0, 100.0, 200.0, 1)), 2: _blob((0, 100.0, 200.0, 1))})
    _add_run(conn, "b", {
        1: _blob((0, 101.0, 200.0, 1)),   # other pixel
        2: _blob((1, 100.0, 200.0, 1)),   # other camera
        3: _blob((0, 100.0, 200.0, 1)),   # other time
    })
    _add_run(conn, "c", {1: _blob((0, 100.0, 200.0, 0))})  # same pixel but not used
    assert validate.double_claims(conn, ["a", "b", "c"]) == 0


def test_parse_tracker_output_reads_the_headline_numbers():
    text = (
        "  Rigid-body init: Kabsch/Umeyama fit over 4 markers, RMS residual = 0.0018 m\n"
        "tracking_run_id: 020952f5-7edf-426e-887d-e526c0b19ca1\n"
        "  Tracked: 1747/1798 steps (97.2%)\n"
        "  Lost: 0 steps\n"
    )
    parsed = validate.parse_tracker_output(text)
    assert parsed == {
        "run_id": "020952f5-7edf-426e-887d-e526c0b19ca1",
        "tracked": 1747, "total": 1798, "tracked_pct": 97.2, "lost": 0, "init_rms_mm": pytest.approx(1.8),
    }

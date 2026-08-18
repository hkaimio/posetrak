# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the obs_blob pad-field (index 7) mode flag added for the
hierarchical solver's patch_obs_results() -- see
docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md.

Verifies decode_obs_blob() and its MCP consumers (get_camera_coverage,
get_observation_gaps) surface the flag rather than silently ignoring it.
"""

from __future__ import annotations

import sqlite3
import struct

import numpy as np
import pytest

from app.mcp.db import (
    OBS_MODE_ABSOLUTE,
    OBS_MODE_PAIR_DIFF_RECONSTRUCTED,
    OBS_PAD,
    decode_obs_blob,
)
from app.mcp.tools.coverage import get_camera_coverage
from app.mcp.tools.diagnostics import get_observation_gaps

RUN_ID = "run1"
CAMERA_LABELS = ["cam0", "cam1"]
MARKER_NAMES = ["MRK-wrist", "MRK-index_1"]


def _make_obs_blob(slots: dict[tuple[int, int], tuple[float, ...]]) -> bytes:
    """Build a float32[n_cam, n_mrk, 8] blob; unset slots are all-NaN."""
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
    conn.execute("""
        CREATE TABLE tracking_runs (
            id TEXT PRIMARY KEY, active_camera_ids TEXT, marker_names TEXT
        );
    """)
    conn.execute("""
        CREATE TABLE camera_instances (id TEXT PRIMARY KEY, label TEXT);
    """)
    conn.execute("""
        CREATE TABLE tracking_results (
            run_id TEXT, person_id INTEGER, tracker_step INTEGER,
            is_smoothed INTEGER, timestamp_s REAL
        );
    """)
    conn.execute("""
        CREATE TABLE tracking_obs_results (
            run_id TEXT, person_id INTEGER, tracker_step INTEGER, obs_blob BLOB
        );
    """)
    conn.execute(
        "INSERT INTO tracking_runs (id, active_camera_ids, marker_names) VALUES (?, ?, ?)",
        (RUN_ID, '["cam0","cam1"]', '["MRK-wrist","MRK-index_1"]'),
    )
    for i, label in enumerate(CAMERA_LABELS):
        conn.execute(
            "INSERT INTO camera_instances (id, label) VALUES (?, ?)", (f"cam-uuid-{i}", label)
        )
    conn.execute(
        "INSERT INTO tracking_results (run_id, person_id, tracker_step, is_smoothed, timestamp_s) "
        "VALUES (?, 0, 1, 0, 0.1)",
        (RUN_ID,),
    )
    yield conn
    conn.close()


def test_obs_mode_constants_are_distinct():
    assert OBS_MODE_ABSOLUTE != OBS_MODE_PAIR_DIFF_RECONSTRUCTED
    assert OBS_PAD == 7


def test_decode_obs_blob_roundtrips_pad_field():
    blob = _make_obs_blob({
        (0, 0): (10, 20, 11, 19, 0.5, 1, 0, OBS_MODE_ABSOLUTE),
        (0, 1): (30, 40, 29, 41, 2.2, 0, 1, OBS_MODE_PAIR_DIFF_RECONSTRUCTED),
    })
    decoded = decode_obs_blob(blob, len(CAMERA_LABELS), len(MARKER_NAMES))
    assert decoded.shape == (2, 2, 8)
    assert decoded[0, 0, OBS_PAD] == OBS_MODE_ABSOLUTE
    assert decoded[0, 1, OBS_PAD] == OBS_MODE_PAIR_DIFF_RECONSTRUCTED


def test_get_camera_coverage_marks_reconstructed_entries_lowercase(conn):
    blob = _make_obs_blob({
        # cam0: native inlier for both markers.
        (0, 0): (10, 20, 11, 19, 0.5, 1, 0, OBS_MODE_ABSOLUTE),
        (0, 1): (10, 20, 11, 19, 0.5, 1, 0, OBS_MODE_ABSOLUTE),
        # cam1: reconstructed (child stage) inlier and outlier.
        (1, 0): (10, 20, 11, 19, 0.5, 1, 0, OBS_MODE_PAIR_DIFF_RECONSTRUCTED),
        (1, 1): (10, 20, 11, 19, 9.0, 0, 1, OBS_MODE_PAIR_DIFF_RECONSTRUCTED),
    })
    conn.execute(
        "INSERT INTO tracking_obs_results (run_id, person_id, tracker_step, obs_blob) "
        "VALUES (?, 0, 1, ?)",
        (RUN_ID, blob),
    )

    text = get_camera_coverage(conn, RUN_ID, 0.0, 1.0, MARKER_NAMES, stride=1)

    assert "lowercase i/x" in text
    lines = [l for l in text.splitlines() if l.strip().startswith("1 ")]
    assert len(lines) == 1
    row = lines[0]
    # cam0 (native) columns use uppercase; cam1 (reconstructed) uses lowercase.
    cells = row.split("|", 1)[1].split()
    # markers x cameras = 2 x 2 = 4 cells: [wrist/cam0, wrist/cam1, index/cam0, index/cam1]
    assert cells == ["I", "i", "I", "x"]


def test_get_observation_gaps_flags_reconstructed_status(conn):
    blob = _make_obs_blob({
        (0, 0): (10, 20, 11, 19, 0.5, 1, 0, OBS_MODE_ABSOLUTE),
        (1, 1): (30, 40, 29, 41, 0.5, 1, 0, OBS_MODE_PAIR_DIFF_RECONSTRUCTED),
    })
    conn.execute(
        "INSERT INTO tracking_obs_results (run_id, person_id, tracker_step, obs_blob) "
        "VALUES (?, 0, 1, ?)",
        (RUN_ID, blob),
    )

    text = get_observation_gaps(conn, RUN_ID, 0.0, 1.0, MARKER_NAMES, stride=1)

    assert "status 'r' = reconstructed" in text
    wrist_section = text.split("=== MRK-wrist ===")[1].split("=== MRK-index_1 ===")[0]
    assert wrist_section.rstrip().endswith(" I")  # native: no 'r' suffix
    index_section = text.split("=== MRK-index_1 ===")[1]
    assert index_section.rstrip().endswith(" Ir")  # reconstructed: 'r' suffix

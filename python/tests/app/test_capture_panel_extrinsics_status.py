"""Tests for CapturePanel's Extrinsics… button status refresh (UX Phase 2,
see docs/roadmap/features/extrinsics-improvements/
extrinsics-ux-redesign.md): mirrors _refresh_sync()'s pattern of querying
current DB state and updating a toolbar control's text/tooltip.

Uses this test suite's established __new__ bypass (see e.g. test_phase5.py)
rather than real CapturePanel(...) construction: __init__/_build() would
try to open capture_videos' file_path for real to build a MultiVideoScrubber,
which hangs against this file's fake paths. _refresh_extrinsics() only
touches self._conn/_capture_id/_ext_btn, so the bypass is exact for what's
under test here.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QPushButton, QWidget

from app.ui.content_panels import CapturePanel
from posetrak.db.db import create_session


@pytest.fixture()
def session_db(tmp_path: Path):
    conn = create_session(tmp_path / "capture_panel_test.db")
    yield conn
    conn.close()


def _make_capture(conn: sqlite3.Connection, capture_id: str = "cap1") -> str:
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES (?, 'sess1', 1)",
        (capture_id,),
    )
    conn.commit()
    return capture_id


def _seed_camera(
    conn: sqlite3.Connection, inst_id: str, label: str, capture_id: str = "cap1",
) -> None:
    """Registers a camera *and* gives it a video in *capture_id* --
    _refresh_extrinsics() scopes its camera count to cameras with a video
    in this panel's own capture, not every camera the session has ever
    registered (2026-08-14 fix)."""
    conn.execute(
        "INSERT OR IGNORE INTO camera_models (id, manufacturer, model_name) "
        "VALUES ('model1', 'Test', 'Cam')"
    )
    conn.execute(
        "INSERT INTO camera_instances (id, camera_model_id, label) VALUES (?, 'model1', ?)",
        (inst_id, label),
    )
    conn.execute(
        "INSERT INTO capture_videos "
        "(id, shot_id, camera_instance_id, file_path, first_video_frame, "
        " last_video_frame, actual_fps) "
        "VALUES (?, ?, ?, '/fake/video.mp4', 0, 1000, 30.0)",
        (f"vid_{inst_id}", capture_id, inst_id),
    )
    conn.commit()


def _seed_calibration(conn: sqlite3.Connection, calib_id: str = "calib1") -> None:
    conn.execute(
        "INSERT INTO extrinsic_calibrations (id, session_id, calibrated_at, method) "
        "VALUES (?, 'sess1', '2026-08-12', 'rig-anchor')",
        (calib_id,),
    )
    conn.commit()


def _seed_entry(conn: sqlite3.Connection, calib_id: str, camera_instance_id: str) -> None:
    R_blob = struct.pack("<9d", *np.eye(3).flatten())
    t_blob = struct.pack("<3d", *np.zeros(3))
    conn.execute(
        "INSERT INTO extrinsic_entries (extrinsic_calibration_id, camera_instance_id, R, t) "
        "VALUES (?, ?, ?, ?)",
        (calib_id, camera_instance_id, R_blob, t_blob),
    )
    conn.commit()


def _make_panel(conn: sqlite3.Connection, capture_id: str) -> CapturePanel:
    panel = CapturePanel.__new__(CapturePanel)
    QWidget.__init__(panel)
    panel._conn = conn
    panel._capture_id = capture_id
    panel._ext_btn = QPushButton("Extrinsics…")
    return panel


def test_no_session_shows_default_text(qapp, tmp_path: Path) -> None:
    conn = create_session(tmp_path / "no_session.db")
    try:
        panel = _make_panel(conn, "nonexistent-capture")
        panel._refresh_extrinsics()
        assert panel._ext_btn.text() == "Extrinsics…"
    finally:
        conn.close()


def test_session_with_no_calibration_shows_not_set(qapp, session_db) -> None:
    capture_id = _make_capture(session_db)
    panel = _make_panel(session_db, capture_id)
    panel._refresh_extrinsics()
    assert panel._ext_btn.text() == "Extrinsics (not set)"
    assert "No extrinsics" in panel._ext_btn.toolTip()


def test_session_with_full_calibration_shows_counts(qapp, session_db) -> None:
    capture_id = _make_capture(session_db)
    _seed_camera(session_db, "inst1", "cam_A")
    _seed_camera(session_db, "inst2", "cam_B")
    _seed_calibration(session_db)
    _seed_entry(session_db, "calib1", "inst1")
    _seed_entry(session_db, "calib1", "inst2")

    panel = _make_panel(session_db, capture_id)
    panel._refresh_extrinsics()
    assert panel._ext_btn.text() == "Extrinsics ✓ (2/2)"


def test_session_with_partial_calibration_shows_counts(qapp, session_db) -> None:
    capture_id = _make_capture(session_db)
    _seed_camera(session_db, "inst1", "cam_A")
    _seed_camera(session_db, "inst2", "cam_B")
    _seed_calibration(session_db)
    _seed_entry(session_db, "calib1", "inst1")  # only one of two solved

    panel = _make_panel(session_db, capture_id)
    panel._refresh_extrinsics()
    assert panel._ext_btn.text() == "Extrinsics ✓ (1/2)"


def test_camera_count_scoped_to_this_capture_not_whole_session(qapp, session_db) -> None:
    """A camera registered for the session but only ever used in a
    *different* capture shouldn't count here (2026-08-14 fix)."""
    capture_id = _make_capture(session_db, "cap1")
    session_db.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('cap2', 'sess1', 2)"
    )
    session_db.commit()
    _seed_camera(session_db, "inst1", "cam_A", capture_id="cap1")
    _seed_camera(session_db, "inst2", "cam_B", capture_id="cap2")  # a different capture
    _seed_calibration(session_db)
    _seed_entry(session_db, "calib1", "inst1")
    _seed_entry(session_db, "calib1", "inst2")

    panel = _make_panel(session_db, capture_id)
    panel._refresh_extrinsics()
    assert panel._ext_btn.text() == "Extrinsics ✓ (1/1)"  # only cam_A is this capture's


def test_refresh_extrinsics_reflects_new_calibration(qapp, session_db) -> None:
    """Calling _refresh_extrinsics() again (as _open_extrinsics does after
    the status dialog closes) picks up newly-written state."""
    capture_id = _make_capture(session_db)
    _seed_camera(session_db, "inst1", "cam_A")
    panel = _make_panel(session_db, capture_id)
    panel._refresh_extrinsics()
    assert panel._ext_btn.text() == "Extrinsics (not set)"

    _seed_calibration(session_db)
    _seed_entry(session_db, "calib1", "inst1")
    panel._refresh_extrinsics()

    assert panel._ext_btn.text() == "Extrinsics ✓ (1/1)"

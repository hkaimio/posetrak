# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for StandaloneRunPanel's object (marker) branch.

Regression coverage for the bug where opening a marker detection run in the
main viewer always built a StitcherPanel -- the person track-to-person
assignment UI -- which has no person tracks to show for an object run and no
way to proceed. StandaloneRunPanel must instead show a small object summary
with a "Finalise" action (if not finalised yet) or a "Review corners..."
action (once it is), reusing finalise_object_to_db and navigating into
ObjectPanel via sequence id, never StitcherPanel.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from PySide6.QtWidgets import QPushButton

from app.pose.db_cache import create_marker_detection_run, MarkerKeypointWriter
from app.pose.finalise import finalise_object_to_db
from posetrak.db.db import create_session, generate_id
from posetrak.db.manage_capture_object import create_capture_object
from posetrak.db.manage_marker_body import import_marker_body_str

_SHOT_ID = "test-shot-id"
_SYNC_ID = "test-sync-id"
_SVID = "test-sv-id"
_CAM_ID = "test-cam-id"

_MARKER_BODY_YAML = """\
name: test-bokken
units: meters
markers:
  - name: hilt
    type: aruco
    dictionary: DICT_4X4_50
    id: "3"
    size: 0.05
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
"""


@pytest.fixture
def object_run(tmp_path):
    """A real marker detection run with real detection_keypoints, not yet
    finalised into a pose_observation_sequence -- the state a run is in
    right after RunDetectionDialog's job finishes but before (or if)
    auto-finalisation runs."""
    db_path = tmp_path / "test.db"
    conn = create_session(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF")
    session_id = generate_id()
    conn.executescript(f"""
        INSERT INTO mocap_sessions (id, recorded_at) VALUES ('{session_id}', '2026-01-01');
        INSERT INTO captures (id, session_id, capture_number, label)
            VALUES ('{_SHOT_ID}', '{session_id}', 1, 'test');
        INSERT INTO camera_instances (id, camera_model_id, label)
            VALUES ('{_CAM_ID}', 'cm1', 'cam_A');
        INSERT INTO sync_configs (id, shot_id, created_by)
            VALUES ('{_SYNC_ID}', '{_SHOT_ID}', 'test');
        INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,
                                 first_video_frame, last_video_frame, actual_fps)
            VALUES ('{_SVID}', '{_SHOT_ID}', '{_CAM_ID}', '/dev/null', 0, 1000, 30.0);
        INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id,
                                 video_frame, timestamp_s)
            VALUES ('{_SYNC_ID}', '{_CAM_ID}', '{_SVID}', 0, 0.0);
    """)
    conn.commit()

    body_id = import_marker_body_str(conn, _MARKER_BODY_YAML, name="Test Bokken")
    object_id = create_capture_object(conn, _SHOT_ID, "bokken-A", body_id)

    run_id = create_marker_detection_run(
        conn, shot_id=_SHOT_ID, sync_config_id=_SYNC_ID, time_start_s=0.0, time_end_s=1.0,
        dictionary="DICT_4X4_50", marker_ids=["3"],
        marker_body_definition_id=body_id, capture_object_id=object_id,
    )
    writer = MarkerKeypointWriter(conn, run_id, _SVID, marker_ids=["3"])
    writer.add_frame(0, [])
    writer.finalise()
    kp = np.zeros((4, 3), dtype=np.float32)
    kp[:, 2] = 1.0
    conn.execute(
        "UPDATE detection_keypoints SET keypoints=? "
        "WHERE detection_run_id=? AND shot_video_id=? AND video_frame=0",
        (kp.tobytes(), run_id, _SVID),
    )
    conn.commit()

    yield conn, run_id, object_id
    conn.close()


def _find_button(panel, text: str) -> QPushButton | None:
    for btn in panel.findChildren(QPushButton):
        if btn.text() == text:
            return btn
    return None


def test_object_run_not_finalised_shows_finalise_button(qapp, object_run):
    from app.ui.content_panels import StandaloneRunPanel

    conn, run_id, _object_id = object_run
    panel = StandaloneRunPanel(conn, run_id)

    assert panel._stitcher_panel is None  # never built for an object run
    finalise_btn = _find_button(panel, "Finalise")
    assert finalise_btn is not None
    assert _find_button(panel, "Review corners…") is None


def test_finalise_button_finalises_and_navigates_to_object_panel(qapp, object_run):
    from app.ui.content_panels import StandaloneRunPanel

    conn, run_id, _object_id = object_run
    panel = StandaloneRunPanel(conn, run_id)

    navigated = []
    panel.navigate_object_track.connect(navigated.append)
    changed = []
    panel.data_changed.connect(lambda: changed.append(True))

    finalise_btn = _find_button(panel, "Finalise")
    finalise_btn.click()

    seq_row = conn.execute(
        "SELECT id FROM pose_observation_sequences WHERE detection_run_id = ?", (run_id,)
    ).fetchone()
    assert seq_row is not None
    assert navigated == [seq_row["id"]]
    assert changed == [True]


def test_object_run_already_finalised_shows_review_button(qapp, object_run):
    from app.ui.content_panels import StandaloneRunPanel

    conn, run_id, _object_id = object_run
    seq_id = finalise_object_to_db(conn, run_id)

    panel = StandaloneRunPanel(conn, run_id)
    assert panel._stitcher_panel is None
    assert _find_button(panel, "Finalise") is None
    review_btn = _find_button(panel, "Review corners…")
    assert review_btn is not None

    navigated = []
    panel.navigate_object_track.connect(navigated.append)
    review_btn.click()
    assert navigated == [seq_id]


def test_finalise_failure_shows_error_and_does_not_navigate(qapp, object_run, monkeypatch):
    from app.ui.content_panels import StandaloneRunPanel
    from PySide6.QtWidgets import QMessageBox

    conn, run_id, _object_id = object_run

    def _boom(session, rid):
        raise RuntimeError("boom")

    critical_calls = []
    monkeypatch.setattr(
        QMessageBox, "critical",
        lambda *a, **k: critical_calls.append(a) or QMessageBox.StandardButton.Ok,
    )

    panel = StandaloneRunPanel(conn, run_id)
    navigated = []
    panel.navigate_object_track.connect(navigated.append)

    import app.pose.finalise as finalise_mod
    monkeypatch.setattr(finalise_mod, "finalise_object_to_db", _boom)

    finalise_btn = _find_button(panel, "Finalise")
    finalise_btn.click()

    assert len(critical_calls) == 1
    assert navigated == []
    assert conn.execute(
        "SELECT COUNT(*) FROM pose_observation_sequences WHERE detection_run_id = ?", (run_id,)
    ).fetchone()[0] == 0

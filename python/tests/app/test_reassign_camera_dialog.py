# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""ReassignCameraDialog exchanges the cameras of two videos through reassign_cameras."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.ui.reassign_camera_dialog import ReassignCameraDialog
from posetrak.db.db import create_session


@pytest.fixture()
def conn(tmp_path: Path):
    c = create_session(tmp_path / "dlg.db")
    c.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES ('s', '2026-01-01')")
    c.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('cap', 's', 1)")
    c.execute("INSERT INTO camera_models (id, manufacturer, model_name) VALUES ('m', 'T', 'C')")
    for cam in ("c1", "c2"):
        c.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES (?, 'm', ?)", (cam, cam))
    for vid, cam in (("v1", "c1"), ("v2", "c2")):
        c.execute(
            "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path, first_video_frame, "
            "last_video_frame, actual_fps) VALUES (?, 'cap', ?, ?, 0, 99, 30.0)", (vid, cam, f"/x/{vid}.mp4"))
    c.commit()
    yield c
    c.close()


def test_exchange_cameras(qapp, conn):
    dlg = ReassignCameraDialog(conn, "cap")
    assert dlg.assignments() == []
    dlg._combos[0].setCurrentIndex(dlg._combos[0].findData("c2"))
    dlg._combos[1].setCurrentIndex(dlg._combos[1].findData("c1"))
    assert {a.video_id: a.camera_instance_id for a in dlg.assignments()} == {"v1": "c2", "v2": "c1"}

    dlg._apply()

    assert dlg.changed
    assert dict(conn.execute("SELECT id, camera_instance_id FROM capture_videos")) == {"v1": "c2", "v2": "c1"}
    assert dict(conn.execute("SELECT id, file_path FROM capture_videos")) == {"v1": "/x/v1.mp4", "v2": "/x/v2.mp4"}

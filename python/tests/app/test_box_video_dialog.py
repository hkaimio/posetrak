# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""BoxVideoDialog saves a calibration only when the verdict passed."""

from __future__ import annotations

import numpy as np
import pytest

from app.setup.box_video import BoxVideoResult, CameraRow
from app.setup.box_video_dialog import BoxVideoDialog
from app.setup.extrinsics_solver import CalibResult, CamCalibState
from posetrak.db.db import create_mocap_session, create_session


def _result(ok: bool) -> BoxVideoResult:
    state = CamCalibState(video_id="cam1", label="cam1", K=np.eye(3), K_orig=np.eye(3),
                          dist=np.zeros((1, 4)), fisheye=False, R=np.eye(3), t=np.zeros((3, 1)))
    return BoxVideoResult(
        calibration=CalibResult(cameras={"cam1": state}, points_3d=[], reprojection_errors={},
                                unsolved=[], pair_matches={}),
        rows=[CameraRow("cam1", True, 3, 40, 1.0 if ok else 80.0, 1.5 if ok else 90.0, 1.0, 2.0, ok)], windows=[], dropped=[],
        anchor_ts=1.0, final_ts=1.0, aligned_to_final=False, alignment_rms_m=None, residuals=[],
        max_median_error_px=5.0, messages=[])


@pytest.fixture()
def dialog(qapp, tmp_path):
    conn = create_session(tmp_path / "dlg.db")
    session_id = create_mocap_session(conn, location="x")
    conn.execute("INSERT INTO captures (id, session_id, capture_number) VALUES ('cap1', ?, 1)", (session_id,))
    conn.execute("INSERT INTO camera_models (id, manufacturer, model_name) VALUES ('m', 'T', 'C')")
    conn.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('i1', 'm', 'cam1')")
    conn.execute("INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path, first_video_frame, "
                 "last_video_frame, actual_fps) VALUES ('v1', 'cap1', 'i1', 'x.mp4', 0, 9, 30.0)")
    conn.commit()
    yield BoxVideoDialog(conn, session_id, "cap1"), conn
    conn.close()


def test_failed_calibration_cannot_be_saved(dialog):
    dlg, conn = dialog
    dlg._on_result(_result(False))
    assert not dlg._save_btn.isEnabled()
    assert "FAILED" in dlg._report.toPlainText() and "80.00" in dlg._report.toPlainText()
    dlg._save()  # even a direct call must not write
    assert conn.execute("SELECT COUNT(*) FROM extrinsic_calibrations").fetchone()[0] == 0


def test_passing_calibration_is_saved_and_linked(dialog):
    dlg, conn = dialog
    dlg._on_result(_result(True))
    assert dlg._save_btn.isEnabled()
    dlg._save()
    assert conn.execute("SELECT COUNT(*) FROM extrinsic_calibrations").fetchone()[0] == 1
    assert conn.execute("SELECT extrinsic_calibration_id FROM captures").fetchone()[0]

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for extrinsics_solver.write_extrinsics_to_db.

Extracted from page_extrinsics.py's previously-private
_write_extrinsics_to_db (2026-08-12) so a non-GUI caller (e.g. a CLI
command) can use the exact same DB write path the GUI does, rather than
re-deriving it. See docs/roadmap/features/extrinsics-improvements/
status.md's 2026-08-12 entries.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from app.setup.extrinsics_solver import CalibResult, CamCalibState, write_extrinsics_to_db
from posetrak.db.db import create_session


@pytest.fixture()
def session_with_camera(tmp_path):
    conn = create_session(tmp_path / "session.db")
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO camera_models (id, manufacturer, model_name) VALUES ('model1', 'Test', 'Cam')"
    )
    conn.execute(
        "INSERT INTO camera_instances (id, camera_model_id, label) "
        "VALUES ('inst1', 'model1', 'cam_a')"
    )
    conn.commit()
    yield conn
    conn.close()


def _solved_state(label: str, R: np.ndarray, t: np.ndarray) -> CamCalibState:
    K = np.eye(3)
    s = CamCalibState(
        video_id=label, label=label, K=K, K_orig=K.copy(), dist=np.zeros((1, 4)), fisheye=False,
    )
    s.R = R
    s.t = t.reshape(3, 1)
    return s


def test_write_extrinsics_to_db_inserts_calibration_and_entry(session_with_camera) -> None:
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    result = CalibResult(
        cameras={"cam_a": _solved_state("cam_a", R, t)},
        points_3d=[], reprojection_errors={}, unsolved=[], pair_matches={},
    )
    calib_id = write_extrinsics_to_db(result, session_with_camera, "sess1", {"cam_a": "inst1"})

    calib_row = session_with_camera.execute(
        "SELECT session_id, method FROM extrinsic_calibrations WHERE id = ?", (calib_id,)
    ).fetchone()
    assert calib_row is not None
    assert calib_row[0] == "sess1"
    assert calib_row[1] == "auto-sift"

    entry_row = session_with_camera.execute(
        "SELECT camera_instance_id, R, t FROM extrinsic_entries "
        "WHERE extrinsic_calibration_id = ?", (calib_id,)
    ).fetchone()
    assert entry_row is not None
    assert entry_row[0] == "inst1"
    R_read = np.array(struct.unpack("<9d", bytes(entry_row[1]))).reshape(3, 3)
    t_read = np.array(struct.unpack("<3d", bytes(entry_row[2])))
    np.testing.assert_allclose(R_read, R)
    np.testing.assert_allclose(t_read, t)


def test_write_extrinsics_to_db_custom_method(session_with_camera) -> None:
    result = CalibResult(
        cameras={"cam_a": _solved_state("cam_a", np.eye(3), np.zeros(3))},
        points_3d=[], reprojection_errors={}, unsolved=[], pair_matches={},
    )
    calib_id = write_extrinsics_to_db(
        result, session_with_camera, "sess1", {"cam_a": "inst1"}, method="rig-anchor",
    )
    method = session_with_camera.execute(
        "SELECT method FROM extrinsic_calibrations WHERE id = ?", (calib_id,)
    ).fetchone()[0]
    assert method == "rig-anchor"


def test_write_extrinsics_to_db_skips_unsolved_camera(session_with_camera) -> None:
    unsolved = CamCalibState(
        video_id="cam_b", label="cam_b", K=np.eye(3), K_orig=np.eye(3),
        dist=np.zeros((1, 4)), fisheye=False,
    )  # R/t left None
    result = CalibResult(
        cameras={"cam_a": _solved_state("cam_a", np.eye(3), np.zeros(3)), "cam_b": unsolved},
        points_3d=[], reprojection_errors={}, unsolved=["cam_b"], pair_matches={},
    )
    calib_id = write_extrinsics_to_db(result, session_with_camera, "sess1", {"cam_a": "inst1"})
    n_entries = session_with_camera.execute(
        "SELECT COUNT(*) FROM extrinsic_entries WHERE extrinsic_calibration_id = ?", (calib_id,)
    ).fetchone()[0]
    assert n_entries == 1


def test_write_extrinsics_to_db_skips_camera_with_no_instance_mapping(session_with_camera) -> None:
    result = CalibResult(
        cameras={"cam_unmapped": _solved_state("cam_unmapped", np.eye(3), np.zeros(3))},
        points_3d=[], reprojection_errors={}, unsolved=[], pair_matches={},
    )
    calib_id = write_extrinsics_to_db(result, session_with_camera, "sess1", {})
    n_entries = session_with_camera.execute(
        "SELECT COUNT(*) FROM extrinsic_entries WHERE extrinsic_calibration_id = ?", (calib_id,)
    ).fetchone()[0]
    assert n_entries == 0


def test_write_extrinsics_to_db_falls_back_to_video_id(session_with_camera) -> None:
    """label_to_instance_id may be keyed by video_id when it differs from label."""
    result = CalibResult(
        cameras={"video123": _solved_state("cam_a", np.eye(3), np.zeros(3))},
        points_3d=[], reprojection_errors={}, unsolved=[], pair_matches={},
    )
    # Keyed by video_id, not the (different) label -- still resolves via fallback.
    calib_id = write_extrinsics_to_db(result, session_with_camera, "sess1", {"video123": "inst1"})
    n_entries = session_with_camera.execute(
        "SELECT COUNT(*) FROM extrinsic_entries WHERE extrinsic_calibration_id = ?", (calib_id,)
    ).fetchone()[0]
    assert n_entries == 1

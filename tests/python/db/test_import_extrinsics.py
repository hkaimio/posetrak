"""Tests for scripts/db/import_extrinsics.py."""

from __future__ import annotations

import math
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))  # project root

from scripts.db.import_extrinsics import ExtrinsicsImportResult, import_extrinsics, _rodrigues_to_matrix
from scripts.db.posetrak_db import create_mocap_session, create_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(tmp_path: Path) -> tuple[sqlite3.Connection, str]:
    """Create a minimal session DB and return (conn, session_id)."""
    sess_path = tmp_path / "ext_test.db"
    conn = create_session(sess_path)
    session_id = create_mocap_session(conn, location="test")
    return conn, session_id


# ---------------------------------------------------------------------------
# Rodrigues conversion (unit tests independent of DB)
# ---------------------------------------------------------------------------


def test_import_extrinsics_rodrigues_identity() -> None:
    """A zero rotation vector should produce the identity matrix."""
    R = _rodrigues_to_matrix(np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_rodrigues_rotation_about_z_90deg() -> None:
    """90-degree rotation about z-axis should match expected matrix."""
    theta = math.pi / 2.0
    rvec = np.array([0.0, 0.0, theta])
    R = _rodrigues_to_matrix(rvec)
    expected = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    np.testing.assert_allclose(R, expected, atol=1e-12)


def test_rodrigues_is_orthogonal() -> None:
    """R * R^T should be identity (rotation matrix is orthogonal)."""
    rvec = np.array([0.1, 0.2, 0.3])
    R = _rodrigues_to_matrix(rvec)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_rodrigues_det_is_one() -> None:
    """det(R) should be 1.0 for a proper rotation."""
    rvec = np.array([-0.1, 0.15, 0.25])
    R = _rodrigues_to_matrix(rvec)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# import_extrinsics — DB integration tests
# ---------------------------------------------------------------------------


def test_import_extrinsics_returns_result(
    tmp_path: Path,
    sample_calib_toml: Path,
) -> None:
    """import_extrinsics() should return an ExtrinsicsImportResult."""
    conn, session_id = _make_session(tmp_path)
    result = import_extrinsics(
        conn, session_id, sample_calib_toml,
        {"cam1": "inst-uuid-1", "cam2": "inst-uuid-2"},
    )
    conn.close()
    assert isinstance(result, ExtrinsicsImportResult)
    assert result.extrinsic_calibration_id


def test_import_extrinsics_creates_calibration_row(
    tmp_path: Path,
    sample_calib_toml: Path,
) -> None:
    """One extrinsic_calibrations row should be created."""
    conn, session_id = _make_session(tmp_path)
    result = import_extrinsics(
        conn, session_id, sample_calib_toml,
        {"cam1": "inst-uuid-1", "cam2": "inst-uuid-2"},
    )
    row = conn.execute(
        "SELECT id, session_id FROM extrinsic_calibrations WHERE id = ?",
        (result.extrinsic_calibration_id,),
    ).fetchone()
    conn.close()
    assert row is not None
    assert row["session_id"] == session_id


def test_import_extrinsics_creates_entry_rows(
    tmp_path: Path,
    sample_calib_toml: Path,
) -> None:
    """Two extrinsic_entries rows should be created for a two-camera TOML."""
    conn, session_id = _make_session(tmp_path)
    result = import_extrinsics(
        conn, session_id, sample_calib_toml,
        {"cam1": "inst-uuid-1", "cam2": "inst-uuid-2"},
    )
    count = conn.execute(
        "SELECT COUNT(*) FROM extrinsic_entries WHERE extrinsic_calibration_id = ?",
        (result.extrinsic_calibration_id,),
    ).fetchone()[0]
    conn.close()
    assert count == 2


def test_import_extrinsics_r_blob_is_float64_9(
    tmp_path: Path,
    sample_calib_toml: Path,
) -> None:
    """R blob should decode to 9 float64 values (3×3 rotation matrix, row-major)."""
    conn, session_id = _make_session(tmp_path)
    result = import_extrinsics(
        conn, session_id, sample_calib_toml,
        {"cam1": "inst-uuid-1", "cam2": "inst-uuid-2"},
    )
    row = conn.execute(
        "SELECT R FROM extrinsic_entries "
        "WHERE extrinsic_calibration_id = ? AND camera_instance_id = 'inst-uuid-1'",
        (result.extrinsic_calibration_id,),
    ).fetchone()
    conn.close()
    blob: bytes = row["R"]
    assert len(blob) == 9 * 8  # 9 × 8 bytes per float64
    values = struct.unpack("<9d", blob)
    assert len(values) == 9


def test_import_extrinsics_t_blob_is_float64_3(
    tmp_path: Path,
    sample_calib_toml: Path,
) -> None:
    """t blob should decode to 3 float64 values."""
    conn, session_id = _make_session(tmp_path)
    result = import_extrinsics(
        conn, session_id, sample_calib_toml,
        {"cam1": "inst-uuid-1", "cam2": "inst-uuid-2"},
    )
    row = conn.execute(
        "SELECT t FROM extrinsic_entries "
        "WHERE extrinsic_calibration_id = ? AND camera_instance_id = 'inst-uuid-1'",
        (result.extrinsic_calibration_id,),
    ).fetchone()
    conn.close()
    blob: bytes = row["t"]
    assert len(blob) == 3 * 8
    values = struct.unpack("<3d", blob)
    assert len(values) == 3
    # cam1 translation in sample_calib_toml is [0.5, 0.0, 2.0]
    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(0.0)
    assert values[2] == pytest.approx(2.0)


def test_import_extrinsics_per_camera_mapping(
    tmp_path: Path,
    sample_calib_toml: Path,
) -> None:
    """Per-camera mapping stores the correct instance IDs in result."""
    conn, session_id = _make_session(tmp_path)
    result = import_extrinsics(
        conn, session_id, sample_calib_toml,
        {"cam1": "inst-aaa", "cam2": "inst-bbb"},
    )
    conn.close()
    assert result.camera_instance_ids["cam1"] == "inst-aaa"
    assert result.camera_instance_ids["cam2"] == "inst-bbb"


def test_import_extrinsics_skip_unlisted(
    tmp_path: Path,
    sample_calib_toml: Path,
) -> None:
    """Cameras not in the per-camera mapping are skipped and appear in result.skipped."""
    conn, session_id = _make_session(tmp_path)
    result = import_extrinsics(
        conn, session_id, sample_calib_toml,
        {"cam1": "inst-aaa"},
    )
    conn.close()
    assert "cam1" in result.camera_instance_ids
    assert "cam2" not in result.camera_instance_ids
    assert "cam2" in result.skipped

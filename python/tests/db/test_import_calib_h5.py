# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for import_calib_h5."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from posetrak.db.db import create_registry, open_registry
from posetrak.db.import_calib_h5 import CalibH5ImportResult, import_calib_h5


pytest.importorskip("h5py")
import h5py  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_h5(
    path: Path,
    *,
    with_maps: bool = True,
    model_type: str = "standard",
    rms: float = 0.42,
    width: int = 1920,
    height: int = 1080,
    camera_name: str = "cam1",
) -> Path:
    """Write a minimal calibration HDF5 file for testing."""
    matrix = np.array([[800.0, 0, 960.0], [0, 800.0, 540.0], [0, 0, 1.0]], dtype=np.float64)
    matrix_undist = np.array([[798.0, 0, 961.0], [0, 798.0, 541.0], [0, 0, 1.0]], dtype=np.float64)
    dist = np.array([-0.1, 0.05, 0.001, -0.002], dtype=np.float64)

    with h5py.File(path, "w") as hf:
        intr = hf.create_group("intrinsics")
        intr.create_dataset("matrix", data=matrix)
        intr.create_dataset("matrix_undistorted", data=matrix_undist)
        intr.create_dataset("distortions", data=dist)
        intr.attrs["size"] = (width, height)
        intr.attrs["model_type"] = model_type
        intr.attrs["error"] = rms
        intr.attrs["camera_name"] = camera_name

        if with_maps:
            maps = hf.create_group("undistortion_maps")
            mapx = np.zeros((height, width), dtype=np.float32)
            mapy = np.zeros((height, width), dtype=np.float32)
            maps.create_dataset("mapx", data=mapx)
            maps.create_dataset("mapy", data=mapy)

    return path


@pytest.fixture()
def registry_db(tmp_path: Path):
    db_path = tmp_path / "registry.db"
    conn = create_registry(db_path)
    yield conn
    conn.close()


@pytest.fixture()
def camera_mode_id(registry_db, tmp_path):
    from posetrak.db.db import create_camera_model, create_camera_mode

    model_id = create_camera_model(registry_db, manufacturer="Test", model_name="Cam")
    return create_camera_mode(registry_db, model_id, width_px=1920, height_px=1080, nominal_fps=60.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_import_creates_row(tmp_path, registry_db, camera_mode_id):
    h5 = _make_h5(tmp_path / "calib.h5")
    result = import_calib_h5(registry_db, h5, camera_mode_id)
    assert isinstance(result, CalibH5ImportResult)
    assert len(result.intrinsics_id) == 36
    assert result.camera_name == "cam1"

    row = registry_db.execute(
        "SELECT * FROM intrinsics_calibrations WHERE id = ?", (result.intrinsics_id,)
    ).fetchone()
    assert row is not None
    assert abs(row["fx"] - 798.0) < 1e-9
    assert abs(row["fy"] - 798.0) < 1e-9
    assert abs(row["cx"] - 961.0) < 1e-9
    assert abs(row["cy"] - 541.0) < 1e-9
    assert row["distortion_model"] == "radtan"
    assert abs(row["rms_error"] - 0.42) < 1e-9
    assert row["image_width"] == 1920
    assert row["image_height"] == 1080
    assert row["camera_mode_id"] == camera_mode_id


def test_import_stores_original_matrix(tmp_path, registry_db, camera_mode_id):
    h5 = _make_h5(tmp_path / "calib.h5")
    result = import_calib_h5(registry_db, h5, camera_mode_id)
    row = registry_db.execute(
        "SELECT matrix_original FROM intrinsics_calibrations WHERE id = ?",
        (result.intrinsics_id,),
    ).fetchone()
    blob = bytes(row["matrix_original"])
    vals = struct.unpack("<9d", blob)
    assert abs(vals[0] - 800.0) < 1e-9   # fx of original matrix
    assert abs(vals[4] - 800.0) < 1e-9   # fy of original matrix


def test_import_stores_compressed_maps(tmp_path, registry_db, camera_mode_id):
    h5 = _make_h5(tmp_path / "calib.h5", with_maps=True)
    result = import_calib_h5(registry_db, h5, camera_mode_id)
    row = registry_db.execute(
        "SELECT undistort_mapx, undistort_mapy FROM intrinsics_calibrations WHERE id = ?",
        (result.intrinsics_id,),
    ).fetchone()
    # Decompress and check shape
    mapx_bytes = zlib.decompress(bytes(row["undistort_mapx"]))
    mapy_bytes = zlib.decompress(bytes(row["undistort_mapy"]))
    mapx = np.frombuffer(mapx_bytes, dtype=np.float32).reshape(1080, 1920)
    mapy = np.frombuffer(mapy_bytes, dtype=np.float32).reshape(1080, 1920)
    assert mapx.shape == (1080, 1920)
    assert mapy.shape == (1080, 1920)


def test_no_maps_skips_maps(tmp_path, registry_db, camera_mode_id):
    h5 = _make_h5(tmp_path / "calib.h5", with_maps=True)
    result = import_calib_h5(registry_db, h5, camera_mode_id, store_maps=False)
    row = registry_db.execute(
        "SELECT undistort_mapx, undistort_mapy FROM intrinsics_calibrations WHERE id = ?",
        (result.intrinsics_id,),
    ).fetchone()
    assert row["undistort_mapx"] is None
    assert row["undistort_mapy"] is None


def test_missing_maps_in_h5_is_tolerated(tmp_path, registry_db, camera_mode_id):
    h5 = _make_h5(tmp_path / "calib.h5", with_maps=False)
    result = import_calib_h5(registry_db, h5, camera_mode_id, store_maps=True)
    row = registry_db.execute(
        "SELECT undistort_mapx FROM intrinsics_calibrations WHERE id = ?",
        (result.intrinsics_id,),
    ).fetchone()
    assert row["undistort_mapx"] is None


def test_fisheye_model_type(tmp_path, registry_db, camera_mode_id):
    h5 = _make_h5(tmp_path / "calib.h5", model_type="fisheye")
    result = import_calib_h5(registry_db, h5, camera_mode_id)
    row = registry_db.execute(
        "SELECT distortion_model FROM intrinsics_calibrations WHERE id = ?",
        (result.intrinsics_id,),
    ).fetchone()
    assert row["distortion_model"] == "fisheye"


def test_file_not_found(tmp_path, registry_db, camera_mode_id):
    with pytest.raises(FileNotFoundError):
        import_calib_h5(registry_db, tmp_path / "nonexistent.h5", camera_mode_id)


def test_camera_instance_id_in_notes(tmp_path, registry_db, camera_mode_id):
    h5 = _make_h5(tmp_path / "calib.h5")
    result = import_calib_h5(
        registry_db, h5, camera_mode_id, camera_instance_id="test-uuid-123"
    )
    row = registry_db.execute(
        "SELECT notes FROM intrinsics_calibrations WHERE id = ?",
        (result.intrinsics_id,),
    ).fetchone()
    assert "test-uuid-123" in (row["notes"] or "")


def test_open_registry_migrates_v1(tmp_path):
    """open_registry should auto-migrate a v1 registry to v2."""
    import sqlite3

    db_path = tmp_path / "old_registry.db"
    # Create a v1 registry (without the new columns) by running the old schema directly
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    # Minimal v1 schema
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS camera_models (id TEXT PRIMARY KEY, manufacturer TEXT, model_name TEXT, sensor_size TEXT);
        CREATE TABLE IF NOT EXISTS camera_modes (id TEXT PRIMARY KEY, camera_model_id TEXT NOT NULL, width_px INTEGER NOT NULL DEFAULT 0, height_px INTEGER NOT NULL DEFAULT 0, nominal_fps REAL NOT NULL DEFAULT 0.0, codec TEXT, notes TEXT);
        CREATE TABLE IF NOT EXISTS camera_instances (id TEXT PRIMARY KEY, camera_model_id TEXT NOT NULL, serial_number TEXT, label TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS intrinsics_calibrations (id TEXT PRIMARY KEY, camera_mode_id TEXT NOT NULL, calibrated_at TEXT NOT NULL, calibration_tool TEXT, distortion_model TEXT NOT NULL DEFAULT 'radtan', fx REAL NOT NULL, fy REAL NOT NULL, cx REAL NOT NULL, cy REAL NOT NULL, dist_coeffs BLOB, rms_error REAL, notes TEXT);
        CREATE TABLE IF NOT EXISTS skeletons (id TEXT PRIMARY KEY, name TEXT NOT NULL, parent_id TEXT, person_label TEXT, source TEXT, yaml_content TEXT NOT NULL, created_at TEXT NOT NULL, notes TEXT);
        CREATE TABLE IF NOT EXISTS tracker_configs (id TEXT PRIMARY KEY, name TEXT NOT NULL, parent_id TEXT, created_at TEXT NOT NULL, alpha REAL, beta REAL, kappa REAL, process_noise_std REAL, measurement_noise_std REAL, outlier_threshold REAL, tracker_fps REAL, ik_max_iterations INTEGER, ik_tolerance REAL, init_position_std REAL, init_orientation_std REAL, init_joint_std REAL, init_velocity_std REAL, min_cameras_for_init INTEGER, notes TEXT);
        PRAGMA user_version = 1;
    """)
    conn.close()

    migrated = open_registry(db_path)
    from posetrak.db.db import get_schema_version, REGISTRY_SCHEMA_VERSION
    assert get_schema_version(migrated) == REGISTRY_SCHEMA_VERSION
    # Check new columns exist
    cols = {row[1] for row in migrated.execute("PRAGMA table_info(intrinsics_calibrations)")}
    assert "image_width" in cols
    assert "undistort_mapx" in cols
    migrated.close()

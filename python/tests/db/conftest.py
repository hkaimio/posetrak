# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the posetrak DB test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from posetrak.db.db import (
    add_session_camera,
    create_camera_model,
    create_camera_mode,
    create_mocap_session,
    create_registry,
    create_session,
)


@pytest.fixture()
def registry_db(tmp_path: Path):
    """Create a temporary registry database; yield the connection; close after test."""
    db_path = tmp_path / "test_registry.db"
    conn = create_registry(db_path)
    yield conn
    conn.close()


@pytest.fixture()
def session_db(tmp_path: Path):
    """Create a temporary session database; yield the connection; close after test."""
    db_path = tmp_path / "test_session.db"
    conn = create_session(db_path)
    yield conn
    conn.close()


@pytest.fixture()
def camera_mode_id(registry_db) -> str:
    """Create a camera model and mode in the registry; return the mode ID."""
    model_id = create_camera_model(
        registry_db,
        manufacturer="TestCo",
        model_name="Test Cam",
    )
    return create_camera_mode(
        registry_db,
        model_id,
        width_px=1280,
        height_px=720,
        nominal_fps=60.0,
    )


@pytest.fixture()
def sample_calib_toml(tmp_path: Path) -> Path:
    """Write a two-camera Pose2Sim calibration TOML and return its path."""
    toml_content = """\
[cam1]
name = "Camera1"
matrix = [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]
rotation = [0.1, 0.2, 0.3]
translation = [0.5, 0.0, 2.0]
distortions = [-0.1, 0.05, 0.001, -0.002]

[cam2]
name = "Camera2"
matrix = [[810.0, 0.0, 645.0], [0.0, 810.0, 362.0], [0.0, 0.0, 1.0]]
rotation = [-0.1, 0.15, 0.25]
translation = [-0.5, 0.0, 2.1]
distortions = [-0.12, 0.06, 0.0, 0.001]
"""
    path = tmp_path / "Calib_test.toml"
    path.write_text(toml_content, encoding="utf-8")
    return path


@pytest.fixture()
def session_db_path(tmp_path: Path) -> Path:
    """Path for a session database (not yet created)."""
    return tmp_path / "test_session.db"


@pytest.fixture()
def session_db_full(tmp_path: Path, registry_db, camera_mode_id):
    """Session DB with two cameras registered.

    Returns (session_conn, session_id, inst1, inst2).
    """
    from posetrak.db.import_calib_toml import import_calib_toml

    toml = tmp_path / "calib2.toml"
    toml.write_text(
        "[cam1]\nname = \"Camera1\"\n"
        "matrix = [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]\n"
        "rotation = [0.1, 0.0, 0.0]\ntranslation = [0.0, 0.0, 2.0]\n"
        "[cam2]\nname = \"Camera2\"\n"
        "matrix = [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]\n"
        "rotation = [-0.1, 0.0, 0.0]\ntranslation = [0.0, 0.0, 2.0]\n",
        encoding="utf-8",
    )
    result = import_calib_toml(registry_db, toml, camera_mode_id)
    inst1 = result.camera_instance_ids["Camera1"]
    inst2 = result.camera_instance_ids["Camera2"]
    intr1 = result.intrinsics_ids["Camera1"]
    intr2 = result.intrinsics_ids["Camera2"]

    sess_path = tmp_path / "test_full.db"
    session_conn = create_session(sess_path)
    session_id = create_mocap_session(session_conn, location="test gym")
    add_session_camera(session_conn, registry_db, session_id, inst1, camera_mode_id, intr1, label="cam1")
    add_session_camera(session_conn, registry_db, session_id, inst2, camera_mode_id, intr2, label="cam2")
    yield session_conn, session_id, inst1, inst2
    session_conn.close()


@pytest.fixture()
def sample_sync_json(tmp_path: Path) -> Path:
    """Write a minimal two-camera sync_data.json and return its path."""
    data = {
        "cam1": {
            "fps": 120.0,
            "syncpoints": [
                {"frame": 0, "timestamp": 0.0},
                {"frame": 1, "timestamp": 0.00833},
            ],
        },
        "cam2": {
            "fps": 120.0,
            "syncpoints": [
                {"frame": 0, "timestamp": 0.004},
                {"frame": 1, "timestamp": 0.01233},
            ],
        },
    }
    path = tmp_path / "sync_data.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture()
def sample_pose_dir(tmp_path: Path) -> Path:
    """Create a minimal pose directory with 3 frames for cam1 and cam2."""
    pose_dir = tmp_path / "pose"
    for cam in ["cam1", "cam2"]:
        (pose_dir / cam).mkdir(parents=True)
        for frame in range(3):
            kps = [float(i) for i in range(133 * 3)]  # 133 keypoints x 3
            data = {
                "version": 1.3,
                "people": [{"person_id": [0], "pose_keypoints_2d": kps}],
            }
            (pose_dir / cam / f"{cam}_{frame:06d}.json").write_text(
                json.dumps(data), encoding="utf-8"
            )
    return pose_dir

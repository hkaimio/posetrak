"""Shared pytest fixtures for the posetrak CLI test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from posetrak.db.db import (
    create_camera_model,
    create_camera_mode,
    create_registry,
    create_session,
    generate_id,
    open_registry,
)
from posetrak.cli.main import main


# IDs used by the seeded session fixture (detect tests reference these).
_SESSION_ID = generate_id()
_CAPTURE_ID = generate_id()
_SYNC_ID = generate_id()
_CAM_INSTANCE_ID = generate_id()
_CAM_MODEL_ID = generate_id()
_VIDEO_ID = generate_id()


@pytest.fixture()
def cli_runner() -> CliRunner:
    """Return a Click CliRunner."""
    return CliRunner()


@pytest.fixture()
def registry_db_path(tmp_path: Path) -> Path:
    """Create a temporary registry database; return its path."""
    db_path = tmp_path / "registry.db"
    conn = create_registry(db_path)
    conn.close()
    return db_path


@pytest.fixture()
def session_db_path(tmp_path: Path) -> Path:
    """Create a temporary, empty session database; return its path."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    conn.close()
    return db_path


@pytest.fixture()
def seeded_session_db_path(tmp_path: Path) -> Path:
    """Session DB seeded with one session, capture, sync config, and camera; return path."""
    db_path = tmp_path / "seeded_session.db"
    conn = create_session(db_path)

    conn.execute(
        "INSERT OR IGNORE INTO camera_models (id, manufacturer, model_name) VALUES (?,?,?)",
        (_CAM_MODEL_ID, "TestCo", "TestCam"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO camera_instances (id, camera_model_id, label) VALUES (?,?,?)",
        (_CAM_INSTANCE_ID, _CAM_MODEL_ID, "cam1"),
    )

    conn.executescript(f"""
        INSERT INTO mocap_sessions (id, recorded_at) VALUES ('{_SESSION_ID}', '2026-01-01');

        INSERT INTO captures (id, session_id, capture_number, label)
            VALUES ('{_CAPTURE_ID}', '{_SESSION_ID}', 1, 'test-capture');

        INSERT INTO sync_configs (id, shot_id, created_by)
            VALUES ('{_SYNC_ID}', '{_CAPTURE_ID}', 'test');

        INSERT INTO capture_videos
            (id, shot_id, camera_instance_id, file_path,
             first_video_frame, last_video_frame, actual_fps)
            VALUES ('{_VIDEO_ID}', '{_CAPTURE_ID}', '{_CAM_INSTANCE_ID}',
                    '/fake/video.mp4', 0, 1000, 30.0);

        INSERT INTO sync_points
            (sync_config_id, camera_instance_id, shot_video_id, video_frame, timestamp_s)
            VALUES ('{_SYNC_ID}', '{_CAM_INSTANCE_ID}', '{_VIDEO_ID}', 0, 0.0);
    """)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def registry_db(registry_db_path: Path):
    """Open the registry database; yield the connection; close after test."""
    conn = open_registry(registry_db_path)
    yield conn
    conn.close()


@pytest.fixture()
def session_db(session_db_path: Path):
    """Open the session database; yield the connection; close after test."""
    from posetrak.db.db import open_session
    conn = open_session(session_db_path)
    yield conn
    conn.close()


@pytest.fixture()
def camera_model_id(registry_db_path: Path) -> str:
    """Create a camera model in the registry; return the model ID."""
    conn = open_registry(registry_db_path)
    model_id = create_camera_model(
        conn,
        manufacturer="TestCo",
        model_name="Test Cam",
    )
    conn.close()
    return model_id


@pytest.fixture()
def camera_mode_id(registry_db_path: Path, camera_model_id: str) -> str:
    """Create a camera mode in the registry; return the mode ID."""
    conn = open_registry(registry_db_path)
    mode_id = create_camera_mode(
        conn,
        camera_model_id,
        width_px=1920,
        height_px=1080,
        nominal_fps=60.0,
    )
    conn.close()
    return mode_id


@pytest.fixture()
def capture_id() -> str:
    return _CAPTURE_ID


@pytest.fixture()
def sync_id() -> str:
    return _SYNC_ID

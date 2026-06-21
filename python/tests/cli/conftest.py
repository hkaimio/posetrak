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
    open_registry,
)
from posetrak.cli.main import main


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
    """Create a temporary session database; return its path."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
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

"""Shared pytest fixtures for the posetrak DB test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so that ``scripts.db`` is importable.
_PROJECT_ROOT = Path(__file__).parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.db.posetrak_db import create_registry, create_session  # noqa: E402


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

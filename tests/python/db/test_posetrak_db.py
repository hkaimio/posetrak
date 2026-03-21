"""Tests for scripts/db/posetrak_db.py public API."""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))  # project root

from scripts.db.posetrak_db import (
    REGISTRY_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
    create_registry,
    create_session,
    generate_id,
    get_project_root,
    get_schema_version,
    open_registry,
    open_session,
    resolve_path,
    set_project_root,
)

# ---------------------------------------------------------------------------
# generate_id
# ---------------------------------------------------------------------------

_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def test_generate_id_is_uuid4() -> None:
    """generate_id() should return a well-formed UUID v4 string."""
    uid = generate_id()
    assert _UUID4_RE.match(uid), f"Not a valid UUID-4: {uid!r}"


def test_generate_id_unique() -> None:
    """Two successive calls to generate_id() should produce different values."""
    assert generate_id() != generate_id()


# ---------------------------------------------------------------------------
# create_registry
# ---------------------------------------------------------------------------


def test_create_registry_creates_file(tmp_path: Path) -> None:
    """create_registry() should create a file at the given path."""
    db_path = tmp_path / "reg.db"
    conn = create_registry(db_path)
    conn.close()
    assert db_path.exists()


def test_create_registry_sets_schema_version(tmp_path: Path) -> None:
    """The newly created registry should have the current schema version."""
    db_path = tmp_path / "reg.db"
    conn = create_registry(db_path)
    version = get_schema_version(conn)
    conn.close()
    assert version == REGISTRY_SCHEMA_VERSION


def test_create_registry_fails_if_exists(tmp_path: Path) -> None:
    """create_registry() should raise FileExistsError if the file already exists."""
    db_path = tmp_path / "reg.db"
    conn = create_registry(db_path)
    conn.close()
    with pytest.raises(FileExistsError):
        create_registry(db_path)


# ---------------------------------------------------------------------------
# open_registry
# ---------------------------------------------------------------------------


def test_open_registry_fails_if_missing(tmp_path: Path) -> None:
    """open_registry() should raise FileNotFoundError for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        open_registry(tmp_path / "nonexistent.db")


def test_open_registry_wrong_version(tmp_path: Path) -> None:
    """open_registry() should raise ValueError if the schema version mismatches."""
    db_path = tmp_path / "reg.db"
    conn = create_registry(db_path)
    # Manually corrupt the user_version
    conn.execute("PRAGMA user_version = 999")
    conn.close()
    with pytest.raises(ValueError, match="registry"):
        open_registry(db_path)


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


def test_create_session_creates_file(tmp_path: Path) -> None:
    """create_session() should create a file at the given path."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    conn.close()
    assert db_path.exists()


def test_create_session_sets_schema_version(tmp_path: Path) -> None:
    """The newly created session DB should have the current schema version."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    version = get_schema_version(conn)
    conn.close()
    assert version == SESSION_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# open_session
# ---------------------------------------------------------------------------


def test_open_session_fails_if_missing(tmp_path: Path) -> None:
    """open_session() should raise FileNotFoundError for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        open_session(tmp_path / "nonexistent.db")


# ---------------------------------------------------------------------------
# PRAGMA foreign_keys
# ---------------------------------------------------------------------------


def test_foreign_keys_enabled(tmp_path: Path) -> None:
    """Foreign key enforcement should be ON for connections returned by create/open functions."""
    db_path = tmp_path / "reg.db"
    conn = create_registry(db_path)
    row = conn.execute("PRAGMA foreign_keys").fetchone()
    conn.close()
    assert row[0] == 1


# ---------------------------------------------------------------------------
# get_schema_version
# ---------------------------------------------------------------------------


def test_get_schema_version(tmp_path: Path) -> None:
    """get_schema_version() should return the value stored via PRAGMA user_version."""
    db_path = tmp_path / "reg.db"
    conn = create_registry(db_path)
    assert get_schema_version(conn) == REGISTRY_SCHEMA_VERSION
    conn.close()


# ---------------------------------------------------------------------------
# project_root helpers
# ---------------------------------------------------------------------------


def test_set_and_get_project_root(registry_db: sqlite3.Connection) -> None:
    """set_project_root() stores a path that get_project_root() can retrieve."""
    root = Path("/some/project/root")
    set_project_root(registry_db, root)
    retrieved = get_project_root(registry_db)
    assert retrieved == root


def test_get_project_root_returns_none_when_not_set(registry_db: sqlite3.Connection) -> None:
    """get_project_root() should return None when no project_root has been set."""
    assert get_project_root(registry_db) is None


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------


def test_resolve_path_absolute(registry_db: sqlite3.Connection) -> None:
    """resolve_path() should return absolute paths unchanged."""
    result = resolve_path("/absolute/path/file.db", registry_db)
    assert result == Path("/absolute/path/file.db")


def test_resolve_path_relative_with_root(registry_db: sqlite3.Connection) -> None:
    """resolve_path() should resolve relative paths against project_root."""
    set_project_root(registry_db, Path("/project"))
    result = resolve_path("data/session.db", registry_db)
    assert result == Path("/project/data/session.db")


def test_resolve_path_relative_no_root_raises(registry_db: sqlite3.Connection) -> None:
    """resolve_path() should raise ValueError if path is relative and root is not set."""
    with pytest.raises(ValueError, match="project_root"):
        resolve_path("relative/path.db", registry_db)


# ---------------------------------------------------------------------------
# Table existence — registry
# ---------------------------------------------------------------------------

_REGISTRY_TABLES = {
    "settings",
    "camera_models",
    "camera_modes",
    "camera_instances",
    "intrinsics_calibrations",
    "skeletons",
    "tracker_configs",
}


def test_registry_has_expected_tables(registry_db: sqlite3.Connection) -> None:
    """The registry schema should create all expected tables."""
    rows = registry_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    actual = {row["name"] for row in rows}
    for table in _REGISTRY_TABLES:
        assert table in actual, f"Missing registry table: {table!r}"


# ---------------------------------------------------------------------------
# Table existence — session
# ---------------------------------------------------------------------------

_SESSION_TABLES = {
    "mocap_sessions",
    "session_cameras",
    "extrinsic_calibrations",
    "extrinsic_entries",
    "shots",
    "shot_videos",
    "sync_configs",
    "sync_points",
    "pose_observation_sequences",
    "pose_observations",
    "tracking_runs",
    "tracking_run_persons",
    "tracking_results",
    "tracking_obs_results",
}


def test_session_has_expected_tables(session_db: sqlite3.Connection) -> None:
    """The session schema should create all expected tables."""
    rows = session_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    actual = {row["name"] for row in rows}
    for table in _SESSION_TABLES:
        assert table in actual, f"Missing session table: {table!r}"

"""posetrak_db.py — Core database access layer for the posetrak SQLite registry and session databases.

This module handles:
- Creating and opening registry databases (shared project-wide metadata: cameras, skeletons, configs).
- Creating and opening per-session databases (mocap sessions, shots, tracking results).
- Schema versioning via SQLite PRAGMA user_version.
- Utility helpers for project-root-relative path resolution.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Schema version constants
# ---------------------------------------------------------------------------

REGISTRY_SCHEMA_VERSION: Final[int] = 1
SESSION_SCHEMA_VERSION: Final[int] = 1

#: Default registry database location — shared across all projects on the machine.
DEFAULT_REGISTRY_PATH: Final[Path] = Path.home() / ".posetrak" / "registry.db"

# ---------------------------------------------------------------------------
# SQL file paths (resolved relative to this source file)
# ---------------------------------------------------------------------------

_REGISTRY_SCHEMA_SQL: Final[Path] = Path(__file__).parents[2] / "db" / "registry_schema.sql"
_SESSION_SCHEMA_SQL: Final[Path] = Path(__file__).parents[2] / "db" / "session_schema.sql"


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def generate_id() -> str:
    """Return a new random UUID v4 string.

    Returns
    -------
    str
        A lowercase UUID-4 string, e.g. ``"550e8400-e29b-41d4-a716-446655440000"``.
    """
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Schema version helpers
# ---------------------------------------------------------------------------


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Read the schema version stored in PRAGMA user_version.

    Parameters
    ----------
    conn:
        An open SQLite connection.

    Returns
    -------
    int
        The current user_version (0 if never set).
    """
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0])


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Set PRAGMA user_version to *version*.

    Parameters
    ----------
    conn:
        An open SQLite connection.
    version:
        The version integer to store.
    """
    # PRAGMA user_version cannot use parameter binding; version is always an
    # internal integer constant so safe to interpolate directly.
    conn.execute(f"PRAGMA user_version = {version}")


def _apply_schema(conn: sqlite3.Connection, sql_path: Path, version: int) -> None:
    """Execute the SQL schema file against *conn* and record *version*.

    Parameters
    ----------
    conn:
        An open SQLite connection (must not have an active transaction).
    sql_path:
        Path to the ``.sql`` file containing the CREATE TABLE statements.
    version:
        Schema version to record via PRAGMA user_version after applying.

    Raises
    ------
    FileNotFoundError
        If *sql_path* does not exist.
    """
    sql = sql_path.read_text(encoding="utf-8")
    conn.executescript(sql)
    _set_schema_version(conn, version)
    conn.commit()


# ---------------------------------------------------------------------------
# Low-level connection helpers
# ---------------------------------------------------------------------------


def _connect(path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with project-standard settings.

    Sets ``row_factory = sqlite3.Row``, enables foreign-key enforcement,
    and switches the journal mode to WAL for better concurrency.

    Parameters
    ----------
    path:
        Filesystem path of the SQLite database file.

    Returns
    -------
    sqlite3.Connection
        A configured open connection.
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _check_schema_version(
    conn: sqlite3.Connection,
    expected: int,
    kind: str,
) -> None:
    """Assert that the database has the expected schema version.

    Parameters
    ----------
    conn:
        An open SQLite connection.
    expected:
        The required schema version number.
    kind:
        Human-readable label for the database type (e.g. ``"registry"``).

    Raises
    ------
    ValueError
        If the actual user_version does not match *expected*.
    """
    actual = get_schema_version(conn)
    if actual != expected:
        raise ValueError(
            f"{kind} database schema version mismatch: "
            f"expected {expected}, got {actual}. "
            "The database may have been created by a different version of posetrak."
        )


# ---------------------------------------------------------------------------
# Registry database
# ---------------------------------------------------------------------------


def create_registry(path: Path) -> sqlite3.Connection:
    """Create a new registry database at *path* and return an open connection.

    Parameters
    ----------
    path:
        Destination path for the new ``.db`` file. Parent directories are
        created automatically.

    Returns
    -------
    sqlite3.Connection
        An open connection to the newly created registry database.

    Raises
    ------
    FileExistsError
        If a file already exists at *path*.
    """
    if path.exists():
        raise FileExistsError(f"Registry database already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(path)
    _apply_schema(conn, _REGISTRY_SCHEMA_SQL, REGISTRY_SCHEMA_VERSION)
    return conn


def open_registry(path: Path) -> sqlite3.Connection:
    """Open an existing registry database and verify its schema version.

    Parameters
    ----------
    path:
        Path to an existing registry ``.db`` file.

    Returns
    -------
    sqlite3.Connection
        An open connection to the registry database.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the schema version does not match :data:`REGISTRY_SCHEMA_VERSION`.
    """
    if not path.exists():
        raise FileNotFoundError(f"Registry database not found: {path}")
    conn = _connect(path)
    _check_schema_version(conn, REGISTRY_SCHEMA_VERSION, "registry")
    return conn


# ---------------------------------------------------------------------------
# Session database
# ---------------------------------------------------------------------------


def create_session(path: Path) -> sqlite3.Connection:
    """Create a new session database at *path* and return an open connection.

    Parameters
    ----------
    path:
        Destination path for the new session ``.db`` file. Parent directories
        are created automatically.

    Returns
    -------
    sqlite3.Connection
        An open connection to the newly created session database.

    Raises
    ------
    FileExistsError
        If a file already exists at *path*.
    """
    if path.exists():
        raise FileExistsError(f"Session database already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(path)
    _apply_schema(conn, _SESSION_SCHEMA_SQL, SESSION_SCHEMA_VERSION)
    return conn


def open_session(path: Path) -> sqlite3.Connection:
    """Open an existing session database and verify its schema version.

    Parameters
    ----------
    path:
        Path to an existing session ``.db`` file.

    Returns
    -------
    sqlite3.Connection
        An open connection to the session database.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the schema version does not match :data:`SESSION_SCHEMA_VERSION`.
    """
    if not path.exists():
        raise FileNotFoundError(f"Session database not found: {path}")
    conn = _connect(path)
    _check_schema_version(conn, SESSION_SCHEMA_VERSION, "session")
    return conn


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------


def get_project_root(registry: sqlite3.Connection) -> Path | None:
    """Read the ``project_root`` setting from the registry, if set.

    Parameters
    ----------
    registry:
        An open connection to a registry database.

    Returns
    -------
    Path | None
        The project root path, or ``None`` if the setting has not been stored.
    """
    row = registry.execute(
        "SELECT value FROM settings WHERE key = 'project_root'"
    ).fetchone()
    if row is None:
        return None
    return Path(row["value"])


def set_project_root(registry: sqlite3.Connection, root: Path) -> None:
    """Store or update the ``project_root`` setting in the registry.

    Parameters
    ----------
    registry:
        An open connection to a registry database.
    root:
        The project root directory path to store.
    """
    registry.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES ('project_root', ?)",
        (str(root),),
    )
    registry.commit()


# ---------------------------------------------------------------------------
# Camera model / mode management
# ---------------------------------------------------------------------------


def create_camera_model(
    registry: sqlite3.Connection,
    *,
    manufacturer: str = "",
    model_name: str = "",
    sensor_size: str | None = None,
    notes: str | None = None,
) -> str:
    """Insert a new camera_models row and return its ID.

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.
    manufacturer:
        Camera manufacturer name (e.g. ``"GoPro"``).
    model_name:
        Camera model name (e.g. ``"Hero 10 Black"``).
    sensor_size:
        Optional sensor size descriptor (e.g. ``"1/2.3\""``).
    notes:
        Optional free-text notes.

    Returns
    -------
    str
        The UUID of the newly created row.
    """
    model_id = generate_id()
    with registry:
        registry.execute(
            "INSERT INTO camera_models (id, manufacturer, model_name, sensor_size) "
            "VALUES (?, ?, ?, ?)",
            (model_id, manufacturer, model_name, sensor_size),
        )
    return model_id


def create_camera_mode(
    registry: sqlite3.Connection,
    camera_model_id: str,
    *,
    width_px: int = 0,
    height_px: int = 0,
    nominal_fps: float = 0.0,
    codec: str | None = None,
    notes: str | None = None,
) -> str:
    """Insert a new camera_modes row and return its ID.

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.
    camera_model_id:
        ID of the parent ``camera_models`` row.
    width_px:
        Image width in pixels (0 = unknown).
    height_px:
        Image height in pixels (0 = unknown).
    nominal_fps:
        Nominal frame rate in frames per second (0.0 = unknown).
    codec:
        Optional codec identifier string (e.g. ``"h264"``).
    notes:
        Optional free-text notes.

    Returns
    -------
    str
        The UUID of the newly created row.

    Raises
    ------
    sqlite3.IntegrityError
        If *camera_model_id* does not refer to an existing camera_models row.
    """
    mode_id = generate_id()
    with registry:
        registry.execute(
            "INSERT INTO camera_modes "
            "(id, camera_model_id, width_px, height_px, nominal_fps, codec, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mode_id, camera_model_id, width_px, height_px, nominal_fps, codec, notes),
        )
    return mode_id


def list_camera_models(registry: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all rows from the camera_models table.

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.

    Returns
    -------
    list[sqlite3.Row]
        All camera model rows, ordered by rowid.
    """
    return registry.execute("SELECT * FROM camera_models ORDER BY rowid").fetchall()


def list_camera_modes(
    registry: sqlite3.Connection,
    camera_model_id: str | None = None,
) -> list[sqlite3.Row]:
    """Return rows from the camera_modes table.

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.
    camera_model_id:
        If provided, return only modes belonging to this camera model.

    Returns
    -------
    list[sqlite3.Row]
        Matching camera mode rows, ordered by rowid.
    """
    if camera_model_id is not None:
        return registry.execute(
            "SELECT * FROM camera_modes WHERE camera_model_id = ? ORDER BY rowid",
            (camera_model_id,),
        ).fetchall()
    return registry.execute(
        "SELECT * FROM camera_modes ORDER BY rowid"
    ).fetchall()


# ---------------------------------------------------------------------------
# Session database management
# ---------------------------------------------------------------------------


def create_mocap_session(
    session: sqlite3.Connection,
    *,
    recorded_at: str | None = None,
    location: str = "",
    notes: str = "",
) -> str:
    """Insert a mocap_sessions row and return its ID.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    recorded_at:
        ISO-format date/datetime string for when the session was recorded.
        Defaults to today's date (``datetime.date.today().isoformat()``).
    location:
        Optional human-readable description of the recording location.
    notes:
        Optional free-text notes.

    Returns
    -------
    str
        UUID of the newly created ``mocap_sessions`` row.
    """
    import datetime as _dt
    if recorded_at is None:
        recorded_at = _dt.date.today().isoformat()
    session_id = generate_id()
    with session:
        session.execute(
            "INSERT INTO mocap_sessions (id, recorded_at, location, notes) "
            "VALUES (?, ?, ?, ?)",
            (session_id, recorded_at, location, notes),
        )
    return session_id


def add_session_camera(
    session: sqlite3.Connection,
    session_id: str,
    camera_instance_id: str,
    camera_mode_id: str,
    intrinsics_calibration_id: str,
    *,
    label: str = "",
) -> None:
    """Insert a session_cameras row linking a camera to a session.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    session_id:
        ID of the parent ``mocap_sessions`` row.
    camera_instance_id:
        Registry ``camera_instances.id`` for the camera.
    camera_mode_id:
        Registry ``camera_modes.id`` describing the capture mode.
    intrinsics_calibration_id:
        Registry ``intrinsics_calibrations.id`` used for this camera.
    label:
        Optional short label for the camera within this session (e.g. ``"cam1"``).

    Raises
    ------
    sqlite3.IntegrityError
        If the (session_id, camera_instance_id) pair already exists.
    """
    with session:
        session.execute(
            "INSERT INTO session_cameras "
            "(session_id, camera_instance_id, camera_mode_id, "
            "intrinsics_calibration_id, label) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, camera_instance_id, camera_mode_id,
             intrinsics_calibration_id, label),
        )


def create_shot(
    session: sqlite3.Connection,
    session_id: str,
    extrinsic_calibration_id: str,
    *,
    shot_number: int | None = None,
    label: str = "",
    notes: str = "",
) -> str:
    """Insert a shots row and return its ID.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    session_id:
        ID of the parent ``mocap_sessions`` row.
    extrinsic_calibration_id:
        ID of the ``extrinsic_calibrations`` row used for this shot.
    shot_number:
        Explicit shot number. If ``None``, auto-increments from the highest
        existing ``shot_number`` within this session (starting at 1).
    label:
        Optional short label for the shot.
    notes:
        Optional free-text notes.

    Returns
    -------
    str
        UUID of the newly created ``shots`` row.
    """
    if shot_number is None:
        row = session.execute(
            "SELECT MAX(shot_number) FROM shots WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        max_num = row[0]
        shot_number = 1 if max_num is None else max_num + 1

    shot_id = generate_id()
    with session:
        session.execute(
            "INSERT INTO shots "
            "(id, session_id, extrinsic_calibration_id, shot_number, label, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (shot_id, session_id, extrinsic_calibration_id, shot_number, label, notes),
        )
    return shot_id


def add_shot_video(
    session: sqlite3.Connection,
    shot_id: str,
    camera_instance_id: str,
    file_path: str,
    first_frame: int,
    last_frame: int,
    fps: float,
) -> str:
    """Insert a shot_videos row and return its ID.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    shot_id:
        ID of the parent ``shots`` row.
    camera_instance_id:
        Registry ``camera_instances.id`` for the camera that recorded this video.
    file_path:
        Path string to the video file (stored as-is, not validated).
    first_frame:
        Index of the first video frame in the file (0-based).
    last_frame:
        Index of the last video frame in the file (inclusive).
    fps:
        Actual recorded frames per second.

    Returns
    -------
    str
        UUID of the newly created ``shot_videos`` row.
    """
    video_id = generate_id()
    with session:
        session.execute(
            "INSERT INTO shot_videos "
            "(id, shot_id, camera_instance_id, file_path, "
            "first_video_frame, last_video_frame, actual_fps) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (video_id, shot_id, camera_instance_id, file_path,
             first_frame, last_frame, fps),
        )
    return video_id


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_path(path_str: str, registry: sqlite3.Connection) -> Path:
    """Resolve a path string relative to the registry ``project_root`` if necessary.

    If *path_str* is absolute, it is returned as-is. If relative, it is
    resolved against the ``project_root`` setting in *registry*.

    Parameters
    ----------
    path_str:
        The path string to resolve (may be absolute or relative).
    registry:
        An open connection to a registry database (used to look up
        ``project_root`` for relative paths).

    Returns
    -------
    Path
        The resolved absolute (or relative-to-cwd) path.

    Raises
    ------
    ValueError
        If *path_str* is relative and ``project_root`` has not been set in
        the registry.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    project_root = get_project_root(registry)
    if project_root is None:
        raise ValueError(
            f"Cannot resolve relative path '{path_str}': "
            "'project_root' has not been set in the registry. "
            "Use set_project_root() first."
        )
    return project_root / p

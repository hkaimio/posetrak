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

REGISTRY_SCHEMA_VERSION: Final[int] = 3
SESSION_SCHEMA_VERSION: Final[int] = 6

#: Default registry database location — shared across all projects on the machine.
DEFAULT_REGISTRY_PATH: Final[Path] = Path.home() / ".posetrak" / "registry.db"

# ---------------------------------------------------------------------------
# SQL file paths (resolved relative to this source file)
# ---------------------------------------------------------------------------

_DB_DIR: Final[Path] = Path(__file__).parents[3] / "db"
_REGISTRY_SCHEMA_SQL: Final[Path] = _DB_DIR / "registry_schema.sql"
_SESSION_SCHEMA_SQL: Final[Path] = _DB_DIR / "session_schema.sql"


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
    actual = get_schema_version(conn)
    if actual == 1:
        _migrate_registry_v1_to_v2(conn)
        actual = 2
    if actual == 2:
        _migrate_registry_v2_to_v3(conn)
    _check_schema_version(conn, REGISTRY_SCHEMA_VERSION, "registry")
    return conn


# ---------------------------------------------------------------------------
# Session database
# ---------------------------------------------------------------------------


def create_session(path: Path) -> sqlite3.Connection:
    """Create a new session database at *path* and return an open connection.

    The session database is self-contained: it embeds a full copy of the
    registry tables (camera_models, camera_modes, camera_instances,
    intrinsics_calibrations, skeletons, tracker_configs) so the DB remains
    usable even when the registry file is not accessible.

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
    registry_sql = _REGISTRY_SCHEMA_SQL.read_text(encoding="utf-8")
    session_sql = _SESSION_SCHEMA_SQL.read_text(encoding="utf-8")
    conn.executescript(registry_sql + "\n" + session_sql)
    _set_schema_version(conn, SESSION_SCHEMA_VERSION)
    conn.commit()
    return conn


def _migrate_registry_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 1 to 2.

    v2 adds image_width, image_height, matrix_original, undistort_mapx,
    undistort_mapy columns to intrinsics_calibrations. All nullable.
    """
    sql = (_DB_DIR / "migrations" / "001_registry_intrinsics.sql").read_text(encoding="utf-8")
    conn.executescript(sql)


def _migrate_registry_v2_to_v3(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 2 to 3.

    v3 adds process_noise_vel_std to tracker_configs. Existing rows receive NULL
    (fall back to using process_noise_std for both position and velocity blocks).
    """
    conn.executescript("""
        BEGIN;
        ALTER TABLE tracker_configs ADD COLUMN process_noise_vel_std REAL;
        PRAGMA user_version = 3;
        COMMIT;
    """)


def _migrate_session_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 1 to 2.

    v2 changes: sync_points PRIMARY KEY changed from (sync_config_id, camera_instance_id)
    to (sync_config_id, camera_instance_id, video_frame) to allow multiple sync points
    per camera per sync config.

    SQLite does not support ALTER TABLE to change a primary key, so we recreate the table.
    Existing single-anchor rows are preserved.
    """
    conn.executescript(
        """
        BEGIN;
        CREATE TABLE sync_points_v2 (
            sync_config_id     TEXT    NOT NULL REFERENCES sync_configs(id),
            camera_instance_id TEXT    NOT NULL,
            shot_video_id      TEXT    NOT NULL REFERENCES shot_videos(id),
            video_frame        INTEGER NOT NULL,
            timestamp_s        REAL    NOT NULL,
            PRIMARY KEY (sync_config_id, camera_instance_id, video_frame)
        );
        INSERT INTO sync_points_v2 SELECT * FROM sync_points;
        DROP TABLE sync_points;
        ALTER TABLE sync_points_v2 RENAME TO sync_points;
        PRAGMA user_version = 2;
        COMMIT;
    """
    )


def _migrate_session_v2_to_v3(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 2 to 3.

    v3 makes shots.extrinsic_calibration_id nullable so shots can be created
    before extrinsics are imported (e.g. during YAML project import).
    """
    sql = (_DB_DIR / "migrations" / "002_session_nullable_extrinsics.sql").read_text(
        encoding="utf-8"
    )
    conn.executescript(sql)


def _migrate_session_v3_to_v4(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 3 to 4.

    v4 adds pixels_are_undistorted to pose_observation_sequences.
    Existing rows default to 1 (undistorted) because all prior captures
    used pre-undistorted video.
    """
    sql = (_DB_DIR / "migrations" / "003_session_pixels_are_undistorted.sql").read_text(
        encoding="utf-8"
    )
    conn.executescript(sql)


def _migrate_session_v4_to_v5(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 4 to 5.

    v5 adds nis_value and nis_dof columns to tracking_results for UKF
    consistency monitoring. Existing rows receive NULL.
    """
    sql = (_DB_DIR / "migrations" / "004_tracking_results_nis.sql").read_text(encoding="utf-8")
    conn.executescript(sql)


def _migrate_session_v5_to_v6(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 5 to 6.

    v6 adds process_noise_vel_std to tracker_configs so the velocity DOF
    process noise can be tuned independently from the position/angle DOF noise.
    Existing rows receive NULL (fall back to process_noise_std for both blocks).
    """
    sql = (_DB_DIR / "migrations" / "005_tracker_configs_vel_noise.sql").read_text(
        encoding="utf-8"
    )
    conn.executescript(sql)


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
    actual = get_schema_version(conn)
    if actual == 1:
        _migrate_session_v1_to_v2(conn)
        actual = 2
    if actual == 2:
        _migrate_session_v2_to_v3(conn)
        actual = 3
    if actual == 3:
        _migrate_session_v3_to_v4(conn)
        actual = 4
    if actual == 4:
        _migrate_session_v4_to_v5(conn)
        actual = 5
    if actual == 5:
        _migrate_session_v5_to_v6(conn)
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


def create_camera_instance(
    registry: sqlite3.Connection,
    camera_model_id: str,
    *,
    label: str,
    serial_number: str | None = None,
) -> str:
    """Insert a new camera_instances row and return its ID.

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.
    camera_model_id:
        ID of the parent ``camera_models`` row.
    label:
        Short human-readable label for this camera unit (e.g. ``"cam1"``).
        Used for lookup by label in session YAML import.
    serial_number:
        Optional manufacturer serial number.

    Returns
    -------
    str
        The UUID of the newly created row.

    Raises
    ------
    sqlite3.IntegrityError
        If *camera_model_id* does not exist.
    """
    instance_id = generate_id()
    with registry:
        registry.execute(
            "INSERT INTO camera_instances (id, camera_model_id, serial_number, label) "
            "VALUES (?, ?, ?, ?)",
            (instance_id, camera_model_id, serial_number, label),
        )
    return instance_id


def list_camera_instances(
    registry: sqlite3.Connection,
    camera_model_id: str | None = None,
) -> list[sqlite3.Row]:
    """Return rows from the camera_instances table.

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.
    camera_model_id:
        If provided, return only instances belonging to this camera model.

    Returns
    -------
    list[sqlite3.Row]
        Matching camera instance rows, ordered by rowid.
    """
    if camera_model_id is not None:
        return registry.execute(
            "SELECT * FROM camera_instances WHERE camera_model_id = ? ORDER BY rowid",
            (camera_model_id,),
        ).fetchall()
    return registry.execute("SELECT * FROM camera_instances ORDER BY rowid").fetchall()


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
# Cross-database row copy helper
# ---------------------------------------------------------------------------


def _copy_rows_if_missing(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    table: str,
    ids: list[str],
) -> None:
    """Copy rows from *src* to *dst* by ID using INSERT OR IGNORE.

    Allows idempotent copy: if the row already exists in *dst* it is silently
    skipped.  Raises ValueError if a row is not found in *src*.

    Parameters
    ----------
    src:
        Source database connection (rows are read from here).
    dst:
        Destination database connection (rows are inserted here).
    table:
        Name of the table to copy rows from/to.
    ids:
        List of primary key values (``id`` column) to copy.  Duplicates are
        automatically deduplicated while preserving order.

    Raises
    ------
    ValueError
        If any ``id`` in *ids* is not found in *src*.
    """
    for row_id in dict.fromkeys(ids):  # deduplicate, preserve order
        row = src.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()
        if row is None:
            raise ValueError(f"Row '{row_id}' not found in {table} of source database")
        dst.execute(
            f"INSERT OR IGNORE INTO {table} VALUES ({', '.join('?' * len(row))})",
            tuple(row),
        )


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
    registry: sqlite3.Connection,
    session_id: str,
    camera_instance_id: str,
    camera_mode_id: str,
    intrinsics_calibration_id: str,
    *,
    label: str = "",
) -> None:
    """Insert a session_cameras row linking a camera to a session.

    Registry rows for the camera model, mode, instance, and intrinsics are
    copied into the session database so the session is self-contained.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    registry:
        Open connection to the posetrak registry database.  Used to look up
        and copy camera rows into *session*.
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
    ValueError
        If *camera_instance_id* or *camera_mode_id* is not found in *registry*.
    sqlite3.IntegrityError
        If the (session_id, camera_instance_id) pair already exists.
    """
    instance_row = registry.execute(
        "SELECT camera_model_id FROM camera_instances WHERE id = ?",
        (camera_instance_id,),
    ).fetchone()
    if instance_row is None:
        raise ValueError(
            f"camera_instance '{camera_instance_id}' not found in registry"
        )

    mode_row = registry.execute(
        "SELECT camera_model_id FROM camera_modes WHERE id = ?",
        (camera_mode_id,),
    ).fetchone()
    if mode_row is None:
        raise ValueError(
            f"camera_mode '{camera_mode_id}' not found in registry"
        )

    camera_model_id = instance_row["camera_model_id"]

    with session:
        # Copy dependency chain: camera_models → camera_modes/instances → intrinsics
        _copy_rows_if_missing(registry, session, "camera_models", [camera_model_id])
        _copy_rows_if_missing(registry, session, "camera_modes", [camera_mode_id])
        _copy_rows_if_missing(registry, session, "camera_instances", [camera_instance_id])
        _copy_rows_if_missing(
            registry, session, "intrinsics_calibrations", [intrinsics_calibration_id]
        )
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
    extrinsic_calibration_id: str | None = None,
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


def set_shot_extrinsics(
    session: sqlite3.Connection,
    shot_id: str,
    extrinsic_calibration_id: str,
) -> None:
    """Set or update the extrinsic_calibration_id on an existing shot.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    shot_id:
        ID of the ``shots`` row to update.
    extrinsic_calibration_id:
        ID of the ``extrinsic_calibrations`` row to link.
    """
    with session:
        session.execute(
            "UPDATE shots SET extrinsic_calibration_id = ? WHERE id = ?",
            (extrinsic_calibration_id, shot_id),
        )


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


def resolve_id_prefix(conn: sqlite3.Connection, table: str, prefix: str) -> str:
    """Resolve a UUID prefix to a full ID, raising if ambiguous or not found.

    Parameters
    ----------
    conn:
        An open SQLite connection to a registry or session database.
    table:
        Table name to search (must have an ``id`` TEXT PRIMARY KEY column).
    prefix:
        Full UUID or a unique prefix thereof.

    Returns
    -------
    str
        The full UUID matching *prefix*.

    Raises
    ------
    ValueError
        If zero or more than one row matches *prefix*.
    """
    rows = conn.execute(
        f"SELECT id FROM {table} WHERE id LIKE ? || '%'", (prefix,)  # noqa: S608
    ).fetchall()
    if len(rows) == 0:
        raise ValueError(f"No {table} record found with id prefix '{prefix}'")
    if len(rows) > 1:
        matches = ", ".join(r[0] for r in rows)
        raise ValueError(
            f"Ambiguous prefix '{prefix}' matches {len(rows)} {table} records: {matches}"
        )
    return rows[0][0]

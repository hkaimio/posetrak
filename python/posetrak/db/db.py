# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""posetrak_db.py — Core database access layer for the posetrak SQLite registry and session databases.

This module handles:
- Creating and opening registry databases (shared project-wide metadata: cameras, skeletons, configs).
- Creating and opening per-session databases (mocap sessions, captures, tracking results).
- Schema versioning via SQLite PRAGMA user_version.
- Utility helpers for project-root-relative path resolution.
"""

from __future__ import annotations

import importlib.resources
import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Final

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version constants
# ---------------------------------------------------------------------------

REGISTRY_SCHEMA_VERSION: Final[int] = 19

#: Seconds a connection waits for another writer's lock before failing. The C++
#: tracker's connections use the same value.
BUSY_TIMEOUT_S: Final[float] = 60.0
SESSION_SCHEMA_VERSION: Final[int] = 70

#: Default registry database location — shared across all projects on the machine.
DEFAULT_REGISTRY_PATH: Final[Path] = Path.home() / ".posetrak" / "registry.db"

# ---------------------------------------------------------------------------
# SQL file paths
# ---------------------------------------------------------------------------

# Package data (posetrak/db/sql/, declared in pyproject.toml's
# [tool.setuptools.package-data]), not a path resolved relative to this
# source file's location on disk -- the previous Path(__file__).parents[3]
# / "db" walk assumed a full git checkout with a repo-root db/ directory
# sibling to python/, which isn't true once posetrak is installed by any
# means that doesn't preserve that exact layout (found via the packaging
# prototype's bootstrap folder, 2026-08-23: a snapshot of just the python/
# tree has no such sibling, so every schema/migration load failed with
# FileNotFoundError). importlib.resources.files() is the same mechanism
# manage_skeleton.py already uses for its own bundled YAML data.
_DB_DIR: Final = importlib.resources.files("posetrak.db.sql")
_REGISTRY_SCHEMA_SQL: Final = _DB_DIR / "registry_schema.sql"
_SESSION_SCHEMA_SQL: Final = _DB_DIR / "session_schema.sql"


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


def _connect(path: Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open a SQLite connection with project-standard settings.

    Sets ``row_factory = sqlite3.Row``, enables foreign-key enforcement,
    and switches the journal mode to WAL for better concurrency.

    Parameters
    ----------
    path:
        Filesystem path of the SQLite database file.
    check_same_thread:
        Passed to :func:`sqlite3.connect`; False for a connection handed
        between threads.

    Returns
    -------
    sqlite3.Connection
        A configured open connection.
    """
    # Several processes write one session database (GUI, CLI, tracker); a writer
    # that meets another waits for it instead of failing with "database is locked".
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_S, check_same_thread=check_same_thread)
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


def _discard_partial_db(conn: sqlite3.Connection, path: Path) -> None:
    """Close *conn* and delete *path* plus any WAL/SHM sidecar files.

    ``sqlite3.connect()`` creates the file at *path* immediately, before a
    single statement has run against it. If schema application or seeding
    then fails partway through, the caller (create_registry()/
    create_session()) would otherwise leave that empty, schema-version-0
    file behind. A later open_or_create_registry() -- or a plain retry --
    sees the leftover file and treats it as an existing (but broken)
    database instead of creating a fresh one, surfacing a confusing
    "expected N, got 0" schema-mismatch error that has nothing to do with
    the actual failure.

    Found via the installer prototype's Windows Sandbox testing
    (2026-08-23): a since-fixed bug in schema-file resolution made
    create_registry() fail this way, and every subsequent launch then hit
    the schema-mismatch error above instead of the original one, because
    the wrecked ~/.posetrak/registry.db from the first failed attempt was
    still sitting there.
    """
    conn.close()
    for suffix in ("", "-wal", "-shm"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)


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
    try:
        _apply_schema(conn, _REGISTRY_SCHEMA_SQL, REGISTRY_SCHEMA_VERSION)
        seed_bundled_defaults(conn)
    except Exception:
        _discard_partial_db(conn, path)
        raise
    return conn


def seed_bundled_defaults(conn: sqlite3.Connection) -> None:
    """Seed the bundled default skeletons and baseline tracker config.

    create_registry() always calls this. create_session() deliberately
    does *not* -- it's also used internally for exports/round-trips that
    need a genuinely empty session (see e.g. trial_export.py), so callers
    that create a session a person will actually start using (the setup
    wizard, `session create`, `session add-camera`, ...) call this
    themselves right after create_session() instead. Idempotent
    (INSERT OR IGNORE, keyed by content hash / a fixed ID), so safe to
    call on an already-seeded connection.
    """
    from posetrak.db.manage_skeleton import seed_default_skeletons
    seed_default_skeletons(conn)
    from posetrak.db.manage_config import seed_baseline_tracker_config
    seed_baseline_tracker_config(conn)


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
        actual = 3
    if actual == 3:
        _migrate_registry_v3_to_v4(conn)
        actual = 4
    if actual == 4:
        _migrate_registry_v4_to_v5(conn)
        actual = 5
    if actual == 5:
        _migrate_registry_v5_to_v6(conn)
        actual = 6
    if actual == 6:
        _migrate_registry_v6_to_v7(conn)
        actual = 7
    if actual == 7:
        _migrate_registry_v7_to_v8(conn)
        actual = 8
    if actual == 8:
        _migrate_registry_v8_to_v9(conn)
        actual = 9
    if actual == 9:
        _migrate_registry_v9_to_v10(conn)
        actual = 10
    if actual == 10:
        _migrate_registry_v10_to_v11(conn)
        actual = 11
    if actual == 11:
        _migrate_registry_v11_to_v12(conn)
        actual = 12
    if actual == 12:
        _migrate_registry_v12_to_v13(conn)
        actual = 13
    if actual == 13:
        _migrate_registry_v13_to_v14(conn)
        actual = 14
    if actual == 14:
        _migrate_registry_v14_to_v15(conn)
        actual = 15
    if actual == 15:
        _migrate_registry_v15_to_v16(conn)
        actual = 16
    if actual == 16:
        _migrate_registry_v16_to_v17(conn)
        actual = 17
    if actual == 17:
        _migrate_registry_v17_to_v18(conn)
        actual = 18
    if actual == 18:
        _migrate_registry_v18_to_v19(conn)
    _check_schema_version(conn, REGISTRY_SCHEMA_VERSION, "registry")
    return conn


# ---------------------------------------------------------------------------
# Session database
# ---------------------------------------------------------------------------


def create_session(path: Path) -> sqlite3.Connection:
    """Create a new session database at *path* and return an open connection.

    The session database is self-contained: it embeds a full copy of the
    registry tables (camera_models, camera_modes, camera_instances,
    intrinsics_calibrations, skeletons, tracker_configs,
    marker_body_definitions) so the DB remains usable even when the
    registry file is not accessible.

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
    try:
        registry_sql = _REGISTRY_SCHEMA_SQL.read_text(encoding="utf-8")
        session_sql = _SESSION_SCHEMA_SQL.read_text(encoding="utf-8")
        conn.executescript(registry_sql + "\n" + session_sql)
        _set_schema_version(conn, SESSION_SCHEMA_VERSION)
        conn.commit()
    except Exception:
        _discard_partial_db(conn, path)
        raise
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


def _migrate_registry_v3_to_v4(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 3 to 4.

    v4 adds velocity_half_life_s to tracker_configs for exponential velocity
    damping. NULL = no damping (backward compatible).
    """
    conn.executescript("""
        BEGIN;
        ALTER TABLE tracker_configs ADD COLUMN velocity_half_life_s REAL;
        PRAGMA user_version = 4;
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


def _migrate_session_v6_to_v7(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 6 to 7.

    v7 adds velocity_half_life_s to tracker_configs for exponential velocity
    damping. NULL = no damping (backward compatible).
    """
    sql = (_DB_DIR / "migrations" / "006_tracker_configs_vel_halflife.sql").read_text(
        encoding="utf-8"
    )
    conn.executescript(sql)


def _migrate_session_v7_to_v8(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 7 to 8.

    v8 adds person_detections, person_tracks, and frame_cache_entries tables
    for the capture pipeline setup application.
    """
    sql = (_DB_DIR / "migrations" / "007_phase2_detection_tables.sql").read_text(
        encoding="utf-8"
    )
    conn.executescript(sql)


def _migrate_session_v8_to_v9(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 8 to 9.

    v9 adds detection_runs and detection_keypoints tables for the integrated
    pose extraction pipeline. person_detections and person_tracks are
    recreated with detection_run_id added to their primary keys. Adds
    detection_run_id to frame_cache_entries and pose_observation_sequences,
    and noise_scale to pose_observations.
    """
    sql = (_DB_DIR / "migrations" / "008_detection_runs.sql").read_text(encoding="utf-8")
    conn.executescript(sql)


def _migrate_registry_v4_to_v5(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 4 to 5.

    v5 adds default_intrinsics_calibration_id (nullable FK) to camera_modes so
    the shot wizard can auto-select the preferred calibration for a mode.
    """
    sql = (_DB_DIR / "migrations" / "009_camera_modes_default_calib.sql").read_text(
        encoding="utf-8"
    )
    conn.executescript(sql)
    conn.execute("PRAGMA user_version = 5")
    conn.commit()


def _migrate_registry_v5_to_v6(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 5 to 6.

    v6 adds velocity_mode_camera_ids and velocity_measurement_noise_std to
    tracker_configs for per-camera velocity measurement mode support.
    Both columns are NULL in existing rows (backward-compatible).
    """
    conn.executescript("""
        BEGIN;
        ALTER TABLE tracker_configs ADD COLUMN velocity_mode_camera_ids TEXT;
        ALTER TABLE tracker_configs ADD COLUMN velocity_measurement_noise_std REAL;
        PRAGMA user_version = 6;
        COMMIT;
    """)


def _migrate_registry_v6_to_v7(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 6 to 7.

    v7 catches up tracker_configs to the full current column set and adds
    is_named (the config-improvements design's named/reusable-config flag --
    see docs/roadmap/features/configuration-improvements/config-improvements-design.md).

    The registry's own migration chain had stopped tracking tracker_configs
    columns after v6 (velocity_mode_camera_ids/velocity_measurement_noise_std)
    -- every column added since (pose_noise_std at session-schema v22 onward,
    ~35 columns total) was only ever added via ALTER TABLE in the *session*
    migration chain, never here. A registry DB created or last opened before
    this point could therefore be missing most of tracker_configs' current
    columns. Fixed generically, the same principle as manage_config.py's
    edit_config() fix: build a reference copy of tracker_configs from the
    *current* registry_schema.sql in an in-memory DB, diff its columns
    against this connection's actual columns, and ALTER TABLE ADD COLUMN
    whatever's missing -- including is_named, with no special-casing needed.
    A future schema addition needs no matching registry migration written
    by hand to stay caught up.
    """
    ref = sqlite3.connect(":memory:")
    try:
        ref.executescript(_REGISTRY_SCHEMA_SQL.read_text(encoding="utf-8"))
        ref_columns = list(ref.execute("PRAGMA table_info(tracker_configs)"))
    finally:
        ref.close()

    existing = {row[1] for row in conn.execute("PRAGMA table_info(tracker_configs)")}
    for row in ref_columns:
        # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk.
        col_name, col_type, notnull, dflt_value = row[1], row[2], row[3], row[4]
        if col_name in existing:
            continue
        ddl = f"{col_name} {col_type}"
        if notnull:
            ddl += " NOT NULL"
        if dflt_value is not None:
            ddl += f" DEFAULT {dflt_value}"
        conn.execute(f"ALTER TABLE tracker_configs ADD COLUMN {ddl}")
    conn.execute("PRAGMA user_version = 7")
    conn.commit()

    # This registry predates the baseline config (it's only ever seeded by
    # create_registry(), which this DB didn't go through); back-fill it here
    # so the default-config resolution chain has somewhere to terminate.
    from posetrak.db.manage_config import seed_baseline_tracker_config
    seed_baseline_tracker_config(conn)


def _migrate_registry_v7_to_v8(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 7 to 8.

    v8 adds marker_body_definitions -- a named, reusable rigid body
    carrying one or more fiducial markers at fixed positions (portable
    calibration rigs, future marker-cluster objects). Same
    content-addressed (SHA-256 of yaml_content) id convention as
    skeletons. See docs/roadmap/features/extrinsics-improvements/
    extrinsics-improvements-design.md, section 10.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS marker_body_definitions ("
        "    id           TEXT PRIMARY KEY,"
        "    name         TEXT NOT NULL,"
        "    yaml_content TEXT NOT NULL,"
        "    source       TEXT,"
        "    created_at   TEXT NOT NULL,"
        "    notes        TEXT"
        ")"
    )
    _set_schema_version(conn, 8)
    conn.commit()


def _migrate_registry_v8_to_v9(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 8 to 9.

    v9 adds streak-derived dot velocity's tuning columns to tracker_configs --
    see docs/roadmap/features/marker-based-mocap/streak-velocity-design.md
    §3/§4. NULL/0 on every column means disabled, matching every other
    adaptive-tracking mechanism's own backward-compatible default.
    """
    existing = _tracker_config_columns(conn)
    if "dot_streak_velocity_enabled" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_streak_velocity_enabled INTEGER")
    if "dot_streak_k_window" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_streak_k_window INTEGER")
    if "dot_streak_k_min_samples" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_streak_k_min_samples INTEGER")
    if "dot_streak_min_displacement_px" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_streak_min_displacement_px REAL")
    if "dot_streak_min_elongation_px" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_streak_min_elongation_px REAL")
    if "dot_streak_velocity_noise_std" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_streak_velocity_noise_std REAL")
    _set_schema_version(conn, 9)
    conn.commit()


def _migrate_registry_v9_to_v10(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 9 to 10.

    v10 adds process_noise_vel_max_multiplier to tracker_configs: the cap on the
    adaptive process noise (Mechanism A) variance-domain multiplier, formerly a
    hardcoded UKF constant (kMaxVelocityNoiseMultiplier = 10.0). It can bind on a
    fast-swing capture: a third of the run's steps, including 100% of a known
    bad-tracking window, already saturated the default gain=4/ref_root=2 tuning's
    multiplier at just ~1.08 rad/s of root angular velocity. NULL means the same
    10.0 default every existing config already got from the hardcoded constant --
    backward-compatible. See
    docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md.
    """
    existing = _tracker_config_columns(conn)
    if "process_noise_vel_max_multiplier" not in existing:
        conn.execute(
            "ALTER TABLE tracker_configs ADD COLUMN process_noise_vel_max_multiplier REAL"
        )
    _set_schema_version(conn, 10)
    conn.commit()


def _migrate_registry_v10_to_v11(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 10 to 11.

    v11 adds the dot-candidate assignment gate and its tracklet-aware
    relaxation to tracker_configs, mirroring the session schema v51->v52
    change. dot_assignment_gate_mahalanobis existed as a
    TOML-only tunable before this migration; it had no DB column at all, so
    a DB-driven tracker_config row could never actually override it -- fixed
    here alongside the new tracklet multiplier since both touch the same
    assignment gate.
    """
    existing = _tracker_config_columns(conn)
    if "dot_assignment_gate_mahalanobis" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_assignment_gate_mahalanobis REAL")
    if "dot_tracklet_gate_multiplier" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_tracklet_gate_multiplier REAL")
    _set_schema_version(conn, 11)
    conn.commit()


def _migrate_registry_v11_to_v12(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 11 to 12.

    v12 adds the experimental per-marker confidence-threshold override to
    tracker_configs, mirroring the session schema v52->v53 change -- see
    _migrate_session_v52_to_v53's docstring.
    """
    existing = _tracker_config_columns(conn)
    if "confidence_threshold_marker_names" not in existing:
        conn.execute(
            "ALTER TABLE tracker_configs ADD COLUMN confidence_threshold_marker_names TEXT"
        )
    if "confidence_threshold_override" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN confidence_threshold_override REAL")
    _set_schema_version(conn, 12)
    conn.commit()


def _migrate_registry_v12_to_v13(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 12 to 13.

    v13 adds the reacquisition gate for the shared dot-assignment phase to
    tracker_configs, mirroring the session schema v53->v54 change -- see
    _migrate_session_v53_to_v54's docstring.
    """
    existing = _tracker_config_columns(conn)
    if "dot_reacquire_gap_frames" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_reacquire_gap_frames INTEGER")
    if "dot_reacquire_max_px" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_reacquire_max_px REAL")
    _set_schema_version(conn, 13)
    conn.commit()


def _migrate_registry_v13_to_v14(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 13 to 14.

    v14 adds cross-view corroboration and per-camera trust for the shared
    dot-assignment phase to tracker_configs, mirroring the session schema
    v54->v55 change -- see _migrate_session_v54_to_v55's docstring.
    """
    existing = _tracker_config_columns(conn)
    if "dot_cross_view_corroboration_px" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_cross_view_corroboration_px REAL")
    if "dot_corroboration_gate_multiplier" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_corroboration_gate_multiplier REAL")
    if "dot_camera_noise_scale" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_camera_noise_scale TEXT")
    _set_schema_version(conn, 14)
    conn.commit()


def _migrate_registry_v14_to_v15(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 14 to 15.

    v15 adds segmentation-mask-based dot-assignment trust to tracker_configs,
    mirroring the session schema v56->v57 change -- see
    _migrate_session_v56_to_v57's docstring.
    """
    existing = _tracker_config_columns(conn)
    if "dot_seg_mask_enabled" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_seg_mask_enabled INTEGER")
    if "dot_seg_mask_dilate_px" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_seg_mask_dilate_px REAL")
    if "dot_seg_mask_tracklet_window" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_seg_mask_tracklet_window INTEGER")
    if "dot_seg_mask_min_tracklet_fraction" not in existing:
        conn.execute(
            "ALTER TABLE tracker_configs ADD COLUMN dot_seg_mask_min_tracklet_fraction REAL"
        )
    if "dot_seg_mask_cost_scale_at_zero_fraction" not in existing:
        conn.execute(
            "ALTER TABLE tracker_configs ADD COLUMN dot_seg_mask_cost_scale_at_zero_fraction REAL"
        )
    _set_schema_version(conn, 15)
    conn.commit()


def _migrate_registry_v15_to_v16(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 15 to 16.

    v16 adds tracker_configs.dot_seg_mask_hard_distance_px, mirroring the
    session schema v60->v61 change -- see _migrate_session_v60_to_v61.
    """
    if "dot_seg_mask_hard_distance_px" not in _tracker_config_columns(conn):
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_seg_mask_hard_distance_px REAL")
    _set_schema_version(conn, 16)
    conn.commit()


def _migrate_registry_v16_to_v17(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 16 to 17.

    v17 adds tracker_configs.active_joint_groups, mirroring the session schema
    v66->v67 change -- see _migrate_session_v66_to_v67.
    """
    if "active_joint_groups" not in _tracker_config_columns(conn):
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN active_joint_groups TEXT")
    _set_schema_version(conn, 17)
    conn.commit()


def _add_kp_candidate_config_columns(conn: sqlite3.Connection) -> None:
    """Add the keypoint-candidate settings to tracker_configs where they are missing."""
    existing = _tracker_config_columns(conn)
    for column, sql_type in (
        ("kp_candidates_enabled", "INTEGER"),
        ("kp_candidate_peak_weight", "REAL"),
        ("kp_candidate_tracklet_multiplier", "REAL"),
    ):
        if column not in existing:
            conn.execute(f"ALTER TABLE tracker_configs ADD COLUMN {column} {sql_type}")


def _migrate_registry_v17_to_v18(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 17 to 18.

    v18 adds tracker_configs.kp_candidates_enabled, kp_candidate_peak_weight and
    kp_candidate_tracklet_multiplier, mirroring the session schema v68->v69 change -- see
    _migrate_session_v68_to_v69.
    """
    _add_kp_candidate_config_columns(conn)
    _set_schema_version(conn, 18)
    conn.commit()


_KP_CHOICE_CONFIG_COLUMNS = (
    ("kp_candidate_choice", "TEXT"),
    ("kp_choice_support_window_s", "REAL"),
    ("kp_choice_delta", "REAL"),
    ("kp_choice_beta", "REAL"),
    ("kp_choice_gamma", "REAL"),
    ("kp_choice_rounds", "INTEGER"),
    ("kp_choice_skip_camera_ids", "TEXT"),
    ("kp_strict_filter", "INTEGER"),
    ("kp_strict_filter_window_frames", "INTEGER"),
    ("kp_strict_filter_min_fraction", "REAL"),
    ("kp_mask_term_weight", "REAL"),
    ("kp_mask_term_window_s", "REAL"),
    ("kp_mask_term_classes", "TEXT"),
)


def _add_kp_choice_config_columns(conn: sqlite3.Connection) -> None:
    """Add the settings of the windowed choice of keypoint candidates to tracker_configs where they are missing."""
    existing = _tracker_config_columns(conn)
    for column, sql_type in _KP_CHOICE_CONFIG_COLUMNS:
        if column not in existing:
            conn.execute(f"ALTER TABLE tracker_configs ADD COLUMN {column} {sql_type}")


def _migrate_registry_v18_to_v19(conn: sqlite3.Connection) -> None:
    """Migrate a registry database from schema version 18 to 19.

    v19 adds the tracker_configs columns of the windowed choice of keypoint candidates (kp_candidate_choice and
    the kp_choice_*, kp_strict_filter* and kp_mask_term_* settings), mirroring the session schema v69->v70
    change -- see _migrate_session_v69_to_v70.
    """
    _add_kp_choice_config_columns(conn)
    _set_schema_version(conn, 19)
    conn.commit()


def _migrate_session_v9_to_v10(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 9 to 10.

    v10 adds default_intrinsics_calibration_id (nullable FK) to the session-local
    copy of camera_modes, mirroring the registry schema v4→v5 change.

    Very old hand-crafted sessions (created before create_session() embedded the
    registry schema) may not have a camera_modes table.  In that case the ALTER
    TABLE is skipped; the column will be present when the registry tables are
    eventually created for that session.
    """
    has_camera_modes = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='camera_modes'"
    ).fetchone() is not None
    if has_camera_modes:
        sql = (_DB_DIR / "migrations" / "009_camera_modes_default_calib.sql").read_text(
            encoding="utf-8"
        )
        conn.executescript(sql)
    conn.execute("PRAGMA user_version = 10")
    conn.commit()


def _migrate_session_v10_to_v11(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 10 to 11.

    v11 moves camera_mode_id and intrinsics_calibration_id from session_cameras
    to shot_videos so that each video can declare its own capture mode and
    intrinsics (supporting mixed-mode sessions).

    The migration uses ALTER TABLE … DROP COLUMN which requires SQLite 3.35+
    (available in CPython 3.12+).
    """
    sql = (_DB_DIR / "migrations" / "010_shot_videos_mode_intrinsics.sql").read_text(
        encoding="utf-8"
    )
    conn.executescript(sql)
    conn.execute("PRAGMA user_version = 11")
    conn.commit()


def _migrate_session_v11_to_v12(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 11 to 12.

    v12 adds the sequence_persons table which maps integer person_id values
    to human-readable names within a pose_observation_sequence.  This allows
    the pose UI to restore assignment colours when a detection run is reopened.
    """
    sql = (_DB_DIR / "migrations" / "011_sequence_persons.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.execute("PRAGMA user_version = 12")
    conn.commit()


def _migrate_session_v12_to_v13(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 12 to 13.

    v13 introduces capture/trial terminology:
    - Renames the shots table to captures (shot_number → capture_number).
    - Renames shot_videos to capture_videos.
    - Adds the trials table for named time windows within a capture.
    - Adds trial_id to detection_runs.
    - Adds name to pose_observation_sequences (person track label).
    - Adds notes to tracking_runs.
    """
    sql = (_DB_DIR / "migrations" / "012_captures_and_trials.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.execute("PRAGMA user_version = 13")
    conn.commit()


def _migrate_session_v13_to_v14(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 13 to 14.

    v14 adds the sync anchor input layer:
    - sync_anchors: one row per shared real-world event visible in 2+ cameras.
    - sync_anchor_observations: per-video frame number (+subframe) for each anchor.
    """
    sql = (_DB_DIR / "migrations" / "013_sync_anchors.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()


def _migrate_session_v14_to_v15(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 14 to 15.

    v15 adds detection_track_assignments: records the explicit track_id →
    person_name mapping from the pose extraction UI so assignments can be
    restored without ambiguous joins through pose_observations.
    """
    sql = (_DB_DIR / "migrations" / "014_detection_track_assignments.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()


def _migrate_session_v15_to_v16(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 15 to 16.

    v16 adds the detection_crops table (superseded by v17).
    """
    sql = (_DB_DIR / "migrations" / "015_detection_crops.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()


def _migrate_session_v16_to_v17(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 16 to 17.

    v17 adds detection_run_id to the frame_cache_entries primary key so that
    PERSON_CROP entries from multiple detection runs on the same shot coexist.
    Non-crop entries use detection_run_id = ''.  Also removes detection_crops
    (its purpose is now covered by frame_cache_entries).
    """
    sql = (_DB_DIR / "migrations" / "016_frame_cache_detection_run.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()


def _migrate_session_v17_to_v18(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 17 to 18.

    v18 adds src_x, src_y, src_w, src_h columns to frame_cache_entries so
    that the exact crop rectangle (in original-frame pixels, before JPEG
    downscale) is stored alongside PERSON_CROP entries.  This lets overlay
    code derive the correct coordinate transform without re-reading the
    detection bounding box.  Existing rows get NULL for all four columns.
    """
    sql = (_DB_DIR / "migrations" / "017_frame_cache_src_rect.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()


def _migrate_session_v18_to_v19(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 18 to 19.

    v19 adds the seg_masks table for the interactive Cutie init widget.
    Each row stores one labeled segmentation mask (indexed PNG blob) per
    (seg_quality_run_id, shot_video_id, frame_idx).
    """
    sql = (_DB_DIR / "migrations" / "018_seg_masks.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()


def _migrate_session_v19_to_v20(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 19 to 20.

    v20 adds velocity_mode_camera_ids and velocity_measurement_noise_std to
    tracker_configs (embedded registry table) for per-camera velocity measurement
    mode support.  Both columns are NULL in existing rows (backward-compatible).
    """
    sql = (_DB_DIR / "migrations" / "019_tracker_velocity_mode.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()


def _migrate_session_v20_to_v21(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 20 to 21.

    v21 adds the pose_observation_edits table and its unique index, providing
    a non-destructive keypoint edit overlay on top of pose_observations.
    """
    sql = (_DB_DIR / "migrations" / "020_pose_observation_edits.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()


def _tracker_config_columns(conn: sqlite3.Connection) -> set[str]:
    """Return the set of column names currently in tracker_configs."""
    return {row[1] for row in conn.execute("PRAGMA table_info(tracker_configs)")}


def _migrate_session_v21_to_v22(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 21 to 22.

    v22 adds pose_noise_std to tracker_configs for the split noise model:
    total_noise = (pose_noise_std * crop_scale + calib_noise_std) / max(conf, 0.1).
    NULL / 0.0 means use calibration-only formula (backward-compatible).
    """
    existing = _tracker_config_columns(conn)
    if "pose_noise_std" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN pose_noise_std REAL")
    _set_schema_version(conn, 22)
    conn.commit()


def _migrate_session_v22_to_v23(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 22 to 23.

    v23 adds use_relative_observations and relative_min_confidence to tracker_configs,
    enabling the RELATIVE measurement mode (child-minus-parent pixel differences).
    NULL means disabled / 0.5 respectively (backward-compatible with v22 configs).
    """
    existing = _tracker_config_columns(conn)
    if "use_relative_observations" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN use_relative_observations INTEGER")
    if "relative_min_confidence" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN relative_min_confidence REAL")
    _set_schema_version(conn, 23)
    conn.commit()


def _migrate_session_v23_to_v24(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 23 to 24.

    v24 adds cross_pair_max_px and cross_pair_max_n to tracker_configs,
    enabling spatial cross-pair RELATIVE observations.
    NULL means disabled / 10 respectively (backward-compatible with v23 configs).
    """
    existing = _tracker_config_columns(conn)
    if "cross_pair_max_px" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN cross_pair_max_px REAL")
    if "cross_pair_max_n" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN cross_pair_max_n INTEGER")
    _set_schema_version(conn, 24)
    conn.commit()


def _migrate_session_v24_to_v25(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 24 to 25.

    v25 adds trial_id to tracking_runs as a direct FK, enabling fast trial lookup
    without the 3-hop join through observation_sequences and detection_runs.
    Existing rows are backfilled via that join.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_runs)")}
    if "trial_id" not in cols:
        conn.execute(
            "ALTER TABLE tracking_runs ADD COLUMN trial_id TEXT REFERENCES trials(id)"
        )
        conn.execute("""
            UPDATE tracking_runs
            SET trial_id = (
                SELECT dr.trial_id
                FROM detection_runs dr
                JOIN pose_observation_sequences s
                    ON s.id = tracking_runs.observation_sequence_id
                WHERE dr.id = s.detection_run_id
            )
        """)
    _set_schema_version(conn, 25)
    conn.commit()


def _migrate_session_v25_to_v26(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 25 to 26.

    v26 adds process_noise_vel_gain_joint/root and process_noise_vel_ref_joint/root
    to tracker_configs, enabling velocity-driven per-DOF process noise (adaptive
    process noise Phase 1). NULL/0 gain means disabled (backward-compatible with
    v25 configs, which get the exact static process noise as before).
    """
    existing = _tracker_config_columns(conn)
    if "process_noise_vel_gain_joint" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN process_noise_vel_gain_joint REAL")
    if "process_noise_vel_ref_joint" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN process_noise_vel_ref_joint REAL")
    if "process_noise_vel_gain_root" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN process_noise_vel_gain_root REAL")
    if "process_noise_vel_ref_root" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN process_noise_vel_ref_root REAL")
    _set_schema_version(conn, 26)
    conn.commit()


def _migrate_session_v26_to_v27(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 26 to 27.

    v27 adds process_noise_vel_joint_names to tracker_configs: a JSON array of
    literal joint names (e.g. "spine1", "thigh.L") the adaptive process noise
    joint gain applies to. NULL/empty means all joints (backward-compatible with
    v26 configs). Added after finding a body-wide joint gain over-loosens
    fast-but-normal limb motion (arms) while barely engaging for the slower
    torso/hip motion it targets. Name-based rather than skeleton-group-based
    since existing skeleton YAMLs don't define groups fine-grained enough for
    this (one "main" group spans the whole body), and adding a finer split would
    mean editing every person's skeleton file.
    """
    existing = _tracker_config_columns(conn)
    if "process_noise_vel_joint_names" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN process_noise_vel_joint_names TEXT")
    _set_schema_version(conn, 27)
    conn.commit()


def _migrate_session_v27_to_v28(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 27 to 28.

    v28 adds pose_reg_joint_names, pose_reg_equal_split_noise_std, and
    pose_reg_rest_pose_noise_std to tracker_configs: pose regularization for a
    kinematically redundant joint chain (e.g. spine1/spine2), fusing two soft
    pseudo-measurements (equal-split and rest-pose pull) into the UKF update so
    one joint in a redundant chain doesn't absorb all available rotation (and
    hit its own limit) while the others stay near neutral. NULL/empty
    pose_reg_joint_names means disabled (backward-compatible with v27
    configs). See
    docs/roadmap/features/pose-regularization/pose-regularization-design.md.
    """
    existing = _tracker_config_columns(conn)
    if "pose_reg_joint_names" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN pose_reg_joint_names TEXT")
    if "pose_reg_equal_split_noise_std" not in existing:
        conn.execute(
            "ALTER TABLE tracker_configs ADD COLUMN pose_reg_equal_split_noise_std REAL"
        )
    if "pose_reg_rest_pose_noise_std" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN pose_reg_rest_pose_noise_std REAL")
    _set_schema_version(conn, 28)
    conn.commit()


def _migrate_session_v28_to_v29(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 28 to 29.

    v29 adds nis_feedback_scopes, nis_feedback_window, nis_feedback_threshold, and
    nis_feedback_max_multiplier to tracker_configs: the NIS-feedback regional fading
    safety net (Mechanism B). Each scope is a named group of joints; when a scope's
    windowed average NIS/DOF (computed from per-observation Mahalanobis distances
    attributed to that scope's joints) exceeds nis_feedback_threshold, a temporary
    variance-domain multiplier (capped at nis_feedback_max_multiplier) widens that
    scope's process noise until the windowed average returns to nominal. NULL/empty
    nis_feedback_scopes means disabled (backward-compatible with v28 configs). See
    docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md.
    """
    existing = _tracker_config_columns(conn)
    if "nis_feedback_scopes" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN nis_feedback_scopes TEXT")
    if "nis_feedback_window" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN nis_feedback_window INTEGER")
    if "nis_feedback_threshold" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN nis_feedback_threshold REAL")
    if "nis_feedback_max_multiplier" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN nis_feedback_max_multiplier REAL")
    _set_schema_version(conn, 29)
    conn.commit()


def _migrate_session_v29_to_v30(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 29 to 30.

    v30 adds process_noise_vel_gain_arms, process_noise_vel_ref_arms, and
    process_noise_vel_joint_names_arms to tracker_configs: a second, independent
    adaptive process noise gain scope (e.g. arms), separate from
    process_noise_vel_gain_joint/process_noise_vel_joint_names. Added after finding
    the NIS-feedback safety net (Mechanism B) alone wasn't enough to keep a fast
    bilateral hand-raise tracked, once pose regularization separately fixed the
    spine issue that originally forced arms to be excluded from the primary gain
    scope. NULL/empty process_noise_vel_joint_names_arms means disabled
    (backward-compatible with v29 configs). See
    docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md.
    """
    existing = _tracker_config_columns(conn)
    if "process_noise_vel_gain_arms" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN process_noise_vel_gain_arms REAL")
    if "process_noise_vel_ref_arms" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN process_noise_vel_ref_arms REAL")
    if "process_noise_vel_joint_names_arms" not in existing:
        conn.execute(
            "ALTER TABLE tracker_configs ADD COLUMN process_noise_vel_joint_names_arms TEXT"
        )
    _set_schema_version(conn, 30)
    conn.commit()


def _migrate_session_v30_to_v31(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 30 to 31.

    v31 replaces v30's single hardcoded "arms" gain scope
    (process_noise_vel_gain_arms/process_noise_vel_ref_arms/
    process_noise_vel_joint_names_arms) with process_noise_vel_scopes: an
    arbitrary JSON list of {name, joint_names, gain, vel_ref}, once a single
    extra split stopped being enough -- distal joints (wrist, ankle) move
    faster than proximal ones (elbow, knee, shoulder, hip) and warrant their
    own reference velocity, not one shared "arms" value. Existing v30 configs
    with a non-empty arms scope are migrated into a single-entry
    process_noise_vel_scopes list (named "arms") so they keep working
    unchanged; the old columns are then dropped. See
    docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md.
    """
    existing = _tracker_config_columns(conn)
    if "process_noise_vel_scopes" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN process_noise_vel_scopes TEXT")

    if "process_noise_vel_gain_arms" in existing:
        rows = conn.execute(
            "SELECT id, process_noise_vel_gain_arms, process_noise_vel_ref_arms,"
            "       process_noise_vel_joint_names_arms"
            " FROM tracker_configs"
            " WHERE process_noise_vel_gain_arms IS NOT NULL"
            "   AND process_noise_vel_gain_arms > 0"
            "   AND process_noise_vel_joint_names_arms IS NOT NULL"
        ).fetchall()
        for config_id, gain, vel_ref, joint_names_json in rows:
            joint_names = json.loads(joint_names_json) if joint_names_json else []
            if not joint_names:
                continue
            scopes = [
                {
                    "name": "arms",
                    "joint_names": joint_names,
                    "gain": gain,
                    "vel_ref": vel_ref if vel_ref is not None else 1.0,
                }
            ]
            conn.execute(
                "UPDATE tracker_configs SET process_noise_vel_scopes = ? WHERE id = ?",
                (json.dumps(scopes), config_id),
            )
        conn.execute("ALTER TABLE tracker_configs DROP COLUMN process_noise_vel_gain_arms")
        conn.execute("ALTER TABLE tracker_configs DROP COLUMN process_noise_vel_ref_arms")
        conn.execute(
            "ALTER TABLE tracker_configs DROP COLUMN process_noise_vel_joint_names_arms"
        )

    _set_schema_version(conn, 31)
    conn.commit()


def _migrate_session_v31_to_v32(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 31 to 32.

    v32 adds soft_limit_joint_names, soft_limit_margin_rad, and
    soft_limit_noise_std to tracker_configs: a pseudo-measurement that
    discourages a joint's angle from approaching its own hard limit, rather
    than only reacting once the hard clamp fires after the fact. Added after
    tracing a sustained "arms completely lost" crisis to upper_arm.L/R
    overshooting their own ball-joint limits during a fast bilateral motion,
    where adaptive process noise (widening the sigma-point spread for a
    fast-moving joint) made the overshoot worse rather than better. NULL/empty
    soft_limit_joint_names means disabled (backward-compatible with v31
    configs). See
    docs/roadmap/features/soft-joint-limits/soft-joint-limits-design.md.
    """
    existing = _tracker_config_columns(conn)
    if "soft_limit_joint_names" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN soft_limit_joint_names TEXT")
    if "soft_limit_margin_rad" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN soft_limit_margin_rad REAL")
    if "soft_limit_noise_std" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN soft_limit_noise_std REAL")
    _set_schema_version(conn, 32)
    conn.commit()


def _migrate_session_v32_to_v33(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 32 to 33.

    v33 adds near_limit_damping_joint_names, near_limit_margin_rad,
    near_limit_spread_sigma, and near_limit_damping_factor to
    tracker_configs: shrinks process noise for a joint whose current
    covariance-implied spread already reaches close to one of its
    configured hard limits, regardless of velocity -- the inverse of the
    existing velocity-driven adaptive process noise. Added after tracing
    two unrelated tracking-crisis events to the same mechanism: a wide
    sigma-point cloud straddling a box-constraint corner (2-3 axes near
    their limits simultaneously) breaks the UKF's local-linearity
    assumption, causing a discontinuous multi-radian jump. The soft
    joint-limit pseudo-measurement (v32) steers the *mean* away from the
    corner but was shown insufficient alone -- this targets the sigma
    *spread* instead. NULL/empty near_limit_damping_joint_names means
    disabled (backward-compatible with v32 configs). See
    docs/roadmap/features/tracking-crisis-debugging-log.md, "Proposals".
    """
    existing = _tracker_config_columns(conn)
    if "near_limit_damping_joint_names" not in existing:
        conn.execute(
            "ALTER TABLE tracker_configs ADD COLUMN near_limit_damping_joint_names TEXT"
        )
    if "near_limit_margin_rad" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN near_limit_margin_rad REAL")
    if "near_limit_spread_sigma" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN near_limit_spread_sigma REAL")
    if "near_limit_damping_factor" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN near_limit_damping_factor REAL")
    _set_schema_version(conn, 33)
    conn.commit()


def _migrate_session_v33_to_v34(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 33 to 34.

    v34 adds edited_kp_noise_std to tracker_configs (Phase 0 of trusted
    keypoint edits): when > 0, a keypoint slot overridden by a
    pose_observation_edits row (human-placed, is_outlier=false) gets this
    value as its measurement noise instead of the usual pose/calibration
    formula, and is exempted from the tracker's outlier gate entirely.
    NULL/0 means disabled (identical to pre-v34 behaviour). There is no
    principled default value -- see
    docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md,
    "Measurement noise for edited and automated observations".
    """
    existing = _tracker_config_columns(conn)
    if "edited_kp_noise_std" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN edited_kp_noise_std REAL")
    _set_schema_version(conn, 34)
    conn.commit()


def _migrate_session_v34_to_v35(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 34 to 35.

    v35 adds `source` to the pose_observations primary key (Phase 2 of
    hand-detection refinement): existing rows become source='body', and a
    new detection_run_id column is backfilled from each row's parent
    pose_observation_sequences.detection_run_id. This lets a refined hand
    pass (source='hand_l'/'hand_r') contribute its own row with its own
    noise_scale instead of being patched into the whole-body kp_blob in
    place, so the C++ tracker can model each source's measurement noise
    correctly. See
    docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md.
    """
    sql = (_DB_DIR / "migrations" / "024_pose_observations_source.sql").read_text(
        encoding="utf-8"
    )
    conn.executescript(sql)


def _migrate_session_v35_to_v36(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 35 to 36.

    v36 adds cross_person_max_world_mm, cross_person_min_confidence, and
    cross_person_max_n to tracker_configs, enabling cross-person PAIR_DIFF
    anchoring for MultiPersonTracker (Phase 5 of error-improvements). NULL
    means disabled / 0.5 / 10 respectively (backward-compatible with v35
    configs). See
    docs/roadmap/features/error-improvements/phase5-cross-person-plan.md.
    """
    existing = _tracker_config_columns(conn)
    if "cross_person_max_world_mm" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN cross_person_max_world_mm REAL")
    if "cross_person_min_confidence" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN cross_person_min_confidence REAL")
    if "cross_person_max_n" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN cross_person_max_n INTEGER")
    _set_schema_version(conn, 36)
    conn.commit()


def _migrate_session_v36_to_v37(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 36 to 37.

    v37 adds tracking_run_stages (per-stage run bookkeeping) and
    tracker_config_stages (per-stage tuning, NULL inherits from the parent
    tracker_configs row) for the hierarchical body/hand solver. See
    docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md.
    """
    sql = (_DB_DIR / "migrations" / "026_hierarchical_solver_stages.sql").read_text(
        encoding="utf-8"
    )
    conn.executescript(sql)


def _migrate_session_v37_to_v38(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 37 to 38.

    v38 adds tracker_configs.is_named (explicit flag distinguishing a
    user-saved, browsable named config from an auto-generated per-run
    snapshot) and captures.default_tracker_config_id /
    trials.default_tracker_config_id (the default-config-per-scope
    resolution chain: trial falls through to capture, then to a checked-in
    baseline config). See
    docs/roadmap/features/configuration-improvements/config-improvements-design.md.
    """
    existing = _tracker_config_columns(conn)
    if "is_named" not in existing:
        conn.execute(
            "ALTER TABLE tracker_configs ADD COLUMN is_named INTEGER NOT NULL DEFAULT 0"
        )
    existing_captures = {row[1] for row in conn.execute("PRAGMA table_info(captures)")}
    if "default_tracker_config_id" not in existing_captures:
        conn.execute("ALTER TABLE captures ADD COLUMN default_tracker_config_id TEXT")
    existing_trials = {row[1] for row in conn.execute("PRAGMA table_info(trials)")}
    if "default_tracker_config_id" not in existing_trials:
        conn.execute("ALTER TABLE trials ADD COLUMN default_tracker_config_id TEXT")
    _set_schema_version(conn, 38)
    conn.commit()


def _migrate_session_v38_to_v39(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 38 to 39.

    v39 adds capture_persons (named performers defined once per capture,
    replacing the previous per-detection-run-only free-text person_name)
    plus a nullable capture_persons.id link on sequence_persons and
    detection_track_assignments -- additive, not a replacement: person_name
    keeps working unchanged for rows that predate this feature. See
    docs/roadmap/features/configuration-improvements/config-improvements-design.md,
    "Person model: promote identity to capture level".
    """
    existing_tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if "capture_persons" not in existing_tables:
        conn.execute(
            "CREATE TABLE capture_persons ("
            "    id                  TEXT PRIMARY KEY,"
            "    capture_id          TEXT NOT NULL REFERENCES captures(id),"
            "    name                TEXT NOT NULL,"
            "    default_skeleton_id TEXT,"
            "    notes               TEXT,"
            "    created_at          TEXT NOT NULL"
            ")"
        )
    existing_seq_persons = {row[1] for row in conn.execute("PRAGMA table_info(sequence_persons)")}
    if "capture_person_id" not in existing_seq_persons:
        conn.execute(
            "ALTER TABLE sequence_persons ADD COLUMN capture_person_id TEXT "
            "REFERENCES capture_persons(id)"
        )
    existing_assignments = {
        row[1] for row in conn.execute("PRAGMA table_info(detection_track_assignments)")
    }
    if "capture_person_id" not in existing_assignments:
        conn.execute(
            "ALTER TABLE detection_track_assignments ADD COLUMN capture_person_id TEXT "
            "REFERENCES capture_persons(id)"
        )
    _set_schema_version(conn, 39)
    conn.commit()


def _migrate_session_v39_to_v40(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 39 to 40.

    v40 adds marker_body_definitions (mirrored from the registry, same
    embedding create_session() already does for camera_models/skeletons/
    tracker_configs -- see the SELF-CONTAINMENT REQUIREMENT header in
    session_schema.sql) and scene_marker_bodies (this session's solved
    marker-body poses: the portable calibration rig's own anchor pose,
    and/or ordinary scattered scene tags). One row per solved body
    instance, not per marker. See docs/roadmap/features/
    extrinsics-improvements/extrinsics-improvements-design.md, section 10
    -- supersedes that design doc's earlier scene_fiducial_markers sketch,
    which was never implemented.
    """
    existing_tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if "marker_body_definitions" not in existing_tables:
        conn.execute(
            "CREATE TABLE marker_body_definitions ("
            "    id           TEXT PRIMARY KEY,"
            "    name         TEXT NOT NULL,"
            "    yaml_content TEXT NOT NULL,"
            "    source       TEXT,"
            "    created_at   TEXT NOT NULL,"
            "    notes        TEXT"
            ")"
        )
    if "scene_marker_bodies" not in existing_tables:
        conn.execute(
            "CREATE TABLE scene_marker_bodies ("
            "    id                               TEXT PRIMARY KEY,"
            "    session_id                       TEXT NOT NULL REFERENCES mocap_sessions(id),"
            "    label                            TEXT NOT NULL,"
            "    marker_body_definition_id        TEXT,"
            "    marker_type                      TEXT,"
            "    dictionary                       TEXT,"
            "    marker_id                        TEXT,"
            "    marker_size                      REAL,"
            "    R                                BLOB NOT NULL,"
            "    t                                BLOB NOT NULL,"
            "    is_primary_anchor                INTEGER NOT NULL DEFAULT 0,"
            "    source_extrinsic_calibration_id  TEXT REFERENCES extrinsic_calibrations(id),"
            "    updated_at                       TEXT NOT NULL"
            ")"
        )
        conn.execute(
            "CREATE UNIQUE INDEX scene_marker_bodies_unique "
            "ON scene_marker_bodies (session_id, label)"
        )
    _set_schema_version(conn, 40)
    conn.commit()


def _migrate_session_v40_to_v41(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 40 to 41.

    v41 adds ``scene_marker_bodies.group_name`` -- a user-chosen name
    grouping every marker anchored together in one physical space (e.g.
    "room7"), so a later capture in the same room can pick that name from
    a list instead of every room's markers in the session loading
    together indiscriminately (see docs/roadmap/features/
    extrinsics-improvements/extrinsics-improvements-design.md, section 9,
    and status.md's 2026-08-12 "how do I select which scene markers"
    entry). ``''`` (not NULL -- see session_schema.sql's column comment
    for why) for legacy/ungrouped rows written before this existed. The
    table's uniqueness moves from (session_id, label) to (session_id,
    group_name, label) so two rooms can reuse the same tag id without
    colliding.
    """
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(scene_marker_bodies)")
    }
    if "group_name" not in existing_cols:
        conn.execute(
            "ALTER TABLE scene_marker_bodies ADD COLUMN group_name TEXT NOT NULL DEFAULT ''"
        )
        conn.execute("DROP INDEX IF EXISTS scene_marker_bodies_unique")
        conn.execute(
            "CREATE UNIQUE INDEX scene_marker_bodies_unique "
            "ON scene_marker_bodies (session_id, group_name, label)"
        )
    _set_schema_version(conn, 41)
    conn.commit()


def _migrate_session_v41_to_v42(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 41 to 42.

    v42 makes seg_quality_runs a time-range-scoped, capture-level resource
    instead of being permanently 1:1 with one detection run -- adds
    shot_id/trial_id/time_start_s/time_end_s (mirroring detection_runs'
    own columns), drops the NOT NULL detection_run_id FK. Lets a
    segmentation be created (and its masks used to drive pose extraction)
    before any detection run exists for the capture -- previously the
    only way to open the segmentation UI at all was to already have run
    YOLO detection once, purely to get a detection_runs row to hang the
    segmentation off of. See docs/roadmap/features/segmentation-reuse/
    segmentation-reuse-design.md.

    SQLite can't drop a NOT NULL column with the ALTER syntax this
    codebase otherwise uses for additive migrations (v38-v41 above), so
    the table is rebuilt (same shape as
    db/migrations/024_pose_observations_source.sql). Existing rows
    backfill their new shot_id/trial_id/time_start_s/time_end_s from
    their (until now, 1:1) owning detection_runs row -- a reasonable
    default since today's relationship means the owning run's own range
    *is* the segmentation's range in every existing session. A
    seg_quality_runs row whose detection_run_id no longer resolves (should
    not happen given detection runs are append-only/never deleted, but not
    airtight-verified across every existing session -- see the design
    doc's own open question 5) is dropped by the join rather than kept
    with a fabricated range.

    seg_quality_runs itself may not exist at all yet: unlike every other
    table this table's own introduction was never given its own numbered
    migration (only added to session_schema.sql for fresh sessions), so a
    session that has been incrementally migrated all the way from an old
    schema version genuinely never got it created (confirmed via
    test_session_v2_migrates_to_v3, a v2-schema session run through the
    full migration chain to current). Handled here rather than filed as
    yet another gap: if the table doesn't exist, just create it directly
    in its v42 shape -- nothing to rebuild or backfill.
    """
    existing_tables = {
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    target_ddl = """
        CREATE TABLE seg_quality_runs (
            id             TEXT PRIMARY KEY,
            shot_id        TEXT NOT NULL REFERENCES captures(id),
            trial_id       TEXT REFERENCES trials(id),
            time_start_s   REAL NOT NULL,
            time_end_s     REAL NOT NULL,
            created_at     TEXT NOT NULL,
            quality_source TEXT NOT NULL DEFAULT 'cutie',
            erosion_px     INTEGER NOT NULL DEFAULT 5,
            mask_dir       TEXT,
            notes          TEXT
        );
    """
    if "seg_quality_runs" not in existing_tables:
        conn.executescript(target_ddl)
        _set_schema_version(conn, 42)
        conn.commit()
        return

    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(seg_quality_runs)")}
    if "shot_id" in existing_cols:
        _set_schema_version(conn, 42)
        conn.commit()
        return
    # seg_masks.seg_quality_run_id REFERENCES seg_quality_runs(id), and any
    # real session has seg_masks rows -- with PRAGMA foreign_keys=ON (the
    # default for every connection this codebase opens), DROP TABLE below
    # raises "FOREIGN KEY constraint failed" the moment a child row still
    # points at it, before the RENAME below ever gets a chance to restore
    # the relationship under the same table name. Same off/on-around-the-
    # rebuild pattern trial_export.py/registry.py already use for bulk
    # copies; can't be set mid-transaction, so it brackets the whole
    # executescript call rather than living inside its own BEGIN/COMMIT.
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.executescript(
        "BEGIN;\n"
        + target_ddl.replace("seg_quality_runs", "seg_quality_runs_new")
        + """
        INSERT INTO seg_quality_runs_new
            (id, shot_id, trial_id, time_start_s, time_end_s, created_at,
             quality_source, erosion_px, mask_dir, notes)
        SELECT sqr.id, dr.shot_id, dr.trial_id, dr.time_start_s, dr.time_end_s,
               sqr.created_at, sqr.quality_source, sqr.erosion_px, sqr.mask_dir, sqr.notes
        FROM seg_quality_runs sqr
        JOIN detection_runs dr ON dr.id = sqr.detection_run_id;
        DROP TABLE seg_quality_runs;
        ALTER TABLE seg_quality_runs_new RENAME TO seg_quality_runs;
        COMMIT;
        """
    )
    conn.execute("PRAGMA foreign_keys = ON")
    _set_schema_version(conn, 42)
    conn.commit()


def _migrate_session_v42_to_v43(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 42 to 43.

    v43 adds seg_quality_runs.persons_json -- a JSON array of person
    names, index i = mask label i+1 (same "ordered string list as JSON in
    one TEXT column" convention tracking_runs.marker_names/
    active_camera_ids already use). Captures the ordinal->name mapping
    baked into a segmentation's own mask labels at creation time, so a
    *different* caller reusing an existing segmentation later (gap 2,
    RunDetectionDialog's "use existing segmentation" bbox source) doesn't
    have to assume today's capture_persons order still matches whatever
    order was in effect when the masks were made. See docs/roadmap/
    features/segmentation-reuse/segmentation-reuse-design.md. NULL for
    existing rows -- there's no way to recover what order was actually
    used after the fact; callers fall back to today's capture_persons
    order for those (posetrak.db.manage_person.persons_ordered_for_seg_run).
    """
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(seg_quality_runs)")}
    if "persons_json" not in existing_cols:
        conn.execute("ALTER TABLE seg_quality_runs ADD COLUMN persons_json TEXT")
    _set_schema_version(conn, 43)
    conn.commit()


def _migrate_session_v43_to_v44(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 43 to 44.

    v44 adds seg_quality_runs.name -- a user-settable display name,
    distinct from the existing free-text notes column. Needed so
    segmentations can be listed and renamed in the session tree the same
    way detection_runs/tracking_runs already are. See
    docs/roadmap/features/segmentation-ui-improvements/
    segmentation-ui-improvements-design.md, Issue 1. NULL for existing
    rows and any row the user hasn't renamed -- callers fall back to a
    generated label (covered time range + created_at) when NULL.
    """
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(seg_quality_runs)")}
    if "name" not in existing_cols:
        conn.execute("ALTER TABLE seg_quality_runs ADD COLUMN name TEXT")
    _set_schema_version(conn, 44)
    conn.commit()


def _migrate_session_v44_to_v45(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 44 to 45.

    v45 adds capture_segmentation_hints -- user-marked "split points":
    frames where propagating a single Cutie tracking pass through would
    likely diverge (e.g. two people crossing paths), so segmentation
    should instead be seeded independently on each side. Deliberately a
    capture-scoped table, not a column on seg_quality_runs: a hard moment
    is a property of the footage itself, not of any one attempt at
    segmenting it, so it must survive a segmentation being deleted or
    superseded (segmentation-ui-improvements design doc, Issue 4).
    time_s is global time (consistent with trials/seg_quality_runs'
    own ranges) -- "two people cross paths here" is one synchronized
    real-world event and has to mean the same instant on every camera.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS capture_segmentation_hints ("
        "    id         TEXT PRIMARY KEY,"
        "    capture_id TEXT NOT NULL REFERENCES captures(id),"
        "    time_s     REAL NOT NULL,"
        "    note       TEXT,"
        "    created_at TEXT NOT NULL"
        ")"
    )
    _set_schema_version(conn, 45)
    conn.commit()


def _migrate_session_v45_to_v46(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 45 to 46.

    v46 adds capture_segmentation_hints.shot_video_id: split points were
    originally modeled capture-wide (one moment shared across every
    camera), but real use immediately showed the whole point of a split
    is usually camera-angle-dependent -- two people can occlude each
    other from one camera's viewpoint at a moment they're clearly
    separated in another's parallax. Nullable rather than NOT NULL: any
    row from the brief window v45 was live has no camera to attribute it
    to and is left as an orphan (harmless -- every real query filters by
    shot_video_id, so a NULL row just never matches and needs
    recreating) rather than guessing or deleting outright.
    """
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(capture_segmentation_hints)")}
    if "shot_video_id" not in existing_cols:
        conn.execute(
            "ALTER TABLE capture_segmentation_hints ADD COLUMN shot_video_id TEXT"
        )
    _set_schema_version(conn, 46)
    conn.commit()


def _migrate_session_v46_to_v47(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 46 to 47.

    v47 adds detection_runs.detector_type and .config_json, the first
    piece of the marker-based-mocap data model (design §4.1): a marker
    detection run (initially ArUco) is a `detection_runs` row like any pose-detection
    run, distinguished by `detector_type` rather than a parallel table, so
    every existing run/status/provenance query keeps working unchanged.
    `detector_type` defaults to 'pose' so every pre-existing row is
    correctly classified with no backfill. `config_json` carries detector
    parameters plus, for coded markers, the `marker_ids` list that fixes
    the corner-blob decode order in detection_keypoints (one physical
    prop's corner slots are ordered list-position-major by this list, per
    marker-mocap-design.md §4.1) -- NULL for existing (pose) runs, which
    have no such config today.
    """
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(detection_runs)")}
    if "detector_type" not in existing_cols:
        conn.execute(
            "ALTER TABLE detection_runs ADD COLUMN detector_type TEXT NOT NULL DEFAULT 'pose'"
        )
    if "config_json" not in existing_cols:
        conn.execute("ALTER TABLE detection_runs ADD COLUMN config_json TEXT")
    _set_schema_version(conn, 47)
    conn.commit()


def _migrate_session_v47_to_v48(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 47 to 48.

    v48 adds the marker-based-mocap object data model (design §4.2):

    - `capture_objects`: the object analog of `capture_persons` -- a
      physical prop instance participating in a capture, linking to its
      registry `marker_body_definitions` row. Carries no geometry itself
      (that lives in the definition); adding one to a capture is the same
      kind of action as adding a named performer.
    - `tracking_run_persons.capture_object_id`: makes a tracking run's
      object subjects explicit rather than a convention (design §4.2);
      set when a subject of the run is an object rather than a person.
    - `detection_runs.capture_object_id`: the same "make it explicit"
      reasoning applied to the detection layer, which the design doc
      doesn't spell out directly -- needed because a marker detection
      run's `config_json.marker_ids` alone can't disambiguate two
      `capture_objects` rows that reference the *same* marker body
      definition (e.g. two physically-identical props in one capture).
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS capture_objects ("
        "    id                         TEXT PRIMARY KEY,"
        "    capture_id                 TEXT NOT NULL REFERENCES captures(id),"
        "    name                       TEXT NOT NULL,"
        "    marker_body_definition_id  TEXT NOT NULL REFERENCES marker_body_definitions(id),"
        "    notes                      TEXT,"
        "    created_at                 TEXT NOT NULL"
        ")"
    )
    tr_persons_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_run_persons)")}
    if "capture_object_id" not in tr_persons_cols:
        conn.execute(
            "ALTER TABLE tracking_run_persons ADD COLUMN capture_object_id "
            "TEXT REFERENCES capture_objects(id)"
        )
    detection_runs_cols = {row[1] for row in conn.execute("PRAGMA table_info(detection_runs)")}
    if "capture_object_id" not in detection_runs_cols:
        conn.execute(
            "ALTER TABLE detection_runs ADD COLUMN capture_object_id "
            "TEXT REFERENCES capture_objects(id)"
        )
    _set_schema_version(conn, 48)
    conn.commit()


def _migrate_session_v48_to_v49(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 48 to 49.

    v49 adds `pose_sequence_keypoints` (design §4.3), the keypoint-manifest
    table sketched as `docs/data-model-and-storage.md` §3's extensibility
    seam: a sequence *without* manifest rows keeps today's implied-by-
    `pose_model` layout (fully backward compatible -- every existing
    person sequence needs no backfill), while a sequence *with* manifest
    rows (first used by marker-based-mocap object sequences)
    declares its own per-slot name and source, e.g. `hilt:c0`..`c3` /
    'aruco'. Landmark names derive from the marker body definition, so the
    same physical prop always yields the same names regardless of which
    ArUco ids it happens to carry.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pose_sequence_keypoints ("
        "    sequence_id   TEXT NOT NULL REFERENCES pose_observation_sequences(id),"
        "    keypoint_idx  INTEGER NOT NULL,"
        "    name          TEXT NOT NULL,"
        "    source        TEXT NOT NULL,"
        "    PRIMARY KEY (sequence_id, keypoint_idx)"
        ")"
    )
    _set_schema_version(conn, 49)
    conn.commit()


def _migrate_session_v49_to_v50(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 49 to 50.

    v50 adds streak-derived dot velocity's tuning columns to tracker_configs,
    mirroring the registry schema v8->v9 change -- see
    docs/roadmap/features/marker-based-mocap/streak-velocity-design.md §3/§4.
    """
    existing = _tracker_config_columns(conn)
    if "dot_streak_velocity_enabled" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_streak_velocity_enabled INTEGER")
    if "dot_streak_k_window" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_streak_k_window INTEGER")
    if "dot_streak_k_min_samples" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_streak_k_min_samples INTEGER")
    if "dot_streak_min_displacement_px" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_streak_min_displacement_px REAL")
    if "dot_streak_min_elongation_px" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_streak_min_elongation_px REAL")
    if "dot_streak_velocity_noise_std" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_streak_velocity_noise_std REAL")
    _set_schema_version(conn, 50)
    conn.commit()


def _migrate_session_v50_to_v51(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 50 to 51.

    v51 adds process_noise_vel_max_multiplier to tracker_configs, mirroring the
    registry schema v9->v10 change -- see
    docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md.
    """
    existing = _tracker_config_columns(conn)
    if "process_noise_vel_max_multiplier" not in existing:
        conn.execute(
            "ALTER TABLE tracker_configs ADD COLUMN process_noise_vel_max_multiplier REAL"
        )
    _set_schema_version(conn, 51)
    conn.commit()


def _migrate_session_v51_to_v52(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 51 to 52.

    v52 adds the dot-candidate assignment gate and its tracklet-aware
    relaxation to tracker_configs, mirroring the registry schema v10->v11
    change. dot_assignment_gate_mahalanobis existed as a
    TOML-only tunable before this migration; it had no DB column at all, so
    a DB-driven tracker_config row could never actually override it -- fixed
    here alongside the new tracklet multiplier since both touch the same
    assignment gate.
    """
    existing = _tracker_config_columns(conn)
    if "dot_assignment_gate_mahalanobis" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_assignment_gate_mahalanobis REAL")
    if "dot_tracklet_gate_multiplier" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_tracklet_gate_multiplier REAL")
    _set_schema_version(conn, 52)
    conn.commit()


def _migrate_session_v52_to_v53(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 52 to 53.

    v53 adds an experimental per-marker confidence-threshold override to
    tracker_configs: confidence_threshold_marker_names (JSON string array)
    names which markers are gated by confidence_threshold_override instead
    of the existing global min_confidence/measurement_noise_std path.
    Motivated by nose/ear.L/ear.R, which keep passing the normal confidence
    gate even when occluded by the back of the head, because occlusion only
    lowers ViTPose's reported confidence, it doesn't zero it out.
    """
    existing = _tracker_config_columns(conn)
    if "confidence_threshold_marker_names" not in existing:
        conn.execute(
            "ALTER TABLE tracker_configs ADD COLUMN confidence_threshold_marker_names TEXT"
        )
    if "confidence_threshold_override" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN confidence_threshold_override REAL")
    _set_schema_version(conn, 53)
    conn.commit()


def _migrate_session_v53_to_v54(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 53 to 54.

    v54 adds a reacquisition gate to the shared dot-assignment phase:
    dot_reacquire_gap_frames (consecutive unresolved steps before a slot
    needs independent evidence to resolve again) and dot_reacquire_max_px
    (the cross-camera "near the prediction" radius that evidence is judged
    against). See ReacquisitionGateModifier (dot_assignment.hpp). Motivated by
    the twice-observed "confidently wrong candidate" failure (ball tracking,
    leg-marker ankle reacquisition) that a per-frame Mahalanobis gate alone
    cannot distinguish from a genuine reacquisition.
    """
    existing = _tracker_config_columns(conn)
    if "dot_reacquire_gap_frames" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_reacquire_gap_frames INTEGER")
    if "dot_reacquire_max_px" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_reacquire_max_px REAL")
    _set_schema_version(conn, 54)
    conn.commit()


def _migrate_session_v54_to_v55(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 54 to 55.

    v55 adds cross-view corroboration and per-camera trust to the shared
    dot-assignment phase: dot_cross_view_corroboration_px (epipolar tolerance
    for CrossViewCorroborationModifier), dot_corroboration_gate_multiplier
    (its cost discount), and dot_camera_noise_scale (a per-camera factor on
    resolved dot observations' measurement noise, keyed by the tracker's own
    integer camera_id). See dot_assignment.hpp and TrackerConfig's own field
    doc comments (config.hpp).
    """
    existing = _tracker_config_columns(conn)
    if "dot_cross_view_corroboration_px" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_cross_view_corroboration_px REAL")
    if "dot_corroboration_gate_multiplier" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_corroboration_gate_multiplier REAL")
    if "dot_camera_noise_scale" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_camera_noise_scale TEXT")
    _set_schema_version(conn, 55)
    conn.commit()


def _migrate_session_v55_to_v56(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 55 to 56.

    v56 adds tracking_assignment_trace: every (candidate, marker slot) pair
    the shared dot-assignment phase costed for a frame, not just the winner
    tracking_obs_results records -- the "why was detection N (not) assigned
    to slot X in frame F" instrumentation. See dot_assignment.hpp's
    AssignmentTrace and 027_tracking_assignment_trace.sql.
    """
    sql = (_DB_DIR / "migrations" / "027_tracking_assignment_trace.sql").read_text(
        encoding="utf-8"
    )
    conn.executescript(sql)


def _migrate_session_v56_to_v57(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 56 to 57.

    v57 adds segmentation-mask-based dot-assignment trust to tracker_configs.
    Every session database embeds its own full copy of the registry tables,
    including tracker_configs, so it needs this same column addition kept in
    sync -- see create_session()'s own docstring. Mirrors the registry schema
    v14->v15 change -- see SegMaskSubjectConfig and TrackerConfig's own
    dot_seg_mask_* field doc comments (dot_assignment.hpp / config.hpp).
    """
    existing = _tracker_config_columns(conn)
    if "dot_seg_mask_enabled" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_seg_mask_enabled INTEGER")
    if "dot_seg_mask_dilate_px" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_seg_mask_dilate_px REAL")
    if "dot_seg_mask_tracklet_window" not in existing:
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_seg_mask_tracklet_window INTEGER")
    if "dot_seg_mask_min_tracklet_fraction" not in existing:
        conn.execute(
            "ALTER TABLE tracker_configs ADD COLUMN dot_seg_mask_min_tracklet_fraction REAL"
        )
    if "dot_seg_mask_cost_scale_at_zero_fraction" not in existing:
        conn.execute(
            "ALTER TABLE tracker_configs ADD COLUMN dot_seg_mask_cost_scale_at_zero_fraction REAL"
        )
    _set_schema_version(conn, 57)
    conn.commit()


def _migrate_session_v60_to_v61(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 60 to 61.

    v61 adds tracker_configs.dot_seg_mask_hard_distance_px: a candidate farther
    than this from its subject's mask label in the current frame cannot be
    assigned to the subject's slots, whatever its tracklet's history. Mirrors
    the registry schema v15->v16 change, since every session embeds a copy of
    tracker_configs.
    """
    if "dot_seg_mask_hard_distance_px" not in _tracker_config_columns(conn):
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN dot_seg_mask_hard_distance_px REAL")
    _set_schema_version(conn, 61)
    conn.commit()


def _migrate_session_v61_to_v62(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 61 to 62.

    v62 adds dot_assignment_proposals (automatic dot tracklet to marker slot
    proposals) and dot_assignment_edits (human assign/forbid/reject decisions
    on them, per trial). See 029_dot_assignment.sql.
    """
    sql = (_DB_DIR / "migrations" / "029_dot_assignment.sql").read_text(encoding="utf-8")
    conn.executescript(sql)


def _migrate_session_v62_to_v63(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 62 to 63.

    v63 adds tracking_runs.name, a display name the user gives a run to find it again
    (NULL until named).
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tracking_runs)")}
    if "name" not in columns:
        conn.execute("ALTER TABLE tracking_runs ADD COLUMN name TEXT")
    _set_schema_version(conn, 63)
    conn.commit()


def _migrate_session_v63_to_v64(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 63 to 64.

    v64 adds tracking_runs.status / completed_at / failure (existing runs are
    'complete'), the segmentation columns of tracking_run_persons (left NULL) and
    the run_progress table that dates the newest activity of a running job.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tracking_runs)")}
    if "status" not in columns:
        conn.execute(
            "ALTER TABLE tracking_runs ADD COLUMN status TEXT NOT NULL DEFAULT 'complete'"
            " CHECK (status IN ('running', 'complete', 'failed'))"
        )
    if "completed_at" not in columns:
        conn.execute("ALTER TABLE tracking_runs ADD COLUMN completed_at TEXT")
    if "failure" not in columns:
        conn.execute("ALTER TABLE tracking_runs ADD COLUMN failure TEXT")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tracking_run_persons)")}
    for name, decl in (
        ("seg_quality_run_id", "TEXT REFERENCES seg_quality_runs(id)"),
        ("seg_mask_label", "INTEGER"),
        ("seg_revision", "INTEGER"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE tracking_run_persons ADD COLUMN {name} {decl}")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS run_progress ("
        " run_id TEXT PRIMARY KEY, updated_at TEXT NOT NULL)"
    )
    _set_schema_version(conn, 64)
    conn.commit()


def _migrate_session_v64_to_v65(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 64 to 65.

    v65 adds pose_sequence_keypoints.obs_source and makes it part of the primary key, so a
    sequence can name the slots of each of its observation sources (a SOMA body blob and the
    hand21 blobs of refined hands). Existing rows get 'body'. See
    030_pose_sequence_keypoints_obs_source.sql.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(pose_sequence_keypoints)")}
    if "obs_source" not in columns:
        sql = (_DB_DIR / "migrations" / "030_pose_sequence_keypoints_obs_source.sql").read_text(encoding="utf-8")
        conn.executescript(sql)
    _set_schema_version(conn, 65)
    conn.commit()


def _migrate_session_v65_to_v66(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 65 to 66.

    v66 adds the run inputs of detection_runs: seg_quality_run_id, seg_revision and
    source_detection_run_id. They stay NULL for existing runs, except that a run whose
    config_json names ``source_detection_run`` (the pose-model spike's prototype runs) gets
    that run in source_detection_run_id when it exists.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(detection_runs)")}
    for name, decl in (
        ("seg_quality_run_id", "TEXT REFERENCES seg_quality_runs(id)"),
        ("seg_revision", "INTEGER"),
        ("source_detection_run_id", "TEXT REFERENCES detection_runs(id)"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE detection_runs ADD COLUMN {name} {decl}")
    conn.execute(
        "UPDATE detection_runs SET source_detection_run_id = json_extract(config_json, '$.source_detection_run')"
        " WHERE source_detection_run_id IS NULL AND config_json IS NOT NULL AND json_valid(config_json)"
        " AND json_extract(config_json, '$.source_detection_run') IN (SELECT id FROM detection_runs)"
    )
    _set_schema_version(conn, 66)
    conn.commit()


def _migrate_session_v66_to_v67(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 66 to 67.

    v67 adds tracker_configs.active_joint_groups: the joint groups the tracker's
    main (parent) stage tracks, so that a hierarchical configuration states the
    form it was made for (parent stage restricted to its group) instead of
    depending on a command-line option. NULL keeps the previous behaviour, the
    whole skeleton. Mirrors the registry schema v16->v17 change, since every
    session embeds a copy of tracker_configs.
    """
    if "active_joint_groups" not in _tracker_config_columns(conn):
        conn.execute("ALTER TABLE tracker_configs ADD COLUMN active_joint_groups TEXT")
    _set_schema_version(conn, 67)
    conn.commit()


def _migrate_session_v67_to_v68(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 67 to 68.

    v68 makes segmentation masks append-only by revision: seg_mask_revisions, seg_masks.revision
    in the primary key (mask_blob nullable), the seg_masks_latest view, and a trigger giving every
    new segmentation run a closed revision 0. SQLite cannot change a primary key in place, so
    seg_masks is rebuilt: every row is copied once with revision 0, run by run, reporting progress
    through the ``posetrak.db.db`` logger. The whole rebuild is one transaction. See
    031_seg_mask_revisions.sql and the segmentation versioning design.
    """
    if "revision" not in {row[1] for row in conn.execute("PRAGMA table_info(seg_masks)")}:
        sql = (_DB_DIR / "migrations" / "031_seg_mask_revisions.sql").read_text(encoding="utf-8")
        before, after = sql.split("-- @@ after copy")
        # The rebuild drops seg_masks while seg_quality_runs rows point at it; foreign-key
        # enforcement cannot change inside a transaction, so it brackets the whole rebuild.
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute("BEGIN")
            for stmt in _split_statements(before):
                conn.execute(stmt)
            runs = conn.execute("SELECT id, created_at FROM seg_quality_runs ORDER BY created_at").fetchall()
            for i, (run_id, created_at) in enumerate(runs, 1):
                conn.execute(
                    "INSERT INTO seg_mask_revisions (seg_quality_run_id, revision, created_at, closed_at)"
                    " VALUES (?, 0, ?, ?)",
                    (run_id, created_at, created_at),
                )
                n = conn.execute(
                    "INSERT INTO seg_masks_new (seg_quality_run_id, shot_video_id, frame_idx, revision, mask_blob)"
                    " SELECT seg_quality_run_id, shot_video_id, frame_idx, 0, mask_blob FROM seg_masks"
                    " WHERE seg_quality_run_id = ?",
                    (run_id,),
                ).rowcount
                _log.info("seg_masks migration: run %d/%d %s, %d masks copied", i, len(runs), run_id, n)
            conn.execute("DROP TABLE seg_masks")
            conn.execute("ALTER TABLE seg_masks_new RENAME TO seg_masks")
            for stmt in _split_statements(after):
                conn.execute(stmt)
            _set_schema_version(conn, 68)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.execute("PRAGMA foreign_keys = ON")
    else:
        _set_schema_version(conn, 68)
        conn.commit()


def _migrate_session_v68_to_v69(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 68 to 69.

    v69 adds tracker_configs.kp_candidates_enabled: whether a tracking run takes the keypoint candidates of its
    sequence (pose_observations rows with source 'kp_candidates') instead of the fixed body observations for the
    markers bound to that pose model. NULL keeps the previous behaviour, the fixed observations. Mirrors the
    registry schema v17->v18 change, since every session embeds a copy of tracker_configs. It also adds
    kp_candidate_peak_weight and kp_candidate_tracklet_multiplier, the settings of the assignment cost of those
    candidates (NULL: 0.0 and 1.0, the cost without a peak term and without tracklet continuity).
    """
    _add_kp_candidate_config_columns(conn)
    _set_schema_version(conn, 69)
    conn.commit()


def _migrate_session_v69_to_v70(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 69 to 70.

    v70 adds the tracker_configs settings of the windowed choice of keypoint candidates (mirroring registry
    v18->v19) and the tracking_kp_picks table, in which a run that chose candidates over a time window records
    the candidate it picked for each keypoint of each frame. A configuration from before keeps NULL in the new
    columns, which means the per-step assignment ('step').
    """
    _add_kp_choice_config_columns(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tracking_kp_picks (
            run_id                      TEXT    NOT NULL REFERENCES tracking_runs(id),
            person_id                   INTEGER NOT NULL,
            camera_instance_id          TEXT    NOT NULL,
            n_frames                    INTEGER NOT NULL,
            n_keypoints                 INTEGER NOT NULL,
            video_frames                BLOB    NOT NULL,
            tracklet_ids                BLOB    NOT NULL,
            ranks                       BLOB    NOT NULL,
            n_candidate_keypoint_frames INTEGER NOT NULL,
            n_filter_emptied            INTEGER NOT NULL,
            n_picked                    INTEGER NOT NULL,
            PRIMARY KEY (run_id, person_id, camera_instance_id)
        )
        """
    )
    _set_schema_version(conn, 70)
    conn.commit()


def _split_statements(sql: str) -> list[str]:
    """Split a migration script into complete statements (triggers keep their ``;`` inside BEGIN..END)."""
    out, buf = [], ""
    for line in sql.splitlines(keepends=True):
        if line.lstrip().startswith("--"):
            continue
        buf += line
        if sqlite3.complete_statement(buf):
            out.append(buf.strip())
            buf = ""
    return out


def _migrate_session_v57_to_v58(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 57 to 58.

    v58 adds capture_subjects (a capture-scoped registry giving every person
    and object a permanent integer label, used as its segmentation-mask
    label), the triggers that keep it in step with capture_persons/
    capture_objects, and seg_quality_run_subjects (exception rows for
    segmentations whose labels differ from the registry labels). Existing
    subjects are backfilled. See 028_capture_subjects.sql and
    docs/roadmap/features/subject-identity/subject-identity-design.md.
    """
    sql = (_DB_DIR / "migrations" / "028_capture_subjects.sql").read_text(encoding="utf-8")
    conn.executescript(sql)


def _migrate_session_v58_to_v59(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 58 to 59.

    v59 adds a nullable ``subject_id`` (a capture_subjects.id) to
    sequence_persons, tracking_run_persons and detection_track_assignments,
    so a sequence, a tracked run and a stitched track each point at the
    capture's subject registry instead of matching names. Existing rows are
    backfilled: from capture_person_id where set (a person's registry id is
    its capture_persons id), else by name within the capture; a run's
    subject is its tracked object or, for a person, its sequence's person.
    Tracking runs made before the tracker recorded tracking_run_persons get
    that row too. See docs/roadmap/features/subject-identity/
    subject-identity-design.md.
    """
    for table in ("sequence_persons", "tracking_run_persons", "detection_track_assignments"):
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if "subject_id" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN subject_id TEXT")
    conn.executescript(
        """
        UPDATE sequence_persons SET subject_id = (
            SELECT cs.id FROM capture_subjects cs WHERE cs.id = sequence_persons.capture_person_id)
        WHERE subject_id IS NULL AND capture_person_id IS NOT NULL;
        UPDATE sequence_persons SET subject_id = (
            SELECT cs.id FROM capture_subjects cs
            JOIN pose_observation_sequences pos ON pos.shot_id = cs.capture_id
            WHERE pos.id = sequence_persons.sequence_id
              AND cs.name = sequence_persons.person_name AND cs.deleted = 0)
        WHERE subject_id IS NULL;

        UPDATE detection_track_assignments SET subject_id = (
            SELECT cs.id FROM capture_subjects cs
            WHERE cs.id = detection_track_assignments.capture_person_id)
        WHERE subject_id IS NULL AND capture_person_id IS NOT NULL;
        UPDATE detection_track_assignments SET subject_id = (
            SELECT cs.id FROM capture_subjects cs
            JOIN detection_runs dr ON dr.shot_id = cs.capture_id
            WHERE dr.id = detection_track_assignments.detection_run_id
              AND cs.name = detection_track_assignments.person_name AND cs.deleted = 0)
        WHERE subject_id IS NULL;

        INSERT OR IGNORE INTO tracking_run_persons
            (run_id, person_id, skeleton_id, capture_object_id, subject_id)
        SELECT tr.id, res.person_id, tr.skeleton_id, dr.capture_object_id,
               COALESCE(dr.capture_object_id,
                        (SELECT sp.subject_id FROM sequence_persons sp
                         WHERE sp.sequence_id = tr.observation_sequence_id
                           AND sp.person_id = res.person_id))
        FROM tracking_runs tr
        JOIN (SELECT DISTINCT run_id, person_id FROM tracking_results) res
             ON res.run_id = tr.id
        LEFT JOIN pose_observation_sequences pos ON pos.id = tr.observation_sequence_id
        LEFT JOIN detection_runs dr ON dr.id = pos.detection_run_id;

        UPDATE tracking_run_persons SET subject_id = capture_object_id
        WHERE subject_id IS NULL AND capture_object_id IS NOT NULL;
        UPDATE tracking_run_persons SET subject_id = (
            SELECT sp.subject_id FROM sequence_persons sp
            WHERE sp.sequence_id = (SELECT observation_sequence_id FROM tracking_runs
                                    WHERE id = tracking_run_persons.run_id)
              AND sp.person_id = tracking_run_persons.person_id)
        WHERE subject_id IS NULL;
        """
    )
    _set_schema_version(conn, 59)
    conn.commit()


def _migrate_session_v59_to_v60(conn: sqlite3.Connection) -> None:
    """Migrate a session database from schema version 59 to 60.

    v60 retires ``seg_quality_runs.persons_json``: label meanings now come
    only from the subject registry and ``seg_quality_run_subjects``. So every
    name still recorded in a persons_json list must be reachable from there.
    For each entry (index i = mask label i+1) the subject is the capture's
    live subject of that name, else its most recently deleted one, else a
    person created now for the name (which gets the next registry label, so
    its old mask label needs an exception row). An exception row is written
    unless the subject is live and already has that label. Existing exception
    rows are kept. The column itself stays, unused. See
    docs/roadmap/features/subject-identity/subject-identity-design.md.
    """
    import datetime
    import json

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    runs = conn.execute(
        "SELECT id, shot_id, persons_json FROM seg_quality_runs WHERE persons_json IS NOT NULL"
    ).fetchall()
    for run_id, capture_id, persons_json in runs:
        try:
            names = json.loads(persons_json)
        except ValueError:
            continue
        for index, name in enumerate(names):
            if not name:
                continue
            label = index + 1
            row = conn.execute(
                "SELECT id, label, deleted FROM capture_subjects WHERE capture_id = ? AND name = ? "
                "ORDER BY deleted, label DESC LIMIT 1",
                (capture_id, name),
            ).fetchone()
            if row is None:
                subject_id = generate_id()
                conn.execute(
                    "INSERT INTO capture_persons (id, capture_id, name, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (subject_id, capture_id, name, now),
                )
                row = conn.execute(
                    "SELECT id, label, deleted FROM capture_subjects WHERE id = ?", (subject_id,)
                ).fetchone()
            subject_id, subject_label, deleted = row
            if deleted or subject_label != label:
                conn.execute(
                    "INSERT OR IGNORE INTO seg_quality_run_subjects "
                    "(seg_quality_run_id, label, subject_id) VALUES (?, ?, ?)",
                    (run_id, label, subject_id),
                )
    _set_schema_version(conn, 60)
    conn.commit()


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
        actual = 6
    if actual == 6:
        _migrate_session_v6_to_v7(conn)
        actual = 7
    if actual == 7:
        _migrate_session_v7_to_v8(conn)
        actual = 8
    if actual == 8:
        _migrate_session_v8_to_v9(conn)
        actual = 9
    if actual == 9:
        _migrate_session_v9_to_v10(conn)
        actual = 10
    if actual == 10:
        _migrate_session_v10_to_v11(conn)
        actual = 11
    if actual == 11:
        _migrate_session_v11_to_v12(conn)
        actual = 12
    if actual == 12:
        _migrate_session_v12_to_v13(conn)
        actual = 13
    if actual == 13:
        _migrate_session_v13_to_v14(conn)
        actual = 14
    if actual == 14:
        _migrate_session_v14_to_v15(conn)
        actual = 15
    if actual == 15:
        _migrate_session_v15_to_v16(conn)
        actual = 16
    if actual == 16:
        _migrate_session_v16_to_v17(conn)
        actual = 17
    if actual == 17:
        _migrate_session_v17_to_v18(conn)
        actual = 18
    if actual == 18:
        _migrate_session_v18_to_v19(conn)
        actual = 19
    if actual == 19:
        _migrate_session_v19_to_v20(conn)
        actual = 20
    if actual == 20:
        _migrate_session_v20_to_v21(conn)
        actual = 21
    if actual == 21:
        _migrate_session_v21_to_v22(conn)
        actual = 22
    if actual == 22:
        _migrate_session_v22_to_v23(conn)
        actual = 23
    if actual == 23:
        _migrate_session_v23_to_v24(conn)
        actual = 24
    if actual == 24:
        _migrate_session_v24_to_v25(conn)
        actual = 25
    if actual == 25:
        _migrate_session_v25_to_v26(conn)
        actual = 26
    if actual == 26:
        _migrate_session_v26_to_v27(conn)
        actual = 27
    if actual == 27:
        _migrate_session_v27_to_v28(conn)
        actual = 28
    if actual == 28:
        _migrate_session_v28_to_v29(conn)
        actual = 29
    if actual == 29:
        _migrate_session_v29_to_v30(conn)
        actual = 30
    if actual == 30:
        _migrate_session_v30_to_v31(conn)
        actual = 31
    if actual == 31:
        _migrate_session_v31_to_v32(conn)
        actual = 32
    if actual == 32:
        _migrate_session_v32_to_v33(conn)
        actual = 33
    if actual == 33:
        _migrate_session_v33_to_v34(conn)
        actual = 34
    if actual == 34:
        _migrate_session_v34_to_v35(conn)
        actual = 35
    if actual == 35:
        _migrate_session_v35_to_v36(conn)
        actual = 36
    if actual == 36:
        _migrate_session_v36_to_v37(conn)
        actual = 37
    if actual == 37:
        _migrate_session_v37_to_v38(conn)
        actual = 38
    if actual == 38:
        _migrate_session_v38_to_v39(conn)
        actual = 39
    if actual == 39:
        _migrate_session_v39_to_v40(conn)
        actual = 40
    if actual == 40:
        _migrate_session_v40_to_v41(conn)
        actual = 41
    if actual == 41:
        _migrate_session_v41_to_v42(conn)
        actual = 42
    if actual == 42:
        _migrate_session_v42_to_v43(conn)
        actual = 43
    if actual == 43:
        _migrate_session_v43_to_v44(conn)
        actual = 44
    if actual == 44:
        _migrate_session_v44_to_v45(conn)
        actual = 45
    if actual == 45:
        _migrate_session_v45_to_v46(conn)
        actual = 46
    if actual == 46:
        _migrate_session_v46_to_v47(conn)
        actual = 47
    if actual == 47:
        _migrate_session_v47_to_v48(conn)
        actual = 48
    if actual == 48:
        _migrate_session_v48_to_v49(conn)
        actual = 49
    if actual == 49:
        _migrate_session_v49_to_v50(conn)
        actual = 50
    if actual == 50:
        _migrate_session_v50_to_v51(conn)
        actual = 51
    if actual == 51:
        _migrate_session_v51_to_v52(conn)
        actual = 52
    if actual == 52:
        _migrate_session_v52_to_v53(conn)
        actual = 53
    if actual == 53:
        _migrate_session_v53_to_v54(conn)
        actual = 54
    if actual == 54:
        _migrate_session_v54_to_v55(conn)
        actual = 55
    if actual == 55:
        _migrate_session_v55_to_v56(conn)
        actual = 56
    if actual == 56:
        _migrate_session_v56_to_v57(conn)
        actual = 57
    if actual == 57:
        _migrate_session_v57_to_v58(conn)
        actual = 58
    if actual == 58:
        _migrate_session_v58_to_v59(conn)
        actual = 59
    if actual == 59:
        _migrate_session_v59_to_v60(conn)
        actual = 60
    if actual == 60:
        _migrate_session_v60_to_v61(conn)
        actual = 61
    if actual == 61:
        _migrate_session_v61_to_v62(conn)
        actual = 62
    if actual == 62:
        _migrate_session_v62_to_v63(conn)
        actual = 63
    if actual == 63:
        _migrate_session_v63_to_v64(conn)
        actual = 64
    if actual == 64:
        _migrate_session_v64_to_v65(conn)
        actual = 65
    if actual == 65:
        _migrate_session_v65_to_v66(conn)
        actual = 66
    if actual == 66:
        _migrate_session_v66_to_v67(conn)
        actual = 67
    if actual == 67:
        _migrate_session_v67_to_v68(conn)
        actual = 68
    if actual == 68:
        _migrate_session_v68_to_v69(conn)
        actual = 69
    if actual == 69:
        _migrate_session_v69_to_v70(conn)
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
        # camera_modes.default_intrinsics_calibration_id references
        # intrinsics_calibrations(id), which itself references back via
        # intrinsics_calibrations.camera_mode_id -- a genuine circular FK
        # between the two tables, not just an ordering problem. No
        # sequence of plain INSERTs satisfies both directions at once, so
        # defer FK enforcement to COMMIT (unlike "PRAGMA foreign_keys",
        # "PRAGMA defer_foreign_keys" is specifically designed to be
        # toggled mid-transaction -- see the comment near
        # seg_quality_runs's migration for why the two pragmas differ).
        # Real integrity violations still fail, just at COMMIT instead of
        # at the offending INSERT; SQLite resets this pragma to off
        # automatically once the transaction ends either way.
        session.execute("PRAGMA defer_foreign_keys = ON")
        # Copy dependency chain: camera_models → camera_modes/instances → intrinsics
        _copy_rows_if_missing(registry, session, "camera_models", [camera_model_id])
        _copy_rows_if_missing(registry, session, "camera_modes", [camera_mode_id])
        _copy_rows_if_missing(registry, session, "camera_instances", [camera_instance_id])
        _copy_rows_if_missing(
            registry, session, "intrinsics_calibrations", [intrinsics_calibration_id]
        )
        # session_cameras no longer stores mode/intrinsics (moved to shot_videos in v11).
        # camera_mode_id and intrinsics_calibration_id are still copied into the session
        # so the session stays self-contained; they are written to shot_videos by the caller.
        session.execute(
            "INSERT INTO session_cameras (session_id, camera_instance_id, label) "
            "VALUES (?, ?, ?)",
            (session_id, camera_instance_id, label),
        )


def create_capture(
    session: sqlite3.Connection,
    session_id: str,
    extrinsic_calibration_id: str | None = None,
    *,
    capture_number: int | None = None,
    label: str = "",
    notes: str = "",
) -> str:
    """Insert a captures row and return its ID.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    session_id:
        ID of the parent ``mocap_sessions`` row.
    extrinsic_calibration_id:
        ID of the ``extrinsic_calibrations`` row used for this capture.
    capture_number:
        Explicit capture number. If ``None``, auto-increments from the highest
        existing ``capture_number`` within this session (starting at 1).
    label:
        Optional short label for the capture.
    notes:
        Optional free-text notes.

    Returns
    -------
    str
        UUID of the newly created ``captures`` row.
    """
    if capture_number is None:
        row = session.execute(
            "SELECT MAX(capture_number) FROM captures WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        max_num = row[0]
        capture_number = 1 if max_num is None else max_num + 1

    capture_id = generate_id()
    with session:
        session.execute(
            "INSERT INTO captures "
            "(id, session_id, extrinsic_calibration_id, capture_number, label, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (capture_id, session_id, extrinsic_calibration_id, capture_number, label, notes),
        )
    return capture_id


# Backwards-compatible alias used by older call sites being migrated incrementally.
create_shot = create_capture


def add_capture_video(
    session: sqlite3.Connection,
    capture_id: str,
    camera_instance_id: str,
    file_path: str,
    first_frame: int,
    last_frame: int,
    fps: float,
) -> str:
    """Insert a capture_videos row and return its ID.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    capture_id:
        ID of the parent ``captures`` row.
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
        UUID of the newly created ``capture_videos`` row.
    """
    video_id = generate_id()
    with session:
        session.execute(
            "INSERT INTO capture_videos "
            "(id, shot_id, camera_instance_id, file_path, "
            "first_video_frame, last_video_frame, actual_fps) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (video_id, capture_id, camera_instance_id, file_path,
             first_frame, last_frame, fps),
        )
    return video_id


# Backwards-compatible alias.
add_shot_video = add_capture_video


def set_capture_extrinsics(
    session: sqlite3.Connection,
    capture_id: str,
    extrinsic_calibration_id: str,
) -> None:
    """Set or update the extrinsic_calibration_id on an existing capture.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    capture_id:
        ID of the ``captures`` row to update.
    extrinsic_calibration_id:
        ID of the ``extrinsic_calibrations`` row to link.
    """
    with session:
        session.execute(
            "UPDATE captures SET extrinsic_calibration_id = ? WHERE id = ?",
            (extrinsic_calibration_id, capture_id),
        )


# Backwards-compatible alias.
set_shot_extrinsics = set_capture_extrinsics


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

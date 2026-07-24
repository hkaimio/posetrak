"""posetrak_db.py — Core database access layer for the posetrak SQLite registry and session databases.

This module handles:
- Creating and opening registry databases (shared project-wide metadata: cameras, skeletons, configs).
- Creating and opening per-session databases (mocap sessions, captures, tracking results).
- Schema versioning via SQLite PRAGMA user_version.
- Utility helpers for project-root-relative path resolution.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Schema version constants
# ---------------------------------------------------------------------------

REGISTRY_SCHEMA_VERSION: Final[int] = 7
SESSION_SCHEMA_VERSION: Final[int] = 39

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
    from posetrak.db.manage_skeleton import seed_default_skeletons
    seed_default_skeletons(conn)
    from posetrak.db.manage_config import seed_baseline_tracker_config
    seed_baseline_tracker_config(conn)
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

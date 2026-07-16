"""Tests for scripts/db/posetrak_db.py public API."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest


from posetrak.db.db import (
    DEFAULT_REGISTRY_PATH,
    REGISTRY_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
    _copy_rows_if_missing,
    add_session_camera,
    create_camera_model,
    create_camera_mode,
    create_mocap_session,
    create_registry,
    create_session,
    generate_id,
    get_project_root,
    list_camera_models,
    list_camera_modes,
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


def test_default_registry_path_under_home() -> None:
    """DEFAULT_REGISTRY_PATH should be inside the user's home directory."""
    from pathlib import Path
    assert DEFAULT_REGISTRY_PATH == Path.home() / ".posetrak" / "registry.db"


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


def test_migrate_session_v34_to_v35_adds_source_to_pose_observations(tmp_path: Path) -> None:
    """v34→v35 rebuilds pose_observations with `source` in the PK, preserving data.

    Existing rows must survive as source='body' with detection_run_id
    backfilled from their parent sequence, and the row count must be
    unchanged (see db/migrations/024_pose_observations_source.sql).
    """
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)

    # Downgrade to the pre-migration (v34) pose_observations shape.
    conn.executescript("""
        BEGIN;
        CREATE TABLE pose_observations_old (
            sequence_id        TEXT    NOT NULL REFERENCES pose_observation_sequences(id),
            camera_instance_id TEXT    NOT NULL,
            video_frame        INTEGER NOT NULL,
            timestamp_s        REAL    NOT NULL,
            person_id          INTEGER NOT NULL,
            kp_blob            BLOB    NOT NULL,
            noise_scale        REAL,
            PRIMARY KEY (sequence_id, camera_instance_id, video_frame, person_id)
        );
        DROP TABLE pose_observations;
        ALTER TABLE pose_observations_old RENAME TO pose_observations;
        PRAGMA user_version = 34;
        COMMIT;
    """)

    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('shot1', 'sess1', 1)"
    )
    conn.execute(
        "INSERT INTO sync_configs (id, shot_id) VALUES ('sync1', 'shot1')"
    )
    conn.execute(
        "INSERT INTO detection_runs (id, shot_id, sync_config_id, time_start_s, time_end_s,"
        " detector_model, pose_model, status, created_at)"
        " VALUES ('run1', 'shot1', 'sync1', 0.0, 1.0, 'yolo', 'rtmpose', 'complete', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO pose_observation_sequences"
        " (id, shot_id, sync_config_id, time_start_s, time_end_s, detection_run_id,"
        "  pixels_are_undistorted)"
        " VALUES ('seq1', 'shot1', 'sync1', 0.0, 1.0, 'run1', 0)"
    )
    conn.execute(
        "INSERT INTO pose_observations"
        " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob,"
        "  noise_scale)"
        " VALUES ('seq1', 'ci1', 10, 0.1, 0, X'00', 1.5)"
    )
    conn.commit()
    conn.close()

    conn = open_session(db_path)
    assert get_schema_version(conn) == SESSION_SCHEMA_VERSION

    cols = {row[1] for row in conn.execute("PRAGMA table_info(pose_observations)")}
    assert {"source", "detection_run_id"} <= cols

    rows = conn.execute(
        "SELECT source, detection_run_id, noise_scale FROM pose_observations"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["source"] == "body"
    assert rows[0]["detection_run_id"] == "run1"
    assert rows[0]["noise_scale"] == 1.5
    conn.close()


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
    # session-specific tables
    "mocap_sessions",
    "session_cameras",
    "extrinsic_calibrations",
    "extrinsic_entries",
    "captures",
    "trials",
    "capture_videos",
    "sync_configs",
    "sync_points",
    "pose_observation_sequences",
    "pose_observations",
    "tracking_runs",
    "tracking_run_persons",
    "tracking_results",
    "tracking_obs_results",
    "person_detections",
    "person_tracks",
    "frame_cache_entries",
    # registry tables embedded in every session DB
    "camera_models",
    "camera_modes",
    "camera_instances",
    "intrinsics_calibrations",
    "skeletons",
    "tracker_configs",
}


def test_session_has_expected_tables(session_db: sqlite3.Connection) -> None:
    """The session schema should create all expected tables, including registry tables."""
    rows = session_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    actual = {row["name"] for row in rows}
    for table in _SESSION_TABLES:
        assert table in actual, f"Missing session table: {table!r}"


# ---------------------------------------------------------------------------
# Camera model management
# ---------------------------------------------------------------------------


def test_create_camera_model_returns_id(registry_db: sqlite3.Connection) -> None:
    """create_camera_model() should return a non-empty UUID string."""
    model_id = create_camera_model(registry_db, manufacturer="Acme", model_name="Cam X")
    assert model_id
    row = registry_db.execute(
        "SELECT manufacturer, model_name FROM camera_models WHERE id = ?", (model_id,)
    ).fetchone()
    assert row["manufacturer"] == "Acme"
    assert row["model_name"] == "Cam X"


def test_create_camera_model_optional_fields(registry_db: sqlite3.Connection) -> None:
    """create_camera_model() with no arguments should create a row with empty strings."""
    model_id = create_camera_model(registry_db)
    row = registry_db.execute(
        "SELECT id FROM camera_models WHERE id = ?", (model_id,)
    ).fetchone()
    assert row is not None


def test_list_camera_models_empty(registry_db: sqlite3.Connection) -> None:
    """list_camera_models() should return an empty list when no models are registered."""
    assert list_camera_models(registry_db) == []


def test_list_camera_models_returns_all(registry_db: sqlite3.Connection) -> None:
    """list_camera_models() should return one row per registered model."""
    create_camera_model(registry_db, model_name="Alpha")
    create_camera_model(registry_db, model_name="Beta")
    rows = list_camera_models(registry_db)
    assert len(rows) == 2
    names = {row["model_name"] for row in rows}
    assert names == {"Alpha", "Beta"}


# ---------------------------------------------------------------------------
# Camera mode management
# ---------------------------------------------------------------------------


def test_create_camera_mode_returns_id(registry_db: sqlite3.Connection) -> None:
    """create_camera_mode() should return a UUID and create a row with correct fields."""
    model_id = create_camera_model(registry_db, model_name="Cam")
    mode_id = create_camera_mode(
        registry_db, model_id, width_px=1920, height_px=1080, nominal_fps=60.0
    )
    assert mode_id
    row = registry_db.execute(
        "SELECT width_px, height_px, nominal_fps, camera_model_id "
        "FROM camera_modes WHERE id = ?",
        (mode_id,),
    ).fetchone()
    assert row["width_px"] == 1920
    assert row["height_px"] == 1080
    assert row["nominal_fps"] == pytest.approx(60.0)
    assert row["camera_model_id"] == model_id


def test_create_camera_mode_invalid_model_raises(registry_db: sqlite3.Connection) -> None:
    """create_camera_mode() with a non-existent camera_model_id should raise IntegrityError."""
    import sqlite3 as _sqlite3
    with pytest.raises(_sqlite3.IntegrityError):
        create_camera_mode(registry_db, "00000000-0000-0000-0000-000000000000")


def test_list_camera_modes_empty(registry_db: sqlite3.Connection) -> None:
    """list_camera_modes() should return an empty list when no modes are registered."""
    assert list_camera_modes(registry_db) == []


def test_list_camera_modes_filtered(registry_db: sqlite3.Connection) -> None:
    """list_camera_modes() should filter by camera_model_id when provided."""
    m1 = create_camera_model(registry_db, model_name="M1")
    m2 = create_camera_model(registry_db, model_name="M2")
    create_camera_mode(registry_db, m1, width_px=1920, height_px=1080)
    create_camera_mode(registry_db, m1, width_px=3840, height_px=2160)
    create_camera_mode(registry_db, m2, width_px=1280, height_px=720)

    modes_m1 = list_camera_modes(registry_db, camera_model_id=m1)
    modes_m2 = list_camera_modes(registry_db, camera_model_id=m2)
    assert len(modes_m1) == 2
    assert len(modes_m2) == 1
    assert len(list_camera_modes(registry_db)) == 3


# ---------------------------------------------------------------------------
# New tests: self-contained session DB and _copy_rows_if_missing
# ---------------------------------------------------------------------------


def test_session_db_has_registry_tables(session_db: sqlite3.Connection) -> None:
    """A newly created session DB should contain all registry tables."""
    rows = session_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    actual = {row["name"] for row in rows}
    for table in ("camera_models", "camera_modes", "camera_instances",
                  "intrinsics_calibrations", "skeletons", "tracker_configs"):
        assert table in actual, f"Registry table missing from session DB: {table!r}"


def test_add_session_camera_copies_camera_rows(
    tmp_path: Path,
    registry_db: sqlite3.Connection,
    session_db: sqlite3.Connection,
) -> None:
    """After add_session_camera, registry rows are present in the session DB."""
    import struct, datetime as _dt
    model_id = create_camera_model(registry_db, manufacturer="Acme", model_name="C1")
    mode_id = create_camera_mode(registry_db, model_id, width_px=1920, height_px=1080)
    dist_blob = struct.pack("<4d", 0.0, 0.0, 0.0, 0.0)
    inst_id = "inst-copy-test"
    registry_db.execute(
        "INSERT INTO camera_instances (id, camera_model_id, serial_number, label) "
        "VALUES (?, ?, '', 'c1')",
        (inst_id, model_id),
    )
    intr_id = "intr-copy-test"
    registry_db.execute(
        "INSERT INTO intrinsics_calibrations "
        "(id, camera_mode_id, calibrated_at, distortion_model, fx, fy, cx, cy, dist_coeffs) "
        "VALUES (?, ?, ?, 'radtan', 800.0, 800.0, 320.0, 240.0, ?)",
        (intr_id, mode_id, _dt.date.today().isoformat(), dist_blob),
    )
    registry_db.commit()

    session_id = create_mocap_session(session_db)
    add_session_camera(
        session_db, registry_db, session_id, inst_id, mode_id, intr_id, label="c1"
    )

    # All four registry rows should now exist in the session DB.
    assert session_db.execute(
        "SELECT id FROM camera_models WHERE id = ?", (model_id,)
    ).fetchone() is not None

    assert session_db.execute(
        "SELECT id FROM camera_modes WHERE id = ?", (mode_id,)
    ).fetchone() is not None

    assert session_db.execute(
        "SELECT id FROM camera_instances WHERE id = ?", (inst_id,)
    ).fetchone() is not None

    assert session_db.execute(
        "SELECT id FROM intrinsics_calibrations WHERE id = ?", (intr_id,)
    ).fetchone() is not None


def test_copy_rows_if_missing_idempotent(
    registry_db: sqlite3.Connection,
    session_db: sqlite3.Connection,
) -> None:
    """Calling _copy_rows_if_missing twice with the same ID should not raise."""
    model_id = create_camera_model(registry_db, manufacturer="Idempotent", model_name="X")
    # First copy
    _copy_rows_if_missing(registry_db, session_db, "camera_models", [model_id])
    # Second copy — should be silently skipped (INSERT OR IGNORE)
    _copy_rows_if_missing(registry_db, session_db, "camera_models", [model_id])
    count = session_db.execute(
        "SELECT COUNT(*) FROM camera_models WHERE id = ?", (model_id,)
    ).fetchone()[0]
    assert count == 1


def test_copy_rows_if_missing_missing_row_raises(
    registry_db: sqlite3.Connection,
    session_db: sqlite3.Connection,
) -> None:
    """_copy_rows_if_missing should raise ValueError when the source row is absent."""
    with pytest.raises(ValueError, match="not found in camera_models"):
        _copy_rows_if_missing(
            registry_db, session_db, "camera_models", ["nonexistent-uuid"]
        )

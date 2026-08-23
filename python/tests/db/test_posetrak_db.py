# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/db/posetrak_db.py public API."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

import posetrak.db.db as db_module
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
    seed_bundled_defaults,
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


def test_create_registry_removes_partial_file_on_schema_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A schema-application failure must not leave a broken file behind.

    sqlite3.connect() creates the file at *path* immediately, before a
    single statement runs. Without cleanup, a failure partway through
    create_registry() leaves an empty, schema-version-0 file there --
    which a later open_or_create_registry() (or a plain retry) would then
    treat as an existing but broken registry instead of creating a fresh
    one, surfacing a confusing "expected N, got 0" mismatch instead of
    the real error.
    """
    db_path = tmp_path / "reg.db"
    monkeypatch.setattr(db_module, "_REGISTRY_SCHEMA_SQL", Path("does-not-exist.sql"))
    with pytest.raises(FileNotFoundError):
        create_registry(db_path)
    assert not db_path.exists()


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


def test_migrate_registry_v6_to_v7_catches_up_stale_columns_and_adds_is_named(
    tmp_path: Path,
) -> None:
    """v6->v7 catches up tracker_configs to every column added since v6 via
    the session migration chain (never previously mirrored into the
    registry's own chain -- see db.py's _migrate_registry_v6_to_v7 doc
    comment) and adds is_named. Simulates a registry created back when v6
    was current: drop a handful of representative post-v6 columns and
    is_named, then confirm open_registry() adds them all back, preserving
    existing data.
    """
    db_path = tmp_path / "reg.db"
    conn = create_registry(db_path)
    conn.execute(
        "INSERT INTO tracker_configs (id, name, parent_id, created_at, alpha, is_named) "
        "VALUES ('cfg1', 'old-config', NULL, '2020-01-01', 0.5, 1)"
    )
    conn.commit()

    # Downgrade to a pre-v22 tracker_configs shape (drop every column added
    # since, plus is_named) and roll the version pragma back to 6.
    conn.executescript("""
        BEGIN;
        CREATE TABLE tracker_configs_old (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            parent_id TEXT REFERENCES tracker_configs(id),
            created_at TEXT NOT NULL,
            alpha REAL, beta REAL, kappa REAL,
            process_noise_std REAL, process_noise_vel_std REAL,
            velocity_half_life_s REAL, measurement_noise_std REAL,
            outlier_threshold REAL, tracker_fps REAL,
            ik_max_iterations INTEGER, ik_tolerance REAL,
            init_position_std REAL, init_orientation_std REAL,
            init_joint_std REAL, init_velocity_std REAL,
            min_cameras_for_init INTEGER,
            velocity_mode_camera_ids TEXT, velocity_measurement_noise_std REAL,
            notes TEXT
        );
        INSERT INTO tracker_configs_old
            (id, name, parent_id, created_at, alpha)
            SELECT id, name, parent_id, created_at, alpha FROM tracker_configs;
        DROP TABLE tracker_configs;
        ALTER TABLE tracker_configs_old RENAME TO tracker_configs;
        PRAGMA user_version = 6;
        COMMIT;
    """)
    conn.close()

    conn = open_registry(db_path)
    assert get_schema_version(conn) == REGISTRY_SCHEMA_VERSION

    cols = {row[1] for row in conn.execute("PRAGMA table_info(tracker_configs)")}
    assert {"pose_noise_std", "cross_person_max_n", "is_named",
            "pose_reg_joint_names"} <= cols

    row = conn.execute(
        "SELECT alpha, is_named, pose_noise_std FROM tracker_configs WHERE id = 'cfg1'"
    ).fetchone()
    assert row["alpha"] == pytest.approx(0.5)
    assert row["is_named"] == 0  # column re-added with its schema default, not the old value
    assert row["pose_noise_std"] is None

    # The baseline config gets backfilled for pre-existing registries too.
    from posetrak.db.manage_config import BASELINE_CONFIG_ID
    baseline = conn.execute(
        "SELECT id FROM tracker_configs WHERE id = ?", (BASELINE_CONFIG_ID,)
    ).fetchone()
    assert baseline is not None
    conn.close()


def test_create_registry_includes_marker_body_definitions(tmp_path: Path) -> None:
    """A freshly created registry DB should have marker_body_definitions
    (section 10 of the extrinsics-improvements design doc), same column
    shape as skeletons' content-addressed-id convention."""
    db_path = tmp_path / "reg.db"
    conn = create_registry(db_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(marker_body_definitions)")}
    assert cols == {"id", "name", "yaml_content", "source", "created_at", "notes"}
    conn.close()


def test_migrate_registry_v7_to_v8_adds_marker_body_definitions(tmp_path: Path) -> None:
    """v7->v8 adds marker_body_definitions to a registry created before
    this feature existed. See db.py's _migrate_registry_v7_to_v8."""
    db_path = tmp_path / "reg.db"
    conn = create_registry(db_path)
    conn.executescript("""
        BEGIN;
        DROP TABLE marker_body_definitions;
        PRAGMA user_version = 7;
        COMMIT;
    """)
    conn.close()

    conn = open_registry(db_path)
    assert get_schema_version(conn) == REGISTRY_SCHEMA_VERSION

    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "marker_body_definitions" in tables

    conn.execute(
        "INSERT INTO marker_body_definitions (id, name, yaml_content, created_at) "
        "VALUES ('body1', 'test-rig', 'name: test-rig\\nmarkers: []\\n', '2026-01-01')"
    )
    conn.commit()
    row = conn.execute(
        "SELECT name FROM marker_body_definitions WHERE id = 'body1'"
    ).fetchone()
    assert row["name"] == "test-rig"
    conn.close()


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


def test_create_session_creates_file(tmp_path: Path) -> None:
    """create_session() should create a file at the given path."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    conn.close()
    assert db_path.exists()


def test_create_session_removes_partial_file_on_schema_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A schema-application failure must not leave a broken file behind.

    See test_create_registry_removes_partial_file_on_schema_failure --
    same failure mode, same fix (_discard_partial_db).
    """
    db_path = tmp_path / "session.db"
    monkeypatch.setattr(db_module, "_REGISTRY_SCHEMA_SQL", Path("does-not-exist.sql"))
    with pytest.raises(FileNotFoundError):
        create_session(db_path)
    assert not db_path.exists()


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


def test_create_session_includes_hierarchical_solver_tables(tmp_path: Path) -> None:
    """A freshly created session DB should have tracking_run_stages and
    tracker_config_stages (hierarchical body/hand solver, v37) with the
    columns the design doc specifies."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)

    stage_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_run_stages)")}
    assert stage_cols == {
        "run_id",
        "person_id",
        "group_name",
        "status",
        "started_at",
        "completed_at",
    }

    config_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracker_config_stages)")}
    assert {
        "tracker_config_id",
        "group_name",
        "process_noise_std",
        "process_noise_vel_std",
        "velocity_half_life_s",
        "pose_noise_std",
        "calib_noise_std",
        "outlier_threshold",
        "min_inliers_ratio",
        "max_innovation_norm",
        "init_joint_std",
        "init_velocity_std",
    } <= config_cols
    conn.close()


def test_migrate_session_v36_to_v37_adds_hierarchical_solver_tables(tmp_path: Path) -> None:
    """v36→v37 adds tracking_run_stages and tracker_config_stages.

    See db/migrations/026_hierarchical_solver_stages.sql.
    """
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)

    # Downgrade to the pre-migration (v36) shape: drop the new tables and
    # roll the version pragma back, simulating a database created before
    # this feature existed.
    conn.executescript("""
        BEGIN;
        DROP TABLE tracking_run_stages;
        DROP TABLE tracker_config_stages;
        PRAGMA user_version = 36;
        COMMIT;
    """)
    conn.close()

    conn = open_session(db_path)
    assert get_schema_version(conn) == SESSION_SCHEMA_VERSION

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"tracking_run_stages", "tracker_config_stages"} <= tables

    # Status defaults and the CHECK constraint's allowed values round-trip.
    # tracking_runs has real FKs to pose_observation_sequences/
    # extrinsic_calibrations/sync_configs (foreign_keys is ON), so build the
    # minimal parent chain rather than disabling enforcement.
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('shot1', 'sess1', 1)"
    )
    conn.execute(
        "INSERT INTO extrinsic_calibrations (id, session_id, calibrated_at)"
        " VALUES ('extr1', 'sess1', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO sync_configs (id, shot_id) VALUES ('sync1', 'shot1')"
    )
    conn.execute(
        "INSERT INTO pose_observation_sequences"
        " (id, shot_id, sync_config_id, time_start_s, time_end_s)"
        " VALUES ('seq1', 'shot1', 'sync1', 0.0, 1.0)"
    )
    conn.execute(
        "INSERT INTO tracking_runs (id, observation_sequence_id, tracker_config_id,"
        " skeleton_id, extrinsic_calibration_id, sync_config_id, ran_at,"
        " posetrak_version, active_camera_ids, marker_names)"
        " VALUES ('run1', 'seq1', 'cfg1', 'skel1', 'extr1', 'sync1', '2026-01-01',"
        " '0.0.0', '[]', '[]')"
    )
    conn.execute(
        "INSERT INTO tracking_run_stages (run_id, person_id, group_name)"
        " VALUES ('run1', 0, 'HandL')"
    )
    row = conn.execute(
        "SELECT status FROM tracking_run_stages WHERE run_id='run1' AND group_name='HandL'"
    ).fetchone()
    assert row["status"] == "pending"

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO tracking_run_stages (run_id, person_id, group_name, status)"
            " VALUES ('run1', 0, 'HandR', 'not-a-real-status')"
        )
    conn.close()


def test_create_session_includes_config_default_columns(tmp_path: Path) -> None:
    """A freshly created session DB should have tracker_configs.is_named and
    captures/trials.default_tracker_config_id (v38, config-improvements)."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)

    config_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracker_configs)")}
    assert "is_named" in config_cols

    capture_cols = {row[1] for row in conn.execute("PRAGMA table_info(captures)")}
    assert "default_tracker_config_id" in capture_cols

    trial_cols = {row[1] for row in conn.execute("PRAGMA table_info(trials)")}
    assert "default_tracker_config_id" in trial_cols
    conn.close()


def test_migrate_session_v37_to_v38_adds_config_default_columns(tmp_path: Path) -> None:
    """v37->v38 adds tracker_configs.is_named and captures/trials.
    default_tracker_config_id to a session DB created before this feature.

    See docs/roadmap/features/configuration-improvements/config-improvements-design.md.
    """
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)

    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('shot1', 'sess1', 1)"
    )
    conn.execute(
        "INSERT INTO trials (id, capture_id, name) VALUES ('trial1', 'shot1', 'take 1')"
    )
    conn.commit()

    # Downgrade to the pre-v38 shape: drop the new columns and roll the
    # version pragma back, simulating a session created before this feature.
    # Goes via "CREATE TABLE ... AS SELECT" (an auto-generated, comment-free
    # schema) rather than ALTER TABLE ... DROP COLUMN directly against the
    # real tables: SQLite's DROP COLUMN does a naive text rewrite of the
    # table's *stored* CREATE TABLE SQL, and a bare comma inside one of this
    # schema's own descriptive `--` comments (there are many) can corrupt
    # that rewrite -- confirmed independent of anything this migration
    # touches. CREATE TABLE ... AS SELECT strips comments entirely, sidestepping it.
    # Foreign keys off for this block only: dropping captures/tracker_configs
    # while trials/tracking_runs still reference them (even transiently,
    # mid-script) trips FK enforcement; PRAGMA foreign_keys can't be toggled
    # inside a transaction, so it's set outside the executescript() below.
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.executescript("""
        BEGIN;
        CREATE TABLE tracker_configs_old AS SELECT * FROM tracker_configs;
        ALTER TABLE tracker_configs_old DROP COLUMN is_named;
        DROP TABLE tracker_configs;
        ALTER TABLE tracker_configs_old RENAME TO tracker_configs;

        CREATE TABLE captures_old AS SELECT * FROM captures;
        ALTER TABLE captures_old DROP COLUMN default_tracker_config_id;
        DROP TABLE captures;
        ALTER TABLE captures_old RENAME TO captures;

        CREATE TABLE trials_old AS SELECT * FROM trials;
        ALTER TABLE trials_old DROP COLUMN default_tracker_config_id;
        DROP TABLE trials;
        ALTER TABLE trials_old RENAME TO trials;

        PRAGMA user_version = 37;
        COMMIT;
    """)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.close()

    conn = open_session(db_path)
    assert get_schema_version(conn) == SESSION_SCHEMA_VERSION

    config_cols = {row[1] for row in conn.execute("PRAGMA table_info(tracker_configs)")}
    assert "is_named" in config_cols
    capture_cols = {row[1] for row in conn.execute("PRAGMA table_info(captures)")}
    assert "default_tracker_config_id" in capture_cols
    trial_cols = {row[1] for row in conn.execute("PRAGMA table_info(trials)")}
    assert "default_tracker_config_id" in trial_cols

    # Existing rows survive the migration untouched.
    trial_row = conn.execute(
        "SELECT name, default_tracker_config_id FROM trials WHERE id = 'trial1'"
    ).fetchone()
    assert trial_row["name"] == "take 1"
    assert trial_row["default_tracker_config_id"] is None
    conn.close()


def test_migrate_session_v38_to_v39_adds_capture_persons(tmp_path: Path) -> None:
    """v38->v39 adds capture_persons plus a nullable capture_person_id link
    on sequence_persons/detection_track_assignments to a session DB created
    before this feature.

    See docs/roadmap/features/configuration-improvements/config-improvements-design.md,
    "Person model: promote identity to capture level".
    """
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)

    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('cap1', 'sess1', 1)"
    )
    conn.execute(
        "INSERT INTO sync_configs (id, shot_id) VALUES ('sync1', 'cap1')"
    )
    conn.execute(
        "INSERT INTO pose_observation_sequences "
        "(id, shot_id, sync_config_id, time_start_s, time_end_s) "
        "VALUES ('seq1', 'cap1', 'sync1', 0.0, 1.0)"
    )
    conn.execute(
        "INSERT INTO sequence_persons (sequence_id, person_id, person_name) "
        "VALUES ('seq1', 0, 'Alice')"
    )
    conn.commit()

    # Downgrade to the pre-v39 shape: drop capture_persons entirely and the
    # capture_person_id column, roll the version pragma back. See
    # test_migrate_session_v37_to_v38_adds_config_default_columns's own
    # comment above for why this goes via CREATE TABLE ... AS SELECT rather
    # than ALTER TABLE ... DROP COLUMN directly.
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.executescript("""
        BEGIN;
        DROP TABLE capture_persons;

        CREATE TABLE sequence_persons_old AS SELECT * FROM sequence_persons;
        ALTER TABLE sequence_persons_old DROP COLUMN capture_person_id;
        DROP TABLE sequence_persons;
        ALTER TABLE sequence_persons_old RENAME TO sequence_persons;

        PRAGMA user_version = 38;
        COMMIT;
    """)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.close()

    conn = open_session(db_path)
    assert get_schema_version(conn) == SESSION_SCHEMA_VERSION

    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "capture_persons" in tables
    seq_persons_cols = {row[1] for row in conn.execute("PRAGMA table_info(sequence_persons)")}
    assert "capture_person_id" in seq_persons_cols
    assignment_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(detection_track_assignments)")
    }
    assert "capture_person_id" in assignment_cols

    # Existing rows survive the migration untouched.
    seq_person_row = conn.execute(
        "SELECT person_name, capture_person_id FROM sequence_persons "
        "WHERE sequence_id = 'seq1' AND person_id = 0"
    ).fetchone()
    assert seq_person_row["person_name"] == "Alice"
    assert seq_person_row["capture_person_id"] is None
    conn.close()


def test_create_session_includes_marker_body_tables(tmp_path: Path) -> None:
    """A freshly created session DB should have marker_body_definitions
    (embedded from the registry, same as camera_models/skeletons/
    tracker_configs) and scene_marker_bodies (session-scoped solved
    poses). See extrinsics-improvements-design.md, section 10."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)

    definition_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(marker_body_definitions)")
    }
    assert definition_cols == {"id", "name", "yaml_content", "source", "created_at", "notes"}

    body_cols = {row[1] for row in conn.execute("PRAGMA table_info(scene_marker_bodies)")}
    assert body_cols == {
        "id", "session_id", "label", "group_name", "marker_body_definition_id",
        "marker_type", "dictionary", "marker_id", "marker_size",
        "R", "t", "is_primary_anchor", "source_extrinsic_calibration_id", "updated_at",
    }
    conn.close()


def test_migrate_session_v39_to_v40_adds_marker_body_tables(tmp_path: Path) -> None:
    """v39->v40 adds marker_body_definitions and scene_marker_bodies to a
    session DB created before this feature existed. See db.py's
    _migrate_session_v39_to_v40."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)

    conn.executescript("""
        BEGIN;
        DROP TABLE scene_marker_bodies;
        DROP TABLE marker_body_definitions;
        PRAGMA user_version = 39;
        COMMIT;
    """)
    conn.close()

    conn = open_session(db_path)
    assert get_schema_version(conn) == SESSION_SCHEMA_VERSION

    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"marker_body_definitions", "scene_marker_bodies"} <= tables

    # Exercise the real FK chain (mocap_sessions -> scene_marker_bodies,
    # extrinsic_calibrations -> scene_marker_bodies) plus both the
    # rig-anchor case (a real marker_body_definitions row) and the lone-tag
    # case (marker_body_definition_id NULL, inline dictionary/id/size) --
    # both scene_marker_bodies "modes" the design doc describes.
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO extrinsic_calibrations (id, session_id, calibrated_at) "
        "VALUES ('extr1', 'sess1', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO marker_body_definitions (id, name, yaml_content, created_at) "
        "VALUES ('body1', 'calib-box', 'name: calib-box\\nmarkers: []\\n', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO scene_marker_bodies "
        "(id, session_id, label, marker_body_definition_id, R, t, "
        " is_primary_anchor, source_extrinsic_calibration_id, updated_at) "
        "VALUES ('smb1', 'sess1', 'calib-box', 'body1', X'00', X'00', "
        " 1, 'extr1', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO scene_marker_bodies "
        "(id, session_id, label, marker_type, dictionary, marker_id, marker_size, "
        " R, t, source_extrinsic_calibration_id, updated_at) "
        "VALUES ('smb2', 'sess1', 'wall-tag-north', 'aruco', 'DICT_5X5_50', '3', 0.1, "
        " X'00', X'00', 'extr1', '2026-01-01')"
    )
    conn.commit()

    rig_row = conn.execute(
        "SELECT marker_body_definition_id, is_primary_anchor FROM scene_marker_bodies "
        "WHERE id = 'smb1'"
    ).fetchone()
    assert rig_row["marker_body_definition_id"] == "body1"
    assert rig_row["is_primary_anchor"] == 1

    tag_row = conn.execute(
        "SELECT marker_body_definition_id, dictionary, marker_id FROM scene_marker_bodies "
        "WHERE id = 'smb2'"
    ).fetchone()
    assert tag_row["marker_body_definition_id"] is None
    assert tag_row["dictionary"] == "DICT_5X5_50"
    assert tag_row["marker_id"] == "3"

    # session_id + label is unique.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO scene_marker_bodies "
            "(id, session_id, label, R, t, updated_at) "
            "VALUES ('smb3', 'sess1', 'calib-box', X'00', X'00', '2026-01-01')"
        )
    conn.close()


def test_migrate_session_v40_to_v41_adds_group_name(tmp_path: Path) -> None:
    """v40->v41 adds scene_marker_bodies.group_name, defaulting existing
    rows to '' (not NULL -- see db.py's _migrate_session_v40_to_v41 for
    why), and widens the unique index to (session_id, group_name, label)
    so two same-named groups' tags with the same id can't collide but two
    *different* groups' tags with the same id now can coexist."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    conn.execute("PRAGMA user_version = 40")
    conn.commit()
    conn.close()

    conn = open_session(db_path)
    assert get_schema_version(conn) == SESSION_SCHEMA_VERSION

    cols = {row[1] for row in conn.execute("PRAGMA table_info(scene_marker_bodies)")}
    assert "group_name" in cols

    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    # Legacy-style insert with no group_name given -- must default to ''.
    conn.execute(
        "INSERT INTO scene_marker_bodies (id, session_id, label, R, t, updated_at) "
        "VALUES ('smb1', 'sess1', 'tag:3', X'00', X'00', '2026-01-01')"
    )
    conn.commit()
    row = conn.execute(
        "SELECT group_name FROM scene_marker_bodies WHERE id = 'smb1'"
    ).fetchone()
    assert row["group_name"] == ""

    # Two different named groups may reuse the same label without colliding.
    conn.execute(
        "INSERT INTO scene_marker_bodies (id, session_id, label, group_name, R, t, updated_at) "
        "VALUES ('smb2', 'sess1', 'tag:3', 'room7', X'00', X'00', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO scene_marker_bodies (id, session_id, label, group_name, R, t, updated_at) "
        "VALUES ('smb3', 'sess1', 'tag:3', 'room8', X'00', X'00', '2026-01-01')"
    )
    conn.commit()

    # But the same group_name + label combination is still unique.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO scene_marker_bodies (id, session_id, label, group_name, R, t, updated_at) "
            "VALUES ('smb4', 'sess1', 'tag:3', 'room7', X'00', X'00', '2026-01-01')"
        )
    conn.close()


def test_migrate_session_v41_to_v42_makes_seg_quality_runs_capture_scoped(
    tmp_path: Path,
) -> None:
    """v41->v42 gives seg_quality_runs its own shot_id/trial_id/time_start_s/
    time_end_s (mirroring detection_runs' own columns) and drops the NOT
    NULL detection_run_id FK, so a segmentation can be created before any
    detection run exists (see docs/roadmap/features/segmentation-reuse/
    segmentation-reuse-design.md). Existing rows backfill their new
    columns from their (until now, 1:1) owning detection_runs row."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    conn.execute("DROP TABLE seg_quality_runs")
    conn.execute(
        "CREATE TABLE seg_quality_runs ("
        "    id TEXT PRIMARY KEY, detection_run_id TEXT NOT NULL, created_at TEXT NOT NULL,"
        "    quality_source TEXT NOT NULL DEFAULT 'cutie',"
        "    erosion_px INTEGER NOT NULL DEFAULT 5, mask_dir TEXT, notes TEXT"
        ")"
    )
    conn.execute("PRAGMA user_version = 41")

    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')")
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('cap1', 'sess1', 1)"
    )
    conn.execute(
        "INSERT INTO trials (id, capture_id, name, time_start_s, time_end_s) "
        "VALUES ('trial1', 'cap1', 'T', 1.0, 2.0)"
    )
    conn.execute(
        "INSERT INTO sync_configs (id, shot_id, created_by) VALUES ('sync1', 'cap1', 'x')"
    )
    conn.execute(
        "INSERT INTO detection_runs "
        "(id, shot_id, sync_config_id, trial_id, time_start_s, time_end_s, "
        " detector_model, pose_model, created_at) "
        "VALUES ('run1', 'cap1', 'sync1', 'trial1', 5.0, 10.0, 'd', 'p', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO seg_quality_runs (id, detection_run_id, created_at) "
        "VALUES ('seg1', 'run1', '2026-01-01')"
    )
    conn.commit()
    conn.close()

    conn = open_session(db_path)
    assert get_schema_version(conn) == SESSION_SCHEMA_VERSION

    cols = {row[1] for row in conn.execute("PRAGMA table_info(seg_quality_runs)")}
    assert "detection_run_id" not in cols
    assert {"shot_id", "trial_id", "time_start_s", "time_end_s"} <= cols

    row = conn.execute(
        "SELECT shot_id, trial_id, time_start_s, time_end_s FROM seg_quality_runs WHERE id='seg1'"
    ).fetchone()
    assert row["shot_id"] == "cap1"
    assert row["trial_id"] == "trial1"
    assert row["time_start_s"] == 5.0
    assert row["time_end_s"] == 10.0

    # A segmentation can now be created directly, with no detection_runs
    # row involved at all -- the whole point of the migration.
    conn.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at) "
        "VALUES ('seg2', 'cap1', 0.0, 1e9, '2026-01-02')"
    )
    conn.commit()
    conn.close()


def test_migrate_session_v41_to_v42_with_seg_masks_present(tmp_path: Path) -> None:
    """seg_masks.seg_quality_run_id REFERENCES seg_quality_runs(id) -- every
    real session has seg_masks rows once any segmentation has actually been
    used. DROP TABLE seg_quality_runs (part of the rebuild) previously
    raised "FOREIGN KEY constraint failed" the instant a child row existed,
    with PRAGMA foreign_keys=ON (the default for every connection this
    codebase opens) -- caught live, 2026-08-16, app failing at startup
    against a real session. The migration's own test above never had
    seg_masks rows, so it never exercised this path. Regression test for
    the fix: bracket the rebuild in PRAGMA foreign_keys=OFF/ON."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")  # setup only -- camera_models isn't the point here
    conn.execute("DROP TABLE seg_quality_runs")
    conn.execute(
        "CREATE TABLE seg_quality_runs ("
        "    id TEXT PRIMARY KEY, detection_run_id TEXT NOT NULL, created_at TEXT NOT NULL,"
        "    quality_source TEXT NOT NULL DEFAULT 'cutie',"
        "    erosion_px INTEGER NOT NULL DEFAULT 5, mask_dir TEXT, notes TEXT"
        ")"
    )
    conn.execute("PRAGMA user_version = 41")

    conn.execute("INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')")
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('cap1', 'sess1', 1)"
    )
    conn.execute(
        "INSERT INTO sync_configs (id, shot_id, created_by) VALUES ('sync1', 'cap1', 'x')"
    )
    conn.execute(
        "INSERT INTO detection_runs "
        "(id, shot_id, sync_config_id, time_start_s, time_end_s, "
        " detector_model, pose_model, created_at) "
        "VALUES ('run1', 'cap1', 'sync1', 5.0, 10.0, 'd', 'p', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO seg_quality_runs (id, detection_run_id, created_at) "
        "VALUES ('seg1', 'run1', '2026-01-01')"
    )
    conn.execute("INSERT INTO camera_models (id) VALUES ('cm1')")
    conn.execute(
        "INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('ci1', 'cm1', 'camA')"
    )
    conn.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,"
        " first_video_frame, last_video_frame, actual_fps)"
        " VALUES ('sv1', 'cap1', 'ci1', '/x.mp4', 0, 100, 30.0)"
    )
    conn.execute(
        "INSERT INTO seg_masks (seg_quality_run_id, shot_video_id, frame_idx, mask_blob) "
        "VALUES ('seg1', 'sv1', 0, X'00')"
    )
    conn.commit()
    conn.close()

    conn = open_session(db_path)  # must not raise IntegrityError
    assert get_schema_version(conn) == SESSION_SCHEMA_VERSION
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    # The mask row's FK is still satisfied against the rebuilt table.
    row = conn.execute(
        "SELECT seg_quality_run_id FROM seg_masks WHERE shot_video_id='sv1'"
    ).fetchone()
    assert row["seg_quality_run_id"] == "seg1"
    conn.close()


def test_migrate_session_v41_to_v42_creates_table_when_missing(tmp_path: Path) -> None:
    """seg_quality_runs was never given its own numbered migration when it
    was originally introduced (only added to session_schema.sql for fresh
    sessions) -- a session incrementally migrated from an old-enough
    schema version genuinely never got the table created at all (caught
    via test_session_v2_migrates_to_v3 failing when this migration was
    first added). Simulates that: no seg_quality_runs table present at
    v41, migration should create it fresh in its v42 shape rather than
    trying (and failing) to rebuild a table that was never there."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    conn.execute("DROP TABLE seg_quality_runs")
    conn.execute("PRAGMA user_version = 41")
    conn.commit()
    conn.close()

    conn = open_session(db_path)
    assert get_schema_version(conn) == SESSION_SCHEMA_VERSION
    cols = {row[1] for row in conn.execute("PRAGMA table_info(seg_quality_runs)")}
    assert {"shot_id", "trial_id", "time_start_s", "time_end_s"} <= cols
    assert "detection_run_id" not in cols
    conn.close()


def test_migrate_session_v42_to_v43_adds_persons_json(tmp_path: Path) -> None:
    """v42->v43 adds seg_quality_runs.persons_json (nullable) -- the
    ordinal->name mapping baked into a segmentation's mask labels, so a
    later caller reusing it doesn't have to assume today's
    capture_persons order still matches (see docs/roadmap/features/
    segmentation-reuse/segmentation-reuse-design.md, gap 2)."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    conn.execute("PRAGMA user_version = 42")
    conn.commit()
    conn.close()

    conn = open_session(db_path)
    assert get_schema_version(conn) == SESSION_SCHEMA_VERSION
    cols = {row[1] for row in conn.execute("PRAGMA table_info(seg_quality_runs)")}
    assert "persons_json" in cols
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


def test_create_session_does_not_seed_bundled_defaults(
    session_db: sqlite3.Connection,
) -> None:
    """Unlike create_registry(), create_session() applies only the schema --
    it's also used internally for exports/round-trips that need a
    genuinely empty session (see trial_export.py). Callers representing a
    person actually starting a new session call seed_bundled_defaults()
    themselves right after (setup wizard, `session create`, `session
    add-camera`, ...)."""
    assert session_db.execute("SELECT COUNT(*) FROM skeletons").fetchone()[0] == 0
    assert session_db.execute("SELECT COUNT(*) FROM tracker_configs").fetchone()[0] == 0


def test_seed_bundled_defaults_seeds_session(session_db: sqlite3.Connection) -> None:
    seed_bundled_defaults(session_db)
    assert session_db.execute("SELECT COUNT(*) FROM skeletons").fetchone()[0] == 2
    assert session_db.execute(
        "SELECT COUNT(*) FROM tracker_configs WHERE id = 'factory-defaults'"
    ).fetchone()[0] == 1


def test_seed_bundled_defaults_idempotent(session_db: sqlite3.Connection) -> None:
    seed_bundled_defaults(session_db)
    seed_bundled_defaults(session_db)
    assert session_db.execute("SELECT COUNT(*) FROM skeletons").fetchone()[0] == 2
    assert session_db.execute(
        "SELECT COUNT(*) FROM tracker_configs WHERE id = 'factory-defaults'"
    ).fetchone()[0] == 1


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


def test_add_session_camera_with_default_intrinsics_on_mode(
    registry_db: sqlite3.Connection,
    session_db: sqlite3.Connection,
) -> None:
    """camera_modes.default_intrinsics_calibration_id and
    intrinsics_calibrations.camera_mode_id reference each other -- a real
    circular FK, not just an insertion-order problem (see the comment in
    add_session_camera). A camera_mode with its default set (true for any
    camera that's actually been calibrated) used to raise
    "FOREIGN KEY constraint failed" here regardless of copy order."""
    import struct, datetime as _dt
    model_id = create_camera_model(registry_db, manufacturer="Acme", model_name="C2")
    mode_id = create_camera_mode(registry_db, model_id, width_px=1920, height_px=1080)
    dist_blob = struct.pack("<4d", 0.0, 0.0, 0.0, 0.0)
    inst_id = "inst-circular-fk-test"
    registry_db.execute(
        "INSERT INTO camera_instances (id, camera_model_id, serial_number, label) "
        "VALUES (?, ?, '', 'c2')",
        (inst_id, model_id),
    )
    intr_id = "intr-circular-fk-test"
    registry_db.execute(
        "INSERT INTO intrinsics_calibrations "
        "(id, camera_mode_id, calibrated_at, distortion_model, fx, fy, cx, cy, dist_coeffs) "
        "VALUES (?, ?, ?, 'radtan', 800.0, 800.0, 320.0, 240.0, ?)",
        (intr_id, mode_id, _dt.date.today().isoformat(), dist_blob),
    )
    registry_db.execute(
        "UPDATE camera_modes SET default_intrinsics_calibration_id = ? WHERE id = ?",
        (intr_id, mode_id),
    )
    registry_db.commit()

    session_id = create_mocap_session(session_db)
    add_session_camera(  # must not raise sqlite3.IntegrityError
        session_db, registry_db, session_id, inst_id, mode_id, intr_id, label="c2"
    )

    assert session_db.execute(
        "SELECT default_intrinsics_calibration_id FROM camera_modes WHERE id = ?", (mode_id,)
    ).fetchone()[0] == intr_id
    assert session_db.execute("PRAGMA foreign_key_check").fetchall() == []


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

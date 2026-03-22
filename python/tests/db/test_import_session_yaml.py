"""Tests for import_session_yaml."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from posetrak.db.db import (
    create_camera_model,
    create_camera_mode,
    create_registry,
    create_session,
    open_session,
)
from posetrak.db.import_calib_toml import import_calib_toml
from posetrak.db.import_session_yaml import (
    SessionYamlImportResult,
    import_session_yaml,
)


pytest.importorskip("yaml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry_db(tmp_path: Path):
    db_path = tmp_path / "registry.db"
    conn = create_registry(db_path)
    yield conn
    conn.close()


@pytest.fixture()
def session_db(tmp_path: Path):
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    yield conn
    conn.close()


@pytest.fixture()
def two_camera_registry(registry_db, tmp_path):
    """Registry with two camera instances (cam1, cam2) and intrinsics for each."""
    import io
    import tomllib

    model_id = create_camera_model(registry_db, manufacturer="Test", model_name="Cam")
    mode_id = create_camera_mode(
        registry_db, model_id, width_px=1280, height_px=720, nominal_fps=120.0
    )

    # Import intrinsics via TOML so each camera gets an instance + intrinsics row
    toml_content = textwrap.dedent("""\
        [cam1]
        name = "cam1"
        matrix = [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]
        rotation = [0.0, 0.0, 0.0]
        translation = [0.0, 0.0, 2.0]
        distortions = [-0.1, 0.05, 0.0, 0.0]

        [cam2]
        name = "cam2"
        matrix = [[810.0, 0.0, 645.0], [0.0, 810.0, 362.0], [0.0, 0.0, 1.0]]
        rotation = [0.1, 0.0, 0.0]
        translation = [0.5, 0.0, 2.1]
        distortions = [-0.12, 0.06, 0.0, 0.0]
    """)
    toml_path = tmp_path / "calib.toml"
    toml_path.write_text(toml_content, encoding="utf-8")
    result = import_calib_toml(registry_db, toml_path, mode_id)

    cam1_instance = result.camera_instance_ids["cam1"]
    cam2_instance = result.camera_instance_ids["cam2"]
    cam1_intrinsics = result.intrinsics_ids["cam1"]
    cam2_intrinsics = result.intrinsics_ids["cam2"]

    return {
        "mode_id": mode_id,
        "cam1_instance": cam1_instance,
        "cam2_instance": cam2_instance,
        "cam1_intrinsics": cam1_intrinsics,
        "cam2_intrinsics": cam2_intrinsics,
    }


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "project.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests — happy path
# ---------------------------------------------------------------------------


def test_import_creates_session(tmp_path, registry_db, session_db, two_camera_registry):
    cam1 = two_camera_registry["cam1_instance"]
    cam2 = two_camera_registry["cam2_instance"]

    yaml_path = _write_yaml(tmp_path, f"""\
        name: "test-session"
        location: "lab"
        cameras:
          cam1:
            video_path: "/videos/cam1.mp4"
            fps: 120.0
            sync_frame: 1000
            camera_instance_id: {cam1}
            camera_mode_id: {two_camera_registry["mode_id"]}
            intrinsics_calibration_id: {two_camera_registry["cam1_intrinsics"]}
          cam2:
            video_path: "/videos/cam2.mp4"
            fps: 120.0
            sync_frame: 1002
            camera_instance_id: {cam2}
            camera_mode_id: {two_camera_registry["mode_id"]}
            intrinsics_calibration_id: {two_camera_registry["cam2_intrinsics"]}
        scenes:
          - label: "scene1"
            cameras:
              cam1:
                first_frame: 1100
                last_frame: 2400
              cam2:
                first_frame: 1102
                last_frame: 2402
    """)

    result = import_session_yaml(session_db, registry_db, yaml_path)

    assert isinstance(result, SessionYamlImportResult)
    assert len(result.session_id) == 36
    assert "scene1" in result.shot_ids
    assert "scene1" in result.sync_config_ids

    # Verify session row
    row = session_db.execute(
        "SELECT * FROM mocap_sessions WHERE id = ?", (result.session_id,)
    ).fetchone()
    assert row is not None
    assert row["notes"] == "test-session"
    assert row["location"] == "lab"


def test_import_creates_cameras(tmp_path, registry_db, session_db, two_camera_registry):
    cam1 = two_camera_registry["cam1_instance"]
    yaml_path = _write_yaml(tmp_path, f"""\
        name: "s"
        cameras:
          cam1:
            video_path: "/v/c1.mp4"
            fps: 60.0
            sync_frame: 500
            camera_instance_id: {cam1}
            camera_mode_id: {two_camera_registry["mode_id"]}
            intrinsics_calibration_id: {two_camera_registry["cam1_intrinsics"]}
        scenes:
          - label: "sc1"
            cameras:
              cam1: {{first_frame: 600, last_frame: 1200}}
    """)
    result = import_session_yaml(session_db, registry_db, yaml_path)

    sc = session_db.execute(
        "SELECT * FROM session_cameras WHERE session_id = ?", (result.session_id,)
    ).fetchall()
    assert len(sc) == 1
    assert sc[0]["camera_instance_id"] == cam1


def test_import_creates_shot_and_videos(tmp_path, registry_db, session_db, two_camera_registry):
    cam1 = two_camera_registry["cam1_instance"]
    cam2 = two_camera_registry["cam2_instance"]
    yaml_path = _write_yaml(tmp_path, f"""\
        name: "s"
        cameras:
          cam1:
            video_path: "/v/c1.mp4"
            fps: 120.0
            sync_frame: 1000
            camera_instance_id: {cam1}
            camera_mode_id: {two_camera_registry["mode_id"]}
            intrinsics_calibration_id: {two_camera_registry["cam1_intrinsics"]}
          cam2:
            video_path: "/v/c2.mp4"
            fps: 120.0
            sync_frame: 1005
            camera_instance_id: {cam2}
            camera_mode_id: {two_camera_registry["mode_id"]}
            intrinsics_calibration_id: {two_camera_registry["cam2_intrinsics"]}
        scenes:
          - label: "run1"
            cameras:
              cam1: {{first_frame: 1100, last_frame: 2000}}
              cam2: {{first_frame: 1105, last_frame: 2005}}
          - label: "run2"
            cameras:
              cam1: {{first_frame: 3000, last_frame: 4000}}
              cam2: {{first_frame: 3005, last_frame: 4005}}
    """)
    result = import_session_yaml(session_db, registry_db, yaml_path)

    assert len(result.shot_ids) == 2
    assert "run1" in result.shot_ids
    assert "run2" in result.shot_ids

    # Shots have nullable extrinsic_calibration_id
    for shot_id in result.shot_ids.values():
        shot = session_db.execute("SELECT * FROM shots WHERE id = ?", (shot_id,)).fetchone()
        assert shot is not None
        assert shot["extrinsic_calibration_id"] is None

    # Videos
    for shot_id in result.shot_ids.values():
        videos = session_db.execute(
            "SELECT * FROM shot_videos WHERE shot_id = ?", (shot_id,)
        ).fetchall()
        assert len(videos) == 2


def test_import_creates_sync_config(tmp_path, registry_db, session_db, two_camera_registry):
    cam1 = two_camera_registry["cam1_instance"]
    yaml_path = _write_yaml(tmp_path, f"""\
        name: "s"
        cameras:
          cam1:
            video_path: "/v/c1.mp4"
            fps: 60.0
            sync_frame: 777
            camera_instance_id: {cam1}
            camera_mode_id: {two_camera_registry["mode_id"]}
            intrinsics_calibration_id: {two_camera_registry["cam1_intrinsics"]}
        scenes:
          - label: "sc1"
            cameras:
              cam1: {{first_frame: 800, last_frame: 1200}}
    """)
    result = import_session_yaml(session_db, registry_db, yaml_path)

    sync_id = result.sync_config_ids["sc1"]
    sc = session_db.execute(
        "SELECT * FROM sync_configs WHERE id = ?", (sync_id,)
    ).fetchone()
    assert sc is not None
    assert sc["created_by"] == "yaml-import-rough"

    pts = session_db.execute(
        "SELECT * FROM sync_points WHERE sync_config_id = ?", (sync_id,)
    ).fetchall()
    assert len(pts) == 1
    assert pts[0]["video_frame"] == 777
    assert pts[0]["timestamp_s"] == pytest.approx(0.0)
    assert pts[0]["camera_instance_id"] == cam1


def test_camera_lookup_by_label(tmp_path, registry_db, session_db, two_camera_registry):
    """When camera_instance_id is absent, look up by YAML key (label)."""
    yaml_path = _write_yaml(tmp_path, f"""\
        name: "s"
        cameras:
          cam1:
            video_path: "/v/c1.mp4"
            fps: 60.0
            sync_frame: 500
            camera_mode_id: {two_camera_registry["mode_id"]}
            intrinsics_calibration_id: {two_camera_registry["cam1_intrinsics"]}
        scenes:
          - label: "sc1"
            cameras:
              cam1: {{first_frame: 600, last_frame: 900}}
    """)
    result = import_session_yaml(session_db, registry_db, yaml_path)
    assert result.camera_instance_ids["cam1"] == two_camera_registry["cam1_instance"]


def test_session_label_override(tmp_path, registry_db, session_db, two_camera_registry):
    cam1 = two_camera_registry["cam1_instance"]
    yaml_path = _write_yaml(tmp_path, f"""\
        name: "original-name"
        cameras:
          cam1:
            video_path: "/v/c1.mp4"
            fps: 60.0
            sync_frame: 500
            camera_instance_id: {cam1}
            camera_mode_id: {two_camera_registry["mode_id"]}
            intrinsics_calibration_id: {two_camera_registry["cam1_intrinsics"]}
        scenes:
          - label: "sc1"
            cameras:
              cam1: {{first_frame: 600, last_frame: 900}}
    """)
    result = import_session_yaml(
        session_db, registry_db, yaml_path, session_label="overridden-name"
    )
    row = session_db.execute(
        "SELECT notes FROM mocap_sessions WHERE id = ?", (result.session_id,)
    ).fetchone()
    assert row["notes"] == "overridden-name"


def test_dry_run_writes_nothing(tmp_path, registry_db, session_db, two_camera_registry):
    cam1 = two_camera_registry["cam1_instance"]
    yaml_path = _write_yaml(tmp_path, f"""\
        name: "s"
        cameras:
          cam1:
            video_path: "/v/c1.mp4"
            fps: 60.0
            sync_frame: 500
            camera_instance_id: {cam1}
            camera_mode_id: {two_camera_registry["mode_id"]}
            intrinsics_calibration_id: {two_camera_registry["cam1_intrinsics"]}
        scenes:
          - label: "sc1"
            cameras:
              cam1: {{first_frame: 600, last_frame: 900}}
    """)
    result = import_session_yaml(session_db, registry_db, yaml_path, dry_run=True)
    assert result.session_id == ""  # nothing written
    rows = session_db.execute("SELECT COUNT(*) FROM mocap_sessions").fetchone()[0]
    assert rows == 0


def test_file_not_found(tmp_path, registry_db, session_db):
    with pytest.raises(FileNotFoundError):
        import_session_yaml(session_db, registry_db, tmp_path / "missing.yaml")


def test_missing_cameras_section(tmp_path, registry_db, session_db):
    yaml_path = _write_yaml(tmp_path, "name: s\nscenes:\n  - label: sc1\n    cameras: {}\n")
    with pytest.raises(ValueError, match="cameras"):
        import_session_yaml(session_db, registry_db, yaml_path)


def test_session_v2_migrates_to_v3(tmp_path):
    """open_session should auto-migrate a v2 session to v3 (nullable extrinsics)."""
    import sqlite3

    db_path = tmp_path / "old_session.db"
    # Simulate a v2 session DB (shots.extrinsic_calibration_id NOT NULL)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS mocap_sessions (id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, location TEXT, notes TEXT);
        CREATE TABLE IF NOT EXISTS session_cameras (session_id TEXT NOT NULL, camera_instance_id TEXT NOT NULL, camera_mode_id TEXT NOT NULL, intrinsics_calibration_id TEXT NOT NULL, label TEXT, PRIMARY KEY (session_id, camera_instance_id));
        CREATE TABLE IF NOT EXISTS extrinsic_calibrations (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, calibrated_at TEXT NOT NULL, method TEXT, rms_error REAL);
        CREATE TABLE IF NOT EXISTS extrinsic_entries (extrinsic_calibration_id TEXT NOT NULL, camera_instance_id TEXT NOT NULL, R BLOB NOT NULL, t BLOB NOT NULL, PRIMARY KEY (extrinsic_calibration_id, camera_instance_id));
        CREATE TABLE IF NOT EXISTS shots (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, extrinsic_calibration_id TEXT NOT NULL, shot_number INTEGER NOT NULL, label TEXT, notes TEXT);
        CREATE TABLE IF NOT EXISTS shot_videos (id TEXT PRIMARY KEY, shot_id TEXT NOT NULL, camera_instance_id TEXT NOT NULL, file_path TEXT NOT NULL, first_video_frame INTEGER NOT NULL, last_video_frame INTEGER NOT NULL, actual_fps REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS sync_configs (id TEXT PRIMARY KEY, shot_id TEXT NOT NULL, created_by TEXT, notes TEXT);
        CREATE TABLE IF NOT EXISTS sync_points (sync_config_id TEXT NOT NULL, camera_instance_id TEXT NOT NULL, shot_video_id TEXT NOT NULL, video_frame INTEGER NOT NULL, timestamp_s REAL NOT NULL, PRIMARY KEY (sync_config_id, camera_instance_id, video_frame));
        CREATE TABLE IF NOT EXISTS pose_observation_sequences (id TEXT PRIMARY KEY, shot_id TEXT NOT NULL, sync_config_id TEXT NOT NULL, time_start_s REAL NOT NULL, time_end_s REAL NOT NULL, pose_model TEXT, notes TEXT);
        CREATE TABLE IF NOT EXISTS pose_observations (sequence_id TEXT NOT NULL, camera_instance_id TEXT NOT NULL, video_frame INTEGER NOT NULL, timestamp_s REAL NOT NULL, person_id INTEGER NOT NULL, kp_blob BLOB NOT NULL, PRIMARY KEY (sequence_id, camera_instance_id, video_frame, person_id));
        CREATE TABLE IF NOT EXISTS tracking_runs (id TEXT PRIMARY KEY, observation_sequence_id TEXT NOT NULL, tracker_config_id TEXT NOT NULL, skeleton_id TEXT NOT NULL, extrinsic_calibration_id TEXT NOT NULL, sync_config_id TEXT NOT NULL, ran_at TEXT NOT NULL, posetrak_version TEXT NOT NULL, active_camera_ids TEXT NOT NULL, marker_names TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tracking_run_persons (run_id TEXT NOT NULL, person_id INTEGER NOT NULL, skeleton_id TEXT NOT NULL, PRIMARY KEY (run_id, person_id));
        CREATE TABLE IF NOT EXISTS tracking_results (run_id TEXT NOT NULL, person_id INTEGER NOT NULL, tracker_step INTEGER NOT NULL, is_smoothed INTEGER NOT NULL DEFAULT 0, timestamp_s REAL NOT NULL, tracking_lost INTEGER NOT NULL DEFAULT 0, n_inlier_observations INTEGER, cov_condition_number REAL, state BLOB NOT NULL, cov_diag BLOB NOT NULL, PRIMARY KEY (run_id, person_id, tracker_step, is_smoothed));
        CREATE TABLE IF NOT EXISTS tracking_obs_results (run_id TEXT NOT NULL, person_id INTEGER NOT NULL, tracker_step INTEGER NOT NULL, obs_blob BLOB NOT NULL, PRIMARY KEY (run_id, person_id, tracker_step));
        PRAGMA user_version = 2;
    """)
    conn.close()

    migrated = open_session(db_path)
    from posetrak.db.db import get_schema_version, SESSION_SCHEMA_VERSION
    assert get_schema_version(migrated) == SESSION_SCHEMA_VERSION
    migrated.close()

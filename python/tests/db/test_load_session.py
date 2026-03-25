"""Tests for scripts/db/load_session.py."""

from __future__ import annotations

import json
import sqlite3
import struct
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from posetrak.db.load_session import (
    load_cameras_from_session,
    load_observations_from_session,
    load_sync_from_session,
    load_tracking_run_data,
    load_tracking_run_with_markers,
)
from posetrak.db.db import create_session

# ---------------------------------------------------------------------------
# Minimal skeleton YAML for tests (n_dof = 4: spine ball=3, head revolute=1)
# ---------------------------------------------------------------------------

MINIMAL_YAML = """\
name: minimal
joints:
  - name: root
    type: root
    parent: null
    offset: [0.0, 0.0, 0.0]
  - name: spine
    type: ball
    parent: root
    offset: [0.0, 0.1, 0.0]
    limits:
      x: [-0.5, 0.5]
      y: [-0.3, 0.3]
      z: [-0.2, 0.2]
  - name: head
    type: revolute
    parent: spine
    offset: [0.0, 0.15, 0.0]
    axis: [1.0, 0.0, 0.0]
    limits: [-0.5, 0.5]
markers:
  - name: nose
    parent: head
    offset: [0.0, 0.05, 0.0]
    openpose_keypoint: 0
"""

N_DOF = 4  # spine(3) + head(1)


# ---------------------------------------------------------------------------
# Helpers to build minimal session DBs in memory
# ---------------------------------------------------------------------------


def _new_id() -> str:
    return str(uuid.uuid4())


def _float64_blob(arr) -> bytes:
    return np.asarray(arr, dtype="<f8").tobytes()


def _float32_blob(arr) -> bytes:
    return np.asarray(arr, dtype="<f4").tobytes()


def _make_state_blob(pos=(0.0, 0.0, 0.0), aa=(0.0, 0.0, 0.0),
                     joint_angles=None, root_vel=(0.0, 0.0, 0.0),
                     root_angvel=(0.0, 0.0, 0.0), joint_vels=None) -> bytes:
    """Build a state blob matching State::to_error_vector() format."""
    ja = joint_angles if joint_angles is not None else [0.0] * N_DOF
    jv = joint_vels if joint_vels is not None else [0.0] * N_DOF
    arr = list(pos) + list(aa) + list(ja) + list(root_vel) + list(root_angvel) + list(jv)
    return _float64_blob(arr)


def _make_cov_diag_blob(n_dof=N_DOF) -> bytes:
    state_size = 12 + 2 * n_dof
    return _float64_blob([0.01] * state_size)


def _insert_session_scaffolding(conn: sqlite3.Connection) -> dict:
    """Insert the minimum rows needed for a tracking_run record.

    Returns a dict with all created IDs.
    """
    session_id = _new_id()
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at, location) VALUES (?, ?, ?)",
        (session_id, "2026-01-01T00:00:00Z", "test lab"),
    )

    ext_cal_id = _new_id()
    conn.execute(
        "INSERT INTO extrinsic_calibrations (id, session_id, calibrated_at) VALUES (?, ?, ?)",
        (ext_cal_id, session_id, "2026-01-01T00:00:00Z"),
    )

    shot_id = _new_id()
    conn.execute(
        "INSERT INTO shots (id, session_id, extrinsic_calibration_id, shot_number) "
        "VALUES (?, ?, ?, ?)",
        (shot_id, session_id, ext_cal_id, 1),
    )

    sync_config_id = _new_id()
    conn.execute(
        "INSERT INTO sync_configs (id, shot_id) VALUES (?, ?)",
        (sync_config_id, shot_id),
    )

    seq_id = _new_id()
    conn.execute(
        "INSERT INTO pose_observation_sequences "
        "(id, shot_id, sync_config_id, time_start_s, time_end_s) "
        "VALUES (?, ?, ?, ?, ?)",
        (seq_id, shot_id, sync_config_id, 0.0, 1.0),
    )

    skeleton_id = _new_id()
    conn.execute(
        "INSERT INTO skeletons (id, name, yaml_content, created_at) VALUES (?, ?, ?, ?)",
        (skeleton_id, "minimal", MINIMAL_YAML, "2026-01-01T00:00:00Z"),
    )

    tracker_config_id = _new_id()
    conn.execute(
        "INSERT OR IGNORE INTO tracker_configs "
        "(id, name, created_at) VALUES (?, ?, ?)",
        (tracker_config_id, "default", "2026-01-01T00:00:00Z"),
    )

    return {
        "session_id": session_id,
        "ext_cal_id": ext_cal_id,
        "shot_id": shot_id,
        "sync_config_id": sync_config_id,
        "seq_id": seq_id,
        "skeleton_id": skeleton_id,
        "tracker_config_id": tracker_config_id,
    }


def _insert_tracking_run(conn: sqlite3.Connection, ids: dict,
                          run_id: str | None = None) -> str:
    if run_id is None:
        run_id = _new_id()
    marker_names = json.dumps(["nose"])
    active_camera_ids = json.dumps(["cam1", "cam2"])
    conn.execute(
        "INSERT INTO tracking_runs "
        "(id, observation_sequence_id, tracker_config_id, skeleton_id, "
        " extrinsic_calibration_id, sync_config_id, ran_at, posetrak_version, "
        " active_camera_ids, marker_names) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            ids["seq_id"],
            ids["tracker_config_id"],
            ids["skeleton_id"],
            ids["ext_cal_id"],
            ids["sync_config_id"],
            "2026-01-01T00:00:00Z",
            "dev",
            active_camera_ids,
            marker_names,
        ),
    )
    return run_id


def _insert_tracking_results(conn: sqlite3.Connection, run_id: str,
                              n_frames: int = 3,
                              is_smoothed: int = 0) -> None:
    state_blob = _make_state_blob()
    cov_diag_blob = _make_cov_diag_blob()
    for i in range(n_frames):
        conn.execute(
            "INSERT INTO tracking_results "
            "(run_id, person_id, tracker_step, is_smoothed, timestamp_s, "
            " tracking_lost, n_inlier_observations, cov_condition_number, "
            " state, cov_diag) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, 0, i, is_smoothed, float(i) * 0.01,
             0, 4, 100.0, state_blob, cov_diag_blob),
        )


def _make_session_db(tmp_path: Path) -> tuple[sqlite3.Connection, str, dict]:
    """Create a minimal session DB file; return (conn, db_path_str, ids)."""
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    conn.row_factory = sqlite3.Row
    ids = _insert_session_scaffolding(conn)
    conn.commit()
    return conn, str(db_path), ids


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def session_db_with_run(tmp_path):
    """Session DB with 3 unsmoothed tracking result frames."""
    conn, db_path, ids = _make_session_db(tmp_path)
    run_id = _insert_tracking_run(conn, ids)
    _insert_tracking_results(conn, run_id, n_frames=3)
    conn.commit()
    conn.close()
    yield db_path, run_id, ids


@pytest.fixture()
def session_db_empty_run(tmp_path):
    """Session DB with a tracking run but no result frames."""
    conn, db_path, ids = _make_session_db(tmp_path)
    run_id = _insert_tracking_run(conn, ids)
    # Intentionally insert no tracking_results rows
    conn.commit()
    conn.close()
    yield db_path, run_id, ids


# ---------------------------------------------------------------------------
# load_tracking_run_data
# ---------------------------------------------------------------------------


class TestLoadTrackingRunData:
    def test_returns_expected_keys(self, session_db_with_run):
        db_path, run_id, _ = session_db_with_run
        result = load_tracking_run_data(db_path, run_id)
        expected_keys = {
            "root_pose_df", "joint_angles_df", "cov_diag_df", "tracking_stats_df",
            "skeleton_yaml", "marker_names", "n_dof", "run_row",
        }
        assert set(result.keys()) == expected_keys

    def test_n_dof_matches_skeleton(self, session_db_with_run):
        db_path, run_id, _ = session_db_with_run
        result = load_tracking_run_data(db_path, run_id)
        assert result["n_dof"] == N_DOF

    def test_skeleton_yaml_contains_minimal_name(self, session_db_with_run):
        db_path, run_id, _ = session_db_with_run
        result = load_tracking_run_data(db_path, run_id)
        assert "minimal" in result["skeleton_yaml"]

    def test_marker_names_loaded(self, session_db_with_run):
        db_path, run_id, _ = session_db_with_run
        result = load_tracking_run_data(db_path, run_id)
        assert result["marker_names"] == ["nose"]

    def test_tracking_stats_has_n_rows(self, session_db_with_run):
        db_path, run_id, _ = session_db_with_run
        result = load_tracking_run_data(db_path, run_id)
        assert len(result["tracking_stats_df"]) == 3

    def test_tracking_stats_columns(self, session_db_with_run):
        db_path, run_id, _ = session_db_with_run
        result = load_tracking_run_data(db_path, run_id)
        df = result["tracking_stats_df"]
        for col in ("frame", "timestamp", "tracking_lost", "num_inliers", "cov_condition_number"):
            assert col in df.columns

    def test_root_pose_df_has_rows(self, session_db_with_run):
        db_path, run_id, _ = session_db_with_run
        result = load_tracking_run_data(db_path, run_id)
        assert len(result["root_pose_df"]) == 3

    def test_joint_angles_df_has_rows(self, session_db_with_run):
        db_path, run_id, _ = session_db_with_run
        result = load_tracking_run_data(db_path, run_id)
        # 3 frames × joints_per_frame (spine + head = 2 non-root joints)
        assert len(result["joint_angles_df"]) > 0

    def test_smoothed_false_loads_unsmoothed(self, tmp_path):
        conn, db_path, ids = _make_session_db(tmp_path)
        run_id = _insert_tracking_run(conn, ids)
        _insert_tracking_results(conn, run_id, n_frames=2, is_smoothed=0)
        _insert_tracking_results(conn, run_id, n_frames=2, is_smoothed=1)
        conn.commit()
        conn.close()
        result = load_tracking_run_data(db_path, run_id, smoothed=False)
        assert len(result["tracking_stats_df"]) == 2

    def test_smoothed_true_loads_smoothed(self, tmp_path):
        conn, db_path, ids = _make_session_db(tmp_path)
        run_id = _insert_tracking_run(conn, ids)
        _insert_tracking_results(conn, run_id, n_frames=2, is_smoothed=0)
        _insert_tracking_results(conn, run_id, n_frames=5, is_smoothed=1)
        conn.commit()
        conn.close()
        result = load_tracking_run_data(db_path, run_id, smoothed=True)
        assert len(result["tracking_stats_df"]) == 5

    def test_missing_run_id_raises(self, session_db_with_run):
        db_path, _, _ = session_db_with_run
        with pytest.raises((ValueError, Exception)):
            load_tracking_run_data(db_path, "nonexistent-run-id")

    def test_empty_run_returns_empty_dfs(self, session_db_empty_run):
        db_path, run_id, _ = session_db_empty_run
        with pytest.warns(UserWarning):
            result = load_tracking_run_data(db_path, run_id)
        assert result["root_pose_df"].empty
        assert result["joint_angles_df"].empty
        assert result["tracking_stats_df"].empty

    def test_run_row_contains_id(self, session_db_with_run):
        db_path, run_id, _ = session_db_with_run
        result = load_tracking_run_data(db_path, run_id)
        assert result["run_row"]["id"] == run_id

    def test_timestamps_in_stats(self, session_db_with_run):
        db_path, run_id, _ = session_db_with_run
        result = load_tracking_run_data(db_path, run_id)
        ts = result["tracking_stats_df"]["timestamp"].tolist()
        assert ts == pytest.approx([0.0, 0.01, 0.02])


# ---------------------------------------------------------------------------
# load_cameras_from_session
# ---------------------------------------------------------------------------


def _insert_cameras(conn: sqlite3.Connection, ids: dict) -> dict:
    """Insert two cameras with intrinsics and extrinsics; return camera_instance_ids."""
    cam_model_id = _new_id()
    conn.execute(
        "INSERT INTO camera_models (id, manufacturer, model_name) VALUES (?, ?, ?)",
        (cam_model_id, "TestCo", "CamModel"),
    )
    cam_mode_id = _new_id()
    conn.execute(
        "INSERT INTO camera_modes (id, camera_model_id, width_px, height_px, nominal_fps) "
        "VALUES (?, ?, ?, ?, ?)",
        (cam_mode_id, cam_model_id, 1280, 720, 60.0),
    )

    inst1_id, inst2_id = _new_id(), _new_id()
    for inst_id, label in [(inst1_id, "cam_a"), (inst2_id, "cam_b")]:
        conn.execute(
            "INSERT INTO camera_instances (id, camera_model_id, label) VALUES (?, ?, ?)",
            (inst_id, cam_model_id, label),
        )

    dist_blob = _float64_blob([-0.1, 0.05, 0.0, 0.0])
    intr1_id, intr2_id = _new_id(), _new_id()
    conn.execute(
        "INSERT INTO intrinsics_calibrations "
        "(id, camera_mode_id, calibrated_at, fx, fy, cx, cy, dist_coeffs) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (intr1_id, cam_mode_id, "2026-01-01T00:00:00Z", 800.0, 800.0, 320.0, 240.0, dist_blob),
    )
    conn.execute(
        "INSERT INTO intrinsics_calibrations "
        "(id, camera_mode_id, calibrated_at, fx, fy, cx, cy, dist_coeffs) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (intr2_id, cam_mode_id, "2026-01-01T00:00:00Z", 810.0, 810.0, 325.0, 245.0, dist_blob),
    )

    session_id = ids["session_id"]
    ext_cal_id = ids["ext_cal_id"]

    for inst_id, intr_id, label in [
        (inst1_id, intr1_id, "cam_a"),
        (inst2_id, intr2_id, "cam_b"),
    ]:
        conn.execute(
            "INSERT INTO session_cameras "
            "(session_id, camera_instance_id, camera_mode_id, intrinsics_calibration_id, label) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, inst_id, cam_mode_id, intr_id, label),
        )

    R_identity = _float64_blob(np.eye(3).flatten())
    t_zero = _float64_blob([0.0, 0.0, 2.0])
    t_other = _float64_blob([0.5, 0.0, 2.0])
    for inst_id, t_blob in [(inst1_id, t_zero), (inst2_id, t_other)]:
        conn.execute(
            "INSERT INTO extrinsic_entries "
            "(extrinsic_calibration_id, camera_instance_id, R, t) VALUES (?, ?, ?, ?)",
            (ext_cal_id, inst_id, R_identity, t_blob),
        )

    return {"inst1_id": inst1_id, "inst2_id": inst2_id,
            "intr1_id": intr1_id, "intr2_id": intr2_id}


@pytest.fixture()
def session_db_with_cameras(tmp_path):
    conn, db_path, ids = _make_session_db(tmp_path)
    cam_ids = _insert_cameras(conn, ids)
    conn.commit()
    conn.close()
    return db_path, ids, cam_ids


class TestLoadCamerasFromSession:
    def test_returns_two_cameras(self, session_db_with_cameras):
        db_path, ids, _ = session_db_with_cameras
        cams = load_cameras_from_session(db_path, ids["ext_cal_id"], ids["session_id"])
        assert len(cams) == 2

    def test_sorted_by_label(self, session_db_with_cameras):
        db_path, ids, _ = session_db_with_cameras
        cams = load_cameras_from_session(db_path, ids["ext_cal_id"], ids["session_id"])
        labels = [c["label"] for c in cams]
        assert labels == sorted(labels)

    def test_camera_id_is_sequential(self, session_db_with_cameras):
        db_path, ids, _ = session_db_with_cameras
        cams = load_cameras_from_session(db_path, ids["ext_cal_id"], ids["session_id"])
        for i, c in enumerate(cams):
            assert c["camera_id"] == i

    def test_K_shape(self, session_db_with_cameras):
        db_path, ids, _ = session_db_with_cameras
        cams = load_cameras_from_session(db_path, ids["ext_cal_id"], ids["session_id"])
        for c in cams:
            assert c["K"].shape == (3, 3)

    def test_K_diagonal_values(self, session_db_with_cameras):
        db_path, ids, _ = session_db_with_cameras
        cams = load_cameras_from_session(db_path, ids["ext_cal_id"], ids["session_id"])
        # cam_a (label alphabetically first) has fx=800
        cam_a = next(c for c in cams if c["label"] == "cam_a")
        assert cam_a["K"][0, 0] == pytest.approx(800.0)
        assert cam_a["K"][1, 1] == pytest.approx(800.0)
        assert cam_a["K"][0, 2] == pytest.approx(320.0)
        assert cam_a["K"][1, 2] == pytest.approx(240.0)

    def test_R_shape(self, session_db_with_cameras):
        db_path, ids, _ = session_db_with_cameras
        cams = load_cameras_from_session(db_path, ids["ext_cal_id"], ids["session_id"])
        for c in cams:
            assert c["R"].shape == (3, 3)

    def test_R_is_identity(self, session_db_with_cameras):
        db_path, ids, _ = session_db_with_cameras
        cams = load_cameras_from_session(db_path, ids["ext_cal_id"], ids["session_id"])
        for c in cams:
            np.testing.assert_allclose(c["R"], np.eye(3), atol=1e-9)

    def test_t_shape(self, session_db_with_cameras):
        db_path, ids, _ = session_db_with_cameras
        cams = load_cameras_from_session(db_path, ids["ext_cal_id"], ids["session_id"])
        for c in cams:
            assert c["t"].shape == (3,)

    def test_t_values(self, session_db_with_cameras):
        db_path, ids, _ = session_db_with_cameras
        cams = load_cameras_from_session(db_path, ids["ext_cal_id"], ids["session_id"])
        cam_a = next(c for c in cams if c["label"] == "cam_a")
        np.testing.assert_allclose(cam_a["t"], [0.0, 0.0, 2.0], atol=1e-9)

    def test_dist_shape(self, session_db_with_cameras):
        db_path, ids, _ = session_db_with_cameras
        cams = load_cameras_from_session(db_path, ids["ext_cal_id"], ids["session_id"])
        for c in cams:
            assert c["dist"].shape == (4,)

    def test_P_shape(self, session_db_with_cameras):
        db_path, ids, _ = session_db_with_cameras
        cams = load_cameras_from_session(db_path, ids["ext_cal_id"], ids["session_id"])
        for c in cams:
            assert c["P"].shape == (3, 4)

    def test_P_matches_K_R_t(self, session_db_with_cameras):
        db_path, ids, _ = session_db_with_cameras
        cams = load_cameras_from_session(db_path, ids["ext_cal_id"], ids["session_id"])
        for c in cams:
            expected_P = c["K"] @ np.hstack([c["R"], c["t"].reshape(3, 1)])
            np.testing.assert_allclose(c["P"], expected_P, atol=1e-9)

    def test_unknown_calibration_returns_empty(self, session_db_with_cameras):
        db_path, ids, _ = session_db_with_cameras
        cams = load_cameras_from_session(db_path, "nonexistent-id", ids["session_id"])
        assert cams == []


# ---------------------------------------------------------------------------
# load_sync_from_session
# ---------------------------------------------------------------------------


def _insert_sync_points(conn: sqlite3.Connection, ids: dict,
                         cam_inst_ids: list[str], cam_labels: list[str]) -> None:
    """Insert shot_videos and sync_points for given camera instances."""
    shot_id = ids["shot_id"]
    sync_config_id = ids["sync_config_id"]
    for inst_id, label in zip(cam_inst_ids, cam_labels):
        sv_id = _new_id()
        conn.execute(
            "INSERT INTO shot_videos "
            "(id, shot_id, camera_instance_id, file_path, first_video_frame, "
            " last_video_frame, actual_fps) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sv_id, shot_id, inst_id, f"/data/{label}.mp4", 0, 99, 60.0),
        )
        for frame, ts in [(0, 0.0), (1, 1.0 / 60.0)]:
            conn.execute(
                "INSERT INTO sync_points "
                "(sync_config_id, camera_instance_id, shot_video_id, video_frame, timestamp_s) "
                "VALUES (?, ?, ?, ?, ?)",
                (sync_config_id, inst_id, sv_id, frame, ts),
            )
        # Update label on camera_instances for the JOIN to work
        conn.execute(
            "UPDATE camera_instances SET label = ? WHERE id = ?",
            (label, inst_id),
        )


@pytest.fixture()
def session_db_with_sync(tmp_path):
    conn, db_path, ids = _make_session_db(tmp_path)
    cam_model_id = _new_id()
    conn.execute(
        "INSERT INTO camera_models (id, manufacturer, model_name) VALUES (?, ?, ?)",
        (cam_model_id, "TestCo", "Model"),
    )
    inst1, inst2 = _new_id(), _new_id()
    for iid, lbl in [(inst1, "sync_cam1"), (inst2, "sync_cam2")]:
        conn.execute(
            "INSERT INTO camera_instances (id, camera_model_id, label) VALUES (?, ?, ?)",
            (iid, cam_model_id, lbl),
        )
    _insert_sync_points(conn, ids, [inst1, inst2], ["sync_cam1", "sync_cam2"])
    conn.commit()
    conn.close()
    return db_path, ids


class TestLoadSyncFromSession:
    def test_returns_two_cameras(self, session_db_with_sync):
        db_path, ids = session_db_with_sync
        sync = load_sync_from_session(db_path, ids["sync_config_id"])
        assert len(sync) == 2

    def test_cam_labels_in_result(self, session_db_with_sync):
        db_path, ids = session_db_with_sync
        sync = load_sync_from_session(db_path, ids["sync_config_id"])
        assert "sync_cam1" in sync
        assert "sync_cam2" in sync

    def test_syncpoints_is_list(self, session_db_with_sync):
        db_path, ids = session_db_with_sync
        sync = load_sync_from_session(db_path, ids["sync_config_id"])
        for cam_data in sync.values():
            assert isinstance(cam_data["syncpoints"], list)

    def test_syncpoints_count(self, session_db_with_sync):
        db_path, ids = session_db_with_sync
        sync = load_sync_from_session(db_path, ids["sync_config_id"])
        for cam_data in sync.values():
            assert len(cam_data["syncpoints"]) == 2

    def test_syncpoint_fields(self, session_db_with_sync):
        db_path, ids = session_db_with_sync
        sync = load_sync_from_session(db_path, ids["sync_config_id"])
        sp = sync["sync_cam1"]["syncpoints"][0]
        assert "frame" in sp
        assert "timestamp" in sp

    def test_syncpoint_values(self, session_db_with_sync):
        db_path, ids = session_db_with_sync
        sync = load_sync_from_session(db_path, ids["sync_config_id"])
        sp = sync["sync_cam1"]["syncpoints"][0]
        assert sp["frame"] == 0
        assert sp["timestamp"] == pytest.approx(0.0)

    def test_unknown_sync_config_returns_empty(self, session_db_with_sync):
        db_path, _ = session_db_with_sync
        result = load_sync_from_session(db_path, "nonexistent-sync-id")
        assert result == {}


# ---------------------------------------------------------------------------
# load_observations_from_session
# ---------------------------------------------------------------------------


def _insert_pose_observations(conn: sqlite3.Connection, ids: dict,
                               inst_id: str, n_frames: int = 3,
                               n_kp: int = 5) -> None:
    seq_id = ids["seq_id"]
    for frame in range(n_frames):
        kp = np.zeros((n_kp, 3), dtype="<f4")
        # Keypoint 0: (100, 200, 0.9), keypoint 1: (0, 0, 0.0) — filtered out
        kp[0] = [100.0, 200.0, 0.9]
        kp[1] = [0.0, 0.0, 0.0]
        kp[2] = [50.0, 75.0, 0.5]
        conn.execute(
            "INSERT INTO pose_observations "
            "(sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (seq_id, inst_id, frame, float(frame) * 0.01, 0, kp.tobytes()),
        )


@pytest.fixture()
def session_db_with_obs(tmp_path):
    conn, db_path, ids = _make_session_db(tmp_path)
    cam_model_id = _new_id()
    conn.execute(
        "INSERT INTO camera_models (id, manufacturer, model_name) VALUES (?, ?, ?)",
        (cam_model_id, "TestCo", "Model"),
    )
    inst_id = _new_id()
    conn.execute(
        "INSERT INTO camera_instances (id, camera_model_id, label) VALUES (?, ?, ?)",
        (inst_id, cam_model_id, "obs_cam"),
    )
    _insert_pose_observations(conn, ids, inst_id, n_frames=3, n_kp=5)
    conn.commit()
    conn.close()
    return db_path, ids, inst_id


class TestLoadObservationsFromSession:
    def test_returns_dataframe(self, session_db_with_obs):
        db_path, ids, inst_id = session_db_with_obs
        camera_label_map = {inst_id: 0}
        df = load_observations_from_session(db_path, ids["seq_id"], camera_label_map)
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self, session_db_with_obs):
        db_path, ids, inst_id = session_db_with_obs
        camera_label_map = {inst_id: 0}
        df = load_observations_from_session(db_path, ids["seq_id"], camera_label_map)
        for col in ("frame", "timestamp", "camera_id", "keypoint_index", "pixel_x", "pixel_y", "confidence"):
            assert col in df.columns

    def test_zero_confidence_excluded(self, session_db_with_obs):
        """Keypoints with confidence=0.0 should not appear in the result."""
        db_path, ids, inst_id = session_db_with_obs
        camera_label_map = {inst_id: 0}
        df = load_observations_from_session(db_path, ids["seq_id"], camera_label_map)
        # keypoint index 1 has confidence 0.0 and should be absent
        assert (df["confidence"] > 0.0).all()

    def test_pixel_values_correct(self, session_db_with_obs):
        db_path, ids, inst_id = session_db_with_obs
        camera_label_map = {inst_id: 0}
        df = load_observations_from_session(db_path, ids["seq_id"], camera_label_map)
        row = df[(df["keypoint_index"] == 0) & (df["frame"] == 0)].iloc[0]
        assert row["pixel_x"] == pytest.approx(100.0, abs=0.01)
        assert row["pixel_y"] == pytest.approx(200.0, abs=0.01)
        assert row["confidence"] == pytest.approx(0.9, abs=0.01)

    def test_camera_id_assigned(self, session_db_with_obs):
        db_path, ids, inst_id = session_db_with_obs
        camera_label_map = {inst_id: 7}  # arbitrary int
        df = load_observations_from_session(db_path, ids["seq_id"], camera_label_map)
        assert (df["camera_id"] == 7).all()

    def test_unknown_camera_excluded(self, session_db_with_obs):
        db_path, ids, _ = session_db_with_obs
        # Provide a map that doesn't include the actual inst_id
        df = load_observations_from_session(db_path, ids["seq_id"], {})
        assert df.empty

    def test_n_frames_in_result(self, session_db_with_obs):
        db_path, ids, inst_id = session_db_with_obs
        camera_label_map = {inst_id: 0}
        df = load_observations_from_session(db_path, ids["seq_id"], camera_label_map)
        assert df["frame"].nunique() == 3


# ---------------------------------------------------------------------------
# load_tracking_run_with_markers
# ---------------------------------------------------------------------------


class TestLoadTrackingRunWithMarkers:
    def test_returns_marker_positions_df_key(self, session_db_with_run):
        db_path, run_id, _ = session_db_with_run
        result = load_tracking_run_with_markers(db_path, run_id)
        assert "marker_positions_df" in result

    def test_marker_positions_df_not_empty(self, session_db_with_run):
        db_path, run_id, _ = session_db_with_run
        result = load_tracking_run_with_markers(db_path, run_id)
        assert not result["marker_positions_df"].empty

    def test_marker_positions_columns(self, session_db_with_run):
        db_path, run_id, _ = session_db_with_run
        result = load_tracking_run_with_markers(db_path, run_id)
        df = result["marker_positions_df"]
        for col in ("frame", "timestamp", "marker_name", "x_3d", "y_3d", "z_3d"):
            assert col in df.columns

    def test_marker_name_in_result(self, session_db_with_run):
        db_path, run_id, _ = session_db_with_run
        result = load_tracking_run_with_markers(db_path, run_id)
        names = set(result["marker_positions_df"]["marker_name"])
        assert "nose" in names

    def test_n_frames_in_marker_df(self, session_db_with_run):
        db_path, run_id, _ = session_db_with_run
        result = load_tracking_run_with_markers(db_path, run_id)
        # 3 frames × 1 marker = 3 rows
        assert len(result["marker_positions_df"]) == 3

    def test_empty_run_gives_empty_marker_df(self, session_db_empty_run):
        db_path, run_id, _ = session_db_empty_run
        with pytest.warns(UserWarning):
            result = load_tracking_run_with_markers(db_path, run_id)
        assert result["marker_positions_df"].empty

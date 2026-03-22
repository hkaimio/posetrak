"""Tests for scripts/db/import_pose_json.py."""

from __future__ import annotations

import json
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))  # project root

from scripts.db.import_pose_json import PoseImportResult, import_pose_json
from scripts.db.import_extrinsics import import_extrinsics
from scripts.db.import_sync_json import import_sync_json
from scripts.db.posetrak_db import add_shot_video, create_shot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_setup(
    session_conn: sqlite3.Connection,
    session_id: str,
    inst1: str,
    inst2: str,
    sample_calib_toml: Path,
    sample_sync_json: Path,
) -> tuple[str, str]:
    """Create extrinsics, shot, shot_videos, and sync config. Returns (shot_id, sync_config_id)."""
    ext_result = import_extrinsics(
        session_conn, session_id, sample_calib_toml,
        {"cam1": inst1, "cam2": inst2},
    )
    shot_id = create_shot(session_conn, session_id, ext_result.extrinsic_calibration_id)

    add_shot_video(session_conn, shot_id, inst1, "/videos/cam1.mp4", 0, 100, 120.0)
    add_shot_video(session_conn, shot_id, inst2, "/videos/cam2.mp4", 0, 100, 120.0)

    sync_result = import_sync_json(
        session_conn, shot_id, sample_sync_json,
        {"cam1": inst1, "cam2": inst2},
    )
    return shot_id, sync_result.sync_config_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pose_import_returns_result(
    session_db_full, sample_calib_toml: Path,
    sample_sync_json: Path, sample_pose_dir: Path,
) -> None:
    """import_pose_json() should return a PoseImportResult."""
    session_conn, session_id, inst1, inst2 = session_db_full
    shot_id, sync_config_id = _full_setup(
        session_conn, session_id, inst1, inst2, sample_calib_toml, sample_sync_json
    )
    result = import_pose_json(
        session_conn, shot_id, sync_config_id, sample_pose_dir,
        {"cam1": inst1, "cam2": inst2},
    )
    assert isinstance(result, PoseImportResult)
    assert result.sequence_id


def test_pose_import_creates_sequence(
    session_db_full, sample_calib_toml: Path,
    sample_sync_json: Path, sample_pose_dir: Path,
) -> None:
    """One pose_observation_sequences row should be created."""
    session_conn, session_id, inst1, inst2 = session_db_full
    shot_id, sync_config_id = _full_setup(
        session_conn, session_id, inst1, inst2, sample_calib_toml, sample_sync_json
    )
    result = import_pose_json(
        session_conn, shot_id, sync_config_id, sample_pose_dir,
        {"cam1": inst1, "cam2": inst2},
    )
    row = session_conn.execute(
        "SELECT id, shot_id FROM pose_observation_sequences WHERE id = ?",
        (result.sequence_id,),
    ).fetchone()
    assert row is not None
    assert row["shot_id"] == shot_id


def test_pose_import_creates_observations(
    session_db_full, sample_calib_toml: Path,
    sample_sync_json: Path, sample_pose_dir: Path,
) -> None:
    """The correct number of pose_observations rows should be created."""
    session_conn, session_id, inst1, inst2 = session_db_full
    shot_id, sync_config_id = _full_setup(
        session_conn, session_id, inst1, inst2, sample_calib_toml, sample_sync_json
    )
    result = import_pose_json(
        session_conn, shot_id, sync_config_id, sample_pose_dir,
        {"cam1": inst1, "cam2": inst2},
    )
    # 2 cameras × 3 frames = 6 observations
    assert result.n_observations == 6
    count = session_conn.execute(
        "SELECT COUNT(*) FROM pose_observations WHERE sequence_id = ?",
        (result.sequence_id,),
    ).fetchone()[0]
    assert count == 6


def test_pose_import_kp_blob_shape(
    session_db_full, sample_calib_toml: Path,
    sample_sync_json: Path, sample_pose_dir: Path,
) -> None:
    """kp_blob should decode to a float32 array of shape [133, 3]."""
    session_conn, session_id, inst1, inst2 = session_db_full
    shot_id, sync_config_id = _full_setup(
        session_conn, session_id, inst1, inst2, sample_calib_toml, sample_sync_json
    )
    result = import_pose_json(
        session_conn, shot_id, sync_config_id, sample_pose_dir,
        {"cam1": inst1, "cam2": inst2},
    )
    row = session_conn.execute(
        "SELECT kp_blob FROM pose_observations WHERE sequence_id = ? LIMIT 1",
        (result.sequence_id,),
    ).fetchone()
    blob: bytes = row["kp_blob"]
    arr = np.frombuffer(blob, dtype=np.float32).reshape(-1, 3)
    assert arr.shape == (133, 3)


def test_pose_import_timestamp_computed_from_sync(
    session_db_full, sample_calib_toml: Path,
    sample_sync_json: Path, sample_pose_dir: Path,
) -> None:
    """Timestamps should be computed as ref_ts + (frame - ref_frame) / fps."""
    session_conn, session_id, inst1, inst2 = session_db_full
    shot_id, sync_config_id = _full_setup(
        session_conn, session_id, inst1, inst2, sample_calib_toml, sample_sync_json
    )
    result = import_pose_json(
        session_conn, shot_id, sync_config_id, sample_pose_dir,
        {"cam1": inst1, "cam2": inst2},
    )
    # cam1: ref_frame=0, ref_ts=0.0, fps=120.0
    # frame 0 → ts=0.0; frame 1 → ts=1/120; frame 2 → ts=2/120
    rows = session_conn.execute(
        "SELECT video_frame, timestamp_s FROM pose_observations "
        "WHERE sequence_id = ? AND camera_instance_id = ? "
        "ORDER BY video_frame",
        (result.sequence_id, inst1),
    ).fetchall()
    assert len(rows) == 3
    assert rows[0]["timestamp_s"] == pytest.approx(0.0)
    assert rows[1]["timestamp_s"] == pytest.approx(1.0 / 120.0)
    assert rows[2]["timestamp_s"] == pytest.approx(2.0 / 120.0)


def test_pose_import_time_range_filter(
    session_db_full, sample_calib_toml: Path,
    sample_sync_json: Path, sample_pose_dir: Path,
) -> None:
    """time_start / time_end filter should exclude out-of-range frames.

    cam1 sync anchor: ref_frame=0, ref_ts=0.0, fps=120 → frame 1 ts≈0.00833
    cam2 sync anchor: ref_frame=0, ref_ts=0.004, fps=120 → frame 1 ts≈0.01233

    Filtering [0.005, 0.012] selects:
      cam1 frame 1 (ts≈0.00833) → included
      cam2 frame 1 (ts≈0.01233) → excluded (above 0.012)
    → 1 observation total
    """
    session_conn, session_id, inst1, inst2 = session_db_full
    shot_id, sync_config_id = _full_setup(
        session_conn, session_id, inst1, inst2, sample_calib_toml, sample_sync_json
    )
    result = import_pose_json(
        session_conn, shot_id, sync_config_id, sample_pose_dir,
        {"cam1": inst1, "cam2": inst2},
        time_start=0.005,
        time_end=0.012,
    )
    # cam1 frame 1 passes, cam2 frame 1 does not (ts≈0.01233 > 0.012)
    assert result.n_observations == 1


def test_pose_import_skip_missing_person(
    session_db_full, sample_calib_toml: Path,
    sample_sync_json: Path, tmp_path: Path,
) -> None:
    """Frames where the requested person_id is absent are skipped."""
    session_conn, session_id, inst1, inst2 = session_db_full
    # Create pose dir with person_id=0 only; request person_id=1
    pose_dir = tmp_path / "pose_no_pid1"
    (pose_dir / "cam1").mkdir(parents=True)
    for frame in range(2):
        kps = [0.0] * (133 * 3)
        data = {"version": 1.3, "people": [{"person_id": [0], "pose_keypoints_2d": kps}]}
        (pose_dir / "cam1" / f"cam1_{frame:06d}.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
    # Also create cam2 dir (empty) to avoid missing dir issues
    (pose_dir / "cam2").mkdir(parents=True)

    shot_id, sync_config_id = _full_setup(
        session_conn, session_id, inst1, inst2, sample_calib_toml, sample_sync_json
    )
    result = import_pose_json(
        session_conn, shot_id, sync_config_id, pose_dir,
        {"cam1": inst1, "cam2": inst2},
        person_ids=[1],  # no one has person_id=1
    )
    assert result.n_observations == 0


def test_pose_import_skip_unlisted_camera(
    session_db_full, sample_calib_toml: Path,
    sample_sync_json: Path, sample_pose_dir: Path,
) -> None:
    """Cameras not in the per-camera mapping are skipped and appear in result.skipped_cameras."""
    session_conn, session_id, inst1, inst2 = session_db_full
    shot_id, sync_config_id = _full_setup(
        session_conn, session_id, inst1, inst2, sample_calib_toml, sample_sync_json
    )
    result = import_pose_json(
        session_conn, shot_id, sync_config_id, sample_pose_dir,
        {"cam1": inst1},  # cam2 not listed
    )
    # Only cam1 imported: 3 frames
    assert result.n_observations == 3
    assert "cam2" in result.skipped_cameras

"""Tests for scripts/db/import_sync_json.py."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))  # project root

from scripts.db.import_sync_json import SyncImportResult, import_sync_json
from scripts.db.import_extrinsics import import_extrinsics
from scripts.db.posetrak_db import add_shot_video, create_shot


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _setup_shot_with_videos(
    session_conn: sqlite3.Connection,
    session_id: str,
    inst1: str,
    inst2: str,
    sample_calib_toml: Path,
) -> tuple[str, str]:
    """Create extrinsics, a shot, and two shot_videos. Returns (shot_id, ext_id)."""
    ext_result = import_extrinsics(
        session_conn, session_id, sample_calib_toml,
        {"cam1": inst1, "cam2": inst2},
    )
    ext_id = ext_result.extrinsic_calibration_id

    shot_id = create_shot(session_conn, session_id, ext_id)

    add_shot_video(
        session_conn, shot_id, inst1,
        "/videos/cam1.mp4", 0, 1000, 120.0,
    )
    add_shot_video(
        session_conn, shot_id, inst2,
        "/videos/cam2.mp4", 0, 1000, 120.0,
    )
    return shot_id, ext_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sync_import_returns_result(
    session_db_full, sample_calib_toml: Path, sample_sync_json: Path
) -> None:
    """import_sync_json() should return a SyncImportResult."""
    session_conn, session_id, inst1, inst2 = session_db_full
    shot_id, _ = _setup_shot_with_videos(
        session_conn, session_id, inst1, inst2, sample_calib_toml
    )
    result = import_sync_json(
        session_conn, shot_id, sample_sync_json,
        {"cam1": inst1, "cam2": inst2},
    )
    assert isinstance(result, SyncImportResult)
    assert result.sync_config_id


def test_sync_import_creates_sync_config(
    session_db_full, sample_calib_toml: Path, sample_sync_json: Path
) -> None:
    """One sync_configs row should be created."""
    session_conn, session_id, inst1, inst2 = session_db_full
    shot_id, _ = _setup_shot_with_videos(
        session_conn, session_id, inst1, inst2, sample_calib_toml
    )
    result = import_sync_json(
        session_conn, shot_id, sample_sync_json,
        {"cam1": inst1, "cam2": inst2},
    )
    row = session_conn.execute(
        "SELECT id, shot_id FROM sync_configs WHERE id = ?",
        (result.sync_config_id,),
    ).fetchone()
    assert row is not None
    assert row["shot_id"] == shot_id


def test_sync_import_creates_two_sync_points(
    session_db_full, sample_calib_toml: Path, sample_sync_json: Path
) -> None:
    """Two sync_points rows should be created for a two-camera sync JSON."""
    session_conn, session_id, inst1, inst2 = session_db_full
    shot_id, _ = _setup_shot_with_videos(
        session_conn, session_id, inst1, inst2, sample_calib_toml
    )
    result = import_sync_json(
        session_conn, shot_id, sample_sync_json,
        {"cam1": inst1, "cam2": inst2},
    )
    count = session_conn.execute(
        "SELECT COUNT(*) FROM sync_points WHERE sync_config_id = ?",
        (result.sync_config_id,),
    ).fetchone()[0]
    assert count == 2


def test_sync_import_uses_first_syncpoint_frame(
    session_db_full, sample_calib_toml: Path, sample_sync_json: Path
) -> None:
    """sync_points.video_frame should be the first syncpoint frame (0 for cam1)."""
    session_conn, session_id, inst1, inst2 = session_db_full
    shot_id, _ = _setup_shot_with_videos(
        session_conn, session_id, inst1, inst2, sample_calib_toml
    )
    result = import_sync_json(
        session_conn, shot_id, sample_sync_json,
        {"cam1": inst1, "cam2": inst2},
    )
    row = session_conn.execute(
        "SELECT video_frame FROM sync_points "
        "WHERE sync_config_id = ? AND camera_instance_id = ?",
        (result.sync_config_id, inst1),
    ).fetchone()
    assert row["video_frame"] == 0


def test_sync_import_uses_first_syncpoint_timestamp(
    session_db_full, sample_calib_toml: Path, sample_sync_json: Path
) -> None:
    """sync_points.timestamp_s for cam1 should be 0.0 (first syncpoint timestamp)."""
    session_conn, session_id, inst1, inst2 = session_db_full
    shot_id, _ = _setup_shot_with_videos(
        session_conn, session_id, inst1, inst2, sample_calib_toml
    )
    result = import_sync_json(
        session_conn, shot_id, sample_sync_json,
        {"cam1": inst1, "cam2": inst2},
    )
    row = session_conn.execute(
        "SELECT timestamp_s FROM sync_points "
        "WHERE sync_config_id = ? AND camera_instance_id = ?",
        (result.sync_config_id, inst1),
    ).fetchone()
    assert row["timestamp_s"] == pytest.approx(0.0)


def test_sync_import_links_correct_shot_video(
    session_db_full, sample_calib_toml: Path, sample_sync_json: Path
) -> None:
    """sync_points.shot_video_id should reference the correct shot_videos row."""
    session_conn, session_id, inst1, inst2 = session_db_full
    shot_id, _ = _setup_shot_with_videos(
        session_conn, session_id, inst1, inst2, sample_calib_toml
    )
    result = import_sync_json(
        session_conn, shot_id, sample_sync_json,
        {"cam1": inst1, "cam2": inst2},
    )
    # Get the shot_video_id from the sync_points row and verify it exists.
    row = session_conn.execute(
        "SELECT shot_video_id FROM sync_points "
        "WHERE sync_config_id = ? AND camera_instance_id = ?",
        (result.sync_config_id, inst1),
    ).fetchone()
    sv_row = session_conn.execute(
        "SELECT id FROM shot_videos WHERE id = ?",
        (row["shot_video_id"],),
    ).fetchone()
    assert sv_row is not None


def test_sync_import_skip_unlisted_camera(
    session_db_full, sample_calib_toml: Path, sample_sync_json: Path
) -> None:
    """Cameras not in the per-camera mapping are skipped."""
    session_conn, session_id, inst1, inst2 = session_db_full
    shot_id, _ = _setup_shot_with_videos(
        session_conn, session_id, inst1, inst2, sample_calib_toml
    )
    result = import_sync_json(
        session_conn, shot_id, sample_sync_json,
        {"cam1": inst1},  # cam2 not listed
    )
    assert "cam2" in result.skipped
    count = session_conn.execute(
        "SELECT COUNT(*) FROM sync_points WHERE sync_config_id = ?",
        (result.sync_config_id,),
    ).fetchone()[0]
    assert count == 1


def test_sync_import_fails_without_shot_video(
    session_db_full, sample_calib_toml: Path, sample_sync_json: Path
) -> None:
    """ValueError is raised when no shot_videos row exists for a listed camera."""
    session_conn, session_id, inst1, inst2 = session_db_full
    ext_result = import_extrinsics(
        session_conn, session_id, sample_calib_toml,
        {"cam1": inst1, "cam2": inst2},
    )
    shot_id = create_shot(session_conn, session_id, ext_result.extrinsic_calibration_id)
    # No shot_videos added — should raise ValueError
    with pytest.raises(ValueError, match="shot_videos"):
        import_sync_json(
            session_conn, shot_id, sample_sync_json,
            {"cam1": inst1, "cam2": inst2},
        )

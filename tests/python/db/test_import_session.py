"""Tests for session CRUD functions added to scripts/db/posetrak_db.py."""

from __future__ import annotations

import datetime
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))  # project root

from scripts.db.posetrak_db import (
    add_session_camera,
    add_shot_video,
    create_camera_model,
    create_camera_mode,
    create_mocap_session,
    create_registry,
    create_session,
    create_shot,
)


# ---------------------------------------------------------------------------
# create_mocap_session
# ---------------------------------------------------------------------------


def test_create_mocap_session_returns_id(session_db: sqlite3.Connection) -> None:
    """create_mocap_session() should return a non-empty UUID string."""
    session_id = create_mocap_session(session_db)
    assert session_id
    row = session_db.execute(
        "SELECT id FROM mocap_sessions WHERE id = ?", (session_id,)
    ).fetchone()
    assert row is not None


def test_create_mocap_session_default_date(session_db: sqlite3.Connection) -> None:
    """When recorded_at is None, it defaults to today's ISO date."""
    today = datetime.date.today().isoformat()
    session_id = create_mocap_session(session_db)
    row = session_db.execute(
        "SELECT recorded_at FROM mocap_sessions WHERE id = ?", (session_id,)
    ).fetchone()
    assert row["recorded_at"] == today


def test_create_mocap_session_custom_date(session_db: sqlite3.Connection) -> None:
    """A custom recorded_at string is stored as given."""
    session_id = create_mocap_session(session_db, recorded_at="2025-06-01")
    row = session_db.execute(
        "SELECT recorded_at FROM mocap_sessions WHERE id = ?", (session_id,)
    ).fetchone()
    assert row["recorded_at"] == "2025-06-01"


def test_create_mocap_session_stores_location_and_notes(
    session_db: sqlite3.Connection,
) -> None:
    """location and notes are stored correctly."""
    session_id = create_mocap_session(
        session_db, location="Lab A", notes="warmup session"
    )
    row = session_db.execute(
        "SELECT location, notes FROM mocap_sessions WHERE id = ?", (session_id,)
    ).fetchone()
    assert row["location"] == "Lab A"
    assert row["notes"] == "warmup session"


# ---------------------------------------------------------------------------
# add_session_camera
# ---------------------------------------------------------------------------


def _make_registry_with_camera(tmp_path: Path) -> tuple[sqlite3.Connection, str, str, str]:
    """Create a registry with one camera model, mode, instance, and intrinsics row.

    Returns (registry_conn, camera_instance_id, camera_mode_id, intrinsics_id).
    """
    import struct
    import datetime
    reg_path = tmp_path / "reg_for_session.db"
    reg = create_registry(reg_path)
    model_id = create_camera_model(reg, manufacturer="TestCo", model_name="Cam")
    mode_id = create_camera_mode(reg, model_id, width_px=1280, height_px=720)
    inst_id = reg.execute(
        "INSERT INTO camera_instances (id, camera_model_id, serial_number, label) "
        "VALUES ('inst-uuid-1', ?, '', 'cam1') RETURNING id",
        (model_id,),
    ).fetchone()[0]
    reg.commit()
    dist_blob = struct.pack("<4d", 0.0, 0.0, 0.0, 0.0)
    intr_id = "intr-uuid-1"
    reg.execute(
        "INSERT INTO intrinsics_calibrations "
        "(id, camera_mode_id, calibrated_at, distortion_model, fx, fy, cx, cy, dist_coeffs) "
        "VALUES (?, ?, ?, 'radtan', 800.0, 800.0, 320.0, 240.0, ?)",
        (intr_id, mode_id, datetime.date.today().isoformat(), dist_blob),
    )
    reg.commit()
    return reg, inst_id, mode_id, intr_id


def test_add_session_camera_creates_row(
    session_db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    """add_session_camera() inserts a session_cameras row."""
    reg, inst_id, mode_id, intr_id = _make_registry_with_camera(tmp_path)
    session_id = create_mocap_session(session_db)
    add_session_camera(
        session_db,
        reg,
        session_id,
        inst_id,
        mode_id,
        intr_id,
        label="cam1",
    )
    reg.close()
    row = session_db.execute(
        "SELECT * FROM session_cameras WHERE session_id = ?", (session_id,)
    ).fetchone()
    assert row is not None
    assert row["camera_instance_id"] == inst_id
    assert row["label"] == "cam1"


def test_add_session_camera_duplicate_raises(
    session_db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    """Inserting the same (session_id, camera_instance_id) pair twice raises IntegrityError."""
    reg, inst_id, mode_id, intr_id = _make_registry_with_camera(tmp_path)
    session_id = create_mocap_session(session_db)
    add_session_camera(session_db, reg, session_id, inst_id, mode_id, intr_id)
    with pytest.raises(sqlite3.IntegrityError):
        add_session_camera(session_db, reg, session_id, inst_id, mode_id, intr_id)
    reg.close()


# ---------------------------------------------------------------------------
# create_shot
# ---------------------------------------------------------------------------

def _make_extrinsic(session_db: sqlite3.Connection, session_id: str) -> str:
    """Insert a minimal extrinsic_calibrations row and return its ID."""
    import uuid
    ext_id = str(uuid.uuid4())
    session_db.execute(
        "INSERT INTO extrinsic_calibrations (id, session_id, calibrated_at) "
        "VALUES (?, ?, '2025-01-01')",
        (ext_id, session_id),
    )
    session_db.commit()
    return ext_id


def test_create_shot_returns_id(session_db: sqlite3.Connection) -> None:
    """create_shot() should return a non-empty UUID string."""
    session_id = create_mocap_session(session_db)
    ext_id = _make_extrinsic(session_db, session_id)
    shot_id = create_shot(session_db, session_id, ext_id)
    assert shot_id
    row = session_db.execute(
        "SELECT id FROM shots WHERE id = ?", (shot_id,)
    ).fetchone()
    assert row is not None


def test_create_shot_auto_number(session_db: sqlite3.Connection) -> None:
    """create_shot() without shot_number auto-increments from 1."""
    session_id = create_mocap_session(session_db)
    ext_id = _make_extrinsic(session_db, session_id)
    id1 = create_shot(session_db, session_id, ext_id)
    id2 = create_shot(session_db, session_id, ext_id)
    n1 = session_db.execute(
        "SELECT shot_number FROM shots WHERE id = ?", (id1,)
    ).fetchone()["shot_number"]
    n2 = session_db.execute(
        "SELECT shot_number FROM shots WHERE id = ?", (id2,)
    ).fetchone()["shot_number"]
    assert n1 == 1
    assert n2 == 2


def test_create_shot_explicit_number(session_db: sqlite3.Connection) -> None:
    """create_shot() with an explicit shot_number stores that number."""
    session_id = create_mocap_session(session_db)
    ext_id = _make_extrinsic(session_db, session_id)
    shot_id = create_shot(session_db, session_id, ext_id, shot_number=42)
    row = session_db.execute(
        "SELECT shot_number FROM shots WHERE id = ?", (shot_id,)
    ).fetchone()
    assert row["shot_number"] == 42


# ---------------------------------------------------------------------------
# add_shot_video
# ---------------------------------------------------------------------------


def test_add_shot_video_returns_id(session_db: sqlite3.Connection) -> None:
    """add_shot_video() should return a non-empty UUID string."""
    session_id = create_mocap_session(session_db)
    ext_id = _make_extrinsic(session_db, session_id)
    shot_id = create_shot(session_db, session_id, ext_id)
    video_id = add_shot_video(
        session_db, shot_id, "cam-inst-1", "/data/video.mp4", 0, 999, 119.88
    )
    assert video_id
    row = session_db.execute(
        "SELECT id FROM shot_videos WHERE id = ?", (video_id,)
    ).fetchone()
    assert row is not None


def test_add_shot_video_stores_path_and_fps(session_db: sqlite3.Connection) -> None:
    """add_shot_video() stores the file path and fps correctly."""
    session_id = create_mocap_session(session_db)
    ext_id = _make_extrinsic(session_db, session_id)
    shot_id = create_shot(session_db, session_id, ext_id)
    video_id = add_shot_video(
        session_db, shot_id, "cam-inst-1", "/mnt/d/videos/take1.mp4", 10, 500, 119.88
    )
    row = session_db.execute(
        "SELECT file_path, actual_fps, first_video_frame, last_video_frame "
        "FROM shot_videos WHERE id = ?",
        (video_id,),
    ).fetchone()
    assert row["file_path"] == "/mnt/d/videos/take1.mp4"
    assert row["actual_fps"] == pytest.approx(119.88)
    assert row["first_video_frame"] == 10
    assert row["last_video_frame"] == 500

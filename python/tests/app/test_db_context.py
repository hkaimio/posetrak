"""Tests for app.setup.db_context (DBContext, SyncTable, typed return types)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from posetrak.db.db import (
    create_session,
    create_mocap_session,
    SESSION_SCHEMA_VERSION,
)
from app.setup.db_context import (
    DBContext,
    ExtrinsicEntry,
    ShotVideoInfo,
    SyncPoint,
    SyncTable,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def session_conn(tmp_path: Path) -> sqlite3.Connection:
    """Open a fresh session DB; close after test."""
    conn = create_session(tmp_path / "test.db")
    yield conn
    conn.close()


@pytest.fixture()
def ctx(session_conn: sqlite3.Connection) -> DBContext:
    """DBContext wired to a freshly created session."""
    session_id = create_mocap_session(session_conn, location="test")
    return DBContext(session_conn, session_id)


@pytest.fixture()
def cam_instance_id(session_conn: sqlite3.Connection) -> str:
    """Insert a minimal camera_instances row and return its ID."""
    cid = "cam-instance-test"
    session_conn.execute(
        "INSERT OR IGNORE INTO camera_models (id, manufacturer, model_name) "
        "VALUES ('mdl-1', 'TestCo', 'TestCam')"
    )
    session_conn.execute(
        "INSERT OR IGNORE INTO camera_instances (id, camera_model_id, label) "
        "VALUES (?, 'mdl-1', 'cam1')",
        (cid,),
    )
    session_conn.commit()
    return cid


# ---------------------------------------------------------------------------
# SyncTable unit tests
# ---------------------------------------------------------------------------


def test_sync_table_single_anchor_extrapolates() -> None:
    """With one anchor, lookup extrapolates forward using fps."""
    pts = [SyncPoint("cam1", "vid1", video_frame=100, timestamp_s=1.0)]
    table = SyncTable(pts, fps_by_video={"vid1": 30.0})

    assert table.lookup(1.0, "vid1") == 100
    assert table.lookup(2.0, "vid1") == 130   # +1 s × 30 fps
    assert table.lookup(0.0, "vid1") == 70    # −1 s × 30 fps


def test_sync_table_two_anchors_interpolates() -> None:
    """With two anchors, lookup interpolates linearly between them."""
    pts = [
        SyncPoint("cam1", "vid1", video_frame=0,   timestamp_s=0.0),
        SyncPoint("cam1", "vid1", video_frame=120,  timestamp_s=1.0),
    ]
    table = SyncTable(pts, fps_by_video={"vid1": 120.0})

    assert table.lookup(0.0, "vid1") == 0
    assert table.lookup(0.5, "vid1") == 60
    assert table.lookup(1.0, "vid1") == 120


def test_sync_table_unknown_video_returns_none() -> None:
    pts = [SyncPoint("cam1", "vid1", video_frame=0, timestamp_s=0.0)]
    table = SyncTable(pts, fps_by_video={"vid1": 30.0})
    assert table.lookup(0.0, "vid-unknown") is None


def test_sync_table_no_fps_snaps_to_nearest() -> None:
    """Without fps, lookup snaps to the nearest anchor frame."""
    pts = [
        SyncPoint("cam1", "vid1", video_frame=10, timestamp_s=1.0),
        SyncPoint("cam1", "vid1", video_frame=20, timestamp_s=2.0),
    ]
    table = SyncTable(pts, fps_by_video={"vid1": 0.0})

    assert table.lookup(1.4, "vid1") == 10
    assert table.lookup(1.6, "vid1") == 20


# ---------------------------------------------------------------------------
# DBContext.create_shot
# ---------------------------------------------------------------------------


def test_create_shot_inserts_row(ctx: DBContext, session_conn: sqlite3.Connection) -> None:
    shot_id = ctx.create_shot("test-shot", shot_number=1)
    row = session_conn.execute(
        "SELECT label, shot_number FROM shots WHERE id = ?", (shot_id,)
    ).fetchone()
    assert row is not None
    assert row["label"] == "test-shot"
    assert row["shot_number"] == 1


# ---------------------------------------------------------------------------
# DBContext.create_shot_video
# ---------------------------------------------------------------------------


def test_create_shot_video_inserts_row(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
    cam_instance_id: str,
) -> None:
    shot_id = ctx.create_shot("s1", shot_number=1)
    vid_id = ctx.create_shot_video(
        shot_id, cam_instance_id, "/path/to/video.mp4",
        fps=120.0, frame_count=1000, width=1920, height=1080,
    )
    row = session_conn.execute(
        "SELECT file_path, actual_fps, first_video_frame, last_video_frame "
        "FROM shot_videos WHERE id = ?",
        (vid_id,),
    ).fetchone()
    assert row is not None
    assert row["file_path"] == "/path/to/video.mp4"
    assert row["actual_fps"] == pytest.approx(120.0)
    assert row["first_video_frame"] == 0
    assert row["last_video_frame"] == 999


# ---------------------------------------------------------------------------
# DBContext.get_shot_videos
# ---------------------------------------------------------------------------


def test_get_shot_videos_returns_list(
    ctx: DBContext,
    cam_instance_id: str,
) -> None:
    shot_id = ctx.create_shot("s1", shot_number=1)
    ctx.create_shot_video(shot_id, cam_instance_id, "/v1.mp4", 120.0, 500, 1920, 1080)

    videos = ctx.get_shot_videos(shot_id)
    assert len(videos) == 1
    v = videos[0]
    assert isinstance(v, ShotVideoInfo)
    assert v.file_path == "/v1.mp4"
    assert v.actual_fps == pytest.approx(120.0)


def test_get_shot_videos_empty(ctx: DBContext) -> None:
    shot_id = ctx.create_shot("empty", shot_number=1)
    assert ctx.get_shot_videos(shot_id) == []


# ---------------------------------------------------------------------------
# DBContext.write_sync_config + get_active_sync
# ---------------------------------------------------------------------------


def test_write_and_read_sync_config(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
    cam_instance_id: str,
) -> None:
    shot_id = ctx.create_shot("s1", 1)
    vid_id = ctx.create_shot_video(shot_id, cam_instance_id, "/v.mp4", 120.0, 1000, 1920, 1080)

    pt = SyncPoint(cam_instance_id, vid_id, video_frame=50, timestamp_s=0.5)
    config_id = ctx.write_sync_config(shot_id, "manual-rough", {cam_instance_id: [pt]})

    row = session_conn.execute(
        "SELECT created_by FROM sync_configs WHERE id = ?", (config_id,)
    ).fetchone()
    assert row["created_by"] == "manual-rough"

    sync_table = ctx.get_active_sync(shot_id)
    assert sync_table is not None
    assert vid_id in sync_table.video_ids()
    assert sync_table.lookup(0.5, vid_id) == 50


def test_get_active_sync_prefers_led_auto(
    ctx: DBContext,
    cam_instance_id: str,
) -> None:
    shot_id = ctx.create_shot("s1", 1)
    vid_id = ctx.create_shot_video(shot_id, cam_instance_id, "/v.mp4", 120.0, 1000, 1920, 1080)

    pt_rough = SyncPoint(cam_instance_id, vid_id, video_frame=10, timestamp_s=0.1)
    pt_led   = SyncPoint(cam_instance_id, vid_id, video_frame=20, timestamp_s=0.1)

    ctx.write_sync_config(shot_id, "manual-rough", {cam_instance_id: [pt_rough]})
    ctx.write_sync_config(shot_id, "led-auto",     {cam_instance_id: [pt_led]})

    table = ctx.get_active_sync(shot_id)
    assert table is not None
    # led-auto has frame=20 at timestamp 0.1
    assert table.lookup(0.1, vid_id) == 20


def test_get_active_sync_no_config_returns_none(ctx: DBContext) -> None:
    shot_id = ctx.create_shot("s1", 1)
    assert ctx.get_active_sync(shot_id) is None


# ---------------------------------------------------------------------------
# DBContext.write_extrinsics
# ---------------------------------------------------------------------------


def test_write_extrinsics_inserts_rows(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
    cam_instance_id: str,
) -> None:
    shot_id = ctx.create_shot("s1", 1)
    R = np.eye(3, dtype=np.float64)
    t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    entry = ExtrinsicEntry(cam_instance_id, R, t)

    calib_id = ctx.write_extrinsics(shot_id, [entry], rms_error=1.5)

    row = session_conn.execute(
        "SELECT rms_error FROM extrinsic_calibrations WHERE id = ?", (calib_id,)
    ).fetchone()
    assert row is not None
    assert row["rms_error"] == pytest.approx(1.5)

    ee = session_conn.execute(
        "SELECT R, t FROM extrinsic_entries "
        "WHERE extrinsic_calibration_id = ? AND camera_instance_id = ?",
        (calib_id, cam_instance_id),
    ).fetchone()
    assert ee is not None
    R_read = np.frombuffer(ee["R"], dtype=np.float64).reshape(3, 3)
    t_read = np.frombuffer(ee["t"], dtype=np.float64)
    np.testing.assert_array_almost_equal(R_read, R)
    np.testing.assert_array_almost_equal(t_read, t)

    # Shot should now point to the new calibration
    shot_row = session_conn.execute(
        "SELECT extrinsic_calibration_id FROM shots WHERE id = ?", (shot_id,)
    ).fetchone()
    assert shot_row["extrinsic_calibration_id"] == calib_id


# ---------------------------------------------------------------------------
# Page transaction helpers
# ---------------------------------------------------------------------------


def test_rollback_page_undoes_writes(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
) -> None:
    ctx.begin_page()
    shot_id = ctx.create_shot("will-be-rolled-back", 99)
    ctx.rollback_page()

    row = session_conn.execute(
        "SELECT id FROM shots WHERE id = ?", (shot_id,)
    ).fetchone()
    assert row is None


def test_commit_page_preserves_writes(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
) -> None:
    ctx.begin_page()
    shot_id = ctx.create_shot("keeper", 1)
    ctx.commit_page()

    row = session_conn.execute(
        "SELECT id FROM shots WHERE id = ?", (shot_id,)
    ).fetchone()
    assert row is not None

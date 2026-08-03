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
    CaptureVideoInfo,
    DBContext,
    ExtrinsicEntry,
    ShotVideoInfo,  # backwards-compat alias
    SyncAnchorObservation,
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


def test_sync_table_two_anchors_local_fps_overrides_stored_zero() -> None:
    """Two anchors provide a local slope even when stored fps is 0."""
    pts = [
        SyncPoint("cam1", "vid1", video_frame=10, timestamp_s=1.0),
        SyncPoint("cam1", "vid1", video_frame=20, timestamp_s=2.0),
    ]
    table = SyncTable(pts, fps_by_video={"vid1": 0.0})
    # local fps = (20-10)/(2.0-1.0) = 10
    assert table.lookup(1.4, "vid1") == 14
    assert table.lookup(1.6, "vid1") == 16


def test_sync_table_single_anchor_zero_fps_snaps_to_anchor() -> None:
    """Single anchor with fps=0 snaps to the anchor frame."""
    pts = [SyncPoint("cam1", "vid1", video_frame=10, timestamp_s=1.0)]
    table = SyncTable(pts, fps_by_video={"vid1": 0.0})
    assert table.lookup(1.4, "vid1") == 10
    assert table.lookup(0.5, "vid1") == 10


# ---------------------------------------------------------------------------
# DBContext.create_shot
# ---------------------------------------------------------------------------


def test_create_shot_inserts_row(ctx: DBContext, session_conn: sqlite3.Connection) -> None:
    shot_id = ctx.create_shot("test-shot")
    row = session_conn.execute(
        "SELECT label, capture_number FROM captures WHERE id = ?", (shot_id,)
    ).fetchone()
    assert row is not None
    assert row["label"] == "test-shot"
    assert row["capture_number"] == 1


# ---------------------------------------------------------------------------
# DBContext.create_shot_video
# ---------------------------------------------------------------------------


def test_create_shot_video_inserts_row(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
    cam_instance_id: str,
) -> None:
    shot_id = ctx.create_shot("s1")
    vid_id = ctx.create_shot_video(
        shot_id, cam_instance_id, "/path/to/video.mp4",
        fps=120.0, frame_count=1000, width=1920, height=1080,
    )
    row = session_conn.execute(
        "SELECT file_path, actual_fps, first_video_frame, last_video_frame "
        "FROM capture_videos WHERE id = ?",
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
    shot_id = ctx.create_shot("s1")
    ctx.create_shot_video(shot_id, cam_instance_id, "/v1.mp4", 120.0, 500, 1920, 1080)

    videos = ctx.get_shot_videos(shot_id)
    assert len(videos) == 1
    v = videos[0]
    assert isinstance(v, CaptureVideoInfo)
    assert v.file_path == "/v1.mp4"
    assert v.actual_fps == pytest.approx(120.0)


def test_get_shot_videos_empty(ctx: DBContext) -> None:
    shot_id = ctx.create_shot("empty")
    assert ctx.get_shot_videos(shot_id) == []


# ---------------------------------------------------------------------------
# DBContext.write_sync_config + get_active_sync
# ---------------------------------------------------------------------------


def test_write_and_read_sync_config(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
    cam_instance_id: str,
) -> None:
    shot_id = ctx.create_shot("s1")
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
    shot_id = ctx.create_shot("s1")
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
    shot_id = ctx.create_shot("s1")
    assert ctx.get_active_sync(shot_id) is None


# ---------------------------------------------------------------------------
# DBContext.write_extrinsics
# ---------------------------------------------------------------------------


def test_write_extrinsics_inserts_rows(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
    cam_instance_id: str,
) -> None:
    shot_id = ctx.create_shot("s1")
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
        "SELECT extrinsic_calibration_id FROM captures WHERE id = ?", (shot_id,)
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
    shot_id = ctx.create_shot("will-be-rolled-back")
    ctx.rollback_page()

    row = session_conn.execute(
        "SELECT id FROM captures WHERE id = ?", (shot_id,)
    ).fetchone()
    assert row is None


def test_commit_page_preserves_writes(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
) -> None:
    ctx.begin_page()
    shot_id = ctx.create_shot("keeper")
    ctx.commit_page()

    row = session_conn.execute(
        "SELECT id FROM captures WHERE id = ?", (shot_id,)
    ).fetchone()
    assert row is not None


def test_commit_page_survives_external_commit_on_shared_connection(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
) -> None:
    """Dialogs opened from a wizard page (e.g. inline camera creation) share
    the session connection and call ``conn.commit()`` directly, which ends
    the whole transaction and drops our savepoint out from under us.
    ``commit_page()`` must not raise "no such savepoint" in that case.
    """
    ctx.begin_page()
    shot_id = ctx.create_shot("keeper")
    session_conn.commit()  # simulates an inline dialog's direct commit()
    ctx.commit_page()

    row = session_conn.execute(
        "SELECT id FROM captures WHERE id = ?", (shot_id,)
    ).fetchone()
    assert row is not None


def test_rollback_page_survives_external_commit_on_shared_connection(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
) -> None:
    ctx.begin_page()
    shot_id = ctx.create_shot("already-durable")
    session_conn.commit()  # simulates an inline dialog's direct commit()
    ctx.rollback_page()  # must not raise; write is already durable

    row = session_conn.execute(
        "SELECT id FROM captures WHERE id = ?", (shot_id,)
    ).fetchone()
    assert row is not None


# ---------------------------------------------------------------------------
# DBContext sync anchor CRUD
# ---------------------------------------------------------------------------


@pytest.fixture()
def shot_and_videos(
    ctx: DBContext, cam_instance_id: str, session_conn: sqlite3.Connection
) -> tuple[str, str, str]:
    """Return (shot_id, video_id_a, video_id_b) for anchor tests."""
    shot_id = ctx.create_shot("anchor-test")
    vid_a = ctx.create_shot_video(shot_id, cam_instance_id, "/a.mp4", 30.0, 900, 1920, 1080)
    # second camera instance
    session_conn.execute(
        "INSERT OR IGNORE INTO camera_instances (id, camera_model_id, label) "
        "VALUES ('cam-b', 'mdl-1', 'cam2')"
    )
    session_conn.commit()
    vid_b = ctx.create_shot_video(shot_id, "cam-b", "/b.mp4", 30.0, 900, 1920, 1080)
    return shot_id, vid_a, vid_b


def test_create_sync_anchor_inserts_row(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
    shot_and_videos: tuple,
) -> None:
    shot_id, _, _ = shot_and_videos
    anchor_id = ctx.create_sync_anchor(shot_id, notes="clap at start")
    row = session_conn.execute(
        "SELECT shot_id, notes FROM sync_anchors WHERE id = ?", (anchor_id,)
    ).fetchone()
    assert row is not None
    assert row["shot_id"] == shot_id
    assert row["notes"] == "clap at start"


def test_create_sync_anchor_no_notes(
    ctx: DBContext, session_conn: sqlite3.Connection, shot_and_videos: tuple
) -> None:
    shot_id, _, _ = shot_and_videos
    anchor_id = ctx.create_sync_anchor(shot_id)
    row = session_conn.execute(
        "SELECT notes FROM sync_anchors WHERE id = ?", (anchor_id,)
    ).fetchone()
    assert row["notes"] is None


def test_add_anchor_observation_inserts_row(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
    shot_and_videos: tuple,
) -> None:
    shot_id, vid_a, _ = shot_and_videos
    anchor_id = ctx.create_sync_anchor(shot_id)
    obs_id = ctx.add_anchor_observation(anchor_id, vid_a, video_frame=42)
    row = session_conn.execute(
        "SELECT sync_anchor_id, shot_video_id, video_frame, subframe "
        "FROM sync_anchor_observations WHERE id = ?",
        (obs_id,),
    ).fetchone()
    assert row is not None
    assert row["sync_anchor_id"] == anchor_id
    assert row["shot_video_id"] == vid_a
    assert row["video_frame"] == 42
    assert row["subframe"] == pytest.approx(0.0)


def test_anchor_observation_default_subframe(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
    shot_and_videos: tuple,
) -> None:
    shot_id, vid_a, _ = shot_and_videos
    anchor_id = ctx.create_sync_anchor(shot_id)
    obs_id = ctx.add_anchor_observation(anchor_id, vid_a, video_frame=10)
    row = session_conn.execute(
        "SELECT subframe FROM sync_anchor_observations WHERE id = ?", (obs_id,)
    ).fetchone()
    assert row["subframe"] == pytest.approx(0.0)


def test_add_anchor_observation_with_subframe(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
    shot_and_videos: tuple,
) -> None:
    shot_id, vid_a, _ = shot_and_videos
    anchor_id = ctx.create_sync_anchor(shot_id)
    obs_id = ctx.add_anchor_observation(anchor_id, vid_a, video_frame=100, subframe=0.37)
    row = session_conn.execute(
        "SELECT subframe FROM sync_anchor_observations WHERE id = ?", (obs_id,)
    ).fetchone()
    assert row["subframe"] == pytest.approx(0.37)


def test_get_anchor_observations_returns_grouped(
    ctx: DBContext,
    shot_and_videos: tuple,
) -> None:
    shot_id, vid_a, vid_b = shot_and_videos
    anchor_id = ctx.create_sync_anchor(shot_id)
    ctx.add_anchor_observation(anchor_id, vid_a, video_frame=100)
    ctx.add_anchor_observation(anchor_id, vid_b, video_frame=55)

    result = ctx.get_anchor_observations(shot_id)
    assert len(result) == 1
    aid, obs = result[0]
    assert aid == anchor_id
    assert len(obs) == 2
    assert all(isinstance(o, SyncAnchorObservation) for o in obs)
    video_ids = {o.shot_video_id for o in obs}
    assert video_ids == {vid_a, vid_b}
    frames = {o.shot_video_id: o.video_frame for o in obs}
    assert frames[vid_a] == 100
    assert frames[vid_b] == 55


def test_get_anchor_observations_multiple_anchors(
    ctx: DBContext,
    shot_and_videos: tuple,
) -> None:
    shot_id, vid_a, vid_b = shot_and_videos
    a1 = ctx.create_sync_anchor(shot_id)
    a2 = ctx.create_sync_anchor(shot_id)
    ctx.add_anchor_observation(a1, vid_a, 10)
    ctx.add_anchor_observation(a1, vid_b, 20)
    ctx.add_anchor_observation(a2, vid_a, 500)
    ctx.add_anchor_observation(a2, vid_b, 510)

    result = ctx.get_anchor_observations(shot_id)
    assert len(result) == 2
    anchor_ids = [aid for aid, _ in result]
    assert a1 in anchor_ids
    assert a2 in anchor_ids


def test_get_anchor_observations_empty(ctx: DBContext, shot_and_videos: tuple) -> None:
    shot_id, _, _ = shot_and_videos
    assert ctx.get_anchor_observations(shot_id) == []


def test_delete_sync_anchor_removes_anchor_and_observations(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
    shot_and_videos: tuple,
) -> None:
    shot_id, vid_a, vid_b = shot_and_videos
    anchor_id = ctx.create_sync_anchor(shot_id)
    ctx.add_anchor_observation(anchor_id, vid_a, 100)
    ctx.add_anchor_observation(anchor_id, vid_b, 55)

    ctx.delete_sync_anchor(anchor_id)
    session_conn.commit()

    assert session_conn.execute(
        "SELECT id FROM sync_anchors WHERE id = ?", (anchor_id,)
    ).fetchone() is None
    assert session_conn.execute(
        "SELECT id FROM sync_anchor_observations WHERE sync_anchor_id = ?", (anchor_id,)
    ).fetchall() == []


def test_update_anchor_observation_changes_frame(
    ctx: DBContext,
    session_conn: sqlite3.Connection,
    shot_and_videos: tuple,
) -> None:
    shot_id, vid_a, _ = shot_and_videos
    anchor_id = ctx.create_sync_anchor(shot_id)
    obs_id = ctx.add_anchor_observation(anchor_id, vid_a, video_frame=42)

    ctx.update_anchor_observation(obs_id, video_frame=99, subframe=0.5)

    row = session_conn.execute(
        "SELECT video_frame, subframe FROM sync_anchor_observations WHERE id = ?", (obs_id,)
    ).fetchone()
    assert row["video_frame"] == 99
    assert row["subframe"] == pytest.approx(0.5)

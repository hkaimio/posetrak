# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Camera relabelling keeps sync points with their videos."""

from __future__ import annotations

import pytest

from posetrak.db.db import (
    add_capture_video,
    add_session_camera,
    create_camera_mode,
    create_camera_model,
    create_capture,
)
from posetrak.db.import_calib_toml import import_calib_toml
from posetrak.db.reassign_camera import (
    Assignment,
    CameraReassignError,
    reassign_cameras,
    runs_referencing_capture,
)


@pytest.fixture()
def capture(session_db_full):
    """A capture with two videos (cam1, cam2) whose sync points use the same frame numbers."""
    conn, session_id, inst1, inst2 = session_db_full
    cap = create_capture(conn, session_id)
    v1 = add_capture_video(conn, cap, inst1, "a.mp4", 0, 999, 60.0)
    v2 = add_capture_video(conn, cap, inst2, "b.mp4", 0, 999, 60.0)
    conn.execute("INSERT INTO sync_configs (id, shot_id) VALUES ('sc', ?)", (cap,))
    for vid, cam, offset in ((v1, inst1, 0.0), (v2, inst2, 0.25)):
        for frame in (0, 600):
            conn.execute(
                "INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id, "
                "video_frame, timestamp_s) VALUES ('sc', ?, ?, ?, ?)",
                (cam, vid, frame, offset + frame / 60.0))
    conn.commit()
    return conn, cap, (v1, inst1), (v2, inst2)


def _state(conn):
    videos = conn.execute(
        "SELECT id, camera_instance_id, file_path FROM capture_videos ORDER BY id").fetchall()
    points = conn.execute(
        "SELECT shot_video_id, camera_instance_id, video_frame, timestamp_s FROM sync_points "
        "ORDER BY shot_video_id, video_frame").fetchall()
    return [tuple(r) for r in videos], [tuple(r) for r in points]


def test_swap_keeps_sync_points_with_their_videos(capture):
    conn, cap, (v1, inst1), (v2, inst2) = capture
    videos_before, points_before = _state(conn)

    reassign_cameras(conn, cap, [Assignment(v1, inst2), Assignment(v2, inst1)])

    videos, points = _state(conn)
    # file paths stay with their videos; the cameras are exchanged
    assert {v[0]: v[2] for v in videos} == {v[0]: v[2] for v in videos_before}
    assert {v[0]: v[1] for v in videos} == {v1: inst2, v2: inst1}
    # sync points: same video, frame and time; only the camera id follows the video
    assert [(p[0], p[2], p[3]) for p in points] == [(p[0], p[2], p[3]) for p in points_before]
    assert {p[0]: p[1] for p in points} == {v1: inst2, v2: inst1}


def test_two_videos_cannot_share_a_camera(capture):
    conn, cap, (v1, inst1), (v2, inst2) = capture
    before = _state(conn)
    with pytest.raises(CameraReassignError, match="same camera"):
        reassign_cameras(conn, cap, [Assignment(v2, inst1)])
    assert _state(conn) == before


def test_refuses_when_results_exist_unless_confirmed(capture):
    conn, cap, (v1, inst1), (v2, inst2) = capture
    conn.execute(
        "INSERT INTO pose_observation_sequences (id, shot_id, sync_config_id, time_start_s, time_end_s) "
        "VALUES ('seq', ?, 'sc', 0, 1)", (cap,))
    conn.commit()
    assert runs_referencing_capture(conn, cap)["pose observation sequences"] == 1
    swap = [Assignment(v1, inst2), Assignment(v2, inst1)]
    before = _state(conn)

    with pytest.raises(CameraReassignError, match="recomputed"):
        reassign_cameras(conn, cap, swap)
    assert _state(conn) == before

    warnings = reassign_cameras(conn, cap, swap, confirm_recompute=True)
    assert any("Recompute" in w for w in warnings)
    assert _state(conn) != before


def test_other_model_takes_its_mode_and_intrinsics(capture, registry_db, tmp_path):
    conn, cap, (v1, inst1), _ = capture
    model2 = create_camera_model(registry_db, manufacturer="Other", model_name="Cam2")
    mode2 = create_camera_mode(registry_db, model2, width_px=1920, height_px=1080, nominal_fps=60.0)
    toml = tmp_path / "c3.toml"
    toml.write_text(
        '[cam3]\nname = "Camera3"\nmatrix = [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]\n'
        "rotation = [0.0, 0.0, 0.0]\ntranslation = [0.0, 0.0, 2.0]\n", encoding="utf-8")
    res = import_calib_toml(registry_db, toml, mode2)
    inst3, intr3 = res.camera_instance_ids["Camera3"], res.intrinsics_ids["Camera3"]
    session_id = conn.execute("SELECT session_id FROM captures WHERE id = ?", (cap,)).fetchone()[0]
    add_session_camera(conn, registry_db, session_id, inst3, mode2, intr3, label="cam3")
    conn.commit()

    reassign_cameras(conn, cap, [Assignment(v1, inst3)])

    row = conn.execute(
        "SELECT camera_instance_id, camera_mode_id, intrinsics_calibration_id FROM capture_videos WHERE id = ?",
        (v1,)).fetchone()
    assert tuple(row) == (inst3, mode2, intr3)


def test_sync_points_holding_the_video_id_are_left_alone(capture):
    """The synchronisation dialog stores the video id in sync_points.camera_instance_id."""
    conn, cap, (v1, inst1), (v2, inst2) = capture
    conn.execute("UPDATE sync_points SET camera_instance_id = shot_video_id")
    conn.commit()
    before = _state(conn)[1]

    reassign_cameras(conn, cap, [Assignment(v1, inst2), Assignment(v2, inst1)])

    assert _state(conn)[1] == before

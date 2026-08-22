# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for posetrak.db.video_export — sync-mapped clip planning."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from posetrak.db import video_export as ve
from posetrak.db.db import create_capture, create_mocap_session, create_session, generate_id


@pytest.fixture()
def capture_with_sync(tmp_path: Path):
    """A session with one capture, one camera video, and 2 sync points
    (video_frame 100 @ t=10.0, video_frame 700 @ t=15.0 -> 120 fps mapping).

    Returns (conn, capture_id, sync_config_id, camera_instance_id).
    """
    conn = create_session(tmp_path / "s.db")
    session_id = create_mocap_session(conn)
    capture_id = create_capture(conn, session_id, label="cap1")
    cam_id = generate_id()
    conn.execute(
        "INSERT INTO camera_models (id, manufacturer, model_name) VALUES (?, 'Acme', 'X')",
        (generate_id(),),
    )
    model_id = conn.execute("SELECT id FROM camera_models").fetchone()[0]
    conn.execute(
        "INSERT INTO camera_instances (id, camera_model_id, label) VALUES (?, ?, 'cam1')",
        (cam_id, model_id),
    )
    video_id = generate_id()
    conn.execute(
        "INSERT INTO capture_videos "
        "(id, shot_id, camera_instance_id, file_path, first_video_frame, "
        " last_video_frame, actual_fps) "
        "VALUES (?, ?, ?, '/fake/cam1.mp4', 0, 10000, 120.0)",
        (video_id, capture_id, cam_id),
    )
    sync_id = generate_id()
    conn.execute(
        "INSERT INTO sync_configs (id, shot_id, created_by) VALUES (?, ?, 'test')",
        (sync_id, capture_id),
    )
    conn.executemany(
        "INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id, "
        "video_frame, timestamp_s) VALUES (?, ?, ?, ?, ?)",
        [
            (sync_id, cam_id, video_id, 100, 10.0),
            (sync_id, cam_id, video_id, 700, 15.0),
        ],
    )
    conn.commit()
    return conn, capture_id, sync_id, cam_id


class TestResolveSyncConfig:
    def test_auto_resolves_single_config(self, capture_with_sync) -> None:
        conn, capture_id, sync_id, _cam_id = capture_with_sync
        assert ve.resolve_sync_config(conn, capture_id) == sync_id

    def test_explicit_id_passes_through(self, capture_with_sync) -> None:
        conn, capture_id, sync_id, _cam_id = capture_with_sync
        assert ve.resolve_sync_config(conn, capture_id, sync_id) == sync_id

    def test_errors_when_none_found(self, capture_with_sync) -> None:
        conn, _capture_id, _sync_id, _cam_id = capture_with_sync
        with pytest.raises(ve.VideoExportError, match="No sync config"):
            ve.resolve_sync_config(conn, "no-such-capture")

    def test_errors_when_ambiguous(self, capture_with_sync) -> None:
        conn, capture_id, _sync_id, _cam_id = capture_with_sync
        conn.execute(
            "INSERT INTO sync_configs (id, shot_id, created_by) VALUES (?, ?, 'test2')",
            (generate_id(), capture_id),
        )
        conn.commit()
        with pytest.raises(ve.VideoExportError, match="--sync-config"):
            ve.resolve_sync_config(conn, capture_id)


class TestPlanClip:
    def test_computes_frame_range_and_container_time(self, capture_with_sync) -> None:
        conn, capture_id, sync_id, cam_id = capture_with_sync
        # mapping: frame = 100 + (700-100)/(15-10) * (t-10) = 100 + 120*(t-10)
        # at t=11.0 -> frame 220; at t=12.0 -> frame 340
        plan = ve.plan_clip(
            conn,
            capture_id=capture_id,
            sync_config_id=sync_id,
            camera_instance_id=cam_id,
            camera_label="cam1",
            master_start_s=11.0,
            master_end_s=12.0,
            probe_fps=lambda _p: 120.0,
        )
        assert plan.frame_start == pytest.approx(220.0)
        assert plan.frame_end == pytest.approx(340.0)
        assert plan.container_start_s == pytest.approx(220.0 / 120.0)
        assert plan.container_duration_s == pytest.approx((340.0 - 220.0) / 120.0)
        assert Path(plan.source_path) == Path("/fake/cam1.mp4")

    def test_uses_container_fps_not_actual_fps_for_seek_time(self, capture_with_sync) -> None:
        """A 4x slow-motion-labeled container (declared fps far below the
        true capture rate, e.g. Android's capture.fps quirk) must seek
        using its own declared fps, not capture_videos.actual_fps."""
        conn, capture_id, sync_id, cam_id = capture_with_sync
        plan = ve.plan_clip(
            conn,
            capture_id=capture_id,
            sync_config_id=sync_id,
            camera_instance_id=cam_id,
            camera_label="cam1",
            master_start_s=11.0,
            master_end_s=12.0,
            probe_fps=lambda _p: 30.0,  # declared container fps, not actual_fps=120.0
        )
        assert plan.container_start_s == pytest.approx(220.0 / 30.0)
        assert plan.container_start_s != pytest.approx(220.0 / 120.0)

    def test_monkeypatches_module_level_probe_fps(
        self, capture_with_sync, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Callers (like the CLI) that don't pass probe_fps explicitly
        should still pick up a monkeypatched video_export.probe_container_fps."""
        conn, capture_id, sync_id, cam_id = capture_with_sync
        monkeypatch.setattr(ve, "probe_container_fps", lambda _p: 120.0)
        plan = ve.plan_clip(
            conn,
            capture_id=capture_id,
            sync_config_id=sync_id,
            camera_instance_id=cam_id,
            camera_label="cam1",
            master_start_s=11.0,
            master_end_s=12.0,
        )
        assert plan.container_start_s == pytest.approx(220.0 / 120.0)

    def test_errors_on_missing_video(self, capture_with_sync) -> None:
        conn, capture_id, sync_id, _cam_id = capture_with_sync
        with pytest.raises(ve.VideoExportError, match="no video registered"):
            ve.plan_clip(
                conn, capture_id=capture_id, sync_config_id=sync_id,
                camera_instance_id="no-such-camera", camera_label="ghost",
                master_start_s=11.0, master_end_s=12.0, probe_fps=lambda _p: 120.0,
            )

    def test_errors_when_end_before_start(self, capture_with_sync) -> None:
        conn, capture_id, sync_id, cam_id = capture_with_sync
        with pytest.raises(ve.VideoExportError, match="must be after"):
            ve.plan_clip(
                conn, capture_id=capture_id, sync_config_id=sync_id,
                camera_instance_id=cam_id, camera_label="cam1",
                master_start_s=12.0, master_end_s=11.0, probe_fps=lambda _p: 120.0,
            )

    def test_errors_when_start_frame_negative(self, capture_with_sync) -> None:
        conn, capture_id, sync_id, cam_id = capture_with_sync
        # frame = 100 + 120*(t-10); frame<0 for t < 10 - 100/120 ≈ 9.167
        with pytest.raises(ve.VideoExportError, match="before the start"):
            ve.plan_clip(
                conn, capture_id=capture_id, sync_config_id=sync_id,
                camera_instance_id=cam_id, camera_label="cam1",
                master_start_s=0.0, master_end_s=5.0, probe_fps=lambda _p: 120.0,
            )

    def test_errors_on_fewer_than_two_sync_points(self, capture_with_sync) -> None:
        conn, capture_id, sync_id, cam_id = capture_with_sync
        conn.execute(
            "DELETE FROM sync_points WHERE video_frame = 700"
        )
        conn.commit()
        with pytest.raises(ve.VideoExportError, match="fewer than 2 sync points"):
            ve.plan_clip(
                conn, capture_id=capture_id, sync_config_id=sync_id,
                camera_instance_id=cam_id, camera_label="cam1",
                master_start_s=11.0, master_end_s=12.0, probe_fps=lambda _p: 120.0,
            )


class TestRunFfmpegExtract:
    def test_builds_expected_command(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(ve.subprocess, "run", fake_run)
        monkeypatch.setattr(ve.shutil, "which", lambda name: f"/usr/bin/{name}")

        plan = ve.ClipPlan(
            camera_label="cam1", camera_instance_id="c1", source_path="/fake/cam1.mp4",
            container_start_s=1.5, container_duration_s=3.25,
            frame_start=100.0, frame_end=200.0,
        )
        out = tmp_path / "cam1.mp4"
        ve.run_ffmpeg_extract(plan, out, overwrite=True)

        cmd = captured["cmd"]
        assert cmd[0] == "/usr/bin/ffmpeg"
        assert "-y" in cmd
        assert "-n" not in cmd
        assert cmd[cmd.index("-ss") + 1] == "1.500000"
        assert cmd[cmd.index("-i") + 1] == "/fake/cam1.mp4"
        assert cmd[cmd.index("-t") + 1] == "3.250000"
        assert cmd[-1] == str(out)

    def test_raises_video_export_error_on_ffmpeg_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def fake_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, stderr="boom")

        monkeypatch.setattr(ve.subprocess, "run", fake_run)
        monkeypatch.setattr(ve.shutil, "which", lambda name: f"/usr/bin/{name}")

        plan = ve.ClipPlan(
            camera_label="cam1", camera_instance_id="c1", source_path="/fake/cam1.mp4",
            container_start_s=1.5, container_duration_s=3.25,
            frame_start=100.0, frame_end=200.0,
        )
        with pytest.raises(ve.VideoExportError, match="boom"):
            ve.run_ffmpeg_extract(plan, tmp_path / "cam1.mp4")

    def test_raises_when_ffmpeg_not_on_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(ve.shutil, "which", lambda name: None)
        plan = ve.ClipPlan(
            camera_label="cam1", camera_instance_id="c1", source_path="/fake/cam1.mp4",
            container_start_s=0.0, container_duration_s=1.0,
            frame_start=0.0, frame_end=1.0,
        )
        with pytest.raises(ve.VideoExportError, match="ffmpeg not found"):
            ve.run_ffmpeg_extract(plan, tmp_path / "cam1.mp4")

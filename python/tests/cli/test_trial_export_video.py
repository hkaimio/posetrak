# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for `posetrak trial export-video` — dry-run planning only; real
ffmpeg/ffprobe invocation is covered at the unit level in
python/tests/db/test_video_export.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.db import video_export as ve
from posetrak.db.db import create_capture, create_mocap_session, create_session, generate_id


@pytest.fixture()
def session_with_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, str]:
    """Session DB with one capture, one trial, one camera with 2 sync
    points (120 fps mapping: frame = 100 + 120*(t-10)).

    Returns (db_path, trial_id, capture_id). Also monkeypatches
    probe_container_fps to 120.0 so tests never need real ffprobe.
    """
    monkeypatch.setattr(ve, "probe_container_fps", lambda _p: 120.0)

    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    session_id = create_mocap_session(conn)
    capture_id = create_capture(conn, session_id, label="cap1")

    conn.execute(
        "INSERT INTO camera_models (id, manufacturer, model_name) VALUES (?, 'Acme', 'X')",
        (generate_id(),),
    )
    model_id = conn.execute("SELECT id FROM camera_models").fetchone()[0]
    cam_id = generate_id()
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
    trial_id = generate_id()
    conn.execute(
        "INSERT INTO trials (id, capture_id, name, time_start_s, time_end_s) "
        "VALUES (?, ?, 'trial1', 11.0, 12.0)",
        (trial_id, capture_id),
    )
    conn.commit()
    conn.close()
    return db_path, trial_id, capture_id


class TestExportVideoDryRun:
    def test_dry_run_from_trial(
        self, cli_runner: CliRunner, session_with_trial: tuple[Path, str, str]
    ) -> None:
        db_path, trial_id, _capture_id = session_with_trial
        result = cli_runner.invoke(main, [
            "--session", str(db_path),
            "trial", "export-video",
            "--trial", trial_id,
            "--camera", "cam1",
            "--output-dir", "unused",
            "--dry-run",
        ])
        assert result.exit_code == 0, result.output
        assert "Master range: [11.000, 12.000]s" in result.output
        assert "cam1" in result.output
        assert "Dry run" in result.output

    def test_padding_widens_master_range(
        self, cli_runner: CliRunner, session_with_trial: tuple[Path, str, str]
    ) -> None:
        db_path, trial_id, _capture_id = session_with_trial
        result = cli_runner.invoke(main, [
            "--session", str(db_path),
            "trial", "export-video",
            "--trial", trial_id,
            "--before", "1",
            "--after", "2",
            "--camera", "cam1",
            "--output-dir", "unused",
            "--dry-run",
        ])
        assert result.exit_code == 0, result.output
        assert "Master range: [10.000, 14.000]s" in result.output

    def test_explicit_start_end_without_trial(
        self, cli_runner: CliRunner, session_with_trial: tuple[Path, str, str]
    ) -> None:
        db_path, _trial_id, capture_id = session_with_trial
        result = cli_runner.invoke(main, [
            "--session", str(db_path),
            "trial", "export-video",
            "--capture", capture_id,
            "--start", "11.0",
            "--end", "12.0",
            "--camera", "cam1",
            "--output-dir", "unused",
            "--dry-run",
        ])
        assert result.exit_code == 0, result.output
        assert "Master range: [11.000, 12.000]s" in result.output

    def test_requires_trial_or_capture_range(
        self, cli_runner: CliRunner, session_with_trial: tuple[Path, str, str]
    ) -> None:
        db_path, _trial_id, _capture_id = session_with_trial
        result = cli_runner.invoke(main, [
            "--session", str(db_path),
            "trial", "export-video",
            "--camera", "cam1",
            "--output-dir", "unused",
            "--dry-run",
        ])
        assert result.exit_code != 0

    def test_unknown_camera_label_fails_clearly(
        self, cli_runner: CliRunner, session_with_trial: tuple[Path, str, str]
    ) -> None:
        db_path, trial_id, _capture_id = session_with_trial
        result = cli_runner.invoke(main, [
            "--session", str(db_path),
            "trial", "export-video",
            "--trial", trial_id,
            "--camera", "no-such-camera",
            "--output-dir", "unused",
            "--dry-run",
        ])
        assert result.exit_code != 0
        assert "No camera instance found" in result.output

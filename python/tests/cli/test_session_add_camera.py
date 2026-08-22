# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for `posetrak session add-camera` — cloning camera registry rows
(model/mode/instance/intrinsics) from another session or registry DB into a
session, without touching captures/videos/trials.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
import struct
from pathlib import Path

import pytest
from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.db.db import (
    create_camera_instance,
    create_camera_model,
    create_camera_mode,
    create_capture,
    create_mocap_session,
    create_registry,
    create_session,
    generate_id,
)


def _add_intrinsics(conn: sqlite3.Connection, mode_id: str, *, set_as_default: bool = True) -> str:
    intr_id = generate_id()
    dist_blob = struct.pack("<4d", 0.0, 0.0, 0.0, 0.0)
    conn.execute(
        "INSERT INTO intrinsics_calibrations "
        "(id, camera_mode_id, calibrated_at, distortion_model, fx, fy, cx, cy, dist_coeffs) "
        "VALUES (?, ?, ?, 'radtan', 800.0, 800.0, 320.0, 240.0, ?)",
        (intr_id, mode_id, _dt.date.today().isoformat(), dist_blob),
    )
    if set_as_default:
        conn.execute(
            "UPDATE camera_modes SET default_intrinsics_calibration_id = ? WHERE id = ?",
            (intr_id, mode_id),
        )
    conn.commit()
    return intr_id


@pytest.fixture()
def source_session_one_calibration(tmp_path: Path) -> tuple[Path, str]:
    """A session DB with one camera used in one capture, one calibration.

    Returns (db_path, camera_label).
    """
    db_path = tmp_path / "source_one.db"
    conn = create_session(db_path)
    session_id = create_mocap_session(conn)
    model_id = create_camera_model(conn, manufacturer="Acme", model_name="Cam1")
    mode_id = create_camera_mode(conn, model_id, width_px=1920, height_px=1080)
    inst_id = create_camera_instance(conn, model_id, label="pixel9")
    intr_id = _add_intrinsics(conn, mode_id)
    capture_id = create_capture(conn, session_id, label="only-capture")
    conn.execute(
        "INSERT INTO capture_videos "
        "(id, shot_id, camera_instance_id, file_path, first_video_frame, "
        " last_video_frame, actual_fps, camera_mode_id, intrinsics_calibration_id) "
        "VALUES (?, ?, ?, '/fake/pixel9.mp4', 0, 100, 30.0, ?, ?)",
        (generate_id(), capture_id, inst_id, mode_id, intr_id),
    )
    conn.commit()
    conn.close()
    return db_path, "pixel9"


@pytest.fixture()
def source_session_two_calibrations(tmp_path: Path) -> tuple[Path, str, str, str]:
    """A session DB where one camera was recalibrated between two captures.

    Returns (db_path, camera_label, older_capture_id, newer_capture_id).
    """
    db_path = tmp_path / "source_two.db"
    conn = create_session(db_path)
    session_id = create_mocap_session(conn)
    model_id = create_camera_model(conn, manufacturer="Acme", model_name="Cam2")
    inst_id = create_camera_instance(conn, model_id, label="insta_ace2_pro")

    mode_id_old = create_camera_mode(conn, model_id, width_px=1920, height_px=1080)
    intr_id_old = _add_intrinsics(conn, mode_id_old)
    capture_old = create_capture(conn, session_id, label="older")
    conn.execute(
        "INSERT INTO capture_videos "
        "(id, shot_id, camera_instance_id, file_path, first_video_frame, "
        " last_video_frame, actual_fps, camera_mode_id, intrinsics_calibration_id) "
        "VALUES (?, ?, ?, '/fake/old.mp4', 0, 100, 30.0, ?, ?)",
        (generate_id(), capture_old, inst_id, mode_id_old, intr_id_old),
    )

    mode_id_new = create_camera_mode(conn, model_id, width_px=3840, height_px=2160)
    intr_id_new = _add_intrinsics(conn, mode_id_new)
    capture_new = create_capture(conn, session_id, label="newer")
    conn.execute(
        "INSERT INTO capture_videos "
        "(id, shot_id, camera_instance_id, file_path, first_video_frame, "
        " last_video_frame, actual_fps, camera_mode_id, intrinsics_calibration_id) "
        "VALUES (?, ?, ?, '/fake/new.mp4', 0, 100, 60.0, ?, ?)",
        (generate_id(), capture_new, inst_id, mode_id_new, intr_id_new),
    )
    conn.commit()
    conn.close()
    return db_path, "insta_ace2_pro", capture_old, capture_new


@pytest.fixture()
def source_registry_only(tmp_path: Path) -> tuple[Path, str]:
    """A bare registry (no capture_videos at all) with one calibrated camera."""
    db_path = tmp_path / "registry_only.db"
    conn = create_registry(db_path)
    model_id = create_camera_model(conn, manufacturer="Acme", model_name="Cam3")
    mode_id = create_camera_mode(conn, model_id, width_px=1920, height_px=1080)
    inst_id = create_camera_instance(conn, model_id, label="gopro-11_mini_02")
    _add_intrinsics(conn, mode_id)
    conn.close()
    return db_path, "gopro-11_mini_02"


def _target_row_counts(db_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(db_path)
    tables = ["camera_models", "camera_modes", "camera_instances",
              "intrinsics_calibrations", "session_cameras", "mocap_sessions"]
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    conn.close()
    return counts


class TestAddCameraAutoResolve:
    def test_clones_camera_into_fresh_session(
        self, cli_runner: CliRunner, tmp_path: Path,
        source_session_one_calibration: tuple[Path, str],
    ) -> None:
        src_path, label = source_session_one_calibration
        target_path = tmp_path / "target.db"

        result = cli_runner.invoke(main, [
            "--session", str(target_path),
            "session", "add-camera",
            "--from", str(src_path),
            "--camera", label,
        ])
        assert result.exit_code == 0, result.output
        assert f"Added camera {label!r}" in result.output

        counts = _target_row_counts(target_path)
        assert counts["mocap_sessions"] == 1
        assert counts["session_cameras"] == 1
        assert counts["camera_instances"] == 1
        assert counts["camera_modes"] == 1
        assert counts["intrinsics_calibrations"] == 1

        # No capture/video/trial data should have been copied.
        conn = sqlite3.connect(target_path)
        assert conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM capture_videos").fetchone()[0] == 0
        conn.close()

    def test_no_fk_violation_with_default_intrinsics_set(
        self, cli_runner: CliRunner, tmp_path: Path,
        source_session_one_calibration: tuple[Path, str],
    ) -> None:
        """Regression test for the circular FK bug: camera_modes has its
        default_intrinsics_calibration_id set before the clone (as any
        real, calibrated camera does), which used to raise a FOREIGN KEY
        constraint error regardless of copy order."""
        src_path, label = source_session_one_calibration
        target_path = tmp_path / "target.db"

        result = cli_runner.invoke(main, [
            "--session", str(target_path),
            "session", "add-camera",
            "--from", str(src_path),
            "--camera", label,
        ])
        assert result.exit_code == 0, result.output

        conn = sqlite3.connect(target_path)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        conn.close()


class TestAddCameraAmbiguity:
    def test_errors_without_capture_when_recalibrated(
        self, cli_runner: CliRunner, tmp_path: Path,
        source_session_two_calibrations: tuple[Path, str, str, str],
    ) -> None:
        src_path, label, _old, _new = source_session_two_calibrations
        target_path = tmp_path / "target.db"

        result = cli_runner.invoke(main, [
            "--session", str(target_path),
            "session", "add-camera",
            "--from", str(src_path),
            "--camera", label,
        ])
        assert result.exit_code != 0
        assert "--capture" in result.output

    def test_capture_disambiguates(
        self, cli_runner: CliRunner, tmp_path: Path,
        source_session_two_calibrations: tuple[Path, str, str, str],
    ) -> None:
        src_path, label, older, newer = source_session_two_calibrations
        target_path = tmp_path / "target.db"

        result = cli_runner.invoke(main, [
            "--session", str(target_path),
            "session", "add-camera",
            "--from", str(src_path),
            "--camera", label,
            "--capture", newer,
        ])
        assert result.exit_code == 0, result.output

        conn = sqlite3.connect(target_path)
        mode = conn.execute("SELECT width_px, height_px FROM camera_modes").fetchone()
        conn.close()
        assert mode == (3840, 2160)  # the "newer" capture's mode, not "older"'s 1920x1080


class TestAddCameraFromRegistry:
    def test_falls_back_to_mode_default(
        self, cli_runner: CliRunner, tmp_path: Path,
        source_registry_only: tuple[Path, str],
    ) -> None:
        src_path, label = source_registry_only
        target_path = tmp_path / "target.db"

        result = cli_runner.invoke(main, [
            "--session", str(target_path),
            "session", "add-camera",
            "--from", str(src_path),
            "--camera", label,
        ])
        assert result.exit_code == 0, result.output
        counts = _target_row_counts(target_path)
        assert counts["intrinsics_calibrations"] == 1


class TestAddCameraMultipleMocapSessions:
    def test_requires_explicit_session_when_ambiguous(
        self, cli_runner: CliRunner, tmp_path: Path,
        source_session_one_calibration: tuple[Path, str],
    ) -> None:
        src_path, label = source_session_one_calibration
        target_path = tmp_path / "target.db"
        conn = create_session(target_path)
        create_mocap_session(conn)
        create_mocap_session(conn)
        conn.close()

        result = cli_runner.invoke(main, [
            "--session", str(target_path),
            "session", "add-camera",
            "--from", str(src_path),
            "--camera", label,
        ])
        assert result.exit_code != 0
        assert "--session" in result.output

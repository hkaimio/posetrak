# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for session, capture, extrinsics, and sync CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.db.db import (
    create_camera_instance,
    create_camera_model,
    create_camera_mode,
    create_mocap_session,
    create_registry,
    create_session,
    open_session,
)


@pytest.fixture()
def session_db_path_new(tmp_path: Path) -> Path:
    """Return a path for a session DB that does not yet exist."""
    return tmp_path / "session_new.db"


@pytest.fixture()
def registry_with_cameras(tmp_path: Path):
    """Registry with one camera model, mode, and instance."""
    db_path = tmp_path / "registry_cameras.db"
    conn = create_registry(db_path)
    model_id = create_camera_model(conn, manufacturer="TestCo", model_name="Cam X")
    mode_id = create_camera_mode(conn, model_id, width_px=1920, height_px=1080, nominal_fps=60.0)
    instance_id = create_camera_instance(conn, model_id, label="cam1")
    conn.close()
    return db_path, model_id, mode_id, instance_id


class TestSessionCreate:
    def test_creates_session_in_new_db(
        self,
        cli_runner: CliRunner,
        session_db_path_new: Path,
    ) -> None:
        result = cli_runner.invoke(
            main,
            [
                "--session", str(session_db_path_new),
                "session", "create",
                "--date", "2024-01-15",
                "--location", "test gym",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "session_id:" in result.output
        assert session_db_path_new.exists()

    def test_creates_session_in_new_db_seeds_bundled_defaults(
        self,
        cli_runner: CliRunner,
        session_db_path_new: Path,
    ) -> None:
        """create_session() alone doesn't seed anything (also used for
        exports/round-trips that need a genuinely empty session) -- the
        CLI command, which represents a person actually starting a new
        session, seeds bundled defaults itself so the skeleton picker and
        baseline tracker config aren't empty (2026-08-23 e2e-testing
        follow-up)."""
        import sqlite3

        result = cli_runner.invoke(
            main,
            ["--session", str(session_db_path_new), "session", "create"],
        )
        assert result.exit_code == 0, result.output

        conn = sqlite3.connect(session_db_path_new)
        assert conn.execute("SELECT COUNT(*) FROM skeletons").fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM tracker_configs WHERE id = 'factory-defaults'"
        ).fetchone()[0] == 1
        conn.close()

    def test_creates_session_in_existing_db(
        self,
        cli_runner: CliRunner,
        session_db_path: Path,
    ) -> None:
        result = cli_runner.invoke(
            main,
            [
                "--session", str(session_db_path),
                "session", "create",
                "--location", "warehouse",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "session_id:" in result.output

    def test_requires_session_path(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["session", "create"])
        assert result.exit_code != 0


class TestSessionList:
    def test_empty(
        self, cli_runner: CliRunner, session_db_path: Path
    ) -> None:
        result = cli_runner.invoke(
            main, ["--session", str(session_db_path), "session", "list"]
        )
        assert result.exit_code == 0
        assert "(no sessions)" in result.output

    def test_lists_sessions(
        self, cli_runner: CliRunner, session_db_path: Path
    ) -> None:
        # Create a session directly via library.
        conn = open_session(session_db_path)
        create_mocap_session(conn, location="studio A")
        conn.close()

        result = cli_runner.invoke(
            main, ["--session", str(session_db_path), "session", "list"]
        )
        assert result.exit_code == 0
        assert "studio A" in result.output

    def test_json_mode(
        self, cli_runner: CliRunner, session_db_path: Path
    ) -> None:
        conn = open_session(session_db_path)
        create_mocap_session(conn, location="studio B")
        conn.close()

        result = cli_runner.invoke(
            main,
            ["--session", str(session_db_path), "--json", "session", "list"],
        )
        assert result.exit_code == 0
        obj = json.loads(result.output.strip())
        assert "id" in obj
        assert "location" in obj


class TestCaptureCreate:
    def _create_prerequisites(
        self, cli_runner: CliRunner, session_db_path: Path, registry_db_path: Path
    ):
        """Create a session + extrinsics placeholder so capture can be created."""
        # Create mocap session.
        r = cli_runner.invoke(
            main,
            [
                "--session", str(session_db_path),
                "session", "create",
                "--location", "test",
            ],
        )
        assert r.exit_code == 0, r.output
        session_id = r.output.split("session_id:")[-1].strip()

        # We need an extrinsic_calibrations row. Do a direct DB insert.
        from posetrak.db.db import generate_id
        conn = open_session(session_db_path)
        ext_id = generate_id()
        conn.execute(
            "INSERT INTO extrinsic_calibrations (id, session_id, calibrated_at, method) "
            "VALUES (?, ?, '2024-01-01', 'manual')",
            (ext_id, session_id),
        )
        conn.commit()
        conn.close()

        return session_id, ext_id

    def test_creates_capture(
        self,
        cli_runner: CliRunner,
        session_db_path: Path,
        registry_db_path: Path,
    ) -> None:
        session_id, ext_id = self._create_prerequisites(
            cli_runner, session_db_path, registry_db_path
        )
        result = cli_runner.invoke(
            main,
            [
                "--session", str(session_db_path),
                "capture", "create",
                "--session", session_id,
                "--extrinsics", ext_id,
                "--number", "1",
                "--label", "warmup run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "capture_id:" in result.output


class TestCaptureList:
    def test_empty(
        self, cli_runner: CliRunner, session_db_path: Path
    ) -> None:
        result = cli_runner.invoke(
            main, ["--session", str(session_db_path), "capture", "list"]
        )
        assert result.exit_code == 0
        assert "(no captures)" in result.output

    def test_requires_session_path(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(main, ["capture", "list"])
        assert result.exit_code != 0


class TestCaptureAddVideo:
    def test_add_video(
        self,
        cli_runner: CliRunner,
        session_db_path: Path,
        registry_db_path: Path,
        tmp_path: Path,
    ) -> None:
        # Create the chain: session → extrinsics → capture → camera_instance → video.
        from posetrak.db.db import generate_id, create_camera_model, create_camera_instance
        from posetrak.db.db import create_mocap_session as _create_mocap_session

        # Insert a camera instance into the session DB.
        conn = open_session(session_db_path)
        model_id = generate_id()
        conn.execute(
            "INSERT INTO camera_models (id, manufacturer, model_name) VALUES (?, 'Co', 'Cam')",
            (model_id,),
        )
        instance_id = generate_id()
        conn.execute(
            "INSERT INTO camera_instances (id, camera_model_id, label) VALUES (?, ?, 'cam1')",
            (instance_id, model_id),
        )
        session_id = _create_mocap_session(conn, location="lab")
        ext_id = generate_id()
        conn.execute(
            "INSERT INTO extrinsic_calibrations (id, session_id, calibrated_at, method) "
            "VALUES (?, ?, '2024-01-01', 'manual')",
            (ext_id, session_id),
        )
        cap_id = generate_id()
        conn.execute(
            "INSERT INTO captures (id, session_id, extrinsic_calibration_id, capture_number, label) "
            "VALUES (?, ?, ?, 1, 'run1')",
            (cap_id, session_id, ext_id),
        )
        conn.commit()
        conn.close()

        video_file = tmp_path / "vid.mp4"
        video_file.touch()

        result = cli_runner.invoke(
            main,
            [
                "--session", str(session_db_path),
                "capture", "add-video",
                "--shot", cap_id,
                "--camera-instance", instance_id,
                "--file", str(video_file),
                "--first-frame", "0",
                "--last-frame", "1000",
                "--fps", "60",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "capture_video_id:" in result.output

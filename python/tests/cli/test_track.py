"""Tests for the posetrak track CLI commands."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.db.db import generate_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(args: list[str], session_path: Path | None = None) -> "click.testing.Result":
    runner = CliRunner()
    base_args: list[str] = []
    if session_path is not None:
        base_args = ["--session", str(session_path)]
    return runner.invoke(main, base_args + args, catch_exceptions=False)


def _seed_run(session_db_path: Path) -> str:
    conn = sqlite3.connect(str(session_db_path))
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    run_id = generate_id()
    conn.execute(
        "INSERT INTO tracking_runs "
        "(id, observation_sequence_id, tracker_config_id, skeleton_id, "
        " extrinsic_calibration_id, sync_config_id, ran_at, posetrak_version, "
        " active_camera_ids, marker_names) "
        "VALUES (?,?,?,?,?,?,?,'0.1.0','[]','[]')",
        (run_id, generate_id(), generate_id(), generate_id(),
         generate_id(), generate_id(), now),
    )
    conn.commit()
    conn.close()
    return run_id


# ---------------------------------------------------------------------------
# track list
# ---------------------------------------------------------------------------


class TestTrackList:
    def test_empty(self, seeded_session_db_path: Path) -> None:
        result = _invoke(["track", "list"], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        assert result.output.strip() == ""

    def test_lists_run(self, seeded_session_db_path: Path) -> None:
        run_id = _seed_run(seeded_session_db_path)
        result = _invoke(["track", "list"], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        assert run_id[:8] in result.output

    def test_json_mode(self, seeded_session_db_path: Path) -> None:
        run_id = _seed_run(seeded_session_db_path)
        result = _invoke(["--json", "track", "list"], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        lines = [l for l in result.output.strip().splitlines() if l]
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["id"] == run_id
        assert "skeleton_id" in obj
        assert "ran_at" in obj
        assert "posetrak_version" in obj

    def test_no_session(self) -> None:
        result = _invoke(["track", "list"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# track show
# ---------------------------------------------------------------------------


class TestTrackShow:
    def test_show_by_prefix(self, seeded_session_db_path: Path) -> None:
        run_id = _seed_run(seeded_session_db_path)
        result = _invoke(["track", "show", run_id[:8]], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        assert run_id in result.output
        assert "0.1.0" in result.output

    def test_show_json(self, seeded_session_db_path: Path) -> None:
        run_id = _seed_run(seeded_session_db_path)
        result = _invoke(["--json", "track", "show", run_id], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        obj = json.loads(result.output.strip())
        assert obj["id"] == run_id

    def test_unknown_id(self, seeded_session_db_path: Path) -> None:
        result = _invoke(["track", "show", "nonexistent"], seeded_session_db_path)
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# track export
# ---------------------------------------------------------------------------


class TestTrackExport:
    @patch("posetrak.cli.track.export_bvh")
    def test_export_bvh(self, mock_export, seeded_session_db_path: Path, tmp_path: Path) -> None:
        run_id = _seed_run(seeded_session_db_path)
        output = str(tmp_path / "take.bvh")
        result = _invoke(
            ["track", "export", "bvh", run_id[:8], output],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output
        mock_export.assert_called_once()
        call_kwargs = mock_export.call_args
        assert call_kwargs.args[0] == output
        assert call_kwargs.kwargs["run_id"] == run_id
        assert call_kwargs.kwargs["session_db"] == str(seeded_session_db_path)

    @patch("posetrak.cli.track.export_gltf")
    def test_export_gltf(self, mock_export, seeded_session_db_path: Path, tmp_path: Path) -> None:
        run_id = _seed_run(seeded_session_db_path)
        output = str(tmp_path / "take.glb")
        result = _invoke(
            ["track", "export", "gltf", run_id[:8], output],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output
        mock_export.assert_called_once()
        call_kwargs = mock_export.call_args
        assert call_kwargs.args[0] == output
        assert call_kwargs.kwargs["run_id"] == run_id
        assert call_kwargs.kwargs["session_db"] == str(seeded_session_db_path)

    @patch("posetrak.cli.track.export_usd")
    def test_export_usd(self, mock_export, seeded_session_db_path: Path, tmp_path: Path) -> None:
        run_id = _seed_run(seeded_session_db_path)
        output = str(tmp_path / "take.usda")
        result = _invoke(
            ["track", "export", "usd", run_id[:8], output],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output
        mock_export.assert_called_once()
        call_kwargs = mock_export.call_args
        assert call_kwargs.args[0] == output
        assert call_kwargs.kwargs["run_id"] == run_id
        assert call_kwargs.kwargs["session_db"] == str(seeded_session_db_path)

    @patch("posetrak.cli.track.export_usd", side_effect=ImportError("usd-core not installed"))
    def test_export_usd_missing_package(
        self, mock_export, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        run_id = _seed_run(seeded_session_db_path)
        output = str(tmp_path / "take.usda")
        result = _invoke(
            ["track", "export", "usd", run_id[:8], output],
            seeded_session_db_path,
        )
        assert result.exit_code != 0

    def test_export_no_session(self, tmp_path: Path) -> None:
        result = _invoke(["track", "export", "bvh", "abc123", str(tmp_path / "out.bvh")])
        assert result.exit_code != 0

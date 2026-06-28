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


# ---------------------------------------------------------------------------
# track run
# ---------------------------------------------------------------------------


def _seed_run_prerequisites(db_path: Path) -> tuple[str, str]:
    """Seed a skeleton and a pose_observation_sequence; return (seq_id, skel_id)."""
    import datetime as dt
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    skel_id = generate_id()
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO skeletons (id, name, yaml_content, created_at)"
        " VALUES (?, 'TestSkel', '{}', ?)",
        (skel_id, now),
    )

    # Re-use the capture and sync_config seeded by the fixture
    capture_row = conn.execute("SELECT id FROM captures LIMIT 1").fetchone()
    sync_row = conn.execute("SELECT id FROM sync_configs LIMIT 1").fetchone()

    seq_id = generate_id()
    conn.execute(
        "INSERT INTO pose_observation_sequences"
        " (id, shot_id, sync_config_id, time_start_s, time_end_s)"
        " VALUES (?, ?, ?, 0.0, 10.0)",
        (seq_id, capture_row["id"], sync_row["id"]),
    )
    conn.commit()
    conn.close()
    return seq_id, skel_id


class TestTrackRun:
    @patch("posetrak.cli.track.run_tracker")
    def test_run_basic(self, mock_run, seeded_session_db_path: Path, tmp_path: Path) -> None:
        from posetrak.tracker.runner import TrackerResult
        run_id = generate_id()
        mock_run.return_value = TrackerResult(exit_code=0, run_id=run_id)

        fake_binary = tmp_path / "fake-binary"
        fake_binary.touch()

        seq_id, skel_id = _seed_run_prerequisites(seeded_session_db_path)
        result = _invoke(
            [
                "track", "run",
                "--sequence", seq_id[:8],
                "--skeleton", skel_id[:8],
                "--output-dir", str(tmp_path / "out"),
                "--binary", str(fake_binary),
            ],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        # sequence_id and skeleton_id are positional args 1 and 2
        assert call_kwargs.args[1] == seq_id
        assert call_kwargs.args[2] == skel_id

    @patch("posetrak.cli.track.run_tracker")
    @patch("posetrak.cli.track.default_binary_path")
    def test_run_creates_config_row(
        self, mock_bin, mock_run, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        from posetrak.tracker.runner import TrackerResult
        mock_bin.return_value = tmp_path / "fake"
        (tmp_path / "fake").touch()
        mock_run.return_value = TrackerResult(exit_code=0, run_id=generate_id())

        seq_id, skel_id = _seed_run_prerequisites(seeded_session_db_path)
        result = _invoke(
            [
                "track", "run",
                "--sequence", seq_id,
                "--skeleton", skel_id,
                "--output-dir", str(tmp_path / "out"),
                "--calib-noise-std", "42.0",
            ],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output

        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row
        cfg = conn.execute(
            "SELECT * FROM tracker_configs WHERE name = 'cli-run' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        assert cfg is not None
        assert cfg["measurement_noise_std"] == pytest.approx(42.0)

    @patch("posetrak.cli.track.run_tracker")
    @patch("posetrak.cli.track.default_binary_path")
    def test_run_prints_run_id(
        self, mock_bin, mock_run, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        from posetrak.tracker.runner import TrackerResult
        mock_bin.return_value = tmp_path / "fake"
        (tmp_path / "fake").touch()
        expected_run_id = generate_id()
        mock_run.return_value = TrackerResult(exit_code=0, run_id=expected_run_id)

        seq_id, skel_id = _seed_run_prerequisites(seeded_session_db_path)
        result = _invoke(
            ["track", "run", "--sequence", seq_id, "--skeleton", skel_id,
             "--output-dir", str(tmp_path / "out")],
            seeded_session_db_path,
        )
        assert result.exit_code == 0, result.output
        assert expected_run_id in result.output

    @patch("posetrak.cli.track.run_tracker")
    @patch("posetrak.cli.track.default_binary_path")
    def test_run_nonzero_exit(
        self, mock_bin, mock_run, seeded_session_db_path: Path, tmp_path: Path
    ) -> None:
        from posetrak.tracker.runner import TrackerResult
        mock_bin.return_value = tmp_path / "fake"
        (tmp_path / "fake").touch()
        mock_run.return_value = TrackerResult(exit_code=1, run_id=None)

        seq_id, skel_id = _seed_run_prerequisites(seeded_session_db_path)
        result = _invoke(
            ["track", "run", "--sequence", seq_id, "--skeleton", skel_id,
             "--output-dir", str(tmp_path / "out")],
            seeded_session_db_path,
        )
        assert result.exit_code == 1

    def test_run_no_session(self) -> None:
        result = _invoke(["track", "run", "--sequence", "abc", "--skeleton", "def"])
        assert result.exit_code != 0

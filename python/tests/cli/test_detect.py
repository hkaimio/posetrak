# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the posetrak detect CLI commands."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.detection.backends import PersonDetection, PoseResult
from posetrak.detection.pipeline import PipelineResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(args: list[str], session_path: Path | None = None) -> "click.testing.Result":
    runner = CliRunner()
    base_args: list[str] = []
    if session_path is not None:
        base_args = ["--session", str(session_path)]
    return runner.invoke(main, base_args + args, catch_exceptions=False)


# ---------------------------------------------------------------------------
# detect list
# ---------------------------------------------------------------------------


class TestDetectList:
    def test_empty_session(self, seeded_session_db_path: Path) -> None:
        """Empty session produces no output and exits 0."""
        result = _invoke(["detect", "list"], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        assert result.output.strip() == ""

    def test_empty_session_json(self, seeded_session_db_path: Path) -> None:
        """Empty session with --json flag produces no output and exits 0."""
        result = _invoke(["detect", "list", "--json"], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        assert result.output.strip() == ""

    def test_list_after_run(self, seeded_session_db_path: Path, capture_id: str, sync_id: str) -> None:
        """After a detection run, detect list shows one row."""
        # Seed a detection run directly in the DB
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        from posetrak.db.db import generate_id
        run_id = generate_id()
        conn.execute(
            "INSERT INTO detection_runs "
            "(id, shot_id, sync_config_id, time_start_s, time_end_s, "
            " detector_model, pose_model, status, created_at) "
            "VALUES (?,?,?,?,?,?,?,'complete',?)",
            (run_id, capture_id, sync_id, 0.0, 10.0, "yolo11x", "rtmpose-l-133kp", now),
        )
        conn.commit()
        conn.close()

        result = _invoke(["detect", "list"], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        assert run_id[:8] in result.output
        assert "yolo11x" in result.output

    def test_list_json(self, seeded_session_db_path: Path, capture_id: str, sync_id: str) -> None:
        """--json flag produces valid JSONL with expected keys."""
        # Seed a run
        conn = sqlite3.connect(str(seeded_session_db_path))
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        from posetrak.db.db import generate_id
        run_id = generate_id()
        conn.execute(
            "INSERT INTO detection_runs "
            "(id, shot_id, sync_config_id, time_start_s, time_end_s, "
            " detector_model, pose_model, status, created_at) "
            "VALUES (?,?,?,?,?,?,?,'complete',?)",
            (run_id, capture_id, sync_id, 0.0, 10.0, "yolo11x", "rtmpose-l-133kp", now),
        )
        conn.commit()
        conn.close()

        result = _invoke(["detect", "list", "--json"], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        lines = [l for l in result.output.strip().splitlines() if l]
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["id"] == run_id
        assert obj["detector"] == "yolo11x"
        assert "capture_id" in obj

    def test_list_filter_by_capture(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        """--capture filter includes only matching runs."""
        conn = sqlite3.connect(str(seeded_session_db_path))
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        from posetrak.db.db import generate_id

        # Run matching the capture
        run_id_match = generate_id()
        conn.execute(
            "INSERT INTO detection_runs "
            "(id, shot_id, sync_config_id, time_start_s, time_end_s, "
            " detector_model, pose_model, status, created_at) "
            "VALUES (?,?,?,?,?,?,?,'complete',?)",
            (run_id_match, capture_id, sync_id, 0.0, 5.0, "yolo11x", "rtmpose-l-133kp", now),
        )

        # Run with a different (non-existent) capture — needs a captures row too
        other_capture_id = generate_id()
        session_row = conn.execute(
            "SELECT id FROM mocap_sessions LIMIT 1"
        ).fetchone()
        session_id = session_row[0]
        conn.execute(
            "INSERT INTO captures (id, session_id, capture_number, label) "
            "VALUES (?,?,2,'other')",
            (other_capture_id, session_id),
        )
        other_sync_id = generate_id()
        conn.execute(
            "INSERT INTO sync_configs (id, shot_id, created_by) VALUES (?,?,'test')",
            (other_sync_id, other_capture_id),
        )
        run_id_other = generate_id()
        conn.execute(
            "INSERT INTO detection_runs "
            "(id, shot_id, sync_config_id, time_start_s, time_end_s, "
            " detector_model, pose_model, status, created_at) "
            "VALUES (?,?,?,?,?,?,?,'complete',?)",
            (run_id_other, other_capture_id, other_sync_id,
             0.0, 5.0, "yolo11x", "rtmpose-l-133kp", now),
        )
        conn.commit()
        conn.close()

        result = _invoke(["detect", "list", "--capture", capture_id[:8]], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        assert run_id_match[:8] in result.output
        assert run_id_other[:8] not in result.output

    def test_list_no_session(self) -> None:
        """detect list without --session exits non-zero."""
        result = _invoke(["detect", "list"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# detect run (mocked)
# ---------------------------------------------------------------------------


class TestDetectRun:
    def _make_mock_pipeline_result(self, run_id: str) -> PipelineResult:
        return PipelineResult(
            detection_run_id=run_id,
            cameras_processed=["cam1"],
            frames_processed=10,
            status="complete",
        )

    @patch("posetrak.cli.detect.DetectionPipeline")
    @patch("posetrak.cli.detect.RTMPoseEstimator")
    @patch("posetrak.cli.detect.YOLOXDetector")
    def test_run_creates_db_row(
        self,
        mock_det_cls,
        mock_rtm_cls,
        mock_pipeline_cls,
        seeded_session_db_path: Path,
        capture_id: str,
        sync_id: str,
    ) -> None:
        """detect run creates a detection_runs row and prints its ID to stdout."""
        from posetrak.db.db import generate_id as _gen_id

        expected_run_id = _gen_id()

        # Mock detector
        mock_det = MagicMock()
        mock_det.name = "yolo11x"
        mock_det.version = "8.0.0"
        mock_det._conf = 0.3
        mock_det_cls.return_value = mock_det

        # Mock estimator
        mock_est = MagicMock()
        mock_est.name = "rtmpose-l-133kp"
        mock_est.version = "0.0.15"
        mock_est.input_size = (288, 384)
        mock_rtm_cls.return_value = mock_est

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = self._make_mock_pipeline_result(expected_run_id)
        mock_pipeline_cls.return_value = mock_pipeline

        result = _invoke(
            [
                "detect", "run",
                "--capture", capture_id,
                "--sync", sync_id,
                "--start", "0",
                "--end", "10",
            ],
            seeded_session_db_path,
        )

        assert result.exit_code == 0, result.output
        # The run ID is printed as the last non-empty line of output.
        # (Progress/info lines go to stderr but CliRunner may mix them.)
        output_lines = [l for l in result.output.strip().splitlines() if l.strip()]
        assert expected_run_id in output_lines

        # Pipeline was instantiated and run() was called
        mock_pipeline_cls.assert_called_once()
        mock_pipeline.run.assert_called_once()

    @patch("posetrak.cli.detect.DetectionPipeline")
    @patch("posetrak.cli.detect.RTMPoseEstimator")
    @patch("posetrak.cli.detect.YOLOXDetector")
    def test_run_uses_correct_ids(
        self,
        mock_det_cls,
        mock_rtm_cls,
        mock_pipeline_cls,
        seeded_session_db_path: Path,
        capture_id: str,
        sync_id: str,
    ) -> None:
        """detect run passes the resolved capture and sync IDs to the pipeline."""
        from posetrak.db.db import generate_id as _gen_id

        mock_det = MagicMock()
        mock_det.name = "yolo11x"
        mock_det.version = "8.0.0"
        mock_det._conf = 0.3
        mock_det_cls.return_value = mock_det

        mock_est = MagicMock()
        mock_est.name = "rtmpose-l-133kp"
        mock_est.version = "0.0.15"
        mock_est.input_size = (288, 384)
        mock_rtm_cls.return_value = mock_est

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = PipelineResult(
            detection_run_id=_gen_id(),
            cameras_processed=[],
            frames_processed=0,
            status="complete",
        )
        mock_pipeline_cls.return_value = mock_pipeline

        _invoke(
            [
                "detect", "run",
                "--capture", capture_id[:8],  # prefix
                "--sync", sync_id[:8],        # prefix
                "--start", "5",
                "--end", "15",
            ],
            seeded_session_db_path,
        )

        call_kwargs = mock_pipeline_cls.call_args.kwargs
        assert call_kwargs["shot_id"] == capture_id
        assert call_kwargs["sync_config_id"] == sync_id
        assert call_kwargs["time_start_s"] == 5.0
        assert call_kwargs["time_end_s"] == 15.0

    def test_run_no_session(self) -> None:
        """detect run without --session exits non-zero."""
        result = _invoke(["detect", "run", "--capture", "x", "--sync", "y",
                          "--start", "0", "--end", "1"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# detect show
# ---------------------------------------------------------------------------


class TestDetectShow:
    def _seed_run(self, seeded_session_db_path: Path, capture_id: str, sync_id: str) -> str:
        conn = sqlite3.connect(str(seeded_session_db_path))
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        from posetrak.db.db import generate_id
        run_id = generate_id()
        conn.execute(
            "INSERT INTO detection_runs "
            "(id, shot_id, sync_config_id, time_start_s, time_end_s, "
            " detector_model, pose_model, status, created_at) "
            "VALUES (?,?,?,?,?,?,?,'complete',?)",
            (run_id, capture_id, sync_id, 0.0, 10.0, "yolo11x", "rtmpose-l-133kp", now),
        )
        conn.commit()
        conn.close()
        return run_id

    def test_show_by_prefix(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        """detect show with an ID prefix prints full details."""
        run_id = self._seed_run(seeded_session_db_path, capture_id, sync_id)

        result = _invoke(["detect", "show", run_id[:8]], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        assert run_id in result.output
        assert "yolo11x" in result.output

    def test_show_json(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        """detect show --json outputs parseable JSON with expected fields."""
        run_id = self._seed_run(seeded_session_db_path, capture_id, sync_id)

        result = _invoke(["detect", "show", run_id, "--json"], seeded_session_db_path)
        assert result.exit_code == 0, result.output
        obj = json.loads(result.output.strip())
        assert obj["id"] == run_id
        assert obj["detector"] == "yolo11x"
        assert obj["status"] == "complete"

    def test_show_unknown_id(self, seeded_session_db_path: Path) -> None:
        """detect show with a non-existent ID exits non-zero."""
        result = _invoke(["detect", "show", "nonexistent"], seeded_session_db_path)
        assert result.exit_code != 0

    def test_show_no_session(self) -> None:
        """detect show without --session exits non-zero."""
        result = _invoke(["detect", "show", "abc123"])
        assert result.exit_code != 0

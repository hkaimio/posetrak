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


# ---------------------------------------------------------------------------
# detect run --type aruco / dots
# ---------------------------------------------------------------------------

_ONE_MARKER_BODY_YAML = """\
name: test-bokken
units: meters
markers:
  - name: hilt
    type: aruco
    dictionary: DICT_4X4_50
    id: "3"
    size: 0.05
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
"""
_DOT_ONLY_BODY_YAML = (
    "name: dot-prop\nunits: meters\nmarkers:\n  - name: a\n    type: reflective_dot\n    center: [0,0,0]\n"
)


def _dot_frames(path, first_frame, last_frame):
    """A dark frame with one bright dot at (50, 60) on every frame."""
    import cv2

    for i in range(first_frame, last_frame):
        frame = np.full((300, 300, 3), 20, dtype=np.uint8)
        cv2.circle(frame, (50, 60), 6, (250, 250, 250), thickness=-1)
        yield i, frame


def _run_markers(session_path: Path, capture_id: str, sync_id: str, *extra: str):
    """Invoke ``detect run`` for a marker run over the seeded capture, with only the frame source faked."""
    with patch("posetrak.detection.marker_pipeline.iter_frames", _dot_frames):
        return _invoke(
            ["detect", "run", "--capture", capture_id, "--sync", sync_id, "--start", "0", "--end", "1", *extra],
            session_path,
        )


def _last_run(session_path: Path) -> sqlite3.Row:
    conn = sqlite3.connect(str(session_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM detection_runs ORDER BY created_at DESC LIMIT 1").fetchone()
    conn.close()
    return row


class TestDetectRunMarkers:
    def test_dots_run_stores_dots_and_no_coded_markers(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        from app.pose.db_cache import read_dot_candidates_for_run, read_marker_keypoints_for_run

        result = _run_markers(
            seeded_session_db_path, capture_id, sync_id, "--type", "dots", "--dots-camera", "cam1",
        )

        assert result.exit_code == 0, result.output
        run = _last_run(seeded_session_db_path)
        assert run["id"] in result.output.split()
        assert run["detector_type"] == "aruco"
        assert run["capture_object_id"] is None
        config = json.loads(run["config_json"])
        assert config["marker_ids"] == []
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.row_factory = sqlite3.Row
        video_id = conn.execute("SELECT id FROM capture_videos").fetchone()["id"]
        assert read_dot_candidates_for_run(conn, run["id"], video_id)[0].shape == (1, 9)
        assert read_marker_keypoints_for_run(conn, run["id"], video_id) == {}
        conn.close()

    def test_unbound_aruco_run_records_marker_ids_and_dictionary(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        result = _run_markers(
            seeded_session_db_path, capture_id, sync_id,
            "--type", "aruco", "--marker-ids", "3, 7", "--dictionary", "DICT_5X5_50", "--frame-step", "2",
        )

        assert result.exit_code == 0, result.output
        config = json.loads(_last_run(seeded_session_db_path)["config_json"])
        assert config["marker_ids"] == ["3", "7"]
        assert config["dictionary"] == "DICT_5X5_50"
        assert config["frame_step"] == 2
        assert "dot_detection" not in config

    def test_per_camera_dot_settings_are_recorded_by_camera_id(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        result = _run_markers(
            seeded_session_db_path, capture_id, sync_id,
            "--type", "dots", "--dots-camera", "cam1", "--dot-background-mode", "blacklist",
            "--dot-threshold", "200", "--dot-threshold-by-camera", "cam1=180",
            "--dot-max-saturation-by-camera", "cam1=120", "--dot-blacklist-frac-by-camera", "cam1=0.9",
        )

        assert result.exit_code == 0, result.output
        dots = json.loads(_last_run(seeded_session_db_path)["config_json"])["dot_detection"]
        assert dots["background_mode"] == "blacklist"
        assert dots["threshold"] == 200
        (camera_id,) = dots["cameras"]
        assert dots["threshold_by_camera"] == {camera_id: 180}
        assert dots["max_saturation_by_camera"] == {camera_id: 120.0}
        assert dots["blacklist_frac_by_camera"] == {camera_id: 0.9}

    def test_object_bound_aruco_run_with_dots(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        from posetrak.db.db import open_session
        from posetrak.db.manage_capture_object import create_capture_object
        from posetrak.db.manage_marker_body import import_marker_body_str

        session = open_session(seeded_session_db_path)
        body_id = import_marker_body_str(session, _ONE_MARKER_BODY_YAML, name="Test Bokken")
        object_id = create_capture_object(session, capture_id, "bokken-A", body_id)
        session.close()

        by_name = _run_markers(
            seeded_session_db_path, capture_id, sync_id,
            "--type", "aruco", "--object", "bokken-A", "--dots-camera", "cam1",
        )
        by_prefix = _run_markers(
            seeded_session_db_path, capture_id, sync_id, "--type", "aruco", "--object", object_id[:8],
        )

        assert by_name.exit_code == 0, by_name.output
        assert by_prefix.exit_code == 0, by_prefix.output
        conn = sqlite3.connect(str(seeded_session_db_path))
        rows = conn.execute("SELECT capture_object_id, config_json FROM detection_runs").fetchall()
        conn.close()
        assert [r[0] for r in rows] == [object_id, object_id]
        assert json.loads(rows[0][1])["marker_ids"] == ["3"]
        assert "dot_detection" in json.loads(rows[0][1])

    def test_dots_run_may_bind_to_a_dot_only_object(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        from posetrak.db.db import open_session
        from posetrak.db.manage_capture_object import create_capture_object
        from posetrak.db.manage_marker_body import import_marker_body_str

        session = open_session(seeded_session_db_path)
        body_id = import_marker_body_str(session, _DOT_ONLY_BODY_YAML, name="Dot Prop")
        object_id = create_capture_object(session, capture_id, "ball", body_id)
        session.close()

        result = _run_markers(
            seeded_session_db_path, capture_id, sync_id,
            "--type", "dots", "--object", "ball", "--dots-camera", "cam1",
        )

        assert result.exit_code == 0, result.output
        assert _last_run(seeded_session_db_path)["capture_object_id"] == object_id

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["--type", "aruco"], "--object, or --marker-ids"),
            (["--type", "dots"], "at least one --dots-camera"),
            (["--type", "dots", "--dots-camera", "cam1", "--marker-ids", "3"], "--marker-ids: only for --type aruco"),
            (["--type", "aruco", "--marker-ids", "3", "--dot-threshold", "100"], "--dot-threshold: only used with --dots-camera"),
            (["--type", "dots", "--dots-camera", "cam1", "--conf", "0.5"], "--conf: only for --type pose"),
            (["--type", "dots", "--dots-camera", "nope"], "no camera_instances with label 'nope'"),
            (["--type", "dots", "--dots-camera", "cam1", "--dot-threshold-by-camera", "cam1"], "expected LABEL=VALUE"),
            (["--type", "dots", "--dots-camera", "cam1", "--dot-threshold-by-camera", "cam1=high"], "bad value"),
            (["--type", "aruco", "--object", "nope"], "no capture object 'nope' in this capture (objects: none)"),
            (["--dots-camera", "cam1"], "--dots-camera: only for --type aruco or dots"),
            (["--object", "x", "--frame-step", "2"], "--object, --frame-step: only for --type aruco or dots"),
        ],
    )
    def test_invalid_option_combinations_are_rejected_before_any_run(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str, args: list[str], message: str
    ) -> None:
        result = _run_markers(seeded_session_db_path, capture_id, sync_id, *args)

        assert result.exit_code != 0
        assert message in result.output
        assert _last_run(seeded_session_db_path) is None

    def test_marker_ids_with_an_object_are_rejected(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        from posetrak.db.db import open_session
        from posetrak.db.manage_capture_object import create_capture_object
        from posetrak.db.manage_marker_body import import_marker_body_str

        session = open_session(seeded_session_db_path)
        body_id = import_marker_body_str(session, _ONE_MARKER_BODY_YAML, name="Test Bokken")
        create_capture_object(session, capture_id, "bokken-A", body_id)
        session.close()

        result = _run_markers(
            seeded_session_db_path, capture_id, sync_id,
            "--type", "aruco", "--object", "bokken-A", "--marker-ids", "3",
        )

        assert result.exit_code != 0
        assert "--marker-ids: taken from the marker body when --object is given" in result.output

    def test_cli_defaults_match_the_pipeline_defaults(self) -> None:
        """The dot options repeat the pipeline's defaults so --help can show them; they must not drift."""
        import inspect

        from posetrak.cli.detect import cmd_run
        from posetrak.detection.marker_pipeline import MarkerDetectionPipeline

        pipeline_defaults = {
            n: p.default for n, p in inspect.signature(MarkerDetectionPipeline.__init__).parameters.items()
        }
        for option in cmd_run.params:
            if option.name.startswith("dot_") and option.name in pipeline_defaults and not option.multiple:
                assert option.default == pipeline_defaults[option.name], option.name
        assert {o.name: o.default for o in cmd_run.params}["dictionary"] == pipeline_defaults["dictionary"]
        assert {o.name: o.default for o in cmd_run.params}["frame_step"] == pipeline_defaults["frame_step"]

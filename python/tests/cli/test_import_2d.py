# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``detect import-2d``: 2D point tracks made in another tool."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from app.pose.db_cache import read_dot_candidates_for_run
from posetrak.cli.main import main
from posetrak.db.manage_marker_body import import_marker_body_str

_DOT_BODY_YAML = (
    "name: ball\nunits: meters\nmarkers:\n  - name: a\n    type: reflective_dot\n    center: [0,0,0]\n"
)


def _invoke(args: list[str], session_path: Path):
    return CliRunner().invoke(main, ["--session", str(session_path), *args], catch_exceptions=False)


def _track(path: Path, points: list[tuple[int, float, float]], *, header: str = "scene_frame,video_frame,pixel_x,pixel_y") -> Path:
    """A track file in the format blender_export_2d_tracks.py writes."""
    lines = [header] + [f"{f + 1},{f},{x},{y}" for f, x, y in points]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _import(session_path: Path, capture_id: str, sync_id: str, *args: str):
    return _invoke(["detect", "import-2d", "--capture", capture_id, "--sync", sync_id, *args], session_path)


def _query(session_path: Path, sql: str, *params):
    conn = sqlite3.connect(str(session_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


def _candidates(session_path: Path, run_id: str):
    conn = sqlite3.connect(str(session_path))
    conn.row_factory = sqlite3.Row
    video_id = conn.execute("SELECT id FROM capture_videos").fetchone()["id"]
    try:
        return read_dot_candidates_for_run(conn, run_id, video_id)
    finally:
        conn.close()


class TestImport2D:
    def test_a_track_becomes_an_external_run_of_dot_candidates(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str, tmp_path: Path
    ) -> None:
        csv_path = _track(tmp_path / "ball.csv", [(3, 100.5, 200.25), (4, 101.0, 201.0), (9, 150.0, 260.0)])

        result = _import(seeded_session_db_path, capture_id, sync_id, "--camera", "cam1", str(csv_path),
                         "--source", "blender")

        assert result.exit_code == 0, result.output
        run_id = result.output.strip().splitlines()[-1]
        (run,) = _query(seeded_session_db_path, "SELECT * FROM detection_runs WHERE id = ?", run_id)
        assert (run["detector_type"], run["status"], run["capture_object_id"]) == ("external_2d", "complete", None)
        config = json.loads(run["config_json"])
        assert (config["layout"], config["source"]) == ("anonymous", "blender")
        assert config["files"] == [{"camera": "cam1", "file": "ball.csv"}]
        assert (run["time_start_s"], run["time_end_s"]) == pytest.approx((3 / 30.0, 9 / 30.0))

        candidates = _candidates(seeded_session_db_path, run_id)
        assert sorted(candidates) == [3, 4, 9]      # a gap is a missing frame, not an empty row
        assert candidates[3].shape == (1, 9)
        assert candidates[3][0, :2] == pytest.approx([100.5, 200.25])
        assert candidates[3][0, 8] == 0.0           # tracklet id of the file's track

    def test_several_tracks_of_a_camera_keep_separate_tracklet_ids(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str, tmp_path: Path
    ) -> None:
        a = _track(tmp_path / "a.csv", [(1, 10.0, 10.0), (2, 11.0, 11.0)])
        b = _track(tmp_path / "b.csv", [(2, 500.0, 400.0), (3, 501.0, 401.0)])

        result = _import(seeded_session_db_path, capture_id, sync_id,
                         "--camera", "cam1", str(a), "--camera", "cam1", str(b))

        assert result.exit_code == 0, result.output
        candidates = _candidates(seeded_session_db_path, result.output.strip().splitlines()[-1])
        assert sorted(candidates) == [1, 2, 3]
        assert sorted(candidates[2][:, 8]) == [0.0, 1.0]
        assert candidates[1][0, 8] == 0.0 and candidates[3][0, 8] == 1.0

    def test_a_bound_run_finalises_into_an_object_sequence_with_dots(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str, tmp_path: Path
    ) -> None:
        conn = sqlite3.connect(str(seeded_session_db_path))
        body_id = import_marker_body_str(conn, _DOT_BODY_YAML, name="Ball")
        conn.close()
        assert _invoke(["capture", "object", "add", "--capture", capture_id, "--name", "ball",
                        "--marker-body", body_id], seeded_session_db_path).exit_code == 0
        csv_path = _track(tmp_path / "ball.csv", [(1, 10.0, 10.0), (2, 11.0, 11.0)])
        imported = _import(seeded_session_db_path, capture_id, sync_id, "--object", "ball",
                           "--camera", "cam1", str(csv_path))
        run_id = imported.output.strip().splitlines()[-1]

        result = _invoke(["sequence", "finalise-object", "--detection-run", run_id], seeded_session_db_path)

        assert result.exit_code == 0, result.output
        sequence_id = result.output.strip().split(": ")[1]
        rows = _query(seeded_session_db_path,
                      "SELECT source, detection_run_id FROM pose_observations WHERE sequence_id = ?", sequence_id)
        assert [(r["source"], r["detection_run_id"]) for r in rows] == [("dots", run_id)] * 2

    def test_an_unbound_imported_run_is_not_finalised_as_an_object(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str, tmp_path: Path
    ) -> None:
        run_id = _import(seeded_session_db_path, capture_id, sync_id, "--camera", "cam1",
                         str(_track(tmp_path / "t.csv", [(1, 1.0, 1.0)]))).output.strip().splitlines()[-1]

        result = _invoke(["sequence", "finalise-object", "--detection-run", run_id], seeded_session_db_path)

        assert result.exit_code != 0 and "no capture_object_id" in result.output

    @pytest.mark.parametrize(
        ("camera", "content", "message"),
        [
            ("nope", "video_frame,pixel_x,pixel_y\n1,1,1\n", "no video of camera 'nope' in this capture (cameras: cam1)"),
            ("cam1", "frame,pixel_x,pixel_y\n1,1,1\n", "missing column(s) video_frame"),
            ("cam1", "video_frame,pixel_x,pixel_y\n1,abc,1\n", "line 2: video_frame, pixel_x and pixel_y must be numbers"),
            ("cam1", "video_frame,pixel_x,pixel_y\n1,nan,1\n", "not finite"),
            ("cam1", "video_frame,pixel_x,pixel_y\n", "none of the points falls on a frame"),
            ("cam1", "video_frame,pixel_x,pixel_y\n-5,1,1\n", "frame -5 is outside the frames of camera 'cam1' (0-1000)"),
            ("cam1", "video_frame,pixel_x,pixel_y\n1001,1,1\n", "frame 1001 is outside"),
        ],
    )
    def test_bad_input_is_rejected_and_writes_nothing(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str, tmp_path: Path,
        camera: str, content: str, message: str,
    ) -> None:
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text(content, encoding="utf-8")

        result = _import(seeded_session_db_path, capture_id, sync_id, "--camera", camera, str(csv_path))

        assert result.exit_code != 0
        assert message in result.output
        assert _query(seeded_session_db_path, "SELECT id FROM detection_runs") == []

    def test_a_missing_file_and_an_unknown_object_are_rejected(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str, tmp_path: Path
    ) -> None:
        missing = _import(seeded_session_db_path, capture_id, sync_id, "--camera", "cam1", str(tmp_path / "none.csv"))
        csv_path = _track(tmp_path / "t.csv", [(1, 1.0, 1.0)])
        unknown = _import(seeded_session_db_path, capture_id, sync_id, "--object", "nope",
                          "--camera", "cam1", str(csv_path))

        assert missing.exit_code != 0
        assert unknown.exit_code != 0 and "no capture object 'nope'" in unknown.output
        assert _query(seeded_session_db_path, "SELECT id FROM detection_runs") == []

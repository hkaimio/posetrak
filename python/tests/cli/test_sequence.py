# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``sequence`` CLI commands.

The workflow tests run the real commands end to end (object, marker run,
finalise); only the video frame source is faked.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.db.manage_marker_body import import_marker_body_str

_BODY_YAML = """\
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


def _dot_frames(path, first_frame, last_frame):
    """A dark frame with one bright dot on every frame."""
    for i in range(first_frame, last_frame):
        frame = np.full((300, 300, 3), 20, dtype=np.uint8)
        cv2.circle(frame, (50, 60), 6, (250, 250, 250), thickness=-1)
        yield i, frame


def _invoke(args: list[str], session_path: Path):
    with patch("posetrak.detection.marker_pipeline.iter_frames", _dot_frames):
        return CliRunner().invoke(main, ["--session", str(session_path), *args], catch_exceptions=False)


def _detect(session_path: Path, capture_id: str, sync_id: str, *extra: str) -> str:
    """Run `detect run` over the seeded capture and return the new run's ID."""
    result = _invoke(
        ["detect", "run", "--capture", capture_id, "--sync", sync_id, "--start", "0", "--end", "1", *extra],
        session_path,
    )
    assert result.exit_code == 0, result.output
    return result.output.strip().splitlines()[-1]


def _query(session_path: Path, sql: str, *params):
    conn = sqlite3.connect(str(session_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


def _person_sequence(session_path: Path, capture_id: str, sync_id: str) -> str:
    """A pose sequence with no dots, as `finalise_to_db` would leave for a person."""
    conn = sqlite3.connect(str(session_path))
    conn.execute(
        "INSERT INTO pose_observation_sequences (id, shot_id, sync_config_id, time_start_s, time_end_s) "
        "VALUES ('person-seq', ?, ?, 0, 1)", (capture_id, sync_id),
    )
    conn.commit()
    conn.close()
    return "person-seq"


class TestFinaliseObject:
    def test_object_workflow_from_capture_object_to_sequence(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        conn = sqlite3.connect(str(seeded_session_db_path))
        body_id = import_marker_body_str(conn, _BODY_YAML, name="Test Bokken")
        conn.close()
        add = _invoke(["capture", "object", "add", "--capture", capture_id, "--name", "bokken",
                       "--marker-body", body_id], seeded_session_db_path)
        assert add.exit_code == 0, add.output
        run_id = _detect(seeded_session_db_path, capture_id, sync_id,
                         "--type", "aruco", "--object", "bokken", "--dots-camera", "cam1")

        result = _invoke(["sequence", "finalise-object", "--detection-run", run_id[:8]], seeded_session_db_path)

        assert result.exit_code == 0, result.output
        (seq,) = _query(seeded_session_db_path, "SELECT id, detection_run_id FROM pose_observation_sequences")
        assert result.output.strip() == f"sequence_id: {seq['id']}"
        assert seq["detection_run_id"] == run_id
        sources = {r["source"]: r["n"] for r in _query(
            seeded_session_db_path,
            "SELECT source, COUNT(*) n FROM pose_observations WHERE sequence_id = ? GROUP BY source", seq["id"])}
        assert sources["dots"] > 0 and sources["markers"] > 0
        names = [r["name"] for r in _query(
            seeded_session_db_path,
            "SELECT name FROM pose_sequence_keypoints WHERE sequence_id = ? ORDER BY keypoint_idx", seq["id"])]
        assert names == ["hilt:c0", "hilt:c1", "hilt:c2", "hilt:c3"]

    def test_a_run_bound_to_no_object_is_refused(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        run_id = _detect(seeded_session_db_path, capture_id, sync_id, "--type", "dots", "--dots-camera", "cam1")

        result = _invoke(["sequence", "finalise-object", "--detection-run", run_id], seeded_session_db_path)

        assert result.exit_code != 0
        assert "no capture_object_id" in result.output
        assert _query(seeded_session_db_path, "SELECT id FROM pose_observation_sequences") == []

    def test_an_unknown_run_is_reported(self, seeded_session_db_path: Path) -> None:
        result = _invoke(["sequence", "finalise-object", "--detection-run", "nope"], seeded_session_db_path)
        assert result.exit_code != 0


class TestAddDots:
    def test_dots_are_added_to_a_person_sequence(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        seq = _person_sequence(seeded_session_db_path, capture_id, sync_id)
        run_id = _detect(seeded_session_db_path, capture_id, sync_id, "--type", "dots", "--dots-camera", "cam1")

        result = _invoke(["sequence", "add-dots", "--detection-run", run_id, "--sequence", seq],
                         seeded_session_db_path)

        assert result.exit_code == 0, result.output
        rows = _query(seeded_session_db_path,
                      "SELECT source, detection_run_id FROM pose_observations WHERE sequence_id = ?", seq)
        assert rows and {(r["source"], r["detection_run_id"]) for r in rows} == {("dots", run_id)}
        assert f"Added {len(rows)} dot rows" in result.output

    def test_dots_already_there_are_kept_unless_replaced(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        seq = _person_sequence(seeded_session_db_path, capture_id, sync_id)
        first = _detect(seeded_session_db_path, capture_id, sync_id, "--type", "dots", "--dots-camera", "cam1")
        second = _detect(seeded_session_db_path, capture_id, sync_id, "--type", "dots", "--dots-camera", "cam1")
        args = ["sequence", "add-dots", "--sequence", seq]
        assert _invoke([*args, "--detection-run", first], seeded_session_db_path).exit_code == 0

        refused = _invoke([*args, "--detection-run", second], seeded_session_db_path)
        replaced = _invoke([*args, "--detection-run", second, "--replace"], seeded_session_db_path)

        assert refused.exit_code != 0 and "already has" in refused.output
        assert replaced.exit_code == 0, replaced.output
        assert {r["detection_run_id"] for r in _query(
            seeded_session_db_path, "SELECT detection_run_id FROM pose_observations WHERE sequence_id = ?", seq
        )} == {second}

    def test_replace_is_refused_once_the_sequence_is_edited(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        seq = _person_sequence(seeded_session_db_path, capture_id, sync_id)
        run_id = _detect(seeded_session_db_path, capture_id, sync_id, "--type", "dots", "--dots-camera", "cam1")
        args = ["sequence", "add-dots", "--detection-run", run_id, "--sequence", seq]
        assert _invoke(args, seeded_session_db_path).exit_code == 0
        camera_id = _query(seeded_session_db_path, "SELECT id FROM camera_instances")[0]["id"]
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.execute(
            "INSERT INTO pose_observation_edits (id, sequence_id, camera_instance_id, video_frame, kp_blob, kp_mask)"
            " VALUES ('e1', ?, ?, 0, ?, ?)", (seq, camera_id, b"", b"\x01"),
        )
        conn.commit()
        conn.close()

        result = _invoke([*args, "--replace"], seeded_session_db_path)

        assert result.exit_code != 0
        assert "manual edits" in result.output
        assert _query(seeded_session_db_path, "SELECT COUNT(*) n FROM pose_observations WHERE source='dots'")[0]["n"] > 0

    def test_a_run_without_dots_is_refused(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        seq = _person_sequence(seeded_session_db_path, capture_id, sync_id)
        run_id = _detect(seeded_session_db_path, capture_id, sync_id, "--type", "aruco", "--marker-ids", "3")

        result = _invoke(["sequence", "add-dots", "--detection-run", run_id, "--sequence", seq],
                         seeded_session_db_path)

        assert result.exit_code != 0
        assert "no dot candidates" in result.output

    def test_a_run_of_another_sync_config_is_refused(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        seq = _person_sequence(seeded_session_db_path, capture_id, sync_id)
        run_id = _detect(seeded_session_db_path, capture_id, sync_id, "--type", "dots", "--dots-camera", "cam1")
        conn = sqlite3.connect(str(seeded_session_db_path))
        conn.execute("INSERT INTO sync_configs (id, shot_id, created_by) VALUES ('other-sync', ?, 't')", (capture_id,))
        conn.execute("UPDATE detection_runs SET sync_config_id = 'other-sync' WHERE id = ?", (run_id,))
        conn.commit()
        conn.close()

        result = _invoke(["sequence", "add-dots", "--detection-run", run_id, "--sequence", seq],
                         seeded_session_db_path)

        assert result.exit_code != 0
        assert "sync_config_id mismatch" in result.output

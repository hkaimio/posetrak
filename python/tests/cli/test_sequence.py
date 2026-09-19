# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``sequence`` CLI commands.

The workflow tests run the real commands end to end (object, marker run,
finalise); only the video frame source is faked.
"""
from __future__ import annotations

import json
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


_TIP_BODY_YAML = _BODY_YAML.replace("test-bokken", "test-tip").replace("hilt", "tip").replace('id: "3"', 'id: "7"')


def _two_objects(session_path: Path, capture_id: str) -> None:
    """Objects "hilt-prop" (marker 3) and "tip-prop" (marker 7)."""
    conn = sqlite3.connect(str(session_path))
    hilt = import_marker_body_str(conn, _BODY_YAML, name="Hilt")
    tip = import_marker_body_str(conn, _TIP_BODY_YAML, name="Tip")
    conn.close()
    for name, body in (("hilt-prop", hilt), ("tip-prop", tip)):
        result = _invoke(["capture", "object", "add", "--capture", capture_id, "--name", name,
                          "--marker-body", body], session_path)
        assert result.exit_code == 0, result.output


def _shared_run(session_path: Path, capture_id: str, sync_id: str, *extra: str) -> str:
    """One unbound run over markers 3 and 7. Marker 3 is seen on even frames, marker 7 on odd ones."""
    run_id = _detect(session_path, capture_id, sync_id, "--type", "aruco", "--marker-ids", "3,7", *extra)
    conn = sqlite3.connect(str(session_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT shot_video_id, video_frame FROM detection_keypoints "
        "WHERE detection_run_id = ? AND region_type = 'markers'", (run_id,),
    ).fetchall()
    for r in rows:
        blob = np.full((8, 3), np.nan, dtype=np.float32)
        blob[:, 2] = 0.0
        seen = slice(0, 4) if r["video_frame"] % 2 == 0 else slice(4, 8)
        blob[seen] = [[100.0 + r["video_frame"], 200.0, 1.0]] * 4
        conn.execute(
            "UPDATE detection_keypoints SET keypoints = ? WHERE detection_run_id = ? AND shot_video_id = ? "
            "AND video_frame = ? AND region_type = 'markers'",
            (blob.tobytes(), run_id, r["shot_video_id"], r["video_frame"]),
        )
    conn.commit()
    conn.close()
    return run_id


def _marker_frames(session_path: Path, run_id: str) -> dict[int, np.ndarray]:
    return {
        r["video_frame"]: np.frombuffer(r["keypoints"], dtype=np.float32).reshape(-1, 3)
        for r in _query(session_path, "SELECT video_frame, keypoints FROM detection_keypoints "
                                      "WHERE detection_run_id = ? AND region_type = 'markers'", run_id)
    }


class TestFinaliseObjectFromASharedRun:
    def test_each_prop_gets_its_own_run_and_sequence_and_the_shared_run_is_unchanged(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        _two_objects(seeded_session_db_path, capture_id)
        shared = _shared_run(seeded_session_db_path, capture_id, sync_id)
        before = {f: b.copy() for f, b in _marker_frames(seeded_session_db_path, shared).items()}

        results = {}
        for name in ("hilt-prop", "tip-prop"):
            result = _invoke(["sequence", "finalise-object", "--detection-run", shared, "--object", name],
                             seeded_session_db_path)
            assert result.exit_code == 0, result.output
            results[name] = dict(line.split(": ") for line in result.output.strip().splitlines())

        for name, marker, parity in (("hilt-prop", "3", 0), ("tip-prop", "7", 1)):
            (run,) = _query(seeded_session_db_path, "SELECT * FROM detection_runs WHERE id = ?",
                            results[name]["detection_run_id"])
            object_id = _query(seeded_session_db_path, "SELECT id FROM capture_objects WHERE name = ?", name)[0]["id"]
            config = json.loads(run["config_json"])
            assert run["capture_object_id"] == object_id
            assert config["marker_ids"] == [marker]
            assert config["derived_from_detection_run_id"] == shared

            frames = _marker_frames(seeded_session_db_path, run["id"])
            assert frames and all(f % 2 == parity for f in frames)
            for frame, blob in frames.items():
                assert blob.shape == (4, 3)
                assert np.allclose(blob, before[frame][4 * parity:4 * parity + 4])

            (seq,) = _query(seeded_session_db_path,
                            "SELECT id, detection_run_id FROM pose_observation_sequences WHERE id = ?",
                            results[name]["sequence_id"])
            assert seq["detection_run_id"] == run["id"]
            landmark = "hilt" if marker == "3" else "tip"
            assert [r["name"] for r in _query(
                seeded_session_db_path,
                "SELECT name FROM pose_sequence_keypoints WHERE sequence_id = ? ORDER BY keypoint_idx", seq["id"],
            )] == [f"{landmark}:c{i}" for i in range(4)]

        after = _marker_frames(seeded_session_db_path, shared)
        assert after.keys() == before.keys()
        assert all(np.array_equal(after[f], before[f], equal_nan=True) for f in after)

    def test_dots_are_copied_only_when_asked(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        _two_objects(seeded_session_db_path, capture_id)
        shared = _shared_run(seeded_session_db_path, capture_id, sync_id, "--dots-camera", "cam1")

        plain = _invoke(["sequence", "finalise-object", "--detection-run", shared, "--object", "hilt-prop"],
                        seeded_session_db_path)
        with_dots = _invoke(["sequence", "finalise-object", "--detection-run", shared, "--object", "tip-prop",
                             "--with-dots"], seeded_session_db_path)

        def sources(output: str) -> set[str]:
            seq = output.strip().splitlines()[-1].split(": ")[1]
            return {r["source"] for r in _query(
                seeded_session_db_path, "SELECT DISTINCT source FROM pose_observations WHERE sequence_id = ?", seq)}

        assert sources(plain.output) == {"markers"}
        assert sources(with_dots.output) == {"markers", "dots"}

    def test_an_object_none_of_whose_markers_the_run_looked_for_is_refused(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        _two_objects(seeded_session_db_path, capture_id)
        run_id = _detect(seeded_session_db_path, capture_id, sync_id, "--type", "aruco", "--marker-ids", "3")

        result = _invoke(["sequence", "finalise-object", "--detection-run", run_id, "--object", "tip-prop"],
                         seeded_session_db_path)

        assert result.exit_code != 0
        assert "none of the marker ids of object 'tip-prop'" in result.output
        assert len(_query(seeded_session_db_path, "SELECT id FROM detection_runs")) == 1

    def test_finalising_the_same_object_twice_points_at_the_existing_run(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        _two_objects(seeded_session_db_path, capture_id)
        shared = _shared_run(seeded_session_db_path, capture_id, sync_id)
        args = ["sequence", "finalise-object", "--detection-run", shared, "--object", "hilt-prop"]
        first = _invoke(args, seeded_session_db_path)
        derived = first.output.splitlines()[0].split(": ")[1]

        again = _invoke(args, seeded_session_db_path)

        assert again.exit_code != 0
        assert f"already has run {derived}" in again.output
        refinalised = _invoke(["sequence", "finalise-object", "--detection-run", derived], seeded_session_db_path)
        assert refinalised.exit_code == 0, refinalised.output

    def test_an_unknown_object_and_a_stray_with_dots_are_rejected(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        _two_objects(seeded_session_db_path, capture_id)
        shared = _shared_run(seeded_session_db_path, capture_id, sync_id)

        unknown = _invoke(["sequence", "finalise-object", "--detection-run", shared, "--object", "nope"],
                          seeded_session_db_path)
        stray = _invoke(["sequence", "finalise-object", "--detection-run", shared, "--with-dots"],
                        seeded_session_db_path)

        assert unknown.exit_code != 0 and "no capture object 'nope'" in unknown.output
        assert stray.exit_code != 0 and "--with-dots needs --object" in stray.output


def _add_second_camera(session_path: Path, capture_id: str, sync_id: str) -> None:
    conn = sqlite3.connect(str(session_path))
    model_id = conn.execute("SELECT camera_model_id FROM camera_instances").fetchone()[0]
    conn.execute("INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('cam2-id', ?, 'cam2')", (model_id,))
    conn.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path, first_video_frame, "
        "last_video_frame, actual_fps) VALUES ('sv2', ?, 'cam2-id', '/fake/video2.mp4', 0, 1000, 30.0)",
        (capture_id,),
    )
    conn.execute(
        "INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id, video_frame, timestamp_s) "
        "VALUES (?, 'cam2-id', 'sv2', 0, 0.0)", (sync_id,),
    )
    conn.commit()
    conn.close()


class TestAddDotsFromSeveralRuns:
    def test_cameras_come_from_different_runs_and_a_covered_camera_needs_replace(
        self, seeded_session_db_path: Path, capture_id: str, sync_id: str
    ) -> None:
        _add_second_camera(seeded_session_db_path, capture_id, sync_id)
        seq = _person_sequence(seeded_session_db_path, capture_id, sync_id)
        both = ("--type", "dots", "--dots-camera", "cam1", "--dots-camera", "cam2")
        run_a = _detect(seeded_session_db_path, capture_id, sync_id, *both)
        run_b = _detect(seeded_session_db_path, capture_id, sync_id, *both)
        add = ["sequence", "add-dots", "--sequence", seq]

        assert _invoke([*add, "--detection-run", run_a, "--camera", "cam1"], seeded_session_db_path).exit_code == 0
        assert _invoke([*add, "--detection-run", run_b, "--camera", "cam2"], seeded_session_db_path).exit_code == 0
        covered = _invoke([*add, "--detection-run", run_b, "--camera", "cam1"], seeded_session_db_path)
        swapped = _invoke([*add, "--detection-run", run_b, "--camera", "cam1", "--replace"], seeded_session_db_path)

        by_camera = {r["label"]: r["detection_run_id"] for r in _query(
            seeded_session_db_path,
            "SELECT DISTINCT ci.label, po.detection_run_id FROM pose_observations po "
            "JOIN camera_instances ci ON ci.id = po.camera_instance_id WHERE po.sequence_id = ?", seq)}
        assert covered.exit_code != 0 and "already has" in covered.output
        assert swapped.exit_code == 0, swapped.output
        assert by_camera == {"cam1": run_b, "cam2": run_b}

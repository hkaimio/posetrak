# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for `detect assign`: from detection keypoints to detection_track_assignments rows."""
from __future__ import annotations

from pathlib import Path

import pytest

from posetrak.detection.assign_run import AssignError, AssignResult, assign_run, format_report, write_assignment
from tests.detection.synthetic_people import build_session

STARTS = [(-2.0, -0.6), (2.0, 0.6)]
VELOCITIES = [(1.0, 0.0), (-1.0, 0.0)]


def _rows(conn, run_id):
    return conn.execute(
        "SELECT shot_video_id, track_id, person_name, first_frame, last_frame, capture_person_id, subject_id "
        "FROM detection_track_assignments WHERE detection_run_id = ? ORDER BY shot_video_id, first_frame",
        (run_id,)).fetchall()


def _person_of(rows, video, tid, frame):
    return next(r["person_name"] for r in rows if r["shot_video_id"] == video and r["track_id"] == tid
                and r["first_frame"] <= frame <= r["last_frame"])


@pytest.fixture()
def swapped(tmp_path: Path):
    """Two persons crossing, whose detector track ids swap at frame 45."""
    conn, run_id, videos = build_session(tmp_path / "s.db", STARTS, VELOCITIES, 90, swap_at=45)
    yield conn, run_id, videos
    conn.close()


def test_a_person_keeps_one_name_through_a_swap_of_detector_ids(swapped) -> None:
    conn, run_id, videos = swapped

    result = assign_run(conn, run_id, persons=2)
    write_assignment(conn, run_id, result, "refuse")

    rows = _rows(conn, run_id)
    for video in videos:
        walker_right = _person_of(rows, video, 0, 10)          # id 0 is the first person until frame 44 ...
        walker_left = _person_of(rows, video, 1, 10)
        assert walker_right != walker_left
        assert _person_of(rows, video, 1, 80) == walker_right    # ... and the second person's id 1 afterwards
        assert _person_of(rows, video, 0, 80) == walker_left
    assert {r["person_name"] for r in rows} == {"Person 1", "Person 2"}
    assert all(r["capture_person_id"] and r["capture_person_id"] == r["subject_id"] for r in rows)
    assert result.persons[0]["name"] == "Person 1"


def test_a_soma_run_is_assigned_from_its_own_trunk_keypoints(tmp_path: Path) -> None:
    conn, run_id, videos = build_session(tmp_path / "s.db", STARTS, VELOCITIES, 90, swap_at=45, soma=True)

    result = assign_run(conn, run_id, persons=2)
    write_assignment(conn, run_id, result, "refuse")

    rows = _rows(conn, run_id)
    for video in videos:
        assert _person_of(rows, video, 0, 10) == _person_of(rows, video, 1, 80)
        assert _person_of(rows, video, 0, 10) != _person_of(rows, video, 1, 10)
    conn.close()


def test_the_count_of_persons_is_found_when_not_given(swapped) -> None:
    conn, run_id, _ = swapped

    result = assign_run(conn, run_id)

    assert len(result.persons) == 2


def test_a_run_with_assignment_rows_is_left_alone_unless_told_otherwise(swapped) -> None:
    conn, run_id, videos = swapped
    conn.execute("INSERT INTO detection_track_assignments (detection_run_id, shot_video_id, track_id, person_name, "
                 "first_frame, last_frame) VALUES (?, ?, 0, 'Alice', 0, 44)", (run_id, videos[0]))

    with pytest.raises(AssignError, match="--fill-only .* --replace"):
        assign_run(conn, run_id, persons=2)


def test_fill_only_keeps_every_row_and_the_hand_made_row_names_its_person(swapped) -> None:
    conn, run_id, videos = swapped
    conn.execute("INSERT INTO detection_track_assignments (detection_run_id, shot_video_id, track_id, person_name, "
                 "first_frame, last_frame) VALUES (?, ?, 0, 'Alice', 0, 44)", (run_id, videos[0]))

    result = assign_run(conn, run_id, persons=2, existing="fill")
    write_assignment(conn, run_id, result, "fill")

    rows = _rows(conn, run_id)
    hand = [r for r in rows if r["person_name"] == "Alice" and r["shot_video_id"] == videos[0] and r["track_id"] == 0]
    assert [(r["first_frame"], r["last_frame"]) for r in hand] == [(0, 44)]
    for video in videos[1:]:
        assert _person_of(rows, video, 0, 10) == "Alice"          # the same person in the other cameras
        assert _person_of(rows, video, 1, 80) == "Alice"
    assert _person_of(rows, videos[0], 1, 80) == "Alice"          # after the swap, in the hand-made row's camera
    assert _person_of(rows, videos[0], 1, 10) == "Person 1"
    # nothing of the hand-made row is written twice: the person has one row for its first stretch in that camera
    assert sum(1 for r in rows if r["shot_video_id"] == videos[0] and r["first_frame"] <= 10 <= r["last_frame"]
               and r["person_name"] == "Alice") == 1


def test_replace_rewrites_the_rows_and_keeps_the_seeded_name(swapped) -> None:
    conn, run_id, videos = swapped
    conn.execute("INSERT INTO detection_track_assignments (detection_run_id, shot_video_id, track_id, person_name, "
                 "first_frame, last_frame) VALUES (?, ?, 0, 'Alice', 0, 10)", (run_id, videos[0]))

    result = assign_run(conn, run_id, persons=2, existing="replace")
    write_assignment(conn, run_id, result, "replace")

    rows = _rows(conn, run_id)
    assert not any(r["first_frame"] == 0 and r["last_frame"] == 10 for r in rows)
    assert _person_of(rows, videos[0], 0, 5) == "Alice" and {r["person_name"] for r in rows} == {"Alice", "Person 1"}


def test_subjects_name_the_persons_in_order_of_appearance(swapped) -> None:
    conn, run_id, _ = swapped

    result = assign_run(conn, run_id, subjects=["Alice", "Bob"])

    assert sorted(p["name"] for p in result.persons) == ["Alice", "Bob"]


def test_a_single_person_in_a_capture_with_one_person_takes_that_persons_name(tmp_path: Path) -> None:
    conn, run_id, _ = build_session(tmp_path / "s.db", [(-1.0, 0.0)], [(1.0, 0.0)], 60, existing_person="Harri")

    result = assign_run(conn, run_id)
    write_assignment(conn, run_id, result, "refuse")

    assert {r["person_name"] for r in _rows(conn, run_id)} == {"Harri"}
    assert conn.execute("SELECT COUNT(*) FROM capture_persons").fetchone()[0] == 1
    conn.close()


def test_the_report_lists_the_cameras_and_flags_a_hand_over(tmp_path: Path) -> None:
    conn, run_id, videos = build_session(tmp_path / "s.db", STARTS, [(1.0, 0.0), (-1.0, 0.0)], 90, swap_at=45)

    result = assign_run(conn, run_id, persons=2)

    assert set(result.cameras) == set(videos)
    assert all(c["clustered_fraction"] > 0.9 for c in result.cameras.values())
    assert result.report["run"] == run_id
    conn.close()


def test_a_pose_model_without_trunk_keypoints_is_refused(swapped) -> None:
    conn, run_id, _ = swapped
    conn.execute("UPDATE detection_runs SET pose_model = 'unknown-model' WHERE id = ?", (run_id,))

    with pytest.raises(AssignError, match="no registered trunk keypoints"):
        assign_run(conn, run_id)


def test_the_report_marks_a_camera_that_agrees_much_less_than_the_others() -> None:
    cameras = {name: {"boxes": 100, "clustered_fraction": fraction, "median_rel_reproj_err": 0.015}
               for name, fraction in (("a", 0.95), ("b", 0.9), ("c", 0.3))}
    result = AssignResult([], [], [], [], cameras, [], {})

    lines = format_report(result, {}).splitlines()

    assert [("calibration or sync?" in line) for line in lines if line.startswith("  ") and "boxes" in line] == [
        False, False, True]

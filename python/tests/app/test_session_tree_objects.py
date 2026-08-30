# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for SessionTreeWidget's object-sequence listing (marker-based-
mocap design doc §7.1 sub-phase 1e): a finalised object sequence must show
up under its own OBJECT_TRACK branch, labeled by the capture_objects row's
name, and must never appear (or be mistaken for a person) under the
PERSON_TRACK branch built by _add_person_tracks.
"""
from __future__ import annotations

import sqlite3

import pytest

from posetrak.db.db import create_session, generate_id
from app.pose.db_cache import create_marker_detection_run, MarkerKeypointWriter
from app.pose.finalise import finalise_object_to_db, finalise_to_db, TrackAssignment
from posetrak.db.manage_capture_object import create_capture_object
from posetrak.db.manage_marker_body import import_marker_body_str

_SHOT_ID = "cap1"
_SYNC_ID = "sync1"
_SVID = "sv1"
_CAM_ID = "cam1"

_MARKER_BODY_YAML = """\
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


@pytest.fixture()
def session_db(tmp_path):
    conn = create_session(tmp_path / "tree_objects_test.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF")
    session_id = generate_id()
    conn.executescript(f"""
        INSERT INTO mocap_sessions (id, recorded_at) VALUES ('{session_id}', '2026-01-01');
        INSERT INTO captures (id, session_id, capture_number, label)
            VALUES ('{_SHOT_ID}', '{session_id}', 1, 'test');
        INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('{_CAM_ID}', 'cm1', 'A');
        INSERT INTO sync_configs (id, shot_id, created_by) VALUES ('{_SYNC_ID}', '{_SHOT_ID}', 't');
        INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,
                                 first_video_frame, last_video_frame, actual_fps)
            VALUES ('{_SVID}', '{_SHOT_ID}', '{_CAM_ID}', '/dev/null', 0, 1000, 30.0);
        INSERT INTO sync_points (sync_config_id, camera_instance_id, shot_video_id,
                                 video_frame, timestamp_s)
            VALUES ('{_SYNC_ID}', '{_CAM_ID}', '{_SVID}', 0, 0.0);
        INSERT INTO trials (id, capture_id, name, time_start_s, time_end_s)
            VALUES ('trial1', '{_SHOT_ID}', 'Trial 1', 0.0, 1.0);
    """)
    conn.commit()
    yield conn
    conn.close()


def _make_object_sequence(conn) -> str:
    body_id = import_marker_body_str(conn, _MARKER_BODY_YAML, name="Test Bokken")
    object_id = create_capture_object(conn, _SHOT_ID, "bokken-A", body_id)
    run_id = create_marker_detection_run(
        conn, shot_id=_SHOT_ID, sync_config_id=_SYNC_ID, time_start_s=0.0, time_end_s=1.0,
        dictionary="DICT_4X4_50", marker_ids=["3"],
        marker_body_definition_id=body_id, capture_object_id=object_id,
    )
    writer = MarkerKeypointWriter(conn, run_id, _SVID, marker_ids=["3"])
    writer.add_frame(0, [])
    writer.finalise()
    conn.commit()
    conn.execute("UPDATE detection_runs SET trial_id=? WHERE id=?", ("trial1", run_id))
    conn.commit()
    seq_id = finalise_object_to_db(conn, run_id)
    return run_id, seq_id


def _tree_for(conn):
    from app.ui.session_tree import SessionTreeWidget
    tree = SessionTreeWidget()
    tree.load(conn)
    return tree


def _find_child(item, kind):
    from app.ui import session_tree as st_mod
    return [
        item.child(i) for i in range(item.childCount())
        if item.child(i).data(0, st_mod._KIND) == kind
    ]


def _find_trial_item(tree):
    from app.ui import session_tree as st_mod
    from app.ui.session_tree import ItemKind
    for item in _iter_all(tree):
        if item.data(0, st_mod._KIND) == ItemKind.TRIAL:
            return item
    raise AssertionError("no TRIAL item found in tree")


def _iter_all(tree):
    def walk(item):
        yield item
        for i in range(item.childCount()):
            yield from walk(item.child(i))
    yield from walk(tree.invisibleRootItem())


def test_object_sequence_appears_as_object_track(qapp, session_db):
    from app.ui.session_tree import ItemKind
    from app.ui import session_tree as st_mod

    _run_id, seq_id = _make_object_sequence(session_db)
    tree = _tree_for(session_db)

    trial_item = _find_trial_item(tree)
    det_items = _find_child(trial_item, ItemKind.DETECTION_RUN)
    assert len(det_items) == 1
    obj_items = _find_child(det_items[0], ItemKind.OBJECT_TRACK)
    assert len(obj_items) == 1
    assert obj_items[0].text(0) == "bokken-A"
    assert obj_items[0].data(0, st_mod._ID) == seq_id

    # Never also listed as a person track.
    person_items = _find_child(det_items[0], ItemKind.PERSON_TRACK)
    assert person_items == []


def test_person_sequence_still_appears_as_person_track(qapp, session_db):
    """Sanity check the split doesn't break the ordinary person case --
    _add_person_tracks' new JOIN must not exclude a normal pose run just
    because it has no capture_object_id (NULL should still match)."""
    from app.pose.db_cache import create_detection_run
    from app.ui.session_tree import ItemKind

    run_id = create_detection_run(
        session_db, shot_id=_SHOT_ID, sync_config_id=_SYNC_ID, time_start_s=0.0, time_end_s=1.0,
        detector_model="yolox-x", pose_model="rtmpose-l-133kp",
    )
    session_db.execute(
        "INSERT INTO detection_keypoints"
        " (detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints)"
        " VALUES (?, ?, 0, 1, 'full_body', ?)",
        (run_id, _SVID, b"\x00" * (133 * 12)),
    )
    session_db.commit()
    assignment = TrackAssignment(
        shot_video_id=_SVID, track_id=1, person_name="Alice", first_frame=0, last_frame=0,
    )
    finalise_to_db(
        session_db, detection_run_id=run_id, shot_id=_SHOT_ID, sync_config_id=_SYNC_ID,
        assignments=[assignment], pose_model="rtmpose-l-133kp",
    )
    session_db.execute("UPDATE detection_runs SET trial_id=? WHERE id=?", ("trial1", run_id))
    session_db.commit()

    tree = _tree_for(session_db)
    trial_item = _find_trial_item(tree)
    det_items = [
        d for d in _find_child(trial_item, ItemKind.DETECTION_RUN)
    ]
    assert len(det_items) == 1
    person_items = _find_child(det_items[0], ItemKind.PERSON_TRACK)
    assert len(person_items) == 1
    assert person_items[0].text(0) == "Alice"
    assert _find_child(det_items[0], ItemKind.OBJECT_TRACK) == []

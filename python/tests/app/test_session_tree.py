# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for SessionTreeWidget's segmentation-run listing (segmentation-ui-
improvements design doc, Issue 1) -- the panel had zero prior test
coverage, so these exercise both the new segmentation code and enough of
the existing capture/trial population it sits alongside to catch
regressions.
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def session_db(tmp_path):
    """Session DB with one capture, two trials, and (deliberately) no
    detection runs -- the fixture only needs enough to exercise
    segmentation-run listing/rename/delete."""
    from posetrak.db.db import create_session

    conn = create_session(tmp_path / "tree_test.db")
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('cap1', 'sess1', 1)"
    )
    conn.execute(
        "INSERT INTO trials (id, capture_id, name, time_start_s, time_end_s) "
        "VALUES ('trial1', 'cap1', 'Warmup', 0.0, 20.0)"
    )
    conn.execute(
        "INSERT INTO trials (id, capture_id, name, time_start_s, time_end_s) "
        "VALUES ('trial2', 'cap1', 'Heitot', 25.0, 50.0)"
    )
    conn.commit()
    yield conn
    conn.close()


def _tree_for(conn):
    from app.ui.session_tree import SessionTreeWidget

    tree = SessionTreeWidget()
    tree.load(conn)
    return tree


def test_segmentation_run_lists_at_capture_level_not_under_a_trial(qapp, session_db):
    """A segmentation whose recorded range spans both trials must appear
    as a direct child of the capture item, never nested under either
    trial -- the core Issue-1 tree-placement decision."""
    from app.ui.session_tree import ItemKind

    session_db.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at) "
        "VALUES ('seg1', 'cap1', 0.0, 50.0, '2026-01-01T00:00:00Z')"
    )
    session_db.commit()

    tree = _tree_for(session_db)
    cap_item = tree.topLevelItem(0)
    from app.ui import session_tree as st_mod
    direct_seg_children = [
        cap_item.child(i) for i in range(cap_item.childCount())
        if cap_item.child(i).data(0, st_mod._KIND) == ItemKind.SEGMENTATION_RUN
    ]
    assert len(direct_seg_children) == 1
    assert direct_seg_children[0].data(0, st_mod._ID) == "seg1"

    # And not nested under either trial item.
    for i in range(cap_item.childCount()):
        child = cap_item.child(i)
        if child.data(0, st_mod._KIND) == ItemKind.TRIAL:
            for j in range(child.childCount()):
                assert child.child(j).data(0, st_mod._KIND) != ItemKind.SEGMENTATION_RUN


def test_segmentation_run_label_includes_name_range_and_mask_count(qapp, session_db):
    import cv2
    import numpy as np
    from app.ui import session_tree as st_mod
    from app.ui.session_tree import ItemKind

    session_db.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, name, time_start_s, time_end_s, created_at) "
        "VALUES ('seg1', 'cap1', 'Main pass', 6.5, 13.9, '2026-08-23T14:15:05Z')"
    )
    session_db.execute(
        "INSERT INTO camera_models (id, manufacturer, model_name) VALUES ('cm1', 'x', 'y')"
    )
    session_db.execute(
        "INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('ci1', 'cm1', 'cam_A')"
    )
    session_db.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,"
        " first_video_frame, last_video_frame, actual_fps)"
        " VALUES ('sv1', 'cap1', 'ci1', '/dev/null', 0, 100, 30.0)"
    )
    ok, buf = cv2.imencode(".png", np.ones((4, 4), dtype=np.uint8))
    assert ok
    for frame in (0, 1, 2):
        session_db.execute(
            "INSERT INTO seg_masks (seg_quality_run_id, shot_video_id, frame_idx, mask_blob) "
            "VALUES ('seg1', 'sv1', ?, ?)",
            (frame, buf.tobytes()),
        )
    session_db.commit()

    tree = _tree_for(session_db)
    cap_item = tree.topLevelItem(0)
    seg_item = next(
        cap_item.child(i) for i in range(cap_item.childCount())
        if cap_item.child(i).data(0, st_mod._KIND) == ItemKind.SEGMENTATION_RUN
    )
    label = seg_item.text(0)
    assert "Main pass" in label
    assert "6.5s" in label and "13.9s" in label
    assert "3 masks" in label


def test_segmentation_run_falls_back_to_generated_label_without_name(qapp, session_db):
    from app.ui import session_tree as st_mod
    from app.ui.session_tree import ItemKind

    session_db.execute(
        "INSERT INTO seg_quality_runs "
        "(id, shot_id, quality_source, time_start_s, time_end_s, created_at) "
        "VALUES ('seg1', 'cap1', 'cutie-interactive', 0.0, 10.0, '2026-01-01T00:00:00Z')"
    )
    session_db.commit()

    tree = _tree_for(session_db)
    cap_item = tree.topLevelItem(0)
    seg_item = next(
        cap_item.child(i) for i in range(cap_item.childCount())
        if cap_item.child(i).data(0, st_mod._KIND) == ItemKind.SEGMENTATION_RUN
    )
    assert "Segmentation" in seg_item.text(0)
    assert "cutie-interactive" in seg_item.text(0)


def test_overlapping_trial_names_computed_not_stored():
    from app.ui.session_tree import _overlapping_trial_names

    sq = {"time_start_s": 10.0, "time_end_s": 30.0}
    trials = [
        {"name": "Warmup", "time_start_s": 0.0, "time_end_s": 20.0},   # overlaps
        {"name": "Heitot", "time_start_s": 25.0, "time_end_s": 50.0},  # overlaps
        {"name": "Cooldown", "time_start_s": 60.0, "time_end_s": 70.0},  # no overlap
    ]
    assert _overlapping_trial_names(sq, trials) == ["Warmup", "Heitot"]


def test_rename_segmentation_run_updates_name(qapp, session_db, monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    from app.ui.session_tree import SessionTreeWidget

    session_db.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at) "
        "VALUES ('seg1', 'cap1', 0.0, 10.0, '2026-01-01T00:00:00Z')"
    )
    session_db.commit()

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Renamed", True)))

    tree = SessionTreeWidget()
    tree.load(session_db)
    tree._rename_segmentation_run("seg1")

    row = session_db.execute(
        "SELECT name FROM seg_quality_runs WHERE id='seg1'"
    ).fetchone()
    assert row["name"] == "Renamed"


def test_delete_segmentation_run_removes_masks_first(qapp, session_db, monkeypatch):
    """Real FK enforcement (seg_masks.seg_quality_run_id REFERENCES
    seg_quality_runs(id)) means deleting the parent row before its masks
    would raise IntegrityError -- this exercises the actual delete order,
    not just that both tables end up empty."""
    import cv2
    import numpy as np
    from PySide6.QtWidgets import QMessageBox
    from app.ui.session_tree import SessionTreeWidget

    assert session_db.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    session_db.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at) "
        "VALUES ('seg1', 'cap1', 0.0, 10.0, '2026-01-01T00:00:00Z')"
    )
    session_db.execute(
        "INSERT INTO camera_models (id, manufacturer, model_name) VALUES ('cm1', 'x', 'y')"
    )
    session_db.execute(
        "INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('ci1', 'cm1', 'cam_A')"
    )
    session_db.execute(
        "INSERT INTO capture_videos (id, shot_id, camera_instance_id, file_path,"
        " first_video_frame, last_video_frame, actual_fps)"
        " VALUES ('sv1', 'cap1', 'ci1', '/dev/null', 0, 100, 30.0)"
    )
    ok, buf = cv2.imencode(".png", np.ones((4, 4), dtype=np.uint8))
    assert ok
    session_db.execute(
        "INSERT INTO seg_masks (seg_quality_run_id, shot_video_id, frame_idx, mask_blob) "
        "VALUES ('seg1', 'sv1', 0, ?)",
        (buf.tobytes(),),
    )
    session_db.commit()

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )

    tree = SessionTreeWidget()
    tree.load(session_db)
    tree._delete_segmentation_run("seg1")

    assert session_db.execute(
        "SELECT COUNT(*) FROM seg_quality_runs WHERE id='seg1'"
    ).fetchone()[0] == 0
    assert session_db.execute(
        "SELECT COUNT(*) FROM seg_masks WHERE seg_quality_run_id='seg1'"
    ).fetchone()[0] == 0

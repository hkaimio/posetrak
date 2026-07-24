"""Tests for StitcherPanel._populate_known_persons() (config-improvements
design doc, "Person model", phase 5 gap): the main-viewer app's embedded
stitching/assignment panel (app/pose/stitcher_panel.py) has its own,
separate person combo from PoseExtractionWindow's (app/pose/main.py) --
missed in the original phase 5 pass, so a capture's already-defined
capture_persons never showed up here even though the standalone pose
extraction window's combo was fixed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from posetrak.db.db import create_session
from posetrak.db.manage_person import create_person


@pytest.fixture()
def session_db(tmp_path: Path):
    conn = create_session(tmp_path / "session.db")
    yield conn
    conn.close()


def _seed_detection_run(conn: sqlite3.Connection, capture_id: str = "cap1") -> str:
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES (?, 'sess1', 1)",
        (capture_id,),
    )
    conn.execute(
        "INSERT INTO sync_configs (id, shot_id) VALUES ('sync1', ?)", (capture_id,)
    )
    run_id = "run1"
    conn.execute(
        "INSERT INTO detection_runs "
        "(id, shot_id, sync_config_id, time_start_s, time_end_s, "
        " detector_model, pose_model, created_at) "
        "VALUES (?, ?, 'sync1', 0.0, 1.0, 'yolo', 'rtmpose', '2026-01-01')",
        (run_id, capture_id),
    )
    conn.commit()
    return run_id


def test_populate_known_persons_offers_existing_capture_persons(qapp, session_db) -> None:
    from app.pose.stitcher_panel import StitcherPanel

    run_id = _seed_detection_run(session_db)
    create_person(session_db, "cap1", "Alice")
    create_person(session_db, "cap1", "Bob")

    panel = StitcherPanel(session_db, run_id)
    names = [panel._person_combo.itemText(i) for i in range(panel._person_combo.count())]
    assert set(names) == {"Alice", "Bob"}


def test_populate_known_persons_empty_capture_leaves_combo_empty(qapp, session_db) -> None:
    from app.pose.stitcher_panel import StitcherPanel

    run_id = _seed_detection_run(session_db)

    panel = StitcherPanel(session_db, run_id)
    assert panel._person_combo.count() == 0


def test_populate_known_persons_syncs_to_stitcher(qapp, session_db) -> None:
    """set_known_persons() on the underlying FilmstripStitcherWidget should
    reflect the capture's persons too, not just the combo -- that's what
    actually drives the assignment context menu."""
    from app.pose.stitcher_panel import StitcherPanel

    run_id = _seed_detection_run(session_db)
    create_person(session_db, "cap1", "Alice")

    panel = StitcherPanel(session_db, run_id)
    assert "Alice" in panel._stitcher._persons

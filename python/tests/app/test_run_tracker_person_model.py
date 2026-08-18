# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for phase 5 of the config-improvements design doc: RunTrackerWidget's
people table switching its data source to a trial's capture's
capture_persons when any are defined, falling back to the original
free-text-name discovery when none are (existing captures that haven't
adopted the person model yet).

Pure widget-construction and DB-state assertions -- no display needed (qapp
fixture forces the offscreen Qt platform, see conftest.py).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from PySide6.QtWidgets import QCheckBox, QComboBox

from posetrak.db.db import create_session
from posetrak.db.manage_person import create_person


@pytest.fixture()
def session_db(tmp_path: Path):
    conn = create_session(tmp_path / "session.db")
    yield conn
    conn.close()


def _make_trial(conn: sqlite3.Connection, capture_id: str = "cap1", trial_id: str = "trial1") -> None:
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES (?, 'sess1', 1)",
        (capture_id,),
    )
    conn.execute(
        "INSERT INTO trials (id, capture_id, name, time_start_s, time_end_s) "
        "VALUES (?, ?, 'take 1', 0.0, 1.0)",
        (trial_id, capture_id),
    )
    conn.commit()


def _make_detection_run_with_sequence(
    conn: sqlite3.Connection,
    *,
    detection_run_id: str,
    seq_id: str,
    capture_id: str,
    trial_id: str,
    person_name: str,
    capture_person_id: str | None,
) -> None:
    """Add a detection run + sequence + sequence_persons row a person's
    row can be discovered from, in either mode."""
    conn.execute(
        "INSERT OR IGNORE INTO sync_configs (id, shot_id) VALUES ('sync1', ?)", (capture_id,)
    )
    conn.execute(
        "INSERT INTO detection_runs "
        "(id, shot_id, sync_config_id, trial_id, time_start_s, time_end_s, "
        " detector_model, pose_model, created_at) "
        "VALUES (?, ?, 'sync1', ?, 0.0, 1.0, 'yolo', 'rtmpose', '2026-01-01')",
        (detection_run_id, capture_id, trial_id),
    )
    conn.execute(
        "INSERT INTO pose_observation_sequences "
        "(id, shot_id, sync_config_id, time_start_s, time_end_s, detection_run_id) "
        "VALUES (?, ?, 'sync1', 0.0, 1.0, ?)",
        (seq_id, capture_id, detection_run_id),
    )
    conn.execute(
        "INSERT INTO sequence_persons (sequence_id, person_id, person_name, capture_person_id) "
        "VALUES (?, 0, ?, ?)",
        (seq_id, person_name, capture_person_id),
    )
    conn.commit()


def _make_skeleton(conn: sqlite3.Connection, skel_id: str, name: str) -> None:
    conn.execute(
        "INSERT INTO skeletons (id, name, yaml_content, created_at) "
        "VALUES (?, ?, 'name: x', '2026-01-01')",
        (skel_id, name),
    )
    conn.commit()


def _select_trial(widget) -> None:
    """Select trial1 the same way the real combo would (added directly,
    bypassing _refresh_trials()'s own detection-run/sequence-join query --
    irrelevant to what these tests are checking)."""
    widget._trial_combo.addItem("trial1", {"trial_id": "trial1"})
    widget._trial_combo.setCurrentIndex(widget._trial_combo.count() - 1)


def test_capture_persons_used_when_defined(qapp, session_db) -> None:
    from app.pose.run_tracker import RunTrackerWidget

    _make_trial(session_db)
    _make_skeleton(session_db, "skel-a", "Skeleton A")
    person_id = create_person(session_db, "cap1", "Alice", default_skeleton_id="skel-a")
    _make_detection_run_with_sequence(
        session_db, detection_run_id="dr1", seq_id="seq1", capture_id="cap1",
        trial_id="trial1", person_name="Alice", capture_person_id=person_id,
    )

    w = RunTrackerWidget()
    w.set_session(session_db, str(session_db.execute("PRAGMA database_list").fetchone()[2]))
    _select_trial(w)

    assert w._people_table.rowCount() == 1
    chk = w._people_table.cellWidget(0, 0)
    assert isinstance(chk, QCheckBox)
    assert chk.text() == "Alice"
    assert chk.isChecked() is True
    assert w._add_person_btn.isHidden() is True

    # Default skeleton pre-filled from the person's own default.
    skel_combo = w._people_table.cellWidget(0, 2)
    assert skel_combo.currentData() == "skel-a"

    # Single detection run -- picker present but not needed, so disabled.
    dr_combo = w._people_table.cellWidget(0, 1)
    assert isinstance(dr_combo, QComboBox)
    assert dr_combo.count() == 1
    assert dr_combo.isEnabled() is False


def test_legacy_free_text_mode_when_no_capture_persons_defined(qapp, session_db) -> None:
    """A capture that hasn't defined any capture_persons yet must keep
    working exactly as before (combo-based row 0, "Add person…" visible)."""
    from app.pose.run_tracker import RunTrackerWidget

    _make_trial(session_db)
    _make_skeleton(session_db, "skel-a", "Skeleton A")
    _make_detection_run_with_sequence(
        session_db, detection_run_id="dr1", seq_id="seq1", capture_id="cap1",
        trial_id="trial1", person_name="Alice", capture_person_id=None,
    )

    w = RunTrackerWidget()
    w.set_session(session_db, str(session_db.execute("PRAGMA database_list").fetchone()[2]))
    _select_trial(w)

    assert w._people_table.rowCount() == 1
    widget = w._people_table.cellWidget(0, 0)
    assert isinstance(widget, QComboBox)  # legacy person picker, not a checkbox
    assert w._add_person_btn.isHidden() is False


def test_capture_person_with_multiple_detection_runs_enables_picker(qapp, session_db) -> None:
    from app.pose.run_tracker import RunTrackerWidget

    _make_trial(session_db)
    person_id = create_person(session_db, "cap1", "Alice")
    _make_detection_run_with_sequence(
        session_db, detection_run_id="dr1", seq_id="seq1", capture_id="cap1",
        trial_id="trial1", person_name="Alice", capture_person_id=person_id,
    )
    _make_detection_run_with_sequence(
        session_db, detection_run_id="dr2", seq_id="seq2", capture_id="cap1",
        trial_id="trial1", person_name="Alice", capture_person_id=person_id,
    )

    w = RunTrackerWidget()
    w.set_session(session_db, str(session_db.execute("PRAGMA database_list").fetchone()[2]))
    _select_trial(w)

    dr_combo = w._people_table.cellWidget(0, 1)
    assert dr_combo.count() == 2
    assert dr_combo.isEnabled() is True


def test_capture_person_with_no_observations_in_trial_is_skipped(qapp, session_db) -> None:
    """A defined capture person with no detection-run observations in this
    particular trial shouldn't produce an unusable row."""
    from app.pose.run_tracker import RunTrackerWidget

    _make_trial(session_db)
    create_person(session_db, "cap1", "Alice")  # no detection run for her at all

    w = RunTrackerWidget()
    w.set_session(session_db, str(session_db.execute("PRAGMA database_list").fetchone()[2]))
    _select_trial(w)

    assert w._people_table.rowCount() == 0


def test_unchecked_row_excluded_from_skeleton_ids_and_run(qapp, session_db) -> None:
    from app.pose.run_tracker import RunTrackerWidget

    _make_trial(session_db)
    _make_skeleton(session_db, "skel-a", "Skeleton A")
    person_id = create_person(session_db, "cap1", "Alice", default_skeleton_id="skel-a")
    _make_detection_run_with_sequence(
        session_db, detection_run_id="dr1", seq_id="seq1", capture_id="cap1",
        trial_id="trial1", person_name="Alice", capture_person_id=person_id,
    )

    w = RunTrackerWidget()
    w.set_session(session_db, str(session_db.execute("PRAGMA database_list").fetchone()[2]))
    _select_trial(w)

    assert w._current_skeleton_ids() == ["skel-a"]
    assert w._run_btn.isEnabled() is True

    chk = w._people_table.cellWidget(0, 0)
    chk.setChecked(False)

    assert w._current_skeleton_ids() == []
    assert w._run_btn.isEnabled() is False

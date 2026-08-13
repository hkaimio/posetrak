"""Tests for ExtrinsicsStatusDialog and _open_auto_calibrate_dialog
(UX Phase 2, see docs/roadmap/features/extrinsics-improvements/
extrinsics-ux-redesign.md): the status-first entry point that replaced
unconditionally launching a TOML-editing screen.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

from app.setup.page_extrinsics import ExtrinsicsStatusDialog, _open_auto_calibrate_dialog
from posetrak.db.db import create_session


@pytest.fixture()
def fake_conn(tmp_path: Path):
    conn = create_session(tmp_path / "status_dialog_test.db")
    yield conn
    conn.close()


def _seed_session(conn: sqlite3.Connection, session_id: str = "sess1") -> None:
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES (?, '2026-01-01')", (session_id,)
    )
    conn.commit()


def _seed_camera(conn: sqlite3.Connection, session_id: str, inst_id: str, label: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO camera_models (id, manufacturer, model_name) "
        "VALUES ('model1', 'Test', 'Cam')"
    )
    conn.execute(
        "INSERT INTO camera_instances (id, camera_model_id, label) VALUES (?, 'model1', ?)",
        (inst_id, label),
    )
    conn.execute(
        "INSERT INTO session_cameras (session_id, camera_instance_id, label) VALUES (?, ?, ?)",
        (session_id, inst_id, label),
    )
    conn.commit()


def _seed_calibration(
    conn: sqlite3.Connection, session_id: str, calib_id: str = "calib1",
    method: str = "rig-anchor", calibrated_at: str = "2026-08-12",
) -> None:
    conn.execute(
        "INSERT INTO extrinsic_calibrations (id, session_id, calibrated_at, method) "
        "VALUES (?, ?, ?, ?)",
        (calib_id, session_id, calibrated_at, method),
    )
    conn.commit()


def _seed_entry(
    conn: sqlite3.Connection, calib_id: str, camera_instance_id: str,
    R: np.ndarray = np.eye(3), t: np.ndarray = np.zeros(3),
) -> None:
    R_blob = struct.pack("<9d", *np.asarray(R, dtype=np.float64).flatten())
    t_blob = struct.pack("<3d", *np.asarray(t, dtype=np.float64).flatten())
    conn.execute(
        "INSERT INTO extrinsic_entries (extrinsic_calibration_id, camera_instance_id, R, t) "
        "VALUES (?, ?, ?, ?)",
        (calib_id, camera_instance_id, R_blob, t_blob),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# ExtrinsicsStatusDialog
# ---------------------------------------------------------------------------


def test_no_calibration_shows_not_set_summary(qapp, fake_conn) -> None:
    _seed_session(fake_conn)
    _seed_camera(fake_conn, "sess1", "inst1", "cam_A")
    dlg = ExtrinsicsStatusDialog(fake_conn, "sess1")
    try:
        assert "No extrinsics calibration yet" in dlg._summary_label.text()
        assert dlg._table.rowCount() == 1
        assert dlg._table.item(0, 0).text() == "cam_A"
        assert dlg._table.item(0, 1).text() == "—"
        assert dlg._table.item(0, 2).text() == "not solved"
    finally:
        dlg.done(0)


def test_calibration_with_all_cameras_solved(qapp, fake_conn) -> None:
    _seed_session(fake_conn)
    _seed_camera(fake_conn, "sess1", "inst1", "cam_A")
    _seed_camera(fake_conn, "sess1", "inst2", "cam_B")
    _seed_calibration(fake_conn, "sess1", method="rig-anchor", calibrated_at="2026-08-12T10:00:00")
    _seed_entry(fake_conn, "calib1", "inst1", t=np.array([1.0, 2.0, 3.0]))
    _seed_entry(fake_conn, "calib1", "inst2", t=np.array([4.0, 5.0, 6.0]))

    dlg = ExtrinsicsStatusDialog(fake_conn, "sess1")
    try:
        assert "2 / 2" in dlg._summary_label.text()
        assert "rig-anchor" in dlg._summary_label.text()
        assert "2026-08-12" in dlg._summary_label.text()

        rows = {
            dlg._table.item(i, 0).text(): (dlg._table.item(i, 1).text(), dlg._table.item(i, 2).text())
            for i in range(dlg._table.rowCount())
        }
        assert rows["cam_A"][1] == "rig-anchor"
        assert rows["cam_B"][1] == "rig-anchor"
        # Camera center = -R^T @ t; R = identity here, so center = -t.
        assert rows["cam_A"][0] == "-1.00, -2.00, -3.00"
    finally:
        dlg.done(0)


def test_calibration_with_partial_solve(qapp, fake_conn) -> None:
    _seed_session(fake_conn)
    _seed_camera(fake_conn, "sess1", "inst1", "cam_A")
    _seed_camera(fake_conn, "sess1", "inst2", "cam_B")
    _seed_calibration(fake_conn, "sess1")
    _seed_entry(fake_conn, "calib1", "inst1")  # only cam_A solved

    dlg = ExtrinsicsStatusDialog(fake_conn, "sess1")
    try:
        assert "1 / 2" in dlg._summary_label.text()
        rows = {
            dlg._table.item(i, 0).text(): dlg._table.item(i, 2).text()
            for i in range(dlg._table.rowCount())
        }
        assert rows["cam_A"] != "not solved"
        assert rows["cam_B"] == "not solved"
    finally:
        dlg.done(0)


def test_uses_most_recent_calibration(qapp, fake_conn) -> None:
    _seed_session(fake_conn)
    _seed_camera(fake_conn, "sess1", "inst1", "cam_A")
    _seed_calibration(fake_conn, "sess1", calib_id="old", method="toml-import", calibrated_at="2026-01-01")
    _seed_calibration(fake_conn, "sess1", calib_id="new", method="rig-anchor", calibrated_at="2026-08-12")
    _seed_entry(fake_conn, "new", "inst1")

    dlg = ExtrinsicsStatusDialog(fake_conn, "sess1")
    try:
        assert "rig-anchor" in dlg._summary_label.text()
        assert "toml-import" not in dlg._summary_label.text()
    finally:
        dlg.done(0)


def test_refresh_after_reopen_reflects_new_calibration(qapp, fake_conn) -> None:
    """_refresh() (called by _on_calibrate's callback and _on_import_toml
    after their dialog closes) picks up newly-written state, not a stale
    snapshot from __init__."""
    _seed_session(fake_conn)
    _seed_camera(fake_conn, "sess1", "inst1", "cam_A")
    dlg = ExtrinsicsStatusDialog(fake_conn, "sess1")
    try:
        assert "No extrinsics calibration yet" in dlg._summary_label.text()

        _seed_calibration(fake_conn, "sess1")
        _seed_entry(fake_conn, "calib1", "inst1")
        dlg._refresh()

        assert "1 / 1" in dlg._summary_label.text()
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# _open_auto_calibrate_dialog (shared launcher, guard clauses)
# ---------------------------------------------------------------------------


def test_open_auto_calibrate_dialog_no_shot_ids_warns(qapp, fake_conn, monkeypatch) -> None:
    warned = []
    monkeypatch.setattr(
        "app.setup.page_extrinsics.QMessageBox.warning",
        lambda *a, **kw: warned.append(a),
    )
    called = []
    _open_auto_calibrate_dialog(None, fake_conn, "sess1", [], lambda cid: called.append(cid))
    assert len(warned) == 1
    assert called == []


def test_open_auto_calibrate_dialog_no_cameras_warns(qapp, fake_conn, monkeypatch) -> None:
    _seed_session(fake_conn)
    conn2 = fake_conn
    conn2.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('cap1', 'sess1', 1)"
    )
    conn2.commit()
    warned = []
    monkeypatch.setattr(
        "app.setup.page_extrinsics.QMessageBox.warning",
        lambda *a, **kw: warned.append(a),
    )
    called = []
    _open_auto_calibrate_dialog(None, fake_conn, "sess1", ["cap1"], lambda cid: called.append(cid))
    assert len(warned) == 1  # no capture_videos rows -> no states -> "No cameras"
    assert called == []

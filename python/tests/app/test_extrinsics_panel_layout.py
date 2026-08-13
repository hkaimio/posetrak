"""Tests for the right-hand panel layout fixes in ExtrinsicsAutoCalibDialog
(collapsible sections, scroll fallback, multi-line intrinsics rows).

Raised directly via UI testing (2026-08-09): once the ArUco and ChArUco
panels joined the pre-existing Control Points / World Position / Camera
Intrinsics sections, the fixed-width sidebar no longer fit everything --
text and tables were clipped vertically, and the intrinsics combo box was
too narrow to read.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QGroupBox, QScrollArea

from app.setup.extrinsics_solver import CamCalibState
from app.setup.page_extrinsics import ExtrinsicsAutoCalibDialog


def _make_state(label: str) -> CamCalibState:
    K = np.eye(3)
    return CamCalibState(
        video_id=label, label=label, K=K, K_orig=K.copy(),
        dist=np.zeros((1, 4)), fisheye=False,
    )


@pytest.fixture()
def fake_conn(tmp_path):
    from posetrak.db.db import create_session
    conn = create_session(tmp_path / "panel_layout_test.db")
    yield conn
    conn.close()


def _find_group(dlg: ExtrinsicsAutoCalibDialog, title: str) -> QGroupBox:
    for g in dlg.findChildren(QGroupBox):
        if g.title() == title:
            return g
    raise AssertionError(f"no QGroupBox titled {title!r} found")


# ---------------------------------------------------------------------------
# Collapsible sections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("title", ["ArUco Markers", "ChArUco Board", "Camera Intrinsics"])
def test_section_is_collapsible_and_starts_expanded(qapp, fake_conn, title) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        group = _find_group(dlg, title)
        assert group.isCheckable()
        assert group.isChecked()
    finally:
        dlg.done(0)


def test_unchecking_aruco_group_hides_its_content(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    dlg.show()
    try:
        group = _find_group(dlg, "ArUco Markers")
        assert not dlg._marker_table.isHidden()
        group.setChecked(False)
        assert dlg._marker_table.isHidden()
        group.setChecked(True)
        assert not dlg._marker_table.isHidden()
    finally:
        dlg.done(0)


def test_unchecking_charuco_group_hides_its_content(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    dlg.show()
    try:
        group = _find_group(dlg, "ChArUco Board")
        assert not dlg._charuco_status_label.isHidden()
        group.setChecked(False)
        assert dlg._charuco_status_label.isHidden()
    finally:
        dlg.done(0)


def test_unchecking_intrinsics_group_hides_its_content(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    dlg.show()
    try:
        group = _find_group(dlg, "Camera Intrinsics")
        combo = dlg._intrinsics_combos["cam_A"]
        assert not combo.isHidden()
        group.setChecked(False)
        assert combo.isHidden()
    finally:
        dlg.done(0)


def test_control_points_and_world_position_are_not_collapsible(qapp, fake_conn) -> None:
    """Only the three sections that were actually reported as crowded are
    collapsible -- Control Points and World Position stay always-visible
    primary controls."""
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        assert not _find_group(dlg, "Control Points").isCheckable()
        assert not _find_group(dlg, "World position (optional)").isCheckable()
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# Scroll fallback
# ---------------------------------------------------------------------------


def test_right_panel_is_wrapped_in_a_scroll_area(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        scroll_areas = dlg.findChildren(QScrollArea)
        # The camera grid already uses one (Phase 1); the right-hand panel
        # must now be wrapped in a second one, sized to be resizable.
        panel_scrolls = [s for s in scroll_areas if s.widgetResizable()]
        assert len(panel_scrolls) >= 1
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# Multi-line intrinsics rows
# ---------------------------------------------------------------------------


def test_intrinsics_combo_not_squeezed_by_fixed_width_label(qapp, fake_conn) -> None:
    """Regression for the unreadable-combo complaint: the old layout gave
    the camera label a fixed 80px width in the same row as the combo and
    three checkboxes, leaving too little room for the combo's own text.
    The label is no longer fixed-width and shares no row with the combo."""
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        combo = dlg._intrinsics_combos["cam_A"]
        assert combo.minimumWidth() == 0  # not artificially constrained
    finally:
        dlg.done(0)


def test_intrinsics_group_has_separator_between_multiple_cameras(qapp, fake_conn) -> None:
    from PySide6.QtWidgets import QFrame

    dlg = ExtrinsicsAutoCalibDialog(
        [_make_state("cam_A"), _make_state("cam_B")], fake_conn, "sess1"
    )
    try:
        group = _find_group(dlg, "Camera Intrinsics")
        separators = [
            w for w in group.findChildren(QFrame)
            if w.frameShape() == QFrame.Shape.HLine
        ]
        assert len(separators) == 1  # one separator between the two camera blocks
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# Intrinsics combo: notes lead, technical details move to a detail label
# (2026-08-13) -- notes are what a user actually recognises a calibration
# by; date/RMS/model were the only thing shown before, and weren't useful
# for telling two calibrations of the same camera apart.
# ---------------------------------------------------------------------------


def _seed_intrinsics(
    conn, label: str = "cam_A", *, notes: str | None, rms: float = 0.5, is_default: bool = False,
) -> str:
    from posetrak.db.db import generate_id
    model_id, mode_id, inst_id, calib_id = (generate_id() for _ in range(4))
    conn.execute(
        "INSERT OR IGNORE INTO camera_models (id, manufacturer, model_name) VALUES (?, 'Test', 'Cam')",
        (model_id,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO camera_instances (id, camera_model_id, label) VALUES (?, ?, ?)",
        (inst_id, model_id, label),
    )
    conn.execute(
        "INSERT INTO camera_modes (id, camera_model_id, width_px, height_px, nominal_fps) "
        "VALUES (?, ?, 1920, 1080, 30.0)",
        (mode_id, model_id),
    )
    conn.execute(
        "INSERT INTO intrinsics_calibrations "
        "(id, camera_mode_id, calibrated_at, fx, fy, cx, cy, rms_error, notes) "
        "VALUES (?, ?, '2026-08-01', 1000.0, 1000.0, 960.0, 540.0, ?, ?)",
        (calib_id, mode_id, rms, notes),
    )
    if is_default:
        conn.execute(
            "UPDATE camera_modes SET default_intrinsics_calibration_id = ? WHERE id = ?",
            (calib_id, mode_id),
        )
    conn.commit()
    return calib_id


def test_intrinsics_combo_leads_with_notes(qapp, fake_conn) -> None:
    _seed_intrinsics(fake_conn, notes="tripod, wide lens")
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        combo = dlg._intrinsics_combos["cam_A"]
        assert combo.itemText(0) == "tripod, wide lens"
    finally:
        dlg.done(0)


def test_intrinsics_combo_falls_back_to_technical_summary_when_no_notes(qapp, fake_conn) -> None:
    _seed_intrinsics(fake_conn, notes=None, rms=1.23)
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        combo = dlg._intrinsics_combos["cam_A"]
        assert "1.23px" in combo.itemText(0)
        assert "2026-08-01" in combo.itemText(0)
    finally:
        dlg.done(0)


def test_intrinsics_combo_default_star_precedes_notes(qapp, fake_conn) -> None:
    _seed_intrinsics(fake_conn, notes="main rig", is_default=True)
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        combo = dlg._intrinsics_combos["cam_A"]
        assert combo.itemText(0) == "★ main rig"
    finally:
        dlg.done(0)


def test_intrinsics_detail_label_shows_technical_summary_for_selected_item(
    qapp, fake_conn,
) -> None:
    _seed_intrinsics(fake_conn, notes="tripod, wide lens", rms=0.87)
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        detail = dlg._intrinsics_detail_labels["cam_A"]
        assert "0.87px" in detail.text()
        assert "2026-08-01" in detail.text()
    finally:
        dlg.done(0)


def test_intrinsics_detail_label_updates_on_selection_change(qapp, fake_conn) -> None:
    _seed_intrinsics(fake_conn, notes="calib one", rms=0.10)
    _seed_intrinsics(fake_conn, notes="calib two", rms=9.99)
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        combo = dlg._intrinsics_combos["cam_A"]
        detail = dlg._intrinsics_detail_labels["cam_A"]
        assert combo.count() == 2

        for i in range(combo.count()):
            combo.setCurrentIndex(i)
            expected_rms = "0.10px" if combo.itemText(i) == "calib one" else "9.99px"
            assert expected_rms in detail.text()
    finally:
        dlg.done(0)

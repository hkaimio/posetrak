"""Tests for the right-hand panel layout fixes in ExtrinsicsAutoCalibDialog
(collapsible sections, scroll fallback) and the per-camera results table.

Raised directly via UI testing (2026-08-09): once the ArUco and ChArUco
panels joined the pre-existing Control Points / World Position / Camera
Intrinsics sections, the fixed-width sidebar no longer fit everything --
text and tables were clipped vertically, and the intrinsics combo box was
too narrow to read.

UX Phase 4 (2026-08-13, see docs/roadmap/features/extrinsics-improvements/
extrinsics-ux-redesign.md) later removed "Camera Intrinsics" as a sidebar
section entirely, folding it into the always-visible, full-width
per-camera results table instead (Intrinsics/Refine/Lock/Excl columns).
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


@pytest.mark.parametrize("title", ["ArUco Markers", "ChArUco Board"])
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


def test_intrinsics_has_no_collapsible_sidebar_group(qapp, fake_conn) -> None:
    """UX Phase 4 (see docs/roadmap/features/extrinsics-improvements/
    extrinsics-ux-redesign.md) removed the "Camera Intrinsics" sidebar
    section entirely -- it's folded into the always-visible, full-width
    per-camera results table instead (see the "cam pos table" tests
    below)."""
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        with pytest.raises(AssertionError):
            _find_group(dlg, "Camera Intrinsics")
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
# Per-camera results table: Intrinsics/Refine/Lock/Excl columns (UX Phase 4)
# ---------------------------------------------------------------------------


def test_cam_pos_table_has_nine_columns(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        assert dlg._cam_pos_table.columnCount() == 9
        headers = [dlg._cam_pos_table.horizontalHeaderItem(i).text() for i in range(9)]
        assert headers == [
            "Camera", "X (m)", "Y (m)", "Z (m)", "CP error",
            "Intrinsics", "Refine", "Lock", "Excl",
        ]
    finally:
        dlg.done(0)


def test_cam_pos_table_always_visible_before_any_solve(qapp, fake_conn) -> None:
    """Unlike before UX Phase 4, the table is populated (one row per
    camera, position/CP-error columns showing "—") and visible from
    dialog construction, not just after a solve/DB load -- it's now also
    where Intrinsics/Refine/Lock/Excl live, which are needed before
    solving, not only after."""
    dlg = ExtrinsicsAutoCalibDialog(
        [_make_state("cam_A"), _make_state("cam_B")], fake_conn, "sess1"
    )
    try:
        assert not dlg._cam_pos_table.isHidden()
        assert dlg._cam_pos_table.rowCount() == 2
        assert dlg._cam_pos_table.item(0, 1).text() == "—"
    finally:
        dlg.done(0)


def test_cam_pos_table_rows_have_intrinsics_combo_and_checkboxes(qapp, fake_conn) -> None:
    from PySide6.QtWidgets import QCheckBox, QComboBox

    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        assert isinstance(dlg._cam_pos_table.cellWidget(0, 5), QComboBox)
        assert dlg._cam_pos_table.cellWidget(0, 5) is dlg._intrinsics_combos["cam_A"]
        for col in (6, 7, 8):
            wrapper = dlg._cam_pos_table.cellWidget(0, col)
            assert wrapper.findChild(QCheckBox) is not None
    finally:
        dlg.done(0)


def test_cam_pos_table_lock_checkbox_disabled_before_solve(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        assert dlg._lock_cbs["cam_A"].isEnabled() is False
    finally:
        dlg.done(0)


def test_cam_pos_table_refine_checkbox_updates_refine_set(qapp, fake_conn) -> None:
    from PySide6.QtWidgets import QCheckBox

    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        refine_cb = dlg._cam_pos_table.cellWidget(0, 6).findChild(QCheckBox)
        refine_cb.setChecked(True)
        assert "cam_A" in dlg._refine_intrinsics
        refine_cb.setChecked(False)
        assert "cam_A" not in dlg._refine_intrinsics
    finally:
        dlg.done(0)


def test_cam_pos_table_excl_checkbox_updates_excluded_set(qapp, fake_conn) -> None:
    from PySide6.QtWidgets import QCheckBox

    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        excl_cb = dlg._cam_pos_table.cellWidget(0, 8).findChild(QCheckBox)
        excl_cb.setChecked(True)
        assert "cam_A" in dlg._excluded_cameras
    finally:
        dlg.done(0)


def test_cam_pos_table_row_count_stable_across_refresh(qapp, fake_conn) -> None:
    """_refresh_cam_pos_table() (called again after every solve/DB load)
    must not disturb the Intrinsics/Refine/Lock/Excl cell widgets built
    once at dialog construction."""
    from PySide6.QtWidgets import QComboBox

    dlg = ExtrinsicsAutoCalibDialog(
        [_make_state("cam_A"), _make_state("cam_B")], fake_conn, "sess1"
    )
    try:
        combo_before = dlg._cam_pos_table.cellWidget(1, 5)
        dlg._refresh_cam_pos_table()
        assert dlg._cam_pos_table.rowCount() == 2
        assert dlg._cam_pos_table.cellWidget(1, 5) is combo_before
        assert isinstance(dlg._cam_pos_table.cellWidget(1, 5), QComboBox)
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


def test_intrinsics_detail_tooltip_shows_technical_summary_for_selected_item(
    qapp, fake_conn,
) -> None:
    """The technical summary moved from a separate label to the combo's
    own tooltip in UX Phase 4 (see docs/roadmap/features/
    extrinsics-improvements/extrinsics-ux-redesign.md) -- table cells
    don't have room for a second line of text the way the old sidebar
    block did."""
    _seed_intrinsics(fake_conn, notes="tripod, wide lens", rms=0.87)
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        combo = dlg._intrinsics_combos["cam_A"]
        assert "0.87px" in combo.toolTip()
        assert "2026-08-01" in combo.toolTip()
    finally:
        dlg.done(0)


def test_intrinsics_detail_tooltip_updates_on_selection_change(qapp, fake_conn) -> None:
    _seed_intrinsics(fake_conn, notes="calib one", rms=0.10)
    _seed_intrinsics(fake_conn, notes="calib two", rms=9.99)
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        combo = dlg._intrinsics_combos["cam_A"]
        assert combo.count() == 2

        for i in range(combo.count()):
            combo.setCurrentIndex(i)
            expected_rms = "0.10px" if combo.itemText(i) == "calib one" else "9.99px"
            assert expected_rms in combo.toolTip()
    finally:
        dlg.done(0)

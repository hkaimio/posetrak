# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the right-hand panel layout fixes in ExtrinsicsAutoCalibDialog
(section grouping, scroll fallback) and the per-camera results table.

Raised directly via UI testing (2026-08-09): once the ArUco and ChArUco
panels joined the pre-existing Control Points / World Position / Camera
Intrinsics sections, the fixed-width sidebar no longer fit everything --
text and tables were clipped vertically, and the intrinsics combo box was
too narrow to read. The original fix made the crowded sections
collapsible; UX Phase 6 (2026-08-14) replaced that with an "Actions"/
"Anchoring" regrouping; a 2026-08-14 follow-up moved most of that
content onto a button bar instead; a 2026-08-15 follow-up (this file's
current state, Harri's second sidebar-cleanup round) moved the button
bar back into the sidebar as 4 flat top-level sections -- Control
Points, "Calibration rig setup", "Markers", "Solve" -- replacing the
Actions/Anchoring split entirely. No collapsing, no tabs, at any point
in that history, per Harri's original steer against progressive
disclosure for an iterative workflow.

UX Phase 4 (2026-08-13, see docs/roadmap/features/extrinsics-improvements/
extrinsics-ux-redesign.md) removed "Camera Intrinsics" as a sidebar
section entirely, folding it into the always-visible, full-width
per-camera results table instead (Intrinsics/Refine/Lock/Excl columns).
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QPushButton, QScrollArea, QSplitter

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
# 4 top-level sidebar sections (2026-08-15 follow-up) -- Control Points,
# "Calibration rig setup", "Markers", "Solve". No collapsing, no tabs;
# every section is always visible. Replaces the Actions/Anchoring split
# entirely -- see the module docstring for the full history.
# ---------------------------------------------------------------------------


_TOP_LEVEL_SECTIONS = ["Control Points", "Calibration rig setup", "Markers", "Solve"]


@pytest.mark.parametrize("title", _TOP_LEVEL_SECTIONS)
def test_no_sidebar_section_is_collapsible(qapp, fake_conn, title) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        assert not _find_group(dlg, title).isCheckable()
    finally:
        dlg.done(0)


def test_sidebar_has_exactly_the_four_top_level_sections(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        groups = {g.title() for g in dlg.findChildren(QGroupBox) if g.title() in _TOP_LEVEL_SECTIONS}
        assert groups == set(_TOP_LEVEL_SECTIONS)
        # None of the 4 is nested inside another -- they're siblings.
        for title in _TOP_LEVEL_SECTIONS:
            group = _find_group(dlg, title)
            others = [_find_group(dlg, t) for t in _TOP_LEVEL_SECTIONS if t != title]
            for other in others:
                assert group not in other.findChildren(QGroupBox)
    finally:
        dlg.done(0)


def test_world_position_group_lives_in_the_detail_pane_not_the_sidebar(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        wp_group = _find_group(dlg, "World position (optional)")
        assert not wp_group.isCheckable()
        # Not a child of any of the 4 top-level sidebar sections...
        for title in _TOP_LEVEL_SECTIONS:
            assert wp_group not in _find_group(dlg, title).findChildren(QGroupBox)
        # ...but is a page of the Data tab's detail pane stack.
        from PySide6.QtWidgets import QStackedWidget
        stack = dlg.findChild(QStackedWidget)
        assert stack is not None
        assert wp_group in stack.findChildren(QGroupBox)
    finally:
        dlg.done(0)


def test_aruco_settings_have_no_sidebar_group(qapp, fake_conn) -> None:
    """"ArUco Markers" removed entirely (2026-08-14 follow-up) -- its
    dictionary/default-size/min-marker-% settings are headless state now
    (_init_aruco_detect_settings), edited via the "Markers" section's
    "Detect markers…" dialog instead of a permanently-visible sidebar
    section."""
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        with pytest.raises(AssertionError):
            _find_group(dlg, "ArUco Markers")
        # The settings still exist, just not parented into any layout.
        assert dlg._aruco_dict_combo.parent() is None
    finally:
        dlg.done(0)


def test_charuco_settings_and_anchor_section_have_no_sidebar_group(qapp, fake_conn) -> None:
    """"ChArUco Board" (detection settings) and "ChArUco Anchor" (Set
    origin & axes/Clear board detections) are both gone entirely
    (2026-08-15 follow-up) -- per Harri, ChArUco is close enough to a
    calibration rig that "Calibration rig setup"'s own "Calib rig…"
    button (its ChArUco Board tab) is the only anchor entry point now.
    _charuco_status_label is headless too, same reasoning as the ArUco
    settings above -- see _init_charuco_detect_settings."""
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        with pytest.raises(AssertionError):
            _find_group(dlg, "ChArUco Board")
        with pytest.raises(AssertionError):
            _find_group(dlg, "ChArUco Anchor")
        assert dlg._charuco_dict_combo.parent() is None
        assert dlg._charuco_status_label.parent() is None
    finally:
        dlg.done(0)


def test_rig_loading_settings_have_no_sidebar_group(qapp, fake_conn) -> None:
    """"Marker Rig / Scene Markers" (loading settings) removed entirely
    -- folded into "Calibration rig setup"'s "Calib rig…" dialog instead
    (_init_rig_detect_settings). The rig anchor status/actions
    themselves (_rig_status_label, min-cameras, Anchor Rig/Clear) DO
    still have a visible home -- in "Calibration rig setup" now, not a
    separate "Rig Anchor" section; see
    test_calibration_rig_setup_group_contents."""
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        with pytest.raises(AssertionError):
            _find_group(dlg, "Marker Rig / Scene Markers")
        assert dlg._rig_min_marker_pct_spin.parent() is None
    finally:
        dlg.done(0)


def test_calibration_rig_setup_group_contents(qapp, fake_conn) -> None:
    """"Calib rig…" (moved back from the button bar) + the rig-anchor
    status/actions that used to be their own "Rig Anchor" section + a
    new "Manage rigs…" (2026-08-15 follow-up)."""
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        group = _find_group(dlg, "Calibration rig setup")
        assert dlg._calib_rig_btn in group.findChildren(QPushButton)
        assert dlg._rig_status_label in group.findChildren(type(dlg._rig_status_label))
        manage_btns = [b for b in group.findChildren(QPushButton) if b.text() == "Manage rigs…"]
        assert len(manage_btns) == 1
        # "Save markers…" is NOT here -- it isn't rig-specific (Markers section instead).
        assert dlg._save_markers_btn not in group.findChildren(QPushButton)
    finally:
        dlg.done(0)


def test_markers_group_contents(qapp, fake_conn) -> None:
    """"Detect markers…"/"Load markers…"/"Save markers…" (moved back
    from the button bar) + a new "Manage markers…" (renamed/relocated
    from "Manage Scene Markers…" in the old "Rig Anchor" section, 2026-08-15
    follow-up -- Harri: "the 'manage scene markers' is bit odd - I don't
    really understand what it does")."""
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        group = _find_group(dlg, "Markers")
        buttons = group.findChildren(QPushButton)
        assert dlg._detect_markers_btn in buttons
        assert dlg._load_markers_btn in buttons
        assert dlg._save_markers_btn in buttons
        manage_btns = [b for b in buttons if b.text() == "Manage markers…"]
        assert len(manage_btns) == 1
        for btn in (dlg._detect_markers_btn, dlg._load_markers_btn, dlg._save_markers_btn):
            assert not btn.isHidden()
    finally:
        dlg.done(0)


def test_solve_group_contents(qapp, fake_conn) -> None:
    """SIFT/RANSAC settings, Solve/Cancel, "Load from DB…", and the
    shared status label (moved back from the button bar, 2026-08-15
    follow-up)."""
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        group = _find_group(dlg, "Solve")
        assert dlg._sift_check in group.findChildren(type(dlg._sift_check))
        assert dlg._ransac_px_spin in group.findChildren(type(dlg._ransac_px_spin))
        assert dlg._solve_btn in group.findChildren(QPushButton)
        assert dlg._cancel_btn in group.findChildren(QPushButton)
        assert dlg._load_db_btn in group.findChildren(QPushButton)
        assert dlg._status_label in group.findChildren(type(dlg._status_label))
    finally:
        dlg.done(0)


def test_intrinsics_has_no_sidebar_group(qapp, fake_conn) -> None:
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
# Cameras/Data tab container + adjustable-height splitter (2026-08-14
# follow-up) -- the two tables used to stack on top of each other, each
# capped at a fixed setMaximumHeight; now they share one QTabWidget in its
# own pane of a vertical QSplitter, so height is user-adjustable by
# dragging the splitter handle instead of a fixed guess neither table's
# row count may match.
# ---------------------------------------------------------------------------


def test_cam_pos_and_data_tables_share_a_tab_container(qapp, fake_conn) -> None:
    from PySide6.QtWidgets import QTabWidget

    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        tabs = dlg.findChild(QTabWidget)
        assert tabs is not None
        assert tabs.count() == 2
        assert tabs.widget(0) is dlg._cam_pos_table
        # Tab 1 wraps the Data table plus its detail pane (2026-08-14
        # follow-up), not the bare table.
        assert dlg._data_table in tabs.widget(1).findChildren(type(dlg._data_table))
        assert [tabs.tabText(i) for i in range(2)] == ["Cameras", "Data"]
    finally:
        dlg.done(0)


def test_cameras_tab_shown_by_default(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        assert not dlg._cam_pos_table.isHidden()
    finally:
        dlg.done(0)


def test_tab_container_height_is_adjustable_via_splitter(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        splitters = [
            s for s in dlg.findChildren(QSplitter)
            if s.orientation() == Qt.Orientation.Vertical
        ]
        assert len(splitters) == 1
        vsplit = splitters[0]
        assert vsplit.indexOf(dlg._tables_tabs) >= 0
        # Neither table has a fixed setMaximumHeight cap anymore -- Qt's
        # QWIDGETSIZE_MAX sentinel means "no cap set".
        assert dlg._cam_pos_table.maximumHeight() >= 16000000
        assert dlg._data_table.maximumHeight() >= 16000000
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# Per-camera results table: Intrinsics/Refine/Lock/Excl columns (UX Phase 4)
# ---------------------------------------------------------------------------


def test_cam_pos_table_has_eleven_columns(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        assert dlg._cam_pos_table.columnCount() == 11
        headers = [dlg._cam_pos_table.horizontalHeaderItem(i).text() for i in range(11)]
        assert headers == [
            "Camera", "X (m)", "Y (m)", "Z (m)", "CP error",
            "Intrinsics", "Calib Date", "Calib RMS", "Refine", "Lock", "Excl",
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
        for col in (8, 9, 10):
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
        refine_cb = dlg._cam_pos_table.cellWidget(0, 8).findChild(QCheckBox)
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
        excl_cb = dlg._cam_pos_table.cellWidget(0, 10).findChild(QCheckBox)
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


# ---------------------------------------------------------------------------
# Calib Date / Calib RMS columns (2026-08-14 follow-up) -- date and RMS
# moved out of the combo's tooltip into their own visible table columns,
# updating live as the intrinsics selection changes.
# ---------------------------------------------------------------------------


def test_calib_date_and_rms_columns_populated_on_build(qapp, fake_conn) -> None:
    _seed_intrinsics(fake_conn, notes="tripod, wide lens", rms=0.87)
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        assert dlg._cam_pos_table.item(0, 6).text() == "2026-08-01"
        assert dlg._cam_pos_table.item(0, 7).text() == "0.87px"
    finally:
        dlg.done(0)


def test_calib_date_and_rms_columns_update_on_selection_change(qapp, fake_conn) -> None:
    _seed_intrinsics(fake_conn, notes="calib one", rms=0.10)
    _seed_intrinsics(fake_conn, notes="calib two", rms=9.99)
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        combo = dlg._intrinsics_combos["cam_A"]
        assert combo.count() == 2

        for i in range(combo.count()):
            combo.setCurrentIndex(i)
            expected_rms = "0.10px" if combo.itemText(i) == "calib one" else "9.99px"
            assert dlg._cam_pos_table.item(0, 7).text() == expected_rms
    finally:
        dlg.done(0)


def test_calib_date_and_rms_columns_correct_row_for_multiple_cameras(qapp, fake_conn) -> None:
    _seed_intrinsics(fake_conn, label="cam_A", notes="a", rms=1.11)
    _seed_intrinsics(fake_conn, label="cam_B", notes="b", rms=2.22)
    dlg = ExtrinsicsAutoCalibDialog(
        [_make_state("cam_A"), _make_state("cam_B")], fake_conn, "sess1"
    )
    try:
        row_a = dlg._cam_pos_row_by_vid["cam_A"]
        row_b = dlg._cam_pos_row_by_vid["cam_B"]
        assert dlg._cam_pos_table.item(row_a, 7).text() == "1.11px"
        assert dlg._cam_pos_table.item(row_b, 7).text() == "2.22px"
    finally:
        dlg.done(0)

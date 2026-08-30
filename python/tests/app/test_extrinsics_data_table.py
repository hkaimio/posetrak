# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified Data table (UX Phase 7, see docs/roadmap/features/
extrinsics-improvements/extrinsics-ux-redesign.md): one row per data point
currently contributing to (or available to) the solve -- control points,
detected ArUco markers, ChArUco board corners, rig corners, and camera-
position observations. Replaces the sidebar's old _cp_list/_marker_table.

Column layout: 0 Type, 1 Label, 2 Cameras, 3 World position, 4 Source,
5 Size (m) (a deliberate deviation from the design doc's literal column
list -- see _refresh_data_table's docstring for why).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QDoubleSpinBox

from app.setup.extrinsics_solver import CalibResult, CamCalibState, MarkerGroup, MarkerPoseResult
from app.setup.fiducial_markers import ARUCO_DICTIONARIES
from app.setup.page_extrinsics import ExtrinsicsAutoCalibDialog
from posetrak.db.manage_marker_body import upsert_scene_marker_body

_ONE_MARKER_RIG_YAML = """\
name: test-rig
units: meters
markers:
  - name: front
    type: aruco
    dictionary: DICT_4X4_50
    id: "3"
    size: 0.1
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
"""


def _render_marker_image(marker_id: int, dictionary: str = "DICT_4X4_50") -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary])
    gray = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)
    padded = cv2.copyMakeBorder(gray, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


def _make_state(label: str, image: np.ndarray | None = None) -> CamCalibState:
    K = np.array([[900.0, 0.0, 150.0], [0.0, 900.0, 150.0], [0.0, 0.0, 1.0]])
    return CamCalibState(
        video_id=label, label=label, K=K, K_orig=K.copy(),
        dist=np.zeros((1, 4)), fisheye=False, image=image,
    )


@pytest.fixture()
def fake_conn(tmp_path):
    from posetrak.db.db import create_session
    conn = create_session(tmp_path / "data_table_test.db")
    yield conn
    conn.close()


@pytest.fixture()
def rig_yaml_path(tmp_path: Path) -> Path:
    p = tmp_path / "test_rig.yaml"
    p.write_text(_ONE_MARKER_RIG_YAML, encoding="utf-8")
    return p


def _rows_by_type(dlg, kind: str) -> list[int]:
    return [
        row for row in range(dlg._data_table.rowCount())
        if dlg._data_table.item(row, 0).text() == kind
    ]


# ---------------------------------------------------------------------------
# Population from every source
# ---------------------------------------------------------------------------


def test_empty_dialog_has_no_data_rows(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        assert dlg._data_table.rowCount() == 0
    finally:
        dlg.done(0)


def test_control_point_appears_as_a_row(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        dlg._add_control_point()

        rows = _rows_by_type(dlg, "CP")
        assert len(rows) == 1
        row = rows[0]
        assert dlg._data_table.item(row, 1).text() == "CP1"
        assert dlg._data_table.item(row, 2).text() == ""  # no cameras yet
        assert dlg._data_table.item(row, 3).text() == ""
        assert dlg._data_table.item(row, 4).text() == "manual"
    finally:
        dlg.done(0)


def test_cp_row_cameras_count_updates_on_placement(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        dlg._add_control_point()
        dlg._on_cam_click("cam_A", 5.0, 6.0)

        row = _rows_by_type(dlg, "CP")[0]
        assert dlg._data_table.item(row, 2).text() == "1"
    finally:
        dlg.done(0)


def test_cameras_column_shows_camera_order_numbers_not_a_count(qapp, fake_conn) -> None:
    """1-based order number matching each camera's own row in the Cameras
    table (_cam_pos_row_by_vid), not a bare count -- so the column
    identifies *which* cameras, not just how many (2026-08-14 follow-up).
    A CP seen only by the 2nd and 3rd of three cameras must show "2, 3",
    not a re-numbered "1, 2" of just the observing subset."""
    dlg = ExtrinsicsAutoCalibDialog(
        [_make_state("cam_A"), _make_state("cam_B"), _make_state("cam_C")], fake_conn, "sess1",
    )
    try:
        dlg._add_control_point()
        dlg._on_cam_click("cam_B", 1.0, 1.0)
        dlg._on_cam_click("cam_C", 2.0, 2.0)

        row = _rows_by_type(dlg, "CP")[0]
        assert dlg._data_table.item(row, 2).text() == "2, 3"
    finally:
        dlg.done(0)


def test_cp_row_world_position_updates_on_apply_xyz(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        dlg._add_control_point()  # auto-selects -> arms xyz panel
        dlg._xyz_x.setText("1.0")
        dlg._xyz_y.setText("2.0")
        dlg._xyz_z.setText("3.0")
        dlg._apply_xyz()

        row = _rows_by_type(dlg, "CP")[0]
        assert dlg._data_table.item(row, 3).text() == "1.000, 2.000, 3.000"
    finally:
        dlg.done(0)


def test_xyz_field_accepts_decimal_point_regardless_of_system_locale(qapp, fake_conn) -> None:
    """QDoubleValidator validates the "." keystroke against the OS locale's
    decimal separator by default (e.g. "," on a Finnish Windows install),
    silently dropping every "." a user types -- while _xyz_field_value()
    parses the result with Python's locale-independent float(), which only
    ever accepts ".". Reported as "keyboard does nothing" in e2e testing
    (2026-08-21) on a Finnish-locale machine; reproduced independent of the
    "Fix 3-D position in BA" enable-gating (also caught in the same report)
    by driving real key events through QTest rather than setText().
    """
    from PySide6.QtTest import QTest

    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        dlg._add_control_point()  # auto-selects -> arms xyz panel
        dlg._xyz_enabled.setChecked(True)
        dlg._xyz_x.setFocus()
        dlg._xyz_x.selectAll()
        QTest.keyClicks(dlg._xyz_x, "12.5")
        assert dlg._xyz_x.text() == "12.5"
        assert dlg._xyz_field_value(dlg._xyz_x) == 12.5
    finally:
        dlg.done(0)


def test_marker_row_appears_after_detection(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_aruco_clicked("cam_A")

        rows = _rows_by_type(dlg, "Marker")
        assert len(rows) == 1
        row = rows[0]
        assert dlg._data_table.item(row, 1).text() == "3"
        assert dlg._data_table.item(row, 2).text() == "1"
        assert dlg._data_table.item(row, 4).text() == "DICT_4X4_50"
        assert isinstance(dlg._data_table.cellWidget(row, 5), QDoubleSpinBox)
    finally:
        dlg.done(0)


def test_marker_row_world_position_populates_after_solve(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(9))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._marker_groups["9"] = MarkerGroup(marker_id="9", size=0.1, dictionary="DICT_5X5_50")
        rvec, _ = cv2.Rodrigues(np.eye(3))
        mp = MarkerPoseResult(
            rvec=rvec, tvec=np.array([1.0, 2.0, 3.0]), size=0.1, rms_reprojection_px=0.5
        )
        states[0].R = np.eye(3)
        states[0].t = np.zeros((3, 1))
        result = CalibResult(
            cameras={"cam_A": states[0]}, points_3d=[], reprojection_errors={},
            unsolved=[], pair_matches={}, marker_poses={"9": mp},
        )
        dlg._on_solve_done(result)

        row = _rows_by_type(dlg, "Marker")[0]
        assert dlg._data_table.item(row, 3).text() == "1.000, 2.000, 3.000"
    finally:
        if dlg._solve_thread is not None:
            dlg._solve_thread.wait(2000)
        dlg.done(0)


def test_board_corner_rows_appear_after_charuco_detection(qapp, fake_conn) -> None:
    from app.setup.fiducial_markers import CharucoBoardDetection, CharucoCornerObs

    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        dlg._charuco_detections["cam_A"] = CharucoBoardDetection(corners=[
            CharucoCornerObs(corner_id=0, video_id="cam_A", frame_idx=0, px=1.0, py=1.0,
                              local_xyz=np.zeros(3)),
        ])
        dlg._refresh_charuco_status()

        rows = _rows_by_type(dlg, "Board corner")
        assert len(rows) == 1
        assert dlg._data_table.item(rows[0], 4).text() == "charuco"
        assert dlg._data_table.item(rows[0], 3).text() == ""  # not anchored yet
    finally:
        dlg.done(0)


def _render_board_image(squares_x=5, squares_y=7, square_length=0.04, marker_length=0.02) -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    gray = board.generateImage((500, 700), marginSize=30)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_board_corner_rows_world_position_populates_once_anchored(qapp, fake_conn) -> None:
    """Companion to test_board_corner_rows_appear_after_charuco_detection
    (free state) -- checks the anchored transition too, mirroring
    test_rig_corner_rows_appear_once_anchored below for the rig case."""
    states = [_make_state("cam_A", _render_board_image())]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_charuco_clicked("cam_A")
        rows_before = _rows_by_type(dlg, "Board corner")
        assert rows_before
        assert all(dlg._data_table.item(r, 3).text() == "" for r in rows_before)

        dlg._on_anchor_from_board()

        rows_after = _rows_by_type(dlg, "Board corner")
        assert rows_after
        assert all(dlg._data_table.item(r, 3).text() != "" for r in rows_after)
    finally:
        dlg.done(0)


def test_rig_corner_rows_appear_once_anchored(qapp, fake_conn, rig_yaml_path) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)  # auto-anchors
        assert dlg._rig_anchored

        rows = _rows_by_type(dlg, "Rig corner")
        assert len(rows) == 4  # one marker's 4 corners
        for row in rows:
            assert dlg._data_table.item(row, 4).text() == "rig:test-rig"
            assert dlg._data_table.item(row, 3).text() != ""  # already fixed
    finally:
        dlg.done(0)


def test_loaded_marker_config_rows_not_labeled_rig_corner(qapp, fake_conn) -> None:
    """A "Load Markers…" config (Tier B, no physical rig) reuses the same
    detector/anchor mechanism as a genuine rig -- but must not be shown
    as "Rig corner" in the Data table, which reads as if a rig had been
    saved/loaded too (Harri's 2026-08-14 report: saved only sized ArUco
    markers, no rig, then saw "Rig corner" rows after loading that
    config into another capture)."""
    fake_conn.execute(
        "INSERT OR IGNORE INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    fake_conn.commit()
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:3", R=np.eye(3), t=np.zeros(3), group_name="room7",
        marker_type="aruco", dictionary="DICT_4X4_50", marker_id="3", marker_size=0.1,
    )
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_scene_marker_group("room7")
        assert dlg._rig_anchored
        assert dlg._rig_source == "scene_markers"

        assert _rows_by_type(dlg, "Rig corner") == []
        rows = _rows_by_type(dlg, "Loaded marker corner")
        assert len(rows) == 4  # one marker's 4 corners
        for row in rows:
            assert dlg._data_table.item(row, 4).text() == "scene markers (room7)"
    finally:
        dlg.done(0)


def test_detail_pane_clear_works_for_loaded_marker_corner_row(qapp, fake_conn) -> None:
    fake_conn.execute(
        "INSERT OR IGNORE INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    fake_conn.commit()
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:3", R=np.eye(3), t=np.zeros(3), group_name="room7",
        marker_type="aruco", dictionary="DICT_4X4_50", marker_id="3", marker_size=0.1,
    )
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_scene_marker_group("room7")
        row = _rows_by_type(dlg, "Loaded marker corner")[0]
        dlg._data_table.selectRow(row)

        assert dlg._detail_stack.currentIndex() == 3
        assert dlg._detail_group_label.text() == "Loaded marker corner"

        dlg._detail_group_clear_fn()

        assert not dlg._rig_anchored
        assert dlg._rig_detections_by_camera == {}
    finally:
        dlg.done(0)


def test_cam_pos_obs_row_appears_after_set(qapp, fake_conn) -> None:
    """First list representation for cam-pos observations -- previously
    only visible as an image overlay."""
    dlg = ExtrinsicsAutoCalibDialog(
        [_make_state("cam_A"), _make_state("cam_B")], fake_conn, "sess1",
    )
    try:
        dlg._on_cam_pos_set("cam_A", "cam_B", 42.0, 24.0)

        rows = _rows_by_type(dlg, "Cam pos obs")
        assert len(rows) == 1
        assert "cam_B" in dlg._data_table.item(rows[0], 1).text()
        assert "cam_A" in dlg._data_table.item(rows[0], 1).text()
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# CP-placement-via-table-selection: behavioral parity with the old
# _cp_list-driven flow
# ---------------------------------------------------------------------------


def test_selecting_cp_row_arms_click_to_place(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        dlg._add_control_point()
        dlg._add_control_point()  # now CP2 is selected, CP1 is not

        cp1_row = next(
            row for row in _rows_by_type(dlg, "CP")
            if dlg._data_table.item(row, 1).text() == "CP1"
        )
        dlg._data_table.selectRow(cp1_row)

        assert dlg._selected_cp_idx == 0
        assert dlg._xyz_enabled.isEnabled()
    finally:
        dlg.done(0)


def test_selecting_non_cp_row_disarms_placement(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._add_control_point()
        dlg._on_detect_aruco_clicked("cam_A")
        assert dlg._selected_cp_idx == 0

        marker_row = _rows_by_type(dlg, "Marker")[0]
        dlg._data_table.selectRow(marker_row)

        assert dlg._selected_cp_idx is None
        assert not dlg._xyz_enabled.isEnabled()

        # A camera click with nothing armed must not touch the CP.
        dlg._on_cam_click("cam_A", 1.0, 1.0)
        assert "cam_A" not in dlg._control_points[0].obs
    finally:
        dlg.done(0)


def test_double_click_cp_row_renames_it(qapp, fake_conn, monkeypatch) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        dlg._add_control_point()
        row = _rows_by_type(dlg, "CP")[0]

        from PySide6.QtWidgets import QInputDialog
        monkeypatch.setattr(QInputDialog, "getText", lambda *a, **kw: ("renamed", True))
        dlg._on_data_table_double_clicked(row, 1)

        assert dlg._control_points[0].name == "renamed"
        new_row = _rows_by_type(dlg, "CP")[0]
        assert dlg._data_table.item(new_row, 1).text() == "renamed"
    finally:
        dlg.done(0)


def test_double_click_non_cp_row_is_a_noop(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_aruco_clicked("cam_A")
        row = _rows_by_type(dlg, "Marker")[0]
        dlg._on_data_table_double_clicked(row, 1)  # must not raise
        assert "3" in dlg._marker_groups
    finally:
        dlg.done(0)


def test_delete_selected_cp_removes_its_row(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        dlg._add_control_point()
        assert len(_rows_by_type(dlg, "CP")) == 1

        dlg._delete_control_point()

        assert _rows_by_type(dlg, "CP") == []
        assert dlg._selected_cp_idx is None
    finally:
        dlg.done(0)


def test_delete_with_nothing_selected_is_a_noop(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        dlg._add_control_point()
        dlg._data_table.clearSelection()
        dlg._selected_cp_idx = None

        dlg._delete_control_point()  # must not raise or delete anything

        assert len(dlg._control_points) == 1
    finally:
        dlg.done(0)


def test_selection_survives_an_unrelated_data_table_refresh(qapp, fake_conn) -> None:
    """_refresh_data_table() rebuilds every row from scratch -- the
    currently-selected CP must still be selected afterward (silently, no
    re-emitted selection signal -- see the method's own docstring for why
    a plain selectRow() there would loop back into itself)."""
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._add_control_point()
        dlg._add_control_point()
        assert dlg._selected_cp_idx == 1

        dlg._on_detect_aruco_clicked("cam_A")  # unrelated mutation, triggers a refresh

        assert dlg._selected_cp_idx == 1
        selected_rows = {i.row() for i in dlg._data_table.selectedIndexes()}
        expected_row = next(
            row for row, idx in dlg._data_table_cp_rows.items() if idx == 1
        )
        assert selected_rows == {expected_row}
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# Per-row-type detail pane (2026-08-14 follow-up, see
# _build_detail_pane/_refresh_detail_pane): a QStackedWidget beside the
# Data table -- index 0 empty placeholder, 1 CP (World position, moved
# here from the old standalone sidebar groupbox), 2 Marker (Clear just
# this one), 3 Rig/Board corner (Clear the whole detected feature), 4 Cam
# pos obs (Remove just this one).
# ---------------------------------------------------------------------------


def test_detail_pane_shows_placeholder_with_nothing_selected(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        assert dlg._detail_stack.currentIndex() == 0
    finally:
        dlg.done(0)


def test_detail_pane_shows_world_position_for_cp_row(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        dlg._add_control_point()  # auto-selects
        assert dlg._detail_stack.currentIndex() == 1
        assert dlg._xyz_enabled.isEnabled()
    finally:
        dlg.done(0)


def test_detail_pane_shows_clear_for_marker_row(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_aruco_clicked("cam_A")
        row = _rows_by_type(dlg, "Marker")[0]
        dlg._data_table.selectRow(row)

        assert dlg._detail_stack.currentIndex() == 2
        assert "3" in dlg._detail_marker_label.text()
    finally:
        dlg.done(0)


def test_detail_pane_marker_clear_removes_just_that_marker(qapp, fake_conn) -> None:
    states = [
        _make_state("cam_A", _render_marker_image(3)),
    ]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._marker_groups["3"] = MarkerGroup(marker_id="3", size=0.1, dictionary="DICT_4X4_50")
        dlg._marker_groups["9"] = MarkerGroup(marker_id="9", size=0.1, dictionary="DICT_4X4_50")
        dlg._refresh_data_table()
        row = next(
            r for r in _rows_by_type(dlg, "Marker") if dlg._data_table.item(r, 1).text() == "3"
        )
        dlg._data_table.selectRow(row)

        dlg._on_clear_single_marker(dlg._detail_marker_id)

        assert set(dlg._marker_groups) == {"9"}
    finally:
        dlg.done(0)


def test_detail_pane_shows_clear_for_rig_corner_row(qapp, fake_conn, rig_yaml_path) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)  # auto-anchors
        row = _rows_by_type(dlg, "Rig corner")[0]
        dlg._data_table.selectRow(row)

        assert dlg._detail_stack.currentIndex() == 3
        assert dlg._detail_group_label.text() == "Rig corner"

        dlg._detail_group_clear_fn()  # same as clicking the pane's Clear button

        assert not dlg._rig_anchored
        assert dlg._rig_detections_by_camera == {}
    finally:
        dlg.done(0)


def test_detail_pane_shows_clear_for_board_corner_row(qapp, fake_conn) -> None:
    from app.setup.fiducial_markers import CharucoBoardDetection, CharucoCornerObs

    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        dlg._charuco_detections["cam_A"] = CharucoBoardDetection(corners=[
            CharucoCornerObs(corner_id=0, video_id="cam_A", frame_idx=0, px=1.0, py=1.0,
                              local_xyz=np.zeros(3)),
        ])
        dlg._refresh_charuco_status()
        row = _rows_by_type(dlg, "Board corner")[0]
        dlg._data_table.selectRow(row)

        assert dlg._detail_stack.currentIndex() == 3
        assert dlg._detail_group_label.text() == "Board corner"

        dlg._detail_group_clear_fn()

        assert dlg._charuco_detections == {}
    finally:
        dlg.done(0)


def test_detail_pane_shows_remove_for_cam_pos_obs_row(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog(
        [_make_state("cam_A"), _make_state("cam_B")], fake_conn, "sess1",
    )
    try:
        dlg._on_cam_pos_set("cam_A", "cam_B", 1.0, 2.0)
        row = _rows_by_type(dlg, "Cam pos obs")[0]
        dlg._data_table.selectRow(row)

        assert dlg._detail_stack.currentIndex() == 4
        assert dlg._detail_camobs_payload == ("cam_A", "cam_B")

        dlg._on_remove_cam_pos_obs(*dlg._detail_camobs_payload)

        assert dlg._cam_pos_obs == []
        assert _rows_by_type(dlg, "Cam pos obs") == []
    finally:
        dlg.done(0)


def test_removing_cam_pos_obs_only_clears_that_subject_marker(qapp, fake_conn) -> None:
    """An observer that marked two different subjects must keep the other
    one's marker after removing just one (remove_user_cam_pos_marker, not
    clear_user_cam_pos_markers)."""
    dlg = ExtrinsicsAutoCalibDialog(
        [_make_state("cam_A"), _make_state("cam_B"), _make_state("cam_C")], fake_conn, "sess1",
    )
    try:
        dlg._on_cam_pos_set("cam_A", "cam_B", 1.0, 2.0)
        dlg._on_cam_pos_set("cam_A", "cam_C", 3.0, 4.0)

        dlg._on_remove_cam_pos_obs("cam_A", "cam_B")

        remaining = {(o.observer, o.subject) for o in dlg._cam_pos_obs}
        assert remaining == {("cam_A", "cam_C")}
        w = dlg._cam_widgets["cam_A"]
        assert "cam_B" not in w._user_cam_pos_markers
        assert "cam_C" in w._user_cam_pos_markers
    finally:
        dlg.done(0)


def test_detail_pane_resets_after_clearing_the_selected_marker_via_bulk_clear(
    qapp, fake_conn,
) -> None:
    """Clearing ALL markers (the ArUco panel's own "Clear markers", not
    the detail pane's single-marker Clear) drops the marker row the
    detail pane was showing -- it must fall back to the placeholder, not
    keep pointing at a marker that no longer exists."""
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_aruco_clicked("cam_A")
        row = _rows_by_type(dlg, "Marker")[0]
        dlg._data_table.selectRow(row)
        assert dlg._detail_stack.currentIndex() == 2

        dlg._on_clear_markers()

        assert dlg._detail_stack.currentIndex() == 0
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# Selecting a Data table row highlights the corresponding point(s) in
# camera views (2026-08-15 follow-up) -- previously CP-only
# (_selected_cp_idx), now every row type. _ClickableImageWidget._markers
# entries are (x, y, color, label, selected) tuples; "selected" drives
# the lighter-fill/thicker-ring paint treatment (_lighten_color).
# ---------------------------------------------------------------------------


def test_selecting_marker_row_highlights_its_corners_in_every_camera(qapp, fake_conn) -> None:
    states = [
        _make_state("cam_A", _render_marker_image(3)),
        _make_state("cam_B", _render_marker_image(3)),
    ]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_aruco_clicked("cam_A")
        dlg._on_detect_aruco_clicked("cam_B")
        row = _rows_by_type(dlg, "Marker")[0]

        dlg._data_table.selectRow(row)

        for vid in ("cam_A", "cam_B"):
            markers = dlg._cam_widgets[vid]._markers
            assert markers  # something was drawn
            assert all(m[4] for m in markers)  # every corner of this marker is selected
    finally:
        dlg.done(0)


def test_selecting_board_corner_row_highlights_only_that_one_corner(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog([_make_state("cam_A")], fake_conn, "sess1")
    try:
        from app.setup.fiducial_markers import CharucoBoardDetection, CharucoCornerObs

        dlg._charuco_detections["cam_A"] = CharucoBoardDetection(corners=[
            CharucoCornerObs(corner_id=0, video_id="cam_A", frame_idx=0, px=1.0, py=1.0,
                              local_xyz=np.zeros(3)),
            CharucoCornerObs(corner_id=1, video_id="cam_A", frame_idx=0, px=2.0, py=2.0,
                              local_xyz=np.zeros(3)),
        ])
        dlg._refresh_charuco_status()
        rows = _rows_by_type(dlg, "Board corner")
        assert len(rows) == 2
        dlg._data_table.selectRow(rows[0])

        markers = dlg._cam_widgets["cam_A"]._markers
        selected = [m for m in markers if m[4]]
        not_selected = [m for m in markers if not m[4]]
        assert len(selected) == 1
        assert len(not_selected) == 1
    finally:
        dlg.done(0)


def test_selecting_loaded_marker_corner_row_highlights_only_that_one_corner(
    qapp, fake_conn,
) -> None:
    fake_conn.execute(
        "INSERT OR IGNORE INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    fake_conn.commit()
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:3", R=np.eye(3), t=np.zeros(3), group_name="room7",
        marker_type="aruco", dictionary="DICT_4X4_50", marker_id="3", marker_size=0.1,
    )
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_scene_marker_group("room7")
        rows = _rows_by_type(dlg, "Loaded marker corner")
        assert len(rows) == 4
        dlg._data_table.selectRow(rows[0])

        markers = dlg._cam_widgets["cam_A"]._markers
        selected = [m for m in markers if m[4]]
        assert len(selected) == 1
    finally:
        dlg.done(0)


def test_selecting_rig_corner_row_highlights_only_that_one_corner(
    qapp, fake_conn, rig_yaml_path,
) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)  # auto-anchors
        rows = _rows_by_type(dlg, "Rig corner")
        assert len(rows) == 4
        dlg._data_table.selectRow(rows[0])

        markers = dlg._cam_widgets["cam_A"]._markers
        selected = [m for m in markers if m[4]]
        assert len(selected) == 1
    finally:
        dlg.done(0)


def test_selecting_cam_pos_obs_row_highlights_that_marker(qapp, fake_conn) -> None:
    dlg = ExtrinsicsAutoCalibDialog(
        [_make_state("cam_A"), _make_state("cam_B")], fake_conn, "sess1",
    )
    try:
        dlg._on_cam_pos_set("cam_A", "cam_B", 1.0, 2.0)
        row = _rows_by_type(dlg, "Cam pos obs")[0]

        dlg._data_table.selectRow(row)

        assert dlg._cam_widgets["cam_A"]._selected_cam_pos_subject == "cam_B"
    finally:
        dlg.done(0)


def test_selection_highlight_clears_when_a_different_row_is_selected(qapp, fake_conn) -> None:
    """Selecting a CP row after a Cam pos obs row must not leave the
    latter's highlight stuck (_refresh_markers resets it for every
    widget every call)."""
    dlg = ExtrinsicsAutoCalibDialog(
        [_make_state("cam_A"), _make_state("cam_B")], fake_conn, "sess1",
    )
    try:
        dlg._on_cam_pos_set("cam_A", "cam_B", 1.0, 2.0)
        dlg._data_table.selectRow(_rows_by_type(dlg, "Cam pos obs")[0])
        assert dlg._cam_widgets["cam_A"]._selected_cam_pos_subject == "cam_B"

        dlg._add_control_point()  # different row, different type

        assert dlg._cam_widgets["cam_A"]._selected_cam_pos_subject is None
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# Detail pane "Remove Corner"/"Remove Marker" for "Loaded marker corner"
# rows (2026-08-15 follow-up, Harri: "add to the side pane option to
# remove individual corner"). A physical rig's "Rig corner" rows only get
# the blanket Clear -- see _build_detail_pane's docstring for why.
# ---------------------------------------------------------------------------


def _load_room7_markers(dlg, fake_conn) -> None:
    fake_conn.execute(
        "INSERT OR IGNORE INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    fake_conn.commit()
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:3", R=np.eye(3), t=np.zeros(3), group_name="room7",
        marker_type="aruco", dictionary="DICT_4X4_50", marker_id="3", marker_size=0.1,
    )
    dlg._load_rig_config_from_scene_marker_group("room7")


def test_remove_buttons_visible_only_for_loaded_marker_corner(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        _load_room7_markers(dlg, fake_conn)
        row = _rows_by_type(dlg, "Loaded marker corner")[0]
        dlg._data_table.selectRow(row)

        assert not dlg._detail_remove_corner_btn.isHidden()
        assert not dlg._detail_remove_marker_btn.isHidden()
    finally:
        dlg.done(0)


def test_remove_buttons_hidden_for_physical_rig_corner(qapp, fake_conn, rig_yaml_path) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)
        row = _rows_by_type(dlg, "Rig corner")[0]
        dlg._data_table.selectRow(row)

        assert dlg._detail_remove_corner_btn.isHidden()
        assert dlg._detail_remove_marker_btn.isHidden()
    finally:
        dlg.done(0)


def test_remove_corner_excludes_just_that_corner(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        _load_room7_markers(dlg, fake_conn)
        assert len(_rows_by_type(dlg, "Loaded marker corner")) == 4
        row = _rows_by_type(dlg, "Loaded marker corner")[0]
        dlg._data_table.selectRow(row)
        removed_key = dlg._detail_rig_corner_key

        dlg._on_remove_rig_corner()

        assert removed_key in dlg._rig_excluded_corners
        assert len(_rows_by_type(dlg, "Loaded marker corner")) == 3
        remaining_keys = {
            info[1] for row2, info in dlg._data_table_row_info.items()
            if info[0] == "Loaded marker corner"
        }
        assert removed_key not in remaining_keys
    finally:
        dlg.done(0)


def test_remove_marker_excludes_all_its_corners(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        _load_room7_markers(dlg, fake_conn)
        row = _rows_by_type(dlg, "Loaded marker corner")[0]
        dlg._data_table.selectRow(row)

        dlg._on_remove_rig_marker()

        assert "3" in dlg._rig_excluded_markers
        assert _rows_by_type(dlg, "Loaded marker corner") == []
    finally:
        dlg.done(0)


def test_remove_corner_leaves_the_solve_unaffected_by_that_corner(qapp, fake_conn) -> None:
    """The exclusion is applied in _rig_control_points() itself, so
    _on_solve's all_cps (built from that same method) doesn't include a
    removed corner either -- not just the Data table display."""
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        _load_room7_markers(dlg, fake_conn)
        assert len(dlg._rig_control_points()) == 4
        row = _rows_by_type(dlg, "Loaded marker corner")[0]
        dlg._data_table.selectRow(row)

        dlg._on_remove_rig_corner()

        assert len(dlg._rig_control_points()) == 3
    finally:
        dlg.done(0)


def test_rig_exclusions_reset_when_a_new_config_is_loaded(
    qapp, fake_conn, rig_yaml_path, monkeypatch,
) -> None:
    from PySide6.QtWidgets import QMessageBox

    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        _load_room7_markers(dlg, fake_conn)
        row = _rows_by_type(dlg, "Loaded marker corner")[0]
        dlg._data_table.selectRow(row)
        dlg._on_remove_rig_corner()
        assert dlg._rig_excluded_corners

        # Already anchored from room7 -- loading a different config asks
        # to confirm replacing it first (_confirm_replace_existing_anchor).
        monkeypatch.setattr(
            "app.setup.page_extrinsics.QMessageBox.question",
            lambda *a, **kw: QMessageBox.StandardButton.Yes,
        )
        dlg._load_rig_config_from_path(rig_yaml_path)  # a genuinely different config

        assert dlg._rig_excluded_corners == set()
        assert dlg._rig_excluded_markers == set()
    finally:
        dlg.done(0)


def test_rig_exclusions_persist_across_clear_and_redetect(qapp, fake_conn) -> None:
    """Clear (redetect) keeps the same loaded config -- a per-corner
    exclusion is a judgment about that config's own markers, not about
    one detection pass, so it should survive a Clear."""
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        _load_room7_markers(dlg, fake_conn)
        row = _rows_by_type(dlg, "Loaded marker corner")[0]
        dlg._data_table.selectRow(row)
        dlg._on_remove_rig_corner()
        excluded = set(dlg._rig_excluded_corners)
        assert excluded

        dlg._on_clear_rig()
        dlg._on_anchor_from_rig()

        assert dlg._rig_excluded_corners == excluded
        assert len(_rows_by_type(dlg, "Loaded marker corner")) == 3
    finally:
        dlg.done(0)

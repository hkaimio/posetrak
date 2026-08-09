"""Dialog-level tests for Phase 4's ChArUco board detection + coordinate-
system anchoring UI in ExtrinsicsAutoCalibDialog.

See docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md, section 4.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.setup.extrinsics_solver import CamCalibState
from app.setup.page_extrinsics import ExtrinsicsAutoCalibDialog

_REAL_BOARD_IMAGE = Path(__file__).parent.parent / "data" / "charuco_board_sample.png"


def _render_board_image(squares_x=5, squares_y=7, square_length=0.04, marker_length=0.02) -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    gray = board.generateImage((500, 700), marginSize=30)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _make_state(label: str, image: np.ndarray | None = None) -> CamCalibState:
    K = np.array([[900.0, 0.0, 250.0], [0.0, 900.0, 350.0], [0.0, 0.0, 1.0]])
    return CamCalibState(
        video_id=label, label=label, K=K, K_orig=K.copy(),
        dist=np.zeros((1, 4)), fisheye=False, image=image,
    )


@pytest.fixture()
def fake_conn(tmp_path):
    from posetrak.db.db import create_session
    conn = create_session(tmp_path / "charuco_ui_test.db")
    yield conn
    conn.close()


def test_dialog_starts_with_no_board_detections(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_board_image())]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        assert dlg._charuco_detections == {}
        assert not dlg._charuco_anchored
        assert dlg._charuco_status_label.text() == "No board detected yet."
    finally:
        dlg.done(0)


def test_detect_charuco_button_finds_board(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_board_image())]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        # Defaults (5x7, 0.04/0.02) match the rendered board.
        dlg._on_detect_charuco_clicked("cam_A")

        assert "cam_A" in dlg._charuco_detections
        assert len(dlg._charuco_detections["cam_A"].corners) == 24
        assert "24 corner" in dlg._charuco_status_label.text()
        assert "not yet anchored" in dlg._charuco_status_label.text()
    finally:
        dlg.done(0)


def test_detect_with_no_image_shows_warning_not_crash(qapp, fake_conn, monkeypatch) -> None:
    states = [_make_state("cam_A", image=None)]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        warned = []
        monkeypatch.setattr(
            "app.setup.page_extrinsics.QMessageBox.warning",
            lambda *a, **kw: warned.append(a),
        )
        dlg._on_detect_charuco_clicked("cam_A")
        assert len(warned) == 1
        assert dlg._charuco_detections == {}
    finally:
        dlg.done(0)


def test_no_board_found_updates_status_not_crash(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", np.full((300, 300, 3), 255, dtype=np.uint8))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_charuco_clicked("cam_A")
        assert dlg._charuco_detections == {}
        assert "No ChArUco board detected" in dlg._status_label.text()
    finally:
        dlg.done(0)


def test_mismatched_board_settings_finds_nothing_useful(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_board_image(squares_x=5, squares_y=7))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._charuco_squares_x_spin.setValue(8)
        dlg._charuco_squares_y_spin.setValue(8)
        dlg._on_detect_charuco_clicked("cam_A")
        # Either nothing detected, or far fewer corners than the real 24 --
        # never a crash, never mistaken full-confidence detection.
        if "cam_A" in dlg._charuco_detections:
            assert len(dlg._charuco_detections["cam_A"].corners) < 24
    finally:
        dlg.done(0)


def test_detect_records_current_scrub_frame(qapp, fake_conn) -> None:
    states = [CamCalibState(
        video_id="cam_A", label="cam_A",
        K=np.array([[900.0, 0.0, 250.0], [0.0, 900.0, 350.0], [0.0, 0.0, 1.0]]),
        K_orig=np.eye(3), dist=np.zeros((1, 4)), fisheye=False,
        file_path="/nonexistent/cam_A.mp4", first_frame=0, last_frame=99,
    )]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._scrub_bars["cam_A"].seek(17)
        dlg._states_by_id["cam_A"].image = _render_board_image()
        dlg._on_detect_charuco_clicked("cam_A")

        det = dlg._charuco_detections["cam_A"]
        assert all(c.frame_idx == 17 for c in det.corners)
    finally:
        dlg.done(0)


def test_redetect_same_camera_overwrites(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_board_image())]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_charuco_clicked("cam_A")
        dlg._on_detect_charuco_clicked("cam_A")
        assert len(dlg._charuco_detections) == 1
    finally:
        dlg.done(0)


def test_detecting_across_two_cameras_accumulates(qapp, fake_conn) -> None:
    states = [
        _make_state("cam_A", _render_board_image()),
        _make_state("cam_B", _render_board_image()),
    ]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_charuco_clicked("cam_A")
        dlg._on_detect_charuco_clicked("cam_B")
        assert set(dlg._charuco_detections) == {"cam_A", "cam_B"}
        cps = {cp.name: cp for cp in dlg._charuco_control_points()}
        # Every corner both cameras share should have both cameras in obs.
        shared = [cp for cp in cps.values() if set(cp.obs) == {"cam_A", "cam_B"}]
        assert len(shared) == 24
    finally:
        dlg.done(0)


def test_control_points_free_before_anchor(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_board_image())]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_charuco_clicked("cam_A")
        cps = dlg._charuco_control_points()
        assert len(cps) == 24
        assert all(cp.world_xyz is None for cp in cps)
    finally:
        dlg.done(0)


def test_anchor_without_detection_shows_warning(qapp, fake_conn, monkeypatch) -> None:
    states = [_make_state("cam_A", _render_board_image())]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        warned = []
        monkeypatch.setattr(
            "app.setup.page_extrinsics.QMessageBox.warning",
            lambda *a, **kw: warned.append(a),
        )
        dlg._on_anchor_from_board()
        assert len(warned) == 1
        assert not dlg._charuco_anchored
    finally:
        dlg.done(0)


def test_anchor_fixes_world_xyz_on_all_corners(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_board_image())]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_charuco_clicked("cam_A")
        dlg._on_anchor_from_board()

        assert dlg._charuco_anchored
        cps = dlg._charuco_control_points()
        assert len(cps) == 24
        assert all(cp.world_xyz is not None for cp in cps)
        assert "anchored" in dlg._charuco_status_label.text()
    finally:
        dlg.done(0)


def test_anchor_face_up_vs_face_down_flips_axes(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_board_image())]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_charuco_clicked("cam_A")

        dlg._charuco_face_up_cb.setChecked(True)
        dlg._on_anchor_from_board()
        up_xyz = {cp.name: cp.world_xyz.copy() for cp in dlg._charuco_control_points()}

        dlg._charuco_face_up_cb.setChecked(False)
        dlg._on_anchor_from_board()
        down_xyz = {cp.name: cp.world_xyz.copy() for cp in dlg._charuco_control_points()}

        name = next(iter(up_xyz))
        np.testing.assert_allclose(down_xyz[name], up_xyz[name] * np.array([1.0, -1.0, -1.0]))
    finally:
        dlg.done(0)


def test_clear_charuco_resets_state(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_board_image())]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_charuco_clicked("cam_A")
        dlg._on_anchor_from_board()

        dlg._on_clear_charuco()

        assert dlg._charuco_detections == {}
        assert not dlg._charuco_anchored
        assert dlg._charuco_status_label.text() == "No board detected yet."
    finally:
        dlg.done(0)


def test_board_corners_drawn_as_overlay_markers(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_board_image())]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_charuco_clicked("cam_A")
        widget = dlg._cam_widgets["cam_A"]
        assert len(widget._markers) == 24
    finally:
        dlg.done(0)


def test_solve_includes_charuco_control_points(qapp, fake_conn, monkeypatch) -> None:
    states = [_make_state("cam_A", _render_board_image())]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_charuco_clicked("cam_A")
        dlg._on_anchor_from_board()

        captured = {}
        from app.setup import page_extrinsics as module
        orig = module._SolveThread.__init__

        def spy_init(self, states_, control_points, *args, **kwargs):
            captured["control_points"] = control_points
            return orig(self, states_, control_points, *args, **kwargs)

        monkeypatch.setattr(module._SolveThread, "__init__", spy_init)
        dlg._sift_check.setChecked(False)
        dlg._on_solve()

        names = {cp.name for cp in captured["control_points"]}
        assert "charuco_c0" in names
        assert len(captured["control_points"]) == 24
    finally:
        if dlg._solve_thread is not None:
            dlg._solve_thread.wait(2000)
        dlg.done(0)


# ---------------------------------------------------------------------------
# Legacy-pattern checkbox (found necessary via a real UI test against a
# calib.io-generated board, 2026-08-09 -- see CharucoDetector's docstring
# and status.md's Phase 4 notes).
# ---------------------------------------------------------------------------


def test_legacy_pattern_checkbox_defaults_unchecked(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_board_image())]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        assert not dlg._charuco_legacy_pattern_cb.isChecked()
    finally:
        dlg.done(0)


@pytest.mark.skipif(not _REAL_BOARD_IMAGE.exists(), reason="real board fixture image not present")
def test_legacy_pattern_checkbox_required_for_real_calibio_board(qapp, fake_conn) -> None:
    """End-to-end through the actual dialog widgets: the same board that
    failed to detect at all during live testing until legacy_pattern was
    set."""
    img = cv2.imread(str(_REAL_BOARD_IMAGE))
    states = [_make_state("cam_A", img)]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._charuco_squares_x_spin.setValue(11)
        dlg._charuco_squares_y_spin.setValue(8)
        dlg._charuco_square_length_spin.setValue(0.02)
        dlg._charuco_marker_length_spin.setValue(0.015)

        dlg._charuco_legacy_pattern_cb.setChecked(False)
        dlg._on_detect_charuco_clicked("cam_A")
        assert "cam_A" not in dlg._charuco_detections

        dlg._charuco_legacy_pattern_cb.setChecked(True)
        dlg._on_detect_charuco_clicked("cam_A")
        assert "cam_A" in dlg._charuco_detections
        assert len(dlg._charuco_detections["cam_A"].corners) >= 8
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# Min marker size (%) -- found necessary via a second live-testing round
# against a full 4K camera frame (2026-08-09), where the board's markers
# were too small relative to the frame for cv2.aruco's own default
# minMarkerPerimeterRate. See CharucoDetector's docstring and status.md's
# Phase 4 notes.
# ---------------------------------------------------------------------------


_REAL_BOARD_SMALL_IN_4K_IMAGE = (
    Path(__file__).parent.parent / "data" / "charuco_board_small_in_4k_frame.png"
)


def test_min_marker_size_spin_defaults_lower_than_opencv_default(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_board_image())]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        assert dlg._charuco_min_marker_pct_spin.value() < 3.0  # cv2's own default is 3%
    finally:
        dlg.done(0)


@pytest.mark.skipif(
    not _REAL_BOARD_SMALL_IN_4K_IMAGE.exists(), reason="real 4K-frame fixture image not present"
)
def test_min_marker_size_required_for_board_small_in_full_frame(qapp, fake_conn) -> None:
    """End-to-end through the actual dialog widgets: the same real 4K frame
    that found nothing at cv2's own default even with axis/legacy-pattern
    already correct."""
    img = cv2.imread(str(_REAL_BOARD_SMALL_IN_4K_IMAGE))
    states = [_make_state("cam_A", img)]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._charuco_squares_x_spin.setValue(11)
        dlg._charuco_squares_y_spin.setValue(8)
        dlg._charuco_square_length_spin.setValue(0.02)
        dlg._charuco_marker_length_spin.setValue(0.015)
        dlg._charuco_legacy_pattern_cb.setChecked(True)

        dlg._charuco_min_marker_pct_spin.setValue(3.0)  # cv2's own default
        dlg._on_detect_charuco_clicked("cam_A")
        assert "cam_A" not in dlg._charuco_detections

        dlg._charuco_min_marker_pct_spin.setValue(1.0)
        dlg._on_detect_charuco_clicked("cam_A")
        assert "cam_A" in dlg._charuco_detections
        assert len(dlg._charuco_detections["cam_A"].corners) >= 8
    finally:
        dlg.done(0)

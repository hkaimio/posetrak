"""Dialog-level tests for Phase 3's ArUco detection UI in
ExtrinsicsAutoCalibDialog.

See docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md, section 3 ("Fiducial marker detection
framework"), the "UI" bullet of the Phase 3 phased-plan entry.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.setup.extrinsics_solver import CamCalibState, MarkerGroup
from app.setup.fiducial_markers import ARUCO_DICTIONARIES
from app.setup.page_extrinsics import ExtrinsicsAutoCalibDialog


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
    conn = create_session(tmp_path / "aruco_ui_test.db")
    yield conn
    conn.close()


def _marker_rows(dlg) -> list[int]:
    """Row indices of every "Marker"-type row in the unified Data table
    (UX Phase 7) -- markers no longer get their own sidebar table."""
    return [
        row for row in range(dlg._data_table.rowCount())
        if dlg._data_table.item(row, 0).text() == "Marker"
    ]


def _marker_row(dlg, marker_id: str) -> int:
    for row in _marker_rows(dlg):
        if dlg._data_table.item(row, 1).text() == marker_id:
            return row
    raise AssertionError(f"no Marker row for {marker_id!r} in the Data table")


def test_dialog_starts_with_no_marker_groups(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        assert dlg._marker_groups == {}
        assert _marker_rows(dlg) == []
    finally:
        dlg.done(0)


def test_detect_button_finds_marker_and_populates_table(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_aruco_clicked("cam_A")

        assert set(dlg._marker_groups) == {"3"}
        row = _marker_row(dlg, "3")
        assert dlg._data_table.item(row, 2).text() == "1"  # 1 camera
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
        dlg._on_detect_aruco_clicked("cam_A")
        assert len(warned) == 1
        assert dlg._marker_groups == {}
    finally:
        dlg.done(0)


def test_detect_records_current_scrub_frame(qapp, fake_conn) -> None:
    states = [CamCalibState(
        video_id="cam_A", label="cam_A",
        K=np.eye(3), K_orig=np.eye(3), dist=np.zeros((1, 4)), fisheye=False,
        file_path="/nonexistent/cam_A.mp4", first_frame=0, last_frame=99,
    )]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._scrub_bars["cam_A"].seek(42)
        dlg._states_by_id["cam_A"].image = _render_marker_image(1)
        dlg._on_detect_aruco_clicked("cam_A")

        mg = dlg._marker_groups["1"]
        assert mg.obs["cam_A"][0].frame_idx == 42
    finally:
        dlg.done(0)


def test_default_size_flows_into_new_marker_group(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._aruco_default_size_spin.setValue(0.08)
        dlg._on_detect_aruco_clicked("cam_A")
        assert dlg._marker_groups["3"].size == pytest.approx(0.08)
    finally:
        dlg.done(0)


def test_zero_default_size_means_unknown(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._aruco_default_size_spin.setValue(0.0)
        dlg._on_detect_aruco_clicked("cam_A")
        assert dlg._marker_groups["3"].size is None
    finally:
        dlg.done(0)


def test_redetecting_same_camera_overwrites_not_duplicates(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_aruco_clicked("cam_A")
        dlg._on_detect_aruco_clicked("cam_A")

        assert len(dlg._marker_groups) == 1
        assert len(_marker_rows(dlg)) == 1
        row = _marker_row(dlg, "3")
        assert dlg._data_table.item(row, 2).text() == "1"
    finally:
        dlg.done(0)


def test_detecting_across_two_cameras_accumulates_one_group(qapp, fake_conn) -> None:
    states = [
        _make_state("cam_A", _render_marker_image(3)),
        _make_state("cam_B", _render_marker_image(3)),
    ]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_aruco_clicked("cam_A")
        dlg._on_detect_aruco_clicked("cam_B")

        assert set(dlg._marker_groups) == {"3"}
        assert set(dlg._marker_groups["3"].obs) == {"cam_A", "cam_B"}
        row = _marker_row(dlg, "3")
        assert dlg._data_table.item(row, 2).text() == "2"
    finally:
        dlg.done(0)


def test_per_marker_size_override_widget_updates_group(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._aruco_default_size_spin.setValue(0.05)
        dlg._on_detect_aruco_clicked("cam_A")
        assert dlg._marker_groups["3"].size == pytest.approx(0.05)

        override_spin = dlg._data_table.cellWidget(_marker_row(dlg, "3"), 5)
        override_spin.setValue(0.2)
        assert dlg._marker_groups["3"].size == pytest.approx(0.2)

        override_spin.setValue(0.0)  # back to "use default"
        assert dlg._marker_groups["3"].size == pytest.approx(0.05)
    finally:
        dlg.done(0)


def test_size_override_persists_across_redetect(qapp, fake_conn) -> None:
    """A marker already given a custom size in the table must keep it on
    re-detection, not silently revert to the default (see
    _on_detect_aruco_clicked's re-resolve-before-merge comment)."""
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._aruco_default_size_spin.setValue(0.05)
        dlg._on_detect_aruco_clicked("cam_A")
        dlg._data_table.cellWidget(_marker_row(dlg, "3"), 5).setValue(0.3)

        dlg._on_detect_aruco_clicked("cam_A")  # re-detect same camera -- rebuilds the table

        assert dlg._marker_groups["3"].size == pytest.approx(0.3)
        assert dlg._data_table.cellWidget(_marker_row(dlg, "3"), 5).value() == pytest.approx(0.3)
    finally:
        dlg.done(0)


def test_clear_markers_empties_groups_and_table(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_aruco_clicked("cam_A")
        assert dlg._marker_groups

        dlg._on_clear_markers()
        assert dlg._marker_groups == {}
        assert _marker_rows(dlg) == []
    finally:
        dlg.done(0)


def test_marker_corners_drawn_as_overlay_markers(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_aruco_clicked("cam_A")
        widget = dlg._cam_widgets["cam_A"]
        assert len(widget._markers) == 4  # one dot per corner
    finally:
        dlg.done(0)


def test_manual_cp_and_aruco_markers_coexist_after_refresh(qapp, fake_conn) -> None:
    """_refresh_markers()'s single clear-then-redraw pass must not let one
    kind of marker wipe out the other."""
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._add_control_point()
        dlg._on_cam_click("cam_A", 5.0, 5.0)
        dlg._on_detect_aruco_clicked("cam_A")

        widget = dlg._cam_widgets["cam_A"]
        assert len(widget._markers) == 1 + 4  # 1 manual CP + 4 ArUco corners
    finally:
        dlg.done(0)


def test_solve_thread_receives_marker_groups(qapp, fake_conn, monkeypatch) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_aruco_clicked("cam_A")

        captured = {}
        real_init = None
        from app.setup import page_extrinsics as module

        orig = module._SolveThread.__init__

        def spy_init(self, *args, **kwargs):
            captured["marker_groups"] = kwargs.get("marker_groups")
            return orig(self, *args, **kwargs)

        monkeypatch.setattr(module._SolveThread, "__init__", spy_init)
        dlg._sift_check.setChecked(False)  # cp_only, avoid a real SIFT/solve pass
        dlg._on_solve()

        assert captured["marker_groups"] is not None
        assert isinstance(captured["marker_groups"][0], MarkerGroup)
    finally:
        if dlg._solve_thread is not None:
            dlg._solve_thread.wait(2000)
        dlg.done(0)


# ---------------------------------------------------------------------------
# Min marker size (%) -- same underlying cv2 setting and default-too-strict
# problem as ChArUco's (see test_extrinsics_charuco_ui.py and
# CharucoDetector's docstring); ArucoDetector needed the identical fix
# since a ChArUco board's markers are ordinary ArUco markers under the hood.
# ---------------------------------------------------------------------------


def test_min_marker_size_spin_defaults_lower_than_opencv_default(qapp, fake_conn) -> None:
    states = [_make_state("cam_A")]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        assert dlg._aruco_min_marker_pct_spin.value() < 3.0  # cv2's own default is 3%
    finally:
        dlg.done(0)


def test_min_marker_size_reaches_arucodetector(qapp, fake_conn, monkeypatch) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._aruco_min_marker_pct_spin.setValue(2.5)

        captured = {}
        from app.setup import page_extrinsics as module
        orig = module.ArucoDetector.__init__

        def spy_init(self, *args, **kwargs):
            captured["min_marker_perimeter_rate"] = kwargs.get("min_marker_perimeter_rate")
            return orig(self, *args, **kwargs)

        monkeypatch.setattr(module.ArucoDetector, "__init__", spy_init)
        dlg._on_detect_aruco_clicked("cam_A")

        assert captured["min_marker_perimeter_rate"] == pytest.approx(0.025)
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# ChArUco board markers double-counted as plain ArUco markers -- raised
# directly (2026-08-09): a ChArUco board's own markers are ordinary ArUco
# markers of the same dictionary, so "Detect ArUco" would otherwise also
# decode every one of the board's own sub-markers as if they were separate
# standalone markers.
# ---------------------------------------------------------------------------


def _fake_charuco_detection() -> "CharucoBoardDetection":
    from app.setup.fiducial_markers import CharucoBoardDetection, CharucoCornerObs
    import numpy as _np
    return CharucoBoardDetection(corners=[
        CharucoCornerObs(corner_id=0, video_id="cam_B", frame_idx=0, px=1.0, py=1.0,
                          local_xyz=_np.zeros(3))
    ])


def test_aruco_marker_not_excluded_when_charuco_never_used(qapp, fake_conn) -> None:
    """A fresh dialog's ArUco and ChArUco panels default to the same
    dictionary -- that alone must not exclude anything, since the user may
    never touch the ChArUco panel at all."""
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        assert dlg._aruco_dict_combo.currentText() == dlg._charuco_dict_combo.currentText()
        assert not dlg._charuco_detections  # never used

        dlg._on_detect_aruco_clicked("cam_A")
        assert "3" in dlg._marker_groups
    finally:
        dlg.done(0)


def test_aruco_marker_excluded_once_matching_board_detected(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        # Same default dictionary on both panels, and evidence the board
        # has genuinely been used (a real detection recorded somewhere).
        dlg._charuco_detections["cam_B"] = _fake_charuco_detection()

        dlg._on_detect_aruco_clicked("cam_A")

        assert "3" not in dlg._marker_groups
        assert "belonging to the ChArUco board/rig excluded" in dlg._status_label.text()
    finally:
        dlg.done(0)


def test_aruco_marker_not_excluded_when_dictionaries_differ(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._charuco_detections["cam_B"] = _fake_charuco_detection()
        # Point the ChArUco panel at a different dictionary than ArUco's.
        other = next(
            i for i in range(dlg._charuco_dict_combo.count())
            if dlg._charuco_dict_combo.itemText(i) != dlg._aruco_dict_combo.currentText()
        )
        dlg._charuco_dict_combo.setCurrentIndex(other)

        dlg._on_detect_aruco_clicked("cam_A")

        assert "3" in dlg._marker_groups
    finally:
        dlg.done(0)

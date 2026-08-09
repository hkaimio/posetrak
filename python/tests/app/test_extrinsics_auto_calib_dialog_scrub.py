"""Tests for the per-camera VideoScrubBar wiring in ExtrinsicsAutoCalibDialog.

See docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md ("Frame source & scrubbing", Phase 1).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.setup.extrinsics_solver import CamCalibState
from app.setup.page_extrinsics import ExtrinsicsAutoCalibDialog


def _make_state(label: str, *, file_path: str | None) -> CamCalibState:
    K = np.eye(3)
    return CamCalibState(
        video_id=label,
        label=label,
        K=K,
        K_orig=K.copy(),
        dist=np.zeros((1, 4)),
        fisheye=False,
        file_path=file_path,
        first_frame=0,
        last_frame=99,
    )


@pytest.fixture()
def fake_conn(tmp_path):
    """A real (empty) session DB -- the dialog queries intrinsics_calibrations
    while building its per-camera panel even when no rows match."""
    from posetrak.db.db import create_session

    conn = create_session(tmp_path / "dialog_scrub_test.db")
    yield conn
    conn.close()


def test_video_backed_camera_gets_a_scrub_bar(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", file_path="/nonexistent/cam_A.mp4")]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        assert "cam_A" in dlg._scrub_bars
        assert dlg._scrub_bars["cam_A"].is_loaded
        assert dlg._scrub_bars["cam_A"].total_frames == 100  # last_frame - first_frame + 1
    finally:
        dlg.done(0)


def test_image_only_camera_gets_no_scrub_bar(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", file_path=None)]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        assert "cam_A" not in dlg._scrub_bars
        assert dlg._cam_panes["cam_A"] is dlg._cam_widgets["cam_A"]
    finally:
        dlg.done(0)


def test_scrub_frame_ready_updates_widget_and_state(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", file_path="/nonexistent/cam_A.mp4")]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        dlg._on_scrub_frame_ready("cam_A", dlg._cam_widgets["cam_A"], frame)
        assert dlg._states_by_id["cam_A"].image is frame
        assert dlg._cam_widgets["cam_A"]._img_bgr is frame
    finally:
        dlg.done(0)


def test_done_shuts_down_scrub_bars_without_crashing(qapp, fake_conn) -> None:
    states = [
        _make_state("cam_A", file_path="/nonexistent/cam_A.mp4"),
        _make_state("cam_B", file_path="/nonexistent/cam_B.mp4"),
    ]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    dlg.done(0)
    assert not any(sb.is_loaded for sb in dlg._scrub_bars.values())


def test_mixed_video_and_image_cameras(qapp, fake_conn) -> None:
    states = [
        _make_state("cam_A", file_path="/nonexistent/cam_A.mp4"),
        _make_state("cam_B", file_path=None),
    ]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        assert set(dlg._scrub_bars) == {"cam_A"}
        assert set(dlg._cam_panes) == {"cam_A", "cam_B"}
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# Phase 2 — per-control-point, per-frame observations
# ---------------------------------------------------------------------------


def test_current_frame_for_video_backed_camera_returns_scrub_position(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", file_path="/nonexistent/cam_A.mp4")]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._scrub_bars["cam_A"].seek(37)
        assert dlg._current_frame_for("cam_A") == 37
    finally:
        dlg.done(0)


def test_current_frame_for_image_only_camera_returns_zero(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", file_path=None)]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        assert dlg._current_frame_for("cam_A") == 0
    finally:
        dlg.done(0)


def test_on_cam_click_records_current_scrub_frame(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", file_path="/nonexistent/cam_A.mp4")]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._add_control_point()
        dlg._scrub_bars["cam_A"].seek(50)  # within [0, last_frame=99]
        dlg._on_cam_click("cam_A", 10.0, 20.0)

        cp = dlg._control_points[0]
        assert cp.obs["cam_A"].frame_idx == 50
        assert cp.obs["cam_A"].px == 10.0
        assert cp.obs["cam_A"].py == 20.0
    finally:
        dlg.done(0)


def test_on_cam_click_twice_at_different_frames_overwrites_frame_idx(qapp, fake_conn) -> None:
    """R4: placing the same point again on the same camera at a different scrub
    position overwrites that camera's ObsPoint (new frame, new pixel)."""
    states = [_make_state("cam_A", file_path="/nonexistent/cam_A.mp4")]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._add_control_point()
        dlg._scrub_bars["cam_A"].seek(20)
        dlg._on_cam_click("cam_A", 10.0, 20.0)

        dlg._scrub_bars["cam_A"].seek(80)
        dlg._on_cam_click("cam_A", 15.0, 25.0)

        cp = dlg._control_points[0]
        assert cp.obs["cam_A"].frame_idx == 80
        assert cp.obs["cam_A"].px == 15.0
        assert cp.obs["cam_A"].py == 25.0
    finally:
        dlg.done(0)


def test_on_cam_click_different_cameras_keep_independent_frames(qapp, fake_conn) -> None:
    states = [
        _make_state("cam_A", file_path="/nonexistent/cam_A.mp4"),
        _make_state("cam_B", file_path="/nonexistent/cam_B.mp4"),
    ]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._add_control_point()
        dlg._scrub_bars["cam_A"].seek(20)
        dlg._on_cam_click("cam_A", 1.0, 1.0)
        dlg._scrub_bars["cam_B"].seek(80)
        dlg._on_cam_click("cam_B", 2.0, 2.0)

        cp = dlg._control_points[0]
        assert cp.obs["cam_A"].frame_idx == 20
        assert cp.obs["cam_B"].frame_idx == 80
    finally:
        dlg.done(0)


def test_on_cam_click_image_only_camera_records_frame_zero(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", file_path=None)]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._add_control_point()
        dlg._on_cam_click("cam_A", 5.0, 6.0)
        assert dlg._control_points[0].obs["cam_A"].frame_idx == 0
    finally:
        dlg.done(0)

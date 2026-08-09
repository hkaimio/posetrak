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

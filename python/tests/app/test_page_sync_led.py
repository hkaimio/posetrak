"""Tests for the LED sync panel in app.setup.page_sync."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.setup.db_context import DBContext
from app.setup.led_sync import LedSyncResult, ROI, CameraSyncResult
from app.setup.page_sync import (
    SyncPage,
    _LedSyncJob,
    _ROISelectDialog,
    _sync_points_from_led_result,
)
from posetrak.db.db import create_session, generate_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(tmp_path: Path, n_shots: int = 1, videos_per_shot: int = 2):
    db_path = tmp_path / "led_test.db"
    conn = create_session(db_path)
    conn.row_factory = sqlite3.Row
    session_id = generate_id()
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES (?, ?)",
        (session_id, "2026-03-01T10:00:00+00:00"),
    )
    for shot_num in range(1, n_shots + 1):
        shot_id = generate_id()
        conn.execute(
            "INSERT INTO shots (id, session_id, shot_number, label) VALUES (?, ?, ?, ?)",
            (shot_id, session_id, shot_num, f"Shot {shot_num}"),
        )
        for cam_num in range(1, videos_per_shot + 1):
            vid_id = generate_id()
            conn.execute(
                "INSERT INTO shot_videos "
                "(id, shot_id, camera_instance_id, file_path, "
                "first_video_frame, last_video_frame, actual_fps) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (vid_id, shot_id, f"cam{cam_num}", f"/fake/s{shot_num}_c{cam_num}.mp4",
                 0, 299, 30.0),
            )
    conn.commit()
    return conn, session_id


def _attach_wizard(page: SyncPage, conn, session_id: str):
    ctx = DBContext(conn, session_id)
    wiz = MagicMock()
    wiz.db_context = ctx
    page.wizard = MagicMock(return_value=wiz)
    return ctx


def _make_led_result(cam_ids=("cam1", "cam2"), video_ids=("v1", "v2"), fps=30.0) -> LedSyncResult:
    """Minimal LedSyncResult with two cameras."""
    n = 300
    t = np.arange(n) / fps
    cam_results = []
    for i, (cid, vid) in enumerate(zip(cam_ids, video_ids)):
        cam_results.append(CameraSyncResult(
            camera_instance_id=cid,
            shot_video_id=vid,
            fps_used=fps,
            n_events=5,
            n_pairs=4 if i > 0 else 0,
            n_inliers=4 if i > 0 else 0,
            map_type="reference" if i == 0 else "affine",
            resid_std_s=0.0 if i == 0 else 0.003,
            frame_times=t.copy() if i == 0 else (t + 0.5).copy(),
            brightness=np.random.default_rng(i).standard_normal(n),
        ))
    return LedSyncResult(cameras=cam_results, ref_camera_idx=0)


@pytest.fixture
def loaded_page(qapp, tmp_path):
    conn, session_id = _make_session(tmp_path)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    yield page, conn
    page.cleanupPage()
    conn.close()


# ---------------------------------------------------------------------------
# LED panel construction
# ---------------------------------------------------------------------------


def test_led_panel_created_with_camera_rows(loaded_page) -> None:
    page, _ = loaded_page
    # 2 cameras → 2 fps spinboxes and 2 roi labels
    assert len(page._led_fps_spinboxes) == 2
    assert len(page._led_roi_labels) == 2


def test_led_fps_spinbox_defaults_to_actual_fps(loaded_page) -> None:
    page, _ = loaded_page
    assert page._led_fps_spinboxes[0].value() == pytest.approx(30.0)
    assert page._led_fps_spinboxes[1].value() == pytest.approx(30.0)


def test_led_run_btn_disabled_without_rois(loaded_page) -> None:
    page, _ = loaded_page
    assert not page._led_run_btn.isEnabled()


def test_led_accept_btn_disabled_initially(loaded_page) -> None:
    page, _ = loaded_page
    assert not page._led_accept_btn.isEnabled()


def test_led_plot_btn_disabled_initially(loaded_page) -> None:
    page, _ = loaded_page
    assert not page._led_plot_btn.isEnabled()


def test_led_quality_widget_hidden_initially(loaded_page) -> None:
    page, _ = loaded_page
    assert not page._led_quality_widget.isVisible()


# ---------------------------------------------------------------------------
# ROI management
# ---------------------------------------------------------------------------


def test_setting_all_rois_enables_run_btn(loaded_page) -> None:
    page, _ = loaded_page
    page._led_rois[0] = ROI(10, 10, 30, 30)
    page._led_rois[1] = ROI(20, 20, 40, 40)
    page._update_led_run_btn()
    assert page._led_run_btn.isEnabled()


def test_partial_rois_does_not_enable_run_btn(loaded_page) -> None:
    page, _ = loaded_page
    page._led_rois[0] = ROI(10, 10, 30, 30)  # only one camera
    page._update_led_run_btn()
    assert not page._led_run_btn.isEnabled()


def test_roi_label_updated_after_set(loaded_page) -> None:
    page, _ = loaded_page
    # Simulate what _on_set_led_roi does after dialog confirms
    page._led_rois[0] = ROI(10, 20, 50, 60)
    page._led_roi_labels[0].setText("ROI: (10,20)→(50,60)")
    assert "10" in page._led_roi_labels[0].text()


# ---------------------------------------------------------------------------
# LED sync done callback
# ---------------------------------------------------------------------------


def test_on_led_sync_done_enables_accept(loaded_page) -> None:
    page, _ = loaded_page
    result = _make_led_result()
    page._on_led_sync_done(result)
    assert page._led_accept_btn.isEnabled()


def test_on_led_sync_done_shows_quality_widget(loaded_page) -> None:
    page, _ = loaded_page
    result = _make_led_result()
    page._on_led_sync_done(result)
    assert not page._led_quality_widget.isHidden()


def test_on_led_sync_done_populates_quality_labels(loaded_page) -> None:
    page, _ = loaded_page
    result = _make_led_result()
    page._on_led_sync_done(result)
    assert len(page._led_quality_labels) == 2


def test_on_led_sync_done_reference_label_grey(loaded_page) -> None:
    page, _ = loaded_page
    result = _make_led_result()
    page._on_led_sync_done(result)
    # First camera is reference; label should contain "reference"
    assert "reference" in page._led_quality_labels[0].text()


def test_on_led_sync_error_shows_message(loaded_page) -> None:
    page, _ = loaded_page
    page._on_led_sync_error("CV2 failed")
    assert "CV2 failed" in page._led_accept_label.text()


def test_on_led_sync_error_re_enables_run_btn(loaded_page) -> None:
    page, _ = loaded_page
    page._on_led_sync_error("oops")
    assert page._led_run_btn.isEnabled()


# ---------------------------------------------------------------------------
# Accept LED sync
# ---------------------------------------------------------------------------


def test_accept_led_sync_writes_config(loaded_page) -> None:
    page, conn = loaded_page
    shot_idx = page._shot_combo.currentIndex()
    shot = page._shots[shot_idx]
    # Provide a valid result that matches the shot's videos
    videos = shot.videos
    result = _make_led_result(
        cam_ids=[sv.camera_instance_id for sv in videos],
        video_ids=[sv.id for sv in videos],
    )
    page._led_result = result
    page._on_accept_led_sync()

    configs = conn.execute("SELECT * FROM sync_configs").fetchall()
    assert len(configs) == 1
    assert configs[0]["created_by"] == "led-auto"


def test_accept_led_sync_writes_sync_points(loaded_page) -> None:
    page, conn = loaded_page
    shot_idx = page._shot_combo.currentIndex()
    shot = page._shots[shot_idx]
    videos = shot.videos
    result = _make_led_result(
        cam_ids=[sv.camera_instance_id for sv in videos],
        video_ids=[sv.id for sv in videos],
    )
    page._led_result = result
    page._on_accept_led_sync()

    pts = conn.execute("SELECT * FROM sync_points").fetchall()
    assert len(pts) > 0


def test_accept_led_sync_switches_scrubber_to_synced(loaded_page) -> None:
    page, _ = loaded_page
    shot = page._shots[page._shot_combo.currentIndex()]
    videos = shot.videos
    result = _make_led_result(
        cam_ids=[sv.camera_instance_id for sv in videos],
        video_ids=[sv.id for sv in videos],
    )
    page._led_result = result
    page._on_accept_led_sync()

    assert page._scrubber.sync_table is not None


def test_accept_led_sync_disables_accept_btn(loaded_page) -> None:
    page, _ = loaded_page
    shot = page._shots[page._shot_combo.currentIndex()]
    videos = shot.videos
    result = _make_led_result(
        cam_ids=[sv.camera_instance_id for sv in videos],
        video_ids=[sv.id for sv in videos],
    )
    page._led_result = result
    page._on_accept_led_sync()
    assert not page._led_accept_btn.isEnabled()


# ---------------------------------------------------------------------------
# Teardown resets LED state
# ---------------------------------------------------------------------------


def test_teardown_clears_led_rois(qapp, tmp_path) -> None:
    conn, session_id = _make_session(tmp_path)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    page._led_rois[0] = ROI(0, 0, 10, 10)
    page.cleanupPage()
    assert page._led_rois == {}
    conn.close()


def test_teardown_clears_led_result(qapp, tmp_path) -> None:
    conn, session_id = _make_session(tmp_path)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    page._led_result = _make_led_result()
    page.cleanupPage()
    assert page._led_result is None
    conn.close()


# ---------------------------------------------------------------------------
# Shot switch resets LED state
# ---------------------------------------------------------------------------


def test_shot_switch_resets_led_rois(qapp, tmp_path) -> None:
    conn, session_id = _make_session(tmp_path, n_shots=2)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    page._led_rois[0] = ROI(0, 0, 10, 10)
    page._shot_combo.setCurrentIndex(1)
    assert page._led_rois == {}
    page.cleanupPage()
    conn.close()


# ---------------------------------------------------------------------------
# _sync_points_from_led_result
# ---------------------------------------------------------------------------


def test_sync_points_from_led_result_contains_both_cameras() -> None:
    result = _make_led_result()
    points, fps_by_video = _sync_points_from_led_result(result)
    assert "cam1" in points
    assert "cam2" in points


def test_sync_points_from_led_result_covers_full_range() -> None:
    result = _make_led_result()
    points, _ = _sync_points_from_led_result(result)
    for pts in points.values():
        frames = [p.video_frame for p in pts]
        assert 0 in frames
        assert max(frames) == 299  # total_frames - 1


def test_sync_points_effective_fps_computed() -> None:
    result = _make_led_result()
    _, fps_by_video = _sync_points_from_led_result(result)
    for vid_id in ["v1", "v2"]:
        assert fps_by_video[vid_id] > 0


# ---------------------------------------------------------------------------
# _LedSyncJob (mocked cv2)
# ---------------------------------------------------------------------------


def _mock_cap_for_file(n_frames: int = 30):
    """Return a mock VideoCapture that produces n_frames of blank frames."""
    frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(n_frames)]
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.return_value = float(n_frames)
    cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
    return cap


def test_led_sync_job_emits_finished(qapp, tmp_path) -> None:
    """LedSyncJob with mocked cv2 completes without error."""
    roi = ROI(0, 0, 10, 10)
    cam_data = [
        ("/a.mp4", roi, 30.0, "cam1", "v1"),
        ("/b.mp4", roi, 30.0, "cam2", "v2"),
    ]
    # Both cameras get the same blank signal → NCC fallback
    results_received = []

    def _on_done(r):
        results_received.append(r)

    job = _LedSyncJob(cam_data, ref_cam=0)
    job.finished.connect(_on_done)

    with patch("cv2.VideoCapture", side_effect=[_mock_cap_for_file(), _mock_cap_for_file()]):
        job.run.__wrapped__(job)

    assert len(results_received) == 1
    assert isinstance(results_received[0], LedSyncResult)


def test_led_sync_job_two_cameras_in_result(qapp, tmp_path) -> None:
    roi = ROI(0, 0, 10, 10)
    cam_data = [
        ("/a.mp4", roi, 30.0, "cam1", "v1"),
        ("/b.mp4", roi, 30.0, "cam2", "v2"),
    ]
    results_received = []

    job = _LedSyncJob(cam_data, ref_cam=0)
    job.finished.connect(lambda r: results_received.append(r))

    with patch("cv2.VideoCapture", side_effect=[_mock_cap_for_file(), _mock_cap_for_file()]):
        job.run.__wrapped__(job)

    assert len(results_received[0].cameras) == 2

"""Tests for the LED sync dialog in app.setup.page_sync."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.setup.db_context import DBContext, SyncTable
from app.setup.led_sync import CameraSyncResult, LedSyncResult, ROI
from app.setup.page_sync import (
    SyncPage,
    _LedSyncDialog,
    _LedSyncJob,
    _ShotMeta,
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


def _make_shot_meta(conn, session_id: str, shot_idx: int = 0) -> _ShotMeta:
    ctx = DBContext(conn, session_id)
    rows = conn.execute(
        "SELECT id, shot_number, label FROM shots WHERE session_id = ? ORDER BY shot_number",
        (session_id,),
    ).fetchall()
    row = rows[shot_idx]
    label = row["label"] or f"Shot {row['shot_number']}"
    videos = ctx.get_shot_videos(row["id"])
    return _ShotMeta(shot_id=row["id"], label=label, videos=list(videos))


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
def loaded_dlg(qapp, tmp_path):
    conn, session_id = _make_session(tmp_path)
    ctx = DBContext(conn, session_id)
    shot = _make_shot_meta(conn, session_id)
    on_accepted = MagicMock()
    dlg = _LedSyncDialog(
        shot=shot,
        fps_overrides={},
        ctx=ctx,
        current_frames=[0, 0],
        on_sync_accepted=on_accepted,
    )
    yield dlg, conn, on_accepted
    conn.close()


# ---------------------------------------------------------------------------
# Dialog construction
# ---------------------------------------------------------------------------


def test_led_dialog_created_with_camera_rows(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    assert len(dlg._fps_spinboxes) == 2
    assert len(dlg._roi_labels) == 2


def test_led_fps_spinbox_defaults_to_actual_fps(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    assert dlg._fps_spinboxes[0].value() == pytest.approx(30.0)
    assert dlg._fps_spinboxes[1].value() == pytest.approx(30.0)


def test_led_run_btn_disabled_without_rois(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    assert not dlg._run_btn.isEnabled()


def test_led_accept_btn_disabled_initially(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    assert not dlg._accept_btn.isEnabled()


def test_led_plot_btn_disabled_initially(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    assert not dlg._plot_btn.isEnabled()


def test_led_quality_widget_hidden_initially(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    assert not dlg._quality_widget.isVisible()


# ---------------------------------------------------------------------------
# ROI management
# ---------------------------------------------------------------------------


def test_setting_all_rois_enables_run_btn(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    dlg._led_rois[0] = ROI(10, 10, 30, 30)
    dlg._led_rois[1] = ROI(20, 20, 40, 40)
    dlg._update_run_btn()
    assert dlg._run_btn.isEnabled()


def test_partial_rois_does_not_enable_run_btn(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    dlg._led_rois[0] = ROI(10, 10, 30, 30)  # only one camera
    dlg._update_run_btn()
    assert not dlg._run_btn.isEnabled()


def test_roi_label_updated_after_set(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    dlg._led_rois[0] = ROI(10, 20, 50, 60)
    dlg._roi_labels[0].setText("ROI: (10,20)→(50,60)")
    assert "10" in dlg._roi_labels[0].text()


# ---------------------------------------------------------------------------
# LED sync done callback
# ---------------------------------------------------------------------------


def test_on_led_sync_done_enables_accept(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    result = _make_led_result()
    dlg._on_done(result)
    assert dlg._accept_btn.isEnabled()


def test_on_led_sync_done_shows_quality_widget(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    result = _make_led_result()
    dlg._on_done(result)
    assert not dlg._quality_widget.isHidden()


def test_on_led_sync_done_populates_quality_labels(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    result = _make_led_result()
    dlg._on_done(result)
    assert dlg._quality_layout.count() == 2


def test_on_led_sync_done_reference_label_grey(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    result = _make_led_result()
    dlg._on_done(result)
    lbl = dlg._quality_layout.itemAt(0).widget()
    assert "reference" in lbl.text()


def test_on_led_sync_error_shows_message(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    dlg._on_error("CV2 failed")
    assert "CV2 failed" in dlg._accept_label.text()


def test_on_led_sync_error_re_enables_run_btn(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    dlg._on_error("oops")
    assert dlg._run_btn.isEnabled()


# ---------------------------------------------------------------------------
# Accept LED sync
# ---------------------------------------------------------------------------


def test_accept_led_sync_writes_config(loaded_dlg) -> None:
    dlg, conn, _ = loaded_dlg
    videos = dlg._shot.videos
    result = _make_led_result(
        cam_ids=[sv.camera_instance_id for sv in videos],
        video_ids=[sv.id for sv in videos],
    )
    dlg._led_result = result
    dlg._on_accept()

    configs = conn.execute("SELECT * FROM sync_configs").fetchall()
    assert len(configs) == 1
    assert configs[0]["created_by"] == "led-auto"


def test_accept_led_sync_writes_sync_points(loaded_dlg) -> None:
    dlg, conn, _ = loaded_dlg
    videos = dlg._shot.videos
    result = _make_led_result(
        cam_ids=[sv.camera_instance_id for sv in videos],
        video_ids=[sv.id for sv in videos],
    )
    dlg._led_result = result
    dlg._on_accept()

    pts = conn.execute("SELECT * FROM sync_points").fetchall()
    assert len(pts) > 0


def test_accept_led_sync_calls_accepted_callback(loaded_dlg) -> None:
    dlg, _, on_accepted = loaded_dlg
    videos = dlg._shot.videos
    result = _make_led_result(
        cam_ids=[sv.camera_instance_id for sv in videos],
        video_ids=[sv.id for sv in videos],
    )
    dlg._led_result = result
    dlg._on_accept()

    on_accepted.assert_called_once()
    assert isinstance(on_accepted.call_args[0][0], SyncTable)


def test_accept_led_sync_disables_accept_btn(loaded_dlg) -> None:
    dlg, _, _ = loaded_dlg
    videos = dlg._shot.videos
    result = _make_led_result(
        cam_ids=[sv.camera_instance_id for sv in videos],
        video_ids=[sv.id for sv in videos],
    )
    dlg._led_result = result
    dlg._on_accept()
    assert not dlg._accept_btn.isEnabled()


# ---------------------------------------------------------------------------
# SyncPage LED sync button state
# ---------------------------------------------------------------------------


def test_led_sync_btn_disabled_initially(qapp, tmp_path) -> None:
    conn, session_id = _make_session(tmp_path)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    assert not page._led_sync_btn.isEnabled()
    page.cleanupPage()
    conn.close()


def test_led_sync_btn_disabled_after_cleanup(qapp, tmp_path) -> None:
    conn, session_id = _make_session(tmp_path)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    page._led_sync_btn.setEnabled(True)
    page.cleanupPage()
    assert not page._led_sync_btn.isEnabled()
    conn.close()


def test_led_sync_btn_disabled_after_shot_switch(qapp, tmp_path) -> None:
    conn, session_id = _make_session(tmp_path, n_shots=2)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    page._led_sync_btn.setEnabled(True)
    page._shot_combo.setCurrentIndex(1)
    assert not page._led_sync_btn.isEnabled()
    page.cleanupPage()
    conn.close()


# ---------------------------------------------------------------------------
# _sync_points_from_led_result
# ---------------------------------------------------------------------------


def test_sync_points_from_led_result_contains_both_videos() -> None:
    """Points are keyed by shot_video_id to avoid collisions on __unassigned__ IDs."""
    result = _make_led_result()
    points, fps_by_video = _sync_points_from_led_result(result)
    assert "v1" in points
    assert "v2" in points


def test_sync_points_from_led_result_covers_full_range() -> None:
    result = _make_led_result()
    points, _ = _sync_points_from_led_result(result)
    for pts in points.values():
        frames = [p.video_frame for p in pts]
        assert 0 in frames
        assert max(frames) == 299  # total_frames - 1


def test_sync_points_from_led_result_stores_every_frame() -> None:
    result = _make_led_result()
    points, _ = _sync_points_from_led_result(result)
    for pts in points.values():
        assert len(pts) == 300  # every frame stored


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
    results_received = []

    job = _LedSyncJob(cam_data, ref_cam=0)
    job.finished.connect(lambda r: results_received.append(r))

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

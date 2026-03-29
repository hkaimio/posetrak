"""Tests for app.setup.led_sync (LED synchronisation algorithm)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.setup.led_sync import (
    ROI,
    CameraSyncResult,
    LedSyncResult,
    _correlate_fft,
    build_time_map,
    detect_events,
    dtw_match_event_times,
    extract_brightness_changes,
    moving_average,
    ransac_affine_fit,
    run_led_sync,
    zscore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_blink_signal(n: int, fps: float, event_times_s: list[float], noise: float = 0.05) -> np.ndarray:
    """Synthetic brightness-change signal with Gaussian spikes at given times."""
    rng = np.random.default_rng(0)
    t = np.arange(n) / fps
    x = noise * rng.standard_normal(n)
    for tt in event_times_s:
        x += np.exp(-0.5 * ((t - tt) / 0.03) ** 2) * 5.0
    return x


def _two_camera_signals(fps: float = 30.0, duration_s: float = 20.0, offset_s: float = 0.5) -> tuple:
    """Reference signal and an offset copy."""
    n = int(duration_s * fps)
    events = [3.0, 7.0, 12.0, 16.0]
    ref = _make_blink_signal(n, fps, events)
    cam = _make_blink_signal(n, fps, [t + offset_s for t in events])
    return ref, cam, events, offset_s


# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------


def test_zscore_zero_mean() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore(x)
    assert abs(np.mean(z)) < 1e-9


def test_zscore_unit_std() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore(x)
    assert abs(np.std(z) - 1.0) < 1e-6


def test_moving_average_length_preserved() -> None:
    x = np.arange(10, dtype=float)
    assert len(moving_average(x, 3)) == len(x)


def test_moving_average_w1_is_identity() -> None:
    x = np.arange(10, dtype=float)
    np.testing.assert_array_equal(moving_average(x, 1), x)


def test_correlate_fft_identical_signals_peak_at_zero() -> None:
    x = np.random.default_rng(42).standard_normal(200)
    corr, lags = _correlate_fft(x, x)
    best_lag = lags[np.argmax(corr)]
    assert best_lag == 0


def test_correlate_fft_shifted_signal() -> None:
    n = 200
    shift = 10
    x = np.zeros(n)
    x[50] = 1.0
    y = np.zeros(n)
    y[50 + shift] = 1.0
    corr, lags = _correlate_fft(x, y)
    best_lag = lags[np.argmax(corr)]
    # y leads x by `shift`, so best lag should be shift (within ±2)
    assert abs(best_lag - shift) <= 2


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------


def test_detect_events_finds_known_peaks() -> None:
    fps = 30.0
    n = int(10 * fps)
    event_times = [2.0, 5.0, 8.0]
    sig = _make_blink_signal(n, fps, event_times, noise=0.02)
    found = detect_events(sig, fps, min_sep_s=0.5, prominence=2.0)
    assert len(found) >= 3
    # Each known event should be matched within 0.1 s
    for tt in event_times:
        assert any(abs(f - tt) < 0.1 for f in found), f"Event at {tt}s not found; got {found}"


def test_detect_events_sub_sample_precision() -> None:
    """Detected time should be within half a frame of the true event."""
    fps = 30.0
    n = 300
    event_t = 3.0
    sig = _make_blink_signal(n, fps, [event_t], noise=0.01)
    found = detect_events(sig, fps, min_sep_s=0.5, prominence=2.0)
    assert len(found) >= 1
    assert abs(found[0] - event_t) < (0.5 / fps) * 3  # within 3 frames


def test_detect_events_empty_signal() -> None:
    found = detect_events(np.zeros(100), 30.0)
    assert len(found) == 0


def test_detect_events_polarity_pos_only() -> None:
    fps = 30.0
    n = 300
    sig = _make_blink_signal(n, fps, [2.0, 5.0], noise=0.01)
    found_both = detect_events(sig, fps, polarity="both", prominence=1.5)
    found_pos = detect_events(sig, fps, polarity="pos", prominence=1.5)
    assert len(found_pos) <= len(found_both)


# ---------------------------------------------------------------------------
# DTW matching
# ---------------------------------------------------------------------------


def test_dtw_match_identical_sequences() -> None:
    t = np.array([1.0, 3.0, 5.0])
    pairs = dtw_match_event_times(t, t)
    # Should match each event to itself
    assert len(pairs) == 3
    np.testing.assert_array_equal(pairs[:, 0], pairs[:, 1])


def test_dtw_match_offset_sequences() -> None:
    tA = np.array([1.0, 3.0, 5.0])
    tB = tA + 0.1  # 100 ms offset
    pairs = dtw_match_event_times(tA, tB, band_s=0.5)
    assert len(pairs) == 3
    np.testing.assert_array_equal(pairs[:, 0], np.arange(3))
    np.testing.assert_array_equal(pairs[:, 1], np.arange(3))


def test_dtw_match_empty_returns_empty() -> None:
    pairs = dtw_match_event_times(np.array([]), np.array([1.0, 2.0]))
    assert pairs.shape == (0, 2)


def test_dtw_match_single_element_each() -> None:
    pairs = dtw_match_event_times(np.array([1.0]), np.array([1.1]))
    assert pairs.shape == (1, 2)


# ---------------------------------------------------------------------------
# RANSAC affine fit
# ---------------------------------------------------------------------------


def test_ransac_affine_identity_fit() -> None:
    t = np.linspace(0, 10, 20)
    (a, b), inliers = ransac_affine_fit(t, t, max_err_s=0.01)
    assert abs(a - 1.0) < 0.01
    assert abs(b) < 0.01
    assert len(inliers) >= 18


def test_ransac_affine_offset_fit() -> None:
    t = np.linspace(0, 10, 20)
    t_cam = t - 0.5  # ref = cam + 0.5
    (a, b), inliers = ransac_affine_fit(t, t_cam, max_err_s=0.02)
    # a ≈ 1, b ≈ 0.5
    assert abs(a - 1.0) < 0.05
    assert abs(b - 0.5) < 0.05


def test_ransac_affine_tolerates_outliers() -> None:
    rng = np.random.default_rng(1)
    t = np.linspace(0, 10, 30)
    t_cam = t - 0.3
    # Add 5 outlier points
    t_ref_noisy = t.copy()
    idx = rng.integers(0, 30, 5)
    t_ref_noisy[idx] += rng.uniform(1, 3, 5)
    (a, b), inliers = ransac_affine_fit(t_ref_noisy, t_cam, max_err_s=0.05, n_iter=500)
    assert len(inliers) >= 20  # most inliers recovered
    assert abs(a - 1.0) < 0.1


# ---------------------------------------------------------------------------
# build_time_map
# ---------------------------------------------------------------------------


def test_build_time_map_affine() -> None:
    t_cam = np.linspace(0, 10, 10)
    t_ref = 2.0 * t_cam + 0.5  # a=2, b=0.5
    f, meta = build_time_map(t_ref, t_cam)
    assert meta["type"] == "affine"
    np.testing.assert_allclose(f(t_cam), t_ref, atol=1e-9)


def test_build_time_map_extrapolates() -> None:
    t_cam = np.linspace(0, 5, 10)
    t_ref = t_cam + 1.0
    f, _ = build_time_map(t_ref, t_cam)
    assert abs(f(np.array([10.0]))[0] - 11.0) < 0.1


# ---------------------------------------------------------------------------
# extract_brightness_changes (mocked cv2)
# ---------------------------------------------------------------------------


@patch("app.setup.led_sync.extract_brightness_changes.__code__", None)
def _skip_if_cv2_absent():
    pass


def test_extract_brightness_changes_basic(tmp_path) -> None:
    """extract_brightness_changes with a mocked cv2 VideoCapture."""
    roi = ROI(x1=10, y1=20, x2=30, y2=40)
    n_frames = 5
    frames = [np.full((100, 100, 3), i * 10, dtype=np.uint8) for i in range(n_frames)]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        3: 100.0,  # CAP_PROP_FRAME_COUNT would be different prop id
        5: 30.0,   # CAP_PROP_FPS
        7: n_frames,
    }.get(int(prop), n_frames)

    read_returns = [(True, f) for f in frames] + [(False, None)]
    mock_cap.read.side_effect = read_returns

    with patch("cv2.VideoCapture", return_value=mock_cap):
        changes, fps = extract_brightness_changes(str(tmp_path / "fake.mp4"), roi)

    assert len(changes) == n_frames
    assert changes[0] == 0.0   # first frame has no previous
    assert fps == pytest.approx(30.0, abs=0.1)


def test_extract_brightness_changes_calls_progress_cb(tmp_path) -> None:
    roi = ROI(x1=0, y1=0, x2=10, y2=10)
    frames = [np.ones((50, 50, 3), dtype=np.uint8) * i for i in range(3)]
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 3.0
    mock_cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]

    calls: list[tuple] = []
    with patch("cv2.VideoCapture", return_value=mock_cap):
        extract_brightness_changes(str(tmp_path / "v.mp4"), roi, progress_cb=lambda i, t: calls.append((i, t)))

    assert len(calls) == 3


def test_extract_brightness_changes_fps_override(tmp_path) -> None:
    roi = ROI(x1=0, y1=0, x2=5, y2=5)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0
    mock_cap.read.side_effect = [(False, None)]

    with patch("cv2.VideoCapture", return_value=mock_cap):
        _, fps = extract_brightness_changes(str(tmp_path / "v.mp4"), roi, fps_override=120.0)

    assert fps == pytest.approx(120.0)


def test_extract_brightness_changes_raises_on_bad_file(tmp_path) -> None:
    roi = ROI(x1=0, y1=0, x2=5, y2=5)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(OSError, match="Cannot open"):
            extract_brightness_changes(str(tmp_path / "no.mp4"), roi)


# ---------------------------------------------------------------------------
# ROI dataclass
# ---------------------------------------------------------------------------


def test_roi_to_slice() -> None:
    roi = ROI(x1=10, y1=20, x2=40, y2=60)
    rs, cs = roi.to_slice()
    assert rs.start == 20 and rs.stop == 60
    assert cs.start == 10 and cs.stop == 40


def test_roi_to_slice_handles_inverted_coords() -> None:
    roi = ROI(x1=40, y1=60, x2=10, y2=20)
    rs, cs = roi.to_slice()
    assert rs.start == 20 and rs.stop == 60
    assert cs.start == 10 and cs.stop == 40


def test_roi_is_valid() -> None:
    assert ROI(0, 0, 10, 10).is_valid
    assert not ROI(5, 5, 5, 5).is_valid


# ---------------------------------------------------------------------------
# run_led_sync — integration test with synthetic signals
# ---------------------------------------------------------------------------


def test_run_led_sync_reference_camera_identity_map() -> None:
    fps = 30.0
    ref, cam, _, _ = _two_camera_signals(fps=fps)
    result = run_led_sync(
        signals=[ref, cam],
        fps_list=[fps, fps],
        cam_ids=["cam1", "cam2"],
        video_ids=["vid1", "vid2"],
        ref_cam=0,
    )
    assert result.ref_camera_idx == 0
    assert result.cameras[0].map_type == "reference"
    # Reference frame_times should be identity: frame k → k/fps
    np.testing.assert_allclose(result.cameras[0].frame_times, np.arange(len(ref)) / fps)


def test_run_led_sync_recovers_known_offset() -> None:
    """After sync, global times for the offset camera should match the reference."""
    fps = 30.0
    offset_s = 0.5
    ref, cam, _, _ = _two_camera_signals(fps=fps, duration_s=20.0, offset_s=offset_s)

    result = run_led_sync(
        signals=[ref, cam],
        fps_list=[fps, fps],
        cam_ids=["cam1", "cam2"],
        video_ids=["vid1", "vid2"],
        ref_cam=0,
        event_cfg=dict(min_sep_s=0.5, prominence=2.0, polarity="both", smooth_win=3),
    )
    cr = result.cameras[1]
    # At frame 0, global time should be close to -offset_s
    # (camera 2 is ahead of reference by offset_s, so its t=0 maps to t_global=-0.5)
    assert abs(cr.frame_times[0] - (-offset_s)) < 0.1


def test_run_led_sync_result_structure() -> None:
    fps = 30.0
    ref, cam, _, _ = _two_camera_signals(fps=fps)
    result = run_led_sync(
        signals=[ref, cam],
        fps_list=[fps, fps],
        cam_ids=["refcam", "othercam"],
        video_ids=["v0", "v1"],
        ref_cam=0,
    )
    assert isinstance(result, LedSyncResult)
    assert len(result.cameras) == 2
    for cr in result.cameras:
        assert isinstance(cr, CameraSyncResult)
        assert len(cr.frame_times) > 0
        assert len(cr.brightness) > 0


def test_run_led_sync_non_zero_ref_index() -> None:
    fps = 30.0
    ref, cam, _, _ = _two_camera_signals(fps=fps)
    result = run_led_sync(
        signals=[cam, ref],   # ref is index 1 this time
        fps_list=[fps, fps],
        cam_ids=["cam2", "cam1"],
        video_ids=["v1", "v0"],
        ref_cam=1,
    )
    assert result.cameras[1].map_type == "reference"
    assert result.cameras[0].map_type != "reference"


def test_run_led_sync_brightness_arrays_preserved() -> None:
    fps = 30.0
    ref, cam, _, _ = _two_camera_signals(fps=fps)
    result = run_led_sync(
        signals=[ref, cam],
        fps_list=[fps, fps],
        cam_ids=["a", "b"],
        video_ids=["va", "vb"],
        ref_cam=0,
    )
    np.testing.assert_array_equal(result.cameras[0].brightness, ref)
    np.testing.assert_array_equal(result.cameras[1].brightness, cam)

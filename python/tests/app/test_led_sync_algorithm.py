"""Regression tests for the LED sync algorithm correctness.

These tests validate that run_led_sync recovers the correct per-camera time
offset, including cases where cameras were started many seconds apart — the
scenario that caused incorrect sync when the unconstrained DTW fallback was
used without a rough-sync seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.setup.led_sync import run_led_sync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_led_signal(n_frames: int, blink_frames: list[int], rng: np.random.Generator) -> np.ndarray:
    """Synthetic brightness-change signal with LED blinks at given frame indices."""
    sig = rng.standard_normal(n_frames) * 0.05
    for f in blink_frames:
        if 0 <= f < n_frames:
            sig[f] += 4.0   # bright flash
        if 0 <= f + 3 < n_frames:
            sig[f + 3] -= 4.0  # LED off
    return sig


def _build_two_camera_signals(
    fps: float,
    offset_s: float,
    n_ref: int,
    n_cam: int,
    blink_interval_s: float = 1.0,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create (sig_ref, sig_cam) with cam starting *offset_s* seconds after ref.

    Blinks occur every *blink_interval_s* seconds in global time, so both
    cameras see the same LED events.
    """
    rng = np.random.default_rng(rng_seed)
    duration_ref = n_ref / fps
    global_blinks = np.arange(blink_interval_s, duration_ref, blink_interval_s)

    ref_blinks = [int(t * fps) for t in global_blinks if int(t * fps) < n_ref]
    cam_blinks = [
        int((t - offset_s) * fps)
        for t in global_blinks
        if 0 <= int((t - offset_s) * fps) < n_cam
    ]
    sig_ref = _make_led_signal(n_ref, ref_blinks, rng)
    sig_cam = _make_led_signal(n_cam, cam_blinks, rng)
    return sig_ref, sig_cam


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sync_zero_offset() -> None:
    """Cameras started simultaneously should give offset ≈ 0."""
    fps = 60.0
    sig_ref, sig_cam = _build_two_camera_signals(fps, offset_s=0.0, n_ref=1800, n_cam=1800)

    result = run_led_sync(
        signals=[sig_ref, sig_cam],
        fps_list=[fps, fps],
        cam_ids=["ref", "cam"],
        video_ids=["v_ref", "v_cam"],
        ref_cam=0,
        rough_offsets=[0.0, 0.0],
    )

    offset = result.cameras[1].frame_times[0]
    assert abs(offset) < 0.2, f"Expected near-zero offset, got {offset:.3f} s"


@pytest.mark.parametrize("offset_s", [5.0, 16.0, 30.0])
def test_large_offset_recovered_with_rough_hint(offset_s: float) -> None:
    """LED sync must recover large per-camera start offsets when rough_offsets is provided."""
    fps = 60.0
    n_ref = int(fps * 60)   # 60 s reference
    n_cam = int(fps * 30)   # 30 s camera clip (started offset_s into global time)

    sig_ref, sig_cam = _build_two_camera_signals(fps, offset_s=offset_s, n_ref=n_ref, n_cam=n_cam)

    result = run_led_sync(
        signals=[sig_ref, sig_cam],
        fps_list=[fps, fps],
        cam_ids=["ref", "cam"],
        video_ids=["v_ref", "v_cam"],
        ref_cam=0,
        rough_offsets=[0.0, offset_s],
    )

    estimated_offset = result.cameras[1].frame_times[0]
    assert abs(estimated_offset - offset_s) < 0.5, (
        f"Expected offset ~{offset_s} s, got {estimated_offset:.3f} s"
    )


@pytest.mark.parametrize("offset_s", [16.0, 30.0])
def test_large_offset_wrong_without_rough_hint(offset_s: float) -> None:
    """Without a rough offset hint, unconstrained DTW gives the wrong answer for large delays.

    This test documents the original bug: unconstrained DTW matches the
    beginning of both event sequences, yielding b≈0 instead of b≈offset_s.
    """
    fps = 60.0
    n_ref = int(fps * 60)
    n_cam = int(fps * 30)

    sig_ref, sig_cam = _build_two_camera_signals(fps, offset_s=offset_s, n_ref=n_ref, n_cam=n_cam)

    result = run_led_sync(
        signals=[sig_ref, sig_cam],
        fps_list=[fps, fps],
        cam_ids=["ref", "cam"],
        video_ids=["v_ref", "v_cam"],
        ref_cam=0,
        # no rough_offsets → unconstrained DTW fallback
    )

    estimated_offset = result.cameras[1].frame_times[0]
    # Without the hint the algorithm gets the wrong answer
    assert abs(estimated_offset - offset_s) > 2.0, (
        f"Expected wrong answer without rough hint for offset={offset_s} s, "
        f"but got estimated_offset={estimated_offset:.3f} s which is suspiciously correct"
    )

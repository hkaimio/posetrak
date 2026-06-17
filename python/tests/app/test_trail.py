"""Tests for compute_trail() — pure-function tests, no Qt or DB required."""
from __future__ import annotations

import numpy as np
import pytest

from app.pose.crop_editor import (
    _CONF_THRESHOLD,
    _TRAIL_N,
    _FrameSlot,
    _TrailData,
    _TrailPoint,
    compute_trail,
)

N_KP = 3
CAM = "ci1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frames(n: int) -> list[_FrameSlot]:
    """n slots; each maps CAM → video_frame equal to the slot index."""
    return [_FrameSlot(timestamp_s=float(i), per_cam={CAM: i}) for i in range(n)]


def _kp_dict(
    frames: list[_FrameSlot],
    x_fn=lambda vf: float(vf * 10),
    y_fn=lambda vf: float(vf * 5),
    conf: float = 0.9,
    include: set[int] | None = None,  # restrict to these video_frame values
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for fs in frames:
        vf = fs.per_cam.get(CAM)
        if vf is None:
            continue
        if include is not None and vf not in include:
            continue
        kp = np.zeros((N_KP, 3), dtype=np.float32)
        kp[:, 0] = x_fn(vf)
        kp[:, 1] = y_fn(vf)
        kp[:, 2] = conf
        result[vf] = kp
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_trail_empty_for_single_frame():
    frames = _frames(1)
    kp = _kp_dict(frames)
    trail = compute_trail(frames, kp, CAM, 0, 0)
    assert trail.kp_idx == 0
    assert trail.past == []
    assert trail.future == []


def test_trail_past_contains_real_points():
    frames = _frames(6)
    kp = _kp_dict(frames)
    trail = compute_trail(frames, kp, CAM, 4, 0, n=10)
    # Slots 0,1,2,3 are in the past window
    assert len(trail.past) == 4
    assert all(not p.is_ghost for p in trail.past)


def test_trail_future_contains_real_points():
    frames = _frames(6)
    kp = _kp_dict(frames)
    trail = compute_trail(frames, kp, CAM, 1, 0, n=10)
    # Slots 2,3,4,5 are in the future window
    assert len(trail.future) == 4
    assert all(not p.is_ghost for p in trail.future)


def test_trail_capped_at_n():
    frames = _frames(30)
    kp = _kp_dict(frames)
    trail = compute_trail(frames, kp, CAM, 15, 0, n=5)
    assert len(trail.past) == 5
    assert len(trail.future) == 5


def test_trail_positions_match_observations():
    frames = _frames(5)
    kp = _kp_dict(frames)
    trail = compute_trail(frames, kp, CAM, 2, 0, n=10)
    # past = [slot 0, slot 1]; slot 0 → vf 0 → x=0, slot 1 → vf 1 → x=10
    assert abs(trail.past[0].x - 0.0) < 0.01
    assert abs(trail.past[1].x - 10.0) < 0.01
    # future = [slot 3, slot 4]; slot 3 → vf 3 → x=30, slot 4 → vf 4 → x=40
    assert abs(trail.future[0].x - 30.0) < 0.01
    assert abs(trail.future[1].x - 40.0) < 0.01


def test_trail_ghost_interpolated_between_two_real_anchors():
    """A gap between two real observations yields an interpolated ghost point."""
    frames = _frames(10)
    # Only frames 0 and 2 have observations; frame 1 is a gap
    kp = _kp_dict(frames, include={0, 2})
    trail = compute_trail(frames, kp, CAM, 5, 0, n=10)
    # past indices = [0,1,2,3,4]
    # slot 0 → real (x=0), slot 1 → ghost (midpoint 0..2 = x=10), slot 2 → real (x=20)
    # slots 3,4 → gap with no right anchor → omitted
    real_pts  = [p for p in trail.past if not p.is_ghost]
    ghost_pts = [p for p in trail.past if p.is_ghost]
    assert len(real_pts) == 2
    assert len(ghost_pts) == 1
    assert abs(ghost_pts[0].x - 10.0) < 0.01   # t=1/2 → x = 0 + ½*(20-0)
    assert abs(ghost_pts[0].y - 5.0) < 0.01    # y: 0 + ½*(10-0)


def test_trail_ghost_not_created_without_right_anchor():
    """A gap with only a left anchor produces no ghost (can't interpolate)."""
    frames = _frames(10)
    # Only frame 0 has an observation; 1-4 are gaps (no right anchor within window)
    kp = _kp_dict(frames, include={0})
    trail = compute_trail(frames, kp, CAM, 5, 0, n=5)
    ghost_pts = [p for p in trail.past if p.is_ghost]
    assert len(ghost_pts) == 0


def test_trail_ghost_not_created_without_left_anchor():
    """Trailing gap at the start of the window with no left anchor → no ghost."""
    frames = _frames(10)
    kp = _kp_dict(frames, include={4})
    trail = compute_trail(frames, kp, CAM, 5, 0, n=5)
    # past = [0,1,2,3,4]; only slot 4 is real; no left anchor → no ghost
    ghost_pts = [p for p in trail.past if p.is_ghost]
    assert len(ghost_pts) == 0


def test_trail_outlier_treated_as_gap():
    """Outlier (confidence == 0) keypoint is treated the same as a missing observation."""
    frames = _frames(10)
    kp = _kp_dict(frames, include={0, 1, 2})
    # Overwrite frame 1's confidence to 0 (outlier)
    kp[1][:, 2] = 0.0
    trail = compute_trail(frames, kp, CAM, 5, 0, n=5)
    # Slot 1 is outlier → ghost, interpolated between slot 0 (x=0) and slot 2 (x=20)
    ghost_pts = [p for p in trail.past if p.is_ghost]
    assert len(ghost_pts) == 1
    assert abs(ghost_pts[0].x - 10.0) < 0.01


def test_select_keypoint_triggers_trail(qapp, seq_db):
    """select_keypoint() sets _selected_kp and doesn't crash on redraw."""
    from app.pose.crop_editor import PersonCropGridWidget
    w = PersonCropGridWidget()
    w.load_sequence(seq_db, "seq1")

    assert w._selected_kp is None
    w.select_keypoint(0)
    assert w._selected_kp == 0

    # Deselect
    w.select_keypoint(None)
    assert w._selected_kp is None

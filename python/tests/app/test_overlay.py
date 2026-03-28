"""Tests for app.setup.overlay (ROIDrawOverlay state, SyncAnchorOverlay state)."""

from __future__ import annotations

from app.setup.overlay import (
    AnnotationPointOverlay,
    Rect,
    ReprojectionOverlay,
    ROIDrawOverlay,
    SyncAnchorOverlay,
)


# ---------------------------------------------------------------------------
# Rect helper
# ---------------------------------------------------------------------------


def test_rect_normalised() -> None:
    r = Rect(x1=10, y1=20, x2=5, y2=8)
    n = r.normalised
    assert n.x1 == 5 and n.y1 == 8
    assert n.x2 == 10 and n.y2 == 20


def test_rect_is_valid() -> None:
    assert Rect(0, 0, 10, 10).is_valid
    assert not Rect(0, 0, 0, 10).is_valid   # zero width
    assert not Rect(0, 0, 0, 0).is_valid    # zero size


# ---------------------------------------------------------------------------
# ROIDrawOverlay
# ---------------------------------------------------------------------------


def test_roi_drag_commits_on_release() -> None:
    overlay = ROIDrawOverlay(active=True)
    assert overlay.roi is None

    overlay.mouse_press(10, 20)
    overlay.mouse_move(50, 60)
    overlay.mouse_release(50, 60)

    assert overlay.roi is not None
    assert overlay.roi.x1 == 10
    assert overlay.roi.y1 == 20
    assert overlay.roi.x2 == 50
    assert overlay.roi.y2 == 60


def test_roi_normalises_negative_drag() -> None:
    """Dragging from bottom-right to top-left should produce a normalised ROI."""
    overlay = ROIDrawOverlay(active=True)
    overlay.mouse_press(80, 90)
    overlay.mouse_release(10, 20)

    roi = overlay.roi
    assert roi is not None
    assert roi.x1 == 10 and roi.y1 == 20
    assert roi.x2 == 80 and roi.y2 == 90


def test_roi_inactive_ignores_mouse() -> None:
    overlay = ROIDrawOverlay(active=False)
    overlay.mouse_press(0, 0)
    overlay.mouse_move(100, 100)
    overlay.mouse_release(100, 100)
    assert overlay.roi is None


def test_roi_clear() -> None:
    overlay = ROIDrawOverlay(active=True)
    overlay.mouse_press(0, 0)
    overlay.mouse_release(50, 50)
    assert overlay.roi is not None
    overlay.clear()
    assert overlay.roi is None


def test_roi_zero_size_drag_not_committed() -> None:
    """A click without drag (zero-size rect) should not commit an ROI."""
    overlay = ROIDrawOverlay(active=True)
    overlay.mouse_press(20, 20)
    overlay.mouse_release(20, 20)
    assert overlay.roi is None


# ---------------------------------------------------------------------------
# SyncAnchorOverlay
# ---------------------------------------------------------------------------


def test_sync_anchor_initial_state() -> None:
    overlay = SyncAnchorOverlay()
    assert overlay.anchor_frame is None


def test_sync_anchor_set() -> None:
    overlay = SyncAnchorOverlay(total_frames=1000)
    overlay.set_anchor(250)
    assert overlay.anchor_frame == 250


def test_sync_anchor_mouse_events_are_noop() -> None:
    """Mouse events on SyncAnchorOverlay must not raise."""
    overlay = SyncAnchorOverlay()
    overlay.mouse_press(0, 0)
    overlay.mouse_move(10, 10)
    overlay.mouse_release(10, 10)


# ---------------------------------------------------------------------------
# Stub overlays satisfy the Overlay protocol shape
# ---------------------------------------------------------------------------


def test_stubs_have_required_methods() -> None:
    for cls in (AnnotationPointOverlay, ReprojectionOverlay):
        obj = cls()
        assert callable(obj.paint)
        assert callable(obj.mouse_press)
        assert callable(obj.mouse_move)
        assert callable(obj.mouse_release)
        # Stubs must not raise when called
        obj.mouse_press(0, 0)
        obj.mouse_move(1, 1)
        obj.mouse_release(1, 1)

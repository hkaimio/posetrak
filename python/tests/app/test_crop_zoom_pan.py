"""Tests for the crop-cell zoom/pan geometry in app.ui.content_panels.

Covers the pure functions behind "Zoom and pan in the camera crop views"
(keypoint-editing-design.md): scaling a view rect around an anchor, panning
it, and computing the full-frame-to-display affine transform (with
clamp-to-available-pixels and fall-back-to-fit behavior). QWidget/QPainter
mechanics (actual wheel/mouse events, paintEvent) follow the project's usual
manual-validation convention and are not covered here.
"""
from __future__ import annotations

from app.ui.content_panels import (
    _compute_view_transform,
    _panned_rect,
    _zoomed_rect,
)


# ---------------------------------------------------------------------------
# _zoomed_rect
# ---------------------------------------------------------------------------

def test_zoomed_rect_zoom_in_shrinks_toward_anchor():
    # 100x100 rect, zoom in 2x centered on its top-left corner (0, 0).
    rect = _zoomed_rect((0, 0, 100, 100), factor=2.0, anchor=(0, 0))
    assert rect == (0, 0, 50, 50)


def test_zoomed_rect_zoom_in_centered_shrinks_symmetrically():
    rect = _zoomed_rect((0, 0, 100, 100), factor=2.0, anchor=(50, 50))
    assert rect == (25, 25, 75, 75)


def test_zoomed_rect_zoom_out_grows():
    rect = _zoomed_rect((25, 25, 75, 75), factor=0.5, anchor=(50, 50))
    assert rect == (0, 0, 100, 100)


def test_zoomed_rect_rejects_degenerate_zoom():
    # Zooming in far enough to go below min_size should be rejected (None),
    # not silently produce a near-zero-area rect.
    rect = _zoomed_rect((0, 0, 100, 100), factor=1000.0, anchor=(50, 50), min_size=20.0)
    assert rect is None


def test_zoomed_rect_accepts_zoom_at_the_min_size_floor():
    # 100px / factor 5 = 20px -- exactly at the floor, still accepted.
    rect = _zoomed_rect((0, 0, 100, 100), factor=5.0, anchor=(50, 50), min_size=20.0)
    assert rect == (40, 40, 60, 60)


# ---------------------------------------------------------------------------
# _panned_rect
# ---------------------------------------------------------------------------

def test_panned_rect_translates_all_corners():
    assert _panned_rect((0, 0, 100, 50), dx=10, dy=-5) == (10, -5, 110, 45)


# ---------------------------------------------------------------------------
# _compute_view_transform
# ---------------------------------------------------------------------------

def test_view_transform_fits_whole_pixmap_when_unzoomed():
    scale, ox, oy, view = _compute_view_transform(
        cell_w=200, cell_h=100, pixmap_extent=(0, 0, 400, 200), zoom_rect=None,
    )
    assert scale == 0.5
    assert view == (0, 0, 400, 200)
    # (0,0) full-frame -> (0,0) display; (400,200) -> (200,100) display.
    assert (0 * scale + ox, 0 * scale + oy) == (0, 0)
    assert (400 * scale + ox, 200 * scale + oy) == (200, 100)


def test_view_transform_uses_zoom_rect_when_it_fully_overlaps():
    scale, ox, oy, view = _compute_view_transform(
        cell_w=100, cell_h=100,
        pixmap_extent=(0, 0, 400, 400),
        zoom_rect=(100, 100, 200, 200),
    )
    assert view == (100, 100, 200, 200)
    assert scale == 1.0  # 100x100 view into a 100x100 cell


def test_view_transform_clamps_zoom_rect_to_pixmap_extent():
    # zoom_rect extends past the pixmap's own extent on the right/bottom.
    scale, ox, oy, view = _compute_view_transform(
        cell_w=100, cell_h=100,
        pixmap_extent=(0, 0, 300, 300),
        zoom_rect=(200, 200, 400, 400),
    )
    assert view == (200, 200, 300, 300)  # clamped to the pixmap's own extent


def test_view_transform_falls_back_to_fit_when_zoom_rect_has_no_overlap():
    # zoom_rect entirely outside the current frame's pixmap (e.g. an epoch
    # boundary shifted the underlying crop) -- fall back to fit, don't blank.
    scale, ox, oy, view = _compute_view_transform(
        cell_w=100, cell_h=100,
        pixmap_extent=(0, 0, 300, 300),
        zoom_rect=(1000, 1000, 1100, 1100),
    )
    assert view == (0, 0, 300, 300)


def test_view_transform_degenerate_cell_size_does_not_crash():
    scale, ox, oy, view = _compute_view_transform(
        cell_w=0, cell_h=0, pixmap_extent=(0, 0, 100, 100), zoom_rect=None,
    )
    assert scale == 1.0 and ox == 0.0 and oy == 0.0

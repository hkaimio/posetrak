# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Overlay protocol and concrete overlay classes for CameraCell widgets.

Each CameraCell holds a ``list[Overlay]``.  After rendering the video frame
it calls ``paint()`` on each overlay in order.  Mouse events are forwarded in
reverse order so the top-most overlay gets first priority.

All pixel coordinates in mouse events are in **video-frame space** (mapped
from display space by CameraCell before forwarding).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

try:
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QColor, QPainter, QPen
    from PySide6.QtCore import Qt
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Overlay Protocol
# ---------------------------------------------------------------------------


class Overlay(Protocol):
    """Typed protocol for all CameraCell overlays.

    Implementations do not need to inherit from this class — structural
    subtyping (duck typing checked by the type checker) is sufficient.
    """

    def paint(
        self,
        painter: "QPainter",
        frame_w: int,
        frame_h: int,
        cell_w: int,
        cell_h: int,
    ) -> None:
        """Draw this overlay onto *painter*.

        Parameters
        ----------
        painter:
            Active QPainter on the cell widget.
        frame_w, frame_h:
            Source video frame dimensions in pixels.
        cell_w, cell_h:
            Display cell dimensions in pixels (after scaling).
        """
        ...

    def mouse_press(self, x_px: int, y_px: int) -> None:
        """Handle a mouse press at video-frame coordinates *(x_px, y_px)*."""
        ...

    def mouse_move(self, x_px: int, y_px: int) -> None:
        """Handle a mouse move at video-frame coordinates *(x_px, y_px)*."""
        ...

    def mouse_release(self, x_px: int, y_px: int) -> None:
        """Handle a mouse release at video-frame coordinates *(x_px, y_px)*."""
        ...


# ---------------------------------------------------------------------------
# SyncAnchorOverlay
# ---------------------------------------------------------------------------


class SyncAnchorOverlay:
    """Draws a vertical tick mark at the user-set sync anchor frame.

    Only ``paint()`` does anything; mouse events are no-ops.

    Parameters
    ----------
    anchor_frame:
        Video frame index where the anchor is set.  ``None`` = not yet set
        (nothing is drawn).
    total_frames:
        Total frame count of the video (used to compute the tick x position).
    """

    def __init__(
        self,
        anchor_frame: int | None = None,
        total_frames: int = 1,
    ) -> None:
        self.anchor_frame = anchor_frame
        self.total_frames = max(total_frames, 1)

    def set_anchor(self, frame: int) -> None:
        self.anchor_frame = frame

    def paint(
        self,
        painter: "QPainter",
        frame_w: int,
        frame_h: int,
        cell_w: int,
        cell_h: int,
    ) -> None:
        if not _QT_AVAILABLE or self.anchor_frame is None:
            return
        # Draw a vertical cyan tick in the bottom strip of the cell
        x = int(self.anchor_frame / self.total_frames * cell_w)
        strip_h = max(8, cell_h // 16)
        pen = QPen(QColor(0, 220, 220), 2)
        painter.setPen(pen)
        painter.drawLine(x, cell_h - strip_h, x, cell_h)

    def mouse_press(self, x_px: int, y_px: int) -> None:
        pass

    def mouse_move(self, x_px: int, y_px: int) -> None:
        pass

    def mouse_release(self, x_px: int, y_px: int) -> None:
        pass


# ---------------------------------------------------------------------------
# ROIDrawOverlay
# ---------------------------------------------------------------------------


@dataclass
class Rect:
    """Axis-aligned rectangle in video-frame pixel coordinates."""
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0

    @property
    def normalised(self) -> "Rect":
        """Return a Rect with x1 ≤ x2 and y1 ≤ y2."""
        return Rect(
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
        )

    @property
    def is_valid(self) -> bool:
        n = self.normalised
        return n.x2 > n.x1 and n.y2 > n.y1


class ROIDrawOverlay:
    """Rubber-band rectangle overlay for LED ROI selection.

    The user drags a rectangle on the video cell.  The committed ROI is
    available via the ``roi`` property after mouse release.

    Parameters
    ----------
    active:
        Whether the overlay is currently accepting mouse input.  When
        ``False``, ``paint()`` still draws the committed ROI if one exists.
    """

    def __init__(self, active: bool = True) -> None:
        self.active = active
        self._dragging = False
        self._current = Rect()
        self._committed: Rect | None = None

    @property
    def roi(self) -> Rect | None:
        """The last committed (mouse-release) ROI, or ``None``."""
        return self._committed

    def clear(self) -> None:
        """Remove the committed ROI."""
        self._committed = None
        self._dragging = False

    def paint(
        self,
        painter: "QPainter",
        frame_w: int,
        frame_h: int,
        cell_w: int,
        cell_h: int,
    ) -> None:
        if not _QT_AVAILABLE:
            return
        scale_x = cell_w / max(frame_w, 1)
        scale_y = cell_h / max(frame_h, 1)

        def _draw_rect(r: Rect, color: "QColor") -> None:
            n = r.normalised
            x = int(n.x1 * scale_x)
            y = int(n.y1 * scale_y)
            w = int((n.x2 - n.x1) * scale_x)
            h = int((n.y2 - n.y1) * scale_y)
            pen = QPen(color, 2, Qt.DashLine if self._dragging else Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x, y, w, h)

        if self._dragging and self._current.is_valid:
            _draw_rect(self._current, QColor(255, 200, 0))
        elif self._committed is not None and self._committed.is_valid:
            _draw_rect(self._committed, QColor(255, 200, 0))

    def mouse_press(self, x_px: int, y_px: int) -> None:
        if not self.active:
            return
        self._dragging = True
        self._current = Rect(x_px, y_px, x_px, y_px)

    def mouse_move(self, x_px: int, y_px: int) -> None:
        if not self.active or not self._dragging:
            return
        self._current.x2 = x_px
        self._current.y2 = y_px

    def mouse_release(self, x_px: int, y_px: int) -> None:
        if not self.active:
            return
        self._dragging = False
        self._current.x2 = x_px
        self._current.y2 = y_px
        if self._current.is_valid:
            self._committed = self._current.normalised


# ---------------------------------------------------------------------------
# Stubs for future overlays
# ---------------------------------------------------------------------------


class AnnotationPointOverlay:
    """Labelled dots + zoom-refine interaction for extrinsics annotation.

    Not yet implemented — all methods are stubs.
    """

    def paint(
        self,
        painter: "QPainter",
        frame_w: int,
        frame_h: int,
        cell_w: int,
        cell_h: int,
    ) -> None:
        pass

    def mouse_press(self, x_px: int, y_px: int) -> None:
        pass

    def mouse_move(self, x_px: int, y_px: int) -> None:
        pass

    def mouse_release(self, x_px: int, y_px: int) -> None:
        pass


class ReprojectionOverlay:
    """Reprojected circles + residual lines after PnP solve.

    Not yet implemented — all methods are stubs.
    """

    def paint(
        self,
        painter: "QPainter",
        frame_w: int,
        frame_h: int,
        cell_w: int,
        cell_h: int,
    ) -> None:
        pass

    def mouse_press(self, x_px: int, y_px: int) -> None:
        pass

    def mouse_move(self, x_px: int, y_px: int) -> None:
        pass

    def mouse_release(self, x_px: int, y_px: int) -> None:
        pass

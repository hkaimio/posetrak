# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""CameraCell — single-camera video display widget for the setup wizard.

Displays one decoded video frame as a scaled image, then calls each
registered ``Overlay.paint()`` on top.  Mouse events are forwarded to
overlays in reverse order (top-most overlay gets first chance) after mapping
display coordinates to video-frame coordinates.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from app.setup.overlay import Overlay

# Border constants
_BORDER_PX = 3
_SELECTED_COLOR = QColor(220, 60, 0)   # red-orange — scrubber-selected camera
_FOCUS_COLOR = QColor(0, 150, 255)     # blue — Qt keyboard focus (fallback)
_PLACEHOLDER_COLOR = QColor(30, 30, 30)


class CameraCell(QWidget):
    """Widget that displays a single camera's video frame with overlays.

    Frames are stretched to fill the widget area (no letterboxing).  This
    keeps the coordinate mapping between display and frame space linear, with
    no offset:

        frame_x = display_x * frame_w / cell_w
        frame_y = display_y * frame_h / cell_h

    Parameters
    ----------
    label:
        Optional camera label shown in the top-left corner when no frame is
        loaded.
    parent:
        Parent widget.
    """

    #: Emitted when the cell is clicked (before focus changes).
    clicked = Signal()

    def __init__(self, label: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = label
        self._frame: np.ndarray | None = None
        self._overlays: list[Overlay] = []
        self._selected = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(160, 90)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_frame(self, frame: np.ndarray) -> None:
        """Display *frame* (H×W×3 BGR uint8 array) and repaint."""
        self._frame = frame
        self.update()

    def clear_frame(self) -> None:
        """Remove the current frame and show the placeholder."""
        self._frame = None
        self.update()

    def set_selected(self, selected: bool) -> None:
        """Mark this cell as the scrubber-focused camera (red border)."""
        if self._selected != selected:
            self._selected = selected
            self.update()

    def set_overlays(self, overlays: list[Overlay]) -> None:
        """Replace the overlay list.  Triggers a repaint."""
        self._overlays = list(overlays)
        self.update()

    def display_to_frame(self, dx: int, dy: int) -> tuple[int, int]:
        """Map display-pixel coordinates to video-frame pixel coordinates."""
        if self._frame is None:
            return dx, dy
        fh, fw = self._frame.shape[:2]
        x_off, y_off, img_w, img_h = self._letterbox_rect(fw, fh)
        if img_w <= 0 or img_h <= 0:
            return 0, 0
        fx = int((dx - x_off) * fw / img_w)
        fy = int((dy - y_off) * fh / img_h)
        return fx, fy

    def _letterbox_rect(self, fw: int, fh: int) -> tuple[int, int, int, int]:
        """Return (x_off, y_off, img_w, img_h) for the frame scaled to fit the cell."""
        cw, ch = self.width(), self.height()
        scale = min(cw / max(fw, 1), ch / max(fh, 1))
        img_w = int(fw * scale)
        img_h = int(fh * scale)
        x_off = (cw - img_w) // 2
        y_off = (ch - img_h) // 2
        return x_off, y_off, img_w, img_h

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: ARG002
        painter = QPainter(self)
        cw, ch = self.width(), self.height()

        if self._frame is not None:
            fh, fw = self._frame.shape[:2]
            # Convert BGR → RGB for QImage
            rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
            # Ensure contiguous layout for QImage
            if not rgb.flags["C_CONTIGUOUS"]:
                rgb = np.ascontiguousarray(rgb)
            img = QImage(
                rgb.data,
                fw, fh,
                int(rgb.strides[0]),
                QImage.Format.Format_RGB888,
            )
            # Letterbox: scale to fit preserving aspect ratio, centre in cell
            x_off, y_off, img_w, img_h = self._letterbox_rect(fw, fh)
            pixmap = QPixmap.fromImage(img).scaled(
                img_w, img_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.fillRect(0, 0, cw, ch, _PLACEHOLDER_COLOR)
            painter.drawPixmap(x_off, y_off, pixmap)

            # Paint overlays translated to the image origin
            painter.save()
            painter.translate(x_off, y_off)
            for overlay in self._overlays:
                overlay.paint(painter, fw, fh, img_w, img_h)
            painter.restore()
        else:
            # Placeholder
            painter.fillRect(0, 0, cw, ch, _PLACEHOLDER_COLOR)
            if self._label:
                painter.setPen(QColor(120, 120, 120))
                painter.drawText(4, 16, self._label)

        # Border: selected (red) takes priority over Qt keyboard focus (blue)
        if self._selected:
            color = _SELECTED_COLOR
        elif self.hasFocus():
            color = _FOCUS_COLOR
        else:
            color = None
        if color is not None:
            pen = QPen(color, _BORDER_PX)
            painter.setPen(pen)
            b = _BORDER_PX // 2
            painter.drawRect(b, b, cw - _BORDER_PX, ch - _BORDER_PX)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()
        fx, fy = self.display_to_frame(int(event.position().x()), int(event.position().y()))
        for overlay in reversed(self._overlays):
            overlay.mouse_press(fx, fy)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        fx, fy = self.display_to_frame(int(event.position().x()), int(event.position().y()))
        for overlay in reversed(self._overlays):
            overlay.mouse_move(fx, fy)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        fx, fy = self.display_to_frame(int(event.position().x()), int(event.position().y()))
        for overlay in reversed(self._overlays):
            overlay.mouse_release(fx, fy)
        super().mouseReleaseEvent(event)

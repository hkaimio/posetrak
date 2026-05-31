"""video_canvas.py — Scaled video frame display with segmentation mask overlay.

Displays a BGR video frame scaled to fit the widget (letterboxed), with an
optional labeled segmentation mask blended on top using the DAVIS palette.
Mouse clicks are transformed back to image coordinates and emitted as signals.

Coordinate convention:
  - Canvas coords: pixels in the QLabel widget (origin top-left).
  - Image coords:  pixels in the original (possibly scaled) frame.
  - The display scales the frame uniformly to fit, with black bars if the
    aspect ratio doesn't match.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor
from PySide6.QtWidgets import QLabel, QSizePolicy


# DAVIS colour palette: first 16 entries (label 0 = background, 1..N = persons).
# RGB values matching the standard XMem/Cutie visualisation colours.
_DAVIS_RGB: list[tuple[int, int, int]] = [
    (0,   0,   0  ),  # 0  background (not drawn)
    (240, 80,  80 ),  # 1  red-ish
    (80,  200, 120),  # 2  green-ish
    (80,  120, 240),  # 3  blue-ish
    (240, 200, 60 ),  # 4  yellow
    (180, 80,  240),  # 5  purple
    (60,  220, 220),  # 6  cyan
    (240, 140, 60 ),  # 7  orange
    (160, 160, 160),  # 8  gray
    (120, 240, 80 ),  # 9  lime
    (240, 60,  160),  # 10 pink
    (60,  160, 240),  # 11 sky blue
    (200, 200, 60 ),  # 12 olive
    (240, 100, 120),  # 13 salmon
    (100, 240, 200),  # 14 aqua
    (200, 140, 240),  # 15 lavender
]
_MASK_ALPHA = 0.45  # opacity of mask overlay


def _build_palette_lut() -> np.ndarray:
    """Return (256, 3) uint8 LUT mapping label index → BGR colour."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i, (r, g, b) in enumerate(_DAVIS_RGB):
        if i >= 256:
            break
        lut[i] = [b, g, r]  # OpenCV BGR order
    return lut


_PALETTE_LUT = _build_palette_lut()


def label_to_color(label: int) -> tuple[int, int, int]:
    """Return (R, G, B) for a person label (0 = background)."""
    if 0 <= label < len(_DAVIS_RGB):
        return _DAVIS_RGB[label]
    return (128, 128, 128)


def blend_mask(frame_bgr: np.ndarray, labeled_mask: np.ndarray) -> np.ndarray:
    """Return a copy of *frame_bgr* with *labeled_mask* blended on top.

    Parameters
    ----------
    frame_bgr:
        HxWx3 uint8 BGR frame.
    labeled_mask:
        HxW uint8 label map (0 = background, 1..N = person).
    """
    colours = _PALETTE_LUT[labeled_mask]   # HxW x3  BGR colours
    fg = labeled_mask > 0                  # HxW bool, True where there is a mask
    out = frame_bgr.copy()
    out[fg] = (
        out[fg].astype(np.float32) * (1 - _MASK_ALPHA)
        + colours[fg].astype(np.float32) * _MASK_ALPHA
    ).astype(np.uint8)
    return out


class VideoCanvas(QLabel):
    """Letterboxed video frame display with mask overlay and click signals.

    Signals
    -------
    left_clicked(x, y):
        Left mouse press in image coordinates.
    right_clicked(x, y):
        Right mouse press in image coordinates.
    """

    left_clicked = Signal(int, int)
    right_clicked = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: black;")

        self._frame: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._message: str | None = None

        # Transform state: image_coord = (canvas_coord - offset) / scale
        self._scale: float = 1.0
        self._offset_x: int = 0
        self._offset_y: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def display(
        self,
        frame_bgr: np.ndarray | None,
        mask_labeled: np.ndarray | None = None,
        message: str | None = None,
    ) -> None:
        """Update the displayed frame and optional mask overlay.

        If *frame_bgr* is None and *message* is set, the message is shown
        centred on the black canvas (e.g. "Video unavailable").
        """
        self._frame = frame_bgr
        self._mask = mask_labeled
        self._message = message
        self._render()

    def clear(self) -> None:
        """Show a blank black canvas."""
        self._frame = None
        self._mask = None
        self._message = None
        self.setPixmap(QPixmap())

    def canvas_to_image(self, cx: int, cy: int) -> tuple[int, int] | None:
        """Convert canvas pixel coordinates to image pixel coordinates.

        Returns None if the point is outside the displayed image area.
        """
        ix = (cx - self._offset_x) / self._scale
        iy = (cy - self._offset_y) / self._scale
        if self._frame is None:
            return None
        h, w = self._frame.shape[:2]
        if not (0 <= ix < w and 0 <= iy < h):
            return None
        return int(ix), int(iy)

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()

    def mousePressEvent(self, event) -> None:
        pos = event.position()
        cx, cy = int(pos.x()), int(pos.y())
        img_coords = self.canvas_to_image(cx, cy)
        if img_coords is None:
            return
        ix, iy = img_coords
        if event.button() == Qt.MouseButton.LeftButton:
            self.left_clicked.emit(ix, iy)
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit(ix, iy)

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        ww, wh = self.width(), self.height()
        if ww == 0 or wh == 0:
            return

        canvas = QPixmap(ww, wh)
        canvas.fill(QColor(0, 0, 0))

        if self._frame is None:
            if self._message:
                painter = QPainter(canvas)
                painter.setPen(QColor(160, 160, 160))
                painter.drawText(canvas.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
                painter.end()
            self.setPixmap(canvas)
            return

        frame = self._frame
        if self._mask is not None and self._mask.shape[:2] == frame.shape[:2]:
            frame = blend_mask(frame, self._mask)

        # Scale to fit the widget while preserving aspect ratio.
        fh, fw = frame.shape[:2]
        if fw == 0 or fh == 0:
            return

        scale = min(ww / fw, wh / fh)
        dw = int(fw * scale)
        dh = int(fh * scale)
        self._scale = scale
        self._offset_x = (ww - dw) // 2
        self._offset_y = (wh - dh) // 2

        import cv2
        display = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)

        # Convert BGR → RGB for QImage.  Use bytes() to ensure QImage owns
        # the buffer — rgb.data is a memoryview that may be freed before
        # QPixmap.fromImage() copies the pixels.
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(bytes(rgb), w, h, w * ch, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        painter = QPainter(canvas)
        painter.drawPixmap(self._offset_x, self._offset_y, pixmap)
        painter.end()
        self.setPixmap(canvas)

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""video_canvas.py — Scaled video frame display with segmentation mask overlay.

Displays a BGR video frame scaled to fit the widget (letterboxed), with an
optional labeled segmentation mask blended on top using the DAVIS palette.
Mouse clicks are transformed back to image coordinates and emitted as signals.

Coordinate convention:
  - Canvas coords: pixels in the QLabel widget (origin top-left).
  - Image coords:  pixels in the original (possibly scaled) frame.
  - The display scales the frame uniformly to fit, with black bars if the
    aspect ratio doesn't match -- unless zoomed in (see zoom_in_at()), in
    which case the visible region is a crop of the frame instead, still
    letterboxed on whichever axis (if any) the crop doesn't fill.

Zoom/pan (segmentation-ui-improvements design doc, Issue 5): zoom_in_at()/
zoom_out_at() recentre on an image point and scale by ZOOM_STEP, clamped to
[ZOOM_MIN, ZOOM_MAX]. The expensive part of a redraw (video decode already
done by the caller, but the cv2 resize/mask-blend/BGR->RGB conversion here)
is cached as self._base_pixmap and only recomputed when the frame/mask/
zoom actually changes -- mouse-move (tracking the cursor for the brush
tool's preview circle) instead just copies that cached pixmap and draws
the circle on top, cheap enough for interactive drag rates.

The skeleton overlay is only painted at zoom 1.0 -- its own coordinate
math assumes the full frame maps linearly into the display rect, which
stops being true once the display is a crop rather than the whole frame;
deliberately skipped instead of drawing it in the wrong place.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PySide6.QtWidgets import QLabel, QSizePolicy

#: Multiplicative zoom step per zoom_in_at()/zoom_out_at() call.
ZOOM_STEP = 1.6
ZOOM_MIN = 1.0
ZOOM_MAX = 8.0


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
    left_dragged(x, y):
        Mouse moved with the left button held, in image coordinates --
        for a continuous paint/erase brush stroke, not emitted for the
        select/zoom tools (see set_tool()).
    """

    left_clicked = Signal(int, int)
    right_clicked = Signal(int, int)
    left_dragged = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: black;")
        self.setMouseTracking(True)  # mouseMoveEvent fires without a button held too

        self._frame: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._message: str | None = None
        self._skeleton_overlay = None   # SkeletonDetectionOverlay | None
        # Native video resolution for the skeleton overlay.  Keypoints are stored
        # at native resolution (4K) but the display frame from FrameCache is at
        # max_dim=1920p.  The overlay's paint() needs the native dimensions so it
        # can compute the correct scale factors from keypoint px → display px.
        # 0 means "not set; fall back to display frame dimensions".
        self._kp_frame_w: int = 0
        self._kp_frame_h: int = 0

        # Transform state: image_coord = (src_x0 + (canvas_coord - offset) / scale)
        self._scale: float = 1.0
        self._offset_x: int = 0
        self._offset_y: int = 0
        self._src_x0: float = 0.0
        self._src_y0: float = 0.0

        # Zoom/pan -- pan_c{x,y} is None until the first zoom_in_at(), meaning
        # "centered on the frame" (the pre-zoom default).
        self._zoom: float = 1.0
        self._pan_cx: float | None = None
        self._pan_cy: float | None = None

        # Brush tool state -- see set_tool()/set_brush_radius(). The cached
        # expensive render, so mouse-move (tracking the cursor for the brush
        # preview circle) can cheaply redraw just the circle on top instead
        # of redoing the cv2 decode/resize/blend pipeline.
        self._active_tool: str = "select"
        self._brush_radius: int = 10
        self._cursor_img: tuple[int, int] | None = None
        self._base_pixmap: QPixmap | None = None

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

    def set_keypoint_resolution(self, w: int, h: int) -> None:
        """Set the native video resolution used to scale keypoint overlay coordinates.

        Keypoints are stored at the original video resolution (e.g. 4K) while
        the display frame comes from FrameCache at max_dim=1920p.  Call this
        whenever the active camera changes so the overlay uses the correct
        scale factors.
        """
        self._kp_frame_w = w
        self._kp_frame_h = h

    def set_skeleton_overlay(self, overlay) -> None:
        """Attach a SkeletonDetectionOverlay to be painted over every frame.

        Pass None to remove the overlay.  The overlay is not owned by the
        canvas — the caller is responsible for calling set_detections() on
        it before each display() to update the painted data.
        """
        self._skeleton_overlay = overlay
        self._render()

    def clear(self) -> None:
        """Show a blank black canvas."""
        self._frame = None
        self._mask = None
        self._message = None
        self._base_pixmap = None
        self.setPixmap(QPixmap())

    def canvas_to_image(self, cx: int, cy: int) -> tuple[int, int] | None:
        """Convert canvas pixel coordinates to image pixel coordinates.

        Returns None if the point is outside the displayed image area.
        """
        ix = self._src_x0 + (cx - self._offset_x) / self._scale
        iy = self._src_y0 + (cy - self._offset_y) / self._scale
        if self._frame is None:
            return None
        h, w = self._frame.shape[:2]
        if not (0 <= ix < w and 0 <= iy < h):
            return None
        return int(ix), int(iy)

    # ------------------------------------------------------------------
    # Tools (segmentation-ui-improvements design doc, Issue 5)
    # ------------------------------------------------------------------

    def set_tool(self, tool: str) -> None:
        """Set the active tool -- "select" (SAM2 clicks, unchanged), "paint",
        "erase", or "zoom". Only affects the brush-preview circle and
        left_dragged emission here; the actual paint/erase/zoom behavior
        lives in the caller (CutieInitPanel), which decides what left_clicked/
        right_clicked/left_dragged mean based on the same tool."""
        self._active_tool = tool
        self._update_cursor_overlay()

    def set_brush_radius(self, radius: int) -> None:
        """Brush radius in *image* pixels (not screen pixels), so painting
        stays consistent across zoom levels -- only the on-screen preview
        circle's radius scales with the current zoom."""
        self._brush_radius = max(1, radius)
        self._update_cursor_overlay()

    def zoom_in_at(self, ix: int, iy: int) -> None:
        """Zoom in by ZOOM_STEP, recentring on image point (ix, iy)."""
        self._zoom = min(ZOOM_MAX, self._zoom * ZOOM_STEP)
        self._pan_cx, self._pan_cy = float(ix), float(iy)
        self._render()

    def zoom_out_at(self, ix: int, iy: int) -> None:
        """Zoom out by ZOOM_STEP, recentring on image point (ix, iy)."""
        self._zoom = max(ZOOM_MIN, self._zoom / ZOOM_STEP)
        self._pan_cx, self._pan_cy = float(ix), float(iy)
        self._render()

    def reset_zoom(self) -> None:
        """Back to fit-to-widget, centred on the whole frame."""
        self._zoom = 1.0
        self._pan_cx = None
        self._pan_cy = None
        self._render()

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

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        cx, cy = int(pos.x()), int(pos.y())
        img_coords = self.canvas_to_image(cx, cy)
        self._cursor_img = img_coords
        self._update_cursor_overlay()
        if img_coords is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self.left_dragged.emit(*img_coords)

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._cursor_img = None
        self._update_cursor_overlay()

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        """Recompute self._base_pixmap (the expensive path: cv2 crop/
        resize/mask-blend/BGR->RGB) and display it. Called whenever the
        frame, mask, zoom, or widget size actually changes -- NOT on
        every mouse move, see _update_cursor_overlay()."""
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
            self._base_pixmap = canvas
            self.setPixmap(canvas)
            return

        import cv2
        frame = self._frame
        if self._mask is not None:
            mask = self._mask
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(
                    mask, (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            frame = blend_mask(frame, mask)

        fh, fw = frame.shape[:2]
        if fw == 0 or fh == 0:
            return

        fit_scale = min(ww / fw, wh / fh)
        scale = fit_scale * self._zoom
        self._scale = scale

        pan_cx = self._pan_cx if self._pan_cx is not None else fw / 2.0
        pan_cy = self._pan_cy if self._pan_cy is not None else fh / 2.0

        # Visible image-space viewport, per axis independently: letterbox
        # (centered offset, no crop) if the whole frame fits that axis at
        # this scale, otherwise crop to a pan-clamped window of that size.
        view_w = ww / scale
        view_h = wh / scale

        if view_w >= fw:
            src_x0, src_w = 0.0, float(fw)
            dst_x0 = int(round((ww - fw * scale) / 2))
        else:
            src_x0 = max(0.0, min(pan_cx - view_w / 2, fw - view_w))
            src_w = view_w
            dst_x0 = 0

        if view_h >= fh:
            src_y0, src_h = 0.0, float(fh)
            dst_y0 = int(round((wh - fh * scale) / 2))
        else:
            src_y0 = max(0.0, min(pan_cy - view_h / 2, fh - view_h))
            src_h = view_h
            dst_y0 = 0

        self._offset_x = dst_x0
        self._offset_y = dst_y0
        self._src_x0 = src_x0
        self._src_y0 = src_y0

        sx0, sy0 = int(round(src_x0)), int(round(src_y0))
        sx1 = min(fw, sx0 + max(1, int(round(src_w))))
        sy1 = min(fh, sy0 + max(1, int(round(src_h))))
        crop = frame[sy0:sy1, sx0:sx1]
        dw = max(1, int(round((sx1 - sx0) * scale)))
        dh = max(1, int(round((sy1 - sy0) * scale)))

        display = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_LINEAR)

        # Convert BGR → RGB for QImage.  Use bytes() to ensure QImage owns
        # the buffer — rgb.data is a memoryview that may be freed before
        # QPixmap.fromImage() copies the pixels.
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(bytes(rgb), w, h, w * ch, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        painter = QPainter(canvas)
        painter.drawPixmap(self._offset_x, self._offset_y, pixmap)
        # Skeleton overlay only at zoom 1.0 -- its own coordinate math
        # assumes the whole frame maps linearly into the display rect,
        # which stops being true once the display is a crop (see module
        # docstring) rather than drawing it in the wrong place.
        if self._skeleton_overlay is not None and self._zoom == 1.0:
            kp_w = self._kp_frame_w if self._kp_frame_w > 0 else fw
            kp_h = self._kp_frame_h if self._kp_frame_h > 0 else fh
            painter.save()
            painter.translate(self._offset_x, self._offset_y)
            self._skeleton_overlay.paint(painter, kp_w, kp_h, dw, dh)
            painter.restore()
        painter.end()
        self._base_pixmap = canvas
        self.setPixmap(canvas)
        self._update_cursor_overlay()

    def _update_cursor_overlay(self) -> None:
        """Cheap redraw: copy the cached base pixmap and draw the brush
        preview circle on top, without touching frame/mask data at all.
        Safe to call on every mouse-move."""
        if self._base_pixmap is None:
            return
        show_circle = (
            self._active_tool in ("paint", "erase")
            and self._cursor_img is not None
        )
        if not show_circle:
            self.setPixmap(self._base_pixmap)
            return

        pixmap = self._base_pixmap.copy()
        ix, iy = self._cursor_img
        cx = self._offset_x + (ix - self._src_x0) * self._scale
        cy = self._offset_y + (iy - self._src_y0) * self._scale
        r = self._brush_radius * self._scale

        painter = QPainter(pixmap)
        color = QColor(240, 60, 60) if self._active_tool == "erase" else QColor(240, 240, 240)
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawEllipse(QPoint(int(cx), int(cy)), int(r), int(r))
        painter.end()
        self.setPixmap(pixmap)

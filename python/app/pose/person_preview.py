# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""person_preview.py — PersonPreviewWidget: zoomed crop of the selected track's bbox."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from app.pose.colors import person_color

# COCO 17-keypoint skeleton connections used for stick-figure drawing.
# For models with more keypoints (e.g. COCO133) the extra keypoints are drawn
# as dots only; full 133-kp connections can be added here later.
_BODY_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

_MIN_CROP_PX = 4     # minimum crop dimension after clamping
_CONF_HIGH = 0.5
_CONF_MED = 0.3


# ---------------------------------------------------------------------------
# Pure functions (testable without Qt)
# ---------------------------------------------------------------------------

def compute_crop_rect(
    bbox_cx: float,
    bbox_cy: float,
    bbox_w: float,
    bbox_h: float,
    frame_w: int,
    frame_h: int,
    margin: float = 0.15,
) -> tuple[int, int, int, int]:
    """Return a (x1, y1, x2, y2) crop region in pixel coordinates.

    The region is the bounding box expanded by *margin* on each side (as a
    fraction of the bbox dimension) and clamped to the frame boundaries.
    Always at least *_MIN_CROP_PX* wide and tall.

    Args:
        bbox_cx: Bbox centre x in pixels (distorted frame space).
        bbox_cy: Bbox centre y in pixels.
        bbox_w: Bbox width in pixels.
        bbox_h: Bbox height in pixels.
        frame_w: Full frame width in pixels.
        frame_h: Full frame height in pixels.
        margin: Fractional margin added on each side (default 0.15 = 15%).

    Returns:
        (x1, y1, x2, y2) integer pixel coordinates, clamped to frame bounds.
    """
    mx = bbox_w * margin
    my = bbox_h * margin
    x1 = int(bbox_cx - bbox_w / 2 - mx)
    y1 = int(bbox_cy - bbox_h / 2 - my)
    x2 = int(bbox_cx + bbox_w / 2 + mx)
    y2 = int(bbox_cy + bbox_h / 2 + my)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    # Guarantee minimum size
    if x2 - x1 < _MIN_CROP_PX:
        x2 = min(frame_w, x1 + _MIN_CROP_PX)
    if y2 - y1 < _MIN_CROP_PX:
        y2 = min(frame_h, y1 + _MIN_CROP_PX)

    return x1, y1, x2, y2


def bbox_from_detections(
    track_id: int,
    detections: list[dict],
) -> tuple[float, float, float, float] | None:
    """Return (cx, cy, w, h) for *track_id* in *detections*, or None if not found.

    Args:
        track_id: The track ID to look up.
        detections: List of detection dicts with keys bbox_x, bbox_y, bbox_w, bbox_h
            (centre-format, pixel coords) and track_id.

    Returns:
        (cx, cy, w, h) or None.
    """
    for det in detections:
        if det["track_id"] == track_id:
            return (
                float(det["bbox_x"]),
                float(det["bbox_y"]),
                float(det["bbox_w"]),
                float(det["bbox_h"]),
            )
    return None


def draw_skeleton_on_crop(
    crop_bgr: np.ndarray,
    keypoints: np.ndarray,
    crop_x1: int,
    crop_y1: int,
) -> np.ndarray:
    """Draw skeleton keypoints and connections onto a bbox crop (in-place).

    Keypoint coordinates are in original frame space; they are translated to
    crop space before drawing.  All N keypoints are drawn as dots (coloured by
    confidence); COCO-17 body connections are drawn as lines.  Extra keypoints
    beyond index 16 (e.g. COCO133 face/hand points) are drawn as dots only.

    Args:
        crop_bgr: numpy uint8 array (h, w, 3) — the cropped region, modified in place.
        keypoints: float32 array (N, 3) — columns are (x, y, conf) in frame pixels.
        crop_x1: x offset of the crop's top-left corner in frame space.
        crop_y1: y offset of the crop's top-left corner in frame space.

    Returns:
        The same *crop_bgr* array after drawing.
    """
    if keypoints is None or keypoints.shape[0] == 0:
        return crop_bgr

    n_kp = keypoints.shape[0]

    # Draw body skeleton connections (COCO-17 subset)
    for a, b in _BODY_SKELETON:
        if a >= n_kp or b >= n_kp:
            continue
        if keypoints[a, 2] < _CONF_MED or keypoints[b, 2] < _CONF_MED:
            continue
        xa = int(keypoints[a, 0]) - crop_x1
        ya = int(keypoints[a, 1]) - crop_y1
        xb = int(keypoints[b, 0]) - crop_x1
        yb = int(keypoints[b, 1]) - crop_y1
        cv2.line(crop_bgr, (xa, ya), (xb, yb), (255, 255, 255), 1, cv2.LINE_AA)

    # Draw all keypoints as coloured dots
    for i in range(n_kp):
        conf = float(keypoints[i, 2])
        if conf < 0.1:
            continue
        x = int(keypoints[i, 0]) - crop_x1
        y = int(keypoints[i, 1]) - crop_y1
        if conf >= _CONF_HIGH:
            color = (60, 220, 0)   # BGR green
        elif conf >= _CONF_MED:
            color = (0, 200, 255)  # BGR yellow
        else:
            color = (40, 40, 220)  # BGR red
        cv2.circle(crop_bgr, (x, y), 3, color, -1, cv2.LINE_AA)

    return crop_bgr


def draw_skeleton_qt(
    painter: QPainter,
    kp: np.ndarray,
    offset_x: float,
    offset_y: float,
    scale: float,
) -> None:
    """Draw skeleton keypoints and connections using Qt's QPainter.

    Maps keypoints from full-frame pixel coordinates into the display surface
    described by *offset_x*, *offset_y*, and *scale*::

        display_x = (frame_x - offset_x) * scale
        display_y = (frame_y - offset_y) * scale

    This is the Qt equivalent of :func:`draw_skeleton_on_crop` and matches the
    style used in ``SkeletonDetectionOverlay._draw_skeleton()`` (frame_view.py).

    Args:
        painter: An active QPainter targeting the destination QPixmap / widget.
        kp: float32 array ``(N, 3)`` — ``(x, y, conf)`` in full-frame pixel coords.
        offset_x: x coordinate of the crop's top-left corner in frame space.
        offset_y: y coordinate of the crop's top-left corner in frame space.
        scale: Uniform scale factor mapping crop-space pixels to display pixels.
    """
    if kp is None or kp.shape[0] == 0:
        return

    n_kp = kp.shape[0]
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Skeleton connections
    painter.setPen(QPen(QColor(255, 255, 255, 160), 1))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for a, b in _BODY_SKELETON:
        if a >= n_kp or b >= n_kp:
            continue
        if kp[a, 2] < _CONF_MED or kp[b, 2] < _CONF_MED:
            continue
        xa = int((kp[a, 0] - offset_x) * scale)
        ya = int((kp[a, 1] - offset_y) * scale)
        xb = int((kp[b, 0] - offset_x) * scale)
        yb = int((kp[b, 1] - offset_y) * scale)
        painter.drawLine(xa, ya, xb, yb)

    # Keypoint dots — colour-coded by confidence
    painter.setPen(Qt.PenStyle.NoPen)
    for i in range(n_kp):
        conf = float(kp[i, 2])
        if conf < 0.1:
            continue
        x = int((kp[i, 0] - offset_x) * scale)
        y = int((kp[i, 1] - offset_y) * scale)
        if conf >= _CONF_HIGH:
            color = QColor(0, 220, 60)    # green  — high confidence
        elif conf >= _CONF_MED:
            color = QColor(255, 200, 0)   # yellow — medium confidence
        else:
            color = QColor(220, 40, 40)   # red    — low confidence
        painter.setBrush(color)
        painter.drawEllipse(x - 3, y - 3, 6, 6)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class PersonPreviewWidget(QWidget):
    """Persistent panel showing a zoomed, aspect-correct crop of the selected track.

    Updated on every ``update_frame()`` call.  Shows empty state when no track
    is selected or when the selected track has no detection in the current frame.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._track_id: int | None = None
        self._person_name: str | None = None

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setMinimumSize(160, 120)

        self._name_label = QLabel()
        self._name_label.setAlignment(Qt.AlignCenter)
        self._name_label.setStyleSheet("font-weight: bold; font-size: 11px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.addWidget(self._image_label, stretch=1)
        layout.addWidget(self._name_label)

        self._show_empty("No track selected")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_track(self, track_id: int | None, person_name: str | None = None) -> None:
        """Set the track to preview.

        Immediately clears the display; the crop appears on the next
        ``update_frame()`` call.

        Args:
            track_id: Detection track ID to preview, or None to clear.
            person_name: Assigned person name for the colour tint label, or None.
        """
        self._track_id = track_id
        self._person_name = person_name
        if track_id is None:
            self._show_empty("No track selected")
        else:
            self._show_empty("–")

    def update_frame(
        self,
        frame_bgr: np.ndarray | None,
        detections: list[dict],
        keypoints: dict[int, np.ndarray],
    ) -> None:
        """Update the displayed crop from the current frame's detection data.

        Shows empty state if the selected track is absent from this frame or if
        *frame_bgr* is None.

        Args:
            frame_bgr: Full video frame as a BGR uint8 numpy array, or None if
                the frame could not be decoded.
            detections: All person detections in this frame.
            keypoints: Keypoint arrays keyed by track_id.
        """
        if self._track_id is None or frame_bgr is None:
            self._show_empty("–")
            return

        bbox = bbox_from_detections(self._track_id, detections)
        if bbox is None:
            self._show_empty("–")
            return

        cx, cy, bw, bh = bbox
        fh, fw = frame_bgr.shape[:2]
        x1, y1, x2, y2 = compute_crop_rect(cx, cy, bw, bh, fw, fh)

        crop = frame_bgr[y1:y2, x1:x2].copy()

        kp = keypoints.get(self._track_id)
        if kp is not None:
            draw_skeleton_on_crop(crop, kp, x1, y1)

        self._show_crop(crop)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _show_empty(self, text: str) -> None:
        """Display a grey placeholder with *text*."""
        self._image_label.clear()
        self._image_label.setText(text)
        self._image_label.setStyleSheet("color: #888; font-size: 11px;")
        self._name_label.setText("")

    def _show_crop(self, crop_bgr: np.ndarray) -> None:
        """Convert *crop_bgr* to a scaled QPixmap and display it."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w = crop_rgb.shape[:2]
        qimg = QImage(crop_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to widget size, keeping aspect ratio; background fills the rest
        target = self._image_label.size()
        scaled = pixmap.scaled(
            target, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._image_label.setPixmap(scaled)
        self._image_label.setStyleSheet("")

        # Name label: use person colour as text colour for a subtle tint
        if self._person_name:
            c = person_color(self._person_name)
            self._name_label.setText(self._person_name)
            self._name_label.setStyleSheet(
                f"font-weight: bold; font-size: 11px; color: {c.name()};"
            )
        else:
            self._name_label.setText(f"track {self._track_id}")
            self._name_label.setStyleSheet("font-size: 11px; color: #555;")

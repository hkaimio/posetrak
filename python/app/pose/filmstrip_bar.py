"""filmstrip_bar.py — FilmstripBarItem: one detection-track segment rendered as a filmstrip."""
from __future__ import annotations

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsObject

from app.pose.colors import person_color

# Bottom colour strip height (pixels).
LABEL_H: int = 6

_BG_COLOR = QColor(50, 50, 50)
_EMPTY_COLOR = QColor(80, 80, 80)           # bar with no thumbnails yet
_SEL_PEN = QPen(QColor(0, 0, 0), 2)
_SEL_FILL = QColor(100, 160, 255, 80)       # blue tint — selected sub-range
_CONFLICT_FILL = QColor(255, 60, 60, 80)    # red tint — overlap conflict
_NAME_FONT = QFont("monospace", 7, QFont.Weight.Bold)


class FilmstripBarItem(QGraphicsObject):
    """A QGraphicsObject representing one detection-track segment.

    Visual layers (bottom to top):
      1. Dark background
      2. JPEG thumbnails tiled left-to-right (consecutive, variable-width)
      3. Person colour strip + name at bottom (LABEL_H px, only when assigned)
      4. Conflict overlay — semi-transparent red
      5. Selection sub-range overlay — semi-transparent blue rect
      6. Selection border — black, 2 px

    Thumbnails are injected by the parent widget via ``set_thumbnails()``;
    this item never queries the database itself.

    Coordinate helpers ``frame_to_local_x`` and ``local_x_to_frame`` use a
    linear interpolation between seg_first/seg_last.  They are used by the
    parent widget to position the selection overlay and to map mouse-click
    positions to frame indices.
    """

    def __init__(
        self,
        svid: str,
        tid: int,
        seg_first: int,
        seg_last: int,
        width: float,
        row_h: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.svid = svid
        self.tid = tid
        self.seg_first = seg_first
        self.seg_last = seg_last

        self._width = max(2.0, float(width))
        self._row_h = row_h

        # Thumbnail cache: frame_idx → QPixmap already scaled to (row_h - LABEL_H) tall
        self._thumbs: dict[int, QPixmap] = {}
        # Frame indices in display order (sorted ascending = left to right)
        self._thumb_order: list[int] = []

        self._person_name: str | None = None
        # List of (frame_first, frame_last) ranges with conflict overlaps.
        # Only those ranges are highlighted; the rest of the bar is unaffected.
        self._conflict_ranges: list[tuple[int, int]] = []

        # Sub-range selection (None when full bar is selected or nothing selected)
        self._sel_first: int | None = None
        self._sel_last: int | None = None
        self._is_selected: bool = False

        self.setAcceptHoverEvents(False)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self._width, float(self._row_h))

    def resize(self, new_width: float, new_row_h: int | None = None) -> None:
        """Update bar dimensions; triggers a repaint."""
        self.prepareGeometryChange()
        self._width = max(2.0, float(new_width))
        if new_row_h is not None:
            self._row_h = new_row_h
        self.update()

    # ------------------------------------------------------------------
    # State setters (all trigger update())
    # ------------------------------------------------------------------

    def set_assignment(self, person_name: str | None) -> None:
        self._person_name = person_name
        self.update()

    def set_thumbnails(self, thumbs: dict[int, QPixmap]) -> None:
        """Replace the thumbnail cache and repaint."""
        self._thumbs = dict(thumbs)
        self._thumb_order = sorted(thumbs.keys())
        self.update()

    def set_conflict_ranges(self, ranges: list[tuple[int, int]]) -> None:
        """Set the frame ranges within this bar that have conflict overlaps.

        Only these ranges are highlighted red; the remainder of the bar is
        unaffected.  Pass an empty list to clear all conflict highlights.
        """
        self._conflict_ranges = list(ranges)
        self.update()

    def set_conflict(self, is_conflict: bool) -> None:
        """Convenience: mark the whole bar or clear all conflict ranges."""
        self._conflict_ranges = [(self.seg_first, self.seg_last)] if is_conflict else []
        self.update()

    def set_selection(
        self,
        sel_first: int | None,
        sel_last: int | None,
        is_selected: bool,
    ) -> None:
        """Update selection state.

        *sel_first* / *sel_last* define a highlighted sub-range within the bar
        (used when the user drag-selects a portion).  Pass both as None to
        indicate the full bar is selected (border only, no range overlay).
        *is_selected* controls whether the selection border is drawn.
        """
        self._sel_first = sel_first
        self._sel_last = sel_last
        self._is_selected = is_selected
        self.update()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def frame_to_local_x(self, frame: int) -> float:
        """Map a frame index to a local x coordinate (linear interpolation)."""
        span = max(1, self.seg_last - self.seg_first)
        frac = (frame - self.seg_first) / span
        return max(0.0, min(self._width, frac * self._width))

    def local_x_to_frame(self, local_x: float) -> int:
        """Map a local x coordinate to the nearest frame index."""
        frac = max(0.0, min(1.0, local_x / self._width))
        return self.seg_first + round(frac * (self.seg_last - self.seg_first))

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paint(self, painter, option, widget) -> None:  # noqa: ARG002
        w = self._width
        h = float(self._row_h)
        label_h = float(LABEL_H) if self._person_name else 0.0
        film_h = h - label_h

        # 1. Background
        painter.fillRect(QRectF(0.0, 0.0, w, h), _BG_COLOR)

        # 2. Thumbnails — placed consecutively left-to-right, clipped to bar width
        if self._thumb_order:
            x_cursor = 0.0
            for fi in self._thumb_order:
                pix = self._thumbs.get(fi)
                if pix is None or pix.isNull() or pix.height() == 0:
                    continue
                # Natural width at film_h height
                thumb_w = float(pix.width()) * film_h / float(pix.height())
                if thumb_w <= 0:
                    continue
                avail_w = w - x_cursor
                if avail_w <= 0:
                    break
                if thumb_w <= avail_w:
                    # Full thumbnail fits
                    painter.drawPixmap(
                        QRectF(x_cursor, 0.0, thumb_w, film_h),
                        pix,
                        QRectF(0.0, 0.0, pix.width(), pix.height()),
                    )
                else:
                    # Clip at bar right edge — only draw left portion of this thumb
                    src_clip_w = int(avail_w * pix.width() / thumb_w)
                    painter.drawPixmap(
                        QRectF(x_cursor, 0.0, avail_w, film_h),
                        pix,
                        QRectF(0.0, 0.0, src_clip_w, pix.height()),
                    )
                    x_cursor += avail_w
                    break
                x_cursor += thumb_w
        else:
            # No thumbnails yet — fill with slightly lighter background
            painter.fillRect(QRectF(0.0, 0.0, w, film_h), _EMPTY_COLOR)

        # 3. Person colour strip + name
        if self._person_name and label_h > 0.0:
            color = person_color(self._person_name)
            painter.fillRect(QRectF(0.0, film_h, w, label_h), color)
            # Name text sits above the colour strip, inside the filmstrip area
            painter.setFont(_NAME_FONT)
            painter.setPen(QColor(240, 240, 240))
            text_rect = QRectF(4.0, max(0.0, film_h - 13.0), max(0.0, w - 6.0), 13.0)
            if text_rect.width() > 10:
                painter.drawText(
                    text_rect,
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    self._person_name,
                )

        # 4. Conflict overlay — drawn only over the actual conflicting frame ranges
        for cf, cl in self._conflict_ranges:
            x1 = self.frame_to_local_x(cf)
            x2 = self.frame_to_local_x(cl)
            if x2 > x1:
                painter.fillRect(QRectF(x1, 0.0, x2 - x1, h), _CONFLICT_FILL)

        # 5. Sub-range selection overlay (only when both bounds are set)
        if self._sel_first is not None and self._sel_last is not None:
            x1 = self.frame_to_local_x(self._sel_first)
            x2 = self.frame_to_local_x(self._sel_last)
            if x2 > x1:
                painter.fillRect(QRectF(x1, 0.0, x2 - x1, h), _SEL_FILL)

        # 6. Selection border
        if self._is_selected:
            painter.setPen(_SEL_PEN)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(1.0, 1.0, w - 2.0, h - 2.0))


def decode_jpeg_to_pixmap(jpeg_bytes: bytes, target_h: int) -> QPixmap | None:
    """Decode *jpeg_bytes* and scale to *target_h* height, preserving aspect ratio.

    Returns None if the bytes cannot be decoded.
    """
    pix = QPixmap()
    if not pix.loadFromData(QByteArray(jpeg_bytes)):
        return None
    if pix.isNull() or pix.height() == 0:
        return None
    target_w = int(pix.width() * target_h / pix.height())
    if target_w <= 0:
        return None
    return pix.scaled(
        target_w,
        target_h,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )

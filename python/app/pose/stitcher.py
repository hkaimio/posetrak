"""stitcher.py — Timeline widget showing per-camera track segments."""
from __future__ import annotations

import sqlite3

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from app.pose.db_cache import read_track_spans


ROW_HEIGHT = 16
ROW_GAP = 4          # gap between cameras
LABEL_WIDTH = 80
PX_PER_FRAME = 1     # default horizontal scale

_UNASSIGNED_COLOR = QColor(120, 120, 120)


def _person_color(name: str) -> QColor:
    hue = hash(name) % 360
    color = QColor.fromHsvF(hue / 360.0, 0.7, 0.9)
    return color


class StitcherWidget(QGraphicsView):
    """QGraphicsView-based timeline showing track segments per camera."""

    segment_clicked = Signal(str, int, int, int)  # shot_video_id, track_id, first_frame, last_frame

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # (shot_video_id, track_id) -> QGraphicsRectItem
        self._items: dict[tuple[str, int], QGraphicsRectItem] = {}
        # (shot_video_id, track_id) -> (first_frame, last_frame)
        self._spans: dict[tuple[str, int], tuple[int, int]] = {}
        # assignments: (shot_video_id, track_id) -> person_name
        self._assignments: dict[tuple[str, int], str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_run(self, session: sqlite3.Connection, detection_run_id: str) -> None:
        """Load track spans for all cameras in this run."""
        self.clear()

        # Get all shot_video_ids for this run
        rows = session.execute(
            "SELECT DISTINCT shot_video_id FROM person_tracks "
            "WHERE detection_run_id = ? ORDER BY shot_video_id",
            (detection_run_id,),
        ).fetchall()
        svids = [r["shot_video_id"] for r in rows]

        # Get camera labels
        cam_labels: dict[str, str] = {}
        for svid in svids:
            row = session.execute(
                "SELECT ci.label FROM shot_videos sv "
                "JOIN camera_instances ci ON ci.id = sv.camera_instance_id "
                "WHERE sv.id = ?",
                (svid,),
            ).fetchone()
            cam_labels[svid] = row["label"] if row and row["label"] else svid[:8]

        y = 0
        for svid in svids:
            spans = read_track_spans(session, detection_run_id, svid)

            # Camera header label
            label_item = self._scene.addText(cam_labels[svid])
            label_item.setPos(0, y)
            label_item.setDefaultTextColor(QColor(220, 220, 220))

            for span in spans:
                tid = span["track_id"]
                first = span["first_frame"]
                last = span["last_frame"]
                self._spans[(svid, tid)] = (first, last)

                x = LABEL_WIDTH + first * PX_PER_FRAME
                w = max(2, (last - first) * PX_PER_FRAME)
                rect = self._scene.addRect(
                    x, y, w, ROW_HEIGHT,
                    QPen(Qt.NoPen),
                    QBrush(_UNASSIGNED_COLOR),
                )
                rect.setToolTip(f"track {tid}  frames {first}–{last}")
                rect.setData(0, svid)
                rect.setData(1, tid)
                rect.setData(2, first)
                rect.setData(3, last)
                rect.setAcceptHoverEvents(True)
                self._items[(svid, tid)] = rect

                y += ROW_HEIGHT + 2

            y += ROW_GAP

        self._scene.setSceneRect(self._scene.itemsBoundingRect())

    def set_assignment(self, shot_video_id: str, track_id: int, person_name: str | None) -> None:
        """Colour a track segment by person assignment (None = unassigned)."""
        key = (shot_video_id, track_id)
        if person_name is None:
            self._assignments.pop(key, None)
            color = _UNASSIGNED_COLOR
        else:
            self._assignments[key] = person_name
            color = _person_color(person_name)

        item = self._items.get(key)
        if item is not None:
            item.setBrush(QBrush(color))

    def clear(self) -> None:
        """Clear all items and internal state."""
        self._scene.clear()
        self._items.clear()
        self._spans.clear()
        self._assignments.clear()

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        pos = self.mapToScene(event.pos())
        item = self._scene.itemAt(pos, self.transform())
        if item is not None and isinstance(item, QGraphicsRectItem):
            svid = item.data(0)
            tid = item.data(1)
            first = item.data(2)
            last = item.data(3)
            if svid is not None:
                self.segment_clicked.emit(svid, tid, first, last)
        super().mousePressEvent(event)

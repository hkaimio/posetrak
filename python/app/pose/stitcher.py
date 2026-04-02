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
ROW_GAP = 4          # vertical gap between cameras
LABEL_WIDTH = 80
PX_PER_SEC = 30      # horizontal scale: pixels per second of global time

_UNASSIGNED_COLOR = QColor(120, 120, 120)


def _person_color(name: str) -> QColor:
    hue = hash(name) % 360
    return QColor.fromHsvF(hue / 360.0, 0.7, 0.9)


def _build_frame_to_time(
    session: sqlite3.Connection,
    sync_config_id: str,
    shot_video_ids: list[str],
) -> dict[str, tuple[int, float, float]]:
    """Return {shot_video_id: (ref_frame, ref_timestamp_s, fps)}.

    Uses the first sync_point per video as the anchor; fps from shot_videos.
    """
    result: dict[str, tuple[int, float, float]] = {}
    for svid in shot_video_ids:
        row = session.execute(
            "SELECT sp.video_frame, sp.timestamp_s, sv.actual_fps "
            "FROM sync_points sp "
            "JOIN shot_videos sv ON sv.id = sp.shot_video_id "
            "WHERE sp.shot_video_id = ? AND sp.sync_config_id = ? "
            "ORDER BY sp.video_frame ASC LIMIT 1",
            (svid, sync_config_id),
        ).fetchone()
        if row:
            fps = float(row["actual_fps"] or 30.0)
            result[svid] = (int(row["video_frame"]), float(row["timestamp_s"]), fps)
    return result


def _frame_to_time(anchor: tuple[int, float, float], frame: int) -> float:
    ref_frame, ref_ts, fps = anchor
    return ref_ts + (frame - ref_frame) / fps


class StitcherWidget(QGraphicsView):
    """QGraphicsView-based timeline showing track segments per camera.

    The x-axis is global time in seconds (from the sync config).
    """

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
        self._time_origin: float = 0.0   # global time at x=LABEL_WIDTH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_run(self, session: sqlite3.Connection, detection_run_id: str) -> None:
        """Load track spans for all cameras in this run."""
        self.clear()

        run_row = session.execute(
            "SELECT sync_config_id, time_start_s FROM detection_runs WHERE id = ?",
            (detection_run_id,),
        ).fetchone()
        if run_row is None:
            return
        sync_config_id = run_row["sync_config_id"]
        self._time_origin = float(run_row["time_start_s"])

        rows = session.execute(
            "SELECT DISTINCT shot_video_id FROM person_tracks "
            "WHERE detection_run_id = ? ORDER BY shot_video_id",
            (detection_run_id,),
        ).fetchall()
        svids = [r["shot_video_id"] for r in rows]

        anchors = _build_frame_to_time(session, sync_config_id, svids)

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
            anchor = anchors.get(svid)

            label_item = self._scene.addText(cam_labels[svid])
            label_item.setPos(0, y)
            label_item.setDefaultTextColor(QColor(220, 220, 220))

            for span in spans:
                tid = span["track_id"]
                first = span["first_frame"]
                last = span["last_frame"]
                self._spans[(svid, tid)] = (first, last)

                if anchor is not None:
                    t0 = _frame_to_time(anchor, first)
                    t1 = _frame_to_time(anchor, last)
                else:
                    t0 = float(first)
                    t1 = float(last)

                x = LABEL_WIDTH + (t0 - self._time_origin) * PX_PER_SEC
                w = max(2, (t1 - t0) * PX_PER_SEC)

                rect = self._scene.addRect(
                    x, y, w, ROW_HEIGHT,
                    QPen(Qt.NoPen),
                    QBrush(_UNASSIGNED_COLOR),
                )
                t0_mm = int(t0 // 60)
                t0_ss = t0 % 60
                t1_mm = int(t1 // 60)
                t1_ss = t1 % 60
                rect.setToolTip(
                    f"track {tid}  frames {first}–{last}\n"
                    f"{t0_mm:02d}:{t0_ss:05.2f} – {t1_mm:02d}:{t1_ss:05.2f}"
                )
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
        key = (shot_video_id, track_id)
        color = _person_color(person_name) if person_name else _UNASSIGNED_COLOR
        if person_name is None:
            self._assignments.pop(key, None)  # type: ignore[attr-defined]
        item = self._items.get(key)
        if item is not None:
            item.setBrush(QBrush(color))

    def clear(self) -> None:
        self._scene.clear()
        self._items.clear()
        self._spans.clear()

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        pos = self.mapToScene(event.pos())
        item = self._scene.itemAt(pos, self.transform())
        if isinstance(item, QGraphicsRectItem):
            svid = item.data(0)
            tid = item.data(1)
            first = item.data(2)
            last = item.data(3)
            if svid is not None:
                self.segment_clicked.emit(svid, tid, first, last)
        super().mousePressEvent(event)

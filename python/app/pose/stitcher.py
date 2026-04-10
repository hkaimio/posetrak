"""stitcher.py — Timeline widget showing per-camera track segments."""
from __future__ import annotations

import sqlite3

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QInputDialog,
    QMenu,
)

from app.pose.colors import UNASSIGNED_COLOR, person_color
from app.pose.db_cache import read_track_spans


ROW_HEIGHT = 16
ROW_GAP = 4          # vertical gap between cameras
LABEL_WIDTH = 80
_PX_PER_SEC_MIN = 5
_PX_PER_SEC_MAX = 500

_SELECTED_PEN = QPen(QColor(0, 0, 0), 2)
_NAME_FONT = QFont("monospace", 8, QFont.Bold)


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

    Signals:
        segment_clicked(shot_video_id, track_id, first_frame, last_frame):
            emitted on left-click of a track bar.
        assignment_changed(shot_video_id, track_id, person_name_or_None):
            emitted when the user assigns or detaches a person via the
            context menu (this segment only).  person_name is None for "Detach".
        assignment_from_here(shot_video_id, track_id, person_name):
            emitted when the user picks "From here onwards" in the context
            menu; person_name is never None for this signal.
    """

    segment_clicked = Signal(str, int, int, int)
    assignment_changed = Signal(str, int, object)   # svid, tid, str|None
    assignment_from_here = Signal(str, int, str)    # svid, tid, person_name
    time_clicked = Signal(float)                    # global_s at the clicked x position

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

        # (shot_video_id, track_id) -> QGraphicsRectItem
        self._items: dict[tuple[str, int], QGraphicsRectItem] = {}
        # (shot_video_id, track_id) -> (first_frame, last_frame)
        self._spans: dict[tuple[str, int], tuple[int, int]] = {}
        # (shot_video_id, track_id) -> person_name
        self._assignments: dict[tuple[str, int], str] = {}
        # (shot_video_id, track_id) -> QGraphicsSimpleTextItem
        self._name_items: dict[tuple[str, int], QGraphicsSimpleTextItem] = {}

        self._selected_key: tuple[str, int] | None = None
        self._time_line: QGraphicsLineItem | None = None
        self._time_origin: float = 0.0
        self._total_duration_s: float = 0.0

        # Known persons for context menu; updated by set_known_persons()
        self._persons: list[str] = []

        # Saved args for rebuild on resize
        self._last_session: sqlite3.Connection | None = None
        self._last_run_id: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_spans(self) -> dict[tuple[str, int], tuple[int, int]]:
        """Return {(shot_video_id, track_id): (first_frame, last_frame)} for all loaded tracks."""
        return dict(self._spans)

    def set_known_persons(self, persons: list[str]) -> None:
        """Update the person list shown in the assignment context menu."""
        self._persons = list(persons)

    def set_current_time(self, global_s: float) -> None:
        """Move the red playhead line to *global_s*."""
        if self._time_line is None:
            return
        pps = self._px_per_sec
        x = LABEL_WIDTH + (global_s - self._time_origin) * pps
        h = self._scene.sceneRect().height()
        self._time_line.setLine(x, 0, x, h)

    @property
    def _px_per_sec(self) -> float:
        """Pixels per second, fitted to the current widget width."""
        available = max(1, self.viewport().width() - LABEL_WIDTH)
        dur = self._total_duration_s
        if dur > 0:
            fitted = available / dur
            return max(_PX_PER_SEC_MIN, min(_PX_PER_SEC_MAX, fitted))
        return 30.0

    def load_run(self, session: sqlite3.Connection, detection_run_id: str) -> None:
        """Load track spans for all cameras in this run."""
        self._last_session = session
        self._last_run_id = detection_run_id
        self._assignments.clear()
        self._selected_key = None
        self._rebuild()

    def set_assignment(self, shot_video_id: str, track_id: int, person_name: str | None) -> None:
        """Update the colour, name label, and internal state for one track."""
        key = (shot_video_id, track_id)
        if person_name:
            self._assignments[key] = person_name
        else:
            self._assignments.pop(key, None)

        color = person_color(person_name) if person_name else UNASSIGNED_COLOR
        item = self._items.get(key)
        if item is None:
            return
        item.setBrush(QBrush(color))

        # Update name label — create if needed, remove if detached
        name_item = self._name_items.get(key)
        if person_name:
            r = item.rect()
            if name_item is None:
                name_item = QGraphicsSimpleTextItem()
                name_item.setFont(_NAME_FONT)
                name_item.setBrush(QBrush(QColor(255, 255, 255)))
                self._scene.addItem(name_item)
                self._name_items[key] = name_item
            name_item.setText(person_name)
            # Draw inside bar with small left margin; clip visually by bar width
            name_item.setPos(r.x() + 4, r.y() + 1)
        else:
            if name_item is not None:
                self._scene.removeItem(name_item)
                del self._name_items[key]

    def clear(self) -> None:
        self._scene.clear()
        self._items.clear()
        self._spans.clear()
        self._name_items.clear()
        self._time_line = None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._last_session is not None and self._last_run_id is not None:
            self._rebuild()

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        """Rebuild the scene from stored session + run_id."""
        session = self._last_session
        detection_run_id = self._last_run_id
        # Preserve assignments across rebuilds
        saved_assignments = dict(self._assignments)
        self.clear()
        if session is None or detection_run_id is None:
            return

        run_row = session.execute(
            "SELECT sync_config_id, time_start_s, time_end_s FROM detection_runs WHERE id = ?",
            (detection_run_id,),
        ).fetchone()
        if run_row is None:
            return
        sync_config_id = run_row["sync_config_id"]
        self._time_origin = float(run_row["time_start_s"])
        self._total_duration_s = max(0.0, float(run_row["time_end_s"]) - float(run_row["time_start_s"]))

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

        pps = self._px_per_sec

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

                x = LABEL_WIDTH + (t0 - self._time_origin) * pps
                w = max(2, (t1 - t0) * pps)

                key = (svid, tid)
                is_selected = (key == self._selected_key)
                pen = _SELECTED_PEN if is_selected else QPen(Qt.NoPen)

                rect = self._scene.addRect(x, y, w, ROW_HEIGHT, pen, QBrush(UNASSIGNED_COLOR))
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
                self._items[key] = rect

                y += ROW_HEIGHT + 2

            y += ROW_GAP

        self._scene.setSceneRect(self._scene.itemsBoundingRect())

        # Restore assignments (colour + name labels)
        self._assignments = saved_assignments
        for key, name in saved_assignments.items():
            svid, tid = key
            self.set_assignment(svid, tid, name)

        # Red playhead line (added last so it draws on top)
        scene_h = max(self._scene.sceneRect().height(), 1)
        x0 = LABEL_WIDTH + 0.0
        self._time_line = self._scene.addLine(x0, 0, x0, scene_h, QPen(QColor(220, 40, 40), 2))
        self._time_line.setZValue(10)

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            item = self._scene.itemAt(scene_pos, self.transform())
            if isinstance(item, QGraphicsRectItem):
                svid = item.data(0)
                tid = item.data(1)
                first = item.data(2)
                last = item.data(3)
                if svid is not None:
                    self._set_selected(svid, tid)
                    # Seek to the clicked time position, not necessarily the bar start
                    pps = self._px_per_sec
                    global_s = self._time_origin + (scene_pos.x() - LABEL_WIDTH) / max(pps, 1e-6)
                    self.time_clicked.emit(global_s)
                    self.segment_clicked.emit(svid, tid, first, last)
        super().mousePressEvent(event)

    def _on_context_menu(self, pos) -> None:
        scene_pos = self.mapToScene(pos)
        item = self._scene.itemAt(scene_pos, self.transform())
        if not isinstance(item, QGraphicsRectItem):
            return
        svid = item.data(0)
        tid = item.data(1)
        if svid is None:
            return

        self._set_selected(svid, tid)

        menu = QMenu(self)
        current = self._assignments.get((svid, tid))

        # ---- This segment only ----
        for name in self._persons:
            action = menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.triggered.connect(lambda checked, n=name: self._assign(svid, tid, n))

        menu.addSeparator()
        new_action = menu.addAction("New person…")
        new_action.triggered.connect(lambda: self._new_person(svid, tid))
        if current:
            detach_action = menu.addAction("Detach")
            detach_action.triggered.connect(lambda: self._assign(svid, tid, None))

        # ---- From here onwards submenu ----
        menu.addSeparator()
        from_here_menu = menu.addMenu("From here onwards")
        for name in self._persons:
            action = from_here_menu.addAction(name)
            action.triggered.connect(lambda checked, n=name: self._assign_from_here(svid, tid, n))
        from_here_menu.addSeparator()
        new_fh = from_here_menu.addAction("New person…")
        new_fh.triggered.connect(lambda: self._new_person_from_here(svid, tid))

        menu.exec(self.viewport().mapToGlobal(pos))

    def _assign(self, svid: str, tid: int, person_name: str | None) -> None:
        # Emit only — main window handles conflict check and calls set_assignment back.
        self.assignment_changed.emit(svid, tid, person_name)

    def _assign_from_here(self, svid: str, tid: int, person_name: str) -> None:
        self.assignment_from_here.emit(svid, tid, person_name)

    def _new_person(self, svid: str, tid: int) -> None:
        name, ok = QInputDialog.getText(self, "New person", "Person name:")
        if ok and name.strip():
            name = name.strip()
            if name not in self._persons:
                self._persons.append(name)
            self._assign(svid, tid, name)

    def _new_person_from_here(self, svid: str, tid: int) -> None:
        name, ok = QInputDialog.getText(self, "New person", "Person name:")
        if ok and name.strip():
            name = name.strip()
            if name not in self._persons:
                self._persons.append(name)
            self._assign_from_here(svid, tid, name)

    def _set_selected(self, svid: str, tid: int) -> None:
        """Highlight the given bar; remove highlight from previously selected."""
        prev = self._selected_key
        new_key = (svid, tid)
        if prev == new_key:
            return
        # Deselect old
        if prev is not None:
            old_item = self._items.get(prev)
            if old_item is not None:
                old_item.setPen(QPen(Qt.NoPen))
        # Select new
        self._selected_key = new_key
        new_item = self._items.get(new_key)
        if new_item is not None:
            new_item.setPen(_SELECTED_PEN)

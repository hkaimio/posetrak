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

from app.setup.db_context import SyncTable
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
            "JOIN capture_videos sv ON sv.id = sp.shot_video_id "
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

    Each detection track is displayed as one or more segments on the same row.
    A segment is a contiguous frame range within one track; tracks start as a
    single segment and can be split at the playhead position into two
    independently assignable segments.

    Signals
    -------
    segment_clicked(svid, tid, first_frame, last_frame):
        Emitted on left-click of a segment bar.  ``first_frame`` and
        ``last_frame`` are the segment's own frame range (not the full track).
    assignment_changed(svid, tid, seg_first, person_or_None):
        Emitted when the user assigns or detaches a segment via the context
        menu.  ``seg_first`` identifies the segment; ``person_or_None`` is
        ``None`` for a detach operation.
    split_requested(svid, tid, seg_first, split_frame):
        Emitted when the user selects "Split here" from the context menu.
        ``seg_first`` identifies the segment; ``split_frame`` is the frame
        at which to split (becomes the first frame of the right half).
    time_clicked(global_s):
        Emitted with the global timestamp corresponding to the clicked
        x position, regardless of which segment was clicked.
    """

    segment_clicked = Signal(str, int, int, int)        # svid, tid, first_frame, last_frame
    assignment_changed = Signal(str, int, int, object)  # svid, tid, seg_first, person|None
    split_requested = Signal(str, int, int, int)        # svid, tid, seg_first, split_frame
    time_clicked = Signal(float)                        # global_s at clicked x

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

        # (svid, tid) → [(seg_first, seg_last), ...] sorted by seg_first
        self._segments: dict[tuple[str, int], list[tuple[int, int]]] = {}
        # (svid, tid, seg_first) → person_name
        self._seg_assignments: dict[tuple[str, int, int], str] = {}
        # (svid, tid, seg_first) → QGraphicsRectItem
        self._items: dict[tuple[str, int, int], QGraphicsRectItem] = {}
        # (svid, tid, seg_first) → QGraphicsSimpleTextItem
        self._name_items: dict[tuple[str, int, int], QGraphicsSimpleTextItem] = {}
        # svid → (ref_frame, ref_ts, fps)  — fallback when no SyncTable
        self._anchors: dict[str, tuple[int, float, float]] = {}
        self._sync_table: SyncTable | None = None
        # (svid, tid) → y pixel coordinate in scene for this detection's row
        self._row_y: dict[tuple[str, int], float] = {}

        self._selected_key: tuple[str, int, int] | None = None
        self._current_time_s: float = 0.0
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

    def set_sync_table(self, table: SyncTable | None) -> None:
        """Set the SyncTable used for frame → global-time conversion in the timeline.

        When set, segment bar positions are computed from piecewise-linear
        SyncTable interpolation rather than the single-anchor + nominal-fps
        approximation stored in _anchors.  Must be called before load_run.
        """
        self._sync_table = table

    def get_spans(self) -> dict[tuple[str, int, int], tuple[int, int]]:
        """Return segment-keyed frame spans: {(svid, tid, seg_first): (first, last)}."""
        return {
            (svid, tid, sf): (sf, sl)
            for (svid, tid), segs in self._segments.items()
            for sf, sl in segs
        }

    def get_time_spans(self) -> dict[tuple[str, int, int], tuple[float, float]]:
        """Return segment-keyed time spans: {(svid, tid, seg_first): (t0_s, t1_s)}."""
        result: dict[tuple[str, int, int], tuple[float, float]] = {}
        for (svid, tid), segs in self._segments.items():
            anchor = self._anchors.get(svid)
            for sf, sl in segs:
                if self._sync_table is not None:
                    t0 = self._sync_table.frame_to_global_time(sf, svid)
                    t1 = self._sync_table.frame_to_global_time(sl, svid)
                    t0 = t0 if t0 is not None else float(sf)
                    t1 = t1 if t1 is not None else float(sl)
                elif anchor is not None:
                    t0 = _frame_to_time(anchor, sf)
                    t1 = _frame_to_time(anchor, sl)
                else:
                    t0, t1 = float(sf), float(sl)
                result[(svid, tid, sf)] = (t0, t1)
        return result

    def set_known_persons(self, persons: list[str]) -> None:
        """Update the person list shown in the assignment context menu."""
        self._persons = list(persons)

    def set_current_time(self, global_s: float) -> None:
        """Move the red playhead line to *global_s* and store the current time."""
        self._current_time_s = global_s
        if self._time_line is None:
            return
        pps = self._px_per_sec
        x = LABEL_WIDTH + (global_s - self._time_origin) * pps
        h = self._scene.sceneRect().height()
        self._time_line.setLine(x, 0, x, h)

    def set_segment_assignment(
        self,
        svid: str,
        tid: int,
        seg_first: int,
        person_name: str | None,
    ) -> None:
        """Update the colour, name label, and internal state for one segment.

        Parameters
        ----------
        svid: Shot video ID.
        tid: Track ID.
        seg_first: First frame of the segment (its identifier within the track).
        person_name: Person to assign, or None to detach.
        """
        seg_key = (svid, tid, seg_first)
        if person_name:
            self._seg_assignments[seg_key] = person_name
        else:
            self._seg_assignments.pop(seg_key, None)
        self._update_seg_visuals(svid, tid, seg_first, person_name)

    def split_segment(self, svid: str, tid: int, seg_first: int, split_frame: int) -> None:
        """Split the segment at *split_frame*, creating two independently assignable segments.

        The left half [seg_first, split_frame-1] inherits the original
        segment's assignment.  The right half [split_frame, seg_last] starts
        unassigned.  Both halves remain on the same row in the timeline.

        Parameters
        ----------
        svid: Shot video ID.
        tid: Track ID.
        seg_first: First frame of the segment to split.
        split_frame: Frame at which to split; becomes the first frame of the
            right half.  Must satisfy seg_first < split_frame <= seg_last.
        """
        key = (svid, tid)
        segs = self._segments.get(key, [])
        for i, (sf, sl) in enumerate(segs):
            if sf != seg_first:
                continue
            if not (sf < split_frame <= sl):
                return  # split point not strictly inside segment

            old_seg_key = (svid, tid, sf)

            # Remove old visual items
            old_item = self._items.pop(old_seg_key, None)
            if old_item is not None:
                self._scene.removeItem(old_item)
            old_name = self._name_items.pop(old_seg_key, None)
            if old_name is not None:
                self._scene.removeItem(old_name)

            # Update segment list in place
            segs[i] = (sf, split_frame - 1)
            segs.insert(i + 1, (split_frame, sl))

            # Right half inherits no assignment (no entry in _seg_assignments)
            # Left half keeps its existing entry (same key sf == seg_first)

            y = self._row_y.get(key, 0.0)
            self._draw_segment(svid, tid, sf, split_frame - 1, y)
            self._draw_segment(svid, tid, split_frame, sl, y)

            # Restore name label on the left half (removed with the old item)
            left_person = self._seg_assignments.get(old_seg_key)
            self._update_seg_visuals(svid, tid, sf, left_person)
            break

    def load_run(self, session: sqlite3.Connection, detection_run_id: str) -> None:
        """Load track spans for all cameras in this run; clear all segment and assignment state.

        Parameters
        ----------
        session: SQLite connection.
        detection_run_id: ID of the detection run to display.
        """
        self._last_session = session
        self._last_run_id = detection_run_id
        self._segments.clear()
        self._seg_assignments.clear()
        self._selected_key = None
        self._rebuild()

    def clear(self) -> None:
        """Remove all visual items from the scene.

        Does **not** clear segment or assignment state so that ``_rebuild()``
        can restore the display after a resize.
        """
        self._scene.clear()
        self._items.clear()
        self._name_items.clear()
        self._row_y.clear()
        self._time_line = None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._last_session is not None and self._last_run_id is not None:
            self._rebuild()

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    @property
    def _px_per_sec(self) -> float:
        """Pixels per second, fitted to the current widget width."""
        available = max(1, self.viewport().width() - LABEL_WIDTH)
        dur = self._total_duration_s
        if dur > 0:
            fitted = available / dur
            return max(_PX_PER_SEC_MIN, min(_PX_PER_SEC_MAX, fitted))
        return 30.0

    def _rebuild(self) -> None:
        """Rebuild all visual items from stored session data.

        Preserves existing segment splits and assignments across calls (e.g.
        on resize).  New detections that have no entry in ``_segments`` are
        initialised as a single full-range segment.
        """
        session = self._last_session
        detection_run_id = self._last_run_id

        self.clear()
        if session is None or detection_run_id is None:
            return

        run_row = session.execute(
            "SELECT shot_id, sync_config_id, time_start_s, time_end_s FROM detection_runs WHERE id = ?",
            (detection_run_id,),
        ).fetchone()
        if run_row is None:
            return

        sync_config_id = run_row["sync_config_id"]
        self._time_origin = float(run_row["time_start_s"])
        self._total_duration_s = max(
            0.0, float(run_row["time_end_s"]) - float(run_row["time_start_s"])
        )

        rows = session.execute(
            "SELECT id FROM capture_videos WHERE shot_id = ? ORDER BY id",
            (run_row["shot_id"],),
        ).fetchall()
        svids = [r["id"] for r in rows]

        self._anchors = _build_frame_to_time(session, sync_config_id, svids)

        cam_labels: dict[str, str] = {}
        for svid in svids:
            row = session.execute(
                "SELECT ci.label FROM capture_videos sv "
                "JOIN camera_instances ci ON ci.id = sv.camera_instance_id "
                "WHERE sv.id = ?",
                (svid,),
            ).fetchone()
            cam_labels[svid] = row["label"] if row and row["label"] else svid[:8]

        y = 0.0
        for svid in svids:
            db_spans = read_track_spans(session, detection_run_id, svid)

            label_item = self._scene.addText(cam_labels[svid])
            label_item.setPos(0, y)
            label_item.setDefaultTextColor(QColor(0, 0, 0))

            if db_spans:
                for span in db_spans:
                    tid = span["track_id"]
                    first = span["first_frame"]
                    last = span["last_frame"]
                    det_key = (svid, tid)

                    if det_key not in self._segments:
                        # Fresh load — initialise as one segment covering the full track
                        self._segments[det_key] = [(first, last)]

                    self._row_y[det_key] = y
                    for seg_first, seg_last in self._segments[det_key]:
                        self._draw_segment(svid, tid, seg_first, seg_last, y)

                    y += ROW_HEIGHT + 2
            else:
                y += ROW_HEIGHT

            y += ROW_GAP

        self._scene.setSceneRect(self._scene.itemsBoundingRect())

        # Restore assignment colours and name labels
        for (svid, tid, sf), name in self._seg_assignments.items():
            self._update_seg_visuals(svid, tid, sf, name)

        # Red playhead line (added last so it draws on top)
        scene_h = max(self._scene.sceneRect().height(), 1)
        x0 = LABEL_WIDTH + 0.0
        self._time_line = self._scene.addLine(x0, 0, x0, scene_h, QPen(QColor(220, 40, 40), 2))
        self._time_line.setZValue(10)

    def _draw_segment(
        self,
        svid: str,
        tid: int,
        seg_first: int,
        seg_last: int,
        y: float,
    ) -> QGraphicsRectItem:
        """Create and register one rect item for a segment.

        Parameters
        ----------
        svid: Shot video ID.
        tid: Track ID.
        seg_first: First frame of the segment.
        seg_last: Last frame of the segment.
        y: Vertical scene coordinate for the row.
        """
        pps = self._px_per_sec
        if self._sync_table is not None:
            t0 = self._sync_table.frame_to_global_time(seg_first, svid)
            t1 = self._sync_table.frame_to_global_time(seg_last, svid)
            t0 = t0 if t0 is not None else float(seg_first)
            t1 = t1 if t1 is not None else float(seg_last)
        else:
            anchor = self._anchors.get(svid)
            if anchor is not None:
                t0 = _frame_to_time(anchor, seg_first)
                t1 = _frame_to_time(anchor, seg_last)
            else:
                t0, t1 = float(seg_first), float(seg_last)

        x = LABEL_WIDTH + (t0 - self._time_origin) * pps
        w = max(2, (t1 - t0) * pps)

        seg_key = (svid, tid, seg_first)
        is_selected = (seg_key == self._selected_key)
        pen = _SELECTED_PEN if is_selected else QPen(Qt.NoPen)
        person = self._seg_assignments.get(seg_key)
        color = person_color(person) if person else UNASSIGNED_COLOR

        t0_mm, t0_ss = int(t0 // 60), t0 % 60
        t1_mm, t1_ss = int(t1 // 60), t1 % 60
        rect = self._scene.addRect(x, y, w, ROW_HEIGHT, pen, QBrush(color))
        rect.setToolTip(
            f"track {tid}  frames {seg_first}–{seg_last}\n"
            f"{t0_mm:02d}:{t0_ss:05.2f} – {t1_mm:02d}:{t1_ss:05.2f}"
        )
        rect.setData(0, svid)
        rect.setData(1, tid)
        rect.setData(2, seg_first)
        rect.setData(3, seg_last)
        rect.setAcceptHoverEvents(True)
        self._items[seg_key] = rect
        return rect

    def _update_seg_visuals(
        self,
        svid: str,
        tid: int,
        seg_first: int,
        person_name: str | None,
    ) -> None:
        """Update the colour and name label of one segment without changing state.

        Parameters
        ----------
        svid: Shot video ID.
        tid: Track ID.
        seg_first: First frame of the segment.
        person_name: Assigned person, or None to show the unassigned colour.
        """
        seg_key = (svid, tid, seg_first)
        item = self._items.get(seg_key)
        if item is None:
            return

        color = person_color(person_name) if person_name else UNASSIGNED_COLOR
        item.setBrush(QBrush(color))

        name_item = self._name_items.get(seg_key)
        if person_name:
            r = item.rect()
            if name_item is None:
                name_item = QGraphicsSimpleTextItem()
                name_item.setFont(_NAME_FONT)
                name_item.setBrush(QBrush(QColor(255, 255, 255)))
                # Don't intercept mouse events — let them fall through to the
                # rect item below so clicking on the name still selects the bar.
                name_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self._scene.addItem(name_item)
                self._name_items[seg_key] = name_item
            name_item.setText(person_name)
            name_item.setPos(r.x() + 4, r.y() + 1)
        else:
            if name_item is not None:
                self._scene.removeItem(name_item)
                del self._name_items[seg_key]

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            # Iterate all items under the cursor so that clicking a name text
            # label (which sits on top) still falls through to the rect below.
            for item in self._scene.items(scene_pos):
                if isinstance(item, QGraphicsRectItem):
                    svid = item.data(0)
                    tid = item.data(1)
                    seg_first = item.data(2)
                    seg_last = item.data(3)
                    if svid is not None:
                        self._set_selected(svid, tid, seg_first)
                        pps = self._px_per_sec
                        global_s = self._time_origin + (scene_pos.x() - LABEL_WIDTH) / max(pps, 1e-6)
                        # segment_clicked must fire before time_clicked so that
                        # main.py sets the preview track_id before seek_frame()
                        # triggers update_frame().
                        self.segment_clicked.emit(svid, tid, seg_first, seg_last)
                        self.time_clicked.emit(global_s)
                    break
        super().mousePressEvent(event)

    def _on_context_menu(self, pos) -> None:
        scene_pos = self.mapToScene(pos)
        # Use items() so that right-clicking a name text label finds the rect.
        rect_item = None
        for item in self._scene.items(scene_pos):
            if isinstance(item, QGraphicsRectItem):
                rect_item = item
                break
        if rect_item is None:
            return
        svid = rect_item.data(0)
        tid = rect_item.data(1)
        seg_first = rect_item.data(2)
        seg_last = rect_item.data(3)
        if svid is None:
            return

        self._set_selected(svid, tid, seg_first)

        menu = QMenu(self)
        current = self._seg_assignments.get((svid, tid, seg_first))

        for name in self._persons:
            action = menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.triggered.connect(
                lambda checked, n=name: self._assign(svid, tid, seg_first, n)
            )

        menu.addSeparator()
        new_action = menu.addAction("New person…")
        new_action.triggered.connect(lambda: self._new_person(svid, tid, seg_first))
        if current:
            detach_action = menu.addAction("Detach")
            detach_action.triggered.connect(
                lambda: self._assign(svid, tid, seg_first, None)
            )

        # "Split here" — only offered when the playhead is strictly inside this segment
        anchor = self._anchors.get(svid)
        if anchor is not None:
            t0 = _frame_to_time(anchor, seg_first)
            t1 = _frame_to_time(anchor, seg_last)
            if t0 < self._current_time_s < t1:
                # Compute the video frame corresponding to the current playhead
                split_frame = int(round(
                    anchor[0] + (self._current_time_s - anchor[1]) * anchor[2]
                ))
                if seg_first < split_frame <= seg_last:
                    menu.addSeparator()
                    split_action = menu.addAction("Split here")
                    split_action.triggered.connect(
                        lambda: self.split_requested.emit(svid, tid, seg_first, split_frame)
                    )

        menu.exec(self.viewport().mapToGlobal(pos))

    def _assign(self, svid: str, tid: int, seg_first: int, person_name: str | None) -> None:
        """Emit assignment_changed for one segment; main window handles conflict check and state update."""
        self.assignment_changed.emit(svid, tid, seg_first, person_name)

    def _new_person(self, svid: str, tid: int, seg_first: int) -> None:
        """Prompt for a new person name and emit an assignment for one segment.

        Parameters
        ----------
        svid: Shot video ID.
        tid: Track ID.
        seg_first: First frame of the segment to assign.
        """
        name, ok = QInputDialog.getText(self, "New person", "Person name:")
        if ok and name.strip():
            name = name.strip()
            if name not in self._persons:
                self._persons.append(name)
            self._assign(svid, tid, seg_first, name)

    def _set_selected(self, svid: str, tid: int, seg_first: int) -> None:
        """Highlight the given segment and remove the highlight from the previously selected one.

        Parameters
        ----------
        svid: Shot video ID.
        tid: Track ID.
        seg_first: First frame of the segment to select.
        """
        prev = self._selected_key
        new_key = (svid, tid, seg_first)
        if prev == new_key:
            return
        if prev is not None:
            old_item = self._items.get(prev)
            if old_item is not None:
                old_item.setPen(QPen(Qt.NoPen))
        self._selected_key = new_key
        new_item = self._items.get(new_key)
        if new_item is not None:
            new_item.setPen(_SELECTED_PEN)

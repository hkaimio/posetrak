"""filmstrip_stitcher.py — FilmstripStitcherWidget: timeline with filmstrip bars."""
from __future__ import annotations

import sqlite3

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QInputDialog,
    QLabel,
    QMenu,
    QSizePolicy,
)

from app.pose.filmstrip_bar import FilmstripBarItem, decode_jpeg_to_pixmap, LABEL_H
from app.pose.db_cache import read_track_spans
from app.setup.db_context import SyncTable


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

ROW_H_DEFAULT: int = 56          # default bar height in pixels
ROW_H_MIN: int = 28
ROW_H_MAX: int = 120
LABEL_WIDTH: int = 90            # camera-label column width
ROW_GAP: int = 5                 # gap between camera groups
TRACK_GAP: int = 2               # gap between tracks within a camera

_PX_PER_SEC_MIN: float = 5.0
_PX_PER_SEC_MAX: float = 500.0

# Maximum thumbnails loaded per bar (caps memory / decode time)
MAX_THUMBS_PER_BAR: int = 120

_PLAYHEAD_PEN = QPen(QColor(220, 40, 40), 2)


def _build_frame_to_time(
    session: sqlite3.Connection,
    sync_config_id: str,
    shot_video_ids: list[str],
) -> dict[str, tuple[int, float, float]]:
    """Return {svid: (ref_frame, ref_ts_s, fps)} from the first sync point per video."""
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


class FilmstripStitcherWidget(QGraphicsView):
    """QGraphicsView timeline where each detection bar is a filmstrip of JPEG crops.

    Segment state management (``_segments``, ``_seg_assignments``, etc.) mirrors
    ``StitcherWidget`` so that ``StitcherPanel`` can use the same ``get_spans()``,
    ``split_segment()``, and ``set_segment_assignment()`` API.  New additions:

    * ``segment_selected`` — 6-argument signal including the user's selected
      sub-range within the bar (for partial-range assignment / auto-split).
    * ``time_hovered`` — emitted on mouse-move for the status bar.
    * ``merge_segments()`` — merge two adjacent segments (used by auto-merge).
    * ``set_conflict_segments()`` — highlight overlap-conflict bars.

    Signals
    -------
    segment_selected(svid, tid, seg_first, seg_last, sel_first, sel_last):
        Left-click or drag-release on a bar.  *sel_first/sel_last* are the
        selected frame sub-range (equal to seg_first/seg_last on a plain click).
    assignment_changed(svid, tid, seg_first, person_or_None):
        User assigns or detaches a segment via the context menu.
    time_hovered(global_s):
        Emitted on every mouse-move (for status bar updates).
    time_clicked(global_s):
        Emitted on left-press (for playhead sync).
    """

    segment_selected = Signal(str, int, int, int, int, int)
    assignment_changed = Signal(str, int, int, object)
    time_hovered = Signal(float)
    time_clicked = Signal(float)

    def __init__(self, row_h: int = ROW_H_DEFAULT, parent=None) -> None:
        super().__init__(parent)
        self._row_h = row_h
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

        # --- Segment state (mirrors StitcherWidget) ---
        # (svid, tid) → [(seg_first, seg_last), ...] sorted by seg_first
        self._segments: dict[tuple[str, int], list[tuple[int, int]]] = {}
        # (svid, tid, seg_first) → person_name
        self._seg_assignments: dict[tuple[str, int, int], str] = {}
        # (svid, tid, seg_first) → FilmstripBarItem
        self._items: dict[tuple[str, int, int], FilmstripBarItem] = {}
        # (svid, tid) → y coordinate in scene
        self._row_y: dict[tuple[str, int], float] = {}
        # svid → (ref_frame, ref_ts, fps)
        self._anchors: dict[str, tuple[int, float, float]] = {}
        self._sync_table: SyncTable | None = None
        # Set of seg_keys with overlap conflicts
        self._conflict_keys: set[tuple[str, int, int]] = set()

        self._selected_key: tuple[str, int, int] | None = None
        self._current_time_s: float = 0.0
        self._time_line: QGraphicsLineItem | None = None
        self._time_origin: float = 0.0
        self._total_duration_s: float = 0.0
        self._persons: list[str] = []

        self._last_session: sqlite3.Connection | None = None
        self._last_run_id: str | None = None

        # --- Drag-to-select state ---
        self._drag_active: bool = False
        self._drag_bar: FilmstripBarItem | None = None
        self._drag_seg_key: tuple[str, int, int] | None = None
        self._drag_start_pos: QPointF | None = None
        self._drag_sel_first: int | None = None
        self._drag_sel_last: int | None = None

        # --- Hover tooltip ---
        self._tooltip = QLabel(self.viewport())
        self._tooltip.setWindowFlags(Qt.WindowType.ToolTip)
        self._tooltip.setStyleSheet(
            "background: #111; border: 1px solid #555; padding: 2px;"
        )
        self._tooltip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._tooltip.hide()
        self._tooltip_timer = QTimer(self)
        self._tooltip_timer.setSingleShot(True)
        self._tooltip_timer.setInterval(120)
        self._tooltip_timer.timeout.connect(self._show_tooltip_for_cursor)
        self._tooltip_frame: int | None = None   # frame shown in current tooltip
        self._tooltip_bar: FilmstripBarItem | None = None

    # ------------------------------------------------------------------
    # Public API (compatible with StitcherWidget)
    # ------------------------------------------------------------------

    def set_sync_table(self, table: SyncTable | None) -> None:
        self._sync_table = table

    def get_spans(self) -> dict[tuple[str, int, int], tuple[int, int]]:
        return {
            (svid, tid, sf): (sf, sl)
            for (svid, tid), segs in self._segments.items()
            for sf, sl in segs
        }

    def get_time_spans(self) -> dict[tuple[str, int, int], tuple[float, float]]:
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
        self._persons = list(persons)

    def set_current_time(self, global_s: float) -> None:
        self._current_time_s = global_s
        if self._time_line is None:
            return
        x = LABEL_WIDTH + (global_s - self._time_origin) * self._px_per_sec
        h = self._scene.sceneRect().height()
        self._time_line.setLine(x, 0, x, h)

    def set_segment_assignment(
        self, svid: str, tid: int, seg_first: int, person_name: str | None
    ) -> None:
        seg_key = (svid, tid, seg_first)
        if person_name:
            self._seg_assignments[seg_key] = person_name
        else:
            self._seg_assignments.pop(seg_key, None)
        bar = self._items.get(seg_key)
        if bar is not None:
            bar.set_assignment(person_name)

    def split_segment(self, svid: str, tid: int, seg_first: int, split_frame: int) -> None:
        """Split segment [seg_first, seg_last] at split_frame.

        Left half [seg_first, split_frame-1] inherits the existing assignment.
        Right half [split_frame, seg_last] is unassigned.
        """
        key = (svid, tid)
        segs = self._segments.get(key, [])
        for i, (sf, sl) in enumerate(segs):
            if sf != seg_first:
                continue
            if not (sf < split_frame <= sl):
                return

            old_key = (svid, tid, sf)
            old_bar = self._items.pop(old_key, None)
            if old_bar is not None:
                self._scene.removeItem(old_bar)

            segs[i] = (sf, split_frame - 1)
            segs.insert(i + 1, (split_frame, sl))

            # Right half is unassigned
            y = self._row_y.get(key, 0.0)
            self._draw_bar(svid, tid, sf, split_frame - 1, y)
            self._draw_bar(svid, tid, split_frame, sl, y)

            # Restore left-half assignment
            left_person = self._seg_assignments.get(old_key)
            bar = self._items.get(old_key)
            if bar is not None:
                bar.set_assignment(left_person)
            break

    def merge_segments(self, svid: str, tid: int, sf1: int, sf2: int) -> None:
        """Merge two adjacent segments where sf2 == sl1 + 1.

        The merged segment keeps the key (svid, tid, sf1) and the assignment
        of the first segment.  The second segment's assignment entry must be
        removed by the caller before calling this.
        """
        key = (svid, tid)
        segs = self._segments.get(key, [])
        idx1 = next((i for i, (sf, _) in enumerate(segs) if sf == sf1), None)
        idx2 = next((i for i, (sf, _) in enumerate(segs) if sf == sf2), None)
        if idx1 is None or idx2 is None or idx2 != idx1 + 1:
            return

        _, sl1 = segs[idx1]
        _, sl2 = segs[idx2]

        # Validate adjacency
        if sl1 + 1 != sf2:
            return

        # Remove both bar items
        for sf in (sf1, sf2):
            bar = self._items.pop((svid, tid, sf), None)
            if bar is not None:
                self._scene.removeItem(bar)

        # Merge in segment list
        segs[idx1] = (sf1, sl2)
        del segs[idx2]

        # Redraw merged bar
        y = self._row_y.get(key, 0.0)
        self._draw_bar(svid, tid, sf1, sl2, y)

        # Restore assignment on merged bar
        person = self._seg_assignments.get((svid, tid, sf1))
        bar = self._items.get((svid, tid, sf1))
        if bar is not None:
            bar.set_assignment(person)

    def set_conflict_segments(self, conflict_keys: set[tuple[str, int, int]]) -> None:
        """Mark/unmark bars as overlap-conflict based on the provided key set."""
        changed = conflict_keys.symmetric_difference(self._conflict_keys)
        self._conflict_keys = set(conflict_keys)
        for key in changed:
            bar = self._items.get(key)
            if bar is not None:
                bar.set_conflict(key in self._conflict_keys)

    def load_run(self, session: sqlite3.Connection, detection_run_id: str) -> None:
        self._last_session = session
        self._last_run_id = detection_run_id
        self._segments.clear()
        self._seg_assignments.clear()
        self._selected_key = None
        self._rebuild(session, detection_run_id)

    def clear(self) -> None:
        self._scene.clear()
        self._items.clear()
        self._row_y.clear()
        self._time_line = None

    def set_row_height(self, row_h: int) -> None:
        self._row_h = max(ROW_H_MIN, min(ROW_H_MAX, row_h))
        if self._last_session and self._last_run_id:
            self._rebuild(self._last_session, self._last_run_id)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._last_session is not None and self._last_run_id is not None:
            self._rebuild(self._last_session, self._last_run_id)

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    @property
    def _px_per_sec(self) -> float:
        available = max(1, self.viewport().width() - LABEL_WIDTH)
        dur = self._total_duration_s
        if dur > 0:
            fitted = available / dur
            return max(_PX_PER_SEC_MIN, min(_PX_PER_SEC_MAX, fitted))
        return 30.0

    def _rebuild(
        self, session: sqlite3.Connection, detection_run_id: str
    ) -> None:
        self.clear()

        run_row = session.execute(
            "SELECT shot_id, sync_config_id, time_start_s, time_end_s "
            "FROM detection_runs WHERE id = ?",
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

            lbl = self._scene.addText(cam_labels[svid])
            lbl.setPos(0, y)
            lbl.setDefaultTextColor(QColor(0, 0, 0))

            if db_spans:
                for span in db_spans:
                    tid = span["track_id"]
                    first = span["first_frame"]
                    last = span["last_frame"]
                    det_key = (svid, tid)
                    if det_key not in self._segments:
                        self._segments[det_key] = [(first, last)]
                    self._row_y[det_key] = y
                    for seg_first, seg_last in self._segments[det_key]:
                        self._draw_bar(svid, tid, seg_first, seg_last, y)
                    y += self._row_h + TRACK_GAP
            else:
                y += self._row_h

            y += ROW_GAP

        self._scene.setSceneRect(self._scene.itemsBoundingRect())

        # Restore assignment colours
        for (svid, tid, sf), name in self._seg_assignments.items():
            bar = self._items.get((svid, tid, sf))
            if bar is not None:
                bar.set_assignment(name)

        # Restore conflict highlights
        for key in self._conflict_keys:
            bar = self._items.get(key)
            if bar is not None:
                bar.set_conflict(True)

        # Playhead
        scene_h = max(self._scene.sceneRect().height(), 1)
        x0 = float(LABEL_WIDTH)
        self._time_line = self._scene.addLine(
            x0, 0.0, x0, scene_h, _PLAYHEAD_PEN
        )
        self._time_line.setZValue(20)

        # Load thumbnails for all bars
        for bar in list(self._items.values()):
            self._load_thumbnails(bar, session, detection_run_id)

    def _draw_bar(
        self,
        svid: str,
        tid: int,
        seg_first: int,
        seg_last: int,
        y: float,
    ) -> FilmstripBarItem:
        """Create and register a FilmstripBarItem for one segment."""
        pps = self._px_per_sec
        t0, t1 = self._seg_time_range(svid, seg_first, seg_last)
        x = LABEL_WIDTH + (t0 - self._time_origin) * pps
        w = max(2.0, (t1 - t0) * pps)

        bar = FilmstripBarItem(svid, tid, seg_first, seg_last, w, self._row_h)
        bar.setPos(x, y)
        bar.setZValue(1)

        seg_key = (svid, tid, seg_first)
        is_sel = (seg_key == self._selected_key)
        bar.set_selection(None, None, is_sel)
        bar.set_conflict(seg_key in self._conflict_keys)

        self._scene.addItem(bar)
        self._items[seg_key] = bar
        return bar

    def _seg_time_range(self, svid: str, seg_first: int, seg_last: int) -> tuple[float, float]:
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
        return t0, t1

    def _load_thumbnails(
        self,
        bar: FilmstripBarItem,
        session: sqlite3.Connection,
        detection_run_id: str,
    ) -> None:
        """Query frame_cache_entries and populate bar with scaled thumbnails."""
        film_h = max(1, bar._row_h - LABEL_H)
        bar_w = max(1, int(bar._width))
        # Estimate how many thumbnails fit (assuming average aspect ~0.5 width/height)
        N = max(4, bar_w // max(1, film_h // 2))
        N = min(N, MAX_THUMBS_PER_BAR)

        rows = session.execute(
            "SELECT frame_idx, image_data "
            "FROM frame_cache_entries "
            "WHERE shot_video_id = ? AND cache_type = 'person_crop' "
            "  AND track_id = ? AND detection_run_id = ? "
            "  AND frame_idx BETWEEN ? AND ? "
            "ORDER BY frame_idx",
            (bar.svid, bar.tid, detection_run_id, bar.seg_first, bar.seg_last),
        ).fetchall()

        if not rows:
            return

        step = max(1, len(rows) // N)
        selected = rows[::step]

        thumbs: dict[int, QPixmap] = {}
        for row in selected:
            pix = decode_jpeg_to_pixmap(bytes(row["image_data"]), film_h)
            if pix is not None:
                thumbs[row["frame_idx"]] = pix

        if thumbs:
            bar.set_thumbnails(thumbs)

    def _bar_at_scene_pos(self, scene_pos: QPointF) -> FilmstripBarItem | None:
        """Return the FilmstripBarItem under scene_pos, or None."""
        for item in self._scene.items(scene_pos):
            if isinstance(item, FilmstripBarItem):
                return item
        return None

    def _scene_x_to_global_time(self, scene_x: float) -> float:
        return self._time_origin + (scene_x - LABEL_WIDTH) / max(self._px_per_sec, 1e-6)

    def _frame_at_scene_pos(self, bar: FilmstripBarItem, scene_pos: QPointF) -> int:
        """Map scene_pos.x to the nearest frame index within bar.

        Uses SyncTable when available; otherwise falls back to linear
        interpolation between seg_first and seg_last.
        """
        global_t = self._scene_x_to_global_time(scene_pos.x())
        if self._sync_table is not None:
            frame = self._sync_table.lookup(global_t, bar.svid)
            if frame is not None:
                return max(bar.seg_first, min(bar.seg_last, frame))
        # Fallback: linear interpolation via bar coordinates
        local_x = scene_pos.x() - bar.pos().x()
        return bar.local_x_to_frame(local_x)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _set_selected(self, svid: str, tid: int, seg_first: int) -> None:
        prev = self._selected_key
        new_key = (svid, tid, seg_first)
        if prev == new_key:
            return
        if prev is not None:
            old_bar = self._items.get(prev)
            if old_bar is not None:
                old_bar.set_selection(None, None, False)
        self._selected_key = new_key
        new_bar = self._items.get(new_key)
        if new_bar is not None:
            new_bar.set_selection(None, None, True)

    def _emit_segment_selected(
        self,
        bar: FilmstripBarItem,
        sel_first: int,
        sel_last: int,
        scene_pos: QPointF,
    ) -> None:
        self.segment_selected.emit(
            bar.svid, bar.tid, bar.seg_first, bar.seg_last, sel_first, sel_last
        )
        global_t = self._scene_x_to_global_time(scene_pos.x())
        self.time_clicked.emit(global_t)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            bar = self._bar_at_scene_pos(scene_pos)
            if bar is not None:
                self._drag_active = False
                self._drag_bar = bar
                self._drag_seg_key = (bar.svid, bar.tid, bar.seg_first)
                self._drag_start_pos = scene_pos
                frame = self._frame_at_scene_pos(bar, scene_pos)
                self._drag_sel_first = frame
                self._drag_sel_last = frame
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.pos())

        # --- Hover tooltip debounce ---
        bar = self._bar_at_scene_pos(scene_pos)
        if bar is not None:
            self._tooltip_bar = bar
            self._tooltip_timer.start()  # restart debounce
        else:
            self._tooltip_timer.stop()
            self._tooltip.hide()
            self._tooltip_bar = None

        # --- Status bar ---
        global_t = self._scene_x_to_global_time(scene_pos.x())
        if global_t >= 0:
            self.time_hovered.emit(global_t)

        # --- Drag-to-select ---
        if (
            self._drag_bar is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            dx = (scene_pos - self._drag_start_pos).x() if self._drag_start_pos else 0.0
            if abs(dx) > 4 or self._drag_active:
                self._drag_active = True
                frame = self._frame_at_scene_pos(self._drag_bar, scene_pos)
                start_frame = self._drag_sel_first or frame
                sel_first = min(start_frame, frame)
                sel_last = max(start_frame, frame)
                self._drag_sel_first = start_frame
                self._drag_sel_last = frame
                self._drag_bar.set_selection(sel_first, sel_last, True)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_bar is not None:
            bar = self._drag_bar
            scene_pos = self.mapToScene(event.pos())

            if self._drag_active:
                # Drag-to-select: emit sub-range
                sf = self._drag_sel_first or bar.seg_first
                sl = self._drag_sel_last or bar.seg_last
                sel_first = min(sf, sl)
                sel_last = max(sf, sl)
            else:
                # Plain click: full bar
                sel_first = bar.seg_first
                sel_last = bar.seg_last

            self._set_selected(bar.svid, bar.tid, bar.seg_first)
            bar.set_selection(
                None if (sel_first == bar.seg_first and sel_last == bar.seg_last) else sel_first,
                None if (sel_first == bar.seg_first and sel_last == bar.seg_last) else sel_last,
                True,
            )
            self._emit_segment_selected(bar, sel_first, sel_last, scene_pos)

            # Reset drag state
            self._drag_active = False
            self._drag_bar = None
            self._drag_seg_key = None
            self._drag_start_pos = None
            self._drag_sel_first = None
            self._drag_sel_last = None

        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._tooltip_timer.stop()
        self._tooltip.hide()
        super().leaveEvent(event)

    # ------------------------------------------------------------------
    # Hover tooltip
    # ------------------------------------------------------------------

    def _show_tooltip_for_cursor(self) -> None:
        """Called by debounce timer; loads exact frame and shows tooltip."""
        if self._tooltip_bar is None or self._last_session is None:
            self._tooltip.hide()
            return

        bar = self._tooltip_bar
        cursor_scene = self.mapToScene(self.viewport().mapFromGlobal(QCursor.pos()))
        frame = self._frame_at_scene_pos(bar, cursor_scene)

        # Check if we already have this frame in the thumbnail cache
        pix = bar._thumbs.get(frame)
        if pix is None:
            # Fetch from DB (fast — single row lookup)
            row = self._last_session.execute(
                "SELECT image_data FROM frame_cache_entries "
                "WHERE shot_video_id=? AND cache_type='person_crop' "
                "  AND track_id=? AND detection_run_id=? "
                "  AND frame_idx BETWEEN ? AND ? "
                "ORDER BY ABS(frame_idx - ?) LIMIT 1",
                (bar.svid, bar.tid, self._last_run_id,
                 frame - 3, frame + 3, frame),
            ).fetchone()
            if row is not None:
                pix = decode_jpeg_to_pixmap(bytes(row["image_data"]), 160)

        if pix is None:
            self._tooltip.hide()
            return

        self._tooltip.setPixmap(pix)
        self._tooltip.adjustSize()

        # Position tooltip above the bar in viewport coords
        vp_pos = self.viewport().mapFromGlobal(QCursor.pos())
        tip_x = max(0, min(vp_pos.x() - pix.width() // 2,
                           self.viewport().width() - pix.width() - 4))
        tip_y = max(0, vp_pos.y() - pix.height() - 6)
        self._tooltip.move(tip_x, tip_y)
        self._tooltip.show()
        self._tooltip.raise_()

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _on_context_menu(self, pos) -> None:
        scene_pos = self.mapToScene(pos)
        bar = self._bar_at_scene_pos(scene_pos)
        if bar is None:
            return

        svid, tid, seg_first = bar.svid, bar.tid, bar.seg_first
        seg_key = (svid, tid, seg_first)
        self._set_selected(svid, tid, seg_first)

        menu = QMenu(self)
        current = self._seg_assignments.get(seg_key)

        for name in self._persons:
            action = menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.triggered.connect(
                lambda _checked, n=name: self.assignment_changed.emit(svid, tid, seg_first, n)
            )

        menu.addSeparator()
        new_action = menu.addAction("New person…")
        new_action.triggered.connect(lambda: self._new_person(svid, tid, seg_first))
        if current:
            detach_action = menu.addAction("Detach")
            detach_action.triggered.connect(
                lambda: self.assignment_changed.emit(svid, tid, seg_first, None)
            )

        menu.exec(self.viewport().mapToGlobal(pos))

    def _new_person(self, svid: str, tid: int, seg_first: int) -> None:
        name, ok = QInputDialog.getText(self, "New person", "Person name:")
        if ok and name.strip():
            name = name.strip()
            if name not in self._persons:
                self._persons.append(name)
            self.assignment_changed.emit(svid, tid, seg_first, name)

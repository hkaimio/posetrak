"""filmstrip_stitcher.py — FilmstripStitcherWidget: timeline with filmstrip bars."""
from __future__ import annotations

import sqlite3
from collections import defaultdict

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QInputDialog,
    QMenu,
)

from app.pose.colors import person_color
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
ROW_GAP: int = 5                 # gap between camera/person groups
TRACK_GAP: int = 2               # gap between tracks within a camera
PERSON_GAP: int = 12             # extra gap between person sections (by-person view)

_PX_PER_SEC_MIN: float = 5.0
_PX_PER_SEC_MAX: float = 500.0

# Maximum thumbnails loaded per bar (caps memory / decode time)
MAX_THUMBS_PER_BAR: int = 120

_PLAYHEAD_PEN = QPen(QColor(220, 40, 40), 2)
_CAM_SEP_PEN = QPen(QColor(120, 120, 120, 100), 1, Qt.PenStyle.SolidLine)
_PERSON_SEP_PEN = QPen(QColor(80, 80, 80, 120), 1, Qt.PenStyle.DashLine)
_SECTION_LABEL_COLOR = QColor(30, 30, 30)
_PERSON_LABEL_COLOR = QColor(10, 60, 120)
_PERSON_LABEL_FONT = QFont()

_PERSON_BG_ALPHA = 35  # opacity of the person section tint (0–255)


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
    * ``bar_hovered`` — emitted on mouse-move over a bar (drives side crop panel).
    * ``time_hovered`` — emitted on mouse-move for the status bar.
    * ``merge_segments()`` — merge two adjacent segments (used by auto-merge).
    * ``set_conflict_segments()`` — highlight overlap-conflict bars.
    * ``refresh_person_view()`` — rebuild scene if currently in person view mode.

    View modes
    ----------
    ``"detection"`` (default): one row per track, grouped by camera.
        Camera groups are separated by horizontal lines.
    ``"person"``: only assigned segments, grouped by person name then camera.
        Each person section has a tinted background.  Non-overlapping segments
        within a person+camera group are packed onto a single row using a greedy
        interval scheduling algorithm.

    Conflict highlights
    -------------------
    Only the *overlapping time ranges* within each conflicted bar are highlighted
    red (via ``FilmstripBarItem.set_conflict_ranges``).  This is more precise than
    highlighting the entire bar.
    """

    segment_selected = Signal(str, int, int, int, int, int)
    assignment_changed = Signal(str, int, int, object)
    bar_hovered = Signal(str, int, int, int)   # svid, tid, seg_first, frame_idx
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
        # (svid, tid, seg_first) → y coordinate in scene (per-segment, for split/merge)
        self._bar_row_y: dict[tuple[str, int, int], float] = {}
        # svid → (ref_frame, ref_ts, fps)
        self._anchors: dict[str, tuple[int, float, float]] = {}
        self._sync_table: SyncTable | None = None

        # Conflict state: both the set of keys and the exact per-bar ranges
        self._conflict_keys: set[tuple[str, int, int]] = set()
        # (svid, tid, seg_first) → [(frame_first, frame_last), ...]
        self._conflict_ranges_data: dict[tuple[str, int, int], list[tuple[int, int]]] = {}

        self._selected_key: tuple[str, int, int] | None = None
        self._current_time_s: float = 0.0
        self._time_line: QGraphicsLineItem | None = None
        self._time_origin: float = 0.0
        self._total_duration_s: float = 0.0
        self._persons: list[str] = []

        self._last_session: sqlite3.Connection | None = None
        self._last_run_id: str | None = None

        # --- View mode ---
        self._view_mode: str = "detection"   # "detection" or "person"

        # --- Thumbnail cache: survives view-mode switches and rebuilds ---
        # (svid, tid, seg_first) → {frame_idx: QPixmap}
        self._thumb_cache: dict[tuple[str, int, int], dict[int, QPixmap]] = {}

        # --- Drag-to-select state ---
        self._drag_active: bool = False
        self._drag_bar: FilmstripBarItem | None = None
        self._drag_seg_key: tuple[str, int, int] | None = None
        self._drag_start_pos: QPointF | None = None
        self._drag_sel_first: int | None = None
        self._drag_sel_last: int | None = None

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
            old_y = self._bar_row_y.get(old_key, 0.0)
            old_bar = self._items.pop(old_key, None)
            self._bar_row_y.pop(old_key, None)
            if old_bar is not None:
                self._scene.removeItem(old_bar)

            segs[i] = (sf, split_frame - 1)
            segs.insert(i + 1, (split_frame, sl))

            self._draw_bar(svid, tid, sf, split_frame - 1, old_y)
            self._draw_bar(svid, tid, split_frame, sl, old_y)

            # Restore left-half assignment
            left_person = self._seg_assignments.get(old_key)
            bar = self._items.get(old_key)
            if bar is not None:
                bar.set_assignment(left_person)
            break

    def merge_segments(self, svid: str, tid: int, sf1: int, sf2: int) -> None:
        """Merge two adjacent segments where sf2 == sl1 + 1."""
        key = (svid, tid)
        segs = self._segments.get(key, [])
        idx1 = next((i for i, (sf, _) in enumerate(segs) if sf == sf1), None)
        idx2 = next((i for i, (sf, _) in enumerate(segs) if sf == sf2), None)
        if idx1 is None or idx2 is None or idx2 != idx1 + 1:
            return

        _, sl1 = segs[idx1]
        _, sl2 = segs[idx2]

        if sl1 + 1 != sf2:
            return

        y = self._bar_row_y.get((svid, tid, sf1), 0.0)

        for sf in (sf1, sf2):
            bar = self._items.pop((svid, tid, sf), None)
            self._bar_row_y.pop((svid, tid, sf), None)
            if bar is not None:
                self._scene.removeItem(bar)

        segs[idx1] = (sf1, sl2)
        del segs[idx2]

        # Merge thumbnail caches
        cache1 = self._thumb_cache.pop((svid, tid, sf1), {})
        cache2 = self._thumb_cache.pop((svid, tid, sf2), {})
        merged_cache = {**cache1, **cache2}
        if merged_cache:
            self._thumb_cache[(svid, tid, sf1)] = merged_cache

        self._draw_bar(svid, tid, sf1, sl2, y)

        person = self._seg_assignments.get((svid, tid, sf1))
        bar = self._items.get((svid, tid, sf1))
        if bar is not None:
            bar.set_assignment(person)

    def set_conflict_segments(
        self,
        conflict_ranges: dict[tuple[str, int, int], list[tuple[int, int]]],
    ) -> None:
        """Update per-bar conflict highlights.

        Args:
            conflict_ranges: mapping from seg_key to a list of (frame_first, frame_last)
                ranges that overlap another assignment for the same person and camera.
                Only those ranges are highlighted red within each bar.
                Pass an empty dict to clear all conflicts.
        """
        new_keys = set(conflict_ranges.keys())

        # Clear bars that are no longer conflicted
        for key in self._conflict_keys - new_keys:
            bar = self._items.get(key)
            if bar is not None:
                bar.set_conflict_ranges([])

        # Set/update conflicted bars
        for key, ranges in conflict_ranges.items():
            bar = self._items.get(key)
            if bar is not None:
                bar.set_conflict_ranges(ranges)

        self._conflict_keys = new_keys
        self._conflict_ranges_data = dict(conflict_ranges)

    def load_run(self, session: sqlite3.Connection, detection_run_id: str) -> None:
        self._last_session = session
        self._last_run_id = detection_run_id
        self._segments.clear()
        self._seg_assignments.clear()
        self._thumb_cache.clear()
        self._conflict_ranges_data.clear()
        self._conflict_keys.clear()
        self._selected_key = None
        self._rebuild(session, detection_run_id)

    def clear(self) -> None:
        """Clear scene items and visual lookup dicts; preserves logical state."""
        self._scene.clear()
        self._items.clear()
        self._bar_row_y.clear()
        self._time_line = None

    def set_row_height(self, row_h: int) -> None:
        self._row_h = max(ROW_H_MIN, min(ROW_H_MAX, row_h))
        if self._last_session and self._last_run_id:
            self._rebuild(self._last_session, self._last_run_id)

    def set_view_mode(self, mode: str) -> None:
        """Switch between 'detection' and 'person' view modes."""
        if mode == self._view_mode:
            return
        self._view_mode = mode
        if self._last_session and self._last_run_id:
            self._rebuild(self._last_session, self._last_run_id)

    def refresh_person_view(self) -> None:
        """Rebuild scene if in 'person' view mode (called after assignment changes)."""
        if self._view_mode == "person" and self._last_session and self._last_run_id:
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
        """Rebuild the entire scene from current segment state."""
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

        # Ensure segment state is initialised from DB for any tracks not yet split
        for svid in svids:
            db_spans = read_track_spans(session, detection_run_id, svid)
            for span in db_spans:
                tid = span["track_id"]
                det_key = (svid, tid)
                if det_key not in self._segments:
                    self._segments[det_key] = [
                        (span["first_frame"], span["last_frame"])
                    ]

        # Estimated scene width (used for backgrounds and separators)
        scene_w = LABEL_WIDTH + max(1.0, self._total_duration_s) * self._px_per_sec + 20.0

        # Dispatch to view-specific layout, collecting decoration positions
        if self._view_mode == "person":
            sep_ys, person_bg_sections = self._rebuild_by_person(
                session, detection_run_id, svids, cam_labels
            )
        else:
            sep_ys, person_bg_sections = self._rebuild_by_detection(
                session, detection_run_id, svids, cam_labels
            )

        # Draw person-section backgrounds (z = -1, behind bars)
        for y_start, y_end, pname in person_bg_sections:
            c = person_color(pname)
            bg = QColor(c.red(), c.green(), c.blue(), _PERSON_BG_ALPHA)
            ri = QGraphicsRectItem(QRectF(0, y_start, scene_w, y_end - y_start))
            ri.setBrush(QBrush(bg))
            ri.setPen(QPen(Qt.PenStyle.NoPen))
            ri.setZValue(-1)
            self._scene.addItem(ri)

        # Draw camera separator lines (z = 0.5, above backgrounds, below bars)
        for sep_y, sep_style in sep_ys:
            pen = _CAM_SEP_PEN if sep_style == "camera" else _PERSON_SEP_PEN
            line = self._scene.addLine(0, sep_y, scene_w, sep_y, pen)
            line.setZValue(0.5)

        self._scene.setSceneRect(self._scene.itemsBoundingRect())

        # Restore conflict highlights from stored range data
        for key, ranges in self._conflict_ranges_data.items():
            bar = self._items.get(key)
            if bar is not None:
                bar.set_conflict_ranges(ranges)

        # Playhead
        scene_h = max(self._scene.sceneRect().height(), 1)
        x0 = float(LABEL_WIDTH)
        self._time_line = self._scene.addLine(
            x0, 0.0, x0, scene_h, _PLAYHEAD_PEN
        )
        self._time_line.setZValue(20)

        # Load thumbnails for all bars (uses cache to avoid DB round-trips)
        for bar in list(self._items.values()):
            self._load_thumbnails(bar, session, detection_run_id)

    def _rebuild_by_detection(
        self,
        session: sqlite3.Connection,
        detection_run_id: str,
        svids: list[str],
        cam_labels: dict[str, str],
    ) -> tuple[list[tuple[float, str]], list[tuple[float, float, str]]]:
        """Draw bars grouped by camera then track (default view).

        Returns (sep_ys, person_bg_sections).  sep_ys is a list of
        (y, style) tuples; style is 'camera' for camera boundaries.
        """
        sep_ys: list[tuple[float, str]] = []
        y = 0.0
        first_cam = True

        for svid in svids:
            if not first_cam:
                sep_ys.append((y - ROW_GAP * 0.5, "camera"))
            first_cam = False

            lbl = self._scene.addText(cam_labels[svid])
            lbl.setPos(0, y)
            lbl.setDefaultTextColor(_SECTION_LABEL_COLOR)

            track_keys = sorted(
                {tid for (s, tid) in self._segments if s == svid}
            )
            if track_keys:
                for tid in track_keys:
                    segs = self._segments.get((svid, tid), [])
                    for seg_first, seg_last in segs:
                        self._draw_bar(svid, tid, seg_first, seg_last, y)
                    y += self._row_h + TRACK_GAP
            else:
                y += self._row_h

            y += ROW_GAP

        # Restore assignment colours
        for (svid, tid, sf), name in self._seg_assignments.items():
            bar = self._items.get((svid, tid, sf))
            if bar is not None:
                bar.set_assignment(name)

        return sep_ys, []  # no person backgrounds in detection view

    def _rebuild_by_person(
        self,
        session: sqlite3.Connection,
        detection_run_id: str,
        svids: list[str],
        cam_labels: dict[str, str],
    ) -> tuple[list[tuple[float, str]], list[tuple[float, float, str]]]:
        """Draw only assigned segments, grouped by person name then camera.

        Returns (sep_ys, person_bg_sections).
        """
        sep_ys: list[tuple[float, str]] = []
        person_bg_sections: list[tuple[float, float, str]] = []

        if not self._seg_assignments:
            lbl = self._scene.addText("No assignments yet")
            lbl.setPos(LABEL_WIDTH, 0)
            lbl.setDefaultTextColor(_SECTION_LABEL_COLOR)
            return sep_ys, person_bg_sections

        # Group assigned keys: person → svid → [seg_key]
        person_cam: dict[str, dict[str, list[tuple[str, int, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for key, pname in self._seg_assignments.items():
            svid = key[0]
            person_cam[pname][svid].append(key)

        y = 0.0
        for pname in sorted(person_cam):
            y_person_start = y

            # Person header
            person_lbl = self._scene.addText(f"▶ {pname}")
            font = person_lbl.font()
            font.setBold(True)
            person_lbl.setFont(font)
            person_lbl.setPos(0, y)
            person_lbl.setDefaultTextColor(_PERSON_LABEL_COLOR)
            y += self._row_h

            first_cam_for_person = True
            for svid in svids:
                seg_keys = person_cam[pname].get(svid)
                if not seg_keys:
                    continue

                if not first_cam_for_person:
                    sep_ys.append((y - ROW_GAP * 0.5, "camera"))
                first_cam_for_person = False

                # Camera sub-label
                cam_lbl = self._scene.addText(f"  {cam_labels[svid]}")
                cam_lbl.setPos(0, y)
                cam_lbl.setDefaultTextColor(_SECTION_LABEL_COLOR)

                # Pack segments onto rows using greedy interval scheduling
                rows = self._pack_into_rows(seg_keys)
                for row_keys in rows:
                    for seg_key in row_keys:
                        svid_k, tid, sf = seg_key
                        sl = next(
                            (sl_ for (sf_, sl_) in self._segments.get((svid_k, tid), [])
                             if sf_ == sf),
                            sf,
                        )
                        bar = self._draw_bar(svid_k, tid, sf, sl, y)
                        bar.set_assignment(pname)
                    y += self._row_h + TRACK_GAP

                y += ROW_GAP

            y_person_end = y
            person_bg_sections.append((y_person_start, y_person_end, pname))

            # Separator between person sections
            sep_ys.append((y - PERSON_GAP * 0.5, "person"))
            y += PERSON_GAP

        return sep_ys, person_bg_sections

    def _pack_into_rows(
        self, seg_keys: list[tuple[str, int, int]]
    ) -> list[list[tuple[str, int, int]]]:
        """Greedy interval scheduling: pack seg_keys into rows so that
        non-overlapping segments share a row."""
        def start_time(key: tuple[str, int, int]) -> float:
            svid, tid, sf = key
            segs = self._segments.get((svid, tid), [])
            sf_range = next((r for r in segs if r[0] == sf), None)
            if sf_range is None:
                return float(sf)
            return self._seg_time_range(svid, sf_range[0], sf_range[1])[0]

        def end_time(key: tuple[str, int, int]) -> float:
            svid, tid, sf = key
            segs = self._segments.get((svid, tid), [])
            sf_range = next((r for r in segs if r[0] == sf), None)
            if sf_range is None:
                return float(sf)
            return self._seg_time_range(svid, sf_range[0], sf_range[1])[1]

        sorted_keys = sorted(seg_keys, key=start_time)
        rows: list[list[tuple[str, int, int]]] = []
        row_end_times: list[float] = []

        for key in sorted_keys:
            t0 = start_time(key)
            t1 = end_time(key)
            placed = False
            for i, row_end in enumerate(row_end_times):
                if t0 >= row_end:
                    rows[i].append(key)
                    row_end_times[i] = t1
                    placed = True
                    break
            if not placed:
                rows.append([key])
                row_end_times.append(t1)

        return rows

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

        # Restore conflict ranges for this bar (if any)
        ranges = self._conflict_ranges_data.get(seg_key, [])
        if ranges:
            bar.set_conflict_ranges(ranges)

        self._scene.addItem(bar)
        self._items[seg_key] = bar
        self._bar_row_y[seg_key] = y
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
        """Populate bar thumbnails from cache (or DB on first load)."""
        seg_key = (bar.svid, bar.tid, bar.seg_first)
        cached = self._thumb_cache.get(seg_key)
        if cached is not None:
            relevant = {f: p for f, p in cached.items()
                        if bar.seg_first <= f <= bar.seg_last}
            if relevant:
                bar.set_thumbnails(relevant)
            return

        film_h = max(1, bar._row_h - LABEL_H)
        bar_w = max(1, int(bar._width))
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
            self._thumb_cache[seg_key] = {}
            return

        step = max(1, len(rows) // N)
        selected = rows[::step]

        thumbs: dict[int, QPixmap] = {}
        for row in selected:
            pix = decode_jpeg_to_pixmap(bytes(row["image_data"]), film_h)
            if pix is not None:
                thumbs[row["frame_idx"]] = pix

        self._thumb_cache[seg_key] = thumbs
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
        """Map scene_pos.x to the nearest frame index within bar."""
        global_t = self._scene_x_to_global_time(scene_pos.x())
        if self._sync_table is not None:
            frame = self._sync_table.lookup(global_t, bar.svid)
            if frame is not None:
                return max(bar.seg_first, min(bar.seg_last, frame))
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

        # --- Bar hover: emit bar_hovered for side crop panel ---
        bar = self._bar_at_scene_pos(scene_pos)
        if bar is not None:
            frame = self._frame_at_scene_pos(bar, scene_pos)
            self.bar_hovered.emit(bar.svid, bar.tid, bar.seg_first, frame)

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
                sf = self._drag_sel_first or bar.seg_first
                sl = self._drag_sel_last or bar.seg_last
                sel_first = min(sf, sl)
                sel_last = max(sf, sl)
            else:
                sel_first = bar.seg_first
                sel_last = bar.seg_last

            self._set_selected(bar.svid, bar.tid, bar.seg_first)
            bar.set_selection(
                None if (sel_first == bar.seg_first and sel_last == bar.seg_last) else sel_first,
                None if (sel_first == bar.seg_first and sel_last == bar.seg_last) else sel_last,
                True,
            )
            self._emit_segment_selected(bar, sel_first, sel_last, scene_pos)

            self._drag_active = False
            self._drag_bar = None
            self._drag_seg_key = None
            self._drag_start_pos = None
            self._drag_sel_first = None
            self._drag_sel_last = None

        super().mouseReleaseEvent(event)

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

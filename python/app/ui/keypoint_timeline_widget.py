"""keypoint_timeline_widget.py — dope-sheet style timeline for keypoint editing.

Phases 12-13 of the keypoint-editing timeline view (see
docs/roadmap/features/keypoint-editing/keypoint-editing-design.md,
"Improvements" section), plus four rounds of follow-up UX fixes requested
after Phase 13 landed:

- Phase 12: a custom-painted tree of keypoint/group rows colored by
  `app.pose.timeline_status` axis-1 status, scoped to one camera at a time
  (matching `PersonCropGridWidget._sel_cam_idx`), with a playhead.
- Phase 13: rubber-band drag selects keypoints + a frame range, and
  Ctrl+click toggles a keyframe — both only emit signals, the actual DB
  writes and `_sel_kp_indices`/`_range_start_v` state live on the host
  widget (`PersonCropGridWidget`), matching the "state is owned by the
  host, not duplicated per-widget" rule from the design doc.
- Round 1: the standalone scrub slider was removed — the timeline became
  the trial's only clock; zoom (Ctrl+wheel / +/-/Fit buttons) and a
  horizontal scrollbar for panning; collapsible to save space.
- Round 2: seeking and selecting turned out to conflict when both lived in
  the row-tree's click handler (clicking to scrub silently nuked most of
  a multi-keypoint selection). Seeking now lives exclusively in a always-
  visible `_RulerWidget` (tick marks + playhead) above the row tree;
  clicking the row tree is purely a selection gesture (click clears,
  drag selects). Zoom now anchors on the playhead, not the cursor/click
  position. At high zoom, adjacent frame cells get a small gap so
  individual frames are visually distinguishable.
- Round 3: ruler ticks now show the same capture-global timestamp as the
  overlay row's current-time label, instead of time-from-trial-start (two
  different clocks looked like a bug). A track-area click on the row tree
  clears the selection *and* moves the playhead — round 2 made clicking
  selection-only, which felt unresponsive since clicking to deselect did
  nothing else. The active-range overlay snaps to whole-frame pixel bounds
  (frames are what's actually selected, not a continuous ms span) via
  `_TimelineCanvas._range_frame_pixel_bounds`. The collapse/expand arrow
  moved from the tab row onto the ruler, since the ruler is the row that
  stays visible while collapsed. Along the way, `_RulerWidget` and
  `_TimelineCanvas` were found to be silently misaligned whenever the row
  tree grows a vertical scrollbar (the ruler, outside that scroll area,
  didn't shrink to match) — fixed by giving both widgets their own
  width-aware time<->pixel mapping (module-level `_x_at_time_v`/
  `_time_v_at_x`) plus a dynamic right-margin on the ruler that tracks the
  canvas's scrollbar width.
- Round 4: per-row visibility. Each row gets a small eye icon (hand-drawn,
  not a font glyph) at the right edge of the label column; clicking it
  hides/shows that keypoint (or every keypoint in a group row at once).
  Hidden keypoints are excluded from drag-select and Ctrl+click keyframe
  toggling here, and from drawing/hit-testing/selection in the crop grid
  (`PersonCropGridWidget._hidden_kp_indices`, the actual source of truth —
  this widget only renders it and emits `visibility_toggled`).
- Round 5: a single timestamp marker, shown as a red flag in the ruler.
  Right-click the ruler to set/clear it (`_RulerWidget.contextMenuEvent`).
  Right-click a row in the canvas to "Select to marker", which selects that
  row's keypoint(s) and the frame range between the marker and the current
  playhead (whichever order they're in) — the actual selection/DB state
  still lives on the host widget, this only emits `marker_set` /
  `select_to_marker_requested`, matching the existing
  state-lives-on-the-host convention from Phase 13.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMenu,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.pose.kp_models import PoseModel
from app.pose.timeline_status import (
    STATUS_BLUE,
    STATUS_GREEN,
    STATUS_GREY,
    STATUS_ORANGE,
    STATUS_YELLOW,
)

ROW_H = 16
RULER_H = 22
LABEL_W = 140
COL_W = 2  # px per sampled time bucket when drawing status cells
MIN_VIEW_SPAN_MS = 100  # can't zoom in past a 100ms window
_EXPANDED_HEIGHT = 200  # default canvas height when first expanded
_FRAME_GAP_PX_THRESHOLD = 6  # start drawing gaps once a frame is wider than this
_FRAME_GAP_PX = 2
_EYE_ICON_W = 18          # px — clickable/drawn width of the per-row visibility toggle
_EYE_ICON_MARGIN = 4      # px gap between the icon and the label column's right edge

_STATUS_COLORS = {
    STATUS_GREEN: QColor(80, 170, 80),
    STATUS_YELLOW: QColor(210, 190, 60),
    STATUS_ORANGE: QColor(220, 140, 50),  # Idea 3: auto-redetected, not yet human-verified
    STATUS_BLUE: QColor(70, 130, 220),
    STATUS_GREY: QColor(120, 120, 120),
}
_NO_DATA_COLOR = QColor(45, 45, 45)
_BG_COLOR = QColor(30, 30, 30)
_LABEL_COLOR = QColor(220, 220, 220)
_SELECTED_ROW_BG = QColor(60, 90, 130, 130)
_RANGE_OVERLAY = QColor(255, 255, 255, 60)
_PLAYHEAD_COLOR = QColor(255, 80, 80)
_MARKER_COLOR = QColor(230, 40, 40)
_MARKER_FLAG_W = 9   # px — width of the flag triangle
_MARKER_FLAG_H = 7   # px — height of the flag triangle
_INLIER_BAR_COLOR = QColor(180, 180, 180)
_DRAG_RECT_COLOR = QColor(120, 170, 255, 60)
_HIDDEN_ROW_OVERLAY = QColor(0, 0, 0, 140)
_EYE_ICON_COLOR = QColor(200, 200, 200)
_EYE_ICON_HIDDEN_COLOR = QColor(110, 110, 110)

# "Nice" tick intervals for the ruler, in ms — smallest that keeps labels
# legibly spaced (see _pick_tick_interval_ms) is picked for the current zoom.
_NICE_INTERVALS_MS = (
    10, 20, 50, 100, 200, 500,
    1000, 2000, 5000, 10000, 15000, 30000,
    60000, 120000, 300000, 600000,
)


def _fmt_tick(t_abs: float) -> str:
    """Format a ruler tick label — same '<seconds>.<ms>' style as the overlay
    row's current-time label (`_fmt_time` in content_panels.py), using the
    capture's global timestamp rather than time-from-trial-start, so the two
    rows agree on what "the time" is."""
    return f"{t_abs:.3f}"


def _x_at_time_v(v: int, view_start: int, view_end: int, width: int) -> float:
    """Map a time value (ms from t_start) to a widget-local x pixel.

    A free function (not a method) so `_TimelineCanvas` and `_RulerWidget`
    can each map using *their own* width — they used to share the canvas's
    width, which silently drifted out of alignment whenever the canvas's
    QScrollArea grew a vertical scrollbar (shrinking the canvas's viewport)
    while the ruler, being outside that scroll area, didn't shrink to match.
    """
    span = max(1, view_end - view_start)
    w = max(1, width - LABEL_W)
    return LABEL_W + ((v - view_start) / span) * w


def _time_v_at_x(x: float, view_start: int, view_end: int, width: int) -> int:
    """Inverse of `_x_at_time_v`; see its docstring for why *width* is explicit."""
    span = max(1, view_end - view_start)
    w = max(1, width - LABEL_W)
    frac = (x - LABEL_W) / w
    frac = min(1.0, max(0.0, frac))
    return int(round(view_start + frac * span))


@dataclass(frozen=True)
class Row:
    """One row of the timeline tree: a keypoint group header, or a leaf keypoint."""

    kind: str                 # "group" | "leaf"
    label: str
    kp_indices: tuple[int, ...]
    depth: int = 0


def build_rows(pose_model: PoseModel, expanded: set[str]) -> list[Row]:
    """Build the tree row list from `pose_model.tree_groups` (a non-overlapping partition).

    Group rows are always shown; leaf (per-keypoint) rows only appear for
    groups whose name is in *expanded*. Any keypoint indices not covered by
    `tree_groups` (a pose model that doesn't fully partition) fall back into
    a flat "Other" group so no keypoint is silently dropped.
    """
    rows: list[Row] = []
    covered: set[int] = set()
    for group_name in pose_model.tree_groups:
        idx = tuple(sorted(pose_model.group_indices(group_name)))
        covered.update(idx)
        rows.append(Row(kind="group", label=group_name, kp_indices=idx))
        if group_name in expanded:
            for kp_idx in idx:
                rows.append(Row(kind="leaf", label=pose_model.name_of(kp_idx),
                                 kp_indices=(kp_idx,), depth=1))

    leftover = tuple(sorted(set(pose_model.all_indices) - covered))
    if leftover:
        rows.append(Row(kind="group", label="Other", kp_indices=leftover))
        if "Other" in expanded:
            for kp_idx in leftover:
                rows.append(Row(kind="leaf", label=pose_model.name_of(kp_idx),
                                 kp_indices=(kp_idx,), depth=1))
    return rows


def _eye_icon_x_range() -> tuple[float, float]:
    """(x_start, x_end) of the visibility-toggle hotspot — same for every row."""
    x = LABEL_W - _EYE_ICON_W - _EYE_ICON_MARGIN
    return (x, x + _EYE_ICON_W)


def _eye_icon_rect(row_y: int) -> tuple[float, float, float, float]:
    """(x, y, w, h) of the visibility-toggle icon for a row at *row_y*."""
    x, _x_end = _eye_icon_x_range()
    h = ROW_H - 4
    y = row_y + 2
    return (x, y, _EYE_ICON_W, h)


def _draw_eye_icon(painter: QPainter, rect: tuple[float, float, float, float], visible: bool) -> None:
    """A small hand-drawn eye (outline + pupil), or the same shape with a
    diagonal slash through it when hidden — avoids depending on an emoji /
    icon font being available for a plain QPainter widget."""
    x, y, w, h = rect
    cx, cy = x + w / 2, y + h / 2
    ew, eh = w * 0.75, h * 0.6
    color = _EYE_ICON_COLOR if visible else _EYE_ICON_HIDDEN_COLOR
    painter.setPen(QPen(color, 1.2))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(QPointF(cx, cy), ew / 2, eh / 2)
    if visible:
        painter.setBrush(color)
        painter.drawEllipse(QPointF(cx, cy), eh * 0.2, eh * 0.2)
        painter.setBrush(Qt.BrushStyle.NoBrush)
    else:
        painter.drawLine(QPointF(x + 1, y + 1), QPointF(x + w - 1, y + h - 1))


_DRAG_THRESHOLD = 5  # px — matches _KP_DRAG_THRESHOLD in content_panels.py's _ImageCanvas


class _TimelineCanvas(QWidget):
    """Custom-painted tree + time-colored cells for one camera's keypoint status.

    Purely a selection surface: click clears the selection, drag selects
    keypoints + a frame range, Ctrl+click toggles a keyframe. It does *not*
    move the playhead — that's `_RulerWidget`'s job, so scrubbing never
    disturbs an in-progress multi-keypoint selection.
    """

    # kp_indices(set[int]), range_start_v, range_end_v, ctrl (add-to-selection vs. replace)
    rubber_band_selected = Signal(object, int, int, bool)
    # kp_idx, time_v (ms from t_start) — Ctrl+click on a leaf row's cell
    keyframe_toggled = Signal(int, int)
    # time_v, ms from t_start — any click in the track area also moves the
    # playhead (in addition to whatever selection action it performs), so
    # clicking a keypoint row behaves the way clicking a slider used to.
    time_scrubbed = Signal(int)
    # emitted whenever the visible time window changes (zoom in/out/fit), so the
    # host container can resync its panning scrollbar and the ruler can redraw
    view_changed = Signal()
    # kp_indices(frozenset[int]) — eye-icon click on a row's label column;
    # the host decides show-vs-hide (see PersonCropGridWidget's handler)
    visibility_toggled = Signal(object)
    # kp_indices(tuple[int, ...]) — "Select to marker" chosen from a row's
    # right-click menu; the host computes the actual frame range from the
    # marker and the current playhead (this widget doesn't own DB state)
    select_to_marker_requested = Signal(object)
    # emitted whenever the marker moves/clears, so the ruler (a sibling
    # widget that reads marker_v() directly rather than duplicating it,
    # same as view_range()) knows to repaint
    marker_changed = Signal()
    # no-payload: "Disable selected" / "Enable selected" / "Interpolate
    # missing" from a row's right-click menu, acting on the host's current
    # selection + range — same three actions as the crop-grid canvas menu
    disable_selected_requested = Signal()
    enable_selected_requested = Signal()
    interpolate_missing_requested = Signal()

    def __init__(self, pose_model: PoseModel, parent=None) -> None:
        super().__init__(parent)
        self._pose_model = pose_model
        self._expanded: set[str] = set()
        self._rows: list[Row] = build_rows(pose_model, self._expanded)

        self._status_by_frame: dict[int, object] = {}   # frame -> int8[N] (active camera)
        self._inlier_counts: dict[int, object] = {}      # frame -> int16[N] (cross-camera)
        self._n_cameras: int = 1

        self._t_start = 0.0
        self._t_end = 0.0
        self._svid: str | None = None
        self._sync_table = None

        # Visible time window, in ms from t_start.  Defaults to the full trial
        # span; zoom() / zoom_fit() narrow or reset it.
        self._view_start_v: int = 0
        self._view_end_v: int = 1

        self._sel_kp_indices: set[int] = set()
        self._hidden_kp: frozenset[int] = frozenset()
        self._range_start_v: int | None = None
        self._range_end_v: int | None = None
        self._current_v: int = 0
        self._marker_v: int | None = None

        self._edit_mode = False
        self._drag_start: tuple[float, float] | None = None
        self._drag_current: tuple[float, float] | None = None
        self._drag_ctrl = False
        self._drag_moved = False

        self._resize_to_rows()

    def set_edit_mode(self, enabled: bool) -> None:
        self._edit_mode = enabled
        if not enabled:
            self._drag_start = None
            self._drag_current = None
            self._drag_moved = False

    # ------------------------------------------------------------------
    # Data wiring (called by the host widget)
    # ------------------------------------------------------------------

    def set_pose_model(self, pose_model: PoseModel) -> None:
        self._pose_model = pose_model
        self._expanded = set()
        self._rebuild_rows()

    def toggle_group(self, name: str) -> None:
        if name in self._expanded:
            self._expanded.discard(name)
        else:
            self._expanded.add(name)
        self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        self._rows = build_rows(self._pose_model, self._expanded)
        self._resize_to_rows()
        self.update()

    def _resize_to_rows(self) -> None:
        self.setMinimumHeight(max(1, len(self._rows)) * ROW_H)

    def set_time_range(self, t_start: float, t_end: float, svid: str | None, sync_table) -> None:
        self._t_start = t_start
        self._t_end = t_end
        self._svid = svid
        self._sync_table = sync_table
        self._view_start_v = 0
        self._view_end_v = self._total_ms()
        self.update()

    def set_svid(self, svid: str | None) -> None:
        """Switch which camera's frames `_frame_at_time_v` resolves against,
        without resetting the view window — t_start/t_end/sync_table are the
        same object across all cameras in a sequence, so there's no reason
        for a mere camera switch (or a status refresh after an edit) to reset
        the user's current zoom/pan. See KeypointTimelineWidget.set_svid."""
        self._svid = svid
        self.update()

    def set_status_data(
        self, status_by_frame: dict[int, object], inlier_counts: dict[int, object], n_cameras: int,
    ) -> None:
        self._status_by_frame = status_by_frame
        self._inlier_counts = inlier_counts
        self._n_cameras = max(1, n_cameras)
        self.update()

    def set_selection(
        self, sel_kp_indices: set[int], range_start_v: int | None, range_end_v: int | None,
    ) -> None:
        self._sel_kp_indices = set(sel_kp_indices)
        self._range_start_v = range_start_v
        self._range_end_v = range_end_v
        self.update()

    def set_current_time_v(self, v: int) -> None:
        self._current_v = v
        self.update()

    def set_hidden(self, hidden: frozenset[int]) -> None:
        self._hidden_kp = hidden
        self.update()

    def set_marker(self, v: int | None) -> None:
        self._marker_v = v
        self.update()
        self.marker_changed.emit()

    def marker_v(self) -> int | None:
        return self._marker_v

    # ------------------------------------------------------------------
    # Zoom / pan
    # ------------------------------------------------------------------

    def total_ms(self) -> int:
        return self._total_ms()

    def view_range(self) -> tuple[int, int]:
        return self._view_start_v, self._view_end_v

    def zoom(self, factor: float) -> None:
        """Scale the visible window by *factor* (< 1 zooms in), anchored on
        the playhead — not the cursor — per user feedback: zooming around
        wherever the mouse happens to be was disorienting; the playhead is
        the one fixed reference point you're always looking at."""
        total = self._total_ms()
        span = max(1, self._view_end_v - self._view_start_v)
        anchor_v = self._current_v
        new_span = max(MIN_VIEW_SPAN_MS, min(total, int(round(span * factor))))
        frac = (anchor_v - self._view_start_v) / span
        new_start = int(round(anchor_v - frac * new_span))
        new_start = max(0, min(new_start, total - new_span))
        self._view_start_v = new_start
        self._view_end_v = new_start + new_span
        self.view_changed.emit()
        self.update()

    def zoom_fit(self) -> None:
        self._view_start_v = 0
        self._view_end_v = self._total_ms()
        self.view_changed.emit()
        self.update()

    def set_view_start(self, v: int) -> None:
        """Pan without changing zoom level; driven by the host's scrollbar."""
        span = self._view_end_v - self._view_start_v
        total = self._total_ms()
        v = max(0, min(v, max(0, total - span)))
        self._view_start_v = v
        self._view_end_v = v + span
        self.update()

    # ------------------------------------------------------------------
    # Geometry helpers — pure functions of state, independent of paintEvent
    # so they can be unit tested directly.
    # ------------------------------------------------------------------

    def _total_ms(self) -> int:
        return max(1, int((self._t_end - self._t_start) * 1000))

    def _view_span(self) -> int:
        return max(1, self._view_end_v - self._view_start_v)

    def _x_at_time_v(self, v: int) -> float:
        return _x_at_time_v(v, self._view_start_v, self._view_end_v, self.width())

    def _time_v_at_x(self, x: float) -> int:
        return _time_v_at_x(x, self._view_start_v, self._view_end_v, self.width())

    def _row_index_at_y(self, y: int) -> int | None:
        idx = y // ROW_H
        if 0 <= idx < len(self._rows):
            return idx
        return None

    def _row_at_y(self, y: int) -> Row | None:
        idx = self._row_index_at_y(y)
        return self._rows[idx] if idx is not None else None

    def _frame_at_time_v(self, v: int) -> int | None:
        if self._sync_table is None or self._svid is None:
            return None
        return self._sync_table.lookup(self._t_start + v / 1000.0, self._svid)

    def _estimate_ms_per_frame(self) -> float | None:
        """Probe the sync table to find roughly how many ms one frame spans.

        No public fps accessor exists on the sync table object handed to us
        (it only offers `lookup(t, svid) -> frame`), so this walks forward
        from the current view until the returned frame number changes, then
        binary-searches for a tighter estimate of exactly where.
        """
        f0 = self._frame_at_time_v(self._view_start_v)
        if f0 is None:
            return None
        lo, hi = 0.0, None
        for probe_ms in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
            f1 = self._frame_at_time_v(self._view_start_v + probe_ms)
            if f1 is not None and f1 != f0:
                hi = float(probe_ms)
                break
            lo = float(probe_ms)
        if hi is None:
            return None
        for _ in range(8):
            mid = (lo + hi) / 2
            f_mid = self._frame_at_time_v(self._view_start_v + mid)
            if f_mid is not None and f_mid != f0:
                hi = mid
            else:
                lo = mid
        return hi

    def _should_gap_frames(self) -> bool:
        """True once each frame is wide enough on screen to draw a small gap
        between adjacent frame cells (see _FRAME_GAP_PX_THRESHOLD)."""
        ms_per_frame = self._estimate_ms_per_frame()
        if ms_per_frame is None:
            return False
        w = max(1, self.width() - LABEL_W)
        px_per_frame = ms_per_frame * w / self._view_span()
        return px_per_frame > _FRAME_GAP_PX_THRESHOLD

    def _status_columns(self, row: Row, split_by_frame: bool = False) -> list[tuple[int, int, int]]:
        """Run-length-encoded [(x_px, width_px, status_code)] segments for *row*.

        status_code is the max (highest-precedence) axis-1 code across the
        row's kp_indices at each sampled time; -1 means "no data". When
        *split_by_frame* is set, a new segment also starts whenever the
        sampled frame number changes (even if the status code doesn't), so
        callers can draw a gap at each frame boundary once zoomed in enough
        for that to be legible.
        """
        w = max(0, self.width() - LABEL_W)
        if w <= 0 or not row.kp_indices:
            return []
        segments: list[tuple[int, int, int]] = []
        cur_code: int | None = None
        cur_frame: int | None = None
        cur_start = 0
        x = 0
        while x < w:
            v = self._time_v_at_x(LABEL_W + x)
            frame = self._frame_at_time_v(v)
            status = self._status_by_frame.get(frame) if frame is not None else None
            if status is None:
                code = -1
            else:
                code = max(
                    (int(status[i]) for i in row.kp_indices if i < status.shape[0]),
                    default=-1,
                )
            boundary = code != cur_code or (split_by_frame and frame != cur_frame)
            if boundary:
                if cur_code is not None:
                    segments.append((LABEL_W + cur_start, x - cur_start, cur_code))
                cur_code = code
                cur_frame = frame
                cur_start = x
            x += COL_W
        if cur_code is not None:
            segments.append((LABEL_W + cur_start, w - cur_start, cur_code))
        return segments

    def _inlier_fraction_columns(self, kp_idx: int) -> list[tuple[int, int, float]]:
        """Same run-length scheme as `_status_columns`, for the inlier-count hint bar."""
        w = max(0, self.width() - LABEL_W)
        if w <= 0:
            return []
        segments: list[tuple[int, int, float]] = []
        cur_val: float | None = None
        cur_start = 0
        x = 0
        while x < w:
            v = self._time_v_at_x(LABEL_W + x)
            frame = self._frame_at_time_v(v)
            counts = self._inlier_counts.get(frame) if frame is not None else None
            if counts is not None and kp_idx < counts.shape[0]:
                frac = float(counts[kp_idx]) / self._n_cameras
            else:
                frac = 0.0
            if cur_val is None or abs(frac - cur_val) > 1e-6:
                if cur_val is not None:
                    segments.append((LABEL_W + cur_start, x - cur_start, cur_val))
                cur_val = frac
                cur_start = x
            x += COL_W
        if cur_val is not None:
            segments.append((LABEL_W + cur_start, w - cur_start, cur_val))
        return segments

    def _kp_indices_in_row_range(self, y0: float, y1: float) -> set[int]:
        """Union of kp_indices for every row whose band intersects [y0, y1],
        excluding hidden keypoints — a rubber-band drag over a group with
        some hidden members must not select the hidden ones."""
        if not self._rows:
            return set()
        idx0 = max(0, int(y0) // ROW_H)
        idx1 = min(len(self._rows) - 1, int(y1) // ROW_H)
        result: set[int] = set()
        for i in range(idx0, idx1 + 1):
            result.update(self._rows[i].kp_indices)
        return result - self._hidden_kp

    def _is_row_selected(self, row: Row) -> bool:
        return bool(self._sel_kp_indices) and bool(set(row.kp_indices) & self._sel_kp_indices)

    def _is_row_hidden(self, row: Row) -> bool:
        """True once *every* keypoint in the row is hidden — drives both the
        eye-icon glyph and the row-dimming overlay. A group with only some
        children hidden still reads as visible (open eye), since there's
        real, interactable content left in it."""
        return bool(row.kp_indices) and all(i in self._hidden_kp for i in row.kp_indices)

    def _range_frame_pixel_bounds(self) -> tuple[float, float] | None:
        """Pixel span covering every whole frame in [_range_start_v, _range_end_v].

        The active range is selected in frames, not milliseconds, but
        `_range_start_v`/`_range_end_v` are stored as raw slider ms and don't
        line up with frame boundaries — drawing the overlay directly from
        them highlighted a fractional sliver of the first/last frame instead
        of the whole cell. Snap outward to the frames' full pixel extent
        instead. Falls back to the raw ms mapping when there's no sync table
        (e.g. in isolated unit tests) so the overlay still renders something.
        """
        if self._range_start_v is None or self._range_end_v is None:
            return None
        frame_lo = self._frame_at_time_v(min(self._range_start_v, self._range_end_v))
        frame_hi = self._frame_at_time_v(max(self._range_start_v, self._range_end_v))
        if frame_lo is None or frame_hi is None:
            return (self._x_at_time_v(self._range_start_v), self._x_at_time_v(self._range_end_v))
        if frame_lo > frame_hi:
            frame_lo, frame_hi = frame_hi, frame_lo

        w = max(0, self.width() - LABEL_W)
        if w <= 0:
            return None
        x_min: float | None = None
        x_max: float | None = None
        x = 0
        while x < w:
            v = self._time_v_at_x(LABEL_W + x)
            frame = self._frame_at_time_v(v)
            if frame is not None and frame_lo <= frame <= frame_hi:
                if x_min is None:
                    x_min = float(LABEL_W + x)
                x_max = float(LABEL_W + x + COL_W)
            x += COL_W
        if x_min is None or x_max is None:
            return None
        return (x_min, x_max)

    # ------------------------------------------------------------------
    # Mouse: group-row expand/collapse (always); a track-area click also
    # moves the playhead (always, like clicking a slider used to); rubber-
    # band select / click-to-clear / Ctrl+click keyframe toggle happen only
    # in edit mode. `_RulerWidget` provides the same seeking without any
    # selection side effect, and stays usable while this canvas is hidden
    # behind the timeline's collapse toggle.
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            eye_x0, eye_x1 = _eye_icon_x_range()
            if eye_x0 <= pos.x() < eye_x1:
                row = self._row_at_y(int(pos.y()))
                if row is not None and row.kp_indices:
                    self.visibility_toggled.emit(frozenset(row.kp_indices))
                    return
            if pos.x() < LABEL_W:
                row = self._row_at_y(int(pos.y()))
                if row is not None and row.kind == "group":
                    self.toggle_group(row.label)
                    return
            else:
                self.time_scrubbed.emit(self._time_v_at_x(pos.x()))
            if self._edit_mode:
                ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                self._drag_start = (pos.x(), pos.y())
                self._drag_current = (pos.x(), pos.y())
                self._drag_ctrl = ctrl
                self._drag_moved = False
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_start is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = event.position()
            self._drag_current = (pos.x(), pos.y())
            dx = pos.x() - self._drag_start[0]
            dy = pos.y() - self._drag_start[1]
            if dx * dx + dy * dy >= _DRAG_THRESHOLD ** 2:
                self._drag_moved = True
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._drag_start is None:
            super().mouseReleaseEvent(event)
            return

        x0, y0 = self._drag_start
        x1, y1 = self._drag_current if self._drag_current is not None else self._drag_start

        if self._edit_mode:
            if self._drag_ctrl and not self._drag_moved:
                row = self._row_at_y(int(y0))
                if (row is not None and row.kind == "leaf" and x0 >= LABEL_W
                        and row.kp_indices[0] not in self._hidden_kp):
                    v = self._time_v_at_x(x0)
                    self.keyframe_toggled.emit(row.kp_indices[0], v)
            elif self._drag_moved:
                if x0 >= LABEL_W or x1 >= LABEL_W:
                    kp_indices = self._kp_indices_in_row_range(min(y0, y1), max(y0, y1))
                    if kp_indices:
                        v0 = self._time_v_at_x(min(x0, x1))
                        v1 = self._time_v_at_x(max(x0, x1))
                        self.rubber_band_selected.emit(kp_indices, v0, v1, self._drag_ctrl)
            elif not self._drag_ctrl and x0 >= LABEL_W:
                # Plain click, no drag: clear the selection entirely rather than
                # collapsing it down to just the clicked row — a stray click
                # shouldn't discard the rest of a multi-keypoint selection.
                v0 = self._time_v_at_x(x0)
                self.rubber_band_selected.emit(set(), v0, v0, False)

        self._drag_start = None
        self._drag_current = None
        self._drag_ctrl = False
        self._drag_moved = False
        self.update()

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        """Right-click a row: offer to select that row's keypoint(s) across
        the range between the marker and the current playhead, plus —
        whenever a selection already exists — the same disable/enable/
        interpolate-missing actions the crop-grid canvas's menu has. A
        selection made via "select to marker" naturally continues here,
        so those actions need to be reachable from this menu too, not
        only from the crop grid."""
        if not self._edit_mode:
            return
        pos = event.pos()
        row = self._row_at_y(int(pos.y())) if pos.x() >= LABEL_W else None

        menu = QMenu(self)
        if self._marker_v is not None and row is not None and row.kp_indices:
            kp_indices = tuple(i for i in row.kp_indices if i not in self._hidden_kp)
            if kp_indices:
                label = row.label if row.kind == "leaf" else f"{row.label} (all)"
                action = menu.addAction(f"Select {label} to marker")
                action.triggered.connect(
                    lambda checked=False, idx=kp_indices: self.select_to_marker_requested.emit(idx)
                )
        if self._sel_kp_indices:
            if not menu.isEmpty():
                menu.addSeparator()
            disable_act = menu.addAction("Disable selected")
            disable_act.triggered.connect(lambda: self.disable_selected_requested.emit())
            enable_act = menu.addAction("Enable selected")
            enable_act.triggered.connect(lambda: self.enable_selected_requested.emit())
            if self._range_start_v is not None:
                interp_act = menu.addAction("Interpolate missing")
                interp_act.triggered.connect(lambda: self.interpolate_missing_requested.emit())
        if menu.isEmpty():
            return
        menu.exec(event.globalPos())

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 0.8 if event.angleDelta().y() > 0 else 1.25
            self.zoom(factor)
            event.accept()
            return
        super().wheelEvent(event)

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), _BG_COLOR)
            split_by_frame = self._should_gap_frames()
            range_bounds = self._range_frame_pixel_bounds()
            for row_idx, row in enumerate(self._rows):
                y = row_idx * ROW_H
                is_selected = self._is_row_selected(row)
                is_hidden = self._is_row_hidden(row)
                if is_selected:
                    painter.fillRect(0, y, self.width(), ROW_H, _SELECTED_ROW_BG)

                painter.setPen(_LABEL_COLOR)
                indent = 6 + row.depth * 12
                prefix = ""
                if row.kind == "group":
                    prefix = "▼ " if row.label in self._expanded else "▶ "
                painter.drawText(indent, y + ROW_H - 4, prefix + row.label)
                _draw_eye_icon(painter, _eye_icon_rect(y), visible=not is_hidden)

                for x, w, code in self._status_columns(row, split_by_frame):
                    color = _STATUS_COLORS.get(code, _NO_DATA_COLOR)
                    draw_w = max(1, w - _FRAME_GAP_PX) if split_by_frame else w
                    painter.fillRect(x, y + 1, max(1, draw_w), ROW_H - 4, color)

                if row.kind == "leaf":
                    for x, w, frac in self._inlier_fraction_columns(row.kp_indices[0]):
                        bar_w = max(0, int(round(w * frac)))
                        if bar_w > 0:
                            painter.fillRect(x, y + ROW_H - 3, bar_w, 2, _INLIER_BAR_COLOR)

                # The active-range overlay only makes sense over rows that are
                # actually part of the selection it applies to — painting it
                # across every row (selected or not) made it look like the
                # whole time range was highlighted rather than just the
                # selected keypoints within it. Bounds are frame-snapped (see
                # _range_frame_pixel_bounds) since the range is a set of whole
                # frames, not a continuous span of milliseconds.
                if is_selected and range_bounds is not None:
                    x1, x2 = range_bounds
                    painter.fillRect(int(x1), y, max(1, int(x2 - x1)), ROW_H, _RANGE_OVERLAY)

                # Dim the whole row (label, status cells, everything) once
                # every keypoint in it is hidden — a translucent wash over
                # already-drawn content is simpler than threading a "hidden"
                # branch through every color decision above.
                if is_hidden:
                    painter.fillRect(0, y, self.width(), ROW_H, _HIDDEN_ROW_OVERLAY)

            if (self._marker_v is not None
                    and self._view_start_v <= self._marker_v <= self._view_end_v):
                mx = self._x_at_time_v(self._marker_v)
                painter.setPen(QPen(_MARKER_COLOR, 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(mx), 0, int(mx), self.height())

            if self._view_start_v <= self._current_v <= self._view_end_v:
                px = self._x_at_time_v(self._current_v)
                painter.setPen(QPen(_PLAYHEAD_COLOR, 2))
                painter.drawLine(int(px), 0, int(px), self.height())

            if self._drag_moved and self._drag_start is not None and self._drag_current is not None:
                x0, y0 = self._drag_start
                x1, y1 = self._drag_current
                rect_x, rect_w = min(x0, x1), abs(x1 - x0)
                rect_y, rect_h = min(y0, y1), abs(y1 - y0)
                painter.fillRect(int(rect_x), int(rect_y), max(1, int(rect_w)), max(1, int(rect_h)),
                                  _DRAG_RECT_COLOR)
        finally:
            painter.end()


_COLLAPSE_HOTSPOT_W = 20  # px — clickable width of the collapse arrow, left edge of the ruler


class _RulerWidget(QWidget):
    """Fixed-height timestamp ruler above the row tree: the only place that
    moves the playhead, plus the collapse/expand arrow (moved here from the
    tab row since this is the row that stays visible while collapsed — the
    arrow belongs where it's always reachable). Always visible, even while
    the row tree is collapsed, so scrubbing never requires expanding the
    timeline.

    Reads view/zoom state from a `_TimelineCanvas` instance (same module,
    tightly coupled sibling widgets — see the module docstring) rather than
    duplicating zoom/pan bookkeeping, but maps time <-> pixel using *its own*
    width (see the module-level `_x_at_time_v`/`_time_v_at_x`), since the
    ruler and canvas widths only match when the canvas's QScrollArea has no
    vertical scrollbar — see `set_right_margin`.
    """

    time_scrubbed = Signal(int)  # time_v, ms from t_start
    collapse_clicked = Signal()
    # no-payload: "Select all keypoints to marker" from the ruler's
    # right-click menu; the host selects every visible keypoint and the
    # frame range between the marker and the current playhead
    select_all_to_marker_requested = Signal()

    def __init__(self, canvas: _TimelineCanvas, parent=None) -> None:
        super().__init__(parent)
        self._canvas = canvas
        self._dragging = False
        self._collapsed = True
        # Extra right-side inset so this widget's drawable width matches the
        # canvas's actual (scrollbar-shrunk, when a vertical scrollbar is
        # showing) width — see KeypointTimelineWidget._sync_ruler_margin.
        self._right_margin = 0
        self.setFixedHeight(RULER_H)

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self.update()

    def set_right_margin(self, px: int) -> None:
        if px != self._right_margin:
            self._right_margin = px
            self.update()

    def _x_at_time_v(self, v: int) -> float:
        view_start, view_end = self._canvas.view_range()
        return _x_at_time_v(v, view_start, view_end, self.width() - self._right_margin)

    def _time_v_at_x(self, x: float) -> int:
        view_start, view_end = self._canvas.view_range()
        return _time_v_at_x(x, view_start, view_end, self.width() - self._right_margin)

    def _pick_tick_interval_ms(self, span_ms: int, px_width: int, min_px_gap: int = 60) -> int:
        if px_width <= 0 or span_ms <= 0:
            return _NICE_INTERVALS_MS[-1]
        for interval in _NICE_INTERVALS_MS:
            if px_width * interval / span_ms >= min_px_gap:
                return interval
        return _NICE_INTERVALS_MS[-1]

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            if pos.x() < _COLLAPSE_HOTSPOT_W:
                self.collapse_clicked.emit()
                return
            if pos.x() >= LABEL_W:
                self._dragging = True
                self.time_scrubbed.emit(self._time_v_at_x(pos.x()))
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._dragging and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = event.position()
            self.time_scrubbed.emit(self._time_v_at_x(pos.x()))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._dragging = False
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        """Right-click the ruler: set the marker at the clicked time, clear
        it if one's already set, or select every keypoint across the range
        between the marker and the current playhead. Only one marker at a
        time — setting a new one silently replaces whatever was there."""
        pos = event.pos()
        if pos.x() < LABEL_W:
            return
        menu = QMenu(self)
        v = self._time_v_at_x(pos.x())
        set_act = menu.addAction("Set marker here")
        set_act.triggered.connect(lambda checked=False, mv=v: self._canvas.set_marker(mv))
        if self._canvas.marker_v() is not None:
            clear_act = menu.addAction("Clear marker")
            clear_act.triggered.connect(lambda: self._canvas.set_marker(None))
            menu.addSeparator()
            select_all_act = menu.addAction("Select all keypoints to marker")
            select_all_act.triggered.connect(
                lambda: self.select_all_to_marker_requested.emit()
            )
        menu.exec(event.globalPos())

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 0.8 if event.angleDelta().y() > 0 else 1.25
            self._canvas.zoom(factor)
            event.accept()
            return
        super().wheelEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), _BG_COLOR)

            painter.setPen(_LABEL_COLOR)
            painter.drawText(4, self.height() - 6, "▸" if self._collapsed else "▾")

            view_start, view_end = self._canvas.view_range()
            span = max(1, view_end - view_start)
            w = max(0, self.width() - self._right_margin - LABEL_W)
            interval = self._pick_tick_interval_ms(span, w)
            t_start = self._canvas._t_start

            v = (view_start // interval) * interval
            while v <= view_end:
                if v >= view_start:
                    x = self._x_at_time_v(v)
                    painter.drawLine(int(x), self.height() - 6, int(x), self.height())
                    painter.drawText(int(x) + 2, self.height() - 8, _fmt_tick(t_start + v / 1000.0))
                v += interval

            marker_v = self._canvas.marker_v()
            if marker_v is not None and view_start <= marker_v <= view_end:
                mx = self._x_at_time_v(marker_v)
                painter.setPen(QPen(_MARKER_COLOR, 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(mx), 0, int(mx), self.height())
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(_MARKER_COLOR)
                painter.drawPolygon([
                    QPointF(mx, 0),
                    QPointF(mx + _MARKER_FLAG_W, _MARKER_FLAG_H / 2),
                    QPointF(mx, _MARKER_FLAG_H),
                ])
                painter.setBrush(Qt.BrushStyle.NoBrush)

            cur_v = self._canvas._current_v
            if view_start <= cur_v <= view_end:
                px = self._x_at_time_v(cur_v)
                painter.setPen(QPen(_PLAYHEAD_COLOR, 2))
                painter.drawLine(int(px), 0, int(px), self.height())
        finally:
            painter.end()


class KeypointTimelineWidget(QWidget):
    """Container: collapse toggle + camera tabs + zoom controls + always-
    visible `_RulerWidget` + scrollable `_TimelineCanvas` + a horizontal
    scrollbar for panning when zoomed in.

    Starts collapsed to tab-row + ruler height only (manual keypoint editing
    is the exception, not the common case) — the ruler stays visible even
    collapsed, so the timeline still works as a scrub control without
    expanding it. Once expanded, the row-tree height is controlled by
    whatever QSplitter the host places this widget in.
    """

    camera_changed = Signal(int)
    rubber_band_selected = Signal(object, int, int, bool)
    keyframe_toggled = Signal(int, int)
    time_scrubbed = Signal(int)
    collapsed_changed = Signal(bool)
    visibility_toggled = Signal(object)
    select_to_marker_requested = Signal(object)
    select_all_to_marker_requested = Signal()
    disable_selected_requested = Signal()
    enable_selected_requested = Signal()
    interpolate_missing_requested = Signal()

    def __init__(self, pose_model: PoseModel, cameras: list[dict], parent=None) -> None:
        super().__init__(parent)
        self._cameras = cameras
        self._active_cam_idx = 0
        self._cam_buttons: list[QPushButton] = []
        self._collapsed = True

        # Camera tabs + zoom controls. The collapse/expand arrow lives on the
        # ruler below (not here) — it needs to work while this row's sibling
        # canvas is hidden, and the ruler is the row that stays visible then.
        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(0, 0, 0, 0)
        for i, cam in enumerate(cameras):
            btn = QPushButton(cam.get("label", str(i)))
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.clicked.connect(lambda _checked=False, idx=i: self._on_tab_clicked(idx))
            self._cam_buttons.append(btn)
            tab_row.addWidget(btn)
        tab_row.addStretch()

        self._fit_btn = QPushButton("Fit")
        zoom_out_btn = QPushButton("−")
        zoom_in_btn = QPushButton("+")
        for b in (zoom_out_btn, zoom_in_btn, self._fit_btn):
            b.setFixedWidth(28)
        zoom_out_btn.setToolTip("Zoom out around the playhead (Ctrl+scroll)")
        zoom_in_btn.setToolTip("Zoom in around the playhead (Ctrl+scroll)")
        self._fit_btn.setToolTip("Reset zoom to the full trial")
        zoom_out_btn.clicked.connect(lambda: self._canvas.zoom(1.25))
        zoom_in_btn.clicked.connect(lambda: self._canvas.zoom(0.8))
        self._fit_btn.clicked.connect(self._on_fit_clicked)
        tab_row.addWidget(zoom_out_btn)
        tab_row.addWidget(zoom_in_btn)
        tab_row.addWidget(self._fit_btn)

        self._canvas = _TimelineCanvas(pose_model)
        self._canvas.rubber_band_selected.connect(self.rubber_band_selected)
        self._canvas.keyframe_toggled.connect(self.keyframe_toggled)
        self._canvas.time_scrubbed.connect(self.time_scrubbed)
        self._canvas.visibility_toggled.connect(self.visibility_toggled)
        self._canvas.select_to_marker_requested.connect(self.select_to_marker_requested)
        self._canvas.disable_selected_requested.connect(self.disable_selected_requested)
        self._canvas.enable_selected_requested.connect(self.enable_selected_requested)
        self._canvas.interpolate_missing_requested.connect(self.interpolate_missing_requested)
        self._canvas.view_changed.connect(self._sync_hscroll)
        self._canvas.marker_changed.connect(lambda: self._ruler.update())

        self._ruler = _RulerWidget(self._canvas)
        self._ruler.time_scrubbed.connect(self.time_scrubbed)
        self._ruler.collapse_clicked.connect(self._on_collapse_clicked)
        self._ruler.select_all_to_marker_requested.connect(self.select_all_to_marker_requested)
        self._canvas.view_changed.connect(self._ruler.update)

        self._canvas_scroll = QScrollArea()
        self._canvas_scroll.setWidget(self._canvas)
        self._canvas_scroll.setWidgetResizable(True)
        # Keep the ruler's time<->pixel mapping aligned with the canvas: when
        # the row tree grows tall enough to need a vertical scrollbar, the
        # canvas's viewport (and therefore its usable width) shrinks by the
        # scrollbar's width, but the ruler — being outside this scroll area —
        # wouldn't shrink to match on its own.
        self._canvas_scroll.verticalScrollBar().rangeChanged.connect(self._sync_ruler_margin)

        self._hscroll = QScrollBar(Qt.Orientation.Horizontal)
        self._hscroll.valueChanged.connect(self._on_hscroll)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(tab_row)
        layout.addWidget(self._ruler)
        layout.addWidget(self._canvas_scroll)
        layout.addWidget(self._hscroll)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._apply_collapsed_state()

    def _on_tab_clicked(self, idx: int) -> None:
        self.set_active_camera(idx)
        self.camera_changed.emit(idx)

    def set_active_camera(self, idx: int) -> None:
        self._active_cam_idx = idx
        for i, btn in enumerate(self._cam_buttons):
            btn.setChecked(i == idx)

    def active_camera_index(self) -> int:
        return self._active_cam_idx

    # Collapse/expand -----------------------------------------------------

    def _on_collapse_clicked(self) -> None:
        self.set_collapsed(not self._collapsed)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self._apply_collapsed_state()
        self.collapsed_changed.emit(collapsed)

    def _apply_collapsed_state(self) -> None:
        self._ruler.set_collapsed(self._collapsed)
        self._canvas_scroll.setVisible(not self._collapsed)
        self._hscroll.setVisible(not self._collapsed)
        if self._collapsed:
            bar_h = self._fit_btn.sizeHint().height() + RULER_H + 12
            self.setMaximumHeight(bar_h)
            self.setMinimumHeight(bar_h)
        else:
            self.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
            self.setMinimumHeight(60 + RULER_H)
            if self.height() <= self.minimumHeight():
                self.resize(self.width(), _EXPANDED_HEIGHT)

    # Zoom / pan ------------------------------------------------------------

    def _on_fit_clicked(self) -> None:
        self._canvas.zoom_fit()

    def _sync_hscroll(self) -> None:
        total = self._canvas.total_ms()
        start, end = self._canvas.view_range()
        span = max(1, end - start)
        self._hscroll.blockSignals(True)
        self._hscroll.setRange(0, max(0, total - span))
        self._hscroll.setPageStep(span)
        self._hscroll.setValue(start)
        self._hscroll.blockSignals(False)

    def _on_hscroll(self, value: int) -> None:
        self._canvas.set_view_start(value)
        self._ruler.update()

    def _sync_ruler_margin(self, *_args) -> None:
        """Match the ruler's right inset to the canvas's vertical scrollbar
        width so tick marks line up with the rows underneath (see the
        `_RulerWidget` docstring). Connected to the scrollbar's rangeChanged
        signal, which fires synchronously whenever the row tree's content
        height changes (group expand/collapse), unlike isVisible() checked
        right after a resize, which can still reflect a stale Qt layout."""
        vbar = self._canvas_scroll.verticalScrollBar()
        margin = vbar.sizeHint().width() if vbar.isVisible() else 0
        self._ruler.set_right_margin(margin)

    # Pass-throughs to the canvas ---------------------------------------

    def set_pose_model(self, pose_model: PoseModel) -> None:
        self._canvas.set_pose_model(pose_model)

    def set_time_range(self, t_start: float, t_end: float, svid: str | None, sync_table) -> None:
        self._canvas.set_time_range(t_start, t_end, svid, sync_table)
        self._sync_hscroll()
        self._ruler.update()

    def set_svid(self, svid: str | None) -> None:
        """Switch the active camera's frame source without resetting zoom/pan
        (see `_TimelineCanvas.set_svid`) — used for camera switches and
        post-edit status refreshes, as opposed to `set_time_range`, which is
        only needed once at setup since t_start/t_end/sync_table don't vary
        per camera within a sequence."""
        self._canvas.set_svid(svid)
        self._ruler.update()

    def set_status_data(
        self, status_by_frame: dict[int, object], inlier_counts: dict[int, object], n_cameras: int,
    ) -> None:
        self._canvas.set_status_data(status_by_frame, inlier_counts, n_cameras)

    def set_selection(
        self, sel_kp_indices: set[int], range_start_v: int | None, range_end_v: int | None,
    ) -> None:
        self._canvas.set_selection(sel_kp_indices, range_start_v, range_end_v)

    def set_current_time_v(self, v: int) -> None:
        self._canvas.set_current_time_v(v)
        self._ruler.update()

    def set_hidden(self, hidden: frozenset[int]) -> None:
        self._canvas.set_hidden(hidden)

    def set_marker(self, v: int | None) -> None:
        self._canvas.set_marker(v)

    def marker_v(self) -> int | None:
        return self._canvas.marker_v()

    def set_edit_mode(self, enabled: bool) -> None:
        self._canvas.set_edit_mode(enabled)

"""keypoint_timeline_widget.py — dope-sheet style timeline for keypoint editing.

Phases 12-13 of the keypoint-editing timeline view (see
docs/keypoint-editing/keypoint-editing-design.md, "Improvements" section):

- Phase 12: a custom-painted tree of keypoint/group rows colored by
  `app.pose.timeline_status` axis-1 status, scoped to one camera at a time
  (matching `PersonCropGridWidget._sel_cam_idx`), with a playhead synced to
  the existing time slider.
- Phase 13: rubber-band drag (or a plain click) selects keypoints + a frame
  range, and Ctrl+click toggles a keyframe — both only emit signals, the
  actual DB writes and `_sel_kp_indices`/`_range_start_v` state live on the
  host widget (`PersonCropGridWidget`), matching the "state is owned by the
  host, not duplicated per-widget" rule from the design doc.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QScrollArea, QVBoxLayout, QWidget

from app.pose.kp_models import PoseModel
from app.pose.timeline_status import STATUS_BLUE, STATUS_GREEN, STATUS_GREY, STATUS_YELLOW

ROW_H = 16
LABEL_W = 140
COL_W = 2  # px per sampled time bucket when drawing status cells

_STATUS_COLORS = {
    STATUS_GREEN: QColor(80, 170, 80),
    STATUS_YELLOW: QColor(210, 190, 60),
    STATUS_BLUE: QColor(70, 130, 220),
    STATUS_GREY: QColor(120, 120, 120),
}
_NO_DATA_COLOR = QColor(45, 45, 45)
_BG_COLOR = QColor(30, 30, 30)
_LABEL_COLOR = QColor(220, 220, 220)
_SELECTED_LABEL_BG = QColor(60, 90, 130)
_RANGE_OVERLAY = QColor(255, 255, 255, 40)
_PLAYHEAD_COLOR = QColor(255, 80, 80)
_INLIER_BAR_COLOR = QColor(180, 180, 180)
_DRAG_RECT_COLOR = QColor(120, 170, 255, 60)


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


_DRAG_THRESHOLD = 5  # px — matches _KP_DRAG_THRESHOLD in content_panels.py's _ImageCanvas


class _TimelineCanvas(QWidget):
    """Custom-painted tree + time-colored cells for one camera's keypoint status."""

    # kp_indices(set[int]), range_start_v, range_end_v, ctrl (add-to-selection vs. replace)
    rubber_band_selected = Signal(object, int, int, bool)
    # kp_idx, time_v (ms from t_start) — Ctrl+click on a leaf row's cell
    keyframe_toggled = Signal(int, int)

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

        self._sel_kp_indices: set[int] = set()
        self._range_start_v: int | None = None
        self._range_end_v: int | None = None
        self._current_v: int = 0

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

    # ------------------------------------------------------------------
    # Geometry helpers — pure functions of state, independent of paintEvent
    # so they can be unit tested directly.
    # ------------------------------------------------------------------

    def _total_ms(self) -> int:
        return max(1, int((self._t_end - self._t_start) * 1000))

    def _x_at_time_v(self, v: int) -> float:
        w = max(1, self.width() - LABEL_W)
        return LABEL_W + (v / self._total_ms()) * w

    def _time_v_at_x(self, x: float) -> int:
        w = max(1, self.width() - LABEL_W)
        frac = (x - LABEL_W) / w
        frac = min(1.0, max(0.0, frac))
        return int(round(frac * self._total_ms()))

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

    def _status_columns(self, row: Row) -> list[tuple[int, int, int]]:
        """Run-length-encoded [(x_px, width_px, status_code)] segments for *row*.

        status_code is the max (highest-precedence) axis-1 code across the
        row's kp_indices at each sampled time; -1 means "no data".
        """
        w = max(0, self.width() - LABEL_W)
        if w <= 0 or not row.kp_indices:
            return []
        segments: list[tuple[int, int, int]] = []
        cur_code: int | None = None
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
            if code != cur_code:
                if cur_code is not None:
                    segments.append((LABEL_W + cur_start, x - cur_start, cur_code))
                cur_code = code
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
        """Union of kp_indices for every row whose band intersects [y0, y1]."""
        if not self._rows:
            return set()
        idx0 = max(0, int(y0) // ROW_H)
        idx1 = min(len(self._rows) - 1, int(y1) // ROW_H)
        result: set[int] = set()
        for i in range(idx0, idx1 + 1):
            result.update(self._rows[i].kp_indices)
        return result

    # ------------------------------------------------------------------
    # Mouse: group-row expand/collapse (always), rubber-band select and
    # Ctrl+click keyframe toggle (edit mode only)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            if pos.x() < LABEL_W:
                row = self._row_at_y(int(pos.y()))
                if row is not None and row.kind == "group":
                    self.toggle_group(row.label)
                    return
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

        if self._drag_ctrl and not self._drag_moved:
            row = self._row_at_y(int(y0))
            if row is not None and row.kind == "leaf" and x0 >= LABEL_W:
                v = self._time_v_at_x(x0)
                self.keyframe_toggled.emit(row.kp_indices[0], v)
        elif x0 >= LABEL_W or x1 >= LABEL_W:
            kp_indices = self._kp_indices_in_row_range(min(y0, y1), max(y0, y1))
            if kp_indices:
                v0 = self._time_v_at_x(min(x0, x1))
                v1 = self._time_v_at_x(max(x0, x1))
                self.rubber_band_selected.emit(kp_indices, v0, v1, self._drag_ctrl)

        self._drag_start = None
        self._drag_current = None
        self._drag_ctrl = False
        self._drag_moved = False
        self.update()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), _BG_COLOR)
            for row_idx, row in enumerate(self._rows):
                y = row_idx * ROW_H
                if self._sel_kp_indices and (set(row.kp_indices) & self._sel_kp_indices):
                    painter.fillRect(0, y, LABEL_W, ROW_H, _SELECTED_LABEL_BG)

                painter.setPen(_LABEL_COLOR)
                indent = 6 + row.depth * 12
                prefix = ""
                if row.kind == "group":
                    prefix = "▼ " if row.label in self._expanded else "▶ "
                painter.drawText(indent, y + ROW_H - 4, prefix + row.label)

                for x, w, code in self._status_columns(row):
                    color = _STATUS_COLORS.get(code, _NO_DATA_COLOR)
                    painter.fillRect(x, y + 1, max(1, w), ROW_H - 4, color)

                if row.kind == "leaf":
                    for x, w, frac in self._inlier_fraction_columns(row.kp_indices[0]):
                        bar_w = max(0, int(round(w * frac)))
                        if bar_w > 0:
                            painter.fillRect(x, y + ROW_H - 3, bar_w, 2, _INLIER_BAR_COLOR)

            if self._range_start_v is not None and self._range_end_v is not None:
                x1 = self._x_at_time_v(self._range_start_v)
                x2 = self._x_at_time_v(self._range_end_v)
                painter.fillRect(int(x1), 0, max(1, int(x2 - x1)), self.height(), _RANGE_OVERLAY)

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


class KeypointTimelineWidget(QWidget):
    """Container: camera tab strip + scrollable `_TimelineCanvas`."""

    camera_changed = Signal(int)
    rubber_band_selected = Signal(object, int, int, bool)
    keyframe_toggled = Signal(int, int)

    def __init__(self, pose_model: PoseModel, cameras: list[dict], parent=None) -> None:
        super().__init__(parent)
        self._cameras = cameras
        self._active_cam_idx = 0
        self._cam_buttons: list[QPushButton] = []

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

        self._canvas = _TimelineCanvas(pose_model)
        self._canvas.rubber_band_selected.connect(self.rubber_band_selected)
        self._canvas.keyframe_toggled.connect(self.keyframe_toggled)

        scroll = QScrollArea()
        scroll.setWidget(self._canvas)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(100)
        scroll.setMaximumHeight(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(tab_row)
        layout.addWidget(scroll)

    def _on_tab_clicked(self, idx: int) -> None:
        self.set_active_camera(idx)
        self.camera_changed.emit(idx)

    def set_active_camera(self, idx: int) -> None:
        self._active_cam_idx = idx
        for i, btn in enumerate(self._cam_buttons):
            btn.setChecked(i == idx)

    def active_camera_index(self) -> int:
        return self._active_cam_idx

    # Pass-throughs to the canvas ---------------------------------------

    def set_pose_model(self, pose_model: PoseModel) -> None:
        self._canvas.set_pose_model(pose_model)

    def set_time_range(self, t_start: float, t_end: float, svid: str | None, sync_table) -> None:
        self._canvas.set_time_range(t_start, t_end, svid, sync_table)

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

    def set_edit_mode(self, enabled: bool) -> None:
        self._canvas.set_edit_mode(enabled)

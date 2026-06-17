"""crop_editor.py — PersonCropGridWidget: multi-camera keypoint editing surface.

Shows per-camera JPEG crops from frame_cache_entries with a keypoint overlay
drawn from pose_observations (with pose_observation_edits applied).  Frame
navigation is instantaneous because it reads cached JPEG blobs, not raw video.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QByteArray, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.pose.db_cache import read_observations_with_edits, update_single_keypoint_edit

# Crop cell dimensions
_CELL_W = 320
_CELL_H = 240

# Keypoint display
_KP_RADIUS = 4
_KP_HI_CONF = QColor(50, 220, 50)      # green  — confident inlier
_KP_LO_CONF = QColor(220, 130, 50)     # orange — low confidence
_KP_OUTLIER = QColor(120, 120, 120)    # grey   — outlier (confidence == 0)
_KP_EDITED  = QColor(255, 220, 0)      # yellow — overridden by an edit

_CONF_THRESHOLD = 0.01  # below this → treat as outlier for display

# Mouse interaction
_DRAG_THRESHOLD  = 5    # minimum drag distance in display pixels before a move is registered
_HIT_RADIUS      = _KP_RADIUS + 4  # hit-test tolerance around each keypoint dot

# Trail overlay
_TRAIL_N = 10                               # default half-window in frame-slots
_TRAIL_PAST_COLOR   = QColor(220,  60,  60)       # red   — past frames
_TRAIL_FUTURE_COLOR = QColor( 60, 100, 220)       # blue  — future frames
_TRAIL_GHOST_COLOR  = QColor(140, 140, 140, 160)  # grey, semi-transparent — ghost
_TRAIL_LINE_WIDTH   = 1.5
_TRAIL_DOT_R        = 3


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _CameraSlot:
    """Per-camera metadata for one sequence."""
    camera_instance_id: str
    shot_video_id: str
    label: str
    # Sorted list of (first_frame, last_frame, track_id) for detection_track_assignments.
    # Needed to map video_frame → track_id for frame_cache_entries queries.
    track_ranges: list[tuple[int, int, int]] = field(default_factory=list)
    # Merged keypoints per video_frame (loaded once for the whole sequence).
    kp_by_frame: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class _FrameSlot:
    """One logical time step across all cameras."""
    timestamp_s: float
    per_cam: dict[str, int]  # camera_instance_id → video_frame


@dataclass
class _TrailPoint:
    """One point on the keypoint trail."""
    x: float
    y: float
    is_ghost: bool  # True → linearly interpolated; no real observation at this slot


@dataclass
class _TrailData:
    """Trail context for one selected keypoint in one camera cell."""
    kp_idx: int
    past: list[_TrailPoint]    # oldest first (slots before current)
    future: list[_TrailPoint]  # nearest first (slots after current)


# ---------------------------------------------------------------------------
# Trail computation (pure functions — testable without Qt)
# ---------------------------------------------------------------------------

def _slot_kp_pos(
    slot: _FrameSlot,
    kp_by_frame: dict[int, np.ndarray],
    camera_id: str,
    kp_idx: int,
) -> tuple[float, float] | None:
    """Return (x, y) for kp_idx at this slot if it is a real, non-outlier observation."""
    vf = slot.per_cam.get(camera_id)
    if vf is None:
        return None
    kp = kp_by_frame.get(vf)
    if kp is None or kp_idx >= kp.shape[0] or kp[kp_idx, 2] < _CONF_THRESHOLD:
        return None
    return float(kp[kp_idx, 0]), float(kp[kp_idx, 1])


def _build_trail_segment(
    frames: list[_FrameSlot],
    kp_by_frame: dict[int, np.ndarray],
    camera_id: str,
    slot_indices: list[int],
    kp_idx: int,
) -> list[_TrailPoint]:
    """Build trail points for a window of slot indices, interpolating gaps."""
    if not slot_indices:
        return []

    raw: list[tuple[float, float] | None] = [
        _slot_kp_pos(frames[i], kp_by_frame, camera_id, kp_idx)
        for i in slot_indices
    ]
    n = len(raw)
    result: list[_TrailPoint | None] = [None] * n

    for i, pos in enumerate(raw):
        if pos is not None:
            result[i] = _TrailPoint(pos[0], pos[1], is_ghost=False)

    # Fill gaps via linear interpolation between bracketing real points
    for i in range(n):
        if result[i] is not None:
            continue
        prev_anchor: tuple[int, _TrailPoint] | None = None
        next_anchor: tuple[int, _TrailPoint] | None = None
        for j in range(i - 1, -1, -1):
            if result[j] is not None and not result[j].is_ghost:
                prev_anchor = (j, result[j])
                break
        for j in range(i + 1, n):
            if result[j] is not None and not result[j].is_ghost:
                next_anchor = (j, result[j])
                break
        if prev_anchor is not None and next_anchor is not None:
            pi, pp = prev_anchor
            ni, np_ = next_anchor
            t = (i - pi) / (ni - pi)
            result[i] = _TrailPoint(
                pp.x + t * (np_.x - pp.x),
                pp.y + t * (np_.y - pp.y),
                is_ghost=True,
            )

    return [p for p in result if p is not None]


def compute_trail(
    frames: list[_FrameSlot],
    kp_by_frame: dict[int, np.ndarray],
    camera_id: str,
    current_slot_idx: int,
    kp_idx: int,
    n: int = _TRAIL_N,
) -> _TrailData:
    """Compute past/future keypoint trail for the selected keypoint index.

    Ghost positions (no detection or outlier) are linearly interpolated between
    bracketing real observations.  Points with no anchors on both sides are omitted.
    """
    past_indices  = list(range(max(0, current_slot_idx - n), current_slot_idx))
    future_indices = list(range(current_slot_idx + 1, min(len(frames), current_slot_idx + n + 1)))
    return _TrailData(
        kp_idx=kp_idx,
        past=_build_trail_segment(frames, kp_by_frame, camera_id, past_indices, kp_idx),
        future=_build_trail_segment(frames, kp_by_frame, camera_id, future_indices, kp_idx),
    )


# ---------------------------------------------------------------------------
# Single camera cell widget
# ---------------------------------------------------------------------------

class _CropCellWidget(QWidget):
    """One camera's crop image with a keypoint overlay."""

    # Emitted when user clicks a keypoint dot (kp_idx)
    keypoint_selected  = Signal(int)
    # Emitted when user clicks empty space
    keypoint_deselected = Signal()
    # Emitted on drag release: (kp_idx, new_x_full, new_y_full) in full-frame pixels
    keypoint_moved     = Signal(int, float, float)

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = label
        self._pixmap: QPixmap | None = None
        self._kp: np.ndarray | None = None          # float32[N,3] merged
        self._edited_mask: np.ndarray | None = None  # bool[N] — edited slots
        self._trail: _TrailData | None = None
        # Crop rect in original full-resolution frame (for coordinate transform)
        self._src_x = 0
        self._src_y = 0
        self._src_w = 1
        self._src_h = 1
        # Mouse interaction state
        self._selected_kp: int | None = None
        self._drag_kp_idx: int | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_current: tuple[float, float] | None = None
        self.setFixedSize(_CELL_W, _CELL_H + 20)  # +20 for label strip
        self.setStyleSheet("background: #1a1a1a;")

    def update_frame(
        self,
        jpeg_bytes: bytes | None,
        src_x: int,
        src_y: int,
        src_w: int,
        src_h: int,
        kp: np.ndarray | None,
        edited_mask: np.ndarray | None = None,
        trail: _TrailData | None = None,
    ) -> None:
        self._src_x = src_x
        self._src_y = src_y
        self._src_w = max(src_w, 1)
        self._src_h = max(src_h, 1)
        self._kp = kp
        self._edited_mask = edited_mask
        self._trail = trail

        if jpeg_bytes:
            pix = QPixmap()
            if pix.loadFromData(QByteArray(jpeg_bytes)) and not pix.isNull():
                self._pixmap = pix.scaled(
                    _CELL_W, _CELL_H,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            else:
                self._pixmap = None
        else:
            self._pixmap = None
        self.update()

    def clear(self) -> None:
        self._pixmap = None
        self._kp = None
        self._edited_mask = None
        self._trail = None
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(0, 0, _CELL_W, _CELL_H, QColor(30, 30, 30))

        # JPEG image
        if self._pixmap is not None:
            painter.drawPixmap(0, 0, self._pixmap)
        else:
            # No crop available — show placeholder
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(0, 0, _CELL_W, _CELL_H, Qt.AlignCenter, "no crop")

        # Trail (drawn under keypoint dots so dots appear on top)
        if self._trail is not None:
            self._draw_trail(painter, self._trail)

        # Keypoint overlay
        if self._kp is not None:
            self._draw_keypoints(painter)

        # Camera label strip at bottom
        painter.fillRect(0, _CELL_H, _CELL_W, 20, QColor(20, 20, 20))
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("monospace", 8))
        painter.drawText(4, _CELL_H, _CELL_W - 8, 20, Qt.AlignLeft | Qt.AlignVCenter,
                         self._label)

    def set_selected_kp(self, kp_idx: int | None) -> None:
        """Set which keypoint index is highlighted as selected."""
        self._selected_kp = kp_idx
        self.update()

    def _hit_kp(self, dx: float, dy: float) -> int | None:
        """Return keypoint index if (dx, dy) hits a dot, else None."""
        if self._kp is None:
            return None
        scale_x = _CELL_W / self._src_w
        scale_y = _CELL_H / self._src_h
        for i in range(self._kp.shape[0]):
            x_f, y_f = float(self._kp[i, 0]), float(self._kp[i, 1])
            if not (self._src_x <= x_f < self._src_x + self._src_w and
                    self._src_y <= y_f < self._src_y + self._src_h):
                continue
            cdx = (x_f - self._src_x) * scale_x
            cdy = (y_f - self._src_y) * scale_y
            if (dx - cdx) ** 2 + (dy - cdy) ** 2 <= _HIT_RADIUS ** 2:
                return i
        return None

    def _display_to_full(self, dx: float, dy: float) -> tuple[float, float]:
        """Convert display-crop coords to full-frame pixel coords."""
        return (
            self._src_x + dx * self._src_w / _CELL_W,
            self._src_y + dy * self._src_h / _CELL_H,
        )

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            hit = self._hit_kp(pos.x(), pos.y())
            if hit is not None:
                self._drag_kp_idx = hit
                self._drag_start   = (pos.x(), pos.y())
                self._drag_current = (pos.x(), pos.y())
                self.keypoint_selected.emit(hit)
            else:
                self._drag_kp_idx  = None
                self._drag_start   = None
                self._drag_current = None
                self.keypoint_deselected.emit()
            self.update()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_kp_idx is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = event.position()
            self._drag_current = (pos.x(), pos.y())
            self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._drag_kp_idx is not None:
            if self._drag_start is not None and self._drag_current is not None:
                ddx = self._drag_current[0] - self._drag_start[0]
                ddy = self._drag_current[1] - self._drag_start[1]
                if ddx ** 2 + ddy ** 2 > _DRAG_THRESHOLD ** 2:
                    kp = self._kp
                    if kp is not None and self._drag_kp_idx < kp.shape[0]:
                        orig_x = float(kp[self._drag_kp_idx, 0])
                        orig_y = float(kp[self._drag_kp_idx, 1])
                        new_x = orig_x + ddx * self._src_w / _CELL_W
                        new_y = orig_y + ddy * self._src_h / _CELL_H
                        self.keypoint_moved.emit(self._drag_kp_idx, new_x, new_y)
            self._drag_kp_idx  = None
            self._drag_start   = None
            self._drag_current = None
            self.update()
        else:
            super().mouseReleaseEvent(event)

    def _draw_keypoints(self, painter: QPainter) -> None:
        kp = self._kp
        assert kp is not None
        scale_x = _CELL_W / self._src_w
        scale_y = _CELL_H / self._src_h

        # Live drag offset in display coords
        drag_ddx = drag_ddy = 0.0
        if self._drag_kp_idx is not None and self._drag_start and self._drag_current:
            drag_ddx = self._drag_current[0] - self._drag_start[0]
            drag_ddy = self._drag_current[1] - self._drag_start[1]

        for i in range(kp.shape[0]):
            x_full, y_full, conf = float(kp[i, 0]), float(kp[i, 1]), float(kp[i, 2])

            if not (self._src_x <= x_full < self._src_x + self._src_w and
                    self._src_y <= y_full < self._src_y + self._src_h):
                continue

            dx = (x_full - self._src_x) * scale_x
            dy = (y_full - self._src_y) * scale_y

            # Live drag preview for the dragged dot
            if i == self._drag_kp_idx and self._drag_start is not None:
                dx += drag_ddx
                dy += drag_ddy

            edited = (self._edited_mask is not None and bool(self._edited_mask[i]))

            if conf < _CONF_THRESHOLD:
                color = _KP_OUTLIER
            elif edited:
                color = _KP_EDITED
            elif conf > 0.5:
                color = _KP_HI_CONF
            else:
                color = _KP_LO_CONF

            selected = (i == self._selected_kp)
            radius = _KP_RADIUS + 2 if selected else _KP_RADIUS

            if selected:
                painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            else:
                painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(int(dx) - radius, int(dy) - radius, radius * 2, radius * 2)

    def _draw_trail(self, painter: QPainter, trail: _TrailData) -> None:
        """Draw past/future trail polylines and dots for the selected keypoint."""
        scale_x = _CELL_W / self._src_w
        scale_y = _CELL_H / self._src_h

        def to_display(tp: _TrailPoint) -> tuple[int, int] | None:
            if not (self._src_x <= tp.x < self._src_x + self._src_w and
                    self._src_y <= tp.y < self._src_y + self._src_h):
                return None
            return (
                int((tp.x - self._src_x) * scale_x),
                int((tp.y - self._src_y) * scale_y),
            )

        def draw_segment(points: list[_TrailPoint], color: QColor) -> None:
            if not points:
                return
            display = [to_display(p) for p in points]

            # Polyline through all visible display points
            poly = [QPointF(d[0], d[1]) for d in display if d is not None]
            if len(poly) >= 2:
                pen = QPen(color)
                pen.setWidthF(_TRAIL_LINE_WIDTH)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPolyline(poly)

            # Individual dots (ghost in semi-transparent grey, real in segment color)
            painter.setPen(Qt.PenStyle.NoPen)
            for d, tp in zip(display, points):
                if d is None:
                    continue
                painter.setBrush(_TRAIL_GHOST_COLOR if tp.is_ghost else color)
                painter.drawEllipse(
                    d[0] - _TRAIL_DOT_R, d[1] - _TRAIL_DOT_R,
                    _TRAIL_DOT_R * 2, _TRAIL_DOT_R * 2,
                )

        draw_segment(trail.past, _TRAIL_PAST_COLOR)
        draw_segment(trail.future, _TRAIL_FUTURE_COLOR)


# ---------------------------------------------------------------------------
# Main grid widget
# ---------------------------------------------------------------------------

class PersonCropGridWidget(QWidget):
    """Multi-camera keypoint editing surface for one person sequence.

    Loads JPEG crops from frame_cache_entries and draws merged keypoints
    from read_observations_with_edits().  Frame navigation is instant because
    it reads cached blobs from the DB rather than seeking raw video files.
    """

    frame_changed = Signal(int, float)  # video_frame_index, timestamp_s

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session: sqlite3.Connection | None = None
        self._sequence_id: str | None = None
        self._detection_run_id: str | None = None
        self._cameras: list[_CameraSlot] = []
        self._frames: list[_FrameSlot] = []
        self._frame_idx: int = 0        # index into self._frames
        self._selected_kp: int | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Navigation toolbar
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("◀  Prev (a)")
        self._prev_btn.clicked.connect(self._on_prev)
        nav.addWidget(self._prev_btn)

        self._frame_label = QLabel("—")
        self._frame_label.setAlignment(Qt.AlignCenter)
        self._frame_label.setMinimumWidth(200)
        nav.addWidget(self._frame_label, 1)

        self._next_btn = QPushButton("Next (d)  ▶")
        self._next_btn.clicked.connect(self._on_next)
        nav.addWidget(self._next_btn)
        root.addLayout(nav)

        # Horizontal scroll area for camera cells
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self._cell_container = QWidget()
        self._cells_layout = QHBoxLayout(self._cell_container)
        self._cells_layout.setContentsMargins(2, 2, 2, 2)
        self._cells_layout.setSpacing(4)
        self._cell_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        scroll.setWidget(self._cell_container)
        root.addWidget(scroll, 1)

        self._cells: list[_CropCellWidget] = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_sequence(self, session: sqlite3.Connection, sequence_id: str) -> None:
        """Load all required data for *sequence_id* and display the first frame."""
        self._session = session
        self._sequence_id = sequence_id
        self._cameras.clear()
        self._frames.clear()
        self._frame_idx = 0

        seq = session.execute(
            "SELECT detection_run_id, shot_id FROM pose_observation_sequences WHERE id = ?",
            (sequence_id,),
        ).fetchone()
        if seq is None:
            return
        self._detection_run_id = seq["detection_run_id"]
        shot_id = seq["shot_id"]

        person_row = session.execute(
            "SELECT person_name FROM sequence_persons WHERE sequence_id = ? AND person_id = 0",
            (sequence_id,),
        ).fetchone()
        person_name = person_row["person_name"] if person_row else ""

        # Build camera slots
        sv_rows = session.execute(
            "SELECT sv.id AS svid, sv.camera_instance_id, ci.label"
            " FROM capture_videos sv"
            " JOIN camera_instances ci ON ci.id = sv.camera_instance_id"
            " WHERE sv.shot_id = ?"
            " ORDER BY ci.label",
            (shot_id,),
        ).fetchall()

        for sv in sv_rows:
            slot = _CameraSlot(
                camera_instance_id=sv["camera_instance_id"],
                shot_video_id=sv["svid"],
                label=sv["label"] or sv["camera_instance_id"][:8],
            )
            # Load track ranges for this camera + person
            if self._detection_run_id and person_name:
                tr_rows = session.execute(
                    "SELECT first_frame, last_frame, track_id"
                    " FROM detection_track_assignments"
                    " WHERE detection_run_id = ? AND shot_video_id = ? AND person_name = ?"
                    " ORDER BY first_frame",
                    (self._detection_run_id, sv["svid"], person_name),
                ).fetchall()
                slot.track_ranges = [(r["first_frame"], r["last_frame"], r["track_id"])
                                     for r in tr_rows]
            # Load merged keypoints for all frames (one DB read per camera)
            slot.kp_by_frame = read_observations_with_edits(
                session, sequence_id, sv["camera_instance_id"]
            )
            self._cameras.append(slot)

        # Build frame list from all observations across all cameras, sorted by timestamp_s
        frame_obs = session.execute(
            "SELECT camera_instance_id, video_frame, timestamp_s"
            " FROM pose_observations"
            " WHERE sequence_id = ? AND person_id = 0"
            " ORDER BY timestamp_s, camera_instance_id",
            (sequence_id,),
        ).fetchall()

        # Group by timestamp_s
        by_ts: dict[float, dict[str, int]] = {}
        for row in frame_obs:
            ts = float(row["timestamp_s"])
            by_ts.setdefault(ts, {})[row["camera_instance_id"]] = int(row["video_frame"])

        self._frames = [
            _FrameSlot(timestamp_s=ts, per_cam=per_cam)
            for ts, per_cam in sorted(by_ts.items())
        ]

        # Rebuild cell widgets
        self._rebuild_cells()
        self._show_frame(0)

    def _rebuild_cells(self) -> None:
        """Recreate CropCellWidget instances to match current camera list."""
        while self._cells_layout.count():
            item = self._cells_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cells.clear()

        for cam in self._cameras:
            cell = _CropCellWidget(cam.label)
            self._cells_layout.addWidget(cell)
            self._cells.append(cell)
            cell.keypoint_selected.connect(
                lambda idx, c=cam: self._on_cell_kp_selected(c, idx)
            )
            cell.keypoint_deselected.connect(self._on_cell_kp_deselected)
            cell.keypoint_moved.connect(
                lambda kp_idx, x, y, c=cam: self._on_cell_kp_moved(c, kp_idx, x, y)
            )

        self._cells_layout.addStretch()

    # ------------------------------------------------------------------
    # Frame navigation
    # ------------------------------------------------------------------

    def _on_prev(self) -> None:
        if self._frame_idx > 0:
            self._show_frame(self._frame_idx - 1)

    def _on_next(self) -> None:
        if self._frame_idx < len(self._frames) - 1:
            self._show_frame(self._frame_idx + 1)

    def _show_frame(self, idx: int) -> None:
        if not self._frames or self._session is None:
            self._frame_label.setText("— no data —")
            for cell in self._cells:
                cell.clear()
            return

        idx = max(0, min(idx, len(self._frames) - 1))
        self._frame_idx = idx
        slot = self._frames[idx]

        ts = slot.timestamp_s
        mm = int(ts // 60)
        ss = ts % 60
        self._frame_label.setText(
            f"Frame {idx + 1} / {len(self._frames)}  |  {mm:02d}:{ss:06.3f} s"
        )
        self._prev_btn.setEnabled(idx > 0)
        self._next_btn.setEnabled(idx < len(self._frames) - 1)

        for cam, cell in zip(self._cameras, self._cells):
            video_frame = slot.per_cam.get(cam.camera_instance_id)
            if video_frame is None:
                cell.clear()
                continue

            # Look up track_id for this frame
            track_id = self._track_id_for_frame(cam, video_frame)

            jpeg, src_x, src_y, src_w, src_h = self._load_crop(
                cam.shot_video_id, video_frame, track_id
            )

            kp = cam.kp_by_frame.get(video_frame)
            edited_mask = self._edited_mask(cam.camera_instance_id, video_frame) if kp is not None else None

            trail = None
            if self._selected_kp is not None:
                trail = compute_trail(
                    self._frames, cam.kp_by_frame, cam.camera_instance_id,
                    idx, self._selected_kp,
                )

            cell.update_frame(jpeg, src_x, src_y, src_w, src_h, kp, edited_mask, trail)

        self.frame_changed.emit(idx, slot.timestamp_s)

    # ------------------------------------------------------------------
    # Keypoint selection
    # ------------------------------------------------------------------

    def select_keypoint(self, kp_idx: int | None) -> None:
        """Set the selected keypoint index and refresh the trail display."""
        self._selected_kp = kp_idx
        for cell in self._cells:
            cell.set_selected_kp(kp_idx)
        self._show_frame(self._frame_idx)

    def _on_cell_kp_selected(self, cam: _CameraSlot, kp_idx: int) -> None:
        self._selected_kp = kp_idx
        for cell in self._cells:
            cell.set_selected_kp(kp_idx)
        self._show_frame(self._frame_idx)

    def _on_cell_kp_deselected(self) -> None:
        self._selected_kp = None
        for cell in self._cells:
            cell.set_selected_kp(None)
        self._show_frame(self._frame_idx)

    def _on_cell_kp_moved(
        self, cam: _CameraSlot, kp_idx: int, new_x: float, new_y: float
    ) -> None:
        """Write a single-keypoint edit to DB and refresh the display."""
        if self._session is None or self._sequence_id is None or not self._frames:
            return
        slot = self._frames[self._frame_idx]
        video_frame = slot.per_cam.get(cam.camera_instance_id)
        if video_frame is None:
            return
        update_single_keypoint_edit(
            self._session, self._sequence_id, cam.camera_instance_id,
            video_frame, kp_idx, new_x, new_y, is_outlier=False,
        )
        cam.kp_by_frame = read_observations_with_edits(
            self._session, self._sequence_id, cam.camera_instance_id
        )
        self._show_frame(self._frame_idx)

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _track_id_for_frame(self, cam: _CameraSlot, video_frame: int) -> int | None:
        """Return the track_id covering *video_frame* for *cam*, or None."""
        for first, last, tid in cam.track_ranges:
            if first <= video_frame <= last:
                return tid
        return None

    def _load_crop(
        self, shot_video_id: str, video_frame: int, track_id: int | None
    ) -> tuple[bytes | None, int, int, int, int]:
        """Load JPEG crop blob and src rect from frame_cache_entries.

        Returns (jpeg_bytes, src_x, src_y, src_w, src_h).
        jpeg_bytes is None when no crop is cached.
        """
        if self._session is None or track_id is None:
            return None, 0, 0, 1, 1

        row = self._session.execute(
            "SELECT image_data, src_x, src_y, src_w, src_h"
            " FROM frame_cache_entries"
            " WHERE shot_video_id = ?"
            "   AND frame_idx = ?"
            "   AND detection_run_id = ?"
            "   AND track_id = ?"
            "   AND cache_type = 'person_crop'"
            "   AND region_type = 'full_body'"
            " LIMIT 1",
            (shot_video_id, video_frame, self._detection_run_id, track_id),
        ).fetchone()

        if row is None:
            return None, 0, 0, 1, 1

        jpeg = bytes(row["image_data"])
        src_x = int(row["src_x"] or 0)
        src_y = int(row["src_y"] or 0)
        src_w = int(row["src_w"] or 1)
        src_h = int(row["src_h"] or 1)
        return jpeg, src_x, src_y, src_w, src_h

    def _edited_mask(
        self, camera_instance_id: str, video_frame: int
    ) -> np.ndarray | None:
        """Return a bool[N] mask of edited keypoint slots, or None if no edit exists."""
        if self._session is None or self._sequence_id is None:
            return None

        row = self._session.execute(
            "SELECT kp_mask, kp_blob FROM pose_observation_edits"
            " WHERE sequence_id = ? AND camera_instance_id = ? AND video_frame = ?",
            (self._sequence_id, camera_instance_id, video_frame),
        ).fetchone()
        if row is None:
            return None

        mask_bytes = bytes(row["kp_mask"])
        kp_blob = bytes(row["kp_blob"])
        n_kp = len(kp_blob) // (3 * 4)
        n_bytes = math.ceil(n_kp / 8)
        if len(mask_bytes) < n_bytes:
            return None

        result = np.zeros(n_kp, dtype=bool)
        for i in range(n_kp):
            if (mask_bytes[i // 8] >> (i % 8)) & 1:
                result[i] = True
        return result

    # ------------------------------------------------------------------
    # Keyboard navigation (Phase 5 will add more)
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_A:
            self._on_prev()
        elif key == Qt.Key.Key_D:
            self._on_next()
        else:
            super().keyPressEvent(event)

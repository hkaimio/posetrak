"""page_extrinsics.py — Extrinsic calibration import and auto-calibration.

Public classes
--------------
ExtrinsicsImportWidget
    Reusable core widget.  Call ``set_session(conn, session_id, shot_ids)``
    before showing.  Emits ``imported(str)`` with the new calibration ID on
    success.  Includes an "Auto-calibrate…" button that opens
    ExtrinsicsAutoCalibDialog.

ExtrinsicsAutoCalibDialog
    Semi-automatic extrinsics dialog.  Takes a list of CamCalibState objects
    (with images loaded), runs SIFT matching + bundle adjustment in a background
    thread, lets the user add manual control points, then writes the result
    directly to the session DB.

ExtrinsicsPage
    QWizardPage hosting ExtrinsicsImportWidget.  Reads conn / session_id from
    ``wizard.session_conn`` / ``wizard.session_id`` on initializePage().
    Always completable — extrinsics are optional at wizard time.

ExtrinsicsImportDialog
    Standalone QDialog wrapping ExtrinsicsImportWidget, for use from the pose
    extraction window (or any other context where the wizard is not running).
"""
from __future__ import annotations

import datetime
import re
import sqlite3
import struct
import tomllib
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRect, QThread, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from app.setup.extrinsics_solver import (
    CalibResult,
    CamCalibState,
    ControlPoint,
    run_calibration,
)
from posetrak.db.db import generate_id as _generate_id
from posetrak.db.import_extrinsics import import_extrinsics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CP_COLORS = [
    QColor(220, 80, 80),
    QColor(80, 200, 80),
    QColor(80, 120, 220),
    QColor(220, 180, 0),
    QColor(200, 80, 200),
    QColor(0, 200, 200),
]

# ---------------------------------------------------------------------------
# Label-matching helpers (mirrors calibrate_from_exports.py — kept in sync)
# ---------------------------------------------------------------------------

_FNAME_RE_UI = re.compile(r"^.+?_\d{2}_\d{2}_\d{3}_(.+?)_\d+\.png$", re.IGNORECASE)


def _ui_label_from_filename(fname: str) -> str | None:
    m = _FNAME_RE_UI.match(fname)
    return m.group(1).replace("_", " ") if m else None


def _ui_normalise(label: str) -> str:
    return re.sub(r"[-_.\s]+", " ", label).strip().lower()


def _ui_match_label(file_label: str, db_labels: list[str]) -> str | None:
    fl = _ui_normalise(file_label)
    matches = [db for db in db_labels if _ui_normalise(db) == fl]
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _load_states_from_images(
    images_dir: Path,
    conn: sqlite3.Connection,
    shot_id: str,
) -> list[CamCalibState]:
    """Load CamCalibState list by matching exported PNGs to cameras in the DB.

    Uses the same label-matching logic as calibrate_from_exports.py.
    Returns only cameras that have intrinsics AND a matching PNG.
    """
    old_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT ci.label AS cam_label,
                   ci.id AS cam_instance_id,
                   COALESCE(cv.intrinsics_calibration_id, ic.id) AS intrinsics_calibration_id
            FROM capture_videos cv
            JOIN camera_instances ci ON ci.id = cv.camera_instance_id
            LEFT JOIN camera_modes cm ON cm.id = cv.camera_mode_id
            LEFT JOIN intrinsics_calibrations ic ON ic.camera_mode_id = cm.id
            WHERE cv.shot_id = ?
            """,
            (shot_id,),
        ).fetchall()

        intrinsics: dict[str, dict] = {}
        all_labels: list[str] = []
        for r in rows:
            label = r["cam_label"]
            all_labels.append(label)
            if r["intrinsics_calibration_id"] is None:
                continue
            ic = conn.execute(
                "SELECT * FROM intrinsics_calibrations WHERE id = ?",
                (r["intrinsics_calibration_id"],),
            ).fetchone()
            if ic is None:
                continue
            fx, fy, cx, cy = ic["fx"], ic["fy"], ic["cx"], ic["cy"]
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
            if ic["matrix_original"]:
                vals = struct.unpack("<9d", bytes(ic["matrix_original"]))
                K_orig = np.array(vals).reshape(3, 3)
            else:
                K_orig = K.copy()
            if ic["dist_coeffs"]:
                n = len(bytes(ic["dist_coeffs"])) // 8
                dist = np.array(struct.unpack(f"<{n}d", bytes(ic["dist_coeffs"]))).reshape(1, -1)
            else:
                dist = np.zeros((1, 4))
            fisheye = ic["distortion_model"] == "fisheye"
            intrinsics[label] = {"K": K, "K_orig": K_orig, "dist": dist, "fisheye": fisheye}
    finally:
        conn.row_factory = old_factory

    states: list[CamCalibState] = []
    for png in sorted(images_dir.glob("*.png")):
        file_label = _ui_label_from_filename(png.name)
        if file_label is None:
            continue
        db_label = _ui_match_label(file_label, all_labels)
        if db_label is None or db_label not in intrinsics:
            continue
        img = cv2.imread(str(png))
        if img is None:
            continue
        intr = intrinsics[db_label]
        states.append(CamCalibState(
            video_id=db_label,
            label=db_label,
            K=intr["K"],
            K_orig=intr["K_orig"],
            dist=intr["dist"],
            fisheye=intr["fisheye"],
            image=img,
        ))
    return states


def _write_extrinsics_to_db(
    result: CalibResult,
    conn: sqlite3.Connection,
    session_id: str,
    label_to_instance_id: dict[str, str],
    method: str = "auto-sift",
) -> str:
    """Write CalibResult directly to the DB.  Returns new extrinsic_calibration_id."""
    calib_id = _generate_id()
    calibrated_at = datetime.date.today().isoformat()
    rows: list[tuple[str, str, bytes, bytes]] = []
    for vid, s in result.cameras.items():
        if s.R is None:
            continue
        instance_id = label_to_instance_id.get(s.label) or label_to_instance_id.get(vid)
        if instance_id is None:
            continue
        R_blob = struct.pack("<9d", *s.R.flatten())
        t_blob = struct.pack("<3d", *s.t.flatten())
        rows.append((calib_id, instance_id, R_blob, t_blob))

    with conn:
        conn.execute(
            "INSERT INTO extrinsic_calibrations (id, session_id, calibrated_at, method)"
            " VALUES (?, ?, ?, ?)",
            (calib_id, session_id, calibrated_at, method),
        )
        conn.executemany(
            "INSERT INTO extrinsic_entries"
            " (extrinsic_calibration_id, camera_instance_id, R, t)"
            " VALUES (?, ?, ?, ?)",
            rows,
        )
    return calib_id


# ---------------------------------------------------------------------------
# Clickable camera thumbnail with zoom-on-drag
# ---------------------------------------------------------------------------


class _ClickableImageWidget(QWidget):
    """Camera image widget with zoom-on-drag for precise control-point placement.

    Press-drag-release workflow
    ---------------------------
    Mouse press  → enter zoom mode (image shown at 1:1 pixel ratio, clicked
                   image coordinate stays under cursor)
    Mouse drag   → move placement crosshair; image pans if point goes near edge
    Mouse release→ finalise point, emit ``point_set``, return to fit view

    Any press on the image records/updates the point for the currently selected
    control point (managed by the parent dialog).  Right-click has no effect.

    After calibration, call ``set_calib_status`` to show a per-camera badge.
    """

    point_set = Signal(float, float)   # original image coords on mouse release

    def __init__(self, cam_label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cam_label = cam_label
        self._img_bgr: np.ndarray | None = None
        self._markers: list[tuple[float, float, QColor, str]] = []

        # Fit-mode display rect (set during paintEvent, used for coord mapping)
        self._fit_rect = QRect()
        self._fit_scale: float = 1.0

        # Full-res image cache (kept alive so QImage data pointer stays valid)
        self._full_rgb: np.ndarray | None = None
        self._full_qimage: QImage | None = None

        # Fit-scale thumbnail cache
        self._thumb_size: tuple[int, int] | None = None
        self._thumb_data: np.ndarray | None = None
        self._thumb_qimage: QImage | None = None

        # Zoom-on-drag state
        self._zoom_active: bool = False
        self._zoom_scale: float = 1.0
        self._zoom_ox: float = 0.0   # image-space origin of display_rect top-left
        self._zoom_oy: float = 0.0
        self._drag_img: tuple[float, float] | None = None  # current drag position

        # Calibration status badge
        self._status_text: str | None = None
        self._status_error: bool = False

        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)  # needed for smooth drag

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image(self, img_bgr: np.ndarray) -> None:
        self._img_bgr = img_bgr
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self._full_rgb = np.ascontiguousarray(rgb)
        h, w = img_bgr.shape[:2]
        self._full_qimage = QImage(
            self._full_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888
        )
        self._thumb_size = None  # invalidate thumbnail cache
        self.update()

    def add_marker(self, x: float, y: float, color: QColor, label: str = "") -> None:
        self._markers.append((x, y, color, label))
        self.update()

    def clear_markers(self) -> None:
        self._markers.clear()
        self.update()

    def set_calib_status(self, text: str | None, error: bool = False) -> None:
        self._status_text = text
        self._status_error = error
        self.update()

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _widget_to_img(self, wx: float, wy: float) -> tuple[float, float]:
        if self._zoom_active:
            return (
                self._zoom_ox + wx / self._zoom_scale,
                self._zoom_oy + wy / self._zoom_scale,
            )
        r = self._fit_rect
        if r.width() <= 0 or self._img_bgr is None:
            return 0.0, 0.0
        h, w = self._img_bgr.shape[:2]
        return (wx - r.x()) * w / r.width(), (wy - r.y()) * h / r.height()

    def _img_to_widget(self, ix: float, iy: float) -> tuple[float, float]:
        if self._zoom_active:
            return (
                (ix - self._zoom_ox) * self._zoom_scale,
                (iy - self._zoom_oy) * self._zoom_scale,
            )
        r = self._fit_rect
        if r.width() <= 0 or self._img_bgr is None:
            return 0.0, 0.0
        h, w = self._img_bgr.shape[:2]
        return r.x() + ix / w * r.width(), r.y() + iy / h * r.height()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        ww, wh = self.width(), self.height()
        label_h = 22
        avail_h = wh - label_h

        if self._img_bgr is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(
                QRect(0, 0, ww, avail_h),
                Qt.AlignmentFlag.AlignCenter,
                self._cam_label + "\n(no image)",
            )
        elif self._zoom_active:
            self._paint_zoomed(painter, ww, avail_h)
        else:
            self._paint_fit(painter, ww, avail_h)

        # Camera label bar
        painter.fillRect(0, wh - label_h, ww, label_h, QColor(0, 0, 0, 200))
        painter.setPen(QColor(240, 240, 240))
        painter.drawText(
            QRect(0, wh - label_h, ww, label_h),
            Qt.AlignmentFlag.AlignCenter,
            self._cam_label,
        )

        # Calibration status badge (top-right)
        if self._status_text:
            badge_w, badge_h = 110, 20
            bx = ww - badge_w
            color = QColor(180, 40, 40) if self._status_error else QColor(40, 160, 60)
            painter.fillRect(bx, 0, badge_w, badge_h, color)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                QRect(bx, 0, badge_w, badge_h),
                Qt.AlignmentFlag.AlignCenter,
                self._status_text,
            )
            if self._status_error:
                painter.setPen(QPen(QColor(200, 40, 40), 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(2, 2, ww - 4, wh - 4)

    def _paint_fit(self, painter: QPainter, ww: int, avail_h: int) -> None:
        """Draw image scaled to fit, update _fit_rect and thumbnail cache."""
        h, w = self._img_bgr.shape[:2]
        scale = min(ww / w, avail_h / h)
        self._fit_scale = scale
        dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
        x0 = (ww - dw) // 2
        y0 = (avail_h - dh) // 2
        self._fit_rect = QRect(x0, y0, dw, dh)

        if (dw, dh) != self._thumb_size:
            self._thumb_data = np.ascontiguousarray(
                cv2.resize(self._full_rgb, (dw, dh), interpolation=cv2.INTER_AREA)
            )
            self._thumb_qimage = QImage(
                self._thumb_data.data, dw, dh, dw * 3, QImage.Format.Format_RGB888
            )
            self._thumb_size = (dw, dh)

        painter.drawImage(x0, y0, self._thumb_qimage)
        self._draw_markers(painter)

    def _paint_zoomed(self, painter: QPainter, ww: int, avail_h: int) -> None:
        """Draw image at zoom scale, showing the region around the drag point."""
        h_img, w_img = self._img_bgr.shape[:2]
        scale = self._zoom_scale

        # Visible region in image coordinates
        src_x0 = max(0.0, self._zoom_ox)
        src_y0 = max(0.0, self._zoom_oy)
        src_x1 = min(float(w_img), self._zoom_ox + ww / scale)
        src_y1 = min(float(h_img), self._zoom_oy + avail_h / scale)
        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return

        src_rect = QRect(
            int(src_x0), int(src_y0),
            int(src_x1 - src_x0), int(src_y1 - src_y0),
        )
        dst_x = int((src_x0 - self._zoom_ox) * scale)
        dst_y = int((src_y0 - self._zoom_oy) * scale)
        dst_rect = QRect(
            dst_x, dst_y,
            int((src_x1 - src_x0) * scale),
            int((src_y1 - src_y0) * scale),
        )
        painter.drawImage(dst_rect, self._full_qimage, src_rect)

        # Permanent markers
        self._draw_markers(painter)

        # Drag crosshair
        if self._drag_img is not None:
            ix, iy = self._drag_img
            wx, wy = self._img_to_widget(ix, iy)
            arm = 16
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.drawLine(int(wx) - arm, int(wy), int(wx) + arm, int(wy))
            painter.drawLine(int(wx), int(wy) - arm, int(wx), int(wy) + arm)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.drawEllipse(int(wx) - 9, int(wy) - 9, 18, 18)

        # Zoom indicator
        painter.fillRect(0, 0, 52, 18, QColor(0, 0, 0, 160))
        painter.setPen(QColor(255, 220, 0))
        pct = int(round(scale * 100))
        painter.drawText(2, 13, f"{pct}%")

    def _draw_markers(self, painter: QPainter) -> None:
        for mx, my, color, mlabel in self._markers:
            wx, wy = self._img_to_widget(mx, my)
            painter.setBrush(color)
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            r = 7
            painter.drawEllipse(int(wx) - r, int(wy) - r, r * 2, r * 2)
            if mlabel:
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(int(wx) + r + 3, int(wy) + 5, mlabel)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._img_bgr is None:
            return
        pos = event.position()
        wx, wy = pos.x(), pos.y()

        # Convert click from fit view to image coords
        ix, iy = self._widget_to_img(wx, wy)
        h_img, w_img = self._img_bgr.shape[:2]
        if not (0 <= ix < w_img and 0 <= iy < h_img):
            return

        # Enter zoom mode: compute scale and pan so clicked point stays fixed
        ww, wh = self.width(), self.height()
        avail_h = wh - 22
        fit_scale = min(ww / w_img, avail_h / h_img)
        # Zoom to 1:1 for large images, 2× for smaller images
        self._zoom_scale = max(1.0, fit_scale * 2.0)
        # Pan: set origin so img coord (ix, iy) maps to widget coord (wx, wy)
        self._zoom_ox = ix - wx / self._zoom_scale
        self._zoom_oy = iy - wy / self._zoom_scale
        self._clamp_zoom_origin()

        self._zoom_active = True
        self._drag_img = (ix, iy)
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._zoom_active or self._img_bgr is None:
            return
        pos = event.position()
        ix, iy = self._widget_to_img(pos.x(), pos.y())
        h_img, w_img = self._img_bgr.shape[:2]
        ix = max(0.0, min(float(w_img - 1), ix))
        iy = max(0.0, min(float(h_img - 1), iy))

        # Pan when point approaches widget edge (keep crosshair within 20% of edges)
        ww, wh = self.width(), self.height()
        avail_h = wh - 22
        margin = 0.2
        wx_cur = (ix - self._zoom_ox) * self._zoom_scale
        wy_cur = (iy - self._zoom_oy) * self._zoom_scale
        if wx_cur < ww * margin:
            self._zoom_ox = ix - ww * margin / self._zoom_scale
        elif wx_cur > ww * (1 - margin):
            self._zoom_ox = ix - ww * (1 - margin) / self._zoom_scale
        if wy_cur < avail_h * margin:
            self._zoom_oy = iy - avail_h * margin / self._zoom_scale
        elif wy_cur > avail_h * (1 - margin):
            self._zoom_oy = iy - avail_h * (1 - margin) / self._zoom_scale
        self._clamp_zoom_origin()

        self._drag_img = (ix, iy)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or not self._zoom_active:
            return
        final = self._drag_img
        self._zoom_active = False
        self._drag_img = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()
        if final is not None:
            self.point_set.emit(float(final[0]), float(final[1]))

    def _clamp_zoom_origin(self) -> None:
        """Keep zoom pan within image bounds."""
        if self._img_bgr is None:
            return
        h_img, w_img = self._img_bgr.shape[:2]
        ww, wh = self.width(), self.height()
        avail_h = wh - 22
        max_ox = w_img - ww / self._zoom_scale
        max_oy = h_img - avail_h / self._zoom_scale
        self._zoom_ox = max(0.0, min(max_ox, self._zoom_ox))
        self._zoom_oy = max(0.0, min(max_oy, self._zoom_oy))


# ---------------------------------------------------------------------------
# Background solve thread
# ---------------------------------------------------------------------------


class _SolveThread(QThread):
    finished = Signal(object)      # CalibResult
    error_occurred = Signal(str)

    def __init__(
        self,
        states: list[CamCalibState],
        control_points: list[ControlPoint],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._states = states
        self._control_points = control_points

    def run(self) -> None:
        try:
            result = run_calibration(self._states, self._control_points)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))


# ---------------------------------------------------------------------------
# Auto-calibration dialog
# ---------------------------------------------------------------------------


class ExtrinsicsAutoCalibDialog(QDialog):
    """Semi-automatic extrinsics calibration dialog.

    Shows camera thumbnails (with zoom-on-drag for precise control point
    placement), runs SIFT matching + BA in a background thread, shows per-
    camera reprojection error / disconnection badges, and writes the result
    to the session DB on Accept.
    """

    imported = Signal(str)  # extrinsic_calibration_id

    def __init__(
        self,
        states: list[CamCalibState],
        conn: sqlite3.Connection,
        session_id: str,
        shot_ids: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._states = states
        self._conn = conn
        self._session_id = session_id
        self._shot_ids = shot_ids or []
        self._result: CalibResult | None = None
        self._solve_thread: _SolveThread | None = None
        self._control_points: list[ControlPoint] = []
        self._selected_cp_idx: int | None = None

        self.setWindowTitle("Auto Extrinsics Calibration")
        self.setMinimumSize(960, 640)
        self.resize(1200, 760)

        self._cam_widgets: dict[str, _ClickableImageWidget] = {}
        for state in states:
            w = _ClickableImageWidget(state.label)
            if state.image is not None:
                w.set_image(state.image)
            vid = state.video_id
            w.point_set.connect(lambda x, y, v=vid: self._on_cam_click(v, x, y))
            self._cam_widgets[state.video_id] = w

        self._build_ui()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Camera grid — fills available space
        n = len(self._cam_widgets)
        ncols = 1 if n == 1 else (2 if n <= 4 else 3)

        cam_container = QWidget()
        from PySide6.QtWidgets import QGridLayout
        grid = QGridLayout(cam_container)
        grid.setSpacing(4)
        grid.setContentsMargins(4, 4, 4, 4)
        for col in range(ncols):
            grid.setColumnStretch(col, 1)
        for i, (vid, w) in enumerate(self._cam_widgets.items()):
            row, col = divmod(i, ncols)
            grid.addWidget(w, row, col)
            grid.setRowStretch(row, 1)

        cam_scroll = QScrollArea()
        cam_scroll.setWidget(cam_container)
        cam_scroll.setWidgetResizable(True)
        cam_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Right panel: control points
        cp_panel = self._build_cp_panel()

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(cam_scroll)
        splitter.addWidget(cp_panel)
        splitter.setSizes([900, 280])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        # Solve row
        self._solve_btn = QPushButton("Match && Solve")
        self._solve_btn.clicked.connect(self._on_solve)
        self._status_label = QLabel(
            "Click 'Match & Solve' to run SIFT matching and bundle adjustment.  "
            "Optionally add control points first (press a camera image to place one)."
        )
        self._status_label.setWordWrap(True)

        solve_row = QHBoxLayout()
        solve_row.addWidget(self._solve_btn)
        solve_row.addWidget(self._status_label, 1)

        # Dialog buttons
        btn_box = QDialogButtonBox()
        self._accept_btn = btn_box.addButton("Accept", QDialogButtonBox.ButtonRole.AcceptRole)
        btn_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._accept_btn.setEnabled(False)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addWidget(splitter, 1)
        root.addLayout(solve_row)
        root.addWidget(btn_box)

    def _build_cp_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(280)

        # Control points list
        cp_group = QGroupBox("Control Points")
        cp_layout = QVBoxLayout(cp_group)

        hint = QLabel(
            "Select a point, then press and drag on camera images to place it precisely. "
            "Set World position to fix scale / origin in BA."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 10px;")

        self._cp_list = QListWidget()
        self._cp_list.currentRowChanged.connect(self._on_cp_selected)

        add_del = QHBoxLayout()
        add_btn = QPushButton("Add")
        del_btn = QPushButton("Delete")
        add_btn.clicked.connect(self._add_control_point)
        del_btn.clicked.connect(self._delete_control_point)
        add_del.addWidget(add_btn)
        add_del.addWidget(del_btn)

        cp_layout.addWidget(hint)
        cp_layout.addWidget(self._cp_list, 1)
        cp_layout.addLayout(add_del)

        # World position (optional)
        xyz_group = QGroupBox("World position (optional)")
        xyz_layout = QVBoxLayout(xyz_group)

        self._xyz_enabled = QCheckBox("Fix 3-D position in BA")
        self._xyz_enabled.stateChanged.connect(self._on_xyz_toggle)

        self._xyz_x = QDoubleSpinBox()
        self._xyz_y = QDoubleSpinBox()
        self._xyz_z = QDoubleSpinBox()
        for sb in (self._xyz_x, self._xyz_y, self._xyz_z):
            sb.setRange(-1e6, 1e6)
            sb.setDecimals(4)
            sb.setSingleStep(0.1)
            sb.setEnabled(False)

        xyz_form = QHBoxLayout()
        for lbl, sb in [("X", self._xyz_x), ("Y", self._xyz_y), ("Z", self._xyz_z)]:
            xyz_form.addWidget(QLabel(lbl))
            xyz_form.addWidget(sb)

        self._xyz_apply_btn = QPushButton("Apply")
        self._xyz_apply_btn.setEnabled(False)
        self._xyz_apply_btn.clicked.connect(self._apply_xyz)

        xyz_layout.addWidget(self._xyz_enabled)
        xyz_layout.addLayout(xyz_form)
        xyz_layout.addWidget(self._xyz_apply_btn)

        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(cp_group, 1)
        v.addWidget(xyz_group)
        return panel

    # ------------------------------------------------------------------
    # Control-point slots
    # ------------------------------------------------------------------

    def _add_control_point(self) -> None:
        default = f"CP{len(self._control_points) + 1}"
        name, ok = QInputDialog.getText(self, "Add Control Point", "Name:", text=default)
        if not ok or not name.strip():
            return
        cp = ControlPoint(name=name.strip())
        self._control_points.append(cp)
        self._cp_list.addItem(name.strip())
        self._cp_list.setCurrentRow(len(self._control_points) - 1)

    def _delete_control_point(self) -> None:
        row = self._cp_list.currentRow()
        if row < 0:
            return
        self._control_points.pop(row)
        self._cp_list.takeItem(row)
        self._selected_cp_idx = None
        self._refresh_markers()

    def _on_cp_selected(self, row: int) -> None:
        self._selected_cp_idx = row if row >= 0 else None
        has_cp = self._selected_cp_idx is not None
        self._xyz_enabled.setEnabled(has_cp)
        if has_cp:
            cp = self._control_points[row]
            fixed = cp.world_xyz is not None
            self._xyz_enabled.setChecked(fixed)
            for sb in (self._xyz_x, self._xyz_y, self._xyz_z):
                sb.setEnabled(fixed)
            self._xyz_apply_btn.setEnabled(fixed)
            if fixed:
                self._xyz_x.setValue(float(cp.world_xyz[0]))
                self._xyz_y.setValue(float(cp.world_xyz[1]))
                self._xyz_z.setValue(float(cp.world_xyz[2]))
        else:
            self._xyz_enabled.setChecked(False)
            for sb in (self._xyz_x, self._xyz_y, self._xyz_z):
                sb.setEnabled(False)
            self._xyz_apply_btn.setEnabled(False)
        self._refresh_markers()

    def _on_xyz_toggle(self, state: int) -> None:
        enabled = state == Qt.CheckState.Checked.value
        for sb in (self._xyz_x, self._xyz_y, self._xyz_z):
            sb.setEnabled(enabled)
        self._xyz_apply_btn.setEnabled(enabled)
        if not enabled and self._selected_cp_idx is not None:
            self._control_points[self._selected_cp_idx].world_xyz = None

    def _apply_xyz(self) -> None:
        if self._selected_cp_idx is None:
            return
        cp = self._control_points[self._selected_cp_idx]
        cp.world_xyz = np.array([
            self._xyz_x.value(),
            self._xyz_y.value(),
            self._xyz_z.value(),
        ])

    # ------------------------------------------------------------------
    # Camera click → record observation
    # ------------------------------------------------------------------

    def _on_cam_click(self, vid: str, x: float, y: float) -> None:
        if self._selected_cp_idx is None:
            return
        cp = self._control_points[self._selected_cp_idx]
        cp.obs[vid] = (x, y)
        self._refresh_markers()

    def _refresh_markers(self) -> None:
        for w in self._cam_widgets.values():
            w.clear_markers()
        for i, cp in enumerate(self._control_points):
            color = _CP_COLORS[i % len(_CP_COLORS)]
            is_selected = (i == self._selected_cp_idx)
            mlabel = cp.name if is_selected else ""
            for vid, (x, y) in cp.obs.items():
                if vid in self._cam_widgets:
                    self._cam_widgets[vid].add_marker(x, y, color, mlabel)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def _on_solve(self) -> None:
        if self._solve_thread and self._solve_thread.isRunning():
            return
        for s in self._states:
            s.R = None
            s.t = None
        for w in self._cam_widgets.values():
            w.set_calib_status(None)

        self._solve_btn.setEnabled(False)
        self._accept_btn.setEnabled(False)
        self._status_label.setText("Running SIFT matching and bundle adjustment…")

        self._solve_thread = _SolveThread(self._states, self._control_points, parent=self)
        self._solve_thread.finished.connect(self._on_solve_done)
        self._solve_thread.error_occurred.connect(self._on_solve_error)
        self._solve_thread.start()

    def _on_solve_done(self, result: CalibResult) -> None:
        self._result = result
        self._solve_btn.setEnabled(True)
        n_total = len(result.cameras)
        n_solved = n_total - len(result.unsolved)

        lines = [f"Solved: {n_solved}/{n_total} cameras"]
        for vid, stats in result.reprojection_errors.items():
            s = result.cameras[vid]
            lines.append(
                f"  {s.label}: {stats['mean']:.2f} ± {stats['std']:.2f} px"
                f"  (max {stats['max']:.1f}, n={stats['n']})"
            )
        if result.unsolved:
            lines.append(
                f"  Disconnected: {', '.join(result.unsolved)}"
                f" — add control points shared with a solved camera to connect them."
            )
        self._status_label.setText("\n".join(lines))
        self._accept_btn.setEnabled(n_solved > 0)

        # Update per-camera badges
        for state in self._states:
            vid = state.video_id
            if vid not in self._cam_widgets:
                continue
            w = self._cam_widgets[vid]
            if vid in result.unsolved:
                w.set_calib_status("Disconnected", error=True)
            elif vid in result.reprojection_errors:
                err = result.reprojection_errors[vid]["mean"]
                w.set_calib_status(f"err {err:.2f} px", error=err > 5.0)
            else:
                w.set_calib_status(None)

    def _on_solve_error(self, msg: str) -> None:
        self._solve_btn.setEnabled(True)
        self._status_label.setText(f"Error: {msg}")

    # ------------------------------------------------------------------
    # Accept → write to DB
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        if self._result is None:
            return
        old_factory = self._conn.row_factory
        self._conn.row_factory = sqlite3.Row
        try:
            rows = self._conn.execute(
                "SELECT id, label FROM camera_instances"
            ).fetchall()
        finally:
            self._conn.row_factory = old_factory
        label_to_id = {r["label"]: r["id"] for r in rows}

        try:
            calib_id = _write_extrinsics_to_db(
                self._result, self._conn, self._session_id, label_to_id
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Write failed", str(exc))
            return

        if self._shot_ids:
            with self._conn:
                self._conn.executemany(
                    "UPDATE captures SET extrinsic_calibration_id = ? WHERE id = ?",
                    [(calib_id, sid) for sid in self._shot_ids],
                )

        self.imported.emit(calib_id)
        self.accept()


# ---------------------------------------------------------------------------
# Core widget
# ---------------------------------------------------------------------------


class ExtrinsicsImportWidget(QWidget):
    """File picker + camera matching table + import button."""

    imported = Signal(str)  # extrinsic_calibration_id

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._conn: sqlite3.Connection | None = None
        self._session_id: str | None = None
        self._shot_ids: list[str] = []
        self._cam_keys: list[str] = []
        self._toml_names: dict[str, str] = {}
        self._toml_path: Path | None = None
        self._instances: list[sqlite3.Row] = []

        # ---- File row ----
        self._path_label = QLabel("No file selected.")
        self._path_label.setStyleSheet("color: grey;")
        browse_btn = QPushButton("Browse TOML…")
        browse_btn.clicked.connect(self._browse)
        self._auto_btn = QPushButton("Auto-calibrate…")
        self._auto_btn.clicked.connect(self._on_auto_calibrate)
        self._auto_btn.setToolTip(
            "Run SIFT-based automatic calibration from exported PNG frames."
        )

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("TOML file:"))
        file_row.addWidget(self._path_label, 1)
        file_row.addWidget(browse_btn)
        file_row.addWidget(self._auto_btn)

        # ---- Existing calibrations info ----
        self._existing_label = QLabel()
        self._existing_label.setStyleSheet("color: #555; font-size: 11px;")
        self._existing_label.setVisible(False)

        # ---- Matching table ----
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["TOML entry", "Name in file", "Session camera"])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(100)

        match_box = QGroupBox("Camera assignment")
        match_layout = QVBoxLayout(match_box)
        match_layout.addWidget(
            QLabel("Assign each TOML camera entry to a camera instance in the session.")
        )
        match_layout.addWidget(self._table)

        # ---- Status / error ----
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setVisible(False)

        # ---- Import button ----
        self._import_btn = QPushButton("Import")
        self._import_btn.setEnabled(False)
        self._import_btn.clicked.connect(self._do_import)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self._import_btn)

        # ---- Layout ----
        root = QVBoxLayout(self)
        root.addLayout(file_row)
        root.addWidget(self._existing_label)
        root.addWidget(match_box)
        root.addWidget(self._status_label)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_session(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        shot_ids: list[str] | None = None,
    ) -> None:
        """Supply session connection and ID.  Safe to call multiple times."""
        self._conn = conn
        self._session_id = session_id
        self._shot_ids = shot_ids or []
        self._instances = self._load_instances()
        self._refresh_existing_label()
        self._rebuild_table()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Pose2Sim calibration TOML", "", "TOML files (*.toml);;All files (*)"
        )
        if path:
            self._load_toml(Path(path))

    def _on_auto_calibrate(self) -> None:
        if self._conn is None or self._session_id is None:
            QMessageBox.warning(self, "No session", "Open a session before auto-calibrating.")
            return
        if not self._shot_ids:
            QMessageBox.warning(
                self, "No shot",
                "No capture/shot ID available — cannot look up camera intrinsics."
            )
            return

        images_dir = QFileDialog.getExistingDirectory(
            self, "Select exported frames directory"
        )
        if not images_dir:
            return

        states = _load_states_from_images(
            Path(images_dir), self._conn, self._shot_ids[0]
        )
        if not states:
            QMessageBox.warning(
                self, "No cameras",
                "No cameras with matching intrinsics found in the selected directory.\n\n"
                "Make sure you have:\n"
                " • exported frames using 'Export frames…' in the sync page\n"
                " • intrinsics calibrated for the cameras in this session",
            )
            return

        dlg = ExtrinsicsAutoCalibDialog(
            states,
            self._conn,
            self._session_id,
            self._shot_ids,
            parent=self,
        )
        dlg.imported.connect(self._on_auto_imported)
        dlg.exec()

    def _on_auto_imported(self, calib_id: str) -> None:
        self._refresh_existing_label()
        self.imported.emit(calib_id)

    def _do_import(self) -> None:
        self._set_status(None)
        if self._conn is None or self._session_id is None:
            self._set_status("No session open.", error=True)
            return

        assignment: dict[str, str] = {}
        for row_idx in range(self._table.rowCount()):
            cam_key = self._table.item(row_idx, 0).text()
            combo: QComboBox = self._table.cellWidget(row_idx, 2)
            instance_id = combo.currentData()
            if instance_id is not None:
                assignment[cam_key] = instance_id

        if not assignment:
            self._set_status("Assign at least one camera before importing.", error=True)
            return

        dupes = [iid for iid in assignment.values() if list(assignment.values()).count(iid) > 1]
        if dupes:
            self._set_status("Each session camera can only be assigned to one TOML entry.", error=True)
            return

        try:
            result = import_extrinsics(
                self._conn,
                self._session_id,
                self._toml_path,
                assignment,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Import failed: {exc}", error=True)
            return

        n = len(result.camera_instance_ids)
        msg = f"Imported {n} camera{'s' if n != 1 else ''}."
        if result.skipped:
            msg += f"  Skipped: {', '.join(sorted(result.skipped))}."

        self._refresh_existing_label()
        QMessageBox.information(self, "Import successful", msg)
        self.imported.emit(result.extrinsic_calibration_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_instances(self) -> list[sqlite3.Row]:
        if self._conn is None:
            return []
        return self._conn.execute(
            "SELECT ci.id, ci.label, ci.serial_number,"
            "       COALESCE(cm.model_name, '') AS model_name"
            " FROM camera_instances ci"
            " LEFT JOIN camera_models cm ON cm.id = ci.camera_model_id"
            " ORDER BY ci.label"
        ).fetchall()

    def _refresh_existing_label(self) -> None:
        if self._conn is None:
            self._existing_label.setVisible(False)
            return
        rows = self._conn.execute(
            "SELECT id, calibrated_at, method FROM extrinsic_calibrations"
            " ORDER BY calibrated_at DESC"
        ).fetchall()
        if not rows:
            self._existing_label.setVisible(False)
            return
        parts = [f"{r['calibrated_at']}  [{r['method'] or '?'}]  {r['id'][:8]}…" for r in rows]
        self._existing_label.setText("Existing calibrations: " + " | ".join(parts))
        self._existing_label.setVisible(True)

    def _instance_display(self, row: sqlite3.Row) -> str:
        parts: list[str] = []
        if row["model_name"]:
            parts.append(row["model_name"])
        if row["label"]:
            parts.append(row["label"])
        if row["serial_number"]:
            parts.append(f"S/N {row['serial_number']}")
        return "  —  ".join(parts) if parts else row["id"][:8]

    def _load_toml(self, path: Path) -> None:
        self._set_status(None)
        try:
            with path.open("rb") as fh:
                raw = tomllib.load(fh)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Cannot read TOML: {exc}", error=True)
            return

        cam_keys = sorted(
            (k for k in raw if k.startswith("cam") and k != "metadata"),
            key=lambda k: int(k[3:]) if k[3:].isdigit() else float("inf"),
        )
        if not cam_keys:
            self._set_status("No camera sections (cam1, cam2, …) found in TOML.", error=True)
            return

        self._toml_path = path
        self._cam_keys = cam_keys
        self._toml_names = {k: str(raw[k].get("name", "")) for k in cam_keys}

        self._path_label.setText(path.name)
        self._path_label.setToolTip(str(path))
        self._path_label.setStyleSheet("")

        self._rebuild_table()
        self._import_btn.setEnabled(True)

    def _rebuild_table(self) -> None:
        self._table.setRowCount(0)
        if not self._cam_keys:
            return

        used_ids: set[str] = set()
        for row_idx, cam_key in enumerate(self._cam_keys):
            self._table.insertRow(row_idx)

            key_item = QTableWidgetItem(cam_key)
            key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row_idx, 0, key_item)

            toml_name = self._toml_names.get(cam_key, "")
            name_item = QTableWidgetItem(toml_name if toml_name else "—")
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row_idx, 1, name_item)

            combo = QComboBox()
            combo.addItem("(unassigned)", None)
            for inst in self._instances:
                combo.addItem(self._instance_display(inst), inst["id"])

            best = self._auto_match(cam_key, toml_name, used_ids)
            if best is not None:
                idx = combo.findData(best)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                    used_ids.add(best)

            self._table.setCellWidget(row_idx, 2, combo)

    def _auto_match(self, cam_key: str, toml_name: str, used_ids: set[str]) -> str | None:
        """Return instance id that best matches this TOML entry, or None."""
        candidates = [i for i in self._instances if i["id"] not in used_ids]
        if not candidates:
            return None

        if toml_name:
            name_lower = toml_name.lower()
            matches = [
                i for i in candidates
                if name_lower in i["label"].lower() or i["label"].lower() in name_lower
            ]
            if len(matches) == 1:
                return matches[0]["id"]

        if len(self._cam_keys) == len(self._instances):
            pos = self._cam_keys.index(cam_key)
            if pos < len(candidates):
                return candidates[pos]["id"]

        return None

    def _set_status(self, msg: str | None, *, error: bool = False) -> None:
        if msg is None:
            self._status_label.setVisible(False)
            return
        self._status_label.setText(msg)
        self._status_label.setStyleSheet("color: red;" if error else "color: green;")
        self._status_label.setVisible(True)


# ---------------------------------------------------------------------------
# Wizard page
# ---------------------------------------------------------------------------


class ExtrinsicsPage(QWizardPage):
    """Wizard page 4 — import extrinsic calibration (optional step)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("Extrinsic Calibration")
        self.setSubTitle(
            "Import a Pose2Sim cameras.toml to add camera positions to the session. "
            "This step is optional — you can import extrinsics later from the pose window."
        )
        self._widget = ExtrinsicsImportWidget()
        self._widget.imported.connect(self._on_imported)
        layout = QVBoxLayout(self)
        layout.addWidget(self._widget)

    def initializePage(self) -> None:  # noqa: N802
        wiz = self.wizard()
        conn = getattr(wiz, "session_conn", None)
        sid = getattr(wiz, "session_id", None)
        shot_ids: list[str] = getattr(wiz, "new_shot_ids", [])
        if conn is not None and sid is not None:
            self._widget.set_session(conn, sid, shot_ids)

    def _on_imported(self, calib_id: str) -> None:
        wiz = self.wizard()
        conn = getattr(wiz, "session_conn", None)
        shot_ids: list[str] = getattr(wiz, "new_shot_ids", [])
        if conn is None or not shot_ids:
            return
        with conn:
            conn.executemany(
                "UPDATE captures SET extrinsic_calibration_id = ? WHERE id = ?",
                [(calib_id, sid) for sid in shot_ids],
            )

    def isComplete(self) -> bool:  # noqa: N802
        return True


# ---------------------------------------------------------------------------
# Standalone dialog
# ---------------------------------------------------------------------------


class ExtrinsicsImportDialog(QDialog):
    """Dialog for importing extrinsics outside the wizard (e.g. pose window)."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        shot_ids: list[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Extrinsic Calibration")
        self.setMinimumWidth(560)

        self._conn = conn
        self._shot_ids = shot_ids or []

        self._widget = ExtrinsicsImportWidget()
        self._widget.set_session(conn, session_id, self._shot_ids)
        self._widget.imported.connect(self._on_imported)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._widget)
        layout.addWidget(buttons)

    def _on_imported(self, calib_id: str) -> None:
        if not self._shot_ids:
            return
        with self._conn:
            self._conn.executemany(
                "UPDATE captures SET extrinsic_calibration_id = ? WHERE id = ?",
                [(calib_id, sid) for sid in self._shot_ids],
            )

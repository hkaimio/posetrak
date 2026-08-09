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
import logging
import re
import sqlite3
import struct
import threading
import tomllib
from pathlib import Path

_log = logging.getLogger(__name__)

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
    QSpinBox,
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
    CamPosObs,
    ControlPoint,
    MarkerGroup,
    ObsPoint,
    _Cancelled,
    _proj_matrix,
    _undistort_pts,
    load_control_points,
    run_calibration,
    save_control_points,
)
from app.setup.fiducial_markers import (
    ArucoDetector,
    CharucoBoardDetection,
    CharucoDetector,
    anchor_from_charuco_board,
    merge_detections_into_groups,
)
from app.setup.video_scrub_bar import VideoScrubBar
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

# Curated subset of cv2.aruco's dictionaries -- the full ARUCO_DICTIONARIES
# map (fiducial_markers.py) has ~20 entries; these are the ones actually
# common in printed calibration markers.
_ARUCO_DICTIONARY_CHOICES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250",
    "DICT_ARUCO_ORIGINAL",
]

# Gold, distinct from _CP_COLORS, for drawn ArUco marker corner overlays.
_ARUCO_MARKER_COLOR = QColor(255, 200, 40)

# Cyan, distinct from both _CP_COLORS and _ARUCO_MARKER_COLOR, for drawn
# ChArUco board corner overlays.
_CHARUCO_CORNER_COLOR = QColor(0, 210, 230)

# ---------------------------------------------------------------------------
# Label-matching helpers (mirrors calibrate_from_exports.py — kept in sync)
# ---------------------------------------------------------------------------

_FNAME_RE_UI = re.compile(r"^.+?_\d{1,2}_\d{1,2}_\d{1,3}_(.+?)_\d+\.png$", re.IGNORECASE)


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
    """Load CamCalibState list by matching PNG files to calibrated cameras.

    Searches ALL camera instances in the session DB that have any intrinsics
    calibration — not just those with capture_videos entries for the current shot.
    The session DB must be self-contained (all cameras and intrinsics used by the
    session must be present there).  When adding intrinsics via the camera registry
    dialog, they are mirrored to the session DB automatically.

    Priority for picking the calibration:
      1. Per-video override in capture_videos.intrinsics_calibration_id
      2. Mode default (camera_modes.default_intrinsics_calibration_id)
      3. Latest calibration for the camera's mode
    """
    final_calib: dict[str, str] = {}  # cam_label → calibration_id

    old_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        # Diagnostic: log session DB state before querying
        n_models = conn.execute("SELECT COUNT(*) FROM camera_models").fetchone()[0]
        n_modes  = conn.execute("SELECT COUNT(*) FROM camera_modes").fetchone()[0]
        n_cals   = conn.execute("SELECT COUNT(*) FROM intrinsics_calibrations").fetchone()[0]
        n_insts  = conn.execute("SELECT COUNT(*) FROM camera_instances").fetchone()[0]
        _log.debug(
            "_load_states: session DB has %d models, %d modes, %d calibrations, %d instances",
            n_models, n_modes, n_cals, n_insts,
        )
        inst_labels = [r[0] for r in conn.execute("SELECT label FROM camera_instances").fetchall()]
        _log.debug("  camera_instances labels: %s", inst_labels)

        # Registry-wide search: any camera instance whose model has a calibrated mode.
        # ORDER BY puts default calibrations first, then newest — first row per label wins.
        for r in conn.execute(
            """
            SELECT ci.label AS cam_label, ic.id AS calib_id,
                   (cm.default_intrinsics_calibration_id = ic.id) AS is_default
            FROM camera_instances ci
            JOIN camera_modes cm ON cm.camera_model_id = ci.camera_model_id
            JOIN intrinsics_calibrations ic ON ic.camera_mode_id = cm.id
            ORDER BY is_default DESC, ic.calibrated_at DESC
            """
        ).fetchall():
            if r["cam_label"] not in final_calib:
                final_calib[r["cam_label"]] = r["calib_id"]

        _log.debug("  cameras with calibration via JOIN: %s", list(final_calib.keys()))

        # Per-video override from capture_videos for this shot takes highest priority.
        for r in conn.execute(
            """
            SELECT ci.label AS cam_label, cv.intrinsics_calibration_id
            FROM capture_videos cv
            JOIN camera_instances ci ON ci.id = cv.camera_instance_id
            WHERE cv.shot_id = ? AND cv.intrinsics_calibration_id IS NOT NULL
            """,
            (shot_id,),
        ).fetchall():
            final_calib[r["cam_label"]] = r["intrinsics_calibration_id"]

        # Collect all labels we know about (with or without intrinsics) for matching.
        all_labels: set[str] = set(final_calib.keys())
        for r in conn.execute(
            """
            SELECT ci.label AS cam_label
            FROM capture_videos cv
            JOIN camera_instances ci ON ci.id = cv.camera_instance_id
            WHERE cv.shot_id = ?
            """,
            (shot_id,),
        ).fetchall():
            all_labels.add(r["cam_label"])

        intrinsics: dict[str, dict] = {}
        for label, calib_id in final_calib.items():
            ic = conn.execute(
                "SELECT * FROM intrinsics_calibrations WHERE id = ?", (calib_id,)
            ).fetchone()
            if ic is None:
                continue
            intrinsics[label] = _resolve_intrinsics(ic)
    finally:
        conn.row_factory = old_factory

    all_labels_list = list(all_labels)
    states: list[CamCalibState] = []
    for png in sorted(images_dir.glob("*.png")):
        file_label = _ui_label_from_filename(png.name)
        if file_label is None:
            print(f"  SKIP {png.name} — filename doesn't match expected pattern")
            continue
        db_label = _ui_match_label(file_label, all_labels_list)
        if db_label is None:
            print(f"  SKIP {png.name} — '{file_label}' not found in camera registry")
            continue
        if db_label not in intrinsics:
            print(f"  SKIP {png.name} — '{db_label}' has no intrinsics calibration")
            continue
        img = cv2.imread(str(png))
        if img is None:
            print(f"  SKIP {png.name} — could not read image")
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
            calib_id=final_calib.get(db_label),
        ))
        print(f"  LOAD {db_label} ({img.shape[1]}×{img.shape[0]}) from {png.name}")
    return states


def _resolve_intrinsics(ic: sqlite3.Row) -> dict:
    """Build the K/K_orig/dist/fisheye dict used by CamCalibState from an
    intrinsics_calibrations row."""
    fx, fy, cx, cy = ic["fx"], ic["fy"], ic["cx"], ic["cy"]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    K_orig = K.copy()
    if ic["matrix_original"]:
        vals = struct.unpack("<9d", bytes(ic["matrix_original"]))
        K_orig = np.array(vals).reshape(3, 3)
    if ic["dist_coeffs"]:
        n = len(bytes(ic["dist_coeffs"])) // 8
        dist = np.array(struct.unpack(f"<{n}d", bytes(ic["dist_coeffs"]))).reshape(1, -1)
    else:
        dist = np.zeros((1, 4))
    return {
        "K": K, "K_orig": K_orig, "dist": dist,
        "fisheye": ic["distortion_model"] == "fisheye",
    }


def _load_states_from_capture(
    conn: sqlite3.Connection,
    shot_id: str,
) -> list[CamCalibState]:
    """Load CamCalibState list directly from this capture's video files.

    Unlike ``_load_states_from_images``, no filename matching is needed:
    ``capture_videos.camera_instance_id`` already links each video file
    directly to its camera instance, so intrinsics resolve without going
    through a camera label round-trip.

    ``CamCalibState.image`` is left ``None`` — the caller is expected to
    populate it by scrubbing to an initial frame per camera (see
    ``ExtrinsicsAutoCalibDialog``, which wires a ``VideoScrubBar`` per camera
    using the ``file_path``/``first_frame``/``last_frame`` fields set here).

    Priority for picking the intrinsics calibration (same as
    ``_load_states_from_images``):
      1. Per-video override in capture_videos.intrinsics_calibration_id
      2. Mode default (camera_modes.default_intrinsics_calibration_id)
      3. Latest calibration for the camera's mode
    """
    old_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT cv.file_path, cv.first_video_frame, cv.last_video_frame,
                   cv.intrinsics_calibration_id AS cv_calib_id,
                   ci.label AS cam_label,
                   cm.id AS camera_mode_id,
                   cm.default_intrinsics_calibration_id AS mode_default_calib_id
            FROM capture_videos cv
            JOIN camera_instances ci ON ci.id = cv.camera_instance_id
            LEFT JOIN camera_modes cm ON cm.id = cv.camera_mode_id
            WHERE cv.shot_id = ?
            ORDER BY ci.label
            """,
            (shot_id,),
        ).fetchall()

        states: list[CamCalibState] = []
        for r in rows:
            calib_id = r["cv_calib_id"] or r["mode_default_calib_id"]
            if calib_id is None and r["camera_mode_id"] is not None:
                latest = conn.execute(
                    "SELECT id FROM intrinsics_calibrations WHERE camera_mode_id = ?"
                    " ORDER BY calibrated_at DESC LIMIT 1",
                    (r["camera_mode_id"],),
                ).fetchone()
                calib_id = latest["id"] if latest else None
            if calib_id is None:
                _log.warning(
                    "  SKIP %s — no intrinsics calibration available", r["cam_label"]
                )
                continue

            ic = conn.execute(
                "SELECT * FROM intrinsics_calibrations WHERE id = ?", (calib_id,)
            ).fetchone()
            if ic is None:
                continue

            states.append(CamCalibState(
                video_id=r["cam_label"],
                label=r["cam_label"],
                calib_id=calib_id,
                file_path=r["file_path"],
                first_frame=r["first_video_frame"],
                last_frame=r["last_video_frame"],
                **_resolve_intrinsics(ic),
            ))
        return states
    finally:
        conn.row_factory = old_factory


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

    point_set = Signal(float, float)        # original image coords on mouse release
    cam_pos_set = Signal(str, float, float) # subject_label, image_x, image_y on cam-marker drag
    hovered = Signal(bool)                  # True = mouse entered, False = left
    sift_feature_hovered = Signal(int)      # index in _sift_pts nearest mouse, -1 = none

    def __init__(self, cam_label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cam_label = cam_label
        self._img_bgr: np.ndarray | None = None
        self._markers: list[tuple[float, float, QColor, str, bool]] = []  # x,y,color,label,selected
        # Image-space position of the currently-selected CP in this camera (if any).
        # Used to initialise _drag_img on press so releasing immediately keeps old position.
        self._selected_marker_pos: tuple[float, float] | None = None

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

        # SIFT feature overlay (set by parent dialog on hover)
        self._sift_pts: np.ndarray | None = None  # Nx2 undistorted image coords
        self._near_sift_idx: int = -1             # index of nearest SIFT pt to cursor

        # Per-feature highlight (set by parent on feature hover)
        self._sift_hi_obs: np.ndarray | None = None   # (1, 2) observed position
        self._sift_hi_proj: np.ndarray | None = None  # (1, 2) projected position

        # Reprojection markers for control points (open circle+cross, same color as CP)
        self._proj_markers: list[tuple[float, float, QColor, bool]] = []

        # Camera-position markers: projected world position of other cameras (auto, gold)
        self._cam_pos_markers: list[tuple[float, float, str]] = []  # x, y, label
        # User-placed camera-position markers (cyan) — persists across solves
        self._user_cam_pos_markers: dict[str, tuple[float, float]] = {}  # label → (x, y)
        # Label of camera currently being dragged (None = normal CP drag)
        self._dragging_cam: str | None = None

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

    def add_marker(self, x: float, y: float, color: QColor, label: str = "",
                   selected: bool = False) -> None:
        self._markers.append((x, y, color, label, selected))
        self.update()

    def set_selected_marker(self, pos: tuple[float, float] | None) -> None:
        """Set existing image-space position of the selected CP for this camera."""
        self._selected_marker_pos = pos

    def add_proj_marker(self, x: float, y: float, color: QColor, selected: bool = False) -> None:
        """Add a reprojection marker (open circle + crosshair) at image position (x, y)."""
        self._proj_markers.append((x, y, color, selected))
        self.update()

    def add_cam_pos_marker(self, x: float, y: float, label: str) -> None:
        self._cam_pos_markers.append((x, y, label))
        self.update()

    def set_user_cam_pos_marker(self, label: str, x: float, y: float) -> None:
        """Record a user-placed camera-position marker (cyan).  Persists across solves."""
        self._user_cam_pos_markers[label] = (x, y)
        self.update()

    def clear_user_cam_pos_markers(self) -> None:
        self._user_cam_pos_markers.clear()
        self.update()

    def clear_markers(self) -> None:
        self._markers.clear()
        self._proj_markers.clear()
        self._cam_pos_markers.clear()
        self._selected_marker_pos = None
        self.update()

    def set_calib_status(self, text: str | None, error: bool = False) -> None:
        self._status_text = text
        self._status_error = error
        self.update()

    def set_sift_overlay(self, pts: np.ndarray | None) -> None:
        """Set SIFT match points to overlay (Nx2 undistorted image coords, or None)."""
        self._sift_pts = pts
        self._near_sift_idx = -1
        self._sift_hi_obs = None
        self._sift_hi_proj = None
        self.update()

    def set_sift_highlight(
        self, obs: np.ndarray | None, proj: np.ndarray | None
    ) -> None:
        """Highlight one SIFT feature: observed position (orange) and/or projected (magenta)."""
        self._sift_hi_obs = obs
        self._sift_hi_proj = proj
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
        # SIFT match overlay (small cyan diamonds)
        if self._sift_pts is not None and len(self._sift_pts) > 0:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            r = 5
            for i, (ix, iy) in enumerate(self._sift_pts):
                wx, wy = self._img_to_widget(float(ix), float(iy))
                # Hovered feature drawn larger in a brighter colour
                if i == self._near_sift_idx:
                    painter.setPen(QPen(QColor(0, 255, 255), 2))
                    r2 = 7
                else:
                    painter.setPen(QPen(QColor(0, 200, 200), 1.5))
                    r2 = r
                diamond = [
                    (int(wx), int(wy) - r2),
                    (int(wx) + r2, int(wy)),
                    (int(wx), int(wy) + r2),
                    (int(wx) - r2, int(wy)),
                ]
                for j in range(4):
                    painter.drawLine(diamond[j][0], diamond[j][1],
                                     diamond[(j + 1) % 4][0], diamond[(j + 1) % 4][1])

        # Per-feature highlight: observed (orange crosshair) and projected (magenta X)
        # with a dashed line connecting them
        wx_o = wy_o = wx_p = wy_p = None
        if self._sift_hi_obs is not None and len(self._sift_hi_obs) > 0:
            ox, oy = float(self._sift_hi_obs[0, 0]), float(self._sift_hi_obs[0, 1])
            wx_o, wy_o = self._img_to_widget(ox, oy)
            ro = 9
            painter.setPen(QPen(QColor(255, 160, 0), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(int(wx_o) - ro, int(wy_o) - ro, ro * 2, ro * 2)
            painter.drawLine(int(wx_o) - ro, int(wy_o), int(wx_o) + ro, int(wy_o))
            painter.drawLine(int(wx_o), int(wy_o) - ro, int(wx_o), int(wy_o) + ro)

        if self._sift_hi_proj is not None and len(self._sift_hi_proj) > 0:
            px_, py_ = float(self._sift_hi_proj[0, 0]), float(self._sift_hi_proj[0, 1])
            wx_p, wy_p = self._img_to_widget(px_, py_)
            rp = 7
            painter.setPen(QPen(QColor(255, 60, 220), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(int(wx_p) - rp, int(wy_p) - rp, int(wx_p) + rp, int(wy_p) + rp)
            painter.drawLine(int(wx_p) - rp, int(wy_p) + rp, int(wx_p) + rp, int(wy_p) - rp)

        if wx_o is not None and wx_p is not None:
            painter.setPen(QPen(QColor(255, 120, 120), 1, Qt.PenStyle.DashLine))
            painter.drawLine(int(wx_o), int(wy_o), int(wx_p), int(wy_p))

        # Reprojection markers (open circle + inner crosshair, same CP colour)
        for mx, my, color, sel in self._proj_markers:
            wx, wy = self._img_to_widget(mx, my)
            r = 10 if sel else 8
            arm = 6 if sel else 5
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if sel:
                painter.setPen(QPen(QColor(255, 255, 255), 3.0))
                painter.drawEllipse(int(wx) - r, int(wy) - r, r * 2, r * 2)
            painter.setPen(QPen(color, 2.0 if sel else 1.5))
            painter.drawEllipse(int(wx) - r, int(wy) - r, r * 2, r * 2)
            painter.drawLine(int(wx) - arm, int(wy), int(wx) + arm, int(wy))
            painter.drawLine(int(wx), int(wy) - arm, int(wx), int(wy) + arm)

        # Camera-position markers: small camera-body icon + label
        cam_bg = QColor(0, 0, 0, 160)
        painter.setFont(painter.font())

        def _draw_cam_icon(wx, wy, color, clabel):
            r = 6
            painter.setBrush(color)
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawRect(int(wx) - r, int(wy) - r + 2, r * 2, r * 2 - 2)
            painter.drawRect(int(wx) - r // 2, int(wy) - r, r, 3)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(clabel)
            lx, ly = int(wx) + r + 3, int(wy) + 4
            painter.setBrush(cam_bg)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(lx - 1, ly - fm.ascent(), tw + 2, fm.height())
            painter.setPen(color)
            painter.drawText(lx, ly, clabel)

        # Auto-computed (gold) — drawn first (below user markers)
        for mx, my, clabel in self._cam_pos_markers:
            wx, wy = self._img_to_widget(mx, my)
            _draw_cam_icon(wx, wy, QColor(255, 210, 0), clabel)

        # User-placed (cyan) — drawn on top with a white ring to distinguish
        for clabel, (mx, my) in self._user_cam_pos_markers.items():
            wx, wy = self._img_to_widget(mx, my)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            painter.drawEllipse(int(wx) - 10, int(wy) - 10, 20, 20)
            _draw_cam_icon(wx, wy, QColor(0, 220, 220), clabel)

        # Manual control point markers (solid colored circles) — drawn on top
        for mx, my, color, mlabel, is_sel in self._markers:
            wx, wy = self._img_to_widget(mx, my)
            if is_sel:
                # White outer ring to make selected CP visually distinct
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(255, 255, 255), 2.5))
                painter.drawEllipse(int(wx) - 12, int(wy) - 12, 24, 24)
            r = 9 if is_sel else 7
            painter.setBrush(color)
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
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

        # Check if press is within HIT_RADIUS (widget pixels) of any camera marker.
        # User-placed markers take priority over auto-computed ones.
        _HIT_RADIUS_W = 18.0  # widget pixels
        self._dragging_cam = None
        cam_drag_start: tuple[float, float] | None = None
        all_cam_markers: list[tuple[float, float, str]] = [
            (mx, my, lbl)
            for lbl, (mx, my) in self._user_cam_pos_markers.items()
        ] + self._cam_pos_markers
        for mx, my, clabel in all_cam_markers:
            mwx, mwy = self._img_to_widget(mx, my)
            if ((mwx - wx) ** 2 + (mwy - wy) ** 2) ** 0.5 <= _HIT_RADIUS_W:
                self._dragging_cam = clabel
                cam_drag_start = (mx, my)
                break

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
        if self._dragging_cam is not None:
            # Camera-marker drag: start from the marker's current position so a
            # release without movement leaves it in the original spot.
            self._drag_img = cam_drag_start
        else:
            # Normal CP drag: start from existing observation or click position.
            self._drag_img = self._selected_marker_pos if self._selected_marker_pos is not None else (ix, iy)
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        # SIFT feature proximity detection (fit-view only, not during zoom/drag)
        if not self._zoom_active and self._sift_pts is not None and len(self._sift_pts) > 0:
            wx, wy = pos.x(), pos.y()
            threshold = 15.0
            nearest = -1
            best = threshold
            for i, (ix_, iy_) in enumerate(self._sift_pts):
                sx, sy = self._img_to_widget(float(ix_), float(iy_))
                d = ((sx - wx) ** 2 + (sy - wy) ** 2) ** 0.5
                if d < best:
                    best = d
                    nearest = i
            if nearest != self._near_sift_idx:
                self._near_sift_idx = nearest
                self.sift_feature_hovered.emit(nearest)
                self.update()

        if not self._zoom_active or self._img_bgr is None:
            return
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
        dragging_cam = self._dragging_cam
        self._zoom_active = False
        self._drag_img = None
        self._dragging_cam = None
        self.unsetCursor()
        self.update()
        if final is not None:
            if dragging_cam is not None:
                self.cam_pos_set.emit(dragging_cam, float(final[0]), float(final[1]))
            else:
                self.point_set.emit(float(final[0]), float(final[1]))

    def enterEvent(self, event) -> None:  # noqa: N802
        self.hovered.emit(True)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.hovered.emit(False)
        if self._near_sift_idx != -1:
            self._near_sift_idx = -1
            self.sift_feature_hovered.emit(-1)

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
    progress = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        states: list[CamCalibState],
        control_points: list[ControlPoint],
        cam_pos_obs: list[CamPosObs] | None = None,
        marker_groups: list[MarkerGroup] | None = None,
        refine_intrinsics: set[str] | None = None,
        locked_cameras: set[str] | None = None,
        cp_only: bool = False,
        pnp_ransac_px: float = 8.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._states = states
        self._control_points = control_points
        self._cam_pos_obs = cam_pos_obs or []
        self._marker_groups = marker_groups or []
        self._refine_intrinsics = refine_intrinsics or set()
        self._locked_cameras = locked_cameras or set()
        self._cp_only = cp_only
        self._pnp_ransac_px = pnp_ransac_px
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            result = run_calibration(
                self._states, self._control_points,
                cam_pos_obs=self._cam_pos_obs or None,
                marker_groups=self._marker_groups or None,
                refine_intrinsics=self._refine_intrinsics or None,
                locked_cameras=self._locked_cameras or None,
                progress_cb=lambda msg: self.progress.emit(msg),
                cancel_event=self._cancel_event,
                cp_only=self._cp_only,
                pnp_ransac_px=self._pnp_ransac_px,
            )
        except _Cancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001
            import traceback
            _log.error("Calibration failed: %s\n%s", exc, traceback.format_exc())
            if not self._cancel_event.is_set():
                self.error_occurred.emit(str(exc))
            else:
                self.cancelled.emit()
            return
        if not self._cancel_event.is_set():
            self.finished.emit(result)
        else:
            self.cancelled.emit()


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
        self._intrinsics_combos: dict[str, QComboBox] = {}
        self._cam_pos_obs: list[CamPosObs] = []
        self._refine_intrinsics: set[str] = set()
        self._locked_cameras: set[str] = set()
        self._lock_cbs: dict[str, QCheckBox] = {}
        self._excluded_cameras: set[str] = set()

        self.setWindowTitle("Auto Extrinsics Calibration")
        self.setMinimumSize(960, 640)
        self.resize(1200, 760)

        self._sift_matches: dict[tuple[str, str], object] | None = None
        self._hov_vid: str | None = None
        self._hov_3d_pts: list[tuple[np.ndarray, dict]] = []
        self._hov_cam_pt_idx: dict[str, list[int]] = {}
        self._cp_3d: dict[str, np.ndarray] = {}
        self._states_by_id = {s.video_id: s for s in states}

        self._cam_widgets: dict[str, _ClickableImageWidget] = {}
        self._scrub_bars: dict[str, VideoScrubBar] = {}
        self._cam_panes: dict[str, QWidget] = {}
        self._marker_groups: dict[str, MarkerGroup] = {}
        self._charuco_detections: dict[str, CharucoBoardDetection] = {}
        self._charuco_anchored: bool = False
        self._charuco_board_face_up: bool = True
        for state in states:
            w = _ClickableImageWidget(state.label)
            if state.image is not None:
                w.set_image(state.image)
            vid = state.video_id
            w.point_set.connect(lambda x, y, v=vid: self._on_cam_click(v, x, y))
            w.cam_pos_set.connect(lambda lbl, x, y, v=vid: self._on_cam_pos_set(v, lbl, x, y))
            w.hovered.connect(lambda entered, v=vid: self._on_cam_hover(v, entered))
            w.sift_feature_hovered.connect(
                lambda idx, v=vid: self._on_sift_feature_hovered(v, idx)
            )
            w.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            w.customContextMenuRequested.connect(
                lambda pos, v=vid: self._on_cam_context_menu(v, pos)
            )
            self._cam_widgets[vid] = w

            pane = QWidget()
            pane_layout = QVBoxLayout(pane)
            pane_layout.setContentsMargins(0, 0, 0, 0)
            pane_layout.addWidget(w, 1)

            if state.file_path:
                # Video-sourced camera (see docs/roadmap/features/
                # extrinsics-improvements/extrinsics-improvements-design.md,
                # "Frame source & scrubbing"): add a scrub bar below the
                # image and refresh both the widget and the CamCalibState's
                # `image` as the user scrubs, so control-point placement and
                # the SIFT/BA solve both see whatever frame is displayed.
                scrub = VideoScrubBar()
                scrub.frame_ready.connect(
                    lambda idx, frame, v=vid, widget=w: self._on_scrub_frame_ready(
                        v, widget, frame
                    )
                )
                total_frames = max(state.last_frame - state.first_frame + 1, 1)
                scrub.load(state.file_path, total_frames, initial_frame=0)
                self._scrub_bars[vid] = scrub
                pane_layout.addWidget(scrub)

            # Fiducial detection (Phases 3/4), works on video- and
            # image-sourced cameras alike since it just needs the currently
            # displayed frame: one row of buttons per camera, each acting
            # on that camera only.
            detect_row = QHBoxLayout()
            aruco_btn = QPushButton("Detect ArUco")
            aruco_btn.clicked.connect(lambda _checked, v=vid: self._on_detect_aruco_clicked(v))
            charuco_btn = QPushButton("Detect ChArUco")
            charuco_btn.clicked.connect(lambda _checked, v=vid: self._on_detect_charuco_clicked(v))
            detect_row.addWidget(aruco_btn)
            detect_row.addWidget(charuco_btn)
            pane_layout.addLayout(detect_row)

            self._cam_panes[vid] = pane

        self._build_ui()

    def _on_scrub_frame_ready(
        self, video_id: str, widget: "_ClickableImageWidget", frame: np.ndarray
    ) -> None:
        widget.set_image(frame)
        state = self._states_by_id.get(video_id)
        if state is not None:
            state.image = frame

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Camera grid — fills available space
        n = len(self._cam_panes)
        ncols = 1 if n == 1 else (2 if n <= 4 else 3)

        cam_container = QWidget()
        from PySide6.QtWidgets import QGridLayout
        grid = QGridLayout(cam_container)
        grid.setSpacing(4)
        grid.setContentsMargins(4, 4, 4, 4)
        for col in range(ncols):
            grid.setColumnStretch(col, 1)
        for i, (vid, pane) in enumerate(self._cam_panes.items()):
            row, col = divmod(i, ncols)
            grid.addWidget(pane, row, col)
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
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel_solve)
        self._cancel_btn.setVisible(False)
        self._load_db_btn = QPushButton("Load from DB…")
        self._load_db_btn.clicked.connect(self._on_load_from_db)
        self._load_db_btn.setToolTip("Load a previously saved calibration to inspect camera positions and CP errors.")
        self._sift_check = QCheckBox("SIFT matching")
        self._sift_check.setChecked(True)
        self._sift_check.setToolTip(
            "Use SIFT feature matching to initialise camera poses.\n"
            "Uncheck to use only control points (requires ≥4 world-xyz CPs per camera)."
        )
        self._ransac_px_spin = QDoubleSpinBox()
        self._ransac_px_spin.setRange(1.0, 500.0)
        self._ransac_px_spin.setSingleStep(1.0)
        self._ransac_px_spin.setValue(8.0)
        self._ransac_px_spin.setDecimals(1)
        self._ransac_px_spin.setSuffix(" px")
        self._ransac_px_spin.setToolTip(
            "PnP RANSAC reprojection error threshold.\n"
            "Increase if cameras with bad intrinsics or coplanar CPs fail to solve.\n"
            "Large values allow wrong poses — use only for diagnosis."
        )
        self._ransac_px_spin.setMaximumWidth(90)
        self._status_label = QLabel(
            "Click 'Match & Solve' to run SIFT matching and bundle adjustment.  "
            "Optionally add control points first (press a camera image to place one)."
        )
        self._status_label.setWordWrap(True)

        solve_row = QHBoxLayout()
        solve_row.addWidget(self._solve_btn)
        solve_row.addWidget(self._cancel_btn)
        solve_row.addWidget(self._load_db_btn)
        solve_row.addWidget(self._sift_check)
        solve_row.addWidget(QLabel("RANSAC:"))
        solve_row.addWidget(self._ransac_px_spin)
        solve_row.addWidget(self._status_label, 1)

        # Camera positions table (shown after solve or DB load)
        self._cam_pos_table = QTableWidget(0, 5)
        self._cam_pos_table.setHorizontalHeaderLabels(
            ["Camera", "X (m)", "Y (m)", "Z (m)", "CP error"]
        )
        self._cam_pos_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for col in range(1, 5):
            self._cam_pos_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )
        self._cam_pos_table.setMaximumHeight(140)
        self._cam_pos_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._cam_pos_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._cam_pos_table.setAlternatingRowColors(True)
        self._cam_pos_table.setVisible(False)

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
        root.addWidget(self._cam_pos_table)
        root.addWidget(btn_box)

    def _build_cp_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(280)

        # Control points list
        cp_group = QGroupBox("Control Points")
        cp_layout = QVBoxLayout(cp_group)

        hint = QLabel(
            "Select a point, then press and drag on camera images to place it precisely. "
            "Double-click a point to rename it. "
            "Right-click a camera to remove just that camera's observation. "
            "Set World position to fix scale / origin in BA."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 10px;")

        self._cp_list = QListWidget()
        self._cp_list.currentRowChanged.connect(self._on_cp_selected)
        self._cp_list.itemDoubleClicked.connect(self._rename_control_point)

        add_del = QHBoxLayout()
        add_btn = QPushButton("Add")
        del_btn = QPushButton("Delete")
        load_btn = QPushButton("Load…")
        save_btn = QPushButton("Save…")
        add_btn.clicked.connect(self._add_control_point)
        del_btn.clicked.connect(self._delete_control_point)
        load_btn.clicked.connect(self._load_cp_file)
        save_btn.clicked.connect(self._save_cp_file)
        add_del.addWidget(add_btn)
        add_del.addWidget(del_btn)
        add_del.addStretch()
        add_del.addWidget(load_btn)
        add_del.addWidget(save_btn)

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
        v.addWidget(self._build_aruco_group())
        v.addWidget(self._build_charuco_group())
        v.addWidget(self._build_intrinsics_group())
        return panel

    def _build_aruco_group(self) -> QGroupBox:
        """ArUco marker detection settings + detected-marker list (Phase 3).

        See docs/roadmap/features/extrinsics-improvements/
        extrinsics-improvements-design.md, section 3.
        """
        group = QGroupBox("ArUco Markers")
        layout = QVBoxLayout(group)

        hint = QLabel(
            "Click \"Detect ArUco\" under a camera to find markers in its "
            "current frame. A marker's 4 corners act as one control-point "
            "group; give it a size (below) to also recover its rigid "
            "world pose once ≥2 cameras have seen it."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 10px;")

        dict_row = QHBoxLayout()
        dict_row.addWidget(QLabel("Dictionary:"))
        self._aruco_dict_combo = QComboBox()
        self._aruco_dict_combo.addItems(_ARUCO_DICTIONARY_CHOICES)
        dict_row.addWidget(self._aruco_dict_combo, 1)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Default size:"))
        self._aruco_default_size_spin = QDoubleSpinBox()
        self._aruco_default_size_spin.setRange(0.0, 5.0)
        self._aruco_default_size_spin.setDecimals(4)
        self._aruco_default_size_spin.setSingleStep(0.005)
        self._aruco_default_size_spin.setSuffix(" m")
        self._aruco_default_size_spin.setToolTip(
            "0 = unknown size: marker corners still contribute as free "
            "control points, but no rigid pose is solved for it."
        )
        size_row.addWidget(self._aruco_default_size_spin, 1)

        min_size_row = QHBoxLayout()
        min_size_row.addWidget(QLabel("Min marker size:"))
        self._aruco_min_marker_pct_spin = QDoubleSpinBox()
        self._aruco_min_marker_pct_spin.setRange(0.1, 10.0)
        self._aruco_min_marker_pct_spin.setDecimals(2)
        self._aruco_min_marker_pct_spin.setSingleStep(0.1)
        self._aruco_min_marker_pct_spin.setValue(1.0)
        self._aruco_min_marker_pct_spin.setSuffix(" %")
        self._aruco_min_marker_pct_spin.setToolTip(
            "Smallest marker cv2.aruco will accept, as a percentage of the "
            "frame's larger dimension. OpenCV's own default is 3%, which "
            "misses markers photographed from across a room in a full "
            "4K/similar frame -- lower this if \"Detect\" finds nothing "
            "despite the marker clearly being visible.\n\n"
            "Going too low is not always better: past a point it starts "
            "accepting false-positive/misdecoded quads, which can break "
            "detection just as badly as too few real markers. If the log "
            "shows the same marker id decoded more than once, you've gone "
            "too low -- raise this back up rather than lowering it further."
        )
        min_size_row.addWidget(self._aruco_min_marker_pct_spin, 1)

        self._marker_table = QTableWidget(0, 3)
        self._marker_table.setHorizontalHeaderLabels(["Marker", "Cameras", "Size override (m)"])
        hdr = self._marker_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._marker_table.verticalHeader().setVisible(False)
        self._marker_table.setMinimumHeight(80)
        self._marker_table.setMaximumHeight(140)

        clear_btn = QPushButton("Clear markers")
        clear_btn.clicked.connect(self._on_clear_markers)

        layout.addWidget(hint)
        layout.addLayout(dict_row)
        layout.addLayout(size_row)
        layout.addLayout(min_size_row)
        layout.addWidget(self._marker_table)
        layout.addWidget(clear_btn)
        return group

    def _build_charuco_group(self) -> QGroupBox:
        """ChArUco board detection + coordinate-system anchoring (Phase 4).

        See docs/roadmap/features/extrinsics-improvements/
        extrinsics-improvements-design.md, section 4, and status.md's
        Phase 4 notes for why "Set origin & axes from board" doesn't need
        a reference camera or its intrinsics.
        """
        group = QGroupBox("ChArUco Board")
        layout = QVBoxLayout(group)

        hint = QLabel(
            "Click \"Detect ChArUco\" under a camera to find the board in "
            "its current frame. Once detected in at least one camera, "
            "\"Set origin & axes\" fixes the world coordinate system to "
            "the board's own geometry -- scale, origin, and axes together."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 10px;")

        dict_row = QHBoxLayout()
        dict_row.addWidget(QLabel("Dictionary:"))
        self._charuco_dict_combo = QComboBox()
        self._charuco_dict_combo.addItems(_ARUCO_DICTIONARY_CHOICES)
        dict_row.addWidget(self._charuco_dict_combo, 1)

        squares_row = QHBoxLayout()
        squares_row.addWidget(QLabel("Squares X/Y:"))
        self._charuco_squares_x_spin = QSpinBox()
        self._charuco_squares_x_spin.setRange(2, 50)
        self._charuco_squares_x_spin.setValue(5)
        self._charuco_squares_y_spin = QSpinBox()
        self._charuco_squares_y_spin.setRange(2, 50)
        self._charuco_squares_y_spin.setValue(7)
        squares_row.addWidget(self._charuco_squares_x_spin)
        squares_row.addWidget(self._charuco_squares_y_spin)

        length_row = QHBoxLayout()
        length_row.addWidget(QLabel("Square/marker (m):"))
        self._charuco_square_length_spin = QDoubleSpinBox()
        self._charuco_square_length_spin.setRange(0.001, 5.0)
        self._charuco_square_length_spin.setDecimals(4)
        self._charuco_square_length_spin.setSingleStep(0.005)
        self._charuco_square_length_spin.setValue(0.04)
        self._charuco_marker_length_spin = QDoubleSpinBox()
        self._charuco_marker_length_spin.setRange(0.001, 5.0)
        self._charuco_marker_length_spin.setDecimals(4)
        self._charuco_marker_length_spin.setSingleStep(0.005)
        self._charuco_marker_length_spin.setValue(0.02)
        length_row.addWidget(self._charuco_square_length_spin)
        length_row.addWidget(self._charuco_marker_length_spin)

        self._charuco_face_up_cb = QCheckBox("Board face up (+Z is world up)")
        self._charuco_face_up_cb.setChecked(True)
        self._charuco_face_up_cb.setToolTip(
            "Unchecked: the board is mounted face-down; Y and Z are "
            "negated together so the world frame stays right-handed."
        )

        self._charuco_legacy_pattern_cb = QCheckBox("Legacy pattern (calib.io / older boards)")
        self._charuco_legacy_pattern_cb.setToolTip(
            "Boards generated before OpenCV 4.7's ChArUco marker-placement "
            "change (calib.io's generator among them) need this checked. "
            "If \"Detect ChArUco\" finds nothing, try toggling this before "
            "suspecting the board itself -- ArUco markers still detect "
            "fine either way, only the checkerboard-corner step is affected."
        )

        min_size_row = QHBoxLayout()
        min_size_row.addWidget(QLabel("Min marker size:"))
        self._charuco_min_marker_pct_spin = QDoubleSpinBox()
        self._charuco_min_marker_pct_spin.setRange(0.1, 10.0)
        self._charuco_min_marker_pct_spin.setDecimals(2)
        self._charuco_min_marker_pct_spin.setSingleStep(0.1)
        self._charuco_min_marker_pct_spin.setValue(1.0)
        self._charuco_min_marker_pct_spin.setSuffix(" %")
        self._charuco_min_marker_pct_spin.setToolTip(
            "Smallest marker cv2.aruco will accept, as a percentage of the "
            "frame's larger dimension. OpenCV's own default is 3%, which "
            "misses a board photographed from across a room in a full "
            "4K/similar frame -- lower this if \"Detect ChArUco\" finds "
            "nothing despite the board clearly being visible.\n\n"
            "Going too low is not always better: past a point it starts "
            "accepting false-positive/misdecoded quads, which breaks corner "
            "interpolation just as badly as too few real markers -- often "
            "with MORE markers found overall but still zero corners. If the "
            "log shows the same marker id decoded more than once, you've "
            "gone too low; raise this back up rather than lowering it "
            "further. There is usually only a narrow working range."
        )
        min_size_row.addWidget(self._charuco_min_marker_pct_spin, 1)

        self._charuco_status_label = QLabel("No board detected yet.")
        self._charuco_status_label.setWordWrap(True)
        self._charuco_status_label.setStyleSheet("color: #666; font-size: 10px;")

        anchor_btn = QPushButton("Set origin && axes from board")
        anchor_btn.clicked.connect(self._on_anchor_from_board)
        clear_btn = QPushButton("Clear board detections")
        clear_btn.clicked.connect(self._on_clear_charuco)
        btn_row = QHBoxLayout()
        btn_row.addWidget(anchor_btn)
        btn_row.addWidget(clear_btn)

        layout.addWidget(hint)
        layout.addLayout(dict_row)
        layout.addLayout(squares_row)
        layout.addLayout(length_row)
        layout.addWidget(self._charuco_face_up_cb)
        layout.addWidget(self._charuco_legacy_pattern_cb)
        layout.addLayout(min_size_row)
        layout.addWidget(self._charuco_status_label)
        layout.addLayout(btn_row)
        return group

    def _build_intrinsics_group(self) -> QGroupBox:
        group = QGroupBox("Camera Intrinsics")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        for state in self._states:
            row = QHBoxLayout()
            row.setSpacing(4)
            lbl = QLabel(state.label)
            lbl.setFixedWidth(80)
            lbl.setToolTip(state.label)
            combo = QComboBox()
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self._populate_intrinsics_combo(state, combo)
            vid = state.video_id
            combo.currentIndexChanged.connect(
                lambda _idx, v=vid, c=combo: self._on_intrinsics_changed(v, c.currentData())
            )
            self._intrinsics_combos[state.video_id] = combo
            refine_cb = QCheckBox("Refine")
            refine_cb.setToolTip("Optimise fx/fy for this camera during bundle adjustment")
            refine_cb.toggled.connect(
                lambda checked, v=vid: (
                    self._refine_intrinsics.add(v) if checked
                    else self._refine_intrinsics.discard(v)
                )
            )
            lock_cb = QCheckBox("Lock")
            lock_cb.setToolTip("Keep this camera's pose fixed in the next solve")
            lock_cb.setEnabled(state.R is not None)
            lock_cb.toggled.connect(
                lambda checked, v=vid: (
                    self._locked_cameras.add(v) if checked
                    else self._locked_cameras.discard(v)
                )
            )
            self._lock_cbs[vid] = lock_cb
            excl_cb = QCheckBox("Excl")
            excl_cb.setToolTip("Exclude this camera from the next solve entirely")
            excl_cb.toggled.connect(
                lambda checked, v=vid: (
                    self._excluded_cameras.add(v) if checked
                    else self._excluded_cameras.discard(v)
                )
            )
            row.addWidget(lbl)
            row.addWidget(combo, 1)
            row.addWidget(refine_cb)
            row.addWidget(lock_cb)
            row.addWidget(excl_cb)
            layout.addLayout(row)

        return group

    def _populate_intrinsics_combo(self, state: CamCalibState, combo: QComboBox) -> None:
        old_factory = self._conn.row_factory
        self._conn.row_factory = sqlite3.Row
        try:
            rows = self._conn.execute(
                """
                SELECT ic.id, ic.calibrated_at, ic.rms_error, ic.distortion_model,
                       (cm.default_intrinsics_calibration_id = ic.id) AS is_default
                FROM intrinsics_calibrations ic
                JOIN camera_modes cm ON cm.id = ic.camera_mode_id
                JOIN camera_instances ci ON ci.camera_model_id = cm.camera_model_id
                WHERE ci.label = ?
                ORDER BY is_default DESC, ic.calibrated_at DESC
                """,
                (state.label,),
            ).fetchall()
        finally:
            self._conn.row_factory = old_factory

        combo.blockSignals(True)
        combo.clear()
        for r in rows:
            date = (r["calibrated_at"] or "")[:10]
            rms = f"{r['rms_error']:.2f}px" if r["rms_error"] is not None else "?"
            model = r["distortion_model"] or "standard"
            star = "★ " if r["is_default"] else ""
            text = f"{star}{date}  {rms}  {model}"
            combo.addItem(text, userData=r["id"])
            if r["id"] == state.calib_id:
                combo.setCurrentIndex(combo.count() - 1)
        combo.blockSignals(False)

    def _on_intrinsics_changed(self, video_id: str, calib_id: str | None) -> None:
        if not calib_id:
            return
        state = self._states_by_id.get(video_id)
        if state is None:
            return
        old_factory = self._conn.row_factory
        self._conn.row_factory = sqlite3.Row
        try:
            ic = self._conn.execute(
                "SELECT * FROM intrinsics_calibrations WHERE id = ?", (calib_id,)
            ).fetchone()
        finally:
            self._conn.row_factory = old_factory
        if ic is None:
            return

        fx, fy, cx, cy = ic["fx"], ic["fy"], ic["cx"], ic["cy"]
        K_new = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        K_orig = K_new.copy()
        if ic["matrix_original"]:
            vals = struct.unpack("<9d", bytes(ic["matrix_original"]))
            K_orig = np.array(vals).reshape(3, 3)
        if ic["dist_coeffs"]:
            n = len(bytes(ic["dist_coeffs"])) // 8
            dist = np.array(struct.unpack(f"<{n}d", bytes(ic["dist_coeffs"]))).reshape(1, -1)
        else:
            dist = np.zeros((1, 4))

        state.K = K_new
        state.K_orig = K_orig
        state.dist = dist
        state.fisheye = ic["distortion_model"] == "fisheye"
        state.calib_id = calib_id

    # ------------------------------------------------------------------
    # Control-point slots
    # ------------------------------------------------------------------

    def _cp_list_label(self, cp: "ControlPoint") -> str:
        kind = "fixed" if cp.world_xyz is not None else "free"
        return f"{cp.name}  ({kind}, {len(cp.obs)} cam{'s' if len(cp.obs) != 1 else ''})"

    def _refresh_cp_list_labels(self) -> None:
        for row, cp in enumerate(self._control_points):
            item = self._cp_list.item(row)
            if item is not None:
                item.setText(self._cp_list_label(cp))

    def _add_control_point(self) -> None:
        name = f"CP{len(self._control_points) + 1}"
        cp = ControlPoint(name=name)
        self._control_points.append(cp)
        self._cp_list.addItem(self._cp_list_label(cp))
        self._cp_list.setCurrentRow(len(self._control_points) - 1)

    def _rename_control_point(self, item) -> None:
        row = self._cp_list.row(item)
        if row < 0 or row >= len(self._control_points):
            return
        cp = self._control_points[row]
        name, ok = QInputDialog.getText(self, "Rename Control Point", "Name:", text=cp.name)
        if ok and name.strip():
            cp.name = name.strip()
            item.setText(self._cp_list_label(cp))

    def _delete_control_point(self) -> None:
        row = self._cp_list.currentRow()
        if row < 0:
            return
        self._control_points.pop(row)
        self._cp_list.takeItem(row)
        self._selected_cp_idx = None
        self._refresh_markers()

    def _load_cp_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Control Points", "", "CP files (*.json);;All files (*)"
        )
        if not path:
            return
        # A version-1 file (no per-observation frame_idx) falls back to each
        # camera's current scrub position — see load_control_points'
        # docstring and the design doc's "Per-control-point, per-frame
        # observations" section.
        default_frame_by_id = {vid: self._current_frame_for(vid) for vid in self._states_by_id}
        try:
            cps = load_control_points(path, self._states, default_frame_by_id)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        if not cps:
            QMessageBox.information(self, "Load", "No control points found in file.")
            return
        reply = QMessageBox.question(
            self, "Load Control Points",
            f"Replace existing {len(self._control_points)} CP(s) with "
            f"{len(cps)} loaded from file?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._control_points = cps
        self._cp_list.clear()
        for cp in cps:
            self._cp_list.addItem(self._cp_list_label(cp))
        self._selected_cp_idx = None
        self._refresh_markers()
        # Report how many observations matched current cameras
        matched = sum(len(cp.obs) for cp in cps)
        _log.info("Loaded %d CPs from %s (%d observations matched)", len(cps), path, matched)

    def _save_cp_file(self) -> None:
        if not self._control_points:
            QMessageBox.information(self, "Save", "No control points to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Control Points", "", "CP files (*.json);;All files (*)"
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        try:
            save_control_points(self._control_points, self._states, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        _log.info("Saved %d CPs to %s", len(self._control_points), path)

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
            self._refresh_cp_list_labels()

    def _apply_xyz(self) -> None:
        if self._selected_cp_idx is None:
            return
        cp = self._control_points[self._selected_cp_idx]
        cp.world_xyz = np.array([
            self._xyz_x.value(),
            self._xyz_y.value(),
            self._xyz_z.value(),
        ])
        self._refresh_cp_list_labels()

    # ------------------------------------------------------------------
    # Camera click → record observation
    # ------------------------------------------------------------------

    def _current_frame_for(self, vid: str) -> int:
        """Current scrub position for *vid*, or 0 for an image-only camera."""
        scrub = self._scrub_bars.get(vid)
        return scrub.current_frame if scrub is not None else 0

    def _on_cam_click(self, vid: str, x: float, y: float) -> None:
        if self._selected_cp_idx is None:
            return
        cp = self._control_points[self._selected_cp_idx]
        # Per docs/roadmap/features/extrinsics-improvements/
        # extrinsics-improvements-design.md, "Per-control-point, per-frame
        # observations": record whichever frame this camera is currently
        # scrubbed to, independently of every other camera and every other
        # control point. Re-placing this same point on this same camera at
        # a different scrub position overwrites frame_idx along with px/py.
        cp.obs[vid] = ObsPoint(frame_idx=self._current_frame_for(vid), px=x, py=y)
        self._refresh_cp_list_labels()
        self._refresh_markers()

    def _on_cam_pos_set(self, observer_vid: str, subject_label: str, x: float, y: float) -> None:
        """Called when the user drags a camera-position marker in observer_vid's view."""
        # Update or create a CamPosObs for this (observer, subject) pair.
        for obs in self._cam_pos_obs:
            if obs.observer == observer_vid and obs.subject == subject_label:
                obs.pixel = (x, y)
                break
        else:
            self._cam_pos_obs.append(CamPosObs(observer=observer_vid, subject=subject_label, pixel=(x, y)))
        # Show a cyan user-placed marker on the observer's widget.
        w = self._cam_widgets.get(observer_vid)
        if w is not None:
            w.set_user_cam_pos_marker(subject_label, x, y)

    def _on_cam_context_menu(self, vid: str, pos) -> None:
        if self._selected_cp_idx is None or self._selected_cp_idx >= len(self._control_points):
            return
        cp = self._control_points[self._selected_cp_idx]
        if vid not in cp.obs:
            return
        from PySide6.QtWidgets import QMenu
        state = self._states_by_id.get(vid)
        label = state.label if state else vid
        menu = QMenu(self)
        action = menu.addAction(f"Remove '{cp.name}' from {label}")
        action.triggered.connect(lambda: self._remove_cp_from_camera(vid))
        w = self._cam_widgets[vid]
        menu.exec(w.mapToGlobal(pos))

    def _remove_cp_from_camera(self, vid: str) -> None:
        if self._selected_cp_idx is None or self._selected_cp_idx >= len(self._control_points):
            return
        cp = self._control_points[self._selected_cp_idx]
        cp.obs.pop(vid, None)
        self._refresh_cp_list_labels()
        self._refresh_markers()

    def _refresh_markers(self) -> None:
        for w in self._cam_widgets.values():
            w.clear_markers()
        for i, cp in enumerate(self._control_points):
            color = _CP_COLORS[i % len(_CP_COLORS)]
            is_selected = (i == self._selected_cp_idx)
            mlabel = cp.name if is_selected else ""
            for vid, obs in cp.obs.items():
                x, y = obs.px, obs.py
                if vid in self._cam_widgets:
                    self._cam_widgets[vid].add_marker(x, y, color, mlabel, selected=is_selected)
                    if is_selected:
                        self._cam_widgets[vid].set_selected_marker((x, y))
            # Reprojection marker (open circle+cross) if we have a 3D position
            xyz = self._cp_3d.get(cp.name)
            if xyz is None or self._result is None:
                continue
            for vid, w in self._cam_widgets.items():
                state = self._result.cameras.get(vid)
                if state is None or state.R is None:
                    continue
                pt_cam = state.R @ xyz + state.t.flatten()
                if pt_cam[2] <= 0:
                    continue  # behind this camera
                rvec, _ = cv2.Rodrigues(state.R)
                proj, _ = cv2.projectPoints(
                    xyz.reshape(1, 3), rvec, state.t.reshape(3, 1), state.K, np.zeros(4)
                )
                px_, py_ = proj.reshape(2)
                w.add_proj_marker(float(px_), float(py_), color, selected=is_selected)

        # ArUco marker corners (Phase 3) -- same clear/redraw pass as CPs
        # above (clear_markers() clears both, so they must share one pass
        # rather than each calling clear_markers() independently).
        for mg in self._marker_groups.values():
            for vid, corners in mg.obs.items():
                w = self._cam_widgets.get(vid)
                if w is None:
                    continue
                for corner_idx, obs in corners.items():
                    label = mg.marker_id if corner_idx == 0 else ""
                    w.add_marker(obs.px, obs.py, _ARUCO_MARKER_COLOR, label, selected=False)

        # ChArUco board corners (Phase 4) -- same shared clear/redraw pass.
        # Labeled only once (corner 0 across all cameras) to avoid clutter
        # from potentially dozens of corners.
        labeled_once = False
        for det in self._charuco_detections.values():
            for c in det.corners:
                w = self._cam_widgets.get(c.video_id)
                if w is None:
                    continue
                label = ""
                if not labeled_once:
                    label = "board"
                    labeled_once = True
                w.add_marker(c.px, c.py, _CHARUCO_CORNER_COLOR, label, selected=False)

    # ------------------------------------------------------------------
    # ArUco marker detection (Phase 3)
    # ------------------------------------------------------------------

    def _current_default_size(self) -> float | None:
        """0 in the spin box means "unknown" -- see its tooltip."""
        value = self._aruco_default_size_spin.value()
        return value if value > 0 else None

    def _size_for_marker(self, marker_id: str) -> float | None:
        """Per-marker override from the table, falling back to the default."""
        for row in range(self._marker_table.rowCount()):
            if self._marker_table.item(row, 0).text() != marker_id:
                continue
            spin = self._marker_table.cellWidget(row, 2)
            if spin is not None and spin.value() > 0:
                return spin.value()
            break
        return self._current_default_size()

    def _on_detect_aruco_clicked(self, vid: str) -> None:
        state = self._states_by_id.get(vid)
        if state is None or state.image is None:
            QMessageBox.warning(self, "Detect ArUco", "No image loaded for this camera yet.")
            return

        dictionary = self._aruco_dict_combo.currentText()
        detector = ArucoDetector(
            dictionary=dictionary,
            default_size=self._current_default_size(),
            min_marker_perimeter_rate=self._aruco_min_marker_pct_spin.value() / 100.0,
        )
        frame_idx = self._current_frame_for(vid)
        detections = detector.detect(state.image, video_id=vid, frame_idx=frame_idx)

        # Re-resolve each detection's size against any existing per-marker
        # override *before* merging, so a marker already given a custom
        # size in the table keeps it rather than reverting to the default.
        for det in detections:
            size = self._size_for_marker(det.marker_id)
            merge_detections_into_groups([det], self._marker_groups, size=size)

        self._refresh_marker_table()
        self._refresh_markers()
        self._status_label.setText(
            f"Detected {len(detections)} ArUco marker(s) in "
            f"{self._states_by_id[vid].label} (frame {frame_idx})."
        )

    def _on_clear_markers(self) -> None:
        self._marker_groups.clear()
        self._refresh_marker_table()
        self._refresh_markers()

    def _refresh_marker_table(self) -> None:
        self._marker_table.setRowCount(0)
        for marker_id, mg in sorted(self._marker_groups.items()):
            row = self._marker_table.rowCount()
            self._marker_table.insertRow(row)
            self._marker_table.setItem(row, 0, QTableWidgetItem(marker_id))
            n_cams = len(mg.cameras_observing())
            self._marker_table.setItem(row, 1, QTableWidgetItem(str(n_cams)))

            spin = QDoubleSpinBox()
            spin.setRange(0.0, 5.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.005)
            spin.setToolTip("0 = use the default size above")
            spin.setValue(mg.size or 0.0)
            spin.valueChanged.connect(
                lambda value, m=marker_id: self._on_marker_size_override_changed(m, value)
            )
            self._marker_table.setCellWidget(row, 2, spin)

    def _on_marker_size_override_changed(self, marker_id: str, value: float) -> None:
        mg = self._marker_groups.get(marker_id)
        if mg is not None:
            mg.size = value if value > 0 else self._current_default_size()

    # ------------------------------------------------------------------
    # ChArUco board detection + anchoring (Phase 4)
    # ------------------------------------------------------------------

    def _make_charuco_detector(self) -> CharucoDetector:
        return CharucoDetector(
            dictionary=self._charuco_dict_combo.currentText(),
            squares_x=self._charuco_squares_x_spin.value(),
            squares_y=self._charuco_squares_y_spin.value(),
            square_length=self._charuco_square_length_spin.value(),
            marker_length=self._charuco_marker_length_spin.value(),
            legacy_pattern=self._charuco_legacy_pattern_cb.isChecked(),
            min_marker_perimeter_rate=self._charuco_min_marker_pct_spin.value() / 100.0,
        )

    def _on_detect_charuco_clicked(self, vid: str) -> None:
        state = self._states_by_id.get(vid)
        if state is None or state.image is None:
            QMessageBox.warning(self, "Detect ChArUco", "No image loaded for this camera yet.")
            return

        frame_idx = self._current_frame_for(vid)
        detection = self._make_charuco_detector().detect(state.image, video_id=vid, frame_idx=frame_idx)
        if detection is None:
            self._status_label.setText(
                f"No ChArUco board detected in {state.label} (frame {frame_idx}). "
                f"If the board is definitely visible, try swapping Squares X/Y, "
                f"toggling \"Legacy pattern\", or adjusting \"Min marker size\" "
                f"(small in a full-resolution frame) before suspecting the board "
                f"itself -- see the app log for exactly which markers were found. "
                f"Note: lowering \"Min marker size\" too far can also break "
                f"detection (false-positive markers), not just too high a value."
            )
            return

        self._charuco_detections[vid] = detection
        self._refresh_charuco_status()
        self._refresh_markers()
        self._status_label.setText(
            f"Detected {len(detection.corners)} board corners in "
            f"{state.label} (frame {frame_idx})."
        )

    def _on_anchor_from_board(self) -> None:
        if not self._charuco_detections:
            QMessageBox.warning(
                self, "Set origin & axes",
                "Detect the ChArUco board in at least one camera first.",
            )
            return
        self._charuco_board_face_up = self._charuco_face_up_cb.isChecked()
        self._charuco_anchored = True
        self._refresh_charuco_status()
        self._refresh_markers()
        n_corners = len(self._charuco_control_points())
        self._status_label.setText(
            f"World coordinate system anchored from the ChArUco board "
            f"({n_corners} corners across {len(self._charuco_detections)} camera(s))."
        )

    def _on_clear_charuco(self) -> None:
        self._charuco_detections.clear()
        self._charuco_anchored = False
        self._refresh_charuco_status()
        self._refresh_markers()

    def _refresh_charuco_status(self) -> None:
        if not self._charuco_detections:
            self._charuco_status_label.setText("No board detected yet.")
            return
        n_corners = len(self._charuco_control_points())
        state_str = "anchored (fixed world coordinates)" if self._charuco_anchored else "detected, not yet anchored"
        self._charuco_status_label.setText(
            f"Board {state_str}: {n_corners} corner(s) across "
            f"{len(self._charuco_detections)} camera(s)."
        )

    def _charuco_control_points(self) -> list[ControlPoint]:
        """Every detected board corner as a ControlPoint -- free until
        ``_on_anchor_from_board`` is clicked, fixed (``world_xyz`` set)
        afterward. Always freshly derived from the raw per-camera
        detections rather than cached, so there's no stale duplicate state
        to keep in sync (same pattern as ``_marker_groups``).
        """
        if self._charuco_anchored:
            return anchor_from_charuco_board(
                self._charuco_detections, board_face_up=self._charuco_board_face_up
            )
        cps: dict[int, ControlPoint] = {}
        for det in self._charuco_detections.values():
            for c in det.corners:
                cp = cps.get(c.corner_id)
                if cp is None:
                    cp = ControlPoint(name=f"charuco_c{c.corner_id}")
                    cps[c.corner_id] = cp
                cp.obs[c.video_id] = ObsPoint(frame_idx=c.frame_idx, px=c.px, py=c.py)
        return list(cps.values())

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def _on_solve(self) -> None:
        if self._solve_thread and self._solve_thread.isRunning():
            return
        for s in self._states:
            if s.video_id not in self._locked_cameras and s.video_id not in self._excluded_cameras:
                s.R = None
                s.t = None
        for w in self._cam_widgets.values():
            w.set_calib_status(None)
        self._cp_3d = {}
        self._refresh_markers()

        self._solve_btn.setVisible(False)
        self._cancel_btn.setVisible(True)
        self._cancel_btn.setEnabled(True)
        self._accept_btn.setEnabled(False)
        self._status_label.setText("Starting…")

        cp_only = not self._sift_check.isChecked()
        active_states = [s for s in self._states if s.video_id not in self._excluded_cameras]
        all_cps = self._control_points + self._charuco_control_points()
        self._solve_thread = _SolveThread(
            active_states, all_cps,
            cam_pos_obs=self._cam_pos_obs or None,
            marker_groups=list(self._marker_groups.values()) or None,
            refine_intrinsics=self._refine_intrinsics or None,
            locked_cameras=self._locked_cameras or None,
            cp_only=cp_only,
            pnp_ransac_px=self._ransac_px_spin.value(),
            parent=self,
        )
        self._solve_thread.finished.connect(self._on_solve_done)
        self._solve_thread.error_occurred.connect(self._on_solve_error)
        self._solve_thread.progress.connect(self._status_label.setText)
        self._solve_thread.cancelled.connect(self._on_solve_cancelled)
        self._solve_thread.start()

    def _on_cancel_solve(self) -> None:
        if self._solve_thread and self._solve_thread.isRunning():
            self._solve_thread.cancel()
            self._cancel_btn.setEnabled(False)
            self._status_label.setText("Cancelling…")

    def _on_solve_cancelled(self) -> None:
        self._solve_btn.setVisible(True)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setEnabled(True)
        self._status_label.setText("Solve cancelled.")

    def _on_solve_done(self, result: CalibResult) -> None:
        self._result = result
        self._sift_matches = result.pair_matches
        self._solve_btn.setVisible(True)
        self._cancel_btn.setVisible(False)
        n_total = len(result.cameras)
        n_solved = n_total - len(result.unsolved)

        lines = [f"Solved: {n_solved}/{n_total} cameras"]
        for vid, stats in result.reprojection_errors.items():
            s = result.cameras[vid]
            cp_stats = result.cp_reprojection_errors.get(vid)
            cp_str = (f"  | CP: {cp_stats['mean']:.2f} ± {cp_stats['std']:.2f} px"
                      f" (max {cp_stats['max']:.1f}, n={cp_stats['n']})"
                      if cp_stats else "")
            lines.append(
                f"  {s.label}: {stats['mean']:.2f} ± {stats['std']:.2f} px"
                f"  (max {stats['max']:.1f}, n={stats['n']}){cp_str}"
            )
        if result.unsolved:
            lines.append(
                f"  Disconnected: {', '.join(result.unsolved)}"
                f" — add control points shared with a solved camera to connect them."
            )
        if result.marker_poses:
            for marker_id, mp in sorted(result.marker_poses.items()):
                lines.append(
                    f"  Marker {marker_id}: rms {mp.rms_reprojection_px:.2f} px"
                )
        self._status_label.setText("\n".join(lines))
        self._accept_btn.setEnabled(n_solved > 0)

        for vid in self._excluded_cameras:
            if vid in self._cam_widgets:
                self._cam_widgets[vid].set_calib_status("Excluded")

        for vid, cb in self._lock_cbs.items():
            s = result.cameras.get(vid)
            has_pose = s is not None and s.R is not None
            cb.setEnabled(has_pose)
            if not has_pose:
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
                self._locked_cameras.discard(vid)

        # Compute triangulated 3D positions for all CPs (for reprojection markers)
        self._cp_3d = {}
        state_by_id = result.cameras
        for cp in self._control_points:
            if cp.world_xyz is not None:
                self._cp_3d[cp.name] = cp.world_xyz.astype(np.float64)
                continue
            solved_obs = []
            for vid, obs in cp.obs.items():
                px, py = obs.px, obs.py
                s = state_by_id.get(vid)
                if s is None or s.R is None:
                    continue
                pts_u = _undistort_pts(np.array([[px, py]], dtype=np.float32), s)
                solved_obs.append((vid, float(pts_u[0, 0]), float(pts_u[0, 1])))
            if len(solved_obs) < 2:
                continue
            A_rows = []
            for vid, px, py in solved_obs:
                P = _proj_matrix(state_by_id[vid])
                A_rows.append(px * P[2] - P[0])
                A_rows.append(py * P[2] - P[1])
            A = np.array(A_rows, dtype=np.float64)
            _, _, Vt = np.linalg.svd(A)
            h = Vt[-1]
            if abs(h[3]) > 1e-10:
                self._cp_3d[cp.name] = (h[:3] / h[3]).astype(np.float64)

        self._refresh_markers()
        self._refresh_cam_pos_table()
        self._refresh_cam_pos_markers()

        # Update per-camera badges; clear any stale SIFT overlays and highlights
        self._hov_vid = None
        self._hov_3d_pts = []
        self._hov_cam_pt_idx = {}
        for state in self._states:
            vid = state.video_id
            if vid not in self._cam_widgets:
                continue
            w = self._cam_widgets[vid]
            w.set_sift_overlay(None)
            w.set_sift_highlight(None, None)
            if vid in result.unsolved:
                w.set_calib_status("Disconnected", error=True)
            elif vid in result.reprojection_errors:
                err = result.reprojection_errors[vid]["mean"]
                w.set_calib_status(f"err {err:.2f} px", error=err > 5.0)
            else:
                w.set_calib_status(None)

    def _refresh_cam_pos_table(self) -> None:
        """Populate the camera-positions table from current state R/t values."""
        cp_errors = self._result.cp_reprojection_errors if self._result else {}
        self._cam_pos_table.setRowCount(len(self._states))
        for row, s in enumerate(self._states):
            name_item = QTableWidgetItem(s.label)
            if s.R is None:
                self._cam_pos_table.setItem(row, 0, name_item)
                for col in range(1, 5):
                    item = QTableWidgetItem("—")
                    item.setForeground(QColor(150, 150, 150))
                    self._cam_pos_table.setItem(row, col, item)
                continue
            C = -s.R.T @ s.t.flatten()
            z_ok = C[2] > -0.1
            color = QColor(30, 140, 30) if z_ok else QColor(180, 40, 40)
            name_item.setForeground(color)
            self._cam_pos_table.setItem(row, 0, name_item)
            for col, val in enumerate([C[0], C[1], C[2]], start=1):
                item = QTableWidgetItem(f"{val:.3f}")
                item.setForeground(color)
                self._cam_pos_table.setItem(row, col, item)
            err = cp_errors.get(s.video_id)
            if err:
                err_item = QTableWidgetItem(
                    f"{err['mean']:.1f} ± {err['std']:.1f} px  (max {err['max']:.0f})"
                )
                err_item.setForeground(QColor(180, 40, 40) if err["mean"] > 5.0 else QColor(30, 140, 30))
            else:
                err_item = QTableWidgetItem("—")
                err_item.setForeground(QColor(150, 150, 150))
            self._cam_pos_table.setItem(row, 4, err_item)
        self._cam_pos_table.setVisible(True)

    def _refresh_cam_pos_markers(self) -> None:
        """Project each solved camera's world position into every other camera's view."""
        solved = [(s, -s.R.T @ s.t.flatten()) for s in self._states if s.R is not None]
        for s in self._states:
            w = self._cam_widgets.get(s.video_id)
            if w is None or s.R is None:
                continue
            rvec, _ = cv2.Rodrigues(s.R)
            h_img, w_img = s.image.shape[:2] if s.image is not None else (0, 0)
            for other, C in solved:
                if other.video_id == s.video_id:
                    continue
                # Check camera is in front (positive z in this camera's frame)
                p_cam = s.R @ C + s.t.flatten()
                if p_cam[2] <= 0:
                    continue
                proj, _ = cv2.projectPoints(
                    C.reshape(1, 3), rvec, s.t.reshape(3, 1), s.K, np.zeros(4)
                )
                px, py = proj.reshape(2)
                if not (np.isfinite(px) and np.isfinite(py)):
                    continue
                if w_img > 0 and not (0 <= px < w_img and 0 <= py < h_img):
                    continue
                w.add_cam_pos_marker(float(px), float(py), other.label)

    def _on_load_from_db(self) -> None:
        """Pick an existing calibration from the DB and load it into the current states."""
        rows = self._conn.execute(
            "SELECT id, calibrated_at, method, rms_error"
            " FROM extrinsic_calibrations"
            " WHERE session_id = ?"
            " ORDER BY calibrated_at DESC",
            (self._session_id,),
        ).fetchall()
        if not rows:
            QMessageBox.information(self, "Load from DB", "No calibrations found for this session.")
            return

        # Picker dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Calibration")
        dlg.setMinimumWidth(420)
        combo = QComboBox()
        for r in rows:
            rms = f"  rms={r['rms_error']:.3f}" if r["rms_error"] is not None else ""
            combo.addItem(
                f"{r['calibrated_at']}  [{r['method'] or '?'}]{rms}  ({r['id'][:8]}…)",
                r["id"],
            )
        btn_ok = QPushButton("Load")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        dlg_btns = QHBoxLayout()
        dlg_btns.addWidget(btn_ok)
        dlg_btns.addWidget(btn_cancel)
        dlg_lay = QVBoxLayout(dlg)
        dlg_lay.addWidget(QLabel("Choose calibration to inspect:"))
        dlg_lay.addWidget(combo)
        dlg_lay.addLayout(dlg_btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        calib_id = combo.currentData()
        entries = self._conn.execute(
            "SELECT ci.label, ee.R, ee.t"
            " FROM extrinsic_entries ee"
            " JOIN camera_instances ci ON ci.id = ee.camera_instance_id"
            " WHERE ee.extrinsic_calibration_id = ?",
            (calib_id,),
        ).fetchall()

        label_to_state = {s.label: s for s in self._states}
        loaded = 0
        for entry in entries:
            s = label_to_state.get(entry["label"])
            if s is None:
                continue
            R = np.array(struct.unpack("<9d", entry["R"])).reshape(3, 3)
            t = np.array(struct.unpack("<3d", entry["t"])).reshape(3, 1)
            s.R = R
            s.t = t
            loaded += 1

        if loaded == 0:
            QMessageBox.warning(self, "Load from DB", "No cameras matched the stored calibration.")
            return

        # Clear unsolved cameras if they weren't in this calibration
        calib_labels = {e["label"] for e in entries}
        for s in self._states:
            if s.label not in calib_labels:
                s.R = None
                s.t = None

        # Recompute CP 3D positions so reprojection markers appear
        if self._control_points:
            self._cp_3d = {}
            for cp in self._control_points:
                if cp.world_xyz is not None:
                    self._cp_3d[cp.name] = cp.world_xyz.astype(np.float64)

        self._refresh_markers()
        self._refresh_cam_pos_table()
        self._refresh_cam_pos_markers()

        # Update per-camera error badges using CP errors only
        from app.setup.extrinsics_solver import compute_cp_errors
        cp_errs = compute_cp_errors(self._states, self._control_points) if self._control_points else {}
        for s in self._states:
            w = self._cam_widgets.get(s.video_id)
            if w is None:
                continue
            if s.R is None:
                w.set_calib_status("Disconnected", error=True)
            elif s.video_id in cp_errs:
                err = cp_errs[s.video_id]["mean"]
                w.set_calib_status(f"err {err:.2f} px", error=err > 5.0)
            else:
                w.set_calib_status(None)

        # Refresh the stored result's cp errors so the table has data
        if self._result is not None:
            self._result.cp_reprojection_errors.update(cp_errs)
            self._refresh_cam_pos_table()

        self._status_label.setText(
            f"Loaded calibration from DB: {loaded}/{len(self._states)} cameras.  "
            f"CP reprojection errors shown in table."
        )
        _log.info("Loaded calibration %s from DB (%d cameras)", calib_id[:8], loaded)

    def _on_cam_hover(self, vid: str, entered: bool) -> None:
        """Show/hide SIFT feature overlay when hovering over a camera image.

        The hovered camera shows its actual observed SIFT feature positions.
        Other cameras show the actual matched feature positions (from obs_dict),
        not reprojections — so diamonds in all cameras are real pixel measurements.
        """
        if not entered or self._result is None:
            self._hov_vid = None
            self._hov_3d_pts = []
            self._hov_cam_pt_idx = {}
            for w in self._cam_widgets.values():
                w.set_sift_overlay(None)
                w.set_sift_highlight(None, None)
            return

        self._hov_vid = vid
        self._hov_3d_pts = []
        hov_obs_self: list[tuple[float, float]] = []
        for xyz, obs in self._result.points_3d:
            if vid in obs:
                self._hov_3d_pts.append((xyz, obs))
                hov_obs_self.append(obs[vid])

        if not self._hov_3d_pts:
            self._hov_vid = None
            self._hov_cam_pt_idx = {}
            for w in self._cam_widgets.values():
                w.set_sift_overlay(None)
                w.set_sift_highlight(None, None)
            return

        self._hov_cam_pt_idx = {}
        for v, w in self._cam_widgets.items():
            state = self._result.cameras.get(v)
            if state is None or state.R is None:
                w.set_sift_overlay(None)
                self._hov_cam_pt_idx[v] = []
                continue

            if v == vid:
                # Hovered camera: actual observed positions (already collected)
                pts = np.array(hov_obs_self, dtype=np.float32)
                self._hov_cam_pt_idx[v] = list(range(len(self._hov_3d_pts)))
            else:
                # Other cameras: show actual matched positions for shared 3D points
                pairs = [
                    (obs[v], i)
                    for i, (_, obs) in enumerate(self._hov_3d_pts)
                    if v in obs
                ]
                if pairs:
                    pts = np.array([p for p, _ in pairs], dtype=np.float32)
                    self._hov_cam_pt_idx[v] = [i for _, i in pairs]
                else:
                    pts = None
                    self._hov_cam_pt_idx[v] = []

            w.set_sift_overlay(pts if pts is not None and len(pts) > 0 else None)

    def _on_sift_feature_hovered(self, vid: str, idx: int) -> None:
        """Highlight one SIFT feature across all cameras.

        In every camera: magenta X = reprojected 3D position.
        In cameras that observed this feature: orange crosshair = actual observation.
        Dashed line = residual between observed and reprojected.
        """
        if idx < 0 or self._result is None:
            for w in self._cam_widgets.values():
                w.set_sift_highlight(None, None)
            return

        # Map diamond index in camera vid → index into _hov_3d_pts
        cam_idx_list = self._hov_cam_pt_idx.get(vid, [])
        if idx >= len(cam_idx_list):
            return
        hov3d_idx = cam_idx_list[idx]
        xyz_world, obs_dict = self._hov_3d_pts[hov3d_idx]
        xyz_arr = xyz_world.reshape(1, 3)

        for v, w in self._cam_widgets.items():
            state = self._result.cameras.get(v)
            if state is None or state.R is None or state.t is None:
                w.set_sift_highlight(None, None)
                continue

            pts_cam = (state.R @ xyz_arr.T).T + state.t.flatten()
            if pts_cam[0, 2] <= 0:
                w.set_sift_highlight(None, None)
                continue
            fx, fy = state.K[0, 0], state.K[1, 1]
            cx, cy = state.K[0, 2], state.K[1, 2]
            proj_x = fx * pts_cam[0, 0] / pts_cam[0, 2] + cx
            proj_y = fy * pts_cam[0, 1] / pts_cam[0, 2] + cy
            proj_pos = np.array([[proj_x, proj_y]], dtype=np.float32)

            obs_pos = None
            if v in obs_dict:
                ox, oy = obs_dict[v]
                obs_pos = np.array([[ox, oy]], dtype=np.float32)

            w.set_sift_highlight(obs_pos, proj_pos)

    def _on_solve_error(self, msg: str) -> None:
        self._solve_btn.setVisible(True)
        self._cancel_btn.setVisible(False)
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
                # Persist any per-camera intrinsics selection changes.
                for state in self._states:
                    combo = self._intrinsics_combos.get(state.video_id)
                    if combo is None:
                        continue
                    selected_calib_id = combo.currentData()
                    if not selected_calib_id:
                        continue
                    for shot_id in self._shot_ids:
                        self._conn.execute(
                            "UPDATE capture_videos SET intrinsics_calibration_id = ?"
                            " WHERE shot_id = ? AND camera_instance_id = ("
                            "  SELECT id FROM camera_instances WHERE label = ?)",
                            (selected_calib_id, shot_id, state.label),
                        )

        self.imported.emit(calib_id)
        self.accept()

    def done(self, result: int) -> None:  # noqa: N802
        """Shut down any per-camera video scrub bars before the dialog closes.

        Covers accept, reject, and the window-close button alike, since
        QDialog funnels all of them through ``done()``. Without this, a
        VideoScrubBar's FrameReader QThread can still be running when Qt
        destroys the (parented) widget tree, which aborts the process.
        """
        for scrub in self._scrub_bars.values():
            scrub.unload()
        super().done(result)


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
        self._auto_video_btn = QPushButton("Auto-calibrate…")
        self._auto_video_btn.clicked.connect(self._on_auto_calibrate_video)
        self._auto_video_btn.setToolTip(
            "Run semi-automatic calibration by scrubbing this capture's video "
            "files directly — no still-frame export needed."
        )
        self._auto_btn = QPushButton("Auto-calibrate (image folder)…")
        self._auto_btn.clicked.connect(self._on_auto_calibrate)
        self._auto_btn.setToolTip(
            "Legacy path: run calibration from a directory of previously "
            "exported PNG frames instead of scrubbing video directly."
        )

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("TOML file:"))
        file_row.addWidget(self._path_label, 1)
        file_row.addWidget(browse_btn)
        file_row.addWidget(self._auto_video_btn)
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

    def _on_auto_calibrate_video(self) -> None:
        """Primary auto-calibration path: scrub this capture's own video files.

        See docs/roadmap/features/extrinsics-improvements/
        extrinsics-improvements-design.md, "Frame source & scrubbing" — no
        still-frame export step is needed; each camera gets its own
        VideoScrubBar in the calibration dialog.
        """
        if self._conn is None or self._session_id is None:
            QMessageBox.warning(self, "No session", "Open a session before auto-calibrating.")
            return
        if not self._shot_ids:
            QMessageBox.warning(
                self, "No shot",
                "No capture/shot ID available — cannot look up camera video files."
            )
            return

        states = _load_states_from_capture(self._conn, self._shot_ids[0])
        if not states:
            QMessageBox.warning(
                self, "No cameras",
                "No cameras with both a video file and an intrinsics calibration "
                "were found for this capture.\n\n"
                "Make sure video files are registered for this capture and "
                "intrinsics are calibrated for its cameras.",
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

    def _on_auto_calibrate(self) -> None:
        """Legacy auto-calibration path: load a directory of exported PNG frames."""
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
            Path(images_dir), self._conn, self._shot_ids[0],
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

"""content_panels.py — Right-pane detail panels for each tree item type."""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from pathlib import Path

from math import ceil

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

try:
    import pxr  # noqa: F401
    _USD_AVAILABLE = True
except ImportError:
    _USD_AVAILABLE = False

_USD_TOOLTIP = (
    "USD export requires the 'usd-core' package.\n"
    "Install with:  uv pip install usd-core"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> QGroupBox:
    box = QGroupBox(title)
    box.setLayout(QVBoxLayout())
    box.layout().setSpacing(2)
    return box


def _form_row(label: str, value: str) -> tuple[QLabel, QLabel]:
    lbl = QLabel(label)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    val = QLabel(value)
    val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    val.setWordWrap(True)
    return lbl, val


def _action_btn(text: str, enabled: bool = True) -> QPushButton:
    btn = QPushButton(text)
    btn.setEnabled(enabled)
    btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return btn


def _scrollable(inner: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(inner)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    return area


def _fmt_ts(ts: str | None) -> str:
    return ts[:16].replace("T", " ") if ts else "—"


def _fmt_time(s: float | None) -> str:
    return f"{s:.3f} s" if s is not None else "—"


# ---------------------------------------------------------------------------
# CapturePanel
# ---------------------------------------------------------------------------


class CapturePanel(QWidget):
    """Sync scrubber + pose detection launcher for a capture."""

    data_changed = Signal()  # emitted after a new trial+detection run is created

    def __init__(self, conn: sqlite3.Connection, capture_id: str,
                 session_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._capture_id = capture_id
        self._session_path = session_path
        self._scrubber = None
        self._start_s: float = 0.0
        self._end_s: float = 0.0
        self._start_label: QLabel | None = None
        self._end_label: QLabel | None = None
        self._detect_btn: QPushButton | None = None
        self._build()

    def shutdown(self) -> None:
        if self._scrubber is not None:
            self._scrubber.shutdown()

    def _build(self) -> None:
        from app.setup.db_context import SyncPoint, SyncTable
        from app.setup.frame_cache import FrameCache
        from app.setup.multi_video_scrubber import CellInfo, MultiVideoScrubber

        videos = self._conn.execute(
            "SELECT cv.id, cv.file_path, cv.actual_fps, "
            "       cv.first_video_frame, cv.last_video_frame, "
            "       COALESCE(ci.label, cv.id) AS cam_label "
            "FROM capture_videos cv "
            "LEFT JOIN camera_instances ci ON ci.id = cv.camera_instance_id "
            "WHERE cv.shot_id = ? ORDER BY cam_label",
            (self._capture_id,),
        ).fetchall()

        syncs = self._conn.execute(
            "SELECT id, created_by, notes FROM sync_configs WHERE shot_id = ? ORDER BY rowid",
            (self._capture_id,),
        ).fetchall()
        has_sync = bool(syncs)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if videos:
            cells = []
            for v in videos:
                first = v["first_video_frame"] or 0
                last = v["last_video_frame"] or 0
                total = max(1, last - first + 1)
                cells.append(CellInfo(
                    shot_video_id=v["id"],
                    file_path=v["file_path"] or "",
                    total_frames=total,
                    fps=float(v["actual_fps"] or 30.0),
                    label=v["cam_label"],
                ))

            cache = FrameCache(conn=None)
            self._scrubber = MultiVideoScrubber(cells, cache)

            # Load sync table into the scrubber so global time slider appears
            if has_sync:
                sync_id = syncs[0]["id"]
                sp_rows = self._conn.execute(
                    "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, cv.actual_fps "
                    "FROM sync_points sp "
                    "JOIN capture_videos cv ON cv.id = sp.shot_video_id "
                    "WHERE sp.sync_config_id = ? ORDER BY sp.shot_video_id, sp.video_frame",
                    (sync_id,),
                ).fetchall()
                if sp_rows:
                    fps_by_video = {r["shot_video_id"]: float(r["actual_fps"] or 30.0)
                                    for r in sp_rows}
                    sync_pts = [
                        SyncPoint(
                            camera_instance_id=r["shot_video_id"],
                            shot_video_id=r["shot_video_id"],
                            video_frame=int(r["video_frame"]),
                            timestamp_s=float(r["timestamp_s"]),
                        )
                        for r in sp_rows
                    ]
                    self._scrubber.reload_sync(SyncTable(sync_pts, fps_by_video))

            root.addWidget(self._scrubber, 1)
        else:
            root.addWidget(QLabel("No videos attached to this capture."), 1)

        # Bottom toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(6, 4, 6, 4)
        toolbar.setSpacing(6)

        self._start_label = QLabel("Start: —")
        mark_start_btn = QPushButton("Mark Start")
        mark_start_btn.setMaximumWidth(90)
        mark_start_btn.clicked.connect(self._mark_start)

        self._end_label = QLabel("End: —")
        mark_end_btn = QPushButton("Mark End")
        mark_end_btn.setMaximumWidth(90)
        mark_end_btn.clicked.connect(self._mark_end)

        sync_btn = QPushButton("Set up sync…")
        sync_btn.clicked.connect(self._open_sync)

        ext_btn = QPushButton("Extrinsics…")
        ext_btn.clicked.connect(self._open_extrinsics)

        self._detect_btn = QPushButton("Detect Pose…")
        self._detect_btn.clicked.connect(self._open_detection_dialog)
        self._detect_btn.setEnabled(has_sync and bool(videos))
        if not has_sync:
            self._detect_btn.setToolTip("Set up sync first")

        toolbar.addWidget(self._start_label)
        toolbar.addWidget(mark_start_btn)
        toolbar.addSpacing(12)
        toolbar.addWidget(self._end_label)
        toolbar.addWidget(mark_end_btn)
        toolbar.addStretch()
        toolbar.addWidget(sync_btn)
        toolbar.addWidget(ext_btn)
        toolbar.addWidget(self._detect_btn)

        root.addLayout(toolbar)

    def _mark_start(self) -> None:
        if self._scrubber is not None:
            self._start_s = self._scrubber.current_timestamp
            if self._start_label is not None:
                self._start_label.setText(f"Start: {self._start_s:.3f} s")

    def _mark_end(self) -> None:
        if self._scrubber is not None:
            self._end_s = self._scrubber.current_timestamp
            if self._end_label is not None:
                self._end_label.setText(f"End: {self._end_s:.3f} s")

    def _open_sync(self) -> None:
        from app.setup.db_context import DBContext
        from app.setup.page_sync import SyncDialog
        session_row = self._conn.execute(
            "SELECT id FROM mocap_sessions LIMIT 1"
        ).fetchone()
        if session_row is None:
            return
        ctx = DBContext(self._conn, session_row["id"])
        dlg = SyncDialog(ctx, self._capture_id, parent=self)
        dlg.exec()
        self._refresh_sync()

    def _refresh_sync(self) -> None:
        from app.setup.db_context import SyncPoint, SyncTable
        syncs = self._conn.execute(
            "SELECT id FROM sync_configs WHERE shot_id = ? ORDER BY rowid",
            (self._capture_id,),
        ).fetchall()
        has_sync = bool(syncs)
        if self._detect_btn is not None:
            self._detect_btn.setEnabled(has_sync and self._scrubber is not None)
        if not has_sync or self._scrubber is None:
            return
        sync_id = syncs[0]["id"]
        sp_rows = self._conn.execute(
            "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, cv.actual_fps "
            "FROM sync_points sp "
            "JOIN capture_videos cv ON cv.id = sp.shot_video_id "
            "WHERE sp.sync_config_id = ? ORDER BY sp.shot_video_id, sp.video_frame",
            (sync_id,),
        ).fetchall()
        if not sp_rows:
            return
        fps_by_video = {r["shot_video_id"]: float(r["actual_fps"] or 30.0) for r in sp_rows}
        sync_pts = [
            SyncPoint(
                camera_instance_id=r["shot_video_id"],
                shot_video_id=r["shot_video_id"],
                video_frame=int(r["video_frame"]),
                timestamp_s=float(r["timestamp_s"]),
            )
            for r in sp_rows
        ]
        self._scrubber.reload_sync(SyncTable(sync_pts, fps_by_video))

    def _open_extrinsics(self) -> None:
        from app.setup.page_extrinsics import ExtrinsicsImportDialog
        session_row = self._conn.execute(
            "SELECT id FROM mocap_sessions LIMIT 1"
        ).fetchone()
        if session_row is None:
            return
        dlg = ExtrinsicsImportDialog(
            self._conn, session_row["id"],
            shot_ids=[self._capture_id],
            parent=self,
        )
        dlg.exec()

    def _open_detection_dialog(self) -> None:
        from app.pose.run_detection_dialog import RunDetectionDialog
        dlg = RunDetectionDialog(
            conn=self._conn,
            session_path=self._session_path,
            capture_id=self._capture_id,
            time_start_s=self._start_s if self._start_s > 0 else None,
            time_end_s=self._end_s if self._end_s > 0 else None,
            parent=self,
        )
        dlg.detection_finished.connect(lambda _tid, _rid: self.data_changed.emit())
        dlg.exec()


# ---------------------------------------------------------------------------
# TrialPanel
# ---------------------------------------------------------------------------


class TrialPanel(QWidget):
    """Stitcher + assignment panel for a trial.

    Shows a compact header (trial name, time range, run selector) then embeds
    StitcherPanel for the selected detection run.  Exposes has_unsaved_changes()
    so the main window can prompt before navigating away.
    """

    data_changed = Signal()  # forwarded from StitcherPanel.applied

    def __init__(
        self,
        conn: sqlite3.Connection,
        trial_id: str,
        preselect_run_id: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._conn = conn
        self._trial_id = trial_id
        self._preselect_run_id = preselect_run_id
        self._stitcher_panel: "StitcherPanel | None" = None
        self._build()

    # ------------------------------------------------------------------

    def has_unsaved_changes(self) -> bool:
        return self._stitcher_panel is not None and self._stitcher_panel.is_dirty

    def save_changes(self) -> bool:
        if self._stitcher_panel is not None:
            return self._stitcher_panel.apply()
        return True

    # ------------------------------------------------------------------

    def _build(self) -> None:
        from app.pose.stitcher_panel import StitcherPanel

        trial = self._conn.execute(
            "SELECT id, name, time_start_s, time_end_s FROM trials WHERE id = ?",
            (self._trial_id,),
        ).fetchone()
        if trial is None:
            self.setLayout(QVBoxLayout())
            self.layout().addWidget(QLabel("Trial not found."))
            return

        runs = self._conn.execute(
            "SELECT id, detector_model, pose_model, status, created_at "
            "FROM detection_runs WHERE trial_id = ? ORDER BY created_at DESC",
            (self._trial_id,),
        ).fetchall()

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        # --- Header row ---
        header = QHBoxLayout()
        title = trial["name"] or "Unnamed trial"
        start_s = trial["time_start_s"]
        end_s = trial["time_end_s"]
        time_str = (
            f"  {_fmt_time(start_s)} – {_fmt_time(end_s)}"
            if start_s is not None and end_s is not None else ""
        )
        header.addWidget(QLabel(f"<b>{title}</b>{time_str}"))
        header.addStretch()

        self._run_combo = QComboBox()
        for r in runs:
            label = (
                f"{r['detector_model']}+{r['pose_model']}"
                f"  {_fmt_ts(r['created_at'])}"
                f"  ({r['status']})"
            )
            self._run_combo.addItem(label, r["id"])
        header.addWidget(QLabel("Run:"))
        header.addWidget(self._run_combo)

        self._seg_init_btn = QPushButton("Segmentation…")
        self._seg_init_btn.setToolTip(
            "Open interactive Cutie segmentation initialisation for this detection run"
        )
        self._seg_init_btn.clicked.connect(self._on_open_seg_init)
        self._seg_init_btn.setEnabled(bool(runs))
        header.addWidget(self._seg_init_btn)

        vbox.addLayout(header)

        # --- Stitcher area (fills remaining space) ---
        self._stitcher_container = QVBoxLayout()
        self._stitcher_container.setContentsMargins(0, 0, 0, 0)
        vbox.addLayout(self._stitcher_container, 1)

        # Pre-select run if specified (e.g. clicked on Detection Run node)
        if self._preselect_run_id:
            idx = self._run_combo.findData(self._preselect_run_id)
            if idx >= 0:
                self._run_combo.setCurrentIndex(idx)

        self._run_combo.currentIndexChanged.connect(self._on_run_changed)
        self._load_stitcher(self._run_combo.currentData())

    def _load_stitcher(self, run_id: str | None) -> None:
        from app.pose.stitcher_panel import StitcherPanel

        # Clear previous panel
        while self._stitcher_container.count():
            item = self._stitcher_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._stitcher_panel = None

        if not run_id:
            self._stitcher_container.addWidget(QLabel("No detection run selected."))
            return

        panel = StitcherPanel(self._conn, run_id, parent=self)
        panel.applied.connect(self.data_changed)
        self._stitcher_panel = panel
        self._stitcher_container.addWidget(panel)

    def _on_run_changed(self, index: int) -> None:
        if self._stitcher_panel is not None and self._stitcher_panel.is_dirty:
            ans = QMessageBox.question(
                self, "Unsaved changes",
                "You have unapplied assignments. Discard and switch run?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return
        self._load_stitcher(self._run_combo.itemData(index))

    def _on_open_seg_init(self) -> None:
        run_id = self._run_combo.currentData()
        if not run_id:
            return
        from app.pose.cutie_init_panel import CutieInitPanel
        win = QWidget(self, Qt.WindowType.Window)
        win.setWindowTitle("Cutie Segmentation Init")
        win.resize(1200, 750)
        layout = QVBoxLayout(win)
        layout.setContentsMargins(0, 0, 0, 0)
        panel = CutieInitPanel(self._conn, run_id, parent=win)
        layout.addWidget(panel)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        win.destroyed.connect(panel.shutdown)
        win.show()


# ---------------------------------------------------------------------------
# DetectionRunPanel
# ---------------------------------------------------------------------------


class DetectionRunPanel(QWidget):
    """Detail view for a detection run."""

    def __init__(self, conn: sqlite3.Connection, run_id: str,
                 session_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._run_id = run_id
        self._session_path = session_path
        self._build()

    def _build(self) -> None:
        run = self._conn.execute(
            "SELECT id, shot_id, sync_config_id, detector_model, pose_model, "
            "       status, time_start_s, time_end_s, created_at, completed_at "
            "FROM detection_runs WHERE id = ?", (self._run_id,)
        ).fetchone()
        if run is None:
            return

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setAlignment(Qt.AlignmentFlag.AlignTop)

        vbox.addWidget(QLabel(f"<h2>Detection [{run['detector_model']}]</h2>"))

        form_box = _section("Run info")
        form = QFormLayout()
        form.addRow("Status:", QLabel(run["status"]))
        form.addRow("Detector:", QLabel(run["detector_model"]))
        form.addRow("Pose model:", QLabel(run["pose_model"] or "—"))
        form.addRow("Time range:", QLabel(
            f"{_fmt_time(run['time_start_s'])}  →  {_fmt_time(run['time_end_s'])}"
        ))
        form.addRow("Started:", QLabel(_fmt_ts(run["created_at"])))
        form.addRow("Completed:", QLabel(_fmt_ts(run["completed_at"])))
        form_box.layout().addLayout(form)
        vbox.addWidget(form_box)

        # Person tracks produced by this run
        tracks = self._conn.execute(
            "SELECT pos.id, GROUP_CONCAT(sp.person_name, ', ') AS names "
            "FROM pose_observation_sequences pos "
            "LEFT JOIN sequence_persons sp ON sp.sequence_id = pos.id "
            "WHERE pos.detection_run_id = ? GROUP BY pos.id",
            (self._run_id,),
        ).fetchall()
        tr_box = _section(f"Person tracks ({len(tracks)})")
        for t in tracks:
            tr_box.layout().addWidget(QLabel(t["names"] or "Unnamed track"))
        if not tracks:
            tr_box.layout().addWidget(QLabel(
                "No person tracks yet — finalise to assign persons."
            ))
        vbox.addWidget(tr_box)

        btn_row = QHBoxLayout()
        open_btn = _action_btn("Open in Pose Extraction…")
        open_btn.clicked.connect(self._open_pose_extraction)
        btn_row.addWidget(open_btn)

        finalise_btn = _action_btn("Finalise → person tracks…",
                                   enabled=(run["status"] == "complete" and not tracks))
        finalise_btn.setToolTip(
            "Open Pose Extraction and use the Finalise workflow there."
        )
        finalise_btn.clicked.connect(self._open_pose_extraction)
        btn_row.addWidget(finalise_btn)

        btn_row.addStretch()
        vbox.addLayout(btn_row)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(_scrollable(inner))

    def _open_pose_extraction(self) -> None:
        from app.pose.main import PoseExtractionWindow
        from app.ui.main_window import MainWindow
        self._pose_win = PoseExtractionWindow(
            session_db=str(self._session_path),
            parent=None,
        )
        main = self.window()
        if isinstance(main, MainWindow):
            self._pose_win.data_changed.connect(main.reload_tree)
        self._pose_win.show()


# ---------------------------------------------------------------------------
# StandaloneRunPanel — stitcher for a detection run not linked to a trial
# ---------------------------------------------------------------------------


class StandaloneRunPanel(QWidget):
    """Stitcher wrapper for a detection run that isn't associated with a trial."""

    data_changed = Signal()

    def __init__(self, conn: sqlite3.Connection, run_id: str, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._run_id = run_id
        self._stitcher_panel = None
        self._build()

    def has_unsaved_changes(self) -> bool:
        return self._stitcher_panel is not None and self._stitcher_panel.is_dirty

    def save_changes(self) -> bool:
        if self._stitcher_panel is not None:
            return self._stitcher_panel.apply()
        return True

    def _build(self) -> None:
        from app.pose.stitcher_panel import StitcherPanel

        run = self._conn.execute(
            "SELECT detector_model, pose_model, created_at FROM detection_runs WHERE id = ?",
            (self._run_id,),
        ).fetchone()

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        if run:
            vbox.addWidget(QLabel(
                f"Detection run: {run['detector_model']}+{run['pose_model']}"
                f"  ({_fmt_ts(run['created_at'])})  — not linked to a trial"
            ))

        panel = StitcherPanel(self._conn, self._run_id, parent=self)
        panel.applied.connect(self.data_changed)
        self._stitcher_panel = panel
        vbox.addWidget(panel, 1)


# ---------------------------------------------------------------------------
# PersonCropGridWidget — multi-camera crop preview for PersonPanel
# ---------------------------------------------------------------------------

# COCO-17 skeleton connections (mirrored from person_preview.py)
_BODY_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _project_point_distorted(
    p_world: "np.ndarray",
    R: "np.ndarray",
    t: "np.ndarray",
    K_orig: "np.ndarray",
    dist: "np.ndarray",
) -> "tuple[float, float] | None":
    """Project a 3D world point to distorted pixel coords using K_original + radtan model.

    Supports both standard (k1,k2,p1,p2,k3) and rational (k1-k6,p1,p2) layouts as
    stored by OpenCV (14-element dist_coeffs with trailing zeros).
    Returns None if the point is behind the camera.
    """
    import numpy as np
    p_cam = R @ p_world + t
    if p_cam[2] <= 1e-3:
        return None
    x = p_cam[0] / p_cam[2]
    y = p_cam[1] / p_cam[2]
    r2 = x * x + y * y
    r4, r6 = r2 * r2, r2 * r2 * r2
    k1, k2 = float(dist[0]), float(dist[1])
    p1, p2 = float(dist[2]), float(dist[3])
    k3 = float(dist[4]) if len(dist) > 4 else 0.0
    k4 = float(dist[5]) if len(dist) > 5 else 0.0
    k5 = float(dist[6]) if len(dist) > 6 else 0.0
    k6 = float(dist[7]) if len(dist) > 7 else 0.0
    numer = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    denom = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    radial = numer / denom if denom != 0.0 else numer
    dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    xd = x * radial + dx
    yd = y * radial + dy
    u = K_orig[0, 0] * xd + K_orig[0, 2]
    v = K_orig[1, 1] * yd + K_orig[1, 2]
    return u, v


def _undistorted_to_distorted(
    u_n: float,
    v_n: float,
    K_new: "np.ndarray",
    K_orig: "np.ndarray",
    dist: "np.ndarray",
) -> "tuple[float, float]":
    """Forward-distort an undistorted (K_new) pixel to distorted (K_orig) pixel coords."""
    import numpy as np
    x = (u_n - K_new[0, 2]) / K_new[0, 0]
    y = (v_n - K_new[1, 2]) / K_new[1, 1]
    r2 = x * x + y * y
    r4, r6 = r2 * r2, r2 * r2 * r2
    k1, k2 = float(dist[0]), float(dist[1])
    p1, p2 = float(dist[2]), float(dist[3])
    k3 = float(dist[4]) if len(dist) > 4 else 0.0
    k4 = float(dist[5]) if len(dist) > 5 else 0.0
    k5 = float(dist[6]) if len(dist) > 6 else 0.0
    k6 = float(dist[7]) if len(dist) > 7 else 0.0
    numer = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    denom = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    radial = numer / denom if denom != 0.0 else numer
    dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    xd = x * radial + dx
    yd = y * radial + dy
    return K_orig[0, 0] * xd + K_orig[0, 2], K_orig[1, 1] * yd + K_orig[1, 2]


def _nearest_tracker_step(t: float, timestamps: list[tuple[float, int]]) -> int:
    """Binary-search timestamps (sorted list of (timestamp_s, tracker_step)) for closest step."""
    import bisect
    ts_vals = [ts for ts, _ in timestamps]
    i = bisect.bisect_left(ts_vals, t)
    if i == 0:
        return timestamps[0][1]
    if i >= len(timestamps):
        return timestamps[-1][1]
    before, after = timestamps[i - 1], timestamps[i]
    return before[1] if abs(before[0] - t) <= abs(after[0] - t) else after[1]


class _ImageCanvas(QWidget):
    """Custom painting widget: image + vector overlays drawn with QPainter.

    All overlay coordinates are stored in full-frame pixel space.  The
    coordinate transform (full-frame → JPEG crop → display) is computed
    at paint time so overlays stay sharp regardless of zoom level.
    """

    def __init__(self, min_h: int = 240, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(min_h)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #222;")
        self._pixmap: QPixmap | None = None
        # Crop-to-full-frame transform
        self._x1: float = 0.0
        self._y1: float = 0.0
        self._src_scale: float = 1.0   # full-frame pixels → JPEG crop pixels
        # Detection overlay: (N, 3) float32 — x, y, conf in full-frame pixels
        self._obs_kp = None
        self._outlier_kp_mask = None           # bool array indexed by COCO keypoint ID
        # Tracking overlay
        self._joint_xy: dict | None = None    # joint_name → [u, v] full-frame
        self._bone_pairs: list = []
        self._marker_xy = None                 # (N, 2) float32 full-frame
        # Visibility flags
        self._show_detected: bool = True
        self._show_tracked: bool = True

    def show_empty(self) -> None:
        self._pixmap = None
        self._obs_kp = None
        self._outlier_kp_mask = None
        self._joint_xy = None
        self._marker_xy = None
        self.update()

    def show_image(self, pixmap: QPixmap, x1: float, y1: float, src_scale: float) -> None:
        self._pixmap = pixmap
        self._x1 = x1
        self._y1 = y1
        self._src_scale = src_scale
        self.update()

    def set_overlay(
        self,
        obs_kp,
        joint_xy,
        bone_pairs: list,
        marker_xy,
        show_detected: bool,
        show_tracked: bool,
        outlier_mask=None,
    ) -> None:
        self._obs_kp = obs_kp
        self._joint_xy = joint_xy
        self._bone_pairs = bone_pairs
        self._marker_xy = marker_xy
        self._show_detected = show_detected
        self._show_tracked = show_tracked
        self._outlier_kp_mask = outlier_mask
        self.update()

    def _image_rect(self) -> tuple[int, int, int, int, float]:
        """Return (off_x, off_y, disp_w, disp_h, disp_scale) for the current pixmap."""
        cw, ch = self.width(), self.height()
        if self._pixmap is None or self._pixmap.width() == 0:
            return 0, 0, cw, ch, 1.0
        px_w, px_h = self._pixmap.width(), self._pixmap.height()
        disp_scale = min(cw / px_w, ch / px_h)
        disp_w = int(px_w * disp_scale)
        disp_h = int(px_h * disp_scale)
        off_x = (cw - disp_w) // 2
        off_y = (ch - disp_h) // 2
        return off_x, off_y, disp_w, disp_h, disp_scale

    def paintEvent(self, _event) -> None:
        from math import isnan

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cw, ch = self.width(), self.height()
        painter.fillRect(0, 0, cw, ch, QColor("#222"))

        if self._pixmap is None:
            painter.setPen(QColor("#666"))
            painter.drawText(QRectF(0, 0, cw, ch), Qt.AlignmentFlag.AlignCenter, "—")
            painter.end()
            return

        off_x, off_y, disp_w, disp_h, disp_scale = self._image_rect()
        scaled_pix = self._pixmap.scaled(
            disp_w, disp_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(off_x, off_y, scaled_pix)

        # Combined scale: full-frame pixels → display pixels
        combined = self._src_scale * disp_scale

        def to_pt(u: float, v: float) -> QPointF:
            return QPointF(
                (u - self._x1) * combined + off_x,
                (v - self._y1) * combined + off_y,
            )

        # ---- Detected keypoints (white connections, colour-coded dots) ----
        if self._show_detected and self._obs_kp is not None:
            kp = self._obs_kp
            n_kp = kp.shape[0]

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(210, 210, 210), 1.0))
            for a, b in _BODY_SKELETON:
                if a >= n_kp or b >= n_kp:
                    continue
                if float(kp[a, 2]) < 0.3 or float(kp[b, 2]) < 0.3:
                    continue
                painter.drawLine(
                    to_pt(float(kp[a, 0]), float(kp[a, 1])),
                    to_pt(float(kp[b, 0]), float(kp[b, 1])),
                )

            painter.setPen(Qt.PenStyle.NoPen)
            for i in range(n_kp):
                conf = float(kp[i, 2])
                if conf < 0.1:
                    continue
                if (self._outlier_kp_mask is not None
                        and i < len(self._outlier_kp_mask)
                        and self._outlier_kp_mask[i]):
                    painter.setBrush(QColor(120, 120, 120))  # grey — rejected outlier
                elif conf >= 0.5:
                    painter.setBrush(QColor(0, 220, 60))    # green
                elif conf >= 0.3:
                    painter.setBrush(QColor(255, 200, 0))   # yellow
                else:
                    painter.setBrush(QColor(220, 40, 40))   # red
                painter.drawEllipse(to_pt(float(kp[i, 0]), float(kp[i, 1])), 3.0, 3.0)

        # ---- Tracked skeleton (cyan lines + yellow dots) ----
        if self._show_tracked:
            if self._joint_xy is not None and self._bone_pairs:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(0, 210, 210), 1.5))  # cyan
                for parent_name, child_name in self._bone_pairs:
                    pxy = self._joint_xy.get(parent_name)
                    cxy = self._joint_xy.get(child_name)
                    if pxy is None or cxy is None:
                        continue
                    if isnan(float(pxy[0])) or isnan(float(cxy[0])):
                        continue
                    painter.drawLine(
                        to_pt(float(pxy[0]), float(pxy[1])),
                        to_pt(float(cxy[0]), float(cxy[1])),
                    )

            if self._marker_xy is not None:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(255, 220, 0))  # yellow
                for i in range(self._marker_xy.shape[0]):
                    mu, mv = float(self._marker_xy[i, 0]), float(self._marker_xy[i, 1])
                    if isnan(mu) or isnan(mv):
                        continue
                    painter.drawEllipse(to_pt(mu, mv), 3.0, 3.0)

        painter.end()


class _CropCell(QWidget):
    """One camera cell in the crop grid: name label + image canvas."""

    _IMG_H = 240

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        name_lbl = QLabel(label)
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet("font-size: 10px; font-weight: bold;")
        name_lbl.setMaximumHeight(18)

        self._canvas = _ImageCanvas(min_h=self._IMG_H)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(1)
        vbox.addWidget(name_lbl)
        vbox.addWidget(self._canvas, stretch=1)

        self.show_empty()

    def show_empty(self) -> None:
        self._canvas.show_empty()

    def show_image(self, pixmap: QPixmap, x1: float, y1: float, src_scale: float) -> None:
        self._canvas.show_image(pixmap, x1, y1, src_scale)

    def set_overlay(
        self,
        obs_kp,
        joint_xy,
        bone_pairs: list,
        marker_xy,
        show_detected: bool,
        show_tracked: bool,
        outlier_mask=None,
    ) -> None:
        self._canvas.set_overlay(
            obs_kp, joint_xy, bone_pairs, marker_xy, show_detected, show_tracked,
            outlier_mask=outlier_mask,
        )


class PersonCropGridWidget(QWidget):
    """Grid of per-camera person crop images with a time scrubber.

    Reads JPEG crops from frame_cache_entries, overlays pose_observations
    keypoints, and shows all cameras simultaneously.  One extra placeholder
    cell is reserved for a future 3D tracking view.
    """

    time_changed = Signal(float)  # emitted on every slider move (absolute timestamp_s)

    def __init__(self, conn: sqlite3.Connection, sequence_id: str, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._sequence_id = sequence_id
        self._cells: list[_CropCell] = []
        self._cameras: list[dict] = []
        self._sync_table = None
        self._det_run_id: str | None = None
        self._t_start: float = 0.0
        self._t_end: float = 0.0
        self._current_t: float = 0.0
        self._slider: QSlider | None = None
        self._time_label: QLabel | None = None
        self._show_detected: QCheckBox | None = None   # green pose-detection keypoints
        self._show_tracked: QCheckBox | None = None    # FK skeleton lines + predicted dots
        # Per-camera track segments: svid → [(track_id, first_frame, last_frame)] sorted by first_frame
        self._track_segs: dict[str, list[tuple[int, int, int]]] = {}
        # Pre-loaded per-camera data (indexed by shot_video_id or camera_instance_id)
        self._obs_kp: dict[str, dict[int, object]] = {}   # cam_instance_id→frame→kp
        self._det_bboxes: dict[str, dict[int, tuple]] = {}  # svid→frame→(cx,cy,w,h)
        # Tracking overlay data
        # _marker_proj: cam_instance_id → tracker_step → (n_markers, 2) predicted pixel positions
        # _joint_proj:  cam_instance_id → tracker_step → {joint_name: [u, v]}
        self._marker_proj: dict[str, dict[int, object]] = {}
        self._joint_proj: dict[str, dict[int, dict]] = {}
        self._bone_pairs: list[tuple[str, str]] = []
        self._tracking_timestamps: list[tuple[float, int]] = []  # sorted (ts, step)
        # cam_instance_id → tracker_step → bool array indexed by COCO keypoint ID
        self._outlier_masks: dict[str, dict[int, object]] = {}
        self._build()

    def _build(self) -> None:
        import numpy as np
        from app.setup.db_context import SyncPoint, SyncTable

        seq = self._conn.execute(
            "SELECT detection_run_id, shot_id, sync_config_id, time_start_s, time_end_s "
            "FROM pose_observation_sequences WHERE id = ?",
            (self._sequence_id,),
        ).fetchone()
        if seq is None:
            QVBoxLayout(self).addWidget(QLabel("Sequence not found."))
            return

        self._det_run_id = seq["detection_run_id"]
        self._t_start = float(seq["time_start_s"])
        self._t_end = float(seq["time_end_s"])

        prow = self._conn.execute(
            "SELECT person_name FROM sequence_persons WHERE sequence_id = ? AND person_id = 0",
            (self._sequence_id,),
        ).fetchone()
        person_name = prow["person_name"] if prow else None

        # Collect ALL (track_id, first_frame, last_frame) segments per camera for this person.
        # A person may be assigned to multiple tracks in the same camera (e.g. after a track
        # split), so we must not overwrite — store them as an ordered list and look up
        # the correct track by frame range at display time.
        self._track_segs = {}
        if person_name and self._det_run_id:
            for r in self._conn.execute(
                "SELECT shot_video_id, track_id, first_frame, last_frame "
                "FROM detection_track_assignments "
                "WHERE detection_run_id = ? AND person_name = ? "
                "ORDER BY first_frame",
                (self._det_run_id, person_name),
            ):
                self._track_segs.setdefault(r["shot_video_id"], []).append(
                    (r["track_id"], r["first_frame"], r["last_frame"])
                )

        cam_rows = self._conn.execute(
            "SELECT cv.id, cv.camera_instance_id, cv.file_path, "
            "       COALESCE(ci.label, cv.camera_instance_id) AS label "
            "FROM capture_videos cv "
            "LEFT JOIN camera_instances ci ON ci.id = cv.camera_instance_id "
            "WHERE cv.shot_id = ? ORDER BY label",
            (seq["shot_id"],),
        ).fetchall()

        sp_rows = self._conn.execute(
            "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, cv.actual_fps "
            "FROM sync_points sp "
            "JOIN capture_videos cv ON cv.id = sp.shot_video_id "
            "WHERE sp.sync_config_id = ?",
            (seq["sync_config_id"],),
        ).fetchall()
        if sp_rows:
            points = [
                SyncPoint(
                    camera_instance_id="",
                    shot_video_id=r["shot_video_id"],
                    video_frame=r["video_frame"],
                    timestamp_s=r["timestamp_s"],
                )
                for r in sp_rows
            ]
            fps_by_video = {r["shot_video_id"]: float(r["actual_fps"]) for r in sp_rows}
            self._sync_table = SyncTable(points, fps_by_video)

        self._cameras = [
            {
                "shot_video_id": r["id"],
                "camera_instance_id": r["camera_instance_id"],
                "label": r["label"],
                "file_path": r["file_path"] or "",
            }
            for r in cam_rows
        ]

        # Pre-load pose_observations keypoints: camera_instance_id → frame → kp
        for r in self._conn.execute(
            "SELECT camera_instance_id, video_frame, kp_blob "
            "FROM pose_observations WHERE sequence_id = ? AND person_id = 0",
            (self._sequence_id,),
        ):
            raw = bytes(r["kp_blob"])
            n = len(raw) // 12
            kp = np.frombuffer(raw, dtype=np.float32).reshape(n, 3)
            self._obs_kp.setdefault(r["camera_instance_id"], {})[r["video_frame"]] = kp

        # Pre-load detection bboxes: svid → frame → (cx, cy, w, h)
        # Load for every track_id that is assigned to the person in each camera.
        if self._det_run_id:
            for svid, segs in self._track_segs.items():
                for track_id, _first, _last in segs:
                    for r in self._conn.execute(
                        "SELECT video_frame, bbox_x, bbox_y, bbox_w, bbox_h "
                        "FROM person_detections "
                        "WHERE detection_run_id=? AND shot_video_id=? AND track_id=? "
                        "AND region_type='full_body'",
                        (self._det_run_id, svid, track_id),
                    ):
                        self._det_bboxes.setdefault(svid, {})[r["video_frame"]] = (
                            r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]
                        )

        n_cells = len(self._cameras) + 1  # +1 for 3D placeholder
        ncols = max(2, min(n_cells, 4))
        self._ncols = ncols

        grid = QGridLayout()
        grid.setSpacing(4)
        self._grid = grid

        for i, cam in enumerate(self._cameras):
            row, col = divmod(i, ncols)
            cell = _CropCell(cam["label"])
            self._cells.append(cell)
            grid.addWidget(cell, row, col)
            grid.setColumnStretch(col, 1)

        r3d, c3d = divmod(len(self._cameras), ncols)
        ph = QLabel("3D view\n(coming soon)")
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet("color: #888; border: 1px dashed #555;")
        ph.setMinimumHeight(_CropCell._IMG_H)
        grid.addWidget(ph, r3d, c3d)
        grid.setColumnStretch(c3d, 1)
        self._3d_ph = ph

        nrows = ceil(n_cells / ncols)
        for r in range(nrows):
            grid.setRowStretch(r, 1)

        dur_ms = max(1, int((self._t_end - self._t_start) * 1000))
        _fps_vals = [float(r["actual_fps"]) for r in sp_rows if r["actual_fps"]]
        frame_step_ms = max(1, round(1000.0 / max(_fps_vals))) if _fps_vals else 8
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(dur_ms)
        self._slider.setSingleStep(frame_step_ms)
        self._slider.setValue(0)
        self._slider.valueChanged.connect(self._on_slider)

        self._time_label = QLabel(_fmt_time(self._t_start))
        self._time_label.setMinimumWidth(70)

        slider_row = QHBoxLayout()
        slider_row.addWidget(self._slider)
        slider_row.addWidget(self._time_label)

        self._show_detected = QCheckBox("Detected keypoints")
        self._show_detected.setChecked(True)
        self._show_detected.stateChanged.connect(lambda _: self._load_frame(self._current_t))
        self._show_tracked = QCheckBox("Tracked skeleton")
        self._show_tracked.setChecked(True)
        self._show_tracked.stateChanged.connect(lambda _: self._load_frame(self._current_t))

        overlay_row = QHBoxLayout()
        overlay_row.addWidget(QLabel("Show:"))
        overlay_row.addWidget(self._show_detected)
        overlay_row.addWidget(self._show_tracked)
        overlay_row.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(grid, stretch=1)
        layout.addLayout(overlay_row)
        layout.addLayout(slider_row)

        self._current_t = self._t_start
        self._load_frame(self._t_start)

    def _on_slider(self, value: int) -> None:
        self._current_t = self._t_start + value / 1000.0
        if self._time_label is not None:
            self._time_label.setText(_fmt_time(self._current_t))
        self._load_frame(self._current_t)
        self.time_changed.emit(self._current_t)

    def seek(self, t: float) -> None:
        """Seek to an absolute timestamp without re-emitting time_changed."""
        if self._slider is None:
            return
        ms = int((t - self._t_start) * 1000)
        ms = max(0, min(ms, self._slider.maximum()))
        self._slider.blockSignals(True)
        self._slider.setValue(ms)
        self._slider.blockSignals(False)
        self._current_t = t
        if self._time_label is not None:
            self._time_label.setText(_fmt_time(t))
        self._load_frame(t)

    def set_tracking_run(self, run_id: str | None) -> None:
        """Load tracking run overlay data; called by PersonPanel when run selection changes."""
        self._load_tracking_run(run_id)

    def camera_labels(self) -> list[str]:
        return [c["label"] for c in self._cameras]

    def set_camera_filter(self, label: str | None) -> None:
        """Show only the camera with the given label; pass None to show all."""
        visible_cols: set[int] = set()
        for i, cam in enumerate(self._cameras):
            show = label is None or cam["label"] == label
            self._cells[i].setVisible(show)
            if show:
                visible_cols.add(i % self._ncols)
        show_3d = label is None
        self._3d_ph.setVisible(show_3d)
        if show_3d:
            visible_cols.add(len(self._cameras) % self._ncols)
        for col in range(self._ncols):
            self._grid.setColumnStretch(col, 1 if col in visible_cols else 0)
            self._grid.setColumnMinimumWidth(col, 0)

    def _track_id_at_frame(self, svid: str, frame_idx: int) -> int | None:
        """Return the track_id assigned to the person at *frame_idx* in camera *svid*.

        Searches _track_segs for a segment whose [first_frame, last_frame] range
        covers frame_idx.  Returns None if no assignment exists for that frame.
        """
        for track_id, first_frame, last_frame in self._track_segs.get(svid, []):
            if first_frame <= frame_idx <= last_frame:
                return track_id
        return None

    def _load_tracking_run(self, run_id: str | None) -> None:
        import json
        import numpy as np
        from posetrak.db.skeleton_layout import SkeletonLayout

        self._marker_proj.clear()
        self._joint_proj.clear()
        self._bone_pairs.clear()
        self._tracking_timestamps.clear()
        self._outlier_masks.clear()

        if not run_id:
            self._load_frame(self._current_t)
            return

        run = self._conn.execute(
            "SELECT active_camera_ids, marker_names, skeleton_id, "
            "       extrinsic_calibration_id, observation_sequence_id "
            "FROM tracking_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        if run is None:
            self._load_frame(self._current_t)
            return

        cam_labels: list[str] = json.loads(run["active_camera_ids"] or "[]")
        marker_names: list[str] = json.loads(run["marker_names"] or "[]")
        n_cams, n_markers = len(cam_labels), len(marker_names)
        if n_cams == 0 or n_markers == 0:
            self._load_frame(self._current_t)
            return

        # Map camera label → camera_instance_id
        placeholders = ",".join("?" * n_cams)
        label_to_cam_id: dict[str, str] = {}
        for r in self._conn.execute(
            f"SELECT id, label FROM camera_instances WHERE label IN ({placeholders})",
            cam_labels,
        ):
            label_to_cam_id[r["label"]] = r["id"]

        # Load timestamps for nearest-step lookup
        ts_rows = self._conn.execute(
            "SELECT tracker_step, timestamp_s FROM tracking_results "
            "WHERE run_id=? AND person_id=0 AND is_smoothed=0 ORDER BY tracker_step",
            (run_id,),
        ).fetchall()
        step_to_ts = {r["tracker_step"]: r["timestamp_s"] for r in ts_rows}
        self._tracking_timestamps = sorted(
            (ts, step) for step, ts in step_to_ts.items()
        )

        # Load skeleton → bone pairs
        skel = self._conn.execute(
            "SELECT yaml_content FROM skeletons WHERE id=?",
            (run["skeleton_id"],),
        ).fetchone()
        if not skel or not skel["yaml_content"]:
            self._load_frame(self._current_t)
            return
        layout = SkeletonLayout(skel["yaml_content"])
        self._bone_pairs = layout.bone_pairs

        # marker index → COCO keypoint index (for outlier colouring)
        marker_to_coco = {
            m["name"]: m["openpose_keypoint"]
            for m in layout.markers
            if m["openpose_keypoint"] is not None
        }
        mi_to_coco: dict[int, int] = {
            mi: marker_to_coco[name]
            for mi, name in enumerate(marker_names)
            if name in marker_to_coco
        }
        n_coco_kp = max((c + 1 for c in mi_to_coco.values()), default=17)

        # Load extrinsics: cam_instance_id → (R 3×3, t 3-vector)
        ext_id = run["extrinsic_calibration_id"]
        cam_extrinsics: dict[str, tuple] = {}
        if ext_id:
            for r in self._conn.execute(
                "SELECT ee.camera_instance_id, ee.R, ee.t "
                "FROM extrinsic_entries ee "
                "WHERE ee.extrinsic_calibration_id = ?",
                (ext_id,),
            ):
                R = np.frombuffer(bytes(r["R"]), dtype="<f8").reshape(3, 3)
                t = np.frombuffer(bytes(r["t"]), dtype="<f8")
                cam_extrinsics[r["camera_instance_id"]] = (R, t)

        # Load intrinsics: cam_instance_id → {fx, fy, cx, cy}
        seq = self._conn.execute(
            "SELECT shot_id FROM pose_observation_sequences WHERE id=?",
            (run["observation_sequence_id"],),
        ).fetchone()
        cam_intrinsics: dict[str, dict] = {}
        if seq:
            for r in self._conn.execute(
                "SELECT cv.camera_instance_id, ic.fx, ic.fy, ic.cx, ic.cy, "
                "       ic.dist_coeffs, ic.matrix_original "
                "FROM capture_videos cv "
                "JOIN intrinsics_calibrations ic ON ic.id = cv.intrinsics_calibration_id "
                "WHERE cv.shot_id = ?",
                (seq["shot_id"],),
            ):
                K_orig = None
                dist = None
                if r["matrix_original"]:
                    K_orig = np.frombuffer(bytes(r["matrix_original"]), dtype="<f8").reshape(3, 3)
                if r["dist_coeffs"]:
                    dist = np.frombuffer(bytes(r["dist_coeffs"]), dtype="<f8")
                cam_intrinsics[r["camera_instance_id"]] = {
                    "fx": r["fx"], "fy": r["fy"], "cx": r["cx"], "cy": r["cy"],
                    "K_orig": K_orig, "dist": dist,
                }

        # Load obs blobs for marker dots (predicted positions from tracker output)
        obs_rows = self._conn.execute(
            "SELECT tracker_step, obs_blob FROM tracking_obs_results "
            "WHERE run_id=? AND person_id=0 ORDER BY tracker_step",
            (run_id,),
        ).fetchall()
        expected_obs = n_cams * n_markers * 8
        for obs_row in obs_rows:
            step = obs_row["tracker_step"]
            blob = np.frombuffer(bytes(obs_row["obs_blob"]), dtype="<f4")
            if len(blob) != expected_obs:
                continue
            obs = blob.reshape(n_cams, n_markers, 8)
            for ci, label in enumerate(cam_labels):
                cam_id = label_to_cam_id.get(label)
                if cam_id is None:
                    continue
                pred_xy = obs[ci, :, 2:4].copy()  # undistorted predicted positions
                intr = cam_intrinsics.get(cam_id)
                if intr is not None:
                    K_orig = intr.get("K_orig")
                    dist = intr.get("dist")
                    if K_orig is not None and dist is not None:
                        fx_n = intr["fx"]; fy_n = intr["fy"]
                        cx_n = intr["cx"]; cy_n = intr["cy"]
                        K_new = np.array([[fx_n, 0, cx_n], [0, fy_n, cy_n], [0, 0, 1]])
                        for mi in range(pred_xy.shape[0]):
                            u_n, v_n = float(pred_xy[mi, 0]), float(pred_xy[mi, 1])
                            if np.isfinite(u_n) and np.isfinite(v_n):
                                u_d, v_d = _undistorted_to_distorted(u_n, v_n, K_new, K_orig, dist)
                                pred_xy[mi, 0] = u_d
                                pred_xy[mi, 1] = v_d
                self._marker_proj.setdefault(cam_id, {})[step] = pred_xy

                if mi_to_coco:
                    mask = np.zeros(n_coco_kp, dtype=bool)
                    for mi, coco_id in mi_to_coco.items():
                        is_out = obs[ci, mi, 6]
                        if np.isfinite(is_out) and is_out != 0.0:
                            mask[coco_id] = True
                    self._outlier_masks.setdefault(cam_id, {})[step] = mask

        # Compute joint projections via FK from state blobs
        state_rows = self._conn.execute(
            "SELECT tracker_step, state FROM tracking_results "
            "WHERE run_id=? AND person_id=0 AND is_smoothed=0 ORDER BY tracker_step",
            (run_id,),
        ).fetchall()
        for state_row in state_rows:
            step = state_row["tracker_step"]
            try:
                decoded = layout.decode_state_blob(bytes(state_row["state"]))
                transforms = layout.compute_joint_transforms(decoded)
            except Exception:
                continue
            for label in cam_labels:
                cam_id = label_to_cam_id.get(label)
                if cam_id is None:
                    continue
                ext = cam_extrinsics.get(cam_id)
                intr = cam_intrinsics.get(cam_id)
                if ext is None or intr is None:
                    continue
                R, t = ext
                K_orig = intr.get("K_orig")
                dist = intr.get("dist")
                fx, fy, cx_k, cy_k = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
                use_distortion = K_orig is not None and dist is not None
                joint_xy: dict[str, np.ndarray] = {}
                for jname, T in transforms.items():
                    p_world = T[:3, 3]
                    if use_distortion:
                        uv = _project_point_distorted(p_world, R, t, K_orig, dist)
                        if uv is None:
                            continue
                        u, v = uv
                    else:
                        p_cam = R @ p_world + t
                        if p_cam[2] <= 1e-3:
                            continue
                        u = fx * p_cam[0] / p_cam[2] + cx_k
                        v = fy * p_cam[1] / p_cam[2] + cy_k
                    joint_xy[jname] = np.array([u, v])
                self._joint_proj.setdefault(cam_id, {})[step] = joint_xy

        self._load_frame(self._current_t)

    def _load_frame(self, global_time: float) -> None:
        import cv2
        import numpy as np

        if not self._det_run_id or not self._sync_table:
            for cell in self._cells:
                cell.show_empty()
            return

        show_detected = self._show_detected is None or self._show_detected.isChecked()
        show_tracked = self._show_tracked is None or self._show_tracked.isChecked()

        tracking_step: int | None = None
        if self._tracking_timestamps:
            tracking_step = _nearest_tracker_step(global_time, self._tracking_timestamps)

        for i, cam in enumerate(self._cameras):
            svid = cam["shot_video_id"]
            cam_id = cam["camera_instance_id"]
            cell = self._cells[i]

            frame_idx = self._sync_table.lookup(global_time, svid)
            if frame_idx is None:
                cell.show_empty()
                continue

            # Find which track covers this frame for this camera.
            track_id = self._track_id_at_frame(svid, frame_idx)
            if track_id is None:
                cell.show_empty()
                continue

            row = self._conn.execute(
                "SELECT image_data, height_px, src_x, src_y, src_w, src_h "
                "FROM frame_cache_entries "
                "WHERE shot_video_id=? AND cache_type='person_crop' AND track_id=? "
                "AND region_type='full_body' AND detection_run_id=? "
                "AND frame_idx BETWEEN ? AND ? "
                "ORDER BY ABS(frame_idx - ?) LIMIT 1",
                (svid, track_id, self._det_run_id,
                 frame_idx - 3, frame_idx + 3, frame_idx),
            ).fetchone()

            if row is None:
                cell.show_empty()
                continue

            buf = np.frombuffer(bytes(row["image_data"]), dtype=np.uint8)
            crop_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

            if crop_bgr is None:
                cell.show_empty()
                continue

            # Compute crop-to-full-frame transform for overlay coordinate mapping.
            bbox = self._det_bboxes.get(svid, {}).get(frame_idx)
            if row["src_x"] is not None:
                x1, y1 = float(row["src_x"]), float(row["src_y"])
                src_h = float(row["src_h"])
            elif bbox is not None:
                cx, cy, bw, bh = bbox
                x1, y1 = cx - bw / 2, cy - bh / 2
                src_h = bh
            else:
                x1, y1 = 0.0, 0.0
                src_h = float(crop_bgr.shape[0])
            jpeg_h = float(row["height_px"] or crop_bgr.shape[0])
            src_scale = jpeg_h / src_h if src_h > 0 else 1.0

            # Convert image to QPixmap (no overlays drawn into the image)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            h_img, w_img = crop_rgb.shape[:2]
            qimg = QImage(
                crop_rgb.data, w_img, h_img, 3 * w_img, QImage.Format.Format_RGB888
            )
            cell.show_image(QPixmap.fromImage(qimg), x1, y1, src_scale)

            # Pass overlay data — canvas draws everything at display resolution
            obs_kp = self._obs_kp.get(cam_id, {}).get(frame_idx)
            joint_xy = (
                self._joint_proj.get(cam_id, {}).get(tracking_step)
                if tracking_step is not None else None
            )
            marker_xy = (
                self._marker_proj.get(cam_id, {}).get(tracking_step)
                if tracking_step is not None else None
            )
            outlier_mask = (
                self._outlier_masks.get(cam_id, {}).get(tracking_step)
                if tracking_step is not None else None
            )
            cell.set_overlay(
                obs_kp=obs_kp,
                joint_xy=joint_xy,
                bone_pairs=self._bone_pairs,
                marker_xy=marker_xy,
                outlier_mask=outlier_mask,
                show_detected=show_detected,
                show_tracked=show_tracked,
            )


# ---------------------------------------------------------------------------
# _LineChart — small metric time-series widget with a cursor line
# ---------------------------------------------------------------------------


class _LineChart(QWidget):
    """Compact line chart for a single metric over tracker steps.

    Renders a data line and an optional vertical cursor marking the current
    step.  Y axis auto-scales to the data range; no axes or tick labels are
    drawn to keep the widget compact.
    """

    _PAD_L = 4
    _PAD_R = 4
    _PAD_T = 18   # space for title text
    _PAD_B = 4

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self._steps: list[int] = []
        self._values: list[float | None] = []
        self._cursor_step: int | None = None
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_data(self, steps: list[int], values: list[float | None]) -> None:
        self._steps = steps
        self._values = values
        self._cursor_step = None
        self.update()

    def set_cursor(self, step: int | None) -> None:
        if step != self._cursor_step:
            self._cursor_step = step
            self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        pl, pr, pt, pb = self._PAD_L, self._PAD_R, self._PAD_T, self._PAD_B

        bg = self.palette().color(self.backgroundRole())
        painter.fillRect(0, 0, w, h, bg)

        text_color = self.palette().color(self.foregroundRole())
        painter.setPen(text_color)
        painter.drawText(pl, pt - 4, self._title)

        vals = [v for v in self._values if v is not None and v == v]  # skip None/NaN
        if not self._steps or not vals:
            return

        y_min, y_max = min(vals), max(vals)
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        x_min, x_max = self._steps[0], self._steps[-1]
        if x_min == x_max:
            x_max += 1

        cw = w - pl - pr
        ch = h - pt - pb

        def sx(step: int) -> float:
            return pl + (step - x_min) / (x_max - x_min) * cw

        def sy(val: float) -> float:
            return pt + ch - (val - y_min) / (y_max - y_min) * ch

        # Data line
        path = QPainterPath()
        started = False
        for step, val in zip(self._steps, self._values):
            if val is None or val != val:
                started = False
                continue
            x, y = sx(step), sy(val)
            if not started:
                path.moveTo(x, y)
                started = True
            else:
                path.lineTo(x, y)
        painter.setPen(QPen(QColor(60, 140, 240), 1.5))
        painter.drawPath(path)

        # Cursor
        if self._cursor_step is not None and x_min <= self._cursor_step <= x_max:
            cx = sx(self._cursor_step)
            painter.setPen(QPen(QColor(255, 120, 0), 1.5))
            painter.drawLine(QPointF(cx, pt), QPointF(cx, h - pb))


# ---------------------------------------------------------------------------
# _RunInfoPane — right-side collapsible info pane
# ---------------------------------------------------------------------------


class _RunInfoPane(QWidget):
    """Collapsible right-side pane showing metadata and per-frame stats for a
    tracking run.

    Top section:  static run metadata (skeleton, person, date, config params).
    Middle section: per-frame live stats updated as the scrubber moves.
    Bottom:        NIS/DOF and covariance condition number charts.
    """

    _MIN_WIDTH = 260

    def __init__(self, conn: sqlite3.Connection, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._tracking_timestamps: list[tuple[float, int]] = []
        self._step_stats: dict[int, sqlite3.Row] = {}   # step → tracking_results row
        self._nis_dof_steps: list[int] = []
        self._nis_dof_vals: list[float | None] = []
        self._cov_steps: list[int] = []
        self._cov_vals: list[float | None] = []
        self.setMinimumWidth(self._MIN_WIDTH)
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # --- Run info ---
        run_box = QGroupBox("Run info")
        run_form = QFormLayout(run_box)
        run_form.setHorizontalSpacing(6)
        run_form.setVerticalSpacing(2)
        self._ri_skeleton = QLabel("—")
        self._ri_person = QLabel("—")
        self._ri_ran_at = QLabel("—")
        self._ri_frames = QLabel("—")
        self._ri_cfg = QLabel("—")
        self._ri_cfg.setWordWrap(True)
        run_form.addRow("Skeleton:", self._ri_skeleton)
        run_form.addRow("Person:", self._ri_person)
        run_form.addRow("Tracked at:", self._ri_ran_at)
        run_form.addRow("Frames:", self._ri_frames)
        run_form.addRow("Config:", self._ri_cfg)
        root.addWidget(run_box)

        # --- Current frame ---
        frame_box = QGroupBox("Current frame")
        frame_form = QFormLayout(frame_box)
        frame_form.setHorizontalSpacing(6)
        frame_form.setVerticalSpacing(2)
        self._fi_step = QLabel("—")
        self._fi_time = QLabel("—")
        self._fi_inliers = QLabel("—")
        self._fi_nis = QLabel("—")
        self._fi_cov = QLabel("—")
        frame_form.addRow("Step:", self._fi_step)
        frame_form.addRow("Time:", self._fi_time)
        frame_form.addRow("Inliers:", self._fi_inliers)
        frame_form.addRow("NIS / DOF:", self._fi_nis)
        frame_form.addRow("Cov cond #:", self._fi_cov)
        root.addWidget(frame_box)

        # --- Charts ---
        charts_box = QGroupBox("Metrics")
        charts_v = QVBoxLayout(charts_box)
        charts_v.setSpacing(4)
        self._nis_chart = _LineChart("NIS / DOF")
        self._cov_chart = _LineChart("Covariance condition #")
        charts_v.addWidget(self._nis_chart)
        charts_v.addWidget(self._cov_chart)
        root.addWidget(charts_box)

        root.addStretch(1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_run(self, run_id: str | None) -> None:
        """Populate static run info and load chart time-series data."""
        self._tracking_timestamps.clear()
        self._step_stats.clear()
        self._nis_dof_steps.clear()
        self._nis_dof_vals.clear()
        self._cov_steps.clear()
        self._cov_vals.clear()

        # Clear labels
        for lbl in (self._ri_skeleton, self._ri_person, self._ri_ran_at,
                    self._ri_frames, self._ri_cfg):
            lbl.setText("—")
        for lbl in (self._fi_step, self._fi_time, self._fi_inliers,
                    self._fi_nis, self._fi_cov):
            lbl.setText("—")
        self._nis_chart.set_data([], [])
        self._cov_chart.set_data([], [])

        if not run_id:
            return

        run = self._conn.execute(
            "SELECT tr.ran_at, tr.posetrak_version, tr.tracker_config_id, "
            "       s.name AS skel_name, "
            "       (SELECT GROUP_CONCAT(sp.person_name, ', ') "
            "        FROM sequence_persons sp "
            "        WHERE sp.sequence_id = tr.observation_sequence_id) AS person_names "
            "FROM tracking_runs tr "
            "LEFT JOIN skeletons s ON s.id = tr.skeleton_id "
            "WHERE tr.id = ?",
            (run_id,),
        ).fetchone()
        if not run:
            return

        n_frames = self._conn.execute(
            "SELECT COUNT(*) FROM tracking_results "
            "WHERE run_id=? AND person_id=0 AND is_smoothed=0",
            (run_id,),
        ).fetchone()[0]

        self._ri_skeleton.setText(run["skel_name"] or "—")
        self._ri_person.setText(run["person_names"] or "—")
        self._ri_ran_at.setText(_fmt_ts(run["ran_at"]))
        self._ri_frames.setText(str(n_frames))

        # Try to load tracker config params
        cfg_id = run["tracker_config_id"]
        cfg = self._conn.execute(
            "SELECT name, process_noise_std, process_noise_vel_std, "
            "       measurement_noise_std, outlier_threshold, tracker_fps "
            "FROM tracker_configs WHERE id=?",
            (cfg_id,),
        ).fetchone() if cfg_id else None
        if cfg:
            parts = [cfg["name"] or cfg_id[:8]]
            if cfg["process_noise_std"] is not None:
                parts.append(f"Q={cfg['process_noise_std']}")
            if cfg["measurement_noise_std"] is not None:
                parts.append(f"R={cfg['measurement_noise_std']}")
            if cfg["outlier_threshold"] is not None:
                parts.append(f"thr={cfg['outlier_threshold']}")
            self._ri_cfg.setText("  ".join(parts))
        else:
            self._ri_cfg.setText(cfg_id[:12] + "…" if cfg_id else "—")

        # Load per-frame stats
        rows = self._conn.execute(
            "SELECT tracker_step, timestamp_s, n_inlier_observations, "
            "       nis_value, nis_dof, cov_condition_number "
            "FROM tracking_results "
            "WHERE run_id=? AND person_id=0 AND is_smoothed=0 "
            "ORDER BY tracker_step",
            (run_id,),
        ).fetchall()

        for row in rows:
            step = row["tracker_step"]
            self._tracking_timestamps.append((row["timestamp_s"], step))
            self._step_stats[step] = row

            nis = row["nis_value"]
            dof = row["nis_dof"]
            nis_norm = (nis / dof) if (nis is not None and dof and dof > 0) else None
            self._nis_dof_steps.append(step)
            self._nis_dof_vals.append(nis_norm)

            self._cov_steps.append(step)
            self._cov_vals.append(row["cov_condition_number"])

        self._nis_chart.set_data(self._nis_dof_steps, self._nis_dof_vals)
        self._cov_chart.set_data(self._cov_steps, self._cov_vals)

    def on_time_changed(self, t: float) -> None:
        """Update current-frame labels and chart cursors for timestamp *t*."""
        if not self._tracking_timestamps:
            return
        step = _nearest_tracker_step(t, self._tracking_timestamps)
        row = self._step_stats.get(step)
        if row is None:
            return

        self._fi_step.setText(str(step))
        self._fi_time.setText(f"{row['timestamp_s']:.3f} s")

        n_in = row["n_inlier_observations"]
        self._fi_inliers.setText(str(n_in) if n_in is not None else "—")

        nis = row["nis_value"]
        dof = row["nis_dof"]
        if nis is not None and dof:
            self._fi_nis.setText(f"{nis:.2f} / {dof}  ({nis/dof:.2f} norm)")
        else:
            self._fi_nis.setText("—")

        cov = row["cov_condition_number"]
        self._fi_cov.setText(f"{cov:.1f}" if cov is not None else "—")

        self._nis_chart.set_cursor(step)
        self._cov_chart.set_cursor(step)


# ---------------------------------------------------------------------------
# PersonPanel
# ---------------------------------------------------------------------------


class PersonPanel(QWidget):
    """Person panel: info, tracking history, and tracker launcher."""

    _bvh_export_done = Signal(str, str)  # (out_path, error — empty string = success)
    _usd_export_done = Signal(str, str)  # (out_path, error — empty string = success)

    def __init__(self, conn: sqlite3.Connection, sequence_id: str,
                 session_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._sequence_id = sequence_id
        self._session_path = session_path
        self._bvh_export_done.connect(self._on_bvh_done)
        self._usd_export_done.connect(self._on_usd_done)
        self._crop_grid: PersonCropGridWidget | None = None
        self._info_pane: _RunInfoPane | None = None
        self._build()

    def _build(self) -> None:
        seq = self._conn.execute(
            "SELECT id, name, time_start_s, time_end_s, pose_model, notes "
            "FROM pose_observation_sequences WHERE id = ?",
            (self._sequence_id,),
        ).fetchone()
        if seq is None:
            return

        person_names = self._conn.execute(
            "SELECT GROUP_CONCAT(person_name, ', ') AS names "
            "FROM sequence_persons WHERE sequence_id = ?",
            (self._sequence_id,),
        ).fetchone()["names"]

        n_obs = self._conn.execute(
            "SELECT COUNT(DISTINCT video_frame || camera_instance_id) "
            "FROM pose_observations WHERE sequence_id = ?",
            (self._sequence_id,),
        ).fetchone()[0]

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = person_names or seq["name"] or "Person"
        vbox.addWidget(QLabel(f"<h2>{title}</h2>"))

        form_box = _section("Person info")
        form = QFormLayout()
        form.addRow("Persons:", QLabel(person_names or "—"))
        form.addRow("Time range:", QLabel(
            f"{_fmt_time(seq['time_start_s'])}  →  {_fmt_time(seq['time_end_s'])}"
        ))
        form.addRow("Observations:", QLabel(str(n_obs)))
        form.addRow("Pose model:", QLabel(seq["pose_model"] or "—"))
        form_box.layout().addLayout(form)
        vbox.addWidget(form_box)

        # --- Tracking runs section ---
        self._run_box = _section("Tracking runs (0)")
        box_vbox = self._run_box.layout()

        self._run_list = QListWidget()
        self._run_list.setMaximumHeight(110)
        self._run_list.currentItemChanged.connect(self._on_run_selected)
        box_vbox.addWidget(self._run_list)

        self._run_detail = QLabel("")
        self._run_detail.setWordWrap(True)
        self._run_detail.setVisible(False)
        box_vbox.addWidget(self._run_detail)

        run_act_row = QHBoxLayout()
        run_act_row.setContentsMargins(0, 2, 0, 0)
        self._export_bvh_btn = QPushButton("Export BVH…")
        self._export_bvh_btn.setEnabled(False)
        self._export_bvh_btn.clicked.connect(self._export_bvh)
        self._export_usd_btn = QPushButton("Export USD…")
        self._export_usd_btn.setEnabled(False)
        if not _USD_AVAILABLE:
            self._export_usd_btn.setToolTip(_USD_TOOLTIP)
        self._export_usd_btn.clicked.connect(self._export_usd)
        self._delete_run_btn = QPushButton("Delete run")
        self._delete_run_btn.setEnabled(False)
        self._delete_run_btn.clicked.connect(self._delete_run)
        self._scale_btn = QPushButton("Scale skeleton…")
        self._scale_btn.setEnabled(False)
        self._scale_btn.setToolTip(
            "Measure bone lengths from inlier observations and scale the skeleton"
        )
        self._scale_btn.clicked.connect(self._open_scaling)
        self._info_toggle_btn = QPushButton("Info")
        self._info_toggle_btn.setCheckable(True)
        self._info_toggle_btn.setChecked(False)
        self._info_toggle_btn.setToolTip("Show / hide run info pane  (I)")
        self._info_toggle_btn.toggled.connect(self._toggle_info_pane)
        run_act_row.addStretch()
        run_act_row.addWidget(self._scale_btn)
        run_act_row.addWidget(self._info_toggle_btn)
        run_act_row.addWidget(self._export_bvh_btn)
        run_act_row.addWidget(self._export_usd_btn)
        run_act_row.addWidget(self._delete_run_btn)
        box_vbox.addLayout(run_act_row)

        vbox.addWidget(self._run_box)

        self._refresh_runs()

        btn_row = QHBoxLayout()
        run_btn = _action_btn("Run tracker…")
        run_btn.clicked.connect(self._open_run_tracker)
        btn_row.addWidget(run_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        scroll = _scrollable(inner)
        scroll.setMaximumHeight(320)

        self._crop_grid = PersonCropGridWidget(self._conn, self._sequence_id)
        self._info_pane = _RunInfoPane(self._conn)
        self._info_pane.setVisible(False)
        self._crop_grid.time_changed.connect(self._info_pane.on_time_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._crop_grid)
        splitter.addWidget(self._info_pane)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([700, 280])

        # Keyboard shortcut: I toggles the info pane
        info_shortcut = QShortcut(QKeySequence("I"), self)
        info_shortcut.activated.connect(self._info_toggle_btn.toggle)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        root.addWidget(scroll)
        root.addWidget(splitter, stretch=1)

        # Auto-select the most recent tracking run so the overlay loads immediately
        if self._run_list.count() > 0 and self._run_list.item(0).data(
            Qt.ItemDataRole.UserRole
        ):
            self._run_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    # Tracking runs list
    # ------------------------------------------------------------------

    def _refresh_runs(self) -> None:
        self._run_list.clear()
        self._run_detail.setVisible(False)
        self._export_bvh_btn.setEnabled(False)
        self._export_usd_btn.setEnabled(False)
        self._delete_run_btn.setEnabled(False)
        self._scale_btn.setEnabled(False)

        runs = self._conn.execute(
            "SELECT tr.id, tr.ran_at, s.name AS skel_name "
            "FROM tracking_runs tr "
            "LEFT JOIN skeletons s ON s.id = tr.skeleton_id "
            "WHERE tr.observation_sequence_id = ? ORDER BY tr.ran_at DESC",
            (self._sequence_id,),
        ).fetchall()
        self._run_box.setTitle(f"Tracking runs ({len(runs)})")
        if not runs:
            self._run_list.addItem("No tracking runs yet.")
            return
        for r in runs:
            stats = self._conn.execute(
                "SELECT COUNT(*) AS total, "
                "       SUM(CASE WHEN tracking_lost=0 THEN 1 ELSE 0 END) AS tracked "
                "FROM tracking_results WHERE run_id=? AND person_id=0 AND is_smoothed=0",
                (r["id"],),
            ).fetchone()
            label = f"[{r['skel_name'] or '?'}]  {_fmt_ts(r['ran_at'])}"
            if stats and stats["total"]:
                pct = 100.0 * (stats["tracked"] or 0) / stats["total"]
                label += f"  —  {stats['tracked']}/{stats['total']} frames ({pct:.0f}%)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, r["id"])
            self._run_list.addItem(item)

    def _on_run_selected(self, current: QListWidgetItem, _prev) -> None:
        run_id = current.data(Qt.ItemDataRole.UserRole) if current else None
        if not run_id:
            self._run_detail.setVisible(False)
            self._export_bvh_btn.setEnabled(False)
            self._export_usd_btn.setEnabled(False)
            self._delete_run_btn.setEnabled(False)
            self._scale_btn.setEnabled(False)
            if self._crop_grid is not None:
                self._crop_grid.set_tracking_run(None)
            if self._info_pane is not None:
                self._info_pane.load_run(None)
            return

        run = self._conn.execute(
            "SELECT tr.ran_at, tr.notes, s.name AS skel_name "
            "FROM tracking_runs tr "
            "LEFT JOIN skeletons s ON s.id = tr.skeleton_id "
            "WHERE tr.id = ?",
            (run_id,),
        ).fetchone()
        stats = self._conn.execute(
            "SELECT COUNT(*) AS total, "
            "       SUM(CASE WHEN tracking_lost=0 THEN 1 ELSE 0 END) AS tracked, "
            "       AVG(COALESCE(n_inlier_observations, 0)) AS avg_inliers "
            "FROM tracking_results WHERE run_id=? AND person_id=0 AND is_smoothed=0",
            (run_id,),
        ).fetchone()

        if run:
            if stats and stats["total"]:
                total = stats["total"]
                tracked = stats["tracked"] or 0
                pct = 100.0 * tracked / total
                avg = stats["avg_inliers"] or 0.0
                stat_line = f"{tracked}/{total} frames ({pct:.1f}%)  —  avg inliers: {avg:.1f}"
            else:
                stat_line = "no frame stats"
            self._run_detail.setText(
                f"<b>{run['skel_name'] or '?'}</b>  {_fmt_ts(run['ran_at'])}<br>"
                f"Frames: {stat_line}<br>"
                f"Notes: {run['notes'] or '—'}"
            )
            self._run_detail.setVisible(True)

        self._export_bvh_btn.setEnabled(True)
        self._export_usd_btn.setEnabled(_USD_AVAILABLE)
        self._delete_run_btn.setEnabled(True)
        self._scale_btn.setEnabled(bool(self._session_path))

        if self._crop_grid is not None:
            self._crop_grid.set_tracking_run(run_id)
        if self._info_pane is not None:
            self._info_pane.load_run(run_id)

    def _toggle_info_pane(self, checked: bool) -> None:
        if self._info_pane is not None:
            self._info_pane.setVisible(checked)

    # ------------------------------------------------------------------
    # BVH export
    # ------------------------------------------------------------------

    def _export_bvh(self) -> None:
        item = self._run_list.currentItem()
        run_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not run_id:
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save BVH file", "", "BVH files (*.bvh)"
        )
        if not out_path:
            return
        if not out_path.endswith(".bvh"):
            out_path += ".bvh"

        self._export_bvh_btn.setEnabled(False)

        from posetrak.export.bvh import export_bvh

        def _run() -> None:
            error = ""
            try:
                export_bvh(
                    out_path,
                    session_db=str(self._session_path),
                    run_id=run_id,
                    person_id=0,
                    smoothed=True,
                )
            except Exception as exc:
                error = str(exc)
            self._bvh_export_done.emit(out_path, error)

        threading.Thread(target=_run, daemon=True).start()

    def _on_bvh_done(self, out_path: str, error: str) -> None:
        self._export_bvh_btn.setEnabled(True)
        if error:
            QMessageBox.critical(self, "Export failed", f"BVH export failed:\n\n{error}")
        else:
            QMessageBox.information(self, "Export complete", f"BVH written to:\n{out_path}")

    def _export_usd(self) -> None:
        item = self._run_list.currentItem()
        run_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not run_id:
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save USD file", "",
            "USD files (*.usda *.usdc);;USDA ASCII (*.usda);;USDC binary (*.usdc)"
        )
        if not out_path:
            return
        if not (out_path.endswith(".usda") or out_path.endswith(".usdc")):
            out_path += ".usda"

        self._export_usd_btn.setEnabled(False)

        from posetrak.export.usd import export_usd

        def _run() -> None:
            error = ""
            try:
                export_usd(
                    out_path,
                    session_db=str(self._session_path),
                    run_id=run_id,
                    person_id=0,
                    smoothed=True,
                )
            except Exception as exc:
                error = str(exc)
            self._usd_export_done.emit(out_path, error)

        threading.Thread(target=_run, daemon=True).start()

    def _on_usd_done(self, out_path: str, error: str) -> None:
        self._export_usd_btn.setEnabled(True)
        if error:
            QMessageBox.critical(self, "Export failed", f"USD export failed:\n\n{error}")
        else:
            QMessageBox.information(self, "Export complete", f"USD written to:\n{out_path}")

    # ------------------------------------------------------------------
    # Delete run
    # ------------------------------------------------------------------

    def _delete_run(self) -> None:
        item = self._run_list.currentItem()
        run_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not run_id:
            return
        if QMessageBox.question(
            self,
            "Delete tracking run",
            "Delete this tracking run and all its results?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        with self._conn:
            self._conn.execute(
                "DELETE FROM tracking_results WHERE run_id = ?", (run_id,)
            )
            self._conn.execute(
                "DELETE FROM tracking_obs_results WHERE run_id = ?", (run_id,)
            )
            self._conn.execute(
                "DELETE FROM tracking_runs WHERE id = ?", (run_id,)
            )
        self._refresh_runs()

    # ------------------------------------------------------------------
    # Tracker dialog
    # ------------------------------------------------------------------

    def _open_run_tracker(self) -> None:
        from app.pose.run_tracker import RunTrackerDialog
        dlg = RunTrackerDialog(
            conn=self._conn,
            session_path=str(self._session_path),
            sequence_id=self._sequence_id,
            parent=self,
        )
        dlg.exec()
        self._refresh_runs()

    def _open_scaling(self) -> None:
        item = self._run_list.currentItem()
        run_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not run_id or not self._session_path:
            return
        from app.ui.skeleton_scaling_panel import SkeletonScalingPanel
        dlg = SkeletonScalingPanel(
            self._conn,
            str(self._session_path),
            run_id,
            parent=self,
        )
        dlg.show()


# ---------------------------------------------------------------------------
# TrackingRunPanel
# ---------------------------------------------------------------------------


class TrackingRunPanel(QWidget):
    """Detail view for a tracking run."""

    _export_done = Signal(str, str, str)  # (fmt, out_path, error — empty = success)

    def __init__(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        session_path: Path | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._conn = conn
        self._run_id = run_id
        self._session_path = session_path
        self._export_bvh_btn: QPushButton | None = None
        self._export_usd_btn: QPushButton | None = None
        self._export_done.connect(self._on_export_done)
        self._build()

    def _build(self) -> None:
        run = self._conn.execute(
            "SELECT tr.id, tr.ran_at, tr.notes, tr.posetrak_version, "
            "       tr.active_camera_ids, tr.marker_names, "
            "       tr.observation_sequence_id, "
            "       s.name AS skel_name "
            "FROM tracking_runs tr "
            "LEFT JOIN skeletons s ON s.id = tr.skeleton_id "
            "WHERE tr.id = ?",
            (self._run_id,),
        ).fetchone()
        if run is None:
            return

        n_frames = self._conn.execute(
            "SELECT COUNT(*) FROM tracking_results WHERE run_id = ? AND is_smoothed = 0",
            (self._run_id,),
        ).fetchone()[0]

        skel = run["skel_name"] or "?"

        # Compact header + buttons (non-scrolling top strip)
        header = QWidget()
        header_v = QVBoxLayout(header)
        header_v.setContentsMargins(6, 4, 6, 2)
        header_v.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel(f"<b>Tracking run</b> [{skel}]"))
        title_row.addStretch()

        form = QFormLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(1)
        form.addRow("Ran at:", QLabel(_fmt_ts(run["ran_at"])))
        form.addRow("Frames:", QLabel(f"{n_frames}  |  version {run['posetrak_version'] or '—'}"))
        try:
            cam_ids = json.loads(run["active_camera_ids"] or "[]")
            form.addRow("Cameras:", QLabel(", ".join(cam_ids) or "—"))
        except Exception:
            pass
        if run["notes"]:
            form.addRow("Notes:", QLabel(run["notes"]))

        btn_row = QHBoxLayout()
        self._export_bvh_btn = _action_btn("Export BVH…", enabled=bool(self._session_path))
        self._export_bvh_btn.clicked.connect(self._export_bvh)
        btn_row.addWidget(self._export_bvh_btn)
        self._export_usd_btn = _action_btn(
            "Export USD…", enabled=bool(self._session_path) and _USD_AVAILABLE
        )
        if not _USD_AVAILABLE:
            self._export_usd_btn.setToolTip(_USD_TOOLTIP)
        self._export_usd_btn.clicked.connect(self._export_usd)
        btn_row.addWidget(self._export_usd_btn)
        scale_btn = _action_btn("Scale skeleton…", enabled=bool(self._session_path))
        scale_btn.setToolTip("Measure bone lengths from inlier observations and scale the skeleton")
        scale_btn.clicked.connect(self._open_scaling)
        btn_row.addWidget(scale_btn)
        btn_row.addStretch()

        header_v.addLayout(title_row)
        header_v.addLayout(form)
        header_v.addLayout(btn_row)

        # Video crop grid fills the rest of the panel
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(header)

        seq_id = run["observation_sequence_id"]
        if seq_id:
            crop_grid = PersonCropGridWidget(self._conn, seq_id)
            crop_grid.set_tracking_run(self._run_id)
            root.addWidget(crop_grid, stretch=1)
        else:
            root.addStretch(1)

    def _export_bvh(self) -> None:
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save BVH file", "", "BVH files (*.bvh)"
        )
        if not out_path:
            return
        if not out_path.endswith(".bvh"):
            out_path += ".bvh"
        if self._export_bvh_btn:
            self._export_bvh_btn.setEnabled(False)
        from posetrak.export.bvh import export_bvh
        run_id = self._run_id
        session_db = str(self._session_path)

        def _run() -> None:
            error = ""
            try:
                export_bvh(out_path, session_db=session_db, run_id=run_id,
                           person_id=0, smoothed=True)
            except Exception as exc:
                error = str(exc)
            self._export_done.emit("BVH", out_path, error)

        threading.Thread(target=_run, daemon=True).start()

    def _export_usd(self) -> None:
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save USD file", "",
            "USD files (*.usda *.usdc);;USDA ASCII (*.usda);;USDC binary (*.usdc)"
        )
        if not out_path:
            return
        if not (out_path.endswith(".usda") or out_path.endswith(".usdc")):
            out_path += ".usda"
        if self._export_usd_btn:
            self._export_usd_btn.setEnabled(False)
        from posetrak.export.usd import export_usd
        run_id = self._run_id
        session_db = str(self._session_path)

        def _run() -> None:
            error = ""
            try:
                export_usd(out_path, session_db=session_db, run_id=run_id,
                           person_id=0, smoothed=True)
            except Exception as exc:
                error = str(exc)
            self._export_done.emit("USD", out_path, error)

        threading.Thread(target=_run, daemon=True).start()

    def _on_export_done(self, fmt: str, out_path: str, error: str) -> None:
        if fmt == "BVH" and self._export_bvh_btn:
            self._export_bvh_btn.setEnabled(True)
        elif fmt == "USD" and self._export_usd_btn:
            self._export_usd_btn.setEnabled(True)
        if error:
            QMessageBox.critical(self, "Export failed",
                                 f"{fmt} export failed:\n\n{error}")
        else:
            QMessageBox.information(self, "Export complete",
                                    f"{fmt} written to:\n{out_path}")

    def _open_scaling(self) -> None:
        from app.ui.skeleton_scaling_panel import SkeletonScalingPanel

        dlg = SkeletonScalingPanel(
            self._conn,
            str(self._session_path),
            self._run_id,
            parent=self,
        )
        dlg.show()

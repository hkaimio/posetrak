"""content_panels.py — Right-pane detail panels for each tree item type."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from math import ceil

from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
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
    QVBoxLayout,
    QWidget,
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
                # Revert combo to the current panel's run
                run_id = self._run_combo.findData(
                    self._run_combo.currentData()
                )
                return
        self._load_stitcher(self._run_combo.itemData(index))


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


def _draw_skeleton_lines(
    img,
    joint_xy: dict,     # joint_name → np.ndarray([u, v]) in full-frame coordinates
    bone_pairs: list,   # list of (parent_name, child_name)
    x1: float, y1: float, scale: float,
) -> None:
    import cv2, math
    color = (0, 210, 210)  # cyan

    def to_crop(px: float, py: float) -> tuple[int, int]:
        return (int((px - x1) * scale), int((py - y1) * scale))

    for parent_name, child_name in bone_pairs:
        pxy = joint_xy.get(parent_name)
        cxy = joint_xy.get(child_name)
        if pxy is None or cxy is None:
            continue
        px, py = float(pxy[0]), float(pxy[1])
        cx, cy = float(cxy[0]), float(cxy[1])
        if math.isnan(px) or math.isnan(cx):
            continue
        cv2.line(img, to_crop(px, py), to_crop(cx, cy), color, 1, cv2.LINE_AA)


def _draw_marker_dots(
    img,
    marker_xy,          # np.ndarray (n_markers, 2) full-frame predicted positions
    x1: float, y1: float, scale: float,
) -> None:
    import cv2, math
    color = (0, 210, 210)  # cyan
    n = marker_xy.shape[0]

    def to_crop(px: float, py: float) -> tuple[int, int]:
        return (int((px - x1) * scale), int((py - y1) * scale))

    for i in range(n):
        px, py = float(marker_xy[i, 0]), float(marker_xy[i, 1])
        if math.isnan(px) or math.isnan(py):
            continue
        cv2.circle(img, to_crop(px, py), 3, color, -1, cv2.LINE_AA)


class _CropCell(QWidget):
    """One camera cell in the crop grid: name label + image."""

    _IMG_H = 240

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        name_lbl = QLabel(label)
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet("font-size: 10px; font-weight: bold;")
        name_lbl.setMaximumHeight(18)

        self._img = QLabel()
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setMinimumHeight(self._IMG_H)
        self._img.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._img.setStyleSheet("background: #222;")

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(1)
        vbox.addWidget(name_lbl)
        vbox.addWidget(self._img, stretch=1)

        self.show_empty()

    def show_empty(self) -> None:
        self._img.clear()
        self._img.setText("—")
        self._img.setStyleSheet("background: #222; color: #666;")

    def show_bgr(self, bgr) -> None:
        """Display a BGR numpy array, scaling to fit the cell."""
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self._img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._img.setPixmap(scaled)
        self._img.setStyleSheet("background: #222;")


class PersonCropGridWidget(QWidget):
    """Grid of per-camera person crop images with a time scrubber.

    Reads JPEG crops from frame_cache_entries, overlays pose_observations
    keypoints, and shows all cameras simultaneously.  One extra placeholder
    cell is reserved for a future 3D tracking view.
    """

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
        self._show_skeleton: QCheckBox | None = None
        self._show_keypoints: QCheckBox | None = None
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

        track_by_svid: dict[str, int] = {}
        if person_name and self._det_run_id:
            for r in self._conn.execute(
                "SELECT shot_video_id, track_id FROM detection_track_assignments "
                "WHERE detection_run_id = ? AND person_name = ?",
                (self._det_run_id, person_name),
            ):
                track_by_svid[r["shot_video_id"]] = r["track_id"]

        cam_rows = self._conn.execute(
            "SELECT cv.id, cv.camera_instance_id, "
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
                "track_id": track_by_svid.get(r["id"]),
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
        if self._det_run_id:
            for cam in self._cameras:
                svid = cam["shot_video_id"]
                tid = cam["track_id"]
                if tid is None:
                    continue
                for r in self._conn.execute(
                    "SELECT video_frame, bbox_x, bbox_y, bbox_w, bbox_h "
                    "FROM person_detections "
                    "WHERE detection_run_id=? AND shot_video_id=? AND track_id=? "
                    "AND region_type='full_body'",
                    (self._det_run_id, svid, tid),
                ):
                    self._det_bboxes.setdefault(svid, {})[r["video_frame"]] = (
                        r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]
                    )

        n_cells = len(self._cameras) + 1  # +1 for 3D placeholder
        ncols = max(2, min(n_cells, 4))

        grid = QGridLayout()
        grid.setSpacing(4)

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

        nrows = ceil(n_cells / ncols)
        for r in range(nrows):
            grid.setRowStretch(r, 1)

        dur_ms = max(1, int((self._t_end - self._t_start) * 1000))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(dur_ms)
        self._slider.setValue(0)
        self._slider.valueChanged.connect(self._on_slider)

        self._time_label = QLabel(_fmt_time(self._t_start))
        self._time_label.setMinimumWidth(70)

        slider_row = QHBoxLayout()
        slider_row.addWidget(self._slider)
        slider_row.addWidget(self._time_label)

        self._show_skeleton = QCheckBox("Skeleton")
        self._show_skeleton.setChecked(True)
        self._show_skeleton.stateChanged.connect(lambda _: self._load_frame(self._current_t))
        self._show_keypoints = QCheckBox("Keypoints")
        self._show_keypoints.setChecked(True)
        self._show_keypoints.stateChanged.connect(lambda _: self._load_frame(self._current_t))

        overlay_row = QHBoxLayout()
        overlay_row.addWidget(QLabel("Tracking overlay:"))
        overlay_row.addWidget(self._show_skeleton)
        overlay_row.addWidget(self._show_keypoints)
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

    def set_tracking_run(self, run_id: str | None) -> None:
        """Load tracking run overlay data; called by PersonPanel when run selection changes."""
        self._load_tracking_run(run_id)

    def _load_tracking_run(self, run_id: str | None) -> None:
        import json
        import numpy as np
        from posetrak.db.skeleton_layout import SkeletonLayout

        self._marker_proj.clear()
        self._joint_proj.clear()
        self._bone_pairs.clear()
        self._tracking_timestamps.clear()

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
                "SELECT cv.camera_instance_id, ic.fx, ic.fy, ic.cx, ic.cy "
                "FROM capture_videos cv "
                "JOIN intrinsics_calibrations ic ON ic.id = cv.intrinsics_calibration_id "
                "WHERE cv.shot_id = ?",
                (seq["shot_id"],),
            ):
                cam_intrinsics[r["camera_instance_id"]] = {
                    "fx": r["fx"], "fy": r["fy"], "cx": r["cx"], "cy": r["cy"],
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
                pred_xy = obs[ci, :, 2:4].copy()  # columns 2:4 = predicted 2D positions
                self._marker_proj.setdefault(cam_id, {})[step] = pred_xy

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
                fx, fy, cx_k, cy_k = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
                joint_xy: dict[str, np.ndarray] = {}
                for jname, T in transforms.items():
                    p_cam = R @ T[:3, 3] + t
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
        from app.pose.person_preview import draw_skeleton_on_crop

        if not self._det_run_id or not self._sync_table:
            for cell in self._cells:
                cell.show_empty()
            return

        for i, cam in enumerate(self._cameras):
            svid = cam["shot_video_id"]
            cam_id = cam["camera_instance_id"]
            track_id = cam["track_id"]
            cell = self._cells[i]

            if track_id is None:
                cell.show_empty()
                continue
            frame_idx = self._sync_table.lookup(global_time, svid)
            if frame_idx is None:
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

            # Resolve crop-to-frame transform: prefer stored src_* (exact),
            # fall back to bbox (old thumbnails without margin stored).
            bbox = self._det_bboxes.get(svid, {}).get(frame_idx)
            if row["src_x"] is not None:
                x1, y1 = float(row["src_x"]), float(row["src_y"])
                src_h = float(row["src_h"])
            elif bbox is not None:
                cx, cy, bw, bh = bbox
                x1, y1 = cx - bw / 2, cy - bh / 2
                src_h = bh
            else:
                x1 = y1 = 0.0
                src_h = float(crop_bgr.shape[0])
            jpeg_h = float(row["height_px"] or crop_bgr.shape[0])
            scale = jpeg_h / src_h if src_h > 0 else 1.0

            # Overlay pose_observations keypoints
            kp = self._obs_kp.get(cam_id, {}).get(frame_idx)
            if kp is not None:
                kp_s = kp.copy()
                kp_s[:, 0] = kp[:, 0] * scale
                kp_s[:, 1] = kp[:, 1] * scale
                draw_skeleton_on_crop(crop_bgr, kp_s, int(x1 * scale), int(y1 * scale))

            # Overlay tracking solution (skeleton lines + marker dots)
            if self._tracking_timestamps:
                step = _nearest_tracker_step(global_time, self._tracking_timestamps)
                if self._show_skeleton and self._show_skeleton.isChecked():
                    joint_xy = self._joint_proj.get(cam_id, {}).get(step)
                    if joint_xy is not None:
                        _draw_skeleton_lines(
                            crop_bgr, joint_xy, self._bone_pairs, x1, y1, scale
                        )
                if self._show_keypoints and self._show_keypoints.isChecked():
                    marker_xy = self._marker_proj.get(cam_id, {}).get(step)
                    if marker_xy is not None:
                        _draw_marker_dots(crop_bgr, marker_xy, x1, y1, scale)

            cell.show_bgr(crop_bgr)


# ---------------------------------------------------------------------------
# PersonPanel
# ---------------------------------------------------------------------------


_EXPORT_BVH_SCRIPT = Path(__file__).resolve().parents[3] / "python" / "tools" / "export_bvh.py"


class PersonPanel(QWidget):
    """Person panel: info, tracking history, and tracker launcher."""

    def __init__(self, conn: sqlite3.Connection, sequence_id: str,
                 session_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._sequence_id = sequence_id
        self._session_path = session_path
        self._bvh_proc: QProcess | None = None
        self._crop_grid: PersonCropGridWidget | None = None
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
        self._delete_run_btn = QPushButton("Delete run")
        self._delete_run_btn.setEnabled(False)
        self._delete_run_btn.clicked.connect(self._delete_run)
        run_act_row.addStretch()
        run_act_row.addWidget(self._export_bvh_btn)
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

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        root.addWidget(scroll)
        root.addWidget(self._crop_grid, stretch=1)

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
        self._delete_run_btn.setEnabled(False)

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
            self._delete_run_btn.setEnabled(False)
            if self._crop_grid is not None:
                self._crop_grid.set_tracking_run(None)
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
        self._delete_run_btn.setEnabled(True)

        if self._crop_grid is not None:
            self._crop_grid.set_tracking_run(run_id)

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

        self._bvh_proc = QProcess(self)
        self._bvh_proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._export_bvh_btn.setEnabled(False)

        proc = self._bvh_proc

        def _done(code: int, _status) -> None:
            self._export_bvh_btn.setEnabled(True)
            if code == 0:
                QMessageBox.information(
                    self, "Export complete", f"BVH written to:\n{out_path}"
                )
            else:
                output = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
                QMessageBox.critical(
                    self, "Export failed",
                    f"export_bvh.py exited with code {code}.\n\n{output[-800:]}",
                )

        self._bvh_proc.finished.connect(_done)
        self._bvh_proc.start(
            sys.executable,
            [
                str(_EXPORT_BVH_SCRIPT),
                "--session-db", str(self._session_path),
                "--run-id",     run_id,
                "--person-id",  "0",
                "--smoothed",
                "--output",     out_path,
            ],
        )

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


# ---------------------------------------------------------------------------
# TrackingRunPanel
# ---------------------------------------------------------------------------


class TrackingRunPanel(QWidget):
    """Detail view for a tracking run."""

    def __init__(self, conn: sqlite3.Connection, run_id: str, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._run_id = run_id
        self._build()

    def _build(self) -> None:
        run = self._conn.execute(
            "SELECT tr.id, tr.ran_at, tr.notes, tr.posetrak_version, "
            "       tr.active_camera_ids, tr.marker_names, "
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

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setAlignment(Qt.AlignmentFlag.AlignTop)

        skel = run["skel_name"] or "?"
        vbox.addWidget(QLabel(f"<h2>Tracking run  [{skel}]</h2>"))

        form_box = _section("Run info")
        form = QFormLayout()
        form.addRow("Skeleton:", QLabel(skel))
        form.addRow("Ran at:", QLabel(_fmt_ts(run["ran_at"])))
        form.addRow("Version:", QLabel(run["posetrak_version"] or "—"))
        form.addRow("Frames:", QLabel(str(n_frames)))
        try:
            cam_ids = json.loads(run["active_camera_ids"] or "[]")
            form.addRow("Cameras:", QLabel(", ".join(cam_ids) or "—"))
        except Exception:
            pass
        form.addRow("Notes:", QLabel(run["notes"] or "—"))
        form_box.layout().addLayout(form)
        vbox.addWidget(form_box)

        btn_row = QHBoxLayout()
        view_btn = _action_btn("View results…", enabled=False)
        view_btn.setToolTip("Phase 5: results visualiser")
        btn_row.addWidget(view_btn)

        export_btn = _action_btn("Export BVH…", enabled=False)
        export_btn.setToolTip("Not yet wired in this UI")
        btn_row.addWidget(export_btn)

        btn_row.addStretch()
        vbox.addLayout(btn_row)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(_scrollable(inner))

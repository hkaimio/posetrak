"""content_panels.py — Right-pane detail panels for each tree item type."""

from __future__ import annotations

import collections
import json
import logging
import sqlite3
import sys
import threading
from pathlib import Path

_log = logging.getLogger(__name__)

from math import ceil, isfinite, log10

from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
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

# Target pixel width per camera cell — used to auto-compute column count on resize.
# Portrait-aspect ROIs need narrower cells; 220 gives 3 cols on ~700 px panels.
_TARGET_CELL_W = 220


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _CollapsibleBox(QWidget):
    """Section widget with a clickable title bar that toggles visibility of content."""

    def __init__(self, title: str, expanded: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._btn = QToolButton()
        self._btn.setCheckable(True)
        self._btn.setChecked(expanded)
        self._btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._btn.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._btn.setStyleSheet("QToolButton { border: none; font-weight: bold; padding: 2px; }")
        self._btn.setText(title)
        self._btn.toggled.connect(self._on_toggle)

        self._body = QWidget()
        self._inner = QVBoxLayout(self._body)
        self._inner.setContentsMargins(4, 2, 4, 4)
        self._inner.setSpacing(2)
        self._body.setVisible(expanded)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 2, 0, 2)
        root.setSpacing(0)
        root.addWidget(self._btn)
        root.addWidget(self._body)

    def _on_toggle(self, checked: bool) -> None:
        self._body.setVisible(checked)
        self._btn.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def inner_layout(self) -> QVBoxLayout:
        return self._inner

    def setTitle(self, title: str) -> None:
        self._btn.setText(title)


def _section(title: str, expanded: bool = True) -> _CollapsibleBox:
    return _CollapsibleBox(title, expanded=expanded)


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


def _id_row_widget(full_id: str | None) -> QWidget:
    """A compact widget showing an 8-char ID prefix with a clipboard copy button.

    The button is always created (so _set_id_widget can show/hide it later),
    but starts hidden when full_id is None.
    """
    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(2)
    lbl = QLabel(full_id[:8] + "…" if full_id else "—")
    lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    lbl.setStyleSheet("font-family: monospace; font-size: 10px;")
    if full_id:
        lbl.setToolTip(full_id)
    row.addWidget(lbl)
    btn = QToolButton()
    btn.setText("⎘")
    btn.setToolTip("Copy full ID")
    btn.setFixedSize(18, 18)
    btn.setStyleSheet("font-size: 10px; padding: 0;")
    btn.setVisible(bool(full_id))
    if full_id:
        btn.clicked.connect(lambda checked=False, v=full_id: QApplication.clipboard().setText(v))
    row.addWidget(btn)
    row.addStretch()
    w._lbl = lbl
    w._btn = btn
    return w


def _set_id_widget(w: QWidget, full_id: str | None, extra_tooltip: str = "") -> None:
    """Update an _id_row_widget in place with a new ID value."""
    lbl: QLabel = w._lbl
    btn: QToolButton = w._btn
    if full_id:
        lbl.setText(full_id[:8] + "…")
        tip = full_id + (f"\n{extra_tooltip}" if extra_tooltip else "")
        lbl.setToolTip(tip)
        try:
            btn.clicked.disconnect()
        except RuntimeError:
            pass
        btn.clicked.connect(lambda checked=False, v=full_id: QApplication.clipboard().setText(v))
        btn.setVisible(True)
    else:
        lbl.setText("—")
        lbl.setToolTip("")
        btn.setVisible(False)


def _build_run_ids_group() -> tuple["_CollapsibleBox", dict]:
    """Build a collapsible IDs section and return it with the id_row_widgets dict."""
    box = _CollapsibleBox("IDs", expanded=False)  # collapsed by default — rarely needed
    form = QFormLayout()
    form.setHorizontalSpacing(6)
    form.setVerticalSpacing(1)
    widgets: dict[str, QWidget] = {
        "run":       _id_row_widget(None),
        "skeleton":  _id_row_widget(None),
        "detection": _id_row_widget(None),
        "trial":     _id_row_widget(None),
        "capture":   _id_row_widget(None),
    }
    form.addRow("Run:", widgets["run"])
    form.addRow("Skeleton:", widgets["skeleton"])
    form.addRow("Detection:", widgets["detection"])
    form.addRow("Trial:", widgets["trial"])
    form.addRow("Capture:", widgets["capture"])
    box.inner_layout().addLayout(form)
    return box, widgets


def _populate_run_ids(
    widgets: dict[str, QWidget],
    run: sqlite3.Row,
) -> None:
    """Fill ID widgets from a row that has run_id, skeleton_id, detection_run_id,
    trial_id, trial_name, capture_id, capture_label columns."""
    _set_id_widget(widgets["run"],       run["run_id"])
    _set_id_widget(widgets["skeleton"],  run["skeleton_id"])
    _set_id_widget(widgets["detection"], run["detection_run_id"])
    _set_id_widget(widgets["trial"],     run["trial_id"],
                   extra_tooltip=run["trial_name"] or "")
    _set_id_widget(widgets["capture"],   run["capture_id"],
                   extra_tooltip=run["capture_label"] or "")


def _frame_identifier_text(
    db_path: "str | None",
    run: sqlite3.Row,
    step: "int | None",
    timestamp_s: "float | None",
) -> str:
    """Multi-line, copy-pasteable identifier for one exact frame of one run --
    db path + run/trial/capture IDs + tracker step/timestamp. One button,
    one clipboard paste, instead of separately copying each UUID and reading
    the step/time off the sidebar by hand.
    """
    lines = [f"db: {db_path or '?'}", f"run: {run['run_id']}"]
    trial_id = run["trial_id"]
    if trial_id:
        name = run["trial_name"]
        lines.append(f"trial: {trial_id}" + (f" ({name})" if name else ""))
    capture_id = run["capture_id"]
    if capture_id:
        label = run["capture_label"]
        lines.append(f"capture: {capture_id}" + (f" ({label})" if label else ""))
    if step is not None:
        lines.append(f"step: {step}")
    if timestamp_s is not None:
        lines.append(f"time_s: {timestamp_s:.3f}")
    return "\n".join(lines)


def _db_path_of(conn: sqlite3.Connection) -> "str | None":
    row = conn.execute("PRAGMA database_list").fetchone()
    return row[2] if row else None


_RUN_INFO_SQL = (
    "SELECT tr.id AS run_id, tr.ran_at, tr.posetrak_version, "
    "       tr.tracker_config_id, tr.skeleton_id, tr.notes, "
    "       tr.observation_sequence_id, tr.active_camera_ids, "
    "       s.name AS skel_name, "
    "       (SELECT GROUP_CONCAT(sp.person_name, ', ') "
    "        FROM sequence_persons sp "
    "        WHERE sp.sequence_id = tr.observation_sequence_id) AS person_names, "
    "       dr.id AS detection_run_id, "
    "       t.id AS trial_id, t.name AS trial_name, "
    "       cap.id AS capture_id, cap.label AS capture_label "
    "FROM tracking_runs tr "
    "LEFT JOIN skeletons s ON s.id = tr.skeleton_id "
    "LEFT JOIN pose_observation_sequences pos "
    "       ON pos.id = tr.observation_sequence_id "
    "LEFT JOIN detection_runs dr ON dr.id = pos.detection_run_id "
    "LEFT JOIN trials t ON t.id = dr.trial_id "
    "LEFT JOIN captures cap ON cap.id = t.capture_id "
    "WHERE tr.id = ?"
)

_CFG_SQL = (
    "SELECT name, process_noise_std, process_noise_vel_std, velocity_half_life_s, "
    "       measurement_noise_std, pose_noise_std, outlier_threshold, tracker_fps, "
    "       velocity_mode_camera_ids, velocity_measurement_noise_std, "
    "       use_relative_observations, relative_min_confidence, "
    "       cross_pair_max_px, cross_pair_max_n, "
    "       cross_person_max_world_mm, cross_person_min_confidence, cross_person_max_n "
    "FROM tracker_configs WHERE id=?"
)


def _cfg_text(cfg: sqlite3.Row | None, cfg_id: str | None) -> str:
    if cfg is None:
        return (cfg_id[:12] + "…" if cfg_id else "—")
    parts = [cfg["name"] or (cfg_id[:8] if cfg_id else "?")]
    if cfg["process_noise_std"] is not None:
        parts.append(f"Q={cfg['process_noise_std']}")
    if cfg["process_noise_vel_std"] is not None:
        parts.append(f"Qv={cfg['process_noise_vel_std']}")
    if cfg["velocity_half_life_s"] is not None:
        parts.append(f"vhl={cfg['velocity_half_life_s']}s")
    if cfg["pose_noise_std"] is not None and cfg["pose_noise_std"] != 0.0:
        parts.append(f"Rp={cfg['pose_noise_std']}")
    if cfg["measurement_noise_std"] is not None:
        parts.append(f"Rc={cfg['measurement_noise_std']}")
    if cfg["outlier_threshold"] is not None:
        parts.append(f"thr={cfg['outlier_threshold']}")
    if cfg["velocity_mode_camera_ids"]:
        parts.append(f"vel_cams={cfg['velocity_mode_camera_ids']}")
    if cfg["velocity_measurement_noise_std"] is not None:
        parts.append(f"Rvel={cfg['velocity_measurement_noise_std']}")
    if cfg["use_relative_observations"]:
        rel_conf = cfg["relative_min_confidence"]
        suffix = f"@{rel_conf}" if rel_conf is not None else ""
        parts.append(f"rel{suffix}")
    if cfg["cross_pair_max_px"]:
        n = cfg["cross_pair_max_n"]
        n_suffix = f"×{n}" if n is not None else ""
        parts.append(f"cross@{cfg['cross_pair_max_px']}px{n_suffix}")
    if cfg["cross_person_max_world_mm"]:
        n = cfg["cross_person_max_n"]
        n_suffix = f"×{n}" if n is not None else ""
        parts.append(f"xperson@{cfg['cross_person_max_world_mm']}mm{n_suffix}")
    return "\n".join(parts)


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

        self._detect_btn = QPushButton("New trial…")
        self._detect_btn.clicked.connect(self._open_new_trial_dialog)
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

    def _open_new_trial_dialog(self) -> None:
        dlg = _NewTrialDialog(
            conn=self._conn,
            capture_id=self._capture_id,
            time_start_s=self._start_s if self._start_s > 0 else None,
            time_end_s=self._end_s if self._end_s > 0 else None,
            parent=self,
        )
        dlg.trial_created.connect(lambda _tid: self.data_changed.emit())
        dlg.exec()


# ---------------------------------------------------------------------------
# _NewTrialDialog — simple dialog to create a trial (no detection)
# ---------------------------------------------------------------------------


class _NewTrialDialog(QDialog):
    """Collect trial name and time range; insert a trials row.  No detection."""

    trial_created = Signal(str)  # trial_id

    def __init__(
        self,
        conn: sqlite3.Connection,
        capture_id: str,
        time_start_s: float | None = None,
        time_end_s: float | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Trial")
        self.setMinimumWidth(360)
        self._conn = conn
        self._capture_id = capture_id
        self._build_ui(time_start_s, time_end_s)

    def _build_ui(self, time_start_s: float | None, time_end_s: float | None) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        count = self._conn.execute(
            "SELECT COUNT(*) FROM trials WHERE capture_id = ?", (self._capture_id,)
        ).fetchone()[0]
        self._name = QLineEdit(f"Trial {count + 1}")
        form.addRow("Trial name:", self._name)

        self._start = QDoubleSpinBox()
        self._start.setRange(0.0, 100_000.0)
        self._start.setDecimals(3)
        self._start.setSuffix(" s")
        self._start.setValue(time_start_s if time_start_s is not None else 0.0)
        form.addRow("Start time:", self._start)

        self._end = QDoubleSpinBox()
        self._end.setRange(0.0, 100_000.0)
        self._end.setDecimals(3)
        self._end.setSuffix(" s")
        self._end.setValue(time_end_s if time_end_s is not None else 0.0)
        form.addRow("End time:", self._end)

        layout.addLayout(form)

        btns = QHBoxLayout()
        create_btn = QPushButton("Create Trial")
        create_btn.setDefault(True)
        create_btn.clicked.connect(self._on_create)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(create_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

    def _on_create(self) -> None:
        from posetrak.db.db import generate_id
        name = self._name.text().strip() or "Trial"
        trial_id = generate_id()
        start = self._start.value() if self._start.value() > 0 else None
        end = self._end.value() if self._end.value() > 0 else None
        self._conn.execute(
            "INSERT INTO trials (id, capture_id, name, time_start_s, time_end_s) "
            "VALUES (?, ?, ?, ?, ?)",
            (trial_id, self._capture_id, name, start, end),
        )
        self._conn.commit()
        self.trial_created.emit(trial_id)
        self.accept()


# ---------------------------------------------------------------------------
# TrialPanel
# ---------------------------------------------------------------------------


class TrialPanel(QWidget):
    """Overview panel for a trial: info, segmentation, detection runs, tracking runs."""

    data_changed = Signal()
    navigate_detection = Signal(str)   # detection run_id — open assignment editor
    navigate_tracking = Signal(str)    # tracking run_id — open tracking run panel

    def __init__(
        self,
        conn: sqlite3.Connection,
        trial_id: str,
        session_path: "Path | None" = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._conn = conn
        self._trial_id = trial_id
        self._session_path = session_path
        self._build()

    def _build(self) -> None:
        trial = self._conn.execute(
            "SELECT t.id, t.name, t.time_start_s, t.time_end_s, c.label AS capture_label "
            "FROM trials t JOIN captures c ON c.id = t.capture_id WHERE t.id = ?",
            (self._trial_id,),
        ).fetchone()
        if trial is None:
            self.setLayout(QVBoxLayout())
            self.layout().addWidget(QLabel("Trial not found."))
            return

        detection_runs = self._conn.execute(
            "SELECT id, detector_model, pose_model, status, created_at "
            "FROM detection_runs WHERE trial_id = ? ORDER BY created_at DESC",
            (self._trial_id,),
        ).fetchall()

        tracking_runs = self._conn.execute(
            "SELECT tr.id, COALESCE(sp.person_name, 'Unnamed') AS person_name, "
            "       tr.ran_at "
            "FROM tracking_runs tr "
            "LEFT JOIN pose_observation_sequences pos ON pos.id = tr.observation_sequence_id "
            "LEFT JOIN sequence_persons sp ON sp.sequence_id = pos.id "
            "WHERE tr.trial_id = ? "
            "ORDER BY tr.ran_at DESC",
            (self._trial_id,),
        ).fetchall()

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setAlignment(Qt.AlignmentFlag.AlignTop)
        vbox.setSpacing(8)
        vbox.setContentsMargins(8, 8, 8, 8)

        # Breadcrumb
        bc = QLabel(f"Capture: <b>{trial['capture_label']}</b>")
        bc.setStyleSheet("color: gray; font-size: 11px;")
        vbox.addWidget(bc)

        # Trial title + time range
        title = trial["name"] or "Unnamed trial"
        start_s = trial["time_start_s"]
        end_s = trial["time_end_s"]
        time_str = (
            f"{_fmt_time(start_s)} – {_fmt_time(end_s)}"
            if start_s is not None and end_s is not None else "—"
        )
        vbox.addWidget(QLabel(f"<h2>{title}</h2>"))
        info_form = QFormLayout()
        info_form.addRow("Time range:", QLabel(time_str))
        vbox.addLayout(info_form)

        # Segmentation section
        seg_box = _section("Segmentation")
        self._seg_btn = QPushButton(
            "Create segmentation" if detection_runs else "No detection runs yet"
        )
        self._seg_btn.setEnabled(bool(detection_runs))
        self._seg_btn.setToolTip(
            "Open interactive Cutie segmentation initialisation"
        )
        self._seg_btn.clicked.connect(self._on_open_seg_init)
        seg_box.inner_layout().addWidget(self._seg_btn)
        vbox.addWidget(seg_box)

        # Detection runs section
        det_box = _section(f"Detection runs ({len(detection_runs)})")
        if detection_runs:
            self._det_list = QListWidget()
            self._det_list.setMaximumHeight(140)
            self._det_list.setAlternatingRowColors(True)
            for r in detection_runs:
                label = (
                    f"{r['detector_model']}+{r['pose_model']}"
                    f"  {_fmt_ts(r['created_at'])}"
                    f"  [{r['status']}]"
                )
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, r["id"])
                self._det_list.addItem(item)
            self._det_list.itemDoubleClicked.connect(
                lambda it: self.navigate_detection.emit(it.data(Qt.ItemDataRole.UserRole))
            )
            det_box.inner_layout().addWidget(self._det_list)
        else:
            det_box.inner_layout().addWidget(QLabel("No detection runs yet."))

        run_det_btn = QPushButton("Run detection…")
        run_det_btn.setEnabled(self._session_path is not None)
        run_det_btn.clicked.connect(self._on_run_detection)
        det_box.inner_layout().addWidget(run_det_btn)
        vbox.addWidget(det_box)

        # Tracking runs section
        trk_box = _section(f"Tracking runs ({len(tracking_runs)})")
        if tracking_runs:
            self._trk_list = QListWidget()
            self._trk_list.setMaximumHeight(140)
            self._trk_list.setAlternatingRowColors(True)
            for r in tracking_runs:
                label = (
                    f"{r['person_name']}"
                    f"  {_fmt_ts(r['ran_at'])}"
                )
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, r["id"])
                self._trk_list.addItem(item)
            self._trk_list.itemDoubleClicked.connect(
                lambda it: self.navigate_tracking.emit(it.data(Qt.ItemDataRole.UserRole))
            )
            trk_box.inner_layout().addWidget(self._trk_list)
        else:
            trk_box.inner_layout().addWidget(QLabel("No tracking runs yet."))
        vbox.addWidget(trk_box)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(_scrollable(inner))

    def _on_open_seg_init(self) -> None:
        row = self._conn.execute(
            "SELECT id FROM detection_runs WHERE trial_id = ? ORDER BY created_at DESC LIMIT 1",
            (self._trial_id,),
        ).fetchone()
        if not row:
            return
        from app.pose.cutie_init_panel import CutieInitPanel
        win = QWidget(self, Qt.WindowType.Window)
        win.setWindowTitle("Cutie Segmentation Init")
        win.resize(1200, 750)
        layout = QVBoxLayout(win)
        layout.setContentsMargins(0, 0, 0, 0)
        panel = CutieInitPanel(self._conn, row["id"], parent=win)
        layout.addWidget(panel)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        win.destroyed.connect(panel.shutdown)
        win.show()

    def _on_run_detection(self) -> None:
        if self._session_path is None:
            return
        from app.pose.run_detection_dialog import RunDetectionDialog
        dlg = RunDetectionDialog(
            conn=self._conn,
            session_path=self._session_path,
            trial_id=self._trial_id,
            parent=self,
        )
        dlg.detection_finished.connect(lambda _tid, _rid: self.data_changed.emit())
        dlg.exec()


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
        form_box.inner_layout().addLayout(form)
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
            tr_box.inner_layout().addWidget(QLabel(t["names"] or "Unnamed track"))
        if not tracks:
            tr_box.inner_layout().addWidget(QLabel(
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
# StandaloneRunPanel — assignment editor for a single detection run
# ---------------------------------------------------------------------------


class StandaloneRunPanel(QWidget):
    """Assignment editor (StitcherPanel) for a single detection run.

    Reached by clicking a detection run row in TrialPanel or a detection run
    node in the session tree.  Shows a compact breadcrumb header above the
    full-width StitcherPanel.
    """

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
            "SELECT dr.id, dr.detector_model, dr.pose_model, dr.created_at, "
            "       dr.trial_id, t.name AS trial_name, c.label AS capture_label "
            "FROM detection_runs dr "
            "LEFT JOIN trials t ON t.id = dr.trial_id "
            "LEFT JOIN captures c ON c.id = t.capture_id "
            "WHERE dr.id = ?",
            (self._run_id,),
        ).fetchone()

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        if run:
            # Breadcrumb
            parts = []
            if run["capture_label"]:
                parts.append(run["capture_label"])
            if run["trial_name"]:
                parts.append(run["trial_name"])
            parts.append(
                f"{run['detector_model']}+{run['pose_model']}  {_fmt_ts(run['created_at'])}"
            )
            bc = QLabel("  /  ".join(parts))
            bc.setStyleSheet("color: gray; font-size: 11px;")
            vbox.addWidget(bc)

        panel = StitcherPanel(self._conn, self._run_id, parent=self)
        panel.applied.connect(self.data_changed)
        self._stitcher_panel = panel
        vbox.addWidget(panel, 1)


# ---------------------------------------------------------------------------
# CropBackfillWorker — background thread to generate missing person-crop JPEGs
# ---------------------------------------------------------------------------

class CropBackfillWorker(QThread):
    """Generate missing person-crop JPEG cache entries on a background thread.

    Uses its own SQLite connection so it never contends with the main thread's
    reads.  Emits frame_ready(svid, frame_idx) after each crop is committed.

    Call prioritise(svid, frame_idx) when the user seeks — those frames are
    decoded before the rest of the background queue.
    """

    frame_ready = Signal(str, int)

    def __init__(
        self,
        db_path: str,
        det_run_id: str,
        cameras: list[dict],        # [{shot_video_id, file_path}, ...]
        track_segs: dict,           # svid → [(track_id, first_frame, last_frame)]
        bboxes: dict,               # svid → {frame_idx: (cx, cy, w, h)}
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._db_path = db_path
        self._det_run_id = det_run_id
        self._cameras = cameras
        self._track_segs = track_segs
        self._bboxes = bboxes
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._priority: collections.deque = collections.deque()
        self._normal: collections.deque = collections.deque()
        # In-memory results for frames without a detection (no DB row written).
        # Key: (svid, frame_idx) → (jpeg_bytes, x1, y1, src_scale)
        self._mem_results: dict = {}
        self._mem_lock = threading.Lock()

    def stop(self) -> None:
        """Signal the worker to stop and block until it actually has.

        Never return early: dropping the last reference to a QThread while
        its underlying OS thread is still running is undefined behaviour in
        Qt (observed as "QThread: Destroyed while thread is still running",
        followed by the app exiting). A 3s timeout is normally enough, but if
        a single decode/seek is slow, wait however long it actually takes.
        """
        self._stop_event.set()
        if not self.wait(3000):
            _log.warning("backfill worker: still running 3s after stop() -- waiting for it to finish")
            self.wait()

    def get_mem_result(self, svid: str, frame_idx: int):
        """Return (jpeg_bytes, x1, y1, src_scale) for a frame decoded into memory, or None."""
        with self._mem_lock:
            return self._mem_results.get((svid, frame_idx))

    def prioritise(self, svid: str, frame_idx: int) -> None:
        """Ensure (svid, frame_idx) is next to be processed.

        If the frame is already in the normal queue (has a detection bbox), move
        it to the front.  If it is not in the queue at all (no bbox, or frame
        outside detection range), add a *full-frame* task directly to the
        priority queue so the worker still decodes and delivers the image.
        """
        with self._lock:
            # Check if already queued in priority (avoid duplicates)
            for item in self._priority:
                if item[0] == svid and item[1] == frame_idx:
                    return

            # Try to promote from normal queue
            moved = []
            remaining = collections.deque()
            for item in self._normal:
                if item[0] == svid and item[1] == frame_idx:
                    moved.append(item)
                else:
                    remaining.append(item)
            if moved:
                self._normal = remaining
                self._priority.extendleft(reversed(moved))
                return

            # Frame not queued at all — add as a full-frame / nearest-bbox task.
            # track_id=None, bbox=None signals the worker to use best-effort decode.
            self._priority.append((svid, frame_idx, None, None))

    def run(self) -> None:
        import bisect
        import cv2
        from app.pose.db_cache import _CROP_TARGET_HEIGHT, _CROP_JPEG_QUALITY, _encode_crop

        _log.info("backfill worker: thread started  db=%s", self._db_path)
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
        except Exception:
            _log.exception("backfill worker: failed to open DB")
            return

        file_map = {cam["shot_video_id"]: cam["file_path"] for cam in self._cameras}

        # Precompute sorted bbox frame lists per svid for window-union lookup.
        sorted_bbox_frames: dict[str, list[int]] = {
            svid: sorted(bboxes.keys())
            for svid, bboxes in self._bboxes.items()
        }
        _UNION_N = 10   # frames on each side of the target frame to include

        def _union_bbox(svid: str, frame_idx: int):
            """Return union bbox of all detections within ±N frames, or None.

            The union (min x1/y1, max x2/y2) covers the person's range across
            the context window, padded by _CROP_MARGIN on all sides via
            _encode_crop.  Returned as (cx, cy, w, h).
            """
            bboxes = self._bboxes.get(svid, {})
            sframes = sorted_bbox_frames.get(svid, [])
            if not sframes:
                return None
            lo, hi = frame_idx - _UNION_N, frame_idx + _UNION_N
            i_lo = bisect.bisect_left(sframes, lo)
            i_hi = bisect.bisect_right(sframes, hi)
            window = sframes[i_lo:i_hi]
            if not window:
                # Nothing in window — fall back to the single nearest frame.
                pos = bisect.bisect_left(sframes, frame_idx)
                if pos == 0:
                    window = [sframes[0]]
                elif pos >= len(sframes):
                    window = [sframes[-1]]
                else:
                    before, after = sframes[pos - 1], sframes[pos]
                    window = [before if frame_idx - before <= after - frame_idx else after]
            # Compute union in (x1, y1, x2, y2) space.
            x1 = y1 = float("inf")
            x2 = y2 = float("-inf")
            for fi in window:
                cx, cy, w, h = bboxes[fi]
                x1 = min(x1, cx - w / 2)
                y1 = min(y1, cy - h / 2)
                x2 = max(x2, cx + w / 2)
                y2 = max(y2, cy + h / 2)
            uw, uh = x2 - x1, y2 - y1
            return (x1 + uw / 2, y1 + uh / 2, uw, uh)

        # Build the normal queue: frames WITH a detection bbox that lack a cached crop.
        tasks: list[tuple] = []
        for cam in self._cameras:
            svid = cam["shot_video_id"]
            segs = self._track_segs.get(svid, [])
            bboxes = self._bboxes.get(svid, {})
            _log.debug(
                "backfill worker: svid=%s  segs=%d  bbox_frames=%d",
                svid, len(segs), len(bboxes),
            )
            if not segs or not bboxes:
                _log.debug("backfill worker: skipping svid=%s (no segs or bboxes)", svid)
                continue
            cam_tasks = 0
            for track_id, first, last in segs:
                cached = set(
                    r[0] for r in conn.execute(
                        "SELECT frame_idx FROM frame_cache_entries"
                        " WHERE shot_video_id=? AND cache_type='person_crop'"
                        " AND detection_run_id=? AND region_type='full_body'"
                        " AND track_id=?",
                        (svid, self._det_run_id, track_id),
                    )
                )
                _log.debug(
                    "backfill worker: svid=%s  track=%s  cached=%d  seg=[%d,%d]",
                    svid, track_id, len(cached), first, last,
                )
                for fi in range(first, last + 1):
                    if fi in cached:
                        continue
                    bbox = bboxes.get(fi)
                    if bbox is None:
                        continue
                    tasks.append((svid, fi, track_id, bbox))
                    cam_tasks += 1
            _log.info("backfill worker: svid=%s  missing DB crops=%d", svid, cam_tasks)

        _log.info("backfill worker: total missing DB crops queued=%d", len(tasks))
        tasks.sort(key=lambda t: t[1])
        with self._lock:
            self._normal.extend(tasks)

        # None sentinel in caps means "video failed to open — don't retry"
        caps: dict[str, object] = {}

        def _get_frame(svid: str, frame_idx: int):
            if svid not in caps:
                path = file_map.get(svid, "")
                _log.debug("backfill worker: opening video  svid=%s  path=%s", svid, path)
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    _log.warning("backfill worker: cannot open video  svid=%s  path=%s", svid, path)
                    caps[svid] = None
                    return None
                caps[svid] = cap
                _log.info("backfill worker: opened video  svid=%s", svid)
            cap = caps[svid]
            if cap is None:
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = cap.read()
            if not ok:
                _log.warning("backfill worker: read failed  svid=%s  frame=%d", svid, frame_idx)
            return bgr if ok else None

        def _encode_mem(bgr, svid: str, frame_idx: int):
            """Encode a crop for a frame with no detection.

            Uses the union of bboxes within ±N frames as crop region so the
            result is wide enough to contain the person even if they moved.
            Falls back to a full-frame thumbnail when no bboxes exist.
            Returns same tuple as _encode_crop or None on failure.
            """
            union = _union_bbox(svid, frame_idx)
            if union is not None:
                return _encode_crop(bgr, union)
            # Full-frame fallback: downscale to target height.
            h, w = bgr.shape[:2]
            if h > _CROP_TARGET_HEIGHT:
                scale = _CROP_TARGET_HEIGHT / h
                thumb = cv2.resize(bgr, (int(w * scale), _CROP_TARGET_HEIGHT))
            else:
                thumb = bgr.copy()
            ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, _CROP_JPEG_QUALITY])
            if not ok:
                return None
            return buf.tobytes(), thumb.shape[1], thumb.shape[0], 0, 0, w, h

        n_done = 0
        try:
            while not self._stop_event.is_set():
                with self._lock:
                    if self._priority:
                        task = self._priority.popleft()
                        _log.debug("backfill worker: dequeued priority frame svid=%s frame=%d", task[0], task[1])
                    elif self._normal:
                        task = self._normal.popleft()
                    else:
                        task = None

                if task is None:
                    self._stop_event.wait(0.05)
                    continue

                svid, frame_idx, track_id, bbox = task

                if bbox is None:
                    # No detection for this frame — ghost crop.
                    with self._mem_lock:
                        if (svid, frame_idx) in self._mem_results:
                            continue  # already decoded by a previous priority request

                    # Check if already persisted to DB from a previous session.
                    cached = conn.execute(
                        "SELECT image_data, width_px, height_px, src_x, src_y, src_w, src_h"
                        " FROM frame_cache_entries"
                        " WHERE shot_video_id=? AND cache_type='ghost_crop'"
                        " AND track_id=-1 AND region_type='full_body' AND detection_run_id=''"
                        " AND frame_idx=?",
                        (svid, frame_idx),
                    ).fetchone()
                    if cached is not None:
                        result = (
                            bytes(cached["image_data"]),
                            cached["width_px"], cached["height_px"],
                            cached["src_x"] or 0, cached["src_y"] or 0,
                            cached["src_w"] or 0, cached["src_h"] or 0,
                        )
                        with self._mem_lock:
                            self._mem_results[(svid, frame_idx)] = result
                        _log.debug("backfill worker: loaded ghost from DB  svid=%s  frame=%d", svid, frame_idx)
                        self.frame_ready.emit(svid, frame_idx)
                        continue

                    bgr = _get_frame(svid, frame_idx)
                    if bgr is None:
                        continue
                    result = _encode_mem(bgr, svid, frame_idx)
                    if result is None:
                        _log.warning("backfill worker: mem encode failed  svid=%s  frame=%d", svid, frame_idx)
                        continue
                    with self._mem_lock:
                        self._mem_results[(svid, frame_idx)] = result
                    _log.debug("backfill worker: stored mem crop  svid=%s  frame=%d", svid, frame_idx)

                    # Persist ghost crop to DB so it survives app restarts.
                    jpeg, wpx, hpx, src_x, src_y, src_w, src_h = result
                    try:
                        conn.execute(
                            "INSERT OR REPLACE INTO frame_cache_entries"
                            " (shot_video_id, frame_idx, cache_type, track_id, region_type,"
                            "  width_px, height_px, image_data, detection_run_id,"
                            "  src_x, src_y, src_w, src_h)"
                            " VALUES (?,?,'ghost_crop',-1,'full_body',?,?,?,'',?,?,?,?)",
                            (svid, frame_idx, wpx, hpx, jpeg, src_x, src_y, src_w, src_h),
                        )
                        conn.commit()
                    except Exception:
                        _log.warning("backfill worker: failed to persist ghost crop  svid=%s  frame=%d", svid, frame_idx)

                    self.frame_ready.emit(svid, frame_idx)
                    continue

                # Frame has a detection bbox → write to persistent DB cache.
                already = conn.execute(
                    "SELECT 1 FROM frame_cache_entries"
                    " WHERE shot_video_id=? AND cache_type='person_crop'"
                    " AND detection_run_id=? AND frame_idx=? AND region_type='full_body'",
                    (svid, self._det_run_id, frame_idx),
                ).fetchone()
                if already:
                    continue

                bgr = _get_frame(svid, frame_idx)
                if bgr is None:
                    continue

                result = _encode_crop(bgr, bbox)
                if result is None:
                    _log.warning("backfill worker: encode failed  svid=%s  frame=%d", svid, frame_idx)
                    continue

                jpeg, wpx, hpx, src_x, src_y, src_w, src_h = result
                conn.execute(
                    "INSERT OR REPLACE INTO frame_cache_entries"
                    " (shot_video_id, frame_idx, cache_type, track_id, region_type,"
                    "  width_px, height_px, image_data, detection_run_id,"
                    "  src_x, src_y, src_w, src_h)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (svid, frame_idx, "person_crop", track_id, "full_body",
                     wpx, hpx, jpeg, self._det_run_id,
                     src_x, src_y, src_w, src_h),
                )
                conn.commit()
                n_done += 1
                if n_done % 50 == 1:
                    _log.info("backfill worker: progress  written=%d", n_done)
                _log.debug("backfill worker: wrote crop  svid=%s  frame=%d", svid, frame_idx)
                self.frame_ready.emit(svid, frame_idx)
        except Exception:
            _log.exception("backfill worker: unexpected error")
        finally:
            _log.info("backfill worker: stopping  written=%d", n_done)
            for cap in caps.values():
                if cap is not None:
                    cap.release()
            conn.close()


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

from app.pose.kp_models import PoseModel, get_pose_model as _get_pose_model
from app.pose.timeline_status import STATUS_BLUE, compute_inlier_camera_counts, read_timeline_status
from app.ui.keypoint_timeline_widget import KeypointTimelineWidget


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


# ---------------------------------------------------------------------------
# Keypoint edit-mode overlay constants
# ---------------------------------------------------------------------------

_KP_EDIT_RADIUS = 4
_KP_DRAG_THRESHOLD = 5
_KP_HIT_RADIUS = _KP_EDIT_RADIUS + 4
_TRAIL_PAST_COLOR = QColor(220, 60, 60)
_TRAIL_FUTURE_COLOR = QColor(60, 100, 220)
_TRAIL_GHOST_COLOR = QColor(140, 140, 140, 160)
_TRAIL_LINE_WIDTH = 1.5
_TRAIL_DOT_R = 3
_TRAIL_N = 10

# ---------------------------------------------------------------------------
# Crop-cell zoom/pan constants and pure geometry helpers
#
# See "Zoom and pan in the camera crop views" in the design doc: view state
# is a full-frame-pixel rectangle (not a display-pixel offset), because the
# wide-crop cache's underlying crop can shift between epochs -- the same
# clamp-to-available-pixels shape already built for that feature's
# sub-cropping.
# ---------------------------------------------------------------------------

_ZOOM_PER_WHEEL_STEP = 1.15   # factor per 120-unit angleDelta step (one mouse click)
_MIN_ZOOM_RECT_SIZE = 20.0    # full-frame px floor, prevents a degenerate zoom-in
_DISPLAY_MARGIN_FRAC = 0.15   # margin around the minimum display bbox, default (unzoomed) view


def _zoomed_rect(
    rect: tuple, factor: float, anchor: tuple, min_size: float = _MIN_ZOOM_RECT_SIZE,
) -> "tuple | None":
    """Scale *rect* (x0, y0, x1, y1) by 1/factor around *anchor* (ax, ay).

    factor > 1 zooms in (rect shrinks toward the anchor); factor < 1 zooms
    out. Returns None if the result would be smaller than *min_size* on
    either axis -- the caller should then leave the rect unchanged rather
    than apply a degenerate (near-zero-area) zoom.
    """
    x0, y0, x1, y1 = rect
    ax, ay = anchor
    nx0 = ax + (x0 - ax) / factor
    nx1 = ax + (x1 - ax) / factor
    ny0 = ay + (y0 - ay) / factor
    ny1 = ay + (y1 - ay) / factor
    if nx1 - nx0 < min_size or ny1 - ny0 < min_size:
        return None
    return (nx0, ny0, nx1, ny1)


def _panned_rect(rect: tuple, dx: float, dy: float) -> tuple:
    """Translate *rect* (x0, y0, x1, y1) by (dx, dy)."""
    x0, y0, x1, y1 = rect
    return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)


def _compute_view_transform(
    cell_w: float,
    cell_h: float,
    pixmap_extent: tuple,
    zoom_rect: "tuple | None",
) -> tuple:
    """Return (scale, origin_x, origin_y, view_rect).

    The affine map from full-frame pixel (u, v) to display pixel is
    `(u * scale + origin_x, v * scale + origin_y)`. *view_rect* is the
    full-frame rectangle actually being displayed: *zoom_rect* clamped to
    *pixmap_extent* (both (x0, y0, x1, y1)) if given and it overlaps, else
    the whole *pixmap_extent* -- "fit whatever crop is given", today's
    unzoomed behavior.

    Before fitting, *view* is grown (never shrunk) to match the cell's own
    aspect ratio, then re-clamped to *pixmap_extent* -- fills the cell with
    image content instead of a letterboxed border whenever the cache has the
    extra margin to do so. This applies whether or not *zoom_rect* is set:
    without it, a zoomed-uniformly rect preserves whatever aspect ratio the
    unzoomed crop happened to have, which has no reason to match the cell
    either -- the same problem the wide-crop cache's own initial crop
    selection has, fixed here once for every layer and every zoom level
    instead of separately per crop-selection call site.
    """
    pm_x0, pm_y0, pm_x1, pm_y1 = pixmap_extent
    view = (pm_x0, pm_y0, pm_x1, pm_y1)
    if zoom_rect is not None:
        zx0, zy0, zx1, zy1 = zoom_rect
        cx0, cy0 = max(zx0, pm_x0), max(zy0, pm_y0)
        cx1, cy1 = min(zx1, pm_x1), min(zy1, pm_y1)
        if cx1 > cx0 and cy1 > cy0:
            view = (cx0, cy0, cx1, cy1)
    if cell_w > 0 and cell_h > 0:
        expanded = _expand_rect_to_aspect(view, cell_w / cell_h)
        ex0, ey0, ex1, ey1 = expanded
        cx0, cy0 = max(ex0, pm_x0), max(ey0, pm_y0)
        cx1, cy1 = min(ex1, pm_x1), min(ey1, pm_y1)
        if cx1 > cx0 and cy1 > cy0:
            view = (cx0, cy0, cx1, cy1)
    vx0, vy0, vx1, vy1 = view
    view_w, view_h = vx1 - vx0, vy1 - vy0
    if view_w <= 0 or view_h <= 0 or cell_w <= 0 or cell_h <= 0:
        return 1.0, 0.0, 0.0, view
    scale = min(cell_w / view_w, cell_h / view_h)
    origin_x = (cell_w - view_w * scale) / 2 - vx0 * scale
    origin_y = (cell_h - view_h * scale) / 2 - vy0 * scale
    return scale, origin_x, origin_y, view


def _nearest_segment_track_id(
    segs: "list[tuple[int, int, int]]", frame_idx: int,
) -> "int | None":
    """Track id of whichever `(track_id, first_frame, last_frame)` segment in
    *segs* is closest to *frame_idx* -- 0 distance (and an early return) if
    one already covers it, otherwise the smallest gap to whichever segment's
    nearest edge is closest. None if *segs* is empty.

    Used to resolve a wide-crop cache lookup for a frame in a genuine gap
    between two of a person's assigned track segments (see
    `_nearest_track_id_for_gap`) -- distinct from a frame within a segment
    that simply lacks a real per-frame detection, which the wide-crop
    cache's own gap-search already handles given a resolvable track_id.
    """
    best_id, best_dist = None, None
    for track_id, first_frame, last_frame in segs:
        if frame_idx < first_frame:
            dist = first_frame - frame_idx
        elif frame_idx > last_frame:
            dist = frame_idx - last_frame
        else:
            return track_id
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_id = track_id
    return best_id


# Idea 3 (automated post-edit hand redetection): a debounced timer per
# (camera, frame, side) waits this long after the last wrist/elbow edit
# before firing a redetect request -- an untuned guess (500ms-1s proposed
# in the design doc), meant to be adjusted once tried against real editing.
_HAND_REDETECT_DEBOUNCE_MS = 700


def _hand_side_for_kp_idx(kp_idx: int) -> "str | None":
    """Return "left"/"right" if *kp_idx* is a wrist or elbow index for that
    side, else None -- the trigger condition for Idea 3's hand redetection.
    """
    from posetrak.detection.hand_refinement import _ELBOW_IDX, _WRIST_IDX

    for side, idx in _WRIST_IDX.items():
        if idx == kp_idx:
            return side
    for side, idx in _ELBOW_IDX.items():
        if idx == kp_idx:
            return side
    return None


def _kp_overlay_bbox(obs_kp, hidden_indices: "frozenset[int]"):
    """Bounding box (x0, y0, x1, y1), full-frame pixel coords, of keypoints
    that would actually be drawn for this frame (matches paintEvent's
    conf >= 0.1 visibility cutoff), or None if none qualify.

    Used to make sure the wide-crop cache's sub-crop always covers whatever
    is actually being displayed -- including edited keypoints, which can sit
    far from where the original (possibly wrong) detection placed the
    person, since the crop cache only knows about raw `person_detections`.
    """
    if obs_kp is None:
        return None
    xs, ys = [], []
    for i in range(obs_kp.shape[0]):
        if i in hidden_indices or float(obs_kp[i, 2]) < 0.1:
            continue
        xs.append(float(obs_kp[i, 0]))
        ys.append(float(obs_kp[i, 1]))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _windowed_kp_bbox(
    obs_kp_by_frame: "dict[int, object]",
    frame_idx: int,
    hidden_indices: "frozenset[int]",
    n_frames: int = 10,
) -> "tuple | None":
    """Union of `_kp_overlay_bbox` over the nearest `n_frames` frames *with
    an observation* on each side of `frame_idx` (inclusive of `frame_idx`
    itself), skipping frames with none -- so a long stretch with no
    detection doesn't erode the window down to nothing.

    E.g. frame_idx=100 with no observations for this person in frames
    75-450: the backward side reaches past the gap to whichever 10 frames
    <=100 do have data (e.g. 65-74), and the forward side likewise reaches
    to whichever 10 do (e.g. 451-460) -- not just frames 90-110.

    `obs_kp_by_frame` is the already edit-merged per-frame array (see
    `read_observations_with_edits`), so this single window covers both the
    "raw observation" and "edited keypoint" cases from the original brief:
    edits overwrite a keypoint's x/y in place, they don't add a second,
    separate signal to track.
    """
    if not obs_kp_by_frame:
        return None
    frames = sorted(obs_kp_by_frame.keys())
    backward = [f for f in frames if f <= frame_idx][-n_frames:]
    forward = [f for f in frames if f > frame_idx][:n_frames]
    bbox = None
    for f in backward + forward:
        b = _kp_overlay_bbox(obs_kp_by_frame.get(f), hidden_indices)
        if b is None:
            continue
        if bbox is None:
            bbox = b
        else:
            bbox = (
                min(bbox[0], b[0]), min(bbox[1], b[1]),
                max(bbox[2], b[2]), max(bbox[3], b[3]),
            )
    return bbox


def _tracked_overlay_bbox(
    joint_xy: "dict | None",
    marker_xy,
    video_dims: "tuple[int, int] | None" = None,
) -> "tuple | None":
    """Bounding box (x0, y0, x1, y1), full-frame pixel coords, of the tracked
    skeleton's projected joints/markers for one camera + tracker step, or
    None if neither is available.

    A tracking result can exist and be selected independently of whether the
    person's keypoints have been edited yet (edits are normally done before
    tracking, but nothing prevents editing after a run too), so the wide-crop
    sub-crop must cover this overlay as well as `_kp_overlay_bbox`'s.

    If `video_dims` (full-frame width, height) is given, points projecting
    outside the camera's own frame are dropped before the bbox is built --
    the skeleton can be entirely or partly out of view/behind this camera
    (e.g. occluded during part of a throw), and including those off-frame
    coordinates would force the requested crop area to an unsatisfiable
    extent, permanently starving this camera of a displayable crop instead
    of just showing whatever part of the skeleton is actually visible here.
    """
    from math import isnan

    def _in_view(u: float, v: float) -> bool:
        if video_dims is None:
            return True
        w, h = video_dims
        return 0.0 <= u <= w and 0.0 <= v <= h

    xs, ys = [], []
    if joint_xy:
        for u, v in joint_xy.values():
            u, v = float(u), float(v)
            if isnan(u) or isnan(v) or not _in_view(u, v):
                continue
            xs.append(u)
            ys.append(v)
    if marker_xy is not None:
        for i in range(marker_xy.shape[0]):
            u, v = float(marker_xy[i, 0]), float(marker_xy[i, 1])
            if isnan(u) or isnan(v) or not _in_view(u, v):
                continue
            xs.append(u)
            ys.append(v)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


# Last-resort sanity cap for a display bbox, in full-frame pixels --
# comfortably larger than any real camera resolution. `_tracked_overlay_bbox`
# already drops out-of-frame points when `video_dims` is known, but that
# guard is a no-op when a shot's video dimensions aren't recorded in the DB;
# a numerically diverged tracked-skeleton projection (e.g. from a UKF state
# that's diverged during fast motion -- see the covariance-ill-conditioning
# investigation in the design doc) or a corrupted keypoint edit can then
# reach `desired` unbounded, which previously turned into an attempt to
# allocate a multi-terabyte canvas in `_composite_black_fill`.
_MAX_PLAUSIBLE_BBOX_DIM = 20_000.0


def _sane_bbox(bbox: "tuple | None") -> "tuple | None":
    """Reject a bbox with a non-finite coordinate, an implausibly large
    width/height, or an implausibly large coordinate magnitude, so one bad
    source (edit, detection, or projection) can't blow up the crop area
    unioned from several independent sources.

    The magnitude check matters on its own, separately from the
    width/height check: a single diverged point (e.g. one tracked marker
    projected far from the scene) is a *zero-width* bbox -- (v, v, v, v) --
    so a width/height-only check would never catch it before it's unioned
    with another, normally-sized bbox and the combined extent blows up.
    """
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    if not all(isfinite(v) for v in bbox):
        return None
    if any(abs(v) > _MAX_PLAUSIBLE_BBOX_DIM for v in bbox):
        return None
    if (x1 - x0) > _MAX_PLAUSIBLE_BBOX_DIM or (y1 - y0) > _MAX_PLAUSIBLE_BBOX_DIM:
        return None
    return bbox


def _expand_rect_to_aspect(rect: tuple, target_ar: float) -> tuple:
    """Grow *rect* (x0, y0, x1, y1), centered, to match *target_ar* (w/h).

    Only ever grows, never shrinks, so callers that already ensured some
    minimum content is covered don't lose it. Used so the wide-crop cache's
    displayed sub-crop fills the cell instead of leaving a letterboxed black
    border -- the padded/keypoint-widened window has no inherent reason to
    match the cell's aspect ratio otherwise.
    """
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0 or target_ar <= 0:
        return rect
    cur_ar = w / h
    if cur_ar < target_ar:
        new_w = h * target_ar
        dx = (new_w - w) / 2
        return (x0 - dx, y0, x1 + dx, y1)
    new_h = w / target_ar
    dy = (new_h - h) / 2
    return (x0, y0 - dy, x1, y1 + dy)


_MAX_CANVAS_DIM_PX = 8000  # last-resort allocation cap, see _composite_black_fill


def _composite_black_fill(
    crop_bgr,
    x1: float,
    y1: float,
    src_scale: float,
    target_rect: "tuple[float, float, float, float]",
):
    """Composite *crop_bgr* (a decoded BGR crop whose top-left full-frame
    corner is (x1, y1), at `src_scale` full-frame-px -> crop-px) onto a
    black canvas exactly covering *target_rect*, in full-frame pixel
    coordinates.

    Returns (canvas, new_x1, new_y1, black_filled) -- `black_filled` is True
    iff some part of the canvas wasn't covered by decoded pixels (used only
    for the debug label, see "Debug overlay" in the design doc). Unlike a
    plain sub-crop, *target_rect* is not required to fit inside what's
    actually decoded -- any part of it outside the decoded pixels stays
    black, rather than shrinking the
    display down to whatever *is* available. See "Unified minimum-display
    bbox..." in the design doc.
    """
    import numpy as np

    tx0, ty0, tx1, ty1 = target_rect
    canvas_w = max(1, round((tx1 - tx0) * src_scale))
    canvas_h = max(1, round((ty1 - ty0) * src_scale))
    if canvas_w > _MAX_CANVAS_DIM_PX or canvas_h > _MAX_CANVAS_DIM_PX:
        # Callers are expected to have already sanity-checked target_rect
        # (see _sane_bbox) -- this is a last-resort guard against actually
        # allocating a multi-gigabyte/terabyte array if something upstream
        # still let an implausible rect through, rather than crashing.
        _log.warning(
            "_composite_black_fill: clamping implausible canvas %dx%d px "
            "(target_rect=%r, src_scale=%.4g)",
            canvas_w, canvas_h, target_rect, src_scale,
        )
        canvas_w = min(canvas_w, _MAX_CANVAS_DIM_PX)
        canvas_h = min(canvas_h, _MAX_CANVAS_DIM_PX)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Overlap between the desired window and what's actually decoded, in
    # full-frame coordinates.
    dec_x1 = x1 + crop_bgr.shape[1] / src_scale
    dec_y1 = y1 + crop_bgr.shape[0] / src_scale
    ox0, oy0 = max(tx0, x1), max(ty0, y1)
    ox1, oy1 = min(tx1, dec_x1), min(ty1, dec_y1)
    black_filled = True  # any part of the canvas not proven covered below
    if ox1 > ox0 and oy1 > oy0:
        sx0 = int(round((ox0 - x1) * src_scale))
        sy0 = int(round((oy0 - y1) * src_scale))
        dx0 = int(round((ox0 - tx0) * src_scale))
        dy0 = int(round((oy0 - ty0) * src_scale))
        # Width/height come from whichever of {remaining source, remaining
        # canvas} runs out first, rather than independently rounding
        # (ox1 - x1) and (ox1 - tx0) -- those two roundings can disagree by
        # a pixel (different offsets rounding differently), which previously
        # made the source patch one pixel wider/taller than the canvas slot
        # it was being copied into and crashed with a broadcast-shape error.
        pw = min(crop_bgr.shape[1] - sx0, canvas_w - dx0)
        ph = min(crop_bgr.shape[0] - sy0, canvas_h - dy0)
        if pw > 0 and ph > 0:
            canvas[dy0:dy0 + ph, dx0:dx0 + pw] = crop_bgr[sy0:sy0 + ph, sx0:sx0 + pw]
            black_filled = not (dx0 == 0 and dy0 == 0 and pw == canvas_w and ph == canvas_h)
    return canvas, tx0, ty0, black_filled


def _build_cam_trail(
    kp_by_frame: dict,
    cam_id: str,
    current_frame: int,
    kp_idx: int,
    n: int = _TRAIL_N,
):
    """Compute past/future trail for kp_idx in one camera using sorted frame indices."""
    from app.pose.crop_editor import _FrameSlot, _TrailData, compute_trail
    sorted_frames = sorted(kp_by_frame.keys())
    if not sorted_frames:
        return _TrailData(kp_idx=kp_idx, past=[], future=[])
    cur_pos = min(range(len(sorted_frames)), key=lambda i: abs(sorted_frames[i] - current_frame))
    frames = [_FrameSlot(timestamp_s=float(f), per_cam={cam_id: f}) for f in sorted_frames]
    return compute_trail(frames, kp_by_frame, cam_id, cur_pos, kp_idx, n)


class _ImageCanvas(QWidget):
    """Custom painting widget: image + vector overlays drawn with QPainter.

    All overlay coordinates are stored in full-frame pixel space.  The
    coordinate transform (full-frame → JPEG crop → display) is computed
    at paint time so overlays stay sharp regardless of zoom level.
    """

    keypoint_selected = Signal(int)          # plain left-click on dot
    keypoint_ctrl_clicked = Signal(int)      # Ctrl+left-click on dot (toggle)
    keypoint_deselected = Signal()
    keypoint_moved = Signal(int, float, float)
    # Emitted when the user clicks empty space in edit mode (no kp hit).
    # Carries display-space coords so the widget can convert and place a kp.
    empty_area_clicked = Signal(float, float)
    # Rubber-band drag completed: (x1, y1, x2, y2) display pixels + ctrl_held bool
    rubber_band_selected = Signal(float, float, float, float, bool)
    # Right-click: hit_kp_idx (-1 if empty), display (dx, dy)
    context_menu_requested = Signal(int, float, float)
    # Left-click while placement mode is active (see set_placement_active):
    # display-space coords, regardless of whether the click hit an existing
    # dot -- placement always wins over the normal select/drag/rubber-band
    # flow while armed.
    placement_clicked = Signal(float, float)

    def __init__(self, min_h: int = 240, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(min_h)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #222;")
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
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
        # Edit mode state
        self._edit_mode: bool = False
        self._sel_kp_set: frozenset[int] = frozenset()   # all selected kp indices
        self._hidden_kp: frozenset[int] = frozenset()    # excluded from drawing + hit-testing
        self._primary_kp: int | None = None               # primary (name label, trail)
        self._drag_kp: int | None = None
        self._drag_start_disp: tuple[float, float] | None = None
        self._drag_cur_disp: tuple[float, float] | None = None
        self._drag_moved: bool = False
        self._rubber_band_active: bool = False
        self._rubber_band_ctrl: bool = False   # Ctrl held at rubber-band start
        self._trail = None  # _TrailData | None
        self._selected_kp_name: str | None = None
        self._loading: bool = False
        self._range_pts: list[tuple[float, float]] = []  # frame-space positions to highlight
        # Zoom/pan: desired view, full-frame pixel coords; None = fit whatever
        # crop is given (today's default). See _compute_view_transform.
        self._zoom_rect: tuple[float, float, float, float] | None = None
        self._pan_active: bool = False
        self._pan_start_disp: tuple[float, float] | None = None
        self._pan_start_rect: tuple[float, float, float, float] | None = None
        # Keypoint-placement mode: set via set_placement_active when a
        # keypoint is picked from _KeypointPickerPanel. While True, any
        # left-click places it (see placement_clicked / mousePressEvent)
        # instead of the normal select/drag/rubber-band flow.
        self._placement_active: bool = False
        # Text shown top-center while chain-placement mode is armed, naming
        # the limb keypoint the next click will set. None hides it (also used
        # for the plain one-shot placement mode, which has no chain to name).
        self._placement_label: str | None = None
        # Diagnostic label naming which crop-source layer produced the
        # currently-shown image (and whether black-fill was applied) --
        # see "Debug overlay" in the design doc. None = hidden (checkbox off
        # or nothing to report yet).
        self._debug_label: str | None = None

    def set_placement_active(self, active: bool) -> None:
        self._placement_active = active
        self.setCursor(Qt.CursorShape.CrossCursor if active else Qt.CursorShape.ArrowCursor)

    def set_placement_label(self, text: str | None) -> None:
        self._placement_label = text
        self.update()

    def show_empty(self) -> None:
        self._loading = False
        self._pixmap = None
        self._obs_kp = None
        self._outlier_kp_mask = None
        self._joint_xy = None
        self._marker_xy = None
        self._debug_label = None
        self.update()

    def show_loading(self) -> None:
        """Show a 'generating…' placeholder — called when backfill is in progress."""
        self._loading = True
        self._pixmap = None
        self._obs_kp = None
        self._outlier_kp_mask = None
        self._joint_xy = None
        self._marker_xy = None
        self._debug_label = None
        self.update()

    def show_image(self, pixmap: QPixmap, x1: float, y1: float, src_scale: float) -> None:
        self._loading = False
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

    def set_debug_label(self, text: "str | None") -> None:
        if text != self._debug_label:
            self._debug_label = text
            self.update()

    def set_edit_mode(self, enabled: bool) -> None:
        self._edit_mode = enabled
        self.setMouseTracking(enabled)
        if not enabled:
            self._sel_kp_set = frozenset()
            self._primary_kp = None
            self._drag_kp = None
            self._drag_start_disp = None
            self._drag_cur_disp = None
            self._drag_moved = False
            self._rubber_band_active = False
            self._rubber_band_ctrl = False
            self._trail = None
        self.update()

    def set_trail(self, trail) -> None:
        self._trail = trail
        self.update()

    def set_range_highlights(self, pts: list[tuple[float, float]]) -> None:
        self._range_pts = pts
        self.update()

    def set_selection(
        self, primary: int | None, sel_indices: frozenset[int], name: str | None = None
    ) -> None:
        self._primary_kp = primary
        self._sel_kp_set = sel_indices
        self._selected_kp_name = name
        self.update()

    def set_hidden(self, hidden: frozenset[int]) -> None:
        """Keypoint indices toggled off from the timeline's eye icon: drawn
        nowhere and excluded from hit-testing, regardless of edit mode."""
        self._hidden_kp = hidden
        self.update()

    def set_selected_kp(
        self, idx: int | None, show_ring: bool = True, name: str | None = None
    ) -> None:
        self.set_selection(idx, frozenset({idx}) if idx is not None else frozenset(), name)

    def _pixmap_extent(self) -> "tuple[float, float, float, float] | None":
        """Full-frame pixel extent (x0, y0, x1, y1) the current pixmap covers."""
        if self._pixmap is None or self._pixmap.width() == 0 or self._src_scale <= 0:
            return None
        return (
            self._x1, self._y1,
            self._x1 + self._pixmap.width() / self._src_scale,
            self._y1 + self._pixmap.height() / self._src_scale,
        )

    def _view_transform(self) -> tuple:
        """(scale, origin_x, origin_y, view_rect) -- see _compute_view_transform."""
        extent = self._pixmap_extent()
        if extent is None:
            return 1.0, 0.0, 0.0, (0.0, 0.0, float(self.width()), float(self.height()))
        return _compute_view_transform(
            float(self.width()), float(self.height()), extent, self._zoom_rect,
        )

    def reset_zoom(self) -> None:
        """Return to fit-whatever-crop-is-given (today's unzoomed default)."""
        if self._zoom_rect is not None:
            self._zoom_rect = None
            self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.reset_zoom()

    def _to_pt(self, u: float, v: float) -> QPointF:
        scale, origin_x, origin_y, _view = self._view_transform()
        return QPointF(u * scale + origin_x, v * scale + origin_y)

    def _display_to_full(self, dx: float, dy: float) -> tuple[float, float]:
        scale, origin_x, origin_y, _view = self._view_transform()
        if scale == 0:
            return self._x1, self._y1
        return (dx - origin_x) / scale, (dy - origin_y) / scale

    def _hit_kp(self, dx: float, dy: float) -> int | None:
        if self._obs_kp is None:
            return None
        scale, origin_x, origin_y, _view = self._view_transform()
        best_i, best_d2 = None, float(_KP_HIT_RADIUS ** 2)
        for i in range(self._obs_kp.shape[0]):
            if i in self._hidden_kp:
                continue
            if float(self._obs_kp[i, 2]) < 0.1 and not self._edit_mode:
                continue
            px = float(self._obs_kp[i, 0]) * scale + origin_x
            py = float(self._obs_kp[i, 1]) * scale + origin_y
            d2 = (dx - px) ** 2 + (dy - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def mousePressEvent(self, event) -> None:  # noqa: N802
        # Middle-mouse pan works regardless of edit mode -- it's a view-only
        # operation, not an edit action, and doesn't compete with left/right
        # button handling below.
        if event.button() == Qt.MouseButton.MiddleButton:
            pos = event.position()
            self._pan_active = True
            self._pan_start_disp = (pos.x(), pos.y())
            self._pan_start_rect = self._zoom_rect or self._pixmap_extent()
            return

        if self._edit_mode:
            pos = event.position()
            dx, dy = pos.x(), pos.y()

            if event.button() == Qt.MouseButton.RightButton:
                hit = self._hit_kp(dx, dy)
                self.context_menu_requested.emit(hit if hit is not None else -1, dx, dy)
                return

            if self._placement_active and event.button() == Qt.MouseButton.LeftButton:
                # Placement wins over the normal select/drag/rubber-band flow
                # entirely -- no hit-test, no drag threshold, immediate place.
                self.placement_clicked.emit(dx, dy)
                return

            if event.button() == Qt.MouseButton.LeftButton:
                ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                hit = self._hit_kp(dx, dy)
                if hit is not None:
                    self._drag_kp = hit
                    self._drag_start_disp = (dx, dy)
                    self._drag_cur_disp = (dx, dy)
                    self._drag_moved = False
                    self._rubber_band_active = False
                    self._primary_kp = hit
                    if ctrl:
                        self.keypoint_ctrl_clicked.emit(hit)
                    else:
                        self.keypoint_selected.emit(hit)
                else:
                    self._drag_kp = None
                    self._rubber_band_active = True
                    self._rubber_band_ctrl = ctrl
                    self._drag_start_disp = (dx, dy)
                    self._drag_cur_disp = (dx, dy)
                    self._drag_moved = False
                self.update()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._pan_active and (event.buttons() & Qt.MouseButton.MiddleButton):
            if self._pan_start_disp is not None and self._pan_start_rect is not None:
                pos = event.position()
                scale, _ox, _oy, _view = self._view_transform()
                if scale > 0:
                    fdx = (pos.x() - self._pan_start_disp[0]) / scale
                    fdy = (pos.y() - self._pan_start_disp[1]) / scale
                    # Grab-and-drag: content follows the cursor, so the view
                    # rect moves opposite to the drag direction.
                    self._zoom_rect = _panned_rect(self._pan_start_rect, -fdx, -fdy)
                    self.update()
            return

        if self._drag_kp is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = event.position()
            self._drag_cur_disp = (pos.x(), pos.y())
            if self._drag_start_disp is not None:
                ddx = pos.x() - self._drag_start_disp[0]
                ddy = pos.y() - self._drag_start_disp[1]
                if ddx ** 2 + ddy ** 2 >= _KP_DRAG_THRESHOLD ** 2:
                    self._drag_moved = True
            self.update()
        elif self._rubber_band_active and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = event.position()
            self._drag_cur_disp = (pos.x(), pos.y())
            if self._drag_start_disp is not None:
                ddx = pos.x() - self._drag_start_disp[0]
                ddy = pos.y() - self._drag_start_disp[1]
                if ddx ** 2 + ddy ** 2 >= _KP_DRAG_THRESHOLD ** 2:
                    self._drag_moved = True
            self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_active = False
            self._pan_start_disp = None
            self._pan_start_rect = None
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_kp is not None:
                if self._drag_moved and self._drag_cur_disp is not None:
                    new_x, new_y = self._display_to_full(*self._drag_cur_disp)
                    self.keypoint_moved.emit(self._drag_kp, new_x, new_y)
                self._drag_kp = None
                self._drag_start_disp = None
                self._drag_cur_disp = None
                self._drag_moved = False
                self.update()
            elif self._rubber_band_active:
                if self._drag_moved and self._drag_start_disp and self._drag_cur_disp:
                    x0, y0 = self._drag_start_disp
                    x1, y1 = self._drag_cur_disp
                    self.rubber_band_selected.emit(
                        min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1),
                        self._rubber_band_ctrl,
                    )
                elif self._drag_start_disp:
                    self.empty_area_clicked.emit(*self._drag_start_disp)
                self._rubber_band_active = False
                self._rubber_band_ctrl = False
                self._drag_start_disp = None
                self._drag_cur_disp = None
                self._drag_moved = False
                self.update()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        """Zoom/pan -- see "Zoom and pan in the camera crop views" in the
        design doc.

        Originally dispatched on `QWheelEvent.device().type()` to give mouse
        and touchpad different unmodified defaults, but real-hardware testing
        showed Windows doesn't preserve that distinction by the time Qt sees
        the event -- a Precision Touchpad's two-finger scroll arrives
        indistinguishable from a physical wheel (Windows synthesizes the same
        WM_MOUSEWHEEL/WM_MOUSEHWHEEL messages for both; there's no native
        gesture API in play for a plain two-finger scroll the way there is
        for pinch). So this uses one modifier-based mapping instead of trying
        to tell the devices apart: plain wheel/swipe zooms, `Shift`+wheel/swipe
        pans. That's device-agnostic by construction -- it doesn't matter
        which device produced the event. Mouse users still have middle-drag
        for pan without any modifier, since a physical wheel has no
        horizontal axis to pan with regardless.
        """
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        angle = event.angleDelta()
        pos = event.position()

        if not shift:
            steps = angle.y() / 120.0
            if steps == 0:
                event.ignore()
                return
            factor = _ZOOM_PER_WHEEL_STEP ** steps
            self._apply_zoom(factor, (pos.x(), pos.y()))
            event.accept()
            return

        # Shift+wheel/swipe: pan using whichever axes the event actually
        # carries -- a touchpad swipe supplies both x and y directly; a
        # mouse wheel only ever supplies y (no synthesized remap needed,
        # middle-drag already covers full 2D pan for mouse users).
        self._apply_pan_delta(angle.x() / 8.0, angle.y() / 8.0)
        event.accept()

    def _apply_zoom(self, factor: float, anchor_disp: tuple[float, float]) -> None:
        extent = self._pixmap_extent()
        if extent is None:
            return
        base_rect = self._zoom_rect or extent
        anchor_full = self._display_to_full(*anchor_disp)
        new_rect = _zoomed_rect(base_rect, factor, anchor_full)
        if new_rect is None:
            return
        # Zooming out past the pixmap's own extent is equivalent to fit mode
        # (_compute_view_transform already clamps to it) -- collapse back to
        # None so a subsequent zoom-in starts fresh from the *current* frame's
        # extent rather than a stale rect from a since-changed crop.
        x0, y0, x1, y1 = new_rect
        ex0, ey0, ex1, ey1 = extent
        if x0 <= ex0 and y0 <= ey0 and x1 >= ex1 and y1 >= ey1:
            self._zoom_rect = None
        else:
            self._zoom_rect = new_rect
        self.update()

    def _apply_pan_delta(self, disp_dx: float, disp_dy: float) -> None:
        extent = self._pixmap_extent()
        if extent is None or self._zoom_rect is None:
            # Not zoomed in -- the whole crop already fits the cell, so
            # panning has nothing new to reveal. Let it propagate (e.g. to an
            # enclosing scroll area) instead of silently doing nothing.
            return
        scale, _ox, _oy, _view = self._view_transform()
        if scale <= 0:
            return
        self._zoom_rect = _panned_rect(self._zoom_rect, -disp_dx / scale, -disp_dy / scale)
        self.update()

    def paintEvent(self, _event) -> None:
        from math import isnan

        from PySide6.QtCore import QRect

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cw, ch = self.width(), self.height()
        painter.fillRect(0, 0, cw, ch, QColor("#222"))

        def _draw_debug_label() -> None:
            # Diagnostic only -- names which crop-source layer produced the
            # currently-shown image, and whether black-fill was applied.
            # See "Debug overlay" in the design doc.
            if not self._debug_label:
                return
            fm = painter.fontMetrics()
            pad = 3
            tw = fm.horizontalAdvance(self._debug_label)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.drawRect(QRectF(pad, pad, tw + 2 * pad, fm.height() + 2 * pad))
            painter.setPen(QColor(255, 220, 0))
            painter.drawText(QPointF(2 * pad, pad + fm.ascent() + pad), self._debug_label)

        def _draw_placement_label() -> None:
            # Chain-placement mode: names which keypoint the next click will
            # set, e.g. "Left arm: elbow (2/5)" -- top-center, out of the way
            # of the debug label (top-left) and any per-keypoint name label.
            if not self._placement_label:
                return
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            fm = painter.fontMetrics()
            pad = 4
            text = self._placement_label
            tw = fm.horizontalAdvance(text)
            th = fm.height()
            x = (cw - tw) / 2 - pad
            y = 4.0
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 140, 0, 225))
            painter.drawRoundedRect(QRectF(x, y, tw + 2 * pad, th + 2 * pad), 3.0, 3.0)
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(QPointF(x + pad, y + pad + fm.ascent()), text)
            font.setBold(False)
            painter.setFont(font)

        if self._pixmap is None:
            painter.setPen(QColor("#666"))
            label = "generating…" if self._loading else "—"
            painter.drawText(QRectF(0, 0, cw, ch), Qt.AlignmentFlag.AlignCenter, label)
            _draw_debug_label()
            _draw_placement_label()
            painter.end()
            return

        scale, origin_x, origin_y, view = self._view_transform()
        vx0, vy0, vx1, vy1 = view
        # view is in full-frame pixel coords; convert to this pixmap's own
        # local pixel coords (may be a sub-region of it when zoomed) before
        # cropping -- clamped defensively even though _compute_view_transform
        # already clamps against the same extent this pixmap reports.
        pix_x0 = max(0, int(round((vx0 - self._x1) * self._src_scale)))
        pix_y0 = max(0, int(round((vy0 - self._y1) * self._src_scale)))
        pix_x1 = min(self._pixmap.width(), int(round((vx1 - self._x1) * self._src_scale)))
        pix_y1 = min(self._pixmap.height(), int(round((vy1 - self._y1) * self._src_scale)))
        if pix_x1 > pix_x0 and pix_y1 > pix_y0:
            cropped = self._pixmap.copy(QRect(pix_x0, pix_y0, pix_x1 - pix_x0, pix_y1 - pix_y0))
            disp_x = vx0 * scale + origin_x
            disp_y = vy0 * scale + origin_y
            disp_w = max(1, int(round((vx1 - vx0) * scale)))
            disp_h = max(1, int(round((vy1 - vy0) * scale)))
            scaled_pix = cropped.scaled(
                disp_w, disp_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(int(round(disp_x)), int(round(disp_y)), scaled_pix)

        def to_pt(u: float, v: float) -> QPointF:
            return QPointF(u * scale + origin_x, v * scale + origin_y)

        # ---- Detected keypoints (white connections, colour-coded dots) ----
        if self._show_detected and self._obs_kp is not None:
            kp = self._obs_kp
            n_kp = kp.shape[0]

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(210, 210, 210), 1.0))
            for a, b in _BODY_SKELETON:
                if a >= n_kp or b >= n_kp:
                    continue
                if a in self._hidden_kp or b in self._hidden_kp:
                    continue
                if float(kp[a, 2]) < 0.3 or float(kp[b, 2]) < 0.3:
                    continue
                painter.drawLine(
                    to_pt(float(kp[a, 0]), float(kp[a, 1])),
                    to_pt(float(kp[b, 0]), float(kp[b, 1])),
                )

            painter.setPen(Qt.PenStyle.NoPen)
            for i in range(n_kp):
                if i in self._hidden_kp:
                    continue
                conf = float(kp[i, 2])
                kp_x, kp_y = float(kp[i, 0]), float(kp[i, 1])
                if conf < 0.1:
                    if self._edit_mode and kp_x > 0.0:
                        pt = to_pt(kp_x, kp_y)
                        if 0 <= pt.x() <= cw and 0 <= pt.y() <= ch:
                            painter.setBrush(QColor(80, 80, 80))
                            painter.drawEllipse(pt, 4.0, 4.0)
                    continue
                if (not self._edit_mode
                        and self._outlier_kp_mask is not None
                        and i < len(self._outlier_kp_mask)
                        and self._outlier_kp_mask[i]):
                    painter.setBrush(QColor(120, 120, 120))  # grey — rejected by tracker
                elif conf >= 0.5:
                    painter.setBrush(QColor(0, 220, 60))    # green
                elif conf >= 0.3:
                    painter.setBrush(QColor(255, 200, 0))   # yellow
                else:
                    painter.setBrush(QColor(220, 40, 40))   # red
                painter.drawEllipse(to_pt(kp_x, kp_y), 5.0, 5.0)

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
                    painter.drawEllipse(to_pt(mu, mv), 5.0, 5.0)

        # ---- Edit mode: trail overlay ----
        if self._edit_mode and self._trail is not None:
            trail = self._trail

            def _draw_trail_seg(points, color):
                if not points:
                    return
                pts_d = [to_pt(p.x, p.y) for p in points]
                if len(pts_d) >= 2:
                    pen = QPen(color)
                    pen.setWidthF(_TRAIL_LINE_WIDTH)
                    painter.setPen(pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawPolyline(pts_d)
                painter.setPen(Qt.PenStyle.NoPen)
                for pt, tp in zip(pts_d, points):
                    painter.setBrush(_TRAIL_GHOST_COLOR if tp.is_ghost else color)
                    painter.drawEllipse(pt, float(_TRAIL_DOT_R), float(_TRAIL_DOT_R))

            _draw_trail_seg(trail.past, _TRAIL_PAST_COLOR)
            _draw_trail_seg(trail.future, _TRAIL_FUTURE_COLOR)

        # ---- Edit mode: frame-range highlight rings (white, small) ----
        if self._edit_mode and self._range_pts:
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for rx, ry in self._range_pts:
                painter.drawEllipse(to_pt(rx, ry), 7.0, 7.0)

        # ---- Edit mode: selection rings (all selected kp) ----
        if self._edit_mode and self._sel_kp_set and self._obs_kp is not None:
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for sel in self._sel_kp_set:
                if sel >= self._obs_kp.shape[0]:
                    continue
                if (sel == self._drag_kp and self._drag_moved
                        and self._drag_cur_disp is not None):
                    pt = QPointF(self._drag_cur_disp[0], self._drag_cur_disp[1])
                else:
                    pt = to_pt(float(self._obs_kp[sel, 0]), float(self._obs_kp[sel, 1]))
                painter.drawEllipse(pt, 9.0, 9.0)

            # Name label for primary kp only
            pri = self._primary_kp
            if (pri is not None and pri < self._obs_kp.shape[0]
                    and self._selected_kp_name):
                if (pri == self._drag_kp and self._drag_moved
                        and self._drag_cur_disp is not None):
                    pt = QPointF(self._drag_cur_disp[0], self._drag_cur_disp[1])
                else:
                    pt = to_pt(float(self._obs_kp[pri, 0]), float(self._obs_kp[pri, 1]))
                from PySide6.QtGui import QFont as _QFont
                lbl_font = _QFont()
                lbl_font.setPointSize(8)
                lbl_font.setBold(False)
                painter.setFont(lbl_font)
                fm = painter.fontMetrics()
                label = self._selected_kp_name
                text_w = fm.horizontalAdvance(label)
                text_h = fm.height()
                lx = pt.x() + 10
                ly = pt.y() - text_h / 2
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(0, 0, 0, 170))
                painter.drawRoundedRect(
                    QRectF(lx - 2, ly - 1, text_w + 5, text_h + 2), 2.0, 2.0
                )
                painter.setPen(QColor(255, 255, 255))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawText(QPointF(lx, ly + fm.ascent()), label)

        # ---- Edit mode: rubber-band selection rect ----
        if (self._edit_mode and self._rubber_band_active and self._drag_moved
                and self._drag_start_disp and self._drag_cur_disp):
            x0, y0 = self._drag_start_disp
            x1, y1 = self._drag_cur_disp
            painter.setPen(QPen(QColor(100, 180, 255), 1.0, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(100, 180, 255, 30))
            painter.drawRect(QRectF(
                min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
            ))

        _draw_debug_label()
        _draw_placement_label()
        painter.end()


class _CropCell(QWidget):
    """One camera cell in the crop grid: name label + image canvas."""

    _IMG_H = 300
    maximize_requested = Signal()

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        name_lbl = QLabel(label)
        name_lbl.setStyleSheet("font-size: 10px; font-weight: bold;")

        self._max_btn = QToolButton()
        self._max_btn.setText("⤢")
        self._max_btn.setFixedSize(16, 16)
        self._max_btn.setStyleSheet("font-size: 9px; border: none; padding: 0;")
        self._max_btn.setToolTip("Maximize / restore this camera view")
        self._max_btn.clicked.connect(self.maximize_requested)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(2)
        title_row.addWidget(name_lbl, stretch=1)
        title_row.addWidget(self._max_btn)

        title_w = QWidget()
        title_w.setMaximumHeight(18)
        title_w.setLayout(title_row)

        self._canvas = _ImageCanvas(min_h=self._IMG_H)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(1)
        vbox.addWidget(title_w)
        vbox.addWidget(self._canvas, stretch=1)

        self.show_empty()

    def set_is_maximized(self, maximized: bool) -> None:
        self._max_btn.setText("⤡" if maximized else "⤢")

    def show_empty(self) -> None:
        self._canvas.show_empty()

    def show_loading(self) -> None:
        self._canvas.show_loading()

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

    def set_edit_mode(self, enabled: bool) -> None:
        self._canvas.set_edit_mode(enabled)

    def set_placement_active(self, active: bool) -> None:
        self._canvas.set_placement_active(active)

    def set_placement_label(self, text: str | None) -> None:
        self._canvas.set_placement_label(text)

    def set_trail(self, trail) -> None:
        self._canvas.set_trail(trail)

    def set_range_highlights(self, pts: list[tuple[float, float]]) -> None:
        self._canvas.set_range_highlights(pts)

    def set_selection(
        self, primary: int | None, sel_indices: frozenset[int], name: str | None = None
    ) -> None:
        self._canvas.set_selection(primary, sel_indices, name)

    def set_hidden(self, hidden: frozenset[int]) -> None:
        self._canvas.set_hidden(hidden)

    def set_debug_label(self, text: "str | None") -> None:
        self._canvas.set_debug_label(text)

    def set_selected_kp(
        self, idx: int | None, show_ring: bool = True, name: str | None = None
    ) -> None:
        self._canvas.set_selected_kp(idx, show_ring=show_ring, name=name)


# Keeps running _TrackingRunLoader instances alive at the Python level until their
# OS thread has fully exited.  Without this, dropping the widget's self._loader
# reference while run() is still in its finally-block would GC the QThread Python
# wrapper and call the C++ destructor on a still-running thread.
_ACTIVE_LOADERS: set["_TrackingRunLoader"] = set()


class _TrackingRunLoader(QThread):
    """Loads tracking run overlay data (obs blobs + FK projections) on a background thread.

    All expensive work — blob deserialization, undistortion, forward kinematics — runs
    here so the UI thread stays responsive while the page opens.

    Emits ``loaded`` with the four result dicts when complete, or with empty dicts if
    loading fails.
    """

    loaded = Signal(list, dict, dict, list, dict)
    # args: tracking_timestamps, marker_proj, joint_proj, bone_pairs, outlier_masks

    def __init__(self, db_path: str, run_id: str) -> None:
        super().__init__()   # no Qt parent — lifetime managed via _ACTIVE_LOADERS
        self._db_path = db_path
        self._run_id = run_id
        _ACTIVE_LOADERS.add(self)

    def _on_finished(self) -> None:
        """Connected to finished(). Removes from active set and schedules C++ cleanup."""
        _ACTIVE_LOADERS.discard(self)
        self.deleteLater()

    def run(self) -> None:  # noqa: C901
        import json
        import sqlite3 as _sqlite3
        import numpy as np
        from posetrak.db.skeleton_layout import SkeletonLayout

        conn = None
        try:
            conn = _sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            conn.row_factory = _sqlite3.Row
            self._do_load(conn, np, json, SkeletonLayout)
        except Exception:
            self.loaded.emit([], {}, {}, [], {})
        finally:
            if conn is not None:
                conn.close()

    def _do_load(self, conn, np, json, SkeletonLayout) -> None:
        run_id = self._run_id
        run = conn.execute(
            "SELECT active_camera_ids, marker_names, skeleton_id, "
            "       extrinsic_calibration_id, observation_sequence_id "
            "FROM tracking_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        if run is None:
            self.loaded.emit([], {}, {}, [], {})
            return

        cam_labels: list[str] = json.loads(run["active_camera_ids"] or "[]")
        marker_names: list[str] = json.loads(run["marker_names"] or "[]")
        n_cams, n_markers = len(cam_labels), len(marker_names)
        if n_cams == 0 or n_markers == 0:
            self.loaded.emit([], {}, {}, [], {})
            return

        # Map camera label → camera_instance_id
        placeholders = ",".join("?" * n_cams)
        label_to_cam_id: dict[str, str] = {}
        for r in conn.execute(
            f"SELECT id, label FROM camera_instances WHERE label IN ({placeholders})",
            cam_labels,
        ):
            label_to_cam_id[r["label"]] = r["id"]

        # Tracking timestamps for nearest-step lookup
        ts_rows = conn.execute(
            "SELECT tracker_step, timestamp_s FROM tracking_results "
            "WHERE run_id=? AND person_id=0 AND is_smoothed=0 ORDER BY tracker_step",
            (run_id,),
        ).fetchall()
        tracking_timestamps = sorted(
            (r["timestamp_s"], r["tracker_step"]) for r in ts_rows
        )

        # Skeleton → bone pairs
        skel = conn.execute(
            "SELECT yaml_content FROM skeletons WHERE id=?",
            (run["skeleton_id"],),
        ).fetchone()
        if not skel or not skel["yaml_content"]:
            self.loaded.emit([], {}, {}, [], {})
            return
        layout = SkeletonLayout(skel["yaml_content"])
        bone_pairs = layout.bone_pairs

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

        # Extrinsics: cam_instance_id → (R, t)
        ext_id = run["extrinsic_calibration_id"]
        cam_extrinsics: dict[str, tuple] = {}
        if ext_id:
            for r in conn.execute(
                "SELECT camera_instance_id, R, t FROM extrinsic_entries "
                "WHERE extrinsic_calibration_id = ?",
                (ext_id,),
            ):
                cam_extrinsics[r["camera_instance_id"]] = (
                    np.frombuffer(bytes(r["R"]), dtype="<f8").reshape(3, 3),
                    np.frombuffer(bytes(r["t"]), dtype="<f8"),
                )

        # Intrinsics: cam_instance_id → dict
        seq = conn.execute(
            "SELECT shot_id FROM pose_observation_sequences WHERE id=?",
            (run["observation_sequence_id"],),
        ).fetchone()
        cam_intrinsics: dict[str, dict] = {}
        if seq:
            for r in conn.execute(
                "SELECT cv.camera_instance_id, ic.fx, ic.fy, ic.cx, ic.cy, "
                "       ic.dist_coeffs, ic.matrix_original "
                "FROM capture_videos cv "
                "JOIN intrinsics_calibrations ic ON ic.id = cv.intrinsics_calibration_id "
                "WHERE cv.shot_id = ?",
                (seq["shot_id"],),
            ):
                K_orig = (
                    np.frombuffer(bytes(r["matrix_original"]), dtype="<f8").reshape(3, 3)
                    if r["matrix_original"] else None
                )
                dist = (
                    np.frombuffer(bytes(r["dist_coeffs"]), dtype="<f8")
                    if r["dist_coeffs"] else None
                )
                cam_intrinsics[r["camera_instance_id"]] = {
                    "fx": r["fx"], "fy": r["fy"], "cx": r["cx"], "cy": r["cy"],
                    "K_orig": K_orig, "dist": dist,
                }

        # Obs blobs → outlier flags only. Marker *positions* are projected
        # from the posterior state below, alongside the joints -- obs_blob's
        # own pred_x/pred_y is the UKF's PRE-update prediction (the sigma
        # points' measurement mean before the Kalman correction, used for
        # gating/innovation), which can legitimately sit far from the
        # corrected skeleton on a fast-motion frame. Projecting both markers
        # and joints from the same posterior state keeps the two overlays
        # from ever visually disagreeing.
        outlier_masks: dict[str, dict[int, object]] = {}
        if mi_to_coco:
            obs_rows = conn.execute(
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
                    mask = np.zeros(n_coco_kp, dtype=bool)
                    for mi, coco_id in mi_to_coco.items():
                        is_out = obs[ci, mi, 6]
                        if np.isfinite(is_out) and is_out != 0.0:
                            mask[coco_id] = True
                    outlier_masks.setdefault(cam_id, {})[step] = mask

        # State blobs → joint + marker projections via FK, both from the same
        # posterior (corrected) state.
        joint_proj: dict[str, dict[int, dict]] = {}
        marker_proj: dict[str, dict[int, object]] = {}
        state_rows = conn.execute(
            "SELECT tracker_step, state FROM tracking_results "
            "WHERE run_id=? AND person_id=0 AND is_smoothed=0 ORDER BY tracker_step",
            (run_id,),
        ).fetchall()
        for state_row in state_rows:
            step = state_row["tracker_step"]
            try:
                decoded = layout.decode_state_blob(bytes(state_row["state"]))
                transforms = layout.compute_joint_transforms(decoded)
                marker_world = layout.compute_marker_positions(decoded)
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
                dist_c = intr.get("dist")
                fx, fy, cx_k, cy_k = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
                use_distortion = K_orig is not None and dist_c is not None

                def _project(p_world):
                    if use_distortion:
                        return _project_point_distorted(p_world, R, t, K_orig, dist_c)
                    p_cam = R @ p_world + t
                    if p_cam[2] <= 1e-3:
                        return None
                    return (
                        fx * p_cam[0] / p_cam[2] + cx_k,
                        fy * p_cam[1] / p_cam[2] + cy_k,
                    )

                joint_xy: dict[str, object] = {}
                for jname, T in transforms.items():
                    uv = _project(T[:3, 3])
                    if uv is None:
                        continue
                    joint_xy[jname] = np.array(uv)
                joint_proj.setdefault(cam_id, {})[step] = joint_xy

                marker_xy = np.full((n_markers, 2), np.nan, dtype=np.float64)
                for mi, mname in enumerate(marker_names):
                    p_world = marker_world.get(mname)
                    if p_world is None:
                        continue
                    uv = _project(p_world)
                    if uv is None:
                        continue
                    marker_xy[mi] = uv
                marker_proj.setdefault(cam_id, {})[step] = marker_xy

        self.loaded.emit(tracking_timestamps, marker_proj, joint_proj, bone_pairs, outlier_masks)


class _KeypointPickerPanel(QWidget):
    """Hierarchical keypoint list for the "pick a keypoint, then click a
    camera cell to place/move it there" workflow -- see "Keypoint-placement
    toolbar" in the design doc. Generalizes Phase 7's ghost-frame
    click-to-place (limited to the current primary selection and frames with
    no observation at all) to any keypoint, on any frame.

    Uses the same `pose_model.tree_groups` partition the timeline's row tree
    derives (`build_rows` in keypoint_timeline_widget.py), but as a native
    `QTreeWidget` rather than that module's custom-painted canvas, since this
    panel only needs plain click-to-select, not per-frame status columns.
    """

    keypoint_picked = Signal(int)  # kp_idx
    group_picked = Signal(str)     # group label -- selects the group, doesn't arm placement

    def __init__(self, pose_model: PoseModel, parent=None) -> None:
        super().__init__(parent)
        self._pose_model = pose_model
        self._active_item: QTreeWidgetItem | None = None

        title = QLabel("<b>Place keypoint</b>")
        title.setStyleSheet("font-size: 10px;")
        hint = QLabel("Pick a keypoint, then click a camera view to place it. Esc cancels.")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 9px; color: #888;")

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.itemClicked.connect(self._on_item_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(self._tree, stretch=1)

        self._rebuild()

    def _rebuild(self) -> None:
        self._tree.clear()
        covered: set[int] = set()

        def _add_group(label: str, indices: list[int]) -> None:
            group_item = QTreeWidgetItem([label])
            group_item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # header only, not clickable-to-place
            self._tree.addTopLevelItem(group_item)
            for kp_idx in indices:
                leaf = QTreeWidgetItem([self._pose_model.name_of(kp_idx)])
                leaf.setData(0, Qt.ItemDataRole.UserRole, kp_idx)
                group_item.addChild(leaf)

        for group_name in self._pose_model.tree_groups:
            idx = sorted(self._pose_model.group_indices(group_name))
            covered.update(idx)
            _add_group(group_name, idx)

        leftover = sorted(self._pose_model.all_indices - covered)
        if leftover:
            _add_group("Other", leftover)

        self._tree.collapseAll()

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        kp_idx = item.data(0, Qt.ItemDataRole.UserRole)
        if kp_idx is None:
            # Group header: select the group for the existing multi-select
            # operations (nudge/toggle/etc. on the whole group) -- doesn't
            # arm placement mode, since a click only has one location to
            # place a single keypoint at.
            self.group_picked.emit(item.text(0))
            return
        self.keypoint_picked.emit(int(kp_idx))

    def set_active(self, kp_idx: int | None) -> None:
        """Highlight the currently-pending keypoint (or clear the highlight)."""
        if self._active_item is not None:
            self._active_item.setBackground(0, QColor("transparent"))
            self._active_item = None
        if kp_idx is None:
            return
        root = self._tree.invisibleRootItem()
        for gi in range(root.childCount()):
            group_item = root.child(gi)
            for li in range(group_item.childCount()):
                leaf = group_item.child(li)
                if leaf.data(0, Qt.ItemDataRole.UserRole) == kp_idx:
                    leaf.setBackground(0, QColor(255, 200, 0, 90))
                    self._active_item = leaf
                    group_item.setExpanded(True)
                    return


class PersonCropGridWidget(QWidget):
    """Grid of per-camera person crop images with a time scrubber.

    Reads JPEG crops from frame_cache_entries, overlays pose_observations
    keypoints, and shows all cameras simultaneously.  One extra placeholder
    cell is reserved for a future 3D tracking view.
    """

    time_changed = Signal(float)    # emitted on every slider move (absolute timestamp_s)
    status_message = Signal(str)    # short notification for the main window status bar

    # Idea 3 (automated post-edit hand redetection): class-level defaults so
    # that test fixtures constructing this widget via
    # `PersonCropGridWidget.__new__(...)` (bypassing __init__ -- an
    # established pattern across this test suite) still see these as None/{}
    # rather than raising AttributeError, without needing every such fixture
    # updated individually. Real instances always get their own per-instance
    # values from __init__ below.
    _hand_redetect: "HandRedetectWorker | None" = None
    _hand_redetect_timers: dict = {}
    # "Auto-detect" vs "keep existing state" toggle (see the design doc's
    # "Idea 3" section) -- None until _build() constructs the real checkbox;
    # treated as "on" when None so a test fixture without it still exercises
    # the default, common-case behavior.
    _auto_redetect_chk: "QCheckBox | None" = None

    # Number-key shortcuts for limb groups (see _handle_key): plain key
    # toggles show/hide of the limb's keypoints (same rule as the timeline's
    # eye icon), Shift+key isolates it (hides everything else), Ctrl+key
    # starts "Set limb..." chain placement for it (if the limb has a chain
    # defined -- see kp_models.py's limb_chains; hands don't yet).
    _LIMB_SHORTCUT_KEYS: dict[int, str] = {
        Qt.Key.Key_1: "Face",
        Qt.Key.Key_2: "Left arm",
        Qt.Key.Key_3: "Left hand",
        Qt.Key.Key_4: "Right arm",
        Qt.Key.Key_5: "Right hand",
        Qt.Key.Key_6: "Left leg",
        Qt.Key.Key_7: "Right leg",
    }

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
        self._show_seg: QCheckBox | None = None         # segmentation mask overlay
        self._show_debug: QCheckBox | None = None       # per-cell crop-source-layer label
        self._edit_btn: QPushButton | None = None       # edit mode toggle
        # Edit mode state
        self._edit_mode: bool = False
        self._sel_kp_indices: set[int] = set()   # all selected kp indices
        self._primary_kp_idx: int | None = None   # primary kp (trail, nudge, toggle)
        self._sel_cam_idx: int | None = None      # camera that last emitted keypoint_selected
        # Keypoint-placement mode (see _KeypointPickerPanel): the keypoint
        # picked from the toolbar, waiting for a canvas click to place it.
        # One-shot -- cleared after a placement, by Esc, by picking a
        # different keypoint, or on exiting edit mode. Chain-placement mode
        # (see _start_chain_placement) also drives this same field, one limb
        # keypoint at a time, but re-arms itself after each placement instead
        # of clearing it -- see _chain_limb below.
        self._pending_place_kp_idx: int | None = None
        # Chain-placement mode state: set a whole limb in shoulder->wrist (or
        # hip->toe) order with one click per keypoint. _chain_limb is the
        # active limb name (a key into PoseModel.limb_chains) or None when no
        # chain is in progress; _chain_indices is that limb's ordered
        # keypoint indices (resolved once, at chain start); _chain_pos is the
        # index into _chain_indices of the keypoint the next click will set.
        self._chain_limb: str | None = None
        self._chain_indices: list[int] = []
        self._chain_pos: int = 0
        self._chain_btn: QPushButton | None = None
        self._kp_picker: "_KeypointPickerPanel | None" = None
        # svid → seg_quality_run_id (or None when no masks are available)
        self._seg_sources: dict[str, str | None] = {}
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
        self._backfill: CropBackfillWorker | None = None
        self._wide_crop_mgr: "FrameCropCacheManager | None" = None
        # Idea 3 (automated post-edit hand redetection): background worker +
        # debounce timers keyed by (camera_instance_id, video_frame, side).
        self._hand_redetect: "HandRedetectWorker | None" = None
        self._hand_redetect_timers: dict[tuple[str, int, str], QTimer] = {}
        self._loader: _TrackingRunLoader | None = None
        self._maximized_idx: int | None = None
        self._pose_model: PoseModel = _get_pose_model(None)  # updated in _build
        # Copy/paste clipboard: kp_idx → (x, y); None until first copy
        self._clipboard: dict[int, tuple[float, float]] | None = None
        self._clipboard_cam_idx: int | None = None
        # Frame-range selection for interpolation (slider values in ms from t_start)
        self._range_start_v: int | None = None
        self._range_end_v: int | None = None
        # Timeline dope-sheet widget (Phase 12): cam_instance_id → frame → int8[N] status,
        # plus a cross-camera inlier count shared by all cameras.
        self._timeline: KeypointTimelineWidget | None = None
        self._timeline_status_by_cam: dict[str, dict[int, object]] = {}
        self._timeline_inlier_counts: dict[int, object] = {}
        # Keypoint indices hidden via the timeline's eye icon: drawn nowhere,
        # excluded from selection/hit-testing/interpolation. UI-only, not
        # persisted — resets each time the editor is reopened.
        self._hidden_kp_indices: set[int] = set()
        self._build()

    def _build(self) -> None:
        import numpy as np
        from app.setup.db_context import SyncPoint, SyncTable

        seq = self._conn.execute(
            "SELECT detection_run_id, shot_id, sync_config_id, time_start_s, time_end_s, pose_model "
            "FROM pose_observation_sequences WHERE id = ?",
            (self._sequence_id,),
        ).fetchone()
        if seq is None:
            QVBoxLayout(self).addWidget(QLabel("Sequence not found."))
            return

        self._det_run_id = seq["detection_run_id"]
        self._t_start = float(seq["time_start_s"])
        self._t_end = float(seq["time_end_s"])
        self._pose_model = _get_pose_model(seq["pose_model"])

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

        # Find seg_quality_run_id with the most masks for each camera, and store
        # video dimensions for coordinate scaling (masks may be at a lower resolution
        # than the video, e.g. FHD masks on 4K cameras).
        self._video_dims: dict[str, tuple[int, int]] = {}  # svid → (w, h)
        for cam in self._cameras:
            svid = cam["shot_video_id"]
            row_sq = self._conn.execute(
                "SELECT seg_quality_run_id, COUNT(*) n FROM seg_masks "
                "WHERE shot_video_id=? GROUP BY seg_quality_run_id ORDER BY n DESC LIMIT 1",
                (svid,),
            ).fetchone()
            self._seg_sources[svid] = row_sq["seg_quality_run_id"] if row_sq else None
            dim_row = self._conn.execute(
                "SELECT COALESCE(cm.width_px, ic.image_width) AS vw, "
                "       COALESCE(cm.height_px, ic.image_height) AS vh "
                "FROM capture_videos cv "
                "LEFT JOIN camera_modes cm ON cm.id = cv.camera_mode_id "
                "LEFT JOIN intrinsics_calibrations ic ON ic.id = cv.intrinsics_calibration_id "
                "WHERE cv.id = ?",
                (svid,),
            ).fetchone()
            if dim_row and dim_row["vw"] and dim_row["vh"]:
                self._video_dims[svid] = (int(dim_row["vw"]), int(dim_row["vh"]))

        # Pre-load merged keypoints (pose_observations with pose_observation_edits applied)
        from app.pose.db_cache import read_observations_with_edits
        for cam in self._cameras:
            self._obs_kp[cam["camera_instance_id"]] = read_observations_with_edits(
                self._conn, self._sequence_id, cam["camera_instance_id"]
            )

        # Timeline dope-sheet status, one camera at a time (Phase 12).
        for cam in self._cameras:
            self._refresh_timeline_status(cam["camera_instance_id"])

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
        # Start with 3 columns; resizeEvent adjusts when the widget is shown.
        self._ncols = max(2, min(n_cells, 3))
        ncols = self._ncols

        self._grid_container = QWidget()
        self._grid = QGridLayout(self._grid_container)
        self._grid.setSpacing(4)

        for i, cam in enumerate(self._cameras):
            row, col = divmod(i, ncols)
            cell = _CropCell(cam["label"])
            self._cells.append(cell)
            self._grid.addWidget(cell, row, col)
            self._grid.setColumnStretch(col, 1)
            cell._canvas.keypoint_selected.connect(
                lambda idx, i=i: self._on_kp_selected(i, idx)
            )
            cell._canvas.keypoint_ctrl_clicked.connect(
                lambda idx, i=i: self._on_kp_ctrl_clicked(i, idx)
            )
            cell._canvas.keypoint_deselected.connect(
                lambda i=i: self._on_kp_deselected(i)
            )
            cell._canvas.keypoint_moved.connect(
                lambda idx, x, y, i=i: self._on_kp_moved(i, idx, x, y)
            )
            cell._canvas.empty_area_clicked.connect(
                lambda dx, dy, i=i: self._on_empty_area_clicked(i, dx, dy)
            )
            cell._canvas.rubber_band_selected.connect(
                lambda x1, y1, x2, y2, ctrl, i=i: self._on_rubber_band_selected(i, x1, y1, x2, y2, ctrl)
            )
            cell._canvas.context_menu_requested.connect(
                lambda hit, dx, dy, i=i: self._on_context_menu_requested(i, hit, dx, dy)
            )
            cell._canvas.placement_clicked.connect(
                lambda dx, dy, i=i: self._on_placement_clicked(i, dx, dy)
            )
            cell._canvas.installEventFilter(self)
            cell.maximize_requested.connect(lambda i=i: self._on_maximize_requested(i))

        r3d, c3d = divmod(len(self._cameras), ncols)
        ph = QLabel("3D view\n(coming soon)")
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet("color: #888; border: 1px dashed #555;")
        ph.setMinimumHeight(_CropCell._IMG_H)
        self._grid.addWidget(ph, r3d, c3d)
        self._grid.setColumnStretch(c3d, 1)
        self._3d_ph = ph

        nrows = ceil(n_cells / ncols)
        for r in range(nrows):
            self._grid.setRowStretch(r, 1)

        dur_ms = max(1, int((self._t_end - self._t_start) * 1000))
        _fps_vals = [float(r["actual_fps"]) for r in sp_rows if r["actual_fps"]]
        frame_step_ms = max(1, round(1000.0 / max(_fps_vals))) if _fps_vals else 8
        # Headless value/range holder — the timeline widget is now the only visible
        # scrub control (see _on_timeline_scrub); this keeps _on_slider, seek(),
        # _extend_range_left/right, and _compute_range_highlights unchanged, since
        # they only need value()/setValue()/singleStep()/min/max, not visibility.
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(dur_ms)
        self._slider.setSingleStep(frame_step_ms)
        self._slider.setValue(0)
        self._slider.valueChanged.connect(self._on_slider)

        self._time_label = QLabel(_fmt_time(self._t_start))
        self._time_label.setMinimumWidth(70)

        self._show_detected = QCheckBox("Detected keypoints")
        self._show_detected.setChecked(True)
        self._show_detected.stateChanged.connect(lambda _: self._load_frame(self._current_t))
        self._show_tracked = QCheckBox("Tracked skeleton")
        self._show_tracked.setChecked(True)
        self._show_tracked.stateChanged.connect(lambda _: self._load_frame(self._current_t))
        has_seg = any(v is not None for v in self._seg_sources.values())
        self._show_seg = QCheckBox("Segmentation")
        self._show_seg.setChecked(has_seg)
        self._show_seg.setEnabled(has_seg)
        self._show_seg.stateChanged.connect(lambda _: self._load_frame(self._current_t))
        self._show_debug = QCheckBox("Debug")
        self._show_debug.setChecked(False)
        self._show_debug.setToolTip(
            "Show which crop-source layer produced each cell's image, and "
            "whether black-fill was applied. Diagnostic only."
        )
        self._show_debug.stateChanged.connect(lambda _: self._load_frame(self._current_t))

        self._edit_btn = QPushButton("Edit keypoints")
        self._edit_btn.setCheckable(True)
        self._edit_btn.setChecked(False)
        self._edit_btn.setToolTip("Toggle keypoint editing mode: click to select, drag to move")
        self._edit_btn.toggled.connect(self._set_edit_mode)

        self._chain_btn = QPushButton("Set limb…")
        self._chain_btn.setToolTip(
            "Click through a limb's keypoints in order (shoulder→elbow→wrist→...\n"
            "or hip→knee→ankle→...). Space skips the current keypoint, Esc ends it."
        )
        self._chain_btn.setEnabled(False)
        # Clicking it must not leave it holding keyboard focus -- Space is
        # the "skip keypoint" shortcut during a chain, and a focused
        # QPushButton treats Space as "click me" (reopening this menu)
        # instead of letting the shortcut reach _handle_key.
        self._chain_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._chain_btn.clicked.connect(self._show_chain_menu)

        self._auto_redetect_chk = QCheckBox("Auto-redetect hands")
        self._auto_redetect_chk.setChecked(True)
        self._auto_redetect_chk.setEnabled(False)  # enabled alongside edit mode, in _set_edit_mode
        self._auto_redetect_chk.setToolTip(
            "\"Auto-detect\": editing a wrist/elbow redetects that hand in the\n"
            "background and clears any earlier \"disable\" edits on its fingers\n"
            "(a deliberate reposition edit is never touched). Uncheck for\n"
            "\"keep existing state\": nothing is redetected or touched\n"
            "automatically -- useful while troubleshooting by hand (e.g.\n"
            "disabling a camera's keypoints to see if that alone fixes tracking)."
        )

        overlay_row = QHBoxLayout()
        overlay_row.addWidget(QLabel("Show:"))
        overlay_row.addWidget(self._show_detected)
        overlay_row.addWidget(self._show_tracked)
        overlay_row.addWidget(self._show_seg)
        overlay_row.addWidget(self._show_debug)
        overlay_row.addStretch()
        overlay_row.addWidget(self._time_label)
        overlay_row.addWidget(self._edit_btn)
        overlay_row.addWidget(self._chain_btn)
        overlay_row.addWidget(self._auto_redetect_chk)

        # Maximized-view container: big cell on left, thumbnail strip on right.
        self._max_placeholder = QWidget()  # occupies left slot when not maximized
        self._thumb_container = QWidget()
        self._thumb_layout = QVBoxLayout(self._thumb_container)
        self._thumb_layout.setSpacing(4)
        self._thumb_layout.setContentsMargins(0, 0, 0, 0)
        thumb_scroll = QScrollArea()
        thumb_scroll.setWidget(self._thumb_container)
        thumb_scroll.setWidgetResizable(True)
        self._max_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._max_splitter.addWidget(self._max_placeholder)
        self._max_splitter.addWidget(thumb_scroll)
        self._max_splitter.setStretchFactor(0, 3)
        self._max_splitter.setStretchFactor(1, 1)
        max_container = QWidget()
        max_h = QHBoxLayout(max_container)
        max_h.setContentsMargins(0, 0, 0, 0)
        max_h.addWidget(self._max_splitter)

        # Wrap the grid in a scroll area so the window can always be made smaller
        # without the grid's minimum height forcing the window taller than the screen.
        # The slider and controls below are outside the scroll area and always visible.
        grid_scroll = QScrollArea()
        grid_scroll.setWidget(self._grid_container)
        grid_scroll.setWidgetResizable(True)
        grid_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        grid_scroll.setMinimumHeight(0)

        self._stack = QStackedWidget()
        self._stack.addWidget(grid_scroll)   # page 0: scrollable grid
        self._stack.addWidget(max_container) # page 1: maximized view

        self._timeline = KeypointTimelineWidget(self._pose_model, self._cameras)
        self._timeline.camera_changed.connect(self._on_timeline_camera_changed)
        self._timeline.rubber_band_selected.connect(self._on_timeline_rubber_band)
        self._timeline.keyframe_toggled.connect(self._on_timeline_keyframe_toggle)
        self._timeline.time_scrubbed.connect(self._on_timeline_scrub)
        self._timeline.visibility_toggled.connect(self._on_timeline_visibility_toggled)
        self._timeline.select_to_marker_requested.connect(self._on_timeline_select_to_marker)
        self._timeline.select_all_to_marker_requested.connect(self._on_timeline_select_all_to_marker)
        self._timeline.disable_selected_requested.connect(lambda: self._set_outlier_selected(True))
        self._timeline.enable_selected_requested.connect(lambda: self._set_outlier_selected(False))
        self._timeline.interpolate_missing_requested.connect(self._interpolate_missing_range)
        if self._cameras:
            # set_time_range only needs to run once: t_start/t_end/sync_table
            # are the same for every camera in a sequence, so re-running it on
            # every camera switch or post-edit status refresh would reset the
            # user's zoom/pan for no reason (see _push_timeline_camera_data).
            self._timeline.set_time_range(
                self._t_start, self._t_end, self._cameras[0]["shot_video_id"], self._sync_table
            )
            self._push_timeline_camera_data(0)

        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)
        top_layout.addWidget(self._stack, stretch=1)
        top_layout.addLayout(overlay_row)

        # Keypoint-placement toolbar: a narrow sidebar next to the grid, only
        # useful (and only shown) in edit mode -- see "Keypoint-placement
        # toolbar" in the design doc.
        self._kp_picker = _KeypointPickerPanel(self._pose_model)
        self._kp_picker.keypoint_picked.connect(self._on_kp_picked)
        self._kp_picker.group_picked.connect(self._select_group)
        self._kp_picker.setVisible(False)
        self._kp_picker.setMinimumWidth(120)
        self._kp_picker.setMaximumWidth(220)

        grid_and_picker = QSplitter(Qt.Orientation.Horizontal)
        grid_and_picker.addWidget(top_container)
        grid_and_picker.addWidget(self._kp_picker)
        grid_and_picker.setStretchFactor(0, 1)
        grid_and_picker.setStretchFactor(1, 0)
        grid_and_picker.setCollapsible(0, False)
        grid_and_picker.setSizes([3000, 0])

        # Vertical splitter: the video grid gets all the space by default (the
        # timeline starts collapsed to its tab-row height); dragging the handle
        # resizes the timeline once it's expanded.
        self._main_splitter = QSplitter(Qt.Orientation.Vertical)
        self._main_splitter.addWidget(grid_and_picker)
        self._main_splitter.addWidget(self._timeline)
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 0)
        self._main_splitter.setCollapsible(0, False)
        # QSplitter can squash a "collapsible" pane to 0px on first layout even
        # when it has an explicit min/maxHeight — without this the timeline
        # (including its always-visible tab row + ruler) doesn't render at all
        # until the user drags the splitter handle.
        self._main_splitter.setCollapsible(1, False)
        self._main_splitter.setSizes([3000, self._timeline.maximumHeight()])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self._main_splitter, stretch=1)

        # Low-res backfill runs regardless of edit mode -- a view-mode cache
        # miss should also eventually resolve instead of leaving a permanent
        # placeholder. See "View-mode parity" in the design doc. The heavier
        # wide-crop cluster cache stays edit-mode-only (started/released in
        # _set_edit_mode), per its own cost/scope rationale.
        self._start_backfill()

        self._current_t = self._t_start
        self._load_frame(self._t_start)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._maximized_idx is not None or not self._cells:
            return
        if self.width() < 100:
            return  # skip layout during initial sizing before widget is shown
        new_ncols = max(2, min(len(self._cameras) + 1, self.width() // _TARGET_CELL_W))
        if new_ncols != self._ncols:
            self._ncols = new_ncols
            self._repopulate_grid()

    def _repopulate_grid(self) -> None:
        """Re-insert all cells into the grid with the current column count."""
        for cell in self._cells:
            self._grid.removeWidget(cell)
        self._grid.removeWidget(self._3d_ph)

        # Reset all column/row stretches in a safe range.
        for c in range(8):
            self._grid.setColumnStretch(c, 0)
        for r in range(8):
            self._grid.setRowStretch(r, 0)

        ncols = self._ncols
        for i, cell in enumerate(self._cells):
            row, col = divmod(i, ncols)
            self._grid.addWidget(cell, row, col)
            self._grid.setColumnStretch(col, 1)
            cell.show()

        n_cells = len(self._cameras) + 1
        r3d, c3d = divmod(len(self._cameras), ncols)
        self._grid.addWidget(self._3d_ph, r3d, c3d)
        self._grid.setColumnStretch(c3d, 1)
        self._3d_ph.show()

        nrows = ceil(n_cells / ncols)
        for r in range(nrows):
            self._grid.setRowStretch(r, 1)

    def _on_maximize_requested(self, idx: int) -> None:
        if self._maximized_idx == idx:
            self._leave_maximized()
        elif self._maximized_idx is not None:
            self._leave_maximized()
            self._enter_maximized(idx)
        else:
            self._enter_maximized(idx)

    def _enter_maximized(self, idx: int) -> None:
        self._maximized_idx = idx
        big_cell = self._cells[idx]
        big_cell.set_is_maximized(True)

        # Remove all cells and 3D placeholder from grid.
        for cell in self._cells:
            self._grid.removeWidget(cell)
        self._grid.removeWidget(self._3d_ph)

        # Big cell → left pane of splitter.
        self._max_splitter.replaceWidget(0, big_cell)
        big_cell.show()

        # Remaining cells → thumbnail strip.
        for i, cell in enumerate(self._cells):
            if i == idx:
                continue
            self._thumb_layout.addWidget(cell)
            cell.show()
        self._thumb_layout.addWidget(self._3d_ph)
        self._3d_ph.show()

        self._stack.setCurrentIndex(1)

        # Maximizing a camera almost always means you want to work with that
        # camera's data next -- follow it in the timeline, same rule already
        # used when a keypoint gets selected in a different camera's cell
        # (see _sync_timeline).
        self._sel_cam_idx = idx
        self._sync_timeline(self._current_t)

    def _leave_maximized(self) -> None:
        if self._maximized_idx is None:
            return
        self._cells[self._maximized_idx].set_is_maximized(False)
        self._maximized_idx = None

        # Restore placeholder to left pane (removes big cell from splitter).
        self._max_splitter.replaceWidget(0, self._max_placeholder)

        # Drain thumbnail strip without destroying widgets.
        while self._thumb_layout.count():
            self._thumb_layout.takeAt(0)

        self._stack.setCurrentIndex(0)
        self._repopulate_grid()

    def _set_edit_mode(self, enabled: bool) -> None:
        self._edit_mode = enabled
        if not enabled:
            self._sel_kp_indices = set()
            self._primary_kp_idx = None
            self._sel_cam_idx = None
            self._range_start_v = None
            self._range_end_v = None
            self._cancel_placement()
            # Low-res backfill (self._backfill) is *not* stopped here -- it
            # now runs regardless of edit mode (started once in _build), so
            # view mode keeps benefiting from it after editing ends. Only
            # the heavier, edit-mode-only wide-crop cluster cache is torn
            # down.
            if self._wide_crop_mgr is not None:
                self._wide_crop_mgr.frame_ready.disconnect(self._on_crop_ready)
                self._wide_crop_mgr.release()
                self._wide_crop_mgr = None
            # Idea 3: hand redetection only makes sense while editing --
            # same edit-mode-only lifecycle as the wide-crop cache above.
            for timer in self._hand_redetect_timers.values():
                timer.stop()
            self._hand_redetect_timers.clear()
            if self._hand_redetect is not None:
                self._hand_redetect.result_ready.disconnect(self._on_hand_redetect_ready)
                self._hand_redetect.stop()
                self._hand_redetect = None
        else:
            if self._backfill is None:
                self._start_backfill()
            self._start_wide_crop_cache()
            self._start_hand_redetect()
        for cell in self._cells:
            cell.set_edit_mode(enabled)
            if not enabled:
                cell.set_trail(None)
                cell.set_selection(None, frozenset())
        if self._kp_picker is not None:
            self._kp_picker.setVisible(enabled)
        if self._chain_btn is not None:
            self._chain_btn.setEnabled(enabled)
        if self._auto_redetect_chk is not None:
            self._auto_redetect_chk.setEnabled(enabled)
        if self._timeline is not None:
            self._timeline.set_edit_mode(enabled)
            self._sync_timeline(self._current_t)

    def _start_backfill(self) -> None:
        """Start the background crop-generation worker if there is a detection run."""
        if not self._det_run_id:
            _log.debug("backfill: no detection run — skipping")
            return
        row = self._conn.execute("PRAGMA database_list").fetchone()
        if row is None:
            _log.warning("backfill: PRAGMA database_list returned nothing")
            return
        db_path = row[2]
        if not db_path:
            _log.warning("backfill: could not determine DB path (in-memory DB?)")
            return
        _log.info("backfill: starting worker  db=%s  det_run=%s  cameras=%d",
                  db_path, self._det_run_id, len(self._cameras))
        worker = CropBackfillWorker(
            db_path=db_path,
            det_run_id=self._det_run_id,
            cameras=self._cameras,
            track_segs=self._track_segs,
            bboxes=self._det_bboxes,
        )
        worker.frame_ready.connect(self._on_crop_ready)
        self._backfill = worker
        worker.start()

    def _start_wide_crop_cache(self) -> None:
        """Acquire the shared, detection-run-scoped wide-crop cluster cache.

        Scoped to the detection run (not this panel) via
        `FrameCropCacheManager`, so a second person's panel in the same trial
        reuses the first panel's already-built cache instead of rebuilding it
        -- see "Cache scope and lifecycle" in the design doc.
        """
        if not self._det_run_id:
            return
        row = self._conn.execute("PRAGMA database_list").fetchone()
        if row is None or not row[2]:
            return
        db_path = row[2]
        from app.pose.wide_crop_cache import FrameCropCacheManager

        mgr = FrameCropCacheManager.acquire(db_path, self._det_run_id)
        mgr.frame_ready.connect(self._on_crop_ready)
        self._wide_crop_mgr = mgr

    def _start_hand_redetect(self) -> None:
        """Start the background hand-redetection worker (Idea 3).

        Scoped to this panel's (sequence, person) -- unlike the wide-crop
        cache, there's no cross-panel sharing concern here since a
        redetection write is specific to one person's sequence.
        """
        row = self._conn.execute("PRAGMA database_list").fetchone()
        if row is None or not row[2]:
            return
        db_path = row[2]
        from app.ui.hand_redetect_worker import HandRedetectWorker

        try:
            worker = HandRedetectWorker(db_path, self._sequence_id, 0, self._cameras)
        except ImportError:
            _log.warning("hand-redetect: rtmlib unavailable, skipping")
            return
        worker.result_ready.connect(self._on_hand_redetect_ready)
        self._hand_redetect = worker
        worker.start()

    def _on_crop_ready(self, svid: str, frame_idx: int) -> None:
        """Called on the main thread when the backfill worker writes a new crop."""
        _log.debug("backfill: crop ready  svid=%s  frame=%d", svid, frame_idx)
        if not self._sync_table:
            return
        for cam in self._cameras:
            if cam["shot_video_id"] == svid:
                current_fi = self._sync_table.lookup(self._current_t, svid)
                if current_fi == frame_idx:
                    _log.debug("backfill: refreshing display for current frame %d", frame_idx)
                    self._load_frame(self._current_t)
                break

    def _on_hand_redetect_ready(self, cam_id: str, frame_idx: int) -> None:
        """Called on the main thread when the hand-redetect worker (Idea 3)
        writes a new 'hand_l.refined'/'hand_r.refined' row."""
        from app.pose.db_cache import read_observations_with_edits

        self._obs_kp[cam_id] = read_observations_with_edits(self._conn, self._sequence_id, cam_id)
        self._refresh_timeline_status(cam_id)
        if not self._sync_table:
            return
        for cam in self._cameras:
            if cam["camera_instance_id"] == cam_id:
                current_fi = self._sync_table.lookup(self._current_t, cam["shot_video_id"])
                if current_fi == frame_idx:
                    self._load_frame(self._current_t)
                break

    def _auto_redetect_enabled(self) -> bool:
        """"Auto-detect" vs "keep existing state" (Idea 3's design doc):
        the checkbox is absent (None) only on test-constructed widgets that
        bypass __init__/_build, where the default is "on" so those tests
        exercise the common-case behavior."""
        return self._auto_redetect_chk is None or self._auto_redetect_chk.isChecked()

    def _maybe_queue_hand_redetect(self, cam_id: str, svid: str, frame_idx: int, kp_idx: int) -> None:
        """After any keypoint edit, arm a debounced hand-redetect request if
        *kp_idx* is a wrist/elbow index for either hand side -- the
        converged trigger point every editing operation funnels through
        (see Idea 3's design doc). No-op if the worker isn't running (not
        in edit mode, or rtmlib unavailable).
        """
        if self._hand_redetect is None or not self._auto_redetect_enabled():
            return
        side = _hand_side_for_kp_idx(kp_idx)
        if side is None:
            return
        _log.debug(
            "hand-redetect: armed debounce  cam=%s frame=%d side=%s kp_idx=%d",
            cam_id, frame_idx, side, kp_idx,
        )
        key = (cam_id, frame_idx, side)
        timer = self._hand_redetect_timers.get(key)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(
                lambda: self._fire_hand_redetect(cam_id, svid, frame_idx, side)
            )
            self._hand_redetect_timers[key] = timer
        timer.start(_HAND_REDETECT_DEBOUNCE_MS)

    def _fire_hand_redetect(self, cam_id: str, svid: str, frame_idx: int, side: str) -> None:
        """Debounce timer fired: look up the *current* wrist/elbow position
        (edited if available, tracked otherwise -- same anchor convention as
        Idea 2's batch crop) and queue a single-frame redetect request."""
        from posetrak.detection.hand_refinement import _ELBOW_IDX, _WRIST_IDX

        self._hand_redetect_timers.pop((cam_id, frame_idx, side), None)
        if self._hand_redetect is None or not self._sync_table:
            return
        kp = self._obs_kp.get(cam_id, {}).get(frame_idx)
        if kp is None:
            _log.info(
                "hand-redetect: debounce fired but no observation at all"
                "  cam=%s frame=%d side=%s -- skipping", cam_id, frame_idx, side,
            )
            return
        wrist_idx, elbow_idx = _WRIST_IDX[side], _ELBOW_IDX[side]
        if wrist_idx >= kp.shape[0] or kp[wrist_idx, 2] <= 0.0:
            _log.info(
                "hand-redetect: debounce fired but wrist not confidently known"
                "  cam=%s frame=%d side=%s conf=%.2f -- skipping",
                cam_id, frame_idx, side,
                float(kp[wrist_idx, 2]) if wrist_idx < kp.shape[0] else -1.0,
            )
            return
        wrist = (float(kp[wrist_idx, 0]), float(kp[wrist_idx, 1]))
        elbow = None
        if elbow_idx < kp.shape[0] and kp[elbow_idx, 2] > 0.0:
            elbow = (float(kp[elbow_idx, 0]), float(kp[elbow_idx, 1]))
        timestamp_s = self._sync_table.frame_to_global_time(frame_idx, svid)
        if timestamp_s is None:
            _log.info(
                "hand-redetect: debounce fired but no timestamp for frame"
                "  cam=%s frame=%d side=%s -- skipping", cam_id, frame_idx, side,
            )
            return
        self._hand_redetect.request_frame(svid, frame_idx, timestamp_s, side, wrist, elbow)

    def _queue_hand_redetect_range(
        self, cam_id: str, svid: str, side: str, frames: "set[int]",
    ) -> None:
        """Queue one range redetect request covering *frames* (the
        interpolation-fill case -- see Idea 3's design doc). Reads the
        just-written wrist/elbow positions from `self._obs_kp` directly
        rather than debouncing, since an interpolation batch is already
        settled by the time this is called."""
        from posetrak.detection.hand_refinement import _ELBOW_IDX, _WRIST_IDX

        if self._hand_redetect is None or not frames or not self._sync_table:
            return
        if not self._auto_redetect_enabled():
            _log.debug("hand-redetect: range skipped, auto-detect is off  cam=%s side=%s", cam_id, side)
            return
        kp_by_frame = self._obs_kp.get(cam_id, {})
        wrist_idx, elbow_idx = _WRIST_IDX[side], _ELBOW_IDX[side]
        anchor_by_frame: dict[int, tuple] = {}
        skipped = 0
        for f in frames:
            kp = kp_by_frame.get(f)
            if kp is None or wrist_idx >= kp.shape[0] or kp[wrist_idx, 2] <= 0.0:
                skipped += 1
                continue
            wrist = (float(kp[wrist_idx, 0]), float(kp[wrist_idx, 1]))
            elbow = None
            if elbow_idx < kp.shape[0] and kp[elbow_idx, 2] > 0.0:
                elbow = (float(kp[elbow_idx, 0]), float(kp[elbow_idx, 1]))
            ts = self._sync_table.frame_to_global_time(f, svid)
            if ts is None:
                skipped += 1
                continue
            anchor_by_frame[f] = (ts, wrist, elbow)
        _log.info(
            "hand-redetect: range candidate  cam=%s side=%s requested=%d usable=%d skipped=%d",
            cam_id, side, len(frames), len(anchor_by_frame), skipped,
        )
        self._hand_redetect.request_range(svid, side, anchor_by_frame)

    def _on_kp_selected(self, cam_idx: int, kp_idx: int) -> None:
        """Plain left-click on a dot: sole selection."""
        self._sel_kp_indices = {kp_idx}
        self._primary_kp_idx = kp_idx
        self._sel_cam_idx = cam_idx
        self._load_frame(self._current_t)

    def _on_kp_ctrl_clicked(self, cam_idx: int, kp_idx: int) -> None:
        """Ctrl+click on a dot: toggle membership in selection."""
        if kp_idx in self._sel_kp_indices:
            self._sel_kp_indices.discard(kp_idx)
            if kp_idx == self._primary_kp_idx:
                self._primary_kp_idx = next(iter(self._sel_kp_indices), None)
        else:
            self._sel_kp_indices.add(kp_idx)
            self._primary_kp_idx = kp_idx
        self._sel_cam_idx = cam_idx
        self._load_frame(self._current_t)

    def _on_kp_deselected(self, cam_idx: int) -> None:
        self._sel_kp_indices = set()
        self._primary_kp_idx = None
        self._sel_cam_idx = None
        for cell in self._cells:
            cell.set_trail(None)
            cell.set_selection(None, frozenset())
        self._sync_timeline(self._current_t)

    def _on_rubber_band_selected(
        self, cam_idx: int, x1: float, y1: float, x2: float, y2: float, ctrl: bool
    ) -> None:
        """Rubber-band drag: select all kp inside rect.  Ctrl adds to selection, else replaces."""
        cell = self._cells[cam_idx]
        cam = self._cameras[cam_idx]
        cam_id = cam["camera_instance_id"]
        svid = cam["shot_video_id"]
        if not self._sync_table:
            return
        frame_idx = self._sync_table.lookup(self._current_t, svid)
        obs_kp = self._obs_kp.get(cam_id, {}).get(frame_idx) if frame_idx is not None else None
        if obs_kp is None:
            return
        fx1, fy1 = cell._canvas._display_to_full(x1, y1)
        fx2, fy2 = cell._canvas._display_to_full(x2, y2)
        new_hits: set[int] = set()
        for i in range(obs_kp.shape[0]):
            if i in self._hidden_kp_indices:
                continue
            kx, ky = float(obs_kp[i, 0]), float(obs_kp[i, 1])
            if fx1 <= kx <= fx2 and fy1 <= ky <= fy2:
                new_hits.add(i)
        if ctrl:
            self._sel_kp_indices |= new_hits
        else:
            self._sel_kp_indices = new_hits
        if self._sel_kp_indices and self._primary_kp_idx not in self._sel_kp_indices:
            self._primary_kp_idx = next(iter(self._sel_kp_indices))
        elif not self._sel_kp_indices:
            self._primary_kp_idx = None
        self._sel_cam_idx = cam_idx
        self._load_frame(self._current_t)

    def _on_context_menu_requested(
        self, cam_idx: int, hit_kp_idx: int, dx: float, dy: float
    ) -> None:
        """Right-click: show group-selection context menu."""
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        for group_name in self._pose_model.group_names:
            action = menu.addAction(f"Select {group_name}")
            action.triggered.connect(
                lambda checked=False, g=group_name, ci=cam_idx: self._select_group(g, ci)
            )
        menu.addSeparator()
        all_act = menu.addAction("Select all")
        all_act.triggered.connect(lambda ci=cam_idx: self._select_all(ci))
        clear_act = menu.addAction("Deselect all")
        clear_act.triggered.connect(lambda: self._on_kp_deselected(cam_idx))
        if self._sel_kp_indices:
            menu.addSeparator()
            disable_act = menu.addAction("Disable selected")
            disable_act.triggered.connect(lambda: self._set_outlier_selected(True))
            enable_act = menu.addAction("Enable selected")
            enable_act.triggered.connect(lambda: self._set_outlier_selected(False))
            if self._range_start_v is not None:
                interp_act = menu.addAction("Interpolate missing")
                interp_act.triggered.connect(self._interpolate_missing_range)
        cam = self._cameras[cam_idx]
        cam_id = cam["camera_instance_id"]
        frame_idx = (
            self._sync_table.lookup(self._current_t, cam["shot_video_id"])
            if self._sync_table else None
        )
        refined_sides = self._refined_sides_at(cam_id, frame_idx) if frame_idx is not None else set()
        if refined_sides:
            menu.addSeparator()
            for side in sorted(refined_sides):
                label = "Revert left-hand redetection" if side == "left" else "Revert right-hand redetection"
                revert_act = menu.addAction(label)
                revert_act.triggered.connect(
                    lambda checked=False, ci=cam_id, f=frame_idx, s=side: self._revert_hand_redetect(ci, f, s)
                )
        cell = self._cells[cam_idx]
        if cell._canvas._zoom_rect is not None:
            menu.addSeparator()
            reset_zoom_act = menu.addAction("Reset zoom")
            reset_zoom_act.triggered.connect(lambda ci=cam_idx: self._cells[ci]._canvas.reset_zoom())
        global_pos = cell._canvas.mapToGlobal(QPoint(int(dx), int(dy)))
        menu.exec(global_pos)

    def _select_group(self, group_name: str, cam_idx: int | None = None) -> None:
        self._sel_kp_indices = set(self._pose_model.group_indices(group_name)) - self._hidden_kp_indices
        self._primary_kp_idx = next(iter(self._sel_kp_indices), None)
        if cam_idx is not None:
            self._sel_cam_idx = cam_idx
        self._load_frame(self._current_t)

    def _select_all(self, cam_idx: int | None = None) -> None:
        self._sel_kp_indices = set(self._pose_model.all_indices) - self._hidden_kp_indices
        self._primary_kp_idx = next(iter(self._sel_kp_indices), None)
        if cam_idx is not None:
            self._sel_cam_idx = cam_idx
        self._load_frame(self._current_t)

    def _refined_sides_at(self, cam_id: str, frame_idx: int) -> "set[str]":
        """Which hand sides currently have a 'hand_l.refined'/'hand_r.refined'
        row for this (camera, frame) -- used to only offer "Revert hand
        redetection" (Idea 3) when there's actually something to revert."""
        from app.pose.db_cache import _HAND_REFINED_SOURCE

        rows = self._conn.execute(
            "SELECT source FROM pose_observations"
            " WHERE sequence_id=? AND camera_instance_id=? AND video_frame=?",
            (self._sequence_id, cam_id, frame_idx),
        ).fetchall()
        by_source = {r["source"] for r in rows}
        return {side for side, source in _HAND_REFINED_SOURCE.items() if source in by_source}

    def _revert_hand_redetect(self, cam_id: str, frame_idx: int, side: str) -> None:
        """Revert-hand-redetection context-menu action (Idea 3): delete the
        interactively-redetected row, falling back to whatever the batch
        'hand_l'/'hand_r' row (or nothing) provides for that slot instead."""
        from app.pose.db_cache import read_observations_with_edits, revert_hand_refinement

        revert_hand_refinement(self._conn, self._sequence_id, cam_id, frame_idx, 0, side=side)
        self._obs_kp[cam_id] = read_observations_with_edits(self._conn, self._sequence_id, cam_id)
        self._refresh_timeline_status(cam_id)
        self._load_frame(self._current_t)

    def _on_empty_area_clicked(self, cam_idx: int, dx: float, dy: float) -> None:
        """Canvas click missed all kp.  Place primary kp on ghost frames; otherwise deselect."""
        if (self._edit_mode
                and self._primary_kp_idx is not None
                and self._sync_table is not None):
            cam = self._cameras[cam_idx]
            svid = cam["shot_video_id"]
            cam_id = cam["camera_instance_id"]
            frame_idx = self._sync_table.lookup(self._current_t, svid)
            if (frame_idx is not None
                    and self._obs_kp.get(cam_id, {}).get(frame_idx) is None):
                # Ghost frame with a selected keypoint → place it at the click location.
                full_x, full_y = self._cells[cam_idx]._canvas._display_to_full(dx, dy)
                self._on_kp_moved(cam_idx, self._primary_kp_idx, full_x, full_y)
                return
        self._on_kp_deselected(cam_idx)

    def _on_kp_moved(self, cam_idx: int, kp_idx: int, new_x: float, new_y: float) -> None:
        from app.pose.db_cache import read_observations_with_edits, update_single_keypoint_edit
        cam = self._cameras[cam_idx]
        cam_id = cam["camera_instance_id"]
        svid = cam["shot_video_id"]
        if not self._sync_table:
            return
        frame_idx = self._sync_table.lookup(self._current_t, svid)
        if frame_idx is None:
            return
        update_single_keypoint_edit(
            self._conn, self._sequence_id, cam_id, frame_idx, kp_idx, new_x, new_y
        )
        self._obs_kp[cam_id] = read_observations_with_edits(
            self._conn, self._sequence_id, cam_id
        )
        self._refresh_timeline_status(cam_id)
        self._maybe_queue_hand_redetect(cam_id, svid, frame_idx, kp_idx)
        self._load_frame(self._current_t)

    def _on_kp_picked(self, kp_idx: int) -> None:
        """A keypoint was picked from the placement toolbar: arm placement
        mode on every camera cell. Picking a different keypoint while one is
        already pending just retargets it -- no need to Esc first. Also ends
        any in-progress limb chain, since a single explicit pick overrides it."""
        self._chain_limb = None
        self._chain_indices = []
        self._chain_pos = 0
        self._pending_place_kp_idx = kp_idx
        for cell in self._cells:
            cell.set_placement_active(True)
            cell.set_placement_label(None)
        if self._kp_picker is not None:
            self._kp_picker.set_active(kp_idx)

    def _cancel_placement(self) -> None:
        if self._pending_place_kp_idx is None and self._chain_limb is None:
            return
        self._pending_place_kp_idx = None
        self._chain_limb = None
        self._chain_indices = []
        self._chain_pos = 0
        for cell in self._cells:
            cell.set_placement_active(False)
            cell.set_placement_label(None)
        if self._kp_picker is not None:
            self._kp_picker.set_active(None)

    def _show_chain_menu(self) -> None:
        """"Set limb…" button: pick which limb to place, in shoulder/hip-first order."""
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        for limb in ("Face", "Left arm", "Right arm", "Left leg", "Right leg"):
            if not self._pose_model.limb_chain_indices(limb):
                continue
            action = menu.addAction(limb)
            action.triggered.connect(lambda checked=False, lb=limb: self._start_chain_placement(lb))
        if self._chain_btn is not None:
            menu.exec(self._chain_btn.mapToGlobal(self._chain_btn.rect().bottomLeft()))

    def _start_chain_placement(self, limb: str) -> None:
        indices = self._pose_model.limb_chain_indices(limb)
        if not indices:
            return
        self._chain_limb = limb
        self._chain_indices = indices
        self._chain_pos = 0
        self._arm_chain_step()

    def _arm_chain_step(self) -> None:
        """Arm placement for the current chain keypoint and show its name/position
        on every cell. Wraps back to the first keypoint once the last one has
        been placed, rather than ending the mode -- the whole point is to run
        the same limb again on another frame without re-picking it from the
        "Set limb…" menu each time. Esc (or picking a different limb/keypoint)
        is the actual way to stop."""
        if self._chain_limb is None:
            return
        if self._chain_pos >= len(self._chain_indices):
            self._chain_pos = 0
        kp_idx = self._chain_indices[self._chain_pos]
        self._pending_place_kp_idx = kp_idx
        label = (
            f"{self._chain_limb}: {self._pose_model.name_of(kp_idx)} "
            f"({self._chain_pos + 1}/{len(self._chain_indices)})"
        )
        for cell in self._cells:
            cell.set_placement_active(True)
            cell.set_placement_label(label)
        if self._kp_picker is not None:
            self._kp_picker.set_active(kp_idx)
        # Space/Esc must reach _handle_key immediately, without the user
        # having to click a camera view first -- key events only route there
        # (via the installed eventFilter) when a canvas actually holds
        # keyboard focus, which picking a limb from the menu does not confer.
        if self._cells:
            self._cells[0]._canvas.setFocus()

    def _on_placement_clicked(self, cam_idx: int, dx: float, dy: float) -> None:
        """Canvas click while placement mode is armed: place the pending
        keypoint at the clicked location, on any frame (real detection,
        ghost frame, or already has a dot there) -- a superset of
        _on_empty_area_clicked's ghost-frame-only, primary-selection-only
        placement. Plain toolbar placement is one shot: disarms afterward,
        same as a normal drag-to-move only moves the one keypoint it grabbed.
        Chain placement instead re-arms for the next limb keypoint."""
        if self._pending_place_kp_idx is None:
            return
        kp_idx = self._pending_place_kp_idx
        full_x, full_y = self._cells[cam_idx]._canvas._display_to_full(dx, dy)
        if self._chain_limb is not None:
            self._chain_pos += 1
            self._on_kp_moved(cam_idx, kp_idx, full_x, full_y)
            self._arm_chain_step()
        else:
            self._cancel_placement()
            self._on_kp_moved(cam_idx, kp_idx, full_x, full_y)

    # ------------------------------------------------------------------
    # Keyboard handling
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:
        from PySide6.QtCore import QEvent
        if (event.type() == QEvent.Type.KeyPress
                and any(obj is c._canvas for c in self._cells)):
            return self._handle_key(event)
        return False

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if not self._handle_key(event):
            super().keyPressEvent(event)

    def _handle_key(self, event) -> bool:
        """Dispatch keyboard shortcuts. Returns True when the event is consumed."""
        key = event.key()
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        if key == Qt.Key.Key_A and self._slider is not None:
            if self._edit_mode and shift:
                self._extend_range_left()
            else:
                self._range_start_v = None
                self._range_end_v = None
                self._slider.setValue(self._slider.value() - self._slider.singleStep())
            return True
        if key == Qt.Key.Key_D and self._slider is not None:
            if self._edit_mode and shift:
                self._extend_range_right()
            else:
                self._range_start_v = None
                self._range_end_v = None
                self._slider.setValue(self._slider.value() + self._slider.singleStep())
            return True

        if not self._edit_mode:
            return False

        if key == Qt.Key.Key_Escape:
            if self._pending_place_kp_idx is not None:
                # Cancel placement mode first; a second Esc (nothing pending
                # anymore) falls through to the existing deselect below.
                self._cancel_placement()
                return True
            self._sel_kp_indices = set()
            self._primary_kp_idx = None
            self._sel_cam_idx = None
            self._range_start_v = None
            self._range_end_v = None
            for cell in self._cells:
                cell.set_trail(None)
                cell.set_selection(None, frozenset())
                cell.set_range_highlights([])
            self._sync_timeline(self._current_t)
            return True

        if key == Qt.Key.Key_I and self._range_start_v is not None:
            self._interpolate_range()
            return True
        if key == Qt.Key.Key_M and self._timeline is not None:
            v = int(round((self._current_t - self._t_start) * 1000))
            self._timeline.set_marker(v)
            return True
        if key == Qt.Key.Key_Space and self._chain_limb is not None:
            self._chain_pos += 1
            self._arm_chain_step()
            return True
        if ctrl and key == Qt.Key.Key_C:
            self._copy_keypoints()
            return True
        if ctrl and key == Qt.Key.Key_V:
            self._paste_keypoints()
            return True

        limb = self._LIMB_SHORTCUT_KEYS.get(key)
        if limb is not None:
            self._handle_limb_shortcut(limb, ctrl=ctrl, shift=shift)
            return True

        if not self._sel_kp_indices or self._sel_cam_idx is None:
            return False

        nudge = {
            Qt.Key.Key_Left:  (-1.0,  0.0),
            Qt.Key.Key_Right: ( 1.0,  0.0),
            Qt.Key.Key_Up:    ( 0.0, -1.0),
            Qt.Key.Key_Down:  ( 0.0,  1.0),
        }.get(key)
        if nudge is not None:
            self._nudge_keypoint(*nudge)
            return True

        if key == Qt.Key.Key_Space:
            self._toggle_outlier()
            return True

        return False

    def _handle_limb_shortcut(self, limb: str, *, ctrl: bool, shift: bool) -> None:
        """Number-key shortcut for *limb* (see _LIMB_SHORTCUT_KEYS): plain key
        toggles show/hide, Shift+key isolates it, Ctrl+key starts chain
        placement (if the limb has one -- see kp_models.py's limb_chains)."""
        indices = set(self._pose_model.group_indices(limb))
        if not indices:
            self.status_message.emit(f"No {limb} keypoints in this pose model")
            return

        if ctrl:
            if self._pose_model.limb_chain_indices(limb):
                self._start_chain_placement(limb)
            else:
                self.status_message.emit(f"No limb-placement order defined for {limb} yet")
        elif shift:
            self._hidden_kp_indices = set(self._pose_model.all_indices) - indices
            self._sel_kp_indices -= self._hidden_kp_indices
            if self._primary_kp_idx in self._hidden_kp_indices:
                self._primary_kp_idx = next(iter(self._sel_kp_indices), None)
            if self._timeline is not None:
                self._timeline.set_hidden(frozenset(self._hidden_kp_indices))
            self._load_frame(self._current_t)
            self.status_message.emit(f"Showing only {limb}")
        else:
            self._on_timeline_visibility_toggled(indices)

    def _copy_keypoints(self) -> None:
        """Ctrl+C: copy selected kp positions from current frame, primary camera."""
        if not self._sel_kp_indices or self._sel_cam_idx is None:
            return
        cam = self._cameras[self._sel_cam_idx]
        cam_id = cam["camera_instance_id"]
        svid = cam["shot_video_id"]
        if not self._sync_table:
            return
        frame_idx = self._sync_table.lookup(self._current_t, svid)
        if frame_idx is None:
            return
        kp = self._obs_kp.get(cam_id, {}).get(frame_idx)
        if kp is None:
            return
        self._clipboard = {}
        for kp_idx in self._sel_kp_indices:
            if kp_idx < kp.shape[0]:
                self._clipboard[kp_idx] = (float(kp[kp_idx, 0]), float(kp[kp_idx, 1]))
        self._clipboard_cam_idx = self._sel_cam_idx
        n = len(self._clipboard)
        self.status_message.emit(f"Copied {n} keypoint{'s' if n != 1 else ''}")

    def _paste_keypoints(self) -> None:
        """Ctrl+V: paste clipboard kp into current frame of the clipboard's camera."""
        from app.pose.db_cache import read_observations_with_edits, update_single_keypoint_edit
        if not self._clipboard or self._clipboard_cam_idx is None:
            return
        cam = self._cameras[self._clipboard_cam_idx]
        cam_id = cam["camera_instance_id"]
        svid = cam["shot_video_id"]
        if not self._sync_table:
            return
        frame_idx = self._sync_table.lookup(self._current_t, svid)
        if frame_idx is None:
            return
        for kp_idx, (x, y) in self._clipboard.items():
            update_single_keypoint_edit(
                self._conn, self._sequence_id, cam_id, frame_idx, kp_idx, x, y
            )
            self._maybe_queue_hand_redetect(cam_id, svid, frame_idx, kp_idx)
        self._obs_kp[cam_id] = read_observations_with_edits(
            self._conn, self._sequence_id, cam_id
        )
        self._refresh_timeline_status(cam_id)
        self._load_frame(self._current_t)
        n = len(self._clipboard)
        self.status_message.emit(f"Pasted {n} keypoint{'s' if n != 1 else ''}")

    # ------------------------------------------------------------------
    # Frame-range selection + interpolation (Phase 10)
    # ------------------------------------------------------------------

    def _range_frame_set(self, svid: str) -> set[int]:
        """Return video frames for svid that fall within the active slider range."""
        if (self._range_start_v is None or self._range_end_v is None
                or not self._sync_table or self._slider is None):
            return set()
        step = self._slider.singleStep()
        frames: set[int] = set()
        v = self._range_start_v
        while v <= self._range_end_v:
            f = self._sync_table.lookup(self._t_start + v / 1000.0, svid)
            if f is not None:
                frames.add(f)
            v += step
        return frames

    def _compute_range_highlights(
        self, cam_id: str, svid: str
    ) -> list[tuple[float, float]]:
        """Return frame-space (x, y) positions of selected kp within the active range."""
        if (self._range_start_v is None or not self._sel_kp_indices
                or not self._sync_table or self._slider is None):
            return []
        kp_by_frame = self._obs_kp.get(cam_id, {})
        pts: list[tuple[float, float]] = []
        step = self._slider.singleStep()
        v = self._range_start_v
        while v <= self._range_end_v:
            f = self._sync_table.lookup(self._t_start + v / 1000.0, svid)
            if f is not None:
                kp = kp_by_frame.get(f)
                if kp is not None:
                    for kp_idx in self._sel_kp_indices:
                        if kp_idx < kp.shape[0]:
                            x, y = float(kp[kp_idx, 0]), float(kp[kp_idx, 1])
                            if x != 0.0 or y != 0.0:
                                pts.append((x, y))
            v += step
        return pts

    def _extend_range_left(self) -> None:
        if self._slider is None:
            return
        cur = self._slider.value()
        step = self._slider.singleStep()
        new_val = max(self._slider.minimum(), cur - step)
        if self._range_start_v is None:
            self._range_start_v = new_val
            self._range_end_v = cur
        else:
            self._range_start_v = min(self._range_start_v, new_val)
        self._slider.setValue(new_val)

    def _extend_range_right(self) -> None:
        if self._slider is None:
            return
        cur = self._slider.value()
        step = self._slider.singleStep()
        new_val = min(self._slider.maximum(), cur + step)
        if self._range_end_v is None:
            self._range_start_v = cur
            self._range_end_v = new_val
        else:
            self._range_end_v = max(self._range_end_v, new_val)
        self._slider.setValue(new_val)

    def _interpolate_range(self) -> None:
        """I key: piecewise-linear interpolation of selected kp across the range.

        The range's own boundary frames are always anchors (Phase 10
        behavior). Phase 14 adds interior anchors: any frame inside the range
        that is an explicit keyframe for a given keypoint — i.e. has an edit
        row with is_outlier == 0 (STATUS_BLUE; see timeline_status.py and the
        design doc's *Multi-keyframe interpolation* section) — also anchors
        the interpolation, splitting it into independent segments around it.

        An untouched original detection is never an interior anchor, even if
        it's currently an inlier: with a wide range covering both good and
        bad frames, plain "select range, press I" must still overwrite
        everything between the two ends with one straight line (the Phase 10
        behavior), not silently keep whatever wrong values happen to sit in
        the middle. Deliberately keeping a frame's value (Ctrl+click freeze,
        or re-enabling/moving it) is what turns it into an anchor.
        """
        from app.pose.db_cache import read_observations_with_edits, update_single_keypoint_edit
        if self._range_start_v is None or self._range_end_v is None:
            return
        if not self._sel_kp_indices or not self._sync_table or self._slider is None:
            return

        for cam in self._cameras:
            cam_id = cam["camera_instance_id"]
            svid = cam["shot_video_id"]

            range_frames = sorted(self._range_frame_set(svid))
            if not range_frames:
                continue

            frame_start = range_frames[0]
            frame_end = range_frames[-1]
            kp_by_frame = self._obs_kp.get(cam_id, {})
            kp_l = kp_by_frame.get(frame_start)
            kp_r = kp_by_frame.get(frame_end)
            if kp_l is None or kp_r is None:
                continue

            status_by_frame = read_timeline_status(
                self._conn, self._sequence_id, cam_id,
                shot_video_id=svid,
                seg_run_id=self._seg_sources.get(svid),
                track_id_by_frame=self._track_id_by_frame(svid),
            )

            any_written = False
            # Idea 3: frames actually written for a wrist/elbow index, per
            # hand side -- queued as one redetect range per side below,
            # covering exactly the interpolated (non-anchor) frames, once
            # the whole kp_idx loop below has settled.
            touched_frames_by_side: dict[str, set[int]] = {}
            for kp_idx in self._sel_kp_indices:
                if kp_idx >= kp_l.shape[0] or kp_idx >= kp_r.shape[0]:
                    continue
                if float(kp_l[kp_idx, 2]) < 0.01 or float(kp_r[kp_idx, 2]) < 0.01:
                    continue

                anchors: list[tuple[int, float, float]] = [
                    (frame_start, float(kp_l[kp_idx, 0]), float(kp_l[kp_idx, 1])),
                ]
                for f in range_frames:
                    if f == frame_start or f == frame_end:
                        continue
                    status = status_by_frame.get(f)
                    if status is None or kp_idx >= len(status) or int(status[kp_idx]) != STATUS_BLUE:
                        continue
                    kp_f = kp_by_frame.get(f)
                    if kp_f is not None and kp_idx < kp_f.shape[0]:
                        anchors.append((f, float(kp_f[kp_idx, 0]), float(kp_f[kp_idx, 1])))
                anchors.append((frame_end, float(kp_r[kp_idx, 0]), float(kp_r[kp_idx, 1])))
                anchors.sort(key=lambda a: a[0])
                anchor_frames = {a[0] for a in anchors}

                side = _hand_side_for_kp_idx(kp_idx)
                for (f0, x0, y0), (f1, x1, y1) in zip(anchors, anchors[1:]):
                    span = f1 - f0
                    if span <= 0:
                        continue
                    for f in range_frames:
                        if f <= f0 or f >= f1 or f in anchor_frames:
                            continue
                        t = (f - f0) / span
                        x = x0 + t * (x1 - x0)
                        y = y0 + t * (y1 - y0)
                        update_single_keypoint_edit(
                            self._conn, self._sequence_id, cam_id, f, kp_idx, x, y
                        )
                        any_written = True
                        if side is not None:
                            touched_frames_by_side.setdefault(side, set()).add(f)

            if any_written:
                self._obs_kp[cam_id] = read_observations_with_edits(
                    self._conn, self._sequence_id, cam_id
                )
                self._refresh_timeline_status(cam_id)
                for side, frames in touched_frames_by_side.items():
                    self._queue_hand_redetect_range(cam_id, svid, side, frames)

        self._range_start_v = None
        self._range_end_v = None
        self._load_frame(self._current_t)
        n = len(self._sel_kp_indices)
        self.status_message.emit(
            f"Interpolated {n} keypoint{'s' if n != 1 else ''} over range"
        )

    def _nudge_keypoint(self, dx: float, dy: float) -> None:
        from app.pose.db_cache import read_observations_with_edits, update_single_keypoint_edit
        cam = self._cameras[self._sel_cam_idx]
        cam_id = cam["camera_instance_id"]
        svid = cam["shot_video_id"]
        if not self._sync_table:
            return
        frame_idx = self._sync_table.lookup(self._current_t, svid)
        if frame_idx is None:
            return
        kp = self._obs_kp.get(cam_id, {}).get(frame_idx)
        if kp is None:
            return
        for kp_idx in self._sel_kp_indices:
            if kp_idx >= kp.shape[0]:
                continue
            update_single_keypoint_edit(
                self._conn, self._sequence_id, cam_id, frame_idx,
                kp_idx,
                float(kp[kp_idx, 0]) + dx,
                float(kp[kp_idx, 1]) + dy,
            )
            self._maybe_queue_hand_redetect(cam_id, svid, frame_idx, kp_idx)
        self._obs_kp[cam_id] = read_observations_with_edits(
            self._conn, self._sequence_id, cam_id
        )
        self._refresh_timeline_status(cam_id)
        self._load_frame(self._current_t)

    def _toggle_outlier(self) -> None:
        """Space bar: flip the current frame's primary kp state and apply
        that same new state to the whole selection (see
        `_set_outlier_selected` for the shared apply-to-range logic)."""
        if not self._sel_kp_indices or self._sel_cam_idx is None or not self._sync_table:
            return
        cam = self._cameras[self._sel_cam_idx]
        cam_id = cam["camera_instance_id"]
        svid = cam["shot_video_id"]
        frame_idx = self._sync_table.lookup(self._current_t, svid)
        kp_cur = self._obs_kp.get(cam_id, {}).get(frame_idx) if frame_idx is not None else None
        if kp_cur is None:
            return
        pri = self._primary_kp_idx
        if pri is None or pri >= kp_cur.shape[0]:
            pri = next(iter(self._sel_kp_indices))
        new_outlier = float(kp_cur[pri, 2]) >= 0.01
        self._set_outlier_selected(new_outlier)

    def _set_outlier_selected(self, is_outlier: bool) -> None:
        """Mark every selected keypoint as outlier (disabled) or not, across
        every frame in the active range — or just the current frame if no
        range is selected. Shared by the Space-bar toggle and the "Disable
        selected" / "Enable selected" context-menu actions, which pass an
        explicit target state instead of inferring one from the primary kp."""
        from app.pose.db_cache import read_observations_with_edits, update_single_keypoint_edit
        if not self._sel_kp_indices or self._sel_cam_idx is None:
            return
        cam = self._cameras[self._sel_cam_idx]
        cam_id = cam["camera_instance_id"]
        svid = cam["shot_video_id"]
        if not self._sync_table:
            return
        kp_by_frame = self._obs_kp.get(cam_id, {})

        frame_idx = self._sync_table.lookup(self._current_t, svid)
        target_frames: list[int]
        if self._range_start_v is not None:
            target_frames = sorted(self._range_frame_set(svid))
        else:
            target_frames = [frame_idx] if frame_idx is not None else []

        for f in target_frames:
            kp_f = kp_by_frame.get(f)
            if kp_f is None:
                continue
            for kp_idx in self._sel_kp_indices:
                if kp_idx >= kp_f.shape[0]:
                    continue
                update_single_keypoint_edit(
                    self._conn, self._sequence_id, cam_id, f,
                    kp_idx,
                    float(kp_f[kp_idx, 0]),
                    float(kp_f[kp_idx, 1]),
                    is_outlier=is_outlier,
                )
                self._maybe_queue_hand_redetect(cam_id, svid, f, kp_idx)
        self._obs_kp[cam_id] = read_observations_with_edits(
            self._conn, self._sequence_id, cam_id
        )
        self._refresh_timeline_status(cam_id)
        self._load_frame(self._current_t)

    def _interpolate_missing_range(self) -> None:
        """Right-click "Interpolate missing": like `_interpolate_range`, but
        only fills frames that have *no* value for a selected keypoint
        (confidence < 0.01 — disabled, or a ghost frame with nothing at that
        slot) — every frame that already has a value, disabled or not, is
        left untouched. Anchors are the nearest present values on either
        side of each gap, found independently per keypoint; a gap that
        isn't bounded by a present value on both sides (e.g. touching the
        range's own edge) is left as still-missing rather than extrapolated.
        """
        from app.pose.db_cache import read_observations_with_edits, update_single_keypoint_edit
        if self._range_start_v is None or self._range_end_v is None:
            return
        if not self._sel_kp_indices or not self._sync_table:
            return

        for cam in self._cameras:
            cam_id = cam["camera_instance_id"]
            svid = cam["shot_video_id"]

            range_frames = sorted(self._range_frame_set(svid))
            if not range_frames:
                continue
            kp_by_frame = self._obs_kp.get(cam_id, {})

            any_written = False
            touched_frames_by_side: dict[str, set[int]] = {}
            for kp_idx in self._sel_kp_indices:
                present: list[tuple[int, float, float]] = []
                for f in range_frames:
                    kp_f = kp_by_frame.get(f)
                    if kp_f is None or kp_idx >= kp_f.shape[0]:
                        continue
                    if float(kp_f[kp_idx, 2]) >= 0.01:
                        present.append((f, float(kp_f[kp_idx, 0]), float(kp_f[kp_idx, 1])))
                if len(present) < 2:
                    continue  # nothing to interpolate from

                present_frames = {f for f, _, _ in present}
                side = _hand_side_for_kp_idx(kp_idx)
                for (f0, x0, y0), (f1, x1, y1) in zip(present, present[1:]):
                    span = f1 - f0
                    if span <= 0:
                        continue
                    for f in range_frames:
                        if f <= f0 or f >= f1 or f in present_frames:
                            continue
                        t = (f - f0) / span
                        x = x0 + t * (x1 - x0)
                        y = y0 + t * (y1 - y0)
                        update_single_keypoint_edit(
                            self._conn, self._sequence_id, cam_id, f, kp_idx, x, y
                        )
                        any_written = True
                        if side is not None:
                            touched_frames_by_side.setdefault(side, set()).add(f)

            if any_written:
                self._obs_kp[cam_id] = read_observations_with_edits(
                    self._conn, self._sequence_id, cam_id
                )
                self._refresh_timeline_status(cam_id)
                for side, frames in touched_frames_by_side.items():
                    self._queue_hand_redetect_range(cam_id, svid, side, frames)

        self._load_frame(self._current_t)
        n = len(self._sel_kp_indices)
        self.status_message.emit(
            f"Interpolated missing values for {n} keypoint{'s' if n != 1 else ''} over range"
        )

    def _on_slider(self, value: int) -> None:
        self._current_t = self._t_start + value / 1000.0
        if self._time_label is not None:
            self._time_label.setText(_fmt_time(self._current_t))
        if self._chain_limb is not None and self._chain_pos != 0:
            # A limb chain always restarts at its first keypoint on a frame
            # change (A/D, slider drag, or timeline scrub all land here) --
            # per-frame progress through the chain isn't meaningful since
            # each frame needs the whole limb set from scratch.
            self._chain_pos = 0
            self._arm_chain_step()
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

    def _nearest_track_id_for_gap(self, svid: str, frame_idx: int) -> int | None:
        """Track id of whichever of this person's own assigned segments is
        closest to *frame_idx*, for frames `_track_id_at_frame` can't resolve
        at all -- a true gap between two segments (no track assignment
        covers it), not just a frame within a segment lacking a real
        per-frame detection.

        Used only to pick which wide-crop cluster to *look up*: the cache's
        own gap-search (`_TrackWindow.raw_rect` in wide_crop_cache.py) is
        what actually decides whether a crop exists this far from that
        track's last/next real detection. Querying with a too-distant
        track_id just costs a dict lookup that returns None -- there's no
        need to duplicate the cache's own gap-radius threshold here.
        """
        return _nearest_segment_track_id(self._track_segs.get(svid, []), frame_idx)

    def _track_id_by_frame(self, svid: str) -> dict[int, int]:
        """Expand `_track_segs[svid]` into a flat {frame: track_id} map.

        Used by `read_timeline_status` to resolve `keypoint_obs_quality` rows,
        which are keyed by track_id and can change mid-trial if the person's
        assignment switches tracks (e.g. after a detection-track split).
        """
        result: dict[int, int] = {}
        for track_id, first_frame, last_frame in self._track_segs.get(svid, []):
            for frame in range(first_frame, last_frame + 1):
                result[frame] = track_id
        return result

    def _refresh_timeline_status(self, cam_id: str) -> None:
        """Recompute timeline axis-1 status for one camera + inlier counts for all.

        Called once per camera in `_build()`, and again for the edited camera
        after any keypoint edit (nudge, move, toggle-outlier, interpolate,
        paste) so the timeline reflects the change immediately.
        """
        cam = next((c for c in self._cameras if c["camera_instance_id"] == cam_id), None)
        if cam is None:
            return
        svid = cam["shot_video_id"]
        self._timeline_status_by_cam[cam_id] = read_timeline_status(
            self._conn, self._sequence_id, cam_id,
            shot_video_id=svid,
            seg_run_id=self._seg_sources.get(svid),
            track_id_by_frame=self._track_id_by_frame(svid),
        )
        self._timeline_inlier_counts = compute_inlier_camera_counts(self._obs_kp)
        if self._timeline is not None:
            active_cam = self._cameras[self._timeline.active_camera_index()]
            if active_cam["camera_instance_id"] == cam_id:
                self._push_timeline_camera_data(self._timeline.active_camera_index())

    def _push_timeline_camera_data(self, cam_idx: int) -> None:
        """Push the cached status/inlier data for camera *cam_idx* into the timeline widget.

        Uses set_svid (not set_time_range) so this can be called on every
        post-edit status refresh and camera switch without resetting the
        user's zoom/pan — t_start/t_end/sync_table never change per camera.
        """
        if self._timeline is None or not (0 <= cam_idx < len(self._cameras)):
            return
        cam = self._cameras[cam_idx]
        self._timeline.set_svid(cam["shot_video_id"])
        self._timeline.set_status_data(
            self._timeline_status_by_cam.get(cam["camera_instance_id"], {}),
            self._timeline_inlier_counts,
            n_cameras=max(1, len(self._cameras)),
        )

    def _on_timeline_camera_changed(self, cam_idx: int) -> None:
        """User clicked a camera tab on the timeline: mirror it onto _sel_cam_idx."""
        self._sel_cam_idx = cam_idx
        self._push_timeline_camera_data(cam_idx)

    def _on_timeline_scrub(self, v: int) -> None:
        """Click/drag on the timeline: the timeline is now the only scrub control.

        Routes through the (headless, no longer shown) _slider so _on_slider
        remains the single place that updates _current_t, reloads the frame,
        and emits time_changed for external listeners (PersonPanel, etc).
        """
        if self._slider is not None:
            self._slider.setValue(v)

    def _sync_timeline(self, global_time: float) -> None:
        """Push cheap, per-scrub state (playhead + selection) to the timeline widget."""
        if self._timeline is None:
            return
        # If the user just selected a keypoint in a different camera's crop cell,
        # follow it — editing a keypoint in one camera almost always means you
        # want to see that same camera's timeline, not whichever tab was last
        # clicked directly on the timeline itself.
        if self._sel_cam_idx is not None and self._sel_cam_idx != self._timeline.active_camera_index():
            self._timeline.set_active_camera(self._sel_cam_idx)
            self._push_timeline_camera_data(self._sel_cam_idx)
        v = int(round((global_time - self._t_start) * 1000))
        self._timeline.set_current_time_v(v)
        self._timeline.set_selection(
            set(self._sel_kp_indices), self._range_start_v, self._range_end_v,
        )

    def _on_timeline_rubber_band(
        self, kp_indices: set[int], range_start_v: int, range_end_v: int, ctrl: bool,
    ) -> None:
        """Rubber-band drag (selects) or a plain click (clears) on the timeline row tree.

        The timeline emits an empty `kp_indices` set for a plain, non-Ctrl
        click — clearing the whole selection (keypoints *and* range) rather
        than collapsing it down to just the clicked row, which used to
        silently discard the rest of a multi-keypoint selection on a stray
        click. Seeking is handled separately by the ruler and never reaches
        this method.
        """
        if not kp_indices and not ctrl:
            self._sel_kp_indices = set()
            self._primary_kp_idx = None
            self._range_start_v = None
            self._range_end_v = None
            if self._timeline is not None:
                self._sel_cam_idx = self._timeline.active_camera_index()
            self._load_frame(self._current_t)
            return

        if ctrl:
            self._sel_kp_indices |= set(kp_indices)
        else:
            self._sel_kp_indices = set(kp_indices)
        if self._sel_kp_indices and self._primary_kp_idx not in self._sel_kp_indices:
            self._primary_kp_idx = next(iter(self._sel_kp_indices))
        elif not self._sel_kp_indices:
            self._primary_kp_idx = None
        if self._timeline is not None:
            self._sel_cam_idx = self._timeline.active_camera_index()
        if range_start_v != range_end_v:
            self._range_start_v = min(range_start_v, range_end_v)
            self._range_end_v = max(range_start_v, range_end_v)
        self._load_frame(self._current_t)

    def _on_timeline_select_to_marker(self, kp_indices) -> None:
        """"Select to marker" from a timeline row's right-click menu:
        selects that row's keypoint(s) and the frame range between the
        marker and the current playhead, whichever order they fall in."""
        if self._timeline is None:
            return
        marker_v = self._timeline.marker_v()
        if marker_v is None:
            return
        current_v = int(round((self._current_t - self._t_start) * 1000))
        self._sel_kp_indices = set(kp_indices) - self._hidden_kp_indices
        self._primary_kp_idx = next(iter(self._sel_kp_indices), None)
        self._sel_cam_idx = self._timeline.active_camera_index()
        self._range_start_v = min(marker_v, current_v)
        self._range_end_v = max(marker_v, current_v)
        self._load_frame(self._current_t)

    def _on_timeline_select_all_to_marker(self) -> None:
        """"Select all keypoints to marker" from the ruler's right-click
        menu: every visible keypoint, over the range between the marker
        and the current playhead."""
        if self._timeline is None:
            return
        marker_v = self._timeline.marker_v()
        if marker_v is None:
            return
        current_v = int(round((self._current_t - self._t_start) * 1000))
        self._sel_kp_indices = set(self._pose_model.all_indices) - self._hidden_kp_indices
        self._primary_kp_idx = next(iter(self._sel_kp_indices), None)
        self._sel_cam_idx = self._timeline.active_camera_index()
        self._range_start_v = min(marker_v, current_v)
        self._range_end_v = max(marker_v, current_v)
        self._load_frame(self._current_t)

    def _on_timeline_keyframe_toggle(self, kp_idx: int, time_v: int) -> None:
        """Ctrl+click on the timeline: freeze the current position as a keyframe, or
        un-freeze it if it already is one (see *Multi-keyframe interpolation* in the
        design doc — an interior anchor is any frame with an edit row where
        `is_outlier == 0`, i.e. STATUS_BLUE)."""
        from app.pose.db_cache import (
            clear_single_keypoint_edit,
            read_observations_with_edits,
            update_single_keypoint_edit,
        )
        if self._timeline is None or not self._sync_table:
            return
        cam_idx = self._timeline.active_camera_index()
        if not (0 <= cam_idx < len(self._cameras)):
            return
        cam = self._cameras[cam_idx]
        cam_id = cam["camera_instance_id"]
        svid = cam["shot_video_id"]
        frame_idx = self._sync_table.lookup(self._t_start + time_v / 1000.0, svid)
        if frame_idx is None:
            return

        status = self._timeline_status_by_cam.get(cam_id, {}).get(frame_idx)
        is_keyframe = status is not None and kp_idx < len(status) and int(status[kp_idx]) == STATUS_BLUE
        if is_keyframe:
            clear_single_keypoint_edit(self._conn, self._sequence_id, cam_id, frame_idx, kp_idx)
        else:
            kp = self._obs_kp.get(cam_id, {}).get(frame_idx)
            if kp is None or kp_idx >= kp.shape[0]:
                return
            update_single_keypoint_edit(
                self._conn, self._sequence_id, cam_id, frame_idx, kp_idx,
                float(kp[kp_idx, 0]), float(kp[kp_idx, 1]), is_outlier=False,
            )
        self._maybe_queue_hand_redetect(cam_id, svid, frame_idx, kp_idx)
        self._obs_kp[cam_id] = read_observations_with_edits(self._conn, self._sequence_id, cam_id)
        self._refresh_timeline_status(cam_id)
        self._load_frame(self._current_t)

    def _on_timeline_visibility_toggled(self, kp_indices) -> None:
        """Eye-icon click on the timeline: hide/show a keypoint (leaf row) or
        every keypoint in a group (group row). If every index in *kp_indices*
        is already hidden, this shows them all; otherwise it hides them all —
        the same rule a tri-state "select all" checkbox uses, so clicking a
        partially-hidden group's icon hides the rest rather than leaving the
        group in a confusing mixed state.
        """
        kp_indices = set(kp_indices)
        if not kp_indices:
            return
        if kp_indices <= self._hidden_kp_indices:
            self._hidden_kp_indices -= kp_indices
        else:
            self._hidden_kp_indices |= kp_indices
        # Hidden keypoints can't stay selected, moved, or interpolated.
        self._sel_kp_indices -= self._hidden_kp_indices
        if self._primary_kp_idx in self._hidden_kp_indices:
            self._primary_kp_idx = next(iter(self._sel_kp_indices), None)
        if self._timeline is not None:
            self._timeline.set_hidden(frozenset(self._hidden_kp_indices))
        self._load_frame(self._current_t)

    def _load_tracking_run(self, run_id: str | None) -> None:
        """Start async loading of tracking overlay data.

        Cells are shown in a loading state immediately; the overlay is applied
        once the background thread finishes.
        """
        # Abandon any in-flight load: disconnect data and cleanup signals so stale
        # results are ignored.  _ACTIVE_LOADERS keeps the thread alive until it exits.
        if self._loader is not None:
            for sig, slot in [
                (self._loader.loaded,    self._on_tracking_loaded),
                (self._loader.finished,  self._on_loader_finished),
            ]:
                try:
                    sig.disconnect(slot)
                except RuntimeError:
                    pass
            self._loader = None

        self._marker_proj.clear()
        self._joint_proj.clear()
        self._bone_pairs.clear()
        self._tracking_timestamps.clear()
        self._outlier_masks.clear()

        if not run_id:
            self._load_frame(self._current_t)
            return

        # Get the db file path from the open connection
        db_row = self._conn.execute("PRAGMA database_list").fetchone()
        db_path = db_row[2] if db_row else None
        if not db_path:
            self._load_frame(self._current_t)
            return

        # Show loading indicator on all cells while the thread runs
        for cell in self._cells:
            cell.show_loading()

        self._loader = _TrackingRunLoader(db_path, run_id)
        # _on_finished removes from _ACTIVE_LOADERS and calls deleteLater — must fire
        # even if the widget is gone, so connect as a self-connection on the thread.
        self._loader.finished.connect(self._loader._on_finished)
        # _on_loader_finished clears self._loader only after the thread has fully exited.
        # Qt auto-disconnects this if the widget is destroyed before finished fires.
        self._loader.finished.connect(self._on_loader_finished)
        # Explicit QueuedConnection: _on_tracking_loaded is fired from the UI thread
        # event loop, not directly from run() — critical because loaded is emitted
        # while run() is still executing its finally block.
        self._loader.loaded.connect(
            self._on_tracking_loaded, Qt.ConnectionType.QueuedConnection
        )
        self._loader.start()

    def _on_tracking_loaded(
        self,
        tracking_timestamps: list,
        marker_proj: dict,
        joint_proj: dict,
        bone_pairs: list,
        outlier_masks: dict,
    ) -> None:
        # Do NOT clear self._loader here — the thread's run() may still be executing
        # its finally block.  _on_loader_finished (connected to finished) does it safely.
        self._tracking_timestamps = tracking_timestamps
        self._marker_proj = marker_proj
        self._joint_proj = joint_proj
        self._bone_pairs = bone_pairs
        self._outlier_masks = outlier_masks
        self._load_frame(self._current_t)

    def _on_loader_finished(self) -> None:
        """Connected to _TrackingRunLoader.finished(). Safe to drop the reference here
        because finished() is emitted only after run() has returned and the OS thread
        has fully exited — isRunning() is False by this point."""
        self._loader = None

    def _apply_overlay(
        self,
        cell,
        cam_id: str,
        svid: str,
        frame_idx: int,
        tracking_step: "int | None",
        show_detected: bool,
        show_tracked: bool,
    ) -> None:
        """Set cell's keypoint/tracked-skeleton overlay (and, in edit mode,
        the trail/selection/range-highlight overlays) for one camera/frame.

        Independent of whether an image was actually found for this frame --
        an edited keypoint or a tracked skeleton stays visible (drawn over a
        black background) even on a frame with no raw detection at all, in
        both view and edit mode. See "View-mode parity" in the design doc.
        """
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
        cell.set_hidden(frozenset(self._hidden_kp_indices))
        if self._edit_mode:
            trail = None
            kp_name = None
            if self._primary_kp_idx is not None:
                kp_by_frame = self._obs_kp.get(cam_id, {})
                trail = _build_cam_trail(kp_by_frame, cam_id, frame_idx, self._primary_kp_idx)
                kp_name = self._pose_model.name_of(self._primary_kp_idx)
            cell.set_trail(trail)
            cell.set_selection(
                self._primary_kp_idx,
                frozenset(self._sel_kp_indices),
                name=kp_name,
            )
            cell.set_range_highlights(
                self._compute_range_highlights(cam_id, svid)
            )

    def _display_crop_result(
        self,
        cell,
        cam_id: str,
        svid: str,
        frame_idx: int,
        tracking_step: int | None,
        show_detected: bool,
        show_tracked: bool,
        result: tuple,
        target_rect: "tuple[float, float, float, float] | None" = None,
        layer_label: str = "",
        show_debug: bool = False,
    ) -> None:
        """Decode a (jpeg, wpx, hpx, src_x, src_y, src_w, src_h) crop and render
        it into *cell*, including keypoint/tracking overlays and (in edit mode)
        the trail/selection/range-highlight overlays.

        Shared by the wide-crop cluster cache (see "Background wide-crop frame
        cache" in the design doc) and the Phase 6 in-memory synthetic-crop
        path, since both produce results in this same shape.

        *target_rect*, if given, is the desired display window (full-frame
        pixel coordinates) -- used by the wide-crop cluster cache to sub-crop
        a possibly multi-person cluster image down to just the person being
        edited, widened to cover whatever keypoints/tracked-skeleton overlay
        is actually about to be drawn (see *_windowed_kp_bbox* /
        `_tracked_overlay_bbox`). Unlike a plain sub-crop, *target_rect* is
        not required to fit inside what's actually been decoded: any part of
        it outside the decoded pixels is filled black rather than shrinking
        the display down to whatever *is* available (see "Unified
        minimum-display bbox..." in the design doc) -- this keeps the
        overlay's coordinate mapping (and the requested framing) correct even
        when the cache hasn't caught up with a large edit or a fast-moving
        tracked skeleton yet.

        *layer_label*/*show_debug* feed the "Debug overlay" corner label --
        see the design doc.
        """
        import cv2
        import numpy as np

        jpeg, wpx, hpx, src_x, src_y, src_w, src_h = result
        buf = np.frombuffer(jpeg, dtype=np.uint8)
        crop_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if crop_bgr is None:
            cell.show_empty()
            self._apply_overlay(
                cell, cam_id, svid, frame_idx, tracking_step, show_detected, show_tracked,
            )
            return
        x1 = float(src_x)
        y1 = float(src_y)
        jpeg_h = float(hpx or crop_bgr.shape[0])
        src_scale = jpeg_h / float(src_h) if src_h > 0 else 1.0

        black_filled = False
        if target_rect is not None:
            crop_bgr, x1, y1, black_filled = _composite_black_fill(
                crop_bgr, x1, y1, src_scale, target_rect,
            )

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = crop_rgb.shape[:2]
        qimg = QImage(crop_rgb.data, w_img, h_img, 3 * w_img, QImage.Format.Format_RGB888)
        cell.show_image(QPixmap.fromImage(qimg), x1, y1, src_scale)
        cell.set_debug_label(
            (layer_label + (" +black-fill" if black_filled else "")) if show_debug else None
        )
        self._apply_overlay(
            cell, cam_id, svid, frame_idx, tracking_step, show_detected, show_tracked,
        )

    def _compute_target_rect(
        self,
        cell,
        cam_id: str,
        svid: str,
        frame_idx: int,
        tracking_step: "int | None",
        own_rect: "tuple[float, float, float, float]",
    ) -> "tuple[float, float, float, float]":
        """Minimum display bbox for one camera/frame: *own_rect* (whichever
        crop-source layer's own decoded extent) unioned with the windowed
        keypoint bbox and the tracked-skeleton bbox, padded by a margin, and
        grown to the cell's own aspect ratio.

        Shared by every crop-source layer (wide-crop cluster cache, low-res
        per-track cache, in-memory backfill) so they all frame the same way
        regardless of which layer actually served the image -- see "Unified
        minimum-display bbox..." in the design doc.
        """
        kp_bbox = _sane_bbox(_windowed_kp_bbox(
            self._obs_kp.get(cam_id, {}),
            frame_idx,
            frozenset(self._hidden_kp_indices),
        ))
        tracked_bbox = _sane_bbox(_tracked_overlay_bbox(
            self._joint_proj.get(cam_id, {}).get(tracking_step)
            if tracking_step is not None else None,
            self._marker_proj.get(cam_id, {}).get(tracking_step)
            if tracking_step is not None else None,
            self._video_dims.get(svid),
        ))
        desired = own_rect
        for bbox in (kp_bbox, tracked_bbox):
            if bbox is not None:
                # Widen to cover whatever's actually drawn -- an edit (or a
                # tracking run's own projection) can place a point far from
                # where the raw (possibly wrong) detection said the person was.
                desired = (
                    min(desired[0], bbox[0]), min(desired[1], bbox[1]),
                    max(desired[2], bbox[2]), max(desired[3], bbox[3]),
                )
        # Margin around the minimum display bbox so overlay points never sit
        # flush against the cell edge.
        mx = (desired[2] - desired[0]) * _DISPLAY_MARGIN_FRAC
        my = (desired[3] - desired[1]) * _DISPLAY_MARGIN_FRAC
        desired = (desired[0] - mx, desired[1] - my, desired[2] + mx, desired[3] + my)
        # Fill the cell instead of letterboxing: grow (never shrink) to the
        # *canvas's* own aspect ratio. Uses cell._canvas's own dimensions,
        # not the outer _CropCell's -- _CropCell stacks a title bar above
        # the canvas, so its aspect ratio isn't the canvas's; using it here
        # would under-correct.
        canvas_w, canvas_h = cell._canvas.width(), cell._canvas.height()
        if canvas_w > 0 and canvas_h > 0:
            desired = _expand_rect_to_aspect(desired, canvas_w / canvas_h)
        return desired

    def _load_frame(self, global_time: float) -> None:
        import cv2
        import numpy as np

        if not self._det_run_id or not self._sync_table:
            for cell in self._cells:
                cell.show_empty()
            return

        show_detected = self._show_detected is None or self._show_detected.isChecked()
        show_tracked = self._show_tracked is None or self._show_tracked.isChecked()
        show_seg = self._show_seg is not None and self._show_seg.isChecked()
        show_debug = self._show_debug is not None and self._show_debug.isChecked()

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

            # Preferred layer: wide-crop cluster cache -- generous, higher-quality
            # crop, shared across nearby people where their crop windows overlap.
            # Falls through to the existing chain below if no cluster entry
            # exists yet for this frame/track (worker hasn't reached it), or if
            # the cached crop turns out not to cover what's actually being
            # displayed for this frame (see below).
            if self._edit_mode and self._wide_crop_mgr is not None:
                wide_track_id = self._track_id_at_frame(svid, frame_idx)
                if wide_track_id is None:
                    # A true gap between two assigned segments (no track
                    # covers this frame at all) -- fall back to whichever of
                    # this person's own segments is nearest, so the cache's
                    # own gap-search still gets a chance to serve a crop
                    # anchored to the last/next real detection.
                    wide_track_id = self._nearest_track_id_for_gap(svid, frame_idx)
                if wide_track_id is not None:
                    wide_lookup = self._wide_crop_mgr.get_cluster_result(svid, frame_idx, wide_track_id)
                    if wide_lookup is not None:
                        wide, own_rect = wide_lookup

                        # Sub-crop to this track's own padded window, not the
                        # whole (possibly multi-person) cluster image -- a
                        # cluster spanning several spread-out people would
                        # otherwise show most of the room for someone editing
                        # just one of them. No longer clamped to what this
                        # cluster image actually has decoded: _display_crop_result
                        # now black-fills any part of `desired` outside the
                        # decoded pixels instead of the crop silently shrinking
                        # (or being rejected outright) when an edit or a
                        # tracked-skeleton projection reaches beyond what's
                        # cached so far -- see "Unified minimum-display
                        # bbox..." in the design doc.
                        desired = self._compute_target_rect(
                            cell, cam_id, svid, frame_idx, tracking_step, own_rect,
                        )
                        if desired[2] > desired[0] and desired[3] > desired[1]:
                            self._display_crop_result(
                                cell, cam_id, svid, frame_idx, tracking_step,
                                show_detected, show_tracked, wide, target_rect=desired,
                                layer_label="wide-cache", show_debug=show_debug,
                            )
                            continue
                        # Else: degenerate window (shouldn't normally happen
                        # given a valid own_rect) -- fall through.
                    else:
                        self._wide_crop_mgr.prioritise(svid, frame_idx)

            # Check in-memory results first (frames decoded without a detection bbox).
            if self._backfill is not None:
                mem = self._backfill.get_mem_result(svid, frame_idx)
                if mem is not None:
                    _, _, _, mem_x, mem_y, mem_w, mem_h = mem
                    own_rect = (mem_x, mem_y, mem_x + mem_w, mem_y + mem_h)
                    desired = self._compute_target_rect(
                        cell, cam_id, svid, frame_idx, tracking_step, own_rect,
                    )
                    self._display_crop_result(
                        cell, cam_id, svid, frame_idx, tracking_step,
                        show_detected, show_tracked, mem, target_rect=desired,
                        layer_label="backfill/ghost", show_debug=show_debug,
                    )
                    continue

            # Find which track covers this frame for this camera. Runs in
            # both modes -- the low-res backfill worker (self._backfill) is
            # started once, unconditionally, when the sequence loads (see
            # _build), not only in edit mode, so a view-mode miss also
            # prioritises and eventually fills in instead of leaving a
            # permanent placeholder. See "View-mode parity" in the design doc.
            track_id = self._track_id_at_frame(svid, frame_idx)
            if track_id is None:
                _log.debug(
                    "_load_frame: no track  svid=%s  frame=%d  t=%.3f",
                    svid[-8:], frame_idx, global_time,
                )
                if self._backfill is not None:
                    cell.show_loading()
                    self._backfill.prioritise(svid, frame_idx)
                else:
                    cell.show_empty()
                # An edited keypoint or an active tracking run can still be
                # meaningful here even with zero raw detection for this
                # frame -- keep it visible (drawn over black) rather than
                # blanking it along with the missing image.
                self._apply_overlay(
                    cell, cam_id, svid, frame_idx, tracking_step, show_detected, show_tracked,
                )
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
                _log.debug(
                    "_load_frame: no crop  svid=%s  frame=%d  track=%s",
                    svid[-8:], frame_idx, track_id,
                )
                if self._backfill is not None:
                    cell.show_loading()
                    self._backfill.prioritise(svid, frame_idx)
                else:
                    cell.show_empty()
                self._apply_overlay(
                    cell, cam_id, svid, frame_idx, tracking_step, show_detected, show_tracked,
                )
                continue

            buf = np.frombuffer(bytes(row["image_data"]), dtype=np.uint8)
            crop_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

            if crop_bgr is None:
                cell.show_empty()
                self._apply_overlay(
                    cell, cam_id, svid, frame_idx, tracking_step, show_detected, show_tracked,
                )
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

            # Segmentation mask overlay (blended directly into the JPEG before Qt conversion)
            if show_seg:
                sqr_id = self._seg_sources.get(svid)
                if sqr_id and track_id is not None:
                    mask_row = self._conn.execute(
                        "SELECT mask_blob FROM seg_masks "
                        "WHERE seg_quality_run_id=? AND shot_video_id=? AND frame_idx=?",
                        (sqr_id, svid, frame_idx),
                    ).fetchone()
                    if mask_row:
                        mask_buf = np.frombuffer(bytes(mask_row["mask_blob"]), dtype=np.uint8)
                        full_mask = cv2.imdecode(mask_buf, cv2.IMREAD_UNCHANGED)
                        if full_mask is not None:
                            if full_mask.ndim == 3:
                                full_mask = full_mask[:, :, 0]
                            # Crop mask to the same source region as the JPEG.
                            # Masks may be stored at a lower resolution than the video
                            # (e.g. FHD masks on 4K cameras), so scale src coordinates.
                            m_h, m_w = full_mask.shape[:2]
                            vid_dims = self._video_dims.get(svid)
                            if vid_dims and vid_dims[0] > 0 and vid_dims[1] > 0:
                                sx_scale = m_w / vid_dims[0]
                                sy_scale = m_h / vid_dims[1]
                            else:
                                sx_scale = sy_scale = 1.0
                            src_w_val = row["src_w"] if row["src_w"] is not None else (
                                crop_bgr.shape[1] / src_scale if src_scale > 0 else crop_bgr.shape[1]
                            )
                            src_h_val = float(row["src_h"] or src_h)
                            mx1 = max(0, int(x1 * sx_scale))
                            my1 = max(0, int(y1 * sy_scale))
                            mx2 = max(mx1 + 1, min(m_w, int((x1 + src_w_val) * sx_scale)))
                            my2 = max(my1 + 1, min(m_h, int((y1 + src_h_val) * sy_scale)))
                            mask_crop = full_mask[my1:my2, mx1:mx2]
                            if mask_crop.size > 0:
                                mask_crop = cv2.resize(
                                    mask_crop,
                                    (crop_bgr.shape[1], crop_bgr.shape[0]),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                                # DAVIS palette — BGR
                                _DAVIS_COLORS_BGR = [
                                    (80, 80, 240), (120, 200, 80), (240, 120, 80),
                                    (60, 200, 240), (240, 80, 180), (220, 60, 60),
                                    (80, 240, 140), (60, 140, 240), (240, 200, 60),
                                ]
                                color_idx = (track_id - 1) % len(_DAVIS_COLORS_BGR)
                                color_bgr = _DAVIS_COLORS_BGR[color_idx]
                                fg = mask_crop == track_id
                                if fg.any():
                                    overlay = crop_bgr.astype(np.float32)
                                    for c, cv_ in enumerate(color_bgr):
                                        overlay[:, :, c][fg] = (
                                            overlay[:, :, c][fg] * 0.55 + cv_ * 0.45
                                        )
                                    crop_bgr = overlay.astype(np.uint8)

            # Widen to the same minimum display bbox every other layer uses
            # (own extent unioned with keypoint/tracked-skeleton overlays),
            # black-filling whatever's outside this crop's own decoded
            # extent -- see "Unified minimum-display bbox..." in the design
            # doc. Must happen after the segmentation blend above, which
            # operates on the crop in its original decoded extent/origin.
            own_rect = (x1, y1, x1 + crop_bgr.shape[1] / src_scale, y1 + crop_bgr.shape[0] / src_scale)
            desired = self._compute_target_rect(
                cell, cam_id, svid, frame_idx, tracking_step, own_rect,
            )
            black_filled = False
            if desired[2] > desired[0] and desired[3] > desired[1]:
                crop_bgr, x1, y1, black_filled = _composite_black_fill(
                    crop_bgr, x1, y1, src_scale, desired,
                )

            # Convert image to QPixmap (no vector overlays drawn into the image)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            h_img, w_img = crop_rgb.shape[:2]
            qimg = QImage(
                crop_rgb.data, w_img, h_img, 3 * w_img, QImage.Format.Format_RGB888
            )
            cell.show_image(QPixmap.fromImage(qimg), x1, y1, src_scale)
            cell.set_debug_label(
                ("low-res" + (" +black-fill" if black_filled else "")) if show_debug else None
            )
            self._apply_overlay(
                cell, cam_id, svid, frame_idx, tracking_step, show_detected, show_tracked,
            )

        self._sync_timeline(global_time)


# ---------------------------------------------------------------------------
# _LineChart — small metric time-series widget with a cursor line
# ---------------------------------------------------------------------------

# Matches the posetrak MCP diagnostic server's own condition-number warning
# threshold (app/mcp/tools/diagnostics.py: _COV_COND_WARN) -- drawn as a
# reference line so a spike above it is visible at a glance, not just in
# text form in the "Current frame" box above.
_COV_COND_WARN = 1_000_000.0


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

    def __init__(
        self,
        title: str,
        reference_y: float | None = None,
        log_y: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._reference_y = reference_y
        self._log_y = log_y
        self._steps: list[int] = []
        self._values: list[float | None] = []
        self._cursor_step: int | None = None
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _y_space(self, v: float) -> "float | None":
        """Map a raw value into the space the Y axis is actually drawn in --
        log10 when `log_y` (values <= 0 have no log and are treated as
        missing), otherwise unchanged. A metric like the covariance
        condition number naturally spans several orders of magnitude; on a
        linear axis one bad frame's spike flattens every other problem frame
        into visual insignificance near zero, whereas log scale keeps
        relative differences between large values visible without dropping
        (clipping) any of them.
        """
        if not self._log_y:
            return v
        return log10(v) if v > 0 else None

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

        raw = [v for v in self._values if v is not None and v == v]  # skip None/NaN
        vals = [t for t in (self._y_space(v) for v in raw) if t is not None]
        if not self._steps or not vals:
            return

        ref_y = self._y_space(self._reference_y) if self._reference_y is not None else None
        y_min, y_max = min(vals), max(vals)
        # Extend range to keep reference_y visible even if data is all above/below it
        if ref_y is not None:
            y_min = min(y_min, ref_y)
            y_max = max(y_max, ref_y)
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
        for step, raw_val in zip(self._steps, self._values):
            if raw_val is None or raw_val != raw_val:
                started = False
                continue
            val = self._y_space(raw_val)
            if val is None:
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

        # Reference line (e.g. NIS/DOF = 1 indicating well-calibrated filter)
        if ref_y is not None and y_min <= ref_y <= y_max:
            ry = sy(ref_y)
            painter.setPen(QPen(QColor(220, 50, 50), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(pl, ry), QPointF(w - pr, ry))

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
        self._run_row: sqlite3.Row | None = None
        self._cur_step: int | None = None
        self._cur_ts: float | None = None
        self.setMinimumWidth(self._MIN_WIDTH)
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # --- Run info ---
        run_box = _CollapsibleBox("Run info")
        run_form = QFormLayout()
        run_form.setHorizontalSpacing(6)
        run_form.setVerticalSpacing(2)
        self._ri_skeleton = QLabel("—")
        self._ri_person = QLabel("—")
        self._ri_ran_at = QLabel("—")
        self._ri_version = QLabel("—")
        self._ri_version.setWordWrap(True)
        self._ri_frames = QLabel("—")
        self._ri_cfg = QLabel("—")
        self._ri_cfg.setWordWrap(True)
        self._ri_notes = QLabel("—")
        self._ri_notes.setWordWrap(True)
        run_form.addRow("Skeleton:", self._ri_skeleton)
        run_form.addRow("Person:", self._ri_person)
        run_form.addRow("Tracked at:", self._ri_ran_at)
        run_form.addRow("Binary:", self._ri_version)
        run_form.addRow("Frames:", self._ri_frames)
        run_form.addRow("Config:", self._ri_cfg)
        run_form.addRow("Notes:", self._ri_notes)
        run_box.inner_layout().addLayout(run_form)
        root.addWidget(run_box)

        # --- IDs (UUIDs / SHA for cross-referencing) ---
        ids_box, self._id_widgets = _build_run_ids_group()
        root.addWidget(ids_box)

        # --- Current frame ---
        frame_box = _CollapsibleBox("Current frame")
        frame_form = QFormLayout()
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
        frame_box.inner_layout().addLayout(frame_form)
        copy_frame_btn = _action_btn("⎘ Copy frame ID")
        copy_frame_btn.setToolTip(
            "Copy db path, run/trial/capture IDs, and the current step/timestamp "
            "to the clipboard, in one paste-able block."
        )
        copy_frame_btn.clicked.connect(self._copy_frame_id)
        frame_box.inner_layout().addWidget(copy_frame_btn)
        root.addWidget(frame_box)

        # --- Charts ---
        charts_box = _CollapsibleBox("Metrics")
        self._nis_chart = _LineChart("NIS / DOF", reference_y=1.0)
        self._cov_chart = _LineChart(
            "Covariance condition # (log scale)", reference_y=_COV_COND_WARN, log_y=True,
        )
        charts_box.inner_layout().addWidget(self._nis_chart)
        charts_box.inner_layout().addWidget(self._cov_chart)
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
                    self._ri_version, self._ri_frames, self._ri_cfg, self._ri_notes):
            lbl.setText("—")
        for w in self._id_widgets.values():
            _set_id_widget(w, None)
        for lbl in (self._fi_step, self._fi_time, self._fi_inliers,
                    self._fi_nis, self._fi_cov):
            lbl.setText("—")
        self._nis_chart.set_data([], [])
        self._cov_chart.set_data([], [])
        self._run_row = None
        self._cur_step = None
        self._cur_ts = None

        if not run_id:
            return

        run = self._conn.execute(_RUN_INFO_SQL, (run_id,)).fetchone()
        if not run:
            return
        self._run_row = run

        n_frames = self._conn.execute(
            "SELECT COUNT(*) FROM tracking_results "
            "WHERE run_id=? AND person_id=0 AND is_smoothed=0",
            (run_id,),
        ).fetchone()[0]

        self._ri_skeleton.setText(run["skel_name"] or "—")
        self._ri_person.setText(run["person_names"] or "—")
        self._ri_ran_at.setText(_fmt_ts(run["ran_at"]))
        self._ri_version.setText(run["posetrak_version"] or "—")
        self._ri_frames.setText(str(n_frames))
        notes = run["notes"]
        self._ri_notes.setText(notes if notes else "—")
        self._ri_notes.setVisible(bool(notes))

        # IDs
        _populate_run_ids(self._id_widgets, run)

        # Tracker config params
        cfg_id = run["tracker_config_id"]
        cfg = self._conn.execute(_CFG_SQL, (cfg_id,)).fetchone() if cfg_id else None
        self._ri_cfg.setText(_cfg_text(cfg, cfg_id))

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

        self._cur_step = step
        self._cur_ts = row["timestamp_s"]
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

    def _copy_frame_id(self) -> None:
        if self._run_row is None:
            return
        text = _frame_identifier_text(
            _db_path_of(self._conn), self._run_row, self._cur_step, self._cur_ts,
        )
        QApplication.clipboard().setText(text)


# ---------------------------------------------------------------------------
# PersonPanel
# ---------------------------------------------------------------------------


class PersonPanel(QWidget):
    """Person panel: info, tracking history, and tracker launcher."""

    _bvh_export_done  = Signal(str, str)  # (out_path, error — empty string = success)
    _usd_export_done  = Signal(str, str)
    _gltf_export_done = Signal(str, str)

    def __init__(self, conn: sqlite3.Connection, sequence_id: str,
                 session_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._sequence_id = sequence_id
        self._session_path = session_path
        self._bvh_export_done.connect(self._on_bvh_done)
        self._usd_export_done.connect(self._on_usd_done)
        self._gltf_export_done.connect(self._on_gltf_done)
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

        # ------------------------------------------------------------------
        # Right sidebar content
        # ------------------------------------------------------------------
        title_lbl = QLabel(f"<b>{person_names or seq['name'] or 'Person'}</b>")
        title_lbl.setWordWrap(True)

        hide_btn = QToolButton()
        hide_btn.setText("✕")
        hide_btn.setToolTip("Hide info panel")
        hide_btn.setFixedSize(20, 20)

        title_row = QHBoxLayout()
        title_row.addWidget(title_lbl, stretch=1)
        title_row.addWidget(hide_btn)

        form_box = _section("Person info")
        form = QFormLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(2)
        form.addRow("Persons:", QLabel(person_names or "—"))
        form.addRow("Time range:", QLabel(
            f"{_fmt_time(seq['time_start_s'])}  →  {_fmt_time(seq['time_end_s'])}"
        ))
        form.addRow("Observations:", QLabel(str(n_obs)))
        form.addRow("Pose model:", QLabel(seq["pose_model"] or "—"))
        form_box.inner_layout().addLayout(form)

        # --- Tracking runs section ---
        self._run_box = _section("Tracking runs (0)")
        box_vbox = self._run_box.inner_layout()

        self._run_list = QListWidget()
        self._run_list.setMaximumHeight(130)
        self._run_list.currentItemChanged.connect(self._on_run_selected)
        box_vbox.addWidget(self._run_list)

        self._run_detail = QLabel("")
        self._run_detail.setWordWrap(True)
        self._run_detail.setVisible(False)
        box_vbox.addWidget(self._run_detail)

        # Buttons as vertical column in the narrow sidebar
        self._export_bvh_btn = QPushButton("Export BVH…")
        self._export_bvh_btn.setEnabled(False)
        self._export_bvh_btn.clicked.connect(self._export_bvh)
        self._export_usd_btn = QPushButton("Export USD…")
        self._export_usd_btn.setEnabled(False)
        if not _USD_AVAILABLE:
            self._export_usd_btn.setToolTip(_USD_TOOLTIP)
        self._export_usd_btn.clicked.connect(self._export_usd)
        self._export_gltf_btn = QPushButton("Export glTF…")
        self._export_gltf_btn.setEnabled(False)
        self._export_gltf_btn.clicked.connect(self._export_gltf)
        self._delete_run_btn = QPushButton("Delete run")
        self._delete_run_btn.setEnabled(False)
        self._delete_run_btn.clicked.connect(self._delete_run)
        self._scale_btn = QPushButton("Scale skeleton…")
        self._scale_btn.setEnabled(False)
        self._scale_btn.setToolTip(
            "Measure bone lengths from inlier observations and scale the skeleton"
        )
        self._scale_btn.clicked.connect(self._open_scaling)
        run_btn = _action_btn("Run tracker…")
        run_btn.clicked.connect(self._open_run_tracker)

        for btn in (self._export_bvh_btn, self._export_usd_btn, self._export_gltf_btn,
                    self._scale_btn, self._delete_run_btn):
            box_vbox.addWidget(btn)

        # Run info pane (per-frame stats + charts) lives below the runs section
        self._info_pane = _RunInfoPane(self._conn)

        sidebar_content = QWidget()
        sb_v = QVBoxLayout(sidebar_content)
        sb_v.setContentsMargins(6, 4, 6, 4)
        sb_v.setSpacing(6)
        sb_v.addLayout(title_row)
        sb_v.addWidget(form_box)
        sb_v.addWidget(self._run_box)
        sb_v.addWidget(run_btn)
        sb_v.addWidget(self._info_pane)
        sb_v.addStretch()

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidget(sidebar_content)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setMinimumWidth(220)

        # Show button + top bar (visible only while sidebar is hidden)
        show_btn = QToolButton()
        show_btn.setText("ℹ")
        show_btn.setToolTip("Show info panel")
        show_btn.setVisible(False)

        top_bar = QWidget()
        top_bar.setMaximumHeight(24)
        top_bar_h = QHBoxLayout(top_bar)
        top_bar_h.setContentsMargins(0, 2, 4, 0)
        top_bar_h.addStretch()
        top_bar_h.addWidget(show_btn)
        top_bar.setVisible(False)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        def _toggle_sidebar() -> None:
            visible = not sidebar_scroll.isVisible()
            sidebar_scroll.setVisible(visible)
            top_bar.setVisible(not visible)
            if visible:
                w = self._splitter.width()
                self._splitter.setSizes([w - 300, 300])

        hide_btn.clicked.connect(_toggle_sidebar)
        show_btn.clicked.connect(_toggle_sidebar)

        # ------------------------------------------------------------------
        # Camera grid (left pane)
        # ------------------------------------------------------------------
        self._crop_grid = PersonCropGridWidget(self._conn, self._sequence_id)
        self._crop_grid.time_changed.connect(self._info_pane.on_time_changed)

        self._refresh_runs()

        left_w = QWidget()
        left_v = QVBoxLayout(left_w)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(0)
        left_v.addWidget(top_bar)
        left_v.addWidget(self._crop_grid, stretch=1)

        self._splitter.addWidget(left_w)
        self._splitter.addWidget(sidebar_scroll)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(1, True)
        self._splitter.setSizes([700, 300])

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._splitter)

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
        self._export_gltf_btn.setEnabled(False)
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
            self._export_gltf_btn.setEnabled(False)
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
        self._export_gltf_btn.setEnabled(True)
        self._delete_run_btn.setEnabled(True)
        self._scale_btn.setEnabled(bool(self._session_path))

        if self._crop_grid is not None:
            self._crop_grid.set_tracking_run(run_id)
        if self._info_pane is not None:
            self._info_pane.load_run(run_id)

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

    def _export_gltf(self) -> None:
        item = self._run_list.currentItem()
        run_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not run_id or not self._session_path:
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save glTF file", "",
            "glTF binary (*.glb);;glTF JSON (*.gltf)"
        )
        if not out_path:
            return
        if not (out_path.endswith(".glb") or out_path.endswith(".gltf")):
            out_path += ".glb"

        self._export_gltf_btn.setEnabled(False)
        from posetrak.export.gltf import export_gltf
        session_db = str(self._session_path)

        def _run() -> None:
            error = ""
            try:
                export_gltf(out_path, session_db=session_db, run_id=run_id)
            except Exception as exc:
                error = str(exc)
            self._gltf_export_done.emit(out_path, error)

        threading.Thread(target=_run, daemon=True).start()

    def _on_gltf_done(self, out_path: str, error: str) -> None:
        self._export_gltf_btn.setEnabled(True)
        if error:
            QMessageBox.critical(self, "Export failed", f"glTF export failed:\n\n{error}")
        else:
            QMessageBox.information(self, "Export complete", f"glTF written to:\n{out_path}")

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
        self._export_gltf_btn: QPushButton | None = None
        self._nis_chart: _LineChart | None = None
        self._cov_chart: _LineChart | None = None
        self._trp_tracking_timestamps: list[tuple[float, int]] = []
        self._trp_step_stats: dict[int, object] = {}
        self._trp_fi_step: QLabel | None = None
        self._trp_fi_time: QLabel | None = None
        self._trp_fi_inliers: QLabel | None = None
        self._trp_fi_nis: QLabel | None = None
        self._trp_fi_cov: QLabel | None = None
        self._run_row: sqlite3.Row | None = None
        self._trp_cur_step: int | None = None
        self._trp_cur_ts: float | None = None
        self._export_done.connect(self._on_export_done)
        self._build()

    def _build(self) -> None:
        run = self._conn.execute(_RUN_INFO_SQL, (self._run_id,)).fetchone()
        if run is None:
            return
        self._run_row = run

        n_frames = self._conn.execute(
            "SELECT COUNT(*) FROM tracking_results WHERE run_id = ? AND is_smoothed = 0",
            (self._run_id,),
        ).fetchone()[0]

        skel = run["skel_name"] or "?"

        # Tracker config
        cfg_id = run["tracker_config_id"]
        cfg = self._conn.execute(_CFG_SQL, (cfg_id,)).fetchone() if cfg_id else None

        # ------------------------------------------------------------------
        # Info sidebar (right pane of the splitter)
        # ------------------------------------------------------------------
        info_title = QLabel(f"<b>Tracking run</b>  [{skel}]")
        info_title.setWordWrap(True)

        hide_btn = QToolButton()
        hide_btn.setText("✕")
        hide_btn.setToolTip("Hide info panel")
        hide_btn.setFixedSize(20, 20)

        title_row = QHBoxLayout()
        title_row.addWidget(info_title, stretch=1)
        title_row.addWidget(hide_btn)

        form = QFormLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(2)
        form.addRow("Ran at:", QLabel(_fmt_ts(run["ran_at"])))
        form.addRow("Frames:", QLabel(f"{n_frames}  |  ver. {run['posetrak_version'] or '—'}"))
        form.addRow("Person:", QLabel(run["person_names"] or "—"))
        try:
            cam_ids = json.loads(run["active_camera_ids"] or "[]")
            cam_lbl = QLabel(", ".join(cam_ids) or "—")
            cam_lbl.setWordWrap(True)
            form.addRow("Cameras:", cam_lbl)
        except Exception:
            pass
        cfg_lbl = QLabel(_cfg_text(cfg, cfg_id))
        cfg_lbl.setWordWrap(True)
        form.addRow("Config:", cfg_lbl)
        if run["notes"]:
            notes_lbl = QLabel(run["notes"])
            notes_lbl.setWordWrap(True)
            form.addRow("Notes:", notes_lbl)

        ids_box, id_widgets = _build_run_ids_group()
        _populate_run_ids(id_widgets, run)

        self._export_bvh_btn = _action_btn("Export BVH…", enabled=bool(self._session_path))
        self._export_bvh_btn.clicked.connect(self._export_bvh)
        self._export_usd_btn = _action_btn(
            "Export USD…", enabled=bool(self._session_path) and _USD_AVAILABLE
        )
        if not _USD_AVAILABLE:
            self._export_usd_btn.setToolTip(_USD_TOOLTIP)
        self._export_usd_btn.clicked.connect(self._export_usd)
        self._export_gltf_btn = _action_btn(
            "Export glTF…", enabled=bool(self._session_path)
        )
        self._export_gltf_btn.clicked.connect(self._export_gltf)
        scale_btn = _action_btn("Scale skeleton…", enabled=bool(self._session_path))
        scale_btn.setToolTip("Measure bone lengths from inlier observations and scale the skeleton")
        scale_btn.clicked.connect(self._open_scaling)

        info_content = QWidget()
        info_v = QVBoxLayout(info_content)
        info_v.setContentsMargins(6, 4, 6, 4)
        info_v.setSpacing(6)
        info_v.addLayout(title_row)
        info_v.addLayout(form)
        info_v.addWidget(ids_box)
        info_v.addWidget(self._export_bvh_btn)
        info_v.addWidget(self._export_usd_btn)
        info_v.addWidget(self._export_gltf_btn)
        info_v.addWidget(scale_btn)

        # --- Current frame ---
        frame_box = _CollapsibleBox("Current frame")
        frame_form = QFormLayout()
        frame_form.setHorizontalSpacing(6)
        frame_form.setVerticalSpacing(2)
        self._trp_fi_step     = QLabel("—")
        self._trp_fi_time     = QLabel("—")
        self._trp_fi_inliers  = QLabel("—")
        self._trp_fi_nis      = QLabel("—")
        self._trp_fi_cov      = QLabel("—")
        frame_form.addRow("Step:",       self._trp_fi_step)
        frame_form.addRow("Time:",       self._trp_fi_time)
        frame_form.addRow("Inliers:",    self._trp_fi_inliers)
        frame_form.addRow("NIS / DOF:",  self._trp_fi_nis)
        frame_form.addRow("Cov cond #:", self._trp_fi_cov)
        frame_box.inner_layout().addLayout(frame_form)
        copy_frame_btn = _action_btn("⎘ Copy frame ID")
        copy_frame_btn.setToolTip(
            "Copy db path, run/trial/capture IDs, and the current step/timestamp "
            "to the clipboard, in one paste-able block."
        )
        copy_frame_btn.clicked.connect(self._copy_frame_id)
        frame_box.inner_layout().addWidget(copy_frame_btn)
        info_v.addWidget(frame_box)

        # --- Metrics charts ---
        self._nis_chart = _LineChart("NIS / DOF", reference_y=1.0)
        self._cov_chart = _LineChart(
            "Covariance condition # (log scale)", reference_y=_COV_COND_WARN, log_y=True,
        )
        charts_box = _CollapsibleBox("Metrics")
        charts_box.inner_layout().addWidget(self._nis_chart)
        charts_box.inner_layout().addWidget(self._cov_chart)
        info_v.addWidget(charts_box)
        info_v.addStretch()

        # Load per-frame stats (used by Current frame labels and chart cursors)
        rows = self._conn.execute(
            "SELECT tracker_step, timestamp_s, n_inlier_observations, "
            "       nis_value, nis_dof, cov_condition_number "
            "FROM tracking_results "
            "WHERE run_id=? AND person_id=0 AND is_smoothed=0 "
            "ORDER BY tracker_step",
            (self._run_id,),
        ).fetchall()
        nis_steps, nis_vals, cov_steps, cov_vals = [], [], [], []
        for row in rows:
            step = row["tracker_step"]
            self._trp_tracking_timestamps.append((row["timestamp_s"], step))
            self._trp_step_stats[step] = row
            nis, dof = row["nis_value"], row["nis_dof"]
            nis_steps.append(step)
            nis_vals.append((nis / dof) if (nis is not None and dof and dof > 0) else None)
            cov_steps.append(step)
            cov_vals.append(row["cov_condition_number"])
        self._nis_chart.set_data(nis_steps, nis_vals)
        self._cov_chart.set_data(cov_steps, cov_vals)

        info_scroll = QScrollArea()
        info_scroll.setWidget(info_content)
        info_scroll.setWidgetResizable(True)
        info_scroll.setMinimumWidth(180)

        # "Show info" button — visible only while the panel is hidden.
        show_btn = QToolButton()
        show_btn.setText("ℹ")
        show_btn.setToolTip("Show run info panel")
        show_btn.setVisible(False)

        # Thin bar at top of the camera area that hosts the show button.
        top_bar = QWidget()
        top_bar.setMaximumHeight(24)
        top_bar_h = QHBoxLayout(top_bar)
        top_bar_h.setContentsMargins(0, 2, 4, 0)
        top_bar_h.addStretch()
        top_bar_h.addWidget(show_btn)
        top_bar.setVisible(False)  # hidden while panel is expanded

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._info_scroll = info_scroll

        def _toggle_info() -> None:
            visible = not info_scroll.isVisible()
            info_scroll.setVisible(visible)
            top_bar.setVisible(not visible)
            if visible:
                w = self._splitter.width()
                self._splitter.setSizes([w - 260, 260])

        hide_btn.clicked.connect(_toggle_info)
        show_btn.clicked.connect(_toggle_info)

        # ------------------------------------------------------------------
        # Left pane: camera grid (wrapped so top_bar sits above it)
        # ------------------------------------------------------------------
        seq_id = run["observation_sequence_id"]
        left_w = QWidget()
        left_v = QVBoxLayout(left_w)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(0)
        left_v.addWidget(top_bar)
        if seq_id:
            crop_grid = PersonCropGridWidget(self._conn, seq_id)
            crop_grid.set_tracking_run(self._run_id)
            crop_grid.time_changed.connect(self._on_time_changed)
            left_v.addWidget(crop_grid, stretch=1)
        else:
            left_v.addStretch(1)

        self._splitter.addWidget(left_w)
        self._splitter.addWidget(info_scroll)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(1, True)
        self._splitter.setSizes([800, 260])

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._splitter)

    def _on_time_changed(self, t: float) -> None:
        if not self._trp_tracking_timestamps:
            return
        step = _nearest_tracker_step(t, self._trp_tracking_timestamps)
        if self._nis_chart:
            self._nis_chart.set_cursor(step)
        if self._cov_chart:
            self._cov_chart.set_cursor(step)
        row = self._trp_step_stats.get(step)
        if row is None:
            return
        self._trp_cur_step = step
        self._trp_cur_ts = row["timestamp_s"]
        if self._trp_fi_step:
            self._trp_fi_step.setText(str(step))
        if self._trp_fi_time:
            self._trp_fi_time.setText(f"{row['timestamp_s']:.3f} s")
        if self._trp_fi_inliers:
            n = row["n_inlier_observations"]
            self._trp_fi_inliers.setText(str(n) if n is not None else "—")
        if self._trp_fi_nis:
            nis, dof = row["nis_value"], row["nis_dof"]
            if nis is not None and dof:
                self._trp_fi_nis.setText(f"{nis:.2f} / {dof}  ({nis/dof:.2f} norm)")
            else:
                self._trp_fi_nis.setText("—")
        if self._trp_fi_cov:
            cov = row["cov_condition_number"]
            self._trp_fi_cov.setText(f"{cov:.1f}" if cov is not None else "—")

    def _copy_frame_id(self) -> None:
        if self._run_row is None:
            return
        text = _frame_identifier_text(
            _db_path_of(self._conn), self._run_row, self._trp_cur_step, self._trp_cur_ts,
        )
        QApplication.clipboard().setText(text)

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

    def _export_gltf(self) -> None:
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save glTF file", "",
            "glTF binary (*.glb);;glTF JSON (*.gltf)"
        )
        if not out_path:
            return
        if not (out_path.endswith(".glb") or out_path.endswith(".gltf")):
            out_path += ".glb"
        if self._export_gltf_btn:
            self._export_gltf_btn.setEnabled(False)
        from posetrak.export.gltf import export_gltf
        run_id = self._run_id
        session_db = str(self._session_path) if self._session_path else None

        def _run() -> None:
            error = ""
            try:
                export_gltf(out_path, session_db=session_db, run_id=run_id)
            except Exception as exc:
                error = str(exc)
            self._export_done.emit("glTF", out_path, error)

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
        elif fmt == "glTF" and self._export_gltf_btn:
            self._export_gltf_btn.setEnabled(True)
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

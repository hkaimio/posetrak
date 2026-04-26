"""page_shots.py — Wizard page 2: define shots and add video files.

The user creates one or more shots and assigns video files to each.  After
the user picks a file, a background probe (cv2 + optional exiftool) fills in
camera metadata automatically.

Wizard fields read by this page
---------------------------------
``db_context`` / ``session_conn`` — set by page 1, accessed via
``wizard().db_context``.

Behaviour
---------
- A shot is created for every entry in the shots list when the user clicks
  *Next*.
- Each video file within a shot is written to ``shot_videos`` via
  ``DBContext.create_shot_video()``.
- The whole operation is wrapped in a single ``begin_page()`` savepoint so
  that clicking *Back* rolls everything back.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from app.setup.video_probe import VideoProbeResult, probe_video


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class VideoEntry:
    """One video file within a shot."""
    path: str
    probe: VideoProbeResult | None = None
    error: str | None = None
    camera_instance_id: str | None = None
    camera_mode_id: str | None = None
    intrinsics_calibration_id: str | None = None


@dataclass
class ShotEntry:
    """One shot (take) with zero or more associated video files."""
    shot_number: int
    label: str = ""
    videos: list[VideoEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Background probe worker
# ---------------------------------------------------------------------------


class _ProbeWorker(QThread):
    """Probe a single video file in a background thread."""

    done = Signal(object, object)   # (VideoEntry, VideoProbeResult | None)
    failed = Signal(object, str)    # (VideoEntry, error_message)

    def __init__(self, entry: VideoEntry, parent=None) -> None:
        super().__init__(parent)
        self._entry = entry

    def run(self) -> None:
        try:
            result = probe_video(Path(self._entry.path))
            self.done.emit(self._entry, result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self._entry, str(exc))


# ---------------------------------------------------------------------------
# Video row widget
# ---------------------------------------------------------------------------


class _VideoRow(QWidget):
    """Compact row widget showing one video file + camera/mode pickers."""

    remove_requested = Signal(object)  # emits VideoEntry

    def __init__(self, entry: VideoEntry, db_context, parent=None) -> None:
        super().__init__(parent)
        self._entry = entry
        self._db_context = db_context

        self._path_label = QLabel(Path(entry.path).name)
        self._path_label.setToolTip(entry.path)
        self._path_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        # Camera instance picker
        self._cam_combo = QComboBox()
        self._cam_combo.setFixedWidth(160)
        self._cam_combo.setStyleSheet("font-size: 11px;")
        self._cam_combo.addItem("— select camera —", None)

        # Camera mode picker (enabled after camera is chosen)
        self._mode_combo = QComboBox()
        self._mode_combo.setFixedWidth(140)
        self._mode_combo.setStyleSheet("font-size: 11px;")
        self._mode_combo.setEnabled(False)
        self._mode_combo.addItem("— mode —", None)

        # Calibration status indicator
        self._calib_label = QLabel()
        self._calib_label.setStyleSheet("font-size: 11px;")

        # Probe / error status
        self._meta_label = QLabel("Probing…")
        self._meta_label.setStyleSheet("color: grey; font-size: 11px;")

        self._remove_btn = QPushButton("✕")
        self._remove_btn.setFixedWidth(28)
        self._remove_btn.clicked.connect(lambda: self.remove_requested.emit(self._entry))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(self._path_label)
        layout.addWidget(self._cam_combo)
        layout.addWidget(self._mode_combo)
        layout.addWidget(self._calib_label)
        layout.addWidget(self._meta_label)
        layout.addWidget(self._remove_btn)

        self._cam_combo.currentIndexChanged.connect(self._on_camera_changed)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        # Populate camera list
        self._refresh_cameras()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_probe(self, result: VideoProbeResult) -> None:
        parts: list[str] = []
        if result.mode_hint:
            parts.append(result.mode_hint)
        elif result.width and result.height:
            fps = result.capture_fps or result.container_fps
            fps_str = f"{fps:.0f}fps" if fps == int(fps) else f"{fps:.2f}fps"
            parts.append(f"{result.width}×{result.height} {fps_str}")
        if result.serial_number:
            parts.append(f"S/N {result.serial_number}")
            # Try to auto-select a camera whose serial number matches
            self._try_select_by_serial(result.serial_number)
        self._meta_label.setText(" · ".join(parts) if parts else "OK")
        self._meta_label.setStyleSheet("color: #444; font-size: 11px;")
        # Re-annotate mode combo with probe match hints
        self._annotate_modes(result)

    def set_error(self, msg: str) -> None:
        self._meta_label.setText(f"Probe failed: {msg}")
        self._meta_label.setStyleSheet("color: red; font-size: 11px;")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    _CREATE_SENTINEL = "__create_new__"
    _ADD_MODE_SENTINEL = "__add_mode__"

    def _refresh_cameras(self) -> None:
        if self._db_context is None:
            return
        instances = self._db_context.list_camera_instances()
        self._cam_combo.blockSignals(True)
        self._cam_combo.clear()
        self._cam_combo.addItem("— select camera —", None)
        for row in instances:
            label = row["label"]
            if row["from_registry"]:
                label += " [registry]"
            self._cam_combo.addItem(label, row["id"])
        self._cam_combo.addItem("Create new camera…", self._CREATE_SENTINEL)
        self._cam_combo.blockSignals(False)

    def _on_camera_changed(self) -> None:
        instance_id = self._cam_combo.currentData()

        if instance_id == self._CREATE_SENTINEL:
            # Reset to placeholder before opening dialog so we don't stay on sentinel
            self._cam_combo.blockSignals(True)
            self._cam_combo.setCurrentIndex(0)
            self._cam_combo.blockSignals(False)
            self._open_inline_create_camera()
            return

        self._entry.camera_instance_id = instance_id
        self._entry.camera_mode_id = None
        self._entry.intrinsics_calibration_id = None

        self._mode_combo.blockSignals(True)
        self._mode_combo.clear()
        self._mode_combo.addItem("— mode —", None)
        self._mode_combo.setEnabled(False)
        self._mode_combo.blockSignals(False)
        self._calib_label.setText("")

        if instance_id is None or self._db_context is None:
            return

        model_id = self._db_context.get_camera_model_id(instance_id)
        if model_id is None:
            return

        modes = self._db_context.list_camera_modes(model_id)
        probe = self._entry.probe

        self._mode_combo.blockSignals(True)
        # Count duplicate (w, h, fps) combos so unnamed duplicates can be numbered.
        param_counts: Counter = Counter(
            (m["width_px"], m["height_px"], m["nominal_fps"]) for m in modes
        )
        param_seen: Counter = Counter()
        for mode in modes:
            label = self._mode_label(mode, param_counts, param_seen)
            if probe and self._mode_matches(probe, mode):
                label = "✓ " + label
            self._mode_combo.addItem(label, mode["id"])
        # Always offer to create a new mode even if the model has none yet
        self._mode_combo.addItem("Create new mode…", self._ADD_MODE_SENTINEL)
        self._mode_combo.setEnabled(True)
        self._mode_combo.blockSignals(False)

        # Auto-select if exactly one mode or probe matches one
        if len(modes) == 1:
            self._mode_combo.setCurrentIndex(1)
        elif probe:
            for i, mode in enumerate(modes):
                if self._mode_matches(probe, mode):
                    self._mode_combo.setCurrentIndex(i + 1)
                    break

    def _on_mode_changed(self) -> None:
        mode_id = self._mode_combo.currentData()

        if mode_id == self._ADD_MODE_SENTINEL:
            self._mode_combo.blockSignals(True)
            self._mode_combo.setCurrentIndex(0)
            self._mode_combo.blockSignals(False)
            self._open_add_mode()
            return

        self._entry.camera_mode_id = mode_id
        self._entry.intrinsics_calibration_id = None
        self._calib_label.setText("")

        if mode_id is None or self._db_context is None:
            return

        # Look up default_intrinsics_calibration_id from the selected mode
        conn = self._db_context._conn
        row = conn.execute(
            "SELECT default_intrinsics_calibration_id FROM camera_modes WHERE id = ?",
            (mode_id,),
        ).fetchone()
        if row is None and self._db_context._registry_conn:
            row = self._db_context._registry_conn.execute(
                "SELECT default_intrinsics_calibration_id FROM camera_modes WHERE id = ?",
                (mode_id,),
            ).fetchone()

        if row and row["default_intrinsics_calibration_id"]:
            self._entry.intrinsics_calibration_id = row["default_intrinsics_calibration_id"]
            self._calib_label.setText("calib ✓")
            self._calib_label.setStyleSheet("color: green; font-size: 11px;")
        else:
            self._calib_label.setText("no calib")
            self._calib_label.setStyleSheet("color: orange; font-size: 11px;")

    @staticmethod
    def _mode_label(mode, param_counts: Counter, param_seen: Counter) -> str:
        """Build a human-readable label for a camera mode row.

        Prefers mode notes when present.  Falls back to resolution+fps, with a
        ``#N`` disambiguator when multiple modes share identical parameters.
        """
        w, h = mode["width_px"], mode["height_px"]
        fps = mode["nominal_fps"]
        fps_str = f"{fps:.0f}fps" if fps == int(fps) else f"{fps:.2f}fps"
        res_fps = f"{w}×{h} {fps_str}"
        notes = mode["notes"] or ""
        if notes:
            return f"{notes} ({res_fps})"
        key = (w, h, fps)
        if param_counts[key] > 1:
            param_seen[key] += 1
            return f"{res_fps} #{param_seen[key]}"
        return res_fps

    def _mode_matches(self, probe: VideoProbeResult, mode) -> bool:
        """Return True if *probe* resolution/fps is consistent with *mode*."""
        if probe.width and probe.height:
            if mode["width_px"] != probe.width or mode["height_px"] != probe.height:
                return False
        fps = probe.capture_fps or probe.container_fps
        if fps and mode["nominal_fps"]:
            if abs(mode["nominal_fps"] - fps) > 1.0:
                return False
        return True

    def _try_select_by_serial(self, serial: str) -> None:
        """Select camera instance whose serial_number matches *serial*."""
        if self._db_context is None:
            return
        for i in range(1, self._cam_combo.count()):
            inst_id = self._cam_combo.itemData(i)
            if inst_id is None:
                continue
            row = self._db_context._conn.execute(
                "SELECT serial_number FROM camera_instances WHERE id = ?", (inst_id,)
            ).fetchone()
            if row and row["serial_number"] == serial:
                self._cam_combo.setCurrentIndex(i)
                return
            if self._db_context._registry_conn:
                row = self._db_context._registry_conn.execute(
                    "SELECT serial_number FROM camera_instances WHERE id = ?", (inst_id,)
                ).fetchone()
                if row and row["serial_number"] == serial:
                    self._cam_combo.setCurrentIndex(i)
                    return

    def _annotate_modes(self, probe: VideoProbeResult) -> None:
        """Re-annotate mode combo items with ✓ where probe matches."""
        for i in range(1, self._mode_combo.count()):
            mode_id = self._mode_combo.itemData(i)
            if mode_id is None:
                continue
            conn = self._db_context._conn if self._db_context else None
            if conn is None:
                continue
            row = conn.execute(
                "SELECT width_px, height_px, nominal_fps FROM camera_modes WHERE id = ?",
                (mode_id,),
            ).fetchone()
            if row is None and self._db_context._registry_conn:
                row = self._db_context._registry_conn.execute(
                    "SELECT width_px, height_px, nominal_fps FROM camera_modes WHERE id = ?",
                    (mode_id,),
                ).fetchone()
            if row is None:
                continue
            existing = self._mode_combo.itemText(i)
            base = existing[2:] if existing.startswith("✓ ") else existing
            prefix = "✓ " if self._mode_matches(probe, row) else ""
            self._mode_combo.setItemText(i, prefix + base)

    def _open_inline_create_camera(self) -> None:
        """Open InlineCreateCameraDialog; on accept select the new instance."""
        if self._db_context is None:
            return
        from app.setup.camera_registry import InlineCreateCameraDialog

        dlg = InlineCreateCameraDialog(
            self._db_context._conn,
            self._db_context._registry_conn,
            self,
        )
        if dlg.exec() != InlineCreateCameraDialog.DialogCode.Accepted:
            return

        new_id = dlg.saved_instance_id()
        if not new_id:
            return

        # Refresh combo and select the new instance
        self._refresh_cameras()
        for i in range(self._cam_combo.count()):
            if self._cam_combo.itemData(i) == new_id:
                self._cam_combo.setCurrentIndex(i)
                break

    def _open_add_mode(self) -> None:
        """Open ModeDialog to create a new mode for the currently selected camera's model."""
        if self._db_context is None:
            return
        instance_id = self._cam_combo.currentData()
        if instance_id is None or instance_id == self._CREATE_SENTINEL:
            return

        from PySide6.QtWidgets import QDialog as _QDialog
        from app.setup.camera_registry import ModeDialog

        model_id = self._db_context.get_camera_model_id(instance_id)
        if model_id is None:
            return
        row = self._db_context._conn.execute(
            "SELECT model_name FROM camera_models WHERE id = ?", (model_id,)
        ).fetchone()
        model_name = row["model_name"] if row else "Camera"

        dlg = ModeDialog(self._db_context._conn, model_id, model_name, parent=self)
        if dlg.exec() == _QDialog.DialogCode.Accepted:
            # Retrigger camera selection so mode combo is rebuilt with the new mode
            self._on_camera_changed()


# ---------------------------------------------------------------------------
# Shot panel widget
# ---------------------------------------------------------------------------


class _ShotPanel(QGroupBox):
    """Collapsible panel representing one shot."""

    remove_requested = Signal(object)   # emits ShotEntry

    def __init__(self, entry: ShotEntry, db_context=None, parent=None) -> None:
        super().__init__(parent)
        self._entry = entry
        self._db_context = db_context
        self._video_rows: dict[str, _VideoRow] = {}  # path → row widget
        self._workers: list[_ProbeWorker] = []

        self._num_spin = QSpinBox()
        self._num_spin.setRange(1, 9999)
        self._num_spin.setValue(entry.shot_number)
        self._num_spin.valueChanged.connect(
            lambda v: setattr(self._entry, "shot_number", v)
        )

        self._label_edit = QLineEdit(entry.label)
        self._label_edit.setPlaceholderText("Optional label (e.g. 'walk 1')")
        self._label_edit.textChanged.connect(
            lambda t: setattr(self._entry, "label", t)
        )

        self._add_video_btn = QPushButton("Add Videos…")
        self._add_video_btn.clicked.connect(self._add_videos)

        self._remove_shot_btn = QPushButton("Remove Shot")
        self._remove_shot_btn.clicked.connect(
            lambda: self.remove_requested.emit(self._entry)
        )

        header = QFormLayout()
        header.addRow("Shot #:", self._num_spin)
        header.addRow("Label:", self._label_edit)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._add_video_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._remove_shot_btn)

        self._video_container = QWidget()
        self._video_layout = QVBoxLayout(self._video_container)
        self._video_layout.setContentsMargins(0, 0, 0, 0)
        self._video_layout.setSpacing(2)

        main = QVBoxLayout(self)
        main.addLayout(header)
        main.addLayout(btn_row)
        main.addWidget(self._video_container)

        self._update_title()
        self._num_spin.valueChanged.connect(lambda _: self._update_title())
        self._label_edit.textChanged.connect(lambda _: self._update_title())

    def _update_title(self) -> None:
        label = self._entry.label
        title = f"Shot {self._entry.shot_number}"
        if label:
            title += f" — {label}"
        self.setTitle(title)

    def _add_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            f"Add videos to Shot {self._entry.shot_number}",
            "",
            "Video files (*.mp4 *.mov *.avi *.mkv *.mts *.m2ts);;All files (*)",
        )
        for p in paths:
            if p in self._video_rows:
                continue  # already added
            ve = VideoEntry(path=p)
            self._entry.videos.append(ve)

            row = _VideoRow(ve, self._db_context, self._video_container)
            row.remove_requested.connect(self._remove_video)
            self._video_layout.addWidget(row)
            self._video_rows[p] = row

            # Start background probe
            worker = _ProbeWorker(ve, self)
            worker.done.connect(self._on_probe_done)
            worker.failed.connect(self._on_probe_failed)
            worker.finished.connect(lambda w=worker: self._workers.remove(w))
            self._workers.append(worker)
            worker.start()

    def _remove_video(self, entry: VideoEntry) -> None:
        row = self._video_rows.pop(entry.path, None)
        if row is not None:
            self._video_layout.removeWidget(row)
            row.deleteLater()
        if entry in self._entry.videos:
            self._entry.videos.remove(entry)

    def refresh_camera_combos(self) -> None:
        """Refresh camera dropdowns in all video rows (call after Manage Cameras closes)."""
        for row in self._video_rows.values():
            row._refresh_cameras()

    def _on_probe_done(self, entry: VideoEntry, result: VideoProbeResult) -> None:
        entry.probe = result
        row = self._video_rows.get(entry.path)
        if row:
            row.set_probe(result)

    def _on_probe_failed(self, entry: VideoEntry, msg: str) -> None:
        entry.error = msg
        row = self._video_rows.get(entry.path)
        if row:
            row.set_error(msg)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


class ShotsPage(QWizardPage):
    """Wizard page 2 — define shots and add video files."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("Shots & Videos")
        self.setSubTitle(
            "Create one entry per shot (take). Add the video files captured "
            "by each camera for that shot."
        )

        self._shots: list[ShotEntry] = []
        self._panels: dict[int, _ShotPanel] = {}   # shot_number → panel

        self._add_shot_btn = QPushButton("+ Add Shot")
        self._add_shot_btn.clicked.connect(self._add_shot)

        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red;")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)

        # Scrollable area for shot panels
        self._scroll_widget = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_widget)
        self._scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidget(self._scroll_widget)
        scroll.setWidgetResizable(True)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self._add_shot_btn)
        top_bar.addStretch()

        layout = QVBoxLayout(self)
        layout.addLayout(top_bar)
        layout.addWidget(scroll)
        layout.addWidget(self._error_label)

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def initializePage(self) -> None:  # noqa: N802
        """Called each time the page is shown; begin savepoint and reset UI."""
        self.wizard().db_context.begin_page()
        # Add default shot now that db_context is available.
        if not self._shots:
            self._add_shot()

    def cleanupPage(self) -> None:  # noqa: N802
        """Called when user clicks Back; roll back any writes and reset UI."""
        self.wizard().db_context.rollback_page()
        for entry in list(self._shots):
            self._remove_shot(entry)

    def validatePage(self) -> bool:  # noqa: N802
        """Write shots + shot_videos to DB; return False on error."""
        self._error_label.setVisible(False)

        if not self._shots:
            self._show_error("Add at least one shot before proceeding.")
            return False

        ctx = self.wizard().db_context

        try:
            for entry in self._shots:
                shot_id = ctx.create_shot(
                    label=entry.label or f"Shot {entry.shot_number}",
                    shot_number=entry.shot_number,
                )
                for ve in entry.videos:
                    probe = ve.probe
                    fps = probe.container_fps if probe else 0.0
                    frames = probe.frame_count if probe else 0
                    w = probe.width if probe else 0
                    h = probe.height if probe else 0

                    # Copy camera records from registry into session DB and
                    # record the camera's participation in this session.
                    if ve.camera_instance_id:
                        ctx.upsert_camera_records(
                            ve.camera_instance_id,
                            ve.camera_mode_id,
                            ve.intrinsics_calibration_id,
                        )
                        ctx.upsert_session_camera(ve.camera_instance_id)

                    ctx.create_shot_video(
                        shot_id=shot_id,
                        cam_instance_id=ve.camera_instance_id or Path(ve.path).stem,
                        path=ve.path,
                        fps=fps,
                        frame_count=max(frames, 1),
                        width=w,
                        height=h,
                        camera_mode_id=ve.camera_mode_id,
                        intrinsics_calibration_id=ve.intrinsics_calibration_id,
                    )
            ctx.commit_page()
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Database error: {exc}")
            return False

        return True

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _add_shot(self) -> None:
        shot_number = (max(e.shot_number for e in self._shots) + 1
                       if self._shots else 1)
        entry = ShotEntry(shot_number=shot_number)
        self._shots.append(entry)

        ctx = getattr(self.wizard(), "db_context", None) if self.wizard() else None
        panel = _ShotPanel(entry, db_context=ctx, parent=self._scroll_widget)
        panel.remove_requested.connect(self._remove_shot)
        self._scroll_layout.addWidget(panel)
        self._panels[id(entry)] = panel

    def _remove_shot(self, entry: ShotEntry) -> None:
        panel = self._panels.pop(id(entry), None)
        if panel is not None:
            self._scroll_layout.removeWidget(panel)
            panel.deleteLater()
        if entry in self._shots:
            self._shots.remove(entry)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def refresh_camera_combos(self) -> None:
        """Refresh camera dropdowns in all video rows after camera registry changes."""
        for panel in self._panels.values():
            panel.refresh_camera_combos()

    def _show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setVisible(True)

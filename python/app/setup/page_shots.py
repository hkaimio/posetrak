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
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
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
    """Compact row widget showing one video file + its probe result."""

    remove_requested = Signal(object)  # emits VideoEntry

    def __init__(self, entry: VideoEntry, parent=None) -> None:
        super().__init__(parent)
        self._entry = entry

        self._path_label = QLabel(Path(entry.path).name)
        self._path_label.setToolTip(entry.path)
        self._path_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self._meta_label = QLabel("Probing…")
        self._meta_label.setStyleSheet("color: grey; font-size: 11px;")

        self._remove_btn = QPushButton("✕")
        self._remove_btn.setFixedWidth(28)
        self._remove_btn.clicked.connect(lambda: self.remove_requested.emit(self._entry))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(self._path_label)
        layout.addWidget(self._meta_label)
        layout.addWidget(self._remove_btn)

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
        self._meta_label.setText(" · ".join(parts) if parts else "OK")
        self._meta_label.setStyleSheet("color: #444; font-size: 11px;")

    def set_error(self, msg: str) -> None:
        self._meta_label.setText(f"Probe failed: {msg}")
        self._meta_label.setStyleSheet("color: red; font-size: 11px;")


# ---------------------------------------------------------------------------
# Shot panel widget
# ---------------------------------------------------------------------------


class _ShotPanel(QGroupBox):
    """Collapsible panel representing one shot."""

    remove_requested = Signal(object)   # emits ShotEntry

    def __init__(self, entry: ShotEntry, parent=None) -> None:
        super().__init__(parent)
        self._entry = entry
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

            row = _VideoRow(ve, self._video_container)
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

        # Add one shot by default
        self._add_shot()

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def initializePage(self) -> None:  # noqa: N802
        """Called each time the page is shown; begin savepoint."""
        self.wizard().db_context.begin_page()

    def cleanupPage(self) -> None:  # noqa: N802
        """Called when user clicks Back; roll back any writes."""
        self.wizard().db_context.rollback_page()

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
                    # Use a placeholder camera_instance_id; camera assignment
                    # is handled in a later wizard page.
                    ctx.create_shot_video(
                        shot_id=shot_id,
                        cam_instance_id="__unassigned__",
                        path=ve.path,
                        fps=fps,
                        frame_count=max(frames, 1),
                        width=w,
                        height=h,
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

        panel = _ShotPanel(entry, self._scroll_widget)
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

    def _show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setVisible(True)

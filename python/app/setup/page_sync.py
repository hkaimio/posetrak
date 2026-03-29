"""page_sync.py — Wizard page 3: camera synchronisation.

The page hosts a ``MultiVideoScrubber`` so the user can inspect all cameras
together and mark synchronisation anchors.

D3a foundation — independent scrubbing
---------------------------------------
The scrubber starts in independent mode (no sync table loaded). The user
selects which shot to work on, then scrolls each camera individually to find
a common reference event before setting anchors in D3b.

Anatomy
-------
- Shot selector (``QComboBox``) — one entry per shot in the session.
- ``MultiVideoScrubber`` — fills the centre of the page.
- Per-camera frame counter strip — shows ``cam / frame N of M`` for each
  camera below the scrubber.
- Seek slider — controls the focused camera's frame position.
- Placeholder panels for rough-sync (D3b) and LED sync (D3c) — hidden for now.

Data flow
---------
``initializePage()`` reads ``shots`` + ``shot_videos`` from the DB, builds
``CellInfo`` objects, opens a ``FrameCache``, and passes them to the scrubber.
``cleanupPage()`` releases the ``FrameCache`` and clears the scrubber.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from app.setup.db_context import DBContext
from app.setup.frame_cache import FrameCache
from app.setup.multi_video_scrubber import CellInfo, MultiVideoScrubber


@dataclass
class _ShotMeta:
    shot_id: str
    label: str
    videos: list  # list of ShotVideoInfo


class SyncPage(QWizardPage):
    """Wizard page 3 — camera synchronisation (D3a: independent mode scaffold)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("Camera Synchronisation")
        self.setSubTitle(
            "Scroll each camera to a common reference event before setting sync anchors."
        )

        self._shots: list[_ShotMeta] = []
        self._cache: FrameCache | None = None
        self._scrubber: MultiVideoScrubber | None = None

        # --- shot selector ---
        self._shot_combo = QComboBox()
        self._shot_combo.currentIndexChanged.connect(self._on_shot_selected)

        shot_bar = QHBoxLayout()
        shot_bar.addWidget(QLabel("Shot:"))
        shot_bar.addWidget(self._shot_combo)
        shot_bar.addStretch()

        # --- scrubber area (replaced on shot selection) ---
        self._scrubber_container = QWidget()
        self._scrubber_layout = QVBoxLayout(self._scrubber_container)
        self._scrubber_layout.setContentsMargins(0, 0, 0, 0)

        # --- seek slider (below scrubber) ---
        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setMinimum(0)
        self._seek_slider.setMaximum(0)
        self._seek_slider.setValue(0)
        self._seek_slider.setEnabled(False)
        # NoFocus so keyboard events always reach the scrubber, not the slider
        self._seek_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._seek_slider.sliderMoved.connect(self._on_slider_gesture)
        self._slider_updating = False  # guard: prevent seek when we set value programmatically

        # Debounce timer: decode only after slider has been idle for 80 ms
        self._slider_timer = QTimer(self)
        self._slider_timer.setSingleShot(True)
        self._slider_timer.setInterval(80)
        self._slider_timer.timeout.connect(self._on_slider_debounced)
        self._slider_pending_value: int = 0

        # --- camera status strip ---
        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._status_label.setStyleSheet("font-family: monospace; font-size: 11px;")

        # --- placeholder panels for D3b / D3c (hidden) ---
        self._rough_panel = QGroupBox("Rough sync")
        self._rough_panel.setVisible(False)
        self._led_panel = QGroupBox("LED sync")
        self._led_panel.setVisible(False)

        # --- error label ---
        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red;")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)

        # --- main layout ---
        layout = QVBoxLayout(self)
        layout.addLayout(shot_bar)
        layout.addWidget(self._scrubber_container, stretch=1)
        layout.addWidget(self._seek_slider)
        layout.addWidget(self._status_label)
        layout.addWidget(self._rough_panel)
        layout.addWidget(self._led_panel)
        layout.addWidget(self._error_label)

    # ------------------------------------------------------------------
    # Qt wizard overrides
    # ------------------------------------------------------------------

    def initializePage(self) -> None:  # noqa: N802
        """Load shots from DB and populate the shot combo."""
        self._error_label.setVisible(False)
        self._shots.clear()
        self._shot_combo.blockSignals(True)
        self._shot_combo.clear()

        ctx: DBContext = self.wizard().db_context
        try:
            rows = ctx._conn.execute(
                "SELECT id, shot_number, label FROM shots "
                "WHERE session_id = ? ORDER BY shot_number",
                (ctx._session_id,),
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            self._shot_combo.blockSignals(False)
            self._show_error(f"Could not read shots: {exc}")
            return

        for row in rows:
            label = row["label"] or f"Shot {row['shot_number']}"
            videos = ctx.get_shot_videos(row["id"])
            meta = _ShotMeta(shot_id=row["id"], label=label, videos=videos)
            self._shots.append(meta)
            self._shot_combo.addItem(label)

        self._shot_combo.blockSignals(False)

        if self._shots:
            self._on_shot_selected(0)
        else:
            self._show_error(
                "No shots found. Go back to the Shots & Videos page and add at least one."
            )

    def cleanupPage(self) -> None:  # noqa: N802
        """Release the FrameCache and clear the scrubber on Back."""
        self._teardown_scrubber()

    def isComplete(self) -> bool:  # noqa: N802
        # The sync step is optional — user can advance without setting any sync.
        return True

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_shot_selected(self, index: int) -> None:
        """Rebuild the scrubber for the newly-selected shot."""
        self._teardown_scrubber()
        if index < 0 or index >= len(self._shots):
            return

        shot = self._shots[index]
        if not shot.videos:
            self._show_error(
                f"Shot '{shot.label}' has no videos. "
                "Go back and add video files for this shot."
            )
            return

        # Build CellInfo list
        cells_info = [
            CellInfo(
                shot_video_id=sv.id,
                file_path=sv.file_path,
                total_frames=max(sv.last_video_frame - sv.first_video_frame + 1, 1),
                fps=sv.actual_fps or 30.0,
                label=sv.camera_instance_id,
            )
            for sv in shot.videos
        ]

        # Open a fresh FrameCache (no DB persistence in this session)
        self._cache = FrameCache()

        # Build scrubber
        scrubber = MultiVideoScrubber(cells_info, self._cache, self._scrubber_container)
        scrubber.frame_changed.connect(self._on_frame_changed)
        self._scrubber_layout.addWidget(scrubber)
        self._scrubber = scrubber

        # Initialise seek slider to the reference camera's range
        ref_frames = cells_info[0].total_frames if cells_info else 1
        self._seek_slider.setMaximum(max(ref_frames - 1, 0))
        self._seek_slider.setValue(0)
        self._seek_slider.setEnabled(True)

        self._update_status()
        scrubber.setFocus()

    def _on_frame_changed(self, cell_idx: int, frame_idx: int) -> None:
        """Sync slider to focused camera and refresh the status strip."""
        if self._scrubber is None:
            return
        if cell_idx == self._scrubber.focused_cell:
            self._slider_updating = True
            self._seek_slider.setValue(frame_idx)
            self._slider_updating = False
        self._update_status()

    def _on_slider_gesture(self, value: int) -> None:
        """Fires on every pixel of user drag; debounced before seeking."""
        self._slider_pending_value = value
        self._slider_timer.start()   # restart — fires 80 ms after last movement

    def _on_slider_debounced(self) -> None:
        """Fires once after the slider has been idle for 80 ms."""
        if self._slider_updating or self._scrubber is None:
            return
        fc = self._scrubber.focused_cell
        self._scrubber.seek_camera(fc, self._slider_pending_value)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_status(self) -> None:
        if self._scrubber is None or not self._shots:
            self._status_label.setText("")
            return
        idx = self._shot_combo.currentIndex()
        if idx < 0 or idx >= len(self._shots):
            return
        shot = self._shots[idx]
        frames = self._scrubber.current_frames
        parts: list[str] = []
        for i, sv in enumerate(shot.videos):
            total = max(sv.last_video_frame - sv.first_video_frame + 1, 1)
            cur = frames[i] if i < len(frames) else 0
            cam = sv.camera_instance_id
            focused = " ◄" if (self._scrubber and i == self._scrubber.focused_cell) else ""
            parts.append(f"{cam}: {cur}/{total - 1}{focused}")
        self._status_label.setText("   ".join(parts))

    def _teardown_scrubber(self) -> None:
        if self._scrubber is not None:
            self._slider_timer.stop()
            self._scrubber.shutdown()   # stop background decoder before deleting
            self._scrubber_layout.removeWidget(self._scrubber)
            self._scrubber.deleteLater()
            self._scrubber = None
        if self._cache is not None:
            self._cache.close_all()
            self._cache = None
        self._seek_slider.setEnabled(False)
        self._seek_slider.setValue(0)
        self._status_label.setText("")

    def _show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setVisible(True)

"""page_sync.py — Wizard page for camera synchronisation.

The page lets the user inspect all camera feeds together and establish a
common time reference so that downstream tracking and visualisation can
treat frames from different cameras as simultaneous.

Two sync methods are available (in increasing accuracy):

Rough sync (this page, lower panel)
    The user scrolls each camera to the same physical event — a clap, a
    flash, an LED blink — and presses "Set anchor" for each camera.  Once
    every camera has an anchor, "Apply rough sync" computes per-camera
    frame offsets and switches the scrubber into synced mode.  The result
    is written to the session as a sync config with method "manual-rough".

LED sync (placeholder, to be implemented)
    Automated brightness-peak detection and cross-correlation against a
    reference camera, requiring the user to draw an ROI over a blinking
    LED in each camera feed.

Layout
------
- Shot selector — one entry per shot in the session.
- MultiVideoScrubber — grid of camera cells, each with its own slider and
  frame counter.  Cells scroll independently until a sync config is applied.
- Rough sync panel — set/clear per-camera anchors, apply the sync.
- LED sync panel — placeholder, hidden until implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from app.setup.db_context import DBContext, SyncPoint, SyncTable
from app.setup.frame_cache import FrameCache
from app.setup.multi_video_scrubber import CellInfo, MultiVideoScrubber
from app.setup.overlay import SyncAnchorOverlay


@dataclass
class _ShotMeta:
    shot_id: str
    label: str
    videos: list  # list of ShotVideoInfo


class SyncPage(QWizardPage):
    """Wizard page — camera synchronisation."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("Camera Synchronisation")
        self.setSubTitle(
            "Scroll each camera to a common reference event, set an anchor per "
            "camera, then apply rough sync.  Click a cell to focus it; use "
            "←/→ (±1 frame) or Shift+←/→ (±10 frames) to navigate."
        )

        self._shots: list[_ShotMeta] = []
        self._cache: FrameCache | None = None
        self._scrubber: MultiVideoScrubber | None = None

        # Rough-sync state (reset whenever a different shot is selected)
        self._anchors: dict[int, int] = {}          # cell_idx → frame_idx
        self._anchor_overlays: list[SyncAnchorOverlay] = []
        self._anchor_labels: list[QLabel] = []

        # ---- shot selector ----
        self._shot_combo = QComboBox()
        self._shot_combo.currentIndexChanged.connect(self._on_shot_selected)

        shot_bar = QHBoxLayout()
        shot_bar.addWidget(QLabel("Shot:"))
        shot_bar.addWidget(self._shot_combo)
        shot_bar.addStretch()

        # ---- scrubber area ----
        self._scrubber_container = QWidget()
        self._scrubber_layout = QVBoxLayout(self._scrubber_container)
        self._scrubber_layout.setContentsMargins(0, 0, 0, 0)

        # ---- rough sync panel ----
        self._rough_panel = QGroupBox("Rough synchronisation")
        rough_layout = QVBoxLayout(self._rough_panel)
        rough_layout.setSpacing(4)

        btn_row = QHBoxLayout()
        self._set_anchor_btn = QPushButton("Set anchor for focused camera")
        self._set_anchor_btn.setEnabled(False)
        self._set_anchor_btn.clicked.connect(self._on_set_anchor)
        self._clear_anchors_btn = QPushButton("Clear all anchors")
        self._clear_anchors_btn.setEnabled(False)
        self._clear_anchors_btn.clicked.connect(self._on_clear_anchors)
        btn_row.addWidget(self._set_anchor_btn)
        btn_row.addWidget(self._clear_anchors_btn)
        btn_row.addStretch()
        rough_layout.addLayout(btn_row)

        # Per-camera anchor status labels (populated dynamically)
        self._anchor_status_widget = QWidget()
        self._anchor_status_layout = QHBoxLayout(self._anchor_status_widget)
        self._anchor_status_layout.setContentsMargins(0, 0, 0, 0)
        rough_layout.addWidget(self._anchor_status_widget)

        apply_row = QHBoxLayout()
        self._rough_status_label = QLabel()
        self._rough_status_label.setStyleSheet("color: grey; font-size: 11px;")
        self._apply_rough_btn = QPushButton("Apply rough sync")
        self._apply_rough_btn.setEnabled(False)
        self._apply_rough_btn.clicked.connect(self._on_apply_rough_sync)
        apply_row.addWidget(self._rough_status_label, stretch=1)
        apply_row.addWidget(self._apply_rough_btn)
        rough_layout.addLayout(apply_row)

        # ---- LED sync panel (placeholder) ----
        self._led_panel = QGroupBox("LED sync (not yet available)")
        self._led_panel.setEnabled(False)

        # ---- error label ----
        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red;")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)

        # ---- main layout ----
        layout = QVBoxLayout(self)
        layout.addLayout(shot_bar)
        layout.addWidget(self._scrubber_container, stretch=1)
        layout.addWidget(self._rough_panel)
        layout.addWidget(self._led_panel)
        layout.addWidget(self._error_label)

    # ------------------------------------------------------------------
    # Qt wizard overrides
    # ------------------------------------------------------------------

    def initializePage(self) -> None:  # noqa: N802
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
        self._teardown_scrubber()

    def isComplete(self) -> bool:  # noqa: N802
        return True

    # ------------------------------------------------------------------
    # Slots — shot selection
    # ------------------------------------------------------------------

    def _on_shot_selected(self, index: int) -> None:
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

        self._cache = FrameCache()
        scrubber = MultiVideoScrubber(cells_info, self._cache, self._scrubber_container)
        self._scrubber_layout.addWidget(scrubber)
        self._scrubber = scrubber

        # Create one SyncAnchorOverlay per cell and attach it
        self._anchor_overlays = [
            SyncAnchorOverlay(total_frames=info.total_frames)
            for info in cells_info
        ]
        for i, ov in enumerate(self._anchor_overlays):
            scrubber.set_overlays(i, [ov])

        # Build per-camera anchor status labels
        self._rebuild_anchor_labels(shot)

        self._set_anchor_btn.setEnabled(True)
        self._clear_anchors_btn.setEnabled(True)
        self._update_rough_panel_state()
        scrubber.setFocus()

    # ------------------------------------------------------------------
    # Slots — rough sync
    # ------------------------------------------------------------------

    def _on_set_anchor(self) -> None:
        if self._scrubber is None:
            return
        fc = self._scrubber.focused_cell
        frame = self._scrubber.current_frames[fc]
        self._anchors[fc] = frame
        self._anchor_overlays[fc].set_anchor(frame)
        self._scrubber._cells[fc].update()

        shot = self._shots[self._shot_combo.currentIndex()]
        cam = shot.videos[fc].camera_instance_id
        self._anchor_labels[fc].setText(f"{cam}: {frame}")
        self._update_rough_panel_state()

    def _on_clear_anchors(self) -> None:
        self._anchors.clear()
        for ov in self._anchor_overlays:
            ov.anchor_frame = None
        if self._scrubber:
            for cell in self._scrubber._cells:
                cell.update()
        shot_idx = self._shot_combo.currentIndex()
        if 0 <= shot_idx < len(self._shots):
            shot = self._shots[shot_idx]
            for i, lbl in enumerate(self._anchor_labels):
                cam = shot.videos[i].camera_instance_id
                lbl.setText(f"{cam}: —")
        if self._scrubber:
            self._scrubber.reload_sync(None)
        self._update_rough_panel_state()

    def _on_apply_rough_sync(self) -> None:
        if not self._anchors or self._scrubber is None:
            return

        shot_idx = self._shot_combo.currentIndex()
        shot = self._shots[shot_idx]

        # Reference camera: lowest cell index with an anchor
        ref_cell = min(self._anchors)
        ref_frame = self._anchors[ref_cell]
        ref_sv = shot.videos[ref_cell]
        ref_fps = ref_sv.actual_fps or 30.0
        ref_ts = ref_frame / ref_fps

        # Build one SyncPoint per anchored camera, all sharing the same
        # global timestamp (the moment of the sync event).
        points: dict[str, list[SyncPoint]] = {}
        fps_by_video: dict[str, float] = {}
        for cell_idx, anchor_frame in self._anchors.items():
            sv = shot.videos[cell_idx]
            cam_id = sv.camera_instance_id
            fps = sv.actual_fps or 30.0
            points[cam_id] = [
                SyncPoint(
                    camera_instance_id=cam_id,
                    shot_video_id=sv.id,
                    video_frame=anchor_frame,
                    timestamp_s=ref_ts,
                )
            ]
            fps_by_video[sv.id] = fps

        # Persist to the session database
        ctx: DBContext = self.wizard().db_context
        ctx.write_sync_config(shot.shot_id, "manual-rough", points)
        ctx._conn.commit()

        # Switch scrubber to synced mode
        all_points = [sp for pts in points.values() for sp in pts]
        sync_table = SyncTable(all_points, fps_by_video)
        self._scrubber.reload_sync(sync_table)

        n = len(self._anchors)
        self._rough_status_label.setText(
            f"Rough sync applied ({n} camera{'s' if n != 1 else ''})."
        )
        self._rough_status_label.setStyleSheet("color: green; font-size: 11px;")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rebuild_anchor_labels(self, shot: _ShotMeta) -> None:
        """Replace the per-camera anchor status labels for the given shot."""
        # Remove old labels
        while self._anchor_status_layout.count():
            item = self._anchor_status_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._anchor_labels = []
        for sv in shot.videos:
            lbl = QLabel(f"{sv.camera_instance_id}: —")
            lbl.setStyleSheet("font-size: 11px;")
            self._anchor_status_layout.addWidget(lbl)
            self._anchor_labels.append(lbl)
        self._anchor_status_layout.addStretch()

    def _update_rough_panel_state(self) -> None:
        n_anchored = len(self._anchors)
        shot_idx = self._shot_combo.currentIndex()
        n_total = len(self._shots[shot_idx].videos) if 0 <= shot_idx < len(self._shots) else 0
        can_apply = n_anchored >= 2
        self._apply_rough_btn.setEnabled(can_apply)
        if n_anchored == 0:
            msg = "Set an anchor on at least two cameras."
        elif n_anchored < n_total:
            msg = f"{n_anchored} / {n_total} cameras anchored — can apply (unanchored cameras will not be synced)."
        else:
            msg = f"All {n_total} cameras anchored."
        self._rough_status_label.setText(msg)
        self._rough_status_label.setStyleSheet("color: grey; font-size: 11px;")

    def _teardown_scrubber(self) -> None:
        if self._scrubber is not None:
            self._scrubber.shutdown()
            self._scrubber_layout.removeWidget(self._scrubber)
            self._scrubber.deleteLater()
            self._scrubber = None
        if self._cache is not None:
            self._cache.close_all()
            self._cache = None
        self._anchors.clear()
        self._anchor_overlays.clear()
        self._set_anchor_btn.setEnabled(False)
        self._clear_anchors_btn.setEnabled(False)
        self._apply_rough_btn.setEnabled(False)
        self._rough_status_label.setText("")
        self._rough_status_label.setStyleSheet("color: grey; font-size: 11px;")

    def _show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setVisible(True)

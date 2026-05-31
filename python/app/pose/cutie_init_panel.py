"""cutie_init_panel.py — Interactive Cutie segmentation initialisation panel.

Phase 1: video scrubber, camera selector, mask overlay from stored seg_masks.
Phase 2: click-to-SAM2 interaction (ClickController, not yet implemented).
Phase 3: Cutie worker thread, Track/Stop buttons, mask persistence.
Phase 4: correction workflow, RTMPose post-step.
"""
from __future__ import annotations

import sqlite3

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.pose.frame_cache import FrameCache
from app.pose.video_canvas import VideoCanvas, label_to_color


class CutieInitPanel(QWidget):
    """Video scrubber with mask overlay for interactive Cutie initialisation.

    Takes a detection_run_id to know which cameras, persons, and frame range
    to work with.  In Phase 1 it displays video frames and any seg_masks
    already stored in the DB; editing controls are added in later phases.
    """

    closed = Signal()

    def __init__(
        self,
        conn: sqlite3.Connection,
        detection_run_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._conn = conn
        self._run_id = detection_run_id
        self._frame_cache = FrameCache(max_frames=300, max_dim=1920)

        # Populated by _load_run()
        self._cameras: list[dict] = []          # [{id, label, file_path, first, last, fps, track_first, track_last}]
        self._seg_run_id: str | None = None     # most recent seg_quality_run for this run
        self._persons: list[str] = []           # person names in label order

        self._build_ui()
        self._load_run()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self._frame_cache.close()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # --- Top bar: camera selector + frame info ---
        top = QHBoxLayout()
        top.addWidget(QLabel("Camera:"))
        self._cam_combo = QComboBox()
        self._cam_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._cam_combo.currentIndexChanged.connect(self._on_camera_changed)
        top.addWidget(self._cam_combo)
        top.addStretch()
        self._frame_label = QLabel("Frame —")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self._frame_label)
        root.addLayout(top)

        # --- Video canvas ---
        self._canvas = VideoCanvas()
        root.addWidget(self._canvas, stretch=1)

        # --- Scrubber ---
        scrubber_row = QHBoxLayout()
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setMinimum(0)
        self._scrubber.setMaximum(0)
        self._scrubber.valueChanged.connect(self._on_frame_changed)
        scrubber_row.addWidget(self._scrubber, 1)
        root.addLayout(scrubber_row)

        # --- Person legend (populated once seg run is known) ---
        self._legend_group = QGroupBox("Persons")
        self._legend_layout = QHBoxLayout(self._legend_group)
        self._legend_layout.setContentsMargins(4, 2, 4, 2)
        root.addWidget(self._legend_group)

        # --- Status bar ---
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("font-size: 10px; color: #555;")
        root.addWidget(self._status_label)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_run(self) -> None:
        run_row = self._conn.execute(
            "SELECT shot_id FROM detection_runs WHERE id = ?",
            (self._run_id,),
        ).fetchone()
        if run_row is None:
            self._set_status("Detection run not found.")
            return

        shot_id = run_row["shot_id"]
        cam_rows = self._conn.execute(
            "SELECT cv.id, cv.file_path, cv.first_video_frame, cv.last_video_frame, "
            "       cv.actual_fps, "
            "       COALESCE(ci.label, cv.camera_instance_id) AS label, "
            "       MIN(pt.first_frame) AS track_first, "
            "       MAX(pt.last_frame)  AS track_last "
            "FROM capture_videos cv "
            "LEFT JOIN camera_instances ci ON ci.id = cv.camera_instance_id "
            "LEFT JOIN person_tracks pt "
            "       ON pt.shot_video_id = cv.id AND pt.detection_run_id = ? "
            "WHERE cv.shot_id = ? "
            "GROUP BY cv.id "
            "ORDER BY label",
            (self._run_id, shot_id),
        ).fetchall()

        self._cameras = [
            {
                "id": r["id"],
                "label": r["label"],
                "file_path": r["file_path"],
                "first": int(r["first_video_frame"]),
                "last": int(r["last_video_frame"]),
                "fps": float(r["actual_fps"] or 30.0),
                # Scrubber bounds: use track range when available, else full video.
                "track_first": int(r["track_first"]) if r["track_first"] is not None else int(r["first_video_frame"]),
                "track_last":  int(r["track_last"])  if r["track_last"]  is not None else int(r["last_video_frame"]),
            }
            for r in cam_rows
        ]

        # Find most recent seg_quality_run for this detection run.
        seg_row = self._conn.execute(
            "SELECT id FROM seg_quality_runs "
            "WHERE detection_run_id = ? ORDER BY created_at DESC LIMIT 1",
            (self._run_id,),
        ).fetchone()
        if seg_row:
            self._seg_run_id = seg_row["id"]

        # Derive person list from track assignments (distinct names, sorted).
        person_rows = self._conn.execute(
            "SELECT DISTINCT person_name FROM detection_track_assignments "
            "WHERE detection_run_id = ? ORDER BY person_name",
            (self._run_id,),
        ).fetchall()
        self._persons = [r["person_name"] for r in person_rows]

        self._rebuild_camera_combo()
        self._rebuild_legend()

    def _rebuild_camera_combo(self) -> None:
        self._cam_combo.blockSignals(True)
        self._cam_combo.clear()
        for cam in self._cameras:
            self._cam_combo.addItem(cam["label"], cam)
        self._cam_combo.blockSignals(False)
        if self._cameras:
            self._on_camera_changed(0)

    def _rebuild_legend(self) -> None:
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._persons:
            self._legend_group.setVisible(False)
            return

        self._legend_group.setVisible(True)
        for i, name in enumerate(self._persons):
            r, g, b = label_to_color(i + 1)
            swatch = QLabel("  ")
            swatch.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid #888; min-width:16px;"
            )
            swatch.setMaximumWidth(20)
            self._legend_layout.addWidget(swatch)
            self._legend_layout.addWidget(QLabel(name))
            if i < len(self._persons) - 1:
                sep = QLabel("|")
                sep.setStyleSheet("color: #aaa; padding: 0 4px;")
                self._legend_layout.addWidget(sep)
        self._legend_layout.addStretch()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_camera_changed(self, index: int) -> None:
        cam = self._cam_combo.itemData(index)
        if cam is None:
            return
        first = cam["track_first"]
        last = cam["track_last"]
        self._scrubber.blockSignals(True)
        self._scrubber.setMinimum(first)
        self._scrubber.setMaximum(last)
        self._scrubber.setValue(first)
        self._scrubber.blockSignals(False)
        self._show_frame(first)

    def _on_frame_changed(self, frame_idx: int) -> None:
        self._show_frame(frame_idx)

    def _show_frame(self, frame_idx: int) -> None:
        cam = self._cam_combo.currentData()
        if cam is None:
            return

        frame = self._frame_cache.get_frame(cam["file_path"], frame_idx)
        mask = self._load_mask(cam["id"], frame_idx)

        if frame is None:
            import os
            msg = (
                f"Video not accessible:\n{cam['file_path']}"
                if not os.path.exists(cam["file_path"])
                else f"Could not decode frame {frame_idx}"
            )
            self._canvas.display(None, message=msg)
        else:
            self._canvas.display(frame, mask)

        # Update frame label: show frame index and time relative to track start.
        t = (frame_idx - cam["track_first"]) / cam["fps"]
        mm, ss = divmod(t, 60)
        seg_indicator = " [mask]" if mask is not None else ""
        self._frame_label.setText(
            f"Frame {frame_idx}  ({int(mm):02d}:{ss:05.2f}){seg_indicator}"
        )

    def _load_mask(self, shot_video_id: str, frame_idx: int) -> np.ndarray | None:
        """Load stored seg_mask from DB, or None if not available."""
        if self._seg_run_id is None:
            return None
        row = self._conn.execute(
            "SELECT mask_blob FROM seg_masks "
            "WHERE seg_quality_run_id = ? AND shot_video_id = ? AND frame_idx = ?",
            (self._seg_run_id, shot_video_id, frame_idx),
        ).fetchone()
        if row is None:
            return None
        buf = np.frombuffer(bytes(row["mask_blob"]), dtype=np.uint8)
        mask = _decode_mask_png(buf)
        return mask

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)


def _decode_mask_png(buf: np.ndarray) -> np.ndarray | None:
    """Decode an indexed PNG mask blob to a (H, W) uint8 label array."""
    import cv2
    decoded = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return None
    if decoded.ndim == 3:
        # Shouldn't happen for indexed PNG but handle gracefully.
        decoded = decoded[:, :, 0]
    return decoded

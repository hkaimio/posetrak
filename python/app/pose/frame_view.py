"""frame_view.py — Single-camera frame viewer with pose overlay."""
from __future__ import annotations

import sqlite3

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.setup.camera_cell import CameraCell
from app.pose.db_cache import read_detections_for_run, read_keypoints_for_run


# COCO 17-keypoint skeleton connections
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class SkeletonDetectionOverlay:
    """Overlay drawing bboxes, keypoints, and stick figure."""

    def __init__(self):
        self._detections: list[dict] = []
        self._keypoints: dict[int, np.ndarray] = {}  # track_id -> [N,3]

    def set_detections(
        self,
        detections: list[dict],
        keypoints: dict[int, np.ndarray],
    ) -> None:
        self._detections = detections
        self._keypoints = keypoints

    def clear(self) -> None:
        self._detections = []
        self._keypoints = {}

    def paint(
        self,
        painter: QPainter,
        frame_w: int,
        frame_h: int,
        cell_w: int,
        cell_h: int,
    ) -> None:
        if not self._detections and not self._keypoints:
            return

        sx = cell_w / max(frame_w, 1)
        sy = cell_h / max(frame_h, 1)

        for det in self._detections:
            # bbox is centre-format xywh
            cx = det["bbox_x"] * sx
            cy = det["bbox_y"] * sy
            bw = det["bbox_w"] * sx
            bh = det["bbox_h"] * sy
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)

            pen = QPen(QColor(255, 220, 0), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x1, y1, int(bw), int(bh))

            tid = det["track_id"]
            painter.drawText(x1 + 2, y1 - 4, f"t{tid}")

            kp = self._keypoints.get(tid)
            if kp is None:
                continue

            n_kp = kp.shape[0]

            # Draw sticks
            painter.setPen(QPen(QColor(255, 255, 255, 160), 1))
            for a, b in COCO_SKELETON:
                if a >= n_kp or b >= n_kp:
                    continue
                if kp[a, 2] < 0.1 or kp[b, 2] < 0.1:
                    continue
                xa, ya = int(kp[a, 0] * sx), int(kp[a, 1] * sy)
                xb, yb = int(kp[b, 0] * sx), int(kp[b, 1] * sy)
                painter.drawLine(xa, ya, xb, yb)

            # Draw keypoint dots
            for i in range(n_kp):
                x = int(kp[i, 0] * sx)
                y = int(kp[i, 1] * sy)
                conf = float(kp[i, 2])
                if conf > 0.5:
                    color = QColor(0, 220, 60)
                elif conf > 0.3:
                    color = QColor(255, 200, 0)
                else:
                    color = QColor(220, 40, 40)
                painter.setPen(Qt.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(x - 3, y - 3, 6, 6)

    def mouse_press(self, x_px: int, y_px: int) -> None:
        pass

    def mouse_move(self, x_px: int, y_px: int) -> None:
        pass

    def mouse_release(self, x_px: int, y_px: int) -> None:
        pass


class FrameViewWidget(QWidget):
    """Single-camera frame viewer with navigation and pose overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._shot_video_id: str | None = None
        self._file_path: str | None = None
        self._camera_instance_id: str | None = None
        self._total_frames: int = 0
        self._current_frame: int = 0
        self._session: sqlite3.Connection | None = None
        self._detection_run_id: str | None = None
        self._track_id: int | None = None

        # Per-frame detections loaded from DB
        # frame -> list[dict]
        self._det_by_frame: dict[int, list[dict]] = {}
        # frame -> {track_id: kp array}
        self._kp_by_frame: dict[int, dict[int, np.ndarray]] = {}

        self._overlay = SkeletonDetectionOverlay()
        self._cell = CameraCell(parent=self)
        self._cell.set_overlays([self._overlay])
        self._cell.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        self._info_label = QLabel("frame: -  t: -")
        self._info_label.setAlignment(Qt.AlignLeft)

        self._cam_combo = QComboBox()
        self._cam_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider_changed)

        self._prev_btn = QPushButton("<")
        self._prev_btn.setFixedWidth(30)
        self._prev_btn.clicked.connect(self._on_prev)

        self._next_btn = QPushButton(">")
        self._next_btn.setFixedWidth(30)
        self._next_btn.clicked.connect(self._on_next)

        nav_row = QHBoxLayout()
        nav_row.addWidget(self._prev_btn)
        nav_row.addWidget(self._slider)
        nav_row.addWidget(self._next_btn)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Camera:"))
        top_row.addWidget(self._cam_combo)
        top_row.addStretch()
        top_row.addWidget(self._info_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top_row)
        layout.addWidget(self._cell, stretch=1)
        layout.addLayout(nav_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_camera(self, shot_video_id: str, file_path: str, camera_instance_id: str) -> None:
        self._shot_video_id = shot_video_id
        self._file_path = file_path
        self._camera_instance_id = camera_instance_id
        self._det_by_frame.clear()
        self._kp_by_frame.clear()
        self._overlay.clear()

        # Detect total frames
        cap = cv2.VideoCapture(file_path)
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self._slider.setMaximum(max(0, self._total_frames - 1))
        self._slider.setValue(0)

        # Update combo
        idx = self._cam_combo.findData(shot_video_id)
        if idx < 0:
            self._cam_combo.addItem(camera_instance_id[:12], shot_video_id)
            idx = self._cam_combo.count() - 1
        self._cam_combo.setCurrentIndex(idx)

        self.seek_frame(0)

    def seek_frame(self, frame_idx: int) -> None:
        if self._file_path is None:
            return
        frame_idx = max(0, min(frame_idx, self._total_frames - 1))
        self._current_frame = frame_idx

        cap = cv2.VideoCapture(self._file_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap.read()
        cap.release()

        if ret:
            self._cell.set_frame(img)
        else:
            self._cell.clear_frame()

        self._update_overlay(frame_idx)
        self._update_info_label(frame_idx)

        # Sync slider without triggering seek again
        self._slider.blockSignals(True)
        self._slider.setValue(frame_idx)
        self._slider.blockSignals(False)

    def set_pose_data(
        self,
        session: sqlite3.Connection,
        detection_run_id: str,
        track_id: int | None = None,
    ) -> None:
        """Load detection data for overlay from DB."""
        if self._shot_video_id is None:
            return

        self._session = session
        self._detection_run_id = detection_run_id
        self._track_id = track_id
        self._det_by_frame.clear()
        self._kp_by_frame.clear()

        # Load all detections for this camera
        dets = read_detections_for_run(session, detection_run_id, self._shot_video_id)
        for det in dets:
            f = det["video_frame"]
            self._det_by_frame.setdefault(f, []).append(det)

        # Load keypoints for selected track (or all tracks)
        if track_id is not None:
            kp_map = read_keypoints_for_run(
                session, detection_run_id, self._shot_video_id, track_id
            )
            for frame, kp in kp_map.items():
                self._kp_by_frame.setdefault(frame, {})[track_id] = kp
        else:
            # Load all tracks
            track_ids_rows = session.execute(
                "SELECT DISTINCT track_id FROM person_tracks "
                "WHERE detection_run_id=? AND shot_video_id=?",
                (detection_run_id, self._shot_video_id),
            ).fetchall()
            for row in track_ids_rows:
                tid = row["track_id"]
                kp_map = read_keypoints_for_run(
                    session, detection_run_id, self._shot_video_id, tid
                )
                for frame, kp in kp_map.items():
                    self._kp_by_frame.setdefault(frame, {})[tid] = kp

        self._update_overlay(self._current_frame)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_overlay(self, frame_idx: int) -> None:
        dets = self._det_by_frame.get(frame_idx, [])
        kps = self._kp_by_frame.get(frame_idx, {})
        self._overlay.set_detections(dets, kps)
        self._cell.update()

    def _update_info_label(self, frame_idx: int) -> None:
        total_s = frame_idx  # rough placeholder without fps
        mm = total_s // 60
        ss = total_s % 60
        self._info_label.setText(f"frame: {frame_idx}  t: {mm:02d}:{ss:02d}.00")

    def _on_slider_changed(self, value: int) -> None:
        if value != self._current_frame:
            self.seek_frame(value)

    def _on_prev(self) -> None:
        self.seek_frame(self._current_frame - 1)

    def _on_next(self) -> None:
        self.seek_frame(self._current_frame + 1)

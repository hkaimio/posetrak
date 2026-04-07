"""frame_view.py — Single-camera frame viewer with pose overlay."""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

import cv2

_log = logging.getLogger(__name__)
import numpy as np
from PySide6.QtCore import Qt, Signal
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


class _ComboBox(QComboBox):
    """QComboBox that reliably closes its popup on item selection (see main.py)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.activated.connect(lambda _: self.hidePopup())


# COCO 17-keypoint skeleton connections
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


@dataclass
class _CameraInfo:
    shot_video_id: str
    file_path: str
    camera_instance_id: str
    label: str
    fps: float
    ref_frame: int
    ref_timestamp_s: float


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
    """Single-camera frame viewer with navigation and pose overlay.

    Supports browsing multiple cameras for a shot before a detection run
    exists.  Call ``load_cameras()`` to populate the camera dropdown from
    a list of shot videos; the user can then switch cameras via the combo.

    Signals:
        frame_changed(frame_idx, global_time_s): emitted on every seek.
        camera_switched(shot_video_id): emitted when the user picks a
            different camera from the dropdown.
    """

    frame_changed = Signal(int, float)   # frame_idx, global_time_s
    camera_switched = Signal(str)        # shot_video_id

    def __init__(self, parent=None):
        super().__init__(parent)

        # Per-camera metadata, keyed by shot_video_id
        self._cameras: dict[str, _CameraInfo] = {}

        self._shot_video_id: str | None = None
        self._file_path: str | None = None
        self._total_frames: int = 0
        self._current_frame: int = 0
        self._fps: float = 30.0
        self._ref_frame: int = 0
        self._ref_timestamp_s: float = 0.0
        self._session: sqlite3.Connection | None = None
        self._detection_run_id: str | None = None
        self._track_id: int | None = None

        # Per-frame detections loaded from DB
        self._det_by_frame: dict[int, list[dict]] = {}
        self._kp_by_frame: dict[int, dict[int, np.ndarray]] = {}

        self._overlay = SkeletonDetectionOverlay()
        self._cell = CameraCell(parent=self)
        self._cell.set_overlays([self._overlay])
        self._cell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._info_label = QLabel("frame: -  t: -")
        self._info_label.setAlignment(Qt.AlignLeft)

        self._cam_combo = _ComboBox()
        self._cam_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._cam_combo.currentIndexChanged.connect(self._on_cam_combo_changed)

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

    def load_cameras(self, cameras: list[_CameraInfo]) -> None:
        """Populate the camera dropdown from a list of shot videos.

        Replaces any previously loaded cameras. Selects the first camera.
        Does not require a detection run to be present.
        """
        self._cameras = {c.shot_video_id: c for c in cameras}

        self._cam_combo.blockSignals(True)
        self._cam_combo.clear()
        for c in cameras:
            self._cam_combo.addItem(c.label, c.shot_video_id)
        self._cam_combo.blockSignals(False)

        if cameras:
            self._switch_to_camera(cameras[0].shot_video_id)

    def load_camera(
        self,
        shot_video_id: str,
        file_path: str,
        camera_instance_id: str,
        fps: float = 30.0,
        ref_frame: int = 0,
        ref_timestamp_s: float = 0.0,
    ) -> None:
        """Load a single camera (legacy call path, still used from the stitcher click handler)."""
        label = camera_instance_id[:12]
        cam = _CameraInfo(
            shot_video_id=shot_video_id,
            file_path=file_path,
            camera_instance_id=camera_instance_id,
            label=label,
            fps=fps,
            ref_frame=ref_frame,
            ref_timestamp_s=ref_timestamp_s,
        )
        if shot_video_id not in self._cameras:
            self._cameras[shot_video_id] = cam
            self._cam_combo.blockSignals(True)
            self._cam_combo.addItem(label, shot_video_id)
            self._cam_combo.blockSignals(False)
        else:
            self._cameras[shot_video_id] = cam

        idx = self._cam_combo.findData(shot_video_id)
        if idx >= 0:
            self._cam_combo.blockSignals(True)
            self._cam_combo.setCurrentIndex(idx)
            self._cam_combo.blockSignals(False)

        self._switch_to_camera(shot_video_id, seek_to=0)

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
            _log.warning("seek_frame: cv2 read failed for frame %d in %s", frame_idx, self._file_path)
            self._cell.clear_frame()

        self._update_overlay(frame_idx)
        global_s = self._update_info_label(frame_idx)

        self._slider.blockSignals(True)
        self._slider.setValue(frame_idx)
        self._slider.blockSignals(False)

        self.frame_changed.emit(frame_idx, global_s)

    def seek_global_time(self, global_s: float) -> None:
        """Seek to the frame closest to *global_s* using the current camera's sync anchor."""
        if self._fps <= 0:
            return
        frame_idx = int(self._ref_frame + (global_s - self._ref_timestamp_s) * self._fps)
        self.seek_frame(frame_idx)

    def current_global_time(self) -> float:
        """Return the global timestamp of the currently displayed frame."""
        return self._ref_timestamp_s + (self._current_frame - self._ref_frame) / max(self._fps, 1)

    def current_shot_video_id(self) -> str | None:
        return self._shot_video_id

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

        dets = read_detections_for_run(session, detection_run_id, self._shot_video_id)
        for det in dets:
            f = det["video_frame"]
            self._det_by_frame.setdefault(f, []).append(det)

        if track_id is not None:
            kp_map = read_keypoints_for_run(
                session, detection_run_id, self._shot_video_id, track_id
            )
            for frame, kp in kp_map.items():
                self._kp_by_frame.setdefault(frame, {})[track_id] = kp
        else:
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

    def _switch_to_camera(self, shot_video_id: str, seek_to: int | None = None) -> None:
        """Switch the viewer to the given camera, preserving global time if possible."""
        cam = self._cameras.get(shot_video_id)
        if cam is None:
            return

        # Remember current global time before switching
        prev_global_s = self.current_global_time() if self._shot_video_id else None

        self._shot_video_id = shot_video_id
        self._file_path = cam.file_path
        self._fps = cam.fps
        self._ref_frame = cam.ref_frame
        self._ref_timestamp_s = cam.ref_timestamp_s
        self._det_by_frame.clear()
        self._kp_by_frame.clear()
        self._overlay.clear()

        cap = cv2.VideoCapture(cam.file_path)
        if not cap.isOpened():
            _log.error("_switch_to_camera: could not open %s", cam.file_path)
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, self._total_frames - 1))
        self._slider.blockSignals(False)

        # Seek: explicit override > preserve global time > frame 0
        if seek_to is not None:
            target = seek_to
        elif prev_global_s is not None:
            target = int(cam.ref_frame + (prev_global_s - cam.ref_timestamp_s) * cam.fps)
        else:
            target = 0

        self.seek_frame(target)
        self.camera_switched.emit(shot_video_id)

    def _update_overlay(self, frame_idx: int) -> None:
        dets = self._det_by_frame.get(frame_idx, [])
        kps = self._kp_by_frame.get(frame_idx, {})
        self._overlay.set_detections(dets, kps)
        self._cell.update()

    def _update_info_label(self, frame_idx: int) -> float:
        """Update the info label; return the computed global time in seconds."""
        global_s = self._ref_timestamp_s + (frame_idx - self._ref_frame) / max(self._fps, 1)
        mm = int(global_s // 60)
        ss = global_s % 60
        self._info_label.setText(f"frame: {frame_idx}  t: {mm:02d}:{ss:05.2f}")
        return global_s

    def _on_cam_combo_changed(self, index: int) -> None:
        svid = self._cam_combo.itemData(index)
        if svid and svid != self._shot_video_id:
            self._switch_to_camera(svid)

    def _on_slider_changed(self, value: int) -> None:
        if value != self._current_frame:
            self.seek_frame(value)

    def _on_prev(self) -> None:
        self.seek_frame(self._current_frame - 1)

    def _on_next(self) -> None:
        self.seek_frame(self._current_frame + 1)

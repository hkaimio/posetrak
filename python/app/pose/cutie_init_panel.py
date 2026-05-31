"""cutie_init_panel.py — Interactive Cutie segmentation initialisation panel.

Phase 1: video scrubber, camera selector, mask overlay from stored seg_masks.
Phase 2: click-to-SAM2 interaction — PersonSelector buttons, ClickController.
Phase 3: Cutie worker thread, Track/Stop buttons, mask persistence.
Phase 4: correction workflow, RTMPose post-step.
"""
from __future__ import annotations

import sqlite3

import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QButtonGroup,
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
    """Video scrubber + click-based SAM2 segmentation init panel.

    Camera selector → frame scrubber → VideoCanvas with mask overlay.
    PersonSelector buttons let the user choose which person label to assign
    to each click.  Left-click = positive (foreground), right-click = negative.

    SAM2 encoding is lazy: the image encoder runs on the first click on a frame
    (or immediately when a person button is selected and the video is accessible).
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
        self._cameras: list[dict] = []
        self._seg_run_id: str | None = None
        self._persons: list[str] = []       # person names, index+1 = SAM label

        # Click interaction state
        self._controller = None             # ClickController; created lazily
        self._selected_label: int = 0       # 0 = no person selected
        self._encoded_frame_idx: int = -1   # frame that _controller has encoded
        self._encoded_svid: str = ""        # camera id for encoded frame

        # Debounce timer: encode image 300 ms after last scrub event when
        # a person button is active (pre-warm the encoder for fast first click).
        self._encode_timer = QTimer(self)
        self._encode_timer.setSingleShot(True)
        self._encode_timer.timeout.connect(self._encode_current_frame)

        self._build_ui()
        self._load_run()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self._encode_timer.stop()
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
        self._canvas.left_clicked.connect(self._on_left_click)
        self._canvas.right_clicked.connect(self._on_right_click)
        root.addWidget(self._canvas, stretch=1)

        # --- Scrubber ---
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setMinimum(0)
        self._scrubber.setMaximum(0)
        self._scrubber.valueChanged.connect(self._on_frame_changed)
        root.addLayout(self._make_scrubber_row())

        # --- Person selector (populated after _load_run) ---
        self._person_group = QGroupBox("Person  (click to select, then click on canvas)")
        self._person_layout = QHBoxLayout(self._person_group)
        self._person_layout.setContentsMargins(4, 2, 4, 2)
        self._person_layout.setSpacing(4)
        self._person_btn_group = QButtonGroup(self)
        self._person_btn_group.setExclusive(False)
        root.addWidget(self._person_group)

        # --- Edit action buttons ---
        edit_row = QHBoxLayout()
        self._clear_person_btn = QPushButton("Clear person")
        self._clear_person_btn.setToolTip("Remove all clicks for the selected person")
        self._clear_person_btn.clicked.connect(self._on_clear_person)
        self._clear_person_btn.setEnabled(False)
        edit_row.addWidget(self._clear_person_btn)

        self._clear_all_btn = QPushButton("Clear all")
        self._clear_all_btn.setToolTip("Remove all clicks for all persons")
        self._clear_all_btn.clicked.connect(self._on_clear_all)
        self._clear_all_btn.setEnabled(False)
        edit_row.addWidget(self._clear_all_btn)
        edit_row.addStretch()

        self._sam_status_label = QLabel("")
        self._sam_status_label.setStyleSheet("font-size: 10px; color: #666;")
        edit_row.addWidget(self._sam_status_label)
        root.addLayout(edit_row)

        # --- Status bar ---
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("font-size: 10px; color: #555;")
        root.addWidget(self._status_label)

    def _make_scrubber_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(self._scrubber, 1)
        return row

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
                "track_first": int(r["track_first"]) if r["track_first"] is not None else int(r["first_video_frame"]),
                "track_last":  int(r["track_last"])  if r["track_last"]  is not None else int(r["last_video_frame"]),
            }
            for r in cam_rows
        ]

        seg_row = self._conn.execute(
            "SELECT id FROM seg_quality_runs "
            "WHERE detection_run_id = ? ORDER BY created_at DESC LIMIT 1",
            (self._run_id,),
        ).fetchone()
        if seg_row:
            self._seg_run_id = seg_row["id"]

        person_rows = self._conn.execute(
            "SELECT DISTINCT person_name FROM detection_track_assignments "
            "WHERE detection_run_id = ? ORDER BY person_name",
            (self._run_id,),
        ).fetchall()
        self._persons = [r["person_name"] for r in person_rows]

        self._rebuild_camera_combo()
        self._rebuild_person_selector()
        self._init_controller()

    def _rebuild_camera_combo(self) -> None:
        self._cam_combo.blockSignals(True)
        self._cam_combo.clear()
        for cam in self._cameras:
            self._cam_combo.addItem(cam["label"], cam)
        self._cam_combo.blockSignals(False)
        if self._cameras:
            self._on_camera_changed(0)

    def _rebuild_person_selector(self) -> None:
        # Clear old buttons
        for btn in self._person_btn_group.buttons():
            self._person_btn_group.removeButton(btn)
            btn.deleteLater()
        while self._person_layout.count():
            item = self._person_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._persons:
            self._person_group.setVisible(False)
            return

        self._person_group.setVisible(True)
        for i, name in enumerate(self._persons):
            label = i + 1  # 1-based SAM label
            r, g, b = label_to_color(label)
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "#000" if luma > 140 else "#fff"
            btn = QPushButton(f"{label}: {name}")
            btn.setCheckable(True)
            btn.setProperty("person_label", label)
            btn.setStyleSheet(
                f"QPushButton {{ background-color: rgb({r},{g},{b}); color: {text_color};"
                f" border: 2px solid transparent; padding: 3px 8px; }}"
                f"QPushButton:checked {{ border: 2px solid #000; font-weight: bold; }}"
            )
            # Keyboard shortcut 1..9
            if label <= 9:
                QShortcut(QKeySequence(str(label)), self).activated.connect(
                    lambda checked=False, b=btn: self._toggle_person_btn(b)
                )
            btn.clicked.connect(lambda checked, b=btn: self._on_person_btn_clicked(b))
            self._person_btn_group.addButton(btn, label)
            self._person_layout.addWidget(btn)

        self._person_layout.addStretch()

    def _init_controller(self) -> None:
        """Create the ClickController lazily (imports SAM2 if available)."""
        try:
            from app.pose.cutie_click_controller import ClickController
            self._controller = ClickController()
            if self._controller.available:
                self._sam_status_label.setText("SAM2 ready")
                self._sam_status_label.setStyleSheet("font-size: 10px; color: #080;")
            else:
                self._sam_status_label.setText("SAM2 not available — install ultralytics")
                self._sam_status_label.setStyleSheet("font-size: 10px; color: #c60;")
        except Exception as e:
            self._sam_status_label.setText(f"SAM2 error: {e}")

    # ------------------------------------------------------------------
    # Interaction — person selector
    # ------------------------------------------------------------------

    def _toggle_person_btn(self, btn: QPushButton) -> None:
        btn.setChecked(not btn.isChecked())
        self._on_person_btn_clicked(btn)

    def _on_person_btn_clicked(self, clicked_btn: QPushButton) -> None:
        label = clicked_btn.property("person_label")
        if clicked_btn.isChecked():
            # Deselect all others (manual exclusivity so clicking same btn toggles off)
            for btn in self._person_btn_group.buttons():
                if btn is not clicked_btn:
                    btn.setChecked(False)
            self._selected_label = label
            self._canvas.setCursor(Qt.CursorShape.CrossCursor)
            # Pre-warm encoder when person is selected and frame is accessible
            self._schedule_encode()
        else:
            self._selected_label = 0
            self._canvas.setCursor(Qt.CursorShape.ArrowCursor)
            self._encode_timer.stop()

        self._update_edit_buttons()

    def _update_edit_buttons(self) -> None:
        has_controller = self._controller is not None
        has_selection = self._selected_label > 0
        self._clear_person_btn.setEnabled(has_controller and has_selection)
        self._clear_all_btn.setEnabled(has_controller)

    # ------------------------------------------------------------------
    # Interaction — canvas clicks
    # ------------------------------------------------------------------

    def _on_left_click(self, x: int, y: int) -> None:
        self._handle_click(x, y, positive=True)

    def _on_right_click(self, x: int, y: int) -> None:
        self._handle_click(x, y, positive=False)

    def _handle_click(self, x: int, y: int, positive: bool) -> None:
        if self._selected_label == 0:
            self._set_status("Select a person button first, then click on the frame.")
            return
        if self._controller is None or not self._controller.available:
            self._set_status("SAM2 not available.")
            return

        cam = self._cam_combo.currentData()
        frame_idx = self._scrubber.value()
        if cam is None:
            return

        # Encode if needed (lazy, also handles first click after scrub)
        self._ensure_encoded(cam, frame_idx)

        self._set_status("Running SAM2…")
        mask = self._controller.push_point(self._selected_label, x, y, positive)
        self._set_status(
            f"Person {self._selected_label}: "
            f"{self._controller.click_count(self._selected_label)} click(s)"
        )
        self._refresh_overlay(cam, frame_idx, mask)

    def _on_clear_person(self) -> None:
        if self._controller is None or self._selected_label == 0:
            return
        cam = self._cam_combo.currentData()
        frame_idx = self._scrubber.value()
        mask = self._controller.clear_person(self._selected_label)
        self._set_status(f"Cleared person {self._selected_label}")
        self._refresh_overlay(cam, frame_idx, mask)

    def _on_clear_all(self) -> None:
        if self._controller is None:
            return
        cam = self._cam_combo.currentData()
        frame_idx = self._scrubber.value()
        self._controller.clear_all()
        self._set_status("Cleared all clicks")
        frame = self._frame_cache.get_frame(cam["file_path"], frame_idx) if cam else None
        self._canvas.display(frame)

    # ------------------------------------------------------------------
    # Interaction — scrubbing
    # ------------------------------------------------------------------

    def _on_camera_changed(self, index: int) -> None:
        cam = self._cam_combo.itemData(index)
        if cam is None:
            return
        # Camera switch invalidates the encoded frame
        self._encoded_frame_idx = -1
        self._encoded_svid = ""
        if self._controller:
            self._controller.clear_all()

        self._scrubber.blockSignals(True)
        self._scrubber.setMinimum(cam["track_first"])
        self._scrubber.setMaximum(cam["track_last"])
        self._scrubber.setValue(cam["track_first"])
        self._scrubber.blockSignals(False)
        self._show_frame(cam["track_first"])

    def _on_frame_changed(self, frame_idx: int) -> None:
        # Frame changed: clear click state, show new frame.
        if self._controller:
            self._controller.clear_all()
        self._encoded_frame_idx = -1
        self._show_frame(frame_idx)
        # If a person is selected, pre-warm encoder after scrubbing stops.
        if self._selected_label > 0:
            self._schedule_encode()

    def _schedule_encode(self) -> None:
        """Start/restart the debounce timer to encode the current frame."""
        self._encode_timer.start(300)

    def _encode_current_frame(self) -> None:
        """Called by debounce timer: encode current frame if accessible."""
        cam = self._cam_combo.currentData()
        frame_idx = self._scrubber.value()
        if cam is not None:
            self._ensure_encoded(cam, frame_idx)

    def _ensure_encoded(self, cam: dict, frame_idx: int) -> None:
        """Encode the frame for SAM2 if not already done for this frame."""
        if (
            self._controller is None
            or not self._controller.available
            or (self._encoded_frame_idx == frame_idx and self._encoded_svid == cam["id"])
        ):
            return

        frame = self._frame_cache.get_frame(cam["file_path"], frame_idx)
        if frame is None:
            return

        self._set_status("Encoding frame for SAM2…")
        self._controller.set_image(frame)
        self._encoded_frame_idx = frame_idx
        self._encoded_svid = cam["id"]
        self._set_status("Ready — click to segment")

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _show_frame(self, frame_idx: int) -> None:
        cam = self._cam_combo.currentData()
        if cam is None:
            return

        frame = self._frame_cache.get_frame(cam["file_path"], frame_idx)
        # Prefer live controller mask; fall back to stored DB mask.
        mask = None
        if self._controller and np.any(self._controller.get_mask()):
            mask = self._controller.get_mask()
        else:
            mask = self._load_stored_mask(cam["id"], frame_idx)

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

        t = (frame_idx - cam["track_first"]) / cam["fps"]
        mm, ss = divmod(t, 60)
        seg_indicator = " [mask]" if mask is not None else ""
        self._frame_label.setText(
            f"Frame {frame_idx}  ({int(mm):02d}:{ss:05.2f}){seg_indicator}"
        )

    def _refresh_overlay(
        self, cam: dict | None, frame_idx: int, mask: np.ndarray
    ) -> None:
        """Redraw the canvas with *mask* without reloading the frame."""
        if cam is None:
            return
        frame = self._frame_cache.get_frame(cam["file_path"], frame_idx)
        self._canvas.display(frame, mask if np.any(mask) else None)

    def _load_stored_mask(
        self, shot_video_id: str, frame_idx: int
    ) -> np.ndarray | None:
        """Load a previously saved seg_mask blob from the DB."""
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
        return _decode_mask_png(buf)

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)


def _decode_mask_png(buf: np.ndarray) -> np.ndarray | None:
    """Decode an indexed PNG mask blob to a (H, W) uint8 label array."""
    import cv2
    decoded = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return None
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded

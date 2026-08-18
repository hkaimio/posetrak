# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

import sys
import cv2
import yaml  # <-- Add this import
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QMessageBox, QScrollArea, QGridLayout
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QPoint

import re
import os

def parse_timecode(tc, fps):
    if isinstance(tc, int):
        return tc
    if isinstance(tc, float):
        return int(round(tc))
    if isinstance(tc, str):
        tc = tc.strip()
        m = re.match(r'^(?:(\d+):)?(?:(\d+):)?(\d+)(?:[.,](\d+))?$', tc)
        if not m:
            raise ValueError(f"Invalid timecode format: {tc}")
        h = int(m.group(1)) if m.group(2) else 0
        m_ = int(m.group(2)) if m.group(2) else (int(m.group(1)) if m.group(1) and not m.group(2) else 0)
        s = int(m.group(3))
        f = int(m.group(4)) if m.group(4) else 0
        if m.group(4):
            total_seconds = h * 3600 + m_ * 60 + s + float("0." + m.group(4))
        else:
            total_seconds = h * 3600 + m_ * 60 + s
        return int(round(total_seconds * fps))
    raise ValueError(f"Invalid frame/timecode value: {tc}")

class VideoWidget(QWidget):
    def __init__(self, video_path, sync_frame=0, fps=None):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", f"Cannot open video: {video_path}")
            return
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps is None:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.fps = fps
        self.current_frame = 0
        self.sync_frame = sync_frame  # frame index in this video corresponding to sync event
        self.display_width = 640
        self.display_height = 480
        self.current_full_frame = None  # Store the current frame for zooming
        self.zoom_center = None  # (x, y) in original video coordinates

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.mousePressEvent = self.on_video_click

        # Enlarged view
        self.zoom_label = QLabel()
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setFixedSize(256, 256)
        self.zoom_label.setStyleSheet("border: 1px solid gray;")

        self.frame_label = QLabel()
        self.jump_edit = QLineEdit()
        self.jump_edit.setPlaceholderText("Frame #")
        self.jump_edit.setFixedWidth(60)

        prev_btn = QPushButton("⏮️")
        next_btn = QPushButton("⏭️")
        prev_btn.clicked.connect(self.prev_frame)
        next_btn.clicked.connect(self.next_frame)
        self.jump_edit.returnPressed.connect(self.jump_to_frame)

        controls = QHBoxLayout()
        controls.addWidget(prev_btn)
        controls.addWidget(next_btn)
        controls.addWidget(self.jump_edit)
        controls.addWidget(self.frame_label)

        # Layout for video and zoom side by side
        video_zoom_layout = QHBoxLayout()
        video_zoom_layout.addWidget(self.image_label)
        video_zoom_layout.addWidget(self.zoom_label)

        # Header with filename and FPS
        header_layout = QHBoxLayout()
        filename_label = QLabel(video_path)
        fps_label = QLabel(f"FPS: {self.fps:.2f}")
        fps_label.setStyleSheet("color: gray;")
        header_layout.addWidget(filename_label)
        header_layout.addStretch()
        header_layout.addWidget(fps_label)

        layout = QVBoxLayout()
        layout.addLayout(header_layout)
        layout.addLayout(video_zoom_layout)
        layout.addLayout(controls)
        self.setLayout(layout)

        self.show_frame(self.current_frame)

    def on_video_click(self, event):
        """Handle mouse click on video to set zoom center."""
        if self.current_full_frame is None:
            return

        # Get click position relative to the label
        click_pos = event.pos()
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return

        # Calculate the actual displayed image position within the label
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # Offset to center of displayed image
        x_offset = (label_width - pixmap_width) // 2
        y_offset = (label_height - pixmap_height) // 2

        # Click position relative to displayed image
        img_x = click_pos.x() - x_offset
        img_y = click_pos.y() - y_offset

        # Check if click is within the displayed image
        if img_x < 0 or img_x >= pixmap_width or img_y < 0 or img_y >= pixmap_height:
            return

        # Scale to original frame coordinates
        h, w = self.current_full_frame.shape[:2]
        orig_x = int(img_x * w / pixmap_width)
        orig_y = int(img_y * h / pixmap_height)

        self.zoom_center = (orig_x, orig_y)
        self.update_zoom_view()

    def update_zoom_view(self):
        """Update the enlarged view based on zoom_center."""
        if self.current_full_frame is None or self.zoom_center is None:
            return

        h, w = self.current_full_frame.shape[:2]
        cx, cy = self.zoom_center

        # Extract 128x128 area centered at click point
        half_size = 64
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)

        # Extract and pad if necessary
        crop = self.current_full_frame[y1:y2, x1:x2]

        # Pad if the crop is smaller than 128x128 (near edges)
        crop_h, crop_w = crop.shape[:2]
        if crop_h < 128 or crop_w < 128:
            pad_top = max(0, half_size - cy)
            pad_bottom = max(0, cy + half_size - h)
            pad_left = max(0, half_size - cx)
            pad_right = max(0, cx + half_size - w)
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Resize to 256x256 (2x zoom)
        zoomed = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Convert to Qt image
        rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
        h_z, w_z, ch = rgb.shape
        bytes_per_line = ch * w_z
        qt_img = QImage(rgb.data, w_z, h_z, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.zoom_label.setPixmap(pixmap)

    def show_frame(self, frame_idx):
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_full_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.image_label.setPixmap(pixmap.scaled(self.display_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio))
        self.frame_label.setText(f"Frame: {frame_idx+1}/{self.frame_count}")
        self.current_frame = frame_idx

        # Update zoom view if a location has been selected
        if self.zoom_center is not None:
            self.update_zoom_view()

    def prev_frame(self):
        self.show_frame(max(0, self.current_frame - 1))

    def next_frame(self):
        self.show_frame(min(self.frame_count - 1, self.current_frame + 1))

    def jump_to_frame(self):
        try:
            idx = int(self.jump_edit.text()) - 1
            self.show_frame(idx)
        except ValueError:
            pass

    def goto_synced_frame(self, ref_diff: float):
        """Move this video to the frame with given difference to reference frame.

        Args:
            ref_diff (float): Difference in frames from reference video's sync frame in seconds.
        """
        new_frame = self.sync_frame + ref_diff * self.fps
        self.show_frame(int(round(new_frame)))

    def set_display_size(self, width, height):
        """Set the display size for video frames."""
        self.display_width = width
        self.display_height = height
        self.show_frame(self.current_frame)

class MainWindow(QWidget):
    def __init__(self, video_infos, ref_idx=0):
        super().__init__()
        self.setWindowTitle("Multi Video Frame Stepper")
        self.video_widgets = []
        self.ref_idx = ref_idx
        self.video_infos = video_infos  # Store original video infos
        layout = QVBoxLayout()

        # Create grid layout for videos (max 3 per row)
        grid_layout = QGridLayout()
        videos_per_row = 3
        for idx, info in enumerate(video_infos):
            vw = VideoWidget(info["path"], sync_frame=info["sync_frame"], fps=info["fps"])
            self.video_widgets.append(vw)
            row = idx // videos_per_row
            col = idx % videos_per_row
            grid_layout.addWidget(vw, row, col)

        layout.addLayout(grid_layout)

        # Button controls
        button_layout = QHBoxLayout()
        sync_btn = QPushButton("Synchronize to ref_camera")
        sync_btn.clicked.connect(self.synchronize_videos)
        button_layout.addWidget(sync_btn)

        set_ref_btn = QPushButton("Set sync reference")
        set_ref_btn.clicked.connect(self.set_sync_reference)
        button_layout.addWidget(set_ref_btn)

        save_btn = QPushButton("Save Project")
        save_btn.clicked.connect(self.save_project)
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Adjust video sizes based on number of videos
        self.adjust_video_sizes()

    def adjust_video_sizes(self):
        """Adjust video display sizes based on window size and number of videos."""
        num_videos = len(self.video_widgets)
        videos_per_row = min(3, num_videos)

        # Calculate size per video
        available_width = 1920  # default, will be adjusted on resize
        available_height = 1080

        video_width = available_width // videos_per_row - 40
        video_height = available_height // ((num_videos + videos_per_row - 1) // videos_per_row) - 100

        for vw in self.video_widgets:
            vw.set_display_size(video_width, video_height)

    def synchronize_videos(self):
        ref_widget = self.video_widgets[self.ref_idx]
        ref_frame = ref_widget.current_frame
        ref_sync_frame = ref_widget.sync_frame
        ref_diff = (ref_frame - ref_sync_frame) / ref_widget.fps
        for i, vw in enumerate(self.video_widgets):
            if i == self.ref_idx:
                continue
            vw.goto_synced_frame(ref_diff)

    def set_sync_reference(self):
        """Set the sync frame for all videos to their current frame."""
        for vw in self.video_widgets:
            vw.sync_frame = vw.current_frame
        QMessageBox.information(self, "Sync Reference Set",
                                "All videos' sync frames have been set to their current frames.")

    def save_project(self):
        """Save the current project configuration to a YAML file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )

        if not file_path:
            return

        # Ensure .yaml extension
        if not file_path.endswith(('.yaml', '.yml')):
            file_path += '.yaml'

        # Build the YAML structure
        project_data = {
            "cameras": [],
            "ref_camera": self.video_infos[self.ref_idx]["name"]
        }

        for idx, (vw, info) in enumerate(zip(self.video_widgets, self.video_infos)):
            cam_data = {
                "name": info["name"],
                "path": info["path"],
                "sync_frame": vw.sync_frame,
                "fps": round(vw.fps, 2)
            }
            project_data["cameras"].append(cam_data)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(project_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            QMessageBox.information(self, "Success", f"Project saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project:\n{str(e)}")

def get_video_infos_from_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cameras = data.get("cameras", [])
    ref_camera_name = data.get("ref_camera", cameras[0]["name"] if cameras else None)
    video_infos = []
    ref_idx = 0
    for idx, cam in enumerate(cameras):
        path = cam["path"]

        # Try to get FPS from YAML first, then from video metadata
        if "fps" in cam and cam["fps"] is not None:
            fps = float(cam["fps"])
        else:
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            cap.release()

        sync_frame = parse_timecode(cam.get("sync_frame", 0), fps)
        video_infos.append({"path": path, "sync_frame": sync_frame, "fps": fps, "name": cam.get("name", f"cam{idx+1}")})
        if cam.get("name") == ref_camera_name:
            ref_idx = idx
    return video_infos, ref_idx

def main():
    app = QApplication(sys.argv)
    video_infos = []
    ref_idx = 0
    if len(sys.argv) < 2:
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Videos (*.mp4 *.avi *.mov *.mkv);;YAML (*.yaml *.yml)")
        if file_dialog.exec():
            selected = file_dialog.selectedFiles()
            if selected and (selected[0].endswith(".yaml") or selected[0].endswith(".yml")):
                video_infos, ref_idx = get_video_infos_from_yaml(selected[0])
            else:
                # No sync info, just open as plain videos
                video_infos = [{"path": p, "sync_frame": 0, "fps": 30, "name": f"cam{i+1}"} for i, p in enumerate(selected)]
        else:
            sys.exit(0)
    else:
        arg = sys.argv[1]
        if arg.endswith(".yaml") or arg.endswith(".yml"):
            video_infos, ref_idx = get_video_infos_from_yaml(arg)
        else:
            video_infos = [{"path": p, "sync_frame": 0, "fps": 30, "name": f"cam{i+1}"} for i, p in enumerate(sys.argv[1:])]
    if not video_infos:
        QMessageBox.critical(None, "Error", "No video files found.")
        sys.exit(1)
    window = MainWindow(video_infos, ref_idx=ref_idx)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

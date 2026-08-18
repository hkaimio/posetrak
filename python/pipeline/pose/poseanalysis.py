# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

# poseanalysis.py - Complete pose analysis toolkit

import traceback
import time
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from pathlib import Path

# UI Components
from ipycanvas import Canvas, hold_canvas
import ipywidgets as widgets
from IPython.display import display

# RTMLib imports
from rtmlib.tools.pose_estimation import RTMPose
from rtmlib.tools.object_detection import YOLOX
from rtmlib.visualization import draw

# YOLO tracker imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# PyAV — faster video decoding (hardware-accelerated when available)
try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    print("Warning: PyAV not available; falling back to OpenCV for video reads."
          " Install with: pip install av")

def _av_read_single_frame(video_path: str, frame_num_0based: int) -> Optional[np.ndarray]:
    """
    Read one frame (0-based index) from *video_path* using PyAV.
    Falls back to OpenCV when PyAV is not installed.
    Returns a BGR uint8 numpy array, or None on failure.
    """
    if not AV_AVAILABLE:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_0based)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        seek_ts = int(frame_num_0based / fps / float(stream.time_base))
        container.seek(seek_ts, stream=stream, backward=True, any_frame=False)
        for av_frame in container.decode(stream):
            if av_frame.pts is None:
                continue
            current = int(av_frame.pts * float(stream.time_base) * fps + 0.5)
            if current >= frame_num_0based:
                return av_frame.to_ndarray(format='bgr24')
    return None

import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json
import matplotlib.colors as mcolors

@dataclass
class DetectResult:
    """YOLO detection result for named person timeline"""
    x: float
    y: float
    w: float
    h: float
    confidence: float
    person_name: Optional[str] = None

@dataclass
class VideoPersonData:
    """Data for a single person in a single frame"""
    bbox: np.ndarray
    center: float
    scale: float
    keypoints: np.ndarray
    scores: np.ndarray
    simcc_x: np.ndarray
    simcc_y: np.ndarray
    person_name: Optional[str] = None  # Name of the person if using named persons
    is_manual: bool = False  # Flag to indicate if this was manually added


class NamedPersonTimeline:
    """
    Manages timeline for named persons using YOLO tracker

    This class tracks persons across frames using YOLO object tracking and allows
    assigning YOLO track IDs to named persons.
    """

    def __init__(self, named_persons: List[str], person_colors: Optional[Dict[str, str]] = None):
        """
        Initialize named person timeline manager

        Args:
            named_persons: List of person names to track (e.g., ["Harri", "Tommi", "Jani"])
            person_colors: Optional dict mapping person names to hex colors
        """
        self.named_persons = named_persons

        # Default colors if not provided
        if person_colors is None:
            default_colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1D3", "#F38181", "#AA96DA"]
            self.person_colors = {
                name: default_colors[i % len(default_colors)]
                for i, name in enumerate(named_persons)
            }
            self.person_colors[None] = "#95E1D3"  # Unassigned color
        else:
            self.person_colors = person_colors

        # Timeline: {person_name: {frame_id: yolo_track_id}}
        self.timelines = {name: {} for name in named_persons}

        # YOLO tracking data: {yolo_track_id: Person}
        self.yolo_persons = {}

        # Detection results: {yolo_track_id: {frame_id: DetectResult}}
        self.detections = {}

    def add_yolo_detection(self, frame_num: int, yolo_id: int, detect_result: DetectResult):
        """Add a YOLO detection result"""
        if yolo_id not in self.detections:
            self.detections[yolo_id] = {}
        self.detections[yolo_id][frame_num] = detect_result

    def assign_yolo_to_person(self, yolo_id: int, person_name: str, from_frame: int = 0):
        """
        Assign a YOLO track ID to a named person

        Args:
            yolo_id: YOLO track ID
            person_name: Name of the person
            from_frame: Start assigning from this frame onward
        """
        if person_name not in self.named_persons:
            raise ValueError(f"Unknown person name: {person_name}")

        # Remove this yolo_id from all other named persons (from_frame onward)
        for name in self.named_persons:
            if name != person_name:
                frames_to_remove = [f for f, yid in self.timelines[name].items()
                                  if yid == yolo_id and f >= from_frame]
                for frame in frames_to_remove:
                    del self.timelines[name][frame]
                    # Update DetectResult
                    if yolo_id in self.detections and frame in self.detections[yolo_id]:
                        self.detections[yolo_id][frame].person_name = None

        # Add frames for this yolo_id to the named person (from_frame onward)
        if yolo_id in self.detections:
            for frame_num in self.detections[yolo_id].keys():
                if frame_num >= from_frame:
                    self.timelines[person_name][frame_num] = yolo_id
                    self.detections[yolo_id][frame_num].person_name = person_name

    def get_person_bbox_for_frame(self, person_name: str, frame_num: int) -> Optional[np.ndarray]:
        """Get bounding box for a named person in a specific frame"""
        if frame_num not in self.timelines[person_name]:
            return None

        yolo_id = self.timelines[person_name][frame_num]
        if yolo_id not in self.detections or frame_num not in self.detections[yolo_id]:
            return None

        detect_result = self.detections[yolo_id][frame_num]
        # Convert xywh to x1y1x2y2
        x1 = detect_result.x - detect_result.w / 2
        y1 = detect_result.y - detect_result.h / 2
        x2 = detect_result.x + detect_result.w / 2
        y2 = detect_result.y + detect_result.h / 2

        return np.array([x1, y1, x2, y2])

    def get_persons_in_frame(self, frame_num: int) -> Dict[str, int]:
        """Get dict of {person_name: yolo_id} for all persons in a frame"""
        persons_in_frame = {}
        for person_name in self.named_persons:
            if frame_num in self.timelines[person_name]:
                persons_in_frame[person_name] = self.timelines[person_name][frame_num]
        return persons_in_frame

    def add_named_person_to_frame(self, frame_num: int, person_name: str) -> bool:
        """
        Add a named person to a specific frame using their timeline bbox

        Args:
            frame_num: Frame number
            person_name: Name of the person to add

        Returns:
            True if successfully added, False if person already in frame or no bbox available
        """
        if person_name not in self.named_persons:
            raise ValueError(f"Unknown person name: {person_name}")

        # Check if person already assigned in this frame
        if frame_num in self.timelines[person_name]:
            return False

        return True  # Person can be added

    def add_manual_detection(self, frame_num: int, person_name: str, bbox: np.ndarray):
        """
        Add a manual detection for a named person

        This creates a synthetic YOLO ID for manually added detections

        Args:
            frame_num: Frame number
            person_name: Name of the person
            bbox: Bounding box as [x1, y1, x2, y2]
        """
        if person_name not in self.named_persons:
            raise ValueError(f"Unknown person name: {person_name}")

        # Create a synthetic YOLO ID (negative numbers to distinguish from real YOLO IDs)
        # Use frame_num and person_name to create unique ID
        synthetic_yolo_id = -(1000000 + frame_num * 100 + self.named_persons.index(person_name))

        # Convert bbox from x1y1x2y2 to xywh (center format)
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Create DetectResult
        detect_result = DetectResult(
            x=x_center,
            y=y_center,
            w=w,
            h=h,
            confidence=1.0,  # Manual detections have confidence 1.0
            person_name=person_name
        )

        # Add to detections
        if synthetic_yolo_id not in self.detections:
            self.detections[synthetic_yolo_id] = {}
        self.detections[synthetic_yolo_id][frame_num] = detect_result

        # Add to timeline
        self.timelines[person_name][frame_num] = synthetic_yolo_id

        return synthetic_yolo_id

def analyze_video_with_yolo_tracker(
    video_path: str,
    named_persons: List[str],
    tracker_config: Optional[str] = None,
    person_colors: Optional[Dict[str, str]] = None,
    model_name: str = "yolo11x.pt",
    device: str = "cuda"
) -> Tuple[NamedPersonTimeline, int]:
    """
    Analyze video using YOLO tracker to detect and track persons

    Args:
        video_path: Path to video file
        named_persons: List of person names to track (e.g., ["Harri", "Tommi", "Jani"])
        tracker_config: Path to YOLO tracker config file (None for default)
        person_colors: Optional dict mapping person names to hex colors
        model_name: YOLO model name (default: "yolo11x.pt")
        device: Device to use ("cuda" or "cpu")

    Returns:
        Tuple of (NamedPersonTimeline, total_frames)

    Example:
        >>> timeline, total_frames = analyze_video_with_yolo_tracker(
        ...     "video.mp4",
        ...     ["Harri", "Tommi", "Jani"]
        ... )
        >>> # Later assign YOLO track IDs to persons
        >>> timeline.assign_yolo_to_person(yolo_id=1, person_name="Harri")
    """
    if not YOLO_AVAILABLE:
        raise ImportError("YOLO tracker requires 'ultralytics' package. Install with: pip install ultralytics")

    # Load YOLO model
    model = YOLO(model_name)

    # Initialize timeline manager
    timeline = NamedPersonTimeline(named_persons, person_colors)

    print(f"Analyzing video with YOLO tracker...")
    frame_number = 0

    # ------------------------------------------------------------------ #
    # Prefetch queue: a daemon thread decodes frames on the CPU while the #
    # main thread runs YOLO on the GPU, eliminating the serialization     #
    # that was causing ~35-40 % GPU utilisation.                          #
    # ------------------------------------------------------------------ #
    _QUEUE_MAXSIZE = 8   # tune: more = more RAM buffered ahead
    _SENTINEL = object() # signals end-of-stream
    _q: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)

    def _decode_worker():
        """Background thread: push decoded BGR arrays into _q."""
        try:
            if AV_AVAILABLE:
                with av.open(video_path) as ctx:
                    stream = ctx.streams.video[0]
                    stream.thread_type = 'AUTO'  # multi-threaded SW decode
                    for av_frame in ctx.decode(stream):
                        _q.put(av_frame.to_ndarray(format='bgr24'))
            else:
                cap = cv2.VideoCapture(video_path)
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    _q.put(f)
                cap.release()
        finally:
            _q.put(_SENTINEL)

    _decode_thread = threading.Thread(target=_decode_worker, daemon=True)
    _decode_thread.start()

    # Timing accumulators for the profiling breakdown
    t_queue_wait = 0.0
    t_inference  = 0.0
    t_total_start = time.perf_counter()

    while True:
        # How long does the main thread block waiting for the next frame?
        t0 = time.perf_counter()
        frame = _q.get()
        t_queue_wait += time.perf_counter() - t0

        if frame is _SENTINEL:
            break

        # Run YOLO tracking
        t1 = time.perf_counter()
        if tracker_config:
            results = model.track(frame, persist=True, tracker=tracker_config, device=device)
        else:
            results = model.track(frame, persist=True)
        t_inference += time.perf_counter() - t1

        if results[0].boxes is not None and results[0].boxes.id is not None:
            bboxes = results[0].boxes.xywh.cpu()
            conf = results[0].boxes.conf.cpu()
            ids = results[0].boxes.id.cpu()
            class_ids = results[0].boxes.cls.cpu()

            for cls_id, yolo_id, box, c in zip(class_ids, ids, bboxes, conf):
                if cls_id != 0:  # Only process person class (class ID 0)
                    continue

                x, y, w, h = box.tolist()
                detect_result = DetectResult(
                    x=x, y=y, w=w, h=h,
                    confidence=c.item(),
                    person_name=None
                )

                timeline.add_yolo_detection(frame_number, int(yolo_id.item()), detect_result)

        frame_number += 1
        if frame_number % 100 == 0:
            elapsed = time.perf_counter() - t_total_start
            pct_wait = 100 * t_queue_wait / elapsed if elapsed else 0
            pct_inf  = 100 * t_inference  / elapsed if elapsed else 0
            print(f"  Frame {frame_number:5d} | "
                  f"elapsed {elapsed:6.1f}s | "
                  f"decode-wait {t_queue_wait:5.1f}s ({pct_wait:.0f}%) | "
                  f"YOLO {t_inference:5.1f}s ({pct_inf:.0f}%)")

    _decode_thread.join()
    total_frames = frame_number
    elapsed_total = time.perf_counter() - t_total_start

    print(f"Completed! Processed {total_frames} frames in {elapsed_total:.1f}s")
    print(f"Timing breakdown (main thread):")
    print(f"  Waiting for decoded frame : {t_queue_wait:.2f}s "
          f"({100*t_queue_wait/elapsed_total:.0f}%)  <- decode bound if high")
    print(f"  YOLO inference            : {t_inference:.2f}s "
          f"({100*t_inference/elapsed_total:.0f}%)  <- GPU bound if high")
    print(f"  Other (tracking bookkeep) : "
          f"{elapsed_total - t_queue_wait - t_inference:.2f}s")
    print(f"Detected {len(timeline.detections)} unique person tracks")

    return timeline, total_frames


class NamedPersonStitcher:
    """
    Interactive UI for assigning YOLO track IDs to named persons

    This widget displays a timeline heatmap showing when each YOLO track ID
    is detected, and allows users to click on the timeline to assign track IDs
    to named persons.
    """

    def __init__(self, timeline: NamedPersonTimeline, video_path: str, total_frames: int):
        """
        Initialize the stitcher UI

        Args:
            timeline: NamedPersonTimeline object with YOLO detections
            video_path: Path to the video file
            total_frames: Total number of frames in the video
        """
        self.timeline = timeline
        self.video_path = video_path
        self.total_frames = total_frames

        # Get all YOLO track IDs
        self.yolo_ids = sorted(timeline.detections.keys())

        # Current selection
        self.current_selection = {'yolo_id': None, 'frame_idx': None}

        # Create matplotlib figure for the timeline
        self._create_timeline_figure()

        # Create control buttons
        self._create_controls()

        # Create main UI layout
        self.ui = widgets.VBox([
            widgets.HTML("<h2>Named Person Timeline Stitcher</h2>"),
            widgets.HTML("<p>Click on a YOLO track ID row to select it, then click a person button to assign.</p>"),
            self.output_widget,
            self.controls_box
        ])

    def _create_timeline_figure(self):
        """Create the matplotlib figure with timeline heatmap"""
        # Create presence matrix and color matrix
        presence_matrix = np.zeros((len(self.yolo_ids), self.total_frames))
        color_matrix = np.zeros((len(self.yolo_ids), self.total_frames, 4))  # RGBA

        # Fill matrices
        for yolo_idx, yolo_id in enumerate(self.yolo_ids):
            for frame_num in self.timeline.detections[yolo_id].keys():
                presence_matrix[yolo_idx, frame_num] = 1
                detect_result = self.timeline.detections[yolo_id][frame_num]
                person_name = detect_result.person_name
                hex_color = self.timeline.person_colors.get(person_name, self.timeline.person_colors[None])
                rgba = mcolors.to_rgba(hex_color)
                color_matrix[yolo_idx, frame_num] = rgba

        # Store matrices for updates
        self.presence_matrix = presence_matrix
        self.color_matrix = color_matrix

        # Create figure
        self.fig, (self.ax_timeline, self.ax_image) = plt.subplots(
            1, 2, figsize=(20, max(6, len(self.yolo_ids) * 0.5)),
            gridspec_kw={'width_ratios': [3, 1]}
        )

        # Plot timeline
        self.im = self.ax_timeline.imshow(
            color_matrix, aspect='auto', interpolation='nearest'
        )

        self.ax_timeline.set_xlabel('Frame Number', fontsize=12)
        self.ax_timeline.set_ylabel('YOLO Track ID', fontsize=12)
        self.ax_timeline.set_title(
            'YOLO Track Timeline (Click to select, then assign to person)',
            fontsize=14, fontweight='bold'
        )

        # Set y-axis ticks
        self.ax_timeline.set_yticks(range(len(self.yolo_ids)))
        self.ax_timeline.set_yticklabels([f'ID {yid}' for yid in self.yolo_ids])

        # Add grid
        self.ax_timeline.grid(which='major', axis='y', linestyle='-', linewidth=0.5, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.timeline.person_colors[name], label=name)
            for name in self.timeline.named_persons
        ]
        legend_elements.append(
            Patch(facecolor=self.timeline.person_colors[None], label='Unassigned')
        )
        self.ax_timeline.legend(handles=legend_elements, loc='upper right')

        # Setup image display area
        self.ax_image.set_title('Detection Image', fontsize=12)
        self.ax_image.axis('off')
        self.ax_image.text(
            0.5, 0.5, 'Click on timeline\nto view detection',
            ha='center', va='center', fontsize=12, transform=self.ax_image.transAxes
        )

        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_timeline_click)

        # Wrap in output widget
        self.output_widget = widgets.Output()
        with self.output_widget:
            plt.tight_layout()
            plt.show()

    def _create_controls(self):
        """Create control buttons for person assignment"""
        self.info_label = widgets.HTML(
            value="<b>Select a YOLO track ID by clicking on the timeline</b>"
        )

        # Create assignment buttons
        self.assign_buttons = {}
        self.from_frame_buttons = {}  # Changed from checkboxes to buttons

        button_rows = []
        for person_name in self.timeline.named_persons:
            color = self.timeline.person_colors[person_name]

            # Assign button (all frames)
            btn = widgets.Button(
                description=f'Assign to {person_name}',
                button_style='success',
                disabled=True,
                layout=widgets.Layout(width='150px')
            )
            btn.style.button_color = color
            btn.on_click(lambda b, name=person_name: self._assign_to_person(name, False))
            self.assign_buttons[person_name] = btn

            # Assign from frame button
            btn_from = widgets.Button(
                description=f'→ {person_name} →',
                button_style='warning',
                disabled=True,
                layout=widgets.Layout(width='100px'),
                tooltip='Assign from current frame onward'
            )
            btn_from.style.button_color = color
            btn_from.on_click(lambda b, name=person_name: self._assign_to_person(name, True))
            self.from_frame_buttons[person_name] = btn_from

            row = widgets.HBox([btn, btn_from])
            button_rows.append(row)

        self.controls_box = widgets.VBox([
            self.info_label,
            widgets.HTML("<b>Assignment Options:</b>"),
            *button_rows
        ])

    def _on_timeline_click(self, event):
        """Handle click on timeline"""
        if event.inaxes == self.ax_timeline and event.xdata is not None and event.ydata is not None:
            # Get clicked position
            frame_idx = int(round(event.xdata))
            yolo_idx = int(round(event.ydata))

            # Check validity
            if 0 <= yolo_idx < len(self.yolo_ids) and 0 <= frame_idx < self.total_frames:
                yolo_id = self.yolo_ids[yolo_idx]
                self.current_selection['yolo_id'] = yolo_id
                self.current_selection['frame_idx'] = frame_idx

                # Update info label
                self.info_label.value = (
                    f"<b>Selected: YOLO Track ID {yolo_id}, Frame {frame_idx}</b>"
                )

                # Enable ALL buttons (both assign and from-frame buttons)
                for btn in self.assign_buttons.values():
                    btn.disabled = False
                for btn in self.from_frame_buttons.values():
                    btn.disabled = False

                # Load and display detection image
                if yolo_id in self.timeline.detections and frame_idx in self.timeline.detections[yolo_id]:
                    self._display_detection_image(yolo_id, frame_idx)

    def _display_detection_image(self, yolo_id: int, frame_num: int):
        """Display the detection image for a specific YOLO ID and frame"""
        detect_result = self.timeline.detections[yolo_id][frame_num]

        # Load frame from video (frame_num is 0-based here)
        frame = _av_read_single_frame(self.video_path, frame_num)
        if frame is None:
            return

        # Calculate bbox coordinates
        x1 = int(detect_result.x - detect_result.w / 2)
        y1 = int(detect_result.y - detect_result.h / 2)
        x2 = int(detect_result.x + detect_result.w / 2)
        y2 = int(detect_result.y + detect_result.h / 2)

        # Crop detection area
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        detection_img = frame[y1:y2, x1:x2]

        # Convert BGR to RGB
        detection_img_rgb = cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB)

        # Update image display
        self.ax_image.clear()
        self.ax_image.axis('off')
        self.ax_image.imshow(detection_img_rgb)

        person_name_text = f" ({detect_result.person_name})" if detect_result.person_name else ""
        self.ax_image.set_title(
            f'YOLO ID: {yolo_id}{person_name_text} | Frame: {frame_num}\n'
            f'Confidence: {detect_result.confidence:.3f}',
            fontsize=10
        )

        self.fig.canvas.draw()

    def _assign_to_person(self, person_name: str, from_frame_onward: bool):
        """Assign selected YOLO track ID to a named person"""
        if self.current_selection['yolo_id'] is None:
            return

        yolo_id = self.current_selection['yolo_id']
        frame_idx = self.current_selection['frame_idx'] if from_frame_onward else 0

        # Before assigning, track which other YOLO IDs need to be updated
        # (those that were previously assigned to this person)
        yolo_ids_to_update = set()
        for other_yolo_id, frames_dict in self.timeline.detections.items():
            for frame_num, detect_result in frames_dict.items():
                if detect_result.person_name == person_name and frame_num >= frame_idx:
                    yolo_ids_to_update.add(other_yolo_id)

        # Assign in timeline
        self.timeline.assign_yolo_to_person(yolo_id, person_name, from_frame=frame_idx)

        # Update color matrix for the selected YOLO ID
        for frame_num in self.timeline.detections[yolo_id].keys():
            if frame_num >= frame_idx:
                detect_result = self.timeline.detections[yolo_id][frame_num]
                hex_color = self.timeline.person_colors[detect_result.person_name]
                rgba = mcolors.to_rgba(hex_color)
                yolo_idx = self.yolo_ids.index(yolo_id)
                self.color_matrix[yolo_idx, frame_num] = rgba

        # Update color matrix for other YOLO IDs that were previously assigned to this person
        for other_yolo_id in yolo_ids_to_update:
            if other_yolo_id != yolo_id and other_yolo_id in self.timeline.detections:
                other_yolo_idx = self.yolo_ids.index(other_yolo_id)
                for frame_num in self.timeline.detections[other_yolo_id].keys():
                    if frame_num >= frame_idx:
                        detect_result = self.timeline.detections[other_yolo_id][frame_num]
                        # Get the color (might be None now if unassigned)
                        hex_color = self.timeline.person_colors.get(
                            detect_result.person_name,
                            self.timeline.person_colors[None]
                        )
                        rgba = mcolors.to_rgba(hex_color)
                        self.color_matrix[other_yolo_idx, frame_num] = rgba

        # Update display
        self._update_timeline_display()

        # Update info
        mode_text = f"from frame {frame_idx} onward" if from_frame_onward else "for all frames"
        self.info_label.value = (
            f"<b>Assigned YOLO ID {yolo_id} to {person_name} {mode_text}</b>"
        )

    def _update_timeline_display(self):
        """Redraw the timeline with updated colors"""
        self.ax_timeline.clear()
        self.ax_timeline.imshow(self.color_matrix, aspect='auto', interpolation='nearest')

        self.ax_timeline.set_xlabel('Frame Number', fontsize=12)
        self.ax_timeline.set_ylabel('YOLO Track ID', fontsize=12)
        self.ax_timeline.set_title(
            'YOLO Track Timeline (Click to select, then assign to person)',
            fontsize=14, fontweight='bold'
        )

        self.ax_timeline.set_yticks(range(len(self.yolo_ids)))
        self.ax_timeline.set_yticklabels([f'ID {yid}' for yid in self.yolo_ids])
        self.ax_timeline.grid(which='major', axis='y', linestyle='-', linewidth=0.5, alpha=0.3)

        # Re-add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.timeline.person_colors[name], label=name)
            for name in self.timeline.named_persons
        ]
        legend_elements.append(
            Patch(facecolor=self.timeline.person_colors[None], label='Unassigned')
        )
        self.ax_timeline.legend(handles=legend_elements, loc='upper right')

        self.fig.canvas.draw()

    def display(self):
        """Display the stitcher UI"""
        return self.ui

    def get_summary(self) -> Dict:
        """Get summary of current assignments"""
        summary = {
            'total_yolo_tracks': len(self.yolo_ids),
            'assignments': {}
        }

        for person_name in self.timeline.named_persons:
            frame_count = len(self.timeline.timelines[person_name])
            yolo_ids_used = set(self.timeline.timelines[person_name].values())
            summary['assignments'][person_name] = {
                'frame_count': frame_count,
                'yolo_ids': sorted(list(yolo_ids_used))
            }

        return summary


def create_named_person_stitcher(
    video_path: str,
    named_persons: List[str],
    tracker_config: Optional[str] = None,
    person_colors: Optional[Dict[str, str]] = None,
    yolo_model: str = "yolo11x.pt",
    device: str = "cuda"
) -> NamedPersonStitcher:
    """
    Create an interactive stitcher for assigning YOLO track IDs to named persons

    This is Step 2 in the named person workflow:
    1. analyze_video_with_yolo_tracker() - detects and tracks persons
    2. create_named_person_stitcher() - interactive UI to assign track IDs to names
    3. analyze_video_with_named_persons() - compute poses for named persons

    Args:
        video_path: Path to video file
        named_persons: List of person names (e.g., ["Harri", "Tommi", "Jani"])
        tracker_config: Optional YOLO tracker config file
        person_colors: Optional dict mapping person names to hex colors
        yolo_model: YOLO model name (default: "yolo11x.pt")
        device: Device to use ("cuda" or "cpu")

    Returns:
        NamedPersonStitcher widget for interactive assignment

    Example:
        >>> stitcher = create_named_person_stitcher(
        ...     "video.mp4",
        ...     ["Harri", "Tommi", "Jani"]
        ... )
        >>> display(stitcher.display())
        >>> # After making assignments in the UI:
        >>> summary = stitcher.get_summary()
        >>> print(summary)
        >>> # Then continue with analyze_video_with_named_persons()
    """
    # Step 1: Run YOLO tracker analysis
    timeline, total_frames = analyze_video_with_yolo_tracker(
        video_path, named_persons, tracker_config, person_colors, yolo_model, device
    )

    # Step 2: Create stitcher UI
    stitcher = NamedPersonStitcher(timeline, video_path, total_frames)

    print(f"Found {len(timeline.detections)} YOLO track IDs across {total_frames} frames")
    print("Use the interactive UI to assign track IDs to named persons")

    return stitcher


class VideoData:
    """Video-wide pose analysis data with optional named person support"""

    def __init__(
        self,
        video_path: str,
        det_model,
        pose_model,
        start_frame: int = 1,
        end_frame: Optional[int] = None,
        named_person_timeline: Optional[NamedPersonTimeline] = None
    ):
        """
        Initialize video data by detecting persons and poses in all frames

        Args:
            video_path: Path to video file or frame directory
            det_model: Object detection model (YOLOX)
            pose_model: Pose estimation model (RTMPose)
            start_frame: First frame to process
            end_frame: Last frame to process (None for all frames)
            named_person_timeline: Optional NamedPersonTimeline for using named persons
        """
        self.video_path = Path(video_path)
        self.det_model = det_model
        self.pose_model = pose_model
        self.start_frame = start_frame
        self.named_person_timeline = named_person_timeline
        self.use_named_persons = named_person_timeline is not None

        # Dictionary to store person data
        if self.use_named_persons:
            # {frame_num: {person_name: VideoPersonData}}
            self.frame_data: Dict[int, Dict[str, VideoPersonData]] = {}
        else:
            # {frame_num: [VideoPersonData, ...]}
            self.frame_data: Dict[int, List[VideoPersonData]] = {}

        # Detect if we're working with video file or image sequence
        if self.video_path.is_file():
            self._load_from_video(end_frame)
        else:
            self._load_from_images(end_frame)

    def _get_frame_path(self, frame_num: int) -> Path:
        """Get path for a specific frame (for image sequences)"""
        return self.video_path / f"frame-{frame_num:04d}.jpg"

    def _load_from_images(self, end_frame: Optional[int] = None):
        """Load pose data from image sequence"""
        frame_num = self.start_frame

        while True:
            frame_path = self._get_frame_path(frame_num)
            if not frame_path.exists():
                break

            if end_frame and frame_num > end_frame:
                break

            print(f"Processing frame {frame_num}")
            img = cv2.imread(str(frame_path))
            if img is None:
                break

            self._process_frame(frame_num, img)
            frame_num += 1

    def _load_from_video(self, end_frame: Optional[int] = None):
        """Load pose data from video file using PyAV for faster (hardware-accelerated) decoding."""
        if not AV_AVAILABLE:
            # OpenCV fallback
            cap = cv2.VideoCapture(str(self.video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame - 1)
            frame_num = self.start_frame
            while True:
                ret, img = cap.read()
                if not ret:
                    break
                if end_frame and frame_num > end_frame:
                    break
                print(f"Processing frame {frame_num}")
                self._process_frame(frame_num, img)
                frame_num += 1
            cap.release()
            return

        with av.open(str(self.video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'  # multi-threaded software decode
            fps = float(stream.average_rate)

            # Seek to just before the desired start frame (0-based in OpenCV ↔ 1-based here)
            if self.start_frame > 1:
                seek_ts = int((self.start_frame - 1) / fps / float(stream.time_base))
                container.seek(seek_ts, stream=stream, backward=True, any_frame=False)

            for av_frame in container.decode(stream):
                if av_frame.pts is None:
                    continue
                # Convert PTS to 1-based frame number matching the original convention
                frame_num = int(av_frame.pts * float(stream.time_base) * fps + 0.5) + 1
                if frame_num < self.start_frame:
                    continue
                if end_frame and frame_num > end_frame:
                    break

                img = av_frame.to_ndarray(format='bgr24')
                print(f"Processing frame {frame_num}")
                self._process_frame(frame_num, img)

    def _process_frame(self, frame_num: int, img: np.ndarray):
        """Process a single frame to detect persons and estimate poses"""
        if self.use_named_persons:
            # Use named person timeline for bounding boxes
            persons_in_frame = self.named_person_timeline.get_persons_in_frame(frame_num)
            person_data_dict = {}

            # Process each named person that was assigned in this frame
            for person_name, yolo_id in persons_in_frame.items():
                bbox = self.named_person_timeline.get_person_bbox_for_frame(person_name, frame_num)
                if bbox is not None:
                    try:
                        preproc_img, center, scale = self.pose_model.preprocess(img, bbox)
                        outputs = self.pose_model.inference(preproc_img)
                        keypoints, scores = self.pose_model.postprocess(outputs, center, scale)

                        person_data = VideoPersonData(
                            bbox=bbox.astype(np.float64),
                            center=center.astype(np.float64),
                            scale=scale.astype(np.float64),
                            keypoints=keypoints.astype(np.float64),
                            scores=scores.astype(np.float64),
                            simcc_x=outputs[0].astype(np.float64),
                            simcc_y=outputs[1].astype(np.float64),
                            person_name=person_name
                        )
                        person_data_dict[person_name] = person_data
                    except Exception as e:
                        print(f"  Warning: Failed to process {person_name} in frame {frame_num}: {e}")

            self.frame_data[frame_num] = person_data_dict
            print(f"Frame {frame_num}: {len(person_data_dict)} named persons processed")
        else:
            # Use object detection for bounding boxes
            bboxes = self.det_model(img)
            persons = []

            for bbox in bboxes:
                preproc_img, center, scale = self.pose_model.preprocess(img, bbox[:4])
                outputs = self.pose_model.inference(preproc_img)
                keypoints, scores = self.pose_model.postprocess(outputs, center, scale)

                person_data = VideoPersonData(
                    bbox=bbox.astype(np.float64),
                    center=center.astype(np.float64),
                    scale=scale.astype(np.float64),
                    keypoints=keypoints.astype(np.float64),
                    scores=scores.astype(np.float64),
                    simcc_x=outputs[0].astype(np.float64),
                    simcc_y=outputs[1].astype(np.float64)
                )
                persons.append(person_data)

            self.frame_data[frame_num] = persons
            print(f"Frame {frame_num}: {len(persons)} persons detected")

    def get_frame_count(self) -> int:
        """Get total number of processed frames"""
        return len(self.frame_data)

    def get_frame_numbers(self) -> List[int]:
        """Get list of all frame numbers"""
        return sorted(self.frame_data.keys())

    def get_person_count(self, frame_num: int) -> int:
        """Get number of persons in a specific frame"""
        frame_persons = self.frame_data.get(frame_num, {} if self.use_named_persons else [])
        return len(frame_persons)

    def get_person_names_in_frame(self, frame_num: int) -> List[str]:
        """Get list of person names in a specific frame (for named person mode)"""
        if not self.use_named_persons:
            return []
        return list(self.frame_data.get(frame_num, {}).keys())

    def _load_frame_image(self, frame_num: int) -> np.ndarray:
        """Load the original image for a frame (1-based frame_num)."""
        if self.video_path.is_file():
            # frame_num is 1-based; _av_read_single_frame expects 0-based
            img = _av_read_single_frame(str(self.video_path), frame_num - 1)
            if img is None:
                raise ValueError(f"Could not read frame {frame_num}")
            return img
        else:
            frame_path = self._get_frame_path(frame_num)
            img = cv2.imread(str(frame_path))
            if img is None:
                raise ValueError(f"Could not read frame {frame_num} from {frame_path}")
            return img

    def draw_frame(self, frame_num: int, thr: float = 0.5, persons: Optional[List] = None) -> np.ndarray:
        """
        Draw frame with pose skeletons for specified persons

        Args:
            frame_num: Frame number to draw
            thr: Keypoint confidence threshold
            persons: List of person indices/names to draw (None for all)

        Returns:
            Image with drawn skeletons
        """
        if frame_num not in self.frame_data:
            raise ValueError(f"Frame {frame_num} not found")

        img = self._load_frame_image(frame_num)
        frame_persons = self.frame_data[frame_num]

        if not frame_persons:
            return img

        if self.use_named_persons:
            # Named persons mode
            if persons is None:
                persons = list(frame_persons.keys())

            kps = np.vstack([frame_persons[name].keypoints for name in persons if name in frame_persons])
            scores = np.vstack([frame_persons[name].scores for name in persons if name in frame_persons])
        else:
            # Index-based mode
            if persons is None:
                persons = list(range(len(frame_persons)))

            kps = np.vstack([frame_persons[i].keypoints for i in persons])
            scores = np.vstack([frame_persons[i].scores for i in persons])

        return draw.draw_skeleton(img, kps, scores, kpt_thr=thr)

    def draw_person(self, frame_num: int, person_identifier, thr: float = 0.5) -> np.ndarray:
        """
        Draw cropped image of a specific person with pose skeleton

        Args:
            frame_num: Frame number
            person_identifier: Person index (int) or name (str) depending on mode
            thr: Keypoint confidence threshold

        Returns:
            Cropped image with person's pose skeleton
        """
        if frame_num not in self.frame_data:
            raise ValueError(f"Frame {frame_num} not found")

        frame_persons = self.frame_data[frame_num]

        if self.use_named_persons:
            # person_identifier is a name (str)
            if person_identifier not in frame_persons:
                raise ValueError(f"Person {person_identifier} not found in frame {frame_num}")
            person = frame_persons[person_identifier]
            person_list = [person_identifier]
        else:
            # person_identifier is an index (int)
            if person_identifier >= len(frame_persons):
                raise ValueError(f"Person {person_identifier} not found in frame {frame_num}")
            person = frame_persons[person_identifier]
            person_list = [person_identifier]

        # Draw the full frame with just this person
        full_img = self.draw_frame(frame_num, thr, persons=person_list)

        # Crop to person's bounding box
        bbox = person.bbox
        x1, y1, x2, y2 = bbox[:4].astype(int)
        return full_img[y1:y2, x1:x2, :]

    def update_person_bbox(self, frame_num: int, person_idx: int, x1: float, y1: float, x2: float, y2: float):
        """
        Update bounding box for a person and recompute pose

        Args:
            frame_num: Frame number
            person_idx: Person index to update
            x1, y1, x2, y2: New bounding box coordinates
        """
        if frame_num not in self.frame_data:
            raise ValueError(f"Frame {frame_num} not found")

        frame_persons = self.frame_data[frame_num]
        if person_idx >= len(frame_persons):
            raise ValueError(f"Person {person_idx} not found in frame {frame_num}")

        # Load original frame image
        img = self._load_frame_image(frame_num)

        # Update person data
        person = frame_persons[person_idx]
        person.bbox = np.array([x1, y1, x2, y2])

        # Recompute pose with new bbox
        preproc_img, center, scale = self.pose_model.preprocess(img, person.bbox[:4])
        outputs = self.pose_model.inference(preproc_img)
        keypoints, scores = self.pose_model.postprocess(outputs, center, scale)

        # Update stored data
        person.center = center
        person.scale = scale
        person.keypoints = keypoints
        person.scores = scores
        person.simcc_x = outputs[0]
        person.simcc_y = outputs[1]

    def add_person(self, frame_num: int, x1: float, y1: float, x2: float, y2: float) -> int:
        """
        Add a new person to a frame

        Args:
            frame_num: Frame number
            x1, y1, x2, y2: Bounding box coordinates

        Returns:
            Index of the newly added person
        """
        if frame_num not in self.frame_data:
            self.frame_data[frame_num] = []

        # Load original frame image
        img = self._load_frame_image(frame_num)

        # Create new person data
        bbox = np.array([x1, y1, x2, y2])
        preproc_img, center, scale = self.pose_model.preprocess(img, bbox[:4])
        outputs = self.pose_model.inference(preproc_img)
        keypoints, scores = self.pose_model.postprocess(outputs, center, scale)

        new_person = VideoPersonData(
            bbox=bbox,
            center=center,
            scale=scale,
            keypoints=keypoints,
            scores=scores,
            simcc_x=outputs[0],
            simcc_y=outputs[1]
        )

        self.frame_data[frame_num].append(new_person)
        return len(self.frame_data[frame_num]) - 1

    def remove_person(self, frame_num: int, person_idx: int):
        """Remove a person from a frame"""
        if frame_num not in self.frame_data:
            raise ValueError(f"Frame {frame_num} not found")

        frame_persons = self.frame_data[frame_num]
        if person_idx >= len(frame_persons):
            raise ValueError(f"Person {person_idx} not found in frame {frame_num}")

        del frame_persons[person_idx]

    def get_person_data(self, frame_num: int, person_idx: int) -> VideoPersonData:
        """Get person data for a specific frame and person"""
        if frame_num not in self.frame_data:
            raise ValueError(f"Frame {frame_num} not found")

        frame_persons = self.frame_data[frame_num]
        if person_idx >= len(frame_persons):
            raise ValueError(f"Person {person_idx} not found in frame {frame_num}")

        return frame_persons[person_idx]

    def add_named_person(self, frame_num: int, person_name: str, x1: float, y1: float, x2: float, y2: float):
        """
        Add a named person to a frame (for named person mode)

        Args:
            frame_num: Frame number
            person_name: Name of the person
            x1, y1, x2, y2: Bounding box coordinates

        Returns:
            The person name
        """
        if not self.use_named_persons:
            raise ValueError("add_named_person only works in named person mode")

        if frame_num not in self.frame_data:
            self.frame_data[frame_num] = {}

        # Load original frame image
        img = self._load_frame_image(frame_num)

        # Create new person data
        bbox = np.array([x1, y1, x2, y2])
        preproc_img, center, scale = self.pose_model.preprocess(img, bbox[:4])
        outputs = self.pose_model.inference(preproc_img)
        keypoints, scores = self.pose_model.postprocess(outputs, center, scale)

        new_person = VideoPersonData(
            bbox=bbox.astype(np.float64),
            center=center.astype(np.float64),
            scale=scale.astype(np.float64),
            keypoints=keypoints.astype(np.float64),
            scores=scores.astype(np.float64),
            simcc_x=outputs[0].astype(np.float64),
            simcc_y=outputs[1].astype(np.float64),
            person_name=person_name,
            is_manual=True
        )

        self.frame_data[frame_num][person_name] = new_person

        # Also add to the timeline if it exists
        if self.named_person_timeline is not None:
            self.named_person_timeline.add_manual_detection(frame_num, person_name, bbox)

        return person_name

    def extract_person_data_array(self, person_identifier) -> tuple[np.ndarray, dict]:
        """
        Extract data for a specific person across all frames as a numpy array

        Args:
            person_identifier: Person index (int) or name (str) depending on mode

        Returns:
            tuple containing:
            - data_array: numpy array with shape (n_frames, n_columns) where each row is a frame
            - metadata: dict with video_path, person_identifier, and column_descriptions

        Note:
            If a person is not present in a frame, that row will be filled with NaN values
        """
        frame_numbers = self.get_frame_numbers()

        # Determine array dimensions by checking the first available person data
        sample_person = None

        if self.use_named_persons:
            # person_identifier should be a string (person name)
            person_name = person_identifier
            for frame_num in frame_numbers:
                frame_persons = self.frame_data.get(frame_num, {})
                if person_name in frame_persons:
                    sample_person = frame_persons[person_name]
                    break
        else:
            # person_identifier should be an int (person index)
            person_idx = person_identifier
            for frame_num in frame_numbers:
                frame_persons = self.frame_data.get(frame_num, [])
                if len(frame_persons) > person_idx:
                    sample_person = frame_persons[person_idx]
                    break

        if sample_person is None:
            raise ValueError(f"Person {person_identifier} not found in any frame")

        # Handle keypoints - squeeze out extra dimensions and flatten
        keypoints = sample_person.keypoints.squeeze()
        scores = sample_person.scores.squeeze()

        # Now handle the keypoints properly
        if len(keypoints.shape) == 2:
            # Shape is (n_keypoints, 2)
            n_keypoints = keypoints.shape[0]
            keypoints_flat_len = n_keypoints * 2
        else:
            # Already flattened
            keypoints_flat_len = len(keypoints)
            n_keypoints = keypoints_flat_len // 2

        n_scores = len(scores)
        n_cols = 4 + keypoints_flat_len + n_scores  # bbox(4) + keypoints_xy + scores
        n_frames = len(frame_numbers)

        # Initialize array with NaN
        data_array = np.full((n_frames, n_cols), np.nan)

        # Fill array with person data
        for i, frame_num in enumerate(frame_numbers):
            person = None

            if self.use_named_persons:
                # Named person mode
                frame_persons = self.frame_data.get(frame_num, {})
                person = frame_persons.get(person_name)
            else:
                # Index mode
                frame_persons = self.frame_data.get(frame_num, [])
                if len(frame_persons) > person_idx:
                    person = frame_persons[person_idx]

            if person is not None:
                col_idx = 0

                # Bounding box coordinates (4 values)
                data_array[i, col_idx:col_idx+4] = person.bbox[:4]
                col_idx += 4

                # Keypoint coordinates - squeeze and flatten
                keypoints = person.keypoints.squeeze()
                if len(keypoints.shape) == 2:
                    keypoints_flat = keypoints.flatten()
                else:
                    keypoints_flat = keypoints

                data_array[i, col_idx:col_idx+len(keypoints_flat)] = keypoints_flat
                col_idx += len(keypoints_flat)

                # Keypoint scores - squeeze
                scores = person.scores.squeeze()
                data_array[i, col_idx:col_idx+len(scores)] = scores

        # Create column descriptions
        column_descriptions = []

        # Bounding box columns
        column_descriptions.extend(['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])

        # Keypoint coordinate columns
        for kp_idx in range(n_keypoints):
            column_descriptions.extend([f'keypoint_{kp_idx}_x', f'keypoint_{kp_idx}_y'])

        # Keypoint score columns
        for kp_idx in range(n_scores):
            column_descriptions.append(f'keypoint_{kp_idx}_score')

        # Create metadata
        metadata = {
            'video_path': str(self.video_path),
            'person_identifier': person_identifier,
            'person_name': person_name if self.use_named_persons else None,
            'person_index': person_idx if not self.use_named_persons else None,
            'frame_numbers': frame_numbers,
            'n_frames': n_frames,
            'n_keypoints': n_keypoints,
            'n_scores': n_scores,
            'column_descriptions': column_descriptions,
            'array_shape': data_array.shape,
            'missing_frames_info': 'NaN values indicate frames where person was not detected'
        }

        return data_array, metadata

    def get_all_person_identifiers(self) -> List:
        """
        Get list of all person identifiers (names or indices depending on mode)

        Returns:
            List of person identifiers that can be used with extract_person_data_array()
        """
        if self.use_named_persons:
            # Return all unique person names across all frames
            all_names = set()
            for frame_data in self.frame_data.values():
                all_names.update(frame_data.keys())
            return sorted(list(all_names))
        else:
            # Find maximum number of persons in any frame
            max_persons = 0
            for frame_data in self.frame_data.values():
                max_persons = max(max_persons, len(frame_data))
            return list(range(max_persons))


class MultiVideoPoseDataset:
    """
    HDF5-based storage for pose data from multiple videos

    Structure:
    dataset.h5
    ├── video_0/
    │   ├── metadata (attributes)
    │   ├── person_0/data (array)
    │   ├── person_0/metadata (attributes)
    │   └── person_1/...
    └── global_metadata (attributes)
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def save_video_data(self, video_data, video_name: str, max_persons: Optional[int] = None):
        """
        Save pose data from a VideoData object to HDF5 file

        Args:
            video_data: VideoData object containing pose analysis
            video_name: Unique name for this video (used as group name)
            max_persons: Maximum number of persons to save (None for all)
        """
        with h5py.File(self.filepath, 'a') as f:  # 'a' = append/create
            # Create video group (remove if exists)
            if video_name in f:
                del f[video_name]

            video_group = f.create_group(video_name)

            # Save video metadata
            video_group.attrs['video_path'] = str(video_data.video_path)
            video_group.attrs['n_frames'] = video_data.get_frame_count()
            video_group.attrs['frame_numbers'] = video_data.get_frame_numbers()
            video_group.attrs['use_named_persons'] = video_data.use_named_persons

            # Get all person identifiers
            all_person_ids = video_data.get_all_person_identifiers()

            # Limit to max_persons if specified
            if max_persons is not None:
                all_person_ids = all_person_ids[:max_persons]

            video_group.attrs['n_persons'] = len(all_person_ids)

            print(f"Saving {len(all_person_ids)} persons from video '{video_name}'")

            if video_data.use_named_persons:
                # Save person names for reference
                video_group.attrs['person_names'] = json.dumps(all_person_ids)

            # Save data for each person
            for idx, person_id in enumerate(all_person_ids):
                try:
                    data_array, metadata = video_data.extract_person_data_array(person_id)

                    # Create person group
                    person_group = video_group.create_group(f'person_{idx}')

                    # Save person data array
                    person_group.create_dataset('data', data=data_array,
                                              compression='gzip', compression_opts=9)

                    # Save person metadata as attributes
                    if video_data.use_named_persons:
                        person_group.attrs['person_name'] = person_id
                        person_group.attrs['person_index'] = idx
                    else:
                        person_group.attrs['person_index'] = person_id

                    person_group.attrs['array_shape'] = metadata['array_shape']
                    person_group.attrs['n_keypoints'] = metadata['n_keypoints']
                    person_group.attrs['n_scores'] = metadata['n_scores']

                    # Save column descriptions as JSON string
                    person_group.attrs['column_descriptions'] = json.dumps(metadata['column_descriptions'])

                    if video_data.use_named_persons:
                        print(f"   {person_id}: {data_array.shape}")
                    else:
                        print(f"   Person {person_id}: {data_array.shape}")

                except Exception as e:
                    print(f"  Person {person_id}: Error - {e}")
                    traceback.print_exc()
                    continue

            # Update global metadata
            if 'global_metadata' not in f:
                global_group = f.create_group('global_metadata')
                global_group.attrs['dataset_description'] = 'Multi-video pose analysis dataset'
                global_group.attrs['format_version'] = '1.0'
                global_group.attrs['video_count'] = 0

            global_group = f['global_metadata']
            global_group.attrs['video_count'] = len([key for key in f.keys() if key != 'global_metadata'])

    def load_video_list(self) -> List[str]:
        """Get list of all video names in the dataset"""
        with h5py.File(self.filepath, 'r') as f:
            return [key for key in f.keys() if key != 'global_metadata']

    def load_video_data(self, video_name: str, person_idx: Optional[int] = None) -> Dict:
        """
        Load data for a specific video and optionally a specific person

        Args:
            video_name: Name of the video to load
            person_idx: Specific person to load (None for all persons)

        Returns:
            Dictionary with video data and metadata
        """
        with h5py.File(self.filepath, 'r') as f:
            if video_name not in f:
                raise ValueError(f"Video '{video_name}' not found in dataset")

            video_group = f[video_name]

            # Load video metadata
            result = {
                'video_name': video_name,
                'video_path': video_group.attrs['video_path'],
                'n_frames': video_group.attrs['n_frames'],
                'frame_numbers': video_group.attrs['frame_numbers'].tolist(),
                'n_persons': video_group.attrs['n_persons'],
                'persons': {}
            }

            # Load person data
            if person_idx is not None:
                # Load specific person
                person_key = f'person_{person_idx}'
                if person_key in video_group:
                    person_group = video_group[person_key]
                    result['persons'][person_idx] = {
                        'data': person_group['data'][:],  # Load array
                        'person_index': person_group.attrs['person_index'],
                        'array_shape': person_group.attrs['array_shape'],
                        'n_keypoints': person_group.attrs['n_keypoints'],
                        'n_scores': person_group.attrs['n_scores'],
                        'column_descriptions': json.loads(person_group.attrs['column_descriptions'])
                    }
            else:
                # Load all persons
                for person_key in video_group.keys():
                    if person_key.startswith('person_'):
                        person_idx = int(person_key.split('_')[1])
                        person_group = video_group[person_key]
                        result['persons'][person_idx] = {
                            'data': person_group['data'][:],  # Load array
                            'person_index': person_group.attrs['person_index'],
                            'array_shape': person_group.attrs['array_shape'],
                            'n_keypoints': person_group.attrs['n_keypoints'],
                            'n_scores': person_group.attrs['n_scores'],
                            'column_descriptions': json.loads(person_group.attrs['column_descriptions'])
                        }

            return result

    def get_dataset_info(self) -> Dict:
        """Get overall dataset information"""
        with h5py.File(self.filepath, 'r') as f:
            if 'global_metadata' not in f:
                return {'error': 'No global metadata found'}

            global_group = f['global_metadata']
            return {
                'filepath': str(self.filepath),
                'video_count': global_group.attrs['video_count'],
                'format_version': global_group.attrs['format_version'],
                'dataset_description': global_group.attrs['dataset_description'],
                'video_names': self.load_video_list(),
                'file_size_mb': self.filepath.stat().st_size / (1024*1024) if self.filepath.exists() else 0
            }

    def export_to_openpose_json(self, output_dir: str, video_name: Optional[str] = None):
        """
        Export pose data to OpenPose JSON format

        Args:
            output_dir: Directory where JSON files will be saved
            video_name: Specific video to export (None for all videos)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        video_names = [video_name] if video_name else self.load_video_list()

        for vid_name in video_names:
            print(f"Exporting video: {vid_name}")
            self._export_video_to_json(vid_name, output_path)

    def _export_video_to_json(self, video_name: str, output_path: Path):
        """Export a single video to OpenPose JSON format"""
        video_data = self.load_video_data(video_name)
        frame_numbers = video_data['frame_numbers']

        # Create video-specific directory
        video_dir = output_path / f"{video_name}_json"
        video_dir.mkdir(exist_ok=True)

        for frame_idx, frame_num in enumerate(frame_numbers):
            # Create JSON structure for this frame
            frame_json = {
                "version": 1.3,
                "people": []
            }

            # Process each person in this frame
            for person_idx, person_data in video_data['persons'].items():
                person_array = person_data['data']

                # Check if this person has data for this frame (non-NaN bbox)
                if frame_idx < len(person_array) and not np.isnan(person_array[frame_idx, 0]):
                    person_frame_data = person_array[frame_idx]

                    # Extract keypoints and scores
                    pose_keypoints_2d = self._extract_openpose_keypoints(
                        person_frame_data, person_data['n_keypoints']
                    )

                    person_json = {
                        "person_id": [person_idx],
                        "pose_keypoints_2d": pose_keypoints_2d,
                        "face_keypoints_2d": [],
                        "hand_left_keypoints_2d": [],
                        "hand_right_keypoints_2d": [],
                        "pose_keypoints_3d": [],
                        "face_keypoints_3d": [],
                        "hand_left_keypoints_3d": [],
                        "hand_right_keypoints_3d": []
                    }

                    frame_json["people"].append(person_json)

            # Save frame JSON
            json_filename = f"{video_name}_{frame_idx:06d}.json"
            json_path = video_dir / json_filename

            with open(json_path, 'w') as f:
                json.dump(frame_json, f, indent=4)

        print(f"  Exported {len(frame_numbers)} frames to {video_dir}")

    def _extract_openpose_keypoints(self, person_frame_data: np.ndarray, n_keypoints: int) -> List[float]:
        """
        Extract keypoints in OpenPose format (x, y, confidence) for each keypoint

        Args:
            person_frame_data: Single frame data for one person
            n_keypoints: Number of keypoints

        Returns:
            List of [x, y, confidence] values flattened
        """
        # Data layout: [bbox(4), keypoints_xy(n*2), scores(n)]
        keypoints_start = 4
        keypoints_end = 4 + (n_keypoints * 2)
        scores_start = keypoints_end
        scores_end = scores_start + n_keypoints

        # Extract keypoint coordinates (x, y pairs)
        keypoints_xy = person_frame_data[keypoints_start:keypoints_end]
        # Extract scores
        scores = person_frame_data[scores_start:scores_end]

        # Reshape keypoints to (n_keypoints, 2)
        keypoints_reshaped = keypoints_xy.reshape(n_keypoints, 2)

        # Create OpenPose format: [x1, y1, conf1, x2, y2, conf2, ...]
        openpose_keypoints = []
        for i in range(n_keypoints):
            x, y = keypoints_reshaped[i]
            confidence = scores[i]
            openpose_keypoints.extend([float(x), float(y), float(confidence)])

        return openpose_keypoints

    def export_video_to_openpose_json(self, video_name: str, output_dir: str):
        """
        Export a specific video to OpenPose JSON format

        Args:
            video_name: Name of the video to export
            output_dir: Directory where JSON files will be saved
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Exporting video '{video_name}' to OpenPose JSON format...")
        self._export_video_to_json(video_name, output_path)
        print("Export completed!")

    def get_export_summary(self, video_name: str) -> Dict:
        """
        Get summary information for exporting a video

        Args:
            video_name: Name of the video

        Returns:
            Dictionary with export summary information
        """
        video_data = self.load_video_data(video_name)

        # Count frames with valid data per person
        person_frame_counts = {}
        for person_idx, person_data in video_data['persons'].items():
            person_array = person_data['data']
            # Count non-NaN frames (where bbox_x1 is not NaN)
            valid_frames = np.sum(~np.isnan(person_array[:, 0]))
            person_frame_counts[person_idx] = int(valid_frames)

        return {
            'video_name': video_name,
            'total_frames': len(video_data['frame_numbers']),
            'n_persons': video_data['n_persons'],
            'person_frame_counts': person_frame_counts,
            'n_keypoints': video_data['persons'][0]['n_keypoints'] if video_data['persons'] else 0,
            'frame_numbers': video_data['frame_numbers']
        }

class BBoxEditor:
    """Interactive bounding box editor"""

    def __init__(self, img, bboxes, bbox_labels=None, max_width=800, max_height=600):
        self.original_img = img
        self.original_bboxes = bboxes.copy()
        self.bbox_labels = bbox_labels if bbox_labels else [f'Person {i}' for i in range(len(bboxes))]

        # Calculate scale factor to fit within max dimensions
        h, w = img.shape[:2]
        self.scale_x = min(max_width / w, 1.0)
        self.scale_y = min(max_height / h, 1.0)
        self.scale = min(self.scale_x, self.scale_y)  # Maintain aspect ratio

        # Resize image and bboxes
        self.display_width = int(w * self.scale)
        self.display_height = int(h * self.scale)

        # Actually resize the image for display
        self.img = cv2.resize(img, (self.display_width, self.display_height))

        # Scale bboxes
        self.bboxes = []
        for bbox in bboxes:
            scaled_bbox = bbox.copy()
            scaled_bbox[0] = bbox[0] * self.scale  # x1
            scaled_bbox[1] = bbox[1] * self.scale  # y1
            scaled_bbox[2] = bbox[2] * self.scale  # x2
            scaled_bbox[3] = bbox[3] * self.scale  # y2
            self.bboxes.append(scaled_bbox)

        self.canvas = Canvas(width=self.display_width, height=self.display_height)
        self.selected_bbox = None
        self.dragging = False
        self.resizing = False
        self.resize_corner = None  # 'tl', 'tr', 'bl', 'br'
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.corner_threshold = 5  # pixels

        # Set the actual display size - now matches the scaled image
        self.canvas.layout.width = f'{self.display_width}px'
        self.canvas.layout.height = f'{self.display_height}px'

        # Performance optimizations
        self.last_mouse_pos = None
        self.redraw_pending = False

        self.canvas.on_mouse_down(self.on_mouse_down)
        self.canvas.on_mouse_move(self.on_mouse_move)
        self.canvas.on_mouse_up(self.on_mouse_up)

        self.redraw()

    def get_corner_near_point(self, x, y, bbox):
        """Check if point is near any corner of the bbox"""
        x1, y1, x2, y2 = bbox[:4]
        corners = {
            'tl': (x1, y1),  # top-left
            'tr': (x2, y1),  # top-right
            'bl': (x1, y2),  # bottom-left
            'br': (x2, y2)   # bottom-right
        }

        for corner_name, (cx, cy) in corners.items():
            if abs(x - cx) <= self.corner_threshold and abs(y - cy) <= self.corner_threshold:
                return corner_name
        return None

    def redraw(self):
        # Only redraw if not already pending
        if self.redraw_pending:
            return

        self.redraw_pending = True

        with hold_canvas(self.canvas):
            self.canvas.clear()
            self.canvas.put_image_data(self.img, 0, 0)

            # Draw bboxes
            for i, bbox in enumerate(self.bboxes):
                x1, y1, x2, y2 = bbox[:4]
                self.canvas.stroke_style = 'red' if i != self.selected_bbox else 'blue'
                self.canvas.line_width = 2 if i != self.selected_bbox else 3
                self.canvas.stroke_rect(x1, y1, x2-x1, y2-y1)

                # Draw corner handles for selected bbox
                if i == self.selected_bbox:
                    self.canvas.fill_style = 'blue'
                    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                    for cx, cy in corners:
                        self.canvas.fill_rect(cx-2, cy-2, 4, 4)

                # Draw text with person label
                self.canvas.fill_style = 'red' if i != self.selected_bbox else 'blue'
                self.canvas.font = '14px Arial'
                label = self.bbox_labels[i] if i < len(self.bbox_labels) else f'Person {i}'
                self.canvas.fill_text(label, x1, max(y1-5, 15))

        self.redraw_pending = False

    def on_mouse_down(self, x, y):
        # Check if clicking on existing bbox
        for i, bbox in enumerate(self.bboxes):
            x1, y1, x2, y2 = bbox[:4]
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_bbox = i

                # Check if clicking near a corner
                corner = self.get_corner_near_point(x, y, bbox)
                if corner:
                    self.resizing = True
                    self.resize_corner = corner
                else:
                    self.dragging = True
                    self.drag_offset_x = x - x1
                    self.drag_offset_y = y - y1

                self.last_mouse_pos = (x, y)
                self.redraw()
                break

    def on_mouse_move(self, x, y):
        # Skip if mouse hasn't moved enough (reduce sensitivity)
        if self.last_mouse_pos:
            dx = abs(x - self.last_mouse_pos[0])
            dy = abs(y - self.last_mouse_pos[1])
            if dx < 2 and dy < 2:  # Only update if moved more than 2 pixels
                return

        self.last_mouse_pos = (x, y)

        if self.resizing and self.selected_bbox is not None:
            # Resize the bbox
            bbox = self.bboxes[self.selected_bbox]
            x1, y1, x2, y2 = bbox[:4]

            # Constrain to canvas bounds
            x = max(0, min(x, self.display_width))
            y = max(0, min(y, self.display_height))

            if self.resize_corner == 'tl':  # top-left
                x1, y1 = x, y
            elif self.resize_corner == 'tr':  # top-right
                x2, y1 = x, y
            elif self.resize_corner == 'bl':  # bottom-left
                x1, y2 = x, y
            elif self.resize_corner == 'br':  # bottom-right
                x2, y2 = x, y

            # Ensure valid bbox (min width/height of 10 pixels)
            if x2 - x1 < 10:
                if self.resize_corner in ['tl', 'bl']:
                    x1 = x2 - 10
                else:
                    x2 = x1 + 10
            if y2 - y1 < 10:
                if self.resize_corner in ['tl', 'tr']:
                    y1 = y2 - 10
                else:
                    y2 = y1 + 10

            self.bboxes[self.selected_bbox][:4] = [x1, y1, x2, y2]
            self.redraw()

        elif self.dragging and self.selected_bbox is not None:
            # Move the entire bbox
            bbox = self.bboxes[self.selected_bbox]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            new_x1 = x - self.drag_offset_x
            new_y1 = y - self.drag_offset_y

            # Constrain to canvas bounds
            new_x1 = max(0, min(new_x1, self.display_width - width))
            new_y1 = max(0, min(new_y1, self.display_height - height))

            self.bboxes[self.selected_bbox][:4] = [new_x1, new_y1, new_x1 + width, new_y1 + height]
            self.redraw()

    def on_mouse_up(self, x, y):
        if self.dragging or self.resizing:
            # Update original bbox coordinates
            scaled_bbox = self.bboxes[self.selected_bbox]
            self.original_bboxes[self.selected_bbox][0] = scaled_bbox[0] / self.scale
            self.original_bboxes[self.selected_bbox][1] = scaled_bbox[1] / self.scale
            self.original_bboxes[self.selected_bbox][2] = scaled_bbox[2] / self.scale
            self.original_bboxes[self.selected_bbox][3] = scaled_bbox[3] / self.scale

        self.dragging = False
        self.resizing = False
        self.resize_corner = None
        self.selected_bbox = None
        self.last_mouse_pos = None
        self.redraw()

    def get_original_bboxes(self):
        """Return the bboxes in original image coordinates"""
        return self.original_bboxes


class BBoxEditorWithCallback(BBoxEditor):
    """BBox Editor with callback support for new bbox drawing"""

    def __init__(self, img, bboxes, bbox_labels=None, callback=None, new_bbox_callback=None, max_width=800, max_height=600):
        self.callback = callback
        self.new_bbox_callback = new_bbox_callback
        self.drawing_mode = False
        self.drawing_start_pos = None
        self.drawing_current_pos = None

        super().__init__(img, bboxes, bbox_labels, max_width, max_height)

    def enable_drawing_mode(self):
        """Enable mode for drawing new bounding boxes"""
        self.drawing_mode = True
        self.selected_bbox = None
        self.redraw()

    def on_mouse_down(self, x, y):
        if self.drawing_mode:
            # Start drawing new bbox
            self.drawing_start_pos = (x, y)
            self.drawing_current_pos = (x, y)
            return

        # Normal bbox selection/editing
        super().on_mouse_down(x, y)

    def on_mouse_move(self, x, y):
        if self.drawing_mode and self.drawing_start_pos:
            # Update drawing rectangle
            self.drawing_current_pos = (x, y)
            self.redraw()
            return

        # Normal bbox movement/resizing
        super().on_mouse_move(x, y)

    def on_mouse_up(self, x, y):
        if self.drawing_mode and self.drawing_start_pos:
            # Finish drawing new bbox
            x1, y1 = self.drawing_start_pos
            x2, y2 = x, y

            # Ensure valid rectangle (minimum size and correct order)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            if x2 - x1 > 10 and y2 - y1 > 10:  # Minimum size check
                # Convert to original image coordinates
                orig_x1 = x1 / self.scale
                orig_y1 = y1 / self.scale
                orig_x2 = x2 / self.scale
                orig_y2 = y2 / self.scale

                # Call callback to add new person
                if self.new_bbox_callback:
                    self.new_bbox_callback(orig_x1, orig_y1, orig_x2, orig_y2)

            # Reset drawing mode
            self.drawing_mode = False
            self.drawing_start_pos = None
            self.drawing_current_pos = None
            return

        # Normal mouse up handling
        was_dragging = self.dragging
        was_resizing = self.resizing
        selected_bbox_idx = self.selected_bbox

        super().on_mouse_up(x, y)

        # If we were dragging/resizing and have a callback, notify
        if (was_dragging or was_resizing) and self.callback and selected_bbox_idx is not None:
            bbox = self.original_bboxes[selected_bbox_idx]
            x1, y1, x2, y2 = bbox[:4]
            self.callback(selected_bbox_idx, x1, y1, x2, y2)

    def redraw(self):
        # Only redraw if not already pending
        if self.redraw_pending:
            return

        self.redraw_pending = True

        with hold_canvas(self.canvas):
            self.canvas.clear()
            self.canvas.put_image_data(self.img, 0, 0)

            # Draw existing bboxes
            for i, bbox in enumerate(self.bboxes):
                x1, y1, x2, y2 = bbox[:4]
                self.canvas.stroke_style = 'red' if i != self.selected_bbox else 'blue'
                self.canvas.line_width = 2 if i != self.selected_bbox else 3
                self.canvas.stroke_rect(x1, y1, x2-x1, y2-y1)

                # Draw corner handles for selected bbox
                if i == self.selected_bbox:
                    self.canvas.fill_style = 'blue'
                    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                    for cx, cy in corners:
                        self.canvas.fill_rect(cx-2, cy-2, 4, 4)

                # Draw text with person label
                self.canvas.fill_style = 'red' if i != self.selected_bbox else 'blue'
                self.canvas.font = '14px Arial'
                label = self.bbox_labels[i] if i < len(self.bbox_labels) else f'Person {i}'
                self.canvas.fill_text(label, x1, max(y1-5, 15))

            # Draw new bbox being drawn
            if self.drawing_mode and self.drawing_start_pos and self.drawing_current_pos:
                x1, y1 = self.drawing_start_pos
                x2, y2 = self.drawing_current_pos
                self.canvas.stroke_style = 'green'
                self.canvas.line_width = 2
                self.canvas.stroke_rect(min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))

            # Show instruction text when in drawing mode
            if self.drawing_mode:
                self.canvas.fill_style = 'green'
                self.canvas.font = '16px Arial'
                self.canvas.fill_text('Draw rectangle for new person', 10, 25)

        self.redraw_pending = False


class VideoFrameEditor:
    """Frame editor adapted to work with VideoData"""

    def __init__(self, video_data: VideoData):
        self.video_data = video_data
        self.frame_numbers = video_data.get_frame_numbers()
        self.current_frame_idx = 0
        self.bbox_editor = None
        self.use_named_persons = video_data.use_named_persons
        self.timeline_fig = None  # Store timeline figure for click events

        # Create UI components
        self.frame_slider = widgets.IntSlider(
            value=self.frame_numbers[0],
            min=min(self.frame_numbers),
            max=max(self.frame_numbers),
            step=1,
            description='Frame:',
            style={'description_width': 'initial'}
        )

        # Navigation buttons for person count changes
        self.prev_person_change_button = widgets.Button(
            description='← Prev Change',
            button_style='info',
            icon='arrow-left',
            tooltip='Go to previous frame with different person count'
        )

        self.next_person_change_button = widgets.Button(
            description='Next Change →',
            button_style='info',
            icon='arrow-right',
            tooltip='Go to next frame with different person count'
        )

        # Create add person buttons
        if self.use_named_persons:
            # Create button for each named person
            self.add_person_buttons = {}
            for person_name in video_data.named_person_timeline.named_persons:
                color = video_data.named_person_timeline.person_colors.get(person_name, '#95E1D3')
                btn = widgets.Button(
                    description=f'Add {person_name}',
                    button_style='success',
                    icon='plus',
                    layout=widgets.Layout(width='120px')
                )
                btn.style.button_color = color
                btn.on_click(lambda b, name=person_name: self.on_add_named_person_click(name))
                self.add_person_buttons[person_name] = btn

            add_buttons_box = widgets.HBox(list(self.add_person_buttons.values()))
        else:
            # Single add person button for non-named mode
            self.add_person_button = widgets.Button(
                description='Add Person',
                button_style='success',
                icon='plus'
            )
            self.add_person_button.on_click(self.on_add_person_click)
            add_buttons_box = widgets.HBox([self.add_person_button])

        # Frame controls layout
        self.frame_controls = widgets.VBox([
            self.frame_slider,
            widgets.HBox([
                self.prev_person_change_button,
                self.next_person_change_button
            ]),
            add_buttons_box
        ])

        self.main_canvas_container = widgets.VBox()
        self.timeline_output = widgets.Output()  # For timeline visualization
        self.person_images_container = widgets.VBox()  # Changed to VBox for better layout

        # Create main content area with side-by-side layout
        self.main_content = widgets.HBox([
            widgets.VBox([
                widgets.HTML("<h3>Main Image with Bounding Boxes</h3>"),
                self.main_canvas_container,
                widgets.HTML("<h3>Person Detection Timeline</h3>"),
                self.timeline_output
            ]),
            widgets.VBox([
                widgets.HTML("<h3>Individual Persons</h3>"),
                self.person_images_container
            ])
        ])

        # Layout
        self.ui = widgets.VBox([
            self.frame_controls,
            self.main_content
        ])

        # Event handlers
        self.frame_slider.observe(self.on_frame_change, names='value')
        self.prev_person_change_button.on_click(self.on_prev_person_change)
        self.next_person_change_button.on_click(self.on_next_person_change)

        # Only attach add_person_button handler if it exists (non-named mode)
        if not self.use_named_persons:
            self.add_person_button.on_click(self.on_add_person_click)

        # Initialize display
        self.update_display()

    def on_frame_change(self, change):
        self.current_frame_num = change['new']
        self.update_display()

    def on_add_named_person_click(self, person_name: str):
        """Add a specific named person to the current frame"""
        # Enable drawing mode and store the person name for the new bbox
        if self.bbox_editor:
            self.pending_person_name = person_name
            self.bbox_editor.enable_drawing_mode()

    def on_add_person_click(self, button):
        """Enable drawing mode for adding a new person (non-named mode)"""
        if self.bbox_editor:
            self.bbox_editor.enable_drawing_mode()

    def on_prev_person_change(self, button):
        """Navigate to previous frame with different person count"""
        current_frame = self.frame_slider.value
        current_person_count = self.video_data.get_person_count(current_frame)

        # Find previous frame with different person count
        for frame_num in reversed(self.frame_numbers):
            if frame_num < current_frame:
                frame_person_count = self.video_data.get_person_count(frame_num)
                if frame_person_count != current_person_count:
                    self.frame_slider.value = frame_num
                    return

        # If no change found, go to first frame
        if self.frame_numbers and self.frame_numbers[0] != current_frame:
            self.frame_slider.value = self.frame_numbers[0]

    def on_next_person_change(self, button):
        """Navigate to next frame with different person count"""
        current_frame = self.frame_slider.value
        current_person_count = self.video_data.get_person_count(current_frame)

        # Find next frame with different person count
        for frame_num in self.frame_numbers:
            if frame_num > current_frame:
                frame_person_count = self.video_data.get_person_count(frame_num)
                if frame_person_count != current_person_count:
                    self.frame_slider.value = frame_num
                    return

        # If no change found, go to last frame
        if self.frame_numbers and self.frame_numbers[-1] != current_frame:
            self.frame_slider.value = self.frame_numbers[-1]

    def on_new_bbox_drawn(self, x1, y1, x2, y2):
        """Called when a new bbox is drawn"""
        if self.use_named_persons and hasattr(self, 'pending_person_name'):
            # In named person mode, add the person with the stored name
            person_name = self.pending_person_name
            try:
                self.video_data.add_named_person(
                    self.current_frame_num,
                    person_name,
                    x1, y1, x2, y2
                )
                print(f"Added {person_name} to frame {self.current_frame_num}")
            except Exception as e:
                print(f"Error adding {person_name}: {e}")
            finally:
                del self.pending_person_name
        else:
            # Non-named mode
            self.video_data.add_person(self.current_frame_num, x1, y1, x2, y2)

        # Refresh the display
        self.update_display()

    def update_person_bbox_named(self, person_name: str, x1: float, y1: float, x2: float, y2: float):
        """Update bbox for a named person"""
        frame_num = self.current_frame_num

        if frame_num not in self.video_data.frame_data:
            return

        frame_persons = self.video_data.frame_data[frame_num]
        if person_name not in frame_persons:
            return

        # Load original frame image
        img = self.video_data._load_frame_image(frame_num)

        # Update person data
        person = frame_persons[person_name]
        person.bbox = np.array([x1, y1, x2, y2])

        # Recompute pose with new bbox
        preproc_img, center, scale = self.video_data.pose_model.preprocess(img, person.bbox[:4])
        outputs = self.video_data.pose_model.inference(preproc_img)
        keypoints, scores = self.video_data.pose_model.postprocess(outputs, center, scale)

        # Update stored data
        person.center = center
        person.scale = scale
        person.keypoints = keypoints
        person.scores = scores
        person.simcc_x = outputs[0]
        person.simcc_y = outputs[1]

        # Also update timeline if this is a manual detection
        if person.is_manual and self.video_data.named_person_timeline is not None:
            # Update the bbox in the timeline's detection
            yolo_id = self.video_data.named_person_timeline.timelines[person_name].get(frame_num)
            if yolo_id and yolo_id in self.video_data.named_person_timeline.detections:
                detect_result = self.video_data.named_person_timeline.detections[yolo_id].get(frame_num)
                if detect_result:
                    # Update to new bbox (convert to center format)
                    detect_result.x = (x1 + x2) / 2
                    detect_result.y = (y1 + y2) / 2
                    detect_result.w = x2 - x1
                    detect_result.h = y2 - y1

    def on_bbox_update(self, person_idx_or_name, x1, y1, x2, y2):
        """Called when a bbox is moved/resized"""
        if self.use_named_persons:
            # In named mode, person_idx_or_name is actually a name (from bbox_labels)
            # But we receive an index from the editor, so we need to map it
            frame_persons = self.video_data.frame_data.get(self.current_frame_num, {})
            person_names = list(frame_persons.keys())

            if person_idx_or_name < len(person_names):
                person_name = person_names[person_idx_or_name]
                self.update_person_bbox_named(person_name, x1, y1, x2, y2)
        else:
            # Non-named mode
            self.video_data.update_person_bbox(self.current_frame_num, person_idx_or_name, x1, y1, x2, y2)

        self.update_person_images()

    def update_display(self):
        """Update the entire display for current frame"""
        self.current_frame_num = self.frame_slider.value

        # Update button states for named persons
        if self.use_named_persons:
            persons_in_frame = self.video_data.get_person_names_in_frame(self.current_frame_num)
            for person_name, btn in self.add_person_buttons.items():
                btn.disabled = person_name in persons_in_frame

        self.update_main_canvas()
        self.update_timeline_visualization()
        self.update_person_images()

    def update_main_canvas(self):
        """Update the main canvas with current frame and bboxes"""
        img = self.video_data.draw_frame(self.current_frame_num)

        # Get bboxes and labels for current frame
        frame_persons = self.video_data.frame_data.get(self.current_frame_num,
                                                       {} if self.use_named_persons else [])

        if self.use_named_persons:
            # Named persons mode: frame_persons is a dict {name: VideoPersonData}
            bboxes = [p.bbox for p in frame_persons.values()]
            bbox_labels = list(frame_persons.keys())
        else:
            # Index mode: frame_persons is a list [VideoPersonData, ...]
            bboxes = [p.bbox for p in frame_persons]
            bbox_labels = [f'Person {i}' for i in range(len(frame_persons))]

        # Create new bbox editor with callbacks
        self.bbox_editor = BBoxEditorWithCallback(
            img, bboxes, bbox_labels,
            callback=self.on_bbox_update,
            new_bbox_callback=self.on_new_bbox_drawn,
            max_width=800, max_height=600
        )

        self.main_canvas_container.children = [self.bbox_editor.canvas]

    def update_timeline_visualization(self):
        """Create and update timeline visualization showing person detections"""
        if not self.use_named_persons:
            return  # Only for named persons mode

        with self.timeline_output:
            self.timeline_output.clear_output(wait=True)

            # Close old figure if exists
            if self.timeline_fig is not None:
                plt.close(self.timeline_fig)

            # Create timeline visualization
            self.timeline_fig, ax = plt.subplots(figsize=(12, 3))

            # Get all person names
            person_names = self.video_data.named_person_timeline.named_persons
            colors = self.video_data.named_person_timeline.person_colors

            # Create timeline for each person
            for idx, person_name in enumerate(person_names):
                # Get frames where this person is detected
                frames_with_person = []
                for frame_num in self.frame_numbers:
                    if person_name in self.video_data.get_person_names_in_frame(frame_num):
                        frames_with_person.append(frame_num)

                # Plot as horizontal bars
                if frames_with_person:
                    y_pos = len(person_names) - idx - 1
                    for frame in frames_with_person:
                        ax.barh(y_pos, 1, left=frame, height=0.8,
                               color=colors[person_name], alpha=0.7)

            # Highlight current frame
            ax.axvline(x=self.current_frame_num, color='red', linestyle='--',
                      linewidth=2, alpha=0.7, label='Current Frame')

            # Configure plot
            ax.set_yticks(range(len(person_names)))
            ax.set_yticklabels(reversed(person_names))
            ax.set_xlabel('Frame Number', fontsize=10)
            ax.set_xlim(min(self.frame_numbers), max(self.frame_numbers))
            ax.set_ylim(-0.5, len(person_names) - 0.5)
            ax.legend(loc='upper right')
            ax.grid(axis='x', alpha=0.3)
            ax.set_title('Person Detection Timeline (Click to jump to frame)',
                        fontsize=12, fontweight='bold')

            # Connect click event
            self.timeline_fig.canvas.mpl_connect('button_press_event', self._on_timeline_click)

            plt.tight_layout()
            plt.show()

    def _on_timeline_click(self, event):
        """Handle click on timeline to jump to frame"""
        if event.inaxes is not None and event.xdata is not None:
            # Get clicked frame
            clicked_frame = int(round(event.xdata))

            # Constrain to valid frame range
            clicked_frame = max(min(self.frame_numbers),
                              min(clicked_frame, max(self.frame_numbers)))

            # Update frame slider
            if clicked_frame in self.frame_numbers:
                self.frame_slider.value = clicked_frame

    def update_person_images(self):
        """Update individual person images"""
        person_widgets = []

        if self.use_named_persons:
            # Named persons mode: iterate over person names
            persons_in_frame = self.video_data.get_person_names_in_frame(self.current_frame_num)

            for person_name in persons_in_frame:
                try:
                    person_img = self.video_data.draw_person(self.current_frame_num, person_name)
                    # Convert BGR to RGB for matplotlib
                    person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

                    # Create matplotlib figure with smaller size
                    fig, ax = plt.subplots(figsize=(2.5, 3.5))
                    ax.imshow(person_img_rgb)
                    ax.set_title(f'{person_name}', fontsize=10, fontweight='bold')
                    ax.axis('off')
                    plt.tight_layout()

                    # Convert to widget using Output
                    output = widgets.Output(layout=widgets.Layout(
                        width='auto',
                        height='auto',
                        margin='5px'
                    ))
                    with output:
                        display(fig)

                    person_widgets.append(output)
                    plt.close(fig)

                except Exception as e:
                    # If person image fails, show placeholder
                    output = widgets.Output()
                    with output:
                        print(f"{person_name}:\nError: {e}")
                    person_widgets.append(output)
        else:
            # Index-based mode: iterate over person indices
            person_count = self.video_data.get_person_count(self.current_frame_num)

            for i in range(person_count):
                try:
                    person_img = self.video_data.draw_person(self.current_frame_num, i)
                    # Convert BGR to RGB
                    person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

                    # Create matplotlib figure with smaller size
                    fig, ax = plt.subplots(figsize=(2.5, 3.5))
                    ax.imshow(person_img_rgb)
                    ax.set_title(f'Person {i}', fontsize=10, fontweight='bold')
                    ax.axis('off')
                    plt.tight_layout()

                    # Convert to widget using Output
                    output = widgets.Output(layout=widgets.Layout(
                        width='auto',
                        height='auto',
                        margin='5px'
                    ))
                    with output:
                        display(fig)

                    person_widgets.append(output)
                    plt.close(fig)

                except Exception as e:
                    # If person image fails, show placeholder
                    output = widgets.Output()
                    with output:
                        print(f"Person {i}:\nError: {e}")
                    person_widgets.append(output)

        self.person_images_container.children = person_widgets

    def display(self):
        """Display the UI"""
        return self.ui


# Convenience functions for easy use

def create_models(device="cuda"):
    """Create default YOLOX and RTMPose models"""
    det_model = YOLOX(
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
        model_input_size=(640,640),
        backend="onnxruntime",
        device=device
    )

    # Whole body
    pose_model = RTMPose(
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
        model_input_size=(288, 384),
        backend="onnxruntime",
        device=device
    )

    return det_model, pose_model


def analyze_video(video_path: str, device="cuda", start_frame: int = 1, end_frame: Optional[int] = None):
    """
    Analyze video and return interactive editor

    Args:
        video_path: Path to video file
        device: Device to use for inference ("cuda" or "cpu")
        start_frame: First frame to process
        end_frame: Last frame to process (None for all frames)

    Returns:
        VideoFrameEditor: Interactive editor widget
    """
    det_model, pose_model = create_models(device)
    video_data = VideoData(video_path, det_model, pose_model, start_frame, end_frame)
    editor = VideoFrameEditor(video_data)
    return editor


def analyze_video_with_named_persons(
    video_path: str,
    named_persons: List[str],
    tracker_config: Optional[str] = None,
    person_colors: Optional[Dict[str, str]] = None,
    yolo_model: str = "yolo11x.pt",
    device: str = "cuda",
    start_frame: int = 1,
    end_frame: Optional[int] = None,
    timeline: Optional[NamedPersonTimeline] = None
) -> Tuple[VideoFrameEditor, NamedPersonTimeline]:
    """
    Analyze video with YOLO tracker and named persons support

    This function performs analysis in stages:
    1. Use YOLO tracker to detect and track persons across frames (if timeline not provided)
    2. Use RTMPose to estimate poses for tracked persons

    After analysis, you can assign YOLO track IDs to named persons using the timeline,
    or provide a pre-configured timeline from create_named_person_stitcher().

    Args:
        video_path: Path to video file
        named_persons: List of person names to track (e.g., ["Harri", "Tommi", "Jani"])
        tracker_config: Path to YOLO tracker config file (None for default)
        person_colors: Optional dict mapping person names to hex colors
        yolo_model: YOLO model name (default: "yolo11x.pt")
        device: Device to use for inference ("cuda" or "cpu")
        start_frame: First frame to process
        end_frame: Last frame to process (None for all)
        timeline: Optional pre-configured NamedPersonTimeline (from stitcher)

    Returns:
        Tuple of (VideoFrameEditor, NamedPersonTimeline)

    Example:
        >>> # Option 1: All-in-one (requires manual assignment after)
        >>> editor, timeline = analyze_video_with_named_persons(
        ...     "video.mp4",
        ...     ["Harri", "Tommi", "Jani"]
        ... )
        >>> timeline.assign_yolo_to_person(1, "Harri", from_frame=0)
        >>> display(editor.display())

        >>> # Option 2: Use stitcher first (recommended)
        >>> stitcher = create_named_person_stitcher("video.mp4", ["Harri", "Tommi", "Jani"])
        >>> display(stitcher.display())
        >>> # ... make assignments in UI ...
        >>> editor, timeline = analyze_video_with_named_persons(
        ...     "video.mp4",
        ...     ["Harri", "Tommi", "Jani"],
        ...     timeline=stitcher.timeline  # Use the timeline with assignments
        ... )
        >>> display(editor.display())
    """
    # Step 1: Analyze with YOLO tracker (if timeline not provided)
    if timeline is None:
        timeline, total_frames = analyze_video_with_yolo_tracker(
            video_path, named_persons, tracker_config, person_colors, yolo_model, device
        )

    # Step 2: Create pose estimation models
    det_model, pose_model = create_models(device)

    # Step 3: Create VideoData with named person timeline
    video_data = VideoData(
        video_path, det_model, pose_model,
        start_frame, end_frame,
        named_person_timeline=timeline
    )

    # Step 4: Create editor
    editor = VideoFrameEditor(video_data)

    return editor, timeline


def extract_person_array(video_path: str, person_idx: int, device="cuda", start_frame: int = 1, end_frame: Optional[int] = None):
    """
    Convenience function to extract person data array from video

    Args:
        video_path: Path to video file or image sequence directory
        person_idx: Index of the person to extract (0-based)
        device: Device to use for inference ("cuda" or "cpu")
        start_frame: First frame to process
        end_frame: Last frame to process (None for all)

    Returns:
        tuple containing:
        - data_array: numpy array with person data across frames
        - metadata: dict with video_path, person_index, and column_descriptions
    """
    det_model, pose_model = create_models(device)
    video_data = VideoData(video_path, det_model, pose_model, start_frame, end_frame)
    return video_data.extract_person_data_array(person_idx)

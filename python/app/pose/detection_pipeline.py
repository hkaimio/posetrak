"""detection_pipeline.py — Synchronous detection + pose estimation pipeline."""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import av as _av
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False

import cv2

from app.pose.backends import PersonDetector, PoseEstimator
from app.pose.db_cache import DetectionBatchWriter, create_detection_run, mark_run_complete

ProgressCallback = Callable[[int, int, str], None]  # done, total, camera_id


@dataclass
class CameraInfo:
    shot_video_id: str
    camera_instance_id: str
    file_path: str
    actual_fps: float
    ref_frame: int          # sync anchor frame number
    ref_timestamp_s: float  # sync anchor global time


@dataclass
class PipelineResult:
    detection_run_id: str
    cameras_processed: list[str] = field(default_factory=list)
    frames_processed: int = 0
    status: str = "complete"


class DetectionPipeline:
    """Run person detection and pose estimation for all cameras in a shot.

    Processes frames sequentially (camera by camera), writing results
    directly to the session DB.  No intermediate files are created.

    Coordinates are in original (distorted) pixel space throughout.
    """

    def __init__(
        self,
        session: sqlite3.Connection,
        shot_id: str,
        sync_config_id: str,
        time_start_s: float,
        time_end_s: float,
        detector: PersonDetector,
        estimator: PoseEstimator,
        thumbnail_every_s: float = 0.0,  # 0 = no thumbnails in this run
        stop_event: threading.Event | None = None,
    ) -> None:
        self._session = session
        self._shot_id = shot_id
        self._sync_config_id = sync_config_id
        self._time_start_s = time_start_s
        self._time_end_s = time_end_s
        self._detector = detector
        self._estimator = estimator
        self._thumbnail_every_s = thumbnail_every_s
        self._stop_event = stop_event or threading.Event()
        self._cameras = self._load_cameras()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, on_progress: ProgressCallback | None = None) -> PipelineResult:
        run_id = create_detection_run(
            self._session,
            shot_id=self._shot_id,
            sync_config_id=self._sync_config_id,
            time_start_s=self._time_start_s,
            time_end_s=self._time_end_s,
            detector_model=self._detector.name,
            pose_model=self._estimator.name,
            detector_version=getattr(self._detector, "version", ""),
            pose_version=getattr(self._estimator, "version", ""),
            detector_conf=getattr(self._detector, "_conf", 0.3),
            pose_conf_threshold=0.3,
            pose_input_width=self._estimator.input_size[1],   # (H, W) → W
            pose_input_height=self._estimator.input_size[0],
        )

        result = PipelineResult(detection_run_id=run_id)

        try:
            for cam in self._cameras:
                if self._stop_event.is_set():
                    break
                self._detector.reset_tracker()
                n = self._process_camera(run_id, cam, on_progress)
                result.cameras_processed.append(cam.camera_instance_id)
                result.frames_processed += n

            status = "failed" if self._stop_event.is_set() else "complete"
            result.status = status
            mark_run_complete(self._session, run_id, status)
        except Exception:
            mark_run_complete(self._session, run_id, "failed")
            raise

        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_cameras(self) -> list[CameraInfo]:
        """Load shot videos with sync anchor for each camera."""
        rows = self._session.execute(
            "SELECT sv.id, sv.camera_instance_id, sv.file_path, sv.actual_fps, "
            "       sp.video_frame, sp.timestamp_s "
            "FROM shot_videos sv "
            "JOIN sync_points sp "
            "    ON sp.camera_instance_id = sv.camera_instance_id "
            "    AND sp.sync_config_id = ? "
            "WHERE sv.shot_id = ? "
            "ORDER BY sv.camera_instance_id",
            (self._sync_config_id, self._shot_id),
        ).fetchall()

        cameras = []
        for row in rows:
            fps = float(row["actual_fps"] or 30.0)
            cameras.append(CameraInfo(
                shot_video_id=row["id"],
                camera_instance_id=row["camera_instance_id"],
                file_path=row["file_path"],
                actual_fps=fps,
                ref_frame=int(row["video_frame"]),
                ref_timestamp_s=float(row["timestamp_s"]),
            ))
        return cameras

    def _frame_range(self, cam: CameraInfo) -> tuple[int, int]:
        """Convert global time range to (first_frame, last_frame_exclusive)."""
        fps = cam.actual_fps
        first = cam.ref_frame + int((self._time_start_s - cam.ref_timestamp_s) * fps)
        last = cam.ref_frame + int((self._time_end_s - cam.ref_timestamp_s) * fps)
        first = max(0, first)
        return first, last

    def _process_camera(
        self,
        run_id: str,
        cam: CameraInfo,
        on_progress: ProgressCallback | None,
    ) -> int:
        first_frame, last_frame = self._frame_range(cam)
        total = max(1, last_frame - first_frame)

        writer = DetectionBatchWriter(
            self._session,
            detection_run_id=run_id,
            shot_video_id=cam.shot_video_id,
            pose_input_width=self._estimator.input_size[1],
        )

        frames_done = 0
        try:
            for video_frame, img in self._iter_frames(cam, first_frame, last_frame):
                if self._stop_event.is_set():
                    break

                detections = self._detector.detect_and_track(img, video_frame)
                pose_results = self._estimator.estimate(img, detections) if detections else []

                writer.add_frame(video_frame, detections, pose_results, self._detector.name)
                frames_done += 1

                if on_progress:
                    on_progress(frames_done, total, cam.camera_instance_id)
        finally:
            writer.finalise()

        return frames_done

    def _iter_frames(
        self, cam: CameraInfo, first_frame: int, last_frame: int
    ):
        """Yield (video_frame_index, bgr_array) for frames in [first_frame, last_frame)."""
        path = cam.file_path
        fps = cam.actual_fps

        if _AV_AVAILABLE:
            yield from self._iter_frames_av(path, fps, first_frame, last_frame)
        else:
            yield from self._iter_frames_cv2(path, first_frame, last_frame)

    @staticmethod
    def _iter_frames_av(path: str, fps: float, first_frame: int, last_frame: int):
        import av
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            time_base = float(stream.time_base)

            # Seek slightly before the target to land on a keyframe
            if first_frame > 0:
                seek_s = max(0.0, (first_frame - 1) / fps)
                seek_pts = int(seek_s / time_base)
                container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

            for av_frame in container.decode(stream):
                if av_frame.pts is None:
                    continue
                frame_s = float(av_frame.pts) * time_base
                frame_idx = round(frame_s * fps)
                if frame_idx < first_frame:
                    continue
                if frame_idx >= last_frame:
                    break
                yield frame_idx, av_frame.to_ndarray(format="bgr24")

    @staticmethod
    def _iter_frames_cv2(path: str, first_frame: int, last_frame: int):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        frame_idx = first_frame
        while frame_idx < last_frame:
            ok, img = cap.read()
            if not ok:
                break
            yield frame_idx, img
            frame_idx += 1
        cap.release()

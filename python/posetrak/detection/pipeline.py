# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""pipeline.py — Synchronous detection + pose estimation pipeline."""
from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from posetrak.detection.backends import PersonDetector, PoseEstimator
from posetrak.detection.frame_source import iter_frames
from app.pose.db_cache import DetectionBatchWriter, create_detection_run, mark_run_complete
from app.setup.db_context import SyncPoint, SyncTable

_log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]  # done, total, camera_id


@dataclass
class CameraInfo:
    shot_video_id: str
    camera_instance_id: str
    file_path: str
    actual_fps: float
    ref_frame: int          # sync anchor frame number
    ref_timestamp_s: float  # sync anchor global time
    # Human-readable camera label (e.g. "gopro-11_mini_01") for progress/log
    # messages -- camera_instance_id is a UUID, not something a user reading
    # a progress bar can identify a camera by. Defaults to "" (not the UUID
    # itself) so call sites that predate this field but never display it
    # (pose_worker.py's single-camera background-refinement path, tests)
    # don't need updating; _load_cameras() below always sets a real value.
    label: str = ""


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
        self._cameras, self._sync_table = self._load_cameras()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def cameras(self) -> list[CameraInfo]:
        return self._cameras

    def run(
        self,
        on_progress: ProgressCallback | None = None,
        on_camera_done: Callable[[int, int], None] | None = None,
    ) -> PipelineResult:
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
                _log.info("run: resetting tracker for %s", cam.camera_instance_id)
                self._detector.reset_tracker()
                _log.info("run: tracker reset done, starting camera %s", cam.camera_instance_id)
                n = self._process_camera(run_id, cam, on_progress)
                result.cameras_processed.append(cam.camera_instance_id)
                result.frames_processed += n
                if on_camera_done:
                    on_camera_done(len(result.cameras_processed), len(self._cameras))

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

    def _load_cameras(self) -> tuple[list[CameraInfo], SyncTable]:
        """Load shot videos and build a SyncTable from all sync points."""
        sp_rows = self._session.execute(
            "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, sv.actual_fps "
            "FROM sync_points sp "
            "JOIN capture_videos sv ON sv.id = sp.shot_video_id "
            "WHERE sp.sync_config_id = ? AND sv.shot_id = ? "
            "ORDER BY sp.shot_video_id, sp.video_frame",
            (self._sync_config_id, self._shot_id),
        ).fetchall()

        sync_points: list[SyncPoint] = []
        fps_by_video: dict[str, float] = {}
        # First sync point per video used as the single-anchor fallback in CameraInfo
        anchor_by_video: dict[str, tuple[int, float]] = {}
        for r in sp_rows:
            svid = r["shot_video_id"]
            sync_points.append(SyncPoint(
                camera_instance_id=svid,
                shot_video_id=svid,
                video_frame=int(r["video_frame"]),
                timestamp_s=float(r["timestamp_s"]),
            ))
            fps_by_video.setdefault(svid, float(r["actual_fps"] or 30.0))
            if svid not in anchor_by_video:
                anchor_by_video[svid] = (int(r["video_frame"]), float(r["timestamp_s"]))

        sync_table = SyncTable(sync_points, fps_by_video)

        cam_rows = self._session.execute(
            "SELECT sv.id, sv.camera_instance_id, sv.file_path, sv.actual_fps,"
            "       COALESCE(ci.label, sv.camera_instance_id) AS camera_label "
            "FROM capture_videos sv "
            "LEFT JOIN camera_instances ci ON ci.id = sv.camera_instance_id "
            "WHERE sv.id IN (SELECT DISTINCT shot_video_id FROM sync_points WHERE sync_config_id = ?) "
            "  AND sv.shot_id = ? "
            "ORDER BY sv.camera_instance_id",
            (self._sync_config_id, self._shot_id),
        ).fetchall()

        cameras = []
        for row in cam_rows:
            svid = row["id"]
            fps = float(row["actual_fps"] or 30.0)
            ref_frame, ref_ts = anchor_by_video.get(svid, (0, 0.0))
            cameras.append(CameraInfo(
                shot_video_id=svid,
                camera_instance_id=row["camera_instance_id"],
                file_path=row["file_path"],
                actual_fps=fps,
                ref_frame=ref_frame,
                ref_timestamp_s=ref_ts,
                label=row["camera_label"],
            ))

        _log.info(
            "_load_cameras: shot=%s sync=%s → %d cameras, %d sync points",
            self._shot_id, self._sync_config_id, len(cameras), len(sp_rows),
        )
        if not cameras:
            _log.warning(
                "_load_cameras: no cameras found — check that sync_points exist "
                "for sync_config_id=%s and shot_id=%s",
                self._sync_config_id, self._shot_id,
            )
        return cameras, sync_table

    def _frame_range(self, cam: CameraInfo) -> tuple[int, int]:
        """Convert global time range to (first_frame, last_frame_exclusive).

        Uses the SyncTable (piecewise-linear interpolation through all sync
        points) when available; falls back to single-anchor + fps extrapolation
        only when the SyncTable has no data for this camera.
        """
        first = self._sync_table.lookup(self._time_start_s, cam.shot_video_id)
        last = self._sync_table.lookup(self._time_end_s, cam.shot_video_id)
        if first is None or last is None:
            _log.warning(
                "_frame_range: no sync data for %s — falling back to fps extrapolation",
                cam.shot_video_id,
            )
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
        _log.info(
            "_process_camera: %s (%s)  file=%s  frames %d–%d (%d total)  fps=%.2f",
            cam.label or cam.camera_instance_id, cam.camera_instance_id,
            cam.file_path, first_frame, last_frame, total, cam.actual_fps,
        )

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

                if frames_done == 0:
                    _log.debug(
                        "_process_camera: first frame %d decoded ok, shape=%s, "
                        "%d detections, %d poses",
                        video_frame, img.shape, len(detections), len(pose_results),
                    )

                writer.add_frame(video_frame, detections, pose_results, self._detector.name, img)
                frames_done += 1

                if on_progress:
                    on_progress(frames_done, total, cam.label or cam.camera_instance_id)
        finally:
            writer.finalise()

        _log.info(
            "_process_camera: %s done — %d frames",
            cam.label or cam.camera_instance_id, frames_done,
        )
        return frames_done

    def _iter_frames(
        self, cam: CameraInfo, first_frame: int, last_frame: int
    ):
        """Yield (video_frame_index, bgr_array) for frames in [first_frame, last_frame)."""
        return iter_frames(cam.file_path, first_frame, last_frame)

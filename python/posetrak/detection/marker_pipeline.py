# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""marker_pipeline.py — ArUco marker detection pipeline (design phase 1a).

See docs/roadmap/features/marker-based-mocap/marker-mocap-design.md §7.1
sub-phase 1a. Structurally parallel to ``pipeline.py``'s person
``DetectionPipeline``, but for coded (ArUco) markers on a rigid prop:

- reuses ``app.setup.fiducial_markers.ArucoDetector`` (built stateless and
  per-frame for exactly this kind of consumer);
- writes the fixed-slot corner blob described in the design doc's §4.1
  directly to ``detection_keypoints`` via ``db_cache.MarkerKeypointWriter``.

Deliberately decoupled from ``capture_objects``/``marker_body_definitions``
(that wiring is sub-phase 1c): the caller passes detector configuration
directly (dictionary, the prop's marker ids, perimeter rate), the same way
this module can be driven from a script or test without any GUI or
capture-object plumbing existing yet.

Camera/sync-table loading below duplicates ``pipeline.py``'s
``_load_cameras``/``_frame_range`` rather than sharing them, to keep this
phase's slice self-contained; a shared helper is a reasonable extraction
once both pipelines have settled (design doc's "Option 1 first, revisit if
it proves fiddly" precedent, §5.3).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable

from app.pose.db_cache import MarkerKeypointWriter, create_marker_detection_run, mark_run_complete
from app.setup.db_context import SyncPoint, SyncTable
from app.setup.fiducial_markers import ArucoDetector
from posetrak.detection.frame_source import iter_frames

_log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]  # done, total, camera_id


@dataclass
class MarkerCameraInfo:
    shot_video_id: str
    camera_instance_id: str
    file_path: str
    actual_fps: float
    ref_frame: int          # sync anchor frame number
    ref_timestamp_s: float  # sync anchor global time
    label: str = ""


@dataclass
class MarkerPipelineResult:
    detection_run_id: str
    cameras_processed: list[str] = field(default_factory=list)
    frames_processed: int = 0
    status: str = "complete"


class MarkerDetectionPipeline:
    """Run ArUco marker detection for all cameras covering a shot/trial's
    synced time range.

    Coordinates are in original (distorted) pixel space, matching the
    pose-detection pipeline and every other detection-layer consumer.
    """

    def __init__(
        self,
        session,
        shot_id: str,
        sync_config_id: str,
        time_start_s: float,
        time_end_s: float,
        marker_ids: list[str],
        dictionary: str = "DICT_4X4_50",
        min_marker_perimeter_rate: float | None = None,
        frame_step: int = 1,
        trial_id: str | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        if not marker_ids:
            raise ValueError(
                "marker_ids must list every coded marker id the prop carries -- "
                "it fixes the detection_keypoints corner-slot ordering (design §4.1)"
            )
        if frame_step < 1:
            raise ValueError("frame_step must be >= 1")
        self._session = session
        self._shot_id = shot_id
        self._sync_config_id = sync_config_id
        self._time_start_s = time_start_s
        self._time_end_s = time_end_s
        self._trial_id = trial_id
        self._dictionary = dictionary
        self._marker_ids = list(marker_ids)
        self._min_marker_perimeter_rate = min_marker_perimeter_rate
        self._frame_step = frame_step
        self._stop_event = stop_event or threading.Event()
        self._detector = ArucoDetector(
            dictionary=dictionary,
            min_marker_perimeter_rate=min_marker_perimeter_rate,
        )
        self._cameras, self._sync_table = self._load_cameras()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def cameras(self) -> list[MarkerCameraInfo]:
        return self._cameras

    def run(self, on_progress: ProgressCallback | None = None) -> MarkerPipelineResult:
        run_id = create_marker_detection_run(
            self._session,
            shot_id=self._shot_id,
            sync_config_id=self._sync_config_id,
            time_start_s=self._time_start_s,
            time_end_s=self._time_end_s,
            dictionary=self._dictionary,
            marker_ids=self._marker_ids,
            min_marker_perimeter_rate=self._min_marker_perimeter_rate,
            frame_step=self._frame_step,
            trial_id=self._trial_id,
        )

        result = MarkerPipelineResult(detection_run_id=run_id)

        try:
            for cam in self._cameras:
                if self._stop_event.is_set():
                    break
                n = self._process_camera(run_id, cam, on_progress)
                result.cameras_processed.append(cam.camera_instance_id)
                result.frames_processed += n

            result.status = "failed" if self._stop_event.is_set() else "complete"
            mark_run_complete(self._session, run_id, result.status)
        except Exception:
            mark_run_complete(self._session, run_id, "failed")
            raise

        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_cameras(self) -> tuple[list[MarkerCameraInfo], SyncTable]:
        """Load shot videos and build a SyncTable from all sync points.

        Identical in structure to ``pipeline.DetectionPipeline._load_cameras``
        -- see this module's docstring for why it isn't shared yet.
        """
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
            cameras.append(MarkerCameraInfo(
                shot_video_id=svid,
                camera_instance_id=row["camera_instance_id"],
                file_path=row["file_path"],
                actual_fps=fps,
                ref_frame=ref_frame,
                ref_timestamp_s=ref_ts,
                label=row["camera_label"],
            ))

        _log.info(
            "_load_cameras: shot=%s sync=%s -> %d cameras, %d sync points",
            self._shot_id, self._sync_config_id, len(cameras), len(sp_rows),
        )
        return cameras, sync_table

    def _frame_range(self, cam: MarkerCameraInfo) -> tuple[int, int]:
        """Convert global time range to (first_frame, last_frame_exclusive).

        Uses the SyncTable (piecewise-linear interpolation through all sync
        points) when available; falls back to single-anchor + fps
        extrapolation only when the SyncTable has no data for this camera.
        """
        first = self._sync_table.lookup(self._time_start_s, cam.shot_video_id)
        last = self._sync_table.lookup(self._time_end_s, cam.shot_video_id)
        if first is None or last is None:
            _log.warning(
                "_frame_range: no sync data for %s -- falling back to fps extrapolation",
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
        cam: MarkerCameraInfo,
        on_progress: ProgressCallback | None,
    ) -> int:
        first_frame, last_frame = self._frame_range(cam)
        total = max(1, (last_frame - first_frame + self._frame_step - 1) // self._frame_step)
        _log.info(
            "_process_camera: %s (%s)  file=%s  frames %d-%d (frame_step=%d)",
            cam.label or cam.camera_instance_id, cam.camera_instance_id,
            cam.file_path, first_frame, last_frame, self._frame_step,
        )

        writer = MarkerKeypointWriter(
            self._session,
            detection_run_id=run_id,
            shot_video_id=cam.shot_video_id,
            marker_ids=self._marker_ids,
        )

        frames_done = 0
        try:
            for video_frame, img in iter_frames(cam.file_path, first_frame, last_frame):
                if self._stop_event.is_set():
                    break
                if (video_frame - first_frame) % self._frame_step != 0:
                    continue

                detections = self._detector.detect(
                    img, video_id=cam.camera_instance_id, frame_idx=video_frame
                )
                writer.add_frame(video_frame, detections)
                frames_done += 1

                if on_progress:
                    on_progress(frames_done, total, cam.label or cam.camera_instance_id)
        finally:
            writer.finalise()

        _log.info(
            "_process_camera: %s done -- %d frames",
            cam.label or cam.camera_instance_id, frames_done,
        )
        return frames_done

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""pipeline.py — the detection frame loop and the pipeline that runs it over a capture.

One loop serves every box source and pose model (design: docs/roadmap/features/detection-pipeline/
detection-pipeline-design.md, section 2). It is three stages joined by bounded queues:

1. **decode** — one thread per camera video reads frames (on that thread, or in a child process, see
   ``frame_readers``) and asks the box source to prepare them;
2. **infer** — one thread takes frames from all cameras, asks the box source for their boxes and runs the pose
   estimator on the crops of all persons of all those frames in batches, then the hand pass if enabled;
3. **write** — the thread that called :meth:`DetectionLoop.run` writes the rows (the database connection belongs
   to it) and reports progress.

Coordinates are in original (distorted) pixel space throughout.
"""
from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from posetrak.detection import stage_timing as st
from posetrak.detection.backends import PersonDetection, PersonDetector, PoseEstimator, PoseResult
from posetrak.detection.box_sources import BoxSource, FrameWork
from posetrak.detection.frame_readers import FrameReader, get_reader
from app.pose.db_cache import DetectionBatchWriter, create_detection_run, mark_run_complete
from app.setup.db_context import SyncPoint, SyncTable

_log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]  # done, total, camera_id

#: How the decode producers run unless the caller chooses (see ``frame_readers``).
DEFAULT_DECODE_MODE = "thread"


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
    # itself) so call sites that never display it don't need updating;
    # _load_cameras() below always sets a real value.
    label: str = ""


@dataclass
class CameraJob:
    """One camera's frames to process: ``[first_frame, last_frame)`` of its video."""

    cam: CameraInfo
    first_frame: int
    last_frame: int


@dataclass
class PipelineResult:
    detection_run_id: str
    cameras_processed: list[str] = field(default_factory=list)
    frames_processed: int = 0
    status: str = "complete"


class _Done:
    """Marks the end of a stream of frames (of one camera in the decode queue, of the run in the write queue)."""


@dataclass
class _Written:
    work: FrameWork
    detections: list[PersonDetection]
    poses: list[PoseResult]
    hands: dict[int, list]           # track_id -> (region_type, hand_kp, noise_scale) rows


class DetectionLoop:
    """The three-stage frame loop over a list of cameras, writing into an existing detection run."""

    def __init__(
        self,
        session: sqlite3.Connection,
        run_id: str,
        box_source: BoxSource,
        estimator: PoseEstimator,
        *,
        hand_pass=None,
        reader: FrameReader | None = None,
        batch_frames: int = 6,
        queue_frames: int | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._session, self._run_id = session, run_id
        self._box_source, self._estimator, self._hand_pass = box_source, estimator, hand_pass
        self._reader = reader or get_reader(DEFAULT_DECODE_MODE)
        self._batch_frames, self._queue_frames = batch_frames, queue_frames
        self._stop = stop_event or threading.Event()
        self._abort = threading.Event()      # an error in one stage: the others stop too
        self._closed = threading.Event()     # the writer has left: nobody reads the write queue any more
        self._errors: list[BaseException] = []
        self._infer_thread: threading.Thread | None = None

    # -- queues ------------------------------------------------------------

    def _halted(self) -> bool:
        return self._stop.is_set() or self._abort.is_set()

    def _put(self, q: queue.Queue, item: Any) -> bool:
        """Put with a periodic look at the stop event; False if the loop is stopping."""
        while not self._halted():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _put_end(self, q: queue.Queue) -> None:
        """Put an end marker even while stopping: the consumer of *q* keeps draining it until it has them all."""
        while True:
            try:
                q.put(_Done(), timeout=0.2)
                return
            except queue.Full:
                if self._closed.is_set() or self._infer_thread is None or not self._infer_thread.is_alive():
                    return

    # -- stages ------------------------------------------------------------

    def _decode(self, job: CameraJob, out: queue.Queue) -> None:
        cam = job.cam
        try:
            self._box_source.start_camera(cam, job.first_frame, job.last_frame)
            frames = st.timed_iter("decode", self._reader(cam.file_path, job.first_frame, job.last_frame))
            for frame, img in frames:
                if self._halted():
                    break
                if self._box_source.wants(cam, frame):
                    work = FrameWork(cam, frame, img, self._box_source.prepare(cam, frame, img))
                else:
                    work = FrameWork(cam, frame, None)
                if not self._put(out, work):
                    break
        except BaseException as exc:
            _log.exception("decode of %s failed", cam.file_path)
            self._errors.append(exc)
            self._abort.set()
        finally:
            self._put_end(out)

    def _infer(self, decoded: queue.Queue, written: queue.Queue, n_streams: int) -> None:
        try:
            ended = 0
            while ended < n_streams:
                pending: list[FrameWork] = []
                item = decoded.get()
                while True:
                    if isinstance(item, _Done):
                        ended += 1
                    else:
                        pending.append(item)
                    if len(pending) >= self._batch_frames or ended >= n_streams:
                        break
                    try:
                        item = decoded.get_nowait()
                    except queue.Empty:
                        break
                if pending and not self._halted():
                    for result in self._process(pending):
                        if not self._put(written, result):
                            break
        except BaseException as exc:
            _log.exception("inference failed")
            self._errors.append(exc)
            self._abort.set()
        finally:
            self._put_end(written)

    def _process(self, works: list[FrameWork]) -> list[_Written]:
        wanted = [w for w in works if w.image is not None]
        detections = dict(zip((id(w) for w in wanted), self._box_source.detect(wanted))) if wanted else {}
        todo = [(w, detections[id(w)]) for w in wanted if detections[id(w)]]
        poses = self._estimator.estimate_batch([(w.image, d) for w, d in todo]) if todo else []
        pose_of = {id(w): p for (w, _), p in zip(todo, poses)}
        hands: dict[int, dict[int, list]] = {}
        if self._hand_pass is not None and todo:
            jobs = [(w.image, r.keypoints, id(w), r.track_id) for (w, _), p in zip(todo, poses) for r in p]
            rows = self._hand_pass.refine([(img, kp) for img, kp, _, _ in jobs])
            for (_, _, key, track_id), hand_rows in zip(jobs, rows):
                if hand_rows:
                    hands.setdefault(key, {})[track_id] = hand_rows
        return [_Written(w, detections.get(id(w), []), pose_of.get(id(w), []), hands.get(id(w), {})) for w in works]

    # -- run ---------------------------------------------------------------

    def run(
        self,
        jobs: Sequence[CameraJob],
        on_progress: ProgressCallback | None = None,
        on_camera_done: Callable[[int, int], None] | None = None,
    ) -> int:
        """Process *jobs*; returns the number of frames written. Stops early, without error, when stopped."""
        jobs = list(jobs)
        if not jobs:
            return 0
        width = self._estimator.input_size[1]
        writers = {j.cam.shot_video_id: DetectionBatchWriter(self._session, self._run_id, j.cam.shot_video_id, width)
                   for j in jobs}
        totals = {j.cam.shot_video_id: max(1, j.last_frame - j.first_frame) for j in jobs}
        done = dict.fromkeys(totals, 0)
        decoded: queue.Queue = queue.Queue(self._queue_frames or max(4, 2 * len(jobs)))
        written: queue.Queue = queue.Queue(2 * self._batch_frames)
        self._errors = []
        self._abort.clear()
        self._closed.clear()
        self._infer_thread = threading.Thread(target=self._infer, args=(decoded, written, len(jobs)),
                                              name="detect-infer", daemon=True)
        decoders = [threading.Thread(target=self._decode, args=(j, decoded), name=f"detect-decode-{i}", daemon=True)
                    for i, j in enumerate(jobs)]
        self._infer_thread.start()
        for t in decoders:
            t.start()
        cameras_done, frames = 0, 0
        try:
            while True:
                item = written.get()
                if isinstance(item, _Done):
                    break
                work = item.work
                svid = work.cam.shot_video_id
                writer = writers[svid]
                if item.detections:
                    writer.add_frame(work.video_frame, item.detections, item.poses, self._box_source.name, work.image)
                    for track_id, rows in item.hands.items():
                        writer.add_hand_rows(work.video_frame, track_id, rows)
                done[svid] += 1
                frames += 1
                if on_progress:
                    on_progress(done[svid], totals[svid], work.cam.label or work.cam.camera_instance_id)
                if done[svid] == totals[svid] and on_camera_done:
                    cameras_done += 1
                    on_camera_done(cameras_done, len(jobs))
        finally:
            self._abort.set()
            self._closed.set()
            for t in decoders:
                t.join(10.0)
            self._infer_thread.join(10.0)
            for writer in writers.values():
                writer.finalise()
            recompute_track_spans(self._session, self._run_id, list(writers))
        if self._errors:
            raise self._errors[0]
        return frames


def recompute_track_spans(session: sqlite3.Connection, run_id: str, shot_video_ids: Sequence[str]) -> None:
    """Set each track's first and last frame from all its detections in the run, whatever part this call wrote."""
    for svid in shot_video_ids:
        for row in session.execute(
            "SELECT track_id, MIN(video_frame) AS first_f, MAX(video_frame) AS last_f FROM person_detections"
            " WHERE detection_run_id=? AND shot_video_id=? GROUP BY track_id", (run_id, svid),
        ).fetchall():
            session.execute(
                "UPDATE person_tracks SET first_frame=?, last_frame=?"
                " WHERE detection_run_id=? AND shot_video_id=? AND track_id=?",
                (row[1], row[2], run_id, svid, row[0]),
            )
    session.commit()


def run_config(box_source: BoxSource, estimator: PoseEstimator, hand_pass, decode_mode: str, batch_frames: int) -> dict:
    """The options of a run, as stored in ``detection_runs.config_json``."""
    return {
        "box_source": {"name": box_source.name, **box_source.options()},
        "estimator": estimator.options(),
        "hand_pass": hand_pass is not None,
        "decode_mode": decode_mode,
        "batch_frames": batch_frames,
    }


def record_run_options(session: sqlite3.Connection, run_id: str, box_source: BoxSource, config: dict) -> None:
    """Store *config* and the box source's run inputs on a run that was created without them.

    A run that already has options keeps them (several jobs may write into one run); a run input already set is kept.
    """
    session.execute("UPDATE detection_runs SET config_json = ? WHERE id = ? AND config_json IS NULL",
                    (json.dumps(config), run_id))
    for column, value in box_source.run_inputs().items():
        session.execute(f"UPDATE detection_runs SET {column} = ? WHERE id = ? AND {column} IS NULL", (value, run_id))
    session.commit()


class DetectionPipeline:
    """Run person detection and pose estimation for all cameras in a shot.

    Creates the detection run, runs :class:`DetectionLoop` over the cameras, writes results directly to the session
    DB and marks the run complete or failed. No intermediate files are created. The boxes come from *box_source*
    (or, for callers that have a YOLOX detector, from *detector*, which is wrapped in a
    :class:`~posetrak.detection.box_sources.YoloxBoxSource`).
    """

    def __init__(
        self,
        session: sqlite3.Connection,
        shot_id: str,
        sync_config_id: str,
        time_start_s: float,
        time_end_s: float,
        detector: PersonDetector | None = None,
        estimator: PoseEstimator | None = None,
        *,
        box_source: BoxSource | None = None,
        hand_pass=None,
        decode_mode: str | None = None,
        trial_id: str | None = None,
        stop_event: threading.Event | None = None,
        batch_frames: int = 6,
    ) -> None:
        if box_source is None and detector is not None:
            from posetrak.detection.box_sources import YoloxBoxSource

            box_source = YoloxBoxSource(detector)
        self._session = session
        self._shot_id = shot_id
        self._sync_config_id = sync_config_id
        self._time_start_s = time_start_s
        self._time_end_s = time_end_s
        self._box_source = box_source
        self._estimator = estimator
        self._hand_pass = hand_pass
        self._decode_mode = decode_mode or DEFAULT_DECODE_MODE
        self._trial_id = trial_id
        self._batch_frames = batch_frames
        self._stop_event = stop_event or threading.Event()
        self._cameras, self._sync_table = self._load_cameras()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def cameras(self) -> list[CameraInfo]:
        return self._cameras

    def run_config(self) -> dict:
        """The options of this run, as stored in ``detection_runs.config_json``."""
        return run_config(self._box_source, self._estimator, self._hand_pass, self._decode_mode, self._batch_frames)

    def run(
        self,
        on_progress: ProgressCallback | None = None,
        on_camera_done: Callable[[int, int], None] | None = None,
    ) -> PipelineResult:
        box_source, estimator = self._box_source, self._estimator
        run_id = create_detection_run(
            self._session,
            shot_id=self._shot_id,
            sync_config_id=self._sync_config_id,
            time_start_s=self._time_start_s,
            time_end_s=self._time_end_s,
            detector_model=box_source.name,
            pose_model=estimator.name,
            trial_id=self._trial_id,
            detector_version=getattr(box_source, "version", ""),
            pose_version=getattr(estimator, "version", ""),
            detector_conf=getattr(box_source, "conf", 0.3),
            pose_conf_threshold=0.3,
            pose_input_width=estimator.input_size[1],   # (H, W) → W
            pose_input_height=estimator.input_size[0],
            config_json=json.dumps(self.run_config()),
            **box_source.run_inputs(),
        )

        result = PipelineResult(detection_run_id=run_id)
        try:
            jobs = []
            for cam in self._cameras:
                first, last = self._frame_range(cam)
                _log.info("run: %s (%s)  file=%s  frames %d–%d  fps=%.2f",
                          cam.label or cam.camera_instance_id, cam.camera_instance_id, cam.file_path,
                          first, last, cam.actual_fps)
                jobs.append(CameraJob(cam, first, last))
            loop = DetectionLoop(
                self._session, run_id, box_source, estimator, hand_pass=self._hand_pass,
                reader=get_reader(self._decode_mode), batch_frames=self._batch_frames, stop_event=self._stop_event,
            )
            result.frames_processed = loop.run(jobs, on_progress, on_camera_done)
            result.cameras_processed = [j.cam.camera_instance_id for j in jobs]
            result.status = "failed" if self._stop_event.is_set() else "complete"
            mark_run_complete(self._session, run_id, result.status)
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

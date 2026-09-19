# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""marker_pipeline.py — ArUco marker and reflective-dot detection pipeline.

See docs/roadmap/features/marker-based-mocap/marker-mocap-design.md §7.1.
Structurally parallel to ``pipeline.py``'s person ``DetectionPipeline``,
but for coded (ArUco) markers on a rigid prop.

Two ways to drive ``MarkerDetectionPipeline``, both writing the fixed-slot
corner blob described in the design doc's §4.1 to ``detection_keypoints``
via ``db_cache.MarkerKeypointWriter`` (detector-agnostic -- it only
consumes ``FiducialDetection`` objects, so it works unchanged either way):

- **Standalone**: pass ``marker_ids`` + a single
  ``dictionary`` directly, no ``capture_objects``/``marker_body_definitions``
  involved -- a plain ``ArucoDetector`` does the detecting. This is what a
  script or test drives without any GUI or registered marker body existing.
- **Marker-body-driven** (the GUI path): pass a
  ``rig_config`` (an ``app.setup.fiducial_markers.MarkerRigConfig``, e.g.
  from ``load_pipeline_for_capture_object`` below) -- a ``MarkerRigDetector``
  does the detecting instead, which additionally handles a body spanning
  more than one ArUco dictionary and filters out any marker not actually
  part of this body, such as a tag from a different object in the same
  scene (marker-mocap-algorithms.md §1.1). ``marker_ids`` is derived from the
  config, not given directly.

Either mode can additionally run anonymous reflective-dot blob detection
per camera (see ``__init__``'s ``detect_dots_for_cameras``) -- an
orthogonal, opt-in add-on to whichever coded-marker detector is already
running, not a third mode: same frame, same loop, a second writer
(``db_cache.DotCandidateWriter``) alongside the first.

Camera/sync-table loading below duplicates ``pipeline.py``'s
``_load_cameras``/``_frame_range`` rather than sharing them, which keeps the
two pipelines independent; a shared helper is a reasonable extraction now
that both are stable.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

import cv2

from app.pose.db_cache import (
    DotCandidateWriter,
    MarkerKeypointWriter,
    create_marker_detection_run,
    mark_run_complete,
)
from app.setup.db_context import SyncPoint, SyncTable
from app.setup.fiducial_markers import ArucoDetector, MarkerRigConfig, MarkerRigDetector, load_marker_body_yaml
from posetrak.detection.dot_blob_detector import compute_background, detect_blobs
from posetrak.detection.dot_tracklet import MotionGatedLinker
from posetrak.detection.frame_source import iter_frames
from posetrak.db.manage_capture_object import get_capture_object

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


@dataclass
class _DotDetectionConfig:
    """Plain-data snapshot of MarkerDetectionPipeline's dot-detection
    settings -- picklable, so a ProcessPoolExecutor worker (run_parallel(),
    below) can carry it across the process boundary unchanged, unlike the
    pipeline instance itself (holds a live sqlite3 connection and a
    threading.Event, neither of which survive pickling)."""
    threshold: int
    threshold_by_camera: dict[str, int]
    background_mode: str
    blacklist_frac: float
    blacklist_frac_by_camera: dict[str, float]
    blacklist_radius_px: int
    max_saturation: float
    max_saturation_by_camera: dict[str, float]
    bg_subtract: bool
    bg_sample_count: int


@dataclass
class _DetectorSpec:
    """Picklable recipe to reconstruct either coded-marker detector kind in
    a fresh worker process -- see the class docstring's "Two ways to drive"
    section for what each kind is. `rig_config` is None for the standalone
    ArucoDetector kind."""
    dictionary: str
    min_marker_perimeter_rate: float | None
    rig_config: MarkerRigConfig | None = None


def _build_detector(spec: _DetectorSpec):
    if spec.rig_config is not None:
        return MarkerRigDetector(
            spec.rig_config, dictionary=spec.dictionary, min_marker_perimeter_rate=spec.min_marker_perimeter_rate,
        )
    return ArucoDetector(dictionary=spec.dictionary, min_marker_perimeter_rate=spec.min_marker_perimeter_rate)


def _compute_dot_background_for(file_path: str, first_frame: int, last_frame: int, bg_sample_count: int):
    """Module-level twin of MarkerDetectionPipeline._compute_dot_background --
    see that method's own docstring for the sequential-decode-not-per-sample-
    seek reasoning. Free-standing (not a method) so a ProcessPoolExecutor
    worker can call it without a pipeline instance."""
    span = last_frame - first_frame
    stride = max(1, span // bg_sample_count)
    frames = []
    for video_frame, img in iter_frames(file_path, first_frame, last_frame):
        if (video_frame - first_frame) % stride == 0:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if not frames:
        _log.warning("_compute_dot_background_for: %s -- no frames sampled, dot detection for "
                     "this camera will run without a background", file_path)
        return None
    return compute_background(frames)


def _process_camera_core(
    conn,
    detector,
    run_id: str,
    cam: "MarkerCameraInfo",
    marker_ids: list[str],
    frame_step: int,
    stop_event: threading.Event,
    detect_dots: bool,
    dot_cfg: _DotDetectionConfig,
    first_frame: int,
    last_frame: int,
    on_progress: ProgressCallback | None,
) -> int:
    """The actual per-camera decode+detect+write loop -- shared, unchanged
    logic between MarkerDetectionPipeline._process_camera() (sequential,
    one connection/detector reused across every camera) and run_parallel()'s
    worker function (one fresh connection/detector per camera, run in its
    own process). Takes every dependency as an explicit argument rather
    than reading `self` so it works identically either way."""
    total = max(1, (last_frame - first_frame + frame_step - 1) // frame_step)
    _log.info(
        "_process_camera_core: %s (%s)  file=%s  frames %d-%d (frame_step=%d)",
        cam.label or cam.camera_instance_id, cam.camera_instance_id,
        cam.file_path, first_frame, last_frame, frame_step,
    )

    writer = MarkerKeypointWriter(
        conn, detection_run_id=run_id, shot_video_id=cam.shot_video_id, marker_ids=marker_ids,
    )
    dot_writer = None
    dot_background = None
    dot_linker = None
    if detect_dots:
        dot_writer = DotCandidateWriter(conn, detection_run_id=run_id, shot_video_id=cam.shot_video_id)
        dot_linker = MotionGatedLinker()
        # 'blacklist' mode needs a background image just as much as 'subtract' does
        # (see detect_blobs()'s own docstring) -- gate on either, not bg_subtract
        # alone, so choosing background_mode='blacklist' without separately setting
        # bg_subtract=True doesn't silently run with background=None instead.
        if dot_cfg.bg_subtract or dot_cfg.background_mode == "blacklist":
            dot_background = _compute_dot_background_for(
                cam.file_path, first_frame, last_frame, dot_cfg.bg_sample_count,
            )

    frames_done = 0
    try:
        for video_frame, img in iter_frames(cam.file_path, first_frame, last_frame):
            if stop_event.is_set():
                break
            if (video_frame - first_frame) % frame_step != 0:
                continue

            detections = detector.detect(img, video_id=cam.camera_instance_id, frame_idx=video_frame)
            writer.add_frame(video_frame, detections)

            if dot_writer is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dot_threshold = dot_cfg.threshold_by_camera.get(cam.camera_instance_id, dot_cfg.threshold)
                dot_max_sat = dot_cfg.max_saturation_by_camera.get(cam.camera_instance_id, dot_cfg.max_saturation)
                dot_frac = dot_cfg.blacklist_frac_by_camera.get(cam.camera_instance_id, dot_cfg.blacklist_frac)
                blobs = detect_blobs(
                    gray, threshold=dot_threshold, background=dot_background,
                    background_mode=dot_cfg.background_mode,
                    blacklist_frac=dot_frac,
                    blacklist_radius_px=dot_cfg.blacklist_radius_px,
                    bgr=img, max_saturation=dot_max_sat,
                )
                dot_linker.link_frame(video_frame, blobs)
                dot_writer.add_frame(video_frame, blobs)

            frames_done += 1

            if on_progress:
                on_progress(frames_done, total, cam.label or cam.camera_instance_id)
    finally:
        writer.finalise()
        if dot_writer is not None:
            dot_writer.finalise()

    _log.info("_process_camera_core: %s done -- %d frames", cam.label or cam.camera_instance_id, frames_done)
    return frames_done


@dataclass
class _CameraJob:
    """Picklable unit of work for run_parallel()'s ProcessPoolExecutor --
    everything _run_camera_job needs, carried across the process boundary
    as plain data (no live connection, detector instance, or Event)."""
    session_path: str
    run_id: str
    cam: "MarkerCameraInfo"
    marker_ids: list[str]
    frame_step: int
    detector_spec: _DetectorSpec
    detect_dots: bool
    dot_cfg: _DotDetectionConfig
    first_frame: int
    last_frame: int


def _run_camera_job(job: _CameraJob) -> tuple[str, int]:
    """ProcessPoolExecutor worker entry point (must be module-level to be
    picklable for Windows' spawn start method). Opens its own connection --
    SQLite connections cannot cross a process boundary -- with a real
    busy_timeout, since WAL mode (set once, persisted in the DB file itself
    by create_session) allows concurrent writers but still briefly
    serializes actual commits; the default 0 timeout would surface that as
    an immediate "database is locked" error instead of a short, harmless
    wait. No live stop_event or per-frame on_progress here -- see
    run_parallel()'s own docstring for why those don't cross the process
    boundary."""
    conn = sqlite3.connect(job.session_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    detector = _build_detector(job.detector_spec)
    try:
        frames_done = _process_camera_core(
            conn, detector, job.run_id, job.cam, job.marker_ids, job.frame_step,
            threading.Event(), job.detect_dots, job.dot_cfg, job.first_frame, job.last_frame,
            on_progress=None,
        )
    finally:
        conn.close()
    return job.cam.camera_instance_id, frames_done


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
        marker_ids: list[str] | None = None,
        dictionary: str = "DICT_4X4_50",
        rig_config: MarkerRigConfig | None = None,
        min_marker_perimeter_rate: float | None = None,
        frame_step: int = 1,
        trial_id: str | None = None,
        capture_object_id: str | None = None,
        marker_body_definition_id: str | None = None,
        stop_event: threading.Event | None = None,
        detect_dots_for_cameras: set[str] | None = None,
        dot_bg_subtract: bool = False,
        dot_threshold: int = 235,
        dot_threshold_by_camera: dict[str, int] | None = None,
        dot_background_mode: str = "subtract",
        dot_blacklist_frac: float = 0.7,
        dot_blacklist_frac_by_camera: dict[str, float] | None = None,
        dot_blacklist_radius_px: int = 6,
        dot_max_saturation: float = 255.0,
        dot_max_saturation_by_camera: dict[str, float] | None = None,
        dot_bg_sample_count: int = 40,
    ) -> None:
        """See the class docstring for the two ways to drive this pipeline.

        detect_dots_for_cameras: camera_instance_ids to additionally run
            anonymous reflective-dot blob detection on, alongside the coded-
            marker detection every camera already gets (dot-assignment-
            architecture-design.md §7's write path). None/empty disables it
            for every camera -- the default, so every existing caller is
            unaffected. Per-camera rather than a single on/off switch because
            dot visibility depends on the physical rig (a ring light on the
            GoPros used to validate this detector, not necessarily on every
            camera in the same capture) -- the caller decides which cameras
            actually have one, this pipeline doesn't guess from a label.
        dot_bg_subtract, dot_threshold, dot_max_saturation, dot_bg_sample_count:
            opt-in, off by default (see dot_blob_detector.py's docstring for
            why) -- background-subtracted residual detection plus a chroma
            (saturation) filter, validated on a fast-swing capture but not
            broadly enough to be the default for every capture. dot_threshold
            defaults to
            detect_blobs()'s own raw-brightness default (235) so passing
            nothing changes nothing, but MUST be lowered (e.g. to somewhere
            in the tens, not hundreds) when dot_bg_subtract=True -- it gates
            the *residual* in that mode, a much smaller quantity than raw
            brightness, and 235 is unreachable for any realistic residual.
            When dot_bg_subtract is True, each dot-enabled camera gets one
            extra full sequential decode pass (roughly doubling that
            camera's decode time) to build its own median background frame
            from dot_bg_sample_count frames spread across the camera's own
            time range, before the real detection pass starts.
        dot_threshold_by_camera, dot_background_mode, dot_blacklist_frac,
        dot_blacklist_radius_px:
            'subtract' (the default) has a failure mode on person-worn
            markers -- a marker on a subject in a pose the background
            samples didn't cover gets fused into one large, non-round blob
            with the subject's own limb and shape-rejected as a whole (see
            dot_blob_detector.py's "Background modes"). dot_background_mode=
            'blacklist' instead thresholds each frame's own raw brightness
            (shape classification never sees more than the marker's own
            local contour) and uses the background only to veto a spot that's
            already nearly as bright with no subject present. On real data it
            raises recall and precision together, not one at the expense of
            the other.
            dot_threshold still gates raw brightness in this mode (like its
            un-subtracted default already implies), but two different
            cameras' own sensors/tone-mapping can cap a real marker's peak
            brightness at very different absolute levels even under
            identical lighting -- dot_threshold_by_camera (camera_instance_id
            -> threshold) overrides dot_threshold for specific cameras, so a
            rig with mismatched cameras doesn't have to pick one global
            value that's wrong for some of them. dot_blacklist_frac/
            dot_blacklist_radius_px tune the veto itself; see
            detect_blobs()'s own docstring.
        dot_max_saturation_by_camera, dot_blacklist_frac_by_camera:
            The same per-camera-override need as dot_threshold_by_camera,
            for two more parameters that real data showed also need it. A
            capture mixing camera models can have
            some markers rendering meaningfully colour-tinted (real
            saturation 50-90, not near-0) only on specific cameras --
            dot_max_saturation must be loosened there without loosening it
            (and admitting more skin/fabric) on cameras where it doesn't
            need to be. Separately, dot_blacklist_frac=0.7 (tuned on a
            reflective-prop capture) is too tight for person-worn markers
            passing in front of bright background patches (a window-lit
            floor, say): it vetoes real markers, and a ground-truth sweep
            recovered recall by loosening it to 0.85-0.95 depending on the
            camera. Both dicts override their
            scalar default per camera_instance_id exactly like
            dot_threshold_by_camera; absent from the dict means "use the
            scalar".
        """
        if rig_config is not None:
            if not rig_config.marker_corners:
                raise ValueError(
                    f"marker body {rig_config.rig_id!r} has no coded markers to detect "
                    "-- a dot-only body has no coded markers for this detector"
                )
            marker_ids = list(rig_config.marker_corners.keys())
        elif not marker_ids:
            raise ValueError(
                "marker_ids must list every coded marker id the prop carries -- "
                "it fixes the detection_keypoints corner-slot ordering (design §4.1), "
                "or pass rig_config instead to derive it from a marker body definition"
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
        self._capture_object_id = capture_object_id
        self._marker_body_definition_id = marker_body_definition_id
        self._stop_event = stop_event or threading.Event()
        self._detect_dots_for_cameras = detect_dots_for_cameras or set()
        self._dot_bg_subtract = dot_bg_subtract
        self._dot_threshold = dot_threshold
        self._dot_threshold_by_camera = dot_threshold_by_camera or {}
        self._dot_background_mode = dot_background_mode
        self._dot_blacklist_frac = dot_blacklist_frac
        self._dot_blacklist_frac_by_camera = dot_blacklist_frac_by_camera or {}
        self._dot_blacklist_radius_px = dot_blacklist_radius_px
        self._dot_max_saturation = dot_max_saturation
        self._dot_max_saturation_by_camera = dot_max_saturation_by_camera or {}
        self._dot_bg_sample_count = dot_bg_sample_count
        if rig_config is not None:
            self._detector = MarkerRigDetector(
                rig_config, dictionary=dictionary, min_marker_perimeter_rate=min_marker_perimeter_rate,
            )
        else:
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

    def run(
        self,
        on_progress: ProgressCallback | None = None,
        on_camera_done: Callable[[int, int], None] | None = None,
    ) -> MarkerPipelineResult:
        dot_detection_config = None
        if self._detect_dots_for_cameras:
            dot_detection_config = {
                "cameras": sorted(self._detect_dots_for_cameras),
                "bg_subtract": self._dot_bg_subtract,
                "background_mode": self._dot_background_mode,
                "threshold": self._dot_threshold,
                "threshold_by_camera": dict(self._dot_threshold_by_camera),
                "blacklist_frac": self._dot_blacklist_frac,
                "blacklist_frac_by_camera": dict(self._dot_blacklist_frac_by_camera),
                "blacklist_radius_px": self._dot_blacklist_radius_px,
                "max_saturation": self._dot_max_saturation,
                "max_saturation_by_camera": dict(self._dot_max_saturation_by_camera),
                "bg_sample_count": self._dot_bg_sample_count,
            }
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
            capture_object_id=self._capture_object_id,
            marker_body_definition_id=self._marker_body_definition_id,
            dot_detection_config=dot_detection_config,
        )

        result = MarkerPipelineResult(detection_run_id=run_id)

        try:
            for cam in self._cameras:
                if self._stop_event.is_set():
                    break
                n = self._process_camera(run_id, cam, on_progress)
                result.cameras_processed.append(cam.camera_instance_id)
                result.frames_processed += n
                if on_camera_done:
                    on_camera_done(len(result.cameras_processed), len(self._cameras))

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

    def _dot_config(self) -> _DotDetectionConfig:
        return _DotDetectionConfig(
            threshold=self._dot_threshold,
            threshold_by_camera=self._dot_threshold_by_camera,
            background_mode=self._dot_background_mode,
            blacklist_frac=self._dot_blacklist_frac,
            blacklist_frac_by_camera=self._dot_blacklist_frac_by_camera,
            blacklist_radius_px=self._dot_blacklist_radius_px,
            max_saturation=self._dot_max_saturation,
            max_saturation_by_camera=self._dot_max_saturation_by_camera,
            bg_subtract=self._dot_bg_subtract,
            bg_sample_count=self._dot_bg_sample_count,
        )

    def _detector_spec(self) -> _DetectorSpec:
        rig_config = self._detector.config if isinstance(self._detector, MarkerRigDetector) else None
        return _DetectorSpec(
            dictionary=self._dictionary, min_marker_perimeter_rate=self._min_marker_perimeter_rate,
            rig_config=rig_config,
        )

    def _resolve_session_path(self) -> str:
        """The DB file backing `self._session`, needed because a
        ProcessPoolExecutor worker can't share this connection across a
        process boundary and must open its own (run_parallel())."""
        row = self._session.execute("PRAGMA database_list").fetchone()
        if row is None or not row[2]:
            raise RuntimeError(
                "run_parallel() needs a file-backed session (an in-memory or unnamed DB has no "
                "path a worker process could reopen) -- use run() instead for such a session"
            )
        return row[2]

    def _process_camera(
        self,
        run_id: str,
        cam: MarkerCameraInfo,
        on_progress: ProgressCallback | None,
    ) -> int:
        first_frame, last_frame = self._frame_range(cam)
        return _process_camera_core(
            self._session, self._detector, run_id, cam, self._marker_ids, self._frame_step,
            self._stop_event, cam.camera_instance_id in self._detect_dots_for_cameras,
            self._dot_config(), first_frame, last_frame, on_progress,
        )

    def run_parallel(
        self,
        max_workers: int | None = None,
        on_camera_done: Callable[[int, int], None] | None = None,
    ) -> MarkerPipelineResult:
        """Camera-level parallel variant of run() -- one process per camera
        via ProcessPoolExecutor, cameras being the natural parallel unit
        (each already has its own decoder, background model, and tracklet
        linker state). Ceiling is max(per-camera time) rather than
        sum(per-camera time) -- ~Nx on an N-camera capture, up to however
        many the machine can run at once.

        Profiling a representative camera showed ArUco detection
        (~35ms/frame) and video decode (~26ms/frame) are the two real
        costs, both genuinely per-camera-independent CPU work -- the thing
        this parallelizes. A single camera's own frame sequence is
        NOT parallelized here (MotionGatedLinker.link_frame() is inherently
        sequential -- each frame's linking depends on the previous frame's
        still-open tracklets), so a capture with fewer cameras than
        available cores still leaves some idle. Splitting a camera's
        frames into chunks would use them, but is harder because of that
        linker state, and is worth building only if this ceiling proves
        insufficient in practice.

        Two real trade-offs against run(), both because a worker process
        can't share this instance's live state:
        - No live cancellation -- self._stop_event's a threading.Event,
          meaningless across a process boundary, so a run already
          submitted to the pool runs to completion regardless.
        - No per-frame progress -- on_progress doesn't cross process
          boundaries cheaply, so this reports per-CAMERA completion only
          (on_camera_done), not the finer-grained on_progress run() offers.

        Requires a file-backed session (see _resolve_session_path()) --
        each worker opens its own connection to the same file (WAL mode,
        set once by create_session and persisted in the file itself, plus
        a per-connection busy_timeout here, so concurrent writers retry
        briefly on lock contention rather than failing immediately).
        """
        session_path = self._resolve_session_path()
        dot_detection_config = None
        if self._detect_dots_for_cameras:
            dot_detection_config = {
                "cameras": sorted(self._detect_dots_for_cameras),
                "bg_subtract": self._dot_bg_subtract,
                "background_mode": self._dot_background_mode,
                "threshold": self._dot_threshold,
                "threshold_by_camera": dict(self._dot_threshold_by_camera),
                "blacklist_frac": self._dot_blacklist_frac,
                "blacklist_frac_by_camera": dict(self._dot_blacklist_frac_by_camera),
                "blacklist_radius_px": self._dot_blacklist_radius_px,
                "max_saturation": self._dot_max_saturation,
                "max_saturation_by_camera": dict(self._dot_max_saturation_by_camera),
                "bg_sample_count": self._dot_bg_sample_count,
            }
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
            capture_object_id=self._capture_object_id,
            marker_body_definition_id=self._marker_body_definition_id,
            dot_detection_config=dot_detection_config,
        )

        detector_spec = self._detector_spec()
        dot_cfg = self._dot_config()
        jobs = []
        for cam in self._cameras:
            first_frame, last_frame = self._frame_range(cam)
            jobs.append(_CameraJob(
                session_path=session_path, run_id=run_id, cam=cam, marker_ids=self._marker_ids,
                frame_step=self._frame_step, detector_spec=detector_spec,
                detect_dots=cam.camera_instance_id in self._detect_dots_for_cameras,
                dot_cfg=dot_cfg, first_frame=first_frame, last_frame=last_frame,
            ))

        result = MarkerPipelineResult(detection_run_id=run_id)
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_run_camera_job, job) for job in jobs]
                for future in as_completed(futures):
                    cam_id, n = future.result()
                    result.cameras_processed.append(cam_id)
                    result.frames_processed += n
                    if on_camera_done:
                        on_camera_done(len(result.cameras_processed), len(jobs))
            result.status = "complete"
            mark_run_complete(self._session, run_id, result.status)
        except Exception:
            mark_run_complete(self._session, run_id, "failed")
            raise

        return result


def load_pipeline_for_capture_object(
    session,
    capture_object_id: str,
    sync_config_id: str,
    time_start_s: float,
    time_end_s: float,
    min_marker_perimeter_rate: float | None = None,
    frame_step: int = 1,
    trial_id: str | None = None,
    stop_event: threading.Event | None = None,
    detect_dots_for_cameras: set[str] | None = None,
    dot_bg_subtract: bool = False,
    dot_threshold: int = 235,
    dot_threshold_by_camera: dict[str, int] | None = None,
    dot_background_mode: str = "subtract",
    dot_blacklist_frac: float = 0.7,
    dot_blacklist_frac_by_camera: dict[str, float] | None = None,
    dot_blacklist_radius_px: int = 6,
    dot_max_saturation: float = 255.0,
    dot_max_saturation_by_camera: dict[str, float] | None = None,
    dot_bg_sample_count: int = 40,
) -> MarkerDetectionPipeline:
    """Build a ``MarkerDetectionPipeline`` for an existing ``capture_objects``
    row -- resolves its marker body definition, loads the resolved rig
    geometry, and constructs the pipeline in marker-body-driven mode. This is
    the path the GUI's run-detection dialog uses; the plain constructor stays
    available for the standalone/scripted case.

    detect_dots_for_cameras, dot_bg_subtract, dot_threshold,
    dot_threshold_by_camera, dot_background_mode, dot_blacklist_frac,
    dot_blacklist_frac_by_camera, dot_blacklist_radius_px, dot_max_saturation,
    dot_max_saturation_by_camera, dot_bg_sample_count:
        forwarded to ``MarkerDetectionPipeline`` unchanged -- see its own
        docstring. The GUI's run-detection dialog does not yet expose a way
        to set any of these (no UI wiring exists for it yet); a caller
        building the pipeline directly can already use them.

    Raises
    ------
    ValueError
        If *capture_object_id* or its marker body definition does not
        exist, or the marker body has no coded markers to detect (a
        dot-only body has no coded markers for this detector).
    """
    obj_row = get_capture_object(session, capture_object_id)
    if obj_row is None:
        raise ValueError(f"capture_objects row not found: {capture_object_id!r}")

    body_id = obj_row["marker_body_definition_id"]
    body_row = session.execute(
        "SELECT yaml_content FROM marker_body_definitions WHERE id = ?", (body_id,)
    ).fetchone()
    if body_row is None or body_row["yaml_content"] is None:
        raise ValueError(f"marker_body_definitions row not found or empty: {body_id!r}")

    rig_config = load_marker_body_yaml(body_row["yaml_content"], rig_id=body_id)

    return MarkerDetectionPipeline(
        session,
        shot_id=obj_row["capture_id"],
        sync_config_id=sync_config_id,
        time_start_s=time_start_s,
        time_end_s=time_end_s,
        rig_config=rig_config,
        min_marker_perimeter_rate=min_marker_perimeter_rate,
        frame_step=frame_step,
        trial_id=trial_id,
        capture_object_id=capture_object_id,
        marker_body_definition_id=body_id,
        stop_event=stop_event,
        detect_dots_for_cameras=detect_dots_for_cameras,
        dot_bg_subtract=dot_bg_subtract,
        dot_threshold=dot_threshold,
        dot_threshold_by_camera=dot_threshold_by_camera,
        dot_background_mode=dot_background_mode,
        dot_blacklist_frac=dot_blacklist_frac,
        dot_blacklist_frac_by_camera=dot_blacklist_frac_by_camera,
        dot_blacklist_radius_px=dot_blacklist_radius_px,
        dot_max_saturation=dot_max_saturation,
        dot_max_saturation_by_camera=dot_max_saturation_by_camera,
        dot_bg_sample_count=dot_bg_sample_count,
    )

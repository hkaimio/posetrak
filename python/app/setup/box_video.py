# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""box_video.py — multi-camera extrinsics from a handheld calibration box.

A box carrying ArUco markers (a marker body definition) is moved around the
scene in view of every static camera and set down at the end. One reference
camera is scanned for moments when the box was naturally at rest; each such
*stationary window* is looked up in every other camera through the capture's
sync table, which gives all cameras a simultaneous, unblurred detection of the
box in one fixed pose.

World frame: one window, the *anchor*, supplies fixed control points (the
box's own frame is the world frame, as in ``anchor_from_marker_rig``). Every
other window contributes the box's corners as free points, with per-window
keys so that corners of different windows are never merged (the box moved in
between). ``run_calibration`` bundle-adjusts all of them together.

The anchor is the window seen by the most cameras. When the box's final resting
position is seen by fewer cameras than that, the solved scene is rigidly moved
afterwards so that the world frame is the box at its final resting position,
as it would have been had that window been the anchor.

The result is only trustworthy if every camera sees the box in at least some
window, so the solve is judged by a per-camera table of reprojection errors of
the box corners; ``BoxVideoResult.ok`` is False when any camera is unsolved,
has no residuals, or has a median error above the threshold.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import sqlite3
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.setup.db_context import SyncPoint, SyncTable
from app.setup.extrinsics_solver import (
    CalibResult,
    CamCalibState,
    MarkerGroup,
    _undistort_pts,
    apply_similarity_transform,
    control_point_residuals,
    run_calibration,
    triangulate_dlt,
)
from app.setup.fiducial_markers import (
    FiducialDetection,
    MarkerRigConfig,
    MarkerRigDetector,
    anchor_from_marker_rig,
    merge_detections_into_groups,
)
from posetrak.detection.frame_source import iter_frames, iter_frames_at

# A camera "sees" a window when at least one whole marker (4 corners) is detected.
_MIN_CORNERS_PER_CAMERA = 4

# A camera also fails when its 90th-percentile error exceeds this multiple of the median limit:
# a broken solve can keep most observations close (median of 1-3 px) while a quarter or more of
# them are hundreds of pixels off.
_P90_LIMIT_FACTOR = 4.0


class BoxVideoError(RuntimeError):
    """The calibration could not be attempted (missing sync, no windows, ...)."""


@dataclass
class BoxVideoParams:
    reference_camera: str
    time_start_s: float = 6.0
    time_end_s: float = 33.0
    min_marker_perimeter_rate: float = 0.01
    # ArUco dictionary for markers whose body definition names none.
    rig_dict: str = "DICT_4X4_50"
    # Added to every box corner's Z: puts world Z=0 at e.g. the floor below the
    # box's reference marker.
    world_z_offset: float = 0.0
    # Synced-timeline seconds of windows to drop; each matches the nearest window within 0.5 s.
    exclude_window_ts: list[float] = field(default_factory=list)
    # Windows with fewer detected corners (summed over cameras) are dropped.
    min_window_corners: int = 8
    max_median_error_px: float = 5.0
    sample_stride: int = 6
    max_centroid_move_px: float = 6.0
    min_run_samples: int = 3


@dataclass
class Window:
    timestamp_s: float
    # camera label -> number of box corners detected there
    corners_by_camera: dict[str, int]
    detections: dict[str, list[FiducialDetection]] = field(repr=False, default_factory=dict)

    @property
    def n_cameras(self) -> int:
        return sum(n >= _MIN_CORNERS_PER_CAMERA for n in self.corners_by_camera.values())

    @property
    def n_corners(self) -> int:
        return sum(self.corners_by_camera.values())


@dataclass
class CameraRow:
    label: str
    solved: bool
    n_windows: int       # kept windows in which the camera sees the box
    n_points: int        # box-corner observations with a residual
    median_px: float | None
    p90_px: float | None
    mean_px: float | None
    max_px: float | None
    ok: bool


@dataclass
class BoxVideoResult:
    calibration: CalibResult | None
    rows: list[CameraRow]
    windows: list[Window]            # kept windows, in time order
    dropped: list[tuple[float, str]]  # (timestamp, reason)
    anchor_ts: float | None
    final_ts: float | None
    aligned_to_final: bool
    alignment_rms_m: float | None
    # (box corner name, camera label, px) for every observation, for per-point inspection
    residuals: list[tuple[str, str, float]]
    max_median_error_px: float
    messages: list[str]

    @property
    def ok(self) -> bool:
        return self.calibration is not None and bool(self.rows) and all(r.ok for r in self.rows)


@dataclass
class _Cam:
    label: str
    video_id: str
    file_path: str
    fps: float
    intrinsics_calibration_id: str | None


# ---------------------------------------------------------------------------
# Stationary windows
# ---------------------------------------------------------------------------


def find_stationary_windows(
    video_path: str,
    make_detector: Callable[[], MarkerRigDetector],
    start_frame: int,
    end_frame: int,
    sample_stride: int = 6,
    max_centroid_move_px: float = 6.0,
    min_run_samples: int = 3,
    progress: Callable[[str], None] = lambda _msg: None,
    workers: int | None = None,
) -> list[int]:
    """Frames in the middle of each run where the box's detected corners barely move.

    Motion is measured on the detected marker corners themselves rather than a
    generic blur proxy: the detector whose output is needed anyway decides. A
    sampled frame with no detection breaks a run (the box may have been
    occluded or out of view while it moved), even if the centroid lands in a
    similar place on either side.

    The range is split into about *workers* chunks (default: one per CPU core), each with
    its own decoder that seeks to the chunk's first sample, so decoding runs in parallel
    (FFmpeg's own decoder threads stay off, see ``frame_source``). Chunks start on the
    sample grid, so the samples are those of one sequential pass.
    """
    n_samples = len(range(start_frame, end_frame, sample_stride))
    n_chunks = max(1, min(workers or os.cpu_count() or 1, n_samples))
    bounds = [start_frame + (i * n_samples // n_chunks) * sample_stride for i in range(n_chunks)]
    bounds.append(end_frame)
    done = 0
    lock = threading.Lock()

    def scan_chunk(i: int) -> list[tuple[int, np.ndarray]]:
        nonlocal done
        detector = make_detector()
        found: list[tuple[int, np.ndarray]] = []
        for frame_idx, img in iter_frames(video_path, bounds[i], bounds[i + 1], step=sample_stride):
            dets = detector.detect(img, video_id="__scan__", frame_idx=frame_idx)
            if dets:
                pts = np.array([[c.px, c.py] for d in dets for c in d.corners], dtype=np.float64)
                found.append((frame_idx, pts.mean(axis=0)))
        with lock:
            done += 1
            progress(f"Scanning for stationary windows: {100 * done // n_chunks}%")
        return found

    with ThreadPoolExecutor(max_workers=n_chunks) as pool:
        samples = [s for chunk in pool.map(scan_chunk, range(n_chunks)) for s in chunk]

    runs: list[int] = []
    run_start = None
    for i in range(1, len(samples)):
        gap = (samples[i][0] - samples[i - 1][0]) > sample_stride * 1.5
        moved = np.inf if gap else np.linalg.norm(samples[i][1] - samples[i - 1][1])
        if moved <= max_centroid_move_px:
            if run_start is None:
                run_start = i - 1
        else:
            if run_start is not None and (i - run_start) >= min_run_samples:
                runs.append(samples[(run_start + i - 1) // 2][0])
            run_start = None
    if run_start is not None and (len(samples) - run_start) >= min_run_samples:
        runs.append(samples[(run_start + len(samples) - 1) // 2][0])
    return runs


def _detect_windows(
    times: list[float],
    cams: list[_Cam],
    sync: SyncTable,
    make_detector: Callable[[], MarkerRigDetector],
    workers: int | None = None,
) -> list[dict[str, list[FiducialDetection]]]:
    """Detect the box in every camera at each synced time in *times*.

    Each camera's frames are read in time order through an open container instead of one
    open and seek per window. A camera's frames are split into consecutive chunks so that
    about *workers* decoders run at once (decoding and ArUco detection both release the GIL;
    FFmpeg's own decoder threads stay off, see ``frame_source``). Each entry of the result
    maps camera label to detections, in camera order; a camera whose frame is unavailable
    is missing from it.
    """
    workers = workers or os.cpu_count() or 1
    chunks_per_cam = max(1, workers // len(cams))
    frame_of = [{ts: sync.lookup(ts, cam.video_id) for ts in times} for cam in cams]

    def decode_chunk(task: tuple[_Cam, list[int]]) -> dict[int, list[FiducialDetection]]:
        cam, frames = task
        detector = make_detector()
        return {
            idx: detector.detect(img, video_id=cam.label, frame_idx=idx)
            for idx, img in iter_frames_at(cam.file_path, frames)
        }

    tasks = []
    for cam, lookup in zip(cams, frame_of):
        wanted = sorted({f for f in lookup.values() if f is not None})
        n = min(chunks_per_cam, len(wanted))
        tasks += [(cam, wanted[i * len(wanted) // n:(i + 1) * len(wanted) // n]) for i in range(n)]
    with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as pool:
        chunks = list(pool.map(decode_chunk, tasks))
    by_cam: dict[str, dict[int, list[FiducialDetection]]] = {cam.label: {} for cam in cams}
    for (cam, _), dets in zip(tasks, chunks):
        by_cam[cam.label].update(dets)
    out: list[dict[str, list[FiducialDetection]]] = []
    for i, ts in enumerate(times):
        row: dict[str, list[FiducialDetection]] = {}
        for cam, lookup in zip(cams, frame_of):
            if lookup[ts] in by_cam[cam.label]:
                row[cam.label] = by_cam[cam.label][lookup[ts]]
        out.append(row)
    return out


def _read_window_frames(ts: float, cams: list[_Cam], sync: SyncTable) -> dict[str, np.ndarray]:
    """The frame of every camera at synced time *ts*."""
    frames: dict[str, np.ndarray] = {}
    for cam in cams:
        frame_idx = sync.lookup(ts, cam.video_id)
        if frame_idx is not None:
            frames.update({cam.label: img for _, img in iter_frames_at(cam.file_path, [frame_idx])})
    return frames


def rank_key(w: Window, final_ts: float) -> tuple:
    """Anchor preference: most cameras, then the final rest position, then most corners."""
    return (w.n_cameras, w.timestamp_s == final_ts, w.n_corners)


# ---------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------


def _load_intrinsics(conn: sqlite3.Connection, calib_id: str) -> dict:
    row = conn.execute("SELECT * FROM intrinsics_calibrations WHERE id = ?", (calib_id,)).fetchone()
    if row is None:
        raise BoxVideoError(f"No intrinsics_calibrations row for id {calib_id}")
    K = np.array([[row["fx"], 0.0, row["cx"]], [0.0, row["fy"], row["cy"]], [0.0, 0.0, 1.0]])
    K_orig = K.copy()
    if row["matrix_original"]:
        K_orig = np.array(struct.unpack("<9d", bytes(row["matrix_original"]))).reshape(3, 3)
    if row["dist_coeffs"]:
        n = len(bytes(row["dist_coeffs"])) // 8
        dist = np.array(struct.unpack(f"<{n}d", bytes(row["dist_coeffs"]))).reshape(1, -1)
    else:
        dist = np.zeros((1, 4))
    return {"K": K, "K_orig": K_orig, "dist": dist, "fisheye": row["distortion_model"] == "fisheye"}


def load_capture_cameras(conn: sqlite3.Connection, capture_id: str) -> tuple[list[_Cam], SyncTable]:
    """The capture's videos and the sync table of its first sync configuration."""
    old = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT cv.id, ci.label, cv.file_path, cv.actual_fps, cm.nominal_fps, "
            "       cv.intrinsics_calibration_id "
            "FROM capture_videos cv "
            "JOIN camera_instances ci ON ci.id = cv.camera_instance_id "
            "LEFT JOIN camera_modes cm ON cm.id = cv.camera_mode_id "
            "WHERE cv.shot_id = ? ORDER BY ci.label",
            (capture_id,),
        ).fetchall()
        cams = [
            _Cam(r["label"], r["id"], r["file_path"],
                 float(r["actual_fps"] or r["nominal_fps"] or 120.0), r["intrinsics_calibration_id"])
            for r in rows
        ]
        if not cams:
            raise BoxVideoError("The capture has no videos")
        sync_cfg = conn.execute(
            "SELECT id FROM sync_configs WHERE shot_id = ? ORDER BY rowid", (capture_id,)
        ).fetchone()
        if sync_cfg is None:
            raise BoxVideoError("The capture has no sync configuration; synchronise the videos first")
        points = [
            SyncPoint(camera_instance_id="", shot_video_id=r["shot_video_id"],
                      video_frame=r["video_frame"], timestamp_s=r["timestamp_s"])
            for r in conn.execute(
                "SELECT shot_video_id, video_frame, timestamp_s FROM sync_points "
                "WHERE sync_config_id = ?", (sync_cfg["id"],),
            )
        ]
        unsynced = [c.label for c in cams if not any(p.shot_video_id == c.video_id for p in points)]
        if unsynced:
            raise BoxVideoError(f"No sync points for camera(s): {', '.join(unsynced)}")
        return cams, SyncTable(points, {c.video_id: c.fps for c in cams})
    finally:
        conn.row_factory = old


# ---------------------------------------------------------------------------
# Alignment of the solved scene to the final rest position
# ---------------------------------------------------------------------------


def _rigid_fit(local: np.ndarray, world: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Kabsch: R, t with ``world ≈ R @ local + t`` and the RMS residual."""
    cl, cw = local.mean(axis=0), world.mean(axis=0)
    U, _, Vt = np.linalg.svd((local - cl).T @ (world - cw))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cw - R @ cl
    rms = float(np.sqrt(np.mean(np.sum((world - (local @ R.T + t)) ** 2, axis=1))))
    return R, t, rms


def _align_to_window(
    result: CalibResult,
    window: Window,
    config: MarkerRigConfig,
    states: list[CamCalibState],
) -> float | None:
    """Move the solved scene so that its world frame is the box in *window*.

    Returns the RMS fit residual (m) of the window's triangulated corners against the box
    geometry, or ``None`` when too few corners are seen by two cameras.
    """
    by_id = {s.video_id: s for s in states if s.R is not None}
    obs: dict[tuple[str, int], dict[str, tuple[float, float]]] = {}
    for cam_label, dets in window.detections.items():
        s = by_id.get(cam_label)
        if s is None:
            continue
        for d in dets:
            for c in d.corners:
                u = _undistort_pts(np.array([[c.px, c.py]], dtype=np.float32), s)[0]
                obs.setdefault((d.marker_id, c.corner_index), {})[cam_label] = (float(u[0]), float(u[1]))
    local, world = [], []
    for (marker_id, idx), o in obs.items():
        if len(o) < 2:
            continue
        xyz = triangulate_dlt(o, by_id)
        if xyz is not None:
            local.append(config.marker_corners[marker_id][idx])
            world.append(xyz)
    if len(local) < 4:
        return None
    R, t, rms = _rigid_fit(np.array(local), np.array(world))
    result.points_3d = apply_similarity_transform(states, result.points_3d, 1.0, R.T, -R.T @ t)
    return rms


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_from_box_video(
    conn: sqlite3.Connection,
    capture_id: str,
    config: MarkerRigConfig,
    params: BoxVideoParams,
    progress: Callable[[str], None] = lambda _msg: None,
) -> BoxVideoResult:
    """Solve the extrinsics of every camera of a capture from its box video.

    Nothing is written; the caller stores ``BoxVideoResult.calibration`` only when
    ``BoxVideoResult.ok``.
    """
    config = copy.deepcopy(config)
    if params.world_z_offset:
        for corners in config.marker_corners.values():
            corners[:, 2] += params.world_z_offset

    def make_detector() -> MarkerRigDetector:
        return MarkerRigDetector(
            config, dictionary=params.rig_dict,
            min_marker_perimeter_rate=params.min_marker_perimeter_rate,
        )

    cams, sync = load_capture_cameras(conn, capture_id)
    ref = next((c for c in cams if c.label == params.reference_camera), None)
    if ref is None:
        raise BoxVideoError(
            f"Reference camera {params.reference_camera!r} is not in this capture "
            f"(cameras: {', '.join(c.label for c in cams)})"
        )
    messages: list[str] = []

    # --- stationary windows in the reference camera --------------------------------------
    start = sync.lookup(params.time_start_s, ref.video_id)
    end = sync.lookup(params.time_end_s, ref.video_id)
    if start is None or end is None:
        raise BoxVideoError("Cannot map the time range into the reference camera's frames")
    progress(f"Scanning {ref.label} for stationary windows…")
    frames_found = find_stationary_windows(
        ref.file_path, make_detector, start, end,
        params.sample_stride, params.max_centroid_move_px, params.min_run_samples, progress,
    )
    if not frames_found:
        raise BoxVideoError(
            f"No stationary windows found in {ref.label} between {params.time_start_s} s and "
            f"{params.time_end_s} s; check the time range and the reference camera"
        )
    times = [sync.frame_to_global_time(f, ref.video_id) for f in frames_found]
    dropped: list[tuple[float, str]] = []
    for e in params.exclude_window_ts:
        hit = min(times, key=lambda t: abs(t - e), default=None)
        if hit is not None and abs(hit - e) <= 0.5:
            dropped.append((hit, "excluded by option"))
            times.remove(hit)
    if not times:
        raise BoxVideoError("All stationary windows were excluded")

    # --- detect every window in every camera -----------------------------------------------
    progress(f"Detecting the box in {len(times)} windows in {len(cams)} cameras…")
    windows: list[Window] = []
    for ts, dets in zip(times, _detect_windows(times, cams, sync, make_detector)):
        w = Window(ts, {c: sum(len(d.corners) for d in ds) for c, ds in dets.items()}, dets)
        if w.n_corners < params.min_window_corners:
            dropped.append((ts, f"only {w.n_corners} corners detected"))
        elif w.n_cameras < 2:
            dropped.append((ts, f"seen by {w.n_cameras} camera(s), needs 2"))
        else:
            windows.append(w)
    if not windows:
        raise BoxVideoError("No stationary window has enough box corners in at least two cameras")

    final_ts = windows[-1].timestamp_s
    anchor = max(windows, key=lambda w: rank_key(w, final_ts))
    if anchor.n_cameras < len(cams):
        messages.append(
            f"No window is seen by all {len(cams)} cameras (best: t={anchor.timestamp_s:.2f} s, "
            f"{anchor.n_cameras} cameras)"
        )

    # --- solve ----------------------------------------------------------------------------
    anchor_frames = _read_window_frames(anchor.timestamp_s, cams, sync)
    states = []
    for cam in cams:
        if cam.intrinsics_calibration_id is None:
            raise BoxVideoError(f"{cam.label}: the video has no intrinsics calibration")
        old = conn.row_factory
        conn.row_factory = sqlite3.Row
        try:
            intr = _load_intrinsics(conn, cam.intrinsics_calibration_id)
        finally:
            conn.row_factory = old
        states.append(CamCalibState(
            video_id=cam.label, label=cam.label, K=intr["K"], K_orig=intr["K_orig"],
            dist=intr["dist"], fisheye=intr["fisheye"],
            image=anchor_frames.get(cam.label),
        ))

    anchor_cps = anchor_from_marker_rig(anchor.detections, config)
    groups: dict[str, MarkerGroup] = {}
    for i, w in enumerate(windows):
        if w is anchor:
            continue
        for dets in w.detections.values():
            merge_detections_into_groups(
                [_retag(d, f"win{i}_") for d in dets], groups, size=None,
            )
    progress(f"Solving: {len(anchor_cps)} anchor points, {len(groups)} free marker groups…")
    result = run_calibration(
        states, control_points=anchor_cps, marker_groups=list(groups.values()), cp_only=False,
    )

    # Residuals before the alignment below: it moves the cameras but not the anchor's fixed
    # points, and a rigid move of the whole scene leaves every reprojection error unchanged.
    all_cps = list(anchor_cps) + [cp for g in groups.values() for cp in g.as_control_points()]
    residuals = control_point_residuals(states, all_cps)

    aligned, align_rms = False, None
    if result.unsolved:
        messages.append(f"Unsolved camera(s): {', '.join(result.unsolved)}")
    elif anchor is not windows[-1]:
        align_rms = _align_to_window(result, windows[-1], config, states)
        aligned = align_rms is not None
        if not aligned:
            messages.append(
                f"The final rest position (t={final_ts:.2f} s) has too few corners seen by two "
                f"cameras to align to; the world frame is the box at t={anchor.timestamp_s:.2f} s"
            )

    # --- judge ----------------------------------------------------------------------------
    rows = []
    for cam in cams:
        errs = np.array([e for _, vid, e in residuals if vid == cam.label])
        solved = states[[s.label for s in states].index(cam.label)].R is not None
        n_win = sum(w.corners_by_camera.get(cam.label, 0) >= _MIN_CORNERS_PER_CAMERA for w in windows)
        if len(errs):
            med, p90 = float(np.median(errs)), float(np.percentile(errs, 90))
            ok = (solved and med <= params.max_median_error_px
                  and p90 <= _P90_LIMIT_FACTOR * params.max_median_error_px)
            rows.append(CameraRow(cam.label, solved, n_win, len(errs), med, p90, float(errs.mean()),
                                  float(errs.max()), ok))
        else:
            rows.append(CameraRow(cam.label, solved, n_win, 0, None, None, None, None, False))
    return BoxVideoResult(
        calibration=result, rows=rows, windows=windows, dropped=dropped,
        anchor_ts=anchor.timestamp_s, final_ts=final_ts, aligned_to_final=aligned,
        alignment_rms_m=align_rms, residuals=residuals,
        max_median_error_px=params.max_median_error_px, messages=messages,
    )


def _retag(d: FiducialDetection, prefix: str) -> FiducialDetection:
    mid = prefix + d.marker_id
    return dataclasses.replace(
        d, marker_id=mid,
        corners=[dataclasses.replace(c, marker_id=mid) for c in d.corners],
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def format_report(res: BoxVideoResult) -> str:
    """Windows, the per-camera error table and the verdict, as plain text."""
    lines = ["Stationary windows (anchor *, final rest position F):"]
    for w in res.windows:
        tag = ("*" if w.timestamp_s == res.anchor_ts else " ") + ("F" if w.timestamp_s == res.final_ts else " ")
        cams = "  ".join(f"{c}:{n}" for c, n in sorted(w.corners_by_camera.items()))
        lines.append(f"  {tag} t={w.timestamp_s:7.2f} s  {w.n_cameras} cameras, {w.n_corners} corners  [{cams}]")
    for ts, why in sorted(res.dropped):
        lines.append(f"    t={ts:7.2f} s  dropped: {why}")
    if res.aligned_to_final:
        lines.append(f"World frame aligned to the final rest position (box fit {res.alignment_rms_m * 1000:.1f} mm RMS)")
    lines += [f"NOTE: {m}" for m in res.messages]
    p90_limit = _P90_LIMIT_FACTOR * res.max_median_error_px
    lines += ["", f"{'camera':22s} {'windows':>7s} {'points':>7s} {'median px':>10s} {'p90 px':>8s} "
                  f"{'mean px':>9s} {'max px':>9s}  status"]
    for r in res.rows:
        f = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        if r.ok:
            status = "ok"
        elif not r.solved:
            status = "UNSOLVED"
        elif r.median_px is None:
            status = "NO DATA"
        elif r.median_px > res.max_median_error_px:
            status = f"FAIL (median > {res.max_median_error_px:g})"
        else:
            status = f"FAIL (p90 > {p90_limit:g})"
        lines.append(f"{r.label:22s} {r.n_windows:7d} {r.n_points:7d} {f(r.median_px):>10s} "
                     f"{f(r.p90_px):>8s} {f(r.mean_px):>9s} {f(r.max_px):>9s}  {status}")
    lines.append("")
    lines.append("Calibration OK" if res.ok else
                 f"Calibration FAILED: a camera is unsolved, has no box observations, or its median "
                 f"error exceeds {res.max_median_error_px:g} px or its 90th-percentile error "
                 f"{p90_limit:g} px; nothing is written")
    return "\n".join(lines)

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""video_marker_body.py — solve a rigid marker body from one moving camera.

A single camera moves around a stationary prop that carries ArUco markers and,
optionally, reflective dots. Around the prop stand further ArUco markers, the
anchors, which are not part of it. They only serve to track the camera when the
prop's own markers face away, and their sizes may differ from the prop's. A prop
whose markers cover all its sides needs no anchors.

The method is one joint least-squares fit (bundle adjustment) of every marker's pose
and the camera's pose in every sampled frame, to the marker corners seen:

* The reference marker, one of the prop's, defines the frame the result is written
  in. Fixing its pose removes the only freedom the fit has.
* The known side length of every marker fixes the scale.
* Markers are linked by being seen in the same frame: two markers seen together
  constrain each other's pose, and a chain of such pairs, through anchors as well as
  prop markers, places every marker relative to the reference. A prop marker that no
  chain connects to the reference is left out and reported.

The starting point of the fit is built from the same links: each marker's pose is
measured in every frame, the most representative measurement of each pair of markers
is chained outward from the reference, and the camera pose of a frame follows from
any marker in it. The fit is robust (a soft loss, then a second pass without the
detections that still fit badly), so a wrong corner or a flipped planar-pose estimate
does not spoil it.

Reflective dots follow from the camera track: they are stationary too, so each is a
point seen from many known camera poses. A dot candidate carries no identity, so
candidates are first linked from frame to frame into tracklets, each tracklet is
triangulated on its own, and the triangulated points are clustered into physical dots.
Every dot is then matched again in all frames by projecting it, and triangulated from
all its matches. Three things keep bright things that are not dots of the prop out. Only
dots within a set distance of the reference marker are kept, since a camera orbit also
sees the rest of the room. Candidates inside a marker's quad are ignored, because the
white cells of a printed marker are bright too; the quad is grown only a little, as a
prop's dots are often mounted right beside a marker. And a dot whose views do not agree
on one point is dropped. How well is judged against the camera track itself, since the
track's error is in every view: by default a dot must reproject within 0.8 times the markers'
own reprojection error (and at least 1 pixel). On the sword harness, whose track fit to 1.3
pixels, the dots of the prop reproject within 0.6 to 0.8 pixels and marker cells and
reflections at 1.8 or more; on a ball whose track fit to 3 pixels, its dots come out between
1.1 and 2.0. Background subtraction is not used: it needs a stationary camera.

The stages can be used and tested apart: :func:`collect_video_observations` decodes the
video, :func:`solve_video_body` does the geometry, and
:func:`calibrate_marker_body_from_video` runs both from a video and the camera's intrinsics.
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from app.setup.extrinsics_solver import CamCalibState, _undistort_pts, marker_local_corners
from app.setup.fiducial_markers import ArucoDetector
from posetrak.calibration.rigid_marker_body import cluster_dot_samples, format_marker_body_yaml
from posetrak.calibration.session_cameras import Intrinsics
from posetrak.detection.dot_blob_detector import detect_blobs
from posetrak.detection.dot_tracklet import MotionGatedLinker
from posetrak.detection.frame_source import iter_frames

# Two measurements of a marker pair differ in translation (metres) and rotation (radians);
# this weighs radians against metres when picking the most representative one.
_ROTATION_WEIGHT_M_PER_RAD = 0.1
_MAX_EDGE_MEASUREMENTS = 300
_OUTLIER_FLOOR_PX = 3.0
_DOT_RMS_RATIO = 0.8        # by default a dot must reproject within this fraction of the markers' error ...
_MIN_DOT_RMS_PX = 1.0       # ... but a track better than this still allows a dot this much
_ROUGH_DISTANCE_FACTOR = 2.0   # the distance limit is looser for the first triangulation of a dot from one tracklet
_SIZE_WARNING = 0.05      # a marker whose fitted size differs from the typical one by more than this is reported


@dataclass
class VideoCalibrationOptions:
    """What to calibrate from a video, and how.

    ``body_markers``, ``reference_id`` and ``anchor_markers`` say what is in the scene;
    the rest are settings with working defaults.
    """

    body_markers: dict[str, float]           # marker id -> side length, metres; the prop's markers
    reference_id: str                        # one of the body markers; its frame is the result's frame
    anchor_markers: dict[str, float] = field(default_factory=dict)   # markers around the prop
    dictionary: str = "DICT_4X4_50"
    name: str = "calibrated-rigid-body"
    stride: int = 6                          # process every Nth decoded frame
    first_frame: int | None = None           # default: the start of the video
    last_frame: int | None = None            # exclusive; default: the end of the video
    min_frame_markers: int = 2               # a frame needs this many known markers to be used
    max_pnp_rms_px: float = 3.0              # a marker pose that reprojects worse than this is dropped
    detect_dots: bool = False
    dot_threshold: int = 235
    dot_min_area: float = 4.0
    dot_max_area: float = 400.0
    dot_min_compactness: float = 0.5
    dot_max_distance_m: float = 1.0          # keep dots this close to the reference marker's centre
    dot_min_views: int = 12                  # frames a dot must be matched in
    dot_min_parallax_deg: float = 2.0        # least angle between the rays that see a tracklet
    dot_max_reprojection_px: float = 3.0
    dot_cluster_tolerance_m: float = 0.02
    dot_match_gate_px: float = 8.0
    dot_marker_exclude_scale: float = 1.15   # ignore candidates inside a marker's quad grown by this factor
    dot_max_rms_px: float | None = None      # drop a dot whose views disagree more than this; None: see below

    def validate(self) -> None:
        """Raise ValueError if the scene description or a setting is unusable."""
        if self.reference_id not in self.body_markers:
            raise ValueError("the reference id must be one of the body markers")
        both = sorted(set(self.body_markers) & set(self.anchor_markers))
        if both:
            raise ValueError(f"marker ids cannot be both body and anchor markers: {', '.join(both)}")
        if any(size <= 0 for size in self.marker_sizes().values()):
            raise ValueError("marker sizes must be positive")
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        if self.min_frame_markers < 2:
            raise ValueError("min_frame_markers must be at least 2: a frame with one marker fixes no relation")

    def marker_sizes(self) -> dict[str, float]:
        """Side length of every marker the scene has, prop and anchors."""
        return {**self.anchor_markers, **self.body_markers}


@dataclass
class FrameObservation:
    """What one processed frame showed: the known markers and the dot candidates.

    All pixel positions are undistorted.
    """

    frame: int
    markers: dict[str, np.ndarray]                       # marker id -> (4, 2) corners
    # (u, v, tracklet id, marker scale) per dot candidate. The marker scale is the factor by
    # which the nearest marker's quad must be grown, about its centre, to contain the candidate
    # (below 1: inside the marker; infinite: no marker in the frame).
    dots: list[tuple[float, float, int, float]] = field(default_factory=list)


@dataclass
class MarkerStats:
    """How well the footage supported one marker."""

    frames: int
    rms_px: float


@dataclass
class DotStats:
    """A solved dot and what supports it."""

    center: np.ndarray
    views: int
    rms_px: float


@dataclass
class VideoCalibrationResult:
    """A solved marker body and how well the footage supported it."""

    yaml: str
    frames_used: int
    reprojection_rms_px: float
    marker_stats: dict[str, MarkerStats]
    unconnected: list[str]                   # body markers no chain of shared frames links to the reference
    dots: list[DotStats] = field(default_factory=list)
    # Fitted size relative to the given one, for every marker, relative to the typical marker; empty
    # when fewer than three markers were fitted, since sizes can only be compared with each other.
    size_ratios: dict[str, float] = field(default_factory=dict)


def parse_marker_sizes(spec: str, default_size: float | None, what: str) -> dict[str, float]:
    """Read ``2,3:0.06`` into ``{"2": default_size, "3": 0.06}``.

    Parameters
    ----------
    spec:
        Comma-separated marker ids, each optionally followed by ``:`` and its side
        length in metres.
    default_size:
        Side length of the ids that give none.
    what:
        What the ids are, for error messages ("body markers").

    Raises
    ------
    ValueError
        If an id has no size and there is no default, a size is not a number, or an id
        is listed twice.
    """
    sizes: dict[str, float] = {}
    for item in (part.strip() for part in spec.split(",")):
        if not item:
            continue
        marker_id, _, size_text = item.partition(":")
        marker_id = marker_id.strip()
        try:
            size = float(size_text) if size_text else default_size
        except ValueError:
            raise ValueError(f"{what}: size of marker {marker_id!r} is not a number: {size_text!r}") from None
        if size is None:
            raise ValueError(f"{what}: marker {marker_id!r} has no size; give ID:SIZE or a default size")
        if marker_id in sizes:
            raise ValueError(f"{what}: marker {marker_id!r} is listed twice")
        sizes[marker_id] = size
    return sizes


def calibrate_marker_body_from_video(
    video_path: str,
    intrinsics: Intrinsics,
    options: VideoCalibrationOptions,
    *,
    log: Callable[[str], None] = lambda message: None,
) -> VideoCalibrationResult:
    """Solve a marker body from a video of one moving camera.

    Parameters
    ----------
    video_path:
        The video of the orbit.
    intrinsics:
        The intrinsics calibration of the camera, in the mode that filmed the video. The
        video is not a capture of its own, so there are no extrinsics to look up.
    options:
        What to calibrate and how.
    log:
        Called with a progress line at a time.

    Returns
    -------
    VideoCalibrationResult
        The marker body YAML and statistics.

    Raises
    ------
    ValueError
        If *options* are invalid, the video cannot be opened or is not the image size the
        intrinsics were calibrated for, or the footage does not support a solution (see
        :func:`solve_video_body`).
    """
    options.validate()
    width, height = _video_size(video_path)
    if width == 0:
        raise ValueError(f"cannot open the video {video_path!r}")
    if None not in (intrinsics.image_width, intrinsics.image_height) and (
        (width, height) != (intrinsics.image_width, intrinsics.image_height)
    ):
        raise ValueError(
            f"the video is {width}x{height} but intrinsics calibration {intrinsics.calibration_id[:8]} is for "
            f"{intrinsics.image_width}x{intrinsics.image_height}; use the calibration of the camera in the "
            "mode that filmed the video"
        )
    frames = collect_video_observations(video_path, intrinsics.state, options, log=log)
    return solve_video_body(frames, intrinsics.state, options, log=log)


def _video_size(video_path: str) -> tuple[int, int]:
    capture = cv2.VideoCapture(video_path)
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()


def collect_video_observations(
    video_path: str,
    state: CamCalibState,
    options: VideoCalibrationOptions,
    *,
    detector=None,
    log: Callable[[str], None] = lambda message: None,
) -> list[FrameObservation]:
    """Decode the video and collect the known markers and dot candidates of each sampled frame.

    Parameters
    ----------
    video_path:
        The video.
    state:
        The camera's intrinsics, to undistort the pixel positions.
    options:
        Which markers to look for, the frame range and stride, and the dot settings.
    detector:
        A marker detector with ``detect(image, video_id, frame_idx)``; an ``ArucoDetector``
        for ``options.dictionary`` by default.
    log:
        Called with a progress line at a time.

    Returns
    -------
    list[FrameObservation]
        In frame order, only those in which at least ``options.min_frame_markers`` known
        markers were seen. Dot candidates are linked into tracklets over every sampled frame.
    """
    detector = detector or ArucoDetector(dictionary=options.dictionary)
    known = set(options.marker_sizes())
    first = options.first_frame or 0
    last = options.last_frame if options.last_frame is not None else _frame_count(video_path)
    linker = MotionGatedLinker() if options.detect_dots else None

    frames: list[FrameObservation] = []
    n_sampled = 0
    log(f"Decoding {video_path}: frames {first}-{last}, every {options.stride}th")
    for video_frame, img in iter_frames(video_path, first, last):
        if (video_frame - first) % options.stride != 0:
            continue
        n_sampled += 1
        markers = {}
        detections = detector.detect(img, video_id=state.video_id, frame_idx=video_frame)
        for d in detections:
            if d.marker_id in known:
                pts = np.array([(c.px, c.py) for c in d.corners], dtype=np.float64)
                markers[d.marker_id] = _undistort_pts(pts, state)
        dots: list[tuple[float, float, int, float]] = []
        if linker is not None:
            blobs = detect_blobs(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), threshold=options.dot_threshold,
                min_area=options.dot_min_area, max_area=options.dot_max_area,
                min_compactness=options.dot_min_compactness,
            )
            linker.link_frame(video_frame, blobs)
            if blobs:
                raw = np.array([(b.cx, b.cy) for b in blobs], dtype=np.float64)
                quads = [np.array([(c.px, c.py) for c in d.corners], dtype=np.float64) for d in detections]
                scales = _marker_scales(raw, quads)
                dots = [(float(u), float(v), int(b.tracklet_id), float(s))
                        for (u, v), b, s in zip(_undistort_pts(raw, state), blobs, scales)]
        if len(markers) >= options.min_frame_markers:
            frames.append(FrameObservation(video_frame, markers, dots))
        if n_sampled % 200 == 0:
            log(f"  {n_sampled} frames sampled, {len(frames)} usable")
    log(f"{n_sampled} frames sampled, {len(frames)} show at least {options.min_frame_markers} known markers")
    return frames


def _marker_scales(points: np.ndarray, quads: list[np.ndarray]) -> np.ndarray:
    """For each point, the factor by which the nearest of the convex *quads*, grown about its
    centre, just contains it: below 1 for a point inside a quad, infinite without quads.

    A dot detector also finds the bright cells of a printed marker. Such a point is inside its
    marker's quad, and one just beside a marker, like a dot mounted next to it, is not.
    """
    scales = np.full(len(points), np.inf)
    for quad in quads:
        centre = quad.mean(axis=0)
        worst = np.full(len(points), -np.inf)
        for i in range(4):
            edge = quad[(i + 1) % 4] - quad[i]
            normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
            if normal @ (quad[i] - centre) < 0:
                normal = -normal
            worst = np.maximum(worst, (points - centre) @ normal / (normal @ (quad[i] - centre)))
        scales = np.minimum(scales, worst)
    return scales


def _frame_count(video_path: str) -> int:
    capture = cv2.VideoCapture(video_path)
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


# --------------------------------------------------------------------------- geometry


def _rotations(rvecs: np.ndarray) -> np.ndarray:
    """Rotation matrices (N, 3, 3) of rotation vectors (N, 3)."""
    theta = np.linalg.norm(rvecs, axis=1)
    axis = rvecs / np.where(theta > 1e-12, theta, 1.0)[:, None]
    skew = np.zeros((len(rvecs), 3, 3))
    skew[:, 0, 1], skew[:, 0, 2] = -axis[:, 2], axis[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = axis[:, 2], -axis[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -axis[:, 1], axis[:, 0]
    sin, one_minus_cos = np.sin(theta)[:, None, None], (1.0 - np.cos(theta))[:, None, None]
    return np.eye(3) + sin * skew + one_minus_cos * (skew @ skew)


def _rvec(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(R)[0].ravel()


def _marker_pose_in_camera(
    corners_px: np.ndarray, size: float, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Pose of a square marker from its four undistorted corners: (R, t, reprojection rms in px)."""
    local = marker_local_corners(size)
    ok, rvec, tvec = cv2.solvePnP(local, corners_px, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    projected, _ = cv2.projectPoints(local, rvec, tvec, K, None)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - corners_px) ** 2, axis=1))))
    return cv2.Rodrigues(rvec)[0], tvec.ravel(), rms


@dataclass
class _Measurements:
    """Per-frame marker poses in the camera, and the pairwise poses they imply."""

    in_camera: dict[int, dict[str, tuple[np.ndarray, np.ndarray, float]]]      # frame index -> marker -> (R, t, rms)
    pairs: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]          # (a, b), a < b -> pose of b in a


def _measure(frames: list[FrameObservation], sizes: dict[str, float], K: np.ndarray, max_rms_px: float) -> _Measurements:
    in_camera: dict[int, dict[str, tuple[np.ndarray, np.ndarray, float]]] = {}
    pairs: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for index, frame in enumerate(frames):
        poses = {}
        for marker_id, corners in frame.markers.items():
            pose = _marker_pose_in_camera(corners, sizes[marker_id], K)
            if pose is not None and pose[2] <= max_rms_px:
                poses[marker_id] = pose
        in_camera[index] = poses
        for a in sorted(poses):
            for b in sorted(poses):
                if a < b:
                    Ra, ta, _ = poses[a]
                    Rb, tb, _ = poses[b]
                    pairs[(a, b)].append((Ra.T @ Rb, Ra.T @ (tb - ta)))
    return _Measurements(in_camera, pairs)


def _representative(poses: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """The measurement closest to all the others: robust against flipped planar poses and bad corners."""
    if len(poses) > _MAX_EDGE_MEASUREMENTS:
        poses = [poses[i] for i in np.linspace(0, len(poses) - 1, _MAX_EDGE_MEASUREMENTS).astype(int)]
    features = np.array([np.concatenate([t, _ROTATION_WEIGHT_M_PER_RAD * _rvec(R)]) for R, t in poses])
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2).sum(axis=1)
    return poses[int(np.argmin(distances))]


def _chain_from_reference(pairs, reference_id: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Initial pose (R, t) of every marker reachable from the reference, in the reference's frame.

    Follows the maximum spanning tree of the graph whose edges are the pairs of markers seen
    together, weighted by how often, so the best observed links carry the chain.
    """
    neighbours: dict[str, dict[str, int]] = defaultdict(dict)
    for (a, b), poses in pairs.items():
        neighbours[a][b] = neighbours[b][a] = len(poses)
    placed = {reference_id: (np.eye(3), np.zeros(3))}
    queue = [(-count, reference_id, other) for other, count in neighbours[reference_id].items()]
    heapq.heapify(queue)
    while queue:
        _, a, b = heapq.heappop(queue)
        if b in placed:
            continue
        R_ab, t_ab = _representative(pairs[(a, b)]) if a < b else _inverse(*_representative(pairs[(b, a)]))
        Ra, ta = placed[a]
        placed[b] = (Ra @ R_ab, Ra @ t_ab + ta)
        for other, count in neighbours[b].items():
            if other not in placed:
                heapq.heappush(queue, (-count, b, other))
    return placed


def _inverse(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return R.T, -R.T @ t


def _initial_camera_poses(
    measurements: _Measurements, marker_poses: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """World-to-camera pose of each frame that sees a placed marker, from the marker it fits best."""
    cameras = {}
    for index, poses in measurements.in_camera.items():
        candidates = [(rms, m) for m, (_, _, rms) in poses.items() if m in marker_poses]
        if candidates:
            _, marker = min(candidates)
            R_cm, t_cm, _ = poses[marker]
            R_wm, t_wm = marker_poses[marker]
            R_cw = R_cm @ R_wm.T
            cameras[index] = (R_cw, t_cm - R_cw @ t_wm)
    return cameras


@dataclass
class _Fit:
    marker_poses: dict[str, tuple[np.ndarray, np.ndarray]]      # marker -> (R, t): marker frame to reference frame
    camera_poses: dict[int, tuple[np.ndarray, np.ndarray]]      # frame index -> (R, t): reference frame to camera
    detections: list[tuple[int, str]]                           # the (frame index, marker) pairs that were fitted
    residuals_px: np.ndarray                                    # (N, 4, 2) observed minus projected
    size_ratios: dict[str, float] = field(default_factory=dict)  # only with free sizes: fitted size / given size


def _bundle_adjust(
    detections: list[tuple[int, str]],
    frames: list[FrameObservation],
    sizes: dict[str, float],
    K: np.ndarray,
    marker_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    camera_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    reference_id: str,
    free_sizes: bool = False,
) -> _Fit:
    """Fit marker and camera poses to the observed corners, the reference marker held fixed.

    With *free_sizes* every marker's side length is fitted too, as a factor on the given one.
    The factors are only determined up to a common scale, which a weak constraint fixes (their
    logarithms average to zero); a caller compares them with each other.
    """
    free_markers = sorted(m for m in {m for _, m in detections} if m != reference_id)
    frame_indices = sorted({f for f, _ in detections})
    marker_slot = {m: i for i, m in enumerate(free_markers)}
    frame_slot = {f: i for i, f in enumerate(frame_indices)}
    n_marker_params = 6 * len(free_markers)
    n_camera_params = 6 * len(frame_indices)
    size_markers = sorted({m for _, m in detections}) if free_sizes else []
    size_slot = {m: i for i, m in enumerate(size_markers)}

    observed = np.array([frames[f].markers[m] for f, m in detections])                       # (N, 4, 2)
    local = np.array([marker_local_corners(sizes[m]) for _, m in detections])                # (N, 4, 3)
    det_frame = np.array([frame_slot[f] for f, _ in detections])
    det_marker = np.array([marker_slot.get(m, -1) for _, m in detections])                   # -1: the reference
    det_size = np.array([size_slot.get(m, 0) for _, m in detections])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def unpack(x: np.ndarray):
        markers = x[:n_marker_params].reshape(-1, 6)
        cameras = x[n_marker_params: n_marker_params + n_camera_params].reshape(-1, 6)
        log_sizes = x[n_marker_params + n_camera_params:]
        return markers, cameras, log_sizes

    def project(x: np.ndarray) -> np.ndarray:
        markers, cameras, log_sizes = unpack(x)
        R_m = np.concatenate([np.eye(3)[None], _rotations(markers[:, :3])]) if len(markers) else np.eye(3)[None]
        t_m = np.concatenate([np.zeros((1, 3)), markers[:, 3:]]) if len(markers) else np.zeros((1, 3))
        R_c, t_c = _rotations(cameras[:, :3]), cameras[:, 3:]
        m = det_marker + 1
        corners = local * np.exp(log_sizes[det_size])[:, None, None] if free_sizes else local
        world = np.einsum("nij,nkj->nki", R_m[m], corners) + t_m[m][:, None, :]
        cam = np.einsum("nij,nkj->nki", R_c[det_frame], world) + t_c[det_frame][:, None, :]
        z = np.where(cam[..., 2] > 1e-6, cam[..., 2], 1e-6)
        return np.stack([fx * cam[..., 0] / z + cx, fy * cam[..., 1] / z + cy], axis=-1)

    def residuals(x: np.ndarray) -> np.ndarray:
        out = (observed - project(x)).ravel()
        if free_sizes:
            out = np.append(out, 1e3 * unpack(x)[2].mean())       # fixes the common scale of the sizes
        return out

    x0 = np.concatenate(
        [np.concatenate([_rvec(marker_poses[m][0]), marker_poses[m][1]]) for m in free_markers]
        + [np.concatenate([_rvec(camera_poses[f][0]), camera_poses[f][1]]) for f in frame_indices]
        + [np.zeros(len(size_markers))]
    )
    sparsity = lil_matrix((8 * len(detections) + (1 if free_sizes else 0), len(x0)), dtype=int)
    for i in range(len(detections)):
        rows = slice(8 * i, 8 * i + 8)
        if det_marker[i] >= 0:
            sparsity[rows, 6 * det_marker[i]: 6 * det_marker[i] + 6] = 1
        start = n_marker_params + 6 * det_frame[i]
        sparsity[rows, start: start + 6] = 1
        if free_sizes:
            sparsity[rows, n_marker_params + n_camera_params + det_size[i]] = 1
    if free_sizes:
        sparsity[8 * len(detections), n_marker_params + n_camera_params:] = 1
    solution = least_squares(residuals, x0, jac_sparsity=sparsity, loss="soft_l1", f_scale=2.0,
                             x_scale="jac", method="trf", max_nfev=60)

    markers, cameras, log_sizes = unpack(solution.x)
    fitted_markers = {reference_id: (np.eye(3), np.zeros(3))}
    for m, i in marker_slot.items():
        fitted_markers[m] = (_rotations(markers[i:i + 1, :3])[0], markers[i, 3:])
    fitted_cameras = {f: (_rotations(cameras[i:i + 1, :3])[0], cameras[i, 3:]) for f, i in frame_slot.items()}
    return _Fit(fitted_markers, fitted_cameras, detections, observed - project(solution.x),
                {m: float(np.exp(log_sizes[i])) for m, i in size_slot.items()})


# ------------------------------------------------------------------------------ solving


def solve_video_body(
    frames: list[FrameObservation],
    state: CamCalibState,
    options: VideoCalibrationOptions,
    *,
    log: Callable[[str], None] = lambda message: None,
) -> VideoCalibrationResult:
    """Solve the body's geometry from the observations of a moving camera.

    Parameters
    ----------
    frames:
        From :func:`collect_video_observations`.
    state:
        The camera; only its intrinsics are used.
    options:
        Which markers are the prop's, which is the reference, and the dot settings.
    log:
        Called with a progress line at a time.

    Returns
    -------
    VideoCalibrationResult
        The body as YAML: the reference marker first, then the other prop markers in the
        order of ``options.body_markers``, then the dots. Prop markers the footage does not
        link to the reference are left out and listed in ``unconnected``.

    Raises
    ------
    ValueError
        If no frame shows the reference marker together with another known marker, or too
        few frames remain for a fit.
    """
    options.validate()
    sizes = options.marker_sizes()
    ref_id = options.reference_id
    K = state.K

    measurements = _measure(frames, sizes, K, options.max_pnp_rms_px)
    placed = _chain_from_reference(measurements.pairs, ref_id)
    if len(placed) < 2:
        raise ValueError(
            f"the reference marker {ref_id!r} was never seen in a frame together with another known marker"
        )
    log(f"{len(placed)} markers linked to the reference: {', '.join(sorted(placed))}")
    unconnected = [m for m in options.body_markers if m not in placed]
    for marker_id in unconnected:
        log(f"  body marker '{marker_id}': not linked to the reference by any shared frame, left out")

    cameras = _initial_camera_poses(measurements, placed)
    detections = [(f, m) for f, poses in measurements.in_camera.items() for m in poses if m in placed and f in cameras]
    detections = _drop_thin_frames(detections, options.min_frame_markers)
    if not detections:
        raise ValueError("no frame shows enough linked markers for a fit")

    fit = _bundle_adjust(detections, frames, sizes, K, placed, cameras, ref_id)
    rms = np.sqrt(np.mean(np.sum(fit.residuals_px ** 2, axis=2), axis=1))
    keep = rms <= max(_OUTLIER_FLOOR_PX, 3.0 * float(np.median(rms)))
    if not keep.all():
        log(f"  dropping {int((~keep).sum())} of {len(keep)} marker sightings that fit badly, fitting again")
        detections = _drop_thin_frames([d for d, k in zip(fit.detections, keep) if k], options.min_frame_markers)
        fit = _bundle_adjust(detections, frames, sizes, K, fit.marker_poses, fit.camera_poses, ref_id)
        rms = np.sqrt(np.mean(np.sum(fit.residuals_px ** 2, axis=2), axis=1))

    stats = {}
    for marker_id in fit.marker_poses:
        mask = np.array([m == marker_id for _, m in fit.detections])
        stats[marker_id] = MarkerStats(int(mask.sum()), float(rms[mask].mean()) if mask.any() else float("nan"))
    total_rms = float(np.sqrt(np.mean(rms ** 2)))
    log(f"Fitted {len(fit.camera_poses)} frames and {len(fit.marker_poses)} markers, "
        f"reprojection rms {total_rms:.2f} px")

    size_ratios = _check_sizes(fit, frames, sizes, K, ref_id, log)

    dots = _solve_dots(frames, fit, K, options, total_rms, log) if options.detect_dots else []

    body_ids = [ref_id] + [m for m in options.body_markers if m != ref_id and m in fit.marker_poses]
    markers = []
    for marker_id in body_ids:
        R, t = fit.marker_poses[marker_id]
        markers.append((marker_id, sizes[marker_id], (R @ marker_local_corners(sizes[marker_id]).T).T + t))
    return VideoCalibrationResult(
        yaml=format_marker_body_yaml(options.name, options.dictionary, markers, [d.center for d in dots]),
        frames_used=len(fit.camera_poses),
        reprojection_rms_px=total_rms,
        marker_stats={m: stats[m] for m in body_ids},
        unconnected=unconnected,
        dots=dots,
        size_ratios=size_ratios,
    )


def _check_sizes(
    fit: _Fit, frames: list[FrameObservation], sizes: dict[str, float], K: np.ndarray, reference_id: str,
    log: Callable[[str], None],
) -> dict[str, float]:
    """Fit every marker's size too, and warn about a marker whose size disagrees with the others.

    A marker's size is set by hand, and easy to get wrong: the side that counts is the outer edge
    of the black border, not the printed sheet or the pattern inside it. A wrong size makes the
    marker look nearer or farther than the others, which no pose can explain, so the fit is
    poor and the marker's place in the result is off by a proportional amount. Only the sizes
    relative to each other can be judged, so each is compared with the typical marker.
    """
    if len(fit.marker_poses) < 3:
        return {}
    free = _bundle_adjust(fit.detections, frames, sizes, K, fit.marker_poses, fit.camera_poses, reference_id,
                          free_sizes=True)
    typical = float(np.median(list(free.size_ratios.values())))
    ratios = {m: r / typical for m, r in free.size_ratios.items()}
    for marker_id, ratio in sorted(ratios.items()):
        if abs(ratio - 1.0) > _SIZE_WARNING:
            log(f"  warning: the footage fits marker '{marker_id}' best at {ratio:.2f} times the size given "
                f"({sizes[marker_id] * ratio:.4f} m instead of {sizes[marker_id]:.4f} m), relative to the "
                "other markers. Measure the outer edge of its black border.")
    return ratios


def _drop_thin_frames(detections: list[tuple[int, str]], min_markers: int) -> list[tuple[int, str]]:
    counts: dict[int, int] = defaultdict(int)
    for f, _ in detections:
        counts[f] += 1
    return [(f, m) for f, m in detections if counts[f] >= min_markers]


# ------------------------------------------------------------------------------- dots


def _triangulate(
    poses: list[tuple[np.ndarray, np.ndarray]],
    pixels: np.ndarray,
    K: np.ndarray,
    max_reprojection_px: float,
    min_views: int,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Triangulate a static point from several views, dropping the worst view until all fit.

    Returns
    -------
    tuple | None
        ``(point, inlier mask over the views, rms reprojection in px)``, or None if fewer than
        *min_views* views agree, the system is degenerate, or the point is behind a camera.
    """
    P = np.array([K @ np.hstack([R, t[:, None]]) for R, t in poses])
    keep = np.ones(len(poses), dtype=bool)
    while keep.sum() >= max(2, min_views):
        rows = np.concatenate([
            np.stack([pixels[i, 0] * P[i, 2] - P[i, 0], pixels[i, 1] * P[i, 2] - P[i, 1]]) for i in np.flatnonzero(keep)
        ])
        x = np.linalg.svd(rows)[2][-1]
        if abs(x[3]) < 1e-12:
            return None
        point = x[:3] / x[3]
        homogeneous = P @ np.append(point, 1.0)
        depth = homogeneous[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            errors = np.linalg.norm(homogeneous[:, :2] / depth[:, None] - pixels, axis=1)
        errors = np.where(depth > 0, errors, np.inf)
        worst = int(np.argmax(np.where(keep, errors, -1.0)))
        if errors[worst] <= max_reprojection_px:
            return point, keep, float(np.sqrt(np.mean(errors[keep] ** 2)))
        keep[worst] = False
    return None


def _parallax_deg(poses: list[tuple[np.ndarray, np.ndarray]], point: np.ndarray) -> float:
    """Largest angle between the rays from the camera centres to *point*."""
    rays = np.array([point + R.T @ t for R, t in poses])                 # camera centre is -R^T t
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    return float(np.degrees(np.arccos(np.clip((rays @ rays.T).min(), -1.0, 1.0))))


def _solve_dots(
    frames: list[FrameObservation], fit: _Fit, K: np.ndarray, options: VideoCalibrationOptions,
    fit_rms_px: float, log: Callable[[str], None],
) -> list[DotStats]:
    """Triangulate the prop's dots from the camera track.

    Tracklets give a first point each; clustering turns the points into physical dots;
    matching each dot against every frame's candidates by projection and triangulating
    again gives the result. See the module docstring.
    """
    used = sorted(fit.camera_poses)
    poses = [fit.camera_poses[f] for f in used]
    max_dot_rms = (
        options.dot_max_rms_px if options.dot_max_rms_px is not None
        else max(_MIN_DOT_RMS_PX, _DOT_RMS_RATIO * fit_rms_px)
    )
    usable = [[d for d in frames[f].dots if d[3] >= options.dot_marker_exclude_scale] for f in used]
    candidates = [np.array([(u, v) for u, v, _, _ in dots], dtype=np.float64).reshape(-1, 2) for dots in usable]

    tracklets: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for position, dots in enumerate(usable):
        for u, v, tracklet, _ in dots:
            if tracklet >= 0:
                tracklets[tracklet].append((position, np.array([u, v])))

    points = []
    for views in tracklets.values():
        if len(views) < options.dot_min_views:
            continue
        view_poses = [poses[p] for p, _ in views]
        solved = _triangulate(view_poses, np.array([uv for _, uv in views]), K,
                              options.dot_max_reprojection_px, options.dot_min_views)
        if solved is None:
            continue
        point, inliers, _ = solved
        # A tracklet is short, so its depth is still uncertain: only the final position is held to the
        # distance limit, and this first, rough position to a looser one.
        if np.linalg.norm(point) > _ROUGH_DISTANCE_FACTOR * options.dot_max_distance_m:
            continue
        if _parallax_deg([p for p, k in zip(view_poses, inliers) if k], point) < options.dot_min_parallax_deg:
            continue
        points.append(point)
    log(f"Dots: {len(tracklets)} tracklets, {len(points)} triangulated near the prop")

    centers = [np.mean(cluster, axis=0) for cluster in cluster_dot_samples(points, options.dot_cluster_tolerance_m)]
    results: list[DotStats] = []
    for _ in range(2):
        matches = _match_dots(centers, poses, candidates, K, options.dot_match_gate_px)
        centers, results = [], []
        for views in matches:
            if len(views) < options.dot_min_views:
                continue
            solved = _triangulate([poses[p] for p, _ in views], np.array([uv for _, uv in views]), K,
                                  options.dot_max_reprojection_px, options.dot_min_views)
            if solved is None or np.linalg.norm(solved[0]) > _ROUGH_DISTANCE_FACTOR * options.dot_max_distance_m:
                continue
            point, inliers, dot_rms = solved
            centers.append(point)
            results.append(DotStats(point, int(inliers.sum()), dot_rms))
        centers, results = _merge_duplicates(centers, results, options.dot_cluster_tolerance_m)
    # Only now are the positions accurate enough to be held to the limits.
    results = [d for d in results if np.linalg.norm(d.center) <= options.dot_max_distance_m and d.rms_px <= max_dot_rms]
    results.sort(key=lambda d: tuple(np.round(d.center, 3)))
    log(f"  {len(results)} dots solved: " + ", ".join(f"{d.views} views" for d in results))
    return results


def _match_dots(
    centers: list[np.ndarray], poses: list[tuple[np.ndarray, np.ndarray]], candidates: list[np.ndarray],
    K: np.ndarray, gate_px: float,
) -> list[list[tuple[int, np.ndarray]]]:
    """For every dot, the (frame position, candidate pixel) pairs where a candidate lies at its projection.

    Within a frame the closest pairs win, and a candidate is given to one dot only.
    """
    matches: list[list[tuple[int, np.ndarray]]] = [[] for _ in centers]
    if not centers:
        return matches
    world = np.array(centers)
    for position, ((R, t), cands) in enumerate(zip(poses, candidates)):
        if len(cands) == 0:
            continue
        cam = world @ R.T + t
        in_front = cam[:, 2] > 1e-6
        z = np.where(in_front, cam[:, 2], 1.0)
        projected = np.stack([K[0, 0] * cam[:, 0] / z + K[0, 2], K[1, 1] * cam[:, 1] / z + K[1, 2]], axis=1)
        distance = np.linalg.norm(projected[:, None, :] - cands[None, :, :], axis=2)
        pairs = sorted(
            (distance[d, c], d, c) for d in np.flatnonzero(in_front) for c in np.flatnonzero(distance[d] <= gate_px)
        )
        taken_dots, taken_candidates = set(), set()
        for _, d, c in pairs:
            if d not in taken_dots and c not in taken_candidates:
                taken_dots.add(d)
                taken_candidates.add(c)
                matches[d].append((position, cands[c]))
    return matches


def _merge_duplicates(
    centers: list[np.ndarray], results: list[DotStats], tolerance_m: float
) -> tuple[list[np.ndarray], list[DotStats]]:
    """Of dots closer than *tolerance_m*, keep the one with the most views."""
    order = sorted(range(len(results)), key=lambda i: -results[i].views)
    kept: list[int] = []
    for i in order:
        if all(np.linalg.norm(centers[i] - centers[j]) >= tolerance_m for j in kept):
            kept.append(i)
    return [centers[i] for i in kept], [results[i] for i in kept]


__all__ = [
    "DotStats", "FrameObservation", "MarkerStats", "VideoCalibrationOptions", "VideoCalibrationResult",
    "calibrate_marker_body_from_video", "collect_video_observations", "parse_marker_sizes", "solve_video_body",
]

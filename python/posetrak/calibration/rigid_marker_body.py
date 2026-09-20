# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""rigid_marker_body.py — solve a rigid marker body from ordinary multi-camera footage.

The result is a marker body definition: the geometry of a prop's ArUco markers
and, optionally, its reflective dots, expressed in one marker's own frame. See
docs/roadmap/features/marker-based-mocap/rigid-marker-body-calibration-design.md
for the design and the validation of its core assumption.

No turn-around video is needed, and no two markers have to be visible together
from one camera. All that is required is that, somewhere in the footage, every
marker being placed is seen in the same instant as the chosen reference marker,
possibly by entirely different cameras.

The work has three stages, so that each can be tested and reused on its own:

1. :func:`collect_observations` decodes the footage: per sampled frame and camera,
   the undistorted corners of the body's ArUco markers and, if asked, the reflective
   dot candidates, bucketed by global time.
2. :func:`solve_body` turns the observations into geometry. For every instant in
   which the reference marker is seen by enough cameras, it solves the reference
   pose and that of every other marker seen by enough cameras, expresses the
   others' corners in the reference frame, and averages them robustly over the
   whole capture.
3. The definition is written as marker body YAML in the ``corners:`` form, the
   provenance-preserving form for solved rather than designed geometry.

Reflective dots
---------------
A dot centroid is a point (3 degrees of freedom) where a marker is a rigid
transform (6), but the same co-occurrence mechanism applies. What a dot lacks is
identity: an ArUco corner is named by its decoded id, a dot candidate is not, so
which blob in one camera is the same physical dot as which blob in another is a
sub-problem of its own. It is resolved by using only instants in which at least two
cameras each saw exactly one candidate, then requiring every contributing view to
reproject the triangulated point within a few pixels, since two unrelated stray
bright spots can each be the only candidate of their camera. On real footage of a
reflective prop about 39% of frames have exactly one candidate per camera and
about 11% more than one, so the restriction leaves plenty of samples. Samples are
then clustered by proximity in the reference frame into physical dots
(:func:`cluster_dot_samples`) before averaging.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.setup.db_context import SyncTable
from app.setup.extrinsics_solver import (
    CamCalibState,
    _proj_matrix,
    _undistort_pts,
    marker_local_corners,
    solve_marker_pose,
)
from app.setup.fiducial_markers import ArucoDetector
from posetrak.calibration.session_cameras import load_camera_states, load_sync_table
from posetrak.detection.dot_blob_detector import compute_background, detect_blobs
from posetrak.detection.frame_source import iter_frames

# Frames of different cameras that fall within one bucket count as the same instant.
_BUCKET_S = 0.05


@dataclass
class CalibrationOptions:
    """What to calibrate, and how.

    ``marker_ids``, ``reference_id`` and ``marker_size`` say what the body is; the
    rest are settings with working defaults.
    """

    marker_ids: list[str]
    reference_id: str
    marker_size: float                       # side length of every marker, metres
    dictionary: str = "DICT_4X4_50"
    stride: int = 6                          # process every Nth decoded frame
    min_cameras: int = 2                     # cameras that must see a marker at one instant
    name: str = "calibrated-rigid-body"
    camera_labels: list[str] | None = None   # only these cameras, by label
    detect_dots: bool = False
    dot_threshold: int = 235
    dot_min_area: float = 4.0
    dot_max_area: float = 400.0
    dot_min_compactness: float = 0.5
    dot_cluster_tolerance_m: float = 0.02
    dot_max_reprojection_px: float = 5.0
    dot_bg_subtract: bool = False
    dot_bg_samples: int = 40
    dot_gate_radius_mult: float = 0.0
    dot_quad_exclude_scale: float = 1.5

    def validate(self) -> None:
        """Raise ValueError if the reference is not among the markers or a setting is unusable."""
        if self.reference_id not in self.marker_ids:
            raise ValueError("the reference id must be one of the marker ids")
        if self.stride < 1 or self.min_cameras < 2:
            raise ValueError("stride must be at least 1 and min_cameras at least 2")
        if self.marker_size <= 0:
            raise ValueError("marker size must be positive")


@dataclass
class Observations:
    """What the footage showed, bucketed by global time (see :func:`collect_observations`)."""

    # bucket -> marker id -> camera id -> (4, 2) undistorted corners
    markers: dict[float, dict[str, dict[str, np.ndarray]]] = field(default_factory=dict)
    # bucket -> camera id -> undistorted (u, v) of every dot candidate the camera saw
    dots: dict[float, dict[str, list[np.ndarray]]] = field(default_factory=dict)


@dataclass
class CalibrationResult:
    """A solved marker body and how well the footage supported it."""

    yaml: str
    reference_solved: int                    # instants at which the reference pose was solved
    marker_samples: dict[str, int]           # marker id -> instants it was placed against the reference
    marker_corner_std: dict[str, np.ndarray]  # marker id -> (3,) mean per-axis corner spread, metres
    dot_samples: int = 0
    dot_ambiguous_buckets: int = 0
    dot_cluster_sizes: list[int] = field(default_factory=list)


def calibrate_rigid_marker_body(
    conn: sqlite3.Connection,
    shot_id: str,
    time_start: float,
    time_end: float,
    options: CalibrationOptions,
    *,
    log: Callable[[str], None] = lambda message: None,
) -> CalibrationResult:
    """Solve a marker body from a capture's footage.

    Parameters
    ----------
    conn:
        Connection to a session database with ``sqlite3.Row`` rows. It is only read.
    shot_id:
        The capture.
    time_start, time_end:
        The global time span to use, seconds.
    options:
        What to calibrate and how.
    log:
        Called with a progress line at a time; default discards them.

    Returns
    -------
    CalibrationResult
        The marker body YAML and statistics.

    Raises
    ------
    ValueError
        If *options* are invalid, the capture has no solved extrinsics or sync
        configuration, or the reference marker was never seen by enough cameras.
    """
    options.validate()
    log("Loading camera states and extrinsics...")
    states = load_camera_states(conn, shot_id)
    if options.camera_labels:
        label_by_cam_id = {
            r["camera_instance_id"]: r["label"]
            for r in conn.execute(
                "SELECT DISTINCT cv.camera_instance_id, ci.label FROM capture_videos cv "
                "JOIN camera_instances ci ON ci.id = cv.camera_instance_id WHERE cv.shot_id = ?",
                (shot_id,),
            )
        }
        wanted = set(options.camera_labels)
        states = {c: s for c, s in states.items() if label_by_cam_id.get(c) in wanted}
        log(f"  restricted to {sorted(wanted)}: {len(states)} cameras matched")
    log(f"  {len(states)} cameras with solved extrinsics")
    sync_table, svid_by_cam = load_sync_table(conn, shot_id)

    observations = collect_observations(states, sync_table, svid_by_cam, time_start, time_end, options, log=log)
    return solve_body(observations, states, options, log=log)


def collect_observations(
    states: dict[str, CamCalibState],
    sync_table: SyncTable,
    svid_by_cam: dict[str, str],
    time_start: float,
    time_end: float,
    options: CalibrationOptions,
    *,
    detector=None,
    log: Callable[[str], None] = lambda message: None,
) -> Observations:
    """Decode the footage of each camera and collect marker corners and dot candidates.

    Frames of different cameras are bucketed by global time (rounded to 50 ms), so
    that their independently sampled frames land in the same instant.

    Parameters
    ----------
    states:
        Calibrated cameras by ``camera_instance_id``.
    sync_table:
        Maps a video's frames to global time.
    svid_by_cam:
        ``capture_videos`` id of each camera.
    time_start, time_end:
        The global time span, seconds.
    options:
        What to detect. With ``detect_dots`` the dot candidates of each frame are
        collected too, optionally gated to the neighbourhood of a visible marker.
    detector:
        A marker detector with ``detect(image, video_id, frame_idx)``; an
        ``ArucoDetector`` for ``options.dictionary`` by default.
    log:
        Called with a progress line at a time.

    Returns
    -------
    Observations
        Cameras without sync coverage of the span are skipped.
    """
    detector = detector or ArucoDetector(dictionary=options.dictionary)
    marker_ids = set(options.marker_ids)
    observations = Observations(
        markers=defaultdict(lambda: defaultdict(dict)),
        dots=defaultdict(lambda: defaultdict(list)),
    )

    for cam_id, state in states.items():
        svid = svid_by_cam.get(cam_id)
        if svid is None:
            continue
        first = sync_table.lookup(time_start, svid)
        last = sync_table.lookup(time_end, svid)
        if first is None or last is None:
            log(f"  SKIP {cam_id[:8]}: no sync coverage in [{time_start}, {time_end})")
            continue

        background = None
        if options.detect_dots and options.dot_bg_subtract:
            # Sampled from [first, last) itself, not from the camera's whole span: the
            # decoder has no random access and decodes every frame of the range it is
            # given, so sampling a long shot would decode all of it to keep a few frames.
            step = max(1, (last - first) // options.dot_bg_samples)
            bg_frames = [
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                for video_frame, img in iter_frames(state.file_path, first, last)
                if (video_frame - first) % step == 0
            ]
            background = compute_background(bg_frames)
            log(f"camera {cam_id[:8]}: background from {len(bg_frames)} frames spanning {first}-{last}")

        log(f"camera {cam_id[:8]}: frames {first}-{last} ({state.file_path})")
        n_decoded = 0
        for video_frame, img in iter_frames(state.file_path, first, last):
            n_decoded += 1
            if (video_frame - first) % options.stride != 0:
                continue
            dets = detector.detect(img, video_id=cam_id, frame_idx=video_frame)
            gt = sync_table.frame_to_global_time(video_frame, svid)
            if gt is None:
                continue
            bucket = round(gt / _BUCKET_S) * _BUCKET_S
            body_dets = [d for d in dets if d.marker_id in marker_ids]
            for d in body_dets:
                pts = np.array([(c.px, c.py) for c in d.corners], dtype=np.float64)
                observations.markers[bucket][d.marker_id][cam_id] = _undistort_pts(pts, state)

            if options.detect_dots:
                blobs = detect_blobs(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), threshold=options.dot_threshold,
                    min_area=options.dot_min_area, max_area=options.dot_max_area,
                    min_compactness=options.dot_min_compactness, background=background,
                )
                if options.dot_gate_radius_mult > 0:
                    blobs = _gate_to_markers(blobs, body_dets, options)
                if blobs:
                    pts = np.array([(b.cx, b.cy) for b in blobs], dtype=np.float64)
                    observations.dots[bucket][cam_id].extend(_undistort_pts(pts, state))
        log(f"  {n_decoded} frames decoded")

    log(f"\n{len(observations.markers)} time buckets with >=1 marker detection")
    return observations


def _gate_to_markers(blobs: list, body_dets: list, options: CalibrationOptions) -> list:
    """Keep the dot candidates near a visible marker of the body and outside its own footprint.

    A candidate has to lie within ``dot_gate_radius_mult`` times the on-screen
    diagonal of some visible marker, measured from that marker's centre, which keeps
    the region of interest sane at any camera distance and rejects clutter far from
    the body (a window, a calibration target) that shape and brightness filtering do
    not catch. A frame in which no marker is visible keeps no candidates. Candidates
    inside a marker's own quad, dilated by ``dot_quad_exclude_scale``, are dropped: a
    marker that moved slightly between the background samples leaves its own
    high-contrast edges as background-subtraction residual, which a distance gate
    cannot tell from a real dot next to the marker.
    """
    if not body_dets:
        return []
    anchors, quads = [], []
    for d in body_dets:
        corners = np.array([(c.px, c.py) for c in d.corners], dtype=np.float64)
        anchors.append((corners.mean(axis=0), float(np.linalg.norm(corners[0] - corners[2]))))
        quads.append(corners)
    kept = []
    for b in blobs:
        p = np.array([b.cx, b.cy])
        if options.dot_quad_exclude_scale > 0 and any(
            _point_in_dilated_quad(p, q, options.dot_quad_exclude_scale) for q in quads
        ):
            continue
        if any(np.linalg.norm(p - c) <= options.dot_gate_radius_mult * diag for c, diag in anchors):
            kept.append(b)
    return kept


def solve_body(
    observations: Observations,
    states: dict[str, CamCalibState],
    options: CalibrationOptions,
    *,
    log: Callable[[str], None] = lambda message: None,
) -> CalibrationResult:
    """Solve the body's geometry from collected observations.

    Parameters
    ----------
    observations:
        From :func:`collect_observations`.
    states:
        The calibrated cameras the observations came from.
    options:
        Which marker is the reference, the marker size and the dot settings.
    log:
        Called with a progress line at a time.

    Returns
    -------
    CalibrationResult
        The body as YAML, the reference marker first, then the other markers in the
        order they were first placed, then the dots.

    Raises
    ------
    ValueError
        If the reference marker was never seen by enough cameras at once to solve
        its pose.
    """
    ref_id = options.reference_id
    # Local corners (4, 3) of each marker in the reference marker's frame, one per
    # instant at which both had a solvable pose.
    samples_by_marker: dict[str, list[np.ndarray]] = defaultdict(list)
    # Unlabeled dot positions in the reference frame, one per instant at which the
    # reference was solved and at least two cameras each saw exactly one candidate.
    dot_samples: list[np.ndarray] = []
    n_ref_solved = 0
    n_dot_buckets_ambiguous = 0

    for bucket, by_marker in sorted(observations.markers.items()):
        ref_obs = by_marker.get(ref_id)
        if ref_obs is None or len(ref_obs) < options.min_cameras:
            continue
        try:
            rvec_ref, tvec_ref, _ = solve_marker_pose(ref_obs, states, options.marker_size)
        except (ValueError, RuntimeError):
            continue
        n_ref_solved += 1
        R_ref, _ = cv2.Rodrigues(rvec_ref)

        for mid, obs in by_marker.items():
            if mid == ref_id or len(obs) < options.min_cameras:
                continue
            try:
                rvec_m, tvec_m, _ = solve_marker_pose(obs, states, options.marker_size)
            except (ValueError, RuntimeError):
                continue
            R_m, _ = cv2.Rodrigues(rvec_m)
            world_corners = (R_m @ marker_local_corners(options.marker_size).T).T + tvec_m
            samples_by_marker[mid].append((R_ref.T @ (world_corners - tvec_ref).T).T)

        if options.detect_dots:
            cam_dots = observations.dots.get(bucket, {})
            unambiguous = {cam_id: tuple(pts[0]) for cam_id, pts in cam_dots.items() if len(pts) == 1}
            if any(len(pts) > 1 for pts in cam_dots.values()):
                n_dot_buckets_ambiguous += 1
            if len(unambiguous) >= options.min_cameras:
                world_pt = triangulate_point_multi_view(
                    unambiguous, states, max_reprojection_px=options.dot_max_reprojection_px
                )
                if world_pt is not None:
                    dot_samples.append(R_ref.T @ (world_pt - tvec_ref.flatten()))

    if n_ref_solved == 0:
        raise ValueError(
            f"the reference marker {ref_id!r} was never seen by {options.min_cameras} or more cameras "
            "at once, so no pose could be solved"
        )
    log(f"Reference marker '{ref_id}' solved at {n_ref_solved} instants")
    for mid in options.marker_ids:
        if mid != ref_id and mid not in samples_by_marker:
            log(f"  marker '{mid}': never seen together with the reference, left out")
    for mid, samples in samples_by_marker.items():
        log(f"  marker '{mid}': {len(samples)} co-occurrence samples with reference")

    dot_clusters: list[list[np.ndarray]] = []
    if options.detect_dots:
        log(f"\nDot candidates: {len(dot_samples)} unambiguous triangulated samples "
            f"({n_dot_buckets_ambiguous} instants dropped for having more than one candidate in some camera)")
        dot_clusters = cluster_dot_samples(dot_samples, options.dot_cluster_tolerance_m)
        log(f"  clustered into {len(dot_clusters)} physical dots")
        for i, cluster in enumerate(dot_clusters):
            log(f"  dot{i}: {len(cluster)} samples, per-axis std (m) {np.stack(cluster).std(axis=0)}")

    markers = [(ref_id, options.marker_size, marker_local_corners(options.marker_size))]
    corner_std: dict[str, np.ndarray] = {}
    for mid, samples in samples_by_marker.items():
        stacked = np.stack(samples)  # (N, 4, 3)
        markers.append((mid, options.marker_size, np.stack([robust_mean(stacked[:, i, :]) for i in range(4)])))
        corner_std[mid] = np.stack([stacked[:, i, :].std(axis=0) for i in range(4)]).mean(axis=0)
        log(f"  marker '{mid}' corner std across samples (m): {corner_std[mid]}")

    return CalibrationResult(
        yaml=format_marker_body_yaml(
            options.name, options.dictionary, markers, [robust_mean(np.stack(c)) for c in dot_clusters],
        ),
        reference_solved=n_ref_solved,
        marker_samples={mid: len(s) for mid, s in samples_by_marker.items()},
        marker_corner_std=corner_std,
        dot_samples=len(dot_samples),
        dot_ambiguous_buckets=n_dot_buckets_ambiguous,
        dot_cluster_sizes=[len(c) for c in dot_clusters],
    )


def _format_vector(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"


def format_marker_body_yaml(
    name: str,
    dictionary: str,
    markers: list[tuple[str, float, np.ndarray]],
    dot_centers: list[np.ndarray],
) -> str:
    """Write a solved marker body as marker body definition YAML.

    Parameters
    ----------
    name:
        Name of the body.
    dictionary:
        ArUco dictionary of the markers.
    markers:
        ``(marker id, side length, (4, 3) corners)`` in the order to write them; the
        corners are in the body's frame, metres, in the order of ``marker_local_corners``.
        They are written in the ``corners:`` form, the provenance-preserving form for
        solved rather than designed geometry.
    dot_centers:
        The reflective dots' positions in the body's frame, written as ``dot0``, ``dot1``...

    Returns
    -------
    str
        The YAML text.
    """
    lines = [f"name: {name}", "units: meters", "markers:"]
    for marker_id, size, corners in markers:
        lines += [
            f"  - name: aruco_{marker_id}",
            "    type: aruco",
            f"    dictionary: {dictionary}",
            f'    id: "{marker_id}"',
            f"    size: {size}",
            "    corners:",
        ]
        lines += [f"      - {_format_vector(c)}" for c in corners]
    for i, center in enumerate(dot_centers):
        lines += [f"  - name: dot{i}", "    type: reflective_dot", f"    center: {_format_vector(center)}"]
    return "\n".join(lines) + "\n"


def robust_mean(samples: np.ndarray, trim_frac: float = 0.1) -> np.ndarray:
    """Per-axis trimmed mean across (N, 3) samples.

    Outlier instants from a bad triangulation or pose solve should not bias the result.
    """
    if len(samples) == 0:
        raise ValueError("no samples to average")
    n_trim = int(len(samples) * trim_frac)
    out = np.zeros(3)
    for axis in range(3):
        vals = np.sort(samples[:, axis])
        if n_trim > 0 and len(vals) > 2 * n_trim:
            vals = vals[n_trim:-n_trim]
        out[axis] = vals.mean()
    return out


def triangulate_point_multi_view(
    observations: dict[str, tuple[float, float]], states: dict[str, CamCalibState],
    max_reprojection_px: float = 5.0,
) -> np.ndarray | None:
    """Linear (DLT) triangulation of one unlabeled 3D point from at least two views.

    Two equations per view are stacked and solved by SVD. This is not
    ``extrinsics_solver.triangulate_pair()``, which batch-triangulates many
    already-matched point pairs: calibration has one point per instant seen by any
    number of views.

    Every view's reprojection error is checked against *max_reprojection_px*, and
    that check is required. That every camera saw exactly one candidate does not
    make the candidates the same physical point: two unrelated stray bright spots
    can each be the only candidate of their camera. On real footage, without the
    check a two-camera coincidence triangulated to a point over three metres from
    the body.

    Parameters
    ----------
    observations:
        ``camera id -> undistorted (u, v)``.
    states:
        The calibrated cameras. Observations from a camera not in it are ignored.
    max_reprojection_px:
        Largest acceptable reprojection error in any contributing view.

    Returns
    -------
    numpy.ndarray | None
        The point, or None if fewer than two usable views remain, the system is
        degenerate (near-parallel rays, a camera behind the point) or the
        reprojection check fails.
    """
    proj_matrices: dict[str, np.ndarray] = {}
    rows = []
    for cam_id, (u, v) in observations.items():
        state = states.get(cam_id)
        if state is None or state.R is None:
            continue
        P = _proj_matrix(state)
        proj_matrices[cam_id] = P
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    if len(rows) < 4:  # fewer than 2 real views
        return None
    _, _, vt = np.linalg.svd(np.stack(rows))
    x = vt[-1]
    if abs(x[3]) < 1e-8:
        return None
    point = x[:3] / x[3]

    point_h = np.append(point, 1.0)
    for cam_id, P in proj_matrices.items():
        proj = P @ point_h
        if abs(proj[2]) < 1e-8:
            return None
        u, v = observations[cam_id]
        if np.linalg.norm(proj[:2] / proj[2] - np.array([u, v])) > max_reprojection_px:
            return None
    return point


def _point_in_dilated_quad(point: np.ndarray, quad: np.ndarray, scale: float) -> bool:
    """True if *point* lies inside *quad* (four corners in any winding) dilated by
    *scale* about its centroid. The dilation leaves a margin for the corner
    detector's sub-pixel jitter."""
    centroid = quad.mean(axis=0)
    dilated = (centroid + (quad - centroid) * scale).astype(np.float32)
    return cv2.pointPolygonTest(dilated.reshape(-1, 1, 2), (float(point[0]), float(point[1])), False) >= 0


def cluster_dot_samples(samples: list[np.ndarray], tolerance_m: float = 0.02) -> list[list[np.ndarray]]:
    """Greedy incremental clustering of dot positions into distinct physical dots.

    A triangulated dot carries no identity, so the samples of a whole capture must
    be grouped as "these are the same physical dot" before averaging. Each sample
    joins the nearest cluster, by running centroid, within *tolerance_m*, else
    starts a new one.

    The result depends on the order of the samples, which is acceptable because real
    dots on a rigid prop are centimetres to tens of centimetres apart while
    per-sample triangulation noise is sub-centimetre, so the assignment is not a
    close call and needs no globally optimal method.
    """
    clusters: list[list[np.ndarray]] = []
    centroids: list[np.ndarray] = []
    for sample in samples:
        best_idx = None
        best_dist = tolerance_m
        for i, centroid in enumerate(centroids):
            dist = float(np.linalg.norm(sample - centroid))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is None:
            clusters.append([sample])
            centroids.append(sample)
        else:
            clusters[best_idx].append(sample)
            centroids[best_idx] = np.mean(clusters[best_idx], axis=0)
    return clusters

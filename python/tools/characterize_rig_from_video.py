#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""characterize_rig_from_video.py — Derive a portable calibration rig's
marker geometry from a single video of one camera orbiting the rig.

See docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md, section 9 "Multi-instrument world-frame
anchoring", Tier A subsection "Rig geometry from an orbit video
(self-calibration)" — and status.md's 2026-08-11 entry for the concrete
two-capture plan this is the first step of.

Idea
----
Each sampled video frame is treated as its own "camera" with an unknown
pose (same physical device, same known intrinsics, just a different pose
per frame) — mathematically identical to the ordinary multi-camera
extrinsics-calibration problem `extrinsics_solver.run_calibration()`
already solves, just with N sequential frames from one moving camera
standing in for N simultaneous cameras. ArUco markers detected on the
rig's faces are fed in as *free* (unscaled) control points, exactly the
way Phase 3 already treats any ArUco marker with no configured size — this
lets the ordinary SIFT+BFS+bundle-adjustment pipeline solve all the
per-frame poses jointly, using the rig's own markers as extra
correspondences alongside whatever SIFT features the room offers.

**Why marker size is deliberately withheld from the main solve, and scale
is fixed in a separate step afterward**: a pure SIFT/free-point
reconstruction has no absolute scale — reprojection error is invariant
under uniformly rescaling the whole scene and all camera translations
together, so nothing in the main solve can anchor "how many metres" a
network unit is. `extrinsics_solver.solve_marker_groups()` (the existing
Phase 3 rigid-marker-pose post-pass) assumes the camera network is
*already* metric before it runs, which is true in the shipped Phase 3/4 UI
flow (a ChArUco anchor or manual control point already fixed scale by the
time it runs) but is NOT true here — there is no other metric anchor at
all in a bare orbit video. Mixing a metric `size` into a scale-free camera
network there would produce a mathematically inconsistent PnP-style solve
(camera translations in arbitrary "network units", marker corners pinned
to real metres) with no clean fix. This script sidesteps that by:
  1. Solving the whole frame network with every marker corner *unscaled*
     (free control points only, same as an unknown-size ArUco marker).
  2. Triangulating each marker's 4 corners itself from the solved (still
     network-unit-scale) camera poses.
  3. Measuring the network-unit edge length of every marker's 4 sides
     (nominally equal, since real ArUco markers are square) and comparing
     the median to the *known* physical marker size to get one scalar
     network-units → metres conversion factor.
  4. Rescaling the whole solved network by that factor (reusing
     `apply_similarity_transform`, unchanged from the existing solver).
Only after this is the geometry actually metric — this is the one piece
of new *reasoning* this script adds; everything else reuses
`extrinsics_solver.py`/`fiducial_markers.py` unchanged, per the design
doc's "no new solver machinery, just a new caller" framing.

Output is a rig geometry JSON in the design doc's `"shape": "explicit"`
form: each detected marker's 4 corners, in metres, expressed in a
rig-local frame anchored on one reference marker (default: the
lowest-numbered marker with all 4 corners solved).

Usage
-----
    python characterize_rig_from_video.py VIDEO \\
        --registry-db path/to/registry.db --camera-label oneplus10pro \\
        --marker-size 0.15 --output rig_config.json

Run once per orbit video (one physical camera each) — comparing the two
independent outputs for the same physical rig, shot with two different
cameras, is itself a cross-check of this method's accuracy (see status.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.setup.extrinsics_solver import (  # noqa: E402
    CamCalibState,
    MarkerGroup,
    ObsPoint,
    apply_similarity_transform,
    run_calibration,
)
from app.setup.fiducial_markers import ArucoDetector, merge_detections_into_groups  # noqa: E402

_log = logging.getLogger("characterize_rig_from_video")

# ArUco corner order (top-left, top-right, bottom-right, bottom-left,
# clockwise facing the marker) — same convention as
# extrinsics_solver.marker_local_corners() and fiducial_markers.py,
# verified there against real cv2.aruco output. Consecutive pairs are the
# marker's 4 physical sides.
_EDGE_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 0)]


# ---------------------------------------------------------------------------
# Registry lookup — same JOIN page_extrinsics.py's _load_states use, scoped
# to one camera label instead of every label in a capture.
# ---------------------------------------------------------------------------


def _load_intrinsics(conn: sqlite3.Connection, camera_label: str, camera_mode: str | None) -> dict:
    """Resolve one camera_instances.label (+ optional camera_mode substring)
    to K/K_orig/dist/fisheye.

    **Deliberately does not silently pick "a" calibration when a camera
    model has more than one recording mode.** A camera_instances row can
    have several camera_modes (e.g. this registry's ACE2 Pro has distinct
    "MEGA mode 4K 120 fps" and "4K 120 fps linear" modes -- genuinely
    different FOV/distortion profiles, not just different resolutions), each
    with its own default_intrinsics_calibration_id. Picking "whichever
    default was calibrated most recently" across modes (an earlier version
    of this function did exactly that) can silently apply the wrong mode's
    calibration to footage shot in a different mode -- a wrong-FOV
    intrinsics error that would corrupt the whole rig-geometry solve without
    ever raising an error. So: if more than one camera_modes row exists for
    this camera's model, --camera-mode (a case-insensitive substring against
    camera_modes.notes) is required to disambiguate; if it doesn't narrow
    things to exactly one mode, this lists the candidates and exits rather
    than guessing.
    """
    conn.row_factory = sqlite3.Row
    modes = conn.execute(
        """
        SELECT cm.id, cm.width_px, cm.height_px, cm.nominal_fps, cm.notes,
               cm.default_intrinsics_calibration_id
        FROM camera_instances ci
        JOIN camera_modes cm ON cm.camera_model_id = ci.camera_model_id
        WHERE ci.label = ?
        """,
        (camera_label,),
    ).fetchall()
    if not modes:
        raise SystemExit(
            f"error: no camera_modes found for camera_instances.label={camera_label!r}. "
            f"Check --camera-label matches the registry exactly (case-sensitive)."
        )
    if camera_mode:
        needle = camera_mode.lower()
        modes = [m for m in modes if needle in (m["notes"] or "").lower()]
    if len(modes) != 1:
        lines = "\n".join(
            f"  id={m['id']}  {m['width_px']}x{m['height_px']}@{m['nominal_fps']}fps  "
            f"notes={m['notes']!r}  default_calib={m['default_intrinsics_calibration_id']}"
            for m in (modes if len(modes) > 1 else conn.execute(
                "SELECT cm.id, cm.width_px, cm.height_px, cm.nominal_fps, cm.notes, "
                "cm.default_intrinsics_calibration_id FROM camera_instances ci "
                "JOIN camera_modes cm ON cm.camera_model_id = ci.camera_model_id "
                "WHERE ci.label = ?", (camera_label,)
            ).fetchall())
        )
        raise SystemExit(
            f"error: {camera_label!r} has {len(modes)} matching camera_modes for "
            f"--camera-mode={camera_mode!r} (need exactly 1). Candidates:\n{lines}\n"
            f"Pass --camera-mode with a substring of the intended mode's notes."
        )
    mode = modes[0]
    calib_id = mode["default_intrinsics_calibration_id"]
    if calib_id is not None:
        row = conn.execute("SELECT * FROM intrinsics_calibrations WHERE id = ?", (calib_id,)).fetchone()
    else:
        row = None
    if row is None:
        row = conn.execute(
            "SELECT * FROM intrinsics_calibrations WHERE camera_mode_id = ? "
            "ORDER BY calibrated_at DESC LIMIT 1",
            (mode["id"],),
        ).fetchone()
    if row is None:
        raise SystemExit(
            f"error: camera_modes.id={mode['id']} ({mode['notes']!r}) has no "
            f"intrinsics_calibrations at all."
        )
    print(f"Camera mode: {mode['notes']!r} ({mode['width_px']}x{mode['height_px']}"
          f"@{mode['nominal_fps']}fps), calibration id={row['id']} calibrated_at={row['calibrated_at']}")
    fx, fy, cx, cy = row["fx"], row["fy"], row["cx"], row["cy"]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    K_orig = K.copy()
    if row["matrix_original"]:
        vals = struct.unpack("<9d", bytes(row["matrix_original"]))
        K_orig = np.array(vals).reshape(3, 3)
    if row["dist_coeffs"]:
        n = len(bytes(row["dist_coeffs"])) // 8
        dist = np.array(struct.unpack(f"<{n}d", bytes(row["dist_coeffs"]))).reshape(1, -1)
    else:
        dist = np.zeros((1, 4))
    fisheye = row["distortion_model"] == "fisheye"
    _log.info(
        "Loaded intrinsics for %r: calib_id=%s fx=%.1f fy=%.1f fisheye=%s calibrated_at=%s",
        camera_label, row["id"], fx, fy, fisheye, row["calibrated_at"],
    )
    return {"K": K, "K_orig": K_orig, "dist": dist, "fisheye": fisheye}


# ---------------------------------------------------------------------------
# Rotation-aware frame sampling — duplicated from tools/detect_aruco.py by
# design (see that script's own header): each tool in this directory is
# self-contained rather than sharing a frame-source module, so this can stay
# a standalone prototype without coupling to the UI's video-scrubbing code.
# ---------------------------------------------------------------------------


def _parse_displaymatrix(data: bytes) -> int:
    if len(data) < 36:
        return 0
    m = struct.unpack("<9i", data[:36])
    scale_x = math.hypot(m[0], m[3])
    scale_y = math.hypot(m[1], m[4])
    if scale_x == 0 or scale_y == 0:
        return 0
    return round(-math.atan2(m[1] / scale_y, m[0] / scale_x) * 180 / math.pi) % 360


def _apply_rotation(img: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _video_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _iter_sampled_frames_av(
    path: str, start: int, end: int, stride: int, max_samples: int, rotation_override: int | None,
):
    """Yield up to *max_samples* (frame_idx, bgr) pairs, every *stride*-th
    frame in [start, end). Decodes sequentially (required for long-GOP
    consumer codecs, see the project's other video-scrub code) and stops as
    soon as enough samples are collected rather than decoding the whole
    file.
    """
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base)
        container_fps = float(stream.average_rate)

        if rotation_override is not None:
            rotation = rotation_override
        else:
            rotate_str = (stream.metadata or {}).get("rotate", "0") or "0"
            try:
                rotation = int(rotate_str) % 360
            except (ValueError, TypeError):
                rotation = 0
            if rotation == 0:
                for probe_frame in container.decode(stream):
                    for sd in probe_frame.side_data or []:
                        if "DISPLAYMATRIX" in str(sd.type).upper():
                            rotation = _parse_displaymatrix(bytes(sd))
                    break
                container.seek(0, stream=stream, backward=True, any_frame=False)
        if rotation:
            print(f"  rotation: {rotation}° (from metadata)")

        if start > 0:
            seek_s = max(0.0, (start - 1) / container_fps)
            container.seek(int(seek_s / time_base), stream=stream, backward=True, any_frame=False)

        frame_idx: int | None = None
        n_yielded = 0
        for av_frame in container.decode(stream):
            if av_frame.pts is None:
                continue
            if frame_idx is None:
                pts_idx = round(float(av_frame.pts) * time_base * container_fps)
                if pts_idx < start:
                    continue
                frame_idx = start
            if frame_idx >= end or n_yielded >= max_samples:
                break
            if (frame_idx - start) % stride == 0:
                img = av_frame.to_ndarray(format="bgr24")
                if rotation:
                    img = _apply_rotation(img, rotation)
                yield frame_idx, img
                n_yielded += 1
            frame_idx += 1


def _iter_sampled_frames_cv2(
    path: str, start: int, end: int, stride: int, max_samples: int, rotation_override: int | None,
):
    cap = cv2.VideoCapture(path)
    if rotation_override is not None:
        rotation = rotation_override
    else:
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) % 360
    if rotation:
        print(f"  rotation: {rotation}°")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_idx = start
    n_yielded = 0
    while frame_idx < end and n_yielded < max_samples:
        ok, img = cap.read()
        if not ok:
            break
        if (frame_idx - start) % stride == 0:
            if rotation:
                img = _apply_rotation(img, rotation)
            yield frame_idx, img
            n_yielded += 1
        frame_idx += 1
    cap.release()


def _iter_sampled_frames(path, start, end, stride, max_samples, rotation_override):
    try:
        import av  # noqa: F401
        yield from _iter_sampled_frames_av(path, start, end, stride, max_samples, rotation_override)
    except ImportError:
        yield from _iter_sampled_frames_cv2(path, start, end, stride, max_samples, rotation_override)


# ---------------------------------------------------------------------------
# Post-solve marker-corner triangulation (network-unit scale — see module
# docstring for why this is done locally instead of via
# extrinsics_solver.solve_marker_groups()).
# ---------------------------------------------------------------------------


def _undistort_px(px: float, py: float, state: CamCalibState) -> np.ndarray:
    p = np.array([[[px, py]]], dtype=np.float32)
    if state.fisheye:
        out = cv2.fisheye.undistortPoints(p, state.K_orig, state.dist, None, state.K)
    else:
        out = cv2.undistortPoints(p, state.K_orig, state.dist, None, state.K)
    return out.reshape(2)


def _proj_matrix(state: CamCalibState) -> np.ndarray:
    Rt = np.hstack([state.R, state.t.reshape(3, 1)])
    return state.K @ Rt


def _triangulate_corner(
    obs_by_video: dict[str, ObsPoint], states_by_id: dict[str, CamCalibState],
    max_reprojection_px: float = 10.0,
) -> np.ndarray | None:
    """DLT-triangulate one marker corner from every solved camera that
    observed it. Same SVD approach extrinsics_solver.py already uses twice
    (run_bundle_adjustment's free-CP init, compute_cp_errors) — duplicated
    here in tool-script-local form rather than importing those private
    (underscore-prefixed) internals, per this directory's existing
    standalone-script convention (see detect_aruco.py's header).

    **Unlike either of those two existing call sites, this rejects the
    result outright if any observing camera's reprojection error exceeds
    *max_reprojection_px*** (the same threshold/idea `triangulate_pair`
    already applies to SIFT points, just not something either existing
    free-CP triangulation path currently checks). Found necessary live,
    not speculatively: a marker seen by only 2 solved "frame-cameras" at a
    poor triangulation angle produced multi-metre corner errors (a 0.15m
    marker with a 12m "edge") with nothing to catch it before it silently
    corrupted the scale estimate and the rig config — see status.md's
    orbit-video prototype notes.
    """
    rows = []
    proj_inputs: list[tuple[np.ndarray, np.ndarray]] = []  # (P, undistorted px)
    for vid, obs in obs_by_video.items():
        state = states_by_id.get(vid)
        if state is None or state.R is None:
            continue
        px_u = _undistort_px(obs.px, obs.py, state)
        P = _proj_matrix(state)
        rows.append(px_u[0] * P[2] - P[0])
        rows.append(px_u[1] * P[2] - P[1])
        proj_inputs.append((P, px_u))
    if len(rows) < 4:  # need >= 2 cameras
        return None
    A = np.array(rows, dtype=np.float64)
    if not np.isfinite(A).all():
        return None
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    if abs(h[3]) < 1e-10:
        return None
    xyz = (h[:3] / h[3]).astype(np.float64)

    xyz_h = np.append(xyz, 1.0)
    for P, px_u in proj_inputs:
        proj_h = P @ xyz_h
        if abs(proj_h[2]) < 1e-9:
            return None
        proj = proj_h[:2] / proj_h[2]
        if np.linalg.norm(proj - px_u) > max_reprojection_px:
            return None
    return xyz


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # extrinsics_solver's progress_cb messages and log lines contain non-ASCII
    # characters (e.g. '↔' in match_all_pairs) that crash a plain print() on
    # Windows' default cp1252 console encoding -- reconfigure stdout/stderr to
    # tolerate them instead of erroring out mid-run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", help="Orbit video: one camera moving around the rig")
    parser.add_argument("--registry-db", required=True, help="Path to the camera-registry SQLite DB")
    parser.add_argument("--camera-label", required=True,
                         help="camera_instances.label to look up intrinsics for (must have a "
                              "calibrated camera_modes/intrinsics_calibrations row)")
    parser.add_argument("--camera-mode", default=None,
                         help="Case-insensitive substring of the camera_modes.notes to use "
                              "(e.g. \"linear\" vs \"MEGA\" for an Insta360 ACE2 Pro). Required "
                              "whenever the camera model has more than one recording mode -- the "
                              "script lists candidates and exits rather than guessing.")
    parser.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary (default: DICT_4X4_50)")
    parser.add_argument("--marker-size", type=float, default=0.15,
                         help="Physical marker side length in metres (default: 0.15)")
    parser.add_argument("--min-marker-perimeter-rate", type=float, default=0.01,
                         help="cv2.aruco minMarkerPerimeterRate — default lower than OpenCV's own "
                              "0.03, see fiducial_markers.ArucoDetector's docstring (default: 0.01)")
    parser.add_argument("--num-samples", type=int, default=15,
                         help="Target number of frames to sample across [start, end) (default: 15). "
                              "Cost is roughly O(num_samples^2) via pairwise SIFT matching (each "
                              "pair re-runs full-resolution SIFT on both frames, uncached) -- start "
                              "small and raise it once a run completes in reasonable time.")
    parser.add_argument("--frame-stride", type=int, default=None,
                         help="Explicit stride override; by default computed from --num-samples "
                              "and the video's frame range so samples span the whole orbit")
    parser.add_argument("--start", type=int, default=0, metavar="FRAME")
    parser.add_argument("--end", type=int, default=None, metavar="FRAME")
    parser.add_argument("--rotate", type=int, default=None, choices=[0, 90, 180, 270],
                         help="Override rotation correction; auto-detected from metadata if omitted")
    parser.add_argument("--ref-marker", default=None,
                         help="Marker id whose own frame becomes the rig-local origin/axes "
                              "(default: lowest-id marker with all 4 corners solved)")
    parser.add_argument("--ba-max-nfev", type=int, default=4000,
                         help="Bundle-adjustment iteration cap (default: 4000, raised from "
                              "run_calibration's own 2000 default since this problem has more "
                              "camera-shaped parameters than a typical fixed-rig calibration)")
    parser.add_argument("-o", "--output", default=None,
                         help="Output rig-config JSON path (default: VIDEO.rig_config.json)")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S",
    )

    video_path = args.video
    if not Path(video_path).exists():
        print(f"error: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(f"file:{args.registry_db}?mode=ro", uri=True)
    try:
        intr = _load_intrinsics(conn, args.camera_label, args.camera_mode)
    finally:
        conn.close()

    total_frames = _video_frame_count(video_path)
    start = args.start
    end = args.end if args.end is not None else total_frames
    end = min(end, total_frames)
    if start >= end:
        print(f"error: empty frame range [{start}, {end})", file=sys.stderr)
        sys.exit(1)

    stride = args.frame_stride or max(1, (end - start) // args.num_samples)
    print(f"Video: {video_path}  frames [{start}, {end})  stride={stride}  "
          f"target samples={args.num_samples}")

    detector = ArucoDetector(
        dictionary=args.dict, default_size=None,  # deliberately unscaled — see module docstring
        min_marker_perimeter_rate=args.min_marker_perimeter_rate,
    )

    states: list[CamCalibState] = []
    groups: dict[str, MarkerGroup] = {}
    for frame_idx, img in _iter_sampled_frames(video_path, start, end, stride, args.num_samples, args.rotate):
        video_id = f"f{frame_idx:06d}"
        states.append(CamCalibState(
            video_id=video_id, label=video_id,
            K=intr["K"], K_orig=intr["K_orig"], dist=intr["dist"], fisheye=intr["fisheye"],
            image=img,
        ))
        detections = detector.detect(img, video_id=video_id, frame_idx=frame_idx)
        merge_detections_into_groups(detections, groups, size=None)
        print(f"  sampled frame {frame_idx} ({len(detections)} marker(s) detected)")

    n_frames = len(states)
    if n_frames < 3:
        print(f"error: only {n_frames} frame(s) sampled — need at least 3 for a useful solve",
              file=sys.stderr)
        sys.exit(1)
    print(f"Sampled {n_frames} frames, {len(groups)} distinct marker id(s) seen across them.")

    n_pairs = n_frames * (n_frames - 1) // 2
    print(f"Running run_calibration() — {n_pairs} SIFT pair-matches at full frame resolution "
          f"(each pair re-detects SIFT on both frames, no caching between pairs) — this is the "
          f"slow step, expect it to dominate total runtime...")
    result = run_calibration(
        states, marker_groups=list(groups.values()), cp_only=False, ba_max_nfev=args.ba_max_nfev,
        progress_cb=lambda msg: print(f"  {msg}"),
    )
    states_by_id = {s.video_id: s for s in states}
    if result.unsolved:
        print(f"WARNING: {len(result.unsolved)}/{n_frames} sampled frames did not solve: "
              f"{result.unsolved}")
    n_solved = n_frames - len(result.unsolved)
    if n_solved < 2:
        print("error: fewer than 2 frames solved — cannot triangulate marker corners",
              file=sys.stderr)
        sys.exit(1)

    # --- Triangulate every marker's 4 corners at the network's own (as yet
    # unknown-in-metres) scale. ---
    corners_network: dict[str, list[np.ndarray | None]] = {}
    for marker_id, mg in groups.items():
        corners_network[marker_id] = []
        for corner_idx in range(4):
            obs_by_video = {
                vid: corners[corner_idx]
                for vid, corners in mg.obs.items()
                if corner_idx in corners
            }
            corners_network[marker_id].append(_triangulate_corner(obs_by_video, states_by_id))

    n_corners_total = sum(len(cs) for cs in corners_network.values())
    n_corners_ok = sum(1 for cs in corners_network.values() for c in cs if c is not None)
    print(f"Triangulated {n_corners_ok}/{n_corners_total} marker corners "
          f"({n_corners_total - n_corners_ok} missing or rejected as a bad-conditioned/"
          f"outlier triangulation — see max_reprojection_px in _triangulate_corner).")

    # --- Scale: median network-unit edge length vs. the known marker size. ---
    edge_lengths: list[float] = []
    for marker_id, corners in corners_network.items():
        for i, j in _EDGE_PAIRS:
            if corners[i] is not None and corners[j] is not None:
                edge_lengths.append(float(np.linalg.norm(corners[i] - corners[j])))
    if not edge_lengths:
        print("error: no marker had >= 2 fully-triangulated corners — cannot determine scale. "
              "Try more samples, a lower --min-marker-perimeter-rate, or check --dict.",
              file=sys.stderr)
        sys.exit(1)
    median_edge = float(np.median(edge_lengths))
    scale = args.marker_size / median_edge
    edge_arr = np.array(edge_lengths)
    print(
        f"Scale: {len(edge_lengths)} marker edge(s), median={median_edge:.4f} network-units "
        f"(mean={edge_arr.mean():.4f}, std={edge_arr.std():.4f}) -> {args.marker_size}m each "
        f"=> scale factor k={scale:.6f}"
    )
    rel_std = edge_arr.std() / edge_arr.mean() if edge_arr.mean() > 1e-9 else float("nan")
    if rel_std > 0.05:
        print(
            f"WARNING: edge-length spread is {rel_std * 100:.1f}% of the mean — the solved "
            f"network's internal geometry may be noisy (weak SIFT overlap between samples, too "
            f"few markers per frame, or the oneplus autofocus-intrinsics-drift risk noted in "
            f"status.md). Treat this rig config as provisional until cross-checked."
        )

    apply_similarity_transform(states, [], scale=scale, R_align=np.eye(3), t_align=np.zeros(3))
    corners_metric: dict[str, list[np.ndarray | None]] = {
        marker_id: [c * scale if c is not None else None for c in corners]
        for marker_id, corners in corners_network.items()
    }

    # --- Rig-local frame from the reference marker. ---
    complete_ids = [mid for mid, cs in corners_metric.items() if all(c is not None for c in cs)]
    if not complete_ids:
        print("error: no marker has all 4 corners triangulated — cannot anchor a rig-local frame",
              file=sys.stderr)
        sys.exit(1)
    ref_id = args.ref_marker if args.ref_marker is not None else min(complete_ids, key=lambda x: int(x))
    if ref_id not in complete_ids:
        print(f"error: --ref-marker {ref_id!r} was not fully solved (have: {complete_ids})",
              file=sys.stderr)
        sys.exit(1)
    ref_corners = corners_metric[ref_id]
    origin = np.mean(ref_corners, axis=0)
    ex = ref_corners[1] - ref_corners[0]
    ex /= np.linalg.norm(ex)
    ey_raw = ref_corners[3] - ref_corners[0]
    ey = ey_raw - np.dot(ey_raw, ex) * ex  # Gram-Schmidt: orthogonal even if not exactly square
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    R0 = np.stack([ex, ey, ez], axis=0)  # rows = rig-local axes expressed in the metric world frame
    print(f"Rig-local frame anchored on marker {ref_id} (origin at its corner centroid).")

    # Sanity-check diagnostic: solved "orbit" camera-center distances from the
    # rig, now that apply_similarity_transform above has put states.R/.t in
    # the same metric scale as origin/corners_metric. A real orbit video
    # should show a fairly tight spread (one camera at roughly one working
    # distance) -- wildly inconsistent distances usually mean some sampled
    # frames solved into a disconnected/wrong SIFT component.
    cam_dists = []
    for s in states:
        if s.R is None:
            continue
        C = -s.R.T @ s.t.flatten()
        cam_dists.append(float(np.linalg.norm(C - origin)))
    if cam_dists:
        d = np.array(cam_dists)
        print(f"Solved camera-to-rig distance: min={d.min():.3f}m mean={d.mean():.3f}m "
              f"max={d.max():.3f}m (n={len(d)}) -- sanity-check against the actual orbit radius.")

    marker_corners_out: dict[str, list[list[float]]] = {}
    n_incomplete = 0
    for marker_id, corners in corners_metric.items():
        if any(c is None for c in corners):
            n_incomplete += 1
            continue
        marker_corners_out[marker_id] = [
            (R0 @ (c - origin)).tolist() for c in corners
        ]
    if n_incomplete:
        print(f"NOTE: {n_incomplete} marker(s) had fewer than 2 cameras seeing all 4 corners; "
              f"omitted from output.")

    output_path = args.output or f"{video_path}.rig_config.json"
    payload = {
        "v": 1,
        "shape": "explicit",
        "dict": args.dict,
        "marker_size_m": args.marker_size,
        "ref_marker_id": ref_id,
        "marker_corners": marker_corners_out,
        "_provenance": {
            "source": "orbit_video_self_calibration",
            "video": str(Path(video_path).resolve()),
            "camera_label": args.camera_label,
            "n_frames_sampled": n_frames,
            "n_frames_solved": n_solved,
            "n_markers": len(marker_corners_out),
            "scale_factor_network_to_metric": scale,
            "edge_length_rel_std": rel_std,
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {len(marker_corners_out)} marker(s) to {output_path}")
    print(f"  frames sampled={n_frames} solved={n_solved}  scale k={scale:.6f}  "
          f"edge rel.std={rel_std * 100:.1f}%")


if __name__ == "__main__":
    main()

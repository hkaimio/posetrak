# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""calibrate_harness_from_orbit.py — recalibrate a rigid marker body's ArUco
geometry (and, optionally, the reflective dots mounted on the same rigid
piece) from a single moving camera orbiting a stationary rig, anchored by
one or two extra static reference markers placed in the scene.

Built to fix a real, confirmed problem: `calibrate_rigid_marker_body.py`'s
original method (average independent per-instant PnP solves from several
*simultaneous* fixed cameras) gave a stable ~5.6cm-in-plane bias between the
sword-bokken's two ArUco tags -- confirmed NOT a small-sample artifact by
re-running it at the finest possible temporal stride (9 -> 54 co-occurrence
samples converged onto the exact same biased answer, standard error well
under 0.3cm). A reshoot of the original multi-camera rig wasn't possible
(two of the six cameras broke, the rig is gone) -- but the two ArUco tags'
shared rigid harness still exists, separated from the sword, along with 4 of
the sword's 7 reflective dots (the other 3 were mounted elsewhere on the
sword itself and aren't recoverable this way). This script recalibrates the
harness from a single camera slowly orbiting it, which needs a fundamentally
different data flow than the original multi-fixed-camera script: there is
no pre-solved camera pose to look up, since the camera itself is moving
throughout.

Method: one or two extra static ArUco markers (--anchor-ids) placed near the
harness, visible from different sides of the orbit, define a fixed world
frame (the first anchor's own local frame). For every decoded frame:

1. Solve the camera's own world pose via single-image PnP against whichever
   anchor is visible that frame (bootstrapping the second anchor's own
   world-frame position first, from frames where both anchors are visible
   together, if a second anchor is given).
2. Whenever a harness marker (--marker-ids) is *also* visible that frame,
   solve its pose-in-camera the same way, then compose with the camera's
   now-known world pose to get that marker's corners in world frame.

Every such per-frame world-frame sample for a harness marker is independent
of every other -- unlike the original method, no two markers need to be
seen in the same instant, since each is independently triangulated back
through the same fixed world frame. A single continuous orbit can produce
hundreds to thousands of samples per marker instead of a few dozen.

Dots (--dot-names, --old-body-yaml): a reflective dot carries no identity of
its own, so correspondence across frames still needs *something* to anchor
it to -- solved here by projecting each dot's existing (if biased)
local-frame position from the old calibration through the now-known,
accurate camera pose each frame, and matching the nearest real detected
candidate within --dot-match-gate-px. The old calibration's error only
needs to be small enough not to confuse two different real dots with each
other at any given viewing distance, not accurate in any absolute sense --
the whole point is that the *matched* real detections then get triangulated
completely fresh from scratch (matches, not positions, are what carries
over from the old data). Matched observations across the whole orbit are
triangulated with iterative outlier rejection (_triangulate_robust) rather
than the original all-or-nothing per-view gate (triangulate_point_multi_view
in calibrate_rigid_marker_body.py): with hundreds of contributing views,
demanding *every single one* stay under the reprojection threshold would
reject almost every point outright over an isolated bad frame, where what's
actually needed is to identify and drop just the bad frames.

Any marker present in --old-body-yaml but not touched by this run (the 3
dots not on the harness, or a marker skipped for having too few samples) is
carried through to the output unchanged, so the result is one complete,
ready-to-use marker body rather than a partial file needing manual merging.

Usage:
    python tools/calibrate_harness_from_orbit.py \\
        --session /path/to/session.db \\
        --video "D:\\mocap\\2026-08-20-tutorial\\sword_harness_insta_ace2_4k_120fps.mp4" \\
        --camera-label insta_ace2_pro --reference-shot-id <a real capture using this camera> \\
        --dictionary DICT_4X4_50 \\
        --anchor-ids 0 1 --anchor-marker-size 0.095 \\
        --marker-ids 2 3 --reference-id 2 --marker-size 0.095 \\
        --dot-names dot1 dot2 dot4 dot5 --old-body-yaml scratch/sword_body_with_dots.yaml \\
        --output scratch/sword_harness_recalibrated.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import sqlite3
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.setup.extrinsics_solver import CamCalibState, _undistort_pts, marker_local_corners  # noqa: E402
from app.setup.fiducial_markers import ArucoDetector  # noqa: E402
from posetrak.detection.dot_blob_detector import detect_blobs  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402
from tools.calibrate_rigid_marker_body import _resolve_intrinsics, robust_mean  # noqa: E402


def _load_camera_intrinsics(
    conn: sqlite3.Connection, camera_label: str, reference_shot_id: str
) -> CamCalibState:
    """Reuse an already-solved intrinsics calibration for *camera_label*, as
    used in *reference_shot_id* -- the orbit video isn't itself a registered
    capture, but the same physical camera/lens/settings already has a
    trusted intrinsics calibration from a real capture."""
    cam_row = conn.execute(
        "SELECT id FROM camera_instances WHERE label = ?", (camera_label,)
    ).fetchone()
    if cam_row is None:
        raise SystemExit(f"camera label not found: {camera_label}")
    cam_id = cam_row["id"]

    row = conn.execute(
        "SELECT cv.intrinsics_calibration_id AS cv_calib_id, "
        "       cm.default_intrinsics_calibration_id AS mode_default_calib_id "
        "FROM capture_videos cv LEFT JOIN camera_modes cm ON cm.id = cv.camera_mode_id "
        "WHERE cv.shot_id = ? AND cv.camera_instance_id = ?",
        (reference_shot_id, cam_id),
    ).fetchone()
    if row is None:
        raise SystemExit(f"{camera_label} was not used in shot {reference_shot_id}")
    intr_id = row["cv_calib_id"] or row["mode_default_calib_id"]
    if intr_id is None:
        raise SystemExit(f"{camera_label} has no intrinsics calibration in {reference_shot_id}")
    ic = conn.execute("SELECT * FROM intrinsics_calibrations WHERE id = ?", (intr_id,)).fetchone()
    if ic is None:
        raise SystemExit(f"intrinsics_calibrations row not found: {intr_id}")
    return CamCalibState(video_id=cam_id, label=camera_label, **_resolve_intrinsics(ic))


def _kabsch_fit(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid (no scale, no reflection) fit: B ~= R @ A + t, via SVD.

    A, B: (N, 3) corresponding point sets. Used both to solve a marker's
    own pose from its known local template + averaged world corners, and
    (inverted) to re-express one marker's world position in another's
    local frame.
    """
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    H = (A - ca).T @ (B - cb)
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    return R, t


def _solve_pose(img_pts_undist: np.ndarray, obj_pts: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Single-view PnP: obj_pts (its own local/world frame) -> camera frame.
    Points are already undistorted, so distortion is zero and K is the
    undistorted matrix -- same convention the rest of this codebase uses."""
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts_undist, K, np.zeros(4))
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def _local_to_world(
    R_wc: np.ndarray, t_wc: np.ndarray, R_lc: np.ndarray, t_lc: np.ndarray, local_pts: np.ndarray
) -> np.ndarray:
    """local_pts (N,3), known via their own rigid body's pose-in-camera
    (R_lc, t_lc: local -> camera) this frame, converted to world frame using
    this same frame's camera-pose-in-world (R_wc, t_wc: world -> camera)."""
    cam_pts = (R_lc @ local_pts.T).T + t_lc
    return (R_wc.T @ (cam_pts - t_wc).T).T


def _corners_xy(detection) -> np.ndarray:
    return np.array([(c.px, c.py) for c in detection.corners], dtype=np.float64)


def _triangulate_robust(
    observations: dict[int, tuple[float, float]], poses: dict[int, tuple[np.ndarray, np.ndarray]],
    K: np.ndarray, outlier_px: float = 15.0, max_rounds: int = 5,
) -> tuple[np.ndarray, int, float] | None:
    """DLT triangulation from many (possibly hundreds of) world-pose-known
    views, with iterative outlier rejection instead of
    triangulate_point_multi_view()'s all-or-nothing per-view gate -- see
    module docstring for why that gate doesn't fit a many-view input.

    poses: frame_key -> (R_wc, t_wc), world -> camera. Points are already
    undistorted, so K is the undistorted matrix and distortion is zero.

    Returns (point_world, n_inliers, rms_reprojection_px), or None if fewer
    than 2 views survive.
    """
    keys = list(observations)
    for _round in range(max_rounds):
        rows = []
        proj = {}
        for key in keys:
            R_wc, t_wc = poses[key]
            P = K @ np.hstack([R_wc, t_wc.reshape(3, 1)])
            proj[key] = P
            u, v = observations[key]
            rows.append(u * P[2] - P[0])
            rows.append(v * P[2] - P[1])
        if len(rows) < 4:
            return None
        A = np.stack(rows)
        _, _, vt = np.linalg.svd(A)
        x = vt[-1]
        if abs(x[3]) < 1e-8:
            return None
        point = x[:3] / x[3]

        point_h = np.append(point, 1.0)
        errors = {}
        for key in keys:
            p = proj[key] @ point_h
            if abs(p[2]) < 1e-8:
                errors[key] = np.inf
                continue
            reproj = p[:2] / p[2]
            u, v = observations[key]
            errors[key] = float(np.hypot(reproj[0] - u, reproj[1] - v))

        survivors = [k for k in keys if errors[k] <= outlier_px]
        if len(survivors) == len(keys) or len(survivors) < 2:
            rms = float(np.sqrt(np.mean([errors[k] ** 2 for k in survivors]))) if survivors else float("inf")
            return (point, len(survivors), rms) if len(survivors) >= 2 else None
        keys = survivors
    rms = float(np.sqrt(np.mean([errors[k] ** 2 for k in keys])))
    return point, len(keys), rms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--camera-label", required=True)
    ap.add_argument("--reference-shot-id", required=True,
                     help="A real capture that used this camera, to reuse its intrinsics calibration.")
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    ap.add_argument("--anchor-ids", nargs="+", required=True,
                     help="1 or 2 static reference marker IDs placed in the scene. The first is the world frame origin.")
    ap.add_argument("--anchor-marker-size", type=float, required=True)
    ap.add_argument("--marker-ids", nargs="+", required=True, help="The rigid body's own ArUco IDs.")
    ap.add_argument("--reference-id", required=True, help="Which --marker-ids anchors the body's own local frame.")
    ap.add_argument("--marker-size", type=float, required=True)
    ap.add_argument("--min-marker-perimeter-rate", type=float, default=0.01,
                     help="See ArucoDetector's own docstring: OpenCV's 0.03 default silently misses "
                          "markers seen from across a room; this project's real footage needs it lower.")
    ap.add_argument("--dot-names", nargs="*", default=[],
                     help="Reflective dots also mounted on this rigid body, to recalibrate alongside the ArUco tags.")
    ap.add_argument("--old-body-yaml", default=None,
                     help="Existing marker body YAML: supplies --dot-names' prior positions (for frame-to-frame "
                          "correspondence only, not carried into the result) and every other marker to pass "
                          "through unchanged into the output.")
    ap.add_argument("--dot-match-gate-px", type=float, default=60.0,
                     help="Max distance between a raw detected candidate and a dot's old-calibration-predicted "
                          "position to accept the match.")
    ap.add_argument("--dot-triangulation-outlier-px", type=float, default=15.0)
    ap.add_argument("--dot-threshold", type=int, default=235)
    ap.add_argument("--dot-min-area", type=float, default=4.0)
    ap.add_argument("--dot-max-area", type=float, default=400.0)
    ap.add_argument("--dot-min-compactness", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame.")
    ap.add_argument("--max-frames", type=int, default=None, help="Debug: stop after decoding this many frames.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--name", default="calibrated-rigid-body")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cam_state = _load_camera_intrinsics(conn, args.camera_label, args.reference_shot_id)

    old_doc = None
    old_dot_local: dict[str, np.ndarray] = {}
    if args.old_body_yaml:
        old_doc = yaml.safe_load(Path(args.old_body_yaml).read_text())
        for m in old_doc.get("markers", []):
            if m["name"] in args.dot_names and m.get("type") == "reflective_dot":
                old_dot_local[m["name"]] = np.array(m["center"], dtype=np.float64)
        missing = set(args.dot_names) - set(old_dot_local)
        if missing:
            raise SystemExit(f"--dot-names not found as reflective_dot entries in --old-body-yaml: {missing}")

    all_ids = set(args.anchor_ids) | set(args.marker_ids)
    detector = ArucoDetector(dictionary=args.dictionary, min_marker_perimeter_rate=args.min_marker_perimeter_rate)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"video: {total} frames, processing every {args.stride}")

    anchor0_id, anchor1_id = (args.anchor_ids + [None])[:2]
    anchor0_template = marker_local_corners(args.anchor_marker_size)
    anchor1_template = marker_local_corners(args.anchor_marker_size) if anchor1_id else None
    harness_template = marker_local_corners(args.marker_size)

    detections_by_frame: dict[int, dict[str, np.ndarray]] = {}
    dot_candidates_by_frame: dict[int, np.ndarray] = {}
    n_decoded = 0
    last_frame = min(total, args.max_frames) if args.max_frames else total
    for frame_idx, img in iter_frames(args.video, 0, last_frame):
        n_decoded += 1
        if (frame_idx % args.stride) != 0:
            continue
        dets = detector.detect(img, video_id=args.camera_label, frame_idx=frame_idx)
        by_id = {d.marker_id: _undistort_pts(_corners_xy(d), cam_state) for d in dets if d.marker_id in all_ids}
        if by_id:
            detections_by_frame[frame_idx] = by_id
        if args.dot_names:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blobs = detect_blobs(gray, threshold=args.dot_threshold, min_area=args.dot_min_area,
                                  max_area=args.dot_max_area, min_compactness=args.dot_min_compactness)
            if blobs:
                pts = np.array([(b.cx, b.cy) for b in blobs], dtype=np.float64)
                dot_candidates_by_frame[frame_idx] = _undistort_pts(pts, cam_state)
        if n_decoded % 1000 == 0:
            print(f"  {n_decoded}/{total} decoded, {len(detections_by_frame)} frames with a relevant marker so far")

    print(f"{len(detections_by_frame)} frames saw at least one relevant marker")

    # Pass 1: bootstrap anchor1's world-frame corners, if a second anchor was given.
    anchor1_world: np.ndarray | None = None
    if anchor1_id:
        samples = []
        for by_id in detections_by_frame.values():
            if anchor0_id not in by_id or anchor1_id not in by_id:
                continue
            pose0 = _solve_pose(by_id[anchor0_id], anchor0_template, cam_state.K)
            pose1 = _solve_pose(by_id[anchor1_id], anchor1_template, cam_state.K)
            if pose0 is None or pose1 is None:
                continue
            R_wc, t_wc = pose0
            R_1c, t_1c = pose1
            samples.append(_local_to_world(R_wc, t_wc, R_1c, t_1c, anchor1_template))
        print(f"anchor '{anchor1_id}' bootstrapped from {len(samples)} frames with both anchors visible")
        if samples:
            stacked = np.stack(samples)
            anchor1_world = np.stack([robust_mean(stacked[:, i, :]) for i in range(4)])
        else:
            print(f"  WARNING: anchor '{anchor1_id}' never co-visible with '{anchor0_id}' -- "
                  f"frames seeing only '{anchor1_id}' will be unusable")

    # Pass 2: per-frame camera world pose, then harness marker world-frame samples
    # and (if requested) dot candidate-to-prediction matching.
    harness_samples: dict[str, list[np.ndarray]] = {mid: [] for mid in args.marker_ids}
    camera_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    n_cam_solved = 0
    for frame_idx, by_id in detections_by_frame.items():
        cam_pose = None
        if anchor0_id in by_id:
            cam_pose = _solve_pose(by_id[anchor0_id], anchor0_template, cam_state.K)
        elif anchor1_id and anchor1_id in by_id and anchor1_world is not None:
            cam_pose = _solve_pose(by_id[anchor1_id], anchor1_world, cam_state.K)
        if cam_pose is None:
            continue
        n_cam_solved += 1
        camera_poses[frame_idx] = cam_pose
        R_wc, t_wc = cam_pose
        for mid in args.marker_ids:
            if mid not in by_id:
                continue
            pose_m = _solve_pose(by_id[mid], harness_template, cam_state.K)
            if pose_m is None:
                continue
            R_mc, t_mc = pose_m
            harness_samples[mid].append(_local_to_world(R_wc, t_wc, R_mc, t_mc, harness_template))

    print(f"camera world pose solved in {n_cam_solved} frames")
    for mid, samples in harness_samples.items():
        if samples:
            stacked = np.stack(samples)
            spread = np.stack([np.std(stacked[:, i, :], axis=0) for i in range(4)]).mean(axis=0)
            print(f"  marker '{mid}': {len(samples)} world-frame samples, mean per-axis std (m) {spread}")
        else:
            print(f"  marker '{mid}': 0 samples")

    ref_id = args.reference_id
    if not harness_samples.get(ref_id):
        raise SystemExit(f"reference marker '{ref_id}' has no samples -- cannot anchor the body frame")
    ref_stacked = np.stack(harness_samples[ref_id])
    ref_world = np.stack([robust_mean(ref_stacked[:, i, :]) for i in range(4)])
    R_ref, t_ref = _kabsch_fit(harness_template, ref_world)  # local -> world for the reference marker

    # Dots: match raw candidates to old-calibration predictions, then triangulate fresh.
    dot_world: dict[str, np.ndarray] = {}
    if args.dot_names:
        dot_observations: dict[str, dict[int, tuple[float, float]]] = {name: {} for name in args.dot_names}
        for frame_idx, (R_wc, t_wc) in camera_poses.items():
            candidates = dot_candidates_by_frame.get(frame_idx)
            if candidates is None or len(candidates) == 0:
                continue
            for name in args.dot_names:
                world_guess = R_ref @ old_dot_local[name] + t_ref
                cam_guess = R_wc @ world_guess + t_wc
                if cam_guess[2] <= 0:
                    continue
                pred_px = cam_state.K @ (cam_guess / cam_guess[2])
                pred_px = pred_px[:2]
                d = np.linalg.norm(candidates - pred_px, axis=1)
                best = int(np.argmin(d))
                if d[best] <= args.dot_match_gate_px:
                    dot_observations[name][frame_idx] = tuple(candidates[best])

        for name, obs in dot_observations.items():
            if len(obs) < 2:
                print(f"  dot '{name}': only {len(obs)} matched observations, skipping")
                continue
            poses = {k: camera_poses[k] for k in obs}
            result = _triangulate_robust(obs, poses, cam_state.K, args.dot_triangulation_outlier_px)
            if result is None:
                print(f"  dot '{name}': {len(obs)} matched observations, triangulation failed")
                continue
            point_world, n_inliers, rms = result
            local_pt = R_ref.T @ (point_world - t_ref)
            dot_world[name] = local_pt
            old = old_dot_local[name]
            shift = np.linalg.norm(local_pt - old)
            print(f"  dot '{name}': {len(obs)} matched / {n_inliers} inliers, rms={rms:.2f}px, "
                  f"shift from old calibration={shift * 100:.2f}cm")

    def _fmt(v: np.ndarray) -> str:
        return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"

    lines = [f"name: {args.name}", "units: meters", "markers:"]
    lines.append(f"  - name: aruco_{ref_id}")
    lines.append("    type: aruco")
    lines.append(f"    dictionary: {args.dictionary}")
    lines.append(f'    id: "{ref_id}"')
    lines.append(f"    size: {args.marker_size}")
    lines.append("    corners:")
    for c in harness_template:
        lines.append(f"      - {_fmt(c)}")

    written_names = {f"aruco_{ref_id}"}
    for mid in args.marker_ids:
        if mid == ref_id or not harness_samples[mid]:
            continue
        stacked = np.stack(harness_samples[mid])
        avg_world = np.stack([robust_mean(stacked[:, i, :]) for i in range(4)])
        local_in_ref = (R_ref.T @ (avg_world - t_ref).T).T
        lines.append(f"  - name: aruco_{mid}")
        lines.append("    type: aruco")
        lines.append(f"    dictionary: {args.dictionary}")
        lines.append(f'    id: "{mid}"')
        lines.append(f"    size: {args.marker_size}")
        lines.append("    corners:")
        for c in local_in_ref:
            lines.append(f"      - {_fmt(c)}")
        written_names.add(f"aruco_{mid}")

    for name, pt in dot_world.items():
        lines.append(f"  - name: {name}")
        lines.append("    type: reflective_dot")
        lines.append(f"    center: {_fmt(pt)}")
        written_names.add(name)

    if old_doc is not None:
        for m in old_doc.get("markers", []):
            if m["name"] in written_names:
                continue
            lines.append(f"  - name: {m['name']}")
            lines.append(f"    type: {m['type']}")
            if m["type"] == "aruco":
                lines.append(f"    dictionary: {m['dictionary']}")
                lines.append(f'    id: "{m["id"]}"')
                lines.append(f"    size: {m['size']}")
                lines.append("    corners:")
                for c in m["corners"]:
                    lines.append(f"      - {_fmt(np.asarray(c, dtype=np.float64))}")
            elif m["type"] == "reflective_dot":
                lines.append(f"    center: {_fmt(np.asarray(m['center'], dtype=np.float64))}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

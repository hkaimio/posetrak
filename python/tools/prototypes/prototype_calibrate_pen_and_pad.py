# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_calibrate_pen_and_pad.py -- PROTOTYPE, hardcoded to this
session's pen+pad props (see person-marker-mocap-productization-plan.md's
own "prototype first" convention). Calibrates each rigid object's own
marker geometry from a single dedicated calibration video, before any of
it is wired into the main tracking pipeline.

Pad: two ArUco tags, co-visible in essentially every frame of its own
calibration clip -- straightforward per-frame relative pose + robust
average, no bridging needed.

Pen: a flat paddle with one ArUco tag per face (so the two tags face
opposite directions and are never co-visible), plus two reflective dots
on the rod's own edge/tip -- visible from either side regardless of which
tag is showing. Calibrated in two independent steps, each reusing
prototype_multi_camera_fusion.triangulate_multiview() unmodified: for
whichever tag is visible in a given frame, that tag's own solvePnP pose
IS a full 6-DOF "camera pose relative to a moving reference frame" --
exactly the shape triangulate_multiview() already expects (a list of
projection matrices + one pixel per view), just reinterpreted as "views
across time of a fixed point in the tag's own local frame" instead of
"views across simultaneous cameras". Do this once per dot per tag, then
register the two tags' own local frames together using the two dots as
shared points (Kabsch on the 2-point case) -- see this script's own
--help and status.md for the real, honestly-reported limitation this
leaves: 2 points constrain translation and the axis between them, but
leave one rotational DOF (twist around that axis, i.e. the pen's own
roll) unconstrained. Not resolved here; flagged, not glossed over.

Usage:
    python tools/prototype_calibrate_pen_and_pad.py \\
        --session /path/to/session.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --pad-video /path/to/calib-board-gopro13_02-4k-120fps.MP4 \\
        --pen-video /path/to/calib-pen-gopro13_02-4k-120fps.MP4 \\
        --pad-marker-ids 40 41 --pen-marker-ids 42 43 \\
        --marker-size-m 0.095
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.setup.fiducial_markers import ArucoDetector  # noqa: E402
from posetrak.detection.dot_blob_detector import detect_blobs  # noqa: E402
from tools.prototype_multi_camera_fusion import triangulate_multiview  # noqa: E402

import sqlite3  # noqa: E402


def _resolve_intrinsics(ic) -> dict:
    """Same as calibrate_rigid_marker_body.py's own private helper -- K_orig
    + raw distortion, used directly in solvePnP so no separate undistort
    pass is needed."""
    fx, fy, cx, cy = ic["fx"], ic["fy"], ic["cx"], ic["cy"]
    K_orig = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    if ic["matrix_original"]:
        vals = struct.unpack("<9d", bytes(ic["matrix_original"]))
        K_orig = np.array(vals).reshape(3, 3)
    if ic["dist_coeffs"]:
        n = len(bytes(ic["dist_coeffs"])) // 8
        dist = np.array(struct.unpack(f"<{n}d", bytes(ic["dist_coeffs"]))).reshape(1, -1)
    else:
        dist = np.zeros((1, 4))
    return {"K_orig": K_orig, "dist": dist}


def load_gopro13_02_intrinsics(conn: sqlite3.Connection, shot_id: str) -> dict:
    row = conn.execute(
        """
        SELECT cv.intrinsics_calibration_id AS cv_calib_id,
               cm.default_intrinsics_calibration_id AS mode_default_calib_id
        FROM capture_videos cv
        JOIN camera_instances ci ON ci.id = cv.camera_instance_id
        LEFT JOIN camera_modes cm ON cm.id = cv.camera_mode_id
        WHERE cv.shot_id = ? AND ci.label = 'gopro13_02'
        """,
        (shot_id,),
    ).fetchone()
    if row is None:
        raise ValueError("gopro13_02 not found for this shot")
    calib_id = row["cv_calib_id"] or row["mode_default_calib_id"]
    if calib_id is None:
        raise ValueError("gopro13_02 has no intrinsics calibration")
    ic = conn.execute("SELECT * FROM intrinsics_calibrations WHERE id = ?", (calib_id,)).fetchone()
    return _resolve_intrinsics(ic)


def solve_pnp(detection, K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (R, t) mapping the marker's own local frame -> camera frame,
    or None if PnP fails."""
    obj_pts = np.array(detection.corner_local_xyz).reshape(-1, 1, 3).astype(np.float64)
    img_pts = np.array([[c.px, c.py] for c in detection.corners], dtype=np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.flatten()


def calibrate_pad(video_path: str, marker_ids: tuple[str, str], K, dist, marker_size: float,
                   stride: int, dictionary: str) -> None:
    detector = ArucoDetector(dictionary=dictionary, default_size=marker_size,
                             min_marker_perimeter_rate=0.01)
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rel_translations = []
    rel_rotations = []
    n_both = 0
    for fidx in range(0, n_frames, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        dets = {d.marker_id: d for d in detector.detect(frame)}
        if marker_ids[0] not in dets or marker_ids[1] not in dets:
            continue
        pose0 = solve_pnp(dets[marker_ids[0]], K, dist)
        pose1 = solve_pnp(dets[marker_ids[1]], K, dist)
        if pose0 is None or pose1 is None:
            continue
        R0, t0 = pose0
        R1, t1 = pose1
        # marker1's pose expressed in marker0's own local frame:
        R_rel = R0.T @ R1
        t_rel = R0.T @ (t1 - t0)
        rel_translations.append(t_rel)
        rel_rotations.append(cv2.Rodrigues(R_rel)[0].flatten())
        n_both += 1
    cap.release()

    print(f"\n=== pad: marker {marker_ids[1]} relative to marker {marker_ids[0]} ===")
    print(f"{n_both} frames with both markers detected (of {n_frames // stride} sampled)")
    if n_both == 0:
        print("  no usable frames")
        return
    t_arr = np.array(rel_translations)
    r_arr = np.array(rel_rotations)
    t_med = np.median(t_arr, axis=0)
    r_med = np.median(r_arr, axis=0)
    t_std = t_arr.std(axis=0)
    angle_deg = np.degrees(np.linalg.norm(r_med))
    print(f"  median translation (m): {t_med.round(4)}  (per-axis std: {t_std.round(4)})")
    print(f"  median rotation: axis-angle {r_med.round(4)}  ({angle_deg:.2f} deg)")
    print(f"  translation magnitude: {np.linalg.norm(t_med)*100:.2f} cm")


def collect_tag_frames_and_dots(
    video_path: str, tag_id: str, K, dist, marker_size: float, stride: int, dictionary: str,
    dot_kwargs: dict, gate_px: float = 100.0, bootstrap_compactness: float = 0.7,
) -> tuple[list[np.ndarray], list[tuple[float, float]], list[tuple[float, float]]]:
    """For every frame where *tag_id* is detected: solve its pose and track
    the two dots -- "near" (paddle-mounted) and "far" (rod-tip) -- across
    consecutive frames with a proximity gate, rather than re-searching the
    whole frame each time.

    Why gating, not a one-shot per-frame nearest-to-tag search (the first
    version of this script): this scene has real, extensive clutter --
    sunlit floor glare fragments into dozens of blob candidates, some
    genuinely closer to the tag in pixel space than the real rod-tip dot
    is. A first attempt's per-frame "nearest 2 to the tag" search picked up
    floor glare as the "far" dot in nearly every frame (reprojection error
    ~200-1000px, caught by _robust_triangulate()'s own diagnostic -- see
    status.md). A compactness-only filter fixed the floor-cluster false
    positives but then rejected the real far dot whenever it wasn't
    cleanly round in a given frame, converging on an isolated floor speck
    instead. Gating fixes both failure modes together: a floor blob does
    not move rigidly with the tag between consecutive frames, so it fails
    the gate almost immediately even if briefly nearest; a real dot that's
    momentarily less compact still passes, because gating only cares where
    it was last seen, not its shape.

    Bootstraps on the first frame with >=2 compactness>=`bootstrap_compactness`
    candidates (falls back to the two nearest-to-tag candidates of any
    shape if no frame ever has two compact ones); every frame after that
    is tracked via the gate alone. A dot with no candidate inside
    `gate_px` of its last known position simply doesn't contribute that
    frame -- it is not reseeded from a wrong guess.

    Returns (proj_near, near_dot_px, proj_far, far_dot_px) -- near/far are
    tracked (and can be lost/gated out) independently, so each gets its
    own projection-matrix list paired with its own pixel list; the two
    need not have the same length or come from the same frames.
    """
    detector = ArucoDetector(dictionary=dictionary, default_size=marker_size,
                             min_marker_perimeter_rate=0.01)
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    proj_near, near_px = [], []
    proj_far, far_px = [], []
    prev_near: np.ndarray | None = None
    prev_far: np.ndarray | None = None
    n_seen = n_tag_pose_ok = n_near_tracked = n_far_tracked = 0

    for fidx in range(0, n_frames, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        dets = {d.marker_id: d for d in detector.detect(frame)}
        if tag_id not in dets:
            continue
        n_seen += 1
        pose = solve_pnp(dets[tag_id], K, dist)
        if pose is None:
            continue
        n_tag_pose_ok += 1
        R, t = pose
        P = K @ np.hstack([R, t.reshape(3, 1)])  # see module docstring: tag's own frame as "world"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blobs = detect_blobs(gray, bgr=frame, **dot_kwargs)
        if not blobs:
            continue
        positions = np.array([[b.cx, b.cy] for b in blobs])

        if prev_near is None or prev_far is None:
            tag_center = np.mean([[c.px, c.py] for c in dets[tag_id].corners], axis=0)
            compact = [b for b in blobs if b.compactness >= bootstrap_compactness]
            pool = compact if len(compact) >= 2 else blobs
            pool_sorted = sorted(pool, key=lambda b: np.hypot(b.cx - tag_center[0], b.cy - tag_center[1]))
            if len(pool_sorted) < 2:
                continue
            prev_near = np.array([pool_sorted[0].cx, pool_sorted[0].cy])
            prev_far = np.array([pool_sorted[1].cx, pool_sorted[1].cy])
            continue  # bootstrap frame itself isn't used as a triangulation sample

        d_near = np.hypot(positions[:, 0] - prev_near[0], positions[:, 1] - prev_near[1])
        i_near = int(np.argmin(d_near))
        if d_near[i_near] <= gate_px:
            prev_near = positions[i_near]
            proj_near.append(P)
            near_px.append((float(positions[i_near, 0]), float(positions[i_near, 1])))
            n_near_tracked += 1

        d_far = np.hypot(positions[:, 0] - prev_far[0], positions[:, 1] - prev_far[1])
        i_far = int(np.argmin(d_far))
        if d_far[i_far] <= gate_px:
            prev_far = positions[i_far]
            proj_far.append(P)
            far_px.append((float(positions[i_far, 0]), float(positions[i_far, 1])))
            n_far_tracked += 1
    cap.release()
    print(f"  tag {tag_id}: seen in {n_seen} sampled frames, pose solved in {n_tag_pose_ok}, "
          f"near-dot tracked in {n_near_tracked}, far-dot tracked in {n_far_tracked}")
    return proj_near, near_px, proj_far, far_px


def _reprojection_errors(point_3d: np.ndarray, P_list: list[np.ndarray],
                          px_list: list[tuple[float, float]]) -> np.ndarray:
    errs = []
    for P, (u, v) in zip(P_list, px_list):
        h = P @ np.append(point_3d, 1.0)
        proj = h[:2] / h[2]
        errs.append(np.hypot(proj[0] - u, proj[1] - v))
    return np.array(errs)


def _robust_triangulate(P_list: list[np.ndarray], px_list: list[tuple[float, float]],
                         label: str, max_reproj_px: float = 15.0):
    """DLT triangulation (unmodified, existing triangulate_multiview()), then
    one round of outlier trimming by reprojection error -- plain DLT/SVD has
    no built-in robustness, and this session's very first cross-check run
    found a real, large (73%) inconsistency that turned out to need this."""
    point = triangulate_multiview(P_list, px_list)
    if point is None:
        return None, np.array([])
    errs = _reprojection_errors(point, P_list, px_list)
    keep = errs <= max_reproj_px
    n_bad = (~keep).sum()
    if n_bad > 0:
        print(f"    [{label}] {n_bad}/{len(errs)} frames over {max_reproj_px}px reprojection "
              f"error on the first pass (max {errs.max():.1f}px) -- re-solving without them")
        P_kept = [P for P, k in zip(P_list, keep) if k]
        px_kept = [p for p, k in zip(px_list, keep) if k]
        if len(P_kept) >= 3:
            point = triangulate_multiview(P_kept, px_kept)
            errs = _reprojection_errors(point, P_kept, px_kept)
    print(f"    [{label}] final reprojection error: median={np.median(errs):.2f}px "
          f"p90={np.percentile(errs, 90):.2f}px max={errs.max():.2f}px (n={len(errs)})")
    return point, errs


def calibrate_pen(video_path: str, tag_ids: tuple[str, str], K, dist, marker_size: float,
                   stride: int, dictionary: str, dot_kwargs: dict, gate_px: float = 100.0) -> None:
    print(f"\n=== pen: tags {tag_ids[0]} / {tag_ids[1]} + 2 bridging dots (gate={gate_px}px) ===")
    results = {}
    for tag_id in tag_ids:
        proj_near, near_list, proj_far, far_list = collect_tag_frames_and_dots(
            video_path, tag_id, K, dist, marker_size, stride, dictionary, dot_kwargs, gate_px)
        if len(proj_near) < 3 or len(proj_far) < 3:
            print(f"  tag {tag_id}: only {len(proj_near)}/{len(proj_far)} near/far tracked "
                  "frames -- need >=3 each for a well-conditioned triangulation, skipping")
            continue
        near_3d, _ = _robust_triangulate(proj_near, near_list, f"tag {tag_id} near_dot")
        far_3d, _ = _robust_triangulate(proj_far, far_list, f"tag {tag_id} far_dot")
        if near_3d is None or far_3d is None:
            print(f"  tag {tag_id}: triangulation failed")
            continue
        dist_local = np.linalg.norm(near_3d - far_3d)
        print(f"  tag {tag_id}: near_dot={near_3d.round(4)}  far_dot={far_3d.round(4)}  "
              f"dot-to-dot distance={dist_local*100:.2f} cm  "
              f"(n_near={len(proj_near)} n_far={len(proj_far)} frames)")
        results[tag_id] = (near_3d, far_3d)

    if len(results) < 2:
        print("  could not solve both tags -- stopping")
        return

    (nA, fA), (nB, fB) = results[tag_ids[0]], results[tag_ids[1]]
    print(f"\n  cross-check: dot-to-dot distance from tag {tag_ids[0]}'s own view = "
          f"{np.linalg.norm(nA-fA)*100:.2f} cm, from tag {tag_ids[1]}'s own view = "
          f"{np.linalg.norm(nB-fB)*100:.2f} cm -- should closely agree if both tags' "
          "geometry and the dot detections are self-consistent.")

    # Register B's frame onto A's frame using the 2 shared points. Only
    # translation + the axis between the two points is actually determined
    # by 2 points (Kabsch/Procrustes needs >=3 non-collinear points for a
    # fully-constrained rotation) -- the twist around that axis (the pen's
    # own roll) is NOT resolved by this data and is left at whatever the
    # SVD below picks, not a measured quantity. Real, honest limitation --
    # see this script's own module docstring.
    local_B = np.array([nB, fB])
    local_A = np.array([nA, fA])
    centroid_B, centroid_A = local_B.mean(axis=0), local_A.mean(axis=0)
    H = (local_B - centroid_B).T @ (local_A - centroid_A)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R_AB = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t_AB = centroid_A - R_AB @ centroid_B
    angle_deg = np.degrees(np.linalg.norm(cv2.Rodrigues(R_AB)[0]))
    print(f"\n  tag-{tag_ids[1]}-to-tag-{tag_ids[0]} transform (2-point fit, twist DOF "
          f"unconstrained): translation={ (t_AB*100).round(2) } cm, rotation~{angle_deg:.1f} deg")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--pad-video", required=True)
    ap.add_argument("--pen-video", required=True)
    ap.add_argument("--pad-marker-ids", nargs=2, required=True)
    ap.add_argument("--pen-marker-ids", nargs=2, required=True)
    ap.add_argument("--marker-size-m", type=float, default=0.095)
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    ap.add_argument("--stride", type=int, default=4, help="Pad: sample every Nth frame.")
    ap.add_argument("--pen-stride", type=int, default=2,
                    help="Pen: sample every Nth frame -- smaller than --stride since the "
                         "dot tracking gate needs bounded motion between sampled frames.")
    ap.add_argument("--gate-px", type=float, default=100.0,
                    help="Pen dot tracking: max pixel distance from a dot's last known "
                         "position to still count as the same dot this frame.")
    ap.add_argument("--dot-threshold", type=int, default=235)
    ap.add_argument("--dot-min-area", type=float, default=10.0)
    ap.add_argument("--dot-max-area", type=float, default=4000.0)
    ap.add_argument("--dot-max-saturation", type=float, default=60.0)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    intr = load_gopro13_02_intrinsics(conn, args.shot_id)
    K, dist = intr["K_orig"], intr["dist"]
    print(f"gopro13_02 intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    calibrate_pad(args.pad_video, tuple(args.pad_marker_ids), K, dist, args.marker_size_m,
                 args.stride, args.dictionary)

    dot_kwargs = dict(threshold=args.dot_threshold, min_area=args.dot_min_area,
                      max_area=args.dot_max_area, max_saturation=args.dot_max_saturation)
    calibrate_pen(args.pen_video, tuple(args.pen_marker_ids), K, dist, args.marker_size_m,
                 args.pen_stride, args.dictionary, dot_kwargs, args.gate_px)


if __name__ == "__main__":
    main()

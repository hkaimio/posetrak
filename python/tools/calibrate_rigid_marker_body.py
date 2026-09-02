# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""calibrate_rigid_marker_body.py — Phase A/C1: rigid marker-body
calibration (ArUco corners, and optionally reflective-dot centroids) from
ordinary multi-camera performance footage.

See docs/roadmap/features/marker-based-mocap/rigid-marker-body-calibration-design.md
for the full design and real-data validation of the core assumption this
relies on (§6). No turn-around video needed, and no requirement that any
two markers ever be visible together from a single camera -- only that
across the whole capture, every marker being placed co-occurs with the
chosen reference marker in *some* synchronized frame (possibly seen by
entirely different cameras than the reference).

For each sampled frame, per camera: detect ArUco markers (undistorted
corners), and for every marker seen by >=2 cameras that frame, solve its
own rigid world pose via app.setup.extrinsics_solver.solve_marker_pose()
-- the exact numerics already used, in production, by the (structurally
identical) extrinsics-calibration path. Whichever configured --reference-id
is visible that frame anchors it; every other visible marker's world
corners get expressed in the reference's local frame and accumulated.
Robust-averages (per corner, per axis) each marker's samples across the
whole capture and writes a marker_body_definitions YAML using the
`corners:` form (marker_body definition's provenance-preserving form for
solved, not designed, geometry -- see app/setup/fiducial_markers.py's
load_marker_body_yaml docstring).

**Reflective-dot calibration** (Phase C1, `--detect-dots`; see
docs/roadmap/features/marker-based-mocap/reflective-dot-detection-design.md
§3.1): same co-occurrence-with-the-reference-marker mechanism, just
accumulating dot centroids (3 DOF: a point) instead of ArUco corners (6
DOF: a rigid transform). Multi-view correspondence (which blob in camera A
is the same physical dot as which blob in camera B) is the one genuinely
new sub-problem this needs that ArUco calibration doesn't -- an ArUco
corner's identity comes from the marker's own decoded ID, but a dot
candidate carries no identity at all. Resolved here by restricting to
buckets where at least two cameras *each saw exactly one candidate* that
instant: real GoPro footage from the sword's own validated capture
(reflective-dot-detection-design.md §2.1) has ~39% of frames with exactly
one candidate per camera against ~11% with more than one, so this
restriction still leaves plenty of usable samples per dot without needing
genuine epipolar-consistency correspondence matching (marker-detection-
analysis.md's Question B, layer 2) -- a real trade of some statistical
efficiency for a lot less complexity, not a correctness shortcut: every
sample this *does* use is an unambiguous, correctly-triangulated point.
Since dot identity isn't known ahead of time (unlike an ArUco's decoded
ID), triangulated samples across the whole capture are greedily clustered
by proximity in the reference marker's local frame (see
cluster_dot_samples()) into per-physical-dot groups before averaging.

This is a standalone script, not yet a `posetrak marker-body` CLI
subcommand -- intentionally, until validated against real data (marker-
mocap-design.md's own "Option 1 first, revisit if it proves fiddly"
precedent). Read-only against the session DB; writes only the output YAML.

Phase B (joint least-squares refine) is not implemented here -- see the
design doc's phasing.

Usage:
    python tools/calibrate_rigid_marker_body.py \\
        --session /path/to/session.db \\
        --shot-id <capture_id> \\
        --time-start 34.4 --time-end 100.6 \\
        --dictionary DICT_4X4_50 \\
        --marker-size 0.05 \\
        --marker-ids 2 3 \\
        --reference-id 2 \\
        --detect-dots \\
        --output sword_body.yaml
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.setup.db_context import SyncPoint, SyncTable  # noqa: E402
from app.setup.extrinsics_solver import (  # noqa: E402
    CamCalibState,
    _proj_matrix,
    _undistort_pts,
    marker_local_corners,
    solve_marker_pose,
)
from app.setup.fiducial_markers import ArucoDetector  # noqa: E402
from posetrak.detection.dot_blob_detector import detect_blobs  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402


def _resolve_intrinsics(ic: sqlite3.Row) -> dict:
    """Build the K/K_orig/dist/fisheye dict CamCalibState needs from an
    intrinsics_calibrations row -- copy of app.setup.page_extrinsics'
    private helper of the same name, kept local so this headless batch
    tool doesn't pull in that module's PySide6 dependency."""
    import struct

    fx, fy, cx, cy = ic["fx"], ic["fy"], ic["cx"], ic["cy"]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    K_orig = K.copy()
    if ic["matrix_original"]:
        vals = struct.unpack("<9d", bytes(ic["matrix_original"]))
        K_orig = np.array(vals).reshape(3, 3)
    if ic["dist_coeffs"]:
        n = len(bytes(ic["dist_coeffs"])) // 8
        dist = np.array(struct.unpack(f"<{n}d", bytes(ic["dist_coeffs"]))).reshape(1, -1)
    else:
        dist = np.zeros((1, 4))
    return {
        "K": K, "K_orig": K_orig, "dist": dist,
        "fisheye": ic["distortion_model"] == "fisheye",
    }


def load_camera_states(conn: sqlite3.Connection, shot_id: str) -> dict[str, CamCalibState]:
    """Load one CamCalibState per camera_instance_id covering *shot_id*,
    with R/t already filled from the capture's own solved extrinsics.

    Keyed by camera_instance_id (not camera label, unlike
    page_extrinsics._load_states_from_capture) -- this script needs to
    join against sync_points/capture_videos/extrinsic_entries, which all
    key on camera_instance_id, so using the same key throughout avoids a
    label round-trip.
    """
    row = conn.execute(
        "SELECT extrinsic_calibration_id FROM captures WHERE id = ?", (shot_id,)
    ).fetchone()
    if row is None or row["extrinsic_calibration_id"] is None:
        raise ValueError(f"capture {shot_id!r} has no solved extrinsic_calibration_id")
    calib_id = row["extrinsic_calibration_id"]

    extr_rows = conn.execute(
        "SELECT camera_instance_id, R, t FROM extrinsic_entries WHERE extrinsic_calibration_id = ?",
        (calib_id,),
    ).fetchall()
    extrinsics = {
        r["camera_instance_id"]: (
            np.frombuffer(bytes(r["R"]), dtype=np.float64).reshape(3, 3),
            np.frombuffer(bytes(r["t"]), dtype=np.float64).reshape(3, 1),
        )
        for r in extr_rows
    }

    cam_rows = conn.execute(
        """
        SELECT cv.id AS shot_video_id, cv.camera_instance_id, cv.file_path,
               cv.first_video_frame, cv.last_video_frame, cv.actual_fps,
               cv.intrinsics_calibration_id AS cv_calib_id,
               cm.default_intrinsics_calibration_id AS mode_default_calib_id
        FROM capture_videos cv
        LEFT JOIN camera_modes cm ON cm.id = cv.camera_mode_id
        WHERE cv.shot_id = ?
        """,
        (shot_id,),
    ).fetchall()

    states: dict[str, CamCalibState] = {}
    for r in cam_rows:
        cam_id = r["camera_instance_id"]
        if cam_id not in extrinsics:
            print(f"  SKIP {cam_id[:8]}: no solved extrinsics entry", file=sys.stderr)
            continue
        intr_calib_id = r["cv_calib_id"] or r["mode_default_calib_id"]
        if intr_calib_id is None:
            print(f"  SKIP {cam_id[:8]}: no intrinsics calibration", file=sys.stderr)
            continue
        ic = conn.execute(
            "SELECT * FROM intrinsics_calibrations WHERE id = ?", (intr_calib_id,)
        ).fetchone()
        if ic is None:
            continue
        R, t = extrinsics[cam_id]
        states[cam_id] = CamCalibState(
            video_id=cam_id, label=cam_id, R=R, t=t,
            file_path=r["file_path"], first_frame=r["first_video_frame"],
            last_frame=r["last_video_frame"], **_resolve_intrinsics(ic),
        )
    return states


def load_sync_table(conn: sqlite3.Connection, shot_id: str) -> tuple[SyncTable, dict]:
    row = conn.execute(
        "SELECT id FROM sync_configs WHERE shot_id = ? ORDER BY rowid DESC LIMIT 1", (shot_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"capture {shot_id!r} has no sync_configs row")
    sync_id = row["id"]

    sp_rows = conn.execute(
        "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, cv.actual_fps, "
        "       cv.camera_instance_id "
        "FROM sync_points sp JOIN capture_videos cv ON cv.id = sp.shot_video_id "
        "WHERE sp.sync_config_id = ?",
        (sync_id,),
    ).fetchall()
    sync_points = []
    fps_by_video = {}
    svid_by_cam = {}
    for r in sp_rows:
        svid = r["shot_video_id"]
        sync_points.append(SyncPoint(
            camera_instance_id=svid, shot_video_id=svid,
            video_frame=int(r["video_frame"]), timestamp_s=float(r["timestamp_s"]),
        ))
        fps_by_video[svid] = float(r["actual_fps"])
        svid_by_cam[r["camera_instance_id"]] = svid
    return SyncTable(sync_points, fps_by_video), svid_by_cam


def robust_mean(samples: np.ndarray, trim_frac: float = 0.1) -> np.ndarray:
    """Per-axis trimmed mean across (N, 3) samples -- outlier frames from a
    bad triangulation/pose solve shouldn't bias the seed (marker-mocap
    algorithms.md §5.1's same convention)."""
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
    """Linear (DLT) triangulation of one unlabeled 3D point from >=2
    single-camera undistorted pixel observations -- the standard
    homogeneous linear-least-squares formulation (stack two equations per
    view, solve via SVD), not app.setup.extrinsics_solver's own
    triangulate_pair()/triangulate_all_pairs(): those batch-triangulate
    many *already-matched* point pairs at once (PairMatch), which doesn't
    fit calibration's one-point-per-bucket, N-view (not just 2) shape any
    more simply than writing this directly.

    Every view's own reprojection error is checked against
    *max_reprojection_px* before accepting the result (same filter
    triangulate_pair() applies, same default order of magnitude) --
    *required*, not a nice-to-have: "each camera saw exactly one
    candidate" (the caller's own ambiguity filter) does not imply those
    candidates are the *same physical point* two different stray bright
    spots can each be "the only candidate" in their own camera while being
    completely unrelated. Confirmed necessary against real footage
    (Weapon test 2026-08-20): without this check, one two-camera
    coincidence triangulated to a point over 3 meters from the reference
    marker -- nowhere near the sword.

    Returns None if fewer than 2 usable views, the linear system is
    degenerate (near-parallel rays, camera behind the point, etc.), or the
    reprojection check fails in any contributing view -- the caller drops
    the bucket rather than accepting a garbage point.
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
    A = np.stack(rows)
    _, _, vt = np.linalg.svd(A)
    x = vt[-1]
    if abs(x[3]) < 1e-8:
        return None
    point = x[:3] / x[3]

    point_h = np.append(point, 1.0)
    for cam_id, P in proj_matrices.items():
        proj = P @ point_h
        if abs(proj[2]) < 1e-8:
            return None
        reproj_px = proj[:2] / proj[2]
        u, v = observations[cam_id]
        if np.linalg.norm(reproj_px - np.array([u, v])) > max_reprojection_px:
            return None
    return point


def cluster_dot_samples(samples: list[np.ndarray], tolerance_m: float = 0.02) -> list[list[np.ndarray]]:
    """Greedy incremental clustering of local-frame 3D dot samples into
    distinct physical dots.

    Unlike an ArUco corner (identity comes from the marker's own decoded
    ID), a triangulated dot sample carries no identity at all -- samples
    across the whole capture need grouping into "these N samples are all
    the same physical dot" before they can be averaged. Each sample joins
    the nearest existing cluster's running centroid if within
    *tolerance_m*, else starts a new cluster.

    Order-dependent by construction (greedy, not a global optimum) -- fine
    here because real dot separations are centimeters to tens of
    centimeters apart on a rigid prop, while per-sample triangulation
    noise from real footage is sub-centimeter (marker-mocap-algorithms.md
    §5.1's own robust-average precedent assumes the same kind of
    separation), so this is not a close call needing a globally-optimal
    clustering method.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="Path to the session .db file")
    ap.add_argument("--shot-id", required=True, help="captures.id for the capture to calibrate from")
    ap.add_argument("--time-start", type=float, required=True)
    ap.add_argument("--time-end", type=float, required=True)
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    ap.add_argument("--marker-size", type=float, required=True, help="Marker side length, meters")
    ap.add_argument("--marker-ids", nargs="+", required=True, help="All ArUco IDs on this body")
    ap.add_argument("--reference-id", required=True, help="Which ID anchors the body's local frame")
    ap.add_argument("--stride", type=int, default=6, help="Process every Nth decoded frame")
    ap.add_argument("--min-cameras", type=int, default=2)
    ap.add_argument("--output", required=True, help="Output marker_body_definitions YAML path")
    ap.add_argument("--name", default="calibrated-rigid-body")
    ap.add_argument("--detect-dots", action="store_true",
                    help="Also calibrate reflective-dot centroids (Phase C1) alongside the "
                         "ArUco markers, from the same footage pass")
    ap.add_argument("--dot-threshold", type=int, default=235,
                    help="detect_blobs() threshold override -- see dot_blob_detector.py's own "
                         "module docstring for why the default isn't a starting point to retune")
    ap.add_argument("--dot-min-area", type=float, default=4.0)
    ap.add_argument("--dot-max-area", type=float, default=400.0)
    ap.add_argument("--dot-min-compactness", type=float, default=0.5)
    ap.add_argument("--dot-cluster-tolerance-m", type=float, default=0.02,
                    help="Max distance (meters) for a triangulated dot sample to join an "
                         "existing cluster rather than start a new physical dot")
    ap.add_argument("--dot-max-reprojection-px", type=float, default=5.0,
                    help="Reject a triangulated dot sample if any contributing view's own "
                         "reprojection error exceeds this -- rejects cross-camera "
                         "coincidences (see triangulate_point_multi_view()'s doc comment)")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    print("Loading camera states + extrinsics...")
    states = load_camera_states(conn, args.shot_id)
    print(f"  {len(states)} cameras with solved extrinsics")
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)

    detector = ArucoDetector(dictionary=args.dictionary)
    marker_ids = set(args.marker_ids)
    ref_id = args.reference_id
    if ref_id not in marker_ids:
        raise ValueError("--reference-id must be one of --marker-ids")

    # Per-frame accumulation: bucket by rounded global time so different
    # cameras' independently-sampled frames land in the same "instant".
    # bucket -> marker_id -> camera_instance_id -> (4,2) undistorted corners
    frame_obs: dict[float, dict[str, dict[str, np.ndarray]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    # bucket -> camera_instance_id -> list of undistorted (u, v) dot candidates
    # this camera saw that instant -- kept as a list (not just the count) so
    # the aggregation pass below can still print/inspect ambiguous buckets;
    # only len()==1 buckets ever become a calibration sample (see module
    # docstring's "restrict to unambiguous frames" rationale).
    dot_obs: dict[float, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    for cam_id, state in states.items():
        svid = svid_by_cam.get(cam_id)
        if svid is None:
            continue
        first = sync_table.lookup(args.time_start, svid)
        last = sync_table.lookup(args.time_end, svid)
        if first is None or last is None:
            print(f"  SKIP {cam_id[:8]}: no sync coverage in [{args.time_start}, {args.time_end})")
            continue
        print(f"camera {cam_id[:8]}: frames {first}-{last} ({state.file_path})")
        n_decoded = 0
        for video_frame, img in iter_frames(state.file_path, first, last):
            n_decoded += 1
            if (video_frame - first) % args.stride != 0:
                continue
            dets = detector.detect(img, video_id=cam_id, frame_idx=video_frame)
            gt = sync_table.frame_to_global_time(video_frame, svid)
            if gt is None:
                continue
            bucket = round(gt / 0.05) * 0.05
            for d in dets:
                if d.marker_id not in marker_ids:
                    continue
                pts = np.array([(c.px, c.py) for c in d.corners], dtype=np.float64)
                pts_undist = _undistort_pts(pts, state)
                frame_obs[bucket][d.marker_id][cam_id] = pts_undist

            if args.detect_dots:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blobs = detect_blobs(
                    gray, threshold=args.dot_threshold, min_area=args.dot_min_area,
                    max_area=args.dot_max_area, min_compactness=args.dot_min_compactness,
                )
                if blobs:
                    pts = np.array([(b.cx, b.cy) for b in blobs], dtype=np.float64)
                    pts_undist = _undistort_pts(pts, state)
                    dot_obs[bucket][cam_id].extend(pts_undist)
        print(f"  {n_decoded} frames decoded")

    print(f"\n{len(frame_obs)} time buckets with >=1 marker detection")

    # Per-marker samples: local corners (4,3) expressed in the reference
    # marker's own frame, one sample per bucket where both the reference
    # and this marker had a solvable (>=2 camera) pose.
    samples_by_marker: dict[str, list[np.ndarray]] = defaultdict(list)
    # Unlabeled local-frame 3D dot samples, not yet grouped by physical dot
    # (cluster_dot_samples() does that after this loop) -- one sample per
    # bucket where the reference solved AND >=2 cameras each saw exactly
    # one dot candidate that instant.
    dot_samples: list[np.ndarray] = []
    ref_template = marker_local_corners(args.marker_size)
    n_ref_solved = 0
    n_dot_buckets_ambiguous = 0

    for bucket, by_marker in sorted(frame_obs.items()):
        ref_obs = by_marker.get(ref_id)
        if ref_obs is None or len(ref_obs) < args.min_cameras:
            continue
        try:
            rvec_ref, tvec_ref, rms_ref = solve_marker_pose(ref_obs, states, args.marker_size)
        except (ValueError, RuntimeError):
            continue
        n_ref_solved += 1
        R_ref, _ = cv2.Rodrigues(rvec_ref)

        for mid, obs in by_marker.items():
            if mid == ref_id or len(obs) < args.min_cameras:
                continue
            try:
                rvec_m, tvec_m, rms_m = solve_marker_pose(obs, states, args.marker_size)
            except (ValueError, RuntimeError):
                continue
            R_m, _ = cv2.Rodrigues(rvec_m)
            world_corners = (R_m @ marker_local_corners(args.marker_size).T).T + tvec_m
            local_corners = (R_ref.T @ (world_corners - tvec_ref).T).T
            samples_by_marker[mid].append(local_corners)

        if args.detect_dots:
            cam_dots = dot_obs.get(bucket, {})
            unambiguous = {
                cam_id: tuple(pts[0]) for cam_id, pts in cam_dots.items() if len(pts) == 1
            }
            if len(cam_dots) > 0 and any(len(pts) > 1 for pts in cam_dots.values()):
                n_dot_buckets_ambiguous += 1
            if len(unambiguous) >= args.min_cameras:
                world_pt = triangulate_point_multi_view(
                    unambiguous, states, max_reprojection_px=args.dot_max_reprojection_px
                )
                if world_pt is not None:
                    dot_samples.append(R_ref.T @ (world_pt - tvec_ref.flatten()))

    print(f"Reference marker '{ref_id}' solved in {n_ref_solved} buckets")
    for mid, samples in samples_by_marker.items():
        print(f"  marker '{mid}': {len(samples)} co-occurrence samples with reference")

    dot_clusters: list[list[np.ndarray]] = []
    if args.detect_dots:
        print(f"\nDot candidates: {len(dot_samples)} unambiguous triangulated samples "
              f"({n_dot_buckets_ambiguous} buckets dropped for having >1 candidate in some "
              "camera -- see module docstring)")
        dot_clusters = cluster_dot_samples(dot_samples, args.dot_cluster_tolerance_m)
        print(f"  clustered into {len(dot_clusters)} physical dots")
        for i, cluster in enumerate(dot_clusters):
            spread = np.stack(cluster).std(axis=0)
            print(f"  dot{i}: {len(cluster)} samples, per-axis std (m) {spread}")

    # ---- Aggregate + write output YAML ----
    lines = [f"name: {args.name}", "units: meters", "markers:"]

    def _fmt(v: np.ndarray) -> str:
        return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"

    # Reference marker: its own local corners are just the known template
    # (it defines the body's own frame by construction).
    lines.append(f"  - name: aruco_{ref_id}")
    lines.append("    type: aruco")
    lines.append(f"    dictionary: {args.dictionary}")
    lines.append(f'    id: "{ref_id}"')
    lines.append(f"    size: {args.marker_size}")
    lines.append("    corners:")
    for c in ref_template:
        lines.append(f"      - {_fmt(c)}")

    for mid, samples in samples_by_marker.items():
        stacked = np.stack(samples)  # (N, 4, 3)
        avg_corners = np.stack([robust_mean(stacked[:, i, :]) for i in range(4)])
        lines.append(f"  - name: aruco_{mid}")
        lines.append("    type: aruco")
        lines.append(f"    dictionary: {args.dictionary}")
        lines.append(f'    id: "{mid}"')
        lines.append(f"    size: {args.marker_size}")
        lines.append("    corners:")
        for c in avg_corners:
            lines.append(f"      - {_fmt(c)}")
        # Report per-corner spread as a rough noise indicator.
        spread = np.stack([stacked[:, i, :].std(axis=0) for i in range(4)])
        print(f"  marker '{mid}' corner std across samples (m): "
              f"{spread.mean(axis=0)}")

    for i, cluster in enumerate(dot_clusters):
        center = robust_mean(np.stack(cluster))
        lines.append(f"  - name: dot{i}")
        lines.append("    type: reflective_dot")
        lines.append(f"    center: {_fmt(center)}")

    out_path = Path(args.output)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

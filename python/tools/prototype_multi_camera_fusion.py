# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_multi_camera_fusion.py — PROTOTYPE: phase P7 of
person-marker-assignment-design.md's phased plan (promoted ahead of P2
per discussion 2026-09-09 -- triangulation-based fusion is better-grounded
than a 2D-only normal heuristic and directly attacks P1's quantified
per-camera coverage gap).

This is exactly the "genuine epipolar-consistency correspondence
matching" that calibrate_rigid_marker_body.py's own docstring names and
deliberately defers ("Question B, layer 2" -- that script gets away
without it by restricting to frames where every camera sees exactly one
candidate). With per-frame candidate counts now in the single digits per
camera (after the background_mode='blacklist' fix), exhaustive pairwise
triangulation is cheap enough to just try every candidate combination
directly rather than pre-filtering by epipolar distance first.

Algorithm, per (synchronized) frame:
    1. Undistort every camera's raw candidate pixels.
    2. **Seed 3D point hypotheses**: for every pair of cameras with
       candidates, triangulate every (candidate_a, candidate_b)
       combination (cv2.triangulatePoints), keep pairs with positive
       depth in both cameras and low reprojection error back into BOTH --
       a real 3D point that two different candidates both happen to
       explain well; a coincidental false pairing generally reprojects
       badly in at least one view. Greedily merge accepted 2-view points
       landing within --merge-radius-m of each other in 3D into one seed
       hypothesis (position averaged across whichever pairs merged into
       it) -- this stage exists only to produce candidate 3D positions,
       not a final view/membership list (see step 3).
    3. **Reprojection consensus** (2026-09-10, replacing an earlier
       version that only ever recorded the specific camera pair that
       happened to seed a hypothesis): reproject every seed hypothesis's
       3D position into *every* camera with candidates this frame --not
       only the pair that produced it-- and find each camera's closest
       candidate within --max-reproj-px. Resolve all (hypothesis, camera,
       candidate) matches globally by greedy best-first claiming, sorted
       by reprojection error ascending, so one raw candidate ends up
       supporting at most one hypothesis (whichever it fits best) instead
       of silently supporting two different 3D interpretations of the
       same pixel -- a real bug found 2026-09-09 (two seed hypotheses
       both used the same raw pixel in one camera, paired with two
       different candidates in two other cameras, and each got assigned
       a different marker name downstream). This also recovers support
       from cameras that had a matching candidate but weren't part of
       the winning triangulating pair (Harri, 2026-09-10: "fusion should
       group all detections that reproject close, it does not make sense
       to just apply to the winning pair"). A hypothesis is kept only if
       it ends up with >=2 claimed views after this; its position is then
       refined by a proper N-view triangulation (triangulate_multiview)
       over its final claimed views, rather than kept at the pairwise-
       average estimate from step 2.

    Note this resolves *more* ambiguity at the fusion stage than the
    original P7 design deliberately left unresolved (2026-09-09: "P7
    explicitly does NOT force one-to-one dedup ... competing/ambiguous
    2-view hypotheses are legitimate evidence for the assignment layer to
    weigh"). The greedy claim can make a real second physical point lose
    its own best-fitting camera views to a competing hypothesis that
    happened to fit slightly better globally. Accepted per Harri's
    explicit request; genuinely close, still-separately-visible points
    are the remaining case this doesn't handle perfectly.

Usage:
    python tools/prototype_multi_camera_fusion.py \\
        --session /path/to/session.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --marker-detection-run 01c2e3c1-204b-49f8-9b14-6fd611b765a4 \\
        --time 40.415
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pose.db_cache import decode_dot_candidates  # noqa: E402
from app.setup.extrinsics_solver import CamCalibState, _proj_matrix, _undistort_pts  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402


@dataclass
class FusedPoint:
    xyz: np.ndarray
    views: list[tuple[str, int]] = field(default_factory=list)  # (camera_instance_id, candidate_idx)
    max_reproj_err_px: float = 0.0


def triangulate_candidate_pairs(
    cam_a: str, cam_b: str, pts_a: np.ndarray, pts_b: np.ndarray,
    state_a: CamCalibState, state_b: CamCalibState, max_reproj_px: float,
) -> list[tuple[np.ndarray, int, int, float]]:
    """Every (i, j) candidate combination between two cameras, triangulated
    and reprojection-filtered. Returns (xyz, idx_a, idx_b, max_reproj_err)
    for accepted pairs only."""
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return []
    na, nb = pts_a.shape[0], pts_b.shape[0]
    # Cartesian product, flattened, so cv2.triangulatePoints can batch all
    # combinations for this camera pair in one call.
    ia, ib = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
    ia, ib = ia.ravel(), ib.ravel()
    flat_a, flat_b = pts_a[ia], pts_b[ib]

    P_a, P_b = _proj_matrix(state_a), _proj_matrix(state_b)
    pts4d = cv2.triangulatePoints(P_a, P_b, flat_a.T, flat_b.T)
    w = pts4d[3]
    valid_w = np.abs(w) > 1e-8
    pts3d = np.where(valid_w, pts4d[:3] / np.where(valid_w, w, 1), np.nan).T  # N x 3

    depth_a = (state_a.R @ pts3d.T + state_a.t.reshape(3, 1))[2]
    depth_b = (state_b.R @ pts3d.T + state_b.t.reshape(3, 1))[2]
    keep = valid_w & (depth_a > 0) & (depth_b > 0)

    def _reproj_err(P, pts3d_kept, obs):
        homog = P @ np.hstack([pts3d_kept, np.ones((len(pts3d_kept), 1))]).T
        proj = (homog[:2] / homog[2]).T
        return np.linalg.norm(proj - obs, axis=1)

    out = []
    idxs = np.where(keep)[0]
    if idxs.size == 0:
        return out
    err_a = _reproj_err(P_a, pts3d[idxs], flat_a[idxs])
    err_b = _reproj_err(P_b, pts3d[idxs], flat_b[idxs])
    max_err = np.maximum(err_a, err_b)
    good = max_err <= max_reproj_px
    for k in idxs[good]:
        j = np.where(idxs == k)[0][0]
        out.append((pts3d[k], int(ia[k]), int(ib[k]), float(max_err[j])))
    return out


def fuse_frame(
    candidates_by_camera: dict[str, np.ndarray],  # camera_instance_id -> Nx2 UNDISTORTED pixels
    states: dict[str, CamCalibState],
    max_reproj_px: float = 8.0,
    merge_radius_m: float = 0.03,
) -> list[FusedPoint]:
    cams = [c for c in candidates_by_camera if candidates_by_camera[c].shape[0] > 0]

    # --- Stage 1: seed 3D point hypotheses (pairwise triangulation + 3D-proximity merge) ---
    pair_points: list[tuple[np.ndarray, str, int, str, int, float]] = []
    for i, cam_a in enumerate(cams):
        for cam_b in cams[i + 1:]:
            accepted = triangulate_candidate_pairs(
                cam_a, cam_b, candidates_by_camera[cam_a], candidates_by_camera[cam_b],
                states[cam_a], states[cam_b], max_reproj_px,
            )
            for xyz, ia, ib, err in accepted:
                pair_points.append((xyz, cam_a, ia, cam_b, ib, err))

    seeds: list[np.ndarray] = []
    seed_counts: list[int] = []
    for xyz, _cam_a, _ia, _cam_b, _ib, _err in pair_points:
        target = None
        for si, sxyz in enumerate(seeds):
            if np.linalg.norm(sxyz - xyz) <= merge_radius_m:
                target = si
                break
        if target is None:
            seeds.append(xyz)
            seed_counts.append(1)
        else:
            n = seed_counts[target]
            seeds[target] = (seeds[target] * n + xyz) / (n + 1)
            seed_counts[target] += 1

    if not seeds:
        return []

    # --- Stage 2: reprojection consensus -- see module docstring for why this
    # replaced a version that only ever recorded the seeding pair's own views. ---
    proj_matrices = {c: _proj_matrix(states[c]) for c in cams}
    matches: list[tuple[float, int, str, int]] = []  # (reproj_err, seed_idx, cam_id, cand_idx)
    for si, xyz in enumerate(seeds):
        xyz_h = np.append(xyz, 1.0)
        for cam_id in cams:
            h = proj_matrices[cam_id] @ xyz_h
            if h[2] <= 0:  # behind this camera
                continue
            proj = h[:2] / h[2]
            pts = candidates_by_camera[cam_id]
            d = np.linalg.norm(pts - proj, axis=1)
            j = int(np.argmin(d))
            if d[j] <= max_reproj_px:
                matches.append((float(d[j]), si, cam_id, j))
    matches.sort(key=lambda m: m[0])

    claimed: set[tuple[str, int]] = set()
    seed_views: list[list[tuple[str, int]]] = [[] for _ in seeds]
    seed_max_err: list[float] = [0.0] * len(seeds)
    for err, si, cam_id, j in matches:
        key = (cam_id, j)
        if key in claimed:
            continue
        claimed.add(key)
        seed_views[si].append(key)
        seed_max_err[si] = max(seed_max_err[si], err)

    fused: list[FusedPoint] = []
    for si, views in enumerate(seed_views):
        if len(views) < 2:
            continue  # lost its multi-view confirmation to a competing hypothesis
        refined_P = [proj_matrices[c] for c, _idx in views]
        refined_pt = [tuple(candidates_by_camera[c][idx]) for c, idx in views]
        xyz = triangulate_multiview(refined_P, refined_pt)
        if xyz is None:
            xyz = seeds[si]
        fused.append(FusedPoint(xyz=xyz, views=views, max_reproj_err_px=seed_max_err[si]))

    return fused


def triangulate_multiview(proj_matrices: list[np.ndarray], pts2d: list[tuple[float, float]]) -> np.ndarray | None:
    """Generic N-view (N>=2) linear triangulation (DLT via SVD) -- unlike
    fuse_frame()'s pairwise dot-candidate matching, a pose keypoint's
    identity is already known in every camera (index 13 is 'left_knee'
    everywhere), so there's no correspondence search here at all, just
    triangulating the same named point from however many cameras see it
    this frame. Returns None if fewer than 2 views are given (no
    triangulation possible from a single ray)."""
    if len(proj_matrices) < 2:
        return None
    rows = []
    for P, (u, v) in zip(proj_matrices, pts2d):
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.array(rows)
    _, _, vt = np.linalg.svd(A)
    x = vt[-1]
    if abs(x[3]) < 1e-8:
        return None
    return x[:3] / x[3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--marker-detection-run", required=True)
    ap.add_argument("--time", type=float, required=True)
    ap.add_argument("--max-reproj-px", type=float, default=8.0)
    ap.add_argument("--merge-radius-m", type=float, default=0.03)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    states = load_camera_states(conn, args.shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)
    cam_labels = {r["id"]: r["label"] for r in conn.execute("SELECT id, label FROM camera_instances")}

    candidates_by_camera: dict[str, np.ndarray] = {}
    frame_by_camera: dict[str, int] = {}
    for cam_id, svid in svid_by_cam.items():
        if cam_id not in states:
            continue
        frame_idx = sync_table.lookup(args.time, svid)
        if frame_idx is None:
            continue
        row = conn.execute(
            "SELECT keypoints FROM detection_keypoints WHERE detection_run_id = ? AND shot_video_id = ? "
            "AND video_frame = ? AND region_type = 'dots'",
            (args.marker_detection_run, svid, frame_idx),
        ).fetchone()
        if row is None:
            continue
        dots = decode_dot_candidates(bytes(row["keypoints"]))
        pts_undist = _undistort_pts(dots[:, :2], states[cam_id])
        candidates_by_camera[cam_id] = pts_undist
        frame_by_camera[cam_id] = frame_idx
        print(f"{cam_labels.get(cam_id, cam_id)}: frame={frame_idx}  {dots.shape[0]} raw candidates")

    fused = fuse_frame(candidates_by_camera, states, args.max_reproj_px, args.merge_radius_m)
    print(f"\n{len(fused)} fused 3D points (>= 2-view agreement):")
    for f in sorted(fused, key=lambda f: -len(f.views)):
        view_str = ", ".join(f"{cam_labels.get(c, c)}#{i}" for c, i in f.views)
        print(f"  xyz={f.xyz.round(3).tolist()}  n_views={len(f.views)}  "
              f"max_reproj_err={f.max_reproj_err_px:.2f}px  views=[{view_str}]")


if __name__ == "__main__":
    main()

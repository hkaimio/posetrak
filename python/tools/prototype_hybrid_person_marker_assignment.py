# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_hybrid_person_marker_assignment.py — PROTOTYPE: hybrid of
prototype_fused_person_marker_assignment.py's 3D-primary assignment and
prototype_person_marker_assignment.py's per-camera 2D assignment.

Why a hybrid, not a replacement (2026-09-09 finding): requiring *both* a
catalog marker's anchor keypoint *and* a candidate to independently reach
2+-camera agreement is a much stricter joint condition than either alone
-- confirmed on real data, the pure 3D-primary path genuinely improved
multi-marker-joint coverage (the case it was built for) but collapsed
single-marker-slot coverage (hip dropped from 53% of frames matched to
under 2%), because most real hip sightings come from a keypoint or a
candidate seen by only one camera at a time.

Per frame, per camera:
    1. **3D pass** (once per frame, not per camera): fuse dot candidates
       across all cameras (>=2-view triangulation consistency), triangulate
       each catalog marker's anchor keypoint from every camera that sees it
       (>=2 needed), then one-to-one match markers-with-a-3D-anchor against
       fused candidates by 3D distance. Every accepted match's contributing
       (camera, candidate index) pairs are marked "consumed" in each of
       those cameras specifically.
    2. **2D fallback pass** (per camera, over whatever's left): catalog
       markers not already 3D-assigned, whose anchor keypoint IS visible in
       this camera, one-to-one matched (2D pixel distance) against this
       camera's own not-yet-consumed candidates -- exactly
       prototype_person_marker_assignment.py's mechanism, restricted to the
       leftovers.

Output per camera is {name: (px, py, confidence)} in RAW (distorted) pixel
space (matching the source video directly, for drawing), confidence in
{'3d', '2d'} so a consumer (e.g. the validation video) can tell a
cross-camera-confirmed assignment from a single-view one at a glance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.setup.extrinsics_solver import CamCalibState, _proj_matrix, _undistort_pts  # noqa: E402
from tools.prototype_multi_camera_fusion import fuse_frame, triangulate_multiview  # noqa: E402
from tools.prototype_person_marker_assignment import _CATALOG  # noqa: E402


def hybrid_assign_frame(
    dots_raw_by_cam: dict[str, np.ndarray],   # camera_instance_id -> Nx2 RAW (distorted) pixels
    kp_by_cam: dict[str, np.ndarray],         # camera_instance_id -> (133, 3) RAW pixel + confidence
    states: dict[str, CamCalibState],
    fusion_max_reproj_px: float = 3.0,
    fusion_merge_radius_m: float = 0.03,
    match_radius_m: float = 0.15,
    match_radius_px: float = 80.0,
) -> dict[str, dict[str, tuple[float, float, str]]]:
    """Returns {camera_instance_id: {marker_name: (raw_px_x, raw_px_y, '3d'|'2d')}}."""
    dots_undist_by_cam = {c: _undistort_pts(d, states[c]) for c, d in dots_raw_by_cam.items() if d.shape[0] > 0}
    fused = fuse_frame(dots_undist_by_cam, states, fusion_max_reproj_px, fusion_merge_radius_m)

    marker_anchors: dict[str, np.ndarray] = {}
    for name, (idx, _group) in _CATALOG.items():
        views_P, views_pt = [], []
        for cam_id, kp in kp_by_cam.items():
            if cam_id not in states or kp[idx, 2] <= 0:
                continue
            pt = _undistort_pts(kp[idx:idx + 1, :2], states[cam_id])[0]
            views_P.append(_proj_matrix(states[cam_id]))
            views_pt.append((pt[0], pt[1]))
        if len(views_P) >= 2:
            xyz = triangulate_multiview(views_P, views_pt)
            if xyz is not None:
                marker_anchors[name] = xyz

    result: dict[str, dict[str, tuple[float, float, str]]] = {c: {} for c in dots_raw_by_cam}
    consumed: dict[str, set[int]] = {c: set() for c in dots_raw_by_cam}
    assigned_names: set[str] = set()

    # --- 3D pass ---
    if marker_anchors and fused:
        names = list(marker_anchors)
        cost = np.full((len(names), len(fused)), 1e6)
        for ni, name in enumerate(names):
            for fi, fp in enumerate(fused):
                d = np.linalg.norm(marker_anchors[name] - fp.xyz)
                if d <= match_radius_m:
                    cost[ni, fi] = d
        row_ind, col_ind = linear_sum_assignment(cost)
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] >= 1e6:
                continue
            name = names[ri]
            fp = fused[ci]
            assigned_names.add(name)
            for cam_id, cand_idx in fp.views:
                consumed[cam_id].add(cand_idx)
                px, py = dots_raw_by_cam[cam_id][cand_idx]
                result[cam_id][name] = (float(px), float(py), "3d")

    # --- 2D fallback pass, per camera, over the leftovers ---
    for cam_id, dots_raw in dots_raw_by_cam.items():
        kp = kp_by_cam.get(cam_id)
        if kp is None:
            continue
        leftover_idx = [i for i in range(dots_raw.shape[0]) if i not in consumed[cam_id]]
        if not leftover_idx:
            continue
        leftover_pts = dots_raw[leftover_idx]
        names = [n for n, (idx, _g) in _CATALOG.items() if n not in assigned_names and kp[idx, 2] > 0]
        if not names:
            continue
        cost = np.full((len(names), len(leftover_pts)), 1e6)
        for ni, name in enumerate(names):
            idx, _g = _CATALOG[name]
            kx, ky = kp[idx, 0], kp[idx, 1]
            d = np.hypot(leftover_pts[:, 0] - kx, leftover_pts[:, 1] - ky)
            cost[ni, :] = np.where(d <= match_radius_px, d, 1e6)
        row_ind, col_ind = linear_sum_assignment(cost)
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] >= 1e6:
                continue
            name = names[ri]
            px, py = leftover_pts[ci]
            result[cam_id][name] = (float(px), float(py), "2d")

    return result

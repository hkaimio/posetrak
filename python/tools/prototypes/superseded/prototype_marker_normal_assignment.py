# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_marker_normal_assignment.py — PROTOTYPE: marker-normal-aware
extension of prototype_hybrid_person_marker_assignment.py, for the
ambiguous same-joint marker groups (knee_{L,R}: medial/lateral/front;
ankle_{L,R}: medial/lateral).

Confirmed structural gap in hybrid_assign_frame() this is meant to close
(2026-09-09, real finding, not a guess): every marker name in an ambiguous
group shares one vitpose anchor index (e.g. knee_L_medial, knee_L_lateral,
and knee_L_front are all catalog-mapped to index 13), so their
triangulated 3D anchor is numerically *identical* for all three names.
The existing 3D-pass Hungarian match therefore has zero information to
tell them apart -- whichever candidate a solver happens to pick for
"knee_L_medial" vs "knee_L_lateral" is arbitrary tie-breaking, not a real
decision. This is almost certainly the mechanism behind the medial/
lateral/front flicker Harri reported reviewing the hybrid video.

Approach: rather than requiring per-marker 3D geometric offsets baked
into the catalog ahead of time (the heavier option person-marker-
assignment-design.md #2 flagged as needing its own design pass), derive
each ambiguous joint's medial/lateral/anterior directions *anatomically*,
fresh every frame, from the skeleton's own triangulated pose:

    medial_dir(side) -- unit vector from this leg's hip toward the other
                         leg's hip (the pelvis-width vector), sign-flipped
                         for the right side. Anatomically, "medial" always
                         means "toward the body's own midline", which this
                         vector approximates directly without needing any
                         per-marker catalog data.
    anterior_dir     -- cross(world_up, pelvis_vector), one shared
                         direction for the whole pelvis (both legs, both
                         knee and ankle), *not* derived from either limb's
                         own instantaneous thigh/shank direction -- an
                         earlier version that did was found wrong by
                         rendering it as arrows on a real frame (see
                         render_marker_normal_debug_frame.py): a raised,
                         bent knee's own segment direction swings with
                         hip/knee flexion, but the knee's true anterior-
                         facing direction does not (only hip axial
                         rotation would change it). Anchoring to the
                         pelvis instead matches anatomy and was confirmed
                         against the one case that already looked
                         right (a straight standing leg).

Two places this lets ambiguous-group members be told apart, instead of
matched to an undifferentiated shared anchor:

    1. **3D pass**: a fused candidate's own 3D offset from the joint
       center (fp.xyz - joint_pos) is a real, independent direction that
       can be compared against each name's expected direction -- this
       works with no per-camera projection needed, since fused points
       already carry real 3D positions.
    2. **2D fallback pass**: for a single-camera-only candidate (no fused
       3D position), the anatomical frame's direction vectors are
       projected into that camera's own undistorted image plane, and the
       candidate's own 2D pixel offset from the joint's own projection is
       compared against those projected directions.

Falls back to hybrid_assign_frame()'s original shared-anchor behavior for
any ambiguous group/frame where the anatomical frame itself can't be
computed (insufficient triangulation), so coverage doesn't regress when
normals aren't available -- this is additive disambiguation, not a
replacement for the whole hybrid pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.setup.extrinsics_solver import CamCalibState, _proj_matrix, _undistort_pts  # noqa: E402
from tools.prototypes.superseded.prototype_hybrid_person_marker_assignment import hybrid_assign_frame  # noqa: E402
from tools.prototype_multi_camera_fusion import fuse_frame, triangulate_multiview  # noqa: E402
from tools.prototype_person_marker_assignment import _CATALOG, _GROUP_SIZE, _UNAMBIGUOUS_GROUPS  # noqa: E402

# vitpose-133kp indices for the joints an anatomical frame needs -- shared
# with _CATALOG's own hip/knee/ankle entries (11/12/13/14/15/16), named
# here without the per-marker suffixes since a frame is computed once per
# *joint*, not once per named marker.
_JOINT_IDX = {"hip_L": 11, "hip_R": 12, "knee_L": 13, "knee_R": 14, "ankle_L": 15, "ankle_R": 16}

# Ambiguous groups this module knows how to disambiguate, and which named
# markers belong to each -- both catalog joint-groups with >1 marker.
_AMBIGUOUS_GROUPS = {g for g in _GROUP_SIZE if g not in _UNAMBIGUOUS_GROUPS}


def _camera_center(state: CamCalibState) -> np.ndarray:
    """World-space camera center: solves 0 = R @ C + t for C."""
    return -state.R.T @ state.t.reshape(3)


def _triangulate_joint(idx: int, kp_by_cam: dict[str, np.ndarray], states: dict[str, CamCalibState]) -> np.ndarray | None:
    views_P, views_pt = [], []
    for cam_id, kp in kp_by_cam.items():
        if cam_id not in states or kp[idx, 2] <= 0:
            continue
        pt = _undistort_pts(kp[idx:idx + 1, :2], states[cam_id])[0]
        views_P.append(_proj_matrix(states[cam_id]))
        views_pt.append((pt[0], pt[1]))
    if len(views_P) < 2:
        return None
    return triangulate_multiview(views_P, views_pt)


_WORLD_UP = np.array([0.0, 0.0, 1.0])  # confirmed empirically: triangulated ankle-on-ground ~= Z 0


def compute_leg_frames(
    kp_by_cam: dict[str, np.ndarray], states: dict[str, CamCalibState],
) -> dict[str, dict]:
    """Returns {"knee_L": {...}, "ankle_L": {...}, "knee_R": {...}, "ankle_R": {...}}
    for whichever ambiguous joints have enough triangulated anchors this
    frame -- missing ones are simply absent from the returned dict, not an
    error (caller falls back to the undifferentiated shared-anchor
    behavior for those).

    Each entry: {"joint_pos": xyz, "medial": unit_vec, "anterior": unit_vec}
    -- "lateral" is just -medial, not stored separately.

    Both directions are derived from the *pelvis* alone (hip_L, hip_R, and
    a fixed world-up reference), not from each limb's own instantaneous
    thigh/shank direction -- a real bug caught 2026-09-09 by rendering the
    computed frame as arrows on a real image (see
    render_marker_normal_debug_frame.py) and checking anterior against
    which way the subject was visibly facing: deriving anterior from the
    limb's own segment direction made it swing with hip/knee *flexion*
    (raising a leg), which doesn't anatomically rotate the knee's true
    anterior-facing direction at all -- only hip axial rotation would.
    Anchoring anterior to the pelvis instead reproduced the one case that
    already looked right (the straight standing leg) exactly, and fixed
    the bent raised leg, which had been pointing backwards. This is a
    real, accepted simplification: true anterior would also track hip
    internal/external rotation, which this doesn't model -- revisit if
    that turns out to matter in practice, per this project's "note the
    limitation rather than solve for the untested case" habit.
    """
    joints = {name: _triangulate_joint(idx, kp_by_cam, states) for name, idx in _JOINT_IDX.items()}
    hip_l, hip_r = joints["hip_L"], joints["hip_R"]
    if hip_l is None or hip_r is None:
        return {}  # medial/lateral/anterior are all undefined without both hips

    pelvis_vec = hip_r - hip_l
    pelvis_norm = np.linalg.norm(pelvis_vec)
    if pelvis_norm < 1e-6:
        return {}
    pelvis_unit = pelvis_vec / pelvis_norm

    anterior = np.cross(_WORLD_UP, pelvis_unit)
    a_norm = np.linalg.norm(anterior)
    if a_norm < 1e-6:  # pelvis vector ~parallel to world-up -- no stable anterior this frame
        return {}
    anterior_unit = anterior / a_norm

    frames: dict[str, dict] = {}
    for side, medial_sign in (("L", 1.0), ("R", -1.0)):
        medial_dir = medial_sign * pelvis_unit
        for joint_name in ("knee", "ankle"):
            joint_pos = joints[f"{joint_name}_{side}"]
            if joint_pos is not None:
                frames[f"{joint_name}_{side}"] = {
                    "joint_pos": joint_pos, "medial": medial_dir, "anterior": anterior_unit,
                }

    return frames


def _marker_local_direction(name: str, group: str, frame: dict) -> np.ndarray | None:
    """Expected outward direction for one ambiguous marker name, in world
    space, using its joint's own anatomical frame. Returns None for a
    direction this module doesn't model (there isn't one currently --
    every ambiguous catalog name is medial/lateral/front)."""
    if name.endswith("_medial"):
        return frame["medial"]
    if name.endswith("_lateral"):
        return -frame["medial"]
    if name.endswith("_front"):
        return frame["anterior"]
    return None


def normal_aware_hybrid_assign_frame(
    dots_raw_by_cam: dict[str, np.ndarray],
    kp_by_cam: dict[str, np.ndarray],
    states: dict[str, CamCalibState],
    fusion_max_reproj_px: float = 3.0,
    fusion_merge_radius_m: float = 0.03,
    match_radius_m: float = 0.15,
    match_radius_px: float = 80.0,
    min_direction_score: float = 0.0,
    disambiguate_2d_fallback: bool = False,
) -> dict[str, dict[str, tuple[float, float, str]]]:
    """Same signature/return shape as hybrid_assign_frame(), but ambiguous
    same-joint groups (knee/ankle medial/lateral/front) are matched by
    directional agreement with an anatomically-derived expected normal,
    instead of by distance to an undifferentiated shared anchor.

    *min_direction_score* gates a candidate/name pairing out entirely
    (cost treated as infeasible) below this cosine similarity -- 0.0
    means "don't reject on direction alone, just prefer better-aligned
    pairings", matching this prototype's "drop, don't guess" precedent:
    raise it to actually refuse a poorly-aligned pairing rather than
    accept the least-bad option in a group with no good candidates.

    *disambiguate_2d_fallback* controls whether the *2D-only* fallback
    pass attempts this same directional disambiguation for ambiguous
    groups. Real, diagnosed limitation found 2026-09-09 (Harri: "the knee
    point is assigned incorrectly to medial in the first half of the
    video"): the 2D fallback compares a candidate's *projected* pixel
    offset against each name's *projected* expected direction, which is
    far more sensitive to small joint-position triangulation error and to
    the true marker not sitting exactly on the idealized medial/lateral/
    front axis than the 3D pass is -- confirmed on the real failing case
    (frame 7600): the only nearby candidate's real projected offset was
    ~32 degrees off *every* modeled direction, so "medial" won only for
    being the least-wrong of three bad options (cosine 0.85), not for
    being right (the 3D pass, available a few hundred frames later once
    cross-camera triangulation kicks in, confidently and correctly picks
    "lateral" instead, cosine ~0.9, and every prior joint-purity
    measurement this session used stayed consistent from then on). A
    plain score threshold can't rule this out -- 0.85 looks like a good
    match, it's just anatomically the wrong one. Defaulting this off
    trades reduced ambiguous-group coverage during single-camera-only
    stretches for not confidently mislabeling them; set True only after
    that trade-off has been re-validated against real data.
    """
    dots_undist_by_cam = {c: _undistort_pts(d, states[c]) for c, d in dots_raw_by_cam.items() if d.shape[0] > 0}
    fused = fuse_frame(dots_undist_by_cam, states, fusion_max_reproj_px, fusion_merge_radius_m)
    leg_frames = compute_leg_frames(kp_by_cam, states)

    result: dict[str, dict[str, tuple[float, float, str]]] = {c: {} for c in dots_raw_by_cam}
    consumed: dict[str, set[int]] = {c: set() for c in dots_raw_by_cam}
    assigned_names: set[str] = set()

    # --- unambiguous-group anchors, exactly as hybrid_assign_frame ---
    marker_anchors: dict[str, np.ndarray] = {}
    for name, (idx, group) in _CATALOG.items():
        if group not in _UNAMBIGUOUS_GROUPS:
            continue
        xyz = _triangulate_joint(idx, kp_by_cam, states)
        if xyz is not None:
            marker_anchors[name] = xyz

    # --- 3D pass, unambiguous groups only (plain distance Hungarian) ---
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

    # --- 3D pass, ambiguous groups: directional match against nearby fused points ---
    for group in _AMBIGUOUS_GROUPS:
        frame = leg_frames.get(group)
        names = [n for n, (_idx, g) in _CATALOG.items() if g == group]
        if frame is None or not fused:
            continue
        joint_pos = frame["joint_pos"]
        nearby = [fi for fi, fp in enumerate(fused) if np.linalg.norm(fp.xyz - joint_pos) <= match_radius_m]
        if not nearby:
            continue
        cost = np.full((len(names), len(nearby)), 1e6)
        for ni, name in enumerate(names):
            direction = _marker_local_direction(name, group, frame)
            if direction is None:
                continue
            for ci, fi in enumerate(nearby):
                offset = fused[fi].xyz - joint_pos
                on = np.linalg.norm(offset)
                score = float(np.dot(direction, offset / on)) if on > 1e-6 else 0.0
                if score >= min_direction_score:
                    cost[ni, ci] = 1.0 - score  # lower is better for linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] >= 1e6:
                continue
            name = names[ri]
            fp = fused[nearby[ci]]
            assigned_names.add(name)
            for cam_id, cand_idx in fp.views:
                consumed[cam_id].add(cand_idx)
                px, py = dots_raw_by_cam[cam_id][cand_idx]
                result[cam_id][name] = (float(px), float(py), "3d")

    # --- 2D fallback pass, per camera, over the leftovers ---
    for cam_id, dots_raw in dots_raw_by_cam.items():
        kp = kp_by_cam.get(cam_id)
        if kp is None or cam_id not in states:
            continue
        leftover_idx = [i for i in range(dots_raw.shape[0]) if i not in consumed[cam_id]]
        if not leftover_idx:
            continue
        leftover_pts_raw = dots_raw[leftover_idx]
        leftover_pts_undist = _undistort_pts(leftover_pts_raw, states[cam_id])
        P_cam = _proj_matrix(states[cam_id])

        def _project(xyz: np.ndarray) -> np.ndarray:
            h = P_cam @ np.append(xyz, 1.0)
            return h[:2] / h[2]

        # unambiguous leftovers: unchanged, plain 2D distance Hungarian
        unamb_names = [n for n, (idx, g) in _CATALOG.items()
                        if g in _UNAMBIGUOUS_GROUPS and n not in assigned_names and kp[idx, 2] > 0]
        used_leftover: set[int] = set()
        if unamb_names:
            cost = np.full((len(unamb_names), len(leftover_pts_undist)), 1e6)
            for ni, name in enumerate(unamb_names):
                idx, _g = _CATALOG[name]
                kx, ky = kp[idx, 0], kp[idx, 1]
                kx, ky = _undistort_pts(np.array([[kx, ky]]), states[cam_id])[0]
                d = np.hypot(leftover_pts_undist[:, 0] - kx, leftover_pts_undist[:, 1] - ky)
                cost[ni, :] = np.where(d <= match_radius_px, d, 1e6)
            row_ind, col_ind = linear_sum_assignment(cost)
            for ri, ci in zip(row_ind, col_ind):
                if cost[ri, ci] >= 1e6:
                    continue
                name = unamb_names[ri]
                px, py = leftover_pts_raw[ci]
                result[cam_id][name] = (float(px), float(py), "2d")
                used_leftover.add(ci)

        # ambiguous leftovers: directional match in this camera's own 2D projection.
        # See disambiguate_2d_fallback's docstring -- off by default, this path's
        # projected-direction geometry was found to confidently mislabel a real case.
        for group in _AMBIGUOUS_GROUPS if disambiguate_2d_fallback else []:
            frame = leg_frames.get(group)
            names = [n for n, (idx, g) in _CATALOG.items()
                     if g == group and n not in assigned_names and kp[_CATALOG[n][0], 2] > 0]
            if frame is None or not names:
                continue
            joint_2d = _project(frame["joint_pos"])
            eps = 0.03  # meters -- small enough to stay locally linear through the projection
            dir_2d = {}
            for name in names:
                direction = _marker_local_direction(name, group, frame)
                offset_2d = _project(frame["joint_pos"] + eps * direction) - joint_2d
                n = np.linalg.norm(offset_2d)
                if n > 1e-6:
                    dir_2d[name] = offset_2d / n
            names = [n for n in names if n in dir_2d]
            if not names:
                continue
            candidate_idx = [ci for ci in range(len(leftover_pts_undist)) if ci not in used_leftover]
            candidate_idx = [ci for ci in candidate_idx
                              if np.hypot(*(leftover_pts_undist[ci] - joint_2d)) <= match_radius_px]
            if not candidate_idx:
                continue
            cost = np.full((len(names), len(candidate_idx)), 1e6)
            for ni, name in enumerate(names):
                for ci, cand in enumerate(candidate_idx):
                    offset = leftover_pts_undist[cand] - joint_2d
                    on = np.linalg.norm(offset)
                    score = float(np.dot(dir_2d[name], offset / on)) if on > 1e-6 else 0.0
                    if score >= min_direction_score:
                        cost[ni, ci] = 1.0 - score
            row_ind, col_ind = linear_sum_assignment(cost)
            for ri, ci in zip(row_ind, col_ind):
                if cost[ri, ci] >= 1e6:
                    continue
                name = names[ri]
                cand = candidate_idx[ci]
                px, py = leftover_pts_raw[cand]
                result[cam_id][name] = (float(px), float(py), "2d")
                used_leftover.add(cand)

    return result

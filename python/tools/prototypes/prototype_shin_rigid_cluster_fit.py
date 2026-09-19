# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_shin_rigid_cluster_fit.py -- validates the skeleton-scaling
design doc's core idea (docs/roadmap/features/marker-based-mocap/
skeleton-scaling-and-marker-calibration-design.md §2.1) against real data,
before building anything toward the catalog schema addition (D7) or a
generic module-registry mechanism that idea would eventually need.

PROTOTYPE, hardcoded to this capture's `shin` segment (the one place the
`leg` module has enough markers -- 5 -- for a rigid-cluster fit) and its
5 known marker names, matching this whole session's established
"prototype script first, generalize only once validated" convention. Not
wired into fit_calibrated_attachment_set.py / refit_attachment_set_
single_camera.py, and not meant to be.

Idea being tested: instead of fitting each shin marker's local offset
independently against the *tracked* skeleton's own parent-joint
transform (refit_attachment_set_single_camera.py's approach -- which
inherits whatever error the tracked skeleton has), triangulate each
marker's own 3D position per frame from real multi-camera observations
(2+ cameras, no skeleton involved at all), then iteratively Kabsch-fit
the segment's own per-frame rigid pose against the other markers'
already-triangulated positions, and re-derive each marker's local offset
from that -- entirely independent of the tracked skeleton's own bone
length or joint assumptions. A weak marker (ankle_lat_L, historically)
should get carried by the segment's stronger members instead of needing
its own good data to succeed alone.

Usage:
    python tools/prototype_shin_rigid_cluster_fit.py \\
        --session /path/to/session.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --tracking-run 2e9aa605-3460-4595-9c37-59501c317c72 \\
        --calibrated-attachment-set /path/to/leg.calibrated.refit-v3.yaml
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.mcp.db import (  # noqa: E402
    OBS_ACTUAL_X, OBS_ACTUAL_Y, OBS_OUTLIER, OBS_PAD, OBS_MODE_ABSOLUTE,
    decode_obs_blob, get_run_cameras, get_run_markers,
)
from app.setup.extrinsics_solver import _proj_matrix  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states  # noqa: E402
from tools.fit_calibrated_attachment_set import _TransformCache  # noqa: E402
from tools.prototype_multi_camera_fusion import triangulate_multiview  # noqa: E402

# The one segment this capture's `leg` module actually instruments well
# enough for a rigid-cluster fit -- hardcoded, per this script's own
# prototype scope (see module docstring).
_SHIN_MARKER_BASES = ["knee_lat", "knee_med", "knee_front", "ankle_lat", "ankle_med"]


def _project(P: np.ndarray, world_pt: np.ndarray) -> np.ndarray:
    h = P @ np.append(world_pt, 1.0)
    return h[:2] / h[2]


def _kabsch(local_pts: np.ndarray, world_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standard closed-form rigid registration (Kabsch/Umeyama, rotation
    only -- no scaling): find (R, t) minimizing sum ||R @ local_i + t -
    world_i||^2. Needs >=3 non-collinear points; caller's responsibility."""
    local_centroid = local_pts.mean(axis=0)
    world_centroid = world_pts.mean(axis=0)
    local_c = local_pts - local_centroid
    world_c = world_pts - world_centroid
    H = local_c.T @ world_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = world_centroid - R @ local_centroid
    return R, t


def collect_frame_triangulations(
    conn: sqlite3.Connection, tracking_run_id: str, marker_names_wanted: list[str],
    states: dict, person_id: int = 0,
) -> dict[int, dict[str, np.ndarray]]:
    """{tracker_step: {marker_name: triangulated_xyz}} -- entirely from raw
    multi-camera observations, no skeleton/tracked-pose involved at all."""
    camera_ids, _ = get_run_cameras(conn, tracking_run_id)
    marker_names = get_run_markers(conn, tracking_run_id)
    n_cam, n_mrk = len(camera_ids), len(marker_names)
    wanted_idx = {name: marker_names.index(name) for name in marker_names_wanted
                  if name in marker_names}

    rows = conn.execute(
        "SELECT tracker_step, obs_blob FROM tracking_obs_results "
        "WHERE run_id = ? AND person_id = ? ORDER BY tracker_step",
        (tracking_run_id, person_id),
    ).fetchall()

    out: dict[int, dict[str, np.ndarray]] = {}
    for step, blob in rows:
        obs = decode_obs_blob(blob, n_cam, n_mrk)
        per_marker: dict[str, np.ndarray] = {}
        for name, mi in wanted_idx.items():
            views_P, views_pt = [], []
            for ci, cam_id in enumerate(camera_ids):
                slot = obs[ci, mi]
                if slot[OBS_PAD] != OBS_MODE_ABSOLUTE:
                    continue
                if not np.isfinite(slot[OBS_OUTLIER]) or slot[OBS_OUTLIER] != 0.0:
                    continue
                ax, ay = float(slot[OBS_ACTUAL_X]), float(slot[OBS_ACTUAL_Y])
                if not (np.isfinite(ax) and np.isfinite(ay)) or cam_id not in states:
                    continue
                views_P.append(_proj_matrix(states[cam_id]))
                views_pt.append((ax, ay))
            if len(views_P) >= 2:
                xyz = triangulate_multiview(views_P, views_pt)
                if xyz is not None:
                    per_marker[name] = xyz
        if len(per_marker) >= 3:  # need >=3 points for a determined Kabsch fit
            out[step] = per_marker
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--tracking-run", required=True,
                     help="A dot-augmented tracking run -- only used for its recorded "
                          "observations/camera list, its own tracked pose is never used.")
    ap.add_argument("--calibrated-attachment-set", required=True,
                     help="Supplies the initial local-offset guess and the before/after comparison.")
    ap.add_argument("--side", choices=["L", "R", "both"], default="both")
    ap.add_argument("--iterations", type=int, default=4)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    states = load_camera_states(conn, args.shot_id)
    xform = _TransformCache(conn, args.tracking_run)  # only .bone_length() used -- static geometry

    old_doc = yaml.safe_load(Path(args.calibrated_attachment_set).read_text())
    old_by_name = {m["name"]: m for m in old_doc["markers"]}

    sides = ["L", "R"] if args.side == "both" else [args.side]
    for side in sides:
        marker_names = [f"{base}_{side}" for base in _SHIN_MARKER_BASES]
        parent_joint = f"shin.{side}"
        bone_len = xform.bone_length(parent_joint)

        print(f"\n=== shin.{side} rigid-cluster fit ===")
        frames = collect_frame_triangulations(conn, args.tracking_run, marker_names, states)
        print(f"{len(frames)} frames with >=3 of {len(marker_names)} markers triangulated")
        if not frames:
            print("  no usable frames -- skipping")
            continue

        # Bootstrap local offsets from the existing single-marker refit (already
        # in the parent joint's own local frame -- exactly what this fit produces too).
        local: dict[str, np.ndarray] = {
            name: np.array(old_by_name[name]["offset"], dtype=float) for name in marker_names
        }

        for it in range(args.iterations):
            # Step 1: per-frame rigid pose from whichever markers this frame triangulated.
            frame_pose: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for step, per_marker in frames.items():
                names_present = list(per_marker.keys())
                local_pts = np.array([local[n] for n in names_present])
                world_pts = np.array([per_marker[n] for n in names_present])
                frame_pose[step] = _kabsch(local_pts, world_pts)

            # Step 2: re-derive each marker's own local offset from every frame
            # where it was actually triangulated, via that frame's fitted pose.
            new_local: dict[str, np.ndarray] = {}
            for name in marker_names:
                samples = []
                for step, per_marker in frames.items():
                    if name not in per_marker:
                        continue
                    R, t = frame_pose[step]
                    samples.append(R.T @ (per_marker[name] - t))
                new_local[name] = np.median(np.array(samples), axis=0)
            local = new_local

        # One final pose pass against the *converged* local geometry -- the
        # loop above's own frame_pose was computed from the previous
        # iteration's offsets, one step behind by construction.
        frame_pose = {
            step: _kabsch(
                np.array([local[n] for n in per_marker]),
                np.array([per_marker[n] for n in per_marker]),
            )
            for step, per_marker in frames.items()
        }

        # Reprojection sanity check: project each marker's final local offset
        # through every frame's fitted pose into every camera, compare to the
        # real inlier observation that frame/camera -- same metric used
        # throughout this session's other refit tools.
        camera_ids, _ = get_run_cameras(conn, args.tracking_run)
        marker_names_all = get_run_markers(conn, args.tracking_run)
        wanted_idx = {n: marker_names_all.index(n) for n in marker_names}
        rows = conn.execute(
            "SELECT tracker_step, obs_blob FROM tracking_obs_results WHERE run_id = ? ORDER BY tracker_step",
            (args.tracking_run,),
        ).fetchall()
        n_cam, n_mrk = len(camera_ids), len(marker_names_all)
        resid_by_marker: dict[str, list[float]] = {n: [] for n in marker_names}
        for step, blob in rows:
            if step not in frame_pose:
                continue
            R, t = frame_pose[step]
            obs = decode_obs_blob(blob, n_cam, n_mrk)
            for name in marker_names:
                world_pt = R @ local[name] + t
                mi = wanted_idx[name]
                for ci, cam_id in enumerate(camera_ids):
                    slot = obs[ci, mi]
                    if slot[OBS_PAD] != OBS_MODE_ABSOLUTE:
                        continue
                    if not np.isfinite(slot[OBS_OUTLIER]) or slot[OBS_OUTLIER] != 0.0:
                        continue
                    ax, ay = float(slot[OBS_ACTUAL_X]), float(slot[OBS_ACTUAL_Y])
                    if not (np.isfinite(ax) and np.isfinite(ay)) or cam_id not in states:
                        continue
                    proj = _project(_proj_matrix(states[cam_id]), world_pt)
                    resid_by_marker[name].append(float(np.linalg.norm(proj - np.array([ax, ay]))))

        print(f"\n{'marker':16s} {'n_tri':>6s} {'along old->new':>16s} "
              f"{'lateral old->new':>18s} {'anterior old->new':>18s} {'median_px':>10s} {'n_reproj':>9s}")
        for name in marker_names:
            old = old_by_name[name]
            new_offset = local[name]
            lateral, along, anterior = (
                float(new_offset[0]), float(new_offset[1] / bone_len), float(new_offset[2]),
            )
            n_tri = sum(1 for per_marker in frames.values() if name in per_marker)
            resid = resid_by_marker[name]
            med_px = float(np.median(resid)) if resid else float("nan")
            print(f"{name:16s} {n_tri:6d} {old['along']:6.3f}->{along:6.3f}   "
                  f"{old['lateral']:+7.4f}->{lateral:+7.4f}   "
                  f"{old['anterior']:+7.4f}->{anterior:+7.4f}   {med_px:10.2f} {len(resid):9d}")


if __name__ == "__main__":
    main()

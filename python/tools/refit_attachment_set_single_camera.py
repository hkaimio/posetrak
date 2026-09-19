# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""refit_attachment_set_single_camera.py -- refits a calibrated marker
attachment set (fit_calibrated_attachment_set.py's B4 output) using a
DOT-AUGMENTED TRACKING RUN's own recorded observations, instead of
multi-camera triangulation of tracklet groups.

Why this exists
----------------
B4's fit needed two-or-more cameras to see the *same* dot simultaneously
so it could triangulate a 3D point before solving for the local offset --
`ankle_lat_L` only had 28 frames where that happened (status.md,
2026-09-12), against 500-2200+ for every other slot, and its fit shows
it: a lateral offset (13.4cm) roughly 3x every other ankle marker's, and
an `along` fraction (0.657) well off the ~0.82 the other three
independently-fit ankle markers agree on.

A completed dot-augmented tracking run changes what's available: it
already has the parent joint's own world transform T(t) at every frame,
resolved mostly by the body/hand pose-network markers (largely
independent of the dot markers themselves) -- so a SINGLE camera's 2D
observation, plus the already-known T(t) and that camera's calibration,
is one reprojection constraint on the 3-parameter local offset. Stack
that across every inlier observation of a slot (thousands, for
`ankle_lat_L` specifically 7658 in run d096d14f, vs. 28 triangulated
samples before) and it's a well-conditioned nonlinear least-squares
problem solvable from single-camera data alone.

This intentionally does NOT touch bone length -- each slot's offset is
fit independently against the skeleton's existing (possibly imperfect)
bone lengths, exactly like B4 did. Whether under/over-long bones are
biasing the `along` fractions towards what looks like "too far from the
joint" is a separate, real question (Harri's observation that legs look
too long across every run from this capture, together with the `along`
values clustering suspiciously far from the anatomically-expected 0/1
ends) -- flagged in status.md for its own design discussion, not folded
into this script.

Usage:
    python tools/refit_attachment_set_single_camera.py \\
        --session /path/to/session.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --tracking-run d096d14f-3e66-4875-a843-b3ddf9aab263 \\
        --calibrated-attachment-set /path/to/leg.calibrated.yaml \\
        --output /path/to/leg.calibrated.refit-single-camera.yaml
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mcp.db import (  # noqa: E402
    OBS_ACTUAL_X, OBS_ACTUAL_Y, OBS_OUTLIER, OBS_PAD, OBS_MODE_ABSOLUTE,
    decode_obs_blob, get_run_cameras, get_run_markers,
)
from app.setup.extrinsics_solver import _proj_matrix  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states  # noqa: E402
from tools.fit_calibrated_attachment_set import _TransformCache, _slot_parent_joint  # noqa: E402


@dataclass
class Sample:
    R: np.ndarray       # 3x3, parent joint rotation, world<-local
    trans: np.ndarray   # 3, parent joint translation, world
    P: np.ndarray       # 3x4 camera projection (undistorted pixel space)
    obs: np.ndarray     # 2, observed undistorted pixel (actual_x, actual_y)


def _project(P: np.ndarray, world_pt: np.ndarray) -> np.ndarray:
    h = P @ np.append(world_pt, 1.0)
    return h[:2] / h[2]


def _residuals(offset_local: np.ndarray, samples: list[Sample]) -> np.ndarray:
    out = np.empty(2 * len(samples))
    for i, s in enumerate(samples):
        world_pt = s.R @ offset_local + s.trans
        out[2 * i:2 * i + 2] = _project(s.P, world_pt) - s.obs
    return out


def collect_slot_samples(
    conn: sqlite3.Connection, tracking_run_id: str, slot: str, parent_joint: str,
    xform: _TransformCache, states: dict, person_id: int = 0,
) -> list[Sample]:
    """All inlier single-camera observations of *slot* across the whole run,
    each paired with the parent joint's own tracked transform at that
    instant -- one reprojection constraint per (frame, camera)."""
    camera_ids, _ = get_run_cameras(conn, tracking_run_id)
    marker_names = get_run_markers(conn, tracking_run_id)
    if slot not in marker_names:
        return []
    mi = marker_names.index(slot)
    n_cam, n_mrk = len(camera_ids), len(marker_names)

    rows = conn.execute(
        "SELECT tor.obs_blob, tr.timestamp_s FROM tracking_obs_results tor "
        "JOIN tracking_results tr ON tr.run_id = tor.run_id "
        "  AND tr.person_id = tor.person_id AND tr.tracker_step = tor.tracker_step "
        "  AND tr.is_smoothed = 0 "
        "WHERE tor.run_id = ? AND tor.person_id = ? ORDER BY tor.tracker_step",
        (tracking_run_id, person_id),
    ).fetchall()

    samples: list[Sample] = []
    for row in rows:
        obs = decode_obs_blob(row["obs_blob"], n_cam, n_mrk)
        t = row["timestamp_s"]
        T = xform.at(t)
        if T is None or parent_joint not in T:
            continue
        Tp = T[parent_joint]
        R, trans = Tp[:3, :3], Tp[:3, 3]
        for ci, cam_id in enumerate(camera_ids):
            slot_vals = obs[ci, mi]
            if slot_vals[OBS_PAD] != OBS_MODE_ABSOLUTE:
                continue  # not an absolute-pixel observation -- skip (see decode_obs_blob's own doc)
            if not np.isfinite(slot_vals[OBS_OUTLIER]) or slot_vals[OBS_OUTLIER] != 0.0:
                continue  # absent or outlier
            ax, ay = float(slot_vals[OBS_ACTUAL_X]), float(slot_vals[OBS_ACTUAL_Y])
            if not (np.isfinite(ax) and np.isfinite(ay)):
                continue
            if cam_id not in states:
                continue
            samples.append(Sample(R=R, trans=trans, P=_proj_matrix(states[cam_id]),
                                   obs=np.array([ax, ay])))
    return samples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--tracking-run", required=True,
                     help="A dot-augmented tracking run (has real per-frame dot observations).")
    ap.add_argument("--calibrated-attachment-set", required=True,
                     help="B4's original catalog -- supplies slot list, initial guess, and the "
                          "before/after comparison.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-samples", type=int, default=50,
                     help="Skip refitting a slot with fewer than this many usable observations "
                          "(falls back to the original B4 offset).")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    states = load_camera_states(conn, args.shot_id)
    xform = _TransformCache(conn, args.tracking_run)

    old_doc = yaml.safe_load(Path(args.calibrated_attachment_set).read_text())
    old_by_name = {m["name"]: m for m in old_doc["markers"]}

    print(f"{'slot':16s} {'n':>6s} {'old_px_rms':>11s} {'new_px_rms':>11s} "
          f"{'along old->new':>16s} {'lateral old->new':>18s} {'anterior old->new':>18s}")

    catalog_markers = []
    for slot in sorted(old_by_name):
        old = old_by_name[slot]
        parent_joint = _slot_parent_joint(slot)
        bone_len = xform.bone_length(parent_joint)
        x0 = np.array(old["offset"], dtype=float)

        samples = collect_slot_samples(conn, args.tracking_run, slot, parent_joint, xform, states)
        if len(samples) < args.min_samples:
            print(f"  {slot:16s} only {len(samples)} usable observations (<{args.min_samples}) "
                  "-- keeping the original B4 offset unchanged")
            catalog_markers.append(dict(old))
            continue

        old_rms = float(np.sqrt(np.mean(_residuals(x0, samples) ** 2)))
        result = least_squares(_residuals, x0, args=(samples,), loss="soft_l1", f_scale=15.0)
        new_offset = result.x
        new_rms = float(np.sqrt(np.mean(_residuals(new_offset, samples) ** 2)))

        lateral = float(new_offset[0])
        along = float(new_offset[1] / bone_len)
        anterior = float(new_offset[2])
        normal_vec = np.array([0.0, lateral, anterior])
        norm = float(np.linalg.norm(normal_vec))
        normal = (normal_vec / norm).tolist() if norm > 1e-9 else [0.0, 1.0, 0.0]

        resid_px = np.abs(_residuals(new_offset, samples)).reshape(-1, 2)
        resid_dist = np.linalg.norm(resid_px, axis=1)

        catalog_markers.append({
            "name": slot, "parent_joint": parent_joint,
            "along": round(along, 4), "lateral": round(lateral, 4), "anterior": round(anterior, 4),
            "offset": [round(float(x), 4) for x in new_offset],
            "normal": [round(float(x), 3) for x in normal],
            "n_samples": len(samples),
            "residual_median_px": round(float(np.median(resid_dist)), 2),
            "residual_p90_px": round(float(np.percentile(resid_dist, 90)), 2),
        })

        print(f"  {slot:16s} {len(samples):6d} {old_rms:11.2f} {new_rms:11.2f} "
              f"{old['along']:6.3f}->{along:6.3f}   "
              f"{old['lateral']:+7.4f}->{lateral:+7.4f}   "
              f"{old['anterior']:+7.4f}->{anterior:+7.4f}")

    requires_joints = sorted({j.split(".")[0] for j in {_slot_parent_joint(m["name"]) for m in catalog_markers}})
    out_doc = {
        "module": "leg",
        "requires_topology": xform.topology.name,
        "requires_topology_hash": xform.topology.hash,
        "requires_joints": requires_joints,
        "calibration": {
            "session": args.session, "shot_id": args.shot_id,
            "tracking_run": args.tracking_run,
            "source_attachment_set": args.calibrated_attachment_set,
            "method": "refit_attachment_set_single_camera.py -- nonlinear least-squares "
                      "reprojection fit of the local offset against every inlier single-camera "
                      "dot observation from a dot-augmented tracking run, using that run's own "
                      "tracked parent-joint transform per frame (no multi-camera triangulation "
                      "needed).",
            "date": date.today().isoformat(),
        },
        "markers": catalog_markers,
    }
    # REUSE-IgnoreStart -- builds the *output* file's own SPDX header as a
    # string; concatenated so `reuse lint` doesn't mistake it for a badly-
    # formed header on this .py file itself.
    spdx_tag = "SPDX" "-License-Identifier"
    spdx_header = f"# SPDX-FileCopyrightText: 2026 Harri Kaimio\n#\n# {spdx_tag}: Apache-2.0\n\n"
    # REUSE-IgnoreEnd
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(spdx_header + yaml.safe_dump(out_doc, sort_keys=False))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

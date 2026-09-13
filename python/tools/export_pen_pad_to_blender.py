# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""export_pen_pad_to_blender.py -- PROTOTYPE: convert
prototype_track_pen_trajectory.py's real-capture pen+pad CSV into a
Blender-space keyframe JSON, for blender_add_pen_pad_animation.py (run
inside Blender) to build animated objects from.

Plain Python, no bpy -- same two-stage split as blender_add_cameras.py's
own convention (Blender's bundled interpreter doesn't have this project's
own dependencies), and reuses posetrak.export.common's already-validated
tracker-Z-up -> Blender-Y-up conversion (_coord_matrices("yup"),
quat_to_matrix, matrix_to_quat_components) rather than re-deriving axis
conventions here.

Exports two rigid objects, each with its own per-frame position +
rotation keyframe:
  - "pen": marker 2's own solved world pose.
  - "pad": marker 1's own solved world pose (see status.md, 2026-09-13,
    for why marker 1 and not marker 0).
Plus a static (non-animated) child offset for a "pen_tip" empty, in the
pen's own local frame -- unlike a world position, a *local*, parent-
relative offset needs no coordinate-system conversion at all (the
parent's own already-converted rotation absorbs it; verified algebraically
before trusting it, not assumed): if the parent's Blender transform is
(R_blender = M @ R_tracker, t_blender = M @ t_tracker), then
R_blender @ offset + t_blender = M @ (R_tracker @ offset + t_tracker)
for *any* offset unchanged by M -- so Blender's own parenting produces
the correct world position for the tip automatically, without this
script ever needing to compute a world-space tip trajectory itself.

Only rows where the tracked object's own pose actually solved that
bucket get a keyframe (Blender interpolates the gaps) -- this script
does not fabricate a position for a frame the tracker had no data for.

Usage:
    python tools/export_pen_pad_to_blender.py \\
        --csv pen_pad_trajectory.csv \\
        --output blender_pen_pad_keyframes.json \\
        --fps 20
"""
from __future__ import annotations

import argparse
import csv as csv_module
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posetrak.export.common import (  # noqa: E402
    _coord_matrices, matrix_to_quat_components, quat_to_matrix,
)

# From prototype_pen_band_tracking.py's real-capture fit (see status.md,
# 2026-09-13) -- pen tip's offset in marker 2's own local frame.
TIP_LOCAL_OFFSET = (0.30634122, 0.03274555, -0.00472446)

OUTLIER_THRESHOLD_M = 0.08  # rolling-median residual gate, same as the
                            # trajectory script's own post-hoc cleanup


def _filter_outliers(times: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Boolean keep-mask: reject buckets whose position deviates from a
    size-9 rolling median by more than OUTLIER_THRESHOLD_M -- isolated
    single-frame bad pose solves this standalone pipeline has no proper
    outlier rejection for otherwise (see status.md's caveat)."""
    if len(positions) < 9:
        return np.ones(len(positions), dtype=bool)
    med = np.stack([median_filter(positions[:, i], size=9, mode="nearest") for i in range(3)], axis=1)
    resid = np.linalg.norm(positions - med, axis=1)
    return resid <= OUTLIER_THRESHOLD_M


def _to_blender_pose(pos_tracker: np.ndarray, quat_xyzw: np.ndarray, M: np.ndarray) -> tuple[list, list]:
    """(tracker Z-up position, quaternion xyzw) -> (Blender location, Blender
    rotation_quaternion wxyz), via the project's own established conversion."""
    qx, qy, qz, qw = quat_xyzw
    R = quat_to_matrix(qw, qx, qy, qz)
    t_blender = M @ pos_tracker
    R_blender = M @ R
    w, x, y, z = matrix_to_quat_components(R_blender)
    return [float(v) for v in t_blender], [w, x, y, z]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="prototype_track_pen_trajectory.py's output CSV")
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=float, default=20.0,
                     help="Blender scene fps to convert time_s -> frame number")
    args = ap.parse_args()

    rows = list(csv_module.DictReader(open(args.csv, newline="")))
    times = np.array([float(r["time_s"]) for r in rows])
    # time_s is the capture's own global timestamp (e.g. 98-113s into a much
    # longer multi-minute recording) -- normalize so the animation starts at
    # frame 0 rather than wherever that global time happens to land in frame
    # space (confirmed necessary: an unnormalized first attempt put every
    # keyframe around frame ~1960-2260, silently "working" but making the
    # object appear frozen at every frame before that in Blender's default
    # 1-250 timeline).
    t0 = times.min()

    def _extract(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        has = np.array([r[f"{prefix}_x"] != "" for r in rows])
        idx = np.where(has)[0]
        pos = np.array([[float(rows[i][f"{prefix}_x"]), float(rows[i][f"{prefix}_y"]),
                          float(rows[i][f"{prefix}_z"])] for i in idx])
        quat = np.array([[float(rows[i][f"{prefix}_qx"]), float(rows[i][f"{prefix}_qy"]),
                           float(rows[i][f"{prefix}_qz"]), float(rows[i][f"{prefix}_qw"])] for i in idx])
        return times[idx], pos, quat

    M, _ = _coord_matrices("yup")

    keyframes = {}
    for prefix, obj_name in (("marker2", "pen"), ("pad", "pad")):
        t, pos, quat = _extract(prefix)
        # Filter on the quantity actually validated as clean earlier (status.md,
        # 2026-09-13): for the pen, that's the *tip's* derived world position,
        # not marker 2's own raw translation -- a bad rotation solve can throw
        # the tip's position off a lot while barely moving marker 2's own
        # translation, so filtering on translation alone misses exactly the
        # single-frame glitches the rolling-median check exists to catch.
        if obj_name == "pen":
            tips = np.array([quat_to_matrix(q[3], q[0], q[1], q[2]) @ np.array(TIP_LOCAL_OFFSET) + p
                              for p, q in zip(pos, quat)])
            keep = _filter_outliers(t, tips)
        else:
            keep = _filter_outliers(t, pos)
        n_dropped = int((~keep).sum())
        t, pos, quat = t[keep], pos[keep], quat[keep]
        frames = []
        for ti, pi, qi in zip(t, pos, quat):
            loc, rot = _to_blender_pose(pi, qi, M)
            frames.append({"frame": round((ti - t0) * args.fps), "location": loc, "rotation_quaternion": rot})
        keyframes[obj_name] = frames
        print(f"{obj_name}: {len(frames)} keyframes ({n_dropped} outlier buckets dropped)")

    keyframes["pen_tip_local_offset"] = list(TIP_LOCAL_OFFSET)
    keyframes["fps"] = args.fps

    out_path = Path(args.output)
    out_path.write_text(json.dumps(keyframes, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

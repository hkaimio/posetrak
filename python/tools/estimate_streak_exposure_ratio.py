# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""estimate_streak_exposure_ratio.py — standalone validation for the
streak-velocity design (docs/roadmap/features/marker-based-mocap/
streak-velocity-design.md §3): does a motion-blur streak's length actually
predict a real, resolved frame-to-frame displacement, per camera, well
enough to be worth wiring into the tracker?

For a resolved tracking run, walks consecutive tracking_obs_results steps.
Whenever the same (camera, marker) slot was resolved (used, not an
outlier) in two consecutive steps, that pair's actual_x/y difference is a
real, undistorted-space frame-to-frame displacement -- exactly what the
existing MeasurementMode::VELOCITY residual already uses. The later step's
own raw dot candidate (matched by redistorting the resolved position and
finding the nearest raw candidate in that camera/frame's 'dots' blob) gives
major_axis - minor_axis (the streak's length beyond the dot's own
footprint -- the same "elongation" quantity resolve_dot_assignment()
already uses to inflate measurement noise, dot_assignment.cpp) and
dir_x/dir_y.

k = exposure_time / frame_time is estimated as a ratio of *sums*, not a
mean of per-sample ratios (streak-velocity-design.md §3's own rationale:
a mean of ratios blows up as an individual sample's displacement
approaches zero). Samples below --min-displacement-px are dropped before
summing -- the "only dots with actual movement" gate.

Usage:
    python tools/estimate_streak_exposure_ratio.py \\
        --session /path/to/session.db --run-id <tracking_run_id>
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mcp.db import decode_obs_blob, get_run_cameras, get_run_markers  # noqa: E402
from app.pose.db_cache import decode_dot_candidates  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402
from tools.render_tracking_debug_frames import _redistort_pts  # noqa: E402


@dataclass
class _CamAccum:
    sum_streak_px: float = 0.0
    sum_disp_px: float = 0.0
    n_samples: int = 0
    dir_cos_sum: float = 0.0
    n_dir_samples: int = 0
    # (timestamp, streak_px, disp_px) per accepted sample, for the
    # first-half/second-half stability check.
    samples: list[tuple[float, float, float]] = field(default_factory=list)

    def add(self, timestamp: float, streak_px: float, disp_px: float) -> None:
        self.sum_streak_px += streak_px
        self.sum_disp_px += disp_px
        self.n_samples += 1
        self.samples.append((timestamp, streak_px, disp_px))

    def add_direction_cos(self, cos: float) -> None:
        self.dir_cos_sum += cos
        self.n_dir_samples += 1

    def k(self) -> float | None:
        return self.sum_streak_px / self.sum_disp_px if self.sum_disp_px > 0 else None

    def half_split_k(self) -> tuple[float | None, float | None]:
        """k computed separately over the chronologically-first and
        -second half of accepted samples -- a real per-camera constant
        should agree closely between the two; a wide split is a warning
        the estimate isn't stable enough to trust yet."""
        ordered = sorted(self.samples, key=lambda s: s[0])
        mid = len(ordered) // 2
        if mid == 0:
            return None, None

        def _k_of(rows: list[tuple[float, float, float]]) -> float | None:
            s_streak = sum(r[1] for r in rows)
            s_disp = sum(r[2] for r in rows)
            return s_streak / s_disp if s_disp > 0 else None

        return _k_of(ordered[:mid]), _k_of(ordered[mid:])


def _raw_dot_candidates(
    conn: sqlite3.Connection, sequence_id: str, camera_instance_id: str, video_frame: int
) -> np.ndarray:
    row = conn.execute(
        "SELECT kp_blob FROM pose_observations WHERE sequence_id = ? "
        "AND camera_instance_id = ? AND source = 'dots' AND video_frame = ?",
        (sequence_id, camera_instance_id, video_frame),
    ).fetchone()
    if row is None:
        return np.zeros((0, 9), dtype=np.float32)
    try:
        return decode_dot_candidates(bytes(row["kp_blob"]))
    except ValueError:
        return np.zeros((0, 9), dtype=np.float32)


def estimate(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    min_displacement_px: float,
    min_elongation_px: float,
    match_tolerance_px: float,
) -> dict[str, _CamAccum]:
    run = conn.execute(
        "SELECT observation_sequence_id FROM tracking_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if run is None:
        raise SystemExit(f"tracking run not found: {run_id}")
    sequence_id = run["observation_sequence_id"]

    seq = conn.execute(
        "SELECT shot_id FROM pose_observation_sequences WHERE id = ?", (sequence_id,)
    ).fetchone()
    shot_id = seq["shot_id"]

    camera_ids, cam_names = get_run_cameras(conn, run_id)
    marker_names = get_run_markers(conn, run_id)
    n_cam, n_mrk = len(camera_ids), len(marker_names)

    cam_states = load_camera_states(conn, shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, shot_id)

    stats = {cam_id: _CamAccum() for cam_id in camera_ids}

    steps = conn.execute(
        "SELECT tracker_step, timestamp_s FROM tracking_results "
        "WHERE run_id = ? AND person_id = 0 ORDER BY tracker_step",
        (run_id,),
    ).fetchall()

    prev_actual: dict[tuple[int, int], tuple[float, float]] | None = None
    for step_row in steps:
        step, ts = step_row["tracker_step"], step_row["timestamp_s"]
        obs_row = conn.execute(
            "SELECT obs_blob FROM tracking_obs_results WHERE run_id = ? AND person_id = 0 "
            "AND tracker_step = ?",
            (run_id, step),
        ).fetchone()
        if obs_row is None:
            prev_actual = None  # tracking_lost this step -- continuity broken
            continue

        blob = decode_obs_blob(obs_row["obs_blob"], n_cam, n_mrk)
        cur_actual: dict[tuple[int, int], tuple[float, float]] = {}
        for ci in range(n_cam):
            for mi in range(n_mrk):
                ax, ay, _px, _py, _mahal, used, is_outlier, _pad = blob[ci, mi]
                if used > 0.5 and is_outlier < 0.5 and not np.isnan(ax):
                    cur_actual[(ci, mi)] = (float(ax), float(ay))

        if prev_actual is not None:
            for key, (ax, ay) in cur_actual.items():
                if key not in prev_actual:
                    continue
                px0, py0 = prev_actual[key]
                disp = float(np.hypot(ax - px0, ay - py0))
                if disp < min_displacement_px:
                    continue

                ci, _mi = key
                cam_id = camera_ids[ci]
                svid = svid_by_cam.get(cam_id)
                if svid is None:
                    continue
                frame_idx = sync_table.lookup(ts, svid)
                if frame_idx is None:
                    continue
                raw = _raw_dot_candidates(conn, sequence_id, cam_id, frame_idx)
                if raw.shape[0] == 0:
                    continue

                redist = _redistort_pts(cam_states[cam_id], np.array([[ax, ay]]))[0]
                dists = np.hypot(raw[:, 0] - redist[0], raw[:, 1] - redist[1])
                j = int(np.argmin(dists))
                if dists[j] > match_tolerance_px:
                    continue

                major, minor = float(raw[j, 4]), float(raw[j, 5])
                dir_x, dir_y = float(raw[j, 6]), float(raw[j, 7])
                elongation = major - minor
                if elongation < min_elongation_px:
                    continue

                stats[cam_id].add(ts, elongation, disp)

                if dir_x != 0.0 or dir_y != 0.0:
                    disp_dir = np.array([ax - px0, ay - py0]) / disp
                    streak_dir = np.array([dir_x, dir_y])
                    if np.dot(streak_dir, disp_dir) < 0:
                        streak_dir = -streak_dir
                    stats[cam_id].add_direction_cos(float(np.dot(streak_dir, disp_dir)))

        prev_actual = cur_actual

    return stats, cam_names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--min-displacement-px", type=float, default=3.0,
                    help="Drop samples below this frame-to-frame displacement -- the "
                         "'only dots with actual movement' gate against a noisy near-zero "
                         "denominator (default 3.0).")
    ap.add_argument("--min-elongation-px", type=float, default=1.0,
                    help="Drop candidates whose major_axis-minor_axis is below this -- "
                         "matches resolve_dot_assignment()'s own streak-noise-inflation "
                         "gate (dot_assignment.cpp) (default 1.0).")
    ap.add_argument("--match-tolerance-px", type=float, default=3.0,
                    help="Max redistorted-position-to-raw-candidate distance to accept a "
                         "match (default 3.0).")
    args = ap.parse_args()

    conn = sqlite3.connect(args.session)
    conn.row_factory = sqlite3.Row

    stats, cam_names = estimate(
        conn, args.run_id,
        min_displacement_px=args.min_displacement_px,
        min_elongation_px=args.min_elongation_px,
        match_tolerance_px=args.match_tolerance_px,
    )

    print(f"{'camera':<16} {'n':>6} {'k':>8} {'k(1st half)':>12} {'k(2nd half)':>12} "
          f"{'dir cos':>8}")
    for cam_id, acc in stats.items():
        label = cam_names.get(cam_id, cam_id[:8])
        if acc.n_samples == 0:
            print(f"{label:<16} {0:>6}      --           --           --       --")
            continue
        k = acc.k()
        k1, k2 = acc.half_split_k()
        dir_cos = acc.dir_cos_sum / acc.n_dir_samples if acc.n_dir_samples else None
        fmt = lambda v: f"{v:.4f}" if v is not None else "--"
        print(f"{label:<16} {acc.n_samples:>6} {fmt(k):>8} {fmt(k1):>12} {fmt(k2):>12} "
              f"{fmt(dir_cos):>8}")

    conn.close()


if __name__ == "__main__":
    main()

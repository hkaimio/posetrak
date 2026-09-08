# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_marker_person_filter.py — PROTOTYPE (no production code touched):
first spot-check for filtering the huge raw reflective-dot candidate pool
from a standalone marker detection run down to just the candidates plausibly
attached to one specific person, using that person's already-finalized pose
keypoints (vitpose-l-133kp) as a spatial cue -- rather than segmentation,
since a full-body pose sequence already covering the same time range exists
for this capture.

For each frame/camera this draws every raw dot candidate (small gray ring,
as in render_tracking_debug_frames.py's convention) plus each of the
person's leg keypoints (hip/knee/ankle/big-toe/heel, both sides) as a
colored cross, with a line to its assigned dot candidate within
--match-radius-px (if any). Assignment is a real one-to-one match
(`_assign_keypoints_to_dots`, scipy's `linear_sum_assignment`) over the
whole frame's (keypoints x candidates) cost matrix, NOT each keypoint
independently taking its own nearest candidate -- an earlier version did
that and, confirmed at scale (2026-09-08), let the same real candidate get
"matched" by several different keypoint names at once in 97.3% of frames
with any match at all. Still deliberately crude beyond that one fix -- no
tracking, no use of the marker layout's own known multiplicity (3 markers
at the knee, 2 at the ankle, this only ever considers one candidate per
named keypoint) -- just enough to eyeball whether the separation between
"on this person's legs" and "everything else" is clean before investing in
the real multi-frame calibration optimization.

Leg keypoint indices (vitpose-l-133kp / COCO-WholeBody, see
python/app/pose/kp_models.py's _COCO133_NAMES for the authoritative list):
    11 left_hip     12 right_hip
    13 left_knee    14 right_knee
    15 left_ankle   16 right_ankle
    17 left_big_toe 20 right_big_toe
    19 left_heel    22 right_heel
Note: this trial's physical marker set also has a marker "in front just
below the knee" with no corresponding vitpose keypoint at all -- expected to
show up as an isolated, unmatched dot candidate near (but not on) the knee
keypoint's crosshair; not a bug in this script.

Usage:
    python tools/prototype_marker_person_filter.py \\
        --session D:/mocap/2026-09-06-kare-tests/2026-09-06-kare-tests.db \\
        --marker-detection-run a6431d58-1ac9-4fed-86d7-f3904865e7d0 \\
        --pose-sequence ec1b3e2f-1ef8-4e31-806c-33102a969ecd \\
        --camera-label gopro13_01 pixel9 \\
        --start-time 35.7 --end-time 38.7 \\
        --output-dir scratch/marker_person_filter
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pose.db_cache import decode_dot_candidates  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402
from tools.calibrate_rigid_marker_body import load_sync_table  # noqa: E402

# name -> vitpose-l-133kp index, see module docstring.
_LEG_KEYPOINTS: dict[str, int] = {
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
    "left_big_toe": 17, "right_big_toe": 20,
    "left_heel": 19, "right_heel": 22,
}

_COLORS = {  # BGR, one per keypoint so a mismatch is easy to spot by eye
    "left_hip": (255, 0, 0), "right_hip": (255, 0, 255),
    "left_knee": (0, 255, 0), "right_knee": (0, 200, 200),
    "left_ankle": (0, 0, 255), "right_ankle": (0, 128, 255),
    "left_big_toe": (147, 20, 255), "right_big_toe": (180, 180, 255),
    "left_heel": (255, 128, 0), "right_heel": (128, 255, 255),
}


def _dot_candidates_for_frame(
    conn: sqlite3.Connection, run_id: str, shot_video_id: str, video_frame: int,
) -> np.ndarray:
    row = conn.execute(
        "SELECT keypoints FROM detection_keypoints WHERE detection_run_id = ? "
        "AND shot_video_id = ? AND video_frame = ? AND region_type = 'dots'",
        (run_id, shot_video_id, video_frame),
    ).fetchone()
    if row is None:
        return np.zeros((0, 9), dtype=np.float32)
    return decode_dot_candidates(bytes(row["keypoints"]))


def _leg_keypoints_for_frame(
    conn: sqlite3.Connection, sequence_id: str, camera_instance_id: str, video_frame: int,
) -> dict[str, tuple[float, float, float]]:
    """name -> (x, y, confidence), only for keypoints vitpose actually placed
    (confidence > 0 -- vitpose zeroes x/y/conf together for a keypoint it
    never predicted for this frame, same NaN-free convention as the rest of
    this project's confidence-gated keypoint data)."""
    row = conn.execute(
        "SELECT kp_blob FROM pose_observations WHERE sequence_id = ? "
        "AND camera_instance_id = ? AND video_frame = ? AND source = 'body'",
        (sequence_id, camera_instance_id, video_frame),
    ).fetchone()
    if row is None:
        return {}
    kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    out = {}
    for name, idx in _LEG_KEYPOINTS.items():
        x, y, conf = kp[idx]
        if conf > 0:
            out[name] = (float(x), float(y), float(conf))
    return out


def _assign_keypoints_to_dots(
    dots: np.ndarray, keypoints: dict[str, tuple[float, float, float]], match_radius_px: float,
) -> dict[str, int]:
    """One-to-one (Hungarian/linear-sum-assignment) match between named leg
    keypoints and raw dot candidates -- replaces an earlier version of this
    tool that gave each keypoint its own independent nearest-candidate
    search, which let the SAME real candidate get "matched" by several
    different keypoint names at once (confirmed at scale, 2026-09-08: 97.3%
    of frames with any match had at least one multiply-claimed candidate,
    most with 2-4 simultaneous claims -- exactly the "false positive at
    every toe" Harri flagged from a single-frame example, generalized).
    A real one-to-one assignment can't do that by construction: once a
    candidate is taken, `linear_sum_assignment` can't also give it to a
    second keypoint, it has to look elsewhere or leave that keypoint
    unmatched. Infeasible (too-far) pairs get a cost of 1e6 so the solver
    only ever proposes a pairing within match_radius_px; returns just the
    accepted (name -> dot index) pairs, not scipy's raw all-rows-assigned
    output."""
    names = list(keypoints)
    if not names or dots.shape[0] == 0:
        return {}
    cost = np.full((len(names), dots.shape[0]), 1e6)
    for ki, name in enumerate(names):
        kx, ky, _conf = keypoints[name]
        d = np.hypot(dots[:, 0] - kx, dots[:, 1] - ky)
        cost[ki, :] = np.where(d <= match_radius_px, d, 1e6)
    row_ind, col_ind = linear_sum_assignment(cost)
    return {names[ri]: ci for ri, ci in zip(row_ind, col_ind) if cost[ri, ci] < 1e6}


def _draw_overlay(
    img: np.ndarray, dots: np.ndarray, keypoints: dict[str, tuple[float, float, float]],
    match_radius_px: float,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    out = img.copy()
    all_pts: list[tuple[float, float]] = [(float(x), float(y)) for x, y, *_ in dots]

    for cx, cy, area, *_rest in dots:
        r = max(3, int(np.sqrt(max(area, 1.0) / np.pi)))
        # Bright cyan, not gray: gray sits too close to a real marker's own
        # white dot + dark surrounding ring/elastic to tell "detected" from
        # "not detected" at a glance (Harri's own review feedback on the
        # first version of this overlay) -- cyan is also a cool hue against
        # this footage's warm palette (skin, wood floor, red/orange
        # leggings), which is what actually makes it pop, not just brightness.
        cv2.circle(out, (int(round(cx)), int(round(cy))), r, (255, 255, 0), 2)

    assignment = _assign_keypoints_to_dots(dots, keypoints, match_radius_px)
    for name, (kx, ky, conf) in keypoints.items():
        color = _COLORS[name]
        cv2.drawMarker(out, (int(round(kx)), int(round(ky))), color,
                        markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
        all_pts.append((kx, ky))

        j = assignment.get(name)
        if j is not None:
            dx, dy = float(dots[j, 0]), float(dots[j, 1])
            d = float(np.hypot(dx - kx, dy - ky))
            cv2.line(out, (int(kx), int(ky)), (int(dx), int(dy)), color, 1, cv2.LINE_AA)
            cv2.circle(out, (int(dx), int(dy)), 6, color, 2)
            cv2.putText(out, f"{name} {d:.0f}px", (int(kx) + 8, int(ky) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return out, all_pts


def _crop_to_points(img: np.ndarray, pts: list[tuple[float, float]], pad: int) -> np.ndarray:
    if not pts:
        return img
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = max(0, int(min(xs) - pad)), min(img.shape[1], int(max(xs) + pad))
    y0, y1 = max(0, int(min(ys) - pad)), min(img.shape[0], int(max(ys) + pad))
    if x1 > x0 and y1 > y0:
        return img[y0:y1, x0:x1]
    return img


def process_camera(
    conn: sqlite3.Connection, run_id: str, sequence_id: str, camera_label: str,
    start_time: float, end_time: float, match_radius_px: float, output_dir: Path,
) -> None:
    cam_row = conn.execute("SELECT id FROM camera_instances WHERE label = ?", (camera_label,)).fetchone()
    if cam_row is None:
        raise SystemExit(f"no camera_instances row with label {camera_label!r}")
    camera_instance_id = cam_row["id"]

    video_row = conn.execute(
        "SELECT cv.id AS svid, cv.file_path FROM capture_videos cv "
        "JOIN pose_observation_sequences s ON s.shot_id = cv.shot_id "
        "WHERE s.id = ? AND cv.camera_instance_id = ?",
        (sequence_id, camera_instance_id),
    ).fetchone()
    if video_row is None:
        raise SystemExit(f"no capture_videos row for {camera_label} in this sequence's shot")
    svid, file_path = video_row["svid"], video_row["file_path"]

    sync_table, _ = load_sync_table(conn, conn.execute(
        "SELECT shot_id FROM pose_observation_sequences WHERE id = ?", (sequence_id,)
    ).fetchone()["shot_id"])

    frame_lo = sync_table.lookup(start_time, svid)
    frame_hi = sync_table.lookup(end_time, svid)
    if frame_lo is None or frame_hi is None:
        raise SystemExit(f"{camera_label}: no sync data for [{start_time}, {end_time}]")

    cam_dir = output_dir / camera_label
    cam_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_dots_total = 0
    n_matched_total = 0
    for frame_idx, img in iter_frames(file_path, frame_lo, frame_hi + 1):
        dots = _dot_candidates_for_frame(conn, run_id, svid, frame_idx)
        keypoints = _leg_keypoints_for_frame(conn, sequence_id, camera_instance_id, frame_idx)
        rendered, pts = _draw_overlay(img, dots, keypoints, match_radius_px)
        rendered = _crop_to_points(rendered, pts, pad=250)
        out_path = cam_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(out_path), rendered)
        n_written += 1
        n_dots_total += dots.shape[0]
        if dots.shape[0] > 0 and keypoints:
            for kx, ky, _ in keypoints.values():
                d = np.hypot(dots[:, 0] - kx, dots[:, 1] - ky)
                if d.min() <= match_radius_px:
                    n_matched_total += 1

    print(f"{camera_label}: wrote {n_written} frames -> {cam_dir}  "
          f"(avg {n_dots_total / max(n_written, 1):.1f} raw candidates/frame, "
          f"{n_matched_total / max(n_written, 1):.1f} leg-keypoint matches/frame "
          f"within {match_radius_px:.0f}px)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--marker-detection-run", required=True)
    ap.add_argument("--pose-sequence", required=True)
    ap.add_argument("--camera-label", nargs="+", required=True)
    ap.add_argument("--start-time", type=float, required=True)
    ap.add_argument("--end-time", type=float, required=True)
    ap.add_argument("--match-radius-px", type=float, default=80.0,
                     help="Max pixel distance from a leg keypoint to its nearest dot candidate "
                          "to draw a match line (default 80px -- generous first guess, tune by eye).")
    ap.add_argument("--output-dir", default="scratch/marker_person_filter")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    output_dir = Path(args.output_dir).resolve()
    for label in args.camera_label:
        process_camera(
            conn, args.marker_detection_run, args.pose_sequence, label,
            args.start_time, args.end_time, args.match_radius_px, output_dir,
        )


if __name__ == "__main__":
    main()

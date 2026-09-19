# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_tracklet_smoothed_assignment.py — PROTOTYPE: tracklet-level
majority-vote smoothing on top of prototype_hybrid_person_marker_assignment.py.

Motivation (2026-09-09, a real diagnosed case, see status.md): the hybrid
assignment is still purely per-frame independent -- no tracklet continuity
at all, despite DotCandidateWriter's own detection-time output already
carrying a per-camera tracklet_id for every raw candidate (assigned by the
existing per-camera dot-tracklet linker, already used in production for the prop-dot
case). A real example: heel_L and ankle_L's anchor keypoints came within
2.6cm of each other for a few frames, close enough (given the current
bootstrap-stage 15cm match radius) that they briefly competed for the same
nearby candidates and the per-frame Hungarian solver flipped which name
got which candidate for 3 frames, then flipped back -- the real world
position moved smoothly throughout; only the *label* glitched.

This is exactly the class of error tracklet continuity should catch: the
same physical candidate (same tracklet_id, in one camera) got a different
name for a few frames than the many frames around it. Two-pass approach:

    1. Run hybrid_assign_frame() independently per frame, as before, but
       additionally look up each assigned candidate's own tracklet_id
       (from the raw detection data, not currently threaded through
       hybrid_assign_frame() itself -- recovered here by position lookup
       against the same frame's raw candidates, which is simpler than
       plumbing tracklet_id through every function's signature).
    2. For each (camera, tracklet_id), take the majority-vote marker name
       across every frame that tracklet was assigned *any* name, and
       overwrite every one of that tracklet's per-frame assignments to the
       majority name -- a brief minority-frame flip gets corrected; a
       tracklet that was consistently labeled one way the whole time is
       unaffected.

Known, accepted limitation (matches person-marker-assignment-design.md's
own review UI plan): if the *tracklet linker itself* silently carries one
tracklet_id across what was actually two different physical dots (an
occlusion-recovery mistake), majority voting will force the wrong name
onto whichever portion is the minority -- this is exactly why the design's
UI plan includes a manual "split" action, not something this prototype
tries to detect on its own.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pose.db_cache import decode_dot_candidates  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402
from tools.prototype_hybrid_person_marker_assignment import hybrid_assign_frame  # noqa: E402


def run_tracklet_smoothed(
    conn: sqlite3.Connection, shot_id: str, marker_detection_run: str, pose_sequence: str,
    start_time: float, end_time: float, ref_camera_id: str | None = None, assign_fn=hybrid_assign_frame,
) -> dict[int, dict[str, dict[str, tuple[float, float, str]]]]:
    """Returns {video_frame: {camera_instance_id: {marker_name: (px, py, confidence)}}}
    -- same shape as calling hybrid_assign_frame() per frame, but with
    tracklet-majority-vote smoothing applied.

    video_frame keys are *ref_camera_id*'s own frame numbers (defaults to
    an arbitrary camera if not given) -- pass the camera a caller intends
    to look results up by (e.g. the one a validation video renders) so the
    returned dict's keys actually match what that caller will query.

    *assign_fn* is the per-frame assigner to smooth on top of -- defaults
    to hybrid_assign_frame, but accepts any function with the same
    signature/return shape (e.g.
    prototype_marker_normal_assignment.normal_aware_hybrid_assign_frame).
    """
    states = load_camera_states(conn, shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, shot_id)
    ref_cam = ref_camera_id or next(iter(states))
    ref_svid = svid_by_cam[ref_cam]
    frame_lo = sync_table.lookup(start_time, ref_svid)
    frame_hi = sync_table.lookup(end_time, ref_svid)

    per_frame_results: dict[int, dict[str, dict[str, tuple[float, float, str]]]] = {}
    # (camera_instance_id, tracklet_id) -> Counter of assigned names across frames
    tracklet_votes: dict[tuple[str, int], Counter] = defaultdict(Counter)
    # (video_frame, camera_instance_id) -> every (name, tracklet_id, px, py, conf) assigned that
    # frame, for the rewrite pass. Recording full per-frame membership up front (rather than
    # mutating per_frame_results in place, tracklet by tracklet) matters when two tracklets swap
    # names within the *same* frame: correcting one via pop(old_name)/[new_name]=... on the shared
    # dict then clobbers the other's in-flight correction, since both are fighting over the same
    # two dict keys (a "swap two variables without a temp" bug, caught 2026-09-09 -- see status.md).
    frame_cam_entries: dict[tuple[int, str], list[tuple[str, int, float, float, str]]] = defaultdict(list)

    for ref_frame in range(frame_lo, frame_hi):
        t = sync_table.frame_to_global_time(ref_frame, ref_svid)
        if t is None:
            continue
        dots_raw_by_cam, dots_xy_by_cam, kp_by_cam, frame_by_cam = {}, {}, {}, {}
        for cam_id, svid in svid_by_cam.items():
            if cam_id not in states:
                continue
            fidx = sync_table.lookup(t, svid)
            if fidx is None:
                continue
            frame_by_cam[cam_id] = fidx
            dot_row = conn.execute(
                "SELECT keypoints FROM detection_keypoints WHERE detection_run_id = ? AND shot_video_id = ? "
                "AND video_frame = ? AND region_type = 'dots'",
                (marker_detection_run, svid, fidx),
            ).fetchone()
            dots_full = decode_dot_candidates(bytes(dot_row["keypoints"])) if dot_row is not None else np.zeros((0, 9))
            dots_raw_by_cam[cam_id] = dots_full
            dots_xy_by_cam[cam_id] = dots_full[:, :2]
            pose_row = conn.execute(
                "SELECT kp_blob FROM pose_observations WHERE sequence_id = ? AND camera_instance_id = ? "
                "AND video_frame = ? AND source = 'body'",
                (pose_sequence, cam_id, fidx),
            ).fetchone()
            if pose_row is not None:
                kp_by_cam[cam_id] = np.frombuffer(bytes(pose_row["kp_blob"]), dtype=np.float32).reshape(-1, 3)

        if not kp_by_cam or not any(d.shape[0] > 0 for d in dots_xy_by_cam.values()):
            continue

        result = assign_fn(dots_xy_by_cam, kp_by_cam, states)
        frame_key = frame_by_cam[ref_cam]
        per_frame_results[frame_key] = result

        for cam_id, assigns in result.items():
            dots_full = dots_raw_by_cam[cam_id]
            if dots_full.shape[0] == 0:
                continue
            for name, (px, py, _conf) in assigns.items():
                d = np.hypot(dots_full[:, 0] - px, dots_full[:, 1] - py)
                j = int(np.argmin(d))
                if d[j] > 0.5:  # sanity: should be an exact/near-exact position match
                    continue
                tracklet_id = int(dots_full[j, 8])
                if tracklet_id < 0:
                    continue
                tracklet_votes[(cam_id, tracklet_id)][name] += 1
                frame_cam_entries[(frame_key, cam_id)].append((name, tracklet_id, px, py, _conf))

    # Rewrite pass: rebuild each (frame, camera)'s assignment dict from scratch, mapping every
    # entry's own tracklet_id straight to that tracklet's majority name, instead of mutating the
    # existing dict in place one tracklet at a time (see frame_cam_entries' comment above for why
    # in-place pop/insert corrupts mutual name swaps within a single frame).
    majority_name = {key: counter.most_common(1)[0][0] for key, counter in tracklet_votes.items()}
    n_corrected = 0
    n_collisions = 0
    for (frame_key, cam_id), entries in frame_cam_entries.items():
        new_assigns: dict[str, tuple[float, float, str]] = {}
        for original_name, tracklet_id, px, py, conf in entries:
            best_name = majority_name[(cam_id, tracklet_id)]
            if best_name != original_name:
                n_corrected += 1
            if best_name in new_assigns:
                # Two different tracklets both majority-vote to the same name in this frame --
                # keep whichever arrived first and drop the other rather than silently overwrite.
                n_collisions += 1
                continue
            new_assigns[best_name] = (px, py, conf)
        per_frame_results[frame_key][cam_id] = new_assigns

    if n_collisions:
        print(f"tracklet smoothing: {n_collisions} same-frame name collisions dropped", file=sys.stderr)
    print(f"tracklet smoothing: {len(tracklet_votes)} (camera, tracklet) tracks, "
          f"{n_corrected} per-frame name corrections applied", file=sys.stderr)
    return per_frame_results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--marker-detection-run", required=True)
    ap.add_argument("--pose-sequence", required=True)
    ap.add_argument("--start-time", type=float, required=True)
    ap.add_argument("--end-time", type=float, required=True)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    run_tracklet_smoothed(
        conn, args.shot_id, args.marker_detection_run, args.pose_sequence, args.start_time, args.end_time,
    )


if __name__ == "__main__":
    main()

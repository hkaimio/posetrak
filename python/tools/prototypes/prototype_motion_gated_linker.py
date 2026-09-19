# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_motion_gated_linker.py -- prototype used to design and
validate `MotionGatedLinker`, since ported into production
(posetrak/detection/dot_tracklet.py, 2026-09-11) replacing an earlier
linker that linked purely by a fixed 60px nearest-neighbor gate with no
identity/appearance or motion-consistency check at all. Kept here as the
validation harness (the regression/purity checks below) for tuning the
production linker's parameters against real ground truth in the future --
run it again after any change to `dot_tracklet.py`'s gates.

Motivation (status.md, 2026-09-11): reviewing the 42-48s/5-camera B2/B3
test window by hand, Harri found ~14 of 25 final assigned tracklet
groups needed a manual mid-tracklet split because DotTrackletLinker had
silently switched from tracking one real marker onto a different, nearby
one. The suspected mechanism: an occluded track's "last known position"
in the production linker stays *frozen* while missed, so a different
real marker that later drifts near that stale point gets adopted --
Harri's own suggestion was some temporal mechanism that penalizes
surprising movement, e.g. a simple Kalman/particle filter.

This prototype replaces the fixed-radius gate with a per-tracklet
constant-velocity Kalman filter: an occluded track *coasts* forward
along its own last known velocity instead of freezing in place, and a
candidate is only accepted as a continuation if it's consistent with
where that motion predicts it should be now (a Mahalanobis gate on the
KF innovation), not merely near the last seen pixel. A brand-new track
(no velocity estimate yet) falls back to a small fixed-radius gate for
its second point only, same as production's behavior at track birth --
there is no motion history yet to be inconsistent with.

Harri's explicit priority (2026-09-11): tracklet construction should
strongly favor a low false-link rate over completeness. An unnecessary
break is cheap to fix later (B2's cross-camera reprojection test, or a
human via the B3 scrub-and-split tool); a false link silently corrupts a
slot's whole trajectory. Every gate here defaults tight for that reason.

Validated (no production/detection changes here) against real,
human-produced ground truth from the same B3 review: the real
mid-tracklet splits Harri made while reviewing the 42-48s/5-camera
window are used as a named regression set -- `tracklet_group_
assignments.json`'s 4-element (sub-range) members record exactly where a
human decided one tracklet_id actually contained two different physical
markers. For each, this checks whether the new linker places the frames
on either side of that real cut into two *different* tracklet ids. The
tracklets that did *not* need a split (2-element members) are used as a
purity check in the other direction -- confirming the new linker doesn't
fragment a clean, single-marker tracklet into a lot of needless pieces
chasing the tighter gate.

Usage:
    python tools/prototype_motion_gated_linker.py \\
        --session /path/to/session.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --detection-run 6abcba67-4e7a-4852-87c3-14134599c1e4 \\
        --start-time 42.0 --end-time 48.0 \\
        --groups scratch/dot_ground_truth/tracklet_groups.json \\
        --assignments scratch/dot_ground_truth/tracklet_group_assignments.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.pose.db_cache import decode_dot_candidates  # noqa: E402
from posetrak.detection.dot_blob_detector import BlobCandidate  # noqa: E402
from posetrak.detection.dot_tracklet import MotionGatedLinker  # noqa: E402
from tools.calibrate_rigid_marker_body import load_sync_table  # noqa: E402


def relink_camera_to_tracklets(
    conn, detection_run: str, svid: str, lo: int, hi: int, **linker_kwargs
) -> dict[int, dict[int, tuple[float, float]]]:
    """Same re-link as `_relink_camera`, but shaped exactly like
    `build_tracklet_groups.collect_tracklets()`'s own return value
    ({tracklet_id: {video_frame: (px, py)}}) -- lets B2's own
    prefilter/build_groups/validate run directly against an alternative
    linker parameterization without a new detection run, since B2
    otherwise only ever reads whatever tracklet_id is already stored in
    the DB from whichever linker produced that detection run."""
    rows = conn.execute(
        "SELECT video_frame, keypoints FROM detection_keypoints WHERE detection_run_id = ? AND shot_video_id = ? "
        "AND video_frame BETWEEN ? AND ? AND region_type = 'dots' ORDER BY video_frame",
        (detection_run, svid, lo, hi),
    ).fetchall()
    linker = MotionGatedLinker(**linker_kwargs)
    out: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for r in rows:
        arr = decode_dot_candidates(bytes(r["keypoints"]))
        cands = [
            BlobCandidate(cx=float(c[0]), cy=float(c[1]), area=0.0, compactness=1.0,
                          bbox=(0, 0, 1, 1), major_axis_px=1.0, minor_axis_px=1.0)
            for c in arr
        ]
        linker.link_frame(r["video_frame"], cands)
        for c in cands:
            out[c.tracklet_id][r["video_frame"]] = (c.cx, c.cy)
    return dict(out)


def _relink_camera(conn, detection_run: str, svid: str, lo: int, hi: int, **linker_kwargs) -> list[tuple[int, int, int]]:
    """Returns [(video_frame, old_tid, new_tid), ...] for every candidate
    in the window -- old_tid is production's own from whatever linker made
    this detection run (kept only for matching against ground truth,
    never fed into the new linker). Runs the real, production
    `MotionGatedLinker` (not a local copy) so this stays a true
    regression check against whatever `dot_tracklet.py` actually does."""
    rows = conn.execute(
        "SELECT video_frame, keypoints FROM detection_keypoints WHERE detection_run_id = ? AND shot_video_id = ? "
        "AND video_frame BETWEEN ? AND ? AND region_type = 'dots' ORDER BY video_frame",
        (detection_run, svid, lo, hi),
    ).fetchall()
    linker = MotionGatedLinker(**linker_kwargs)
    out = []
    for r in rows:
        arr = decode_dot_candidates(bytes(r["keypoints"]))
        old_tids = [int(c[8]) for c in arr]
        cands = [
            BlobCandidate(cx=float(c[0]), cy=float(c[1]), area=0.0, compactness=1.0,
                          bbox=(0, 0, 1, 1), major_axis_px=1.0, minor_axis_px=1.0)
            for c in arr
        ]
        linker.link_frame(r["video_frame"], cands)  # mutates cands[i].tracklet_id in place
        new_tids = [c.tracklet_id for c in cands]
        out.extend(zip([r["video_frame"]] * len(cands), old_tids, new_tids))
    return out


def _check_cut(records: list[tuple[int, int, int]], tid: int, cut_frame: int):
    before = [nt for vf, ot, nt in records if ot == tid and vf < cut_frame]
    after = [nt for vf, ot, nt in records if ot == tid and vf >= cut_frame]
    if not before or not after:
        return None
    before_mode, before_n = Counter(before).most_common(1)[0]
    after_mode, after_n = Counter(after).most_common(1)[0]
    return {
        "separated": before_mode != after_mode,
        "before_mode": before_mode, "before_purity": before_n / len(before), "n_before": len(before),
        "after_mode": after_mode, "after_purity": after_n / len(after), "n_after": len(after),
    }


def _check_purity(records: list[tuple[int, int, int]], tid: int) -> tuple[float, int] | None:
    obs = [nt for vf, ot, nt in records if ot == tid]
    if not obs:
        return None
    _, mode_n = Counter(obs).most_common(1)[0]
    return mode_n / len(obs), len(obs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--detection-run", required=True)
    ap.add_argument("--start-time", type=float, required=True)
    ap.add_argument("--end-time", type=float, required=True)
    ap.add_argument("--groups", required=True, help="build_tracklet_groups.py's --output file (for meta, unused otherwise).")
    ap.add_argument("--assignments", required=True, help="label_tracklet_groups_gui.py's --output file.")
    ap.add_argument("--birth-gate-px", type=float, default=25.0)
    ap.add_argument("--mahal-gate", type=float, default=9.21)
    ap.add_argument("--max-missed", type=int, default=3)
    ap.add_argument("--meas-std-px", type=float, default=3.0)
    ap.add_argument("--accel-std-px", type=float, default=2.0)
    def _float_or_none(s: str) -> float | None:
        return None if s.lower() == "none" else float(s)

    ap.add_argument("--max-noise-dt", type=_float_or_none, default=None)
    ap.add_argument("--disambiguation-margin", type=float, default=2.0)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)
    label_to_cam = {r["label"]: r["id"] for r in conn.execute("SELECT id, label FROM camera_instances")}

    assign = json.loads(Path(args.assignments).read_text())

    frags_by_key: dict[tuple[str, int], list[tuple[int | None, int | None]]] = defaultdict(list)
    clean: set[tuple[str, int]] = set()
    for a in assign.values():
        for m in a.get("members", []):
            if len(m) == 4:
                label, tid, lo, hi = m
                frags_by_key[(label, tid)].append((lo, hi))
            else:
                clean.add(tuple(m))

    cuts: set[tuple[str, int, int]] = set()
    for (label, tid), ranges in frags_by_key.items():
        for lo, hi in ranges:
            # both bounds are real human-chosen cut frames -- a fragment
            # missing its other half (the remainder was discarded as
            # noise, not assigned anywhere) still records one real cut.
            if lo is not None:
                cuts.add((label, tid, lo))
            if hi is not None:
                cuts.add((label, tid, hi))
    cuts = sorted(cuts)
    clean = sorted(clean)

    print(f"{len(cuts)} real human-determined cut points, {len(clean)} clean (non-split) tracklets.\n")

    linker_kwargs = dict(
        birth_gate_px=args.birth_gate_px, mahal_gate=args.mahal_gate, max_missed=args.max_missed,
        meas_std_px=args.meas_std_px, accel_std_px=args.accel_std_px, max_noise_dt=args.max_noise_dt,
        disambiguation_margin=args.disambiguation_margin,
    )
    cams_needed = sorted({label for label, *_ in cuts} | {label for label, _ in clean})
    records_by_cam: dict[str, list[tuple[int, int, int]]] = {}
    for label in cams_needed:
        cam_id = label_to_cam[label]
        svid = svid_by_cam[cam_id]
        lo = sync_table.lookup(args.start_time, svid)
        hi = sync_table.lookup(args.end_time, svid)
        records_by_cam[label] = _relink_camera(conn, args.detection_run, svid, lo, hi, **linker_kwargs)

    print("=== Regression: does the new linker separate each real human cut? ===")
    n_sep = 0
    for label, tid, cut in cuts:
        r = _check_cut(records_by_cam[label], tid, cut)
        if r is None:
            print(f"  {label}#{tid} @ {cut}: SKIP (no frames on one side)")
            continue
        n_sep += r["separated"]
        status = "OK  separated" if r["separated"] else "FAIL still merged"
        print(f"  {label}#{tid} @ {cut}: {status}  "
              f"(before: new_tid={r['before_mode']} purity={r['before_purity']:.2f} n={r['n_before']}, "
              f"after: new_tid={r['after_mode']} purity={r['after_purity']:.2f} n={r['n_after']})")
    print(f"\n{n_sep}/{len(cuts)} real cuts correctly separated.\n")

    print("=== Purity check: does the new linker over-fragment clean tracklets? ===")
    purities = []
    for label, tid in clean:
        r = _check_purity(records_by_cam.get(label, []), tid)
        if r is None:
            continue
        purity, n = r
        purities.append(purity)
        flag = "" if purity >= 0.9 else "  <-- fragmented"
        print(f"  {label}#{tid}: purity={purity:.2f} (n={n}){flag}")
    if purities:
        print(f"\nmedian purity={float(np.median(purities)):.3f}  min={min(purities):.3f}  "
              f"({sum(p >= 0.9 for p in purities)}/{len(purities)} >= 0.90)")


if __name__ == "__main__":
    main()

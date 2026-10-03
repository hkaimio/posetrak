# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""evaluate_tracker.py — score a tracking run's dot assignments against a frozen ground truth.

    python -m posetrak.markers.evaluate_tracker SESSION_DB SHOT DETECTION_RUN TRUTH.json RUN [RUN ...]

For every labelled tracklet-frame of the ground truth (every third frame of each segment), the
tracker's used observation (``tracking_obs_results``, inlier, at the tracker step nearest in time)
is looked up to see which marker, if any, took that tracklet's dot:

* *correct*: the labelled marker; *wrong slot*: another marker; *missed*: no marker;
* *clutter given a slot*: a frame labelled not-a-marker or other-subject that took a marker.

A dot is matched to the tracklet by position within 1.5 px of any detection of the same tracklet
in the three video frames either side of the labelled one. The tracker steps at twice the video frame
rate and uses the neighbouring frame's detection in about half of its steps, so an exact position
match understates recall during fast movement by the distance the dot moves in a frame.

Precision is correct / (correct + wrong slot + clutter given a slot); recall is correct / (correct +
wrong slot + missed) over the frames labelled as a marker.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import sqlite3

import numpy as np

from posetrak.calibration.session_cameras import load_camera_states, load_sync_table
from posetrak.markers.evaluate import load_frozen
from posetrak.markers.review import NOT_A_MARKER, OTHER_SUBJECT
from posetrak.markers.tracklets import collect_tracklets

MATCH_PX = 1.5
NEIGHBOUR_FRAMES = 3
MAX_STEP_SKEW_S = 0.006
SAMPLE_EVERY = 3


def score_run(conn: sqlite3.Connection, shot_id: str, detection_run: str, truth, run_prefix: str) -> dict:
    """Counts of correct / wrong_slot / missed / clutter_assigned frames, and per-marker detail."""
    from app.setup.extrinsics_solver import _undistort_pts

    conn.row_factory = sqlite3.Row
    states = load_camera_states(conn, shot_id)
    sync, svid = load_sync_table(conn, shot_id)
    labels = {r["id"]: r["label"] for r in conn.execute("SELECT id, label FROM camera_instances")}
    run = conn.execute("SELECT id, active_camera_ids, marker_names FROM tracking_runs WHERE id LIKE ?",
                       (run_prefix + "%",)).fetchone()
    cams, markers = json.loads(run["active_camera_ids"]), json.loads(run["marker_names"])
    steps = [(r[0], r[1]) for r in conn.execute(
        "SELECT tracker_step, timestamp_s FROM tracking_results WHERE run_id=? AND is_smoothed=0 ORDER BY tracker_step",
        (run["id"],))]
    times = [t for _, t in steps]
    blobs: dict[int, np.ndarray | None] = {}

    def blob(step: int):
        if step not in blobs:
            row = conn.execute("SELECT obs_blob FROM tracking_obs_results WHERE run_id=? AND tracker_step=?",
                               (run["id"], step)).fetchone()
            blobs[step] = None if row is None else np.frombuffer(row[0], dtype=np.float32).reshape(len(cams), len(markers), 8)
        return blobs[step]

    tracklets = {cam: collect_tracklets(conn, detection_run, svid[cam], min(t[2] for t in truth if t[0] == cam),
                                        max(t[3] for t in truth if t[0] == cam)) for cam in {t[0] for t in truth}}
    n: collections.Counter = collections.Counter()
    per_marker: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    confusions: collections.Counter = collections.Counter()
    for cam, tid, first, last, label in truth:
        if labels[cam] not in cams:
            continue
        ci, real = cams.index(labels[cam]), label not in (NOT_A_MARKER, OTHER_SUBJECT)
        for frame in [f for f in sorted(tracklets[cam].get(tid, {})) if first <= f <= last][::SAMPLE_EVERY]:
            t = sync.frame_to_global_time(frame, svid[cam])
            k = min(max(bisect.bisect_left(times, t), 0), len(times) - 1)
            observed = blob(steps[k][0])
            if observed is None or abs(times[k] - t) > MAX_STEP_SKEW_S:
                continue
            near = [tracklets[cam][tid][g] for g in range(frame - NEIGHBOUR_FRAMES, frame + NEIGHBOUR_FRAMES + 1)
                    if g in tracklets[cam][tid]]
            distance = np.full(len(markers), 1e9)
            for u in _undistort_pts(np.array(near), states[cam]):
                d = np.hypot(observed[ci, :, 0] - u[0], observed[ci, :, 1] - u[1])
                distance = np.minimum(distance, np.where(np.isfinite(d) & (observed[ci, :, 5] == 1), d, 1e9))
            best = int(np.argmin(distance))
            got = markers[best] if distance[best] < MATCH_PX else None
            if real:
                per_marker[label]["truth"] += 1
                if got == label:
                    n["correct"] += 1
                    per_marker[label]["found"] += 1
                elif got is None:
                    n["missed"] += 1
                else:
                    n["wrong_slot"] += 1
                    confusions[(got, label)] += 1
            elif got is not None:
                n["clutter_assigned"] += 1
    proposed = n["correct"] + n["wrong_slot"] + n["clutter_assigned"]
    truth_n = n["correct"] + n["wrong_slot"] + n["missed"]
    return {"counts": dict(n), "precision": n["correct"] / max(proposed, 1), "recall": n["correct"] / max(truth_n, 1),
            "confusions": dict(confusions), "per_marker": {k: dict(v) for k, v in per_marker.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_db")
    ap.add_argument("shot")
    ap.add_argument("detection_run")
    ap.add_argument("truth")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    conn = sqlite3.connect(f"file:{a.session_db}?mode=ro", uri=True)
    shot = conn.execute("SELECT id FROM captures WHERE id LIKE ?", (a.shot + "%",)).fetchone()[0]
    det = conn.execute("SELECT id FROM detection_runs WHERE id LIKE ?", (a.detection_run + "%",)).fetchone()[0]
    truth = load_frozen(a.truth)
    for run in a.runs:
        r = score_run(conn, shot, det, truth, run)
        c = r["counts"]
        print(f"{run}: precision {100 * r['precision']:.1f}%  recall {100 * r['recall']:.1f}%  "
              f"(correct {c.get('correct', 0)}, wrong slot {c.get('wrong_slot', 0)}, missed {c.get('missed', 0)}, "
              f"clutter given a slot {c.get('clutter_assigned', 0)})")
        print("   top confusions:", sorted(r["confusions"].items(), key=lambda kv: -kv[1])[:6])


if __name__ == "__main__":
    main()

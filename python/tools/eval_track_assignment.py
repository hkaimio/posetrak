# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""eval_track_assignment.py — score a track-to-person assignment against segmentation-derived identity.

Each box of the assigned detection run is matched, by overlap, to the box of the reference run (a segmentation-
sourced run whose tracks are the persons) and the person assigned to the box is compared with the reference's
person. Reports the agreement overall, where persons are apart (their boxes do not overlap horizontally) and
where they are not, per person and per camera, the share of disagreeing boxes the assignment report listed for
checking, and the stretches of each person that no assigned box covers although the reference sees the person in
two or more cameras.

    python tools/eval_track_assignment.py SESSION.db RUN --reference SEG_RUN [--report FILE]
        [--exclude-camera PREFIX ...] [--match-names]

Person names are matched to the reference's by the pairing that maximises agreement (a seed click per person
gives the same), unless --match-names says they are the same names. The database is opened read-only.
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posetrak.detection.assign_run import load_sync, report_path  # noqa: E402


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def _boxes(conn, run_id):
    return conn.execute(
        "SELECT shot_video_id, video_frame, track_id, bbox_x, bbox_y, bbox_w, bbox_h FROM person_detections "
        "WHERE detection_run_id = ? AND region_type = 'full_body' AND bbox_w > 0", (run_id,))


def _rows(conn, run_id):
    out = collections.defaultdict(list)
    for r in conn.execute("SELECT shot_video_id, track_id, person_name, first_frame, last_frame "
                          "FROM detection_track_assignments WHERE detection_run_id = ?", (run_id,)):
        out[(r[0], r[1])].append((r[3], r[4], r[2]))
    return out


def _person_at(rows, svid, tid, frame):
    return next((p for a, b, p in rows.get((svid, tid), ()) if a <= frame <= b), None)


def _flagged(report, svid, frame, t, person) -> bool:
    for c in report["checks"]:
        if c["kind"] in ("contact", "handover"):
            if c["t0"] <= t <= c["t1"] and (person is None or person in c["persons"]):
                return True
        elif any(s == svid and a <= frame <= b for s, a, b in c["frames"]):
            return True
    return False


def evaluate(conn, run_id, reference_id, report=None, exclude=(), match_names=False) -> dict:
    """The agreement figures (see the module docstring) as a dict, and the per-box records under 'boxes'."""
    sync = load_sync(conn, conn.execute("SELECT sync_config_id FROM detection_runs WHERE id = ?",
                                        (run_id,)).fetchone()[0])
    ref_rows = _rows(conn, reference_id)
    ref: dict[tuple[str, int], dict[str, tuple]] = collections.defaultdict(dict)
    for svid, f, tid, x, y, w, h in _boxes(conn, reference_id):
        person = _person_at(ref_rows, svid, tid, f)
        if person:
            ref[(svid, f)][person] = (x, y, w, h)
    rows = _rows(conn, run_id)
    boxes: list[list] = []
    for svid, f, tid, x, y, w, h in _boxes(conn, run_id):
        if any(svid.startswith(p) for p in exclude):
            continue
        near = ref.get((svid, f))
        if not near:
            continue
        who, box = max(near.items(), key=lambda kv: _iou((x, y, w, h), kv[1]))
        if _iou((x, y, w, h), box) < 0.5:
            continue
        cx = box[0] + box[2] / 2
        apart = all(abs(cx - (o[0] + o[2] / 2)) >= (box[2] + o[2]) / 2 for p, o in near.items() if p != who)
        name = _person_at(rows, svid, tid, f)
        boxes.append([svid, f, tid, who, name, apart, name])      # [..., the run's own name of the person]
    if not match_names:
        refs = sorted({b[3] for b in boxes})
        mine = sorted({b[4] for b in boxes if b[4]})
        M = np.array([[sum(1 for b in boxes if b[3] == g and b[4] == a) for a in mine] for g in refs])
        r, c = linear_sum_assignment(-M)
        pair = {mine[j]: refs[i] for i, j in zip(r, c)}
        for b in boxes:
            b[4] = pair.get(b[4], b[4]) if b[4] else None
    for b in boxes:
        t = sync.frame_to_global_time(b[1], b[0])
        flagged = report is not None and (b[6] is None or _flagged(report, b[0], b[1], t, b[6]))
        b.append(flagged)
        b.append(t)
    out = {"boxes": boxes}

    def rate(sel):
        return len(sel), sum(1 for b in sel if b[3] == b[4])

    out["all"], out["apart"] = rate(boxes), rate([b for b in boxes if b[5]])
    out["close"] = rate([b for b in boxes if not b[5]])
    out["unassigned"] = sum(1 for b in boxes if b[4] is None)
    wrong = [b for b in boxes if b[3] != b[4]]
    out["disagree"] = (len(wrong), sum(1 for b in wrong if b[7]))
    wrong_apart = [b for b in wrong if b[5]]
    out["disagree_apart"] = (len(wrong_apart), sum(1 for b in wrong_apart if b[7]))
    out["per_person"] = {p: rate([b for b in boxes if b[5] and b[3] == p]) for p in sorted({b[3] for b in boxes})}
    out["per_camera"] = {s: (rate([b for b in boxes if b[5] and b[0] == s]), rate([b for b in boxes if b[0] == s]))
                         for s in sorted({b[0] for b in boxes})}
    out["gaps"] = _gaps(ref, sync, exclude, boxes)
    return out


def _gaps(ref, sync, exclude, boxes) -> dict:
    """Per reference person: the stretches where the reference sees the person in two or more cameras but the
    sequence of the paired person has no box within a frame (time resolution 1/30 s)."""
    have = collections.defaultdict(set)                 # person -> time bins at which its sequence has a box
    for svid, f, tid, who, assigned, *_ in boxes:
        if assigned:
            have[assigned].add(round(sync.frame_to_global_time(f, svid) * 30))
    seen = collections.defaultdict(lambda: collections.defaultdict(set))    # person -> time bin -> cameras
    for (svid, f), near in ref.items():
        if not any(svid.startswith(p) for p in exclude):
            for person in near:
                seen[person][round(sync.frame_to_global_time(f, svid) * 30)].add(svid)
    out: dict[str, list[tuple[float, float]]] = {}
    for person, bins in seen.items():
        gap_bins = sorted(b for b, cams in bins.items()
                          if len(cams) >= 2 and not ({b - 1, b, b + 1} & have[person]))
        spans: list[list[int]] = []
        for b in gap_bins:
            if spans and b - spans[-1][1] <= 1:
                spans[-1][1] = b
            else:
                spans.append([b, b])
        out[person] = [(a / 30, b / 30) for a, b in spans]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db")
    ap.add_argument("run", help="the assigned detection run (id prefix)")
    ap.add_argument("--reference", required=True, help="reference detection run (id prefix)")
    ap.add_argument("--report", help="assignment report JSON (default: next to the database)")
    ap.add_argument("--exclude-camera", action="append", default=[], metavar="PREFIX",
                    help="leave out the capture video whose id starts with PREFIX (unreliable reference)")
    ap.add_argument("--match-names", action="store_true", help="names of the run and the reference are the same")
    args = ap.parse_args()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    run_id = conn.execute("SELECT id FROM detection_runs WHERE id LIKE ? || '%'", (args.run,)).fetchone()[0]
    ref_id = conn.execute("SELECT id FROM detection_runs WHERE id LIKE ? || '%'", (args.reference,)).fetchone()[0]
    path = Path(args.report) if args.report else report_path(args.db, run_id)
    report = json.loads(path.read_text()) if path.exists() else None
    r = evaluate(conn, run_id, ref_id, report, args.exclude_camera, args.match_names)

    def pct(t):
        return f"{t[1] / max(1, t[0]):.2%} ({t[1]}/{t[0]})"

    print(f"box-frames matched to the reference: {r['all'][0]}, not assigned: {r['unassigned']}")
    print(f"agreement  all: {pct(r['all'])}   apart: {pct(r['apart'])}   not apart: {pct(r['close'])}")
    if report is not None:
        print(f"disagreeing boxes listed for checking: {pct((r['disagree'][0], r['disagree'][1]))}"
              f"   (apart: {pct((r['disagree_apart'][0], r['disagree_apart'][1]))})")
    print("per person, apart:")
    for p, t in r["per_person"].items():
        print(f"  {p}: {pct(t)}")
    print("per camera (apart / all):")
    for s, (a, b) in r["per_camera"].items():
        print(f"  {s[:8]}: {pct(a)} / {pct(b)}")
    print("stretches the reference sees in two or more cameras with no assigned box:")
    for p, spans in r["gaps"].items():
        print(f"  {p}: " + (", ".join(f"{a:.2f}-{b:.2f} s" for a, b in spans) or "none"))


if __name__ == "__main__":
    main()

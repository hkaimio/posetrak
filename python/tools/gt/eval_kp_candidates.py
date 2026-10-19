# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""eval_kp_candidates.py — score keypoint picks and tracker runs on the frozen evaluation set.

The evaluation set (ranges, hard windows, events, reference runs) is named in ``kp_eval_set.toml``;
its data lives outside the repository. Usage:

    export POSETRAK_KP_EVAL_ROOT=/path/to/kp_eval_set_v1
    export POSETRAK_KP_EVAL_DB=/path/to/session.db
    python python/tools/gt/eval_kp_candidates.py verify
    python python/tools/gt/eval_kp_candidates.py picks [--pick-sequence 1=SEQ --pick-sequence 2=SEQ ...]
    python python/tools/gt/eval_kp_candidates.py tracker --runs proposal_range3
    python python/tools/gt/eval_kp_candidates.py events --runs proposal

``picks`` reports, per range and subject, for the picks of the strongest maximum (or of a stored
pose sequence), the within-3%-of-box-height share, the same-candidate share, the share without a
pick and the false picks on not-visible labels, inside the hard windows and outside them.
``tracker`` is the label-fed comparison: body-joint distance of runs to the reference run.
``events`` is the failure-event comparison of two run groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kp_gt  # noqa: E402

SET_FILE = Path(__file__).with_name("kp_eval_set.toml")
# Tracker distances are scored from this time to the end of the range, the span the design documents' figures refer to.
TRACKER_FROM_S = 52.0


def load_set() -> dict:
    return tomllib.loads(SET_FILE.read_text())


def eval_root(args) -> Path:
    root = args.root or os.environ.get("POSETRAK_KP_EVAL_ROOT")
    if not root:
        sys.exit("give --root or set POSETRAK_KP_EVAL_ROOT (directory of the frozen evaluation set)")
    return Path(root)


def open_db(args) -> sqlite3.Connection:
    path = args.db or os.environ.get("POSETRAK_KP_EVAL_DB")
    if not path:
        sys.exit("give --db or set POSETRAK_KP_EVAL_DB (session database with the tracking runs)")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def cmd_verify(args) -> int:
    root = eval_root(args)
    sums = json.loads((root / load_set()["evaluation_set"]["checksums"]).read_text())
    bad = [name for name, digest in sums.items() if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    print(f"{len(sums) - len(bad)} of {len(sums)} files match their checksum")
    for name in bad:
        print("  differs:", name)
    return 1 if bad else 0


def cmd_picks(args) -> int:
    cfg, root = load_set(), eval_root(args)
    es = cfg["evaluation_set"]
    hard = [(w["t0"], w["t1"]) for w in cfg["hard_windows"]]
    sequences = dict(item.split("=", 1) for item in args.pick_sequence)
    conn = open_db(args) if sequences else None
    print(f"{'range/subject':16s} {'window':8s} {'n':>7s} {'within 3%':>10s} {'same cand.':>11s} {'no pick':>8s} {'false pick':>11s}")
    for name, rng in cfg["ranges"].items():
        for subject in rng["subjects"]:
            d = root / rng["dir"]
            cand = kp_gt.read_candidates(d / f"candidates_subject{subject}.npz")
            labels = kp_gt.read_labels(d / f"labels_subject{subject}.npz", d / f"units_subject{subject}.json", rng["units"])
            picks = (kp_gt.read_sequence_picks(conn, sequences[str(subject)], 0, cand.cameras)
                     if sequences else kp_gt.strongest_picks(cand))
            skip = rng["skip_cameras"].get(str(subject), [])
            outside = _outside_windows(rng, hard)
            for window, windows in (("hard", hard), ("outside", outside)):
                m = kp_gt.pick_metrics(picks, labels, cand, t0=rng["t0"], t1=rng["t1"], tolerance=es["tolerance_box_height"],
                                       same_px=es["same_candidate_px"], skip_cameras=skip, windows=windows)
                print(f"{name + '/' + str(subject):16s} {window:8s} {m['n_visible']:7d} {100 * m['within_tol']:9.1f}% "
                      f"{100 * m['same_candidate']:10.1f}% {100 * m['no_pick']:7.1f}% {100 * m['false_pick']:10.1f}%")
    return 0


def _outside_windows(rng, hard) -> list[tuple[float, float]]:
    """The spans of the range that lie outside the hard windows."""
    spans, start = [], rng["t0"]
    for a, b in sorted(hard):
        if b <= start or a >= rng["t1"]:
            continue
        if a > start:
            spans.append((start, a))
        start = max(start, b)
    if start < rng["t1"]:
        spans.append((start, rng["t1"]))
    return spans


def cmd_tracker(args) -> int:
    cfg, conn = load_set(), open_db(args)
    ranges = cfg["ranges"]["range3"]
    hard = [(w["t0"], w["t1"]) for w in cfg["hard_windows"]]
    ref = cfg["tracker"]["reference"]
    print(f"{'subject':8s} {'run':10s} {'window':8s} {'n':>8s} {'mean cm':>8s} {'median':>8s} {'p90':>8s}")
    for subject, run in cfg["tracker"][args.runs].items():
        for window, spans in (("range", [(TRACKER_FROM_S, ranges["t1"])]), ("hard", hard)):
            r = kp_gt.tracker_comparison(conn, run, ref[subject], spans)
            print(f"{subject:8s} {run:10s} {window:8s} {r['n']:8d} {r['mean']:8.1f} {r['median']:8.1f} {r['p90']:8.1f}")
    return 0


def cmd_events(args) -> int:
    cfg, root, conn = load_set(), eval_root(args), open_db(args)
    es = cfg["evaluation_set"]
    events = json.loads((root / es["events"]).read_text())
    base, other = cfg["tracker"]["baseline"], cfg["tracker"][args.runs]
    dist, dist_any = {}, {}
    print(f"{'event':14s} {'subject':8s} {'window':11s} {'part':6s} {'baseline vs ' + args.runs + ' (cm): part':>34s} {'any joint':>10s}")
    for e in events:
        s = str(e["subject"])
        if s not in other or e["t0"] < 43.5:
            continue
        dist[e["id"]] = kp_gt.event_distance(conn, base[s], other[s], e)
        dist_any[e["id"]] = kp_gt.event_distance(conn, base[s], other[s], {**e, "part": "whole"})
        print(f"{e['id']:14s} {s:8s} {e['t0']:4.1f}-{e['t1']:4.1f} {e['part']:6s} {dist[e['id']]:34.0f} {dist_any[e['id']]:10.0f}")
    for label, d in (("part", dist), ("any joint", dist_any)):
        flagged = kp_gt.events_flagged({k: v for k, v in d.items() if not np.isnan(v)}, es["failure_flag_cm"])
        print(f"flagged at {es['failure_flag_cm']:.0f} cm ({label}): {len(flagged)} of {len(d)} scorable events")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="directory of the frozen evaluation set (default $POSETRAK_KP_EVAL_ROOT)")
    ap.add_argument("--db", help="session database with the tracking runs (default $POSETRAK_KP_EVAL_DB)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    p = sub.add_parser("picks")
    p.add_argument("--pick-sequence", action="append", default=[], metavar="SUBJECT=SEQUENCE",
                   help="score the picks of this pose observation sequence (person 0) for the subject instead of the "
                        "strongest maxima; give one per subject")
    for name in ("tracker", "events"):
        q = sub.add_parser(name)
        q.add_argument("--runs", default="proposal_range3" if name == "tracker" else "proposal",
                       help="group of [tracker.*] runs in kp_eval_set.toml to score")
    args = ap.parse_args(argv)
    return {"verify": cmd_verify, "picks": cmd_picks, "tracker": cmd_tracker, "events": cmd_events}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

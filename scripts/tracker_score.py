# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Accuracy of the tracker baseline runs of the three-person trial, scored with the evaluation harness.

Reads ``<work-dir>/b/results.json`` (written by ``tracker_baseline.py --part b``) and scores every variant's
tracking runs against the label-fed reference run of ``python/tools/gt/kp_eval_set.toml``:

* mean, median and 90th percentile body-joint distance (cm) per person over the labelled range and over the hard
  windows. Only the persons that have a label-fed reference run (subjects 1 and 2) are scored, and only from the
  start of the tracker comparison (``eval_kp_candidates.TRACKER_FROM_S``) to the end of the labelled range;
* the failure events of the evaluation set that fall in that range, as the largest joint distance (cm) to the
  reference run, flagged at ``failure_flag_cm``;
* every failure event as the largest joint distance to the control variant (default ``single120``), which says how far
  a configuration moves the result where no label exists.

Part a (one person, no labels) has no labelled range and is not scored. The scoring reads a snapshot copy of the session
database, because a second connection on a database a tracker run writes to can kill that run; ``--no-snapshot`` reads
the database in place (nothing may be tracking then).

    python scripts/tracker_score.py --work-dir D:/mocap/tracker-s6 --eval-root D:/mocap/vanhaa-spike/kp_eval_set_v1
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tomllib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "python" / "tools" / "gt"))

import kp_gt  # noqa: E402

# Subject (segmentation mask label) of each person of the trial, in the order of tracker_baseline.PART_B["persons"].
SUBJECT_OF_PERSON = {0: 3, 1: 1, 2: 2}
TRACKER_FROM_S = 52.0  # as eval_kp_candidates.TRACKER_FROM_S


def score(conn: sqlite3.Connection, results: dict, eval_set: dict, events: list[dict], control: str) -> dict:
    rng = eval_set["ranges"]["range3"]
    flag = eval_set["evaluation_set"]["failure_flag_cm"]
    reference = eval_set["tracker"]["reference"]
    hard = [(w["t0"], w["t1"]) for w in eval_set["hard_windows"]]
    out: dict = {}
    for name, entry in results.items():
        runs = entry.get("run_ids") or []
        per_person, vs_reference, vs_control = {}, {}, {}
        for idx, run in enumerate(runs):
            subject = str(SUBJECT_OF_PERSON.get(idx))
            if subject in reference:
                per_person[subject] = {
                    "range": kp_gt.tracker_comparison(conn, run, reference[subject], [(TRACKER_FROM_S, rng["t1"])]),
                    "hard": kp_gt.tracker_comparison(conn, run, reference[subject], hard),
                }
            for e in events:
                if e["subject"] != SUBJECT_OF_PERSON.get(idx):
                    continue
                if subject in reference and e["t1"] > rng["t0"] and e["t0"] < rng["t1"]:
                    vs_reference[e["id"]] = kp_gt.event_distance(conn, run, reference[subject], e)
                if name != control and control in results and idx < len(results[control]["run_ids"]):
                    vs_control[e["id"]] = kp_gt.event_distance(conn, run, results[control]["run_ids"][idx], e)
        out[name] = {"per_person": per_person, "events_vs_reference": vs_reference, "events_vs_control": vs_control,
                     "flagged_vs_reference": kp_gt.events_flagged({k: v for k, v in vs_reference.items() if not np.isnan(v)}, flag)}
    return out


def write_markdown(path: Path, scores: dict, control: str, flag: float) -> None:
    lines = ["Body-joint distance (cm) to the label-fed reference run, subjects with a reference run only:", "",
             "| variant | subject | range mean | range median | range p90 | hard-window mean |", "|---|---|---|---|---|---|"]
    for name, s in scores.items():
        for subject, m in s["per_person"].items():
            lines.append(f"| {name} | {subject} | {m['range']['mean']:.1f} | {m['range']['median']:.1f} | {m['range']['p90']:.1f} | {m['hard']['mean']:.1f} |")
    ids = sorted({k for s in scores.values() for k in s["events_vs_reference"]})
    lines += ["", f"Failure events in the labelled range: largest joint distance (cm) to the reference run (flagged at {flag:.0f} cm):", "",
              "| variant | " + " | ".join(ids) + " | flagged |", "|---|" + "---|" * (len(ids) + 1)]
    for name, s in scores.items():
        lines.append(f"| {name} | " + " | ".join(f"{s['events_vs_reference'].get(i, float('nan')):.0f}" for i in ids) + f" | {len(s['flagged_vs_reference'])} |")
    ids = sorted({k for s in scores.values() for k in s["events_vs_control"]})
    lines += ["", f"Every failure event: largest joint distance (cm) to the {control} run:", "",
              "| variant | " + " | ".join(ids) + " |", "|---|" + "---|" * len(ids)]
    for name, s in scores.items():
        if s["events_vs_control"]:
            lines.append(f"| {name} | " + " | ".join(f"{s['events_vs_control'].get(i, float('nan')):.0f}" for i in ids) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--eval-root", required=True, type=Path, help="directory of the frozen evaluation set (events.json)")
    ap.add_argument("--control", default="single120")
    ap.add_argument("--only", help="comma-separated variant names")
    ap.add_argument("--no-snapshot", action="store_true", help="read the session database in place")
    args = ap.parse_args()
    work = args.work_dir / "b"
    db = work / "session.db"
    if not args.no_snapshot:
        snapshot = work / "score-snapshot.db"
        print(f"copying {db} -> {snapshot}", flush=True)
        shutil.copy(db, snapshot)
        db = snapshot
    eval_set = tomllib.loads((REPO / "python" / "tools" / "gt" / "kp_eval_set.toml").read_text())
    events = json.loads((args.eval_root / eval_set["evaluation_set"]["events"]).read_text())
    results = json.loads((work / "results.json").read_text())
    if args.only:
        keep = set(args.only.split(",")) | {args.control}
        results = {k: v for k, v in results.items() if k in keep}
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    scores = score(conn, results, eval_set, events, args.control)
    (work / "accuracy.json").write_text(json.dumps(scores, indent=1))
    write_markdown(work / "accuracy.md", scores, args.control, eval_set["evaluation_set"]["failure_flag_cm"])
    print((work / "accuracy.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())

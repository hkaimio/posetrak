# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tracker baseline for the SOMA-77 skeleton: single-stage against hierarchical, 120 against 60 Hz.

Measurement procedure of docs/roadmap/features/tracker-performance/tracker-performance-design.md §3. Each variant is one
tracker run, executed with the optimized binary in a copy of the session database; the original databases are
never written. The runs take tens of minutes each: run this script in a standalone shell, not as a background task
of a coding agent.

    python scripts/tracker_baseline.py --work-dir D:/mocap/tracker-baseline --part a
    python scripts/tracker_baseline.py --work-dir D:/mocap/tracker-baseline --part b

Variants with the suffix ``m33`` use the parent marker set of the bundled default skeleton: a copy of the skeleton
(a new skeleton version, ``parent_id`` = the original) without the 44 body-crop finger markers, keeping the index and
little-finger first joints of both hands, which sit in the ``main`` group (33 markers in ``main``).

Part a: the 85 s single-person sequence with the SOMA-77 skeleton (variants single120, hier120, single60, hier60,
        and main120, main60: hierarchical with the parent stage restricted to the ``main`` group, ``--joint-groups main``;
        hier* leave the parent stage on the whole skeleton).
Part b: the three-person trial 38-66 s that the failure events were judged on (variants hier120, single120, hier60, main120, main60),
        then the event table of each variant against hier120 (the configuration of the spike's runs).
Both parts write ``<work-dir>/<part>/results.json`` and ``results.md`` and can be resumed: a variant with a
result file is skipped (``--force`` reruns it). ``--only NAME[,NAME]`` selects variants, ``--label L`` keeps a result as ``NAME-L`` (a rerun with another binary), ``--binary`` names the tracker (a copy of optbuild's, so a rebuild cannot replace it mid-run); ``--end-time-s S`` cuts a run
short (a smoke test of the commands, not a measurement).

Peak resident memory is the largest working set of the tracker process, sampled twice a second. The per-step
timings are the columns of the tracker's ``tracking_stats.csv``. The error-state dimension is the length of
``cov_diag`` (float64 entries) of the stored results, and per stage the ``error_state_dim=`` lines the tracker
prints, which are kept with the time they appeared in ``log``.
"""
from __future__ import annotations

import argparse
import csv
import re
import json
import shutil
import sqlite3
import statistics
import sys
import threading
import time
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from posetrak.db.db import open_session  # noqa: E402
from posetrak.tracker.runner import PersonRunSpec, run_multi_person_tracker  # noqa: E402

DEFAULT_BINARY = REPO / "optbuild" / "cpp" / "cli" / ("posetrak-tracker.exe" if sys.platform == "win32" else "posetrak-tracker")
# Tracker output lines kept in the result: stage boundaries, state sizes and phases (with the time they appeared).
LOG_LINES = re.compile(r"error_state_dim|UKF initialized|Hierarchical stage|Will process|Initializing tracker|smooth|Results exported|tracking_run_id", re.I)
BASE_CONFIG = "hier-dualhand-c14786ab2fbe221e"  # hierarchical: tracker_config_stages rows for HandL and HandR

# Part a: single-person SOMA-77 sequence of the real capture, 152.8-238.0 s.
PART_A = {
    "source_db": "D:/3d/ic-tests/table/h/videos/mocap/test-2026-10-15.db",
    "persons": [("523103c6", "8ac54673", 0)],  # sequence, skeleton, person id (prefixes)
    "variants": {"single120": (False, 120.0), "hier120": (True, 120.0), "single60": (False, 60.0), "hier60": (True, 60.0),
                 "main120": (True, 120.0, ("main",)), "main60": (True, 60.0, ("main",)),
                 "main120m33": (True, 120.0, ("main",), "m33"), "single120m33": (False, 120.0, None, "m33")},
    "window": None,
}
# Part b: the three persons of the multi-person trial, strongest-peak sequences (decoder fix), 38.08-66.44 s.
PART_B = {
    "source_db": "D:/mocap/vanhaa-spike/session.db",
    "persons": [("a2cc3890", "b9c65ead", 0), ("3ddc1cd5", "2284a0cb", 0), ("361b8931", "ff9e2118", 0)],
    "variants": {"hier120": (True, 120.0), "single120": (False, 120.0), "hier60": (True, 60.0),
                 "main120": (True, 120.0, ("main",)), "main60": (True, 60.0, ("main",)),
                 "main120m33": (True, 120.0, ("main",), "m33"), "single120m33": (False, 120.0, None, "m33")},
    "window": (38.08, 66.44),
}


def full_id(conn: sqlite3.Connection, table: str, prefix: str) -> str:
    return conn.execute(f"SELECT id FROM {table} WHERE id LIKE ?", (prefix + "%",)).fetchone()[0]


# Body-crop finger markers kept in the parent marker set: the first joints of the index and little fingers.
KEPT_FINGER_MARKERS = {f"{side}HandIndex1" for side in ("Left", "Right")} | {f"{side}HandPinky1" for side in ("Left", "Right")}


def reduced_skeleton(conn: sqlite3.Connection, skeleton_id: str) -> str:
    """Store a copy of the skeleton without the body-crop finger markers; returns its id.

    Every ``<Side>Hand<Finger><n>`` marker is dropped except KEPT_FINGER_MARKERS, which also join the ``main`` group
    (their parent joint is the hand joint, which is in ``main``). The copy is a new skeleton version of the original.
    """
    import hashlib

    import yaml

    row = conn.execute("SELECT name, yaml_content FROM skeletons WHERE id = ?", (skeleton_id,)).fetchone()
    data = yaml.safe_load(row["yaml_content"])
    finger = re.compile(r"^(Left|Right)Hand(Thumb|Index|Middle|Ring|Pinky)\d?\w*$")
    drop = {m["name"] for m in data["markers"] if finger.match(m["name"])} - KEPT_FINGER_MARKERS
    data["markers"] = [m for m in data["markers"] if m["name"] not in drop]
    for group in data["groups"]:
        group["markers"] = [m for m in group["markers"] if m not in drop]
        if group["name"] == "main":
            group["markers"] += sorted(KEPT_FINGER_MARKERS - set(group["markers"]))
    text = yaml.safe_dump(data, sort_keys=False)
    new_id = hashlib.sha256(text.encode()).hexdigest()
    conn.execute(
        "INSERT OR REPLACE INTO skeletons (id, name, parent_id, yaml_content, created_at, notes) "
        "VALUES (?, ?, ?, ?, datetime('now'), ?)",
        (new_id, row["name"] + " - parent marker set (33)", skeleton_id, text, f"{len(drop)} body-crop finger markers removed"))
    conn.commit()
    return new_id


def make_variant(conn: sqlite3.Connection, name: str, hierarchical: bool, fps: float) -> str:
    """Copy BASE_CONFIG as a new tracker_configs row; stage rows (hierarchical mode) only when asked for."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(tracker_configs)") if r[1] not in ("id", "name", "parent_id", "created_at", "is_named", "tracker_fps")]
    new_id = f"baseline-{name}"
    conn.execute("DELETE FROM tracker_config_stages WHERE tracker_config_id = ?", (new_id,))
    conn.execute("DELETE FROM tracker_configs WHERE id = ?", (new_id,))
    conn.execute(
        f"INSERT INTO tracker_configs (id, name, parent_id, created_at, is_named, tracker_fps, {', '.join(cols)}) "
        f"SELECT ?, ?, id, created_at, 0, ?, {', '.join(cols)} FROM tracker_configs WHERE id = ?",
        (new_id, f"baseline {name}", fps, BASE_CONFIG))
    if hierarchical:
        stage_cols = [r[1] for r in conn.execute("PRAGMA table_info(tracker_config_stages)") if r[1] != "tracker_config_id"]
        conn.execute(
            f"INSERT INTO tracker_config_stages (tracker_config_id, {', '.join(stage_cols)}) "
            f"SELECT ?, {', '.join(stage_cols)} FROM tracker_config_stages WHERE tracker_config_id = ?", (new_id, BASE_CONFIG))
    conn.commit()
    return new_id


class PeakMemory(threading.Thread):
    """Largest summed working set (bytes) of this process' child processes while running."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = 0
        self.timeline: list[tuple[float, float]] = []  # (seconds since start, resident GB), every 5 s
        self._stop_flag = threading.Event()

    def run(self) -> None:
        me = psutil.Process()
        started = time.perf_counter()
        next_sample = 0.0
        while not self._stop_flag.wait(0.5):
            try:
                total = sum(p.memory_info().rss for p in me.children(recursive=True))
            except psutil.Error:
                continue
            self.peak = max(self.peak, total)
            if time.perf_counter() - started >= next_sample:
                self.timeline.append((round(time.perf_counter() - started, 1), round(total / 2**30, 2)))
                next_sample += 5.0

    def stop(self) -> int:
        self._stop_flag.set()
        self.join()
        return self.peak


def step_timings(out_dir: Path) -> dict:
    """Per person: mean and max of every timing column of tracking_stats.csv (the *_ms columns, in ms per step, and the observations per step)."""
    out = {}
    for path in sorted(out_dir.glob("person_*/tracking_stats.csv")):
        rows = list(csv.DictReader(open(path, newline="")))
        stats = {"n_rows": len(rows)}
        for key in rows[0] if rows else []:
            try:
                values = [float(r[key]) for r in rows if r[key] not in ("", None)]
            except ValueError:
                continue
            if values and (key.endswith("_ms") or key == "num_observations"):
                stats[key] = {"mean": statistics.fmean(values), "max": max(values)}
        out[path.parent.name] = stats
    return out


def run_stats(conn: sqlite3.Connection, run_id: str) -> dict:
    steps = conn.execute("SELECT COUNT(*) FROM tracking_results WHERE run_id = ? AND is_smoothed = 0", (run_id,)).fetchone()[0]
    lost = conn.execute("SELECT COUNT(*) FROM tracking_results WHERE run_id = ? AND is_smoothed = 0 AND tracking_lost = 1", (run_id,)).fetchone()[0]
    dims = conn.execute(
        "SELECT person_id, MIN(length(cov_diag)) / 8, MAX(length(cov_diag)) / 8 FROM tracking_results WHERE run_id = ? GROUP BY person_id", (run_id,)).fetchall()
    return {"steps": steps, "lost_steps": lost, "error_state_dim_by_person": {str(p): [lo, hi] for p, lo, hi in dims}}


def conn_window(db: Path, sequence_id: str) -> tuple[float, float]:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT time_start_s, time_end_s FROM pose_observation_sequences WHERE id = ?", (sequence_id,)).fetchone()


def run_variant(work: Path, db: Path, part: dict, name: str, end_time: float | None, binary: Path, key: str | None = None) -> dict:
    hierarchical, fps, *rest = part["variants"][name]
    groups = list(rest[0]) if rest and rest[0] else None
    reduced = len(rest) > 1 and rest[1] == "m33"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    config_id = make_variant(conn, name, hierarchical, fps)
    specs = [PersonRunSpec(full_id(conn, "pose_observation_sequences", seq),
                           reduced_skeleton(conn, full_id(conn, "skeletons", skel)) if reduced else full_id(conn, "skeletons", skel), config_id, pid)
             for seq, skel, pid in part["persons"]]
    conn.close()
    key = key or name
    out_dir = work / key
    shutil.rmtree(out_dir, ignore_errors=True)
    start, end = part["window"] or conn_window(db, specs[0].sequence_id)
    if end_time is not None:
        end = start + end_time
    print(f"[{name}] {binary} track ... --tracker-config {config_id} persons={len(specs)} window={start}-{end}", flush=True)
    memory = PeakMemory()
    memory.start()
    started = time.perf_counter()
    log: list[tuple[float, str]] = []

    def keep(line: str) -> None:
        if LOG_LINES.search(line):
            log.append((round(time.perf_counter() - started, 1), line[:160]))

    result = run_multi_person_tracker(db, specs, out_dir, binary_path=binary, start_time=start, end_time=end, joint_groups=groups, on_progress=keep)
    wall = time.perf_counter() - started
    peak = memory.stop()
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    run_ids = [r for r in result.run_ids if r]
    entry = {"variant": key, "hierarchical": hierarchical, "parent_groups": groups, "tracker_fps": fps, "config_id": config_id, "exit_code": result.exit_code,
             "wall_s": wall, "peak_rss_gb": peak / 2**30, "rss_timeline_gb": memory.timeline, "run_ids": run_ids, "binary": str(binary),
             "step_timings": step_timings(out_dir), "log": log, "runs": [run_stats(conn, r) | {"run_id": r} for r in run_ids]}
    conn.close()
    return entry


def event_table(db: Path, results: dict, control: str, events_path: Path | None) -> dict:
    """Largest joint distance (cm) per failure event between each variant's run and the control variant's run."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("kp_gt", REPO / "python" / "tools" / "gt" / "kp_gt.py")
    kp_gt = importlib.util.module_from_spec(spec)
    sys.modules["kp_gt"] = kp_gt
    spec.loader.exec_module(kp_gt)
    if events_path is None or control not in results:
        return {}
    events = json.loads(events_path.read_text())
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    # subject label 3, 1, 2 in the event list = persons in the order of PART_B["persons"] (mask labels 3, 1, 2)
    person_of_subject = {3: 0, 1: 1, 2: 2}
    table = {}
    for variant, entry in results.items():
        if variant == control or not entry.get("run_ids"):
            continue
        per_event = {}
        for e in events:
            idx = person_of_subject.get(e["subject"])
            if idx is None or idx >= len(entry["run_ids"]) or e["t0"] < 38.2:
                continue
            per_event[e["id"]] = kp_gt.event_distance(conn, entry["run_ids"][idx], results[control]["run_ids"][idx], e)
        table[variant] = per_event
    return table


def write_markdown(path: Path, results: dict, events: dict) -> None:
    lines = ["| variant | stages | Hz | wall s | peak RSS GB | steps | lost | error-state dim | mean ms/step (update) |", "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        runs = r.get("runs", [])
        steps = "+".join(str(x["steps"]) for x in runs)
        lost = "+".join(str(x["lost_steps"]) for x in runs)
        dim = ";".join(f"{k}:{v[0]}" if v[0] == v[1] else f"{k}:{v[0]}-{v[1]}" for x in runs[:1] for k, v in x["error_state_dim_by_person"].items())
        upd = r.get("step_timings", {}).get("person_0", {}).get("update_ms", {}).get("mean")
        lines.append(f"| {name} | {('hierarchical, parent ' + '+'.join(r['parent_groups'])) if r.get('parent_groups') else 'hierarchical' if r['hierarchical'] else 'single'} | {r['tracker_fps']:.0f} | {r['wall_s']:.0f} | "
                     f"{r['peak_rss_gb']:.1f} | {steps} | {lost} | {dim} | {upd if upd is None else round(upd, 2)} |")
    for variant, per_event in events.items():
        lines += ["", f"Failure events, largest joint distance (cm) of {variant} to the control run:", "", "| event | cm |", "|---|---|"]
        lines += [f"| {k} | {v:.0f} |" for k, v in per_event.items()]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--part", required=True, choices=["a", "b"])
    ap.add_argument("--only", help="comma-separated variant names")
    ap.add_argument("--end-time-s", type=float, help="cut each run to this many seconds (smoke test)")
    ap.add_argument("--binary", type=Path, default=DEFAULT_BINARY,
                    help="posetrak-tracker executable (a copy of optbuild's, so that a rebuild does not replace it mid-run)")
    ap.add_argument("--label", help="keep the result as <variant>-<label>, so a rerun with another binary does not replace the earlier one")
    ap.add_argument("--force", action="store_true", help="rerun variants that already have a result")
    ap.add_argument("--events", type=Path, help="part b: events.json of the evaluation set (enables the event table)")
    args = ap.parse_args()
    part = PART_A if args.part == "a" else PART_B
    work = args.work_dir / args.part
    work.mkdir(parents=True, exist_ok=True)
    db = work / "session.db"
    if not db.exists():
        print(f"copying {part['source_db']} -> {db}", flush=True)
        shutil.copy(part["source_db"], db)
    open_session(db).close()  # bring a copy made by an older version to the current schema
    results_path = work / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    names = args.only.split(",") if args.only else list(part["variants"])
    for name in names:
        key = f"{name}-{args.label}" if args.label else name
        if key in results and not args.force:
            print(f"[{key}] already done, skipping", flush=True)
            continue
        results[key] = run_variant(work, db, part, name, args.end_time_s, args.binary, key)
        results_path.write_text(json.dumps(results, indent=1))
        print(f"[{key}] exit {results[key]['exit_code']}  {results[key]['wall_s']:.0f} s  peak {results[key]['peak_rss_gb']:.1f} GB", flush=True)
    events = event_table(db, results, "hier120", args.events) if args.part == "b" else {}
    write_markdown(work / "results.md", results, events)
    print((work / "results.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())

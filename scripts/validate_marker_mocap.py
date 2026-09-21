#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Regression check for marker-based motion capture against recorded baselines.

Re-runs the tracker on real captures and compares the outcome with a recorded
tracking run and with expected headline numbers. It is meant to be run on
demand before and after refactoring the marker tracking pipeline, not in CI:
it needs real session databases and an optimized tracker build.

Cases live in a JSON file next to the data, not in the repository, because
they name session databases and run IDs that exist only on the machine that
holds the captures. See ``scripts/validate_marker_mocap.example.json`` for the
format. Point the script at it with ``--cases`` or the environment variable
``POSETRAK_VALIDATION_CASES``. When neither is set, or the file or a session
database used by a selected case is missing, the script says so and exits 0.

Session databases are never modified. Each one is copied (with SQLite's backup
API, so a live WAL is included) into a work directory and migrated to the
current schema there; all tracking runs are written to the copy.

The tracker binary is always the repository's ``optbuild`` build, never the
one ``posetrak.tracker.runner.default_binary_path()`` prefers, which is a
possibly stale install under ``~/.posetrak``.

Usage:
    python scripts/validate_marker_mocap.py --cases /path/to/cases.json
    python scripts/validate_marker_mocap.py --cases cases.json --only pen pad
    python scripts/validate_marker_mocap.py --cases cases.json --reuse-copies
    python scripts/validate_marker_mocap.py --cases cases.json --reuse-copies --reuse-detection

Case fields (all but ``name``, ``session`` and ``baseline_run`` are optional):

    name, session        case label; key into the ``sessions`` map
    baseline_run         prefix of a recorded tracking run in that session. Its
                         sequence, skeleton, tracker config and person index
                         are the recipe for the re-run, and its results are the
                         reference for the comparison
    start_time, end_time restrict the re-run to a time range (seconds)
    smooth               RTS smoothing, default true
    seed_from_baseline   start from the baseline run's first tracked root
                         position (for dots-only subjects, which cannot
                         initialise from observations)
    redetect             {"detection_run": "<prefix>"}: repeat that recorded
                         marker detection run with the current detection
                         pipeline (same object, cameras, time range and dot
                         settings), finalise it into a new sequence and track
                         that instead of the baseline's sequence. Needed when
                         stored detections are in an older blob layout, and
                         it exercises detection as well as tracking. The new
                         sequence is tagged ``validation:<case name>`` so a
                         later run with ``--reuse-detection`` can skip the
                         detection and track it again
    subjects             instead of baseline_run: a list of {"name", "baseline_run",
                         optional "seed_from_baseline"} tracked together. Each
                         subject is also tracked alone over the same
                         start_time/end_time window, the joint run must match
                         the solo runs, and no observed pixel may be used by
                         two subjects at the same time (a physical candidate
                         claimed twice). Needs start_time and end_time so the
                         subjects share one time range
    duplicate_pool       {"donor": <subject>, "into": <subject>, "cameras":
                         [labels]}: repeat the joint run with a copy of the
                         donor's dot candidates on those cameras added to the
                         receiving subject's sequence, and check again that no
                         candidate is claimed twice
    expect               numbers the re-run must meet:
        tracked_pct          percentage of steps tracked
        lost_max             most steps lost
        init_rms_mm          rigid-body initialisation residual
        nis_mean             mean NIS/dof (default: the baseline run's)
        nis_close_to_run     prefix of another recorded run whose mean NIS/dof
                             the re-run must match (for example a markerless run)
        reproj_median_px     {camera: [low, high]} range of the per-camera
                             median reprojection error
    tolerance            overrides for the checks: tracked_pct (points),
                         init_rms_mm, nis_rel, reproj_px, reproj_rel
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

DEFAULT_TOLERANCE = {
    "tracked_pct": 0.5,   # percentage points
    "init_rms_mm": 0.3,
    "nis_rel": 0.10,      # relative to the reference mean NIS/dof
    "reproj_px": 3.0,     # per-camera median reprojection, absolute floor
    "reproj_rel": 0.20,   # ... or this fraction of the reference, whichever is larger
}


def optbuild_binary() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return REPO_ROOT / "optbuild" / "cpp" / "cli" / f"posetrak-tracker{suffix}"


def open_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def make_working_copy(source: Path, dest: Path) -> None:
    """Copy *source* to *dest* with the backup API, then migrate the copy."""
    from posetrak.db.db import open_session

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst, pages=4096)
    finally:
        dst.close()
        src.close()
    open_session(dest).close()  # migrates the copy to the current schema


def run_metrics(conn: sqlite3.Connection, run_id: str) -> dict:
    """Tracked steps, mean NIS/dof and per-camera median reprojection of a run.

    Reprojection is measured on position-mode observations the filter used
    (``obs_blob`` fields ``[actual_x, actual_y, pred_x, pred_y, mahal, used,
    is_outlier, mode]``); velocity and pair-difference observations hold pixel
    deltas and are skipped.
    """
    run = conn.execute("SELECT * FROM tracking_runs WHERE id LIKE ?", (run_id + "%",)).fetchone()
    if run is None:
        raise KeyError(f"no tracking run matching {run_id!r}")
    rid = run["id"]
    rows = conn.execute(
        "SELECT tracking_lost, nis_value, nis_dof FROM tracking_results WHERE run_id = ? AND is_smoothed = 0",
        (rid,),
    ).fetchall()
    tracked_rows = [r for r in rows if not r["tracking_lost"]]
    nis = [r["nis_value"] / r["nis_dof"] for r in tracked_rows if r["nis_dof"]]
    cameras = json.loads(run["active_camera_ids"])
    per_camera: dict[str, list[np.ndarray]] = {}
    for blob in conn.execute("SELECT obs_blob FROM tracking_obs_results WHERE run_id = ?", (rid,)):
        arr = np.frombuffer(bytes(blob["obs_blob"]), dtype=np.float32)
        if len(arr) % (len(cameras) * 8):
            continue
        arr = arr.reshape(len(cameras), -1, 8)
        for ci, name in enumerate(cameras):
            block = arr[ci]
            ok = (block[:, 5] == 1) & np.isfinite(block[:, 0]) & (block[:, 7] == 0)
            per_camera.setdefault(name, []).append(
                np.hypot(block[ok, 0] - block[ok, 2], block[ok, 1] - block[ok, 3])
            )
    medians = {
        name: float(np.median(np.concatenate(parts)))
        for name, parts in per_camera.items()
        if sum(len(p) for p in parts)
    }
    return {
        "run_id": rid,
        "steps_recorded": len(rows),
        "steps_tracked": len(tracked_rows),
        "nis_mean": float(np.mean(nis)) if nis else None,
        "reproj_median_px": medians,
    }


def first_root_position(conn: sqlite3.Connection, run_id: str) -> list[float]:
    row = conn.execute(
        "SELECT state FROM tracking_results WHERE run_id = ? AND is_smoothed = 0 ORDER BY tracker_step LIMIT 1",
        (run_id,),
    ).fetchone()
    return [float(v) for v in np.frombuffer(bytes(row["state"]), dtype=np.float64)[:3]]


def parse_tracker_output(text: str) -> dict:
    out: dict = {}
    m = re.search(r"tracking_run_id:\s*(\S+)", text)
    out["run_id"] = m.group(1) if m else None
    m = re.search(r"Tracked:\s*(\d+)/(\d+) steps \(([\d.]+)%\)", text)
    if m:
        out["tracked"], out["total"], out["tracked_pct"] = int(m.group(1)), int(m.group(2)), float(m.group(3))
    m = re.search(r"Lost:\s*(\d+) steps", text)
    if m:
        out["lost"] = int(m.group(1))
    m = re.search(r"Rigid-body init:.*RMS residual = ([\d.]+) m", text)
    if m:
        out["init_rms_mm"] = float(m.group(1)) * 1000.0
    return out


def run_tracker(binary: Path, db: Path, recipe: dict, out_dir: Path, log_path: Path) -> tuple[int, dict, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [
        str(binary), "track",
        "--session-db", str(db),
        "--sequence", recipe["sequence"],
        "--skeleton", recipe["skeleton"],
        "--tracker-config", recipe["config"],
        "--person-id", str(recipe["person_id"]),
        "--output-dir", str(out_dir),
    ]
    if recipe.get("start_time") is not None:
        args += ["--start-time", str(recipe["start_time"])]
    if recipe.get("end_time") is not None:
        args += ["--end-time", str(recipe["end_time"])]
    if recipe.get("smooth", True):
        args.append("--smooth")
    if recipe.get("seed_position"):
        args += ["--seed-position", *[str(v) for v in recipe["seed_position"]]]
    started = time.time()
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - started
    text = proc.stdout + proc.stderr
    log_path.write_text(text.replace("\r", "\n"), encoding="utf-8")
    return proc.returncode, parse_tracker_output(text.replace("\r", "\n")), elapsed


def run_joint_tracker(binary: Path, db: Path, recipes: list[dict],
                      window: tuple[float, float], out_dir: Path, log_path: Path) -> tuple[int, list[str], float]:
    """One tracker process over several subjects; returns exit code, run ids in subject order, seconds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [str(binary), "track", "--session-db", str(db), "--output-dir", str(out_dir)]
    for r in recipes:
        args += ["--person", r["sequence"], r["skeleton"], r["config"], str(r["person_id"])]
    args += ["--start-time", str(window[0]), "--end-time", str(window[1]), "--smooth"]
    for index, r in enumerate(recipes):
        if r.get("seed_position"):
            args += ["--subject-seed", str(index), *[str(v) for v in r["seed_position"]]]
    started = time.time()
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - started
    text = (proc.stdout + proc.stderr).replace("\r", "\n")
    log_path.write_text(text, encoding="utf-8")
    return proc.returncode, re.findall(r"tracking_run_id:\s*(\S+)", text), elapsed


def used_pixels(conn: sqlite3.Connection, run_id: str) -> dict[tuple[float, str], set[tuple[float, float]]]:
    """Observed pixels a run used, keyed by (timestamp, camera): position-mode observations only."""
    run = conn.execute("SELECT active_camera_ids FROM tracking_runs WHERE id = ?", (run_id,)).fetchone()
    cameras = json.loads(run["active_camera_ids"])
    timestamps = {
        r["tracker_step"]: round(r["timestamp_s"], 5)
        for r in conn.execute(
            "SELECT tracker_step, timestamp_s FROM tracking_results WHERE run_id = ? AND is_smoothed = 0", (run_id,))
    }
    out: dict[tuple[float, str], set[tuple[float, float]]] = {}
    for row in conn.execute("SELECT tracker_step, obs_blob FROM tracking_obs_results WHERE run_id = ?", (run_id,)):
        arr = np.frombuffer(bytes(row["obs_blob"]), dtype=np.float32)
        if row["tracker_step"] not in timestamps or len(arr) % (len(cameras) * 8):
            continue
        arr = arr.reshape(len(cameras), -1, 8)
        for ci, camera in enumerate(cameras):
            block = arr[ci]
            ok = (block[:, 5] == 1) & np.isfinite(block[:, 0]) & (block[:, 7] == 0)
            if ok.any():
                out.setdefault((timestamps[row["tracker_step"]], camera), set()).update(
                    (round(float(x), 3), round(float(y), 3)) for x, y in block[ok, :2])
    return out


def double_claims(conn: sqlite3.Connection, run_ids: list[str]) -> int:
    """Number of (timestamp, camera, pixel) uses shared by two or more of *run_ids*."""
    seen: dict[tuple, int] = {}
    for run_id in run_ids:
        for key, pixels in used_pixels(conn, run_id).items():
            for pixel in pixels:
                seen[key + pixel] = seen.get(key + pixel, 0) + 1
    return sum(1 for n in seen.values() if n > 1)


def clone_sequence_with_donor_dots(db: Path, target_sequence: str, donor_sequence: str,
                                   camera_labels: list[str], window: tuple[float, float], tag: str) -> str:
    """Copy *target_sequence* and add *donor_sequence*'s dot rows on *camera_labels* to the copy."""
    import uuid

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        new_id = str(uuid.uuid4())
        columns = [r["name"] for r in conn.execute("PRAGMA table_info(pose_observation_sequences)")]
        row = conn.execute("SELECT * FROM pose_observation_sequences WHERE id = ?", (target_sequence,)).fetchone()
        values = [new_id if c == "id" else (tag if c == "notes" else row[c]) for c in columns]
        conn.execute(
            f"INSERT INTO pose_observation_sequences ({','.join(columns)}) VALUES ({','.join('?' * len(columns))})", values)
        conn.execute(
            "INSERT INTO pose_observations (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id,"
            " source, detection_run_id, kp_blob, noise_scale)"
            " SELECT ?, camera_instance_id, video_frame, timestamp_s, person_id, source, detection_run_id,"
            " kp_blob, noise_scale FROM pose_observations WHERE sequence_id = ?", (new_id, target_sequence))
        conn.execute(
            "INSERT INTO pose_sequence_keypoints (sequence_id, keypoint_idx, name, source)"
            " SELECT ?, keypoint_idx, name, source FROM pose_sequence_keypoints WHERE sequence_id = ?",
            (new_id, target_sequence))
        labels = ",".join("?" * len(camera_labels))
        conn.execute(
            "INSERT INTO pose_observations (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id,"
            " source, detection_run_id, kp_blob, noise_scale)"
            " SELECT ?, po.camera_instance_id, po.video_frame, po.timestamp_s, po.person_id, po.source,"
            " po.detection_run_id, po.kp_blob, po.noise_scale FROM pose_observations po"
            " JOIN camera_instances ci ON ci.id = po.camera_instance_id"
            f" WHERE po.sequence_id = ? AND po.source = 'dots' AND ci.label IN ({labels})"
            " AND po.timestamp_s BETWEEN ? AND ?",
            (new_id, donor_sequence, *camera_labels, *window))
        conn.commit()
        return new_id
    finally:
        conn.close()


def run_joint_case(case: dict, db: Path, binary: Path, work: Path) -> tuple[list[tuple[str, bool, str]], dict]:
    """Solo runs, the joint run, and the shared-pool checks of a multi-subject case."""
    name = case["name"]
    window = (case["start_time"], case["end_time"])
    tol = {**DEFAULT_TOLERANCE, **case.get("tolerance", {})}
    checks: list[tuple[str, bool, str]] = []
    conn = open_readonly(db)
    recipes, subject_names = [], []
    for subject in case["subjects"]:
        base = conn.execute("SELECT * FROM tracking_runs WHERE id LIKE ?", (subject["baseline_run"] + "%",)).fetchone()
        recipes.append({"sequence": base["observation_sequence_id"], "skeleton": base["skeleton_id"],
                        "config": base["tracker_config_id"], "person_id": 0,
                        "start_time": window[0], "end_time": window[1], "smooth": True,
                        "seed_position": first_root_position(conn, base["id"]) if subject.get("seed_from_baseline") else None})
        subject_names.append(subject["name"])
    conn.close()

    solo = {}
    for subject, recipe in zip(subject_names, recipes):
        code, parsed, _ = run_tracker(binary, db, recipe, work / "out" / f"{name}-{subject}-solo",
                                      work / f"{name}-{subject}-solo.log")
        checks.append((f"solo run {subject}", code == 0 and bool(parsed.get("run_id")), f"exit {code}"))
        solo[subject] = parsed.get("run_id")

    def joint_and_checks(label: str, joint_recipes: list[dict]) -> list[str]:
        code, run_ids, seconds = run_joint_tracker(binary, db, joint_recipes, window,
                                                   work / "out" / f"{name}-{label}", work / f"{name}-{label}.log")
        ok = code == 0 and len(run_ids) == len(joint_recipes)
        checks.append((f"{label} run", ok, f"exit {code}, {len(run_ids)} runs in {seconds:.0f}s"))
        if not ok:
            return []
        conn = open_readonly(db)
        claims = double_claims(conn, run_ids)
        checks.append((f"{label}: candidates used by two subjects", claims == 0, str(claims)))
        conn.close()
        return run_ids

    joint_ids = joint_and_checks("joint", recipes)
    if joint_ids and all(solo.values()):
        conn = open_readonly(db)
        for subject, joint_id in zip(subject_names, joint_ids):
            j, s = run_metrics(conn, joint_id), run_metrics(conn, solo[subject])
            allowed = max(2, 0.02 * s["steps_tracked"])
            checks.append((f"{subject}: tracked steps joint vs solo", abs(j["steps_tracked"] - s["steps_tracked"]) <= allowed,
                           f"{j['steps_tracked']} vs {s['steps_tracked']}"))
            if j["nis_mean"] is not None and s["nis_mean"] is not None:
                checks.append((f"{subject}: mean NIS/dof joint vs solo",
                               abs(j["nis_mean"] - s["nis_mean"]) <= tol["nis_rel"] * s["nis_mean"],
                               f"{j['nis_mean']:.3f} vs {s['nis_mean']:.3f}"))
            for camera, value in sorted(j["reproj_median_px"].items()):
                ref = s["reproj_median_px"].get(camera)
                if ref is not None:
                    allowed_px = max(tol["reproj_px"], tol["reproj_rel"] * ref)
                    checks.append((f"{subject}: reproj median {camera} joint vs solo",
                                   abs(value - ref) <= allowed_px, f"{value:.1f}px vs {ref:.1f}px (+/- {allowed_px:.1f})"))
        conn.close()

    dup = case.get("duplicate_pool")
    if dup:
        donor = subject_names.index(dup["donor"])
        into = subject_names.index(dup["into"])
        variant = clone_sequence_with_donor_dots(db, recipes[into]["sequence"], recipes[donor]["sequence"],
                                                 dup["cameras"], window, f"validation:{name}:duplicate")
        variant_recipes = [dict(r) for r in recipes]
        variant_recipes[into]["sequence"] = variant
        joint_and_checks("duplicate-pool", variant_recipes)
    return checks, {"joint": joint_ids, "solo": solo}


def redetect(db: Path, detection_run: str, case_name: str) -> tuple[str, float]:
    """Repeat a recorded marker detection run in the database copy *db*.

    Returns the new observation sequence id and the detection wall time.
    """
    from app.pose.finalise import finalise_object_to_db
    from posetrak.detection.marker_pipeline import load_pipeline_for_capture_object

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        run = conn.execute("SELECT * FROM detection_runs WHERE id LIKE ?", (detection_run + "%",)).fetchone()
        if run is None or run["capture_object_id"] is None:
            raise KeyError(f"no object-bound detection run matching {detection_run!r}")
        config = json.loads(run["config_json"])
        dots = config.get("dot_detection") or {}
        pipeline = load_pipeline_for_capture_object(
            conn, run["capture_object_id"], run["sync_config_id"], run["time_start_s"], run["time_end_s"],
            min_marker_perimeter_rate=config.get("min_marker_perimeter_rate"),
            frame_step=config.get("frame_step", 1),
            detect_dots_for_cameras=set(dots["cameras"]) if dots.get("cameras") else None,
            dot_bg_subtract=dots.get("bg_subtract", False),
            dot_threshold=dots.get("threshold", 235),
            dot_threshold_by_camera=dots.get("threshold_by_camera") or None,
            dot_background_mode=dots.get("background_mode", "subtract"),
            dot_blacklist_frac=dots.get("blacklist_frac", 0.7),
            dot_blacklist_frac_by_camera=dots.get("blacklist_frac_by_camera") or None,
            dot_blacklist_radius_px=dots.get("blacklist_radius_px", 6),
            dot_max_saturation=dots.get("max_saturation", 255.0),
            dot_max_saturation_by_camera=dots.get("max_saturation_by_camera") or None,
            dot_bg_sample_count=dots.get("bg_sample_count", 40),
        )
        started = time.time()
        result = pipeline.run_parallel(on_camera_done=lambda done, total: print(f"  detection: camera {done}/{total}", flush=True))
        elapsed = time.time() - started
        if result.status != "complete":
            raise RuntimeError(f"detection run finished with status {result.status!r}")
        return finalise_object_to_db(conn, result.detection_run_id, notes=f"validation:{case_name}"), elapsed
    finally:
        conn.close()


def find_validation_sequence(db: Path, case_name: str) -> str | None:
    conn = open_readonly(db)
    try:
        row = conn.execute(
            "SELECT id FROM pose_observation_sequences WHERE notes = ? ORDER BY rowid DESC LIMIT 1",
            (f"validation:{case_name}",),
        ).fetchone()
        return row["id"] if row else None
    finally:
        conn.close()


def evaluate(case: dict, parsed: dict, new: dict, baseline: dict, other: dict | None) -> list[tuple[str, bool, str]]:
    expect = case.get("expect", {})
    tol = {**DEFAULT_TOLERANCE, **case.get("tolerance", {})}
    checks: list[tuple[str, bool, str]] = []

    if "tracked_pct" in expect:
        got = parsed.get("tracked_pct")
        ok = got is not None and abs(got - expect["tracked_pct"]) <= tol["tracked_pct"]
        checks.append(("tracked %", ok, f"{got} (expect {expect['tracked_pct']} +/- {tol['tracked_pct']}; "
                                        f"{parsed.get('tracked')}/{parsed.get('total')})"))
    if "lost_max" in expect:
        got = parsed.get("lost")
        checks.append(("lost steps", got is not None and got <= expect["lost_max"], f"{got} (max {expect['lost_max']})"))
    if "init_rms_mm" in expect:
        got = parsed.get("init_rms_mm")
        ok = got is not None and abs(got - expect["init_rms_mm"]) <= tol["init_rms_mm"]
        checks.append(("rigid init RMS mm", ok, f"{got if got is None else round(got, 2)} "
                                                f"(expect {expect['init_rms_mm']} +/- {tol['init_rms_mm']})"))

    ref_nis = expect.get("nis_mean", baseline["nis_mean"])
    if new["nis_mean"] is not None and ref_nis is not None:
        ok = abs(new["nis_mean"] - ref_nis) <= tol["nis_rel"] * ref_nis
        checks.append(("mean NIS/dof", ok, f"{new['nis_mean']:.3f} (reference {ref_nis:.3f} +/- {tol['nis_rel']:.0%})"))
    if other is not None and other["nis_mean"] is not None and new["nis_mean"] is not None:
        ok = abs(new["nis_mean"] - other["nis_mean"]) <= tol["nis_rel"] * other["nis_mean"]
        checks.append((f"NIS/dof vs run {other['run_id'][:8]}", ok,
                       f"{new['nis_mean']:.3f} vs {other['nis_mean']:.3f} (+/- {tol['nis_rel']:.0%})"))

    ranges = expect.get("reproj_median_px")
    for cam, value in sorted(new["reproj_median_px"].items()):
        ref = baseline["reproj_median_px"].get(cam)
        if ranges is not None:
            if cam not in ranges:
                continue
            lo, hi = ranges[cam]
            ok = lo <= value <= hi
            detail = f"{value:.1f}px (expect {lo}-{hi})"
        elif ref is not None:
            allowed = max(tol["reproj_px"], tol["reproj_rel"] * ref)
            ok = abs(value - ref) <= allowed
            detail = f"{value:.1f}px (baseline run {ref:.1f} +/- {allowed:.1f})"
        else:
            continue
        checks.append((f"reproj median {cam}", ok, detail))
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default=os.environ.get("POSETRAK_VALIDATION_CASES"),
                    help="cases JSON (default: $POSETRAK_VALIDATION_CASES)")
    ap.add_argument("--work-dir", help="directory for database copies and tracker output "
                                       "(default: <cases dir>/validation-work)")
    ap.add_argument("--only", nargs="*", help="run only these case names")
    ap.add_argument("--reuse-copies", action="store_true", help="reuse database copies already in the work dir")
    ap.add_argument("--reuse-detection", action="store_true",
                    help="for cases with 'redetect': track the sequence an earlier run made in the "
                         "reused copy instead of detecting again")
    ap.add_argument("--binary", help="tracker binary (default: this repository's optbuild build)")
    ap.add_argument("--report", help="write a JSON report here")
    args = ap.parse_args()

    if not args.cases or not Path(args.cases).is_file():
        print("SKIP: no cases file (use --cases or POSETRAK_VALIDATION_CASES)")
        return 0
    cases_path = Path(args.cases)
    spec = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = [c for c in spec["cases"] if not args.only or c["name"] in args.only]
    used_sessions = sorted({c["session"] for c in cases})
    missing = [f"{n}: {spec['sessions'][n]}" for n in used_sessions
               if not Path(spec["sessions"][n]).is_file()]
    if missing:
        print("SKIP: session database(s) not available:\n  " + "\n  ".join(missing))
        return 0
    binary = Path(args.binary) if args.binary else Path(spec.get("binary") or optbuild_binary())
    if not binary.is_file():
        print(f"FAIL: tracker binary not found: {binary}\n"
              f"      build it with: meson compile -C optbuild posetrak-tracker")
        return 1
    work = Path(args.work_dir) if args.work_dir else cases_path.parent / "validation-work"
    copies: dict[str, Path] = {}
    for name in used_sessions:
        dest = work / f"{name}.db"
        copies[name] = dest
        if args.reuse_copies and dest.is_file():
            print(f"[{name}] reusing copy {dest}")
            continue
        print(f"[{name}] copying {spec['sessions'][name]} -> {dest} ...", flush=True)
        started = time.time()
        make_working_copy(Path(spec["sessions"][name]), dest)
        print(f"[{name}] copied and migrated in {time.time() - started:.0f}s")

    report = []
    failed = False
    for case in cases:
        name = case["name"]
        db = copies[case["session"]]
        print(f"\n=== {name} " + "=" * max(0, 60 - len(name)))
        if "subjects" in case:
            checks, extra = run_joint_case(case, db, binary, work)
            for label, ok, detail in checks:
                print(f"  {'PASS' if ok else 'FAIL'}  {label}: {detail}")
            failed |= not all(ok for _, ok, _ in checks)
            report.append({"case": name, **extra,
                           "checks": [{"check": l, "pass": ok, "detail": d} for l, ok, d in checks]})
            continue
        conn = open_readonly(db)
        baseline_row = conn.execute("SELECT * FROM tracking_runs WHERE id LIKE ?", (case["baseline_run"] + "%",)).fetchone()
        if baseline_row is None:
            print(f"FAIL: baseline run {case['baseline_run']!r} not found")
            failed = True
            continue
        recipe = {
            "sequence": baseline_row["observation_sequence_id"],
            "skeleton": baseline_row["skeleton_id"],
            "config": baseline_row["tracker_config_id"],
            "person_id": 0,
            "start_time": case.get("start_time"),
            "end_time": case.get("end_time"),
            "smooth": case.get("smooth", True),
        }
        if case.get("seed_from_baseline"):
            recipe["seed_position"] = first_root_position(conn, baseline_row["id"])
            print("seed position from baseline:", [round(v, 3) for v in recipe["seed_position"]])
        baseline = run_metrics(conn, baseline_row["id"])
        other = run_metrics(conn, case["expect"]["nis_close_to_run"]) if case.get("expect", {}).get("nis_close_to_run") else None
        conn.close()
        if case.get("redetect"):
            existing = find_validation_sequence(db, name) if args.reuse_detection else None
            if existing:
                recipe["sequence"] = existing
                print(f"reusing detection: sequence {existing[:8]}")
            else:
                print(f"re-running detection of run {case['redetect']['detection_run']} ...", flush=True)
                recipe["sequence"], detect_seconds = redetect(db, case["redetect"]["detection_run"], name)
                print(f"detection done in {detect_seconds:.0f}s -> new sequence {recipe['sequence'][:8]}")

        out_dir = work / "out" / name
        code, parsed, elapsed = run_tracker(binary, db, recipe, out_dir, work / f"{name}.log")
        print(f"tracker exit {code} in {elapsed:.0f}s: " + ", ".join(f"{k}={v}" for k, v in parsed.items()))
        checks: list[tuple[str, bool, str]] = [("tracker exit code", code == 0, str(code))]
        new = None
        if code == 0 and parsed.get("run_id"):
            conn = open_readonly(db)
            new = run_metrics(conn, parsed["run_id"])
            conn.close()
            checks += evaluate(case, parsed, new, baseline, other)
        else:
            checks.append(("tracker produced a run", False, "no tracking_run_id"))
        for label, ok, detail in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {label}: {detail}")
        failed |= not all(ok for _, ok, _ in checks)
        report.append({"case": name, "seconds": round(elapsed, 1), "parsed": parsed, "new": new, "baseline": baseline,
                       "checks": [{"check": l, "pass": ok, "detail": d} for l, ok, d in checks]})

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    n_pass = sum(1 for r in report if all(c["pass"] for c in r["checks"]))
    print(f"\n{n_pass}/{len(report)} cases pass" + ("" if not failed else "  -- FAILURES ABOVE"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

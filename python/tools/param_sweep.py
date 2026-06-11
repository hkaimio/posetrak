#!/usr/bin/env python3
"""
UKF parameter sweep for posetrak (session-DB mode).

Creates child tracker_configs from a base config, runs the tracker for each
combination, and collects per-camera reprojection errors plus filter-health
metrics from the session DB.

Usage
-----
    python python/tools/param_sweep.py \\
        --session-db /mnt/d/mocap/ukemi-tommi-20260509.db \\
        --config     <base-tracker-config-id-or-prefix> \\
        --sequences  <seq-id-1> [<seq-id-2> ...]  \\
        [--skeleton  <skeleton-id>]  \\
        [--binary    optbuild/cli/posetrak]  \\
        [--out-dir   /tmp/posetrak_sweep]  \\
        [--time-start 38.08 --time-end 66.44]

If --sequences is omitted the script finds the most recent run for the config
and uses its sequence.  If --skeleton is omitted it is taken from that run.

Edit SWEEP_GRID, FIXED_PARAMS, and VELOCITY_MODE_CAMERAS below.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sweep configuration — edit these
# ---------------------------------------------------------------------------

# Axis being swept: velocity measurement noise for the bad-extrinsics camera.
# None = baseline (no velocity mode).
SWEEP_NOISE_VALUES: list[float | None] = [None, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0, 60.0]

# Camera indices (0-based, sorted alphabetically in active_camera_ids) that
# should use velocity measurements.  Set to [] for the baseline.
VELOCITY_CAMERAS: list[int] = [2]  # insta_ace2_pro

# Parameters kept fixed across all sweep runs (passed verbatim to edit_config).
FIXED_PARAMS: dict[str, float] = {
    "measurement_noise_std": 60.0,
    "outlier_threshold":     4.0,
}

# Cameras considered "good" for evaluation (mean inlier reprojection error on
# these is the primary metric).  Labels must match active_camera_ids order.
GOOD_CAMERA_LABELS: set[str] = {"gopro-11_mini_01", "pixel7", "pixel9"}

# ---------------------------------------------------------------------------


def open_db(path: Path) -> sqlite3.Connection:
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from posetrak.db.db import open_session
    conn = open_session(path)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_base_config(conn: sqlite3.Connection, prefix: str) -> str:
    row = conn.execute(
        "SELECT id FROM tracker_configs WHERE id LIKE ? ORDER BY created_at DESC LIMIT 1",
        (prefix + "%",),
    ).fetchone()
    if row is None:
        raise ValueError(f"No tracker_config found matching {prefix!r}")
    return row["id"]


def resolve_sequence_and_skeleton(conn: sqlite3.Connection, config_id: str) -> tuple[str, str]:
    row = conn.execute(
        """SELECT observation_sequence_id, skeleton_id
           FROM tracking_runs WHERE tracker_config_id = ?
           ORDER BY ran_at DESC LIMIT 1""",
        (config_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"No tracking runs found for config {config_id!r}")
    return row["observation_sequence_id"], row["skeleton_id"]


def person_label(conn: sqlite3.Connection, seq_id: str) -> str:
    row = conn.execute(
        "SELECT person_name FROM sequence_persons WHERE sequence_id = ? LIMIT 1",
        (seq_id,),
    ).fetchone()
    return row["person_name"] if row else seq_id[:8]


def create_child_config(
    conn: sqlite3.Connection,
    base_config_id: str,
    velocity_noise: float | None,
) -> str:
    """Create a child tracker_config row and return its ID."""
    from posetrak.db.manage_config import edit_config

    vel_cams = VELOCITY_CAMERAS if velocity_noise is not None else []
    kwargs: dict = {**FIXED_PARAMS}
    if velocity_noise is not None:
        kwargs["velocity_measurement_noise_std"] = velocity_noise
    return edit_config(
        conn,
        base_config_id,
        velocity_mode_camera_ids=vel_cams if vel_cams else None,
        **kwargs,
    )


def decode_obs_blob(
    blob: bytes,
    cam_labels: list[str],
    marker_names: list[str],
) -> list[dict]:
    """Decode one obs_blob into a list of dicts (one per non-absent slot)."""
    n_cams, n_markers = len(cam_labels), len(marker_names)
    data = np.frombuffer(blob, dtype="<f4").reshape(n_cams, n_markers, 8)
    records = []
    for ci, cam in enumerate(cam_labels):
        for mi in range(n_markers):
            slot = data[ci, mi]
            if not np.isfinite(slot[6]):
                continue  # absent slot
            obs_x, obs_y = float(slot[0]), float(slot[1])
            pred_x, pred_y = float(slot[2]), float(slot[3])
            is_outlier = bool(slot[6] != 0.0)
            error = float(np.sqrt((obs_x - pred_x) ** 2 + (obs_y - pred_y) ** 2)) \
                if np.isfinite(obs_x) and np.isfinite(pred_x) else float("nan")
            records.append({
                "cam": cam,
                "is_outlier": is_outlier,
                "error": error,
            })
    return records


def compute_metrics(
    conn: sqlite3.Connection,
    run_id: str,
    cam_labels: list[str],
    marker_names: list[str],
    person_id: int = 0,
) -> dict | None:
    # ── Filter-level metrics ────────────────────────────────────────────────
    rows = conn.execute(
        """SELECT tracking_lost, n_inlier_observations, cov_condition_number,
                  nis_value, nis_dof
           FROM tracking_results
           WHERE run_id = ? AND person_id = ? AND is_smoothed = 0
           ORDER BY tracker_step""",
        (run_id, person_id),
    ).fetchall()
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=[
        "tracking_lost", "n_inlier_observations",
        "cov_condition_number", "nis_value", "nis_dof",
    ])
    n_frames = len(df)
    tracking_lost_pct = 100.0 * df["tracking_lost"].mean()

    nis_rows = df[(df["nis_dof"] > 0) & df["nis_value"].notna()]
    if not nis_rows.empty:
        per_dof = nis_rows["nis_value"] / nis_rows["nis_dof"]
        nis_mean = float(per_dof.mean())
        nis_std  = float(per_dof.std()) if len(per_dof) > 1 else float("nan")
    else:
        nis_mean = nis_std = float("nan")

    cond = df["cov_condition_number"].replace(0, float("nan")).dropna()
    cond_p95 = float(np.nanpercentile(cond, 95)) if not cond.empty else float("nan")

    active = df[df["tracking_lost"] == 0]
    avg_inliers = float(active["n_inlier_observations"].mean()) \
        if not active.empty and active["n_inlier_observations"].notna().any() else float("nan")

    # ── Per-camera reprojection errors ─────────────────────────────────────
    blob_rows = conn.execute(
        """SELECT obs_blob FROM tracking_obs_results
           WHERE run_id = ? AND person_id = ? ORDER BY tracker_step""",
        (run_id, person_id),
    ).fetchall()

    cam_errors: dict[str, list[float]] = {c: [] for c in cam_labels}
    cam_outlier_counts: dict[str, int] = {c: 0 for c in cam_labels}
    cam_total: dict[str, int] = {c: 0 for c in cam_labels}

    for brow in blob_rows:
        for rec in decode_obs_blob(bytes(brow["obs_blob"]), cam_labels, marker_names):
            cam = rec["cam"]
            cam_total[cam] += 1
            if rec["is_outlier"]:
                cam_outlier_counts[cam] += 1
            elif np.isfinite(rec["error"]):
                cam_errors[cam].append(rec["error"])

    per_cam: dict[str, dict] = {}
    for cam in cam_labels:
        errs = cam_errors[cam]
        tot  = cam_total[cam]
        out  = cam_outlier_counts[cam]
        per_cam[cam] = {
            "mean":         float(np.mean(errs))       if errs else float("nan"),
            "median":       float(np.median(errs))     if errs else float("nan"),
            "p95":          float(np.percentile(errs, 95)) if errs else float("nan"),
            "outlier_rate": (out / tot * 100.0)        if tot  else float("nan"),
        }

    good_errors = [e for c, es in cam_errors.items() if c in GOOD_CAMERA_LABELS for e in es]
    good_mean   = float(np.mean(good_errors))   if good_errors else float("nan")
    good_median = float(np.median(good_errors)) if good_errors else float("nan")

    return {
        "n_frames":          n_frames,
        "nis_mean":          nis_mean,
        "nis_std":           nis_std,
        "cond_p95":          cond_p95,
        "avg_inliers":       avg_inliers,
        "tracking_lost_pct": tracking_lost_pct,
        "good_cam_mean":     good_mean,
        "good_cam_median":   good_median,
        "per_cam":           per_cam,
    }


def run_tracker(
    binary: Path,
    db_path: Path,
    sequence_id: str,
    skeleton_id: str,
    config_id: str,
    person_id: int,
    time_start: float,
    time_end: float,
    out_dir: Path,
) -> str | None:
    """Run the tracker; return tracking_run_id or None on failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary), "track",
        "--session-db",     str(db_path),
        "--sequence",       sequence_id,
        "--skeleton",       skeleton_id,
        "--tracker-config", config_id,
        "--person-id",      str(person_id),
        "--start-time",     str(time_start),
        "--end-time",       str(time_end),
        "--output-dir",     str(out_dir),
        "--quiet",
    ]
    import subprocess
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        (out_dir / "stderr.txt").write_text(result.stderr)
        return None

    m = re.search(r"tracking_run_id:\s*(\S+)", result.stdout)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-db", required=True)
    ap.add_argument("--config",     required=True, metavar="CONFIG_ID",
                    help="Base tracker_config ID or unique prefix")
    ap.add_argument("--sequences",  nargs="+", metavar="SEQ_ID",
                    help="Pose observation sequence IDs (default: from most recent run)")
    ap.add_argument("--skeleton",   metavar="SKEL_ID",
                    help="Skeleton ID (default: from most recent run)")
    ap.add_argument("--person-id",  type=int, default=0)
    ap.add_argument("--binary",     default="optbuild/cli/posetrak")
    ap.add_argument("--out-dir",    default="/tmp/posetrak_sweep")
    ap.add_argument("--time-start", type=float, default=0.0)
    ap.add_argument("--time-end",   type=float, default=-1.0)
    args = ap.parse_args()

    db_path = Path(args.session_db)
    binary  = Path(args.binary)
    out_dir = Path(args.out_dir)

    if not db_path.exists():
        print(f"error: DB not found: {db_path}", file=sys.stderr); return 1
    if not binary.exists():
        print(f"error: binary not found: {binary}", file=sys.stderr); return 1

    conn = open_db(db_path)

    base_config_id = resolve_base_config(conn, args.config)
    print(f"Base config : {base_config_id}")

    sequences: list[str] = args.sequences or []
    skeleton_id: str     = args.skeleton or ""

    if not sequences or not skeleton_id:
        seq_fallback, skel_fallback = resolve_sequence_and_skeleton(conn, base_config_id)
        sequences  = sequences  or [seq_fallback]
        skeleton_id = skeleton_id or skel_fallback

    print(f"Sequences   : {sequences}")
    print(f"Skeleton    : {skeleton_id}")

    # Resolve camera labels and marker names from first sequence's run
    seq0_run = conn.execute(
        "SELECT active_camera_ids, marker_names FROM tracking_runs "
        "WHERE observation_sequence_id = ? ORDER BY ran_at DESC LIMIT 1",
        (sequences[0],),
    ).fetchone()
    if seq0_run is None:
        print("error: no tracking runs for first sequence; re-order or add --skeleton",
              file=sys.stderr); return 1
    cam_labels:   list[str] = json.loads(seq0_run["active_camera_ids"] or "[]")
    marker_names: list[str] = json.loads(seq0_run["marker_names"]       or "[]")
    print(f"Cameras     : {list(enumerate(cam_labels))}")
    print(f"Vel cameras : indices {VELOCITY_CAMERAS} "
          f"= {[cam_labels[i] for i in VELOCITY_CAMERAS if i < len(cam_labels)]}")
    print(f"Good cameras: {GOOD_CAMERA_LABELS & set(cam_labels)}")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    run_idx = 0

    for velocity_noise in SWEEP_NOISE_VALUES:
        label = f"vel_noise={velocity_noise}" if velocity_noise is not None else "BASELINE (no velocity mode)"
        child_id = create_child_config(conn, base_config_id, velocity_noise)

        for seq_id in sequences:
            run_idx += 1
            pname = person_label(conn, seq_id)
            print(f"[{run_idx:3d}] {label}  person={pname}", end="  ", flush=True)

            run_out = out_dir / f"run_{run_idx:03d}"
            t0 = time.perf_counter()
            run_id = run_tracker(
                binary, db_path, seq_id, skeleton_id, child_id,
                args.person_id, args.time_start, args.time_end, run_out,
            )
            elapsed = time.perf_counter() - t0

            base_record = {
                "run":          run_idx,
                "velocity_noise": velocity_noise,
                "sequence_id":  seq_id,
                "person":       pname,
                "config_id":    child_id,
            }

            if run_id is None:
                print(f"FAILED ({elapsed:.1f}s)")
                records.append({**base_record, "status": "FAILED"})
                continue

            # Re-open after tracker wrote to DB
            conn.close()
            conn = open_db(db_path)

            metrics = compute_metrics(conn, run_id, cam_labels, marker_names, args.person_id)
            if metrics is None:
                print(f"NO_DATA ({elapsed:.1f}s)")
                records.append({**base_record, "run_id": run_id, "status": "NO_DATA"})
                continue

            per_cam = metrics.pop("per_cam")
            print(
                f"good_mean={metrics['good_cam_mean']:.1f}px  "
                f"good_med={metrics['good_cam_median']:.1f}px  "
                f"NIS={metrics['nis_mean']:.2f}  "
                f"lost={metrics['tracking_lost_pct']:.1f}%  "
                f"({elapsed:.1f}s)"
            )
            cam_flat: dict[str, float] = {}
            for cam, stats in per_cam.items():
                short = cam.replace("-", "_").replace(".", "_")
                for k, v in stats.items():
                    cam_flat[f"{short}_{k}"] = v

            records.append({**base_record, "run_id": run_id, "status": "OK",
                             **metrics, **cam_flat})

    conn.close()

    # ── Summary ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    summary_path = out_dir / "sweep_summary.csv"
    df.to_csv(summary_path, index=False)

    ok = df[df["status"] == "OK"].copy()
    if ok.empty:
        print("\nNo successful runs."); return 1

    # Primary: good-camera mean reprojection error (lower is better)
    # Tiebreak: NIS calibration (closer to 1.0), tracking lost %
    ok["score"] = (
        ok["good_cam_mean"]
        + 5.0 * (ok["nis_mean"] - 1.0).abs()
        + 20.0 * ok["tracking_lost_pct"] / 100.0
    )

    # Aggregate over persons: mean score per velocity_noise value
    agg = ok.groupby("velocity_noise", dropna=False).agg(
        good_mean=("good_cam_mean", "mean"),
        good_median=("good_cam_median", "mean"),
        nis_mean=("nis_mean", "mean"),
        lost_pct=("tracking_lost_pct", "mean"),
        score=("score", "mean"),
        n_persons=("person", "count"),
    ).reset_index().sort_values("score")

    failed = len(df) - len(ok)
    print(f"\n{'─'*90}")
    print(f"Results: {len(ok)}/{len(df)} runs OK" +
          (f", {failed} failed" if failed else ""))
    print("Ranked by good-camera mean reprojection error (primary) + NIS calibration")
    print(f"{'─'*90}")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Per-camera breakdown for best velocity_noise
    best_noise = agg.iloc[0]["velocity_noise"]
    best_runs  = ok[ok["velocity_noise"].isna() if pd.isna(best_noise)
                    else ok["velocity_noise"] == best_noise]
    print(f"\nBest setting: vel_noise={best_noise}  ({len(best_runs)} persons)")
    for cam in cam_labels:
        short = cam.replace("-", "_").replace(".", "_")
        mean_col = f"{short}_mean"
        out_col  = f"{short}_outlier_rate"
        if mean_col in best_runs.columns:
            m = best_runs[mean_col].mean()
            o = best_runs[out_col].mean() if out_col in best_runs.columns else float("nan")
            tag = " ← bad cam" if cam not in GOOD_CAMERA_LABELS else ""
            print(f"  {cam:25s}: mean={m:.1f}px  outlier={o:.1f}%{tag}")

    print(f"\nFull results: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

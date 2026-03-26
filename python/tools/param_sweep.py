#!/usr/bin/env python3
"""
UKF parameter sweep for posetrak (session-DB mode).

Creates child tracker_config rows from a base config, runs the tracker for
each, and ranks results by filter consistency (NIS/dof ≈ 1), inlier rate,
and covariance condition number.  All results are stored in the session DB.

Usage
-----
    uv run python/tools/param_sweep.py \\
        --session-db /mnt/d/mocap/<session>/session.db \\
        --config     <base-tracker-config-id-or-prefix> \\
        [--sequence  <pose-observation-sequence-id>] \\
        [--skeleton  <skeleton-id>] \\
        [--out-dir   /tmp/posetrak_sweep]

If --sequence / --skeleton are omitted the script finds them from the most
recent tracking run that used the given base config.

Edit SWEEP_GRID and FIXED_PARAMS below to change what is varied.
"""

import argparse
import itertools
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameter grid — edit these to change what is swept
# ---------------------------------------------------------------------------

SWEEP_GRID: dict[str, list] = {
    "process_noise_std":     [0.05, 0.1, 0.2],
    "process_noise_vel_std": [0.2, 0.5, 1.0],
    "velocity_half_life_s":  [0.25, 0.5, 1.0],
}

FIXED_PARAMS: dict[str, float | int | None] = {
    "measurement_noise_std": 60.0,
    "outlier_threshold":     4.0,
}

TIME_RANGE = (0.0, 10.0)

# ---------------------------------------------------------------------------


def resolve_base_run(session, config_id: str) -> dict:
    """Return metadata from the most recent tracking run using config_id."""
    row = session.execute(
        """
        SELECT id, observation_sequence_id, skeleton_id,
               active_camera_ids, marker_names
        FROM tracking_runs
        WHERE tracker_config_id = ?
        ORDER BY ran_at DESC
        LIMIT 1
        """,
        (config_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"No tracking runs found for config {config_id!r}")
    return dict(row)


def compute_metrics(session, run_id: str) -> dict | None:
    rows = session.execute(
        """
        SELECT tracking_lost, n_inlier_observations,
               cov_condition_number, nis_value, nis_dof
        FROM tracking_results
        WHERE run_id = ? AND person_id = 0 AND is_smoothed = 0
        ORDER BY tracker_step
        """,
        (run_id,),
    ).fetchall()

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=[
        "tracking_lost", "n_inlier_observations",
        "cov_condition_number", "nis_value", "nis_dof",
    ])

    n_frames = len(df)
    tracking_lost_pct = 100.0 * df["tracking_lost"].mean()

    # NIS / dof
    nis_rows = df[(df["nis_dof"] > 0) & df["nis_value"].notna()]
    if not nis_rows.empty:
        per_dof = nis_rows["nis_value"] / nis_rows["nis_dof"]
        nis_mean = float(per_dof.mean())
        nis_std  = float(per_dof.std()) if len(per_dof) > 1 else float("nan")
    else:
        nis_mean = nis_std = float("nan")

    # Condition number
    cond = df["cov_condition_number"].replace(0, float("nan")).dropna()
    if not cond.empty:
        cond_max = float(cond.max())
        cond_p95 = float(np.nanpercentile(cond, 95))
    else:
        cond_max = cond_p95 = float("nan")

    # Inlier rate (frames that had observations)
    obs_rows = df[df["n_inlier_observations"].notna()]
    # Approximate: need num_observations too, use inlier count as proxy for activity
    # Use n_inlier_observations directly; normalise against max (rough inlier rate proxy)
    active = df[df["tracking_lost"] == 0]
    if not active.empty and active["n_inlier_observations"].notna().any():
        avg_inliers = float(active["n_inlier_observations"].mean())
    else:
        avg_inliers = float("nan")

    return {
        "n_frames":          n_frames,
        "nis_mean":          nis_mean,
        "nis_std":           nis_std,
        "cond_max":          cond_max,
        "cond_p95":          cond_p95,
        "avg_inliers":       avg_inliers,
        "tracking_lost_pct": tracking_lost_pct,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-db", required=True,
                    help="Path to the posetrak session DB")
    ap.add_argument("--config", required=True, metavar="CONFIG_ID",
                    help="Base tracker_config ID (or unique prefix)")
    ap.add_argument("--sequence", metavar="SEQ_ID", default=None,
                    help="Pose observation sequence ID (default: from most recent run)")
    ap.add_argument("--skeleton", metavar="SKEL_ID", default=None,
                    help="Skeleton ID (default: from most recent run)")
    ap.add_argument("--person-id", type=int, default=0,
                    help="Person ID to track (default: 0)")
    ap.add_argument("--binary", default="optbuild/cli/posetrak",
                    help="Path to posetrak binary")
    ap.add_argument("--out-dir", default="/tmp/posetrak_sweep",
                    help="Directory for per-run CSV output")
    args = ap.parse_args()

    db_path  = Path(args.session_db)
    binary   = Path(args.binary)
    out_dir  = Path(args.out_dir)

    if not db_path.exists():
        print(f"error: session DB not found: {db_path}", file=sys.stderr)
        return 1
    if not binary.exists():
        print(f"error: binary not found: {binary}", file=sys.stderr)
        return 1

    # Open session DB (Python layer handles migrations)
    import sqlite3
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from posetrak.db.db import open_session
    from posetrak.db.manage_config import edit_config

    session = open_session(db_path)
    session.row_factory = sqlite3.Row

    # Resolve base config ID prefix
    row = session.execute(
        "SELECT id FROM tracker_configs WHERE id LIKE ? ORDER BY created_at DESC LIMIT 1",
        (args.config + "%",),
    ).fetchone()
    if row is None:
        print(f"error: tracker_config not found: {args.config!r}", file=sys.stderr)
        return 1
    base_config_id = row["id"]
    print(f"Base config : {base_config_id}")

    # Resolve sequence and skeleton from most recent run if not supplied
    sequence_id = args.sequence
    skeleton_id = args.skeleton
    if sequence_id is None or skeleton_id is None:
        try:
            base_run = resolve_base_run(session, base_config_id)
            sequence_id = sequence_id or base_run["observation_sequence_id"]
            skeleton_id = skeleton_id or base_run["skeleton_id"]
        except ValueError as e:
            print(f"error: {e}\n"
                  "Provide --sequence and --skeleton explicitly.", file=sys.stderr)
            return 1

    print(f"Sequence    : {sequence_id}")
    print(f"Skeleton    : {skeleton_id}")

    out_dir.mkdir(parents=True, exist_ok=True)

    param_names  = list(SWEEP_GRID.keys())
    param_values = list(SWEEP_GRID.values())
    grid  = list(itertools.product(*param_values))
    total = len(grid)

    print(f"\nSweep: {total} runs  |  time range: {TIME_RANGE[0]}–{TIME_RANGE[1]} s")
    print(f"Sweep  : {', '.join(f'{k}={v}' for k, v in SWEEP_GRID.items())}")
    print(f"Fixed  : {', '.join(f'{k}={v}' for k, v in FIXED_PARAMS.items())}")
    print()

    records: list[dict] = []

    for idx, values in enumerate(grid, 1):
        sweep_params = dict(zip(param_names, values))
        all_params   = {**sweep_params, **FIXED_PARAMS}

        # Create child tracker_config row
        child_id = edit_config(session, base_config_id, **all_params)

        run_out = out_dir / f"run_{idx:03d}"
        run_out.mkdir(exist_ok=True)

        label = "  ".join(f"{k}={v:g}" for k, v in sweep_params.items())
        print(f"[{idx:3d}/{total}] {label}", end="  ", flush=True)

        cmd = [
            str(binary), "track",
            "--session-db",    str(db_path),
            "--sequence",      sequence_id,
            "--skeleton",      skeleton_id,
            "--tracker-config", child_id,
            "--person-id",     str(args.person_id),
            "--start-time",    str(TIME_RANGE[0]),
            "--end-time",      str(TIME_RANGE[1]),
            "--output-dir",    str(run_out),
            "--quiet",
        ]

        t0 = time.perf_counter()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            elapsed = time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            records.append({"run": idx, **sweep_params, **FIXED_PARAMS,
                             "config_id": child_id, "status": "TIMEOUT"})
            continue

        if result.returncode != 0:
            print(f"FAILED ({elapsed:.1f}s)")
            (run_out / "stderr.txt").write_text(result.stderr)
            records.append({"run": idx, **sweep_params, **FIXED_PARAMS,
                             "config_id": child_id, "status": "FAILED"})
            continue

        # Parse tracking_run_id from stdout
        m = re.search(r"tracking_run_id:\s*(\S+)", result.stdout)
        if not m:
            print(f"NO_RUN_ID ({elapsed:.1f}s)")
            records.append({"run": idx, **sweep_params, **FIXED_PARAMS,
                             "config_id": child_id, "status": "NO_RUN_ID"})
            continue
        run_id = m.group(1)

        # Re-open connection (tracker wrote to the same DB)
        session.close()
        session = open_session(db_path)
        session.row_factory = sqlite3.Row

        metrics = compute_metrics(session, run_id)
        if metrics is None:
            print(f"NO_DATA ({elapsed:.1f}s)")
            records.append({"run": idx, **sweep_params, **FIXED_PARAMS,
                             "config_id": child_id, "run_id": run_id, "status": "NO_DATA"})
            continue

        m2 = metrics
        print(
            f"NIS={m2['nis_mean']:.2f}±{m2['nis_std']:.2f}  "
            f"cond_p95={m2['cond_p95']:.1e}  "
            f"inliers={m2['avg_inliers']:.0f}/frame  "
            f"lost={m2['tracking_lost_pct']:.1f}%  "
            f"({elapsed:.1f}s)"
        )
        records.append({"run": idx, **sweep_params, **FIXED_PARAMS,
                         "config_id": child_id, "run_id": run_id, "status": "OK", **m2})

    session.close()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    df = pd.DataFrame(records)
    summary_path = out_dir / "sweep_summary.csv"
    df.to_csv(summary_path, index=False)

    ok = df[df["status"] == "OK"].copy()
    if ok.empty:
        print("\nNo successful runs.")
        return 1

    ok["score"] = (
        (ok["nis_mean"] - 1.0).abs()
        + 0.5  * (ok["tracking_lost_pct"] / 100.0)
        + ok["cond_p95"].apply(
            lambda x: max(0.0, (np.log10(x) - 6) * 0.05) if np.isfinite(x) and x > 0 else 0.0
        )
    )
    ok = ok.sort_values("score")

    failed = total - len(ok)
    print(f"\n{'─'*110}")
    print(
        f"Results: {len(ok)}/{total} successful"
        + (f", {failed} failed/timeout" if failed else "")
    )
    print("Ranked by |NIS/dof − 1| + penalty for tracking loss and high condition number")
    print(f"{'─'*110}")

    display_cols = (
        param_names
        + list(FIXED_PARAMS.keys())
        + ["nis_mean", "nis_std", "cond_p95", "avg_inliers",
           "tracking_lost_pct", "score", "run_id"]
    )
    display_cols = [c for c in display_cols if c in ok.columns]
    print(ok[display_cols].head(15).to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"\nFull results + run IDs: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

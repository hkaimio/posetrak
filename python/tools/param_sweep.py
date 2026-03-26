#!/usr/bin/env python3
"""
UKF parameter sweep for posetrak.

Runs the tracker over a grid of noise/damping parameters, collects metrics
from the output CSV files, and prints a ranked summary table.

Usage
-----
    uv run python/tools/param_sweep.py \\
        --base  /path/to/complete_config.toml \\
        --out-dir /tmp/posetrak_sweep

Edit SWEEP_GRID and FIXED_PARAMS below to change which parameters are varied.
The script writes a temp config.toml per run (overriding [tracking], [output],
and [processing] time range), runs the tracker, then reads tracking_stats.csv
and marker_projections.csv for metrics.
"""

import argparse
import copy
import itertools
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameter grid — edit these to change what is swept
# ---------------------------------------------------------------------------

SWEEP_GRID: dict[str, list] = {
    "process_noise_std":     [0.05, 0.1, 0.2],
    "process_noise_vel_std": [0.2, 0.5, 1.0],
    "velocity_half_life_s":  [0.5, 1.0, 2.0],
}

FIXED_PARAMS: dict[str, float] = {
    "measurement_noise_std": 60.0,
    "outlier_threshold": 4.0,
}

TIME_RANGE = (0.0, 10.0)

# ---------------------------------------------------------------------------


def _toml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        # Escape backslashes for Windows paths
        return '"' + v.replace("\\", "\\\\") + '"'
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _write_section(lines: list[str], d: dict, header: str) -> None:
    lines.append(f"\n[{header}]")
    for k, v in d.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {_toml_scalar(v)}")
    for k, v in d.items():
        if isinstance(v, dict):
            _write_section(lines, v, f"{header}.{k}")


def write_toml(d: dict) -> str:
    lines: list[str] = []
    for k, v in d.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {_toml_scalar(v)}")
    for k, v in d.items():
        if isinstance(v, dict):
            _write_section(lines, v, k)
    return "\n".join(lines) + "\n"


def _col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    """Return the first column that exists, or None."""
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def compute_metrics(run_dir: Path) -> dict | None:
    stats_path = run_dir / "tracking_stats.csv"
    if not stats_path.exists():
        return None

    stats = pd.read_csv(stats_path)
    if stats.empty:
        return None

    n_frames = len(stats)

    # tracking_lost
    lost_col = _col(stats, "tracking_lost")
    tracking_lost_pct = 100.0 * lost_col.mean() if lost_col is not None else float("nan")

    # NIS / dof
    nis_val = _col(stats, "nis_value")
    nis_dof = _col(stats, "nis_dof")
    if nis_val is not None and nis_dof is not None:
        mask = (nis_dof > 0) & nis_val.notna()
        valid = nis_val[mask] / nis_dof[mask]
        nis_mean = float(valid.mean()) if not valid.empty else float("nan")
        nis_std  = float(valid.std())  if len(valid) > 1 else float("nan")
    else:
        nis_mean = nis_std = float("nan")

    # Covariance condition number
    cond_col = _col(stats, "cov_condition_number")
    if cond_col is not None:
        cond = cond_col.replace(0, float("nan")).dropna()
        cond_max = float(cond.max())  if not cond.empty else float("nan")
        cond_p95 = float(np.nanpercentile(cond, 95)) if not cond.empty else float("nan")
    else:
        cond_max = cond_p95 = float("nan")

    # Inlier rate
    inliers_col  = _col(stats, "num_inliers",  "n_inlier_observations")
    obs_col      = _col(stats, "num_observations")
    if inliers_col is not None and obs_col is not None:
        obs_rows = stats[obs_col > 0]
        if not obs_rows.empty:
            inlier_rate = float((inliers_col[obs_col > 0] / obs_col[obs_col > 0]).mean())
        else:
            inlier_rate = float("nan")
    else:
        inlier_rate = float("nan")

    # Reprojection error from marker_projections.csv (inliers only when available)
    reproj_mean = float("nan")
    proj_path = run_dir / "marker_projections.csv"
    if proj_path.exists():
        proj = pd.read_csv(proj_path)
        if not proj.empty:
            if "error_dist" not in proj.columns:
                ex = _col(proj, "error_x")
                ey = _col(proj, "error_y")
                if ex is not None and ey is not None:
                    proj = proj.copy()
                    proj["error_dist"] = np.sqrt(ex**2 + ey**2)
            if "error_dist" in proj.columns:
                outlier_col = _col(proj, "is_outlier")
                if outlier_col is not None:
                    inlier_proj = proj[~outlier_col.astype(bool)]
                else:
                    inlier_proj = proj
                if not inlier_proj.empty:
                    reproj_mean = float(inlier_proj["error_dist"].mean())

    return {
        "n_frames":          n_frames,
        "nis_mean":          nis_mean,
        "nis_std":           nis_std,
        "cond_max":          cond_max,
        "cond_p95":          cond_p95,
        "inlier_rate":       inlier_rate,
        "tracking_lost_pct": tracking_lost_pct,
        "reproj_error_mean": reproj_mean,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base",    required=True,
                    help="Base TOML config (must have [data] with all file paths)")
    ap.add_argument("--binary",  default="optbuild/cli/posetrak",
                    help="Path to posetrak binary  [default: optbuild/cli/posetrak]")
    ap.add_argument("--out-dir", default="/tmp/posetrak_sweep",
                    help="Root directory for sweep outputs  [default: /tmp/posetrak_sweep]")
    args = ap.parse_args()

    base_path = Path(args.base)
    binary    = Path(args.binary)
    out_dir   = Path(args.out_dir)

    if not base_path.exists():
        print(f"error: base config not found: {base_path}", file=sys.stderr)
        return 1
    if not binary.exists():
        print(f"error: binary not found: {binary}", file=sys.stderr)
        return 1

    with open(base_path, "rb") as fh:
        base_cfg = tomllib.load(fh)

    out_dir.mkdir(parents=True, exist_ok=True)

    param_names  = list(SWEEP_GRID.keys())
    param_values = list(SWEEP_GRID.values())
    grid  = list(itertools.product(*param_values))
    total = len(grid)

    print(f"Sweep: {total} runs  |  time range: {TIME_RANGE[0]}–{TIME_RANGE[1]} s")
    print(f"Sweep params : {', '.join(f'{k}={v}' for k, v in SWEEP_GRID.items())}")
    print(f"Fixed params : {', '.join(f'{k}={v}' for k, v in FIXED_PARAMS.items())}")
    print()

    records: list[dict] = []

    for idx, values in enumerate(grid, 1):
        sweep_params = dict(zip(param_names, values))
        all_params   = {**sweep_params, **FIXED_PARAMS}

        run_dir = out_dir / f"run_{idx:03d}"
        run_dir.mkdir(exist_ok=True)

        cfg = copy.deepcopy(base_cfg)

        tracking = cfg.setdefault("tracking", {})
        for k, v in all_params.items():
            tracking[k] = v

        output = cfg.setdefault("output", {})
        output["directory"] = str(run_dir)
        output.setdefault("export_tracking_results", True)
        output.setdefault("export_statistics", True)

        proc = cfg.setdefault("processing", {})
        proc["start_time"] = float(TIME_RANGE[0])
        proc["end_time"]   = float(TIME_RANGE[1])

        toml_path = run_dir / "config.toml"
        toml_path.write_text(write_toml(cfg))

        label = "  ".join(f"{k}={v:g}" for k, v in sweep_params.items())
        print(f"[{idx:3d}/{total}] {label}", end="  ", flush=True)

        t0 = time.perf_counter()
        try:
            proc_result = subprocess.run(
                [str(binary), "track", str(toml_path)],
                capture_output=True, text=True, timeout=180,
            )
            elapsed = time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            records.append({"run": idx, **sweep_params, **FIXED_PARAMS, "status": "TIMEOUT"})
            continue

        if proc_result.returncode != 0:
            print(f"FAILED ({elapsed:.1f}s)")
            # Save stderr for diagnosis
            (run_dir / "stderr.txt").write_text(proc_result.stderr)
            records.append({"run": idx, **sweep_params, **FIXED_PARAMS, "status": "FAILED"})
            continue

        metrics = compute_metrics(run_dir)
        if metrics is None:
            print(f"NO_DATA ({elapsed:.1f}s)")
            records.append({"run": idx, **sweep_params, **FIXED_PARAMS, "status": "NO_DATA"})
            continue

        m = metrics
        print(
            f"NIS={m['nis_mean']:.2f}±{m['nis_std']:.2f}  "
            f"cond_p95={m['cond_p95']:.1e}  "
            f"inliers={m['inlier_rate']:.0%}  "
            f"reproj={m['reproj_error_mean']:.1f}px  "
            f"lost={m['tracking_lost_pct']:.1f}%  "
            f"({elapsed:.1f}s)"
        )
        records.append({"run": idx, **sweep_params, **FIXED_PARAMS, "status": "OK", **m})

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    df = pd.DataFrame(records)
    summary_path = out_dir / "sweep_summary.csv"
    df.to_csv(summary_path, index=False)

    ok = df[df["status"] == "OK"].copy()
    if ok.empty:
        print("\nNo successful runs — check stderr.txt files in individual run dirs.")
        return 1

    # Score: want NIS/dof close to 1, low tracking_lost, good inliers, sane condition
    ok["score"] = (
        (ok["nis_mean"] - 1.0).abs()
        + 0.5  * (ok["tracking_lost_pct"] / 100.0)
        - 0.2  * ok["inlier_rate"].fillna(0)
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
    print("Ranked by |NIS/dof − 1| + penalties for tracking loss, low inliers, high condition number")
    print(f"{'─'*110}")

    display_cols = (
        param_names
        + list(FIXED_PARAMS.keys())
        + ["nis_mean", "nis_std", "cond_p95", "inlier_rate",
           "reproj_error_mean", "tracking_lost_pct", "score"]
    )
    display_cols = [c for c in display_cols if c in ok.columns]

    fmt = {
        "nis_mean": "{:.3f}", "nis_std": "{:.3f}",
        "cond_p95": "{:.2e}", "inlier_rate": "{:.2f}",
        "reproj_error_mean": "{:.1f}", "tracking_lost_pct": "{:.1f}",
        "score": "{:.3f}",
    }
    print(ok[display_cols].head(15).to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"\nFull results: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

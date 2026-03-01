#!/usr/bin/env python3
"""Compare debug frame data between baseline and HEAD for common columns."""
import sys
import csv
import os
import math

BASE = "/home/harri/projects/posetrak/tracking_tests"
BASELINE_DIR = f"{BASE}/harri-no-palms-baseline/debug"
HEAD_DIR = f"{BASE}/harri-no-palms-head/debug"


def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def compare_csv(frame_dir_name, filename, tol=1e-4):
    bpath = f"{BASELINE_DIR}/{frame_dir_name}/{filename}"
    hpath = f"{HEAD_DIR}/{frame_dir_name}/{filename}"

    if not os.path.exists(bpath):
        print(f"  MISSING baseline: {bpath}")
        return
    if not os.path.exists(hpath):
        print(f"  MISSING head: {hpath}")
        return

    base_rows = load_csv(bpath)
    head_rows = load_csv(hpath)

    base_cols = set(base_rows[0].keys()) if base_rows else set()
    head_cols = set(head_rows[0].keys()) if head_rows else set()
    common_cols = base_cols & head_cols
    baseline_only = base_cols - head_cols
    head_only = head_cols - base_cols

    print(f"\n  --- {filename} ---")
    print(f"  Baseline rows={len(base_rows)}, cols={len(base_cols)}")
    print(f"  HEAD rows={len(head_rows)}, cols={len(head_cols)}")
    print(f"  Common cols: {len(common_cols)}, baseline-only: {len(baseline_only)}, head-only: {len(head_only)}")

    if baseline_only:
        print(f"  Baseline-only cols (first 10): {sorted(baseline_only)[:10]}")
    if head_only:
        print(f"  Head-only cols (first 10): {sorted(head_only)[:10]}")

    # Compare values for common cols
    n_rows = min(len(base_rows), len(head_rows))
    max_diff = 0.0
    max_diff_loc = ""
    n_diffs = 0

    for i in range(n_rows):
        for col in sorted(common_cols):
            try:
                bval = float(base_rows[i][col])
                hval = float(head_rows[i][col])
                diff = abs(bval - hval)
                if diff > tol:
                    n_diffs += 1
                    if diff > max_diff:
                        max_diff = diff
                        max_diff_loc = f"row={i} col={col} base={bval:.6g} head={hval:.6g}"
            except (ValueError, KeyError):
                pass

    if n_diffs == 0:
        print(f"  ✓ All {n_rows*len(common_cols)} common values match (tol={tol})")
    else:
        print(f"  ✗ {n_diffs} values differ (tol={tol}), max_diff={max_diff:.6g}")
        print(f"    Worst: {max_diff_loc}")


def compare_observations(frame_dir_name):
    fname = "all_observations.csv"
    bpath = f"{BASELINE_DIR}/{frame_dir_name}/{fname}"
    hpath = f"{HEAD_DIR}/{frame_dir_name}/{fname}"

    if not os.path.exists(bpath) or not os.path.exists(hpath):
        print(f"  all_observations.csv: missing")
        return

    base_rows = load_csv(bpath)
    head_rows = load_csv(hpath)

    print(f"\n  --- {fname} ---")
    print(f"  Baseline: {len(base_rows)} observations")
    print(f"  HEAD:     {len(head_rows)} observations")

    # Build dicts by (marker_name, camera_id, frame_idx)
    def key(r):
        return (r['marker_name'], r['camera_id'], r.get('frame_idx', r.get('camera_frame_idx', '')))

    base_map = {key(r): r for r in base_rows}
    head_map = {key(r): r for r in head_rows}

    common_keys = set(base_map) & set(head_map)
    print(f"  Common obs: {len(common_keys)}, baseline-only: {len(set(base_map)-set(head_map))}, head-only: {len(set(head_map)-set(base_map))}")

    compare_cols = ['residual_norm', 'mahalanobis_distance', 'predicted_u', 'predicted_v', 'is_outlier']
    diffs = []
    for k in sorted(common_keys):
        br = base_map[k]
        hr = head_map[k]
        for col in compare_cols:
            try:
                bv = float(br[col])
                hv = float(hr[col])
                diff = abs(bv - hv)
                if diff > 1e-3:
                    diffs.append((diff, col, k, bv, hv))
            except (ValueError, KeyError):
                pass

    if not diffs:
        print(f"  ✓ All common observations match closely")
    else:
        diffs.sort(reverse=True)
        print(f"  ✗ {len(diffs)} value differences found")
        print(f"  Top differences:")
        for diff, col, k, bv, hv in diffs[:10]:
            print(f"    {col:30s} marker={k[0]:20s} cam={k[1]} diff={diff:.4g}  base={bv:.4g}  head={hv:.4g}")


if __name__ == "__main__":
    frame = sys.argv[1] if len(sys.argv) > 1 else "frame_0001"
    print(f"=== Comparing {frame} ===")

    compare_csv(frame, "sigma_points.csv")
    compare_csv(frame, "prior_covariance.csv", tol=1e-8)
    compare_csv(frame, "prior_state_computed.csv", tol=1e-4)
    compare_observations(frame)
    compare_csv(frame, "posterior_covariance.csv", tol=1e-4)

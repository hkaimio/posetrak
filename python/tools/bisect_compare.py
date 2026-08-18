#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Compare tracking results across bisect runs.

Usage: python scripts/bisect_compare.py <label_a> <label_b> [--frame-range 680 730]
"""
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
TESTS = ROOT / "tracking_tests"


def compare_labels(a: str, b: str, frame_lo: int, frame_hi: int):
    da = TESTS / f"harri-no-palms-{a}"
    db = TESTS / f"harri-no-palms-{b}"

    # --- Stats comparison ---
    sa = pd.read_csv(da / "tracking_stats.csv")
    sb = pd.read_csv(db / "tracking_stats.csv")

    print(f"\n{'='*60}")
    print(f"INLIER/OUTLIER comparison: {a} vs {b}")
    print(f"{'='*60}")
    print(f"\nOverall (all frames):")
    for col in ["num_inliers", "num_outliers"]:
        if col in sa and col in sb:
            print(f"  {col}: {a}={sa[col].mean():.2f}  {b}={sb[col].mean():.2f}  diff={sb[col].mean()-sa[col].mean():.2f}")

    print(f"\nFrames {frame_lo}–{frame_hi}:")
    wa = sa[(sa["frame"] >= frame_lo) & (sa["frame"] <= frame_hi)]
    wb = sb[(sb["frame"] >= frame_lo) & (sb["frame"] <= frame_hi)]
    for col in ["num_inliers", "num_outliers"]:
        if col in wa and col in wb:
            print(f"  {col}: {a}={wa[col].mean():.2f}  {b}={wb[col].mean():.2f}  diff={wb[col].mean()-wa[col].mean():.2f}")

    # Zero-inlier frames
    za = sa[sa["num_inliers"] == 0]["frame"].tolist()
    zb = sb[sb["num_inliers"] == 0]["frame"].tolist()
    print(f"\n  Zero-inlier frames:")
    print(f"    {a}: {len(za)} frames  (first 10: {za[:10]})")
    print(f"    {b}: {len(zb)} frames  (first 10: {zb[:10]})")

    # --- State vector comparison (joint angles) ---
    sva = pd.read_csv(da / "state_vectors.csv")
    svb = pd.read_csv(db / "state_vectors.csv")

    # Find common joint angle columns
    angle_cols_a = [c for c in sva.columns if "_angle_" in c]
    angle_cols_b = [c for c in svb.columns if "_angle_" in c]
    common_cols = sorted(set(angle_cols_a) & set(angle_cols_b))
    only_a = sorted(set(angle_cols_a) - set(angle_cols_b))
    only_b = sorted(set(angle_cols_b) - set(angle_cols_a))

    print(f"\n{'='*60}")
    print(f"STATE VECTOR comparison: {a} vs {b}")
    print(f"{'='*60}")
    print(f"  Angle cols in {a}: {len(angle_cols_a)}")
    print(f"  Angle cols in {b}: {len(angle_cols_b)}")
    print(f"  Common: {len(common_cols)}")
    if only_a:
        print(f"  Only in {a}: {only_a[:10]}")
    if only_b:
        print(f"  Only in {b}: {only_b[:10]}")

    if common_cols:
        # Merge on frame
        merged = pd.merge(sva[["tracker_frame_idx"] + common_cols],
                          svb[["tracker_frame_idx"] + common_cols],
                          on="tracker_frame_idx", suffixes=(f"_{a}", f"_{b}"))

        print(f"\n  Joint angle RMS differences (top 10 joints by max error, all frames):")
        joint_names = sorted(set(c.rsplit('_angle_',1)[0] for c in common_cols))
        errors = {}
        for jn in joint_names:
            cols = [c for c in common_cols if c.startswith(jn + "_angle_")]
            rms = 0.0
            for c in cols:
                ca, cb = f"{c}_{a}", f"{c}_{b}"
                if ca in merged and cb in merged:
                    rms += (merged[ca] - merged[cb]).pow(2).mean()
            errors[jn] = np.sqrt(rms)
        for jn, err in sorted(errors.items(), key=lambda x: -x[1])[:10]:
            print(f"    {jn}: RMS={err:.4f} rad")

        # Frame-range focus
        mf = merged[(merged["tracker_frame_idx"] >= frame_lo) &
                    (merged["tracker_frame_idx"] <= frame_hi)]
        if len(mf):
            print(f"\n  Joint angle RMS differences (frames {frame_lo}–{frame_hi}):")
            errors_f = {}
            for jn in joint_names:
                cols = [c for c in common_cols if c.startswith(jn + "_angle_")]
                rms = 0.0
                for c in cols:
                    ca, cb = f"{c}_{a}", f"{c}_{b}"
                    if ca in mf and cb in mf:
                        rms += (mf[ca] - mf[cb]).pow(2).mean()
                errors_f[jn] = np.sqrt(rms)
            for jn, err in sorted(errors_f.items(), key=lambda x: -x[1])[:10]:
                print(f"    {jn}: RMS={err:.4f} rad")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("label_a")
    p.add_argument("label_b")
    p.add_argument("--frame-range", nargs=2, type=int, default=[680, 730], metavar=("LO", "HI"))
    args = p.parse_args()
    compare_labels(args.label_a, args.label_b, args.frame_range[0], args.frame_range[1])

if __name__ == "__main__":
    main()

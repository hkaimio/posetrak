# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""skeleton_heldout_check.py — joint-position error of a default skeleton tracked on a reference capture.

Tracks a pose sequence with a bundled default skeleton (SOMA-77 markers included) and prints the error of its
joints against a reference tracking run in one or more time windows, next to the error of a baseline run on the
same sequence. Also prints how often the wrist, elbow and knee sit at a joint limit, which shows whether a limit
is saturated.

The error of a group is the median over time of the mean distance of the group's joints, in cm, with the 90th
percentile in brackets. Groups: legs (thigh, shin, foot), arms (upper arm, hand) and head.

Run it on a COPY of a session database; it adds a skeleton, a tracker configuration and a tracking run::

    python scripts/skeleton_heldout_check.py --session COPY.db --sequence SEQ --reference-run REF \\
        --baseline-run BASE --config CONFIG --binary optbuild/cpp/cli/posetrak-tracker.exe \\
        --window 123.5:135 [--sex male] [--skip-track | --run RUN]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from posetrak.db.db import open_session  # noqa: E402
from posetrak.db.manage_skeleton import seed_default_skeletons  # noqa: E402
from posetrak.db.skeleton_layout import SkeletonLayout  # noqa: E402
from posetrak.markers.bones import BodyPose  # noqa: E402

GROUPS = {
    "legs": ["thigh.L", "shin.L", "foot.L", "thigh.R", "shin.R", "foot.R"],
    "arms": ["upper_arm.L", "hand.L", "upper_arm.R", "hand.R"],
    "head": ["head"],
}
LIMIT_JOINTS = ("hand.L", "hand.R", "forearm.L", "forearm.R", "shin.L", "shin.R")


def _full_id(conn, table: str, prefix: str) -> str:
    return conn.execute(f"SELECT id FROM {table} WHERE id LIKE ?", (prefix + "%",)).fetchone()[0]


def track_with_default(session: Path, sequence: str, sex: str, config: str, binary: str) -> str:
    conn = open_session(session)
    ids = seed_default_skeletons(conn)
    name = f"Default {sex}"
    skeleton = conn.execute(
        f"SELECT id FROM skeletons WHERE name = ? AND id IN ({','.join('?' * len(ids))})", (name, *ids)).fetchone()[0]
    conn.close()
    print(f"tracking sequence {sequence} with {name} ({skeleton[:12]})", flush=True)
    cmd = [sys.executable, "-c", "from posetrak.cli.main import main; main()", "--session", str(session), "track",
           "run", "--sequence", sequence, "--skeleton", skeleton, "--config", config, "--binary", binary]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-1500:], result.stderr[-1500:])
    if result.returncode:
        raise SystemExit(result.returncode)
    conn = open_session(session)
    run = conn.execute("SELECT id FROM tracking_runs WHERE skeleton_id = ? ORDER BY ran_at DESC LIMIT 1",
                       (skeleton,)).fetchone()[0]
    conn.close()
    return run


def group_errors(reference: BodyPose, pose: BodyPose, t0: float, t1: float) -> str:
    cells = []
    for group, joints in GROUPS.items():
        per = []
        for t in np.arange(t0, t1, 0.1):
            ref_frames, frames = reference.joint_frames(float(t)), pose.joint_frames(float(t))
            if ref_frames is None or frames is None:
                continue
            per.append(np.mean([np.linalg.norm(ref_frames[j][:3, 3] - frames[j][:3, 3]) for j in joints]) * 100)
        cells.append(f"{group} {np.median(per):.1f} [{np.percentile(per, 90):.1f}]")
    return "   ".join(cells)


def limit_report(conn, run: str) -> None:
    skeleton_id = conn.execute("SELECT skeleton_id FROM tracking_runs WHERE id = ?", (run,)).fetchone()[0]
    text = conn.execute("SELECT yaml_content FROM skeletons WHERE id = ?", (skeleton_id,)).fetchone()[0]
    limits = {j["name"]: j.get("limits") for j in yaml.safe_load(text)["joints"]}
    layout = SkeletonLayout(text)
    states = conn.execute(
        "SELECT state FROM tracking_results WHERE run_id = ? AND is_smoothed = 1 AND tracking_lost = 0 "
        "ORDER BY tracker_step", (run,)).fetchall()
    angles: dict[str, list] = {}
    for row in states:
        decoded = layout.decode_state_blob(bytes(row[0]))["joint_angles"]
        for joint in LIMIT_JOINTS:
            angles.setdefault(joint, []).append(decoded[joint])
    print(f"share of the {len(states)} smoothed states within 3% of a joint limit")
    for joint, values in angles.items():
        degrees = np.degrees(np.array(values))
        for k, axis in enumerate("xyz"):
            lo, hi = np.degrees(limits[joint][axis])
            if hi - lo < 1e-6:
                continue
            at_limit = np.mean((degrees[:, k] <= lo + 0.03 * (hi - lo)) | (degrees[:, k] >= hi - 0.03 * (hi - lo)))
            p5, p50, p95 = np.percentile(degrees[:, k], [5, 50, 95])
            print(f"  {joint:10s} {axis} limit [{lo:4.0f},{hi:4.0f}] p5/p50/p95 {p5:4.0f}/{p50:4.0f}/{p95:4.0f}"
                  f"  at limit {100 * at_limit:3.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, type=Path, help="a COPY of a session database")
    ap.add_argument("--sequence", required=True, help="pose observation sequence id (prefix)")
    ap.add_argument("--reference-run", required=True, help="tracking run the error is measured against (prefix)")
    ap.add_argument("--baseline-run", required=True, help="tracking run of the same sequence to compare with (prefix)")
    ap.add_argument("--config", required=True, help="tracker configuration id (prefix) of the tracking run")
    ap.add_argument("--binary", help="posetrak-tracker binary (an optimised build)")
    ap.add_argument("--window", action="append", required=True, metavar="T0:T1", help="time window in s; repeatable")
    ap.add_argument("--sex", default="male", choices=("male", "female"))
    ap.add_argument("--run", help="evaluate this tracking run instead of tracking")
    args = ap.parse_args()

    if args.run is None and not args.binary:
        ap.error("--binary is required unless --run is given")
    run = args.run or track_with_default(args.session, args.sequence, args.sex, args.config, args.binary)
    conn = open_session(args.session)
    run = _full_id(conn, "tracking_runs", run)
    reference = BodyPose(conn, _full_id(conn, "tracking_runs", args.reference_run))
    poses = {"baseline": BodyPose(conn, _full_id(conn, "tracking_runs", args.baseline_run)), f"run {run[:8]}": BodyPose(conn, run)}
    print(f"joint position error against {args.reference_run} (cm): median [90th percentile]")
    for window in args.window:
        t0, t1 = (float(v) for v in window.split(":"))
        print(f" window {t0:.1f}-{t1:.1f} s")
        for name, pose in poses.items():
            print(f"   {name:16s} {group_errors(reference, pose, t0, t1)}")
    limit_report(conn, run)


if __name__ == "__main__":
    main()

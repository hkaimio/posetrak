#!/usr/bin/env python3
"""
process_session.py — Batch-process posetrak motion capture sessions.

For each sequence directory under SESSION_ROOT (excluding 'calibration'),
generates tracking configs for all persons, runs the tracker, exports BVH,
creates a mosaic visualization, and copies configs/skeletons to the results dir.

Usage:
    python3 scripts/process_session.py [options]

    python3 scripts/process_session.py \\
        --session-root /mnt/d/mocap/2026-03-10-posetrak-test \\
        --date 2026-03-13 \\
        [--sequences Harri_aihanmi_katatedori_ikkyo Timo_shomenuchi_kotegaeshi_korkea]

Defaults are for the 2026-03-10-posetrak-test session.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Session configuration — edit these for a new recording session
# ---------------------------------------------------------------------------

DEFAULT_SESSION_ROOT = Path("/mnt/d/mocap/2026-03-10-posetrak-test")
DEFAULT_CALIB_TOML   = DEFAULT_SESSION_ROOT / "calibration" / "calib.toml"

# Reallusion skeletns
# (name, person_id, skeleton_yaml_relative_to_project_root)
PERSONS = [
    ("harri", 0,  "/mnt/d/mocap/posetrak-templates/harri-scaled-kevin-2026-03-17.yaml"),
    ("timo",  1, "/mnt/d/mocap/posetrak-templates/girl-scaled-timo-2026-03-17.yaml"),
]

# # Original Rigify skeeltons
# # (name, person_id, skeleton_yaml_relative_to_project_root)
# PERSONS = [
#     ("harri", 0,  "tracking_tests/harri-scaled-skeleton.yaml"),
#     ("timo",  1, "tracking_tests/timo-scaled-skeleton.yaml"),
# ]

TRACKER_FPS    = 120.0
ACTIVE_GROUPS  = ["main", "HandL", "HandR"]

# Directories to skip when discovering sequence directories
SKIP_DIRS = {"calibration", "posetrak_config"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def find_sequences(session_root: Path, only: list[str] | None) -> list[Path]:
    """Return sequence directories sorted by name, skipping excluded dirs."""
    seqs = []
    for d in sorted(session_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name in SKIP_DIRS:
            continue
        if only and d.name not in only:
            continue
        seqs.append(d)
    return seqs


def next_version(version_parent: Path, date_str: str) -> int:
    """Return the next unused version integer for <version_parent>/<date_str>-N."""
    pattern = re.compile(rf"^{re.escape(date_str)}-(\d+)$")
    used = set()
    if version_parent.exists():
        for d in version_parent.iterdir():
            m = pattern.match(d.name)
            if m:
                used.add(int(m.group(1)))
    v = 1
    while v in used:
        v += 1
    return v


def detect_end_time(sync_json: Path) -> float:
    """Read sync_data.json and return the max timestamp across all cameras."""
    with open(sync_json) as f:
        data = json.load(f)
    max_ts = 0.0
    for info in data.values():
        pts = info.get("syncpoints", [])
        if pts:
            max_ts = max(max_ts, pts[-1]["timestamp"])
    # Add a small buffer
    return round(max_ts + 0.5, 1)


def generate_toml(
    *,
    skeleton_rel: str,
    cameras: Path,
    observations_dir: Path,
    sync_json: Path,
    person_id: int,
    output_dir: Path,
    end_time: float,
) -> str:
    """Generate a tracker TOML config string."""
    groups_str = "[" + ", ".join(f'"{g}"' for g in ACTIVE_GROUPS) + "]"
    return f"""\
[data]
skeleton = "{skeleton_rel}"
cameras = "{cameras}"
observations_dir = "{observations_dir}"
sync = "{sync_json}"
person_id = {person_id}
active_joint_groups = {groups_str}

[tracking]
process_noise_std = 0.15
measurement_noise_std = 20.0
outlier_threshold = 4.0

[tracking.initialization]
ik_max_iterations = 1000
ik_tolerance = 0.02
init_position_std = 1.0
init_orientation_std = 1.0
init_joint_std = 0.1
init_velocity_std = 1.0
min_cameras_for_init = 2

[tracking.ukf]
alpha = 0.1
beta = 2.0
kappa = 0.0

[calibration]
enabled = true
prismatic_process_noise_std = 0.0001

[output]
directory = "{output_dir}"
export_tracking_results = true
export_statistics = true

[processing]
start_time = 0.0
end_time = {end_time}
tracker_fps = {TRACKER_FPS}
"""


def run(cmd: list[str], cwd: Path | None = None) -> bool:
    """Run a command, printing it first. Returns True on success."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT)
    if result.returncode != 0:
        print(f"  [FAILED] exit code {result.returncode}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def process_sequence(
    seq_dir: Path,
    date_str: str,
    calib_toml: Path,
    dry_run: bool,
) -> None:
    print(f"\n{'='*70}")
    print(f"Sequence: {seq_dir.name}")
    print(f"{'='*70}")

    pose_dir  = seq_dir / "pose"
    sync_json = seq_dir / "sync_data.json"

    if not pose_dir.exists():
        print(f"  [skip] No pose/ directory found.", file=sys.stderr)
        return
    if not sync_json.exists():
        print(f"  [skip] No sync_data.json found — cannot run tracker.", file=sys.stderr)
        return

    end_time = detect_end_time(sync_json)
    print(f"  Auto-detected end_time: {end_time}s")

    # Determine version
    posetrak_dir = seq_dir / "posetrak"
    version      = next_version(posetrak_dir, date_str)
    version_dir  = posetrak_dir / f"{date_str}-{version}"
    print(f"  Output version dir: {version_dir}")

    if not dry_run:
        version_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Generate configs and run tracker for each person
    # -----------------------------------------------------------------------
    successful_tracking: list[tuple[str, Path, str]] = []  # (name, tracking_dir, skeleton_rel)

    for person_name, person_id, skeleton_rel in PERSONS:
        print(f"\n  -- Person: {person_name} (id={person_id}) --")
        tracking_dir = version_dir / person_name
        toml_content = generate_toml(
            skeleton_rel=skeleton_rel,
            cameras=calib_toml,
            observations_dir=pose_dir,
            sync_json=sync_json,
            person_id=person_id,
            output_dir=tracking_dir,
            end_time=end_time,
        )

        # Write config to version dir
        config_path = version_dir / f"{person_name}.toml"
        print(f"  Writing config: {config_path}")
        if not dry_run:
            config_path.write_text(toml_content)

        # Run tracker
        tracker_bin = PROJECT_ROOT / "optbuild" / "cli" / "posetrak"
        success = dry_run or run(
            [str(tracker_bin), "track", "--smooth", str(config_path)],
            cwd=PROJECT_ROOT,
        )

        if not success:
            print(f"  [warn] Tracking failed for {person_name} — skipping BVH export",
                  file=sys.stderr)
            continue

        successful_tracking.append((person_name, tracking_dir, skeleton_rel))

        # Export BVH
        skeleton_path = PROJECT_ROOT / skeleton_rel
        bvh_name      = f"{person_name}-{date_str}-{version}.bvh"
        bvh_path      = seq_dir / bvh_name
        print(f"  Exporting BVH → {bvh_path}")
        bvh_ok = dry_run or run([
            "uv", "run", "python", "scripts/export_bvh.py",
            str(tracking_dir),
            "--skeleton", str(skeleton_path),
            "--fps", "120",
            "--units", "cm",
            "--coord", "yup",
            "--smoothed",
            "--output", str(bvh_path),
        ], cwd=PROJECT_ROOT)
        if not bvh_ok:
            print(f"  [warn] BVH export failed for {person_name}", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Visualization (only if at least one person tracked successfully)
    # -----------------------------------------------------------------------
    if successful_tracking:
        print(f"\n  -- Visualization --")
        video_dir = seq_dir / "videos"
        if not video_dir.exists():
            print(f"  [warn] No videos/ directory — skipping visualization", file=sys.stderr)
        else:
            mosaic_path = version_dir / "mosaic.mp4"
            viz_cmd = [
                "python3", "scripts/visualize_tracking.py",
            ]
            for person_name, tracking_dir, _ in successful_tracking:
                viz_cmd += ["--tracking-dir", str(tracking_dir)]
            viz_cmd += [
                "--cameras",  str(calib_toml),
                "--sync",     str(sync_json),
                "--video-dir", str(video_dir),
                "--output",   str(mosaic_path),
            ]
            if dry_run:
                print(f"  $ {' '.join(str(c) for c in viz_cmd)}")
            else:
                run(viz_cmd, cwd=PROJECT_ROOT)

    # -----------------------------------------------------------------------
    # Copy skeleton files to version dir
    # -----------------------------------------------------------------------
    print(f"\n  -- Copying skeleton files --")
    for person_name, _tracking_dir, skeleton_rel in PERSONS:
        src = PROJECT_ROOT / skeleton_rel
        dst = version_dir / src.name
        print(f"  {src.name} → {dst}")
        if not dry_run and src.exists():
            shutil.copy2(src, dst)

    print(f"\n  Done: {version_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-process mocap sessions with posetrak.")
    p.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT,
                   help=f"Root directory containing sequence dirs (default: {DEFAULT_SESSION_ROOT})")
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB_TOML,
                   help="Camera calibration TOML (default: <session-root>/calibration/calib.toml)")
    p.add_argument("--date", default=str(date.today()),
                   help="Date string for versioning (default: today, YYYY-MM-DD)")
    p.add_argument("--sequences", nargs="+", metavar="NAME",
                   help="Process only these sequence subdirectories (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without executing anything")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    session_root = args.session_root
    calib_toml   = args.calib if args.calib != DEFAULT_CALIB_TOML else \
                   session_root / "calibration" / "calib.toml"
    date_str     = args.date

    print(f"Session root : {session_root}")
    print(f"Calib TOML  : {calib_toml}")
    print(f"Date        : {date_str}")
    if args.dry_run:
        print("DRY RUN — no files will be written or commands executed")

    if not calib_toml.exists():
        print(f"[error] Calibration TOML not found: {calib_toml}", file=sys.stderr)
        sys.exit(1)

    sequences = find_sequences(session_root, args.sequences)
    if not sequences:
        print("[error] No sequence directories found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(sequences)} sequence(s):")
    for s in sequences:
        sync_ok = (s / "sync_data.json").exists()
        pose_ok = (s / "pose").exists()
        status  = "ok" if (sync_ok and pose_ok) else \
                  "no sync" if (pose_ok and not sync_ok) else \
                  "no pose" if (sync_ok and not pose_ok) else "no pose/sync"
        print(f"  {s.name}  [{status}]")

    for seq in sequences:
        process_sequence(seq, date_str, calib_toml, dry_run=args.dry_run)

    print("\nAll done.")


if __name__ == "__main__":
    main()

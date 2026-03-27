#!/usr/bin/env python3
"""
run_project.py — Batch-track all shots/persons in a session DB, then export
BVH and visualization videos.

For every pose_observation_sequence in the session DB and for every performer
listed on the command line, this script:

  1. Imports the performer's skeleton into the session DB (idempotent).
  2. Creates (or reuses) a tracker config with the given noise parameters.
  3. Runs  ``posetrak track --session-db ...`` for each (sequence × person).
  4. Optionally exports a BVH file via export_bvh.py.
  5. Optionally renders a visualization video via visualize_tracking.py.

Usage
-----
    uv run python/tools/run_project.py \\
        --session-db /mnt/d/mocap/2026-03-10-test/session.db \\
        --person harri 0 /mnt/d/mocap/skeletons/harri.yaml \\
        --person timo  1 /mnt/d/mocap/skeletons/timo.yaml \\
        [--tracker-config <id-or-prefix>]  # reuse an existing config snapshot
        [--process-noise-std       0.1]
        [--process-noise-vel-std   0.5]
        [--velocity-half-life-s    0.25]
        [--measurement-noise-std   60.0]
        [--outlier-threshold       4.0]
        [--tracker-fps             120.0]
        [--joint-groups main HandL HandR]
        [--binary     optbuild/cli/posetrak]
        [--export-bvh]
        [--visualize]
        [--out-dir /mnt/d/mocap/2026-03-10-test/posetrak_results]
        [--dry-run]

Each tracking run, BVH file, and visualization video is written under:
  <out-dir>/<shot-label>/<person-name>/
    tracking/          (CSV outputs from the tracker)
    <label>_<person>.bvh
    <label>_<person>.mp4
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "python"))

from posetrak.db.db import open_session, resolve_id_prefix
from posetrak.db.manage_config import create_config_from_toml, edit_config
from posetrak.db.manage_skeleton import import_skeleton

_TOOLS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Person:
    name: str
    person_id: int
    skeleton_path: Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_sequences(db: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all pose_observation_sequences with their shot label."""
    return db.execute(
        """
        SELECT pos.id AS sequence_id,
               pos.time_start_s, pos.time_end_s,
               sh.id  AS shot_id,
               sh.label AS shot_label,
               sh.shot_number
        FROM pose_observation_sequences pos
        JOIN shots sh ON sh.id = pos.shot_id
        ORDER BY sh.shot_number, pos.time_start_s
        """
    ).fetchall()


def _video_dir_for_shot(db: sqlite3.Connection, shot_id: str) -> Path | None:
    """Infer the video directory from the first shot_video file_path."""
    row = db.execute(
        "SELECT file_path FROM shot_videos WHERE shot_id = ? LIMIT 1",
        (shot_id,),
    ).fetchone()
    if row is None:
        return None
    return Path(row[0]).parent


def _find_or_create_config(
    db: sqlite3.Connection,
    *,
    base_config_id: str | None,
    process_noise_std: float | None,
    process_noise_vel_std: float | None,
    velocity_half_life_s: float | None,
    measurement_noise_std: float | None,
    outlier_threshold: float | None,
    tracker_fps: float | None,
    config_name: str,
) -> str:
    """Return the ID of a tracker config to use.

    If base_config_id is given, derive a new config from it with any supplied
    overrides. Otherwise create a fresh config row with the supplied params.
    """
    if base_config_id:
        full_id = resolve_id_prefix(db, "tracker_configs", base_config_id)
        return edit_config(
            db, full_id,
            process_noise_std=process_noise_std,
            process_noise_vel_std=process_noise_vel_std,
            velocity_half_life_s=velocity_half_life_s,
            measurement_noise_std=measurement_noise_std,
            outlier_threshold=outlier_threshold,
            tracker_fps=tracker_fps,
            notes=f"derived for {config_name}",
        )

    # No base config — insert a new row with the supplied parameters.
    # edit_config needs an existing parent; we do a direct INSERT instead.
    from posetrak.db.db import generate_id
    import datetime
    config_id = generate_id()
    with db:
        db.execute(
            "INSERT INTO tracker_configs "
            "(id, name, parent_id, created_at, "
            " process_noise_std, process_noise_vel_std, velocity_half_life_s, "
            " measurement_noise_std, outlier_threshold, tracker_fps) "
            "VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)",
            (
                config_id,
                config_name,
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                process_noise_std,
                process_noise_vel_std,
                velocity_half_life_s,
                measurement_noise_std,
                outlier_threshold,
                tracker_fps,
            ),
        )
    return config_id


def _run(cmd: list[str], dry_run: bool) -> tuple[bool, str]:
    """Run a command, return (success, stdout). Prints the command first."""
    display = " ".join(str(c) for c in cmd)
    print(f"  $ {display}")
    if dry_run:
        return True, ""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAILED] exit={result.returncode}", file=sys.stderr)
        if result.stderr:
            for line in result.stderr.splitlines()[-10:]:
                print(f"    {line}", file=sys.stderr)
        return False, result.stdout
    return True, result.stdout


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--session-db", required=True, type=Path)
    ap.add_argument(
        "--person", nargs=3, action="append", metavar=("NAME", "PERSON_ID", "SKELETON"),
        dest="persons", required=True,
        help="Performer: name, person_id (int), path to skeleton YAML. "
             "Repeat for multiple performers.",
    )
    # Tracker config
    ap.add_argument("--tracker-config", default=None, metavar="ID",
                    help="Existing tracker_config ID/prefix to derive from. "
                         "If omitted, a new config is created from the noise params.")
    ap.add_argument("--process-noise-std",     type=float, default=0.1)
    ap.add_argument("--process-noise-vel-std", type=float, default=0.5)
    ap.add_argument("--velocity-half-life-s",  type=float, default=0.25)
    ap.add_argument("--measurement-noise-std", type=float, default=60.0)
    ap.add_argument("--outlier-threshold",     type=float, default=4.0)
    ap.add_argument("--tracker-fps",           type=float, default=120.0)
    ap.add_argument("--joint-groups", nargs="+", default=None,
                    metavar="GROUP",
                    help="Active joint groups (default: all groups in skeleton). "
                         "Example: main HandL HandR")
    # Execution
    ap.add_argument("--binary", default="optbuild/cli/posetrak", type=Path,
                    help="Path to posetrak binary [default: optbuild/cli/posetrak]")
    ap.add_argument("--export-bvh", action="store_true",
                    help="Export a BVH file for each tracking run.")
    ap.add_argument("--visualize", action="store_true",
                    help="Render a mosaic visualization video for each tracking run.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output root directory. Default: <session-db-dir>/posetrak_results")
    ap.add_argument("--shots", nargs="+", default=None, metavar="LABEL",
                    help="Only process these shot labels (default: all shots).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing them.")
    args = ap.parse_args()

    db_path = args.session_db.resolve()
    if not db_path.exists():
        print(f"error: session DB not found: {db_path}", file=sys.stderr)
        return 1

    binary = Path(args.binary)
    if not binary.is_absolute():
        binary = (_REPO_ROOT / binary).resolve()
    if not binary.exists():
        print(f"error: posetrak binary not found: {binary}", file=sys.stderr)
        return 1

    out_dir = (args.out_dir or db_path.parent / "posetrak_results").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    persons = [
        Person(name=p[0], person_id=int(p[1]), skeleton_path=Path(p[2]).resolve())
        for p in args.persons
    ]
    for p in persons:
        if not p.skeleton_path.exists():
            print(f"error: skeleton not found for {p.name}: {p.skeleton_path}",
                  file=sys.stderr)
            return 1

    # ------------------------------------------------------------------
    # Open session DB
    # ------------------------------------------------------------------
    db = open_session(db_path)
    db.row_factory = sqlite3.Row

    # ------------------------------------------------------------------
    # Import skeletons (idempotent — same YAML → same SHA-256 ID)
    # ------------------------------------------------------------------
    print("Importing skeletons ...")
    skeleton_ids: dict[str, str] = {}
    for p in persons:
        skel_id = import_skeleton(db, p.skeleton_path,
                                  name=p.skeleton_path.stem,
                                  person_label=p.name)
        skeleton_ids[p.name] = skel_id
        print(f"  {p.name}: {skel_id[:16]}  ({p.skeleton_path.name})")

    # ------------------------------------------------------------------
    # Create tracker config
    # ------------------------------------------------------------------
    print("\nCreating tracker config ...")
    config_id = _find_or_create_config(
        db,
        base_config_id=args.tracker_config,
        process_noise_std=args.process_noise_std,
        process_noise_vel_std=args.process_noise_vel_std,
        velocity_half_life_s=args.velocity_half_life_s,
        measurement_noise_std=args.measurement_noise_std,
        outlier_threshold=args.outlier_threshold,
        tracker_fps=args.tracker_fps,
        config_name="run_project",
    )
    print(f"  Config ID: {config_id[:16]}")
    print(f"  process_noise_std={args.process_noise_std}  "
          f"process_noise_vel_std={args.process_noise_vel_std}  "
          f"velocity_half_life_s={args.velocity_half_life_s}")
    print(f"  measurement_noise_std={args.measurement_noise_std}  "
          f"outlier_threshold={args.outlier_threshold}  "
          f"tracker_fps={args.tracker_fps}")

    # ------------------------------------------------------------------
    # Discover sequences
    # ------------------------------------------------------------------
    sequences = _list_sequences(db)
    if args.shots:
        sequences = [s for s in sequences if s["shot_label"] in args.shots]
    if not sequences:
        print("No pose_observation_sequences found in the session DB.", file=sys.stderr)
        return 1
    print(f"\nFound {len(sequences)} sequence(s) across "
          f"{len({s['shot_id'] for s in sequences})} shot(s).")

    # ------------------------------------------------------------------
    # Track each sequence × person
    # ------------------------------------------------------------------
    failed: list[str] = []
    succeeded: list[tuple[str, str, str]] = []  # (shot_label, person_name, run_id)

    for seq in sequences:
        shot_label  = seq["shot_label"] or f"shot{seq['shot_number']:03d}"
        sequence_id = seq["sequence_id"]
        shot_id     = seq["shot_id"]
        video_dir   = _video_dir_for_shot(db, shot_id)

        print(f"\n{'='*70}")
        print(f"Shot: {shot_label}  |  sequence: {sequence_id[:12]}")

        for p in persons:
            print(f"\n  -- {p.name} (person_id={p.person_id}) --")
            skel_id  = skeleton_ids[p.name]
            run_dir  = out_dir / shot_label / p.name / "tracking"
            run_dir.mkdir(parents=True, exist_ok=True)

            # ── Track ────────────────────────────────────────────────
            track_cmd = [
                str(binary), "track",
                "--session-db",      str(db_path),
                "--sequence",        sequence_id,
                "--skeleton",        skel_id,
                "--tracker-config",  config_id,
                "--person-id",       str(p.person_id),
                "--output-dir",      str(run_dir),
                "--smooth",
                "--quiet",
            ]
            if args.joint_groups:
                track_cmd += ["--joint-groups"] + args.joint_groups

            ok, stdout = _run(track_cmd, args.dry_run)
            if not ok:
                failed.append(f"{shot_label}/{p.name}: tracking failed")
                continue

            # Parse run ID from stdout
            run_id = ""
            if not args.dry_run:
                m = re.search(r"tracking_run_id:\s*(\S+)", stdout)
                if not m:
                    print("  [warn] Could not parse tracking_run_id from output")
                    failed.append(f"{shot_label}/{p.name}: no run_id in output")
                    continue
                run_id = m.group(1)
                print(f"  Tracking run: {run_id[:16]}")
            succeeded.append((shot_label, p.name, run_id))

            # ── BVH export ───────────────────────────────────────────
            if args.export_bvh:
                bvh_path = out_dir / shot_label / p.name / f"{shot_label}_{p.name}.bvh"
                bvh_cmd = [
                    sys.executable, str(_TOOLS_DIR / "export_bvh.py"),
                    "--session-db", str(db_path),
                    "--run-id",     run_id,
                    "--person-id",  str(p.person_id),
                    "--skeleton",   str(p.skeleton_path),
                    "--smoothed",
                    "--output",     str(bvh_path),
                ]
                ok, _ = _run(bvh_cmd, args.dry_run)
                if not ok:
                    failed.append(f"{shot_label}/{p.name}: BVH export failed")

            # ── Visualization ────────────────────────────────────────
            if args.visualize:
                if video_dir is None:
                    print(f"  [warn] No video files found for shot {shot_label}; "
                          "skipping visualization.")
                else:
                    viz_path = out_dir / shot_label / p.name / f"{shot_label}_{p.name}.mp4"
                    viz_cmd = [
                        sys.executable, str(_TOOLS_DIR / "visualize_tracking.py"),
                        "--session-db", str(db_path),
                        "--run-id",     run_id,
                        "--person-id",  str(p.person_id),
                        "--video-dir",  str(video_dir),
                        "--output",     str(viz_path),
                    ]
                    ok, _ = _run(viz_cmd, args.dry_run)
                    if not ok:
                        failed.append(f"{shot_label}/{p.name}: visualization failed")

    db.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Completed: {len(succeeded)} run(s)  |  Failed: {len(failed)}")
    if succeeded and not args.dry_run:
        print("\nSuccessful runs:")
        for shot_label, pname, run_id in succeeded:
            print(f"  {shot_label}/{pname}: {run_id}")
    if failed:
        print("\nFailed steps:")
        for msg in failed:
            print(f"  {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

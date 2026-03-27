#!/usr/bin/env python3
"""
import_project.py — Import an existing mocap project directory into a session DB.

Expected project directory layout:

    <project-dir>/
      calibration/
        intrinsics.toml     # Pose2Sim TOML: intrinsics + extrinsics for all cameras
      <shot1>/              # any subdir with pose/ + sync_data.json is treated as a shot
        pose/
          cam1/
            cam1_000001.json
            ...
          cam2/
            ...
        videos/
          cam1.mp4
          cam2.mp4
          ...
        sync_data.json
      <shot2>/
        ...

Usage
-----
    uv run python/tools/import_project.py \\
        --project-dir /mnt/d/mocap/2026-03-10-posetrak-test \\
        --session-db  /mnt/d/mocap/2026-03-10-posetrak-test/session.db \\
        [--calib      calibration/intrinsics.toml]   # relative to project-dir
        [--camera-model "GoPro Hero 12"]             # hardware description (informational)
        [--camera-mode  "1080p120"]
        [--session-label "2026-03-10 test"]
        [--recorded-at 2026-03-10]                   # ISO date; default: dir mtime
        [--dry-run]

The script creates the session DB (fails if it already exists) and imports
all shots it discovers. Camera model/mode entries are created automatically
in the session DB using the names supplied (or generic defaults). Re-running
with --dry-run shows what would be imported without writing anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve package root so this script works without 'pip install -e .'
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "python"))

from posetrak.db.db import (
    add_session_camera,
    add_shot_video,
    create_camera_model,
    create_camera_mode,
    create_mocap_session,
    create_session,
    create_shot,
    open_session,
)
from posetrak.db.import_calib_toml import import_calib_toml
from posetrak.db.import_extrinsics import import_extrinsics
from posetrak.db.import_pose_json import import_pose_json
from posetrak.db.import_sync_json import import_sync_json

SKIP_DIRS = {"calibration", "posetrak_config", "posetrak_results"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_shot_dirs(project_dir: Path) -> list[Path]:
    """Return all shot subdirectories sorted by name."""
    shots = []
    for d in sorted(project_dir.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name in SKIP_DIRS:
            continue
        if (d / "pose").is_dir() and (d / "sync_data.json").exists():
            shots.append(d)
    return shots


def _read_sync_fps(sync_json: Path) -> dict[str, float]:
    """Return {cam_key: fps} mapping from a sync_data.json file."""
    with open(sync_json) as f:
        data = json.load(f)
    return {k: float(v["fps"]) for k, v in data.items() if "fps" in v}


def _sync_frame_range(sync_json: Path, cam_key: str) -> tuple[int, int]:
    """Return (first_frame, last_frame) for a camera from sync_data.json."""
    with open(sync_json) as f:
        data = json.load(f)
    pts = data.get(cam_key, {}).get("syncpoints", [])
    if not pts:
        return 0, 0
    frames = [p["frame"] for p in pts]
    return min(frames), max(frames)


def _detect_videos(videos_dir: Path) -> dict[str, Path]:
    """Return {cam_key: video_path} by matching stem to cam key pattern."""
    result: dict[str, Path] = {}
    if not videos_dir.is_dir():
        return result
    for f in sorted(videos_dir.iterdir()):
        if f.suffix.lower() in VIDEO_EXTENSIONS:
            result[f.stem] = f
    return result


def _toml_camera_labels(calib_path: Path) -> list[str]:
    """Return camera section keys (e.g. ['cam1','cam2']) from a Pose2Sim TOML."""
    with open(calib_path, "rb") as fh:
        data = tomllib.load(fh)
    return sorted(k for k in data if k.lower().startswith("cam"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--project-dir", required=True, type=Path,
                    help="Root directory of the mocap project.")
    ap.add_argument("--session-db", required=True, type=Path,
                    help="Path for the new session DB (must not already exist).")
    ap.add_argument("--calib", default="calibration/intrinsics.toml",
                    help="Calibration TOML path, relative to --project-dir "
                         "[default: calibration/intrinsics.toml]")
    ap.add_argument("--camera-model", default="",
                    help="Camera model description stored in the DB (informational).")
    ap.add_argument("--camera-mode", default="",
                    help="Camera mode description stored in the DB (informational).")
    ap.add_argument("--session-label", default="",
                    help="Human-readable session label (stored in notes).")
    ap.add_argument("--recorded-at", default=None,
                    help="ISO date of the recording session (default: project dir mtime).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be imported without writing anything.")
    args = ap.parse_args()

    project_dir = args.project_dir.resolve()
    calib_path  = (project_dir / args.calib).resolve()
    db_path     = args.session_db.resolve()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not project_dir.is_dir():
        print(f"error: project directory not found: {project_dir}", file=sys.stderr)
        return 1
    if not calib_path.exists():
        print(f"error: calibration TOML not found: {calib_path}", file=sys.stderr)
        return 1

    cam_labels = _toml_camera_labels(calib_path)
    if not cam_labels:
        print("error: no camera sections found in calibration TOML", file=sys.stderr)
        return 1

    shot_dirs = _detect_shot_dirs(project_dir)
    if not shot_dirs:
        print("warning: no shot directories found (need pose/ + sync_data.json)",
              file=sys.stderr)

    # Recorded-at: use argument or directory modification time
    if args.recorded_at:
        recorded_at = args.recorded_at
    else:
        import datetime
        ts = project_dir.stat().st_mtime
        recorded_at = datetime.datetime.fromtimestamp(ts).date().isoformat()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"Project dir  : {project_dir}")
    print(f"Calibration  : {calib_path}")
    print(f"Session DB   : {db_path}")
    print(f"Recorded at  : {recorded_at}")
    print(f"Cameras      : {', '.join(cam_labels)}")
    print(f"Shots found  : {len(shot_dirs)}")
    for d in shot_dirs:
        print(f"  {d.name}")
    if args.dry_run:
        print("\n[dry-run] No changes written.")
        return 0

    # ------------------------------------------------------------------
    # Create session DB
    # ------------------------------------------------------------------
    try:
        db = create_session(db_path)
    except FileExistsError:
        print(f"error: session DB already exists: {db_path}\n"
              "Delete it first or choose a different path.", file=sys.stderr)
        return 1

    print(f"\nCreated session DB: {db_path}")

    # The session DB includes all registry tables (camera_models, camera_modes,
    # camera_instances, intrinsics_calibrations, skeletons, tracker_configs).
    # We use it as both registry and session throughout.

    # ------------------------------------------------------------------
    # Camera model + mode (one shared set for this import)
    # ------------------------------------------------------------------
    model_name = args.camera_model or "generic"
    mode_name  = args.camera_mode  or "default"
    camera_model_id = create_camera_model(
        db, manufacturer="", model_name=model_name
    )
    camera_mode_id = create_camera_mode(
        db, camera_model_id, notes=mode_name
    )
    print(f"Camera model : {model_name!r} → {camera_model_id[:8]}")
    print(f"Camera mode  : {mode_name!r}  → {camera_mode_id[:8]}")

    # ------------------------------------------------------------------
    # Import intrinsics (creates camera_instances in the session DB)
    # ------------------------------------------------------------------
    print(f"\nImporting intrinsics from {calib_path.name} ...")
    calib_result = import_calib_toml(
        db,
        calib_path,
        camera_mode_id,
        notes=f"imported from {calib_path}",
    )
    if calib_result.skipped:
        print(f"  Skipped TOML sections: {calib_result.skipped}")
    for label, inst_id in calib_result.camera_instance_ids.items():
        intr_id = calib_result.intrinsics_ids[label]
        print(f"  {label}: instance={inst_id[:8]}  intrinsics={intr_id[:8]}")

    # ------------------------------------------------------------------
    # Create mocap session
    # ------------------------------------------------------------------
    notes = args.session_label or project_dir.name
    session_id = create_mocap_session(
        db, recorded_at=recorded_at, notes=notes
    )
    print(f"\nCreated session: {session_id[:8]}  ({notes!r})")

    # ------------------------------------------------------------------
    # Add session cameras
    # ------------------------------------------------------------------
    for cam_label, inst_id in calib_result.camera_instance_ids.items():
        intr_id = calib_result.intrinsics_ids[cam_label]
        add_session_camera(
            db, db,  # session == registry (session DB embeds registry tables)
            session_id, inst_id, camera_mode_id, intr_id,
            label=cam_label,
        )
    print(f"Added {len(calib_result.camera_instance_ids)} session cameras.")

    # ------------------------------------------------------------------
    # Import extrinsics (one shared calibration for all shots)
    # ------------------------------------------------------------------
    print(f"\nImporting extrinsics ...")
    extr_result = import_extrinsics(
        db, session_id, calib_path,
        calib_result.camera_instance_ids,
        registry=db,
    )
    extr_id = extr_result.extrinsic_calibration_id
    print(f"  Extrinsic calibration: {extr_id[:8]}")
    if extr_result.skipped:
        print(f"  Skipped: {extr_result.skipped}")

    # ------------------------------------------------------------------
    # Per-shot import
    # ------------------------------------------------------------------
    for shot_dir in shot_dirs:
        _import_shot(
            db=db,
            shot_dir=shot_dir,
            session_id=session_id,
            extrinsic_calibration_id=extr_id,
            camera_instance_ids=calib_result.camera_instance_ids,
        )

    db.close()
    print(f"\nDone. Session DB: {db_path}")
    return 0


def _import_shot(
    *,
    db,
    shot_dir: Path,
    session_id: str,
    extrinsic_calibration_id: str,
    camera_instance_ids: dict[str, str],
) -> None:
    print(f"\n{'─'*60}")
    print(f"Shot: {shot_dir.name}")

    sync_json  = shot_dir / "sync_data.json"
    pose_dir   = shot_dir / "pose"
    videos_dir = shot_dir / "videos"

    fps_per_cam = _read_sync_fps(sync_json)

    # Create shot
    shot_id = create_shot(
        db, session_id,
        extrinsic_calibration_id=extrinsic_calibration_id,
        label=shot_dir.name,
    )
    print(f"  Created shot: {shot_id[:8]}")

    # Add video files
    video_files = _detect_videos(videos_dir)
    for cam_label, inst_id in camera_instance_ids.items():
        video_path = video_files.get(cam_label)
        if video_path is None:
            print(f"  [warn] No video found for {cam_label} in {videos_dir}")
            continue
        fps = fps_per_cam.get(cam_label, 0.0)
        first_frame, last_frame = _sync_frame_range(sync_json, cam_label)
        add_shot_video(
            db, shot_id, inst_id,
            str(video_path), first_frame, last_frame, fps,
        )
        print(f"  Added video: {cam_label} → {video_path.name} "
              f"(frames {first_frame}–{last_frame}, {fps:.2f} fps)")

    # Import sync
    sync_result = import_sync_json(
        db, shot_id, sync_json,
        camera_instance_ids,
        notes=f"{shot_dir.name}",
    )
    print(f"  Sync config: {sync_result.sync_config_id[:8]}"
          f" ({len(sync_result.camera_instance_ids)} cameras)")
    if sync_result.skipped:
        print(f"  Sync skipped cameras: {sync_result.skipped}")

    # Import pose JSON
    pose_result = import_pose_json(
        db, shot_id,
        sync_result.sync_config_id,
        pose_dir,
        camera_instance_ids,
    )
    print(f"  Pose sequence: {pose_result.sequence_id[:8]}"
          f" ({pose_result.n_observations:,} observations)")
    if pose_result.skipped_cameras:
        print(f"  Pose skipped cameras: {pose_result.skipped_cameras}")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""
import_project.py — Import an existing mocap project directory into a session DB.

Expected project directory layout:

    <project-dir>/
      cameras.toml            # camera hardware description (optional but recommended)
      calibration/
        calib.toml       # Pose2Sim TOML: intrinsics + extrinsics for all cameras
      <shot1>/                # any subdir with pose/ + sync_data.json is treated as a shot
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

cameras.toml format
-------------------
One section per camera label (matching the keys in the calibration TOML and
sync_data.json).  ``model`` and ``mode`` are free-text strings; ``model`` is
used to find or create a ``camera_models`` row so cameras of the same hardware
type share one entry across imports.

    [cam1]
    model = "GoPro Hero 11 Black"
    mode  = "4K Linear 120fps"

    [cam2]
    model = "GoPro Hero 11 Black"
    mode  = "4K Linear 120fps"

    [cam3]
    model = "Sony ZV-E10"
    mode  = "1080p 120fps"

If cameras.toml is absent the script falls back to --camera-model /
--camera-mode (applied to all cameras).

Usage
-----
    uv run python/tools/import_project.py \\
        --project-dir /mnt/d/mocap/2026-03-10-posetrak-test \\
        --session-db  /mnt/d/mocap/2026-03-10-posetrak-test/session.db \\
        [--calib      calibration/calib.toml]   # relative to project-dir
        [--cameras    cameras.toml]                  # relative to project-dir
        [--camera-model "GoPro Hero 12"]             # fallback if cameras.toml absent
        [--camera-mode  "1080p120"]                  # fallback if cameras.toml absent
        [--session-label "2026-03-10 test"]
        [--recorded-at 2026-03-10]                   # ISO date; default: dir mtime
        [--dry-run]

The script creates the session DB (fails if it already exists). Use --dry-run
to preview what would be imported without writing anything.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tomllib
from pathlib import Path

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
    seed_bundled_defaults,
)
from posetrak.db.import_calib_toml import import_calib_toml
from posetrak.db.import_extrinsics import import_extrinsics
from posetrak.db.import_pose_json import import_pose_json
from posetrak.db.import_sync_json import import_sync_json

SKIP_DIRS = {"calibration", "posetrak_config", "posetrak_results"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ---------------------------------------------------------------------------
# Camera model/mode resolution
# ---------------------------------------------------------------------------

def _find_or_create_camera_model(db: sqlite3.Connection, model_name: str) -> str:
    """Return the ID of a camera_models row matching model_name, creating if absent."""
    row = db.execute(
        "SELECT id FROM camera_models WHERE model_name = ? LIMIT 1",
        (model_name,),
    ).fetchone()
    if row:
        return row[0]
    return create_camera_model(db, model_name=model_name)


def _find_or_create_camera_mode(
    db: sqlite3.Connection, camera_model_id: str, mode_name: str
) -> str:
    """Return the ID of a camera_modes row matching (model_id, notes), creating if absent."""
    row = db.execute(
        "SELECT id FROM camera_modes WHERE camera_model_id = ? AND notes = ? LIMIT 1",
        (camera_model_id, mode_name),
    ).fetchone()
    if row:
        return row[0]
    return create_camera_mode(db, camera_model_id, notes=mode_name)


def _build_camera_mode_map(
    db: sqlite3.Connection,
    cam_labels: list[str],
    cameras_toml: Path | None,
    fallback_model: str,
    fallback_mode: str,
) -> dict[str, str]:
    """Return {cam_label: camera_mode_id} for every label in cam_labels.

    If cameras_toml is provided it is used to look up per-camera model/mode.
    Labels missing from the TOML fall back to the fallback_model/mode values.
    """
    per_cam: dict[str, tuple[str, str]] = {}  # label → (model_name, mode_name)

    if cameras_toml and cameras_toml.exists():
        with open(cameras_toml, "rb") as fh:
            toml_data = tomllib.load(fh)
        for label in cam_labels:
            section = toml_data.get(label, {})
            model = section.get("model", fallback_model) or fallback_model
            mode  = section.get("mode",  fallback_mode)  or fallback_mode
            per_cam[label] = (model, mode)
    else:
        for label in cam_labels:
            per_cam[label] = (fallback_model, fallback_mode)

    # Find or create model/mode rows (deduplicated across cameras sharing hardware)
    mode_id_cache: dict[tuple[str, str], str] = {}
    result: dict[str, str] = {}
    for label, (model_name, mode_name) in per_cam.items():
        key = (model_name, mode_name)
        if key not in mode_id_cache:
            model_id = _find_or_create_camera_model(db, model_name)
            mode_id  = _find_or_create_camera_mode(db, model_id, mode_name)
            mode_id_cache[key] = mode_id
        result[label] = mode_id_cache[key]

    return result


# ---------------------------------------------------------------------------
# Project directory helpers
# ---------------------------------------------------------------------------

def _detect_shot_dirs(project_dir: Path) -> list[Path]:
    shots = []
    for d in sorted(project_dir.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name in SKIP_DIRS:
            continue
        if (d / "pose").is_dir() and (d / "sync_data.json").exists():
            shots.append(d)
    return shots


def _read_sync_fps(sync_json: Path) -> dict[str, float]:
    with open(sync_json) as f:
        data = json.load(f)
    return {k: float(v["fps"]) for k, v in data.items() if "fps" in v}


def _sync_frame_range(sync_json: Path, cam_key: str) -> tuple[int, int]:
    with open(sync_json) as f:
        data = json.load(f)
    pts = data.get(cam_key, {}).get("syncpoints", [])
    if not pts:
        return 0, 0
    frames = [p["frame"] for p in pts]
    return min(frames), max(frames)


def _detect_videos(videos_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if not videos_dir.is_dir():
        return result
    for f in sorted(videos_dir.iterdir()):
        if f.suffix.lower() in VIDEO_EXTENSIONS:
            result[f.stem] = f
    return result


def _toml_camera_labels(calib_path: Path) -> list[str]:
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
    ap.add_argument("--project-dir", required=True, type=Path)
    ap.add_argument("--session-db",  required=True, type=Path,
                    help="Path for the new session DB (must not already exist).")
    ap.add_argument("--calib", default="calibration/calib.toml",
                    help="Calibration TOML, relative to --project-dir "
                         "[default: calibration/calib.toml]")
    ap.add_argument("--cameras", default=None,
                    help="Camera hardware TOML, relative to --project-dir "
                         "[default: cameras.toml if it exists]")
    ap.add_argument("--camera-model", default="generic",
                    help="Fallback camera model name when cameras.toml is absent "
                         "[default: generic]")
    ap.add_argument("--camera-mode", default="default",
                    help="Fallback camera mode name when cameras.toml is absent "
                         "[default: default]")
    ap.add_argument("--session-label", default="",
                    help="Human-readable session label (stored in notes).")
    ap.add_argument("--recorded-at", default=None,
                    help="ISO date of the recording (default: project dir mtime).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be imported without writing anything.")
    args = ap.parse_args()

    project_dir = args.project_dir.resolve()
    calib_path  = (project_dir / args.calib).resolve()
    db_path     = args.session_db.resolve()

    # Resolve cameras TOML: explicit arg > default location > absent
    if args.cameras:
        cameras_toml = (project_dir / args.cameras).resolve()
    else:
        cameras_toml = project_dir / "cameras.toml"
    if not cameras_toml.exists():
        cameras_toml = None

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
    print(f"Cameras TOML : {cameras_toml or '(none — using fallback)'}")
    print(f"Session DB   : {db_path}")
    print(f"Recorded at  : {recorded_at}")
    print(f"Cameras      : {', '.join(cam_labels)}")
    print(f"Shots found  : {len(shot_dirs)}")
    for d in shot_dirs:
        print(f"  {d.name}")

    if cameras_toml:
        with open(cameras_toml, "rb") as fh:
            cam_cfg = tomllib.load(fh)
        print("Camera hardware:")
        for label in cam_labels:
            sec = cam_cfg.get(label, {})
            model = sec.get("model", args.camera_model)
            mode  = sec.get("mode",  args.camera_mode)
            print(f"  {label}: {model!r} / {mode!r}")

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
    seed_bundled_defaults(db)

    # The session DB includes all registry tables (camera_models, camera_modes,
    # camera_instances, intrinsics_calibrations, skeletons, tracker_configs).
    # We use it as both registry and session throughout.

    # ------------------------------------------------------------------
    # Camera models & modes (per-camera from cameras.toml or shared fallback)
    # ------------------------------------------------------------------
    camera_mode_ids = _build_camera_mode_map(
        db, cam_labels, cameras_toml, args.camera_model, args.camera_mode
    )
    print("Camera model/mode IDs:")
    for label, mode_id in camera_mode_ids.items():
        mode_row = db.execute(
            "SELECT cm.model_name, cmo.notes "
            "FROM camera_modes cmo "
            "JOIN camera_models cm ON cm.id = cmo.camera_model_id "
            "WHERE cmo.id = ?", (mode_id,)
        ).fetchone()
        print(f"  {label}: {mode_row[0]!r} / {mode_row[1]!r}  ({mode_id[:8]})")

    # ------------------------------------------------------------------
    # Import intrinsics — pass per-camera mode ID mapping
    # ------------------------------------------------------------------
    print(f"\nImporting intrinsics from {calib_path.name} ...")
    calib_result = import_calib_toml(
        db,
        calib_path,
        camera_mode_ids,          # dict[cam_label, camera_mode_id]
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
    session_id = create_mocap_session(db, recorded_at=recorded_at, notes=notes)
    print(f"\nCreated session: {session_id[:8]}  ({notes!r})")

    # ------------------------------------------------------------------
    # Add session cameras (one row per camera)
    # ------------------------------------------------------------------
    for cam_label, inst_id in calib_result.camera_instance_ids.items():
        intr_id  = calib_result.intrinsics_ids[cam_label]
        mode_id  = camera_mode_ids[cam_label]
        add_session_camera(
            db, db,   # session == registry (both are the same self-contained session DB)
            session_id, inst_id, mode_id, intr_id,
            label=cam_label,
        )
    print(f"Added {len(calib_result.camera_instance_ids)} session cameras.")

    # ------------------------------------------------------------------
    # Import extrinsics (one shared calibration reused across all shots)
    # ------------------------------------------------------------------
    print("\nImporting extrinsics ...")
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
    db: sqlite3.Connection,
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

    shot_id = create_shot(
        db, session_id,
        extrinsic_calibration_id=extrinsic_calibration_id,
        label=shot_dir.name,
    )
    print(f"  Created shot: {shot_id[:8]}")

    video_files = _detect_videos(videos_dir)
    for cam_label, inst_id in camera_instance_ids.items():
        video_path = video_files.get(cam_label)
        if video_path is None:
            print(f"  [warn] No video found for {cam_label} in {videos_dir}")
            continue
        fps = fps_per_cam.get(cam_label, 0.0)
        first_frame, last_frame = _sync_frame_range(sync_json, cam_label)
        add_shot_video(db, shot_id, inst_id, str(video_path),
                       first_frame, last_frame, fps)
        print(f"  Added video: {cam_label} → {video_path.name} "
              f"(frames {first_frame}–{last_frame}, {fps:.2f} fps)")

    sync_result = import_sync_json(
        db, shot_id, sync_json, camera_instance_ids,
        notes=shot_dir.name,
    )
    print(f"  Sync config: {sync_result.sync_config_id[:8]}"
          f" ({len(sync_result.camera_instance_ids)} cameras)")
    if sync_result.skipped:
        print(f"  Sync skipped: {sync_result.skipped}")

    pose_result = import_pose_json(
        db, shot_id, sync_result.sync_config_id,
        pose_dir, camera_instance_ids,
    )
    print(f"  Pose sequence: {pose_result.sequence_id[:8]}"
          f" ({pose_result.n_observations:,} observations)")
    if pose_result.skipped_cameras:
        print(f"  Pose skipped cameras: {pose_result.skipped_cameras}")


if __name__ == "__main__":
    sys.exit(main())

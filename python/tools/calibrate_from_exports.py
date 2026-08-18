# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""calibrate_from_exports.py — CLI wrapper for semi-automatic extrinsics calibration.

Takes a directory of PNG frames exported from the sync UI (one per camera) plus
the session DB, runs SIFT matching + BFS chain + bundle adjustment, and writes
a Pose2Sim-compatible cameras.toml.

Usage
-----
    uv run python tools/calibrate_from_exports.py \\
        --images-dir /path/to/exported/frames/ \\
        --session-db /path/to/session.db \\
        --shot-id <shot-uuid> \\
        --output cameras.toml

The tool matches camera labels embedded in filenames
(format: <prefix>_<timestamp>_<camera_label>_<frame>.png) to cameras in the DB.
If the filename convention does not match, use --cam-map to override:
    --cam-map "gopro_mini_01=cam_a,ace2=cam_b"

Optional similarity transform (scale only for now; origin/axis via --floor-z):
    --scale-points "floor-NE,floor-NW" --scale-distance 2.0

Output
------
cameras.toml   Pose2Sim-compatible.  Import with: posetrak-db extrinsics import
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import struct
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.setup.extrinsics_solver import (
    CamCalibState,
    CalibResult,
    run_calibration,
    to_toml_string,
    compute_reprojection_errors,
)


# ---------------------------------------------------------------------------
# Intrinsics loading from session DB
# ---------------------------------------------------------------------------

def _load_intrinsics_from_db(conn: sqlite3.Connection, intrinsics_id: str) -> dict:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM intrinsics_calibrations WHERE id = ?", (intrinsics_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"intrinsics_calibration {intrinsics_id!r} not found")

    fx, fy, cx, cy = row["fx"], row["fy"], row["cx"], row["cy"]
    K_new = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    if row["matrix_original"]:
        vals = struct.unpack("<9d", bytes(row["matrix_original"]))
        K_orig = np.array(vals).reshape(3, 3)
    else:
        K_orig = K_new.copy()

    if row["dist_coeffs"]:
        n = len(bytes(row["dist_coeffs"])) // 8
        dist = np.array(struct.unpack(f"<{n}d", bytes(row["dist_coeffs"]))).reshape(1, -1)
    else:
        dist = np.zeros((1, 4))

    fisheye = (row["distortion_model"] == "fisheye")
    return {"K": K_new, "K_orig": K_orig, "dist": dist, "fisheye": fisheye}


def _load_camera_intrinsics(session_db: Path, shot_id: str) -> tuple[dict[str, dict], set[str]]:
    """Return (intrinsics_by_label, all_labels).

    intrinsics_by_label contains only cameras that have a calibration.
    all_labels contains every camera in the capture (used for match diagnostics).
    """
    conn = sqlite3.connect(str(session_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT ci.label AS cam_label,
               COALESCE(cv.intrinsics_calibration_id, ic.id) AS intrinsics_calibration_id
        FROM capture_videos cv
        JOIN camera_instances ci ON ci.id = cv.camera_instance_id
        LEFT JOIN camera_modes cm ON cm.id = cv.camera_mode_id
        LEFT JOIN intrinsics_calibrations ic ON ic.camera_mode_id = cm.id
        WHERE cv.shot_id = ?
        """,
        (shot_id,),
    ).fetchall()

    intrinsics: dict[str, dict] = {}
    all_labels: set[str] = set()
    for r in rows:
        label = r["cam_label"]
        all_labels.add(label)
        cal_id = r["intrinsics_calibration_id"]
        if cal_id is None:
            print(f"  {label}: no intrinsics calibration")
            continue
        try:
            intr = _load_intrinsics_from_db(conn, cal_id)
            intrinsics[label] = intr
            print(f"  {label}: fx={intr['K'][0,0]:.1f} fisheye={intr['fisheye']}")
        except Exception as e:
            print(f"  WARNING: could not load intrinsics for {label}: {e}")
    conn.close()
    return intrinsics, all_labels


# ---------------------------------------------------------------------------
# Filename → camera label matching
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(r"^.+?_\d{2}_\d{2}_\d{3}_(.+?)_\d+\.png$", re.IGNORECASE)


def _label_from_filename(fname: str) -> str | None:
    m = _FNAME_RE.match(fname)
    return m.group(1).replace("_", " ") if m else None


def _normalise_label(label: str) -> str:
    """Normalise a camera label for comparison: lowercase, collapse -_. and spaces."""
    return re.sub(r"[-_.\s]+", " ", label).strip().lower()


def _match_label(filename_label: str, db_labels: list[str]) -> str | None:
    """Match a filename-derived label to a DB camera label.

    Normalises both sides (case, separator characters) before comparing.
    Returns the DB label on exact normalised match, None otherwise.
    No prefix/substring matching — that causes false positives when camera
    names share a common prefix (e.g. "pixel7" vs "pixel9").
    """
    fl = _normalise_label(filename_label)
    matches = [db for db in db_labels if _normalise_label(db) == fl]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Ambiguous: multiple DB labels normalise to the same string
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", required=True, help="Directory containing exported PNG frames")
    parser.add_argument("--session-db", required=True, help="Path to session SQLite DB")
    parser.add_argument("--shot-id", required=True, help="Shot UUID")
    parser.add_argument("--output", default="cameras.toml", help="Output TOML path")
    parser.add_argument("--cam-map", default="", help="Manual camera mapping: 'file_label=db_label,...'")
    parser.add_argument("--min-inliers", type=int, default=20, help="Minimum SIFT inliers per pair")
    parser.add_argument("--sift-ratio", type=float, default=0.75, help="Lowe ratio test threshold")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    session_db = Path(args.session_db)

    # Parse manual camera map
    cam_map: dict[str, str] = {}
    if args.cam_map:
        for pair in args.cam_map.split(","):
            a, _, b = pair.partition("=")
            cam_map[a.strip()] = b.strip()

    print(f"\n[1/5] Loading intrinsics from {session_db}")
    intrinsics_by_label, all_db_labels = _load_camera_intrinsics(session_db, args.shot_id)
    if not intrinsics_by_label:
        print("ERROR: no cameras with intrinsics found in DB")
        sys.exit(1)

    print(f"\n[2/5] Scanning {images_dir}")
    png_files = sorted(images_dir.glob("*.png"))
    if not png_files:
        print("ERROR: no PNG files found")
        sys.exit(1)

    states: list[CamCalibState] = []
    for png in png_files:
        # Determine camera label from filename
        file_label = _label_from_filename(png.name)
        if file_label and file_label in cam_map:
            file_label = cam_map[file_label]

        if file_label is None:
            print(f"  SKIP {png.name} — filename does not match expected pattern")
            continue

        # Match against ALL DB cameras (with or without intrinsics) for diagnostics
        db_label = _match_label(file_label, list(all_db_labels))
        if db_label is None:
            print(f"  SKIP {png.name} — '{file_label}' not found in DB "
                  f"(use --cam-map to override)")
            continue
        if db_label not in intrinsics_by_label:
            print(f"  SKIP {png.name} — '{db_label}' has no intrinsics calibration")
            continue

        intr = intrinsics_by_label[db_label]
        img = cv2.imread(str(png))
        if img is None:
            print(f"  SKIP {png.name} — cv2.imread failed")
            continue

        state = CamCalibState(
            video_id=db_label,
            label=db_label,
            K=intr["K"],
            K_orig=intr["K_orig"],
            dist=intr["dist"],
            fisheye=intr["fisheye"],
            image=img,
        )
        states.append(state)
        print(f"  {db_label}: {img.shape[1]}×{img.shape[0]} from {png.name}")

    if len(states) < 2:
        print("ERROR: need at least 2 cameras with loaded images")
        sys.exit(1)

    print(f"\n[3/5] Running calibration ({len(states)} cameras)")
    result = run_calibration(
        states,
        sift_ratio=args.sift_ratio,
        sift_min_inliers=args.min_inliers,
    )

    print(f"\n[4/5] Results")
    print(f"  Solved: {len(states) - len(result.unsolved)}/{len(states)} cameras")
    if result.unsolved:
        print(f"  Unsolved: {result.unsolved}")
    print(f"  Triangulated points: {len(result.points_3d)}")
    print()
    for vid, stats in result.reprojection_errors.items():
        s = result.cameras[vid]
        print(f"  {s.label}: {stats['mean']:.2f} ± {stats['std']:.2f} px  "
              f"(max {stats['max']:.1f}, n={stats['n']})")

    if result.unsolved:
        print("\nWARNING: some cameras could not be solved.")
        print("  Add manual control points in the UI, or capture an additional")
        print("  frame showing shared landmarks between isolated cameras.")

    print(f"\n[5/5] Writing {args.output}")
    toml_str = to_toml_string(result)
    Path(args.output).write_text(toml_str)
    print("Done.")
    print(f"\nTo import: posetrak-db extrinsics import --session-db {args.session_db} {args.output}")


if __name__ == "__main__":
    main()

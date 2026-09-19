# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""spot_check_box_calibration.py — PROTOTYPE: visual sanity check for a
solved extrinsic_calibrations row from calibrate_extrinsics_from_moving_box.py.

Projects the calibration box's own known world-frame marker corners
(scratch/calib_box_2026-09-06.yaml) into a handful of real camera frames
using the persisted extrinsics + each camera's own intrinsics, and draws
the projected quads over the real image -- if the calibration is right,
every drawn quad should land exactly on the box's real, visible markers in
that frame. Reprojection-error numbers alone were already checked by the
calibration script itself; this is the actual-pixels check this project's
own established practice calls for before trusting them.

Usage
-----
    python spot_check_box_calibration.py \\
        --session SESSION.db --capture b21fa02d \\
        --extrinsic-calibration 4815f5ac \\
        --timestamp 30.372 \\
        --output-dir scratch/box_calib_spotcheck
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import sqlite3

from posetrak.detection.frame_source import iter_frames
from app.setup.db_context import SyncPoint, SyncTable
from app.setup.fiducial_markers import load_marker_body_yaml_file

BOX_YAML_PATH = str(Path(__file__).resolve().parents[2] / "scratch" / "calib_box_2026-09-06.yaml")


def _load_intrinsics_by_id(conn: sqlite3.Connection, calib_id: str) -> dict:
    row = conn.execute("SELECT * FROM intrinsics_calibrations WHERE id = ?", (calib_id,)).fetchone()
    if row is None:
        raise SystemExit(f"No intrinsics_calibrations row for id {calib_id}")
    fx, fy, cx, cy = row["fx"], row["fy"], row["cx"], row["cy"]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    K_orig = K.copy()
    if row["matrix_original"]:
        K_orig = np.array(struct.unpack("<9d", bytes(row["matrix_original"]))).reshape(3, 3)
    if row["dist_coeffs"]:
        n = len(bytes(row["dist_coeffs"])) // 8
        dist = np.array(struct.unpack(f"<{n}d", bytes(row["dist_coeffs"]))).reshape(1, -1)
    else:
        dist = np.zeros((1, 4))
    return {"K": K, "K_orig": K_orig, "dist": dist, "fisheye": row["distortion_model"] == "fisheye"}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--capture", required=True, help="captures.id (or prefix)")
    ap.add_argument("--extrinsic-calibration", required=True,
                     help="extrinsic_calibrations.id (or prefix) to check")
    ap.add_argument("--timestamp", type=float, required=True,
                     help="Synced-timeline seconds to check (e.g. the box's final position)")
    ap.add_argument("--camera-label", nargs="*", default=None,
                     help="Cameras to check (default: every camera with a solved pose)")
    ap.add_argument("--output-dir", default="scratch/box_calib_spotcheck")
    args = ap.parse_args()

    conn = sqlite3.connect(args.session)
    conn.row_factory = sqlite3.Row
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_row = conn.execute(
        "SELECT id FROM extrinsic_calibrations WHERE id LIKE ? || '%'", (args.extrinsic_calibration,)
    ).fetchone()
    if calib_row is None:
        raise SystemExit(f"No extrinsic_calibrations row matching {args.extrinsic_calibration!r}")
    calib_id = calib_row["id"]

    entry_rows = conn.execute(
        "SELECT camera_instance_id, R, t FROM extrinsic_entries WHERE extrinsic_calibration_id = ?",
        (calib_id,),
    ).fetchall()
    extrinsics_by_instance = {
        r["camera_instance_id"]: (
            np.array(struct.unpack("<9d", bytes(r["R"]))).reshape(3, 3),
            np.array(struct.unpack("<3d", bytes(r["t"]))),
        )
        for r in entry_rows
    }

    capture_row = conn.execute(
        "SELECT id FROM captures WHERE id LIKE ? || '%'", (args.capture,)
    ).fetchone()
    if capture_row is None:
        raise SystemExit(f"No captures row matching {args.capture!r}")
    capture_id = capture_row["id"]

    cam_rows = conn.execute(
        """
        SELECT cv.id AS shot_video_id, ci.label, ci.id AS camera_instance_id, cv.file_path,
               cv.actual_fps, cm.nominal_fps, cv.intrinsics_calibration_id
        FROM capture_videos cv
        JOIN camera_instances ci ON ci.id = cv.camera_instance_id
        JOIN camera_modes cm ON cm.id = cv.camera_mode_id
        WHERE cv.shot_id = ?
        """,
        (capture_id,),
    ).fetchall()
    if args.camera_label:
        cam_rows = [r for r in cam_rows if r["label"] in args.camera_label]

    sync_config = conn.execute(
        "SELECT id FROM sync_configs WHERE shot_id = ?", (capture_id,)
    ).fetchone()
    sp_rows = conn.execute(
        "SELECT shot_video_id, video_frame, timestamp_s FROM sync_points WHERE sync_config_id = ?",
        (sync_config["id"],),
    ).fetchall()
    sync_points = [
        SyncPoint(camera_instance_id="", shot_video_id=r["shot_video_id"],
                  video_frame=r["video_frame"], timestamp_s=r["timestamp_s"])
        for r in sp_rows
    ]
    fps_by_video = {r["shot_video_id"]: float(r["actual_fps"] or r["nominal_fps"] or 120.0) for r in cam_rows}
    sync_table = SyncTable(sync_points, fps_by_video)

    box_config = load_marker_body_yaml_file(BOX_YAML_PATH, rig_id="calib-box-2026-09-06")

    for cam in cam_rows:
        pose = extrinsics_by_instance.get(cam["camera_instance_id"])
        if pose is None:
            print(f"{cam['label']}: no solved extrinsics, skipping")
            continue
        R, t = pose
        intr = _load_intrinsics_by_id(conn, cam["intrinsics_calibration_id"])
        frame_idx = sync_table.lookup(args.timestamp, cam["shot_video_id"])
        if frame_idx is None:
            print(f"{cam['label']}: no sync data at t={args.timestamp}, skipping")
            continue
        img = None
        for _, im in iter_frames(cam["file_path"], frame_idx, frame_idx + 1):
            img = im
            break
        if img is None:
            print(f"{cam['label']}: could not read frame {frame_idx}, skipping")
            continue

        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)
        n_drawn = 0
        for marker_id, corners_world in box_config.marker_corners.items():
            if intr["fisheye"]:
                pts, _ = cv2.fisheye.projectPoints(
                    corners_world.reshape(-1, 1, 3), rvec, tvec, intr["K_orig"], intr["dist"]
                )
            else:
                pts, _ = cv2.projectPoints(corners_world, rvec, tvec, intr["K_orig"], intr["dist"])
            pts = pts.reshape(-1, 2)
            # Only draw if it actually lands within (or near) the frame -- a
            # marker this camera never saw during calibration can still
            # solve to a wildly out-of-frame projection, which is expected,
            # not a bug.
            h, w = img.shape[:2]
            if not np.any((pts[:, 0] > -w) & (pts[:, 0] < 2 * w) & (pts[:, 1] > -h) & (pts[:, 1] < 2 * h)):
                continue
            pts_i = pts.astype(int)
            cv2.polylines(img, [pts_i], isClosed=True, color=(0, 0, 255), thickness=3)
            for p in pts_i:
                cv2.circle(img, tuple(p), 6, (0, 255, 0), -1)
            label_pos = tuple(pts_i[0] + np.array([10, -10]))
            cv2.putText(img, f"id{marker_id}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            n_drawn += 1

        out_path = out_dir / f"{cam['label']}_t{args.timestamp:.3f}.png"
        cv2.imwrite(str(out_path), img)
        print(f"{cam['label']}: drew {n_drawn} marker(s) -> {out_path}")

    conn.close()


if __name__ == "__main__":
    main()

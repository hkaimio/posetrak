# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""render_marker_normal_debug_frame.py — PROTOTYPE: single-frame, all-
cameras debug dump of prototype_marker_normal_assignment.py's per-leg
anatomical frame (medial/lateral/anterior directions), drawn as arrows
from the joint's own projected position.

Built specifically to eyeball-validate the anterior_dir sign convention
(cross(leg_axis, medial_dir)) against a real frame before trusting it in
an assignment run -- a silently-flipped sign would swap front/back
everywhere without ever raising an error, so this needs a real image
check, not just a code review.

Usage:
    python tools/render_marker_normal_debug_frame.py \\
        --session /path/to/session.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --pose-sequence ec1b3e2f-1ef8-4e31-806c-33102a969ecd \\
        --time 42.925 \\
        --out-dir scratch/dot_ground_truth/normal_debug
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.setup.extrinsics_solver import _proj_matrix  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402
from tools.prototypes.superseded.prototype_marker_normal_assignment import compute_leg_frames  # noqa: E402

_ARROW_LEN_M = 0.15
_COLORS = {"medial": (0, 255, 0), "lateral": (0, 0, 255), "anterior": (255, 128, 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--pose-sequence", required=True)
    ap.add_argument("--time", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    states = load_camera_states(conn, args.shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)
    cam_id_by_label = {r["label"]: r["id"] for r in conn.execute("SELECT id, label FROM camera_instances")}
    label_by_cam_id = {v: k for k, v in cam_id_by_label.items()}

    kp_by_cam, fidx_by_cam = {}, {}
    for cam_id, svid in svid_by_cam.items():
        if cam_id not in states:
            continue
        fidx = sync_table.lookup(args.time, svid)
        if fidx is None:
            continue
        fidx_by_cam[cam_id] = fidx
        row = conn.execute(
            "SELECT kp_blob FROM pose_observations WHERE sequence_id = ? AND camera_instance_id = ? "
            "AND video_frame = ? AND source = 'body'",
            (args.pose_sequence, cam_id, fidx),
        ).fetchone()
        if row is not None:
            kp_by_cam[cam_id] = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3)

    leg_frames = compute_leg_frames(kp_by_cam, states)
    print(f"computed leg frames for: {sorted(leg_frames)}")
    if not leg_frames:
        raise SystemExit("no leg frame could be computed at this instant -- try a different --time")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cam_id, fidx in fidx_by_cam.items():
        row = conn.execute("SELECT file_path FROM capture_videos WHERE id = ?", (svid_by_cam[cam_id],)).fetchone()
        img = None
        for vf, frame in iter_frames(row["file_path"], fidx, fidx + 1):
            img = frame
        if img is None:
            continue
        P = _proj_matrix(states[cam_id])

        def _project(xyz: np.ndarray) -> tuple[int, int]:
            h = P @ np.append(xyz, 1.0)
            p = h[:2] / h[2]
            return int(round(p[0])), int(round(p[1]))

        for joint, frame in leg_frames.items():
            joint_pos = frame["joint_pos"]
            p0 = _project(joint_pos)
            cv2.circle(img, p0, 8, (255, 255, 255), -1)
            cv2.putText(img, joint, (p0[0] + 10, p0[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            for tag, vec in (("medial", frame["medial"]), ("lateral", -frame["medial"]),
                              ("anterior", frame["anterior"])):
                p1 = _project(joint_pos + _ARROW_LEN_M * vec)
                cv2.arrowedLine(img, p0, p1, _COLORS[tag], 3, tipLength=0.25)
                cv2.putText(img, tag, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COLORS[tag], 2, cv2.LINE_AA)

        label = label_by_cam_id.get(cam_id, cam_id)
        out_path = out_dir / f"{label}_t{args.time:.3f}.png"
        cv2.imwrite(str(out_path), img)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

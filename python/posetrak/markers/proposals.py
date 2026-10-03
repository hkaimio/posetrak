# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""proposals.py — run the automatic dot-to-slot assignment over a time range and store it.

    python -m posetrak.markers.proposals SESSION_DB SHOT DETECTION_RUN SEG_RUN POSE_RUN ATTACHMENT.yaml T0 T1
        [--mask-label 1] [--skip-camera SUBSTRING ...]

Run ids may be given as unique prefixes. The proposals of the newest run of the pass are what
the review tools show (see review.py).
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

from posetrak.calibration.session_cameras import load_camera_states, load_sync_table
from posetrak.markers.autolabel import (
    Predictor, assign, check_consistency, load_placements, score_tracklets,
)
from posetrak.markers.bones import BodyPose
from posetrak.markers.review import Proposal, save_proposals
from posetrak.markers.tracklets import SubjectMasks, collect_tracklets, filter_tracklets

MASK_GROW_PX = 8.0       # a marker lies on the person: the mask is barely grown
MIN_ON_MASK = 0.5


def _full_id(conn: sqlite3.Connection, table: str, prefix: str) -> str:
    return conn.execute(f"SELECT id FROM {table} WHERE id LIKE ?", (prefix + "%",)).fetchone()[0]


def propose(conn: sqlite3.Connection, shot_id: str, detection_run: str, seg_run: str, pose_run: str,
            attachment: str, t0: float, t1: float, *, mask_label: int = 1, skip_cameras: tuple[str, ...] = ()) -> list[Proposal]:
    """Assign the dot tracklets of the person in [t0, t1] and return the proposals (not stored)."""
    conn.row_factory = sqlite3.Row
    states = load_camera_states(conn, shot_id)
    sync, svid_by_cam = load_sync_table(conn, shot_id)
    labels = {r["id"]: r["label"] for r in conn.execute("SELECT id, label FROM camera_instances")}
    native = {r["id"]: (r["w"], r["h"]) for r in conn.execute(
        "SELECT cv.id, COALESCE(ic.image_width, cm.width_px) w, COALESCE(ic.image_height, cm.height_px) h"
        " FROM capture_videos cv LEFT JOIN camera_modes cm ON cm.id = cv.camera_mode_id"
        " LEFT JOIN intrinsics_calibrations ic ON ic.id = cv.intrinsics_calibration_id WHERE cv.shot_id = ?", (shot_id,))}
    masks = SubjectMasks(conn, seg_run, mask_label, MASK_GROW_PX, native)
    predictor = Predictor(BodyPose(conn, pose_run), load_placements(attachment), states, t_origin=t0)
    scores = []
    for cam, svid in svid_by_cam.items():
        if cam not in states or any(s in labels[cam] for s in skip_cameras):
            continue
        tracklets = collect_tracklets(conn, detection_run, svid, sync.lookup(t0, svid), sync.lookup(t1, svid))
        tracklets, _ = filter_tracklets(tracklets, svid, masks, min_on_subject=MIN_ON_MASK)
        cam_scores = score_tracklets(tracklets, cam, svid, predictor, sync)
        assign(cam_scores)
        scores += cam_scores
    check_consistency(scores, predictor, svid_by_cam, sync)
    return [Proposal(s.cam_id, s.tracklet_id, s.first_frame, s.last_frame,
                     None if s.slot is None else predictor.names[s.slot],
                     float(s.cost[s.slot]) if s.slot is not None else _best_cost(s.cost),
                     float(s.margin), s.support, s.confidence)
            for s in scores]


def _best_cost(cost: np.ndarray) -> float:
    """The cost of the closest slot, a large number if none is visible."""
    return float(cost[np.isfinite(cost)].min()) if np.isfinite(cost).any() else 1e9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_db")
    ap.add_argument("shot")
    ap.add_argument("detection_run")
    ap.add_argument("seg_run")
    ap.add_argument("pose_run")
    ap.add_argument("attachment")
    ap.add_argument("t0", type=float)
    ap.add_argument("t1", type=float)
    ap.add_argument("--mask-label", type=int, default=1)
    ap.add_argument("--skip-camera", action="append", default=[])
    a = ap.parse_args()
    from posetrak.db.db import open_session
    conn = open_session(Path(a.session_db))
    det, seg, pose = (_full_id(conn, t, p) for t, p in (
        ("detection_runs", a.detection_run), ("seg_quality_runs", a.seg_run), ("tracking_runs", a.pose_run)))
    shot = _full_id(conn, "captures", a.shot)
    rows = propose(conn, shot, det, seg, pose, a.attachment, a.t0, a.t1,
                   mask_label=a.mask_label, skip_cameras=tuple(a.skip_camera))
    conn.row_factory = None
    print(f"stored proposal set {save_proposals(conn, det, pose, rows)}: {len(rows)} tracklets")


if __name__ == "__main__":
    main()

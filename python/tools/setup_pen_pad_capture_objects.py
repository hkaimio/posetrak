# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""setup_pen_pad_capture_objects.py -- registers the pen and pad as real
`capture_objects`, each with its own scoped `detection_runs` row and a
finalised `pose_observation_sequences`, so the production tracker
(`posetrak-tracker track`) can be pointed at them directly.

Per CLAUDE.md's own append-only convention: never mutates the existing
ArUco detection run (`75cbf678...`) -- copies its relevant
`detection_keypoints` rows onto a fresh, object-bound run instead, with
the corner blob *re-encoded* (not byte-copied) to the object's own
smaller `marker_ids` list, so the resulting run's `config_json` is a
clean, self-describing 2-marker (pen) / 1-marker (pad) layout rather than
inheriting the original 8-marker wire format.

Pad uses marker "1" ONLY, not "0": the extrinsics calibration box
carries a same-ID ArUco tag on one face by mistake (Harri, 2026-09-13;
see status.md), so "0" sightings are frequently the box, not the pad. A
single ArUco marker already fully determines a rigid body's 6-DOF pose
(4 corners), so dropping "0" costs nothing and sidesteps relying on the
tracker's own outlier rejection to save us from a contamination problem
this script can just avoid outright.

Usage:
    python tools/setup_pen_pad_capture_objects.py \\
        --session D:/mocap/2026-09-06-kare-tests/2026-09-06-kare-tests.db \\
        --shot-id b21fa02d-2a82-4a37-a749-a156116a0aa0 \\
        --source-detection-run 75cbf678-2066-4a58-ab81-d27ea4c58d02 \\
        --time-start 98.0 --time-end 113.0
"""
from __future__ import annotations

import argparse
import datetime
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pose.finalise import finalise_object_to_db  # noqa: E402
from posetrak.db.db import generate_id, open_session  # noqa: E402
from posetrak.db.manage_marker_body import import_marker_body  # noqa: E402
from posetrak.db.manage_skeleton import import_skeleton_str  # noqa: E402
from posetrak.skeleton.marker_body_to_skeleton import generate_prop_skeleton_yaml  # noqa: E402
from app.setup.fiducial_markers import load_marker_body_yaml  # noqa: E402
from tools.calibrate_rigid_marker_body import load_sync_table  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

CATALOG = {
    "pen": {
        "yaml": REPO_ROOT / "catalog" / "pen.calibrated.2026-09-06-kare-tests.real-capture.yaml",
        "marker_ids": ["2", "3"],
    },
    "pad": {
        "yaml": REPO_ROOT / "catalog" / "pad.calibrated.2026-09-06-kare-tests.marker1-only.yaml",
        "marker_ids": ["1"],
    },
}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _reencode_marker_row(kp: np.ndarray, old_marker_ids: list[str], new_marker_ids: list[str]) -> np.ndarray:
    """Re-slot a float32[4*len(old),3] corner blob into float32[4*len(new),3],
    matching MarkerKeypointWriter's own fixed-slot, list-position-major
    layout -- a marker id absent from *new_marker_ids* is simply dropped, one
    present in *new_marker_ids* but not detected that frame keeps the
    NaN/0-confidence occluded-slot convention.
    """
    out = np.full((4 * len(new_marker_ids), 3), np.nan, dtype=np.float32)
    out[:, 2] = 0.0
    old_slot = {mid: i for i, mid in enumerate(old_marker_ids)}
    for new_i, mid in enumerate(new_marker_ids):
        old_i = old_slot.get(mid)
        if old_i is None:
            continue
        out[new_i * 4:new_i * 4 + 4] = kp[old_i * 4:old_i * 4 + 4]
    return out


def setup_object(
    session: sqlite3.Connection,
    shot_id: str,
    source_run_id: str,
    object_name: str,
    marker_ids: list[str],
    yaml_path: Path,
    time_start: float,
    time_end: float,
) -> dict:
    print(f"\n=== {object_name} ===")

    body_id = import_marker_body(
        session, yaml_path, name=f"{object_name}-2026-09-06-kare-tests",
        source="real-capture calibration, marker-based-mocap session 2026-09-13",
    )
    print(f"  marker_body_definition_id: {body_id}")
    config = load_marker_body_yaml(yaml_path.read_text(encoding="utf-8"))

    src = session.execute(
        "SELECT shot_id, sync_config_id, trial_id, detector_model, config_json "
        "FROM detection_runs WHERE id = ?", (source_run_id,),
    ).fetchone()
    if src is None:
        raise ValueError(f"source detection run not found: {source_run_id!r}")
    sync_config_id = src["sync_config_id"]
    old_marker_ids = json.loads(src["config_json"])["marker_ids"]

    object_id = generate_id()
    session.execute(
        "INSERT INTO capture_objects (id, capture_id, name, marker_body_definition_id, notes, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (object_id, shot_id, object_name, body_id,
         "Registered by setup_pen_pad_capture_objects.py, 2026-09-13", _now()),
    )
    print(f"  capture_object_id: {object_id}")

    new_run_id = generate_id()
    new_config = {
        "dictionary": "DICT_4X4_50",
        "marker_ids": marker_ids,
        "min_marker_perimeter_rate": 0.01,
        "marker_body_definition_id": body_id,
    }
    session.execute(
        "INSERT INTO detection_runs "
        "(id, shot_id, sync_config_id, trial_id, time_start_s, time_end_s, detector_model, "
        " pose_model, detector_conf, pose_conf_threshold, status, created_at, completed_at, "
        " detector_type, config_json, capture_object_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (new_run_id, shot_id, sync_config_id, src["trial_id"], time_start, time_end,
         src["detector_model"], "", 0.3, 0.3, "complete", _now(), _now(),
         "aruco", json.dumps(new_config), object_id),
    )
    print(f"  detection_run_id: {new_run_id}")

    sync_table, svid_by_cam = load_sync_table(session, shot_id)
    src_rows = session.execute(
        "SELECT shot_video_id, video_frame, keypoints, noise_scale FROM detection_keypoints "
        "WHERE detection_run_id = ? AND track_id = 0 AND region_type = 'markers' "
        "ORDER BY shot_video_id, video_frame",
        (source_run_id,),
    ).fetchall()

    new_rows = []
    for r in src_rows:
        gt = sync_table.frame_to_global_time(r["video_frame"], r["shot_video_id"])
        if gt is None or not (time_start <= gt <= time_end):
            continue
        kp = np.frombuffer(r["keypoints"], dtype=np.float32).reshape(-1, 3)
        new_kp = _reencode_marker_row(kp, old_marker_ids, marker_ids)
        if not np.any(new_kp[:, 2] > 0.5):
            continue  # none of this object's markers seen this frame -- skip, not a real observation
        new_rows.append((new_run_id, r["shot_video_id"], r["video_frame"], 0, "markers",
                          new_kp.tobytes(), r["noise_scale"]))
    session.executemany(
        "INSERT INTO detection_keypoints "
        "(detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale) "
        "VALUES (?,?,?,?,?,?,?)",
        new_rows,
    )
    session.commit()
    print(f"  copied {len(new_rows)} detection_keypoints rows (re-encoded to {marker_ids})")

    seq_id = finalise_object_to_db(session, new_run_id, notes="setup_pen_pad_capture_objects.py")
    session.commit()
    print(f"  pose_observation_sequence_id: {seq_id}")

    skeleton_yaml = generate_prop_skeleton_yaml(config, name=object_name, marker_body_definition_id=body_id)
    skeleton_id = import_skeleton_str(session, skeleton_yaml, name=object_name, source=f"marker_body:{body_id}")
    session.commit()
    print(f"  skeleton_id: {skeleton_id}")

    return {
        "marker_body_definition_id": body_id, "capture_object_id": object_id,
        "detection_run_id": new_run_id, "sequence_id": seq_id, "skeleton_id": skeleton_id,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--source-detection-run", required=True)
    ap.add_argument("--time-start", type=float, required=True)
    ap.add_argument("--time-end", type=float, required=True)
    args = ap.parse_args()

    session = open_session(Path(args.session))  # read-write -- this script writes new rows

    results = {}
    for name, spec in CATALOG.items():
        results[name] = setup_object(
            session, args.shot_id, args.source_detection_run, name,
            spec["marker_ids"], spec["yaml"], args.time_start, args.time_end,
        )

    print("\n=== summary ===")
    print(json.dumps(results, indent=2))
    session.close()


if __name__ == "__main__":
    main()

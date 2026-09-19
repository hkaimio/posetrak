# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""finalize_ball_cutie_detection.py — write cutie_segment_ball_all_cameras.py's
per-(throw, camera) mask-centroid CSVs into the session DB as a real,
finalized detection: a `capture_objects` row for the ball, its own scoped
`detection_runs` row, a `pose_observation_sequences` row, and the actual
`pose_observations` rows -- so the production tracker
(`posetrak-tracker track --sequence ...`) and the UI's object-track tree
branch (session_tree.py's `_add_object_tracks`) both work exactly the way
they already do for the pen/pad (`setup_pen_pad_capture_objects.py`).

Deliberately does NOT reuse `app.pose.finalise.finalise_object_to_db()`:
that function hard-requires `detection_runs.detector_type == 'aruco'`
(it reads `detection_keypoints` rows written by the ArUco corner writer) --
there is no reflective-dot/segmentation equivalent for a *dots-only* object
today (the existing 'dots' machinery is a body/prop *add-on* alongside a
coded-marker run, not a standalone object detector). So this writes the
same target tables directly instead, using the existing 'dots' wire format
(db_cache.encode_dot_candidates, float32[1,9] per frame: cx, cy, area,
compactness, major_axis_px, minor_axis_px, dir_x, dir_y, tracklet_id) so
the C++ reader's existing `load_unlabeled_candidates()` path (session_
reader.cpp, `WHERE source='dots'`) needs no changes at all to consume it --
scoped to the ball's own new sequence_id, so it can never collide with
Nelli's own real reflective-dot data living under the shared vitpose
sequence.

Usage::

    python tools/finalize_ball_cutie_detection.py \\
        --session /path/to/session.db \\
        --csv-dir <output-dir from cutie_segment_ball_all_cameras.py>
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pose.db_cache import encode_dot_candidates  # noqa: E402
from posetrak.db.db import generate_id, open_session  # noqa: E402
from posetrak.detection.dot_blob_detector import BlobCandidate  # noqa: E402
from posetrak.db.manage_marker_body import import_marker_body  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER_BODY_YAML = REPO_ROOT / "catalog" / "ball.single-reflective-dot.2026-09-06-kare-tests.yaml"

CAPTURE_ID = "b21fa02d-2a82-4a37-a749-a156116a0aa0"
SYNC_CONFIG_ID = "f4acaa06-e663-43ee-870e-13ee4f112892"
TRIAL_ID = "d9ef6092-e629-43e0-9ccb-132761e4a547"

THROW_CORE_WINDOWS_S = {
    # name -> (t_start, t_end), the real (unpadded) throw window: step_to_time()
    # of cutie_segment_ball_all_cameras.py's own THROWS step ranges
    # (33.620 + step/120), NOT that script's own +-0.6s padded search window --
    # this is just descriptive metadata for the finalized sequence/detection_run
    # rows' own time_start_s/time_end_s, not consumed by anything downstream.
    "throw1": (57.087, 57.803),
    "throw2": (63.928, 64.745),
    "throw3": (66.862, 67.570),
    "throw4": (69.487, 70.295),
}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_camera_labels_to_instance_ids(session) -> dict[str, str]:
    rows = session.execute("SELECT id, label FROM camera_instances").fetchall()
    return {label: cid for cid, label in rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--object-name", default="ball")
    args = ap.parse_args()

    session = open_session(Path(args.session))
    csv_dir = Path(args.csv_dir)
    cam_label_to_id = load_camera_labels_to_instance_ids(session)

    body_id = import_marker_body(
        session, MARKER_BODY_YAML, name="ball-2026-09-06-kare-tests",
        source="Cutie/SAM2 mask-centroid segmentation, 2026-09-16 (no real multi-marker "
               "geometry -- single reflective_dot placeholder, see the YAML's own header)",
    )
    print(f"marker_body_definition_id: {body_id}")

    object_id = generate_id()
    session.execute(
        "INSERT INTO capture_objects (id, capture_id, name, marker_body_definition_id, notes, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (object_id, CAPTURE_ID, args.object_name, body_id,
         "Registered by finalize_ball_cutie_detection.py, 2026-09-16 -- position from Cutie/SAM2 "
         "video-object-segmentation mask centroid, not reflective-dot detection.", _now()),
    )
    print(f"capture_object_id: {object_id}")

    all_windows = list(THROW_CORE_WINDOWS_S.values())
    time_start = min(w[0] for w in all_windows)
    time_end = max(w[1] for w in all_windows)

    detection_run_id = generate_id()
    session.execute(
        "INSERT INTO detection_runs "
        "(id, shot_id, sync_config_id, trial_id, time_start_s, time_end_s, detector_model, "
        " pose_model, detector_conf, pose_conf_threshold, status, created_at, completed_at, "
        " detector_type, config_json, capture_object_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (detection_run_id, CAPTURE_ID, SYNC_CONFIG_ID, TRIAL_ID, time_start, time_end,
         "cutie+sam2.1-hiera-base-plus", "", 0.0, 0.0, "complete", _now(), _now(),
         "blob",
         '{"method": "video-object-segmentation mask centroid", '
         '"init": "SAM2 box prompt seeded from a prior constant-velocity UKF tracking run\'s '
         '3D position, projected into each camera", '
         '"script": "tools/cutie_segment_ball_all_cameras.py"}',
         object_id),
    )
    print(f"detection_run_id: {detection_run_id}")

    sequence_id = generate_id()
    session.execute(
        "INSERT INTO pose_observation_sequences "
        "(id, shot_id, sync_config_id, time_start_s, time_end_s, name, pose_model, notes, "
        " pixels_are_undistorted, detection_run_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (sequence_id, CAPTURE_ID, SYNC_CONFIG_ID, time_start, time_end, None,
         "cutie-sam2-centroid", "Registered by finalize_ball_cutie_detection.py, 2026-09-16",
         0, detection_run_id),
    )
    print(f"pose_observation_sequence_id: {sequence_id}")

    n_rows = 0
    n_frames_written = 0
    csv_paths = sorted(csv_dir.glob("throw*_*.csv"))
    for csv_path in csv_paths:
        # filename: {throw}_{camera_label}.csv -- camera_label itself may
        # contain underscores (e.g. gopro-11_mini_01), so split on the throw
        # prefix instead of a naive single split.
        stem = csv_path.stem
        throw_name, _, cam_label = stem.partition("_")
        camera_instance_id = cam_label_to_id.get(cam_label)
        if camera_instance_id is None:
            print(f"  skipping {csv_path.name}: unknown camera label {cam_label!r}")
            continue

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        for row in rows:
            area = float(row["mask_area_px"])
            # Equivalent circular diameter -- the mask is a real object
            # silhouette (not a reflective-dot highlight), so compactness=1.0
            # (round) is the closest honest description this wire format has;
            # there's no motion-blur-streak concept for a full-object mask.
            diameter = 2.0 * math.sqrt(area / math.pi) if area > 0 else 0.0
            cx, cy = float(row["cx"]), float(row["cy"])
            r = diameter / 2.0
            candidate = BlobCandidate(
                cx=cx, cy=cy, area=area,
                compactness=1.0, bbox=(int(cx - r), int(cy - r), int(diameter), int(diameter)),
                major_axis_px=diameter, minor_axis_px=diameter,
                dir_x=0.0, dir_y=0.0, tracklet_id=0.0,
            )
            blob = encode_dot_candidates([candidate])
            session.execute(
                "INSERT INTO pose_observations "
                "(sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, "
                " source, detection_run_id, kp_blob, noise_scale) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (sequence_id, camera_instance_id, int(row["video_frame"]),
                 float(row["timestamp_s"]), 0, "dots", detection_run_id, blob, 1.0),
            )
            n_rows += 1
        n_frames_written += len(rows)
        print(f"  {csv_path.name}: {len(rows)} frames")

    session.commit()
    print(f"\nWrote {n_rows} pose_observations rows across {len(csv_paths)} camera/throw files")
    print(f"\nTrack it with:\n"
          f"  posetrak-tracker track --session-db {args.session} --sequence {sequence_id} "
          f"--skeleton <ball skeleton id> --tracker-config <ball tracker_config id> --person-id 0 "
          f"--output-dir <dir> --smooth")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""finalize_ball_blender_detection.py -- write blender_export_2d_tracks.py's
per-camera CSVs (manually-tracked in Blender's Movie Clip Editor, 2-3
re-inits per camera to survive occlusion/fast motion -- see status.md's
2026-09-16/17 entries) into the session DB as a real, finalized detection:
a `capture_objects` row for the ball, its own scoped `detection_runs` row,
a `pose_observation_sequences` row, and the actual `pose_observations`
rows -- same shape as finalize_ball_cutie_detection.py, minus the
segmentation step (Blender's tracker already gives one (x, y) point per
frame directly, no mask/centroid step needed).

Also folds in gopro13_02's *existing* real reflective-dot data for the
same time span (Harri: good quality there already, no need to hand-track
it in Blender too) -- copied as-is (every raw candidate, unfiltered) from
Nelli's own sequence into this new one. No pre-filtering needed: the
tracker's own Mahalanobis-gated dot assignment already separates the real
ball candidate from body-marker clutter frame by frame, the same
mechanism her leg markers already rely on -- this script doesn't need to
(and shouldn't try to) guess which raw candidate is the ball itself.

Camera video_frame numbers in the Blender export already match
pose_observations.video_frame exactly (verified directly against real
rows before writing this) -- no frame_start/fps correction needed. Each
row's timestamp_s is looked up from an existing 'body' pose_observations
row at that exact (camera, video_frame) rather than recomputed from fps,
so this can't drift even if a camera's actual_fps is slightly imprecise.

Usage::

    python tools/finalize_ball_blender_detection.py \\
        --session /path/to/session.db \\
        --blender-csv-dir /path/to/ball-tracks
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pose.db_cache import encode_dot_candidates  # noqa: E402
from posetrak.db.db import generate_id, open_session  # noqa: E402
from posetrak.db.manage_marker_body import import_marker_body  # noqa: E402
from posetrak.detection.dot_blob_detector import BlobCandidate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER_BODY_YAML = REPO_ROOT / "catalog" / "ball.single-reflective-dot.2026-09-06-kare-tests.yaml"

CAPTURE_ID = "b21fa02d-2a82-4a37-a749-a156116a0aa0"
SYNC_CONFIG_ID = "f4acaa06-e663-43ee-870e-13ee4f112892"
TRIAL_ID = "d9ef6092-e629-43e0-9ccb-132761e4a547"
NELLI_SEQUENCE_ID = "ec1b3e2f-1ef8-4e31-806c-33102a969ecd"  # source of gopro13_02's own dots

# camera label -> Blender export CSV filename (skipping the stray 1-point
# nelli-gopro11_01...__Track.csv -- Track_001 is the real track).
BLENDER_TRACKS = {
    "insta_ace2_pro": "nelli-ace2pro-4k-120fps_mp4__Track.csv",
    "gopro-11_mini_01": "nelli-gopro11_01-4k-120fps_MP4__Track_001.csv",
    "gopro13_01": "nelli-gopro13_01-4k-120fps_MP4__Track.csv",
}
# gopro13_02: copy real dots for this time span instead of a Blender track.
COPY_DOTS_CAMERA = "gopro13_02"
COPY_DOTS_TIME_RANGE = (56.5, 70.5)  # covers all 4 throws with margin

# Nominal candidate size for the encoded dot blob -- unused by dot
# assignment beyond its elongation/noise-inflation term (real streak
# detection doesn't apply to a manually-tracked point at all), so an exact
# value doesn't matter; matches a typical small reflective dot.
_NOMINAL_AREA_PX = 50.0


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _timestamp_for_frame(conn, camera_instance_id: str, video_frame: int) -> float | None:
    row = conn.execute(
        "SELECT timestamp_s FROM pose_observations"
        " WHERE camera_instance_id=? AND video_frame=? AND source='body' LIMIT 1",
        (camera_instance_id, video_frame),
    ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT timestamp_s FROM pose_observations"
            " WHERE camera_instance_id=? AND video_frame=? LIMIT 1",
            (camera_instance_id, video_frame),
        ).fetchone()
    return row[0] if row is not None else None


def _encode_point(cx: float, cy: float) -> bytes:
    diameter = 2.0 * (_NOMINAL_AREA_PX / 3.14159265) ** 0.5
    r = diameter / 2.0
    candidate = BlobCandidate(
        cx=cx, cy=cy, area=_NOMINAL_AREA_PX, compactness=1.0,
        bbox=(int(cx - r), int(cy - r), int(diameter), int(diameter)),
        major_axis_px=diameter, minor_axis_px=diameter,
        dir_x=0.0, dir_y=0.0, tracklet_id=0.0,
    )
    return encode_dot_candidates([candidate])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    ap.add_argument("--blender-csv-dir", required=True)
    ap.add_argument("--object-name", default="ball (blender-tracked, 2026-09-17)")
    args = ap.parse_args()

    session = open_session(Path(args.session))
    csv_dir = Path(args.blender_csv_dir)
    cam_label_to_id = {
        r["label"]: r["id"] for r in session.execute("SELECT id, label FROM camera_instances")
    }

    body_id = import_marker_body(
        session, MARKER_BODY_YAML, name="ball-2026-09-06-kare-tests",
        source="Same marker body as the earlier Cutie attempt -- a single reflective_dot "
               "placeholder, no real multi-marker geometry (see that YAML's own header).",
    )
    print(f"marker_body_definition_id: {body_id}")

    object_id = generate_id()
    session.execute(
        "INSERT INTO capture_objects (id, capture_id, name, marker_body_definition_id, notes, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (object_id, CAPTURE_ID, args.object_name, body_id,
         "Registered by finalize_ball_blender_detection.py, 2026-09-17 -- position from "
         "Blender's Movie Clip Editor 2D tracker (manual, 2-3 re-inits per camera to survive "
         "occlusion), plus gopro13_02's existing real reflective-dot data for the same span.",
         _now()),
    )
    print(f"capture_object_id: {object_id}")

    detection_run_id = generate_id()
    session.execute(
        "INSERT INTO detection_runs "
        "(id, shot_id, sync_config_id, trial_id, time_start_s, time_end_s, detector_model, "
        " pose_model, detector_conf, pose_conf_threshold, status, created_at, completed_at, "
        " detector_type, config_json, capture_object_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (detection_run_id, CAPTURE_ID, SYNC_CONFIG_ID, TRIAL_ID,
         COPY_DOTS_TIME_RANGE[0], COPY_DOTS_TIME_RANGE[1],
         "blender-movie-clip-tracker + gopro13_02-existing-dots", "", 0.0, 0.0, "complete",
         _now(), _now(), "blob",
         '{"method": "manual 2D point tracking in Blender Movie Clip Editor '
         '(insta_ace2_pro, gopro-11_mini_01, gopro13_01), plus gopro13_02 real reflective-dot '
         'candidates copied unfiltered for the same span", '
         '"script": "tools/finalize_ball_blender_detection.py"}',
         object_id),
    )
    print(f"detection_run_id: {detection_run_id}")

    sequence_id = generate_id()
    session.execute(
        "INSERT INTO pose_observation_sequences "
        "(id, shot_id, sync_config_id, time_start_s, time_end_s, name, pose_model, notes, "
        " pixels_are_undistorted, detection_run_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (sequence_id, CAPTURE_ID, SYNC_CONFIG_ID,
         COPY_DOTS_TIME_RANGE[0], COPY_DOTS_TIME_RANGE[1], None,
         "blender-2d-track", "Registered by finalize_ball_blender_detection.py, 2026-09-17",
         0, detection_run_id),
    )
    print(f"pose_observation_sequence_id: {sequence_id}")

    n_rows = 0
    n_skipped_no_timestamp = 0

    for cam_label, csv_name in BLENDER_TRACKS.items():
        camera_instance_id = cam_label_to_id[cam_label]
        csv_path = csv_dir / csv_name
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        written = 0
        for row in rows:
            video_frame = int(row["video_frame"])
            ts = _timestamp_for_frame(session, camera_instance_id, video_frame)
            if ts is None:
                n_skipped_no_timestamp += 1
                continue
            blob = _encode_point(float(row["pixel_x"]), float(row["pixel_y"]))
            session.execute(
                "INSERT INTO pose_observations "
                "(sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, "
                " source, detection_run_id, kp_blob, noise_scale) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (sequence_id, camera_instance_id, video_frame, ts, 0, "dots",
                 detection_run_id, blob, 1.0),
            )
            written += 1
            n_rows += 1
        print(f"  {cam_label:20s} {written}/{len(rows)} rows from {csv_name}")

    # gopro13_02: copy every raw dot candidate for the span, unfiltered.
    copy_cam_id = cam_label_to_id[COPY_DOTS_CAMERA]
    copy_rows = session.execute(
        "SELECT video_frame, timestamp_s, kp_blob, noise_scale FROM pose_observations"
        " WHERE sequence_id=? AND camera_instance_id=? AND source='dots'"
        " AND timestamp_s BETWEEN ? AND ?",
        (NELLI_SEQUENCE_ID, copy_cam_id, *COPY_DOTS_TIME_RANGE),
    ).fetchall()
    for video_frame, ts, blob, noise_scale in copy_rows:
        session.execute(
            "INSERT INTO pose_observations "
            "(sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, "
            " source, detection_run_id, kp_blob, noise_scale) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (sequence_id, copy_cam_id, video_frame, ts, 0, "dots",
             detection_run_id, blob, noise_scale),
        )
        n_rows += 1
    print(f"  {COPY_DOTS_CAMERA:20s} {len(copy_rows)} raw dot candidate rows copied "
          f"(unfiltered, includes body-marker clutter -- assignment sorts it out)")

    session.commit()
    if n_skipped_no_timestamp:
        print(f"\nWARNING: {n_skipped_no_timestamp} Blender-tracked rows skipped -- "
              f"no pose_observations row exists at that exact (camera, video_frame) to "
              f"borrow a timestamp from")
    print(f"\nWrote {n_rows} pose_observations rows")
    print(f"\nTrack it with:\n"
          f"  posetrak-tracker track --session-db {args.session} --sequence {sequence_id} "
          f"--skeleton <ball skeleton id> --tracker-config <ball tracker_config id> --person-id 0 "
          f"--start-time <t0> --end-time <t1> --seed-position <x> <y> <z> "
          f"--output-dir <dir> --smooth")


if __name__ == "__main__":
    main()

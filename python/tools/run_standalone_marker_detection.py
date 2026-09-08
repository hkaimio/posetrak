# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""run_standalone_marker_detection.py — run ArUco + reflective-dot detection
for a trial with no registered capture_objects/marker_body_definitions yet
(design doc's "standalone" sub-phase 1a mode -- see
posetrak.detection.marker_pipeline.MarkerDetectionPipeline's own docstring).

Unlike run_object_marker_detection.py, this does not require
--capture-object-id -- it's for exactly the case this project doesn't have
tooling for yet (per the 2026-09-07 productization-plan doc): raw marker
material (props and/or persons both wearing ArUco/dots) gathered before any
object/body has been characterized or registered, valuable as test-case
data for phase 3 and UC2 prototyping on its own.

Usage:
    python tools/run_standalone_marker_detection.py \\
        --session /path/to/session.db \\
        --shot-id <capture id> --sync-config-id <id> --trial-id <id> \\
        --time-start 33.62 --time-end 130.185 \\
        --dictionary DICT_4X4_50 \\
        --detect-dots-camera-label gopro13_01 gopro13_02 ...
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posetrak.detection.marker_pipeline import MarkerDetectionPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True, help="captures.id")
    ap.add_argument("--sync-config-id", required=True)
    ap.add_argument("--trial-id", default=None)
    ap.add_argument("--time-start", type=float, required=True)
    ap.add_argument("--time-end", type=float, required=True)
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    ap.add_argument("--marker-ids", nargs="+", required=True,
                     help="Every coded marker id the scene may show, e.g. 0 1 2 3 16 17 34 37 -- "
                          "required (not optional) because it fixes the detection_keypoints "
                          "corner-slot layout; an id left out here is silently dropped for the "
                          "whole run, not just skipped opportunistically.")
    ap.add_argument("--min-marker-perimeter-rate", type=float, default=0.01)
    ap.add_argument("--detect-dots-camera-label", nargs="*", default=[],
                     help="camera_instances.label values to run reflective-dot detection on "
                          "(in addition to ArUco, which always runs for every camera).")
    ap.add_argument("--dot-bg-subtract", action="store_true",
                     help="Enables 'subtract' background_mode's residual detection. Not needed for "
                          "--dot-background-mode blacklist, which computes its own background "
                          "regardless of this flag.")
    ap.add_argument("--dot-threshold", type=int, default=235)
    ap.add_argument("--dot-threshold-by-camera", nargs="*", default=[], metavar="LABEL=THRESHOLD",
                     help="Per-camera threshold override, e.g. --dot-threshold-by-camera "
                          "pixel9=180 gopro13_02=200 -- see dot_blob_detector.py's own docstring "
                          "for why two cameras' sensors/tone-mapping can need very different "
                          "absolute thresholds for the same real markers under 'blacklist' mode.")
    ap.add_argument("--dot-background-mode", choices=["subtract", "blacklist"], default="subtract",
                     help="'subtract' (default, unchanged): threshold the background-subtracted "
                          "residual. 'blacklist' (2026-09-08): threshold raw brightness directly "
                          "and use the background only to veto a spot that's already nearly as "
                          "bright with no subject present -- see dot_blob_detector.py's own "
                          "docstring for why 'subtract' fuses a marker into the subject's own "
                          "limb on an atypical pose, and status.md for the real-data validation.")
    ap.add_argument("--dot-blacklist-frac", type=float, default=0.7)
    ap.add_argument("--dot-blacklist-radius-px", type=int, default=6)
    ap.add_argument("--dot-max-saturation", type=float, default=255.0)
    ap.add_argument("--dot-bg-sample-count", type=int, default=40)
    args = ap.parse_args()

    conn = sqlite3.connect(args.session)
    conn.row_factory = sqlite3.Row

    detect_dots_for_cameras = set()
    for label in args.detect_dots_camera_label:
        row = conn.execute("SELECT id FROM camera_instances WHERE label = ?", (label,)).fetchone()
        if row is None:
            raise ValueError(f"no camera_instances row with label {label!r}")
        detect_dots_for_cameras.add(row["id"])
    if args.detect_dots_camera_label:
        print(f"Dot detection enabled for: {args.detect_dots_camera_label} "
              f"({sorted(c[:8] for c in detect_dots_for_cameras)})")

    dot_threshold_by_camera = {}
    for entry in args.dot_threshold_by_camera:
        label, _, value = entry.partition("=")
        row = conn.execute("SELECT id FROM camera_instances WHERE label = ?", (label,)).fetchone()
        if row is None:
            raise ValueError(f"--dot-threshold-by-camera: no camera_instances row with label {label!r}")
        dot_threshold_by_camera[row["id"]] = int(value)

    pipeline = MarkerDetectionPipeline(
        session=conn,
        shot_id=args.shot_id,
        sync_config_id=args.sync_config_id,
        time_start_s=args.time_start,
        time_end_s=args.time_end,
        marker_ids=args.marker_ids,
        dictionary=args.dictionary,
        min_marker_perimeter_rate=args.min_marker_perimeter_rate,
        trial_id=args.trial_id,
        capture_object_id=None,
        detect_dots_for_cameras=detect_dots_for_cameras,
        dot_bg_subtract=args.dot_bg_subtract,
        dot_threshold=args.dot_threshold,
        dot_threshold_by_camera=dot_threshold_by_camera,
        dot_background_mode=args.dot_background_mode,
        dot_blacklist_frac=args.dot_blacklist_frac,
        dot_blacklist_radius_px=args.dot_blacklist_radius_px,
        dot_max_saturation=args.dot_max_saturation,
        dot_bg_sample_count=args.dot_bg_sample_count,
    )

    def on_progress(done, total, cam_label):
        if done % 200 == 0 or done == total:
            print(f"  {cam_label}: {done}/{total}", flush=True)

    def on_camera_done(n_done, n_total):
        print(f"Camera {n_done}/{n_total} done", flush=True)

    result = pipeline.run(on_progress=on_progress, on_camera_done=on_camera_done)
    print(f"\ndetection_run_id: {result.detection_run_id}")
    print(f"status: {result.status}")
    print(f"cameras_processed: {result.cameras_processed}")
    print(f"frames_processed: {result.frames_processed}")
    conn.close()


if __name__ == "__main__":
    main()

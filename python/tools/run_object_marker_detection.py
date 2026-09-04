# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""run_object_marker_detection.py — run coded-marker (and optionally
reflective-dot) detection for an existing capture_objects row, from the
command line.

The GUI's run-detection dialog (python/app/pose/main.py) is the normal
way to do this, but it doesn't expose
posetrak.detection.marker_pipeline.MarkerDetectionPipeline's
detect_dots_for_cameras option yet (deliberately deferred -- see that
module's own docstring). This script is the standalone stand-in, matching
this project's own "standalone script before GUI wiring" precedent
(calibrate_rigid_marker_body.py, annotate_dots_manually.py).

Usage:
    python tools/run_object_marker_detection.py \\
        --session /path/to/session.db \\
        --capture-object-id <id> \\
        --sync-config-id <id> \\
        --time-start 34.376 --time-end 100.618 \\
        --detect-dots-camera-label gopro-11_mini_01 gopro-11_mini_02
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posetrak.detection.marker_pipeline import load_pipeline_for_capture_object  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--capture-object-id", required=True)
    ap.add_argument("--sync-config-id", required=True)
    ap.add_argument("--time-start", type=float, required=True)
    ap.add_argument("--time-end", type=float, required=True)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--min-marker-perimeter-rate", type=float, default=None)
    ap.add_argument("--detect-dots-camera-label", nargs="*", default=[],
                    help="camera_instances.label values to additionally run reflective-dot "
                         "blob detection on (e.g. the ring-lit GoPros) -- resolved to "
                         "camera_instance_id internally. Omit to disable dot detection "
                         "entirely (coded-marker detection always runs for every camera).")
    args = ap.parse_args()

    conn = sqlite3.connect(args.session)
    conn.row_factory = sqlite3.Row

    detect_dots_for_cameras = set()
    for label in args.detect_dots_camera_label:
        row = conn.execute(
            "SELECT id FROM camera_instances WHERE label = ?", (label,)
        ).fetchone()
        if row is None:
            raise ValueError(f"no camera_instances row with label {label!r}")
        detect_dots_for_cameras.add(row["id"])
    if args.detect_dots_camera_label:
        print(f"Dot detection enabled for: {args.detect_dots_camera_label} "
              f"({sorted(c[:8] for c in detect_dots_for_cameras)})")

    pipeline = load_pipeline_for_capture_object(
        conn, args.capture_object_id, args.sync_config_id, args.time_start, args.time_end,
        min_marker_perimeter_rate=args.min_marker_perimeter_rate, frame_step=args.frame_step,
        detect_dots_for_cameras=detect_dots_for_cameras,
    )
    print(f"Cameras: {[c.label or c.camera_instance_id[:8] for c in pipeline.cameras]}")

    def on_progress(done, total, cam_label):
        if done % 200 == 0 or done == total:
            print(f"  {cam_label}: {done}/{total}", flush=True)

    def on_camera_done(n_done, n_total):
        print(f"Camera {n_done}/{n_total} done")

    result = pipeline.run(on_progress=on_progress, on_camera_done=on_camera_done)
    print(f"\ndetection_run_id: {result.detection_run_id}")
    print(f"status: {result.status}")
    print(f"cameras_processed: {result.cameras_processed}")
    print(f"frames_processed: {result.frames_processed}")
    conn.close()


if __name__ == "__main__":
    main()

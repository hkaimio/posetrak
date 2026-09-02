# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""annotate_dots_manually.py — manual reflective-dot calibration by
clicking, for a rigid marker body whose coded (ArUco) markers are already
calibrated.

See docs/roadmap/features/marker-based-mocap/reflective-dot-detection-design.md
§3.1 and calibrate_rigid_marker_body.py's own module docstring for why
this exists alongside the automatic (--detect-dots) path: real per-instant
reference+dot co-occurrence in the "Weapon test 2026-08-20" capture turned
out too sparse for even a correctly-strict automatic approach (2-view
reprojection checking alone isn't a strong enough correspondence
guarantee -- confirmed the hard way against that real footage) to
accumulate enough samples. A human confirming "yes, that bright spot in
camera A and camera B are the same physical dot" sidesteps the
correspondence problem entirely, so only a handful of good instants are
needed per dot rather than the many an automatic approach needs to
average out false positives.

Reuses calibrate_rigid_marker_body.py's own
triangulate_point_multi_view() (same reprojection-checked DLT
triangulation) and the identical reference-marker-local-frame transform --
the only new piece here is getting the per-camera pixel click instead of
an automatic blob detection.

Workflow: for each timestamp you name (a moment you already know shows
the dot clearly in 2+ cameras), the reference marker is solved from that
same instant's ArUco detections, then each camera's frame at that instant
is shown one at a time -- click the dot's pixel, or press 's' to skip
this camera for this dot. Samples across every timestamp for one dot are
averaged into its final local-frame position. Reads an existing
marker_body_definitions YAML (already calibrated for its coded markers,
e.g. calibrate_rigid_marker_body.py's own output) and writes a new one
with the manually-triangulated dots appended.

Usage:
    python tools/annotate_dots_manually.py \\
        --session /path/to/session.db \\
        --shot-id <capture_id> \\
        --dictionary DICT_4X4_50 --marker-size 0.095 \\
        --marker-ids 2 3 --reference-id 2 \\
        --dot-names dot0 dot1 \\
        --timestamps 40.2 55.7 61.0 \\
        --input sword_body.yaml \\
        --output sword_body_with_dots.yaml

Controls while a frame window is shown: left-click to mark the dot (click
again to move the mark before confirming), Enter/Space to confirm and
move on, 's' to skip this camera for this dot, 'q' to abort.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.setup.extrinsics_solver import _undistort_pts, marker_local_corners, solve_marker_pose  # noqa: E402
from app.setup.fiducial_markers import ArucoDetector  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402
from tools.calibrate_rigid_marker_body import (  # noqa: E402
    load_camera_states,
    load_sync_table,
    robust_mean,
    triangulate_point_multi_view,
)


def _grab_frame(file_path: str, video_frame: int) -> np.ndarray | None:
    for _, img in iter_frames(file_path, video_frame, video_frame + 1):
        return img
    return None


def _click_pixel(window_name: str, img: np.ndarray) -> tuple[float, float] | None:
    """Show *img*, let the user click a point, confirm/skip/abort.

    Returns the clicked (u, v) in the image's own pixel space, or None if
    skipped. Raises SystemExit if the user aborts ('q').
    """
    marked: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            marked[:] = [(x, y)]

    display = img.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        frame = display.copy()
        if marked:
            cv2.drawMarker(frame, marked[0], (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, "click dot, Enter=confirm, s=skip, q=quit", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32) and marked:  # Enter or Space
            return float(marked[0][0]), float(marked[0][1])
        if key == ord("s"):
            return None
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Aborted by user")


def annotate_one_instant(
    dot_name: str, timestamp: float, states: dict, sync_table, svid_by_cam: dict,
    detector: ArucoDetector, ref_id: str, marker_size: float, min_cameras: int,
    max_reprojection_px: float,
) -> np.ndarray | None:
    """Solve the reference marker's pose at *timestamp* and let the user
    click *dot_name* in every camera with a frame at that instant.

    Returns the dot's local-frame (reference marker's frame) 3D position
    for this instant, or None if the reference couldn't be solved or fewer
    than *min_cameras* usable clicks were made.
    """
    frames: dict[str, tuple[np.ndarray, int]] = {}
    for cam_id, svid in svid_by_cam.items():
        if cam_id not in states:
            continue
        video_frame = sync_table.lookup(timestamp, svid)
        if video_frame is None:
            continue
        img = _grab_frame(states[cam_id].file_path, video_frame)
        if img is not None:
            frames[cam_id] = (img, video_frame)

    # Solve the reference marker's pose from this instant's own ArUco detections.
    ref_obs: dict[str, np.ndarray] = {}
    for cam_id, (img, video_frame) in frames.items():
        dets = detector.detect(img, video_id=cam_id, frame_idx=video_frame)
        for d in dets:
            if d.marker_id == ref_id:
                pts = np.array([(c.px, c.py) for c in d.corners], dtype=np.float64)
                ref_obs[cam_id] = _undistort_pts(pts, states[cam_id])
    if len(ref_obs) < min_cameras:
        print(f"  t={timestamp}: reference marker '{ref_id}' seen in only "
              f"{len(ref_obs)} camera(s) -- skipping this instant")
        return None
    try:
        rvec_ref, tvec_ref, _rms = solve_marker_pose(ref_obs, states, marker_size)
    except (ValueError, RuntimeError) as e:
        print(f"  t={timestamp}: reference marker pose solve failed ({e}) -- skipping")
        return None
    R_ref, _ = cv2.Rodrigues(rvec_ref)

    print(f"  t={timestamp}: reference solved from {len(ref_obs)} camera(s) -- "
          f"click '{dot_name}' in each frame")
    clicked: dict[str, tuple[float, float]] = {}
    for cam_id, (img, _video_frame) in frames.items():
        pixel = _click_pixel(f"{dot_name} @ t={timestamp} -- {cam_id[:8]}", img)
        cv2.destroyAllWindows()
        if pixel is not None:
            undist = _undistort_pts(np.array([pixel], dtype=np.float64), states[cam_id])
            clicked[cam_id] = (float(undist[0, 0]), float(undist[0, 1]))

    if len(clicked) < min_cameras:
        print(f"  t={timestamp}: only {len(clicked)} camera(s) clicked -- skipping")
        return None
    world_pt = triangulate_point_multi_view(clicked, states, max_reprojection_px=max_reprojection_px)
    if world_pt is None:
        print(f"  t={timestamp}: triangulation/reprojection check failed -- skipping")
        return None
    return R_ref.T @ (world_pt - tvec_ref.flatten())


def _fmt(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"


def write_marker_body_yaml(name: str, markers: list[dict], out_path: Path) -> None:
    """Serialize *markers* in the same inline-numeric-list style
    calibrate_rigid_marker_body.py's own output uses, regardless of
    whether an entry came from the input file or was newly annotated --
    reusing PyYAML's default formatting here would make the two look
    inconsistent."""
    lines = [f"name: {name}", "units: meters", "markers:"]
    for m in markers:
        lines.append(f"  - name: {m['name']}")
        lines.append(f"    type: {m['type']}")
        if m["type"] == "aruco":
            lines.append(f"    dictionary: {m['dictionary']}")
            lines.append(f'    id: "{m["id"]}"')
            lines.append(f"    size: {m['size']}")
            lines.append("    corners:")
            for c in m["corners"]:
                lines.append(f"      - {_fmt(np.asarray(c, dtype=np.float64))}")
        elif m["type"] == "reflective_dot":
            lines.append(f"    center: {_fmt(np.asarray(m['center'], dtype=np.float64))}")
        else:
            raise ValueError(f"unknown marker type {m['type']!r} for {m['name']!r}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--shot-id", required=True)
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    ap.add_argument("--marker-size", type=float, required=True)
    ap.add_argument("--marker-ids", nargs="+", required=True)
    ap.add_argument("--reference-id", required=True)
    ap.add_argument("--dot-names", nargs="+", required=True)
    ap.add_argument("--timestamps", nargs="+", type=float, required=True,
                    help="Instants (seconds) where the dots are expected to be visible "
                         "in >=2 cameras -- each dot is annotated at every timestamp given")
    ap.add_argument("--min-cameras", type=int, default=2)
    ap.add_argument("--max-reprojection-px", type=float, default=8.0,
                    help="Looser than the automatic path's default (5.0) -- a human click "
                         "is a few pixels less precise than a detected blob centroid")
    ap.add_argument("--input", required=True, help="Existing marker_body_definitions YAML "
                    "(coded markers already calibrated)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    print("Loading camera states + extrinsics...")
    states = load_camera_states(conn, args.shot_id)
    print(f"  {len(states)} cameras with solved extrinsics")
    sync_table, svid_by_cam = load_sync_table(conn, args.shot_id)
    detector = ArucoDetector(dictionary=args.dictionary)

    ref_id = args.reference_id
    if ref_id not in set(args.marker_ids):
        raise ValueError("--reference-id must be one of --marker-ids")

    existing = yaml.safe_load(Path(args.input).read_text(encoding="utf-8"))
    markers: list[dict] = list(existing.get("markers", []))
    existing_dot_names = {m["name"] for m in markers if m.get("type") == "reflective_dot"}

    for dot_name in args.dot_names:
        if dot_name in existing_dot_names:
            print(f"'{dot_name}' already exists in {args.input} -- re-annotating will replace it")
        print(f"\n=== Annotating '{dot_name}' ===")
        samples = []
        for t in args.timestamps:
            sample = annotate_one_instant(
                dot_name, t, states, sync_table, svid_by_cam, detector, ref_id,
                args.marker_size, args.min_cameras, args.max_reprojection_px,
            )
            if sample is not None:
                samples.append(sample)
        if not samples:
            print(f"  '{dot_name}': no usable samples across any timestamp -- not written")
            continue
        center = robust_mean(np.stack(samples), trim_frac=0.0) if len(samples) > 1 else samples[0]
        if len(samples) > 1:
            spread = np.stack(samples).std(axis=0)
            print(f"  '{dot_name}': {len(samples)} samples, per-axis std (m) {spread}")
        markers = [m for m in markers if m.get("name") != dot_name]
        markers.append({"name": dot_name, "type": "reflective_dot", "center": center})

    write_marker_body_yaml(existing.get("name", "calibrated-rigid-body"), markers, Path(args.output))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

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
at least one dot clearly in 2+ cameras), the reference marker is solved
from that same instant's ArUco detections, then each camera's frame at
that instant is shown *once* -- label every one of --dot-names that's
visible in that single frame (click a point, press the digit key for
which dot it is, repeat for every dot visible in this camera), then
confirm the whole frame and move to the next camera. Not every dot needs
to be visible in every camera, or at every timestamp -- whatever gets
labeled for a given dot, across however many cameras/timestamps, gets
triangulated and averaged into that dot's final local-frame position.
Reads an existing marker_body_definitions YAML (already calibrated for
its coded markers, e.g. calibrate_rigid_marker_body.py's own output) and
writes a new one with the manually-triangulated dots appended.

Usage:
    python tools/annotate_dots_manually.py \\
        --session /path/to/session.db \\
        --shot-id <capture_id> \\
        --dictionary DICT_4X4_50 --marker-size 0.095 \\
        --marker-ids 2 3 --reference-id 2 \\
        --dot-names dot0 dot1 dot2 \\
        --timestamps 40.2 55.7 61.0 \\
        --input sword_body.yaml \\
        --output sword_body_with_dots.yaml

Controls while a frame window is shown: left-click a point, then press
its digit key (0 for the first --dot-names entry, 1 for the second, ...)
to label it -- repeat for every dot visible in this frame. 'u' undoes the
last labeled point. Enter/Space confirms the whole frame (including zero
labels, if nothing here is visible) and moves to the next camera. 'q'
aborts entirely. At most 10 dot names per run (digit keys 0-9).
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


def _click_multiple_pixels(
    window_name: str, img: np.ndarray, dot_names: list[str]
) -> dict[str, tuple[float, float]]:
    """Show *img*; click a point then press its digit key (0 for
    dot_names[0], 1 for dot_names[1], ...) to label it, repeating for
    every dot visible in this one frame, Enter/Space to confirm the whole
    set and move on.

    Returns {dot_name: (u, v)} in the image's own pixel space, for
    however many of *dot_names* got labeled here (0 or more -- not every
    dot needs to be visible in every camera). Raises SystemExit if the
    user aborts ('q').
    """
    if len(dot_names) > 10:
        raise ValueError("at most 10 dot names supported per run (digit keys 0-9)")

    labeled: dict[str, tuple[int, int]] = {}
    order: list[str] = []  # labeling order, so 'u' undoes the most recent one
    pending: list[tuple[int, int]] = []  # clicked but not yet assigned a digit

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending[:] = [(x, y)]

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    legend = "  ".join(f"{i}:{n}" for i, n in enumerate(dot_names))
    while True:
        frame = img.copy()
        for name, (x, y) in labeled.items():
            cv2.drawMarker(frame, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(frame, name, (x + 14, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0), 2)
        if pending:
            cv2.drawMarker(frame, pending[0], (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"click + digit to label: {legend}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Enter=confirm frame, u=undo, q=quit", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):  # Enter or Space
            return {name: (float(x), float(y)) for name, (x, y) in labeled.items()}
        if key == ord("u"):
            if order:
                del labeled[order.pop()]
        elif key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Aborted by user")
        elif pending and ord("0") <= key <= ord("9"):
            idx = key - ord("0")
            if idx < len(dot_names):
                name = dot_names[idx]
                if name in labeled:
                    order.remove(name)
                labeled[name] = pending[0]
                order.append(name)
                pending.clear()


def annotate_one_instant(
    dot_names: list[str], timestamp: float, states: dict, sync_table, svid_by_cam: dict,
    detector: ArucoDetector, ref_id: str, marker_size: float, min_cameras: int,
    max_reprojection_px: float,
) -> dict[str, np.ndarray]:
    """Solve the reference marker's pose at *timestamp*, show each camera's
    frame once, and let the user label as many of *dot_names* as are
    visible per camera in a single pass.

    Returns {dot_name: local-frame 3D position} for every dot with enough
    usable labels this instant (>=min_cameras cameras, passing the
    reprojection check) -- a dot with too few labels, or none at all,
    simply isn't a key in the result. Empty if the reference marker
    itself couldn't be solved from >=min_cameras cameras.
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
        return {}
    try:
        rvec_ref, tvec_ref, _rms = solve_marker_pose(ref_obs, states, marker_size)
    except (ValueError, RuntimeError) as e:
        print(f"  t={timestamp}: reference marker pose solve failed ({e}) -- skipping")
        return {}
    R_ref, _ = cv2.Rodrigues(rvec_ref)

    print(f"  t={timestamp}: reference solved from {len(ref_obs)} camera(s) -- "
          f"label any of {dot_names} visible in each frame")
    clicked_by_dot: dict[str, dict[str, tuple[float, float]]] = {name: {} for name in dot_names}
    for cam_id, (img, _video_frame) in frames.items():
        points = _click_multiple_pixels(f"t={timestamp} -- {cam_id[:8]}", img, dot_names)
        cv2.destroyAllWindows()
        for dot_name, pixel in points.items():
            undist = _undistort_pts(np.array([pixel], dtype=np.float64), states[cam_id])
            clicked_by_dot[dot_name][cam_id] = (float(undist[0, 0]), float(undist[0, 1]))

    results: dict[str, np.ndarray] = {}
    for dot_name, clicked in clicked_by_dot.items():
        if not clicked:
            continue  # not visible anywhere this instant -- not an error
        if len(clicked) < min_cameras:
            print(f"  t={timestamp}: '{dot_name}' labeled in only {len(clicked)} camera(s) "
                  "-- skipping")
            continue
        world_pt = triangulate_point_multi_view(clicked, states, max_reprojection_px=max_reprojection_px)
        if world_pt is None:
            print(f"  t={timestamp}: '{dot_name}' triangulation/reprojection check failed "
                  "-- skipping")
            continue
        results[dot_name] = R_ref.T @ (world_pt - tvec_ref.flatten())
    return results


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
    for name in args.dot_names:
        if name in existing_dot_names:
            print(f"'{name}' already exists in {args.input} -- re-annotating will replace it")

    # One pass per timestamp (not per dot): each camera's frame at that instant
    # is shown once, and however many of --dot-names are visible get labeled
    # together -- see the module docstring for why (avoids revisiting the same
    # image once per dot when several are visible in one view, which real
    # footage of this sword clearly shows).
    samples_by_dot: dict[str, list[np.ndarray]] = {name: [] for name in args.dot_names}
    for t in args.timestamps:
        results = annotate_one_instant(
            args.dot_names, t, states, sync_table, svid_by_cam, detector, ref_id,
            args.marker_size, args.min_cameras, args.max_reprojection_px,
        )
        for name, sample in results.items():
            samples_by_dot[name].append(sample)

    for name, samples in samples_by_dot.items():
        if not samples:
            print(f"\n'{name}': no usable samples across any timestamp -- not written")
            continue
        center = robust_mean(np.stack(samples), trim_frac=0.0) if len(samples) > 1 else samples[0]
        if len(samples) > 1:
            spread = np.stack(samples).std(axis=0)
            print(f"\n'{name}': {len(samples)} samples, per-axis std (m) {spread}")
        markers = [m for m in markers if m.get("name") != name]
        markers.append({"name": name, "type": "reflective_dot", "center": center})

    write_marker_body_yaml(existing.get("name", "calibrated-rigid-body"), markers, Path(args.output))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

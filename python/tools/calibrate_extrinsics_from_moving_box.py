# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""calibrate_extrinsics_from_moving_box.py — PROTOTYPE: multi-camera
extrinsics calibration from a single handheld ArUco calibration box, moved
around the scene in view of every camera and finally set down in a known
position.

Status: prototype only, not wired into production (mirrors this project's
own precedent -- prototype_streak_detector.py, calibrate_harness_from_orbit.py
-- validate against real footage first, then fold into
posetrak.cli.extrinsics_rig's "anchor-rig" command if it works, rather than
keeping this as a separate one-off forever).

Method (Harri, 2026-09-06)
---------------------------
Filming N static cameras is easy to sync but each one individually only
gets a fleeting, motion-blurred glimpse of a box someone is carrying around
-- not enough for a clean per-camera PnP. Instead of asking for a
deliberate pause-and-hold at every camera, this scans ONE reference
camera's footage for moments the box was *already* naturally stationary
(no motion, no blur -- picked up, set down while adjusting grip, etc.),
and at each such moment looks up the SAME synced instant in every other
camera via this session's already-solved SyncTable. Standing still for even
a fraction of a second gives every camera watching at that moment a clean,
simultaneous detection of the box in one fixed pose -- exactly the kind of
snapshot ordinary "hold up a board" calibration relies on, just harvested
opportunistically from natural handling instead of requested up front.

The box's FINAL resting position anchors the whole solve in a real,
known-by-construction world frame (see BOX geometry section below for the
axis convention) -- ``anchor_from_marker_rig()`` turns that one window's
detections into fixed-world-xyz ``ControlPoint``s, exactly like the
existing single-frame ``anchor-rig`` CLI command already does. Every OTHER
stationary window found earlier in the clip contributes the box's marker
corners as *free* (unfixed) points instead, via ``MarkerGroup`` -- each
window's corners get their own per-window-unique key
(``win{i}_{marker_id}``) so a corner from one window is never confused
with the same physical marker's corner from a different window, which
would be geometrically wrong (the box moved in between). ``run_calibration``
already supports this mix directly: PnP-initialise whichever cameras see
the final position clearly, run the existing SIFT-based background
matching to bootstrap any camera that doesn't, then bundle-adjust
everything (final-position CPs + every window's free marker corners +
background SIFT points) together. No new solver code was needed for the
core idea -- this script is the box-geometry model, the stationary-window
finder, and the per-window bookkeeping around already-built machinery.

The BOX geometry itself (below) is measured, not guessed: derived by
``characterize_rig_from_video.py`` (orbit-video self-calibration, cross-
checked against the box's own known 49cm/33cm/18cm dimensions -- markers
6/8's solved centers are 0.491m apart along their shared axis, matching
the measured long side to within 2mm) then re-expressed into Harri's
world-frame convention by ``reframe_box_rig.py`` (derived Z axis landed
within 1.3 deg of marker 1's own independently-measured outward normal --
confirms the face-role identification, not just internal consistency).
See ``scratch/calib_box_2026-09-06.yaml``'s own ``_provenance`` block for
exactly how it was produced.

Coverage is partial -- only markers 1, 3, 4, 6, 8 got enough multi-camera
corner coverage in the orbit video to solve (5 of the box's 8 markers;
id 0 turned out to be a DICT_4X4_50 marker mixed in by accident, ids 2/9
were seen but not with enough simultaneous multi-camera coverage in either
sampling attempt tried). Real, but not blocking -- five widely-spread
markers (a full top marker, both ends of the long axis, and a full second
face) are already enough to test this script's actual idea end to end;
extend the YAML later if more redundancy turns out to matter.
"""

from __future__ import annotations

import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sqlite3

from posetrak.detection.frame_source import iter_frames
from app.setup.db_context import SyncPoint, SyncTable
from app.setup.extrinsics_solver import (
    CamCalibState,
    MarkerGroup,
    run_calibration,
    write_extrinsics_to_db,
)
from app.setup.fiducial_markers import (
    MarkerRigDetector,
    anchor_from_marker_rig,
    load_marker_body_yaml_file,
    merge_detections_into_groups,
)
# Reusing the CLI's own already-tested intrinsics/label-resolution helpers
# rather than re-deriving them -- see extrinsics_rig.py's own docstring for
# why (this project keeps exactly one way to do this).
from posetrak.cli.extrinsics_rig import _label_to_instance_id


# ---------------------------------------------------------------------------
# Box geometry
# ---------------------------------------------------------------------------
BOX_YAML_PATH = str(Path(__file__).resolve().parents[2] / "scratch" / "calib_box_2026-09-06.yaml")
ARUCO_DICTIONARY = "DICT_5X5_100"


# ---------------------------------------------------------------------------
# Stationary-window detection
# ---------------------------------------------------------------------------


@dataclass
class StationaryWindow:
    timestamp_s: float       # representative instant, in the capture's synced timeline
    n_corners: int           # how many corners the reference camera saw there (diagnostic)
    is_final: bool = False


def find_stationary_windows(
    video_path: str,
    rig_detector: MarkerRigDetector,
    start_frame: int,
    end_frame: int,
    fps: float,
    sample_stride: int = 6,
    max_centroid_move_px: float = 6.0,
    min_run_samples: int = 3,
) -> list[tuple[int, int]]:
    """Scan one camera's frames for runs where the box's detected marker
    corners barely move frame-to-frame -- a natural pause, not a deliberate
    "hold still" cue. Returns (representative_frame_idx, n_corners_seen)
    pairs, one per run, in time order (last = closest to the end of the
    scanned range, i.e. the box's likely final position).

    Deliberately measures motion directly on detected marker corners rather
    than a generic blur/sharpness proxy (e.g. this project's own
    find_sharp_frames in calibrate_intrinsics.py) -- we're already running
    the exact detector whose output we need, so there's no reason to
    introduce a separate heuristic that might not transfer cleanly from the
    ChArUco-frame-quality problem it was built for.
    """
    samples: list[tuple[int, np.ndarray]] = []  # (frame_idx, mean corner xy)
    frame_indices = list(range(start_frame, end_frame, sample_stride))
    wanted = set(frame_indices)
    for frame_idx, img in iter_frames(video_path, start_frame, end_frame):
        if frame_idx not in wanted:
            continue
        dets = rig_detector.detect(img, video_id="__scan__", frame_idx=frame_idx)
        if not dets:
            continue
        pts = np.array([[c.px, c.py] for d in dets for c in d.corners], dtype=np.float64)
        samples.append((frame_idx, pts.mean(axis=0)))

    runs: list[tuple[int, int]] = []
    run_start = None
    for i in range(1, len(samples)):
        prev_idx, prev_c = samples[i - 1]
        cur_idx, cur_c = samples[i]
        # A sampled frame with no detection at all is skipped above rather
        # than recorded, so two consecutive *entries* in `samples` are not
        # necessarily adjacent *sampled frames* -- a gap (box briefly
        # occluded/out of frame mid-motion) must break a run even if the
        # centroid happens to land somewhere similar on either side of it,
        # otherwise a real occlusion could be misread as "didn't move."
        gap = (cur_idx - prev_idx) > sample_stride * 1.5
        moved = np.inf if gap else np.linalg.norm(cur_c - prev_c)
        if moved <= max_centroid_move_px:
            if run_start is None:
                run_start = i - 1
        else:
            if run_start is not None and (i - run_start) >= min_run_samples:
                mid = samples[(run_start + i - 1) // 2][0]
                runs.append((mid, i - run_start))
            run_start = None
    if run_start is not None and (len(samples) - run_start) >= min_run_samples:
        mid = samples[(run_start + len(samples) - 1) // 2][0]
        runs.append((mid, len(samples) - run_start))

    return runs


# ---------------------------------------------------------------------------
# Per-instant multi-camera detection
# ---------------------------------------------------------------------------


@dataclass
class _CamInfo:
    label: str
    shot_video_id: str
    file_path: str
    camera_instance_id: str


def detect_at_instant(
    timestamp_s: float,
    cams: list[_CamInfo],
    sync_table: SyncTable,
    rig_detector: MarkerRigDetector,
    frame_cache: dict[str, np.ndarray] | None = None,
) -> dict[str, list]:
    """Look up *timestamp_s* in every camera via the sync table, read that
    frame, and run the rig detector. Returns {camera_label: detections}.
    *frame_cache*, if given, is filled with one representative BGR frame per
    camera (used for the final window, whose frames also seed SIFT).
    """
    detections_by_camera: dict[str, list] = {}
    for cam in cams:
        frame_idx = sync_table.lookup(timestamp_s, cam.shot_video_id)
        if frame_idx is None:
            continue
        img = None
        for _, im in iter_frames(cam.file_path, frame_idx, frame_idx + 1):
            img = im
            break
        if img is None:
            continue
        if frame_cache is not None:
            frame_cache[cam.label] = img
        dets = rig_detector.detect(img, video_id=cam.label, frame_idx=frame_idx)
        detections_by_camera[cam.label] = dets
    return detections_by_camera


def _load_intrinsics_by_id(conn: sqlite3.Connection, calib_id: str) -> dict:
    """Resolve intrinsics directly by ``intrinsics_calibrations.id`` --
    unlike ``_resolve_intrinsics`` (label + notes-substring lookup, meant
    for a human picking a camera/mode on the command line), this capture's
    own ``capture_videos.intrinsics_calibration_id`` already unambiguously
    names the exact calibration this specific recording used, so there is
    no mode-disambiguation question to ask here at all.
    """
    import struct

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="Path to the session .db file")
    ap.add_argument("--capture", required=True, help="captures.id (or prefix) for 'Nelli 2026-09-06'")
    ap.add_argument("--reference-camera", default="gopro13_02",
                     help="camera_instances.label used to scan for stationary windows")
    ap.add_argument("--time-start", type=float, default=6.0, help="Synced-timeline seconds")
    ap.add_argument("--time-end", type=float, default=33.0, help="Synced-timeline seconds")
    ap.add_argument("--min-marker-perimeter-rate", type=float, default=0.01)
    ap.add_argument("--write", action="store_true",
                     help="Persist the solved calibration to extrinsic_calibrations/"
                          "extrinsic_entries. Without this flag, only prints results "
                          "(dry run) -- the recommended first pass for a new box.")
    args = ap.parse_args()

    conn = sqlite3.connect(args.session)
    conn.row_factory = sqlite3.Row

    capture_row = conn.execute(
        "SELECT id, session_id FROM captures WHERE id LIKE ? || '%'", (args.capture,)
    ).fetchone()
    if capture_row is None:
        raise SystemExit(f"No captures row matching {args.capture!r}")
    capture_id, session_id = capture_row["id"], capture_row["session_id"]

    cam_rows = conn.execute(
        """
        SELECT cv.id AS shot_video_id, ci.label, cv.file_path, ci.id AS camera_instance_id,
               cv.actual_fps, cm.nominal_fps, cv.intrinsics_calibration_id
        FROM capture_videos cv
        JOIN camera_instances ci ON ci.id = cv.camera_instance_id
        JOIN camera_modes cm ON cm.id = cv.camera_mode_id
        WHERE cv.shot_id = ?
        """,
        (capture_id,),
    ).fetchall()
    cams = [
        _CamInfo(label=r["label"], shot_video_id=r["shot_video_id"],
                 file_path=r["file_path"], camera_instance_id=r["camera_instance_id"])
        for r in cam_rows
    ]
    intrinsics_calib_id_by_label = {r["label"]: r["intrinsics_calibration_id"] for r in cam_rows}
    # actual_fps (measured for this specific recording) over nominal_fps
    # (the camera mode's spec value) when available -- same preference
    # ObjectCropGridWidget's own sync table construction uses.
    fps_by_cam = {r["shot_video_id"]: float(r["actual_fps"] or r["nominal_fps"] or 120.0)
                  for r in cam_rows}
    print(f"Cameras: {[c.label for c in cams]}")

    sync_config = conn.execute(
        "SELECT id FROM sync_configs WHERE shot_id = ?", (capture_id,)
    ).fetchone()
    if sync_config is None:
        raise SystemExit(f"No sync_configs row for capture {capture_id}")
    sp_rows = conn.execute(
        "SELECT camera_instance_id, shot_video_id, video_frame, timestamp_s "
        "FROM sync_points WHERE sync_config_id = ?",
        (sync_config["id"],),
    ).fetchall()
    sync_points = [
        SyncPoint(camera_instance_id="", shot_video_id=r["shot_video_id"],
                  video_frame=r["video_frame"], timestamp_s=r["timestamp_s"])
        for r in sp_rows
    ]
    sync_table = SyncTable(sync_points, fps_by_cam)

    box_config = load_marker_body_yaml_file(BOX_YAML_PATH, rig_id="calib-box-2026-09-06")
    rig_detector = MarkerRigDetector(
        box_config, dictionary=ARUCO_DICTIONARY,
        min_marker_perimeter_rate=args.min_marker_perimeter_rate,
    )

    ref_cam = next((c for c in cams if c.label == args.reference_camera), None)
    if ref_cam is None:
        raise SystemExit(f"Reference camera {args.reference_camera!r} not found in this capture")

    ref_start = sync_table.lookup(args.time_start, ref_cam.shot_video_id)
    ref_end = sync_table.lookup(args.time_end, ref_cam.shot_video_id)
    if ref_start is None or ref_end is None:
        raise SystemExit("Could not map --time-start/--time-end into the reference camera's frames")
    print(f"Scanning {ref_cam.label} frames {ref_start}-{ref_end} for stationary windows…")
    runs = find_stationary_windows(ref_cam.file_path, rig_detector, ref_start, ref_end,
                                    fps_by_cam[ref_cam.shot_video_id])
    if not runs:
        raise SystemExit(
            "No stationary windows found -- try loosening --max-centroid-move-px or "
            "check the reference camera actually sees the box moving in this range."
        )
    print(f"Found {len(runs)} stationary window(s) in {ref_cam.label}:")
    windows_ts = []
    # Convert each representative frame back to the common synced timestamp
    # via this camera's own local anchor (reuse the sync table's fps/anchor
    # for ref_cam -- SyncTable only exposes timestamp->frame, so invert
    # locally using the nearest anchor's fps).
    def frame_to_ts(frame_idx: int) -> float:
        # Piecewise-linear inverse of SyncTable.lookup for one video: find
        # the enclosing anchor pair and use its local fps, mirroring
        # SyncTable's own interpolation.
        pts = sorted((p for p in sync_points if p.shot_video_id == ref_cam.shot_video_id),
                     key=lambda p: p.video_frame)
        if len(pts) == 1:
            return pts[0].timestamp_s + (frame_idx - pts[0].video_frame) / fps_by_cam[ref_cam.shot_video_id]
        idx = bisect_right([p.video_frame for p in pts], frame_idx)
        idx = max(1, min(idx, len(pts) - 1))
        a, b = pts[idx - 1], pts[idx]
        local_fps = (b.video_frame - a.video_frame) / (b.timestamp_s - a.timestamp_s)
        return a.timestamp_s + (frame_idx - a.video_frame) / local_fps

    for frame_idx, n_corners in runs:
        ts = frame_to_ts(frame_idx)
        windows_ts.append(ts)
        print(f"  t={ts:.3f}s  ({n_corners} corners seen in {ref_cam.label})")

    final_ts = windows_ts[-1]
    earlier_ts = windows_ts[:-1]
    print(f"\nTreating t={final_ts:.3f}s as the FINAL (world-anchoring) position.")

    states: list[CamCalibState] = []
    final_frames: dict[str, np.ndarray] = {}
    final_detections = detect_at_instant(final_ts, cams, sync_table, rig_detector, final_frames)
    for cam in cams:
        intr = _load_intrinsics_by_id(conn, intrinsics_calib_id_by_label[cam.label])
        states.append(CamCalibState(
            video_id=cam.label, label=cam.label,
            K=intr["K"], K_orig=intr["K_orig"], dist=intr["dist"], fisheye=intr["fisheye"],
            image=final_frames.get(cam.label),
        ))
        n = len(final_detections.get(cam.label, []))
        print(f"  final position: {cam.label}: {n} rig marker(s)")

    final_cps = anchor_from_marker_rig(final_detections, box_config)
    if not final_cps:
        raise SystemExit(
            "Box not detected by any camera at the final position -- cannot anchor world frame."
        )

    free_groups: dict[str, MarkerGroup] = {}
    for i, ts in enumerate(earlier_ts):
        dets_by_cam = detect_at_instant(ts, cams, sync_table, rig_detector)
        n_seen = sum(len(d) for d in dets_by_cam.values())
        print(f"  window {i} (t={ts:.3f}s): {n_seen} rig marker detection(s) across cameras")
        for cam_label, dets in dets_by_cam.items():
            # Give this window's corners a unique key per marker so they
            # are never merged with the same physical marker's corners
            # from a *different* window (the box moved in between -- see
            # module docstring).
            retagged = []
            for d in dets:
                retagged.append(type(d)(
                    marker_type=d.marker_type, marker_id=f"win{i}_{d.marker_id}",
                    corners=[type(c)(marker_type=c.marker_type, marker_id=f"win{i}_{c.marker_id}",
                                      corner_index=c.corner_index, video_id=c.video_id,
                                      frame_idx=c.frame_idx, px=c.px, py=c.py)
                             for c in d.corners],
                ))
            merge_detections_into_groups(retagged, free_groups, size=None)

    print(f"\nRunning calibration: {len(final_cps)} fixed control points, "
          f"{len(free_groups)} free marker groups from {len(earlier_ts)} earlier window(s)…")
    result = run_calibration(
        states, control_points=final_cps, marker_groups=list(free_groups.values()), cp_only=False,
    )
    if result.unsolved:
        print(f"WARNING: {len(result.unsolved)} camera(s) unsolved: {result.unsolved}")

    print("\nSolved camera positions (world XYZ) and reprojection error (px):")
    for s in result.cameras.values():
        if s.R is None:
            print(f"  {s.label:20s}  unsolved")
            continue
        C = -s.R.T @ s.t.flatten()
        err = result.reprojection_errors.get(s.video_id)
        cp_err = result.cp_reprojection_errors.get(s.video_id)
        err_str = f"reproj mean={err['mean']:.2f} max={err['max']:.2f} (n={err['n']})" if err else "reproj: n/a"
        cp_str = f", CP mean={cp_err['mean']:.2f} max={cp_err['max']:.2f} (n={cp_err['n']})" if cp_err else ""
        print(f"  {s.label:20s}  ({C[0]:+.3f}, {C[1]:+.3f}, {C[2]:+.3f})  {err_str}{cp_str}")

    if args.write:
        label_to_instance = _label_to_instance_id(conn)
        calib_id = write_extrinsics_to_db(
            result, conn, session_id, label_to_instance, method="moving-box-anchor",
        )
        conn.execute(
            "UPDATE captures SET extrinsic_calibration_id = ? WHERE id = ?",
            (calib_id, capture_id),
        )
        conn.commit()
        print(f"\nWrote extrinsic_calibration_id: {calib_id} (linked to capture {capture_id})")
    else:
        print("\n(dry run -- pass --write to persist this calibration)")

    conn.close()


if __name__ == "__main__":
    main()

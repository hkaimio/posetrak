#!/usr/bin/env python3
"""test_rig_anchor_capture1.py — Test anchor_from_marker_rig (Phase 8 core)
against a real multi-camera capture, outside the GUI.

See docs/roadmap/features/extrinsics-improvements/status.md's 2026-08-11/12
entries for the plan this is step 2/3 of: anchor a real N-camera capture's
world frame from a portable, non-planar calibration rig's known geometry
(a rig config file — see tools/rig_configs/box_2026-08-10.json for how
that was built), then use the now-solved cameras to recover the world
positions of ordinary scattered ArUco tags (a Phase 5/Tier B precursor —
no scene_fiducial_markers persistence yet, just prints/saves the poses).

Takes exactly one frame per camera — this is a static-rig anchoring test,
not a video-scrubbing one, so no sync/session-DB import is needed at all
(extrinsics calibration frame choice is per-camera and independent, see
the design doc's R2).

Usage
-----
    python test_rig_anchor_capture1.py \\
        --registry-db path/to/registry.db \\
        --rig-config tools/rig_configs/box_2026-08-10.json \\
        --camera "insta_ace2_pro|4K 120 fps linear|VIDEO1.mp4|2069" \\
        --camera "gopro-11_mini_01|HERO11|VIDEO2.mp4|1257" \\
        --camera "oneplus9pro-01|Portrait|VIDEO3.mp4|386"

Each --camera is "label|camera_mode_substring|video_path|frame_idx" ("|"-
separated, not ":" -- a Windows drive letter makes ":" unusable) —
camera_mode_substring may be empty (leave it blank, e.g. "label||video|idx")
if that camera model only has one recording mode.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.setup.extrinsics_solver import (  # noqa: E402
    CamCalibState,
    MarkerGroup,
    ObsPoint,
    run_calibration,
)
from app.setup.fiducial_markers import (  # noqa: E402
    ArucoDetector,
    anchor_from_marker_rig,
    load_rig_config,
    merge_detections_into_groups,
)

_log = logging.getLogger("test_rig_anchor_capture1")


# ---------------------------------------------------------------------------
# Registry lookup — same mode-disambiguation logic as
# characterize_rig_from_video.py's _load_intrinsics (see that script's
# docstring for why a camera model with several recording modes cannot be
# resolved from camera_instances.label alone). Duplicated here rather than
# imported, per this directory's standalone-script convention.
# ---------------------------------------------------------------------------


def _load_intrinsics(conn: sqlite3.Connection, camera_label: str, camera_mode: str | None) -> dict:
    conn.row_factory = sqlite3.Row
    modes = conn.execute(
        """
        SELECT cm.id, cm.width_px, cm.height_px, cm.nominal_fps, cm.notes,
               cm.default_intrinsics_calibration_id
        FROM camera_instances ci
        JOIN camera_modes cm ON cm.camera_model_id = ci.camera_model_id
        WHERE ci.label = ?
        """,
        (camera_label,),
    ).fetchall()
    if not modes:
        raise SystemExit(f"error: no camera_modes found for camera_instances.label={camera_label!r}")
    if camera_mode:
        needle = camera_mode.lower()
        modes = [m for m in modes if needle in (m["notes"] or "").lower()]
    if len(modes) != 1:
        raise SystemExit(
            f"error: {camera_label!r} has {len(modes)} matching camera_modes for "
            f"camera_mode={camera_mode!r} (need exactly 1)."
        )
    mode = modes[0]
    calib_id = mode["default_intrinsics_calibration_id"]
    row = None
    if calib_id is not None:
        row = conn.execute("SELECT * FROM intrinsics_calibrations WHERE id = ?", (calib_id,)).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT * FROM intrinsics_calibrations WHERE camera_mode_id = ? "
            "ORDER BY calibrated_at DESC LIMIT 1",
            (mode["id"],),
        ).fetchone()
    if row is None:
        raise SystemExit(f"error: camera_modes.id={mode['id']} has no intrinsics_calibrations at all.")
    print(f"{camera_label}: mode {mode['notes']!r} ({mode['width_px']}x{mode['height_px']}"
          f"@{mode['nominal_fps']}fps), calibration id={row['id']}")
    fx, fy, cx, cy = row["fx"], row["fy"], row["cx"], row["cy"]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    K_orig = K.copy()
    if row["matrix_original"]:
        vals = struct.unpack("<9d", bytes(row["matrix_original"]))
        K_orig = np.array(vals).reshape(3, 3)
    if row["dist_coeffs"]:
        n = len(bytes(row["dist_coeffs"])) // 8
        dist = np.array(struct.unpack(f"<{n}d", bytes(row["dist_coeffs"]))).reshape(1, -1)
    else:
        dist = np.zeros((1, 4))
    return {"K": K, "K_orig": K_orig, "dist": dist, "fisheye": row["distortion_model"] == "fisheye"}


# ---------------------------------------------------------------------------
# Single-frame, rotation-aware read (duplicated from
# characterize_rig_from_video.py's frame-sampling helpers, trimmed to "just
# grab frame N" — see that script's header for why each tool here is
# self-contained rather than sharing a frame-source module).
# ---------------------------------------------------------------------------


def _parse_displaymatrix(data: bytes) -> int:
    if len(data) < 36:
        return 0
    m = struct.unpack("<9i", data[:36])
    scale_x = math.hypot(m[0], m[3])
    scale_y = math.hypot(m[1], m[4])
    if scale_x == 0 or scale_y == 0:
        return 0
    return round(-math.atan2(m[1] / scale_y, m[0] / scale_x) * 180 / math.pi) % 360


def _apply_rotation(img: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _read_frame_av(path: str, frame_idx: int) -> np.ndarray:
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base)
        container_fps = float(stream.average_rate)

        rotate_str = (stream.metadata or {}).get("rotate", "0") or "0"
        try:
            rotation = int(rotate_str) % 360
        except (ValueError, TypeError):
            rotation = 0
        if rotation == 0:
            for probe_frame in container.decode(stream):
                for sd in probe_frame.side_data or []:
                    if "DISPLAYMATRIX" in str(sd.type).upper():
                        rotation = _parse_displaymatrix(bytes(sd))
                break
            container.seek(0, stream=stream, backward=True, any_frame=False)
        if rotation:
            print(f"  rotation: {rotation}° (from metadata)")

        if frame_idx > 0:
            seek_s = max(0.0, (frame_idx - 1) / container_fps)
            container.seek(int(seek_s / time_base), stream=stream, backward=True, any_frame=False)

        idx: int | None = None
        for av_frame in container.decode(stream):
            if av_frame.pts is None:
                continue
            if idx is None:
                idx = round(float(av_frame.pts) * time_base * container_fps)
                if idx < frame_idx:
                    continue
            if idx >= frame_idx:
                img = av_frame.to_ndarray(format="bgr24")
                if rotation:
                    img = _apply_rotation(img, rotation)
                return img
            idx += 1
    raise RuntimeError(f"could not read frame {frame_idx} from {path}")


def _read_frame_cv2(path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) % 360
    if rotation:
        print(f"  rotation: {rotation}°")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx} from {path}")
    if rotation:
        img = _apply_rotation(img, rotation)
    return img


def _read_frame(path: str, frame_idx: int) -> np.ndarray:
    try:
        import av  # noqa: F401
        return _read_frame_av(path, frame_idx)
    except ImportError:
        return _read_frame_cv2(path, frame_idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass
class _CameraSpec:
    label: str
    camera_mode: str | None
    video_path: str
    frame_idx: int


def _parse_camera_spec(raw: str) -> _CameraSpec:
    # "|"-separated, not ":" -- a Windows drive letter ("D:/...") makes ":"
    # unusable as the field separator for video_path.
    parts = raw.split("|")
    if len(parts) != 4:
        raise SystemExit(
            f"error: --camera must be 'label|camera_mode|video_path|frame_idx', got {raw!r}"
        )
    label, mode, video_path, frame_idx = parts
    return _CameraSpec(label=label, camera_mode=mode or None, video_path=video_path,
                        frame_idx=int(frame_idx))


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--registry-db", required=True)
    parser.add_argument("--rig-config", required=True, help="Path to an 'explicit'-shape rig config JSON")
    parser.add_argument("--camera", action="append", required=True, dest="cameras",
                         help="label:camera_mode:video_path:frame_idx (repeatable, one per camera)")
    parser.add_argument("--rig-dict", default="DICT_4X4_50")
    parser.add_argument("--scattered-dict", default="DICT_5X5_50")
    parser.add_argument("--min-marker-perimeter-rate", type=float, default=0.01)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S",
    )

    specs = [_parse_camera_spec(c) for c in args.cameras]
    rig_config = load_rig_config(args.rig_config)
    print(f"Rig config: {rig_config.rig_id!r}, {len(rig_config.marker_corners)} marker(s): "
          f"{sorted(rig_config.marker_corners)}")

    conn = sqlite3.connect(f"file:{args.registry_db}?mode=ro", uri=True)
    try:
        intrinsics = {s.label: _load_intrinsics(conn, s.label, s.camera_mode) for s in specs}
    finally:
        conn.close()

    rig_detector = ArucoDetector(
        dictionary=args.rig_dict, min_marker_perimeter_rate=args.min_marker_perimeter_rate,
    )
    scattered_detector = ArucoDetector(
        dictionary=args.scattered_dict, min_marker_perimeter_rate=args.min_marker_perimeter_rate,
    )

    states: list[CamCalibState] = []
    rig_detections_by_camera: dict[str, list] = {}
    scattered_groups: dict[str, MarkerGroup] = {}

    for spec in specs:
        print(f"\n--- {spec.label} (frame {spec.frame_idx}) ---")
        img = _read_frame(spec.video_path, spec.frame_idx)
        intr = intrinsics[spec.label]
        states.append(CamCalibState(
            video_id=spec.label, label=spec.label,
            K=intr["K"], K_orig=intr["K_orig"], dist=intr["dist"], fisheye=intr["fisheye"],
            image=img,
        ))

        all_rig_dets = rig_detector.detect(img, video_id=spec.label, frame_idx=spec.frame_idx)
        rig_dets = [d for d in all_rig_dets if d.marker_id in rig_config.marker_corners]
        print(f"  rig markers: {sorted(d.marker_id for d in rig_dets)} "
              f"(of {sorted(d.marker_id for d in all_rig_dets)} total {args.rig_dict} detections)")
        rig_detections_by_camera[spec.label] = rig_dets

        scattered_dets = scattered_detector.detect(img, video_id=spec.label, frame_idx=spec.frame_idx)
        print(f"  scattered tags: {sorted(d.marker_id for d in scattered_dets)}")
        merge_detections_into_groups(scattered_dets, scattered_groups, size=None)

    cps = anchor_from_marker_rig(rig_detections_by_camera, rig_config)
    print(f"\nAnchored {len(cps)} rig control point(s) "
          f"({len({(c.name.split('_')[1]) for c in cps})} distinct marker(s)).")
    if not cps:
        print("error: rig was not detected in any camera -- cannot anchor. Check --rig-dict and "
              "min-marker-perimeter-rate, or that the rig is actually visible at these frames.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Running run_calibration() with {len(scattered_groups)} scattered marker group(s) as "
          f"free control points plus SIFT...")
    result = run_calibration(
        states, control_points=cps, marker_groups=list(scattered_groups.values()),
        cp_only=False,
    )

    print("\n=== Solved camera positions ===")
    for s in states:
        if s.R is None:
            print(f"  {s.label:20s}  UNSOLVED")
            continue
        C = -s.R.T @ s.t.flatten()
        err = result.cp_reprojection_errors.get(s.video_id)
        err_str = f"  CP err {err['mean']:.1f}±{err['std']:.1f}px (max {err['max']:.1f}px)" if err else "  (no CP err)"
        print(f"  {s.label:20s}  ({C[0]:+.3f}, {C[1]:+.3f}, {C[2]:+.3f}){err_str}")

    if result.unsolved:
        print(f"\nWARNING: {len(result.unsolved)} camera(s) unsolved: {result.unsolved}")

    # --- Scattered-tag world positions (Phase 5/Tier B precursor: no
    # persistence, just a diagnostic dump of what solve_marker_groups would
    # feed into scene_fiducial_markers). ---
    print(f"\n=== Scattered tags (free-CP centroid, camera network scale) ===")
    states_by_id = {s.video_id: s for s in states}
    for marker_id, mg in sorted(scattered_groups.items()):
        cameras_seeing = mg.cameras_observing()
        if len(cameras_seeing) < 2:
            print(f"  tag {marker_id}: only {len(cameras_seeing)} camera(s) with all 4 corners -- skipped")
            continue
        cps_for_tag = mg.as_control_points()
        centroids = []
        for cp in cps_for_tag:
            undist = {}
            for vid, obs in cp.obs.items():
                s = states_by_id.get(vid)
                if s is None or s.R is None:
                    continue
                pts_u = cv2.undistortPoints(
                    np.array([[[obs.px, obs.py]]], dtype=np.float32), s.K_orig, s.dist, None, s.K,
                ).reshape(2)
                undist[vid] = pts_u
            if len(undist) < 2:
                continue
            rows = []
            for vid, (px, py) in undist.items():
                s = states_by_id[vid]
                P = s.K @ np.hstack([s.R, s.t.reshape(3, 1)])
                rows.append(px * P[2] - P[0])
                rows.append(py * P[2] - P[1])
            A = np.array(rows)
            _, _, Vt = np.linalg.svd(A)
            h = Vt[-1]
            if abs(h[3]) > 1e-10:
                centroids.append(h[:3] / h[3])
        if centroids:
            centroid = np.mean(centroids, axis=0)
            print(f"  tag {marker_id}: world ({centroid[0]:+.3f}, {centroid[1]:+.3f}, "
                  f"{centroid[2]:+.3f})  ({len(cameras_seeing)} camera(s), {len(centroids)}/4 corners)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""test_reanchor_capture2.py — Test re-anchoring a capture from previously
solved scattered ArUco tags alone, with no physical calibration rig present.

See docs/roadmap/features/extrinsics-improvements/status.md's 2026-08-11/12
entries and extrinsics-improvements-design.md section 9 Tier B ("Scattered
ArUco tags — redundancy / mid-session drift recovery"). This is the direct
test of §9's other stated motivation (see the sixth live-testing round in
status.md): recovering a shared world frame after the primary anchoring
instrument is gone/moved/unavailable, using only ordinary scattered tags
whose world positions were already solved in a *different* capture.

Reuses anchor_from_marker_rig completely unmodified — the scattered tags'
already-solved corner geometry (produced by
test_rig_anchor_capture1.py --save-scattered-tags) is just another rig
config as far as that function is concerned. No new anchoring code needed;
this script only differs from test_rig_anchor_capture1.py in *not* having
a second, physical-rig detection pass, since capture 2 has no box in it.

Usage
-----
    python test_reanchor_capture2.py \\
        --registry-db path/to/registry.db \\
        --rig-config tools/rig_configs/scattered_tags_capture1.json \\
        --camera "insta_ace2_pro|4K 120 fps linear|VIDEO1.mp4|1358" \\
        --camera "gopro-11_mini_01|HERO11|VIDEO2.mp4|588" \\
        --camera "oneplus9pro-01|Portrait|VIDEO3.mp4|219"

Each --camera is "label|camera_mode_substring|video_path|frame_idx", same
convention as test_rig_anchor_capture1.py (see that script for why "|" and
not ":").
"""

from __future__ import annotations

import argparse
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

from app.setup.extrinsics_solver import CamCalibState, run_calibration  # noqa: E402
from app.setup.fiducial_markers import ArucoDetector, anchor_from_marker_rig, load_rig_config  # noqa: E402


_log = logging.getLogger("test_reanchor_capture2")


# ---------------------------------------------------------------------------
# Registry lookup + rotation-aware single-frame read -- identical to
# test_rig_anchor_capture1.py's; duplicated rather than imported per this
# directory's standalone-script convention (see that script's header).
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
    parser.add_argument("--rig-config", required=True,
                         help="Previously-solved scattered-tag geometry (from "
                              "test_rig_anchor_capture1.py --save-scattered-tags)")
    parser.add_argument("--camera", action="append", required=True, dest="cameras")
    parser.add_argument("--tag-dict", default="DICT_5X5_50")
    parser.add_argument("--min-marker-perimeter-rate", type=float, default=0.01)
    parser.add_argument("--refine-intrinsics", action="append", default=[], dest="refine_intrinsics",
                         metavar="LABEL",
                         help="Let run_calibration refine fx/fy for this camera label (repeatable) "
                              "-- diagnostic for suspected intrinsics drift (e.g. autofocus)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S",
    )

    specs = [_parse_camera_spec(c) for c in args.cameras]
    rig_config = load_rig_config(args.rig_config)
    print(f"Re-anchor config: {rig_config.rig_id!r}, {len(rig_config.marker_corners)} known tag(s): "
          f"{sorted(rig_config.marker_corners)}")

    conn = sqlite3.connect(f"file:{args.registry_db}?mode=ro", uri=True)
    try:
        intrinsics = {s.label: _load_intrinsics(conn, s.label, s.camera_mode) for s in specs}
    finally:
        conn.close()

    detector = ArucoDetector(
        dictionary=args.tag_dict, min_marker_perimeter_rate=args.min_marker_perimeter_rate,
    )

    states: list[CamCalibState] = []
    detections_by_camera: dict[str, list] = {}

    for spec in specs:
        print(f"\n--- {spec.label} (frame {spec.frame_idx}) ---")
        img = _read_frame(spec.video_path, spec.frame_idx)
        intr = intrinsics[spec.label]
        states.append(CamCalibState(
            video_id=spec.label, label=spec.label,
            K=intr["K"], K_orig=intr["K_orig"], dist=intr["dist"], fisheye=intr["fisheye"],
            image=img,
        ))
        all_dets = detector.detect(img, video_id=spec.label, frame_idx=spec.frame_idx)
        known_dets = [d for d in all_dets if d.marker_id in rig_config.marker_corners]
        print(f"  known tags seen: {sorted(d.marker_id for d in known_dets)} "
              f"(of {sorted(d.marker_id for d in all_dets)} total {args.tag_dict} detections)")
        detections_by_camera[spec.label] = known_dets

    cps = anchor_from_marker_rig(detections_by_camera, rig_config)
    print(f"\nAnchored {len(cps)} control point(s) from previously-known tags "
          f"({len({c.name.split('_')[1] for c in cps})} distinct tag(s)) -- "
          f"NO physical rig, NO SIFT-free-anchor: this capture is re-anchored purely from tags "
          f"solved in a different capture.")
    if not cps:
        print("error: no known tag was detected in any camera -- cannot re-anchor.", file=sys.stderr)
        sys.exit(1)

    refine = set(args.refine_intrinsics) or None
    if refine:
        print(f"Refining intrinsics (fx/fy) for: {sorted(refine)}")
    result = run_calibration(states, control_points=cps, cp_only=False, refine_intrinsics=refine)

    if refine:
        print("\n=== Refined intrinsics ===")
        for s in states:
            if s.video_id in refine:
                orig = intrinsics[s.label]["K"]
                print(f"  {s.label:20s}  fx {orig[0,0]:.1f}->{s.K[0,0]:.1f}  "
                      f"fy {orig[1,1]:.1f}->{s.K[1,1]:.1f}  "
                      f"({100*(s.K[0,0]/orig[0,0]-1):+.1f}% / {100*(s.K[1,1]/orig[1,1]-1):+.1f}%)")

    print("\n=== Solved camera positions (capture 2, re-anchored) ===")
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


if __name__ == "__main__":
    main()

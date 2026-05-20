#!/usr/bin/env python3
"""detect_aruco.py — Detect ArUco markers in a video and write an annotated output.

Reads frames from the input video (correcting for container rotation metadata),
runs ArUco detection on each frame, draws marker outlines and IDs, and writes
the annotated frames to an output video.

Usage
-----
    python detect_aruco.py INPUT OUTPUT [options]

Examples
--------
    # Full video, default dictionary (6×6 250)
    python detect_aruco.py clip.mp4 clip_aruco.mp4

    # Frames 300–900 only, 5×5 dictionary
    python detect_aruco.py clip.mp4 clip_aruco.mp4 --start 300 --end 900 --dict DICT_5X5_100

    # Force rotation correction if auto-detection fails
    python detect_aruco.py clip.mp4 clip_aruco.mp4 --rotate 180
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# ArUco dictionary names → OpenCV constants
# ---------------------------------------------------------------------------

_DICT_NAMES: dict[str, int] = {
    "DICT_4X4_50":        cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":       cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250":       cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000":      cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50":        cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100":       cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250":       cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000":      cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50":        cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100":       cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250":       cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000":      cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50":        cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100":       cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250":       cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000":      cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


# ---------------------------------------------------------------------------
# Rotation helpers (same logic as detection_pipeline.py)
# ---------------------------------------------------------------------------

def _stream_rotation_av(stream) -> int:
    """Return clockwise rotation degrees from stream metadata (older cameras)."""
    rotate_str = (stream.metadata or {}).get("rotate", "0") or "0"
    try:
        return int(rotate_str) % 360
    except (ValueError, TypeError):
        return 0


def _parse_displaymatrix(data: bytes) -> int:
    """Parse clockwise rotation from a DISPLAYMATRIX side-data blob.

    Modern Android phones (Pixel 7+) store rotation as a 3×3 fixed-point
    matrix in frame side data rather than a plain 'rotate' metadata tag.
    The blob is 36 bytes: nine 32-bit little-endian integers (16.16 fixed).
    """
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


# ---------------------------------------------------------------------------
# Frame iteration
# ---------------------------------------------------------------------------

def _iter_frames_av(path: str, first: int, last: int, rotation_override: int | None):
    """Yield (frame_idx, bgr_array) using PyAV with rotation correction."""
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base)
        container_fps = float(stream.average_rate)

        if rotation_override is not None:
            rotation = rotation_override
            rotation_source = "override"
        else:
            rotation = _stream_rotation_av(stream)
            rotation_source = "metadata"
            if rotation == 0:
                for probe_frame in container.decode(stream):
                    for sd in (probe_frame.side_data or []):
                        if "DISPLAYMATRIX" in str(sd.type).upper():
                            rotation = _parse_displaymatrix(bytes(sd))
                            rotation_source = "DISPLAYMATRIX"
                    break
                container.seek(0, stream=stream, backward=True, any_frame=False)
        if rotation:
            print(f"  rotation: {rotation}° (from {rotation_source})")

        if first > 0:
            seek_s = max(0.0, (first - 1) / container_fps)
            container.seek(int(seek_s / time_base), stream=stream, backward=True, any_frame=False)

        frame_idx: int | None = None
        for av_frame in container.decode(stream):
            if av_frame.pts is None:
                continue
            if frame_idx is None:
                pts_idx = round(float(av_frame.pts) * time_base * container_fps)
                if pts_idx < first:
                    continue
                frame_idx = first
            if frame_idx >= last:
                break
            img = av_frame.to_ndarray(format="bgr24")
            if rotation:
                img = _apply_rotation(img, rotation)
            yield frame_idx, img
            frame_idx += 1


def _iter_frames_cv2(path: str, first: int, last: int, rotation_override: int | None):
    """Yield (frame_idx, bgr_array) using cv2.VideoCapture with rotation correction."""
    cap = cv2.VideoCapture(path)
    if rotation_override is not None:
        rotation = rotation_override
    else:
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) % 360
    if rotation:
        print(f"  rotation: {rotation}°")
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frame_idx = first
    while frame_idx < last:
        ok, img = cap.read()
        if not ok:
            break
        if rotation:
            img = _apply_rotation(img, rotation)
        yield frame_idx, img
        frame_idx += 1
    cap.release()


def _video_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _video_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps or 30.0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

_CORNER_COLOR  = (0, 255, 0)    # green outline
_ID_COLOR      = (0, 255, 255)  # yellow text
_DOT_COLOR     = (0, 0, 255)    # red dot for corner-0 (orientation reference)
_FONT          = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE    = 0.7
_THICKNESS     = 2


def _draw_marker(img: np.ndarray, corners: np.ndarray, marker_id: int) -> None:
    """Draw a single detected marker: outline polygon, ID label, and corner-0 dot."""
    pts = corners.reshape(4, 2).astype(int)

    # Outline
    cv2.polylines(img, [pts], isClosed=True, color=_CORNER_COLOR, thickness=_THICKNESS)

    # Red dot on corner 0 (top-left by ArUco convention) so orientation is visible
    cv2.circle(img, tuple(pts[0]), 6, _DOT_COLOR, -1)

    # ID label at centroid
    cx = int(pts[:, 0].mean())
    cy = int(pts[:, 1].mean())
    label = str(marker_id)
    (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _THICKNESS)
    # Dark background rectangle for legibility
    cv2.rectangle(
        img,
        (cx - tw // 2 - 3, cy - th - baseline - 3),
        (cx + tw // 2 + 3, cy + baseline + 3),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        img, label,
        (cx - tw // 2, cy),
        _FONT, _FONT_SCALE, _ID_COLOR, _THICKNESS, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect ArUco markers in a video and write an annotated output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Available dictionaries:",
            *[f"  {k}" for k in _DICT_NAMES],
        ]),
    )
    parser.add_argument("input",  help="Input video file")
    parser.add_argument("output", help="Output annotated video file")
    parser.add_argument("--start", type=int, default=0,    metavar="FRAME",
                        help="First frame to process (default: 0)")
    parser.add_argument("--end",   type=int, default=None, metavar="FRAME",
                        help="Last frame (exclusive, default: end of video)")
    parser.add_argument("--dict",  default="DICT_6X6_250", choices=list(_DICT_NAMES),
                        metavar="DICT_NAME",
                        help="ArUco dictionary (default: DICT_6X6_250)")
    parser.add_argument("--rotate", type=int, default=None, choices=[0, 90, 180, 270],
                        metavar="DEG",
                        help="Override rotation correction (0/90/180/270); "
                             "auto-detected from metadata if omitted")
    args = parser.parse_args()

    input_path = args.input
    if not Path(input_path).exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = _video_frame_count(input_path)
    fps          = _video_fps(input_path)
    first_frame  = args.start
    last_frame   = args.end if args.end is not None else total_frames
    last_frame   = min(last_frame, total_frames)

    if first_frame >= last_frame:
        print(f"error: empty frame range [{first_frame}, {last_frame})", file=sys.stderr)
        sys.exit(1)

    print(f"Input : {input_path}")
    print(f"Output: {args.output}")
    print(f"Frames: {first_frame}–{last_frame - 1}  ({last_frame - first_frame} frames, {fps:.2f} fps)")
    print(f"Dict  : {args.dict}")

    # --- ArUco detector ---
    aruco_dict   = cv2.aruco.getPredefinedDictionary(_DICT_NAMES[args.dict])
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # --- Frame source ---
    try:
        import av as _av  # noqa: F401
        frame_iter = _iter_frames_av(input_path, first_frame, last_frame, args.rotate)
    except ImportError:
        frame_iter = _iter_frames_cv2(input_path, first_frame, last_frame, args.rotate)

    # --- VideoWriter (opened on first frame so we know the rotated dimensions) ---
    writer: cv2.VideoWriter | None = None

    # --- Stats ---
    frames_with_detections = 0
    detections_per_id: dict[int, int] = defaultdict(int)
    n_processed = 0
    report_every = max(1, (last_frame - first_frame) // 20)

    for frame_idx, img in frame_iter:
        h, w = img.shape[:2]

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
            if not writer.isOpened():
                print(f"error: could not open output video: {args.output}", file=sys.stderr)
                sys.exit(1)
            print(f"Frame size: {w}×{h}")

        corners_list, ids, _ = detector.detectMarkers(img)

        if ids is not None and len(ids) > 0:
            frames_with_detections += 1
            for corners, mid in zip(corners_list, ids.flatten()):
                _draw_marker(img, corners, int(mid))
                detections_per_id[int(mid)] += 1

        writer.write(img)
        n_processed += 1

        if n_processed % report_every == 0:
            pct = 100.0 * n_processed / (last_frame - first_frame)
            print(f"  {n_processed}/{last_frame - first_frame} frames ({pct:.0f}%)", end="\r")

    if writer is not None:
        writer.release()

    print(f"\nDone. {n_processed} frames processed.")
    if detections_per_id:
        print(f"Frames with detections: {frames_with_detections}/{n_processed} "
              f"({100.0 * frames_with_detections / max(n_processed, 1):.1f}%)")
        print("Detected marker IDs:")
        for mid, count in sorted(detections_per_id.items()):
            print(f"  ID {mid:4d} — {count} frames ({100.0 * count / n_processed:.1f}%)")
    else:
        print("No markers detected. Check --dict matches the markers in the video.")


if __name__ == "__main__":
    main()

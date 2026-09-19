# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""make_blender_proxy_videos.py — PROTOTYPE: undistort + downscale a
capture's real videos into lightweight proxies for use as camera
background footage in Blender.

Reuses this project's own stored per-camera undistortion maps
(intrinsics_calibrations.undistort_mapx/mapy, zlib-compressed float32 --
same format/decoding as pipeline/calibration/calibrate_extrinsics.py) so
the proxy's pixel space matches exactly what the rest of the pipeline
already treats as "undistorted" -- not a fresh, possibly-inconsistent
undistortion recomputed from scratch.

Trims to the tracking run's own time range (synced-timeline seconds ->
each camera's own frame range via the session's SyncTable) rather than
processing whole raw recordings.

Output frame rate (Harri, 2026-09-07): the consuming Blender scene's
camera-background/compositor Movie Clip playback maps scene frame to clip
frame by a *plain index offset* -- it never reads a clip's own declared
fps at all. So the proxy's *content* must already advance at exactly
--tracker-fps in real time; relabeling a clip encoded at a camera's real
capture rate (never exactly 120, and never trust any single fps number
for this -- see below) does nothing. Each output frame N is therefore
resolved to a source frame via ``SyncTable.lookup(time_start_s +
N/tracker_fps, shot_video_id)`` -- the *only* ground truth for global-time
<-> video-frame is a capture's own sync anchor pairs (interpolated
locally between the two real anchors bracketing the query, not any
measured/nominal fps value -- see SyncTable.lookup's own local_fps
derivation). A source frame is naturally held (repeated) when the real
camera runs a hair slower than --tracker-fps, or dropped when it runs a
hair faster -- never assumed from an fps ratio.

Usage
-----
    python make_blender_proxy_videos.py \\
        --session SESSION.db --capture b21fa02d \\
        --time-start 33.62 --time-end 130.19 --tracker-fps 120 \\
        --target-width 1280 --output-dir scratch/blender_proxy_videos
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import sqlite3

from posetrak.detection.frame_source import iter_frames
from app.setup.db_context import SyncPoint, SyncTable


def _load_undistort_maps(
    conn: sqlite3.Connection, calib_id: str, fallback_width: int, fallback_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    row = conn.execute("SELECT * FROM intrinsics_calibrations WHERE id = ?", (calib_id,)).fetchone()
    if row is None:
        raise SystemExit(f"No intrinsics_calibrations row for id={calib_id}")
    if row["undistort_mapx"]:
        w, h = row["image_width"], row["image_height"]
        mapx = np.frombuffer(zlib.decompress(bytes(row["undistort_mapx"])), dtype=np.float32).reshape(h, w)
        mapy = np.frombuffer(zlib.decompress(bytes(row["undistort_mapy"])), dtype=np.float32).reshape(h, w)
        return mapx, mapy

    # No maps stored for this calibration (seen for pixel9) -- compute fresh
    # from fx/fy/cx/cy/dist_coeffs, same as calibrate_intrinsics.py's own
    # K_new/initUndistortRectifyMap step.
    print(f"  (no stored undistort maps for {calib_id}, computing fresh from K/dist)")
    w = row["image_width"] or fallback_width
    h = row["image_height"] or fallback_height
    K = np.array([[row["fx"], 0.0, row["cx"]], [0.0, row["fy"], row["cy"]], [0.0, 0.0, 1.0]])
    if row["dist_coeffs"]:
        n = len(bytes(row["dist_coeffs"])) // 8
        import struct
        dist = np.array(struct.unpack(f"<{n}d", bytes(row["dist_coeffs"]))).reshape(1, -1)
    else:
        dist = np.zeros((1, 4))
    if row["distortion_model"] == "fisheye":
        K_new = K.copy()
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, dist, np.eye(3), K_new, (w, h), cv2.CV_32FC1)
    else:
        K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K_new, (w, h), cv2.CV_32FC1)
    return mapx, mapy


def compute_frame_mapping(
    sync_table: SyncTable, shot_video_id: str, time_start_s: float, time_end_s: float, tracker_fps: float,
) -> list[int]:
    """One source-frame index per output tick at exactly tracker_fps,
    resolved purely via SyncTable.lookup() -- see module docstring for why
    this must never be derived from an fps ratio instead.
    """
    n_total = round((time_end_s - time_start_s) * tracker_fps)
    mapping = []
    for i in range(n_total):
        t = time_start_s + i / tracker_fps
        f = sync_table.lookup(t, shot_video_id)
        if f is None:
            raise SystemExit(f"No sync data for {shot_video_id} at t={t}")
        mapping.append(round(f))
    # Sequential single-pass decoding (process_camera below) requires this;
    # a real capture's sync anchors should never produce anything else for
    # a forward time range, but fail loudly rather than silently emit
    # frames out of order if they ever did.
    if any(b < a for a, b in zip(mapping, mapping[1:])):
        raise SystemExit(f"{shot_video_id}: frame mapping is not monotonic -- check sync data")
    return mapping


def process_camera(
    conn: sqlite3.Connection,
    label: str,
    file_path: str,
    intrinsics_calib_id: str,
    frame_mapping: list[int],
    tracker_fps: float,
    target_width: int,
    out_path: Path,
    fallback_width: int = 1920,
    fallback_height: int = 1080,
    log_every: int = 200,
) -> None:
    mapx, mapy = _load_undistort_maps(conn, intrinsics_calib_id, fallback_width, fallback_height)
    src_h, src_w = mapx.shape
    target_height = int(round(target_width * src_h / src_w))
    # Even dimensions -- required by yuv420p, which most players (and
    # Blender's own VSE) expect.
    target_width -= target_width % 2
    target_height -= target_height % 2

    # cv2.VideoWriter's mp4v encoder produces enormous files (mpeg4 at an
    # effectively uncontrolled bitrate -- ~14MB/s at 1280x720 measured here).
    # Piping raw frames into ffmpeg's libx264 instead gives real control
    # over size (CRF) -- this is the same real-world constraint the
    # tracker's own video pipeline already works around elsewhere in this
    # project. Output fps is declared as the exact --tracker-fps the
    # content has now genuinely been resampled to (see compute_frame_mapping)
    # -- Blender itself ignores this value for its own scene-frame mapping,
    # but it's what makes the file's own internal timing honest for anyone
    # (or anything else) that plays it back directly.
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{target_width}x{target_height}", "-r", f"{tracker_fps:.6f}",
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # Group by source frame -> how many times it's needed (usually 1;
    # >1 when the real camera ran a hair slower than tracker_fps and a
    # frame must be held; a source frame absent from this map entirely is
    # naturally skipped, for a camera running a hair faster).
    needed: dict[int, int] = {}
    for raw_idx in frame_mapping:
        needed[raw_idx] = needed.get(raw_idx, 0) + 1
    raw_start, raw_end = min(needed), max(needed) + 1
    n_dupes = sum(c - 1 for c in needed.values() if c > 1)
    if n_dupes:
        print(f"  {label}: {n_dupes} source frame(s) held an extra tick "
              f"(camera running a hair slower than {tracker_fps:.3f}fps)")
    n_skipped = (raw_end - raw_start) - len(needed)
    if n_skipped:
        print(f"  {label}: {n_skipped} source frame(s) skipped "
              f"(camera running a hair faster than {tracker_fps:.3f}fps)")

    n = 0
    total = len(frame_mapping)
    try:
        for frame_idx, img in iter_frames(file_path, raw_start, raw_end):
            count = needed.get(frame_idx)
            if not count:
                continue
            undist = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR)
            small = cv2.resize(undist, (target_width, target_height), interpolation=cv2.INTER_AREA)
            payload = small.tobytes()
            for _ in range(count):
                proc.stdin.write(payload)
                n += 1
            if n % log_every < count or n == total:
                print(f"  {label}: {n}/{total}", flush=True)
    finally:
        proc.stdin.close()
        proc.wait()
    print(f"{label}: wrote {n} frames -> {out_path}")


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--capture", required=True, help="captures.id (or prefix)")
    ap.add_argument("--time-start", type=float, required=True)
    ap.add_argument("--time-end", type=float, required=True)
    ap.add_argument("--tracker-fps", type=float, default=120.0,
                     help="The rate the consuming Blender scene's animation plays at "
                          "(tracker_configs.tracker_fps for the tracking run being synced "
                          "to) -- NOT a per-camera capture rate. Every output frame is "
                          "resolved via the sync table, never assumed from this or any "
                          "other fps number (default: %(default)s).")
    ap.add_argument("--target-width", type=int, default=1280)
    ap.add_argument("--output-dir", default="scratch/blender_proxy_videos")
    ap.add_argument("--camera-label", nargs="*", default=None,
                     help="Restrict to these cameras (default: all in the capture)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.session)
    conn.row_factory = sqlite3.Row
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    capture_row = conn.execute(
        "SELECT id FROM captures WHERE id LIKE ? || '%'", (args.capture,)
    ).fetchone()
    if capture_row is None:
        raise SystemExit(f"No captures row matching {args.capture!r}")
    capture_id = capture_row["id"]

    cam_rows = conn.execute(
        """
        SELECT ci.label, cv.file_path, cv.actual_fps, cm.nominal_fps,
               cv.intrinsics_calibration_id, cv.id AS shot_video_id,
               cm.width_px, cm.height_px
        FROM capture_videos cv
        JOIN camera_instances ci ON ci.id = cv.camera_instance_id
        JOIN camera_modes cm ON cm.id = cv.camera_mode_id
        WHERE cv.shot_id = ?
        """,
        (capture_id,),
    ).fetchall()
    if args.camera_label:
        cam_rows = [r for r in cam_rows if r["label"] in args.camera_label]

    sync_config = conn.execute(
        "SELECT id FROM sync_configs WHERE shot_id = ?", (capture_id,)
    ).fetchone()
    sp_rows = conn.execute(
        "SELECT shot_video_id, video_frame, timestamp_s FROM sync_points WHERE sync_config_id = ?",
        (sync_config["id"],),
    ).fetchall()
    sync_points = [
        SyncPoint(camera_instance_id="", shot_video_id=r["shot_video_id"],
                  video_frame=r["video_frame"], timestamp_s=r["timestamp_s"])
        for r in sp_rows
    ]
    # fps_by_video is only SyncTable's last-resort fallback for a video with
    # a single sync anchor (no second point to derive a real local rate
    # from) -- with >=2 anchors bracketing the query, as here, it is never
    # consulted; see SyncTable.lookup's own local_fps derivation. Do not
    # read anything into this value beyond that.
    fps_by_video = {r["shot_video_id"]: float(r["actual_fps"] or r["nominal_fps"] or 120.0) for r in cam_rows}
    sync_table = SyncTable(sync_points, fps_by_video)

    manifest = []
    for cam in cam_rows:
        frame_mapping = compute_frame_mapping(
            sync_table, cam["shot_video_id"], args.time_start, args.time_end, args.tracker_fps,
        )
        out_path = out_dir / f"{cam['label']}_proxy.mp4"
        print(f"{cam['label']}: {len(frame_mapping)} output frames, source range "
              f"[{frame_mapping[0]}, {frame_mapping[-1]}] -> {out_path}")
        process_camera(
            conn, cam["label"], cam["file_path"], cam["intrinsics_calibration_id"],
            frame_mapping, args.tracker_fps, args.target_width, out_path,
            fallback_width=cam["width_px"], fallback_height=cam["height_px"],
        )
        manifest.append({
            "label": cam["label"], "proxy_path": str(out_path),
            "source_frame_start": frame_mapping[0], "fps": args.tracker_fps,
        })

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest -> {manifest_path}")
    conn.close()


if __name__ == "__main__":
    main()

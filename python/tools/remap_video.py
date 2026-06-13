#!/usr/bin/env python3
"""
remap_video.py — Apply a rectification map (.npz) to a video or image.

Usage
-----
    # Remap a full video:
    uv run python python/tools/remap_video.py \\
        --map    insta_mega_rectmap.npz \\
        --input  mocap.mp4 \\
        --output mocap_rect.mp4

    # Remap a single frame (PNG/JPG → PNG):
    uv run python python/tools/remap_video.py \\
        --map    insta_mega_rectmap.npz \\
        --input  frame.jpg \\
        --output frame_rect.png

    # Preview a few frames (no output file):
    uv run python python/tools/remap_video.py \\
        --map    insta_mega_rectmap.npz \\
        --input  mocap.mp4 \\
        --preview 10

The map must have been produced by build_rectification_map.py for the same
camera mode (same resolution and lens settings).

Output video codec: mp4v (H.264 via ffmpeg is better quality but requires
a separate encode step; mp4v is lossless-equivalent for one-pass workflows).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def load_map(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int,int]]:
    data = np.load(npz_path)
    mapx = data["mapx"]
    mapy = data["mapy"]
    K    = data["K"]
    W, H = int(data["image_size"][0]), int(data["image_size"][1])
    cov  = float(data.get("coverage_pct", 0.0))
    print(f"Map loaded: {W}×{H}  coverage={cov:.1f}%")
    print(f"  K: fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    return mapx, mapy, K, (W, H)


def remap_frame(frame: np.ndarray, mapx: np.ndarray, mapy: np.ndarray) -> np.ndarray:
    return cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def remap_image(args, mapx, mapy, map_size):
    frame = cv2.imread(args.input)
    if frame is None:
        raise IOError(f"Cannot read image: {args.input}")
    h, w = frame.shape[:2]
    W, H = map_size
    if (w, h) != (W, H):
        raise ValueError(
            f"Image size {w}×{h} does not match map size {W}×{H}. "
            "The map must be built for the same camera mode and resolution."
        )
    out = remap_frame(frame, mapx, mapy)
    cv2.imwrite(args.output, out)
    print(f"Saved → {args.output}")


def remap_video(args, mapx, mapy, map_size):
    inp  = Path(args.input)
    cap  = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {inp}")

    W_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = map_size

    if (W_vid, H_vid) != (W, H):
        raise ValueError(
            f"Video size {W_vid}×{H_vid} does not match map size {W}×{H}. "
            "The map must be built for the same camera mode and resolution."
        )

    out_path = Path(args.output) if args.output else None

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
        if not writer.isOpened():
            raise IOError(f"Cannot open output video: {out_path}")
        print(f"Writing {W}×{H} @ {fps:.1f} fps → {out_path}")
    elif args.preview:
        print(f"Preview mode: showing every {args.preview_step} frames (press any key)")

    fi = 0
    t0 = time.perf_counter()
    shown = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preview: subsample frames
        if args.preview and fi % args.preview_step != 0:
            fi += 1
            continue

        out_frame = remap_frame(frame, mapx, mapy)

        if writer:
            writer.write(out_frame)
        elif args.preview:
            # Show side-by-side at half height
            h2 = H // 2
            left  = cv2.resize(frame,     (W // 2, h2))
            right = cv2.resize(out_frame, (W // 2, h2))
            cv2.imshow("Original | Rectified", np.hstack([left, right]))
            if cv2.waitKey(1) & 0xFF != 255:
                break
            shown += 1
            if args.preview > 0 and shown >= args.preview:
                break

        fi += 1
        if fi % 500 == 0 and writer:
            elapsed = time.perf_counter() - t0
            pct = 100.0 * fi / n_frames if n_frames > 0 else 0
            fps_proc = fi / elapsed
            eta = (n_frames - fi) / fps_proc if fps_proc > 0 else 0
            print(f"  {fi}/{n_frames} frames  {pct:.1f}%  {fps_proc:.1f} fps  ETA {eta:.0f}s")

    cap.release()
    if writer:
        writer.release()
        elapsed = time.perf_counter() - t0
        print(f"Done. {fi} frames in {elapsed:.1f}s  ({fi/elapsed:.1f} fps)")
    if args.preview:
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply a build_rectification_map.py warp map to a video or image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--map",    required=True,  help="Rectification map .npz")
    p.add_argument("--input",  required=True,  help="Input video or image path")
    p.add_argument("--output", default=None,   help="Output path (omit for preview only)")
    p.add_argument("--preview", type=int, default=0, metavar="N",
                   help="Show N preview frames instead of writing output (0 = off)")
    p.add_argument("--preview-step", type=int, default=30,
                   help="Show every Nth frame in preview mode (default 30)")
    return p.parse_args()


def main():
    args = parse_args()
    mapx, mapy, K, map_size = load_map(Path(args.map))

    inp = Path(args.input)
    if inp.suffix.lower() in _IMAGE_EXTS:
        if not args.output:
            raise ValueError("--output is required when remapping an image.")
        remap_image(args, mapx, mapy, map_size)
    else:
        if not args.output and not args.preview:
            raise ValueError("Provide --output or --preview N.")
        remap_video(args, mapx, mapy, map_size)


if __name__ == "__main__":
    main()

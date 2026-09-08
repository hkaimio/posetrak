# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""label_dot_ground_truth.py — interactive click-to-label tool for building a
hand-verified ground-truth set of real reflective-dot marker positions, to
validate/develop `dot_blob_detector.py` against in seconds instead of
running the full multi-hour detection pipeline on real data.

Built after a real, humbling lesson from doing this by eye in-conversation
(status.md's 2026-09-08 entry): a first automated-candidate + eyeball pass
missed a genuine marker entirely and, separately, double-counted one real
marker's compression-fragmented highlight as three. Hand-labeling needs
proper zoom and the ability to correct mistakes, not a one-shot screenshot
-- this tool exists so that labeling happens directly against the real
pixels, at whatever zoom is needed, with undo.

Frame list comes from a JSON manifest (see build a fresh one with any
{dataset, camera_label, shot_video_id, video_frame} list) rather than a
fixed hardcoded set, e.g.:
    [{"dataset": "nelli", "camera_label": "gopro-11_mini_01",
      "shot_video_id": "d6495e15-...", "video_frame": 7569,
      "time_s": 40.0, "tag": "person-markers"}, ...]

`dataset` is a short name, not an absolute path -- session DB paths are
machine-specific (see CLAUDE.md), so they're passed at runtime via
`--session name=path`, not embedded in the manifest or the output labels.

Usage:
    python tools/label_dot_ground_truth.py \\
        --manifest scratch/dot_ground_truth/frame_manifest.json \\
        --session nelli=D:/mocap/2026-09-06-kare-tests/2026-09-06-kare-tests.db \\
        --session sword=E:/mocap/vanhaa/ukemi-tommi-20260509.db \\
        --output scratch/dot_ground_truth/labels.json

Controls:
    left click       add a point at the cursor
    right click      remove the nearest point (within 20px of the click, at
                     the current zoom)
    mouse wheel      zoom in/out, centered on the cursor
    left-drag        pan (hold left mouse button and drag on empty space --
                     actually: use arrow keys to pan, see below; drag is not
                     used, to keep plain clicks unambiguous as "add point")
    arrow keys       pan the view
    r                reset zoom/pan to fit the whole frame
    u                undo (remove the most recently added point)
    n / space        next frame (saves this frame's points first)
    p                previous frame (saves this frame's points first)
    s                save progress to --output now, without changing frames
    q / Esc          save and quit

Points are stored in full original-image pixel coordinates regardless of
the current zoom/pan, so the output is meaningful independent of how it was
viewed. Progress is saved incrementally (every 'n'/'p'/'s'/quit) keyed by
(dataset, camera_label, video_frame), so re-running the tool on the same
--output resumes where you left off and lets you revisit/correct any frame
by navigating back to it.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posetrak.detection.frame_source import iter_frames  # noqa: E402

_WINDOW = "label_dot_ground_truth"
_DISPLAY_SIZE = 1400  # window is roughly this wide/tall; view scales to fit / zoom from here
_REMOVE_RADIUS_VIEW_PX = 20  # right-click removal radius, in on-screen (not image) pixels


def _key(entry: dict) -> str:
    return f"{entry['dataset']}::{entry['camera_label']}::{entry['video_frame']}"


def _resolve_file_path(session_conn: sqlite3.Connection, shot_video_id: str) -> str:
    row = session_conn.execute(
        "SELECT file_path FROM capture_videos WHERE id = ?", (shot_video_id,)
    ).fetchone()
    if row is None:
        raise SystemExit(f"no capture_videos row for shot_video_id={shot_video_id!r}")
    return row[0]


class _ViewState:
    """Current zoom/pan for one frame: view_x/view_y is the top-left corner
    (in full-image pixel coordinates) of what's currently shown; scale maps
    image pixels -> display pixels (>1 = zoomed in)."""

    def __init__(self, img_w: int, img_h: int) -> None:
        self.img_w, self.img_h = img_w, img_h
        self.reset()

    def reset(self) -> None:
        self.scale = min(_DISPLAY_SIZE / self.img_w, _DISPLAY_SIZE / self.img_h)
        self.view_x = 0.0
        self.view_y = 0.0

    def to_image(self, dx: float, dy: float) -> tuple[float, float]:
        return self.view_x + dx / self.scale, self.view_y + dy / self.scale

    def zoom(self, factor: float, at_dx: float, at_dy: float) -> None:
        ix, iy = self.to_image(at_dx, at_dy)
        self.scale = max(0.05, min(20.0, self.scale * factor))
        # Keep the point under the cursor fixed on screen.
        self.view_x = ix - at_dx / self.scale
        self.view_y = iy - at_dy / self.scale

    def pan(self, ddx: float, ddy: float) -> None:
        self.view_x += ddx / self.scale
        self.view_y += ddy / self.scale

    def render(self, img: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
        """Nearest-neighbor zoom via explicit index gather -- NOT cv2.resize.

        cv2.resize(INTER_NEAREST)'s own internal source-pixel-per-destination-
        pixel convention doesn't exactly match the linear formula to_image()
        uses to invert a click back to image coordinates (confirmed empirically,
        2026-09-08: a real ~1px systematic bias reported after labeling a real
        ground-truth set, reproduced with a synthetic test and traced to this
        mismatch, not to actual mislabeling). Rather than reverse-engineer
        cv2.resize's exact convention and patch to_image() to match it, this
        computes canvas pixel (px, py)'s source pixel as
        floor(view_x + px/scale) directly -- the exact forward evaluation of
        the same formula to_image() already inverts -- so the two are
        consistent by construction, not by two independent implementations
        happening to agree.
        """
        px = np.arange(_DISPLAY_SIZE)
        src_x = np.clip(np.floor(self.view_x + px / self.scale).astype(np.int64), 0, self.img_w - 1)
        src_y = np.clip(np.floor(self.view_y + px / self.scale).astype(np.int64), 0, self.img_h - 1)
        # Blank out canvas pixels whose source position actually falls outside the image
        # (clip() above would otherwise repeat-smear the edge pixel across the whole margin).
        valid_x = (self.view_x + px / self.scale >= 0) & (self.view_x + px / self.scale < self.img_w)
        valid_y = (self.view_y + px / self.scale >= 0) & (self.view_y + px / self.scale < self.img_h)
        canvas = img[np.ix_(src_y, src_x)].copy()
        canvas[~valid_y, :] = 0
        canvas[:, ~valid_x] = 0
        for x, y in points:
            dx, dy = (x - self.view_x) * self.scale, (y - self.view_y) * self.scale
            if 0 <= dx < _DISPLAY_SIZE and 0 <= dy < _DISPLAY_SIZE:
                p = (int(round(dx)), int(round(dy)))
                cv2.drawMarker(canvas, p, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1)
                cv2.circle(canvas, p, 9, (0, 255, 0), 1)
        return canvas


def _load_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_labels(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    return {_key(e): e for e in entries}


def _save_labels(path: Path, labels: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(labels.values(), key=lambda e: (e["dataset"], e["camera_label"], e["video_frame"]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--session", action="append", required=True, metavar="NAME=PATH",
                     help="Map a manifest 'dataset' name to a session DB path. Repeatable.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    session_paths = {}
    for entry in args.session:
        name, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"--session must be NAME=PATH, got {entry!r}")
        session_paths[name] = path
    session_conns = {name: sqlite3.connect(f"file:{p}?mode=ro", uri=True) for name, p in session_paths.items()}

    manifest = _load_manifest(Path(args.manifest))
    output_path = Path(args.output)
    labels = _load_labels(output_path)

    idx = 0
    view: _ViewState | None = None
    points: list[tuple[float, float]] = []
    img: np.ndarray | None = None
    cur_frame: dict | None = None

    def load_frame(i: int) -> None:
        nonlocal view, points, img, cur_frame
        entry = manifest[i]
        cur_frame = entry
        conn = session_conns.get(entry["dataset"])
        if conn is None:
            raise SystemExit(f"no --session given for dataset {entry['dataset']!r}")
        file_path = _resolve_file_path(conn, entry["shot_video_id"])
        got = None
        for _, decoded in iter_frames(file_path, entry["video_frame"], entry["video_frame"] + 1):
            got = decoded
            break
        if got is None:
            raise SystemExit(f"could not decode frame {entry['video_frame']} from {file_path}")
        img = got
        view = _ViewState(img.shape[1], img.shape[0])
        key = _key(entry)
        points = [tuple(p) for p in labels.get(key, {}).get("points", [])]

    def save_current() -> None:
        if cur_frame is None:
            return
        entry = dict(cur_frame)
        entry["points"] = [list(p) for p in points]
        labels[_key(entry)] = entry
        _save_labels(output_path, labels)

    def on_mouse(event, x, y, flags, _param) -> None:
        nonlocal points
        assert view is not None
        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = view.to_image(x, y)
            points.append((ix, iy))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                ix, iy = view.to_image(x, y)
                dists = [((px - ix) ** 2 + (py - iy) ** 2) ** 0.5 * view.scale for px, py in points]
                j = int(np.argmin(dists))
                if dists[j] <= _REMOVE_RADIUS_VIEW_PX:
                    points.pop(j)
        elif event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.25 if flags > 0 else 0.8
            view.zoom(factor, x, y)

    cv2.namedWindow(_WINDOW)
    cv2.setMouseCallback(_WINDOW, on_mouse)
    load_frame(idx)

    pan_step = 60
    while True:
        assert view is not None and img is not None and cur_frame is not None
        canvas = view.render(img, points)
        done = sum(1 for e in manifest if _key(e) in labels)
        status = (f"[{idx + 1}/{len(manifest)}] {cur_frame['dataset']}/{cur_frame['camera_label']} "
                  f"frame={cur_frame['video_frame']} tag={cur_frame.get('tag', '')}  "
                  f"points={len(points)}  labeled_frames={done}/{len(manifest)}")
        cv2.putText(canvas, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(_WINDOW, canvas)
        k = cv2.waitKey(20) & 0xFF

        if k in (ord("n"), ord(" ")):
            save_current()
            if idx < len(manifest) - 1:
                idx += 1
                load_frame(idx)
        elif k == ord("p"):
            save_current()
            if idx > 0:
                idx -= 1
                load_frame(idx)
        elif k == ord("u"):
            if points:
                points.pop()
        elif k == ord("r"):
            view.reset()
        elif k == ord("s"):
            save_current()
        elif k == 81:  # left arrow
            view.pan(-pan_step, 0)
        elif k == 83:  # right arrow
            view.pan(pan_step, 0)
        elif k == 82:  # up arrow
            view.pan(0, -pan_step)
        elif k == 84:  # down arrow
            view.pan(0, pan_step)
        elif k in (ord("q"), 27):
            save_current()
            break

    cv2.destroyAllWindows()
    print(f"Saved {len(labels)} labeled frames -> {output_path}")


if __name__ == "__main__":
    main()

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

Two modes:

  * **plain** (no --slots): every click is just a point. Output entry is
    `{... "points": [[x, y], ...]}` -- unchanged, and what
    refine_dot_ground_truth.py / validate_dot_detector.py read.

  * **slot mode** (--slots given): each point additionally gets a marker
    **slot name** and a per-camera **track id** (stable across frames for
    one physical dot). Stored in a parallel `point_meta` array aligned to
    `points` by index -- so the `points` array stays exactly the bare
    [[x, y], ...] the other tools already consume, and slot/track are
    purely additive (redesign doc §8 P-C). This is the ground truth the
    per-stage metric harness needs: detection recall/precision per
    camera, tracklet fragmentation/purity vs real tracks, per-slot
    assignment precision/recall.

Usage:
    python tools/label_dot_ground_truth.py \\
        --manifest scratch/dot_ground_truth/frame_manifest.json \\
        --session nelli=/path/to/session.db \\
        --slots "hip_R,knee_med_R,knee_lat_R,knee_ant_R,ankle_med_R,ankle_lat_R,heel_R,toe_R,hip_L,knee_med_L,knee_lat_L,knee_ant_L,ankle_med_L,ankle_lat_L,heel_L,toe_L" \\
        --output scratch/dot_ground_truth/labels.json

    # --slots also accepts @path to a file with one slot name per line.

Controls:
    left click       add a point at the cursor (in slot mode: with the
                     active slot + active track; snaps to and inherits
                     from a carried-forward ghost within 18 view-px)
    right click      remove the nearest point (within 20px of the click)
    mouse wheel      zoom in/out, centered on the cursor
    arrow keys       pan the view
    r                reset zoom/pan to fit the whole frame
    u                undo (remove the most recently added point)
    n / space        next frame (saves this frame's points first)
    p                previous frame (saves first)
    s                save progress to --output now
    q / Esc          save and quit

  slot mode only:
    1..9             select active slot (or, with a point picked via g,
                     reassign that point's slot). Digit maps to the slot
                     shown at that index in the on-screen palette.
    0                select / assign `unlabeled`
    [ ]              page the palette when there are more than 9 slots
    g                pick the nearest point for editing (press again to
                     deselect)
    .                start a NEW track id for subsequent new points
    k                set the active track to the picked point's track
                     (resume that dot's track)
    K (shift-k)      set the picked point's track to the active track
    c                materialise every carried-forward ghost as a real
                     point in this frame (inherits slot + track), so you
                     only nudge / delete rather than re-click each

Points are stored in full original-image pixel coordinates regardless of
the current zoom/pan. Progress is saved incrementally keyed by
(dataset, camera_label, video_frame), so re-running resumes and lets you
revisit/correct any frame.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.pose.db_cache import decode_dot_candidates  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402

_WINDOW = "label_dot_ground_truth"
_DISPLAY_SIZE = 1400
_REMOVE_RADIUS_VIEW_PX = 20
_GHOST_SNAP_VIEW_PX = 18
_UNLABELED = "unlabeled"


def _key(entry: dict) -> str:
    return f"{entry['dataset']}::{entry['camera_label']}::{entry['video_frame']}"


def _slot_color(slot: str | None) -> tuple[int, int, int]:
    if not slot or slot == _UNLABELED:
        return (0, 255, 0)  # plain green, same as the no-slot tool
    h = (abs(hash(slot)) % 997) / 997.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


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
        self.view_x = ix - at_dx / self.scale
        self.view_y = iy - at_dy / self.scale

    def pan(self, ddx: float, ddy: float) -> None:
        self.view_x += ddx / self.scale
        self.view_y += ddy / self.scale

    def render(
        self, img: np.ndarray, points: list[tuple[float, float]],
        metas: list[dict] | None, ghosts: list[dict], selected: int | None,
    ) -> np.ndarray:
        """Nearest-neighbor zoom via explicit index gather -- NOT cv2.resize.

        cv2.resize(INTER_NEAREST)'s own internal source-pixel convention
        doesn't exactly match the linear formula to_image() uses to invert a
        click (confirmed empirically 2026-09-08: a real ~1px systematic bias
        after labeling a real GT set, traced to this mismatch). This computes
        canvas pixel (px, py)'s source as floor(view_x + px/scale) directly
        -- the exact forward evaluation of the formula to_image() inverts --
        so the two are consistent by construction.
        """
        px = np.arange(_DISPLAY_SIZE)
        src_x = np.clip(np.floor(self.view_x + px / self.scale).astype(np.int64), 0, self.img_w - 1)
        src_y = np.clip(np.floor(self.view_y + px / self.scale).astype(np.int64), 0, self.img_h - 1)
        valid_x = (self.view_x + px / self.scale >= 0) & (self.view_x + px / self.scale < self.img_w)
        valid_y = (self.view_y + px / self.scale >= 0) & (self.view_y + px / self.scale < self.img_h)
        canvas = img[np.ix_(src_y, src_x)].copy()
        canvas[~valid_y, :] = 0
        canvas[:, ~valid_x] = 0

        def _screen(x, y):
            return (x - self.view_x) * self.scale, (y - self.view_y) * self.scale

        for g in ghosts:
            gx, gy = g["xy"]
            dx, dy = _screen(gx, gy)
            if 0 <= dx < _DISPLAY_SIZE and 0 <= dy < _DISPLAY_SIZE:
                p = (int(round(dx)), int(round(dy)))
                col = _slot_color(g.get("slot"))
                cv2.circle(canvas, p, 7, col, 1)
                cv2.drawMarker(canvas, p, col, cv2.MARKER_TILTED_CROSS, 10, 1)
                lbl = f"{g.get('slot', '')}"
                if g.get("track") is not None:
                    lbl += f"#{g['track']}"
                cv2.putText(canvas, lbl, (p[0] + 9, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, col, 1, cv2.LINE_AA)

        for i, (x, y) in enumerate(points):
            dx, dy = _screen(x, y)
            if not (0 <= dx < _DISPLAY_SIZE and 0 <= dy < _DISPLAY_SIZE):
                continue
            p = (int(round(dx)), int(round(dy)))
            meta = metas[i] if metas is not None and i < len(metas) else None
            col = _slot_color(meta["slot"] if meta else None)
            cv2.drawMarker(canvas, p, col, cv2.MARKER_CROSS, 14, 1)
            cv2.circle(canvas, p, 9, col, 1)
            if i == selected:
                cv2.circle(canvas, p, 14, (0, 0, 255), 2)
            if meta is not None:
                lbl = meta["slot"]
                if meta.get("track") is not None:
                    lbl += f"#{meta['track']}"
                cv2.putText(canvas, lbl, (p[0] + 11, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, col, 1, cv2.LINE_AA)
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


def _save_labels(path: Path, labels: dict, slot_palette: list[str] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(labels.values(), key=lambda e: (e["dataset"], e["camera_label"], e["video_frame"]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2)


def _parse_slots(spec: str) -> list[str]:
    if spec.startswith("@"):
        names = [ln.strip() for ln in Path(spec[1:]).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    else:
        names = [s.strip() for s in spec.split(",") if s.strip()]
    if _UNLABELED not in names:
        names.append(_UNLABELED)
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--session", action="append", required=True, metavar="NAME=PATH",
                     help="Map a manifest 'dataset' name to a session DB path. Repeatable.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--slots", default=None,
                     help="Comma-separated marker slot names, or @path to a file (one per line). "
                          "Enables slot + track-id labeling. 'unlabeled' is always appended.")
    ap.add_argument("--seed-detection-run", default=None,
                     help="marker detection_runs.id -- for a frame with no saved points yet, "
                          "pre-fill the detector's own dot candidates (all tagged 'unlabeled') "
                          "so labeling is review/tag/delete rather than click-from-blank. "
                          "Saved edits always win; a frame you've touched is never re-seeded.")
    args = ap.parse_args()

    slot_mode = args.slots is not None
    slot_palette = _parse_slots(args.slots) if slot_mode else None

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

    # per-camera next-track high-water mark, seeded from anything already in the file
    next_track: dict[str, int] = {}
    for e in labels.values():
        cam = e["camera_label"]
        for m in e.get("point_meta", []):
            if m.get("track") is not None:
                next_track[cam] = max(next_track.get(cam, 0), int(m["track"]) + 1)

    idx = 0
    view: _ViewState | None = None
    points: list[list[float]] = []
    metas: list[dict] = []
    ghosts: list[dict] = []
    img: np.ndarray | None = None
    cur_frame: dict | None = None
    active_slot = 0
    active_track = [0]  # boxed so nested fns can rebind
    palette_page = [0]
    selected: list[int | None] = [None]

    def _cam_key(entry: dict) -> tuple[str, str]:
        return (entry["dataset"], entry["camera_label"])

    def compute_ghosts(i: int) -> list[dict]:
        if not slot_mode:
            return []
        want = _cam_key(manifest[i])
        for j in range(i - 1, -1, -1):
            if _cam_key(manifest[j]) != want:
                continue
            e = labels.get(_key(manifest[j]))
            if e and e.get("points"):
                out = []
                pm = e.get("point_meta", [])
                for k, xy in enumerate(e["points"]):
                    m = pm[k] if k < len(pm) else {"slot": _UNLABELED, "track": None}
                    out.append({"xy": list(xy), "slot": m.get("slot", _UNLABELED), "track": m.get("track")})
                return out
            return []
        return []

    def load_frame(i: int) -> None:
        nonlocal view, points, metas, img, cur_frame, ghosts
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
        saved = labels.get(key, {})
        points = [list(p) for p in saved.get("points", [])]
        pm = saved.get("point_meta", [])
        metas = []
        if slot_mode:
            for k in range(len(points)):
                m = pm[k] if k < len(pm) else {"slot": _UNLABELED, "track": None}
                metas.append({"slot": m.get("slot", _UNLABELED), "track": m.get("track")})

        # Pre-seed from the detector for a frame not yet touched (key absent entirely).
        if args.seed_detection_run and key not in labels:
            row = conn.execute(
                "SELECT keypoints FROM detection_keypoints WHERE detection_run_id = ? "
                "AND shot_video_id = ? AND video_frame = ? AND region_type = 'dots'",
                (args.seed_detection_run, entry["shot_video_id"], entry["video_frame"]),
            ).fetchone()
            if row is not None:
                for cx, cy in decode_dot_candidates(bytes(row[0]))[:, :2]:
                    points.append([float(cx), float(cy)])
                    if slot_mode:
                        metas.append({"slot": _UNLABELED, "track": None})

        ghosts = compute_ghosts(i)
        selected[0] = None
        cam = entry["camera_label"]
        active_track[0] = next_track.get(cam, 0)

    def save_current() -> None:
        if cur_frame is None:
            return
        entry = dict(cur_frame)
        entry["points"] = [list(p) for p in points]
        if slot_mode:
            entry["point_meta"] = [dict(m) for m in metas]
        labels[_key(entry)] = entry
        _save_labels(output_path, labels, slot_palette)

    def on_mouse(event, x, y, flags, _param) -> None:
        nonlocal points, metas
        assert view is not None
        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = view.to_image(x, y)
            slot = slot_palette[active_slot] if slot_mode else None
            trk = active_track[0] if slot_mode else None
            if slot_mode:
                # snap to a carried-forward ghost, inheriting its slot + track
                best_j, best_d = None, None
                for j, g in enumerate(ghosts):
                    gx, gy = g["xy"]
                    d = np.hypot(gx - ix, gy - iy) * view.scale
                    if best_d is None or d < best_d:
                        best_j, best_d = j, d
                if best_j is not None and best_d <= _GHOST_SNAP_VIEW_PX:
                    g = ghosts[best_j]
                    ix, iy = g["xy"]
                    slot, trk = g["slot"], g["track"]
            points.append([ix, iy])
            if slot_mode:
                metas.append({"slot": slot, "track": trk})
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                ix, iy = view.to_image(x, y)
                dists = [np.hypot(px - ix, py - iy) * view.scale for px, py in points]
                j = int(np.argmin(dists))
                if dists[j] <= _REMOVE_RADIUS_VIEW_PX:
                    points.pop(j)
                    if slot_mode and j < len(metas):
                        metas.pop(j)
                    if selected[0] == j:
                        selected[0] = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            view.zoom(1.25 if flags > 0 else 0.8, x, y)

    def pick_nearest(dx_screen: float, dy_screen: float) -> None:
        assert view is not None
        if not points:
            return
        ix, iy = view.to_image(dx_screen, dy_screen)
        dists = [np.hypot(px - ix, py - iy) for px, py in points]
        selected[0] = int(np.argmin(dists))

    pan_step = 60
    last_mouse = [_DISPLAY_SIZE // 2, _DISPLAY_SIZE // 2]

    def on_mouse_track(event, x, y, flags, param):
        last_mouse[0], last_mouse[1] = x, y
        on_mouse(event, x, y, flags, param)

    cv2.namedWindow(_WINDOW)
    cv2.setMouseCallback(_WINDOW, on_mouse_track)
    load_frame(idx)

    while True:
        assert view is not None and img is not None and cur_frame is not None
        canvas = view.render(img, [tuple(p) for p in points],
                              metas if slot_mode else None, ghosts, selected[0])
        done = sum(1 for e in manifest if _key(e) in labels and labels[_key(e)].get("points"))
        line1 = (f"[{idx + 1}/{len(manifest)}] {cur_frame['dataset']}/{cur_frame['camera_label']} "
                 f"frame={cur_frame['video_frame']} tag={cur_frame.get('tag', '')}  "
                 f"points={len(points)}  labeled={done}/{len(manifest)}")
        cv2.putText(canvas, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        if slot_mode:
            base = palette_page[0] * 9
            shown = slot_palette[base:base + 9]
            pal = "  ".join(f"{n + 1}:{s}" for n, s in enumerate(shown))
            act = slot_palette[active_slot]
            sel = "" if selected[0] is None else f"  [picked #{selected[0]}]"
            cv2.putText(canvas, f"active: {act}  track #{active_track[0]}{sel}", (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, pal + ("  0:unlabeled" if _UNLABELED in slot_palette else ""), (8, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 255), 1, cv2.LINE_AA)
        cv2.imshow(_WINDOW, canvas)
        k = cv2.waitKey(20) & 0xFF

        def set_slot(name: str) -> None:
            nonlocal active_slot
            if selected[0] is not None and selected[0] < len(metas):
                metas[selected[0]]["slot"] = name
            else:
                active_slot = slot_palette.index(name)

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
                if slot_mode and metas:
                    metas.pop()
        elif k == ord("r"):
            view.reset()
        elif k == ord("s"):
            save_current()
        elif slot_mode and k == ord("g"):
            if selected[0] is not None:
                selected[0] = None
            else:
                pick_nearest(last_mouse[0], last_mouse[1])
        elif slot_mode and ord("1") <= k <= ord("9"):
            j = palette_page[0] * 9 + (k - ord("1"))
            if j < len(slot_palette) and slot_palette[j] != _UNLABELED:
                set_slot(slot_palette[j])
        elif slot_mode and k == ord("0"):
            if _UNLABELED in slot_palette:
                set_slot(_UNLABELED)
        elif slot_mode and k == ord("["):
            palette_page[0] = max(0, palette_page[0] - 1)
        elif slot_mode and k == ord("]"):
            if (palette_page[0] + 1) * 9 < len(slot_palette):
                palette_page[0] += 1
        elif slot_mode and k == ord("."):
            cam = cur_frame["camera_label"]
            active_track[0] = next_track.get(cam, 0)
            next_track[cam] = active_track[0] + 1
        elif slot_mode and k == ord("k"):
            if selected[0] is not None and selected[0] < len(metas) and metas[selected[0]].get("track") is not None:
                active_track[0] = int(metas[selected[0]]["track"])
        elif slot_mode and k == ord("K"):
            if selected[0] is not None and selected[0] < len(metas):
                metas[selected[0]]["track"] = active_track[0]
        elif slot_mode and k == ord("c"):
            for g in ghosts:
                points.append(list(g["xy"]))
                metas.append({"slot": g["slot"], "track": g["track"]})
        elif k == 81:
            view.pan(-pan_step, 0)
        elif k == 83:
            view.pan(pan_step, 0)
        elif k == 82:
            view.pan(0, -pan_step)
        elif k == 84:
            view.pan(0, pan_step)
        elif k in (ord("q"), 27):
            save_current()
            break

    cv2.destroyAllWindows()
    print(f"Saved {len(labels)} labeled frames -> {output_path}")


if __name__ == "__main__":
    main()

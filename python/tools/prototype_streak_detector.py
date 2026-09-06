# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""prototype_streak_detector.py — iteration tool for an improved reflective-
dot detector that can find *long* motion-blur streaks during the fastest
sword-swing instants, and (once real candidates are actually found) link
them frame-to-frame into per-camera tracklets.

Background (2026-09-06 investigation, see status.md): `dot_blob_detector.py`
rejects any contour on raw pixel `contourArea` (`max_area=400`) *before*
ever checking its shape, so a long-but-thin streak whose area exceeds that
static-round-dot-sized bound is discarded regardless of whether its width
still looks like a real dot -- and even a streak that survives that gets a
second, separately unvalidated cap (`max_streak_length_px=40`). But checking
the *real* detection data (not just contours) for a known worst-case
70-71s sword cut found something more fundamental first: every one of 6
active cameras shows zero valid ArUco corners and zero dot candidates for
the ~400ms core of the swing -- a real, universal detection blackout, not a
shape filter quietly discarding something otherwise-detectable. So this
tool's first job is to check, empirically, whether a genuinely long streak
even survives thresholding at all during that core (a lower brightness
threshold, since a streak spreads a dot's reflected light over more pixels
and dims its peak below a fixed cutoff) before the shape-filter fix matters.

Not part of the production write path (that's posetrak.detection.dot_blob_
detector) -- this script's `detect_blobs_v2()` is a local, prototype-only
fix (shape-classify before any area-based rejection) to iterate on before
porting anything back. Read-only against the session DB; writes only
annotated PNGs + a CSV to the output dir given on the command line.

Background subtraction (2026-09-06, same session): a plain lower global
brightness threshold does NOT work -- swept 235 down to 140 on this exact
window and got flooded (thousands of spurious tiny contours per camera) by
the room's own whitewashed rough-concrete wall texture well before any real
dimmed streak signal would emerge cleanly. The original algorithm study
(marker-mocap-algorithms.md §1.2 step 4) already anticipated needing this,
though for a different purpose ("static-highlight suppression... a
per-camera median background mask over the run removes anything that never
moves"). Reused here for a second purpose: subtracting a per-camera median
background frame (sampled across the whole sequence, so a transiently-
passing performer/sword gets voted out at any one pixel) cancels the wall's
own static appearance entirely, so detection can threshold the *residual*
(how much brighter this frame is than the normal background at each pixel)
at a much lower level without the wall texture ever contributing -- it has
~0 residual everywhere, moving or not. See `--bg-subtract`.

Usage (detection sweep over a short, known-hard time range):
    python tools/prototype_streak_detector.py \\
        --session /path/to/session.db \\
        --run-id <tracking_run_id> \\
        --camera-label oneplus9pro-01 gopro-11_mini_02 \\
        --start-time 69.6 --end-time 70.1 \\
        --threshold 235 200 180 160 \\
        --out-dir scratch/tracking_debug/streak_proto

Usage (background-subtracted residual thresholding instead of raw brightness
-- --threshold values are now residual/diff levels, typically much smaller
than a raw-brightness threshold):
    python tools/prototype_streak_detector.py \\
        --session /path/to/session.db \\
        --run-id <tracking_run_id> \\
        --camera-label oneplus9pro-01 \\
        --start-time 69.6 --end-time 70.1 \\
        --bg-subtract --bg-start-time 34.5 --bg-end-time 100.5 --bg-sample-count 40 \\
        --threshold 60 40 25 \\
        --out-dir scratch/tracking_debug/streak_proto
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from posetrak.detection.frame_source import iter_frames  # noqa: E402
from tools.render_tracking_debug_frames import (  # noqa: E402
    _decode_pose,
    _fk_predict_all,
    _load_skeleton_markers,
    _RunContext,
    _sequential_frame_lookup,
)


@dataclass
class Candidate:
    cx: float
    cy: float
    area: float
    compactness: float
    major_axis_px: float
    minor_axis_px: float
    kind: str  # "round" | "streak" | "rejected"
    dir_x: float = 0.0
    dir_y: float = 0.0
    mean_hue: float = float("nan")   # OpenCV hue, 0-179; NaN if no color frame was given
    mean_sat: float = float("nan")   # 0-255; NaN if no color frame was given
    local_contrast: float = float("nan")  # mean brightness inside minus median in a surrounding
                                           # ring, in the ORIGINAL (pre-background-subtraction)
                                           # frame; NaN if no original-frame reference was given


# Short, visually-distinct label codes -- "round" and "rejected" both start
# with 'r', so kind[:1] alone (an earlier version of this script) mislabeled
# rejected candidates as "round" in annotated crops.
_LABEL = {"round": "o", "streak": "s", "rejected": "x"}
_COLORS = {"round": (0, 255, 0), "streak": (0, 255, 255), "rejected": (0, 0, 255)}


def _classify_contours(
    mask: np.ndarray,
    hsv: np.ndarray | None,
    *,
    min_area: float,
    max_area: float,
    min_compactness: float,
    max_streak_length_px: float,
    max_saturation: float,
    return_rejected: bool,
    gray_orig: np.ndarray | None = None,
    min_local_contrast: float = -255.0,
    ring_margin_px: int = 6,
) -> list[Candidate]:
    """Shared shape (+ optional chroma) classification for both
    detect_blobs_v2() (raw brightness) and detect_blobs_bgsub() (background-
    subtracted residual) -- the two differ only in how *mask* was produced.

    Shape-classifies (round vs. streak vs. reject) *before* any area-based
    rejection, so a long, thin streak's large raw contourArea can't discard
    it before its width is even checked (dot_blob_detector.py's real bug,
    see this module's own docstring). Round-dot classification (compactness
    >= min_compactness) still uses area directly -- a round blob's area IS
    its size, no shape ambiguity. Streak classification uses the
    minAreaRect's own width (minor axis) against the same round-dot diameter
    range, and length against max_streak_length_px, independent of raw
    pixel-count area.

    Chroma check (2026-09-06, added after background subtraction surfaced a
    real, physically-plausible-looking streak that turned out on inspection
    to be the swinger's own moving wrist, not a marker): a real dot is white/
    near-neutral retroreflective material, but skin in motion produces its
    own bright residual too and passes every shape check just as well. A
    reflective dot's mean saturation (HSV) should be low; skin's is not --
    rejects any otherwise-accepted candidate whose mean saturation exceeds
    max_saturation. Only applied when *hsv* is given (None = chroma check
    disabled, e.g. while first tuning shape-only thresholds).

    Local-contrast check (Harri's own idea, 2026-09-06, proposed after
    reviewing the hard/regression videos): background subtraction judges a
    candidate against its own *historical* value at that pixel, which is a
    genuinely different question from "is this a real, compact bright spot
    right now" -- a broad, slow scene-wide brightness drift (auto-exposure
    settling, a light's own flicker) can clear a residual threshold without
    ever being a real local peak, and a real dot's peak brightness can still
    be locally obvious even where the residual is marginal. Computed on
    *gray_orig* -- deliberately the un-subtracted frame, not the residual --
    as (mean brightness inside the contour) - (median brightness in a
    surrounding ring, contour dilated outward by ring_margin_px). Rejects a
    candidate below min_local_contrast. Only computed when *gray_orig* is
    given (None = disabled, e.g. while first calibrating the residual
    threshold alone) -- logged even for an otherwise-accepted candidate
    (not just to reject) so real vs. known-false-positive values can be
    compared before picking a cutoff, same as max_saturation's own
    calibration.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_diameter = 2.0 * np.sqrt(min_area / np.pi)
    max_diameter = 2.0 * np.sqrt(max_area / np.pi)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1.0:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        compactness = 4 * np.pi * area / (perimeter * perimeter)
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        major_axis_px, minor_axis_px = max(rw, rh), min(rw, rh)
        dir_x = dir_y = 0.0
        if compactness >= min_compactness and min_area <= area <= max_area:
            kind = "round"
        elif min_diameter <= minor_axis_px <= max_diameter and major_axis_px <= max_streak_length_px:
            kind = "streak"
            dir_x, dir_y = _streak_direction(rect)
        else:
            kind = "rejected"

        mean_hue = mean_sat = local_contrast = float("nan")
        if kind != "rejected":
            x, y, w, h = cv2.boundingRect(c)
            if hsv is not None:
                local_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(local_mask, [c - [x, y]], -1, 255, thickness=cv2.FILLED)
                region = hsv[y:y + h, x:x + w]
                sel = local_mask == 255
                if sel.any():
                    mean_hue = float(region[..., 0][sel].mean())
                    mean_sat = float(region[..., 1][sel].mean())
                    if mean_sat > max_saturation:
                        kind = "rejected"
            if gray_orig is not None and kind != "rejected":
                m2 = ring_margin_px
                H, W = gray_orig.shape[:2]
                rx0, ry0 = max(0, x - m2), max(0, y - m2)
                rx1, ry1 = min(W, x + w + m2), min(H, y + h + m2)
                inner_mask = np.zeros((ry1 - ry0, rx1 - rx0), dtype=np.uint8)
                cv2.drawContours(inner_mask, [c - [rx0, ry0]], -1, 255, thickness=cv2.FILLED)
                region = gray_orig[ry0:ry1, rx0:rx1].astype(np.float32)
                inner_sel = inner_mask == 255
                ring_sel = ~inner_sel
                if inner_sel.any() and ring_sel.any():
                    local_contrast = float(region[inner_sel].mean() - np.median(region[ring_sel]))
                    if local_contrast < min_local_contrast:
                        kind = "rejected"

        if kind == "rejected" and not return_rejected:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        out.append(Candidate(cx, cy, area, compactness, major_axis_px, minor_axis_px, kind,
                             dir_x, dir_y, mean_hue, mean_sat, local_contrast))
    return out


def detect_blobs_v2(
    gray: np.ndarray,
    *,
    threshold: int,
    min_area: float,
    max_area: float,
    min_compactness: float,
    max_streak_length_px: float,
    hsv: np.ndarray | None = None,
    max_saturation: float = 255.0,
    min_local_contrast: float = -255.0,
    ring_margin_px: int = 6,
    return_rejected: bool = False,
) -> list[Candidate]:
    """Same detection method as dot_blob_detector.detect_blobs(), but shape-
    (and optionally chroma-/local-contrast-) classifies via
    _classify_contours() -- see that function's own docstring for why and
    how. *gray* itself is already the un-subtracted original frame here (no
    background subtraction in this path), so it doubles as _classify_
    contours()'s gray_orig for the local-contrast check.
    """
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return _classify_contours(mask, hsv, min_area=min_area, max_area=max_area,
                              min_compactness=min_compactness,
                              max_streak_length_px=max_streak_length_px,
                              max_saturation=max_saturation, return_rejected=return_rejected,
                              gray_orig=gray, min_local_contrast=min_local_contrast,
                              ring_margin_px=ring_margin_px)


def compute_background(
    file_path: str, sync_table, svid: str, start_time: float, end_time: float, sample_count: int,
) -> np.ndarray | None:
    """Per-camera median background frame (grayscale), sampled at
    *sample_count* times spread evenly across [start_time, end_time] --
    typically the whole sequence, not just the hard window being
    investigated, so a transiently-passing performer/sword gets outvoted by
    the many samples where that pixel is genuinely empty background. Median
    (not mean) for the same reason a median is usually preferred for
    background estimation: robust to the object actually being at a given
    pixel in a minority of the samples.

    Returns None if no sample time has sync coverage on this camera.
    """
    times = np.linspace(start_time, end_time, sample_count)
    frames = []
    for t in times:
        fidx = sync_table.lookup(float(t), svid)
        if fidx is None:
            continue
        for _vf, img in iter_frames(file_path, fidx, fidx + 1):
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            break
    if not frames:
        return None
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def detect_blobs_bgsub(
    gray: np.ndarray,
    background: np.ndarray,
    *,
    diff_threshold: int,
    min_area: float,
    max_area: float,
    min_compactness: float,
    max_streak_length_px: float,
    hsv: np.ndarray | None = None,
    max_saturation: float = 255.0,
    min_local_contrast: float = -255.0,
    ring_margin_px: int = 6,
    return_rejected: bool = False,
) -> list[Candidate]:
    """Same classification as detect_blobs_v2() (via _classify_contours()),
    but thresholds the positive residual (this frame minus the median
    background) instead of raw brightness -- see this module's own
    "Background subtraction" doc section for why. cv2.subtract() clips
    negative results to 0 rather than wrapping, so only pixels *brighter*
    than their normal background (what a reflective highlight looks like,
    moving or not) can pass -- a pixel that got darker than usual (e.g. the
    performer's own body occluding a bright wall patch) never triggers this
    detector, which is the right direction: that's not a highlight.

    The local-contrast check (see _classify_contours()'s own docstring) is
    computed on *gray* -- the real, un-subtracted frame -- not *diff*: a
    residual can pass threshold from a slow, broad drift with no real local
    peak in the actual image, which is exactly the case this check exists to
    catch, so it must look at the original brightness, not the subtracted one.
    """
    diff = cv2.subtract(gray, background)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    return _classify_contours(mask, hsv, min_area=min_area, max_area=max_area,
                              min_compactness=min_compactness,
                              max_streak_length_px=max_streak_length_px,
                              max_saturation=max_saturation, return_rejected=return_rejected,
                              gray_orig=gray, min_local_contrast=min_local_contrast,
                              ring_margin_px=ring_margin_px)


def _streak_direction(rect: tuple) -> tuple[float, float]:
    box = cv2.boxPoints(rect)
    edge_a = box[1] - box[0]
    edge_b = box[2] - box[1]
    long_edge = edge_a if np.dot(edge_a, edge_a) >= np.dot(edge_b, edge_b) else edge_b
    norm = float(np.linalg.norm(long_edge))
    if norm < 1e-9:
        return 0.0, 0.0
    dx, dy = float(long_edge[0] / norm), float(long_edge[1] / norm)
    if dy < 0.0 or (dy == 0.0 and dx < 0.0):
        dx, dy = -dx, -dy
    return dx, dy


def _draw_candidates(img: np.ndarray, candidates: list[Candidate]) -> np.ndarray:
    annotated = img.copy()
    for c in candidates:
        color = _COLORS[c.kind]
        cv2.circle(annotated, (int(c.cx), int(c.cy)), 6, color, 2)
        sat = f"/sat{c.mean_sat:.0f}" if not np.isnan(c.mean_sat) else ""
        cv2.putText(annotated, f"{_LABEL[c.kind]}:{c.major_axis_px:.0f}x{c.minor_axis_px:.0f}{sat}",
                   (int(c.cx) + 8, int(c.cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated


def _fixed_crop_window(
    ctx: _RunContext, start_time: float, end_time: float, half_size: int,
) -> tuple[int, int, int, int]:
    """One fixed crop window for a whole clip, from the FK-predicted dot
    region's own range across [start_time, end_time] (sampled every raw
    tracker step, cheap -- these are plain SQLite reads), padded generously.
    A per-frame recenter would jump around during exactly the divergence
    this tool exists to inspect, since the prediction itself can drift far
    from truth mid-blackout -- see this module's own docstring.
    """
    rows = ctx.conn.execute(
        "SELECT DISTINCT timestamp_s FROM tracking_results WHERE run_id = ? AND person_id = 0 "
        "AND is_smoothed = 0 AND timestamp_s BETWEEN ? AND ? ORDER BY timestamp_s",
        (ctx.run_id, start_time, end_time),
    ).fetchall()
    pts = [_predicted_dot_region(ctx, r["timestamp_s"]) for r in rows]
    pts = [p for p in pts if p is not None]
    if not pts:
        # Fall back to a fixed-size window around whatever single prediction exists near the
        # clip's midpoint, or the frame center if there's truly nothing to go on.
        mid = _predicted_dot_region(ctx, (start_time + end_time) / 2.0) or (1000.0, 1000.0)
        xs, ys = [mid[0]], [mid[1]]
    else:
        xs, ys = zip(*pts)
    pad = half_size
    x0, y0 = max(0, int(min(xs) - pad)), max(0, int(min(ys) - pad))
    x1, y1 = int(max(xs) + pad), int(max(ys) + pad)
    return x0, y0, x1, y1


def _predicted_dot_region(ctx: _RunContext, timestamp_s: float) -> tuple[float, float] | None:
    """Centroid (cx, cy) of this camera's FK-predicted dot-marker positions
    at the RAW tracker step nearest *timestamp_s* -- a raw state exists even
    through a "lost" stretch (pure prediction coasting), so this still gives
    a usable expected search region deep inside a real detection blackout.
    Returns None if no raw tracking_results row exists at all near this time.
    """
    row = ctx.conn.execute(
        "SELECT state FROM tracking_results WHERE run_id = ? AND person_id = 0 AND is_smoothed = 0 "
        "ORDER BY ABS(timestamp_s - ?) LIMIT 1",
        (ctx.run_id, timestamp_s),
    ).fetchone()
    if row is None or row["state"] is None:
        return None
    pos, rot = _decode_pose(row["state"])
    predicted = _fk_predict_all(ctx, pos, rot)
    dot_pts = [(x, y) for name, (x, y) in predicted.items() if name.startswith("dot")]
    if not dot_pts:
        return None
    xs, ys = zip(*dot_pts)
    return float(np.mean(xs)), float(np.mean(ys))


@dataclass
class _Tracklet:
    points: list[tuple[int, float, float]]  # (frame_idx_position_in_sequence, cx, cy)
    missed: int = 0

    def total_displacement(self) -> float:
        if len(self.points) < 2:
            return 0.0
        (_, x0, y0), (_, x1, y1) = self.points[0], self.points[-1]
        return float(np.hypot(x1 - x0, y1 - y0))


def build_tracklets(
    candidates_by_frame: list[list[Candidate]],
    *,
    max_link_px: float,
    max_missed: int,
    min_total_displacement_px: float,
    min_track_len: int,
) -> list[set[int]]:
    """Link one camera's per-frame candidates into tracklets by nearest-
    neighbor position matching (greedy, not Hungarian -- there are only a
    handful of candidates per frame in this prototype's search-radius-gated
    output, so greedy is good enough), then keep only *(frame, candidate
    index)* pairs belonging to a tracklet that (a) survived >= min_track_len
    frames and (b) moved at least min_total_displacement_px start-to-end.

    Added after finding that plain per-frame shape+chroma filtering leaves a
    real noise floor: small residual leaks at FIXED scene locations (light
    fixture flicker, high-contrast edges the median background doesn't fully
    cancel) pass every per-frame check but never actually move, while every
    real dot on a swinging blade must translate substantially across even a
    ~1s window. Requiring *sustained, real motion* -- not just "looks like a
    dot this one frame" -- rejects that whole noise class for free; this is
    also literally the tracklet-construction step from Harri's own proposal
    (2026-09-06), not a separate mechanism bolted on alongside it.

    Does not yet use the streak's own direction/length as an extra linking
    or confidence signal (Harri's bidirectional-streak-velocity idea) --
    first cut is plain nearest-neighbor position tracking; add that once
    this baseline's own false-positive/negative rate is understood.

    Returns, per input frame, the set of that frame's candidate list indices
    that survived -- same length and order as *candidates_by_frame*.
    """
    open_tracklets: list[_Tracklet] = []
    kept: list[set[int]] = [set() for _ in candidates_by_frame]
    # (tracklet, frame_idx, point_idx_in_that_frame) for every point ever added,
    # so a tracklet confirmed only after the fact can still mark its earlier frames kept.
    membership: list[tuple[_Tracklet, int, int]] = []

    for frame_idx, cands in enumerate(candidates_by_frame):
        unmatched = set(range(len(cands)))
        for tr in open_tracklets:
            if not unmatched:
                break
            _, lx, ly = tr.points[-1]
            best_i, best_d = None, max_link_px
            for i in unmatched:
                d = float(np.hypot(cands[i].cx - lx, cands[i].cy - ly))
                if d <= best_d:
                    best_i, best_d = i, d
            if best_i is not None:
                tr.points.append((frame_idx, cands[best_i].cx, cands[best_i].cy))
                tr.missed = 0
                membership.append((tr, frame_idx, best_i))
                unmatched.discard(best_i)
        for tr in open_tracklets:
            if tr.points[-1][0] != frame_idx:
                tr.missed += 1
        open_tracklets = [tr for tr in open_tracklets if tr.missed <= max_missed]
        for i in unmatched:
            new_tr = _Tracklet(points=[(frame_idx, cands[i].cx, cands[i].cy)])
            membership.append((new_tr, frame_idx, i))
            open_tracklets.append(new_tr)

    confirmed = {id(tr) for tr, _, _ in membership
                 if len(tr.points) >= min_track_len and tr.total_displacement() >= min_total_displacement_px}
    for tr, frame_idx, point_idx in membership:
        if id(tr) in confirmed:
            kept[frame_idx].add(point_idx)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--run-id", required=True, help="Any run on the same sequence/skeleton -- used only "
                     "for FK/calibration context (predicted marker positions), not its own tracking result.")
    ap.add_argument("--camera-label", nargs="+", required=True)
    ap.add_argument("--start-time", type=float, required=True)
    ap.add_argument("--end-time", type=float, required=True)
    ap.add_argument("--threshold", type=int, nargs="+", default=[235, 200, 180, 160])
    ap.add_argument("--min-area", type=float, default=4.0)
    ap.add_argument("--max-area", type=float, default=400.0)
    ap.add_argument("--min-compactness", type=float, default=0.5)
    ap.add_argument("--max-streak-length-px", type=float, default=250.0,
                     help="Prototype default, much larger than production's unvalidated 40px -- "
                          "the point of this tool is to find the real value against real footage.")
    ap.add_argument("--search-radius-px", type=float, default=400.0,
                     help="Only log/crop candidates within this radius of the FK-predicted dot "
                          "region -- keeps output focused on the sword, not unrelated bright scene "
                          "elements (light fixtures, glare) confirmed elsewhere in the frame.")
    ap.add_argument("--crop-half-size", type=int, default=350)
    ap.add_argument("--bg-subtract", action="store_true",
                     help="Threshold the residual against a per-camera median background instead "
                          "of raw brightness -- see this module's own docstring. --threshold values "
                          "become residual/diff levels (typically much smaller than a raw-brightness "
                          "threshold, e.g. 20-60).")
    ap.add_argument("--bg-start-time", type=float, default=None,
                     help="--bg-subtract only: start of the range to sample the background from -- "
                          "should span the whole sequence, not just the hard window being "
                          "investigated. Defaults to --start-time if omitted.")
    ap.add_argument("--bg-end-time", type=float, default=None,
                     help="--bg-subtract only: end of the background-sampling range. Defaults to "
                          "--end-time if omitted.")
    ap.add_argument("--bg-sample-count", type=int, default=40)
    ap.add_argument("--max-saturation", type=float, default=255.0,
                     help="Reject an otherwise-accepted candidate whose mean HSV saturation exceeds "
                          "this (0-255) -- a real retroreflective dot is white/near-neutral (low "
                          "saturation); skin in motion produces its own bright residual that passes "
                          "every shape check but is NOT one of our markers. 255 = disabled (off by "
                          "default so shape-only tuning isn't silently affected).")
    ap.add_argument("--min-local-contrast", type=float, default=-255.0,
                     help="Reject an otherwise-accepted candidate whose mean brightness inside its "
                          "own contour, minus the median brightness in a surrounding ring, both "
                          "measured on the ORIGINAL (pre-background-subtraction) frame, falls below "
                          "this. See _classify_contours()'s own docstring for why this is a "
                          "different question from the residual threshold. -255 = disabled (off by "
                          "default so residual-only tuning isn't silently affected).")
    ap.add_argument("--ring-margin-px", type=int, default=6,
                     help="--min-local-contrast only: width of the surrounding ring, in pixels, "
                          "the contour's own bounding box is dilated by.")
    ap.add_argument("--render-video", action="store_true",
                     help="Also render a slow-motion annotated video for this run, one fixed crop "
                          "per camera computed from the FK-predicted dot region across the whole "
                          "range. Uses a single threshold -- the first value in --threshold -- not "
                          "the full sweep. Written into --out-dir as video_<camera_label>.mp4 -- "
                          "same directory as the CSV and per-hit stills, not a separate path (an "
                          "earlier version wrote it via a separate --video-out prefix, which was "
                          "genuinely confusing to find).")
    ap.add_argument("--tracklet-filter", action="store_true",
                     help="--render-video only: link round/streak candidates frame-to-frame by "
                          "nearest position and only draw ones belonging to a tracklet that (a) "
                          "survived >= --tracklet-min-len frames and (b) moved at least "
                          "--tracklet-min-total-disp-px start-to-end -- see build_tracklets()'s own "
                          "docstring for why this rejects a real false-positive class (static "
                          "residual leaks at fixed scene locations) that per-frame shape/chroma "
                          "checks alone cannot.")
    ap.add_argument("--tracklet-max-link-px", type=float, default=60.0)
    ap.add_argument("--tracklet-max-missed", type=int, default=2)
    ap.add_argument("--tracklet-min-total-disp-px", type=float, default=15.0)
    ap.add_argument("--tracklet-min-len", type=int, default=4)
    ap.add_argument("--slow-factor", type=float, default=4.0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    csv_path = out_dir / "streak_proto_candidates.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["camera_label", "video_frame", "timestamp_s", "threshold", "kind",
                      "cx", "cy", "area", "compactness", "major_axis_px", "minor_axis_px",
                      "mean_hue", "mean_sat", "local_contrast", "dist_to_predicted_px"])

    def detect(gray, hsv, background, thr):
        if background is not None:
            return detect_blobs_bgsub(
                gray, background, diff_threshold=thr, min_area=args.min_area,
                max_area=args.max_area, min_compactness=args.min_compactness,
                max_streak_length_px=args.max_streak_length_px,
                hsv=hsv, max_saturation=args.max_saturation,
                min_local_contrast=args.min_local_contrast, ring_margin_px=args.ring_margin_px,
                return_rejected=True,
            )
        return detect_blobs_v2(
            gray, threshold=thr, min_area=args.min_area, max_area=args.max_area,
            min_compactness=args.min_compactness, max_streak_length_px=args.max_streak_length_px,
            hsv=hsv, max_saturation=args.max_saturation,
            min_local_contrast=args.min_local_contrast, ring_margin_px=args.ring_margin_px,
            return_rejected=True,
        )

    for camera_label in args.camera_label:
        ctx = _RunContext(conn, args.run_id, camera_label, use_smoothed=False)
        frame_lo = ctx.sync_table.lookup(args.start_time, ctx.svid)
        frame_hi = ctx.sync_table.lookup(args.end_time, ctx.svid)
        if frame_lo is None or frame_hi is None:
            print(f"{camera_label}: no sync coverage for this range, skipping")
            continue
        print(f"=== {camera_label} frames {frame_lo}-{frame_hi} ===")

        background = None
        if args.bg_subtract:
            bg_start = args.bg_start_time if args.bg_start_time is not None else args.start_time
            bg_end = args.bg_end_time if args.bg_end_time is not None else args.end_time
            print(f"  computing median background from {args.bg_sample_count} samples "
                  f"in [{bg_start}, {bg_end}]...")
            background = compute_background(ctx.file_path, ctx.sync_table, ctx.svid,
                                             bg_start, bg_end, args.bg_sample_count)
            if background is None:
                print(f"{camera_label}: no sync coverage for background sampling, skipping")
                continue

        # Phase A: one decode pass -- CSV (full threshold sweep), per-hit stills (lowest
        # threshold, unfiltered), and a lightweight per-frame record of the single
        # --render-video threshold's own candidates for phase B below (no images kept).
        thr0 = args.threshold[0]
        frame_records: list[tuple[int, float, list[Candidate]]] = []

        for video_frame, img in iter_frames(ctx.file_path, frame_lo, frame_hi + 1):
            t = ctx.sync_table.frame_to_global_time(video_frame, ctx.svid) or args.start_time
            predicted = _predicted_dot_region(ctx, t)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            best_per_threshold: dict[int, list[Candidate]] = {}
            for thr in args.threshold:
                cands = detect(gray, hsv, background, thr)
                if predicted is not None:
                    px, py = predicted
                    cands = [c for c in cands
                             if np.hypot(c.cx - px, c.cy - py) <= args.search_radius_px]
                best_per_threshold[thr] = cands
                for c in cands:
                    dist = (np.hypot(c.cx - predicted[0], c.cy - predicted[1])
                            if predicted is not None else float("nan"))
                    writer.writerow([camera_label, video_frame, f"{t:.3f}", thr, c.kind,
                                     f"{c.cx:.1f}", f"{c.cy:.1f}", f"{c.area:.1f}",
                                     f"{c.compactness:.3f}", f"{c.major_axis_px:.1f}",
                                     f"{c.minor_axis_px:.1f}", f"{c.mean_hue:.1f}", f"{c.mean_sat:.1f}",
                                     f"{c.local_contrast:.1f}", f"{dist:.1f}"])

            lo_thr = min(args.threshold)
            accepted_any = any(c.kind != "rejected" for cs in best_per_threshold.values() for c in cs)
            if accepted_any or any(best_per_threshold.values()):
                n_accepted = sum(1 for c in best_per_threshold[lo_thr] if c.kind != "rejected")
                n_total = len(best_per_threshold[lo_thr])
                print(f"  f{video_frame} t={t:.3f}: thr={lo_thr} accepted={n_accepted} nearby_total={n_total}")
                if predicted is not None:
                    px, py = predicted
                    x0, y0 = int(px - args.crop_half_size), int(py - args.crop_half_size)
                    x1, y1 = int(px + args.crop_half_size), int(py + args.crop_half_size)
                    x0, y0 = max(0, x0), max(0, y0)
                    annotated = _draw_candidates(img, best_per_threshold[lo_thr])
                    crop = annotated[y0:y1, x0:x1]
                    if crop.size:
                        out_path = out_dir / f"{camera_label}_f{video_frame}_t{t:.3f}_thr{lo_thr}.png"
                        cv2.imwrite(str(out_path), crop)

            if args.render_video:
                frame_records.append((video_frame, t, best_per_threshold[thr0]))

        if not args.render_video or not frame_records:
            continue

        # Phase B: tracklet filtering (optional) + one sequential re-decode pass to render
        # the video -- kept separate from phase A because tracklet filtering is inherently
        # non-causal (a tracklet is only confirmed once it's survived several later frames),
        # so rendering can't start until the whole per-frame candidate sequence is known.
        per_frame_cands = [[c for c in cands if c.kind != "rejected"] for _, _, cands in frame_records]
        if args.tracklet_filter:
            kept = build_tracklets(
                per_frame_cands, max_link_px=args.tracklet_max_link_px,
                max_missed=args.tracklet_max_missed,
                min_total_displacement_px=args.tracklet_min_total_disp_px,
                min_track_len=args.tracklet_min_len,
            )
            n_before = sum(len(cs) for cs in per_frame_cands)
            n_after = sum(len(idxs) for idxs in kept)
            print(f"  tracklet filter: {n_before} candidates -> {n_after} "
                  f"({n_before - n_after} rejected as non-moving)")
            per_frame_cands = [[cs[i] for i in sorted(idxs)] for cs, idxs in zip(per_frame_cands, kept)]

        video_crop = _fixed_crop_window(ctx, args.start_time, args.end_time, args.crop_half_size)
        target_times = [t for _, t, _ in frame_records]
        video_path = out_dir / f"video_{camera_label}.mp4"
        video_writer = None
        out_fps = ctx.native_fps / args.slow_factor
        for idx, ((_, t, _), (_, img)) in enumerate(
            zip(frame_records, _sequential_frame_lookup(ctx, target_times))
        ):
            if img is None:
                continue
            annotated = _draw_candidates(img, per_frame_cands[idx])
            cv2.putText(annotated, f"t={t:.3f}s thr={thr0}"
                       + (" tracklet-filtered" if args.tracklet_filter else ""),
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            x0, y0, x1, y1 = video_crop
            cropped = annotated[y0:y1, x0:x1]
            if video_writer is None:
                h, w = cropped.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(video_path), fourcc, out_fps, (w, h))
                print(f"  writing {video_path} ({w}x{h} @ {out_fps:.1f}fps)")
            video_writer.write(cropped)
        if video_writer is not None:
            video_writer.release()

    csv_file.close()
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""dot_blob_detector.py — anonymous reflective-dot blob detection.

Threshold + connected components + centroid, with a compactness filter to
reject elongated glare streaks (light fixtures, shiny edges) that pass a
brightness+area filter alone but aren't a round dot. See
docs/roadmap/features/marker-based-mocap/reflective-dot-detection-design.md
for the design this implements.

The default threshold/area/compactness values were confirmed against real
GoPro frames under ring-light illumination
(reflective-dot-detection-design.md §2.1); retune them only when a capture
shows they no longer fit.

Motion-blur streaks
-------------------
A fast-moving dot smears into an elongated streak whose compactness drops
well below a round dot's -- the same low-compactness signature the filter
above uses to reject glare, and exactly the moments (fast swings) this
detector matters most for. A streak is told apart from glare (a shiny edge,
which can be arbitrarily long) by its *width* (minor axis of the
minimum-area rectangle), which must fall in the round-dot diameter range
implied by ``min_area``/``max_area``, while its *length* is capped:
motion blur lengthens a real dot but does not widen it, whereas glare is
either the wrong width or unrealistically long.

Shape is classified before any area check, and area only gates the
round-dot path. A long, thin streak's raw pixel area grows with its length,
so bounding it by the round-dot-sized ``max_area`` would discard it before
its width was ever examined.

Streak direction
----------------
A streak's own axis is an otherwise-unused velocity signal (see
reflective-dot-detection-design.md, "Open questions", and
streak-velocity-design.md). It is reported as a unit vector (``dir_x``,
``dir_y``) computed from the minimum-area rectangle's long edge, not from
``cv2.minAreaRect``'s angle, whose convention and range differ between
OpenCV versions. A blur streak is a *line*, not an arrow -- nothing says
which end the dot started from -- so the vector is canonicalized to a fixed
half-plane (dy >= 0, or dx >= 0 when dy == 0), and resolving the remaining
180-degree ambiguity against a predicted velocity is left to the consumer.
(0.0, 0.0) for a round dot, which has no streak axis.

Background modes
----------------
A fixed absolute-brightness ``threshold`` never sees a streak whose light is
spread over many pixels and so is dim per pixel; during the fastest motion
this produced complete detection blackouts on every camera. ``background``
(a per-camera median frame, see `compute_background()`) enables two
alternatives:

* ``background_mode='subtract'`` (default) thresholds the residual --
  how much brighter this frame is than *normal* at each pixel. The residual
  stays near zero on static texture and lighting however low ``threshold``
  is set, so a dim streak can be found without flooding on scene texture.
  It also responds to the subject's own moving skin, so ``max_saturation``
  (below) is needed to reject skin-coloured candidates.

  Its failure mode: when the subject is in a pose the background samples did
  not cover (for example sitting where the samples show an empty chair), a
  large connected part of the subject reads as "brighter than usual" at the
  same residual threshold the marker needs. ``findContours`` then returns the
  marker fused into one large non-round blob with the subject's limb, which
  is shape-rejected as a whole, taking the real marker with it. On a
  person-worn-marker capture this was the largest single source of missed
  markers (about half of them).

* ``background_mode='blacklist'`` thresholds the live frame's raw brightness,
  so shape classification only ever sees the marker's own small local
  contour, and uses ``background`` only to veto a candidate sitting on a
  spot that is already nearly as bright with no subject there -- a fixed
  light or glare source, which is what background subtraction was meant to
  suppress. On the same capture it raised recall (about 65% to 76%) and
  precision (about 50% to 77%) together.

  ``threshold`` should be set per camera in this mode. Different sensors'
  tone mapping caps a real marker's peak brightness at very different
  levels even with identical markers and lighting (one camera's markers
  saturated at 251-254, another's peaked at 194-232). A single threshold
  chosen against the dimmer camera lets markers on the brighter camera fuse
  with nearby moderately bright skin or fabric into non-round blobs; one
  chosen against the brighter camera loses the dimmer camera's markers
  outright. Calibrate it from each camera's own marker brightness floor.

Chroma filter
-------------
``max_saturation`` (needs ``bgr``, the original colour frame) rejects a
candidate whose mean HSV saturation is too high: a retroreflective dot is
white or near-neutral, while skin and coloured fabric are not. The mean is
taken over the candidate's contour mask eroded by one pixel. The contour's
anti-aliased boundary pixels blend with whatever is directly behind the
marker (video chroma subsampling contributes too), and behind a strongly
saturated backdrop -- brightly patterned leggings, say -- that inflates the
mean enough to reject a genuine marker. Erosion strips those boundary
pixels without narrowing what still reads as genuinely saturated: on a real
frame it lowered a missed marker's mean from 46.2 to 28.0 (now accepted)
while a fabric-pattern false positive only went from 116.6 to 105.1 (still
rejected). When erosion empties the mask (very small candidates) the
un-eroded mask is used instead of averaging zero pixels.

The background and chroma options are opt-in (``background=None``,
``max_saturation=255.0``), so a caller that passes neither gets plain
brightness thresholding.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BlobCandidate:
    cx: float
    cy: float
    area: float
    compactness: float  # 4*pi*area / perimeter^2 -- 1.0 for a perfect circle
    bbox: tuple[int, int, int, int]
    # Minimum-area-rectangle axes, in pixels. Equal to the equivalent
    # circular diameter (2*sqrt(area/pi)) for a round dot; for a
    # motion-blur streak accepted via the elongated path below,
    # major_axis_px is the streak's real length and minor_axis_px its
    # width (~the dot's true diameter) -- the pair `resolve_dot_assignment`
    # (dot_assignment.cpp) uses to inflate a streaked candidate's
    # measurement noise, and a velocity-from-streak estimate would use too
    # (see streak-velocity-design.md).
    major_axis_px: float
    minor_axis_px: float
    # Unit vector along a motion-blur streak's own axis (module docstring) --
    # (0.0, 0.0) for a round dot, where major_axis_px == minor_axis_px and no
    # streak axis exists to report.
    dir_x: float = 0.0
    dir_y: float = 0.0
    # Per-camera tracklet id (see dot_tracklet.py) -- assigned by a separate,
    # later linking pass over a camera's frame sequence, NOT by detect_blobs()
    # itself (a per-frame function with no memory of earlier frames). -1 until
    # that pass runs; every candidate that gets linked has *some* id, including
    # a length-1 "singleton" tracklet. Detection only judges whether a
    # candidate looks like a dot in isolation; whether it is real needs motion
    # evidence this function doesn't have.
    tracklet_id: int = -1


def detect_blobs(
    gray: np.ndarray,
    *,
    threshold: int = 235,
    min_area: float = 4.0,
    max_area: float = 400.0,
    min_compactness: float = 0.5,
    max_streak_length_px: float = 60.0,
    background: np.ndarray | None = None,
    background_mode: str = "subtract",
    blacklist_frac: float = 0.7,
    blacklist_radius_px: int = 6,
    bgr: np.ndarray | None = None,
    max_saturation: float = 255.0,
) -> list[BlobCandidate]:
    """Detect anonymous reflective-dot candidates in a grayscale frame.

    Parameters
    ----------
    gray:
        Grayscale frame (as ``cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`` would
        produce).
    threshold, min_area, max_area, min_compactness:
        Detector tuning -- see the module docstring for why the defaults
        aren't starting points to retune without a reason.
    max_streak_length_px:
        Cap on a motion-blur streak's length (see module docstring). A
        deliberately modest default: the only real dot streaks confirmed
        frame by frame against video (fast sword swings) were about 20-22 px
        long, while longer candidates (70-170 px) that looked plausible in
        isolation turned out to be the performer's moving skin or noise.
        Treat it as a starting point to validate per capture, and don't raise
        it without a confirmed real example behind the new value.
    background:
        Per-camera median background frame (see `compute_background()`),
        same shape/dtype as `gray`. Meaning depends on `background_mode`.
        None (default) reproduces plain brightness thresholding exactly,
        regardless of `background_mode`.
    background_mode:
        'subtract' (default): `threshold` gates the positive residual
        (`gray` minus `background`) instead of raw brightness -- see the
        module docstring's "Background modes" section. It has a structural
        failure mode: a subject in a pose the background samples didn't
        cover makes a large part of the subject read as "brighter than
        usual" at the same residual threshold as the marker, so the marker
        is fused into one large non-round blob with the subject's limb and
        shape-rejected along with it.
        'blacklist': threshold raw `gray` directly, exactly like
        `background=None` -- but then reject a candidate if `background`
        is already at least `blacklist_frac` as bright, within
        `blacklist_radius_px` px of the candidate's centroid, as the live
        frame is at that same spot. This targets what background
        subtraction is actually for (suppressing a fixed light/glare source
        that's bright regardless of the subject) without the
        subtract-then-classify step that causes the fusion failure above --
        shape classification always runs on the live frame's own local
        contour, never on a blob that can span the whole subject. On real
        footage it recovers markers 'subtract' fuses away, with both better
        recall and better precision.
    blacklist_frac, blacklist_radius_px:
        Only used when `background_mode='blacklist'`. See above.
    bgr:
        The original color frame (same frame `gray` was derived from,
        before grayscale conversion), needed only for the `max_saturation`
        chroma check below. None (default) disables that check regardless
        of `max_saturation`.
    max_saturation:
        Reject an otherwise-accepted candidate whose mean HSV saturation
        (0-255) exceeds this -- a real retroreflective dot is white/
        near-neutral; skin in motion produces its own bright residual that
        clears every shape check but is not one of our markers. 255.0
        (default) disables this check. Requires `bgr`.
    """
    subtract_mode = background is not None and background_mode == "subtract"
    mask_input = cv2.subtract(gray, background) if subtract_mode else gray
    _, mask = cv2.threshold(mask_input, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_diameter = 2.0 * np.sqrt(min_area / np.pi)
    max_diameter = 2.0 * np.sqrt(max_area / np.pi)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) if bgr is not None and max_saturation < 255.0 else None
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
        rect_major, rect_minor = max(rw, rh), min(rw, rh)

        # Shape-classify BEFORE any area-based rejection (see module docstring):
        # a long, thin streak's raw pixel area scales with its length, so bounding
        # it by the round-dot-sized max_area conflates "long" with "big" -- area
        # only ever gates the round-dot path below, never the streak path.
        dir_x = dir_y = 0.0
        if compactness >= min_compactness and min_area <= area <= max_area:
            equiv_diameter = 2.0 * np.sqrt(area / np.pi)
            major_axis_px = minor_axis_px = equiv_diameter
        elif min_diameter <= rect_minor <= max_diameter and rect_major <= max_streak_length_px:
            major_axis_px, minor_axis_px = rect_major, rect_minor
            dir_x, dir_y = _streak_direction(rect)
        else:
            continue

        if hsv is not None and max_saturation < 255.0:
            x, y, w, h = cv2.boundingRect(c)
            local_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(local_mask, [c - [x, y]], -1, 255, thickness=cv2.FILLED)
            # Erode before averaging (see "Chroma filter" in the module
            # docstring): boundary pixels blend with whatever is behind the
            # candidate (anti-aliasing, chroma subsampling), which inflates the
            # mean over a strongly saturated backdrop. Falls back to the
            # un-eroded mask if erosion empties it (very small candidates),
            # rather than averaging zero pixels.
            eroded_mask = cv2.erode(local_mask, np.ones((3, 3), np.uint8))
            sel = eroded_mask == 255
            if not sel.any():
                sel = local_mask == 255
            if sel.any() and float(hsv[y:y + h, x:x + w, 1][sel].mean()) > max_saturation:
                continue

        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        x, y, w, h = cv2.boundingRect(c)

        if background is not None and background_mode == "blacklist":
            # "This exact spot is basically always this bright, subject or not" --
            # a fixed glare/light source, not a marker. Compared against the live
            # frame's OWN local peak (not a fixed absolute level): a dim streak's
            # peak is dim too, so the bar for calling something "already this
            # bright in the background" scales down with it rather than never
            # tripping for anything darker than a full-brightness round dot.
            r = blacklist_radius_px
            cx_i, cy_i = int(round(cx)), int(round(cy))
            by0, by1 = max(0, cy_i - r), min(gray.shape[0], cy_i + r + 1)
            bx0, bx1 = max(0, cx_i - r), min(gray.shape[1], cx_i + r + 1)
            bg_peak = int(background[by0:by1, bx0:bx1].max())
            live_peak = int(gray[by0:by1, bx0:bx1].max())
            if bg_peak >= blacklist_frac * live_peak:
                continue

        out.append(BlobCandidate(cx, cy, area, compactness, (x, y, w, h),
                                  major_axis_px, minor_axis_px, dir_x, dir_y))
    return out


def compute_background(
    frames: list[np.ndarray],
) -> np.ndarray:
    """Per-camera median background frame from a list of sampled grayscale
    frames -- pass `detect_blobs()`'s own `background` parameter. Sample
    across a whole camera's frame range (not just the moments being
    investigated), so a transiently-passing performer/sword gets outvoted
    by the many samples where a given pixel is genuinely empty background.
    Median (not mean): robust to the object actually being at a given pixel
    in a minority of the samples, the standard reason a median is preferred
    for background estimation.
    """
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def _streak_direction(rect: tuple) -> tuple[float, float]:
    """Unit vector along *rect*'s (a cv2.minAreaRect result) long edge,
    canonicalized to a fixed half-plane -- see the module docstring's
    "Streak direction" section for why this doesn't (and can't) recover a
    forward/backward sense, only the axis.

    Computed from cv2.boxPoints() rather than rect's own angle field
    directly: which side of the rectangle the angle describes, and its
    range, is an OpenCV-version-dependent convention that boxPoints already
    resolves into concrete corner coordinates.
    """
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

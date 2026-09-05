# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""dot_blob_detector.py — anonymous reflective-dot blob detection.

Threshold + connected components + centroid, with a compactness filter to
reject elongated glare streaks (light fixtures, shiny edges) that pass a
brightness+area filter alone but aren't a round dot. See
docs/roadmap/features/marker-based-mocap/reflective-dot-detection-design.md
for the design this implements and marker-detection-analysis.md's Question A
for the original method choice.

The default threshold/area/compactness values are the ones confirmed
2026-09-01 against real GoPro capture frames under ring-light illumination
(reflective-dot-detection-design.md §2.1) -- not starting points to retune,
unless a real capture shows they no longer fit.

Motion-blur streak acceptance (2026-09-04): a fast-moving dot smears into
an elongated streak whose compactness drops well below a round dot's --
exactly the low-compactness signature the filter above uses to reject glare
streaks, and exactly the moments (fast sword swings) this detector exists
to help with most. Distinguished from a genuine glare streak (a shiny
edge, which can be arbitrarily long) by checking the blob's *width*
(minor axis of its minimum-area rectangle) against the same round-dot
diameter range min_area/max_area already imply, while still capping its
*length* -- a real dot's width doesn't change under motion blur, only its
length grows, but a glare streak is either the wrong width or unrealistically
long. `max_streak_length_px`'s default is a first cut (not yet confirmed
against a real blurred-dot example the way the other defaults were) --
narrow or widen it once real fast-swing footage is available to check
against.

Streak direction (2026-09-05): a motion-blur streak's own axis is a real,
otherwise-unused velocity signal (reflective-dot-detection-design.md's
"Open questions" -- the streak-to-speed design this feeds is in
docs/roadmap/features/marker-based-mocap/streak-velocity-design.md).
Reported as a unit vector (`dir_x`, `dir_y`) derived from the minimum-area
rectangle's own long edge rather than trusted from cv2.minAreaRect's angle
output directly (its convention/range has changed across OpenCV versions).
A blur streak is a *line*, not an arrow -- there is no way to tell which
end of it the dot started from -- so the vector is canonicalized to a
fixed half-plane (dy >= 0, or dx >= 0 when dy == 0) rather than left
arbitrary; resolving the resulting 180-degree sign ambiguity against a
predicted velocity is left to the consumer. (0.0, 0.0) for a round dot,
where no streak axis exists.
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
    # measurement noise, and a future velocity-from-streak estimate would
    # use too (see this module's own docstring and status.md's 2026-09-04
    # entry for that and the blinking-LED sub-frame-timing idea).
    major_axis_px: float
    minor_axis_px: float
    # Unit vector along a motion-blur streak's own axis (module docstring) --
    # (0.0, 0.0) for a round dot, where major_axis_px == minor_axis_px and no
    # streak axis exists to report.
    dir_x: float = 0.0
    dir_y: float = 0.0


def detect_blobs(
    gray: np.ndarray,
    *,
    threshold: int = 235,
    min_area: float = 4.0,
    max_area: float = 400.0,
    min_compactness: float = 0.5,
    max_streak_length_px: float = 40.0,
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
        Cap on a motion-blur streak's length (see module docstring) -- a
        first cut, not yet validated against real footage.
    """
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_diameter = 2.0 * np.sqrt(min_area / np.pi)
    max_diameter = 2.0 * np.sqrt(max_area / np.pi)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        compactness = 4 * np.pi * area / (perimeter * perimeter)
        equiv_diameter = 2.0 * np.sqrt(area / np.pi)
        major_axis_px = minor_axis_px = equiv_diameter
        dir_x = dir_y = 0.0
        if compactness < min_compactness:
            rect = cv2.minAreaRect(c)
            (rw, rh) = rect[1]
            major_axis_px, minor_axis_px = max(rw, rh), min(rw, rh)
            is_streak = (
                min_diameter <= minor_axis_px <= max_diameter
                and major_axis_px <= max_streak_length_px
            )
            if not is_streak:
                continue
            dir_x, dir_y = _streak_direction(rect)
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        x, y, w, h = cv2.boundingRect(c)
        out.append(BlobCandidate(cx, cy, area, compactness, (x, y, w, h),
                                  major_axis_px, minor_axis_px, dir_x, dir_y))
    return out


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

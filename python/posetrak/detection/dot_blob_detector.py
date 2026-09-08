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

Shape-before-area ordering fix + background subtraction + chroma filter
(2026-09-06, ported from python/tools/prototype_streak_detector.py after a
real investigation on a fast sword-swing capture -- see status.md's
2026-09-06 entries for the full account): the shape (compactness/axes)
computation used to run *after* an unconditional `area` bound, so a long,
thin motion-blur streak whose raw pixel area exceeds the round-dot-sized
`max_area` got discarded before its width was ever checked -- exactly the
fastest, most violent swing instants this detector exists to help with
most. Area is now only ever checked *within* each shape branch (round:
`min_area <= area <= max_area`; streak: judged by width/length instead,
independent of area) so a real streak's raw pixel count can no longer
disqualify it before its shape is considered.

Checking real detection data (not just contours) during that same
investigation found something more fundamental first, though: during the
most violent ~400ms of a real swing, every active camera showed a
universal detection blackout -- a fixed absolute-brightness `threshold`
simply never sees a streak dim enough (its light spread over more pixels).
`background` (a per-camera median frame) enables residual-based detection
instead: threshold how much brighter this frame is than *normal* at each
pixel, which stays near zero for the room's own static texture/lighting
regardless of how low `threshold` is set, so a real but dim streak can be
found without flooding on scene texture. This alone also picks up the
performer's own moving skin, a real false-positive class a fixed
brightness threshold doesn't encounter -- `max_saturation` (needs `bgr`,
the original color frame) rejects a candidate whose own color is skin-like
rather than the white/near-neutral retroreflective material a real dot is.
Both are opt-in (`background=None`, `max_saturation=255.0` by default) so
a caller that doesn't pass them gets exactly today's brightness-threshold
behavior, aside from the ordering fix above.

Erode the chroma-check mask before averaging (2026-09-08, found on a
person-worn-marker capture with markers sewn onto brightly patterned
red/orange leggings -- see status.md's 2026-09-08 entry): a real,
correctly round, otherwise-obviously-genuine marker (compactness 0.78,
confirmed by eye against the source frame) was rejected at mean
saturation 46.2 against `max_saturation=45.0` -- just barely over. The
inflation traced to video compression's chroma subsampling: a small
marker's own contour mask includes its anti-aliased/chroma-blended
boundary pixels, which sample color from a mix of the marker and
whatever's directly behind it, and a highly saturated backdrop (this
capture's leggings; a neutral backdrop like skin or bare floor doesn't
trigger this) pulls that boundary average up regardless of the marker's
own true (near-neutral) color. Eroding the filled contour mask by one
pixel before computing the mean strips exactly those boundary pixels,
confirmed directly on this same real frame: the miss's mean dropped
46.2 -> 28.0 (would now pass), while a genuine same-frame false positive
(actual fabric-pattern texture, not a marker) stayed correctly rejected
at 116.6 -> 105.1 -- eroding removes the compression-bleed bias without
narrowing what still reads as "genuinely saturated". Falls back to the
un-eroded mask when erosion empties it out entirely (a handful of
very-small candidates, a handful of pixels across), rather than
computing a mean over zero pixels.

`background_mode='blacklist'` (2026-09-08, prototyped against a hand-
labeled ground-truth set after Harri's own diagnosis that background
subtraction was structurally the wrong tool here -- see status.md):
`background_mode='subtract'` (the original mechanism, still the default)
has a real failure mode distinct from the chroma-bleed one above -- a
marker on a subject that's simply in a pose the background model's
samples didn't cover (sitting where the samples mostly show an empty
chair, say) makes a large connected swath of the subject read as
"brighter than usual" at the very same residual threshold the marker
itself needs, so `findContours` returns the marker fused into one large,
obviously-not-round blob together with the subject's own limb -- shape-
rejected as a whole, taking a real marker down with it every time. Ground
truth confirmed this was the single largest recall loss on a real person-
marker capture (roughly half of all misses). `'blacklist'` mode instead
thresholds the live frame's raw brightness directly (so shape
classification only ever sees the marker's own small local contour, never
a blob that can span the whole subject) and uses `background` only to
veto a candidate sitting on a spot that's *already* nearly as bright with
no subject there at all -- a fixed light/glare source, the thing
background subtraction was actually trying to suppress. Validated on the
same real capture: recall 65.1% -> 75.9%, precision 50.0% -> 76.8%
(threshold=200) -- both up together, not a trade-off. A per-camera
`threshold` matters more here than under `'subtract'`: two different
phone/action-camera sensors' own tone-mapping can cap a real marker's
peak brightness at very different absolute levels even with identical
markers and lighting (confirmed: one camera's real markers saturate the
sensor at 251-254, another's cap out at 194-232) -- picking one global
threshold against the dimmer camera's floor lets real markers on the
brighter camera start fusing with nearby moderately-bright skin/fabric
into non-round blobs (the same fusion failure as 'subtract' mode, just
triggered by a threshold set too low rather than by subtraction);
against the brighter camera's floor, the dimmer camera loses real
markers outright. Calibrate `threshold` per camera from that camera's
own real marker brightness floor, not once for a whole multi-camera rig.
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
    # Per-camera tracklet id (2026-09-06, dot_tracklet.py) -- assigned by a
    # separate, later linking pass over a whole camera's frame sequence, NOT
    # by detect_blobs() itself (this is a per-frame function with no memory
    # of earlier frames). -1 until that pass runs; every candidate that does
    # get linked gets *some* id, including a length-1 "singleton" tracklet --
    # detection no longer judges whether a candidate is real (that needs
    # motion evidence this function doesn't have), only whether it looks
    # like a dot in isolation. See dot_tracklet.py's own module docstring.
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
        Cap on a motion-blur streak's length (see module docstring). Raised
        2026-09-06 from an initial, unvalidated 40px guess to 60px -- a
        deliberately modest bump: the only *confirmed* real dot streak found
        during that investigation (a sword-swing capture, background-
        subtracted detection, checked frame-by-frame against the actual
        video) was ~20-22px long; several longer candidates (70-170px) that
        looked plausible in isolation turned out on inspection to be the
        performer's own moving skin or ambiguous noise, not markers. Still a
        first cut -- land on a capture-specific number the way every other
        default here was validated rather than assume this one transfers
        as-is, and don't raise it further without a similarly-confirmed
        real example backing the new value.
    background:
        Per-camera median background frame (see `compute_background()`),
        same shape/dtype as `gray`. Meaning depends on `background_mode`.
        None (default) reproduces plain brightness thresholding exactly,
        regardless of `background_mode`.
    background_mode:
        'subtract' (default): `threshold` gates the positive residual
        (`gray` minus `background`) instead of raw brightness -- see the
        module docstring's "background subtraction" section. Confirmed
        (2026-09-08 person-worn-marker capture, status.md) to have a real
        structural failure mode: a marker sitting on a subject that's
        merely in an atypical *position* (a normal pose the background
        model's samples didn't cover -- e.g. sitting where the samples show
        an empty chair) makes a large connected swath of the subject read
        as "brighter than usual" at the *same* residual threshold as the
        marker itself, so `cv2.findContours` returns the marker fused into
        one large, obviously-non-round blob with the subject's own limb --
        shape-rejected as a whole, taking the real marker down with it.
        'blacklist': threshold raw `gray` directly, exactly like
        `background=None` -- but then reject a candidate if `background`
        is already at least `blacklist_frac` as bright, within
        `blacklist_radius_px` px of the candidate's centroid, as the live
        frame is at that same spot. This targets what background
        subtraction was actually for (suppressing a fixed light/glare
        source that's bright regardless of the subject) without the
        subtract-then-classify step that causes the fusion failure above --
        shape classification always runs on the live frame's own local
        contour, never on a blob that can span the whole subject. Confirmed
        on real footage (status.md's 2026-09-08 entry) to recover markers
        'subtract' fuses away, at both better recall and better precision,
        not a recall/precision trade-off dressed up as an improvement.
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

        # Shape-classify BEFORE any area-based rejection (2026-09-06 fix, see module
        # docstring): a long, thin streak's raw pixel area scales with its length, so
        # bounding it by the round-dot-sized max_area conflates "long" with "big" --
        # area only ever gates the round-dot path below, never the streak path.
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
            # Erode before averaging (module docstring's 2026-09-08 entry): a
            # candidate's boundary pixels blend with whatever's directly behind
            # it (anti-aliasing / video chroma subsampling), which inflates the
            # mean when the backdrop is itself highly saturated -- a real,
            # correctly-shaped marker was rejected this way on a leggings-print
            # backdrop. Falls back to the un-eroded mask if erosion empties it
            # (small candidates, a handful of pixels), rather than average zero.
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

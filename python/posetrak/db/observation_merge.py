"""observation_merge.py — Merge multi-source pose_observations rows into one array.

Phase 2 of hand-detection refinement lets pose_observations hold multiple rows
per (sequence, camera, frame, person) — one per detection source ('body',
'hand_l', 'hand_r') — instead of Phase 1's interim hack of patching refined
hand keypoints into the whole-body blob in place. Every read path (the
keypoint editor, the C++ tracker's session loader) must merge these rows back
into one dense per-marker array before doing anything else with them. This
module is the single place that implements that merge so read-path call
sites can't drift out of sync with each other.

Idea 3 (automated post-edit redetection) adds a second, generic layer on top:
any source of the form '<base>.refined' takes precedence over its plain
'<base>' counterpart, per marker slot, for the same (camera, frame). Hand
redetection ('hand_l' -> 'hand_l.refined', 'hand_r' -> 'hand_r.refined') is
just the first feature to use this — the rule itself has no hand-specific
knowledge, so a later auto-detection-after-edit feature can reuse it without
its own merge special-case.

See docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md.
"""
from __future__ import annotations

import numpy as np

BODY_SOURCE = "body"
_REFINED_SUFFIX = ".refined"

# Keep in sync with posetrak.detection.hand_refinement's _HAND_BASE_IDX /
# _HAND_N_KP (hand21 keypoints are written at COCO-133 offset 91 for the left
# hand, 112 for the right hand). Duplicated rather than imported so this
# low-level db module doesn't depend on the detection pipeline package.
#
# Maps a *base* source name (any '.refined' suffix already stripped) to its
# (start_index, count) placement in the merged array. 'body' isn't listed
# here — it's the base layer that establishes the merged array's width,
# handled separately below; only overlay sources place into a sub-range of it.
_SOURCE_PLACEMENT = {
    "hand_l": (91, 21),
    "hand_r": (112, 21),
}


def _split_source(source: str) -> tuple[str, bool]:
    """Split *source* into (base_name, is_refined)."""
    if source.endswith(_REFINED_SUFFIX):
        return source[: -len(_REFINED_SUFFIX)], True
    return source, False


def merge_observation_sources(
    rows: list[tuple[str, np.ndarray]],
) -> np.ndarray | None:
    """Merge (source, kp[n,3]) rows sharing one (camera, frame) into one array.

    'body' is the base layer — its own row establishes the merged array's
    width. Every other recognised source (`_SOURCE_PLACEMENT`) overlays its
    own index range on top, the same precedence Phase 1 already validated by
    patching hand keypoints directly into the whole-body blob.

    A source '<base>.refined' overrides its plain '<base>' counterpart for
    the same slots — applied as two explicit passes (plain sources, then
    '.refined' sources) rather than relying on *rows*' order, since nothing
    upstream guarantees '.refined' rows arrive after their base row (the DB
    query has no ORDER BY on source). Rows for an unrecognised base source
    are ignored.

    Returns None if no 'body' row is present (nothing to merge onto).
    """
    body: np.ndarray | None = None
    overlay_rows: list[tuple[str, bool, np.ndarray]] = []  # (base, is_refined, kp)
    for source, kp in rows:
        base, is_refined = _split_source(source)
        if base == BODY_SOURCE and not is_refined:
            body = kp
        elif base in _SOURCE_PLACEMENT:
            overlay_rows.append((base, is_refined, kp))

    if body is None:
        return None

    merged = body.copy()
    for pass_is_refined in (False, True):
        for base, is_refined, kp in overlay_rows:
            if is_refined != pass_is_refined:
                continue
            start, count = _SOURCE_PLACEMENT[base]
            n = min(count, kp.shape[0], merged.shape[0] - start)
            if n <= 0:
                continue
            merged[start:start + n] = kp[:n]

    return merged


def refined_indices(rows: list[tuple[str, np.ndarray]]) -> frozenset[int]:
    """Return which merged marker indices are currently backed by a
    '<base>.refined' source among *rows*.

    Used only by timeline_status.read_timeline_status to flag a slot as
    "value came from automated redetection, not yet human-verified"
    (`STATUS_ORANGE`) — `merge_observation_sources` itself doesn't need this
    (its other caller, `read_observations_with_edits`, only wants merged
    coordinates), so it's kept as a separate, small function rather than
    changing that one's return shape.
    """
    refined: set[int] = set()
    for source, kp in rows:
        base, is_refined = _split_source(source)
        if not is_refined or base not in _SOURCE_PLACEMENT:
            continue
        start, count = _SOURCE_PLACEMENT[base]
        n = min(count, kp.shape[0])
        refined.update(range(start, start + n))
    return frozenset(refined)

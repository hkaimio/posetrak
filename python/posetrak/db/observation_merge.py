"""observation_merge.py — Merge multi-source pose_observations rows into one array.

Phase 2 of hand-detection refinement lets pose_observations hold multiple rows
per (sequence, camera, frame, person) — one per detection source ('body',
'hand_l', 'hand_r') — instead of Phase 1's interim hack of patching refined
hand keypoints into the whole-body blob in place. Every read path (the
keypoint editor, the C++ tracker's session loader) must merge these rows back
into one dense per-marker array before doing anything else with them. This
module is the single place that implements that merge so read-path call
sites can't drift out of sync with each other.

See docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md.
"""
from __future__ import annotations

import numpy as np

BODY_SOURCE = "body"

# Keep in sync with posetrak.detection.hand_refinement's _HAND_BASE_IDX /
# _HAND_N_KP (hand21 keypoints are written at COCO-133 offset 91 for the left
# hand, 112 for the right hand). Duplicated rather than imported so this
# low-level db module doesn't depend on the detection pipeline package.
_HAND_BASE_IDX = {"hand_l": 91, "hand_r": 112}
_HAND_N_KP = 21


def merge_observation_sources(
    rows: list[tuple[str, np.ndarray]],
) -> np.ndarray | None:
    """Merge (source, kp[n,3]) rows sharing one (camera, frame) into one array.

    'body' is the base layer — its own row establishes the merged array's
    width. 'hand_l'/'hand_r' rows overwrite their own COCO-133 index range on
    top, the same precedence Phase 1 already validated by patching hand
    keypoints directly into the whole-body blob. Rows for an unrecognised
    source are ignored.

    Returns None if no 'body' row is present (nothing to merge onto).
    """
    body: np.ndarray | None = None
    hand_rows: list[tuple[str, np.ndarray]] = []
    for source, kp in rows:
        if source == BODY_SOURCE:
            body = kp
        elif source in _HAND_BASE_IDX:
            hand_rows.append((source, kp))

    if body is None:
        return None

    merged = body.copy()
    for source, kp in hand_rows:
        base = _HAND_BASE_IDX[source]
        n = min(_HAND_N_KP, kp.shape[0], merged.shape[0] - base)
        if n <= 0:
            continue
        merged[base:base + n] = kp[:n]

    return merged

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""binding.py — whether a skeleton has a marker for the keypoints of a pose sequence.

Mirrors the binding rule of the C++ ``SessionReader::load_observations`` for one question: does any marker
of the skeleton bind to any slot of the sequence's keypoint manifest? A run that binds nothing would only
fail later with "all keypoints filtered out"; :func:`missing_pose_models` lets the command line name the
missing pose model before it starts the tracker.
"""
from __future__ import annotations

import sqlite3

import yaml

from posetrak.detection.pose_models import COCO133_HAND_BASE


def missing_pose_models(conn: sqlite3.Connection, sequence_id: str, skeleton_yaml: str) -> list[str] | None:
    """The pose models of the sequence's manifest if the skeleton binds none of its slots, else None.

    None is also returned for a sequence without manifest rows, whose layout is the implied COCO-133 one
    that the tracker reads itself.
    """
    rows = conn.execute(
        "SELECT obs_source, keypoint_idx, name, source FROM pose_sequence_keypoints WHERE sequence_id = ?",
        (sequence_id,),
    ).fetchall()
    if not rows:
        return None
    markers = yaml.safe_load(skeleton_yaml).get("markers") or []
    landmarks = {m["landmark"] for m in markers if m.get("landmark")}
    coco_ids = {m["openpose_keypoint"] for m in markers if m.get("openpose_keypoint") is not None}
    for obs_source, idx, name, source in rows:
        if name in landmarks or (":" not in name and f"{source}:{name}" in landmarks):
            return None
        if source == "coco133" and obs_source not in ("hand_l", "hand_r") and idx in coco_ids:
            return None
        if source == "hand21" and obs_source in ("hand_l", "hand_r"):
            side = "left" if obs_source == "hand_l" else "right"
            if COCO133_HAND_BASE[side] + idx in coco_ids:
                return None
    return sorted({source for *_, source in rows})


def missing_pose_model_message(models: list[str]) -> str:
    return (f"The skeleton has no marker bound to the keypoints of this sequence (pose model: {', '.join(models)}). "
            f"Add the pose model's markers to the skeleton with 'posetrak skeleton compose --base SKELETON "
            f"--marker-set {models[0]}', or track the sequence with a skeleton that has them.")

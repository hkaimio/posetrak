# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""evaluate.py — freeze the human dot-assignment decisions of a trial as ground truth and
score automatic proposals against them.

A frozen ground truth is a JSON file of segments (camera, tracklet id, frame range, label)
where the label is a slot name, ``NOT_A_MARKER`` or ``OTHER_SUBJECT``. Tracklet ids are those
of one detection run, so a truth file is only valid for that run's dot candidates.
Tracklets nobody decided about are left out and are not scored.

Scores are in tracklet-frames, so a long tracklet counts for its length.
"""
from __future__ import annotations

import collections
import datetime
import json
import sqlite3
from pathlib import Path

from posetrak.markers.review import NOT_A_MARKER, OTHER_SUBJECT, Proposal, load_edits, load_proposals, segments

FORMAT = 1
Segment = tuple[str, int, int, int, str]       # camera id, tracklet id, first frame, last frame, label


def decided_segments(conn: sqlite3.Connection, trial_id: str, detection_run_id: str) -> list[Segment]:
    """The frame ranges of the newest proposal set's tracklets that a human has decided."""
    edits = collections.defaultdict(list)
    for e in load_edits(conn, trial_id, detection_run_id):
        edits[(e.camera_id, e.tracklet_id)].append(e)
    out = []
    for p in load_proposals(conn, detection_run_id):
        for first, last, slot, origin in segments(p.first_frame, p.last_frame, None, edits.get((p.camera_id, p.tracklet_id), [])):
            if origin == "edit":
                out.append((p.camera_id, p.tracklet_id, first, last, slot))
    return out


def freeze(conn: sqlite3.Connection, trial_id: str, detection_run_id: str, path: Path, description: str) -> int:
    """Write the decided segments to *path*; returns their number."""
    rows = decided_segments(conn, trial_id, detection_run_id)
    doc = {
        "format": FORMAT,
        "description": description,
        "trial_id": trial_id,
        "detection_run_id": detection_run_id,
        "frozen_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "not_a_marker": NOT_A_MARKER,
        "other_subject": OTHER_SUBJECT,
        "segments": [dict(zip(("camera_id", "tracklet_id", "first_frame", "last_frame", "label"), r)) for r in rows],
    }
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return len(rows)


def load_frozen(path: Path) -> list[Segment]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    assert doc["format"] == FORMAT
    return [(s["camera_id"], s["tracklet_id"], s["first_frame"], s["last_frame"], s["label"]) for s in doc["segments"]]


def evaluate(proposals: list[Proposal], truth: list[Segment]) -> dict:
    """Score *proposals* on the tracklet-frames of *truth*.

    Returns ``by_confidence`` (frames correct / wrong slot / given a slot though not a marker of this
    subject / given no slot, per proposal confidence), overall ``precision`` and ``recall`` of slots,
    ``per_slot`` counts and the ``confusions`` (proposed, true) -> frames.
    """
    by_key = {(p.camera_id, p.tracklet_id): p for p in proposals}
    by_conf: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    per_slot: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    confusions: collections.Counter = collections.Counter()
    for cam, tid, first, last, label in truth:
        p = by_key.get((cam, tid))
        if p is None:
            continue
        n = last - first + 1
        mine = p.slot
        real = label not in (NOT_A_MARKER, OTHER_SUBJECT)
        outcome = ("correct" if mine == label else "wrong_slot" if mine and real else "wrong_class" if mine else
                   "missed" if real else "correct_none")
        by_conf[p.confidence][outcome] += n
        by_conf[p.confidence]["total"] += n
        if mine:
            per_slot[mine]["proposed"] += n
            per_slot[mine]["right"] += n * (mine == label)
        if real:
            per_slot[label]["truth"] += n
            per_slot[label]["found"] += n * (mine == label)
            if mine and mine != label:
                confusions[(mine, label)] += n
    proposed = sum(s["proposed"] for s in per_slot.values())
    truth_n = sum(s["truth"] for s in per_slot.values())
    return {
        "by_confidence": {k: dict(v) for k, v in by_conf.items()},
        "precision": sum(s["right"] for s in per_slot.values()) / max(proposed, 1),
        "recall": sum(s["found"] for s in per_slot.values()) / max(truth_n, 1),
        "per_slot": {k: dict(v) for k, v in per_slot.items()},
        "confusions": dict(confusions),
    }


def report(result: dict, worst: int = 8) -> str:
    lines = [f"precision {100 * result['precision']:.1f}%  recall {100 * result['recall']:.1f}%"]
    for conf in ("high", "medium", "review", "none"):
        s = result["by_confidence"].get(conf)
        if s:
            t = s["total"]
            lines.append(f"  {conf:7s} {t:6d} frames: correct {100 * s.get('correct', 0) / t:5.1f}%  wrong slot "
                         f"{100 * s.get('wrong_slot', 0) / t:5.1f}%  wrong class {100 * s.get('wrong_class', 0) / t:5.1f}%  "
                         f"missed {100 * s.get('missed', 0) / t:5.1f}%")
    slots = [(k, v) for k, v in result["per_slot"].items() if v.get("truth", 0) >= 300]
    slots.sort(key=lambda kv: kv[1].get("found", 0) / kv[1]["truth"])
    lines.append("  lowest recall: " + ", ".join(
        f"{k} {100 * v.get('found', 0) / v['truth']:.0f}%" for k, v in slots[:worst]))
    top = sorted(result["confusions"].items(), key=lambda kv: -kv[1])[:worst]
    lines.append("  top confusions: " + ", ".join(f"{a}->{b} {n}" for (a, b), n in top))
    return "\n".join(lines)

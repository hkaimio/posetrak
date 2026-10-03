# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""review.py — storage of dot-assignment proposals and the human edits made over them.

A proposal says which slot a dot tracklet probably is (see autolabel.py); an edit is a
human decision. :func:`effective_slot` combines them: the latest assign, reject or other edit
covering a frame wins, then the proposal unless a forbid edit rules it out.

A tracklet a human has classed as clutter or as another subject's marker has the pseudo-slot
:data:`NOT_A_MARKER` or :data:`OTHER_SUBJECT`; one nobody has decided about has no slot.
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass


NOT_A_MARKER = "<not a marker>"
OTHER_SUBJECT = "<other subject>"


@dataclass(frozen=True)
class Edit:
    camera_id: str
    tracklet_id: int
    kind: str                      # assign | forbid | reject | other
    slot: str | None
    first_frame: int | None = None
    last_frame: int | None = None
    created_at: str = ""
    id: str = ""                   # the stored row's id; empty for an edit not stored yet

    def covers(self, frame: int) -> bool:
        return self.first_frame is None or self.first_frame <= frame <= (self.last_frame or self.first_frame)


@dataclass(frozen=True)
class Proposal:
    camera_id: str
    tracklet_id: int
    first_frame: int
    last_frame: int
    slot: str | None            # None: on the subject but no slot fitted
    cost_px: float
    margin_px: float
    support: str
    confidence: str


def save_proposals(conn: sqlite3.Connection, detection_run_id: str, pose_run_id: str | None,
                   rows: list[Proposal], subject_id: str | None = None) -> str:
    """Write *rows* as a new proposal set; returns the set's id."""
    proposal_set = str(uuid.uuid4())
    conn.executemany(
        "INSERT INTO dot_assignment_proposals (id, detection_run_id, subject_id, camera_instance_id, tracklet_id,"
        " first_frame, last_frame, slot_name, cost_px, margin_px, support, confidence, pose_run_id, proposal_set)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(str(uuid.uuid4()), detection_run_id, subject_id, r.camera_id, r.tracklet_id, r.first_frame, r.last_frame,
          r.slot, r.cost_px, r.margin_px, r.support, r.confidence, pose_run_id, proposal_set) for r in rows])
    conn.commit()
    return proposal_set


def load_proposals(conn: sqlite3.Connection, detection_run_id: str, proposal_set: str | None = None) -> list[Proposal]:
    """The newest proposal set of the run, or the given one."""
    if proposal_set is None:
        row = conn.execute("SELECT proposal_set FROM dot_assignment_proposals WHERE detection_run_id=?"
                           " ORDER BY created_at DESC LIMIT 1", (detection_run_id,)).fetchone()
        if row is None:
            return []
        proposal_set = row[0]
    return [Proposal(*r) for r in conn.execute(
        "SELECT camera_instance_id, tracklet_id, first_frame, last_frame, slot_name, cost_px, margin_px,"
        " support, confidence FROM dot_assignment_proposals WHERE proposal_set=?", (proposal_set,))]


def add_edit(conn: sqlite3.Connection, trial_id: str, detection_run_id: str, edit: Edit,
             subject_id: str | None = None) -> str:
    """Store *edit* and return its id."""
    edit_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO dot_assignment_edits (id, trial_id, detection_run_id, subject_id, camera_instance_id,"
        " tracklet_id, first_frame, last_frame, kind, slot_name) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (edit_id, trial_id, detection_run_id, subject_id, edit.camera_id, edit.tracklet_id,
         edit.first_frame, edit.last_frame, edit.kind, edit.slot))
    conn.commit()
    return edit_id


def remove_edit(conn: sqlite3.Connection, edit_id: str) -> None:
    conn.execute("DELETE FROM dot_assignment_edits WHERE id=?", (edit_id,))
    conn.commit()


def load_edits(conn: sqlite3.Connection, trial_id: str, detection_run_id: str) -> list[Edit]:
    """All edits of the trial for the run, oldest first."""
    return [Edit(*r) for r in conn.execute(
        "SELECT camera_instance_id, tracklet_id, kind, slot_name, first_frame, last_frame, created_at, id"
        " FROM dot_assignment_edits WHERE trial_id=? AND detection_run_id=? ORDER BY created_at, rowid",
        (trial_id, detection_run_id))]


def effective_slot(proposal: Proposal | None, edits: list[Edit], frame: int) -> tuple[str | None, str]:
    """(slot, origin) of a tracklet at *frame*; origin is ``edit``, ``proposal`` or ``none``.

    *edits* are those of the tracklet, oldest first.
    """
    covering = [e for e in edits if e.covers(frame)]
    for e in reversed(covering):
        if e.kind == "assign":
            return e.slot, "edit"
        if e.kind == "reject":
            return NOT_A_MARKER, "edit"
        if e.kind == "other":
            return OTHER_SUBJECT, "edit"
    if proposal is not None and not any(e.kind == "forbid" and e.slot == proposal.slot for e in covering):
        return proposal.slot, "proposal"
    return None, "none"


def segments(first_frame: int, last_frame: int, proposal: Proposal | None, edits: list[Edit]) -> list[tuple[int, int, str | None, str]]:
    """The tracklet split into runs of frames with one effective slot: (first, last, slot, origin).

    Edits with a frame range can change the slot part way through a tracklet.
    """
    cuts = {first_frame, last_frame + 1}
    for e in edits:
        if e.first_frame is not None:
            cuts.update((max(first_frame, e.first_frame), min(last_frame, e.last_frame or e.first_frame) + 1))
    bounds = sorted(c for c in cuts if first_frame <= c <= last_frame + 1)
    out: list[tuple[int, int, str | None, str]] = []
    for a, b in zip(bounds, bounds[1:]):
        slot, origin = effective_slot(proposal, edits, a)
        if out and out[-1][2:] == (slot, origin):
            out[-1] = (out[-1][0], b - 1, slot, origin)
        else:
            out.append((a, b - 1, slot, origin))
    return out

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""review.py — storage of dot-assignment proposals and the human edits made over them.

A proposal says which slot a dot tracklet probably is (see autolabel.py); an edit is a
human decision. :func:`effective_slot` combines them: the latest assign edit covering a
frame wins, then a reject, then the proposal unless a forbid edit rules it out.
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class Edit:
    camera_id: str
    tracklet_id: int
    kind: str                      # assign | forbid | reject
    slot: str | None
    first_frame: int | None = None
    last_frame: int | None = None
    created_at: str = ""

    def covers(self, frame: int) -> bool:
        return self.first_frame is None or self.first_frame <= frame <= (self.last_frame or self.first_frame)


@dataclass(frozen=True)
class Proposal:
    camera_id: str
    tracklet_id: int
    first_frame: int
    last_frame: int
    slot: str
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
             subject_id: str | None = None) -> None:
    conn.execute(
        "INSERT INTO dot_assignment_edits (id, trial_id, detection_run_id, subject_id, camera_instance_id,"
        " tracklet_id, first_frame, last_frame, kind, slot_name) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), trial_id, detection_run_id, subject_id, edit.camera_id, edit.tracklet_id,
         edit.first_frame, edit.last_frame, edit.kind, edit.slot))
    conn.commit()


def load_edits(conn: sqlite3.Connection, trial_id: str, detection_run_id: str) -> list[Edit]:
    """All edits of the trial for the run, oldest first."""
    return [Edit(*r) for r in conn.execute(
        "SELECT camera_instance_id, tracklet_id, kind, slot_name, first_frame, last_frame, created_at"
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
            return None, "edit"
    if proposal is not None and not any(e.kind == "forbid" and e.slot == proposal.slot for e in covering):
        return proposal.slot, "proposal"
    return None, "none"

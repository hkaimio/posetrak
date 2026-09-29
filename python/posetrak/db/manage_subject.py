# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Capture-scoped subject identity: persons and objects share one registry
(``capture_subjects``) whose permanent integer ``label`` is the subject's
segmentation-mask label.

See docs/roadmap/features/subject-identity/subject-identity-design.md.

The registry rows are created, renamed and marked deleted by SQL triggers on
``capture_persons``/``capture_objects``, so this module only reads them and
maintains ``seg_quality_run_subjects`` -- the *exceptions* table for
segmentations whose labels are not the subjects' registry labels. A subject
with no row in a run means "the run uses this subject's registry label".
"""

from __future__ import annotations

import sqlite3


def list_subjects(
    session: sqlite3.Connection, capture_id: str, *, include_deleted: bool = False
) -> list[sqlite3.Row]:
    """Return *capture_id*'s subjects (persons and objects), ordered by label."""
    where = "" if include_deleted else " AND deleted = 0"
    return session.execute(
        f"SELECT * FROM capture_subjects WHERE capture_id = ?{where} ORDER BY label",
        (capture_id,),
    ).fetchall()


def get_subject(session: sqlite3.Connection, subject_id: str) -> sqlite3.Row | None:
    return session.execute(
        "SELECT * FROM capture_subjects WHERE id = ?", (subject_id,)
    ).fetchone()


def get_subject_by_name(
    session: sqlite3.Connection, capture_id: str, name: str
) -> sqlite3.Row | None:
    """The live subject named *name* in *capture_id* (exact match), or None."""
    return session.execute(
        "SELECT * FROM capture_subjects WHERE capture_id = ? AND name = ? AND deleted = 0",
        (capture_id, name),
    ).fetchone()


def _run_capture_id(session: sqlite3.Connection, seg_quality_run_id: str) -> str:
    row = session.execute(
        "SELECT shot_id FROM seg_quality_runs WHERE id = ?", (seg_quality_run_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"seg_quality_runs row not found: {seg_quality_run_id!r}")
    return row["shot_id"]


def label_for_subject_in_run(
    session: sqlite3.Connection, seg_quality_run_id: str, subject_id: str
) -> int | None:
    """The mask label *subject_id* has in segmentation *seg_quality_run_id*.

    An exception row for the subject wins. Otherwise the subject's registry
    label, unless an exception row gives that same label to a different
    subject (then this subject has no label in the run: None).
    """
    row = session.execute(
        "SELECT label FROM seg_quality_run_subjects "
        "WHERE seg_quality_run_id = ? AND subject_id = ? ORDER BY label LIMIT 1",
        (seg_quality_run_id, subject_id),
    ).fetchone()
    if row is not None:
        return row["label"]
    subject = get_subject(session, subject_id)
    if subject is None:
        return None
    taken = session.execute(
        "SELECT 1 FROM seg_quality_run_subjects "
        "WHERE seg_quality_run_id = ? AND label = ?",
        (seg_quality_run_id, subject["label"]),
    ).fetchone()
    return None if taken is not None else subject["label"]


def subject_for_label_in_run(
    session: sqlite3.Connection, seg_quality_run_id: str, label: int
) -> sqlite3.Row | None:
    """The subject a mask *label* means in segmentation *seg_quality_run_id*:
    the exception row's subject if there is one (None if it was explicitly
    left unassigned), else the capture's live subject with that registry
    label, else None."""
    row = session.execute(
        "SELECT subject_id FROM seg_quality_run_subjects "
        "WHERE seg_quality_run_id = ? AND label = ?",
        (seg_quality_run_id, label),
    ).fetchone()
    if row is not None:
        return get_subject(session, row["subject_id"]) if row["subject_id"] else None
    capture_id = _run_capture_id(session, seg_quality_run_id)
    # A subject with an exception row elsewhere in this run has been moved
    # off its registry label, so identity must not claim it here.
    return session.execute(
        "SELECT * FROM capture_subjects WHERE capture_id = ? AND label = ? AND deleted = 0 "
        "AND id NOT IN (SELECT subject_id FROM seg_quality_run_subjects "
        "               WHERE seg_quality_run_id = ? AND subject_id IS NOT NULL)",
        (capture_id, label, seg_quality_run_id),
    ).fetchone()


def set_run_label_subject(
    session: sqlite3.Connection,
    seg_quality_run_id: str,
    label: int,
    subject_id: str | None,
) -> None:
    """Record that mask *label* in segmentation *seg_quality_run_id* means
    *subject_id* (None: explicitly unassigned).

    Assigning a subject that already has a different label in this run moves
    it: its old exception row is removed so it never has two labels. When the
    assignment equals the subject's registry label it is stored as no row at
    all (absent means identity).
    """
    capture_id = _run_capture_id(session, seg_quality_run_id)
    subject = None
    if subject_id is not None:
        subject = get_subject(session, subject_id)
        if subject is None or subject["capture_id"] != capture_id:
            raise ValueError(f"subject {subject_id!r} is not in this segmentation's capture")
    with session:
        if subject_id is not None:
            session.execute(
                "DELETE FROM seg_quality_run_subjects "
                "WHERE seg_quality_run_id = ? AND subject_id = ?",
                (seg_quality_run_id, subject_id),
            )
        session.execute(
            "DELETE FROM seg_quality_run_subjects WHERE seg_quality_run_id = ? AND label = ?",
            (seg_quality_run_id, label),
        )
        if subject is None or subject["label"] != label:
            session.execute(
                "INSERT INTO seg_quality_run_subjects (seg_quality_run_id, label, subject_id) "
                "VALUES (?, ?, ?)",
                (seg_quality_run_id, label, subject_id),
            )


def subject_id_for_sequence(
    session: sqlite3.Connection, sequence_id: str, person_id: int = 0
) -> str | None:
    """The capture subject a pose-observation sequence's *person_id* belongs to.

    A person sequence answers from ``sequence_persons.subject_id``. An object
    sequence has no ``sequence_persons`` row (the session tree relies on
    that), so it answers from its detection run's ``capture_object_id`` -- an
    object's registry id is its ``capture_objects.id``. None if neither is
    known.
    """
    row = session.execute(
        "SELECT subject_id FROM sequence_persons WHERE sequence_id = ? AND person_id = ?",
        (sequence_id, person_id),
    ).fetchone()
    if row is not None and row["subject_id"]:
        return row["subject_id"]
    row = session.execute(
        "SELECT dr.capture_object_id FROM pose_observation_sequences pos "
        "JOIN detection_runs dr ON dr.id = pos.detection_run_id "
        "WHERE pos.id = ?",
        (sequence_id,),
    ).fetchone()
    return row["capture_object_id"] if row is not None else None


def seg_mask_label_for_sequence(
    session: sqlite3.Connection, seg_quality_run_id: str, sequence_id: str, person_id: int = 0
) -> int | None:
    """The mask label of the subject tracked in *sequence_id* within
    segmentation *seg_quality_run_id*, or None if the sequence's subject is
    unknown or has no label in that segmentation. This is what the tracker
    launcher passes as ``--seg-mask-label``."""
    subject_id = subject_id_for_sequence(session, sequence_id, person_id)
    if subject_id is None:
        return None
    return label_for_subject_in_run(session, seg_quality_run_id, subject_id)

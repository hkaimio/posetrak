# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Append-only revisions of segmentation masks.

A segmentation run's masks are stored per revision (``seg_masks.revision``); a revision is closed
(``seg_mask_revisions.closed_at`` set) by one explicit save, and only closed revisions are visible
to readers. Jobs that must be reproducible resolve :func:`newest_closed_revision` once at their
start, record it, and read every frame through :func:`mask_at` / :func:`masks_at`. Readers that want
the masks as they are now read the ``seg_masks_latest`` view. See
docs/roadmap/features/segmentation-versioning/segmentation-versioning-design.md.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone


class OpenRevisionError(RuntimeError):
    """Another editor already has a revision of the run open."""

    def __init__(self, run_id: str, revision: int, created_at: str):
        super().__init__(f"segmentation {run_id} has revision {revision} open since {created_at}")
        self.run_id = run_id
        self.revision = revision
        self.created_at = created_at


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def newest_closed_revision(conn: sqlite3.Connection, run_id: str) -> int | None:
    """Highest closed revision of *run_id*; None when the run has no revision rows."""
    row = conn.execute(
        "SELECT MAX(revision) FROM seg_mask_revisions WHERE seg_quality_run_id = ? AND closed_at IS NOT NULL",
        (run_id,),
    ).fetchone()
    return row[0]


def mask_at(conn: sqlite3.Connection, run_id: str, video_id: str, frame_idx: int, revision: int) -> bytes | None:
    """PNG bytes of a frame's mask at *revision*; None when the frame has no mask there."""
    row = conn.execute(
        "SELECT mask_blob FROM seg_masks WHERE seg_quality_run_id = ? AND shot_video_id = ? AND frame_idx = ?"
        " AND revision <= ? ORDER BY revision DESC LIMIT 1",
        (run_id, video_id, frame_idx, revision),
    ).fetchone()
    return None if row is None or row[0] is None else bytes(row[0])


def masks_at(
    conn: sqlite3.Connection, run_id: str, revision: int, video_id: str | None = None
) -> Iterator[tuple[str, int, bytes]]:
    """Yield ``(shot_video_id, frame_idx, mask_blob)`` of every masked frame at *revision*."""
    sql = (
        "SELECT m.shot_video_id, m.frame_idx, m.mask_blob FROM seg_masks m"
        " WHERE m.seg_quality_run_id = ? AND m.mask_blob IS NOT NULL"
        "   AND (? IS NULL OR m.shot_video_id = ?)"
        "   AND m.revision = (SELECT MAX(m2.revision) FROM seg_masks m2"
        "     WHERE m2.seg_quality_run_id = m.seg_quality_run_id AND m2.shot_video_id = m.shot_video_id"
        "       AND m2.frame_idx = m.frame_idx AND m2.revision <= ?)"
        " ORDER BY m.shot_video_id, m.frame_idx"
    )
    for vid, frame, blob in conn.execute(sql, (run_id, video_id, video_id, revision)):
        yield vid, frame, bytes(blob)


def get_open_revision(conn: sqlite3.Connection, run_id: str) -> tuple[int, str] | None:
    """``(revision, created_at)`` of the run's open revision, or None."""
    row = conn.execute(
        "SELECT revision, created_at FROM seg_mask_revisions WHERE seg_quality_run_id = ? AND closed_at IS NULL",
        (run_id,),
    ).fetchone()
    return None if row is None else (row[0], row[1])


def open_revision(conn: sqlite3.Connection, run_id: str, note: str | None = None) -> int:
    """Open the next revision of *run_id* for editing and commit.

    Raises :class:`OpenRevisionError` when one is already open (the partial unique index decides,
    so two processes racing cannot both succeed).
    """
    try:
        cur = conn.execute(
            "INSERT INTO seg_mask_revisions (seg_quality_run_id, revision, created_at, note)"
            " SELECT ?, COALESCE(MAX(revision), -1) + 1, ?, ? FROM seg_mask_revisions WHERE seg_quality_run_id = ?",
            (run_id, _now(), note, run_id),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        existing = get_open_revision(conn, run_id)
        if existing is None:
            raise
        raise OpenRevisionError(run_id, *existing) from None
    return conn.execute(
        "SELECT revision FROM seg_mask_revisions WHERE rowid = ?", (cur.lastrowid,)
    ).fetchone()[0]


def write_mask(
    conn: sqlite3.Connection, run_id: str, video_id: str, frame_idx: int, revision: int, blob: bytes | None
) -> None:
    """Write a frame's mask into the open *revision* (NULL *blob* records a deletion). Does not commit."""
    conn.execute(
        "INSERT OR REPLACE INTO seg_masks (seg_quality_run_id, shot_video_id, frame_idx, revision, mask_blob)"
        " VALUES (?, ?, ?, ?, ?)",
        (run_id, video_id, frame_idx, revision, blob),
    )


def close_revision(conn: sqlite3.Connection, run_id: str, revision: int) -> None:
    """Save *revision*: one write sets ``closed_at``, then commit. Readers see it whole or not at all."""
    conn.execute(
        "UPDATE seg_mask_revisions SET closed_at = ? WHERE seg_quality_run_id = ? AND revision = ? AND closed_at IS NULL",
        (_now(), run_id, revision),
    )
    conn.commit()


def discard_revision(conn: sqlite3.Connection, run_id: str, revision: int) -> None:
    """Delete an open *revision* and its rows."""
    conn.execute(
        "DELETE FROM seg_masks WHERE seg_quality_run_id = ? AND revision = ?"
        " AND revision IN (SELECT revision FROM seg_mask_revisions"
        "   WHERE seg_quality_run_id = ? AND revision = ? AND closed_at IS NULL)",
        (run_id, revision, run_id, revision),
    )
    conn.execute(
        "DELETE FROM seg_mask_revisions WHERE seg_quality_run_id = ? AND revision = ? AND closed_at IS NULL",
        (run_id, revision),
    )
    conn.commit()


def delete_run(conn: sqlite3.Connection, run_id: str) -> None:
    """Delete all masks and revisions of a segmentation run. Does not commit."""
    conn.execute("DELETE FROM seg_masks WHERE seg_quality_run_id = ?", (run_id,))
    conn.execute("DELETE FROM seg_mask_revisions WHERE seg_quality_run_id = ?", (run_id,))

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

import os
import socket
import sqlite3
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import NamedTuple


class OpenRevision(NamedTuple):
    """A revision that is open for editing."""

    revision: int
    created_at: str
    note: str | None  # who opened it, see :func:`owner_note`


class OpenRevisionError(RuntimeError):
    """Another editor already has a revision of the run open."""

    def __init__(self, run_id: str, open_revision: OpenRevision):
        super().__init__(f"segmentation {run_id} has revision {open_revision.revision} open since {open_revision.created_at}")
        self.run_id = run_id
        self.open_revision = open_revision


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


def get_open_revision(conn: sqlite3.Connection, run_id: str) -> OpenRevision | None:
    """The run's open revision, or None."""
    row = conn.execute(
        "SELECT revision, created_at, note FROM seg_mask_revisions WHERE seg_quality_run_id = ? AND closed_at IS NULL",
        (run_id,),
    ).fetchone()
    return None if row is None else OpenRevision(row[0], row[1], row[2])


def owner_note() -> str:
    """Note for a revision opened by this process, so a later opener can tell a live editor from a crashed one."""
    return f"host={socket.gethostname()} pid={os.getpid()}"


def owner_is_alive(note: str | None) -> bool:
    """False only when *note* names a process of this host that is gone (a crashed editor).

    Anything else -- another host, a note not made by :func:`owner_note` -- counts as alive: the safe
    answer is that someone may still be editing.
    """
    try:
        fields = dict(part.split("=", 1) for part in (note or "").split())
        host, pid = fields["host"], int(fields["pid"])
    except (KeyError, ValueError):
        return True
    if host != socket.gethostname():
        return True
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
        return bool(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def masked_frames(conn: sqlite3.Connection, run_id: str, video_id: str, revision: int) -> list[int]:
    """Frame numbers that have a mask at *revision* for one camera view, ascending."""
    rows = conn.execute(
        "SELECT m.frame_idx FROM seg_masks m WHERE m.seg_quality_run_id = ? AND m.shot_video_id = ?"
        " AND m.mask_blob IS NOT NULL AND m.revision = (SELECT MAX(m2.revision) FROM seg_masks m2"
        "   WHERE m2.seg_quality_run_id = m.seg_quality_run_id AND m2.shot_video_id = m.shot_video_id"
        "     AND m2.frame_idx = m.frame_idx AND m2.revision <= ?)"
        " ORDER BY m.frame_idx",
        (run_id, video_id, revision),
    )
    return [r[0] for r in rows]


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
        raise OpenRevisionError(run_id, existing) from None
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
    """Delete all masks and revisions of a segmentation run, and forget it in the results that read it.

    Detection runs and tracking run subjects keep their data but lose the reference to the run (the
    run is about to go; the foreign keys would refuse otherwise). Does not commit and does not delete
    the ``seg_quality_runs`` row.
    """
    conn.execute("UPDATE detection_runs SET seg_quality_run_id = NULL, seg_revision = NULL WHERE seg_quality_run_id = ?", (run_id,))
    conn.execute(
        "UPDATE tracking_run_persons SET seg_quality_run_id = NULL, seg_mask_label = NULL, seg_revision = NULL"
        " WHERE seg_quality_run_id = ?",
        (run_id,),
    )
    conn.execute("DELETE FROM seg_masks WHERE seg_quality_run_id = ?", (run_id,))
    conn.execute("DELETE FROM seg_mask_revisions WHERE seg_quality_run_id = ?", (run_id,))


class PruneResult(NamedTuple):
    """What :func:`prune` removed (or would remove) from one segmentation run."""

    seg_quality_run_id: str
    revisions_removed: list[int]
    rows_removed: int  # mask rows deleted, because a later revision of the same frame covers them
    rows_moved: int  # mask rows carried to the next kept revision, so the kept revisions read unchanged


def referenced_revisions(conn: sqlite3.Connection, run_id: str) -> set[int]:
    """Revisions of *run_id* that a detection run or a tracking run subject recorded, whatever the run's status."""
    rows = conn.execute(
        "SELECT seg_revision FROM detection_runs WHERE seg_quality_run_id = ? AND seg_revision IS NOT NULL"
        " UNION SELECT seg_revision FROM tracking_run_persons WHERE seg_quality_run_id = ? AND seg_revision IS NOT NULL",
        (run_id, run_id),
    )
    return {r[0] for r in rows}


def prune(conn: sqlite3.Connection, run_id: str, *, dry_run: bool = False) -> PruneResult:
    """Remove the revisions of *run_id* that nothing references, keeping the newest closed one.

    A removed revision's rows that are the visible mask of a frame for a later kept revision move to the
    next kept revision (no kept revision reads differently afterwards); rows shadowed by a later revision
    of the same frame are deleted. An open revision is never touched. Commits unless *dry_run*.
    """
    closed = [r[0] for r in conn.execute(
        "SELECT revision FROM seg_mask_revisions WHERE seg_quality_run_id = ? AND closed_at IS NOT NULL ORDER BY revision",
        (run_id,))]
    if not closed:
        return PruneResult(run_id, [], 0, 0)
    keep = referenced_revisions(conn, run_id) | {closed[-1]}
    removed, n_deleted, n_moved = [], 0, 0
    for rev in closed:
        if rev in keep:
            continue
        target = min(k for k in keep if k > rev)       # the newest closed revision is kept and above every other
        # Frames of rev that no row in (rev, target] covers are the visible mask for target and above.
        uncovered = (
            "FROM seg_masks m WHERE m.seg_quality_run_id = ? AND m.revision = ? AND NOT EXISTS ("
            "SELECT 1 FROM seg_masks n WHERE n.seg_quality_run_id = m.seg_quality_run_id"
            " AND n.shot_video_id = m.shot_video_id AND n.frame_idx = m.frame_idx"
            " AND n.revision > m.revision AND n.revision <= ?)"
        )
        moved = conn.execute(f"SELECT COUNT(*) {uncovered}", (run_id, rev, target)).fetchone()[0]
        total = conn.execute(
            "SELECT COUNT(*) FROM seg_masks WHERE seg_quality_run_id = ? AND revision = ?", (run_id, rev)).fetchone()[0]
        n_moved += moved
        n_deleted += total - moved
        removed.append(rev)
        if dry_run:
            continue
        conn.execute(
            "UPDATE seg_masks SET revision = ? WHERE seg_quality_run_id = ? AND revision = ?"
            " AND (shot_video_id, frame_idx) IN (SELECT m.shot_video_id, m.frame_idx " + uncovered + ")",
            (target, run_id, rev, run_id, rev, target),
        )
        conn.execute("DELETE FROM seg_masks WHERE seg_quality_run_id = ? AND revision = ?", (run_id, rev))
        conn.execute("DELETE FROM seg_mask_revisions WHERE seg_quality_run_id = ? AND revision = ?", (run_id, rev))
    if not dry_run:
        conn.commit()
    return PruneResult(run_id, removed, n_deleted, n_moved)

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Segmentation mask revisions: migration to schema 68, the read helper, the latest view."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from posetrak.db import seg_revisions as sr
from posetrak.db.db import create_session, get_schema_version, open_session

RUN, VID = "run-1", "vid-1"


def _add_run(conn: sqlite3.Connection, run_id: str = RUN) -> None:
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at)"
        " VALUES (?, 'cap', 0, 1, '2026-01-01T00:00:00+00:00')",
        (run_id,),
    )
    conn.commit()


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = create_session(tmp_path / "s.db")
    _add_run(c)
    return c


def test_new_run_has_closed_revision_zero(conn: sqlite3.Connection) -> None:
    assert sr.newest_closed_revision(conn, RUN) == 0
    assert sr.get_open_revision(conn, RUN) is None


def test_edit_adds_one_row_and_readers_pin_a_revision(conn: sqlite3.Connection) -> None:
    for f in (1, 2):
        conn.execute("INSERT INTO seg_masks VALUES (?, ?, ?, 0, ?)", (RUN, VID, f, b"a%d" % f))
    conn.commit()
    before = conn.execute("SELECT * FROM seg_masks ORDER BY frame_idx").fetchall()

    rev = sr.open_revision(conn, RUN)
    assert rev == 1
    sr.write_mask(conn, RUN, VID, 2, rev, b"new")
    sr.write_mask(conn, RUN, VID, 1, rev, None)  # deletion
    conn.commit()

    # Open revision is invisible to the latest view and to the newest-closed resolution.
    assert sr.newest_closed_revision(conn, RUN) == 0
    assert {r["frame_idx"]: bytes(r["mask_blob"]) for r in conn.execute("SELECT * FROM seg_masks_latest")} == {
        1: b"a1",
        2: b"a2",
    }
    sr.close_revision(conn, RUN, rev)

    assert {r["frame_idx"]: bytes(r["mask_blob"]) for r in conn.execute("SELECT * FROM seg_masks_latest")} == {
        2: b"new"
    }
    assert sr.mask_at(conn, RUN, VID, 1, 0) == b"a1"
    assert sr.mask_at(conn, RUN, VID, 1, 1) is None
    assert sr.mask_at(conn, RUN, VID, 2, 0) == b"a2"
    assert [(f, b) for _, f, b in sr.masks_at(conn, RUN, 0)] == [(1, b"a1"), (2, b"a2")]
    assert [(f, b) for _, f, b in sr.masks_at(conn, RUN, 1)] == [(2, b"new")]
    # The old revision's rows did not change: two new rows only.
    after = conn.execute("SELECT * FROM seg_masks WHERE revision = 0 ORDER BY frame_idx").fetchall()
    assert [tuple(r) for r in after] == [tuple(r) for r in before]
    assert conn.execute("SELECT COUNT(*) FROM seg_masks").fetchone()[0] == len(before) + 2


def test_second_editor_is_refused_and_discard_frees_the_run(conn: sqlite3.Connection) -> None:
    rev = sr.open_revision(conn, RUN)
    with pytest.raises(sr.OpenRevisionError) as err:
        sr.open_revision(conn, RUN)
    assert err.value.revision == rev
    sr.write_mask(conn, RUN, VID, 5, rev, b"x")
    sr.discard_revision(conn, RUN, rev)
    assert conn.execute("SELECT COUNT(*) FROM seg_masks").fetchone()[0] == 0
    assert sr.open_revision(conn, RUN) == 1  # the number is free again


def test_discard_never_touches_a_closed_revision(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO seg_masks VALUES (?, ?, 1, 0, ?)", (RUN, VID, b"a"))
    conn.commit()
    sr.discard_revision(conn, RUN, 0)
    assert sr.mask_at(conn, RUN, VID, 1, 0) == b"a"


def test_migration_from_67_keeps_every_blob(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    c = create_session(path)
    c.executescript(
        "DROP TRIGGER trg_seg_quality_runs_revision0; DROP VIEW seg_masks_latest;"
        " DROP TABLE seg_masks; DROP TABLE seg_mask_revisions;"
        " CREATE TABLE seg_masks (seg_quality_run_id TEXT NOT NULL REFERENCES seg_quality_runs(id),"
        " shot_video_id TEXT NOT NULL, frame_idx INTEGER NOT NULL, mask_blob BLOB NOT NULL,"
        " PRIMARY KEY (seg_quality_run_id, shot_video_id, frame_idx));"
        " PRAGMA user_version = 67;"
    )
    _add_run(c, "a")
    _add_run(c, "b")
    c.executemany("INSERT INTO seg_masks VALUES (?, ?, ?, ?)", [("a", VID, i, b"m%d" % i) for i in range(5)] + [("b", VID, 0, b"z")])
    c.commit()
    digest = lambda cx, run: hashlib.sha256(  # noqa: E731
        b"".join(bytes(r[0]) for r in cx.execute("SELECT mask_blob FROM seg_masks WHERE seg_quality_run_id = ? ORDER BY frame_idx", (run,)))
    ).hexdigest()
    want = {run: digest(c, run) for run in "ab"}
    c.close()

    m = open_session(path)
    assert get_schema_version(m) == 68
    assert {run: digest(m, run) for run in "ab"} == want
    assert m.execute("SELECT COUNT(*) FROM seg_masks WHERE revision = 0").fetchone()[0] == 6
    assert m.execute("SELECT COUNT(*) FROM seg_mask_revisions WHERE revision = 0 AND closed_at IS NOT NULL").fetchone()[0] == 2
    assert m.execute("SELECT COUNT(*) FROM seg_masks_latest").fetchone()[0] == 6
    m.execute("PRAGMA foreign_keys = OFF")
    m.execute("UPDATE seg_masks SET mask_blob = NULL WHERE frame_idx = 0")  # nullable now

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""The segmentation panel writes its masks as revisions of the segmentation run it edits."""
from __future__ import annotations

import pytest

from posetrak.db import seg_revisions as sr

from .test_cutie_init_panel import capture_db  # noqa: F401  (fixture)


@pytest.fixture()
def seg_db(capture_db):  # noqa: F811
    capture_db.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at) "
        "VALUES ('seg1', 'cap1', 0.0, 1e9, '2026-01-01T00:00:00+00:00')"
    )
    capture_db.executemany(
        "INSERT INTO seg_masks (seg_quality_run_id, shot_video_id, frame_idx, mask_blob) VALUES ('seg1', 'sv1', ?, ?)",
        [(1, b"one"), (2, b"two")],
    )
    capture_db.commit()
    return capture_db


def test_a_flush_writes_changed_frames_into_one_revision_closed_when_the_queue_ends(qapp, seg_db):
    from app.pose.cutie_init_panel import CutieInitPanel

    panel = CutieInitPanel(seg_db, "cap1", seg_init_run_id="seg1")
    try:
        assert panel._open_rev is None
        panel._db_flush_buffer.extend([("sv1", 1, b"one"), ("sv1", 2, b"TWO"), ("sv1", 3, b"three")])
        panel._flush_masks()
        assert panel._open_rev == 1
        # Frame 1 is unchanged and gets no row; nobody else sees revision 1 yet.
        rows = seg_db.execute("SELECT frame_idx FROM seg_masks WHERE revision = 1 ORDER BY 1").fetchall()
        assert [r[0] for r in rows] == [2, 3]
        assert sr.newest_closed_revision(seg_db, "seg1") == 0
        assert seg_db.execute("SELECT COUNT(*) FROM seg_masks_latest WHERE frame_idx = 3").fetchone()[0] == 0
        # The panel itself reads its open revision.
        assert panel._mask_revision("seg1") == 1

        panel._db_flush_buffer.append(("sv1", 3, b"three-b"))
        panel._flush_masks()                       # same revision, the row of frame 3 replaced
        assert seg_db.execute("SELECT COUNT(*) FROM seg_masks WHERE revision = 1").fetchone()[0] == 2

        panel._on_queue_done()                     # job queue ended: saved
        assert panel._open_rev is None and sr.newest_closed_revision(seg_db, "seg1") == 1
        assert sr.mask_at(seg_db, "seg1", "sv1", 3, 1) == b"three-b"
        assert sr.mask_at(seg_db, "seg1", "sv1", 2, 0) == b"two"       # revision 0 untouched
    finally:
        panel.shutdown()


def test_closing_the_panel_saves_the_open_revision(qapp, seg_db):
    from app.pose.cutie_init_panel import CutieInitPanel

    panel = CutieInitPanel(seg_db, "cap1", seg_init_run_id="seg1")
    panel._db_flush_buffer.append(("sv1", 7, b"seven"))
    panel.shutdown()
    assert sr.newest_closed_revision(seg_db, "seg1") == 1
    assert sr.mask_at(seg_db, "seg1", "sv1", 7, 1) == b"seven"


def test_a_second_panel_opens_the_run_read_only(qapp, seg_db):
    from app.pose.cutie_init_panel import CutieInitPanel

    first = CutieInitPanel(seg_db, "cap1", seg_init_run_id="seg1")
    first._db_flush_buffer.append(("sv1", 7, b"seven"))
    first._flush_masks()
    second = CutieInitPanel(seg_db, "cap1", seg_init_run_id="seg1")
    try:
        assert second._read_only and "revision 1" in second._read_only_reason
        assert not second._track_fwd_btn.isEnabled()
        second._db_flush_buffer.append(("sv1", 8, b"eight"))
        second._flush_masks()
        assert seg_db.execute("SELECT COUNT(*) FROM seg_masks WHERE frame_idx = 8").fetchone()[0] == 0
    finally:
        second.shutdown()
        first.shutdown()
    assert sr.newest_closed_revision(seg_db, "seg1") == 1


@pytest.mark.parametrize("choice, saved", [("Save", True), ("Discard", False)])
def test_a_revision_left_by_a_crash_is_offered_for_saving_or_discarding(qapp, seg_db, monkeypatch, choice, saved):
    from PySide6.QtWidgets import QMessageBox

    from app.pose.cutie_init_panel import CutieInitPanel

    rev = sr.open_revision(seg_db, "seg1", note="host=gone pid=1")
    sr.write_mask(seg_db, "seg1", "sv1", 9, rev, b"lost?")
    seg_db.commit()
    monkeypatch.setattr("app.pose.cutie_init_panel.owner_is_alive", lambda note: False)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda self: next(b for b in self.buttons() if b.text() == choice))

    panel = CutieInitPanel(seg_db, "cap1", seg_init_run_id="seg1")
    try:
        assert not panel._read_only
        assert sr.get_open_revision(seg_db, "seg1") is None
        assert (sr.mask_at(seg_db, "seg1", "sv1", 9, 1) == b"lost?") is saved
    finally:
        panel.shutdown()

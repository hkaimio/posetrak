"""Tests for app.setup.page_shots (ShotsPage wizard page)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.setup.db_context import DBContext
from app.setup.page_shots import ShotEntry, ShotsPage, VideoEntry
from app.setup.video_probe import VideoProbeResult
from posetrak.db.db import create_session, generate_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_db(tmp_path: Path) -> tuple[sqlite3.Connection, str]:
    db_path = tmp_path / "session.db"
    conn = create_session(db_path)
    conn.row_factory = sqlite3.Row
    session_id = generate_id()
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES (?, ?)",
        (session_id, "2026-03-01T10:00:00+00:00"),
    )
    conn.commit()
    return conn, session_id


def _make_wizard_mock(page: ShotsPage, conn, session_id: str):
    ctx = DBContext(conn, session_id)
    wiz = MagicMock()
    wiz.db_context = ctx
    page.wizard = MagicMock(return_value=wiz)
    return wiz, ctx


def _fake_probe(width=1920, height=1080, fps=30.0, frames=300) -> VideoProbeResult:
    return VideoProbeResult(
        width=width, height=height,
        container_fps=fps, frame_count=frames,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_page_title(qapp) -> None:
    page = ShotsPage()
    assert page.title() == "Shot & Videos"


def test_initialize_adds_one_shot(qapp, tmp_path) -> None:
    conn, session_id = _make_session_db(tmp_path)
    page = ShotsPage()
    _make_wizard_mock(page, conn, session_id)
    page.initializePage()
    assert len(page._shots) == 1
    conn.close()


# ---------------------------------------------------------------------------
# validatePage — writes to DB
# ---------------------------------------------------------------------------


def test_validate_writes_shot(qapp, tmp_path) -> None:
    conn, session_id = _make_session_db(tmp_path)
    page = ShotsPage()
    _make_wizard_mock(page, conn, session_id)

    # Manually configure shot entry with a probed video
    page._shots.clear()
    entry = ShotEntry(label="walk")
    ve = VideoEntry(path="/fake/cam1.mp4", probe=_fake_probe(frames=600))
    entry.videos.append(ve)
    page._shots.append(entry)

    # initializePage to begin the savepoint
    page.initializePage()
    result = page.validatePage()

    assert result is True

    shot = conn.execute("SELECT * FROM captures WHERE capture_number = 1").fetchone()
    assert shot is not None
    assert shot["label"] == "walk"

    video = conn.execute(
        "SELECT * FROM capture_videos WHERE shot_id = ?", (shot["id"],)
    ).fetchone()
    assert video is not None
    assert video["file_path"] == "/fake/cam1.mp4"
    assert video["last_video_frame"] == 599   # frame_count - 1
    conn.close()


def test_validate_writes_multiple_shots(qapp, tmp_path) -> None:
    conn, session_id = _make_session_db(tmp_path)
    page = ShotsPage()
    _make_wizard_mock(page, conn, session_id)

    page._shots.clear()
    for i in range(3):
        entry = ShotEntry(label=f"shot{i+1}")
        page._shots.append(entry)

    page.initializePage()
    result = page.validatePage()

    assert result is True
    rows = conn.execute(
        "SELECT capture_number FROM captures ORDER BY capture_number"
    ).fetchall()
    assert [r["capture_number"] for r in rows] == [1, 2, 3]
    conn.close()


def test_validate_fails_with_no_shots(qapp, tmp_path) -> None:
    conn, session_id = _make_session_db(tmp_path)
    page = ShotsPage()
    _make_wizard_mock(page, conn, session_id)

    page.initializePage()
    page._shots.clear()   # remove the auto-added shot to test error path
    result = page.validatePage()

    assert result is False
    assert not page._error_label.isHidden()
    conn.close()


def test_cleanup_rolls_back(qapp, tmp_path) -> None:
    """cleanupPage should undo any partial writes."""
    conn, session_id = _make_session_db(tmp_path)
    page = ShotsPage()
    _make_wizard_mock(page, conn, session_id)

    entry = ShotEntry()
    page._shots = [entry]
    page.initializePage()

    # Directly write a shot (simulating partial progress) then clean up
    ctx = DBContext(conn, session_id)
    ctx.begin_page()
    ctx.create_shot("partial")
    ctx.rollback_page()

    count = conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0]
    assert count == 0
    conn.close()


# ---------------------------------------------------------------------------
# VideoEntry / ShotEntry data models
# ---------------------------------------------------------------------------


def test_shot_entry_defaults(qapp) -> None:
    e = ShotEntry()
    assert e.label == ""
    assert e.videos == []


def test_video_entry_stores_path() -> None:
    ve = VideoEntry(path="/path/to/file.mp4")
    assert ve.path == "/path/to/file.mp4"
    assert ve.probe is None
    assert ve.error is None

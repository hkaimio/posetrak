"""Tests for app.setup.page_sync (SyncPage — D3a independent scrubber)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.setup.db_context import DBContext
from app.setup.page_sync import SyncPage, _ShotMeta
from posetrak.db.db import create_session, generate_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_with_shots(
    tmp_path: Path,
    n_shots: int = 2,
    videos_per_shot: int = 2,
) -> tuple[sqlite3.Connection, str]:
    """Create a session DB with *n_shots* shots, each with *videos_per_shot* videos."""
    db_path = tmp_path / "sync_session.db"
    conn = create_session(db_path)
    conn.row_factory = sqlite3.Row
    session_id = generate_id()
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES (?, ?)",
        (session_id, "2026-03-01T10:00:00+00:00"),
    )
    for shot_num in range(1, n_shots + 1):
        shot_id = generate_id()
        conn.execute(
            "INSERT INTO shots (id, session_id, shot_number, label) VALUES (?, ?, ?, ?)",
            (shot_id, session_id, shot_num, f"Shot {shot_num}"),
        )
        for cam_num in range(1, videos_per_shot + 1):
            video_id = generate_id()
            conn.execute(
                "INSERT INTO shot_videos "
                "(id, shot_id, camera_instance_id, file_path, "
                "first_video_frame, last_video_frame, actual_fps) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    video_id, shot_id, f"cam{cam_num}",
                    f"/fake/shot{shot_num}_cam{cam_num}.mp4",
                    0, 299, 30.0,
                ),
            )
    conn.commit()
    return conn, session_id


def _attach_wizard(page: SyncPage, conn, session_id: str):
    ctx = DBContext(conn, session_id)
    wiz = MagicMock()
    wiz.db_context = ctx
    page.wizard = MagicMock(return_value=wiz)
    return ctx


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_page_constructs(qapp) -> None:
    page = SyncPage()
    assert page.title() == "Camera Synchronisation"


def test_page_is_always_complete(qapp) -> None:
    page = SyncPage()
    assert page.isComplete()


# ---------------------------------------------------------------------------
# initializePage — shot loading
# ---------------------------------------------------------------------------


def test_initialize_populates_shot_combo(qapp, tmp_path) -> None:
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=3)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)

    if True:
        page.initializePage()

    assert page._shot_combo.count() == 3
    assert "Shot 1" in page._shot_combo.itemText(0)
    conn.close()


def test_initialize_builds_scrubber_for_first_shot(qapp, tmp_path) -> None:
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=1, videos_per_shot=2)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)

    if True:
        page.initializePage()

    assert page._scrubber is not None
    assert len(page._scrubber._cells) == 2
    conn.close()


def test_initialize_no_shots_shows_error(qapp, tmp_path) -> None:
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=0)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)

    page.initializePage()

    assert page._scrubber is None
    assert not page._error_label.isHidden()
    conn.close()


# ---------------------------------------------------------------------------
# Shot switching
# ---------------------------------------------------------------------------


def test_shot_switch_rebuilds_scrubber(qapp, tmp_path) -> None:
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=2, videos_per_shot=2)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)

    if True:
        page.initializePage()

    first_scrubber = page._scrubber

    if True:
        page._shot_combo.setCurrentIndex(1)

    assert page._scrubber is not first_scrubber
    conn.close()


def test_shot_switch_clears_old_cache(qapp, tmp_path) -> None:
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=2, videos_per_shot=1)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)

    if True:
        page.initializePage()
        first_cache = page._cache

    if True:
        page._shot_combo.setCurrentIndex(1)

    assert page._cache is not first_cache
    conn.close()


# ---------------------------------------------------------------------------
# Seek slider
# ---------------------------------------------------------------------------


def test_seek_slider_enabled_after_shot_load(qapp, tmp_path) -> None:
    conn, session_id = _make_session_with_shots(tmp_path)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)

    if True:
        page.initializePage()

    assert not page._seek_slider.isHidden()
    assert page._seek_slider.isEnabled()
    conn.close()


def test_seek_slider_max_equals_last_frame(qapp, tmp_path) -> None:
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=1, videos_per_shot=1)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)

    if True:
        page.initializePage()

    # first_video_frame=0, last_video_frame=299 → total=300 → max slider = 299
    assert page._seek_slider.maximum() == 299
    conn.close()


def test_slider_move_calls_seek_camera(qapp, tmp_path) -> None:
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=1, videos_per_shot=2)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)

    if True:
        page.initializePage()

    scrubber = page._scrubber
    scrubber.seek_camera = MagicMock()

    page._on_slider_moved(50)

    scrubber.seek_camera.assert_called_once_with(scrubber.focused_cell, 50)
    conn.close()


# ---------------------------------------------------------------------------
# Status strip
# ---------------------------------------------------------------------------


def test_status_shows_camera_labels(qapp, tmp_path) -> None:
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=1, videos_per_shot=2)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)

    if True:
        page.initializePage()

    text = page._status_label.text()
    assert "cam1" in text
    assert "cam2" in text
    conn.close()


def test_status_marks_focused_cell(qapp, tmp_path) -> None:
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=1, videos_per_shot=2)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)

    if True:
        page.initializePage()

    # Default focused cell is 0
    text = page._status_label.text()
    assert "◄" in text
    conn.close()


# ---------------------------------------------------------------------------
# cleanupPage — teardown
# ---------------------------------------------------------------------------


def test_cleanup_releases_scrubber_and_cache(qapp, tmp_path) -> None:
    conn, session_id = _make_session_with_shots(tmp_path)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)

    if True:
        page.initializePage()

    assert page._scrubber is not None
    assert page._cache is not None

    page.cleanupPage()

    assert page._scrubber is None
    assert page._cache is None
    assert not page._seek_slider.isEnabled()
    conn.close()


# ---------------------------------------------------------------------------
# Scrubber independence — no sync table on init
# ---------------------------------------------------------------------------


def test_scrubber_starts_without_sync_table(qapp, tmp_path) -> None:
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=1, videos_per_shot=2)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)

    if True:
        page.initializePage()

    assert page._scrubber is not None
    assert page._scrubber.sync_table is None
    conn.close()

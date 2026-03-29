"""Tests for app.setup.page_sync (SyncPage — D3a independent scrubber)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.setup.db_context import DBContext
from app.setup.page_sync import SyncPage, _ShotMeta
from posetrak.db.db import create_session, generate_id


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_session_with_shots(
    tmp_path: Path,
    n_shots: int = 2,
    videos_per_shot: int = 2,
) -> tuple[sqlite3.Connection, str]:
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


@pytest.fixture
def loaded_page(qapp, tmp_path):
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=2, videos_per_shot=2)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    yield page, conn
    page.cleanupPage()
    conn.close()


@pytest.fixture
def loaded_page_1shot(qapp, tmp_path):
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=1, videos_per_shot=2)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    yield page, conn
    page.cleanupPage()
    conn.close()


@pytest.fixture
def loaded_page_1video(qapp, tmp_path):
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=1, videos_per_shot=1)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    yield page, conn
    page.cleanupPage()
    conn.close()


# ---------------------------------------------------------------------------
# Construction (no DB needed)
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


def test_initialize_populates_shot_combo(loaded_page) -> None:
    page, _ = loaded_page
    assert page._shot_combo.count() == 2
    assert "Shot 1" in page._shot_combo.itemText(0)


def test_initialize_builds_scrubber_for_first_shot(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    assert page._scrubber is not None
    assert len(page._scrubber._cells) == 2


def test_initialize_no_shots_shows_error(qapp, tmp_path) -> None:
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=0)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()

    assert page._scrubber is None
    assert not page._error_label.isHidden()
    conn.close()


# ---------------------------------------------------------------------------
# Per-cell sliders in scrubber
# ---------------------------------------------------------------------------


def test_scrubber_has_per_cell_sliders(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    scrubber = page._scrubber
    assert len(scrubber._sliders) == 2


def test_per_cell_slider_max_equals_last_frame(loaded_page_1video) -> None:
    page, _ = loaded_page_1video
    # first_video_frame=0, last_video_frame=299 → total=300 → max=299
    assert page._scrubber._sliders[0].maximum() == 299


def test_per_cell_slider_updates_on_seek(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    scrubber = page._scrubber
    scrubber._set_cell_frame(0, 50)
    assert scrubber._sliders[0].value() == 50


def test_per_cell_frame_label_updates_on_seek(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    scrubber = page._scrubber
    scrubber._set_cell_frame(1, 42)
    assert "42" in scrubber._frame_labels[1].text()


def test_cell_slider_move_seeks_that_cell(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    scrubber = page._scrubber
    scrubber._on_cell_slider_moved(1, 75)
    assert scrubber.current_frames[1] == 75
    assert scrubber._sliders[1].value() == 75


def test_cell_slider_move_focuses_that_cell(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    scrubber = page._scrubber
    # Default focus is cell 0; drag cell 1's slider
    scrubber._on_cell_slider_moved(1, 10)
    assert scrubber.focused_cell == 1
    assert scrubber._cells[1]._selected is True
    assert scrubber._cells[0]._selected is False


# ---------------------------------------------------------------------------
# Shot switching
# ---------------------------------------------------------------------------


def test_shot_switch_rebuilds_scrubber(loaded_page) -> None:
    page, conn = loaded_page
    first_scrubber = page._scrubber
    page._shot_combo.setCurrentIndex(1)
    assert page._scrubber is not first_scrubber


def test_shot_switch_clears_old_cache(loaded_page) -> None:
    page, conn = loaded_page
    first_cache = page._cache
    page._shot_combo.setCurrentIndex(1)
    assert page._cache is not first_cache


# ---------------------------------------------------------------------------
# cleanupPage — teardown
# ---------------------------------------------------------------------------


def test_cleanup_releases_scrubber_and_cache(qapp, tmp_path) -> None:
    conn, session_id = _make_session_with_shots(tmp_path)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()

    assert page._scrubber is not None
    assert page._cache is not None

    page.cleanupPage()

    assert page._scrubber is None
    assert page._cache is None
    conn.close()


# ---------------------------------------------------------------------------
# Independent mode — no sync table on init
# ---------------------------------------------------------------------------


def test_scrubber_starts_without_sync_table(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    assert page._scrubber.sync_table is None


# ---------------------------------------------------------------------------
# Selected cell highlight
# ---------------------------------------------------------------------------


def test_first_cell_selected_by_default(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    assert page._scrubber._cells[0]._selected is True
    assert page._scrubber._cells[1]._selected is False


def test_click_switches_selected_cell(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    scrubber = page._scrubber
    scrubber._on_cell_clicked(1)
    assert scrubber._cells[0]._selected is False
    assert scrubber._cells[1]._selected is True

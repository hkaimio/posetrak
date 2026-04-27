"""Tests for app.setup.page_sync (sync wizard page)."""

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
            "INSERT INTO captures (id, session_id, capture_number, label) VALUES (?, ?, ?, ?)",
            (shot_id, session_id, shot_num, f"Shot {shot_num}"),
        )
        for cam_num in range(1, videos_per_shot + 1):
            video_id = generate_id()
            conn.execute(
                "INSERT INTO capture_videos "
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
    wiz.new_shot_ids = []   # empty → fallback path loads all shots
    page.wizard = MagicMock(return_value=wiz)
    return ctx


@pytest.fixture
def loaded_page(qapp, tmp_path):
    """2 shots × 2 cameras, page initialized."""
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=2, videos_per_shot=2)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    yield page, conn
    page.cleanupPage()
    conn.close()


@pytest.fixture
def loaded_page_1shot(qapp, tmp_path):
    """1 shot × 2 cameras, page initialized."""
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=1, videos_per_shot=2)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    yield page, conn
    page.cleanupPage()
    conn.close()


@pytest.fixture
def loaded_page_1video(qapp, tmp_path):
    """1 shot × 1 camera, page initialized."""
    conn, session_id = _make_session_with_shots(tmp_path, n_shots=1, videos_per_shot=1)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    yield page, conn
    page.cleanupPage()
    conn.close()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_page_constructs(qapp) -> None:
    page = SyncPage()
    assert page.title() == "Camera Synchronisation"


def test_page_is_always_complete(qapp) -> None:
    assert SyncPage().isComplete()


# ---------------------------------------------------------------------------
# initializePage — shot loading
# ---------------------------------------------------------------------------


def test_initialize_shows_shot_label(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    assert page._shot_label.text() != ""


def test_initialize_builds_scrubber(loaded_page_1shot) -> None:
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


def test_initialize_creates_anchor_overlays(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    assert len(page._anchor_overlays) == 2
    assert all(ov.anchor_frame is None for ov in page._anchor_overlays)


def test_initialize_creates_anchor_labels(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    assert len(page._anchor_labels) == 2
    assert all("—" in lbl.text() for lbl in page._anchor_labels)


# ---------------------------------------------------------------------------
# Per-cell sliders
# ---------------------------------------------------------------------------


def test_scrubber_has_per_cell_sliders(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    assert len(page._scrubber._sliders) == 2


def test_per_cell_slider_max_equals_last_frame(loaded_page_1video) -> None:
    page, _ = loaded_page_1video
    assert page._scrubber._sliders[0].maximum() == 299


def test_per_cell_slider_updates_on_seek(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    page._scrubber._set_cell_frame(0, 50)
    assert page._scrubber._sliders[0].value() == 50


def test_per_cell_frame_label_updates_on_seek(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    page._scrubber._set_cell_frame(1, 42)
    assert "42" in page._scrubber._frame_labels[1].text()


# ---------------------------------------------------------------------------
# Rough sync — set anchor
# ---------------------------------------------------------------------------


def test_set_anchor_records_current_frame(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    scrubber = page._scrubber
    scrubber._set_cell_frame(0, 100)   # seek cell 0 to frame 100
    scrubber._focused_cell = 0
    page._on_set_anchor()
    assert page._anchors[0] == 100


def test_set_anchor_updates_overlay(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    page._scrubber._set_cell_frame(0, 55)
    page._scrubber._focused_cell = 0
    page._on_set_anchor()
    assert page._anchor_overlays[0].anchor_frame == 55


def test_set_anchor_updates_label(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    page._scrubber._set_cell_frame(1, 77)
    page._scrubber._focused_cell = 1
    page._on_set_anchor()
    assert "77" in page._anchor_labels[1].text()
    assert "—" not in page._anchor_labels[1].text()


def test_apply_button_disabled_with_fewer_than_two_anchors(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    # One anchor only
    page._scrubber._focused_cell = 0
    page._on_set_anchor()
    assert not page._apply_rough_btn.isEnabled()


def test_apply_button_enabled_with_two_anchors(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    scrubber = page._scrubber
    scrubber._focused_cell = 0
    page._on_set_anchor()
    scrubber._focused_cell = 1
    page._on_set_anchor()
    assert page._apply_rough_btn.isEnabled()


# ---------------------------------------------------------------------------
# Rough sync — clear anchors
# ---------------------------------------------------------------------------


def test_clear_anchors_resets_state(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    page._scrubber._focused_cell = 0
    page._on_set_anchor()
    page._on_clear_anchors()
    assert page._anchors == {}
    assert all(ov.anchor_frame is None for ov in page._anchor_overlays)
    assert all("—" in lbl.text() for lbl in page._anchor_labels)


def test_clear_anchors_disables_apply(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    scrubber = page._scrubber
    scrubber._focused_cell = 0
    page._on_set_anchor()
    scrubber._focused_cell = 1
    page._on_set_anchor()
    page._on_clear_anchors()
    assert not page._apply_rough_btn.isEnabled()


# ---------------------------------------------------------------------------
# Rough sync — apply
# ---------------------------------------------------------------------------


def test_apply_rough_sync_writes_sync_config(loaded_page_1shot) -> None:
    page, conn = loaded_page_1shot
    scrubber = page._scrubber
    scrubber._set_cell_frame(0, 90)
    scrubber._focused_cell = 0
    page._on_set_anchor()
    scrubber._set_cell_frame(1, 120)
    scrubber._focused_cell = 1
    page._on_set_anchor()

    page._on_apply_rough_sync()

    configs = conn.execute("SELECT * FROM sync_configs").fetchall()
    assert len(configs) == 1
    assert configs[0]["created_by"] == "manual-rough"

    pts = conn.execute("SELECT * FROM sync_points").fetchall()
    assert len(pts) == 2


def test_apply_rough_sync_switches_scrubber_to_synced_mode(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    scrubber = page._scrubber
    scrubber._focused_cell = 0
    page._on_set_anchor()
    scrubber._focused_cell = 1
    page._on_set_anchor()

    page._on_apply_rough_sync()

    assert scrubber.sync_table is not None


def test_apply_rough_sync_shows_confirmation(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    scrubber = page._scrubber
    scrubber._focused_cell = 0
    page._on_set_anchor()
    scrubber._focused_cell = 1
    page._on_set_anchor()

    page._on_apply_rough_sync()

    assert "applied" in page._rough_status_label.text().lower()


def test_apply_rough_sync_correct_frame_offsets(loaded_page_1shot) -> None:
    """After applying, seeking to the reference timestamp shows correct frames."""
    page, _ = loaded_page_1shot
    scrubber = page._scrubber

    # Camera 0 anchor at frame 90 (t = 90/30 = 3.0 s)
    # Camera 1 anchor at frame 120 (also at t = 3.0 s in global time)
    scrubber._set_cell_frame(0, 90)
    scrubber._focused_cell = 0
    page._on_set_anchor()
    scrubber._set_cell_frame(1, 120)
    scrubber._focused_cell = 1
    page._on_set_anchor()
    page._on_apply_rough_sync()

    # At t=3.0s both cameras should show their anchor frames
    scrubber.seek_synced(3.0)
    assert scrubber.current_frames[0] == 90
    assert scrubber.current_frames[1] == 120


# ---------------------------------------------------------------------------
# Shot switching
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# cleanupPage
# ---------------------------------------------------------------------------


def test_cleanup_releases_scrubber_and_cache(qapp, tmp_path) -> None:
    conn, session_id = _make_session_with_shots(tmp_path)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    page.cleanupPage()
    assert page._scrubber is None
    assert page._cache is None
    assert page._anchors == {}
    conn.close()


# ---------------------------------------------------------------------------
# Selected cell highlight
# ---------------------------------------------------------------------------


def test_first_cell_selected_by_default(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    assert page._scrubber._cells[0]._selected is True
    assert page._scrubber._cells[1]._selected is False


def test_click_switches_selected_cell(loaded_page_1shot) -> None:
    page, _ = loaded_page_1shot
    page._scrubber._on_cell_clicked(1)
    assert page._scrubber._cells[0]._selected is False
    assert page._scrubber._cells[1]._selected is True

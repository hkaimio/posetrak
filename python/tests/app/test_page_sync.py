"""Tests for app.setup.page_sync — SyncPage and SyncWidget."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.setup.db_context import DBContext
from app.setup.page_sync import SyncDialog, SyncPage, SyncWidget, _ShotMeta
from posetrak.db.db import create_session, generate_id


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_session(
    tmp_path: Path,
    n_shots: int = 1,
    videos_per_shot: int = 2,
) -> tuple[sqlite3.Connection, str, str]:
    """Return (conn, session_id, first_shot_id)."""
    db_path = tmp_path / "sync_test.db"
    conn = create_session(db_path)
    conn.row_factory = sqlite3.Row
    session_id = generate_id()
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES (?, ?)",
        (session_id, "2026-05-01T10:00:00+00:00"),
    )
    first_shot_id = None
    for sn in range(1, n_shots + 1):
        shot_id = generate_id()
        if first_shot_id is None:
            first_shot_id = shot_id
        conn.execute(
            "INSERT INTO captures (id, session_id, capture_number, label) VALUES (?, ?, ?, ?)",
            (shot_id, session_id, sn, f"Capture {sn}"),
        )
        for cn in range(1, videos_per_shot + 1):
            vid_id = generate_id()
            conn.execute(
                "INSERT OR IGNORE INTO camera_models (id, manufacturer, model_name) "
                "VALUES ('mdl-test', 'TestCo', 'TestCam')"
            )
            conn.execute(
                "INSERT OR IGNORE INTO camera_instances (id, camera_model_id, label) "
                "VALUES (?, 'mdl-test', ?)",
                (f"cam{cn}", f"Cam {cn}"),
            )
            conn.execute(
                "INSERT INTO capture_videos "
                "(id, shot_id, camera_instance_id, file_path, "
                "first_video_frame, last_video_frame, actual_fps) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    vid_id, shot_id, f"cam{cn}",
                    f"/fake/shot{sn}_cam{cn}.mp4",
                    0, 299, 30.0,
                ),
            )
    conn.commit()
    return conn, session_id, first_shot_id


def _make_ctx(conn, session_id):
    return DBContext(conn, session_id)


def _attach_wizard(page: SyncPage, conn, session_id: str):
    ctx = _make_ctx(conn, session_id)
    wiz = MagicMock()
    wiz.db_context = ctx
    wiz.new_shot_ids = []
    page.wizard = MagicMock(return_value=wiz)
    return ctx


@pytest.fixture
def session_1shot(qapp, tmp_path):
    conn, session_id, shot_id = _make_session(tmp_path, n_shots=1, videos_per_shot=2)
    yield conn, session_id, shot_id
    conn.close()


@pytest.fixture
def session_2shots(qapp, tmp_path):
    conn, session_id, shot_id = _make_session(tmp_path, n_shots=2, videos_per_shot=2)
    yield conn, session_id, shot_id
    conn.close()


@pytest.fixture
def widget_1shot(session_1shot):
    conn, session_id, shot_id = session_1shot
    ctx = _make_ctx(conn, session_id)
    w = SyncWidget(ctx, shot_id)
    yield w, ctx, shot_id, conn
    w.shutdown()


# ---------------------------------------------------------------------------
# SyncPage — construction
# ---------------------------------------------------------------------------


def test_sync_page_constructs(qapp) -> None:
    page = SyncPage()
    assert page.title() == "Camera Synchronisation"


def test_sync_page_is_always_complete(qapp) -> None:
    assert SyncPage().isComplete()


# ---------------------------------------------------------------------------
# SyncPage — initializePage
# ---------------------------------------------------------------------------


def test_sync_page_creates_widget_on_init(qapp, tmp_path) -> None:
    conn, session_id, _ = _make_session(tmp_path)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    assert page._widget is not None
    page.cleanupPage()
    conn.close()


def test_sync_page_shot_row_hidden_for_single_shot(qapp, tmp_path) -> None:
    conn, session_id, _ = _make_session(tmp_path, n_shots=1)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    assert page._shot_row_w.isHidden()
    page.cleanupPage()
    conn.close()


def test_sync_page_shot_row_visible_for_multi_shot(qapp, tmp_path) -> None:
    conn, session_id, _ = _make_session(tmp_path, n_shots=2)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    # isHidden() reflects the explicit visibility flag; isVisible() also requires
    # the parent chain to be shown, which is not the case in unit tests.
    assert not page._shot_row_w.isHidden()
    page.cleanupPage()
    conn.close()


def test_sync_page_cleanup_destroys_widget(qapp, tmp_path) -> None:
    conn, session_id, _ = _make_session(tmp_path)
    page = SyncPage()
    _attach_wizard(page, conn, session_id)
    page.initializePage()
    assert page._widget is not None
    page.cleanupPage()
    assert page._widget is None
    conn.close()


# ---------------------------------------------------------------------------
# SyncWidget — construction & initial state
# ---------------------------------------------------------------------------


def test_sync_widget_constructs(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    assert w._pair is not None
    assert w._tree is not None


def test_sync_widget_combos_populated(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    assert w._ref_combo.count() == 2
    assert w._tgt_combo.count() == 2


def test_sync_widget_combos_default_different(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    assert w._ref_combo.currentIndex() != w._tgt_combo.currentIndex()


def test_sync_widget_tree_has_camera_items(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    assert w._tree.topLevelItemCount() == 2


def test_sync_widget_solve_btn_disabled_initially(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    assert not w._solve_btn.isEnabled()


def test_sync_widget_delete_btn_disabled_initially(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    assert not w._delete_btn.isEnabled()


# ---------------------------------------------------------------------------
# SyncWidget — anchor recording
# ---------------------------------------------------------------------------


def test_anchor_recording_writes_to_db(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(1)
    w._on_anchor_requested(100, 50)
    anchors = conn.execute(
        "SELECT COUNT(*) FROM sync_anchors WHERE shot_id = ?", (shot_id,)
    ).fetchone()[0]
    assert anchors == 1
    obs = conn.execute(
        "SELECT COUNT(*) FROM sync_anchor_observations"
    ).fetchone()[0]
    assert obs == 2


def test_anchor_recording_rejected_when_same_camera(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    # Force ref and tgt to same index
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(0)
    w._on_anchor_requested(100, 100)
    anchors = conn.execute(
        "SELECT COUNT(*) FROM sync_anchors WHERE shot_id = ?", (shot_id,)
    ).fetchone()[0]
    assert anchors == 0


def test_anchor_updates_tree(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    before = w._tree.topLevelItem(0).childCount()
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(1)
    w._on_anchor_requested(100, 50)
    after = w._tree.topLevelItem(0).childCount()
    assert after == before + 1


def test_anchor_enables_solve_btn(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(1)
    w._on_anchor_requested(100, 50)
    assert w._solve_btn.isEnabled()


def test_anchor_updates_status_label(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(1)
    w._on_anchor_requested(77, 33)
    assert "77" in w._status_label.text()
    assert "33" in w._status_label.text()


# ---------------------------------------------------------------------------
# SyncWidget — delete anchor
# ---------------------------------------------------------------------------


def test_delete_anchor_removes_from_db(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(1)
    w._on_anchor_requested(100, 50)

    # _reload_tree resets _current_anchor_id; simulate selecting via tree click
    parent_item = w._tree.topLevelItem(0)
    child = parent_item.child(0)
    w._on_tree_item_clicked(child, 0)
    assert w._current_anchor_id is not None

    w._on_delete_anchor()

    anchors = conn.execute(
        "SELECT COUNT(*) FROM sync_anchors WHERE shot_id = ?", (shot_id,)
    ).fetchone()[0]
    assert anchors == 0
    obs = conn.execute(
        "SELECT COUNT(*) FROM sync_anchor_observations"
    ).fetchone()[0]
    assert obs == 0


def test_delete_anchor_disables_delete_btn(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(1)
    w._on_anchor_requested(100, 50)

    parent_item = w._tree.topLevelItem(0)
    child = parent_item.child(0)
    w._on_tree_item_clicked(child, 0)
    assert w._delete_btn.isEnabled()

    w._on_delete_anchor()
    assert not w._delete_btn.isEnabled()


# ---------------------------------------------------------------------------
# SyncWidget — solve
# ---------------------------------------------------------------------------


def test_solve_writes_sync_config(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(1)
    w._on_anchor_requested(300, 150)

    w._on_solve()

    configs = conn.execute(
        "SELECT * FROM sync_configs WHERE shot_id = ?", (shot_id,)
    ).fetchall()
    assert len(configs) == 1
    assert configs[0]["created_by"] == "manual-graph"


def test_solve_writes_sync_points(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(1)
    w._on_anchor_requested(300, 150)

    w._on_solve()

    pts = conn.execute("SELECT * FROM sync_points").fetchall()
    assert len(pts) == 2


def test_solve_sets_sync_table(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(1)
    w._on_anchor_requested(300, 150)
    w._on_solve()
    assert w._sync_table is not None


def test_solve_enables_led_btn(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(1)
    w._on_anchor_requested(300, 150)
    w._on_solve()
    assert w._led_btn.isEnabled()


def test_solve_correct_timestamps(widget_1shot) -> None:
    """Reference camera frame 300 and target frame 150 should map to same time."""
    w, ctx, shot_id, conn = widget_1shot
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(1)
    w._on_anchor_requested(300, 150)
    w._on_solve()

    pts = conn.execute("SELECT timestamp_s FROM sync_points").fetchall()
    timestamps = [r[0] for r in pts]
    assert len(timestamps) == 2
    assert abs(timestamps[0] - timestamps[1]) < 1e-9


# ---------------------------------------------------------------------------
# SyncWidget — connectivity label
# ---------------------------------------------------------------------------


def test_connectivity_label_isolated_when_no_anchors(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    assert "Not connected" in w._connectivity_label.text()


def test_connectivity_label_green_after_anchor(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(1)
    w._on_anchor_requested(100, 50)
    assert "All" in w._connectivity_label.text()
    assert "connected" in w._connectivity_label.text()


# ---------------------------------------------------------------------------
# SyncWidget — tree item click navigates scrubber
# ---------------------------------------------------------------------------


def test_tree_click_sets_current_anchor_id(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(1)
    w._on_anchor_requested(100, 50)

    # Click the child item under first camera
    parent_item = w._tree.topLevelItem(0)
    child = parent_item.child(0)
    w._on_tree_item_clicked(child, 0)
    assert w._current_anchor_id is not None


def test_tree_click_enables_delete_btn(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(1)
    w._on_anchor_requested(100, 50)

    parent_item = w._tree.topLevelItem(0)
    child = parent_item.child(0)
    w._on_tree_item_clicked(child, 0)
    assert w._delete_btn.isEnabled()


def test_tree_click_on_parent_clears_selection(widget_1shot) -> None:
    w, ctx, shot_id, conn = widget_1shot
    w._ref_combo.setCurrentIndex(0)
    w._tgt_combo.setCurrentIndex(1)
    w._on_anchor_requested(100, 50)

    # Click child then parent
    parent_item = w._tree.topLevelItem(0)
    child = parent_item.child(0)
    w._on_tree_item_clicked(child, 0)
    w._on_tree_item_clicked(parent_item, 0)
    assert w._current_anchor_id is None
    assert not w._delete_btn.isEnabled()


# ---------------------------------------------------------------------------
# SyncDialog — construction
# ---------------------------------------------------------------------------


def test_sync_dialog_constructs(session_1shot) -> None:
    conn, session_id, shot_id = session_1shot
    ctx = _make_ctx(conn, session_id)
    dlg = SyncDialog(ctx, shot_id)
    assert "Sync" in dlg.windowTitle()
    dlg._widget.shutdown()

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for SegmentationRunPanel (segmentation-ui-improvements
design doc, Issue 1) -- the read-only summary shown when a segmentation
node is clicked in the session tree.
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def session_db(tmp_path):
    from posetrak.db.db import create_session

    conn = create_session(tmp_path / "seg_panel_test.db")
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number, label) "
        "VALUES ('cap1', 'sess1', 1, 'My Capture')"
    )
    conn.execute(
        "INSERT INTO seg_quality_runs "
        "(id, shot_id, name, time_start_s, time_end_s, created_at, quality_source, notes) "
        "VALUES ('seg1', 'cap1', 'Main pass', 6.5, 13.9, '2026-08-23T14:15:05Z', "
        "'cutie-interactive', 'a note')"
    )
    conn.commit()
    yield conn
    conn.close()


def test_panel_shows_capture_and_segmentation_name(qapp, session_db):
    from app.ui.content_panels import SegmentationRunPanel

    panel = SegmentationRunPanel(session_db, "seg1")
    texts = [
        panel.layout().itemAt(i).widget().text()
        for i in range(panel.layout().count())
        if panel.layout().itemAt(i).widget() is not None
        and hasattr(panel.layout().itemAt(i).widget(), "text")
    ]
    joined = "\n".join(texts)
    assert "My Capture" in joined
    assert "Main pass" in joined
    assert "cutie-interactive" in joined
    assert "a note" in joined


def test_panel_handles_missing_row_without_crashing(qapp, session_db):
    from app.ui.content_panels import SegmentationRunPanel

    panel = SegmentationRunPanel(session_db, "does-not-exist")
    assert panel is not None


def test_open_button_emits_open_requested_with_seg_run_id(qapp, session_db):
    from app.ui.content_panels import SegmentationRunPanel

    panel = SegmentationRunPanel(session_db, "seg1")
    received = []
    panel.open_requested.connect(received.append)

    from PySide6.QtWidgets import QPushButton
    btn = panel.findChildren(QPushButton)[0]
    btn.click()
    assert received == ["seg1"]

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Naming and finding tracking runs in the session tree."""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QInputDialog, QTreeWidgetItem  # noqa: E402

from app.ui import session_tree  # noqa: E402
from app.ui.session_tree import ItemKind, SessionTreeWidget, tracking_run_label  # noqa: E402


def test_label_shows_the_name_or_a_generic_title_and_the_short_id():
    run_id = "80d7fab3-50e4-4a5a-b762-be0d9b154bef"
    assert tracking_run_label(run_id, None, "Tommi", "2026-07-22T05:05:47Z") == (
        "Tracking run  ·80d7fab3·  [Tommi]  2026-07-22 05:05")
    assert tracking_run_label(run_id, "ViTPose tuned", None, None).startswith("ViTPose tuned  ·80d7fab3·  [?]")


def _tree_with_runs(qapp) -> SessionTreeWidget:
    tree = SessionTreeWidget()
    capture = session_tree._make_item(ItemKind.CAPTURE, "cap-1", "Ukemit yritys 6")
    trial = session_tree._make_item(ItemKind.TRIAL, "trial-1", "Ekat ukemit")
    capture.addChild(trial)
    other = session_tree._make_item(ItemKind.TRIAL, "trial-2", "Other trial")
    capture.addChild(other)
    for run_id, label in (("aaaa1111-0000", "ViTPose tuned  ·aaaa1111·  [Tommi]"), ("bbbb2222-0000", "Tracking run  ·bbbb2222·  [Roosa]")):
        trial.addChild(session_tree._make_item(ItemKind.TRACKING_RUN, run_id, label))
    other.addChild(session_tree._make_item(ItemKind.TRACKING_RUN, "cccc3333-0000", "Tracking run  ·cccc3333·  [Timo]"))
    tree.addTopLevelItem(capture)
    return tree


def _visible(tree: SessionTreeWidget) -> list[str]:
    out = []

    def walk(item: QTreeWidgetItem) -> None:
        if not item.isHidden():
            out.append(item.data(0, session_tree._ID))
            for i in range(item.childCount()):
                walk(item.child(i))

    for i in range(tree.topLevelItemCount()):
        walk(tree.topLevelItem(i))
    return out


def test_filter_matches_name_and_id_and_keeps_the_ancestors(qapp):
    tree = _tree_with_runs(qapp)
    tree.set_filter("vitpose")
    assert _visible(tree) == ["cap-1", "trial-1", "aaaa1111-0000"]
    tree.set_filter("BBBB2222")
    assert _visible(tree) == ["cap-1", "trial-1", "bbbb2222-0000"]
    tree.set_filter("cccc")
    assert _visible(tree) == ["cap-1", "trial-2", "cccc3333-0000"]
    tree.set_filter("no such run")
    assert _visible(tree) == []


def test_filter_on_a_matching_trial_shows_its_runs_and_empty_text_shows_everything(qapp):
    tree = _tree_with_runs(qapp)
    tree.set_filter("ekat")
    assert _visible(tree) == ["cap-1", "trial-1", "aaaa1111-0000", "bbbb2222-0000"]
    tree.set_filter("")
    assert len(_visible(tree)) == 6


def test_rename_sets_and_clears_the_name(qapp, monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE tracking_runs (id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO tracking_runs (id) VALUES ('run-1')")
    tree = SessionTreeWidget()
    tree._conn = conn
    monkeypatch.setattr(tree, "reload", lambda: None)
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("  best run  ", True)))
    tree._rename_tracking_run("run-1")
    assert conn.execute("SELECT name FROM tracking_runs").fetchone()[0] == "best run"
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("   ", True)))
    tree._rename_tracking_run("run-1")
    assert conn.execute("SELECT name FROM tracking_runs").fetchone()[0] is None
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("kept", False)))
    tree._rename_tracking_run("run-1")
    assert conn.execute("SELECT name FROM tracking_runs").fetchone()[0] is None


def test_tree_lists_the_runs_of_a_sequence_with_their_names(qapp):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE skeletons (id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE tracking_runs (id TEXT PRIMARY KEY, name TEXT, skeleton_id TEXT, ran_at TEXT, notes TEXT, "
                 "observation_sequence_id TEXT)")
    conn.execute("INSERT INTO skeletons VALUES ('sk', 'Tommi scaled')")
    conn.execute("INSERT INTO tracking_runs VALUES ('run-aaaaaaaa-1', 'labels', 'sk', '2026-10-09T10:00:00Z', 'fed with labels', 'seq')")
    conn.execute("INSERT INTO tracking_runs VALUES ('run-bbbbbbbb-2', NULL, 'sk', '2026-10-09T11:00:00Z', NULL, 'seq')")
    tree = SessionTreeWidget()
    tree._conn = conn
    parent = QTreeWidgetItem(["sequence"])
    tree._add_tracking_runs(parent, "seq")
    texts = [parent.child(i).text(0) for i in range(parent.childCount())]
    assert texts == ["labels  ·run-aaaa·  [Tommi scaled]  2026-10-09 10:00", "Tracking run  ·run-bbbb·  [Tommi scaled]  2026-10-09 11:00"]
    assert parent.child(0).toolTip(0) == "labels\nrun-aaaaaaaa-1\nfed with labels"

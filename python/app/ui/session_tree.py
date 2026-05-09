"""session_tree.py — Tree widget displaying the session hierarchy."""

from __future__ import annotations

import sqlite3
from enum import Enum

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QMessageBox,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
)

_KIND = Qt.ItemDataRole.UserRole
_ID   = Qt.ItemDataRole.UserRole + 1


class ItemKind(str, Enum):
    CAPTURE       = "capture"
    TRIAL         = "trial"
    DETECTION_RUN = "detection_run"
    PERSON_TRACK  = "person_track"
    TRACKING_RUN  = "tracking_run"


class SessionTreeWidget(QTreeWidget):
    """Read-only tree view of one session DB.

    Emits ``*_selected`` signals when the user clicks an item; T3.4 connects
    these to the content panel.
    """

    capture_selected       = Signal(str)  # capture_id
    trial_selected         = Signal(str)  # trial_id
    detection_run_selected = Signal(str)  # run_id
    person_track_selected  = Signal(str)  # sequence_id
    tracking_run_selected  = Signal(str)  # tracking_run_id
    selection_changed      = Signal(str, str)  # kind.value, item_id

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
        self.currentItemChanged.connect(self._on_item_changed)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.reload()

    def unload(self) -> None:
        self._conn = None
        self.clear()

    def reload(self) -> None:
        self.clear()
        if self._conn is None:
            return
        selected_id = self._current_id()
        self._populate()
        self._restore_selection(selected_id)

    # ------------------------------------------------------------------
    # Private — population
    # ------------------------------------------------------------------

    def _populate(self) -> None:
        conn = self._conn
        session_row = conn.execute(
            "SELECT id FROM mocap_sessions ORDER BY recorded_at LIMIT 1"
        ).fetchone()
        if session_row is None:
            return
        session_id = session_row["id"]

        captures = conn.execute(
            "SELECT id, label, capture_number FROM captures "
            "WHERE session_id = ? ORDER BY capture_number",
            (session_id,),
        ).fetchall()

        for cap in captures:
            label = cap["label"] or f"Capture {cap['capture_number']}"
            has_sync = conn.execute(
                "SELECT COUNT(*) FROM sync_configs WHERE shot_id = ?",
                (cap["id"],),
            ).fetchone()[0] > 0
            sync_indicator = "  ✓" if has_sync else "  (no sync)"
            cap_item = _make_item(ItemKind.CAPTURE, cap["id"], label + sync_indicator)
            if not has_sync:
                _grey(cap_item)
            self._add_capture_children(cap_item, cap["id"])
            self.addTopLevelItem(cap_item)
            cap_item.setExpanded(True)

    def _add_capture_children(self, parent: QTreeWidgetItem, capture_id: str) -> None:
        conn = self._conn

        # Trials first
        trials = conn.execute(
            "SELECT id, name, time_start_s, time_end_s "
            "FROM trials WHERE capture_id = ? ORDER BY time_start_s",
            (capture_id,),
        ).fetchall()
        for tr in trials:
            name = tr["name"] or "Unnamed trial"
            if tr["time_start_s"] is not None and tr["time_end_s"] is not None:
                name += f"  ({tr['time_start_s']:.1f}s – {tr['time_end_s']:.1f}s)"
            tr_item = _make_item(ItemKind.TRIAL, tr["id"], name)
            tr_item.setExpanded(True)
            for dr in self._load_detection_runs(capture_id, trial_id=tr["id"]):
                tr_item.addChild(self._make_detection_run_item(dr))
            parent.addChild(tr_item)

        # Detection runs not in any trial
        for dr in self._load_detection_runs(capture_id, trial_id=None):
            parent.addChild(self._make_detection_run_item(dr))

    def _load_detection_runs(self, capture_id: str, trial_id: str | None):
        if trial_id is None:
            return self._conn.execute(
                "SELECT id, detector_model, status, created_at "
                "FROM detection_runs "
                "WHERE shot_id = ? AND trial_id IS NULL ORDER BY created_at",
                (capture_id,),
            ).fetchall()
        return self._conn.execute(
            "SELECT id, detector_model, status, created_at "
            "FROM detection_runs "
            "WHERE shot_id = ? AND trial_id = ? ORDER BY created_at",
            (capture_id, trial_id),
        ).fetchall()

    def _make_detection_run_item(self, dr) -> QTreeWidgetItem:
        ts = _fmt_ts(dr["created_at"])
        status = dr["status"]
        label = f"Detection [{dr['detector_model']}]  {ts}"
        if status != "complete":
            label += f"  ({status})"
        item = _make_item(ItemKind.DETECTION_RUN, dr["id"], label)
        if status == "failed":
            _grey(item)
        item.setExpanded(True)
        self._add_person_tracks(item, dr["id"])
        return item

    def _add_person_tracks(self, parent: QTreeWidgetItem, detection_run_id: str) -> None:
        rows = self._conn.execute(
            "SELECT pos.id, pos.name, "
            "    GROUP_CONCAT(sp.person_name, ', ') AS person_names "
            "FROM pose_observation_sequences pos "
            "LEFT JOIN sequence_persons sp ON sp.sequence_id = pos.id "
            "WHERE pos.detection_run_id = ? "
            "GROUP BY pos.id "
            "ORDER BY pos.time_start_s",
            (detection_run_id,),
        ).fetchall()
        for row in rows:
            label = row["person_names"] or row["name"] or "Person track"
            seq_item = _make_item(ItemKind.PERSON_TRACK, row["id"], label)
            seq_item.setExpanded(True)
            self._add_tracking_runs(seq_item, row["id"])
            parent.addChild(seq_item)

    def _add_tracking_runs(self, parent: QTreeWidgetItem, sequence_id: str) -> None:
        runs = self._conn.execute(
            "SELECT tr.id, s.name AS skel_name, tr.ran_at, tr.notes "
            "FROM tracking_runs tr "
            "LEFT JOIN skeletons s ON s.id = tr.skeleton_id "
            "WHERE tr.observation_sequence_id = ? ORDER BY tr.ran_at",
            (sequence_id,),
        ).fetchall()
        for run in runs:
            skel = run["skel_name"] or "?"
            label = f"Tracking run  [{skel}]  {_fmt_ts(run['ran_at'])}"
            item = _make_item(ItemKind.TRACKING_RUN, run["id"], label)
            if run["notes"]:
                item.setToolTip(0, run["notes"])
            parent.addChild(item)

    # ------------------------------------------------------------------
    # Private — selection
    # ------------------------------------------------------------------

    def _on_item_changed(self, current: QTreeWidgetItem, _prev) -> None:
        if current is None:
            return
        kind: ItemKind = current.data(0, _KIND)
        item_id: str = current.data(0, _ID)
        if kind == ItemKind.CAPTURE:
            self.capture_selected.emit(item_id)
        elif kind == ItemKind.TRIAL:
            self.trial_selected.emit(item_id)
        elif kind == ItemKind.DETECTION_RUN:
            self.detection_run_selected.emit(item_id)
        elif kind == ItemKind.PERSON_TRACK:
            self.person_track_selected.emit(item_id)
        elif kind == ItemKind.TRACKING_RUN:
            self.tracking_run_selected.emit(item_id)
        self.selection_changed.emit(kind.value, item_id)

    def _current_id(self) -> str | None:
        item = self.currentItem()
        return item.data(0, _ID) if item else None

    def restore_selection(self, item_id: str | None) -> None:
        if item_id is None:
            return
        for item in _iter_items(self):
            if item.data(0, _ID) == item_id:
                self.setCurrentItem(item)
                return

    def _restore_selection(self, item_id: str | None) -> None:
        self.restore_selection(item_id)

    # ------------------------------------------------------------------
    # Private — context menu
    # ------------------------------------------------------------------

    def _on_context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is None:
            return
        kind: ItemKind = item.data(0, _KIND)
        item_id: str = item.data(0, _ID)
        menu = QMenu(self)
        {
            ItemKind.CAPTURE:       self._capture_menu,
            ItemKind.TRIAL:         self._trial_menu,
            ItemKind.DETECTION_RUN: self._detection_run_menu,
            ItemKind.PERSON_TRACK:  self._person_track_menu,
            ItemKind.TRACKING_RUN:  self._tracking_run_menu,
        }[kind](menu, item_id)
        menu.exec(self.viewport().mapToGlobal(pos))

    def _capture_menu(self, menu: QMenu, capture_id: str) -> None:
        menu.addAction("New trial…").setEnabled(False)
        menu.addAction("Set up sync…").setEnabled(False)
        menu.addAction("Import extrinsics…").setEnabled(False)
        menu.addSeparator()
        menu.addAction("Delete capture").triggered.connect(
            lambda: self._delete_capture(capture_id)
        )

    def _trial_menu(self, menu: QMenu, trial_id: str) -> None:
        menu.addAction("Run detection…").setEnabled(False)
        menu.addAction("Edit name & notes…").setEnabled(False)
        menu.addSeparator()
        menu.addAction("Delete trial").triggered.connect(
            lambda: self._confirm_delete("trials", trial_id)
        )

    def _detection_run_menu(self, menu: QMenu, run_id: str) -> None:
        menu.addAction("Finalise → person tracks…").setEnabled(False)
        menu.addSeparator()
        menu.addAction("Delete run").triggered.connect(
            lambda: self._confirm_delete("detection_runs", run_id)
        )

    def _person_track_menu(self, menu: QMenu, seq_id: str) -> None:
        menu.addAction("Run tracker…").setEnabled(False)
        menu.addAction("Rename…").setEnabled(False)
        menu.addSeparator()
        menu.addAction("Delete track").triggered.connect(
            lambda: self._confirm_delete("pose_observation_sequences", seq_id)
        )

    def _tracking_run_menu(self, menu: QMenu, run_id: str) -> None:
        menu.addAction("View results…").setEnabled(False)
        menu.addAction("Export BVH…").setEnabled(False)
        menu.addSeparator()
        menu.addAction("Delete run").triggered.connect(
            lambda: self._confirm_delete("tracking_runs", run_id)
        )

    # ------------------------------------------------------------------
    # Private — delete helpers
    # ------------------------------------------------------------------

    def _confirm_delete(self, table: str, item_id: str) -> None:
        if not _ask_delete(self):
            return
        with self._conn:
            self._conn.execute(f"DELETE FROM {table} WHERE id = ?", (item_id,))
        self.reload()

    def _delete_capture(self, capture_id: str) -> None:
        if not _ask_delete(self, "Delete this capture and all its data?"):
            return
        conn = self._conn
        # Manual cascade (no ON DELETE CASCADE in schema)
        with conn:
            seq_ids = [
                r[0] for r in conn.execute(
                    "SELECT id FROM pose_observation_sequences WHERE shot_id = ?",
                    (capture_id,),
                )
            ]
            for sid in seq_ids:
                conn.execute("DELETE FROM tracking_results WHERE run_id IN "
                             "(SELECT id FROM tracking_runs WHERE observation_sequence_id = ?)", (sid,))
                conn.execute("DELETE FROM tracking_obs_results WHERE run_id IN "
                             "(SELECT id FROM tracking_runs WHERE observation_sequence_id = ?)", (sid,))
                conn.execute("DELETE FROM tracking_runs WHERE observation_sequence_id = ?", (sid,))
                conn.execute("DELETE FROM sequence_persons WHERE sequence_id = ?", (sid,))
                conn.execute("DELETE FROM pose_observations WHERE sequence_id = ?", (sid,))
            conn.execute(
                "DELETE FROM pose_observation_sequences WHERE shot_id = ?", (capture_id,)
            )
            dr_ids = [
                r[0] for r in conn.execute(
                    "SELECT id FROM detection_runs WHERE shot_id = ?", (capture_id,)
                )
            ]
            for did in dr_ids:
                conn.execute(
                    "DELETE FROM detection_keypoints WHERE detection_run_id = ?", (did,)
                )
                conn.execute(
                    "DELETE FROM person_detections WHERE detection_run_id = ?", (did,)
                )
                conn.execute(
                    "DELETE FROM person_tracks WHERE detection_run_id = ?", (did,)
                )
            conn.execute("DELETE FROM detection_runs WHERE shot_id = ?", (capture_id,))
            conn.execute("DELETE FROM trials WHERE capture_id = ?", (capture_id,))
            conn.execute("DELETE FROM sync_configs WHERE shot_id = ?", (capture_id,))
            conn.execute("DELETE FROM capture_videos WHERE shot_id = ?", (capture_id,))
            conn.execute("DELETE FROM captures WHERE id = ?", (capture_id,))
        self.reload()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(kind: ItemKind, item_id: str, label: str) -> QTreeWidgetItem:
    item = QTreeWidgetItem([label])
    item.setData(0, _KIND, kind)
    item.setData(0, _ID, item_id)
    return item


def _grey(item: QTreeWidgetItem) -> None:
    from PySide6.QtGui import QColor
    item.setForeground(0, QColor("gray"))


def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return ""
    return ts[:16].replace("T", " ")


def _ask_delete(parent, message: str = "Delete this item and all its data?") -> bool:
    return QMessageBox.question(
        parent,
        "Confirm delete",
        message,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    ) == QMessageBox.StandardButton.Yes


def _iter_items(tree: QTreeWidget):
    """Yield all QTreeWidgetItems in tree."""
    stack = [tree.invisibleRootItem()]
    while stack:
        node = stack.pop()
        for i in range(node.childCount()):
            child = node.child(i)
            yield child
            stack.append(child)

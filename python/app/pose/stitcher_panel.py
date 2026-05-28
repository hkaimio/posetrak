"""stitcher_panel.py — Embeddable stitcher + assignment widget for a detection run.

Used by TrialPanel in the unified shell.  Exposes is_dirty / apply() for
prompt-on-leave and the Apply button.

Uses FilmstripStitcherWidget as its timeline.  FrameViewWidget and
PersonPreviewWidget are not used here; they remain available in
PoseExtractionWindow (main.py).
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.pose.db_cache import list_detection_runs
from app.pose.finalise import TrackAssignment, finalise_to_db
from app.pose.filmstrip_stitcher import (
    FilmstripStitcherWidget,
    ROW_H_DEFAULT,
    ROW_H_MIN,
    ROW_H_MAX,
)
from app.setup.db_context import SyncPoint, SyncTable


@dataclass
class _Selection:
    """Tracks the currently selected segment and optional sub-range within it."""
    svid: str
    tid: int
    seg_first: int      # segment identifier in _assignments / get_spans()
    sel_first: int      # selected frame range start (== seg_first for full-bar click)
    sel_last: int       # selected frame range end   (== seg_last  for full-bar click)


def _load_sync_table(conn: sqlite3.Connection, sync_config_id: str) -> SyncTable:
    rows = conn.execute(
        "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, sv.actual_fps "
        "FROM sync_points sp "
        "JOIN capture_videos sv ON sv.id = sp.shot_video_id "
        "WHERE sp.sync_config_id = ? ORDER BY sp.shot_video_id, sp.video_frame",
        (sync_config_id,),
    ).fetchall()
    fps_by_video: dict[str, float] = {}
    sync_points: list[SyncPoint] = []
    for r in rows:
        svid = r["shot_video_id"]
        fps_by_video.setdefault(svid, float(r["actual_fps"] or 30.0))
        sync_points.append(SyncPoint(
            camera_instance_id=svid,
            shot_video_id=svid,
            video_frame=int(r["video_frame"]),
            timestamp_s=float(r["timestamp_s"]),
        ))
    return SyncTable(sync_points, fps_by_video)


class StitcherPanel(QWidget):
    """Filmstrip timeline + assignment panel for one detection run.

    Signals
    -------
    applied:
        Emitted after a successful Apply (finalise).
    dirty_changed(bool):
        Emitted when the dirty state changes.
    """

    applied = Signal()
    dirty_changed = Signal(bool)

    def __init__(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._conn = conn
        self._run_id = run_id
        self._shot_id: str | None = None
        self._sync_config_id: str | None = None

        # svid → camera info for status bar frame numbers
        self._camera_info: dict[str, dict] = {}   # svid → {label, fps, ...}

        self._assignments: dict[tuple[str, int, int], str] = {}
        self._last_applied: dict[tuple[str, int, int], str] = {}

        self._selection: _Selection | None = None

        self._build_ui()
        self._load_run()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_dirty(self) -> bool:
        return self._assignments != self._last_applied

    def apply(self) -> bool:
        """Finalise assignments to DB.  Returns True on success."""
        if not self._run_id:
            QMessageBox.warning(self, "Warning", "No detection run loaded.")
            return False
        if not self._assignments:
            QMessageBox.warning(self, "Warning", "No track assignments defined.")
            return False
        if not self._shot_id or not self._sync_config_id:
            QMessageBox.warning(self, "Warning", "Run metadata missing.")
            return False

        # Check for unresolved overlap conflicts
        conflicts = self._compute_conflicts()
        if conflicts:
            ans = QMessageBox.question(
                self, "Unresolved conflicts",
                f"{len(conflicts)} overlap conflict(s) remain.\n"
                "Same person is assigned to overlapping time ranges in the same camera.\n"
                "Apply anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return False

        spans = self._stitcher.get_spans()
        assignment_list = []
        for (svid, tid, seg_first), person_name in self._assignments.items():
            seg_range = spans.get((svid, tid, seg_first))
            if seg_range is None:
                continue
            first_frame, last_frame = seg_range
            assignment_list.append(TrackAssignment(
                shot_video_id=svid,
                track_id=tid,
                person_name=person_name,
                first_frame=first_frame,
                last_frame=last_frame,
            ))

        if not assignment_list:
            QMessageBox.warning(self, "Warning", "No valid assignments found.")
            return False

        run_row = self._conn.execute(
            "SELECT pose_model FROM detection_runs WHERE id=?", (self._run_id,)
        ).fetchone()
        pose_model = run_row["pose_model"] if run_row else ""

        try:
            seq_ids = finalise_to_db(
                session=self._conn,
                detection_run_id=self._run_id,
                shot_id=self._shot_id,
                sync_config_id=self._sync_config_id,
                assignments=assignment_list,
                pose_model=pose_model,
            )
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", str(e))
            return False

        was_dirty = self.is_dirty
        self._last_applied = dict(self._assignments)
        if was_dirty:
            self.dirty_changed.emit(False)
        self.applied.emit()
        QMessageBox.information(
            self, "Applied",
            f"Created {len(seq_ids)} person sequence(s):\n" + "\n".join(seq_ids),
        )
        return True

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        # Filmstrip timeline
        self._stitcher = FilmstripStitcherWidget(row_h=ROW_H_DEFAULT)
        self._stitcher.segment_selected.connect(self._on_segment_selected)
        self._stitcher.assignment_changed.connect(self._on_assignment_changed)
        self._stitcher.time_hovered.connect(self._on_time_hovered)
        root.addWidget(self._stitcher, 1)

        # Status bar
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("font-size: 10px; color: #444; padding: 1px 4px;")
        self._status_label.setMaximumHeight(18)
        root.addWidget(self._status_label)

        # Assignment controls
        assign_group = QGroupBox("Selected track")
        assign_layout = QVBoxLayout(assign_group)

        self._selected_label = QLabel("None selected")
        assign_layout.addWidget(self._selected_label)

        person_row = QHBoxLayout()
        person_row.addWidget(QLabel("Person:"))
        self._person_combo = QComboBox()
        self._person_combo.setEditable(True)
        self._person_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._person_combo.setMinimumWidth(120)
        person_row.addWidget(self._person_combo, 1)
        self._add_person_btn = QPushButton("+Add")
        self._add_person_btn.clicked.connect(self._on_add_person)
        person_row.addWidget(self._add_person_btn)
        assign_layout.addLayout(person_row)

        btn_row = QHBoxLayout()
        self._assign_btn = QPushButton("Assign")
        self._assign_btn.clicked.connect(self._on_assign)
        btn_row.addWidget(self._assign_btn)

        self._conflict_label = QLabel("")
        self._conflict_label.setStyleSheet("color: #c00; font-size: 10px;")
        btn_row.addWidget(self._conflict_label)
        btn_row.addStretch()

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setToolTip(
            "Finalise person assignments and create person sequences"
        )
        self._apply_btn.clicked.connect(self.apply)
        btn_row.addWidget(self._apply_btn)
        assign_layout.addLayout(btn_row)

        root.addWidget(assign_group)

    # ------------------------------------------------------------------
    # Run loading
    # ------------------------------------------------------------------

    def _load_run(self) -> None:
        run_row = self._conn.execute(
            "SELECT shot_id, sync_config_id FROM detection_runs WHERE id = ?",
            (self._run_id,),
        ).fetchone()
        if run_row is None:
            return
        self._shot_id = run_row["shot_id"]
        self._sync_config_id = run_row["sync_config_id"]

        sync_table = (
            _load_sync_table(self._conn, self._sync_config_id)
            if self._sync_config_id else None
        )
        self._stitcher.set_sync_table(sync_table)

        # Load camera info for status bar
        if self._shot_id:
            for r in self._conn.execute(
                "SELECT sv.id, sv.actual_fps, "
                "       COALESCE(ci.label, sv.camera_instance_id) AS label "
                "FROM capture_videos sv "
                "LEFT JOIN camera_instances ci ON ci.id = sv.camera_instance_id "
                "WHERE sv.shot_id = ?",
                (self._shot_id,),
            ):
                self._camera_info[r["id"]] = {
                    "label": r["label"],
                    "fps": float(r["actual_fps"] or 30.0),
                }

        self._stitcher.load_run(self._conn, self._run_id)
        self._restore_assignments()
        self._last_applied = dict(self._assignments)

    def _restore_assignments(self) -> None:
        rows = self._conn.execute(
            "SELECT shot_video_id, track_id, person_name, first_frame, last_frame "
            "FROM detection_track_assignments "
            "WHERE detection_run_id = ? "
            "ORDER BY shot_video_id, track_id, first_frame",
            (self._run_id,),
        ).fetchall()
        if not rows:
            return

        by_track: dict[tuple[str, int], list] = defaultdict(list)
        for r in rows:
            by_track[(r["shot_video_id"], r["track_id"])].append(r)

        spans = self._stitcher.get_spans()
        for (svid, tid), track_rows in by_track.items():
            seg_first = next(
                (sf for (s, t, sf) in spans if s == svid and t == tid), None
            )
            if seg_first is None:
                continue
            cur_seg_first = seg_first
            for i, r in enumerate(track_rows):
                if i < len(track_rows) - 1:
                    split_frame = track_rows[i + 1]["first_frame"]
                    self._stitcher.split_segment(svid, tid, cur_seg_first, split_frame)
                    self._apply_assignment(svid, tid, cur_seg_first, r["person_name"])
                    cur_seg_first = split_frame
                else:
                    self._apply_assignment(svid, tid, cur_seg_first, r["person_name"])

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _on_time_hovered(self, global_s: float) -> None:
        if not self._camera_info or self._stitcher._sync_table is None:
            mm, ss = divmod(global_s, 60)
            self._status_label.setText(f"t = {int(mm):02d}:{ss:06.3f}")
            return
        mm, ss = divmod(global_s, 60)
        parts = [f"t = {int(mm):02d}:{ss:06.3f}"]
        for svid, info in sorted(self._camera_info.items(), key=lambda x: x[1]["label"]):
            frame = self._stitcher._sync_table.lookup(global_s, svid)
            if frame is not None:
                parts.append(f"{info['label']}: {frame}")
        self._status_label.setText("  |  ".join(parts))

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_segment_selected(
        self,
        svid: str,
        tid: int,
        seg_first: int,
        seg_last: int,
        sel_first: int,
        sel_last: int,
    ) -> None:
        self._selection = _Selection(svid, tid, seg_first, sel_first, sel_last)
        is_partial = (sel_first != seg_first or sel_last != seg_last)
        if is_partial:
            self._selected_label.setText(
                f"video: {svid[:8]}  track: {tid}  "
                f"selection: {sel_first}–{sel_last}"
            )
        else:
            self._selected_label.setText(
                f"video: {svid[:8]}  track: {tid}  "
                f"frames: {seg_first}–{seg_last}"
            )
        # Pre-fill combo with current assignment
        person_name = self._assignments.get((svid, tid, seg_first))
        if person_name and self._person_combo.findText(person_name) >= 0:
            self._person_combo.setCurrentText(person_name)

    def _on_assignment_changed(
        self, svid: str, tid: int, seg_first: int, person_name: object
    ) -> None:
        if person_name:
            self._do_assign(svid, tid, seg_first, str(person_name))
        else:
            self._detach(svid, tid, seg_first)

    # ------------------------------------------------------------------
    # Assignment with auto-split and auto-merge
    # ------------------------------------------------------------------

    def _on_add_person(self) -> None:
        name, ok = QInputDialog.getText(self, "Add Person", "Person name:")
        if ok and name.strip():
            name = name.strip()
            if self._person_combo.findText(name) < 0:
                self._person_combo.addItem(name)
            self._person_combo.setCurrentText(name)
            self._refresh_persons()

    def _on_assign(self) -> None:
        if self._selection is None:
            QMessageBox.information(self, "No track selected", "Click a track segment first.")
            return
        name = self._person_combo.currentText().strip()
        if not name:
            QMessageBox.warning(self, "No name", "Enter a person name.")
            return
        self._do_assign(
            self._selection.svid,
            self._selection.tid,
            self._selection.seg_first,
            name,
        )

    def _do_assign(self, svid: str, tid: int, seg_first: int, name: str) -> None:
        """Assign *name* to the current selection within the segment.

        If the selection is a sub-range, auto-splits the segment at the
        selection boundaries.  Runs auto-merge after the assignment so that
        adjacent segments with the same person collapse into one.
        """
        # Determine sel_first / sel_last from active selection
        if (
            self._selection is not None
            and self._selection.svid == svid
            and self._selection.tid == tid
            and self._selection.seg_first == seg_first
        ):
            sel_first = self._selection.sel_first
            sel_last = self._selection.sel_last
        else:
            spans = self._stitcher.get_spans()
            seg_range = spans.get((svid, tid, seg_first))
            if seg_range is None:
                return
            sel_first, sel_last = seg_range

        spans = self._stitcher.get_spans()
        seg_range = spans.get((svid, tid, seg_first))
        if seg_range is None:
            return
        cur_first, cur_last = seg_range

        sel_first = max(sel_first, cur_first)
        sel_last = min(sel_last, cur_last)

        # Split left edge if selection starts inside the segment
        target_sf = seg_first
        if sel_first > cur_first:
            self._stitcher.split_segment(svid, tid, cur_first, sel_first)
            target_sf = sel_first

        # Split right edge if selection ends inside the segment
        spans = self._stitcher.get_spans()
        target_range = spans.get((svid, tid, target_sf))
        if target_range and sel_last < target_range[1]:
            self._stitcher.split_segment(svid, tid, target_sf, sel_last + 1)

        self._apply_assignment(svid, tid, target_sf, name)
        self._auto_merge(svid, tid)
        self._refresh_conflicts(svid)

    def _detach(self, svid: str, tid: int, seg_first: int) -> None:
        self._assignments.pop((svid, tid, seg_first), None)
        self._stitcher.set_segment_assignment(svid, tid, seg_first, None)
        if self._selection and (svid, tid, seg_first) == (
            self._selection.svid, self._selection.tid, self._selection.seg_first
        ):
            pass  # selection label stays; user can re-assign
        self._auto_merge(svid, tid)
        self._refresh_conflicts(svid)
        self._emit_dirty()

    def _apply_assignment(self, svid: str, tid: int, seg_first: int, name: str) -> None:
        self._assignments[(svid, tid, seg_first)] = name
        self._stitcher.set_segment_assignment(svid, tid, seg_first, name)
        if self._person_combo.findText(name) < 0:
            self._person_combo.addItem(name)
        self._refresh_persons()
        self._emit_dirty()

    def _auto_merge(self, svid: str, tid: int) -> None:
        """Merge adjacent segments that share the same assignment (or both None)."""
        changed = True
        while changed:
            changed = False
            spans = self._stitcher.get_spans()
            track_segs = sorted(
                [(sf, sl) for (s, t, sf), (_, sl) in spans.items()
                 if s == svid and t == tid],
                key=lambda x: x[0],
            )
            for i in range(len(track_segs) - 1):
                sf1, sl1 = track_segs[i]
                sf2, sl2 = track_segs[i + 1]
                if sf2 != sl1 + 1:
                    continue
                a1 = self._assignments.get((svid, tid, sf1))
                a2 = self._assignments.get((svid, tid, sf2))
                if a1 == a2:
                    self._stitcher.merge_segments(svid, tid, sf1, sf2)
                    self._assignments.pop((svid, tid, sf2), None)
                    # Update selection if it was pointing at the merged-away segment
                    if (self._selection and
                            self._selection.svid == svid and
                            self._selection.tid == tid and
                            self._selection.seg_first == sf2):
                        self._selection = _Selection(svid, tid, sf1, sf1,
                                                     self._stitcher.get_spans().get(
                                                         (svid, tid, sf1), (sf1, sf1)
                                                     )[1])
                    changed = True
                    break

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    def _compute_conflicts(self) -> set[tuple[str, int, int]]:
        """Return the set of seg_keys that overlap with another seg in the same camera
        and share the same person assignment."""
        from app.setup.db_context import SyncTable
        time_spans = self._stitcher.get_time_spans()
        # person+svid → [(t0, t1, key)]
        by_person_cam: dict[tuple[str, str], list] = defaultdict(list)
        for key, person in self._assignments.items():
            ts = time_spans.get(key)
            if ts is None:
                continue
            svid = key[0]
            by_person_cam[(person, svid)].append((ts[0], ts[1], key))

        conflict_keys: set[tuple[str, int, int]] = set()
        for intervals in by_person_cam.values():
            intervals.sort()
            for i, (t0a, t1a, ka) in enumerate(intervals):
                for t0b, t1b, kb in intervals[i + 1:]:
                    if t0b >= t1a:
                        break
                    # Overlap
                    conflict_keys.add(ka)
                    conflict_keys.add(kb)
        return conflict_keys

    def _refresh_conflicts(self, _svid: str | None = None) -> None:
        conflicts = self._compute_conflicts()
        self._stitcher.set_conflict_segments(conflicts)
        if conflicts:
            self._conflict_label.setText(f"⚠ {len(conflicts)} conflict(s)")
        else:
            self._conflict_label.setText("")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_persons(self) -> None:
        persons = [
            self._person_combo.itemText(i)
            for i in range(self._person_combo.count())
        ]
        self._stitcher.set_known_persons(persons)

    def _emit_dirty(self) -> None:
        self.dirty_changed.emit(self.is_dirty)

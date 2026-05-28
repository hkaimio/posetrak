"""stitcher_panel.py — Embeddable stitcher + assignment widget for a detection run.

Used by TrialPanel in the unified shell.  Mirrors the stitcher/assignment half
of PoseExtractionWindow but is a plain QWidget rather than a QMainWindow, and
exposes is_dirty / apply() for prompt-on-leave and the Apply button.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from app.pose.assignment import find_assignment_conflicts
from app.pose.db_cache import list_detection_runs
from app.pose.finalise import TrackAssignment, finalise_to_db
from app.pose.frame_view import CameraInfo, FrameViewWidget
from app.pose.person_preview import PersonPreviewWidget
from app.pose.stitcher import StitcherWidget
from app.setup.db_context import SyncPoint, SyncTable


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
    """Stitcher timeline + frame view + assignment panel for one detection run.

    Signals
    -------
    applied:
        Emitted after a successful Apply (finalise).  Callers should reload
        the session tree to surface new person nodes.
    dirty_changed(bool):
        Emitted whenever the dirty state changes.  TrialPanel / main window
        use this to decide whether to prompt before navigating away.
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

        self._assignments: dict[tuple[str, int, int], str] = {}
        self._last_applied: dict[tuple[str, int, int], str] = {}

        self._current_svid: str | None = None
        self._current_track_id: int | None = None
        self._current_seg_first: int | None = None
        self._current_frame_idx: int = 0

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
        if self._run_id is None:
            QMessageBox.warning(self, "Warning", "No detection run loaded.")
            return False
        if not self._assignments:
            QMessageBox.warning(self, "Warning", "No track assignments defined.")
            return False
        if self._shot_id is None or self._sync_config_id is None:
            QMessageBox.warning(self, "Warning", "Run metadata missing.")
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
        root.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)

        self._frame_view = FrameViewWidget()
        self._frame_view.frame_changed.connect(self._on_frame_changed)
        splitter.addWidget(self._frame_view)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        stitcher_row = QSplitter(Qt.Horizontal)
        self._stitcher = StitcherWidget()
        self._stitcher.setMinimumHeight(150)
        self._stitcher.segment_clicked.connect(self._on_segment_clicked)
        self._stitcher.assignment_changed.connect(self._on_assignment_changed)
        self._stitcher.split_requested.connect(self._on_split_requested)
        self._stitcher.time_clicked.connect(self._frame_view.seek_global_time)
        stitcher_row.addWidget(self._stitcher)

        self._preview = PersonPreviewWidget()
        self._preview.setMinimumWidth(180)
        self._preview.setMaximumWidth(280)
        stitcher_row.addWidget(self._preview)
        stitcher_row.setStretchFactor(0, 1)
        stitcher_row.setStretchFactor(1, 0)
        right_layout.addWidget(stitcher_row, 1)

        self._frame_view.frame_data_ready.connect(self._preview.update_frame)

        assign_group = QGroupBox("Selected track")
        assign_layout = QVBoxLayout(assign_group)

        self._selected_label = QLabel("None selected")
        assign_layout.addWidget(self._selected_label)

        person_row = QHBoxLayout()
        person_row.addWidget(QLabel("Person:"))
        self._person_combo = QComboBox()
        self._person_combo.setEditable(True)
        self._person_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
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
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setToolTip("Finalise person assignments and create person sequences")
        self._apply_btn.clicked.connect(self.apply)
        btn_row.addWidget(self._apply_btn)
        assign_layout.addLayout(btn_row)

        right_layout.addWidget(assign_group)
        splitter.addWidget(right)
        splitter.setSizes([700, 700])

        root.addWidget(splitter, 1)

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
        self._frame_view.set_sync_table(sync_table)

        self._load_cameras()
        self._stitcher.load_run(self._conn, self._run_id)
        self._restore_assignments()
        self._last_applied = dict(self._assignments)

    def _load_cameras(self) -> None:
        if self._shot_id is None:
            return
        rows = self._conn.execute(
            "SELECT sv.id, sv.file_path, sv.actual_fps, ci.label "
            "FROM capture_videos sv "
            "JOIN camera_instances ci ON ci.id = sv.camera_instance_id "
            "WHERE sv.shot_id = ? ORDER BY ci.label",
            (self._shot_id,),
        ).fetchall()

        anchors: dict[str, tuple[int, float]] = {}
        if self._sync_config_id:
            for sp in self._conn.execute(
                "SELECT shot_video_id, video_frame, timestamp_s "
                "FROM sync_points WHERE sync_config_id = ? "
                "ORDER BY shot_video_id, video_frame",
                (self._sync_config_id,),
            ).fetchall():
                svid = sp["shot_video_id"]
                if svid not in anchors:
                    anchors[svid] = (int(sp["video_frame"]), float(sp["timestamp_s"]))

        cameras = []
        for r in rows:
            ref_frame, ref_ts = anchors.get(r["id"], (0, 0.0))
            cameras.append(CameraInfo(
                shot_video_id=r["id"],
                file_path=r["file_path"] or "",
                camera_instance_id=r["id"],
                label=r["label"] or r["id"][:8],
                fps=float(r["actual_fps"] or 30.0),
                ref_frame=ref_frame,
                ref_timestamp_s=ref_ts,
            ))
        self._frame_view.load_cameras(cameras)
        self._frame_view.set_pose_data(self._conn, self._run_id, None)

    def _restore_assignments(self) -> None:
        rows = self._conn.execute(
            "SELECT shot_video_id, track_id, person_name, first_frame, last_frame"
            " FROM detection_track_assignments"
            " WHERE detection_run_id = ?"
            " ORDER BY shot_video_id, track_id, first_frame",
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
    # Frame view / stitcher integration
    # ------------------------------------------------------------------

    def _on_frame_changed(self, frame_idx: int, global_s: float) -> None:
        self._current_frame_idx = frame_idx
        self._stitcher.set_current_time(global_s)
        self._sync_frame_view_assignments()

    def _on_segment_clicked(
        self,
        shot_video_id: str,
        track_id: int,
        first_frame: int,
        last_frame: int,
    ) -> None:
        self._current_svid = shot_video_id
        self._current_track_id = track_id
        self._current_seg_first = first_frame
        self._selected_label.setText(
            f"video: {shot_video_id[:8]}  track: {track_id}  "
            f"frames {first_frame}–{last_frame}"
        )
        person_name = self._assignments.get((shot_video_id, track_id, first_frame))
        self._preview.set_track(track_id, person_name)

        row = self._conn.execute(
            "SELECT file_path, camera_instance_id, actual_fps FROM capture_videos WHERE id=?",
            (shot_video_id,),
        ).fetchone()
        if row is None:
            return
        self._frame_view.load_camera(
            shot_video_id=shot_video_id,
            file_path=row["file_path"] or "",
            camera_instance_id=row["camera_instance_id"] or shot_video_id,
            fps=float(row["actual_fps"] or 30.0),
        )
        self._frame_view.set_pose_data(self._conn, self._run_id, track_id)
        self._frame_view.set_selected_track(track_id)
        self._sync_frame_view_assignments()

    def _sync_frame_view_assignments(self) -> None:
        if self._current_svid is None:
            return
        spans = self._stitcher.get_spans()
        f = self._current_frame_idx
        tid_to_person: dict[int, str] = {}
        for (svid, tid, sf), person in self._assignments.items():
            if svid != self._current_svid:
                continue
            seg_range = spans.get((svid, tid, sf))
            if seg_range and seg_range[0] <= f <= seg_range[1]:
                tid_to_person[tid] = person
        self._frame_view.set_track_assignments(tid_to_person)

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def _on_add_person(self) -> None:
        name, ok = QInputDialog.getText(self, "Add Person", "Person name:")
        if ok and name.strip():
            name = name.strip()
            if self._person_combo.findText(name) < 0:
                self._person_combo.addItem(name)
            self._person_combo.setCurrentText(name)
            persons = [self._person_combo.itemText(i) for i in range(self._person_combo.count())]
            self._stitcher.set_known_persons(persons)

    def _on_assign(self) -> None:
        if (self._current_svid is None
                or self._current_track_id is None
                or self._current_seg_first is None):
            QMessageBox.information(self, "No track selected", "Click a track segment first.")
            return
        name = self._person_combo.currentText().strip()
        if not name:
            QMessageBox.warning(self, "No name", "Enter a person name.")
            return
        self._do_assign(
            self._current_svid, self._current_track_id, self._current_seg_first, name
        )

    def _on_assignment_changed(self, svid: str, tid: int, seg_first: int, person_name: object) -> None:
        if person_name:
            self._do_assign(svid, tid, seg_first, str(person_name))
        else:
            self._detach(svid, tid, seg_first)

    def _on_split_requested(self, svid: str, tid: int, seg_first: int, split_frame: int) -> None:
        self._assignments.pop((svid, tid, split_frame), None)
        self._stitcher.split_segment(svid, tid, seg_first, split_frame)
        self._sync_frame_view_assignments()
        self._emit_dirty()

    def _do_assign(self, svid: str, tid: int, seg_first: int, name: str) -> None:
        seg_key = (svid, tid, seg_first)
        conflicts = find_assignment_conflicts(
            svid, [seg_key], name,
            self._stitcher.get_spans(), self._assignments,
        )
        if conflicts and not self._resolve_conflicts(conflicts, name):
            return
        for conflict_key in conflicts:
            self._detach(*conflict_key)
        self._apply_assignment(svid, tid, seg_first, name)

    def _resolve_conflicts(self, conflicts: list[tuple[str, int, int]], person_name: str) -> bool:
        msg = QMessageBox(self)
        msg.setWindowTitle("Assignment conflict")
        msg.setText(
            f"'{person_name}' is already assigned to {len(conflicts)} "
            f"overlapping segment(s) in this camera:"
        )
        lines = []
        spans = self._stitcher.get_spans()
        for key in conflicts:
            s, t, sf = key
            ff, lf = spans.get(key, (sf, sf))
            lines.append(f"  track {t}  frames {ff}–{lf}")
        msg.setInformativeText("\n".join(lines))
        detach_btn: QAbstractButton = msg.addButton("Detach conflicting segments", QMessageBox.AcceptRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.exec()
        return msg.clickedButton() is detach_btn

    def _apply_assignment(self, svid: str, tid: int, seg_first: int, name: str) -> None:
        self._assignments[(svid, tid, seg_first)] = name
        self._stitcher.set_segment_assignment(svid, tid, seg_first, name)
        if self._person_combo.findText(name) < 0:
            self._person_combo.addItem(name)
        persons = [self._person_combo.itemText(i) for i in range(self._person_combo.count())]
        self._stitcher.set_known_persons(persons)
        if (svid, tid, seg_first) == (self._current_svid, self._current_track_id,
                                       self._current_seg_first):
            self._preview.set_track(tid, name)
        self._sync_frame_view_assignments()
        self._emit_dirty()

    def _detach(self, svid: str, tid: int, seg_first: int) -> None:
        self._assignments.pop((svid, tid, seg_first), None)
        self._stitcher.set_segment_assignment(svid, tid, seg_first, None)
        if (svid, tid, seg_first) == (self._current_svid, self._current_track_id,
                                       self._current_seg_first):
            self._preview.set_track(tid, None)
        self._sync_frame_view_assignments()
        self._emit_dirty()

    def _emit_dirty(self) -> None:
        self.dirty_changed.emit(self.is_dirty)

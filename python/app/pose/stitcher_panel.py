"""stitcher_panel.py — Embeddable stitcher + assignment widget for a detection run.

Used by TrialPanel in the unified shell.  Exposes is_dirty / apply() for
prompt-on-leave and the Apply button.

Uses FilmstripStitcherWidget as its timeline.  FrameViewWidget and
PersonPreviewWidget are not used here; crop preview is handled by the
SegmentCropPanel on the right side of the splitter.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QByteArray, Qt, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.pose.finalise import TrackAssignment, finalise_to_db
from app.pose.filmstrip_stitcher import (
    FilmstripStitcherWidget,
    ROW_H_DEFAULT,
    ROW_H_MIN,
    ROW_H_MAX,
)
from app.pose.person_preview import draw_skeleton_qt
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


# ---------------------------------------------------------------------------
# SegmentCropPanel — right-side preview of the hovered bar frame
# ---------------------------------------------------------------------------

class SegmentCropPanel(QWidget):
    """Displays the person crop for the frame currently under the mouse cursor.

    Reads ``frame_cache_entries`` (JPEG crop) and ``detection_keypoints``
    (float32 blob) from the DB and draws the skeleton overlay using
    ``draw_skeleton_on_crop`` before displaying in a QLabel.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._conn = conn
        self._run_id = run_id

        self.setMinimumWidth(160)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._image_label.setMinimumSize(140, 120)
        self._image_label.setStyleSheet("background: #1a1a1a;")

        self._info_label = QLabel()
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setStyleSheet("font-size: 9px; color: #888;")
        self._info_label.setMaximumHeight(16)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        layout.addWidget(self._image_label, stretch=1)
        layout.addWidget(self._info_label)

        self._show_empty()

    def _show_empty(self, msg: str = "Hover over a bar") -> None:
        self._image_label.clear()
        self._image_label.setText(msg)
        self._image_label.setStyleSheet("background: #1a1a1a; color: #555;")
        self._info_label.setText("")

    def show_frame(
        self,
        svid: str,
        tid: int,
        frame_idx: int,
        person_name: str | None = None,
    ) -> None:
        """Load and display the crop for (svid, tid, frame_idx) with skeleton overlay.

        The JPEG is decoded directly into a QPixmap; keypoints are then drawn
        on top using QPainter (no OpenCV round-trip needed).
        """
        # Fetch the closest cached crop (±3 frames for tolerance)
        row = self._conn.execute(
            "SELECT image_data, height_px, src_x, src_y, src_h "
            "FROM frame_cache_entries "
            "WHERE shot_video_id = ? AND cache_type = 'person_crop' "
            "  AND track_id = ? AND detection_run_id = ? "
            "  AND frame_idx BETWEEN ? AND ? "
            "ORDER BY ABS(frame_idx - ?) LIMIT 1",
            (svid, tid, self._run_id,
             frame_idx - 3, frame_idx + 3, frame_idx),
        ).fetchone()

        if row is None:
            self._show_empty("No crop available")
            return

        # Decode JPEG directly to QPixmap — no OpenCV / numpy needed for the image
        pix = QPixmap()
        if not pix.loadFromData(QByteArray(bytes(row["image_data"]))) or pix.isNull():
            self._show_empty("Decode failed")
            return

        # src_h: height of the source crop region in original frame space.
        # The JPEG may be downscaled; scale = jpeg_h / src_h converts from frame
        # space to JPEG-pixel space.  After display scaling the combined factor is
        # simply disp_h / src_h.
        src_x = float(row["src_x"] or 0.0)
        src_y = float(row["src_y"] or 0.0)
        src_h = float(row["src_h"] or pix.height())

        # Scale to fit label while keeping aspect ratio
        label_size = self._image_label.size()
        if label_size.width() > 4 and label_size.height() > 4:
            pix = pix.scaled(
                label_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # Uniform scale from original-frame-crop coords to displayed pixels
        total_scale = pix.height() / src_h if src_h > 0 else 1.0

        # Fetch keypoints for this exact frame and overlay with QPainter
        kp_row = self._conn.execute(
            "SELECT keypoints FROM detection_keypoints "
            "WHERE detection_run_id = ? AND shot_video_id = ? "
            "  AND track_id = ? AND video_frame = ?",
            (self._run_id, svid, tid, frame_idx),
        ).fetchone()

        if kp_row is not None and kp_row["keypoints"]:
            kp_bytes = bytes(kp_row["keypoints"])
            n = len(kp_bytes) // (3 * 4)   # float32, 3 values per keypoint
            kp = np.frombuffer(kp_bytes, dtype=np.float32).reshape(n, 3)
            painter = QPainter(pix)
            draw_skeleton_qt(painter, kp, src_x, src_y, total_scale)
            painter.end()

        self._image_label.setPixmap(pix)
        self._image_label.setStyleSheet("background: #1a1a1a;")
        cam_short = svid[:8]
        person_str = f" · {person_name}" if person_name else ""
        self._info_label.setText(f"{cam_short} t{tid} f{frame_idx}{person_str}")


# ---------------------------------------------------------------------------
# StitcherPanel
# ---------------------------------------------------------------------------

class StitcherPanel(QWidget):
    """Filmstrip timeline + assignment panel for one detection run.

    Layout: splitter with FilmstripStitcherWidget on the left and
    SegmentCropPanel on the right.  Above the splitter is a header row
    with the view-mode selector; below is the status bar and assign group.

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

        # --- Header row: view mode selector ---
        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItem("By Detection", userData="detection")
        self._view_combo.addItem("By Person", userData="person")
        self._view_combo.currentIndexChanged.connect(self._on_view_mode_changed)
        header_row.addWidget(self._view_combo)
        header_row.addStretch()
        root.addLayout(header_row)

        # --- Splitter: filmstrip on left, crop preview on right ---
        self._stitcher = FilmstripStitcherWidget(row_h=ROW_H_DEFAULT)
        self._stitcher.segment_selected.connect(self._on_segment_selected)
        self._stitcher.assignment_changed.connect(self._on_assignment_changed)
        self._stitcher.time_hovered.connect(self._on_time_hovered)
        self._stitcher.bar_hovered.connect(self._on_bar_hovered)

        self._crop_panel = SegmentCropPanel(self._conn, self._run_id)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._stitcher)
        splitter.addWidget(self._crop_panel)
        splitter.setSizes([700, 200])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter, stretch=1)

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
    # View mode
    # ------------------------------------------------------------------

    def _on_view_mode_changed(self, index: int) -> None:
        mode = self._view_combo.itemData(index)
        self._stitcher.set_view_mode(mode)

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

    def _on_bar_hovered(
        self, svid: str, tid: int, seg_first: int, frame_idx: int
    ) -> None:
        """Forward bar-hover events to the side crop panel."""
        person_name = self._assignments.get((svid, tid, seg_first))
        self._crop_panel.show_frame(svid, tid, frame_idx, person_name)

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
        # Show first frame of selection in crop panel
        self._crop_panel.show_frame(svid, tid, sel_first, person_name)

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

        # Split left edge if selection starts inside the segment.
        # The widget propagates the original assignment to both halves via
        # segment_ops.split(); we mirror that into self._assignments here.
        target_sf = seg_first
        if sel_first > cur_first:
            orig = self._assignments.get((svid, tid, cur_first))
            self._stitcher.split_segment(svid, tid, cur_first, sel_first)
            if orig is not None:
                # Right half [sel_first, ...] inherits original assignment
                self._assignments[(svid, tid, sel_first)] = orig
            target_sf = sel_first

        # Split right edge if selection ends inside the segment.
        spans = self._stitcher.get_spans()
        target_range = spans.get((svid, tid, target_sf))
        if target_range and sel_last < target_range[1]:
            orig = self._assignments.get((svid, tid, target_sf))
            self._stitcher.split_segment(svid, tid, target_sf, sel_last + 1)
            if orig is not None:
                # Right half [sel_last+1, ...] inherits the assignment
                self._assignments[(svid, tid, sel_last + 1)] = orig

        self._apply_assignment(svid, tid, target_sf, name)
        self._auto_merge(svid, tid)
        self._refresh_conflicts(svid)
        # Rebuild by-person layout to reflect the changed assignment grouping
        self._stitcher.refresh_person_view()

    def _detach(self, svid: str, tid: int, seg_first: int) -> None:
        """Remove the person assignment for the active sub-range (or full bar).

        If the user dragged a sub-range selection before detaching, only that
        portion is detached; the rest of the bar retains its original assignment.
        Mirrors the same split logic used by _do_assign().
        """
        spans = self._stitcher.get_spans()
        seg_range = spans.get((svid, tid, seg_first))
        if seg_range is None:
            self._emit_dirty()
            return
        cur_first, cur_last = seg_range

        # Determine which frame range to detach
        if (
            self._selection is not None
            and self._selection.svid == svid
            and self._selection.tid == tid
            and self._selection.seg_first == seg_first
        ):
            sel_first = self._selection.sel_first
            sel_last = self._selection.sel_last
        else:
            sel_first, sel_last = cur_first, cur_last

        sel_first = max(sel_first, cur_first)
        sel_last = min(sel_last, cur_last)

        # Left split: carve off [cur_first, sel_first-1]
        target_sf = seg_first
        if sel_first > cur_first:
            orig = self._assignments.get((svid, tid, cur_first))
            self._stitcher.split_segment(svid, tid, cur_first, sel_first)
            if orig is not None:
                self._assignments[(svid, tid, sel_first)] = orig
            target_sf = sel_first

        # Right split: carve off [sel_last+1, cur_last]
        spans = self._stitcher.get_spans()
        target_range = spans.get((svid, tid, target_sf))
        if target_range is not None and sel_last < target_range[1]:
            orig = self._assignments.get((svid, tid, target_sf))
            self._stitcher.split_segment(svid, tid, target_sf, sel_last + 1)
            if orig is not None:
                self._assignments[(svid, tid, sel_last + 1)] = orig

        # Detach only the target range
        self._assignments.pop((svid, tid, target_sf), None)
        self._stitcher.set_segment_assignment(svid, tid, target_sf, None)

        self._auto_merge(svid, tid)
        self._refresh_conflicts(svid)
        self._stitcher.refresh_person_view()
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

    def _compute_conflicts(
        self,
    ) -> dict[tuple[str, int, int], list[tuple[int, int]]]:
        """Return per-bar conflict frame ranges.

        For each bar whose time range overlaps another bar assigned to the same
        person in the same camera, returns the exact overlapping frame ranges
        within that bar (using SyncTable lookup or linear interpolation).
        """
        time_spans = self._stitcher.get_time_spans()
        frame_spans = self._stitcher.get_spans()
        sync_table = self._stitcher._sync_table

        by_person_cam: dict[tuple[str, str], list] = defaultdict(list)
        for key, person in self._assignments.items():
            ts = time_spans.get(key)
            if ts is None:
                continue
            svid = key[0]
            by_person_cam[(person, svid)].append((ts[0], ts[1], key))

        def _t_to_frame(
            t: float,
            svid: str,
            seg_first: int,
            seg_last: int,
            seg_t0: float,
            seg_t1: float,
        ) -> int:
            if sync_table is not None:
                f = sync_table.lookup(t, svid)
                if f is not None:
                    return max(seg_first, min(seg_last, f))
            span = max(1e-6, seg_t1 - seg_t0)
            frac = (t - seg_t0) / span
            return max(seg_first, min(seg_last,
                       seg_first + round(frac * (seg_last - seg_first))))

        conflict_ranges: dict[tuple[str, int, int], list[tuple[int, int]]] = {}

        for intervals in by_person_cam.values():
            intervals.sort()
            for i, (t0a, t1a, ka) in enumerate(intervals):
                for t0b, t1b, kb in intervals[i + 1:]:
                    if t0b >= t1a:
                        break
                    t_over_s = max(t0a, t0b)
                    t_over_e = min(t1a, t1b)
                    for key, seg_t0, seg_t1 in [(ka, t0a, t1a), (kb, t0b, t1b)]:
                        svid = key[0]
                        fs = frame_spans.get(key)
                        if fs is None:
                            continue
                        seg_first, seg_last = fs
                        f_s = _t_to_frame(t_over_s, svid, seg_first, seg_last,
                                          seg_t0, seg_t1)
                        f_e = _t_to_frame(t_over_e, svid, seg_first, seg_last,
                                          seg_t0, seg_t1)
                        if f_e >= f_s:
                            conflict_ranges.setdefault(key, []).append((f_s, f_e))

        # Merge overlapping ranges within each bar
        for key in conflict_ranges:
            sorted_r = sorted(conflict_ranges[key])
            merged: list[list[int]] = []
            for r in sorted_r:
                if merged and r[0] <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], r[1])
                else:
                    merged.append([r[0], r[1]])
            conflict_ranges[key] = [(r[0], r[1]) for r in merged]

        return conflict_ranges

    def _refresh_conflicts(self, _svid: str | None = None) -> None:
        conflict_ranges = self._compute_conflicts()
        self._stitcher.set_conflict_segments(conflict_ranges)
        n = len(conflict_ranges)
        if n:
            self._conflict_label.setText(f"⚠ {n} conflict(s)")
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

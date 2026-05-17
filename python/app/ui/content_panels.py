"""content_panels.py — Right-pane detail panels for each tree item type."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> QGroupBox:
    box = QGroupBox(title)
    box.setLayout(QVBoxLayout())
    box.layout().setSpacing(2)
    return box


def _form_row(label: str, value: str) -> tuple[QLabel, QLabel]:
    lbl = QLabel(label)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    val = QLabel(value)
    val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    val.setWordWrap(True)
    return lbl, val


def _action_btn(text: str, enabled: bool = True) -> QPushButton:
    btn = QPushButton(text)
    btn.setEnabled(enabled)
    btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return btn


def _scrollable(inner: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(inner)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    return area


def _fmt_ts(ts: str | None) -> str:
    return ts[:16].replace("T", " ") if ts else "—"


def _fmt_time(s: float | None) -> str:
    return f"{s:.3f} s" if s is not None else "—"


# ---------------------------------------------------------------------------
# CapturePanel
# ---------------------------------------------------------------------------


class CapturePanel(QWidget):
    """Detail view for a capture (captures row)."""

    def __init__(self, conn: sqlite3.Connection, capture_id: str,
                 session_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._capture_id = capture_id
        self._session_path = session_path
        self._build()

    def _build(self) -> None:
        conn = self._conn
        cap = conn.execute(
            "SELECT id, label, capture_number, notes, extrinsic_calibration_id "
            "FROM captures WHERE id = ?", (self._capture_id,)
        ).fetchone()
        if cap is None:
            return

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = cap["label"] or f"Capture {cap['capture_number']}"
        lbl = QLabel(f"<h2>{title}</h2>")
        vbox.addWidget(lbl)

        # Basic info
        info = _section("Capture info")
        form = QFormLayout()
        form.addRow("Number:", QLabel(str(cap["capture_number"])))
        form.addRow("Notes:", QLabel(cap["notes"] or "—"))
        ext_id = cap["extrinsic_calibration_id"]
        ext_label = "✓ set" if ext_id else "✗ not set"
        form.addRow("Extrinsics:", QLabel(ext_label))
        info.layout().addLayout(form)
        vbox.addWidget(info)

        # Videos
        videos = conn.execute(
            "SELECT cv.file_path, cv.actual_fps, cv.first_video_frame, cv.last_video_frame, "
            "       ci.label AS cam_label "
            "FROM capture_videos cv "
            "LEFT JOIN camera_instances ci ON ci.id = cv.camera_instance_id "
            "WHERE cv.shot_id = ? ORDER BY ci.label",
            (self._capture_id,),
        ).fetchall()
        vid_box = _section(f"Videos ({len(videos)})")
        for v in videos:
            cam = v["cam_label"] or "unknown camera"
            n_frames = (v["last_video_frame"] or 0) - (v["first_video_frame"] or 0) + 1
            lbl_text = (
                f"<b>{cam}</b>  {v['actual_fps']:.2f} fps  "
                f"{n_frames} frames<br>"
                f"<small>{v['file_path']}</small>"
            )
            lbl = QLabel(lbl_text)
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            vid_box.layout().addWidget(lbl)
        if not videos:
            vid_box.layout().addWidget(QLabel("No videos attached."))
        vbox.addWidget(vid_box)

        # Sync configs
        syncs = conn.execute(
            "SELECT id, created_by, notes FROM sync_configs WHERE shot_id = ? ORDER BY rowid",
            (self._capture_id,),
        ).fetchall()
        sync_box = _section(f"Sync configs ({len(syncs)})")
        for s in syncs:
            sync_box.layout().addWidget(
                QLabel(f"<b>{s['created_by'] or '—'}</b>  {s['notes'] or ''}")
            )
        if not syncs:
            sync_box.layout().addWidget(QLabel("No sync config — set one up before running detection."))
        vbox.addWidget(sync_box)

        # Actions
        btn_row = QHBoxLayout()
        sync_btn = _action_btn("Set up sync…")
        sync_btn.clicked.connect(self._open_sync)
        btn_row.addWidget(sync_btn)

        ext_btn = _action_btn("Import extrinsics…")
        ext_btn.clicked.connect(self._open_extrinsics)
        btn_row.addWidget(ext_btn)

        pose_btn = _action_btn("Open in Pose Extraction…")
        pose_btn.clicked.connect(self._open_pose_extraction)
        btn_row.addWidget(pose_btn)

        btn_row.addStretch()
        vbox.addLayout(btn_row)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(_scrollable(inner))

    def _open_sync(self) -> None:
        from app.setup.db_context import DBContext
        from app.setup.page_sync import SyncDialog
        session_row = self._conn.execute(
            "SELECT id FROM mocap_sessions LIMIT 1"
        ).fetchone()
        if session_row is None:
            return
        ctx = DBContext(self._conn, session_row["id"])
        dlg = SyncDialog(ctx, self._capture_id, parent=self)
        dlg.exec()

    def _open_extrinsics(self) -> None:
        from app.setup.page_extrinsics import ExtrinsicsImportDialog
        session_row = self._conn.execute(
            "SELECT id FROM mocap_sessions LIMIT 1"
        ).fetchone()
        if session_row is None:
            return
        dlg = ExtrinsicsImportDialog(
            self._conn, session_row["id"],
            shot_ids=[self._capture_id],
            parent=self,
        )
        if dlg.exec():
            pass  # tree reloads when user returns to main window

    def _open_pose_extraction(self) -> None:
        from app.pose.main import PoseExtractionWindow
        from app.ui.main_window import MainWindow
        self._pose_win = PoseExtractionWindow(session_db=str(self._session_path), parent=None)
        main = self.window()
        if isinstance(main, MainWindow):
            self._pose_win.data_changed.connect(main.reload_tree)
        self._pose_win.show()


# ---------------------------------------------------------------------------
# TrialPanel
# ---------------------------------------------------------------------------


class TrialPanel(QWidget):
    """Detail view for a trial (trials row)."""

    def __init__(self, conn: sqlite3.Connection, trial_id: str, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._trial_id = trial_id
        self._build()

    def _build(self) -> None:
        trial = self._conn.execute(
            "SELECT id, name, time_start_s, time_end_s, notes "
            "FROM trials WHERE id = ?", (self._trial_id,)
        ).fetchone()
        if trial is None:
            return

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = trial["name"] or "Unnamed trial"
        vbox.addWidget(QLabel(f"<h2>{title}</h2>"))

        form_box = _section("Trial info")
        form = QFormLayout()
        form.addRow("Start:", QLabel(_fmt_time(trial["time_start_s"])))
        form.addRow("End:", QLabel(_fmt_time(trial["time_end_s"])))
        form.addRow("Notes:", QLabel(trial["notes"] or "—"))
        form_box.layout().addLayout(form)
        vbox.addWidget(form_box)

        # Detection runs in this trial
        runs = self._conn.execute(
            "SELECT id, detector_model, status, created_at "
            "FROM detection_runs WHERE trial_id = ? ORDER BY created_at",
            (self._trial_id,),
        ).fetchall()
        dr_box = _section(f"Detection runs ({len(runs)})")
        for r in runs:
            dr_box.layout().addWidget(
                QLabel(f"[{r['detector_model']}]  {_fmt_ts(r['created_at'])}  ({r['status']})")
            )
        if not runs:
            dr_box.layout().addWidget(QLabel("No detection runs yet."))
        vbox.addWidget(dr_box)

        btn_row = QHBoxLayout()
        run_det_btn = _action_btn("Run detection…", enabled=False)  # T3.4 future
        btn_row.addWidget(run_det_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(_scrollable(inner))


# ---------------------------------------------------------------------------
# DetectionRunPanel
# ---------------------------------------------------------------------------


class DetectionRunPanel(QWidget):
    """Detail view for a detection run."""

    def __init__(self, conn: sqlite3.Connection, run_id: str,
                 session_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._run_id = run_id
        self._session_path = session_path
        self._build()

    def _build(self) -> None:
        run = self._conn.execute(
            "SELECT id, shot_id, sync_config_id, detector_model, pose_model, "
            "       status, time_start_s, time_end_s, created_at, completed_at "
            "FROM detection_runs WHERE id = ?", (self._run_id,)
        ).fetchone()
        if run is None:
            return

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setAlignment(Qt.AlignmentFlag.AlignTop)

        vbox.addWidget(QLabel(f"<h2>Detection [{run['detector_model']}]</h2>"))

        form_box = _section("Run info")
        form = QFormLayout()
        form.addRow("Status:", QLabel(run["status"]))
        form.addRow("Detector:", QLabel(run["detector_model"]))
        form.addRow("Pose model:", QLabel(run["pose_model"] or "—"))
        form.addRow("Time range:", QLabel(
            f"{_fmt_time(run['time_start_s'])}  →  {_fmt_time(run['time_end_s'])}"
        ))
        form.addRow("Started:", QLabel(_fmt_ts(run["created_at"])))
        form.addRow("Completed:", QLabel(_fmt_ts(run["completed_at"])))
        form_box.layout().addLayout(form)
        vbox.addWidget(form_box)

        # Person tracks produced by this run
        tracks = self._conn.execute(
            "SELECT pos.id, GROUP_CONCAT(sp.person_name, ', ') AS names "
            "FROM pose_observation_sequences pos "
            "LEFT JOIN sequence_persons sp ON sp.sequence_id = pos.id "
            "WHERE pos.detection_run_id = ? GROUP BY pos.id",
            (self._run_id,),
        ).fetchall()
        tr_box = _section(f"Person tracks ({len(tracks)})")
        for t in tracks:
            tr_box.layout().addWidget(QLabel(t["names"] or "Unnamed track"))
        if not tracks:
            tr_box.layout().addWidget(QLabel(
                "No person tracks yet — finalise to assign persons."
            ))
        vbox.addWidget(tr_box)

        btn_row = QHBoxLayout()
        open_btn = _action_btn("Open in Pose Extraction…")
        open_btn.clicked.connect(self._open_pose_extraction)
        btn_row.addWidget(open_btn)

        finalise_btn = _action_btn("Finalise → person tracks…",
                                   enabled=(run["status"] == "complete" and not tracks))
        finalise_btn.setToolTip(
            "Open Pose Extraction and use the Finalise workflow there."
        )
        finalise_btn.clicked.connect(self._open_pose_extraction)
        btn_row.addWidget(finalise_btn)

        btn_row.addStretch()
        vbox.addLayout(btn_row)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(_scrollable(inner))

    def _open_pose_extraction(self) -> None:
        from app.pose.main import PoseExtractionWindow
        from app.ui.main_window import MainWindow
        self._pose_win = PoseExtractionWindow(
            session_db=str(self._session_path),
            parent=None,
        )
        main = self.window()
        if isinstance(main, MainWindow):
            self._pose_win.data_changed.connect(main.reload_tree)
        self._pose_win.show()


# ---------------------------------------------------------------------------
# PersonPanel
# ---------------------------------------------------------------------------


_EXPORT_BVH_SCRIPT = Path(__file__).resolve().parents[3] / "python" / "tools" / "export_bvh.py"


class PersonPanel(QWidget):
    """Person panel: info, tracking history, and tracker launcher."""

    def __init__(self, conn: sqlite3.Connection, sequence_id: str,
                 session_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._sequence_id = sequence_id
        self._session_path = session_path
        self._bvh_proc: QProcess | None = None
        self._build()

    def _build(self) -> None:
        seq = self._conn.execute(
            "SELECT id, name, time_start_s, time_end_s, pose_model, notes "
            "FROM pose_observation_sequences WHERE id = ?",
            (self._sequence_id,),
        ).fetchone()
        if seq is None:
            return

        person_names = self._conn.execute(
            "SELECT GROUP_CONCAT(person_name, ', ') AS names "
            "FROM sequence_persons WHERE sequence_id = ?",
            (self._sequence_id,),
        ).fetchone()["names"]

        n_obs = self._conn.execute(
            "SELECT COUNT(DISTINCT video_frame || camera_instance_id) "
            "FROM pose_observations WHERE sequence_id = ?",
            (self._sequence_id,),
        ).fetchone()[0]

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = person_names or seq["name"] or "Person"
        vbox.addWidget(QLabel(f"<h2>{title}</h2>"))

        form_box = _section("Person info")
        form = QFormLayout()
        form.addRow("Persons:", QLabel(person_names or "—"))
        form.addRow("Time range:", QLabel(
            f"{_fmt_time(seq['time_start_s'])}  →  {_fmt_time(seq['time_end_s'])}"
        ))
        form.addRow("Observations:", QLabel(str(n_obs)))
        form.addRow("Pose model:", QLabel(seq["pose_model"] or "—"))
        form_box.layout().addLayout(form)
        vbox.addWidget(form_box)

        # --- Tracking runs section ---
        self._run_box = _section("Tracking runs (0)")
        box_vbox = self._run_box.layout()

        self._run_list = QListWidget()
        self._run_list.setMaximumHeight(110)
        self._run_list.currentItemChanged.connect(self._on_run_selected)
        box_vbox.addWidget(self._run_list)

        self._run_detail = QLabel("")
        self._run_detail.setWordWrap(True)
        self._run_detail.setVisible(False)
        box_vbox.addWidget(self._run_detail)

        run_act_row = QHBoxLayout()
        run_act_row.setContentsMargins(0, 2, 0, 0)
        self._export_bvh_btn = QPushButton("Export BVH…")
        self._export_bvh_btn.setEnabled(False)
        self._export_bvh_btn.clicked.connect(self._export_bvh)
        self._delete_run_btn = QPushButton("Delete run")
        self._delete_run_btn.setEnabled(False)
        self._delete_run_btn.clicked.connect(self._delete_run)
        run_act_row.addStretch()
        run_act_row.addWidget(self._export_bvh_btn)
        run_act_row.addWidget(self._delete_run_btn)
        box_vbox.addLayout(run_act_row)

        vbox.addWidget(self._run_box)

        self._refresh_runs()

        btn_row = QHBoxLayout()
        run_btn = _action_btn("Run tracker…")
        run_btn.clicked.connect(self._open_run_tracker)
        btn_row.addWidget(run_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        scroll = _scrollable(inner)
        scroll.setMaximumHeight(320)

        frame_box = self._build_frame_view()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        root.addWidget(scroll)
        root.addWidget(frame_box, stretch=1)

    # ------------------------------------------------------------------
    # Frame view
    # ------------------------------------------------------------------

    def _build_frame_view(self) -> QWidget:
        import numpy as np
        from app.pose.frame_view import CameraInfo, FrameViewWidget
        from app.setup.db_context import SyncPoint, SyncTable

        box = _section("Detection frames")
        self._frame_view = FrameViewWidget()
        box.layout().addWidget(self._frame_view)

        seq = self._conn.execute(
            "SELECT shot_id, sync_config_id, time_start_s "
            "FROM pose_observation_sequences WHERE id = ?",
            (self._sequence_id,),
        ).fetchone()
        if seq is None:
            return box

        # Capture videos for this shot
        videos = self._conn.execute(
            "SELECT cv.id, cv.file_path, cv.actual_fps, cv.camera_instance_id, "
            "       COALESCE(ci.label, cv.camera_instance_id) AS cam_label "
            "FROM capture_videos cv "
            "LEFT JOIN camera_instances ci ON ci.id = cv.camera_instance_id "
            "WHERE cv.shot_id = ? ORDER BY ci.label",
            (seq["shot_id"],),
        ).fetchall()
        if not videos:
            return box

        # Sync table
        sp_rows = self._conn.execute(
            "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, cv.actual_fps "
            "FROM sync_points sp "
            "JOIN capture_videos cv ON cv.id = sp.shot_video_id "
            "WHERE sp.sync_config_id = ?",
            (seq["sync_config_id"],),
        ).fetchall()
        if sp_rows:
            points = [
                SyncPoint(
                    camera_instance_id="",
                    shot_video_id=r["shot_video_id"],
                    video_frame=r["video_frame"],
                    timestamp_s=r["timestamp_s"],
                )
                for r in sp_rows
            ]
            fps_by_video = {r["shot_video_id"]: float(r["actual_fps"]) for r in sp_rows}
            self._frame_view.set_sync_table(SyncTable(points, fps_by_video))

        # Observation keypoints: {shot_video_id: {video_frame: kp [N,3]}}
        self._obs_kp: dict[str, dict[int, np.ndarray]] = {}
        svid_by_cam = {v["camera_instance_id"]: v["id"] for v in videos}
        for row in self._conn.execute(
            "SELECT camera_instance_id, video_frame, kp_blob "
            "FROM pose_observations WHERE sequence_id = ? AND person_id = 0",
            (self._sequence_id,),
        ):
            svid = svid_by_cam.get(row["camera_instance_id"])
            if svid is None:
                continue
            raw = bytes(row["kp_blob"])
            n = len(raw) // 12  # 3 × float32
            kp = np.frombuffer(raw, dtype=np.float32).reshape(n, 3)
            self._obs_kp.setdefault(svid, {})[row["video_frame"]] = kp

        cameras = [
            CameraInfo(
                shot_video_id=v["id"],
                file_path=v["file_path"],
                camera_instance_id=v["camera_instance_id"],
                label=v["cam_label"] or v["id"][:8],
                fps=float(v["actual_fps"]),
                ref_frame=0,
                ref_timestamp_s=0.0,
            )
            for v in videos
        ]

        self._frame_view.load_cameras(cameras)
        # Set keypoints for the initially selected camera before connecting
        # camera_switched so the initial seek already has overlay data
        if cameras:
            self._frame_view.set_observation_keypoints(
                self._obs_kp.get(cameras[0].shot_video_id, {})
            )
        self._frame_view.camera_switched.connect(self._on_camera_switched)

        if seq["time_start_s"] is not None:
            self._frame_view.seek_global_time(float(seq["time_start_s"]))

        return box

    def _on_camera_switched(self, shot_video_id: str) -> None:
        self._frame_view.set_observation_keypoints(
            self._obs_kp.get(shot_video_id, {})
        )

    # ------------------------------------------------------------------
    # Tracking runs list
    # ------------------------------------------------------------------

    def _refresh_runs(self) -> None:
        self._run_list.clear()
        self._run_detail.setVisible(False)
        self._export_bvh_btn.setEnabled(False)
        self._delete_run_btn.setEnabled(False)

        runs = self._conn.execute(
            "SELECT tr.id, tr.ran_at, s.name AS skel_name "
            "FROM tracking_runs tr "
            "LEFT JOIN skeletons s ON s.id = tr.skeleton_id "
            "WHERE tr.observation_sequence_id = ? ORDER BY tr.ran_at DESC",
            (self._sequence_id,),
        ).fetchall()
        self._run_box.setTitle(f"Tracking runs ({len(runs)})")
        if not runs:
            self._run_list.addItem("No tracking runs yet.")
            return
        for r in runs:
            stats = self._conn.execute(
                "SELECT COUNT(*) AS total, "
                "       SUM(CASE WHEN tracking_lost=0 THEN 1 ELSE 0 END) AS tracked "
                "FROM tracking_results WHERE run_id=? AND person_id=0 AND is_smoothed=0",
                (r["id"],),
            ).fetchone()
            label = f"[{r['skel_name'] or '?'}]  {_fmt_ts(r['ran_at'])}"
            if stats and stats["total"]:
                pct = 100.0 * (stats["tracked"] or 0) / stats["total"]
                label += f"  —  {stats['tracked']}/{stats['total']} frames ({pct:.0f}%)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, r["id"])
            self._run_list.addItem(item)

    def _on_run_selected(self, current: QListWidgetItem, _prev) -> None:
        run_id = current.data(Qt.ItemDataRole.UserRole) if current else None
        if not run_id:
            self._run_detail.setVisible(False)
            self._export_bvh_btn.setEnabled(False)
            self._delete_run_btn.setEnabled(False)
            return

        run = self._conn.execute(
            "SELECT tr.ran_at, tr.notes, s.name AS skel_name "
            "FROM tracking_runs tr "
            "LEFT JOIN skeletons s ON s.id = tr.skeleton_id "
            "WHERE tr.id = ?",
            (run_id,),
        ).fetchone()
        stats = self._conn.execute(
            "SELECT COUNT(*) AS total, "
            "       SUM(CASE WHEN tracking_lost=0 THEN 1 ELSE 0 END) AS tracked, "
            "       AVG(COALESCE(n_inlier_observations, 0)) AS avg_inliers "
            "FROM tracking_results WHERE run_id=? AND person_id=0 AND is_smoothed=0",
            (run_id,),
        ).fetchone()

        if run:
            if stats and stats["total"]:
                total = stats["total"]
                tracked = stats["tracked"] or 0
                pct = 100.0 * tracked / total
                avg = stats["avg_inliers"] or 0.0
                stat_line = f"{tracked}/{total} frames ({pct:.1f}%)  —  avg inliers: {avg:.1f}"
            else:
                stat_line = "no frame stats"
            self._run_detail.setText(
                f"<b>{run['skel_name'] or '?'}</b>  {_fmt_ts(run['ran_at'])}<br>"
                f"Frames: {stat_line}<br>"
                f"Notes: {run['notes'] or '—'}"
            )
            self._run_detail.setVisible(True)

        self._export_bvh_btn.setEnabled(True)
        self._delete_run_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # BVH export
    # ------------------------------------------------------------------

    def _export_bvh(self) -> None:
        item = self._run_list.currentItem()
        run_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not run_id:
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save BVH file", "", "BVH files (*.bvh)"
        )
        if not out_path:
            return

        self._bvh_proc = QProcess(self)
        self._bvh_proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._export_bvh_btn.setEnabled(False)

        proc = self._bvh_proc

        def _done(code: int, _status) -> None:
            self._export_bvh_btn.setEnabled(True)
            if code == 0:
                QMessageBox.information(
                    self, "Export complete", f"BVH written to:\n{out_path}"
                )
            else:
                output = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
                QMessageBox.critical(
                    self, "Export failed",
                    f"export_bvh.py exited with code {code}.\n\n{output[-800:]}",
                )

        self._bvh_proc.finished.connect(_done)
        self._bvh_proc.start(
            sys.executable,
            [
                str(_EXPORT_BVH_SCRIPT),
                "--session-db", str(self._session_path),
                "--run-id",     run_id,
                "--person-id",  "0",
                "--smoothed",
                "--output",     out_path,
            ],
        )

    # ------------------------------------------------------------------
    # Delete run
    # ------------------------------------------------------------------

    def _delete_run(self) -> None:
        item = self._run_list.currentItem()
        run_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not run_id:
            return
        if QMessageBox.question(
            self,
            "Delete tracking run",
            "Delete this tracking run and all its results?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        with self._conn:
            self._conn.execute(
                "DELETE FROM tracking_results WHERE run_id = ?", (run_id,)
            )
            self._conn.execute(
                "DELETE FROM tracking_obs_results WHERE run_id = ?", (run_id,)
            )
            self._conn.execute(
                "DELETE FROM tracking_runs WHERE id = ?", (run_id,)
            )
        self._refresh_runs()

    # ------------------------------------------------------------------
    # Tracker dialog
    # ------------------------------------------------------------------

    def _open_run_tracker(self) -> None:
        from app.pose.run_tracker import RunTrackerDialog
        dlg = RunTrackerDialog(
            conn=self._conn,
            session_path=str(self._session_path),
            sequence_id=self._sequence_id,
            parent=self,
        )
        dlg.exec()
        self._refresh_runs()


# ---------------------------------------------------------------------------
# TrackingRunPanel
# ---------------------------------------------------------------------------


class TrackingRunPanel(QWidget):
    """Detail view for a tracking run."""

    def __init__(self, conn: sqlite3.Connection, run_id: str, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._run_id = run_id
        self._build()

    def _build(self) -> None:
        run = self._conn.execute(
            "SELECT tr.id, tr.ran_at, tr.notes, tr.posetrak_version, "
            "       tr.active_camera_ids, tr.marker_names, "
            "       s.name AS skel_name "
            "FROM tracking_runs tr "
            "LEFT JOIN skeletons s ON s.id = tr.skeleton_id "
            "WHERE tr.id = ?",
            (self._run_id,),
        ).fetchone()
        if run is None:
            return

        n_frames = self._conn.execute(
            "SELECT COUNT(*) FROM tracking_results WHERE run_id = ? AND is_smoothed = 0",
            (self._run_id,),
        ).fetchone()[0]

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setAlignment(Qt.AlignmentFlag.AlignTop)

        skel = run["skel_name"] or "?"
        vbox.addWidget(QLabel(f"<h2>Tracking run  [{skel}]</h2>"))

        form_box = _section("Run info")
        form = QFormLayout()
        form.addRow("Skeleton:", QLabel(skel))
        form.addRow("Ran at:", QLabel(_fmt_ts(run["ran_at"])))
        form.addRow("Version:", QLabel(run["posetrak_version"] or "—"))
        form.addRow("Frames:", QLabel(str(n_frames)))
        try:
            cam_ids = json.loads(run["active_camera_ids"] or "[]")
            form.addRow("Cameras:", QLabel(", ".join(cam_ids) or "—"))
        except Exception:
            pass
        form.addRow("Notes:", QLabel(run["notes"] or "—"))
        form_box.layout().addLayout(form)
        vbox.addWidget(form_box)

        btn_row = QHBoxLayout()
        view_btn = _action_btn("View results…", enabled=False)
        view_btn.setToolTip("Phase 5: results visualiser")
        btn_row.addWidget(view_btn)

        export_btn = _action_btn("Export BVH…", enabled=False)
        export_btn.setToolTip("Not yet wired in this UI")
        btn_row.addWidget(export_btn)

        btn_row.addStretch()
        vbox.addLayout(btn_row)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(_scrollable(inner))

"""content_panels.py — Right-pane detail panels for each tree item type."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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
        sync_btn = _action_btn("Set up sync…", enabled=False)  # T3.5
        sync_btn.setToolTip("Available after T3.5 (wizard integration)")
        btn_row.addWidget(sync_btn)

        ext_btn = _action_btn("Import extrinsics…")
        ext_btn.clicked.connect(self._open_extrinsics)
        btn_row.addWidget(ext_btn)

        btn_row.addStretch()
        vbox.addLayout(btn_row)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(_scrollable(inner))

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
        win = PoseExtractionWindow(
            session_db=str(self._session_path),
            parent=None,
        )
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        win.show()


# ---------------------------------------------------------------------------
# PersonTrackPanel
# ---------------------------------------------------------------------------


class PersonTrackPanel(QWidget):
    """Detail view for a person track (pose_observation_sequences row)."""

    def __init__(self, conn: sqlite3.Connection, sequence_id: str,
                 session_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._conn = conn
        self._sequence_id = sequence_id
        self._session_path = session_path
        self._build()

    def _build(self) -> None:
        seq = self._conn.execute(
            "SELECT pos.id, pos.name, pos.time_start_s, pos.time_end_s, pos.pose_model, "
            "       pos.notes, "
            "       GROUP_CONCAT(sp.person_name, ', ') AS person_names, "
            "       COUNT(DISTINCT po.video_frame || po.camera_instance_id) AS n_obs "
            "FROM pose_observation_sequences pos "
            "LEFT JOIN sequence_persons sp ON sp.sequence_id = pos.id "
            "LEFT JOIN pose_observations po ON po.sequence_id = pos.id "
            "WHERE pos.id = ? GROUP BY pos.id",
            (self._sequence_id,),
        ).fetchone()
        if seq is None:
            return

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = seq["person_names"] or seq["name"] or "Person track"
        vbox.addWidget(QLabel(f"<h2>{title}</h2>"))

        form_box = _section("Track info")
        form = QFormLayout()
        form.addRow("Persons:", QLabel(seq["person_names"] or "—"))
        form.addRow("Time range:", QLabel(
            f"{_fmt_time(seq['time_start_s'])}  →  {_fmt_time(seq['time_end_s'])}"
        ))
        form.addRow("Observations:", QLabel(str(seq["n_obs"] or 0)))
        form.addRow("Pose model:", QLabel(seq["pose_model"] or "—"))
        form.addRow("Notes:", QLabel(seq["notes"] or "—"))
        form_box.layout().addLayout(form)
        vbox.addWidget(form_box)

        # Tracking runs for this sequence
        runs = self._conn.execute(
            "SELECT tr.id, tr.ran_at, tr.notes, s.name AS skel_name "
            "FROM tracking_runs tr "
            "LEFT JOIN skeletons s ON s.id = tr.skeleton_id "
            "WHERE tr.observation_sequence_id = ? ORDER BY tr.ran_at",
            (self._sequence_id,),
        ).fetchall()
        run_box = _section(f"Tracking runs ({len(runs)})")
        for r in runs:
            run_box.layout().addWidget(
                QLabel(f"[{r['skel_name'] or '?'}]  {_fmt_ts(r['ran_at'])}")
            )
        if not runs:
            run_box.layout().addWidget(QLabel("No tracking runs yet."))
        vbox.addWidget(run_box)

        btn_row = QHBoxLayout()
        run_btn = _action_btn("Run tracker…")
        run_btn.clicked.connect(self._open_run_tracker)
        btn_row.addWidget(run_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(_scrollable(inner))

    def _open_run_tracker(self) -> None:
        from app.pose.run_tracker import RunTrackerDialog
        dlg = RunTrackerDialog(
            conn=self._conn,
            session_path=str(self._session_path),
            parent=self,
        )
        dlg.exec()


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

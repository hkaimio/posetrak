# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""run_detection_dialog.py — Modal dialog to configure and launch a detection run.

Opens from CapturePanel (creates a new trial) or TrialPanel (uses an existing
trial).  On success it links the new detection_run to the trial and emits
detection_finished(trial_id, run_id) so the caller can refresh the session tree.
"""
from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
)

from posetrak.db.db import generate_id


class RunDetectionDialog(QDialog):
    """Configure model, time range, and (optionally) trial name; run detection in background.

    When *trial_id* is provided the dialog runs detection against an existing
    trial; no new trial row is created.  When *trial_id* is None a new trial row
    is created after detection completes (legacy path from CapturePanel).
    """

    detection_finished = Signal(str, str)  # trial_id, run_id

    def __init__(
        self,
        conn: sqlite3.Connection,
        session_path: Path,
        capture_id: str | None = None,
        time_start_s: float | None = None,
        time_end_s: float | None = None,
        trial_id: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run Detection")
        self.setMinimumWidth(520)
        self._conn = conn
        self._session_path = session_path
        self._trial_id = trial_id
        self._job = None
        self._seg_runner = None
        self._seg_detection_run_id: str | None = None
        self._seg_persons_ordered: list[str] = []
        self._seg_jobs_total = 0
        self._seg_jobs_done = 0

        if trial_id is not None:
            trial = conn.execute(
                "SELECT capture_id, time_start_s, time_end_s FROM trials WHERE id = ?",
                (trial_id,),
            ).fetchone()
            if trial:
                capture_id = capture_id or trial["capture_id"]
                if time_start_s is None:
                    time_start_s = trial["time_start_s"]
                if time_end_s is None:
                    time_end_s = trial["time_end_s"]

        self._capture_id = capture_id
        self._build_ui(time_start_s, time_end_s)

    # ------------------------------------------------------------------

    def _build_ui(self, time_start_s: float | None, time_end_s: float | None) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Trial name — only shown when creating a new trial
        if self._trial_id is None:
            trial_count = self._conn.execute(
                "SELECT COUNT(*) FROM trials WHERE capture_id = ?", (self._capture_id,)
            ).fetchone()[0]
            self._trial_name = QLineEdit(f"Trial {trial_count + 1}")
            form.addRow("Trial name:", self._trial_name)
        else:
            self._trial_name = None

        # Sync config
        syncs = self._conn.execute(
            "SELECT id, created_by, notes FROM sync_configs WHERE shot_id = ? ORDER BY rowid",
            (self._capture_id,),
        ).fetchall()
        self._sync_combo = QComboBox()
        for s in syncs:
            label = s["created_by"] or "sync"
            if s["notes"]:
                label += f" — {s['notes']}"
            self._sync_combo.addItem(label, s["id"])
        form.addRow("Sync config:", self._sync_combo)

        # Time range
        self._start_spin = QDoubleSpinBox()
        self._start_spin.setRange(0.0, 100_000.0)
        self._start_spin.setDecimals(2)
        self._start_spin.setSuffix(" s")
        self._start_spin.setValue(time_start_s if time_start_s is not None else 0.0)

        self._end_spin = QDoubleSpinBox()
        self._end_spin.setRange(0.0, 100_000.0)
        self._end_spin.setDecimals(2)
        self._end_spin.setSuffix(" s")
        self._end_spin.setValue(time_end_s if time_end_s is not None else 0.0)

        time_row = QHBoxLayout()
        time_row.addWidget(self._start_spin)
        time_row.addWidget(QLabel("to"))
        time_row.addWidget(self._end_spin)
        time_widget = self._make_row_widget(time_row)
        form.addRow("Time range:", time_widget)

        # Run for: a person (pose detection, the default) or one of this
        # capture's tracked objects (marker detection instead -- design
        # doc §7.1 sub-phase 1c). Both the combo and its accompanying
        # marker-only fields are only built when the capture actually has
        # at least one object, same "don't show it if it can't apply" rule
        # the bbox-source combo below already follows -- object mode is
        # simply impossible otherwise, so there is nothing to disable-not-
        # hide here (unlike detector/pose-model, which the bbox combo can
        # toggle away from without objects being involved at all).
        from posetrak.db.manage_capture_object import list_capture_objects

        objects = list_capture_objects(self._conn, self._capture_id)
        self._object_combo: QComboBox | None = None
        self._marker_perimeter_spin: QDoubleSpinBox | None = None
        self._marker_frame_step_spin: QSpinBox | None = None
        if objects:
            self._object_combo = QComboBox()
            self._object_combo.addItem("Person (pose)", None)
            for obj in objects:
                self._object_combo.addItem(obj["name"], obj["id"])
            self._object_combo.setToolTip(
                "Choose one of this capture's tracked objects to run marker "
                "detection instead of pose detection -- the fields below "
                "switch to match."
            )
            self._object_combo.currentIndexChanged.connect(self._on_object_source_changed)
            form.addRow("Run for:", self._object_combo)

            # Marker-detection-only fields -- dictionary/marker ids come
            # from the chosen object's own marker body definition, so
            # there is nothing to pick here beyond the detector's own
            # tuning. Disabled (not hidden) until an object is chosen,
            # matching the bbox-source combo's own convention below.
            self._marker_perimeter_spin = QDoubleSpinBox()
            self._marker_perimeter_spin.setRange(0.001, 1.0)
            self._marker_perimeter_spin.setDecimals(3)
            self._marker_perimeter_spin.setSingleStep(0.005)
            self._marker_perimeter_spin.setValue(0.01)
            self._marker_perimeter_spin.setToolTip(
                "cv2.aruco's minMarkerPerimeterRate -- lower catches smaller/"
                "farther markers but risks false positives. The cv2 default "
                "(0.03) misses room-distance markers in 4K frames; 0.01 is a "
                "better starting point for this project's typical setups."
            )
            self._marker_frame_step_spin = QSpinBox()
            self._marker_frame_step_spin.setRange(1, 100)
            self._marker_frame_step_spin.setValue(1)
            self._marker_frame_step_spin.setSuffix(" frame(s)")
            marker_row = QHBoxLayout()
            marker_row.addWidget(QLabel("Perimeter rate:"))
            marker_row.addWidget(self._marker_perimeter_spin)
            marker_row.addWidget(QLabel("Frame step:"))
            marker_row.addWidget(self._marker_frame_step_spin)
            form.addRow("Marker detector:", self._make_row_widget(marker_row))
            self._marker_perimeter_spin.setEnabled(False)
            self._marker_frame_step_spin.setEnabled(False)

        # Bbox source: YOLO (default, always available) or an existing
        # segmentation covering this capture, if any -- gap 2 of
        # docs/roadmap/features/segmentation-reuse/segmentation-reuse-design.md.
        # Only shown when at least one segmentation exists, so the common
        # case (no segmentation, YOLO-only) looks exactly as before.
        seg_runs = self._conn.execute(
            "SELECT id, created_at, persons_json, time_start_s, time_end_s "
            "FROM seg_quality_runs WHERE shot_id = ? ORDER BY created_at DESC",
            (self._capture_id,),
        ).fetchall()
        self._bbox_source_combo: QComboBox | None = None
        if seg_runs:
            import json as _json
            self._bbox_source_combo = QComboBox()
            self._bbox_source_combo.addItem("YOLO detection", None)
            # Default to the most recent segmentation that fully covers the
            # range being detected here, rather than always defaulting to
            # YOLO: a masks-based segmentation gives tighter crops and needs
            # no manual track-to-person stitching afterward, so there's no
            # reason to prefer YOLO when a segmentation already has us
            # covered. A segmentation that only covers *part* of this range
            # is left unselected (picking it silently would leave the
            # uncovered frames without any bboxes at all); with no known
            # range (time_start_s/time_end_s not given -- e.g. a brand new
            # trial before Mark Start/End) there's nothing to check against,
            # so it's left on YOLO too.
            default_index = 0
            for r in seg_runs:
                persons = _json.loads(r["persons_json"]) if r["persons_json"] else []
                who = ", ".join(persons) if persons else "unlabeled"
                created = str(r["created_at"])[:19].replace("T", " ")
                self._bbox_source_combo.addItem(
                    f"Segmentation ({who}) — {created}", r["id"]
                )
                if (
                    default_index == 0
                    and time_start_s is not None
                    and time_end_s is not None
                    and r["time_start_s"] <= time_start_s
                    and r["time_end_s"] >= time_end_s
                ):
                    default_index = self._bbox_source_combo.count() - 1
            self._bbox_source_combo.setCurrentIndex(default_index)
            self._bbox_source_combo.setToolTip(
                "Source bboxes from an existing Cutie segmentation instead of "
                "running YOLO -- masks give tighter, more accurate crops, and "
                "results are finalised automatically (no manual track-to-person "
                "stitching needed, since a segmentation's labels are already "
                "stable per-person identities). Defaults to the most recent "
                "segmentation that fully covers this time range, if any."
            )
            self._bbox_source_combo.currentIndexChanged.connect(self._on_bbox_source_changed)
            form.addRow("Bbox source:", self._bbox_source_combo)

        # Model selection
        self._detector_combo = QComboBox()
        self._detector_combo.addItems(["yolox-x", "yolox-m", "yolox-tiny"])
        form.addRow("Detector:", self._detector_combo)

        self._pose_combo = QComboBox()
        self._pose_combo.addItems(["rtmpose-l-133kp", "vitpose-l-133kp"])
        form.addRow("Pose model:", self._pose_combo)

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.01, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.3)
        form.addRow("Confidence:", self._conf_spin)

        self._refine_hands_check = QCheckBox("Refine hands")
        self._refine_hands_check.setChecked(True)
        self._refine_hands_check.setToolTip(
            "After the full-body pass, re-detect each tracked wrist's hand in a "
            "tight crop (rtmlib.Hand) and patch in the refined finger keypoints. "
            "Only has an effect for 133-keypoint pose models."
        )
        form.addRow("", self._refine_hands_check)

        if self._bbox_source_combo is not None:
            # setCurrentIndex() above (when defaulting to a segmentation)
            # doesn't fire currentIndexChanged (not connected yet at that
            # point), so sync the detector/confidence fields' enabled state
            # by hand now that they exist -- without this, defaulting to a
            # segmentation would leave the YOLO-only fields incorrectly
            # enabled until the user touched the combo themselves.
            self._on_bbox_source_changed(self._bbox_source_combo.currentIndex())

        layout.addLayout(form)

        # Set once by DetectionJob.device_notice if the run falls back to
        # CPU (no GPU / no torch); stays visible for the whole run, unlike
        # _frame_label which gets overwritten by the next progress update.
        self._device_warning_label = QLabel("")
        self._device_warning_label.setStyleSheet("color: #b36b00; font-weight: bold;")
        self._device_warning_label.setWordWrap(True)
        self._device_warning_label.hide()
        layout.addWidget(self._device_warning_label)

        # Progress
        self._frame_bar = QProgressBar()
        self._frame_bar.setRange(0, 100)
        self._frame_label = QLabel("")
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frames:"))
        frame_row.addWidget(self._frame_bar, 1)
        frame_row.addWidget(self._frame_label)
        layout.addLayout(frame_row)

        self._cam_bar = QProgressBar()
        self._cam_bar.setRange(0, 100)
        self._cam_label = QLabel("")
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Cameras:"))
        cam_row.addWidget(self._cam_bar, 1)
        cam_row.addWidget(self._cam_label)
        layout.addLayout(cam_row)

        # Buttons
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Detection")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)
        self._close_btn = QPushButton("Cancel")
        self._close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._close_btn)
        layout.addLayout(btn_row)

        if not syncs:
            self._run_btn.setEnabled(False)
            self._run_btn.setToolTip("No sync config — set one up first")

    @staticmethod
    def _make_row_widget(layout: QHBoxLayout):
        from PySide6.QtWidgets import QWidget
        w = QWidget()
        w.setLayout(layout)
        return w

    # ------------------------------------------------------------------

    def _on_bbox_source_changed(self, _index: int) -> None:
        """Detector/confidence are YOLO-only -- disable them (not hide, so
        the layout doesn't jump) when an existing segmentation is chosen."""
        is_yolo = self._bbox_source_combo.currentData() is None
        self._detector_combo.setEnabled(is_yolo)
        self._conf_spin.setEnabled(is_yolo)

    def _on_object_source_changed(self, _index: int) -> None:
        """Choosing an object switches the dialog to marker-detection mode
        (design §7.1 sub-phase 1c): every person-only field (bbox source,
        detector, pose model, confidence, refine hands) is disabled and
        the marker-only fields are enabled, and vice versa -- same
        disable-not-hide convention as _on_bbox_source_changed above."""
        is_object_mode = self._object_combo.currentData() is not None
        self._marker_perimeter_spin.setEnabled(is_object_mode)
        self._marker_frame_step_spin.setEnabled(is_object_mode)
        self._pose_combo.setEnabled(not is_object_mode)
        self._refine_hands_check.setEnabled(not is_object_mode)
        if self._bbox_source_combo is not None:
            self._bbox_source_combo.setEnabled(not is_object_mode)
            # Reconciles detector/confidence with YOLO-vs-segmentation when
            # returning to person mode; harmlessly overridden again just
            # below when entering object mode.
            self._on_bbox_source_changed(self._bbox_source_combo.currentIndex())
        else:
            self._detector_combo.setEnabled(not is_object_mode)
            self._conf_spin.setEnabled(not is_object_mode)
        if is_object_mode:
            self._detector_combo.setEnabled(False)
            self._conf_spin.setEnabled(False)

    def _controls_enabled(self, enabled: bool) -> None:
        for w in [
            self._sync_combo,
            self._start_spin, self._end_spin,
            self._detector_combo, self._pose_combo, self._conf_spin,
            self._refine_hands_check, self._run_btn, self._close_btn,
        ]:
            w.setEnabled(enabled)
        if self._trial_name is not None:
            self._trial_name.setEnabled(enabled)
        if self._bbox_source_combo is not None:
            self._bbox_source_combo.setEnabled(enabled)
            if enabled:
                self._on_bbox_source_changed(self._bbox_source_combo.currentIndex())
        if self._object_combo is not None:
            self._object_combo.setEnabled(enabled)
            if enabled:
                self._on_object_source_changed(self._object_combo.currentIndex())

    def _on_run(self) -> None:
        sync_id = self._sync_combo.currentData()
        if not sync_id:
            QMessageBox.warning(self, "Missing sync", "No sync config selected.")
            return
        start_s = self._start_spin.value()
        end_s = self._end_spin.value()
        if end_s <= start_s:
            QMessageBox.warning(self, "Invalid range", "End time must be after start time.")
            return

        object_id = self._object_combo.currentData() if self._object_combo else None

        self._controls_enabled(False)
        self._frame_bar.setValue(0)
        self._cam_bar.setValue(0)
        self._cam_label.setText("Starting…")
        self._device_warning_label.hide()
        self._device_warning_label.setText("")

        if object_id is not None:
            self._run_marker_detection(object_id, sync_id, start_s, end_s)
            return

        seg_run_id = self._bbox_source_combo.currentData() if self._bbox_source_combo else None
        if seg_run_id is not None:
            self._run_from_segmentation(seg_run_id, sync_id, start_s, end_s)
            return

        from app.pose.main import DetectionJob
        self._job = DetectionJob(
            session_path=str(self._session_path),
            shot_id=self._capture_id,
            sync_config_id=sync_id,
            time_start_s=start_s,
            time_end_s=end_s,
            detector_name=self._detector_combo.currentText(),
            pose_model_name=self._pose_combo.currentText(),
            detector_conf=self._conf_spin.value(),
            refine_hands=self._refine_hands_check.isChecked(),
        )
        self._job.progress.connect(self._on_progress)
        self._job.camera_progress.connect(self._on_camera_progress)
        self._job.device_notice.connect(self._on_device_notice)
        self._job.finished.connect(self._on_finished)
        self._job.error.connect(self._on_error)
        self._job.start()

    # ------------------------------------------------------------------
    # Marker detection for a tracked object (design §7.1 sub-phase 1c)
    # ------------------------------------------------------------------

    def _run_marker_detection(
        self, object_id: str, sync_id: str, start_s: float, end_s: float,
    ) -> None:
        from app.pose.main import MarkerDetectionJob
        self._job = MarkerDetectionJob(
            session_path=str(self._session_path),
            capture_object_id=object_id,
            sync_config_id=sync_id,
            time_start_s=start_s,
            time_end_s=end_s,
            min_marker_perimeter_rate=self._marker_perimeter_spin.value(),
            frame_step=self._marker_frame_step_spin.value(),
        )
        self._job.progress.connect(self._on_progress)
        self._job.camera_progress.connect(self._on_camera_progress)
        self._job.finished.connect(self._on_finished)
        self._job.error.connect(self._on_error)
        self._job.start()

    # ------------------------------------------------------------------
    # Segmentation bbox source (gap 2, segmentation-reuse-design.md) --
    # invokes the same PoseWorker/PoseExtractionJob/JobQueueRunner
    # machinery CutieInitPanel already uses for its own "Queue Pose",
    # rather than DetectionPipeline/DetectionJob (the design doc's option
    # (a): much less work than teaching DetectionPipeline a second bbox
    # source, and this dialog only needs a producer of detection_keypoints
    # + person_tracks, which PoseWorker already is).
    # ------------------------------------------------------------------

    def _run_from_segmentation(
        self, seg_run_id: str, sync_id: str, start_s: float, end_s: float,
    ) -> None:
        import uuid

        from app.pose.db_cache import create_detection_run
        from app.pose.job_queue_runner import JobQueueRunner
        from app.pose.pose_worker import PoseExtractionJob
        from app.setup.db_context import SyncPoint, SyncTable
        from posetrak.db.manage_person import persons_ordered_for_seg_run

        self._seg_persons_ordered = persons_ordered_for_seg_run(self._conn, seg_run_id)

        self._seg_detection_run_id = create_detection_run(
            self._conn, self._capture_id, sync_id, start_s, end_s,
            detector_model="segmentation", pose_model=self._pose_combo.currentText(),
            trial_id=self._trial_id,
        )
        self._conn.commit()

        # Same global-time -> per-camera-frame conversion
        # posetrak.detection.pipeline.DetectionPipeline._frame_range uses,
        # so both bbox sources cover the same actual frames for a given
        # time range.
        sync_rows = self._conn.execute(
            "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, sv.actual_fps "
            "FROM sync_points sp JOIN capture_videos sv ON sv.id = sp.shot_video_id "
            "WHERE sp.sync_config_id = ?",
            (sync_id,),
        ).fetchall()
        points = [
            SyncPoint(camera_instance_id="", shot_video_id=r["shot_video_id"],
                      video_frame=r["video_frame"], timestamp_s=r["timestamp_s"])
            for r in sync_rows
        ]
        fps_by_video = {r["shot_video_id"]: float(r["actual_fps"]) for r in sync_rows}
        sync_table = SyncTable(points, fps_by_video)

        cam_rows = self._conn.execute(
            "SELECT cv.id, cv.file_path, COALESCE(ci.label, cv.camera_instance_id) AS label "
            "FROM capture_videos cv LEFT JOIN camera_instances ci ON ci.id = cv.camera_instance_id "
            "WHERE cv.shot_id = ?",
            (self._capture_id,),
        ).fetchall()

        db_path = ""
        for row in self._conn.execute("PRAGMA database_list"):
            if row[1] == "main":
                db_path = row[2]
                break

        self._seg_runner = JobQueueRunner(db_path=db_path, parent=self)
        self._seg_jobs_total = 0
        self._seg_jobs_done = 0
        for cam in cam_rows:
            first = sync_table.lookup(start_s, cam["id"])
            last = sync_table.lookup(end_s, cam["id"])
            if first is None or last is None:
                continue
            job = PoseExtractionJob(
                job_id=str(uuid.uuid4())[:8],
                camera_label=cam["label"],
                shot_video_id=cam["id"],
                video_path=cam["file_path"],
                detection_run_id=self._seg_detection_run_id,
                seg_quality_run_id=seg_run_id,
                persons_ordered=self._seg_persons_ordered,
                first_frame=max(0, first),
                last_frame=last,
                pose_model=self._pose_combo.currentText(),
                refine_hands=self._refine_hands_check.isChecked(),
            )
            self._seg_runner.enqueue(job)
            self._seg_jobs_total += 1

        if self._seg_jobs_total == 0:
            QMessageBox.warning(
                self, "No cameras", "No cameras with sync data for this time range."
            )
            self._controls_enabled(True)
            return

        self._cam_label.setText(f"0/{self._seg_jobs_total}")
        self._seg_runner.progress.connect(self._on_seg_progress)
        self._seg_runner.job_finished.connect(self._on_seg_job_finished)
        self._seg_runner.job_failed.connect(self._on_seg_job_failed)
        self._seg_runner.queue_done.connect(self._on_seg_queue_done)
        self._seg_runner.start()

    def _on_seg_progress(self, done: int, total: int) -> None:
        self._frame_bar.setValue(int(done / max(total, 1) * 100))
        self._frame_label.setText(f"{done}/{total} frames")

    def _on_seg_job_finished(self, _job_id: str, _count: int) -> None:
        self._seg_jobs_done += 1
        self._cam_bar.setValue(int(self._seg_jobs_done / max(self._seg_jobs_total, 1) * 100))
        self._cam_label.setText(f"{self._seg_jobs_done}/{self._seg_jobs_total}")

    def _on_seg_job_failed(self, job_id: str, error: str) -> None:
        # Non-fatal to the queue -- JobQueueRunner moves on to the next
        # job on its own; surfaced so a silent partial failure isn't lost.
        QMessageBox.warning(self, "Pose Extraction", f"Camera job {job_id} failed: {error}")

    def _on_seg_queue_done(self) -> None:
        from app.pose.finalise import auto_assign_and_finalise, conf_scale_for_model

        run_row = self._conn.execute(
            "SELECT shot_id, sync_config_id, pose_model FROM detection_runs WHERE id=?",
            (self._seg_detection_run_id,),
        ).fetchone()
        try:
            auto_assign_and_finalise(
                session=self._conn,
                detection_run_id=self._seg_detection_run_id,
                shot_id=run_row["shot_id"],
                sync_config_id=run_row["sync_config_id"],
                persons_ordered=self._seg_persons_ordered,
                pose_model=run_row["pose_model"],
                confidence_scale=conf_scale_for_model(run_row["pose_model"]),
            )
        except Exception as exc:  # noqa: BLE001
            self._controls_enabled(True)
            QMessageBox.critical(self, "Finalise Error", str(exc))
            return
        self._on_finished(self._seg_detection_run_id)

    def _on_progress(self, pct: int, msg: str) -> None:
        self._frame_bar.setValue(pct)
        self._frame_label.setText(msg)

    def _on_device_notice(self, msg: str) -> None:
        self._device_warning_label.setText(f"⚠ {msg}")
        self._device_warning_label.show()

    def _on_camera_progress(self, done: int, total: int) -> None:
        self._cam_bar.setValue(int(done / max(total, 1) * 100))
        self._cam_label.setText(f"{done}/{total}")

    def _on_finished(self, run_id: str) -> None:
        self._frame_bar.setValue(100)
        self._cam_bar.setValue(100)
        self._cam_label.setText("Done")

        if self._trial_id is None:
            # Create a new trial and link the detection run to it
            trial_id = generate_id()
            name = self._trial_name.text().strip() if self._trial_name else "Trial"
            name = name or "Trial"
            self._conn.execute(
                "INSERT INTO trials (id, capture_id, name, time_start_s, time_end_s) "
                "VALUES (?, ?, ?, ?, ?)",
                (trial_id, self._capture_id, name,
                 self._start_spin.value(), self._end_spin.value()),
            )
        else:
            trial_id = self._trial_id

        self._conn.execute(
            "UPDATE detection_runs SET trial_id = ? WHERE id = ?",
            (trial_id, run_id),
        )
        self._conn.commit()

        self.detection_finished.emit(trial_id, run_id)

        self._close_btn.setEnabled(True)
        self._close_btn.setText("Close")
        self._close_btn.clicked.disconnect()
        self._close_btn.clicked.connect(self.accept)

    def _on_error(self, msg: str) -> None:
        self._controls_enabled(True)
        self._cam_label.setText("Error")
        QMessageBox.critical(self, "Detection Error", msg)

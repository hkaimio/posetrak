"""skeleton_scaling_panel.py — Interactive skeleton scaling dialog.

Workflow
--------
1.  Dialog opens from TrackingRunPanel → loads skeleton template measurements.
2.  Background worker: load inlier obs from DB → DLT triangulate → per-step
    distances between marker pairs (femur, shin, upper_arm, …).
3.  Matplotlib canvas shows time series for each measurement (6 plots).
    User drags a SpanSelector band on any plot to pick the "good pose" range.
4.  Medians over the selected span populate editable spinboxes.
5.  "Save scaled skeleton" calls scale_skeleton_yaml() + import_skeleton_str().

Video scrubber
--------------
The left panel shows full video frames via FrameReader.  Clicking any point on
the matplotlib plots seeks the video to that timestamp; the time slider in the
video panel is bi-directionally linked to a vertical cursor line on all plots.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import yaml
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from app.setup.db_context import SyncPoint, SyncTable
from app.setup.video_reader import FrameReader
from posetrak.db.manage_skeleton import import_skeleton_str
from posetrak.db.scale_skeleton import scale_skeleton_yaml, template_measurements

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEAS_KEYS = [
    "femur", "shin", "upper_arm", "lower_arm", "torso_height", "shoulder_width",
]

MEAS_LABELS = {
    "femur":          "Femur  (hip → knee)",
    "shin":           "Shin  (knee → ankle)",
    "upper_arm":      "Upper arm  (shoulder → elbow)",
    "lower_arm":      "Lower arm  (elbow → wrist)",
    "torso_height":   "Torso height  (hip → shoulder midpoint)",
    "shoulder_width": "Shoulder width  (L → R)",
}

# Marker-pair distances that define each measurement.
# For bilateral pairs (L+R) both contribute and we take the mean.
_MEAS_PAIRS: dict[str, list[tuple[str, str]]] = {
    "femur":          [("MRK-hip.L",      "MRK-knee.L"),
                       ("MRK-hip.R",      "MRK-knee.R")],
    "shin":           [("MRK-knee.L",     "MRK-Ankle.L"),
                       ("MRK-knee.R",     "MRK-Ankle.R")],
    "upper_arm":      [("MRK-shoulder.L", "MRK-elbow.L"),
                       ("MRK-shoulder.R", "MRK-elbow.R")],
    "lower_arm":      [("MRK-elbow.L",    "MRK-wrist.L"),
                       ("MRK-elbow.R",    "MRK-wrist.R")],
    "shoulder_width": [("MRK-shoulder.L", "MRK-shoulder.R")],
    # "torso_height": computed from midpoints, handled separately
}

# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


class _MeasWorker(QThread):
    """Load inlier obs → DLT triangulate → per-step measurement DataFrame."""

    finished = Signal(object, str)  # (pd.DataFrame | None, error_msg)

    def __init__(self, db_path: str, run_id: str, parent=None) -> None:
        super().__init__(parent)
        self._db_path = db_path
        self._run_id = run_id

    def run(self) -> None:  # noqa: D401
        import pandas as pd
        from posetrak.db.load_session import (
            load_cameras_from_session,
            load_inlier_obs_from_tracking_run,
        )

        try:
            obs_df = load_inlier_obs_from_tracking_run(
                self._db_path, self._run_id, person_id=0, inliers_only=True
            )
            if obs_df.empty:
                self.finished.emit(None, "No inlier observations found.")
                return

            # Camera projection matrices
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            run = conn.execute(
                "SELECT extrinsic_calibration_id, observation_sequence_id "
                "FROM tracking_runs WHERE id=?",
                (self._run_id,),
            ).fetchone()
            session_row = conn.execute(
                "SELECT session_id FROM extrinsic_calibrations WHERE id=?",
                (run["extrinsic_calibration_id"],),
            ).fetchone()
            conn.close()

            cam_list = load_cameras_from_session(
                self._db_path,
                run["extrinsic_calibration_id"],
                session_row["session_id"],
            )
            P_by_label = {
                c.get("instance_label") or c["label"]: c["P"] for c in cam_list
            }

            # DLT triangulate each (tracker_step, marker_name)
            tri: dict[tuple, np.ndarray] = {}
            step_ts: dict[int, float] = {}
            for (step, mname), grp in obs_df.groupby(["tracker_step", "marker_name"]):
                pts, Ps = [], []
                for row in grp.itertuples(index=False):
                    P = P_by_label.get(row.camera_label)
                    if P is None:
                        continue
                    pts.append((row.pixel_x, row.pixel_y))
                    Ps.append(P)
                if len(pts) < 2:
                    continue
                pos, cond = _dlt(pts, Ps)
                if cond > 200 or not np.all(np.isfinite(pos)):
                    continue
                tri[(step, mname)] = pos
                step_ts[step] = float(grp["timestamp_s"].iloc[0])

            if not tri:
                self.finished.emit(None, "DLT triangulation yielded no valid results.")
                return

            # Aggregate positions per step
            pos_by_step: dict[int, dict] = {}
            for (step, mname), pos in tri.items():
                pos_by_step.setdefault(step, {})
                pos_by_step[step][mname] = pos

            # Compute measurement distances per step
            records = []
            for step in sorted(pos_by_step):
                data = pos_by_step[step]
                rec: dict = {"tracker_step": step, "timestamp_s": step_ts[step]}

                for key, pairs in _MEAS_PAIRS.items():
                    dists = []
                    for m1, m2 in pairs:
                        if m1 in data and m2 in data:
                            dists.append(float(np.linalg.norm(data[m1] - data[m2])))
                    rec[key] = float(np.mean(dists)) if dists else float("nan")

                # torso_height via midpoints
                sl, sr = data.get("MRK-shoulder.L"), data.get("MRK-shoulder.R")
                hl, hr = data.get("MRK-hip.L"), data.get("MRK-hip.R")
                if all(v is not None for v in (sl, sr, hl, hr)):
                    rec["torso_height"] = float(
                        np.linalg.norm((sl + sr) / 2 - (hl + hr) / 2)
                    )
                else:
                    rec["torso_height"] = float("nan")

                records.append(rec)

            self.finished.emit(pd.DataFrame(records), "")

        except Exception:
            import traceback
            self.finished.emit(None, traceback.format_exc())


def _dlt(
    observations: list[tuple[float, float]],
    Ps: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    rows = []
    for (u, v), P in zip(observations, Ps):
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.array(rows, dtype=float)
    _, s, Vt = np.linalg.svd(A)
    X = Vt[-1]
    pos = X[:3] / X[3]
    cond = float(s[0] / s[-2]) if s[-2] > 1e-12 else float("inf")
    return pos, cond


# ---------------------------------------------------------------------------
# Matplotlib plot canvas
# ---------------------------------------------------------------------------


class _PlotCanvas(QWidget):
    """Six measurement time series with linked SpanSelector and cursor line."""

    span_changed = Signal(float, float)   # (t_lo_s, t_hi_s)
    time_clicked = Signal(float)          # plot click → seek video

    _NCOLS = 2
    _NROWS = 3  # ceil(6/2)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._meas_df = None
        self._span_lo: float | None = None
        self._span_hi: float | None = None
        self._cursor_lines: list = []
        self._span_patches: list = []
        self._selectors: list[SpanSelector] = []

        fig = Figure(figsize=(9, 7))
        fig.set_tight_layout(True)
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._axes = fig.subplots(self._NROWS, self._NCOLS).flatten()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._canvas.mpl_connect("button_press_event", self._on_plot_click)

    # ------------------------------------------------------------------
    # Public API

    def load_data(self, meas_df, tmpl: dict[str, float]) -> None:
        """Draw (or redraw) all subplots with measurement data."""
        import pandas as pd

        self._meas_df = meas_df
        self._cursor_lines.clear()
        self._span_patches.clear()
        for sel in self._selectors:
            sel.set_visible(False)
        self._selectors.clear()

        ts = meas_df["timestamp_s"].values if not meas_df.empty else np.array([])

        for i, key in enumerate(MEAS_KEYS):
            ax = self._axes[i]
            ax.clear()
            ax.set_title(MEAS_LABELS[key], fontsize=8, pad=2)
            ax.set_ylabel("cm", fontsize=7)
            ax.tick_params(labelsize=7)

            if not meas_df.empty and key in meas_df.columns:
                vals = meas_df[key].values * 100  # m → cm
                ax.plot(ts, vals, color="lightsteelblue", lw=0.7, alpha=0.7)
                smoothed = (
                    pd.Series(vals).rolling(15, center=True, min_periods=1).median()
                )
                ax.plot(ts, smoothed.values, color="steelblue", lw=1.5)

            if key in tmpl:
                v = tmpl[key] * 100
                ax.axhline(v, color="#888", lw=1, ls="--", alpha=0.8,
                           label=f"tmpl {v:.1f} cm")
                ax.legend(fontsize=7, loc="upper right", framealpha=0.5)

            # Span and cursor placeholders (invisible until used)
            patch = ax.axvspan(0, 0, alpha=0.2, color="gold", visible=False, zorder=0)
            self._span_patches.append(patch)
            cursor = ax.axvline(0, color="crimson", lw=1, alpha=0.7, visible=False)
            self._cursor_lines.append(cursor)

            sel = SpanSelector(
                ax, self._on_span, "horizontal",
                useblit=True, interactive=True,
                drag_from_anywhere=True,
                props=dict(alpha=0.25, facecolor="gold"),
                handle_props=dict(alpha=0.6),
            )
            self._selectors.append(sel)

        for j in range(len(MEAS_KEYS), len(self._axes)):
            self._axes[j].set_visible(False)

        self._canvas.draw_idle()

    def move_cursor(self, t: float) -> None:
        for line in self._cursor_lines:
            line.set_xdata([t, t])
            line.set_visible(True)
        self._canvas.draw_idle()

    def current_span(self) -> tuple[float, float] | None:
        if self._span_lo is None:
            return None
        return self._span_lo, self._span_hi

    # ------------------------------------------------------------------
    # Internal callbacks

    def _on_span(self, xmin: float, xmax: float) -> None:
        if xmax - xmin < 0.05:
            return
        self._span_lo, self._span_hi = xmin, xmax
        # Synchronise gold band across all axes
        for patch in self._span_patches:
            patch.set_visible(True)
            # axvspan xy is [[x0,0],[x0,1],[x1,1],[x1,0],[x0,0]]
            patch.set_xy([
                [xmin, 0], [xmin, 1], [xmax, 1], [xmax, 0], [xmin, 0],
            ])
        self._canvas.draw_idle()
        self.span_changed.emit(xmin, xmax)

    def _on_plot_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        if event.button != 1:
            return
        self.time_clicked.emit(float(event.xdata))


# ---------------------------------------------------------------------------
# Video panel (single camera, full frames)
# ---------------------------------------------------------------------------


class _VideoPanel(QWidget):
    """Camera selector + full-frame video display driven by timestamp."""

    time_changed = Signal(float)

    def __init__(
        self, conn: sqlite3.Connection, run_id: str, parent=None
    ) -> None:
        super().__init__(parent)
        self._conn = conn
        self._run_id = run_id
        self._sync_table: SyncTable | None = None
        self._readers: dict[str, FrameReader] = {}   # svid → reader
        self._svid_map: dict[str, str] = {}           # label → svid
        self._current_svid: str | None = None
        self._t_start = 0.0

        # --- Widgets ---
        self._cam_combo = QComboBox()
        self._cam_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self._frame_lbl = QLabel("Loading…")
        self._frame_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_lbl.setStyleSheet("background: #111; color: #555;")
        self._frame_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._frame_lbl.setMinimumSize(300, 200)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._time_lbl = QLabel("—")
        self._time_lbl.setMinimumWidth(60)

        slider_row = QHBoxLayout()
        slider_row.addWidget(self._slider)
        slider_row.addWidget(self._time_lbl)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        layout.addWidget(self._cam_combo)
        layout.addWidget(self._frame_lbl, stretch=1)
        layout.addLayout(slider_row)

        self._cam_combo.currentIndexChanged.connect(self._on_cam_changed)
        self._slider.valueChanged.connect(self._on_slider_moved)

        self._populate()

    # ------------------------------------------------------------------
    # Initialisation

    def _populate(self) -> None:
        run = self._conn.execute(
            "SELECT observation_sequence_id FROM tracking_runs WHERE id=?",
            (self._run_id,),
        ).fetchone()
        if not run:
            return

        seq = self._conn.execute(
            "SELECT shot_id, time_start_s, time_end_s, sync_config_id "
            "FROM pose_observation_sequences WHERE id=?",
            (run["observation_sequence_id"],),
        ).fetchone()
        if not seq:
            return

        self._t_start = float(seq["time_start_s"])
        t_end = float(seq["time_end_s"])
        dur_ms = max(1, int((t_end - self._t_start) * 1000))

        # Sync table
        sp_rows = self._conn.execute(
            "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, cv.actual_fps "
            "FROM sync_points sp "
            "JOIN capture_videos cv ON cv.id = sp.shot_video_id "
            "WHERE sp.sync_config_id=?",
            (seq["sync_config_id"],),
        ).fetchall()
        if sp_rows:
            pts = [
                SyncPoint("", r["shot_video_id"], r["video_frame"], r["timestamp_s"])
                for r in sp_rows
            ]
            fps_map = {r["shot_video_id"]: float(r["actual_fps"]) for r in sp_rows}
            self._sync_table = SyncTable(pts, fps_map)
            max_fps = max(fps_map.values()) if fps_map else 30.0
        else:
            max_fps = 30.0

        self._slider.setMinimum(0)
        self._slider.setMaximum(dur_ms)
        self._slider.setSingleStep(max(1, round(1000.0 / max_fps)))

        # Cameras
        cam_rows = self._conn.execute(
            "SELECT cv.id, COALESCE(ci.label, cv.camera_instance_id) AS label "
            "FROM capture_videos cv "
            "LEFT JOIN camera_instances ci ON ci.id = cv.camera_instance_id "
            "WHERE cv.shot_id=? ORDER BY label",
            (seq["shot_id"],),
        ).fetchall()
        self._cam_combo.blockSignals(True)
        for row in cam_rows:
            self._svid_map[row["label"]] = row["id"]
            self._cam_combo.addItem(row["label"])
        self._cam_combo.blockSignals(False)
        if cam_rows:
            self._on_cam_changed(0)

    # ------------------------------------------------------------------
    # Slots

    def _on_cam_changed(self, idx: int) -> None:
        label = self._cam_combo.itemText(idx)
        svid = self._svid_map.get(label)
        if svid == self._current_svid:
            return
        # Stop old reader
        if self._current_svid in self._readers:
            self._readers[self._current_svid].shutdown()
        self._current_svid = svid
        if svid and svid not in self._readers:
            fp_row = self._conn.execute(
                "SELECT file_path FROM capture_videos WHERE id=?", (svid,)
            ).fetchone()
            if fp_row and Path(fp_row["file_path"]).exists():
                reader = FrameReader(fp_row["file_path"], parent=self)
                reader.frame_ready.connect(self._on_frame)
                reader.start()
                self._readers[svid] = reader
            else:
                self._frame_lbl.setText("Video file not found")
                return
        self._seek_t(self._t_start + self._slider.value() / 1000.0)

    def _on_slider_moved(self, ms: int) -> None:
        t = self._t_start + ms / 1000.0
        self._time_lbl.setText(f"{t:.2f} s")
        self._seek_t(t)
        self.time_changed.emit(t)

    def _on_frame(self, _frame_idx: int, frame_bgr: object) -> None:
        if not isinstance(frame_bgr, np.ndarray):
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        sz = self._frame_lbl.size()
        self._frame_lbl.setPixmap(
            pixmap.scaled(sz, Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
        )

    # ------------------------------------------------------------------
    # Public API

    def seek(self, t: float) -> None:
        """External seek (plot click): move slider, request frame, no re-emit."""
        ms = int((t - self._t_start) * 1000)
        ms = max(0, min(ms, self._slider.maximum()))
        self._slider.blockSignals(True)
        self._slider.setValue(ms)
        self._slider.blockSignals(False)
        self._time_lbl.setText(f"{t:.2f} s")
        self._seek_t(t)

    def shutdown(self) -> None:
        for reader in self._readers.values():
            reader.shutdown()
        self._readers.clear()

    def _seek_t(self, t: float) -> None:
        svid = self._current_svid
        if not svid or not self._sync_table:
            return
        reader = self._readers.get(svid)
        if not reader:
            return
        frame_idx = self._sync_table.lookup(t, svid)
        if frame_idx is not None:
            reader.request(frame_idx)


# ---------------------------------------------------------------------------
# Summary row widget (one measurement)
# ---------------------------------------------------------------------------


class _MeasRow(QWidget):
    """One row: label | template | measured median | override spinbox."""

    def __init__(self, key: str, tmpl_m: float, parent=None) -> None:
        super().__init__(parent)
        self._key = key
        self._tmpl_m = tmpl_m

        self._lbl_tmpl = QLabel(f"{tmpl_m * 100:.1f} cm" if tmpl_m else "—")
        self._lbl_med = QLabel("—")
        self._spin = QDoubleSpinBox()
        self._spin.setRange(1.0, 300.0)
        self._spin.setDecimals(1)
        self._spin.setSuffix(" cm")
        self._spin.setSingleStep(0.5)
        if tmpl_m:
            self._spin.setValue(round(tmpl_m * 100, 1))

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(MEAS_LABELS[key]), stretch=3)
        row.addWidget(self._lbl_tmpl, stretch=1)
        row.addWidget(self._lbl_med, stretch=1)
        row.addWidget(self._spin, stretch=1)

    def set_median(self, median_m: float) -> None:
        if np.isfinite(median_m):
            self._lbl_med.setText(f"{median_m * 100:.1f} cm")
            self._spin.setValue(round(median_m * 100, 1))
        else:
            self._lbl_med.setText("—")

    @property
    def value_m(self) -> float:
        return self._spin.value() / 100.0


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------


class SkeletonScalingPanel(QDialog):
    """Full skeleton scaling workflow as a resizable dialog."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        run_id: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._conn = conn
        self._db_path = db_path
        self._run_id = run_id
        self._meas_df = None
        self._skel_yaml: str | None = None
        self._tmpl: dict[str, float] = {}
        self._rows: dict[str, _MeasRow] = {}
        self._worker: _MeasWorker | None = None

        self.setWindowTitle("Skeleton Scaling")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(1400, 860)

        self._load_skeleton()
        self._build_ui()
        self._start_worker()

    # ------------------------------------------------------------------
    # Setup

    def _load_skeleton(self) -> None:
        row = self._conn.execute(
            "SELECT s.yaml_content, s.name "
            "FROM tracking_runs tr "
            "JOIN skeletons s ON s.id = tr.skeleton_id "
            "WHERE tr.id=?",
            (self._run_id,),
        ).fetchone()
        if row and row["yaml_content"]:
            self._skel_yaml = row["yaml_content"]
            self._skel_name = row["name"] or "skeleton"
            joints = yaml.safe_load(self._skel_yaml)["joints"]
            self._tmpl = template_measurements(joints)
        else:
            self._skel_name = "skeleton"

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Status bar
        self._status = QLabel("Triangulating inlier observations…  (this may take a moment)")
        self._status.setStyleSheet("color: #aaa; font-style: italic;")
        root.addWidget(self._status)

        # ── Main split: video ← | → plots ──────────────────────────────
        main_split = QSplitter(Qt.Orientation.Horizontal)

        self._video = _VideoPanel(self._conn, self._run_id)
        self._video.setMaximumWidth(480)
        main_split.addWidget(self._video)

        self._plot = _PlotCanvas()
        main_split.addWidget(self._plot)

        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        root.addWidget(main_split, stretch=1)

        # Wire video ↔ plot
        self._video.time_changed.connect(self._plot.move_cursor)
        self._plot.time_clicked.connect(self._video.seek)
        self._plot.time_clicked.connect(self._plot.move_cursor)
        self._plot.span_changed.connect(self._on_span_changed)

        # ── Summary table ───────────────────────────────────────────────
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        summary_layout.setContentsMargins(4, 4, 4, 4)
        summary_layout.setSpacing(2)

        # Header row
        header = QHBoxLayout()
        for text, stretch in [
            ("Measurement", 3), ("Template", 1), ("Span median", 1), ("Override", 1),
        ]:
            lbl = QLabel(f"<b>{text}</b>")
            header.addWidget(lbl, stretch=stretch)
        summary_layout.addLayout(header)

        for key in MEAS_KEYS:
            tmpl_val = self._tmpl.get(key, 0.0)
            row_widget = _MeasRow(key, tmpl_val)
            self._rows[key] = row_widget
            summary_layout.addWidget(row_widget)

        # Buttons
        btn_row = QHBoxLayout()
        self._recompute_btn = QPushButton("Recompute from span")
        self._recompute_btn.setEnabled(False)
        self._recompute_btn.clicked.connect(self._recompute_medians)
        btn_row.addWidget(self._recompute_btn)
        btn_row.addStretch()
        self._save_btn = QPushButton("Save scaled skeleton…")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_skeleton)
        btn_row.addWidget(self._save_btn)
        summary_layout.addLayout(btn_row)

        root.addWidget(summary_widget)

    def _start_worker(self) -> None:
        self._worker = _MeasWorker(self._db_path, self._run_id, parent=self)
        self._worker.finished.connect(self._on_data_ready)
        self._worker.start()

    # ------------------------------------------------------------------
    # Data ready

    def _on_data_ready(self, meas_df, error: str) -> None:
        if error and meas_df is None:
            self._status.setText(f"Error: {error}")
            self._status.setStyleSheet("color: red;")
            return

        self._meas_df = meas_df
        n = len(meas_df) if meas_df is not None and not meas_df.empty else 0
        self._status.setText(
            f"Loaded {n} tracker steps.  "
            "Drag the gold band on any plot to select the frame range for medians."
        )
        self._status.setStyleSheet("color: #ccc;")
        self._save_btn.setEnabled(bool(self._skel_yaml))

        if meas_df is not None and not meas_df.empty:
            self._plot.load_data(meas_df, self._tmpl)
            self._recompute_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Span / median

    def _on_span_changed(self, t_lo: float, t_hi: float) -> None:
        self._update_medians(t_lo, t_hi)

    def _recompute_medians(self) -> None:
        span = self._plot.current_span()
        if span:
            self._update_medians(*span)

    def _update_medians(self, t_lo: float, t_hi: float) -> None:
        if self._meas_df is None or self._meas_df.empty:
            return
        sel = self._meas_df[
            (self._meas_df["timestamp_s"] >= t_lo)
            & (self._meas_df["timestamp_s"] <= t_hi)
        ]
        for key, row_widget in self._rows.items():
            if key in sel.columns:
                med = float(sel[key].median())
                row_widget.set_median(med)

    # ------------------------------------------------------------------
    # Save

    def _save_skeleton(self) -> None:
        if not self._skel_yaml:
            QMessageBox.warning(self, "No skeleton", "No skeleton YAML loaded for this run.")
            return

        measurements = {key: row.value_m for key, row in self._rows.items()}

        name, ok = QInputDialog.getText(
            self, "Skeleton name",
            "Name for the scaled skeleton:",
            text=f"{self._skel_name} (scaled)",
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        # Get parent skeleton id
        parent_id_row = self._conn.execute(
            "SELECT skeleton_id FROM tracking_runs WHERE id=?", (self._run_id,)
        ).fetchone()
        parent_id = parent_id_row["skeleton_id"] if parent_id_row else None

        try:
            scaled_yaml = scale_skeleton_yaml(self._skel_yaml, measurements)
            new_id = import_skeleton_str(
                self._conn,
                scaled_yaml,
                name=name,
                parent_id=parent_id,
                source=f"Scaled from run {self._run_id[:8]}",
            )
            QMessageBox.information(
                self,
                "Saved",
                f"Skeleton '{name}' saved.\nID: {new_id[:16]}…\n\n"
                "Use it by selecting it when creating the next tracking run.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save skeleton:\n{exc}")

    # ------------------------------------------------------------------
    # Cleanup

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
        self._video.shutdown()
        super().closeEvent(event)

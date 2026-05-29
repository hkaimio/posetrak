"""skeleton_scaling_panel.py — Interactive skeleton scaling dialog.

Workflow
--------
1.  Dialog opens from TrackingRunPanel → loads skeleton template measurements.
2.  Background worker: load inlier obs from DB → DLT triangulate → per-step
    distances between marker pairs (femur, shin, upper_arm, …).
3.  Six measurement cards each show a time series.  The user drags a
    SpanSelector band on each graph independently to pick a "good pose" range.
4.  Span median, original template, and editable "new value" are shown inline
    below each graph.  "Use this" copies the span median; "Reset" restores the
    template value.  The current new value is also shown as a dotted line.
5.  "Save scaled skeleton" calls scale_skeleton_yaml() + import_skeleton_str().

Video scrubber
--------------
Left panel: QComboBox to pick one camera + PersonCropGridWidget showing that
camera's person crop with tracking overlay.  Clicking any point in a plot seeks
the video; the widget's time_changed signal moves the cursor line across all plots.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import yaml
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from app.ui.content_panels import PersonCropGridWidget
from posetrak.db.manage_skeleton import import_skeleton_str
from posetrak.db.scale_skeleton import scale_skeleton_yaml, template_measurements

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hard cap on the Y-axis maximum for all measurement graphs (cm).
# A single DLT outlier can produce values of hundreds of cm; capping at this
# value keeps the scale readable while still showing all realistic bone lengths.
Y_AXIS_HARD_CAP_CM: float = 100.0

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

            pos_by_step: dict[int, dict] = {}
            for (step, mname), pos in tri.items():
                pos_by_step.setdefault(step, {})
                pos_by_step[step][mname] = pos

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
# Per-measurement card: one subplot + inline controls
# ---------------------------------------------------------------------------


class _MeasCard(QWidget):
    """One measurement: subplot + controls (orig | span median | new value | buttons)."""

    time_clicked = Signal(float)

    def __init__(self, key: str, parent=None) -> None:
        super().__init__(parent)
        self._key = key
        self._tmpl_m: float = 0.0
        self._ts: np.ndarray = np.array([])
        self._vals_m: np.ndarray = np.array([])
        self._span_median_m: float | None = None
        self._cursor = None
        self._new_val_line = None
        self._selector: SpanSelector | None = None

        fig = Figure(figsize=(4, 2))
        fig.set_tight_layout(True)
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._ax = fig.add_subplot(111)
        self._canvas.mpl_connect("button_press_event", self._on_click)

        self._lbl_tmpl = QLabel("—")
        self._lbl_median = QLabel("—")
        self._lbl_median.setStyleSheet("color: #7af;")

        self._spin = QDoubleSpinBox()
        self._spin.setRange(1.0, 300.0)
        self._spin.setDecimals(1)
        self._spin.setSuffix(" cm")
        self._spin.setSingleStep(0.5)
        self._spin.setMaximumWidth(95)
        self._spin.valueChanged.connect(self._on_new_val_changed)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setMaximumWidth(55)
        self._reset_btn.clicked.connect(self._on_reset)

        self._use_btn = QPushButton("Use this")
        self._use_btn.setMaximumWidth(70)
        self._use_btn.setEnabled(False)
        self._use_btn.clicked.connect(self._on_use_this)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(4, 0, 4, 4)
        ctrl.setSpacing(4)
        ctrl.addWidget(QLabel("Orig:"))
        ctrl.addWidget(self._lbl_tmpl)
        ctrl.addWidget(QLabel("  Span median:"))
        ctrl.addWidget(self._lbl_median)
        ctrl.addWidget(QLabel("  New:"))
        ctrl.addWidget(self._spin)
        ctrl.addWidget(self._reset_btn)
        ctrl.addWidget(self._use_btn)
        ctrl.addStretch()

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(2)
        vbox.addWidget(self._canvas, stretch=1)
        vbox.addLayout(ctrl)

    # ------------------------------------------------------------------
    # Public API

    def load_data(self, ts: np.ndarray, vals_m: np.ndarray, tmpl_m: float) -> None:
        import pandas as pd

        self._ts = ts
        self._vals_m = vals_m
        self._tmpl_m = tmpl_m
        self._span_median_m = None
        self._new_val_line = None

        self._use_btn.setEnabled(False)
        self._lbl_median.setText("—")
        self._lbl_tmpl.setText(f"{tmpl_m * 100:.1f} cm" if tmpl_m else "—")

        if tmpl_m:
            self._spin.blockSignals(True)
            self._spin.setValue(round(tmpl_m * 100, 1))
            self._spin.blockSignals(False)

        ax = self._ax
        ax.clear()
        ax.set_title(MEAS_LABELS[self._key], fontsize=8, pad=2)
        ax.set_ylabel("cm", fontsize=7)
        ax.tick_params(labelsize=7)

        t_min = float(ts.min()) if ts.size else 0.0
        t_max = float(ts.max()) if ts.size else 1.0
        t_margin = max((t_max - t_min) * 0.01, 0.1)

        vals_cm = vals_m * 100
        if ts.size > 0:
            ax.plot(ts, vals_cm, color="lightsteelblue", lw=0.7, alpha=0.7)
            smoothed = (
                pd.Series(vals_cm).rolling(15, center=True, min_periods=1).median()
            )
            ax.plot(ts, smoothed.values, color="steelblue", lw=1.5)

        ax.set_xlim(t_min - t_margin, t_max + t_margin)

        # Y-axis: cap at Y_AXIS_HARD_CAP_CM so a single DLT outlier cannot
        # blow up the scale.  Natural upper bound = 99th-percentile of valid
        # data + 10 % headroom; also keep the template and new-value lines
        # in view.  Never show less than 10 cm of range.
        valid_cm = vals_cm[np.isfinite(vals_cm)] if ts.size > 0 else np.array([])
        data_hi = float(np.percentile(valid_cm, 99)) * 1.1 if valid_cm.size > 0 else 0.0
        ref_hi = max(
            tmpl_m * 100 * 1.25 if tmpl_m else 0.0,
            self._spin.value() * 1.25,
        )
        y_max = min(Y_AXIS_HARD_CAP_CM, max(data_hi, ref_hi, 10.0))
        ax.set_ylim(0.0, y_max)

        if tmpl_m:
            ax.axhline(tmpl_m * 100, color="#888", lw=1, ls="--", alpha=0.8)

        new_v = self._spin.value()
        self._new_val_line = ax.axhline(
            new_v, color="#f90", lw=1.5, ls=":", alpha=0.9
        )

        self._cursor = ax.axvline(
            t_min, color="crimson", lw=1, alpha=0.7, visible=False
        )

        if self._selector is not None:
            self._selector.set_visible(False)
        self._selector = SpanSelector(
            ax, self._on_span, "horizontal",
            useblit=True, interactive=True,
            drag_from_anywhere=True,
            props=dict(alpha=0.25, facecolor="gold"),
            handle_props=dict(alpha=0.6),
        )

        self._canvas.draw_idle()

    def move_cursor(self, t: float) -> None:
        if self._cursor is not None:
            self._cursor.set_xdata([t, t])
            self._cursor.set_visible(True)
            self._canvas.draw_idle()

    @property
    def value_m(self) -> float:
        return self._spin.value() / 100.0

    # ------------------------------------------------------------------
    # Internal

    def _on_span(self, xmin: float, xmax: float) -> None:
        if xmax - xmin < 0.05:
            return
        if self._ts.size > 0 and self._vals_m.size > 0:
            mask = (self._ts >= xmin) & (self._ts <= xmax) & np.isfinite(self._vals_m)
            sub = self._vals_m[mask]
            if sub.size > 0:
                self._span_median_m = float(np.nanmedian(sub))
                self._lbl_median.setText(f"{self._span_median_m * 100:.1f} cm")
                self._use_btn.setEnabled(True)

    def _on_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.button != 1:
            return
        self.time_clicked.emit(float(event.xdata))

    def _on_reset(self) -> None:
        if self._tmpl_m:
            self._spin.setValue(round(self._tmpl_m * 100, 1))

    def _on_use_this(self) -> None:
        if self._span_median_m is not None:
            self._spin.setValue(round(self._span_median_m * 100, 1))

    def _on_new_val_changed(self, v_cm: float) -> None:
        if self._new_val_line is not None:
            self._new_val_line.set_ydata([v_cm, v_cm])
            self._canvas.draw_idle()


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
        self._cards: dict[str, _MeasCard] = {}
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

        self._status = QLabel("Triangulating inlier observations…  (this may take a moment)")
        self._status.setStyleSheet("color: #aaa; font-style: italic;")
        root.addWidget(self._status)

        # ── Main split ──────────────────────────────────────────────────
        main_split = QSplitter(Qt.Orientation.Horizontal)

        # Left: camera selector + crop grid
        left_panel = QWidget()
        left_v = QVBoxLayout(left_panel)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(2)

        seq_row = self._conn.execute(
            "SELECT observation_sequence_id FROM tracking_runs WHERE id=?",
            (self._run_id,),
        ).fetchone()
        seq_id = seq_row["observation_sequence_id"] if seq_row else None

        self._video: PersonCropGridWidget | None = None
        if seq_id:
            self._video = PersonCropGridWidget(self._conn, seq_id)
            self._video.set_tracking_run(self._run_id)

            cam_labels = self._video.camera_labels()
            if cam_labels:
                self._cam_combo = QComboBox()
                for lbl in cam_labels:
                    self._cam_combo.addItem(lbl)
                self._video.set_camera_filter(cam_labels[0])
                self._cam_combo.currentTextChanged.connect(self._video.set_camera_filter)

                cam_row = QHBoxLayout()
                cam_row.setContentsMargins(4, 0, 4, 0)
                cam_row.addWidget(QLabel("Camera:"))
                cam_row.addWidget(self._cam_combo)
                cam_row.addStretch()
                left_v.addLayout(cam_row)

            self._video.time_changed.connect(self._on_time_changed)
            left_v.addWidget(self._video, stretch=1)

        main_split.addWidget(left_panel)

        # Right: 2-column grid of measurement cards in a scroll area
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_content = QWidget()
        cards_grid = QGridLayout(right_content)
        cards_grid.setSpacing(6)
        cards_grid.setContentsMargins(4, 4, 4, 4)

        for i, key in enumerate(MEAS_KEYS):
            row, col = divmod(i, 2)
            card = _MeasCard(key)
            card.time_clicked.connect(self._on_time_clicked)
            self._cards[key] = card
            cards_grid.addWidget(card, row, col)
            cards_grid.setColumnStretch(col, 1)

        for r in range(3):
            cards_grid.setRowStretch(r, 1)

        right_scroll.setWidget(right_content)
        main_split.addWidget(right_scroll)

        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        root.addWidget(main_split, stretch=1)

        # ── Bottom bar ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._save_btn = QPushButton("Save scaled skeleton…")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_skeleton)
        btn_row.addWidget(self._save_btn)
        root.addLayout(btn_row)

    def _start_worker(self) -> None:
        self._worker = _MeasWorker(self._db_path, self._run_id, parent=self)
        self._worker.finished.connect(self._on_data_ready)
        self._worker.start()

    # ------------------------------------------------------------------
    # Signals

    def _on_time_clicked(self, t: float) -> None:
        if self._video is not None:
            self._video.seek(t)
        for card in self._cards.values():
            card.move_cursor(t)

    def _on_time_changed(self, t: float) -> None:
        for card in self._cards.values():
            card.move_cursor(t)

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
            "Drag the gold band on any graph to pick a range; span median fills in automatically."
        )
        self._status.setStyleSheet("color: #ccc;")
        self._save_btn.setEnabled(bool(self._skel_yaml))

        if meas_df is not None and not meas_df.empty:
            ts = meas_df["timestamp_s"].values
            for key, card in self._cards.items():
                vals = meas_df[key].values if key in meas_df.columns else np.full(len(ts), float("nan"))
                card.load_data(ts, vals, self._tmpl.get(key, 0.0))

    # ------------------------------------------------------------------
    # Save

    def _save_skeleton(self) -> None:
        if not self._skel_yaml:
            QMessageBox.warning(self, "No skeleton", "No skeleton YAML loaded for this run.")
            return

        measurements = {key: card.value_m for key, card in self._cards.items()}

        name, ok = QInputDialog.getText(
            self, "Skeleton name",
            "Name for the scaled skeleton:",
            text=f"{self._skel_name} (scaled)",
        )
        if not ok or not name.strip():
            return
        name = name.strip()

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
        super().closeEvent(event)

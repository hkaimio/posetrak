# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""skeleton_scaling_panel.py — Interactive skeleton scaling dialog.

Workflow
--------
1.  Dialog opens from TrackingRunPanel → loads skeleton template measurements.
2.  Background worker: tracking_obs_results says which (camera, marker,
    step) observations the tracker accepted as inliers; the actual 2D point
    triangulated for each is read straight from the original pose_observations
    detection (undistorted the same way the tracker itself would), not from
    tracking_obs_results' own "actual pixel" fields -- see _MeasWorker's
    docstring for why that distinction matters. DLT triangulate → per-step
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
    """tracking_obs_results tells us which (camera, marker, step) observations
    the tracker accepted as inliers; the actual 2D point triangulated for each
    one is then read straight from the original pose detection, not from
    tracking_obs_results itself.

    This split matters: tracking_obs_results.obs_blob's "actual" pixel fields
    are whatever the tracker's own measurement model happened to compare
    against its prediction, and that is *not* always an absolute undistorted
    pixel position -- per posetrak/core/observation.hpp, it can also be a
    frame-to-frame pixel delta (VELOCITY mode,
    tracker_configs.velocity_mode_camera_ids) or a child-minus-parent offset
    (PAIR_DIFF/relative mode, tracker_configs.use_relative_observations) --
    and nothing in the stored data says which one applies to a given slot.
    Confirmed against a real capture, 2026-08-23: triangulating those values
    directly as if they were always positions produced measurements in the
    thousands of centimetres. The original pose_observations detection,
    undistorted the same way the tracker itself would, is unambiguously
    always a real pixel position regardless of which measurement mode the
    tracker used it in -- so that's the source of truth this re-triangulates
    from. See also
    docs/roadmap/features/observation-results-semantics.md.
    """

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
                "SELECT extrinsic_calibration_id, observation_sequence_id, skeleton_id "
                "FROM tracking_runs WHERE id=?",
                (self._run_id,),
            ).fetchone()
            session_row = conn.execute(
                "SELECT session_id FROM extrinsic_calibrations WHERE id=?",
                (run["extrinsic_calibration_id"],),
            ).fetchone()
            skel_row = conn.execute(
                "SELECT yaml_content FROM skeletons WHERE id=?", (run["skeleton_id"],)
            ).fetchone()

            cam_list = load_cameras_from_session(
                self._db_path,
                run["extrinsic_calibration_id"],
                session_row["session_id"],
            )
            cam_by_label = {c.get("instance_label") or c["label"]: c for c in cam_list}

            keypoint_idx_by_marker = _marker_openpose_indices(skel_row["yaml_content"])
            raw_by_instance = _load_raw_observations_by_camera(
                conn, run["observation_sequence_id"]
            )
            conn.close()

            tri: dict[tuple, np.ndarray] = {}
            step_ts: dict[int, float] = {}
            for (step, mname), grp in obs_df.groupby(["tracker_step", "marker_name"]):
                kp_idx = keypoint_idx_by_marker.get(mname)
                if kp_idx is None:
                    continue
                timestamp_s = float(grp["timestamp_s"].iloc[0])
                pts, Ps = [], []
                for row in grp.itertuples(index=False):
                    cam = cam_by_label.get(row.camera_label)
                    if cam is None:
                        continue
                    raw = raw_by_instance.get(cam["camera_instance_id"])
                    if raw is None:
                        continue
                    point = _nearest_raw_keypoint(raw, timestamp_s, kp_idx)
                    if point is None:
                        continue
                    px, py = _undistort_point(point[0], point[1], cam["K"], cam["dist"])
                    pts.append((px, py))
                    Ps.append(cam["P"])
                if len(pts) < 2:
                    continue
                result = _robust_triangulate(pts, Ps)
                if result is None:
                    continue
                pos, _cond = result
                tri[(step, mname)] = pos
                step_ts[step] = timestamp_s

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


def _marker_openpose_indices(yaml_content: str) -> dict[str, int]:
    """Map marker name -> its openpose_keypoint index (the index into
    pose_observations.kp_blob's COCO-133 layout), per the skeleton YAML's own
    markers list. Markers with no openpose_keypoint (rare) are omitted."""
    skel = yaml.safe_load(yaml_content)
    result = {}
    for m in skel.get("markers", []):
        idx = m.get("openpose_keypoint")
        if idx is not None:
            result[m["name"]] = int(idx)
    return result


def _load_raw_observations_by_camera(
    conn: sqlite3.Connection, sequence_id: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load every 'body'-source pose_observations row for *sequence_id*,
    grouped by camera_instance_id.

    Returns {camera_instance_id: (timestamps sorted ascending, kp array
    [n_frames, n_keypoints, 3] aligned with timestamps)}. A whole trial's
    worth of raw detections (a few hundred to low thousands of frames per
    camera) comfortably fits in memory decoded up front, which is much
    simpler than re-querying per (step, marker, camera).
    """
    rows = conn.execute(
        "SELECT camera_instance_id, timestamp_s, kp_blob FROM pose_observations "
        "WHERE sequence_id = ? AND source = 'body' ORDER BY camera_instance_id, timestamp_s",
        (sequence_id,),
    ).fetchall()

    by_instance: dict[str, list] = {}
    for row in rows:
        by_instance.setdefault(row["camera_instance_id"], []).append(row)

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for instance_id, instance_rows in by_instance.items():
        timestamps = np.array([r["timestamp_s"] for r in instance_rows], dtype=float)
        kps = np.stack([
            np.frombuffer(bytes(r["kp_blob"]), dtype="<f4").reshape(-1, 3)
            for r in instance_rows
        ])
        result[instance_id] = (timestamps, kps)
    return result


def _nearest_raw_keypoint(
    raw: tuple[np.ndarray, np.ndarray], timestamp_s: float, keypoint_idx: int
) -> tuple[float, float] | None:
    """Return (px, py) for *keypoint_idx* at the raw detection frame nearest
    *timestamp_s*, or None if that keypoint has zero confidence there (not
    actually detected) or the index is out of range for this camera's layout."""
    timestamps, kps = raw
    i = int(np.searchsorted(timestamps, timestamp_s))
    if i > 0 and (
        i == len(timestamps) or abs(timestamps[i - 1] - timestamp_s) <= abs(timestamps[i] - timestamp_s)
    ):
        i -= 1
    if keypoint_idx >= kps.shape[1]:
        return None
    x, y, conf = kps[i, keypoint_idx]
    return (float(x), float(y)) if conf > 0.0 else None


def _undistort_point(
    px: float, py: float, K: np.ndarray, dist: np.ndarray
) -> tuple[float, float]:
    """Remove lens distortion from a raw detected pixel, mirroring
    Camera::undistort()'s iterative Gauss-Newton method (cpp/src/core/camera.cpp)
    closely enough for this measurement tool's purposes -- this uses K for
    both normalisation and reprojection rather than separate K_original/K_new
    matrices, an approximation that's exact when they're equal and negligible
    otherwise (a calibration refinement pass typically shifts fx/fy/cx/cy by
    well under 1%)."""
    k1, k2, p1, p2 = dist[0], dist[1], dist[2], dist[3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    if abs(k1) < 1e-9 and abs(k2) < 1e-9 and abs(p1) < 1e-9 and abs(p2) < 1e-9:
        return px, py
    xn = (px - cx) / fx
    yn = (py - cy) / fy
    x0, y0 = xn, yn
    for _ in range(5):
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        dx = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        dy = p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        xn = (x0 - dx) / radial
        yn = (y0 - dy) / radial
    return xn * fx + cx, yn * fy + cy


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


def _reprojection_error_px(pos: np.ndarray, pt: tuple[float, float], P: np.ndarray) -> float:
    u, v, w = P @ np.append(pos, 1.0)
    return float(np.hypot(u / w - pt[0], v / w - pt[1]))


# A camera whose recorded "inlier" observation is nonetheless wrong (bad
# detection during fast/blurred motion, a mismatched measurement mode -- see
# the velocity_mode_camera_ids handling in _MeasWorker.run() -- or any other
# per-frame glitch that the tracker's own outlier check didn't happen to
# catch) still passes _dlt()'s cond<200 check as long as the *other* cameras
# agree with each other; the linear system stays well-conditioned even though
# the answer is nonsense. 50px is a coarse but effective backstop: normal
# detection jitter reprojects within a few pixels, while the kind of failure
# that produces a wildly wrong 3D point (e.g. a multi-thousand-cm "femur")
# reprojects hundreds to tens of thousands of pixels off.
_MAX_REPROJECTION_ERROR_PX: float = 50.0


def _robust_triangulate(
    pts: list[tuple[float, float]], Ps: list[np.ndarray]
) -> tuple[np.ndarray, float] | None:
    """DLT-triangulate, iteratively dropping the worst-fitting camera as long
    as the current result reprojects badly in some view and enough cameras
    remain for another attempt.

    More than one camera can have a bad-but-flagged-as-inlier observation in
    the same frame -- e.g. confirmed against a real capture, 2026-08-23: the
    very first tracked frame of a trial had two independently bad detections
    (both low-confidence enough that the tracker's own Mahalanobis outlier
    gate didn't catch either one, since low confidence inflates the assumed
    measurement noise). Dropping only a single camera isn't always enough to
    recover a good answer, so this keeps dropping the current worst offender
    one at a time.

    Returns None if no acceptable (conditioned, finite) solution is found,
    even after dropping cameras down to the minimum of 2.

    Tracks the best (lowest max-reprojection-error) attempt seen rather than
    just the last one: dropping a camera is a heuristic guess at which one
    is bad, and an unlucky guess (e.g. dropping a good camera while a second
    bad one remains) could make a later attempt worse, not better -- that
    shouldn't discard an earlier, better-fitting result.
    """
    best: tuple[np.ndarray, float] | None = None
    best_max_error = float("inf")
    while len(pts) >= 2:
        pos, cond = _dlt(pts, Ps)
        if cond > 200 or not np.all(np.isfinite(pos)):
            break
        errors = [_reprojection_error_px(pos, pt, P) for pt, P in zip(pts, Ps)]
        max_error = max(errors)
        if max_error < best_max_error:
            best, best_max_error = (pos, cond), max_error
        if max_error <= _MAX_REPROJECTION_ERROR_PX or len(pts) <= 2:
            break
        worst = int(np.argmax(errors))
        pts = [p for i, p in enumerate(pts) if i != worst]
        Ps = [P for i, P in enumerate(Ps) if i != worst]
    return best


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

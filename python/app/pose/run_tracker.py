"""run_tracker.py — Widget and dialog for running the posetrak tracker binary."""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from posetrak.tracker.runner import MultiPersonResult, PersonRunSpec, TrackerResult, default_binary_path
from posetrak.tracker.runner import run_multi_person_tracker as _run_multi_person_tracker
from posetrak.tracker.runner import run_tracker as _run_tracker

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOLS_DIR = _REPO_ROOT / "python" / "tools"
_DEFAULT_BINARY = default_binary_path()

# Adaptive process noise (Phase 1) joint gain scopes -- prototyping only.
#
# A single body-wide joint gain over-loosens fast-but-normal limb motion while
# barely engaging for the slower torso motion it's meant to help (see
# docs/roadmap/features/adaptive-process-noise/). Scoping by literal joint name
# rather than skeleton group: existing skeleton YAMLs only define one "main"
# group spanning the whole body, and this is prototyping-stage -- adding a
# finer group split to every person's skeleton file isn't worth it until we
# know whether joint-scoped gain is actually the right fix. Matches the
# "reallusion-no-waist"-style joint names currently in use; revisit (or make
# this configurable) if other skeleton naming shows up.
#
# Three anatomical scopes, not two: distal joints (wrist, ankle) are more
# accurate but move faster than proximal ones (elbow, knee, shoulder, hip), so
# lumping all limb joints into one "arms" scope with one reference velocity
# either over-loosens the fast distal joints or under-engages for the slower
# proximal ones. Torso (spine/neck/head) stays its own scope since it's slower
# again and was the original motivation for Mechanism A in the first place.
ADAPTIVE_NOISE_CORE_JOINTS: list[str] = ["spine1", "spine2", "neck1", "neck2", "head"]

ADAPTIVE_NOISE_PROXIMAL_JOINTS: list[str] = [
    f"{part}.{side}"
    for side in ("L", "R")
    for part in ["shoulder", "upper_arm", "forearm", "thigh", "shin"]
]

_FINGER_CHAINS = ["f_index", "f_middle", "f_ring", "f_pinky", "thumb"]
ADAPTIVE_NOISE_DISTAL_JOINTS: list[str] = [
    f"{part}.{side}" for side in ("L", "R") for part in ["hand", "foot", "toe"]
] + [
    f"{chain}.{segment:02d}.{side}"
    for side in ("L", "R")
    for chain in _FINGER_CHAINS
    for segment in (1, 2, 3)
]

# NIS-feedback (Mechanism B) "limbs" scope -- prototyping only, same rationale as
# ADAPTIVE_NOISE_CORE_JOINTS above. Deliberately the complement of that list
# (everything excluded from the core joint gain, i.e. proximal + distal
# combined): the natural allocation is Mechanism A+B for core, B-only for limbs
# as a safety net where core's own gain doesn't apply -- see
# docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md,
# "Mechanism B", *Case 3*. Kept as one combined bucket for the reactive
# safety net even though Mechanism A's proactive gain is now split finer.
NIS_FEEDBACK_LIMB_JOINTS: list[str] = (
    ADAPTIVE_NOISE_PROXIMAL_JOINTS + ADAPTIVE_NOISE_DISTAL_JOINTS
)

# Pose regularization (kinematic redundancy) Phase 1 chain -- prototyping only, see
# docs/roadmap/features/pose-regularization/pose-regularization-design.md. Scoped to
# spine1/spine2 per that note's Phase 1; extend only if the same pattern is confirmed
# on other chains (e.g. neck).
POSE_REG_SPINE_CHAIN: list[str] = ["spine1", "spine2"]

# Soft joint-limit repulsion Phase 1 scope -- prototyping only, see
# docs/roadmap/features/soft-joint-limits/soft-joint-limits-design.md. Scoped to the
# joints diagnosed as overshooting their own ball-joint limits during a fast bilateral
# motion; extend only if the same saturation pattern is confirmed on other joints.
SOFT_LIMIT_JOINT_NAMES: list[str] = ["upper_arm.L", "upper_arm.R"]


# ---------------------------------------------------------------------------
# Background thread
# ---------------------------------------------------------------------------


class _TrackerThread(QThread):
    """Runs run_tracker() in a background thread and emits Qt signals."""

    line_output = Signal(str)
    # exit_code as `object`, not `int`: on Windows, a process killed before
    # main() runs (e.g. a missing DLL dependency) reports its NTSTATUS code
    # as the return code, which is always > INT32_MAX -- marshalling that
    # into a C++ `int` signal argument overflows (a real crash reported as
    # "libshiboken: Overflow" once, traced to exactly this).
    tracking_finished = Signal(object, str)  # exit_code, run_id (empty str if None)

    def __init__(
        self,
        *,
        session_path: str,
        sequence_id: str,
        skeleton_id: str,
        config_id: str,
        output_dir: Path,
        binary_path: Path,
        person_id: int,
        start_time: float,
        end_time: float,
        smooth: bool,
    ) -> None:
        super().__init__()
        self._kwargs = dict(
            session_path=Path(session_path),
            sequence_id=sequence_id,
            skeleton_id=skeleton_id,
            config_id=config_id,
            output_dir=output_dir,
            binary_path=binary_path,
            person_id=person_id,
            start_time=start_time,
            end_time=end_time,
            smooth=smooth,
        )

    def run(self) -> None:
        result: TrackerResult = _run_tracker(**self._kwargs, on_progress=self.line_output.emit)
        self.tracking_finished.emit(result.exit_code, result.run_id or "")


class _MultiPersonTrackerThread(QThread):
    """Runs run_multi_person_tracker() in a background thread and emits Qt signals.

    Mirrors _TrackerThread above but for the ``--person``-mode multi-person
    CLI path (Stage 1 of the cross-person relative observations plan -- see
    docs/roadmap/features/error-improvements/phase5-cross-person-plan.md).
    """

    line_output = Signal(str)
    # run_ids: list[str | None], one per PersonRunSpec passed in, same order.
    tracking_finished = Signal(object, object)  # exit_code, run_ids

    def __init__(
        self,
        *,
        session_path: str,
        persons: list[PersonRunSpec],
        output_dir: Path,
        binary_path: Path,
        start_time: float,
        end_time: float,
        smooth: bool,
    ) -> None:
        super().__init__()
        self._kwargs = dict(
            session_path=Path(session_path),
            persons=persons,
            output_dir=output_dir,
            binary_path=binary_path,
            start_time=start_time,
            end_time=end_time,
            smooth=smooth,
        )

    def run(self) -> None:
        result: MultiPersonResult = _run_multi_person_tracker(
            **self._kwargs, on_progress=self.line_output.emit
        )
        self.tracking_finished.emit(result.exit_code, result.run_ids)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


class RunTrackerWidget(QWidget):
    """Configure and run the posetrak tracker against an open session database."""

    run_finished = Signal(str)  # emits tracking_run_id on successful completion

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._conn: sqlite3.Connection | None = None
        self._session_path: str | None = None
        self._run_id: str | None = None
        self._thread: _TrackerThread | _MultiPersonTrackerThread | None = None
        self._bvh_process = None
        self._sequence_cameras: list[str] = []
        self._velocity_cam_indices: set[int] = set()
        # (skeleton_id, display name), refreshed once per set_session() and
        # shared by every person row's skeleton combo.
        self._all_skeletons: list[tuple[str, str]] = []
        self._all_sequences: list[sqlite3.Row] = []
        # Display label per person row, captured at run start (for
        # _show_multi_results() -- the table itself may change before the
        # run finishes if the user starts editing it again).
        self._multi_run_labels: list[str] | None = None

        # ---- Configuration group ----------------------------------------
        self._proc_noise_std  = _float_spin(0.1,   0.0, 1000.0, 4)
        self._proc_vel_noise  = _float_spin(0.5,   0.0, 1000.0, 4)
        self._vel_half_life   = _float_spin(0.25,  0.0,   10.0, 4)

        self._vel_noise_gain_joint = _float_spin(0.0, 0.0, 100.0, 3)
        self._vel_noise_gain_joint.setToolTip(
            "Adaptive process noise (Phase 1): scales each joint DOF's own process noise\n"
            "by (1 + gain * |its velocity| / reference velocity). 0 = disabled (static noise,\n"
            "matches pre-Phase-1 behaviour)."
        )
        self._vel_noise_ref_joint = _float_spin(1.0, 1.0e-3, 1000.0, 3)
        self._vel_noise_ref_joint.setToolTip("Reference velocity for the joint gain above (rad/s).")
        self._vel_noise_gain_root = _float_spin(0.0, 0.0, 100.0, 3)
        self._vel_noise_gain_root.setToolTip(
            "Same as the joint gain above, but for the root's position/orientation DOFs\n"
            "(separate knob: root moves in metres/rad, joints in radians)."
        )
        self._vel_noise_ref_root = _float_spin(1.0, 1.0e-3, 1000.0, 3)
        self._vel_noise_ref_root.setToolTip(
            "Reference velocity for the root gain above (m/s for position, rad/s for orientation)."
        )
        self._vel_noise_gain_proximal = _float_spin(0.0, 0.0, 100.0, 3)
        self._vel_noise_gain_proximal.setToolTip(
            "Independent adaptive process noise gain for ADAPTIVE_NOISE_PROXIMAL_JOINTS\n"
            "(shoulder/upper_arm/forearm, hip/knee) -- excluded from the torso gain above\n"
            "to avoid over-loosening fast normal gestures, so give them their own,\n"
            "separately-tuned gain here instead of none at all. 0 = disabled."
        )
        self._vel_noise_ref_proximal = _float_spin(1.0, 1.0e-3, 1000.0, 3)
        self._vel_noise_ref_proximal.setToolTip(
            "Reference velocity for the proximal-limb gain above (rad/s)."
        )
        self._vel_noise_gain_distal = _float_spin(0.0, 0.0, 100.0, 3)
        self._vel_noise_gain_distal.setToolTip(
            "Independent adaptive process noise gain for ADAPTIVE_NOISE_DISTAL_JOINTS\n"
            "(wrist/hand/fingers, ankle/foot/toe) -- kept separate from the proximal scope\n"
            "above since distal joints are typically more accurate but move faster, so\n"
            "warrant their own (likely higher) reference velocity. 0 = disabled."
        )
        self._vel_noise_ref_distal = _float_spin(1.0, 1.0e-3, 1000.0, 3)
        self._vel_noise_ref_distal.setToolTip(
            "Reference velocity for the distal-limb gain above (rad/s)."
        )

        for gain, ref in (
            (self._vel_noise_gain_joint, self._vel_noise_ref_joint),
            (self._vel_noise_gain_root, self._vel_noise_ref_root),
            (self._vel_noise_gain_proximal, self._vel_noise_ref_proximal),
            (self._vel_noise_gain_distal, self._vel_noise_ref_distal),
        ):
            ref.setEnabled(False)
            gain.valueChanged.connect(lambda v, r=ref: r.setEnabled(v > 0.0))

        self._pose_reg_equal_split = _float_spin(0.0, 0.0, 10.0, 4)
        self._pose_reg_equal_split.setToolTip(
            "Pose regularization: pseudo-measurement pulling POSE_REG_SPINE_CHAIN's joint\n"
            "angles toward each other, per axis (stiffness = this std, radians; smaller =\n"
            "stronger pull). 0 = disabled."
        )
        self._pose_reg_rest_pose = _float_spin(0.0, 0.0, 10.0, 4)
        self._pose_reg_rest_pose.setToolTip(
            "Pose regularization: pseudo-measurement pulling POSE_REG_SPINE_CHAIN's joint\n"
            "angles toward zero, per axis (stiffness = this std, radians). 0 = disabled."
        )

        self._soft_limit_margin = _float_spin(0.0, 0.0, 1.5, 4)
        self._soft_limit_margin.setToolTip(
            "Soft joint-limit repulsion: width (radians) of the soft zone just inside\n"
            "each SOFT_LIMIT_JOINT_NAMES axis's hard limit. Only matters if the noise std\n"
            "below is nonzero."
        )
        self._soft_limit_noise_std = _float_spin(0.0, 0.0, 10.0, 4)
        self._soft_limit_noise_std.setToolTip(
            "Soft joint-limit repulsion: pseudo-measurement pulling SOFT_LIMIT_JOINT_NAMES's\n"
            "joint angles away from their own hard limits once inside the margin above\n"
            "(stiffness = this std, radians; smaller = stronger pull). 0 = disabled."
        )

        self._nis_feedback_threshold = _float_spin(1.5, 0.1, 100.0, 2)
        self._nis_feedback_threshold.setToolTip(
            "NIS-feedback safety net (Mechanism B): windowed NIS/DOF for the 'core' and\n"
            "'limbs' scopes above this triggers a temporary process-noise multiplier.\n"
            "Only takes effect if 'Enable NIS feedback' is checked."
        )
        self._nis_feedback_max_mult = _float_spin(10.0, 1.0, 1000.0, 1)
        self._nis_feedback_max_mult.setToolTip("Cap on the variance-domain multiplier above.")
        self._nis_feedback_enabled = QCheckBox()
        self._nis_feedback_enabled.setToolTip(
            "Enable the NIS-feedback safety net, scoped to ADAPTIVE_NOISE_CORE_JOINTS\n"
            "('core') and NIS_FEEDBACK_LIMB_JOINTS ('limbs') -- see\n"
            "docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md."
        )
        self._nis_feedback_enabled.toggled.connect(self._nis_feedback_threshold.setEnabled)
        self._nis_feedback_enabled.toggled.connect(self._nis_feedback_max_mult.setEnabled)
        self._nis_feedback_threshold.setEnabled(False)
        self._nis_feedback_max_mult.setEnabled(False)

        self._pose_noise      = _float_spin(0.0,   0.0, 1.0e6,  2)
        self._calib_noise     = _float_spin(60.0,  0.0, 1.0e6,  2)
        self._outlier_thresh  = _float_spin(4.0,   0.1,   50.0, 2)
        self._tracker_fps     = _float_spin(120.0, 1.0,  500.0, 1)

        self._vel_cam_label = QLabel("None")
        vel_cam_edit_btn = QPushButton("Edit…")
        vel_cam_edit_btn.setFixedWidth(60)
        vel_cam_edit_btn.clicked.connect(self._edit_velocity_cameras)
        vel_cam_row = QHBoxLayout()
        vel_cam_row.addWidget(self._vel_cam_label, 1)
        vel_cam_row.addWidget(vel_cam_edit_btn)

        self._use_relative = QCheckBox()
        self._use_relative.setChecked(False)
        self._use_relative.setToolTip(
            "Emit child-minus-parent pixel observations alongside absolute positions.\n"
            "Calibration error cancels in the difference; requires pose_noise_std > 0."
        )
        self._relative_min_conf = _float_spin(0.5, 0.0, 1.0, 2)
        self._relative_min_conf.setToolTip(
            "Minimum keypoint confidence for both child and parent to form a relative pair."
        )
        self._use_relative.toggled.connect(self._relative_min_conf.setEnabled)
        self._relative_min_conf.setEnabled(False)

        self._cross_pair_max_px = _float_spin(0.0, 0.0, 9999.0, 1)
        self._cross_pair_max_px.setToolTip(
            "Pixel radius for spatial cross-pair relative observations.\n"
            "Pairs of visible markers within this distance and > 2 skeleton hops apart\n"
            "emit an additional RELATIVE observation. 0 = disabled (Phase 4)."
        )
        self._cross_pair_max_n = QSpinBox()
        self._cross_pair_max_n.setRange(1, 999)
        self._cross_pair_max_n.setValue(10)
        self._cross_pair_max_n.setToolTip(
            "Maximum spatial cross-pairs per frame per camera (closest pairs kept)."
        )
        self._cross_pair_max_px.valueChanged.connect(
            lambda v: self._cross_pair_max_n.setEnabled(v > 0.0)
        )
        self._cross_pair_max_n.setEnabled(False)

        self._cross_person_max_world_mm = _float_spin(0.0, 0.0, 99999.0, 1)
        self._cross_person_max_world_mm.setToolTip(
            "3D world-space marker-pair distance gate (mm) for cross-person\n"
            "PAIR_DIFF anchoring between people tracked together below\n"
            "(e.g. ukemi throws, handshakes). 0 = disabled (Phase 5)."
        )
        self._cross_person_min_conf = _float_spin(0.5, 0.0, 1.0, 2)
        self._cross_person_min_conf.setToolTip(
            "Minimum keypoint confidence for both people's detections to form\n"
            "a cross-person anchor."
        )
        self._cross_person_max_n = QSpinBox()
        self._cross_person_max_n.setRange(1, 999)
        self._cross_person_max_n.setValue(10)
        self._cross_person_max_n.setToolTip(
            "Maximum cross-person anchor observations per person pair per\n"
            "camera per frame (closest pairs kept)."
        )
        self._cross_person_max_world_mm.valueChanged.connect(
            lambda v: (
                self._cross_person_min_conf.setEnabled(v > 0.0),
                self._cross_person_max_n.setEnabled(v > 0.0),
            )
        )
        self._cross_person_min_conf.setEnabled(False)
        self._cross_person_max_n.setEnabled(False)

        config_form = QFormLayout()
        config_form.addRow("Process noise std:", self._proc_noise_std)
        config_form.addRow("Velocity noise std:", self._proc_vel_noise)
        config_form.addRow("Velocity half-life (s):", self._vel_half_life)
        config_form.addRow("Adaptive noise gain (joint):", self._vel_noise_gain_joint)
        config_form.addRow("Adaptive noise ref vel (joint, rad/s):", self._vel_noise_ref_joint)
        config_form.addRow("Adaptive noise gain (root):", self._vel_noise_gain_root)
        config_form.addRow("Adaptive noise ref vel (root, m/s, rad/s):", self._vel_noise_ref_root)
        config_form.addRow("Adaptive noise gain (proximal):", self._vel_noise_gain_proximal)
        config_form.addRow(
            "Adaptive noise ref vel (proximal, rad/s):", self._vel_noise_ref_proximal
        )
        config_form.addRow("Adaptive noise gain (distal):", self._vel_noise_gain_distal)
        config_form.addRow("Adaptive noise ref vel (distal, rad/s):", self._vel_noise_ref_distal)
        config_form.addRow("Pose reg equal-split std (spine1/2, rad):", self._pose_reg_equal_split)
        config_form.addRow("Pose reg rest-pose std (spine1/2, rad):", self._pose_reg_rest_pose)
        config_form.addRow("Soft joint-limit margin (upper_arm, rad):", self._soft_limit_margin)
        config_form.addRow(
            "Soft joint-limit noise std (upper_arm, rad):", self._soft_limit_noise_std
        )
        config_form.addRow("Enable NIS feedback (core+limbs):", self._nis_feedback_enabled)
        config_form.addRow("NIS feedback threshold:", self._nis_feedback_threshold)
        config_form.addRow("NIS feedback max multiplier:", self._nis_feedback_max_mult)
        config_form.addRow("Pose noise std (px in model):", self._pose_noise)
        config_form.addRow("Calib noise std (px in video):", self._calib_noise)
        config_form.addRow("Outlier threshold:", self._outlier_thresh)
        config_form.addRow("Tracker FPS:", self._tracker_fps)
        config_form.addRow("Velocity cameras:", vel_cam_row)
        config_form.addRow("Relative observations:", self._use_relative)
        config_form.addRow("Relative min confidence:", self._relative_min_conf)
        config_form.addRow("Cross-pair radius (px):", self._cross_pair_max_px)
        config_form.addRow("Cross-pair max count:", self._cross_pair_max_n)
        config_form.addRow("Cross-person distance (mm):", self._cross_person_max_world_mm)
        config_form.addRow("Cross-person min confidence:", self._cross_person_min_conf)
        config_form.addRow("Cross-person max count:", self._cross_person_max_n)

        config_box = QGroupBox("Tracker configuration")
        config_box.setLayout(config_form)

        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        config_scroll.setWidget(config_box)

        # ---- People group -------------------------------------------------
        # One row per person: each keeps their own sequence AND skeleton (people
        # have different body proportions, and in practice every detected
        # person already lives in their own pose_observation_sequences row --
        # see docs/roadmap/features/error-improvements/phase5-cross-person-plan.md).
        # Row 0 is the primary person and is never removable; its sequence
        # choice pins which trial "Add person…" offers other people from.
        self._people_table = QTableWidget(0, 3)
        self._people_table.setHorizontalHeaderLabels(["Sequence", "Skeleton", ""])
        header = self._people_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._people_table.verticalHeader().setVisible(False)
        self._people_table.setMinimumHeight(90)
        self._people_table.setToolTip(
            "Row 1 is the primary person. \"Add person…\" tracks additional people\n"
            "from the same trial alongside them, interleaved frame-by-frame\n"
            "(enables cross-person anchoring if Cross-person distance above is\n"
            "set > 0). Each person keeps their own sequence and skeleton."
        )

        add_person_btn = QPushButton("Add person…")
        add_person_btn.clicked.connect(self._add_person_row)

        people_layout = QVBoxLayout()
        people_layout.addWidget(self._people_table)
        add_person_row = QHBoxLayout()
        add_person_row.addStretch()
        add_person_row.addWidget(add_person_btn)
        people_layout.addLayout(add_person_row)

        people_box = QGroupBox("People")
        people_box.setLayout(people_layout)

        # ---- Run group --------------------------------------------------
        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setPlaceholderText(
            "Leave empty for <session-dir>/posetrak_results/<shot>/<skeleton>/"
        )
        out_browse_btn = QPushButton("Browse…")
        out_browse_btn.clicked.connect(self._browse_out_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_dir_edit, 1)
        out_row.addWidget(out_browse_btn)

        self._binary_edit = QLineEdit(str(_DEFAULT_BINARY))
        bin_browse_btn = QPushButton("Browse…")
        bin_browse_btn.clicked.connect(self._browse_binary)
        bin_row = QHBoxLayout()
        bin_row.addWidget(self._binary_edit, 1)
        bin_row.addWidget(bin_browse_btn)

        run_form = QFormLayout()
        run_form.addRow("Output directory:", out_row)
        run_form.addRow("Tracker binary:", bin_row)

        run_box = QGroupBox("Run")
        run_box.setLayout(run_form)

        # ---- Run button -------------------------------------------------
        self._run_btn = QPushButton("Run Tracker")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._start_tracking)

        # ---- Progress group (hidden until run starts) -------------------
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._status_label = QLabel("")
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self._log.setFont(mono)

        prog_layout = QVBoxLayout()
        prog_layout.addWidget(self._progress_bar)
        prog_layout.addWidget(self._status_label)
        prog_layout.addWidget(self._log)
        self._prog_box = QGroupBox("Progress")
        self._prog_box.setLayout(prog_layout)
        self._prog_box.setVisible(False)

        # ---- Results group (hidden until run completes) ----------------
        self._results_label = QLabel("")
        self._results_label.setWordWrap(True)
        self._export_bvh_btn = QPushButton("Export BVH…")
        self._export_bvh_btn.clicked.connect(self._export_bvh)

        results_layout = QVBoxLayout()
        results_layout.addWidget(self._results_label)
        results_layout.addWidget(self._export_bvh_btn)
        self._results_box = QGroupBox("Results")
        self._results_box.setLayout(results_layout)
        self._results_box.setVisible(False)

        # ---- Root layout ------------------------------------------------
        # Only the (long) tracker-configuration section scrolls -- people,
        # run controls, the Run button, and progress/results always stay
        # visible without needing to resize the window.
        root = QVBoxLayout(self)
        root.addWidget(people_box)
        root.addWidget(run_box)
        root.addWidget(config_scroll, 1)
        root.addWidget(self._run_btn)
        root.addWidget(self._prog_box)
        root.addWidget(self._results_box)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_session(self, conn: sqlite3.Connection, session_path: str) -> None:
        """Supply open session connection and the path to its .db file."""
        self._conn = conn
        self._session_path = session_path
        self._refresh_skeletons()
        self._refresh_sequences()
        self._update_run_btn()

    def preselect_sequence(self, seq_id: str) -> None:
        """Pre-select and lock the primary person's sequence to *seq_id*.

        Call after set_session().  The combo is disabled so the user cannot
        change the sequence when this widget is embedded in a PersonPanel.
        """
        combo = self._people_table.cellWidget(0, 0)
        if combo is not None:
            for i in range(combo.count()):
                item_seq_id, _, _ = combo.itemData(i)
                if item_seq_id == seq_id:
                    combo.setCurrentIndex(i)
                    break
            combo.setEnabled(False)
        self._update_run_btn()

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def _refresh_skeletons(self) -> None:
        self._all_skeletons = []
        if self._conn is None:
            return
        rows = self._conn.execute(
            "SELECT id, name FROM skeletons ORDER BY name"
        ).fetchall()
        self._all_skeletons = [(r["id"], r["name"] or r["id"][:12]) for r in rows]

    def _skeleton_name(self, skel_id: str | None) -> str | None:
        for sid, name in self._all_skeletons:
            if sid == skel_id:
                return name
        return None

    def _refresh_sequences(self) -> None:
        self._all_sequences = []
        if self._conn is not None:
            self._all_sequences = self._conn.execute(
                "SELECT pos.id AS seq_id, sh.label AS shot_label, sh.capture_number,"
                "       pos.time_start_s, pos.time_end_s,"
                "       sh.extrinsic_calibration_id, pos.sync_config_id"
                " FROM pose_observation_sequences pos"
                " JOIN captures sh ON sh.id = pos.shot_id"
                " ORDER BY sh.capture_number, pos.time_start_s"
            ).fetchall()
        self._rebuild_people_table()

    @staticmethod
    def _sequence_label(r: sqlite3.Row) -> str:
        shot = r["shot_label"] or f"capture{r['capture_number']:03d}"
        duration = r["time_end_s"] - r["time_start_s"]
        label = f"{shot}  [{r['time_start_s']:.1f}–{r['time_end_s']:.1f}s, {duration:.1f}s]"
        missing = []
        if not r["sync_config_id"]:
            missing.append("no sync")
        if not r["extrinsic_calibration_id"]:
            missing.append("no extrinsics")
        if missing:
            label += f"  ⚠ {', '.join(missing)}"
        return label

    def _make_sequence_combo(self, rows: list[sqlite3.Row]) -> QComboBox:
        combo = QComboBox()
        for r in rows:
            combo.addItem(
                self._sequence_label(r), (r["seq_id"], r["time_start_s"], r["time_end_s"])
            )
        return combo

    def _make_skeleton_combo(self) -> QComboBox:
        combo = QComboBox()
        for skel_id, name in self._all_skeletons:
            combo.addItem(name, skel_id)
        return combo

    def _row_sequence_data(self, row: int) -> tuple[str, float, float] | None:
        combo = self._people_table.cellWidget(row, 0)
        return combo.currentData() if combo is not None else None

    def _row_skeleton_id(self, row: int) -> str | None:
        combo = self._people_table.cellWidget(row, 1)
        return combo.currentData() if combo is not None else None

    def _rebuild_people_table(self) -> None:
        """Reset the People table to a single (unremovable) primary row."""
        self._people_table.setRowCount(0)
        self._insert_person_row(self._all_sequences, removable=False)
        primary_combo = self._people_table.cellWidget(0, 0)
        primary_combo.currentIndexChanged.connect(self._on_primary_sequence_changed)
        self._on_primary_sequence_changed()

    def _insert_person_row(self, candidates: list[sqlite3.Row], *, removable: bool) -> int:
        row = self._people_table.rowCount()
        self._people_table.insertRow(row)
        self._people_table.setCellWidget(row, 0, self._make_sequence_combo(candidates))
        self._people_table.setCellWidget(row, 1, self._make_skeleton_combo())
        if removable:
            remove_btn = QPushButton("✕")
            remove_btn.setFixedWidth(28)
            remove_btn.setToolTip("Remove this person")
            remove_btn.clicked.connect(self._make_remove_handler(remove_btn))
            self._people_table.setCellWidget(row, 2, remove_btn)
        self._update_run_btn()
        return row

    def _make_remove_handler(self, button: QPushButton):
        def _remove() -> None:
            for r in range(self._people_table.rowCount()):
                if self._people_table.cellWidget(r, 2) is button:
                    self._people_table.removeRow(r)
                    break
            self._update_run_btn()

        return _remove

    def _add_person_row(self) -> None:
        primary_data = self._row_sequence_data(0)
        if primary_data is None:
            QMessageBox.information(
                self, "Select a sequence first",
                "Choose the primary person's Sequence before adding more people.",
            )
            return
        primary_seq_id = primary_data[0]

        candidates, note = self._trial_scoped_candidates(primary_seq_id)
        used_ids = {
            data[0]
            for row in range(self._people_table.rowCount())
            if (data := self._row_sequence_data(row)) is not None
        }
        candidates = [r for r in candidates if r["seq_id"] not in used_ids]

        if not candidates:
            msg = "No other sequences found in the same trial as the primary person."
            if note:
                msg += "\n\n" + note
            QMessageBox.information(self, "No other people", msg)
            return

        self._insert_person_row(candidates, removable=True)

    def _trial_scoped_candidates(
        self, primary_seq_id: str
    ) -> tuple[list[sqlite3.Row], str | None]:
        """Other sequences from the same trial as *primary_seq_id*.

        Trial is the natural scope for "track together": cross-person
        anchoring needs shared cameras/world space, and sequences from one
        trial share that by construction (same shot_id/sync_config_id).
        Falls back to same-capture + overlapping time range if the primary's
        detection run has no trial_id set (nullable in the schema), with a
        note explaining the approximation.
        """
        primary = self._conn.execute(
            "SELECT pos.shot_id, pos.time_start_s, pos.time_end_s, dr.trial_id"
            " FROM pose_observation_sequences pos"
            " LEFT JOIN detection_runs dr ON dr.id = pos.detection_run_id"
            " WHERE pos.id = ?",
            (primary_seq_id,),
        ).fetchone()
        if primary is None:
            return [], None

        if primary["trial_id"] is not None:
            rows = self._conn.execute(
                "SELECT pos.id AS seq_id, sh.label AS shot_label, sh.capture_number,"
                "       pos.time_start_s, pos.time_end_s,"
                "       sh.extrinsic_calibration_id, pos.sync_config_id"
                " FROM pose_observation_sequences pos"
                " JOIN captures sh ON sh.id = pos.shot_id"
                " JOIN detection_runs dr ON dr.id = pos.detection_run_id"
                " WHERE dr.trial_id = ? AND pos.id != ?"
                " ORDER BY pos.time_start_s",
                (primary["trial_id"], primary_seq_id),
            ).fetchall()
            return list(rows), None

        rows = self._conn.execute(
            "SELECT pos.id AS seq_id, sh.label AS shot_label, sh.capture_number,"
            "       pos.time_start_s, pos.time_end_s,"
            "       sh.extrinsic_calibration_id, pos.sync_config_id"
            " FROM pose_observation_sequences pos"
            " JOIN captures sh ON sh.id = pos.shot_id"
            " WHERE pos.shot_id = ? AND pos.id != ?"
            "       AND pos.time_start_s < ? AND pos.time_end_s > ?"
            " ORDER BY pos.time_start_s",
            (primary["shot_id"], primary_seq_id, primary["time_end_s"], primary["time_start_s"]),
        ).fetchall()
        note = (
            "This capture's detection run has no trial assigned, so sequences were "
            "matched by overlapping time range within the same capture instead of by "
            "trial (less precise -- set a trial for exact scoping)."
        )
        return list(rows), note

    def _update_run_btn(self) -> None:
        ok = (
            len(self._all_skeletons) > 0
            and self._people_table.rowCount() > 0
            and self._row_sequence_data(0) is not None
        )
        self._run_btn.setEnabled(ok)

    def _on_primary_sequence_changed(self) -> None:
        data = self._row_sequence_data(0)
        if data is None or self._conn is None:
            self._sequence_cameras = []
        else:
            seq_id, _, _ = data
            self._sequence_cameras = self._cameras_for_sequence(seq_id)
        self._velocity_cam_indices = set()
        self._update_velocity_cam_label()
        # Additional rows were scoped to the previous primary's trial; once
        # the primary changes that scope may no longer be valid, so drop them
        # rather than leave stale candidates in place.
        while self._people_table.rowCount() > 1:
            self._people_table.removeRow(1)
        self._update_run_btn()

    def _cameras_for_sequence(self, seq_id: str) -> list[str]:
        row = self._conn.execute(
            "SELECT pos.sync_config_id FROM pose_observation_sequences pos WHERE pos.id = ?",
            (seq_id,),
        ).fetchone()
        if not row or not row["sync_config_id"]:
            return []
        sync_id = row["sync_config_id"]
        rows = self._conn.execute(
            "SELECT ci.label"
            " FROM capture_videos sv"
            " JOIN captures sh ON sh.id = sv.shot_id"
            " JOIN sync_configs scfg ON scfg.shot_id = sh.id"
            " JOIN camera_instances ci ON ci.id = sv.camera_instance_id"
            " WHERE scfg.id = ?"
            " ORDER BY ci.label ASC",
            (sync_id,),
        ).fetchall()
        return [r["label"] for r in rows]

    def _update_velocity_cam_label(self) -> None:
        if not self._velocity_cam_indices or not self._sequence_cameras:
            self._vel_cam_label.setText("None")
        else:
            names = [
                self._sequence_cameras[i]
                for i in sorted(self._velocity_cam_indices)
                if i < len(self._sequence_cameras)
            ]
            self._vel_cam_label.setText(", ".join(names) if names else "None")

    def _edit_velocity_cameras(self) -> None:
        if not self._sequence_cameras:
            QMessageBox.information(self, "No cameras", "Select a sequence with cameras first.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Velocity mode cameras")
        layout = QVBoxLayout(dlg)
        label = QLabel(
            "Cameras in velocity mode use keypoint displacement between frames as the "
            "measurement instead of absolute position. Select cameras with poor or "
            "uncertain absolute calibration."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        checkboxes: list[QCheckBox] = []
        for i, cam_label in enumerate(self._sequence_cameras):
            cb = QCheckBox(cam_label)
            cb.setChecked(i in self._velocity_cam_indices)
            checkboxes.append(cb)
            layout.addWidget(cb)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._velocity_cam_indices = {i for i, cb in enumerate(checkboxes) if cb.isChecked()}
            self._update_velocity_cam_label()

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self._out_dir_edit.setText(path)

    def _browse_binary(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select posetrak binary", "", "All files (*)")
        if path:
            self._binary_edit.setText(path)

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def _start_tracking(self) -> None:
        if self._conn is None or self._session_path is None:
            return

        binary = Path(self._binary_edit.text())
        if not binary.exists():
            QMessageBox.critical(
                self,
                "Binary not found",
                f"Cannot find tracker binary:\n{binary}\n\n"
                "Build the optimised release first:\n"
                "  meson setup optbuild --buildtype=release\n"
                "  meson compile -C optbuild",
            )
            return

        people: list[tuple[str, str, float, float, str]] = []
        for row in range(self._people_table.rowCount()):
            data = self._row_sequence_data(row)
            skel_id = self._row_skeleton_id(row)
            if data is None or skel_id is None:
                QMessageBox.critical(
                    self, "Cannot run tracker",
                    "Every person needs both a sequence and a skeleton selected.",
                )
                return
            seq_id, t0, t1 = data
            label = self._people_table.cellWidget(row, 0).currentText()
            people.append((seq_id, skel_id, t0, t1, label))

        primary_seq_id, primary_skel_id, time_start_s, time_end_s, _ = people[0]

        err = self._check_sequence_ready(primary_seq_id)
        if err:
            QMessageBox.critical(self, "Cannot run tracker", err)
            return

        out_dir = self._resolve_out_dir(primary_seq_id, primary_skel_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        config_id = self._create_config()
        self._run_id = None
        self._multi_run_labels = None

        self._progress_bar.setValue(0)
        self._status_label.setText("Starting…")
        self._log.clear()
        self._prog_box.setVisible(True)
        self._results_box.setVisible(False)
        self._run_btn.setEnabled(False)

        if len(people) > 1:
            self._multi_run_labels = [label for _, _, _, _, label in people]
            persons = [
                PersonRunSpec(seq_id, skel_id, config_id, 0)
                for seq_id, skel_id, _, _, _ in people
            ]
            thread = _MultiPersonTrackerThread(
                session_path=self._session_path,
                persons=persons,
                output_dir=out_dir,
                binary_path=binary,
                start_time=time_start_s,
                end_time=time_end_s,
                smooth=True,
            )
            thread.line_output.connect(self._on_output)
            thread.tracking_finished.connect(self._on_multi_finished)
        else:
            thread = _TrackerThread(
                session_path=self._session_path,
                sequence_id=primary_seq_id,
                skeleton_id=primary_skel_id,
                config_id=config_id,
                output_dir=out_dir,
                binary_path=binary,
                person_id=0,
                start_time=time_start_s,
                end_time=time_end_s,
                smooth=True,
            )
            thread.line_output.connect(self._on_output)
            thread.tracking_finished.connect(self._on_finished)

        thread.start()
        self._thread = thread

    def _check_sequence_ready(self, seq_id: str) -> str | None:
        """Return an error message if the sequence is missing sync or extrinsics, else None."""
        row = self._conn.execute(
            "SELECT s.label, s.extrinsic_calibration_id, pos.sync_config_id"
            " FROM pose_observation_sequences pos"
            " JOIN captures s ON s.id = pos.shot_id"
            " WHERE pos.id = ?",
            (seq_id,),
        ).fetchone()
        if row is None:
            return f"Sequence '{seq_id}' not found in the database."
        shot = row["label"] or seq_id[:12]
        if not row["sync_config_id"]:
            return (
                f"Capture \"{shot}\" has no sync configuration.\n\n"
                "Run the setup wizard and complete the Camera Synchronisation step "
                "before tracking."
            )
        if not row["extrinsic_calibration_id"]:
            return (
                f"Capture \"{shot}\" has no extrinsic calibration.\n\n"
                "Run the setup wizard and complete the Extrinsics step before tracking."
            )
        return None

    def _resolve_out_dir(self, seq_id: str, skel_id: str) -> Path:
        explicit = self._out_dir_edit.text().strip()
        if explicit:
            return Path(explicit)
        db_dir = Path(self._session_path).parent
        seq_row = self._conn.execute(
            "SELECT sh.label, sh.capture_number"
            " FROM pose_observation_sequences pos"
            " JOIN captures sh ON sh.id = pos.shot_id"
            " WHERE pos.id = ?",
            (seq_id,),
        ).fetchone()
        shot = (
            seq_row["label"] if seq_row and seq_row["label"]
            else f"capture{seq_row['capture_number']:03d}" if seq_row
            else "capture"
        )
        skel_name = (self._skeleton_name(skel_id) or "skeleton").replace(" ", "_")
        return db_dir / "posetrak_results" / shot / skel_name / "tracking"

    def _create_config(self) -> str:
        import datetime as dt
        import json
        from posetrak.db.db import generate_id
        config_id = generate_id()
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        vel_ids = sorted(self._velocity_cam_indices) if self._velocity_cam_indices else None
        vel_ids_json = json.dumps(vel_ids) if vel_ids is not None else None
        use_rel = 1 if self._use_relative.isChecked() else 0
        rel_min_conf = self._relative_min_conf.value() if use_rel else None
        cross_px = self._cross_pair_max_px.value()
        cross_n = self._cross_pair_max_n.value() if cross_px > 0.0 else None
        cross_px_val = cross_px if cross_px > 0.0 else None
        cross_person_mm = self._cross_person_max_world_mm.value()
        cross_person_mm_val = cross_person_mm if cross_person_mm > 0.0 else None
        cross_person_min_conf = self._cross_person_min_conf.value() if cross_person_mm_val else None
        cross_person_n = self._cross_person_max_n.value() if cross_person_mm_val else None
        joint_gain = self._vel_noise_gain_joint.value()
        joint_names_json = json.dumps(ADAPTIVE_NOISE_CORE_JOINTS) if joint_gain > 0.0 else None
        vel_scopes = []
        if self._vel_noise_gain_proximal.value() > 0.0:
            vel_scopes.append({
                "name": "proximal",
                "joint_names": ADAPTIVE_NOISE_PROXIMAL_JOINTS,
                "gain": self._vel_noise_gain_proximal.value(),
                "vel_ref": self._vel_noise_ref_proximal.value(),
            })
        if self._vel_noise_gain_distal.value() > 0.0:
            vel_scopes.append({
                "name": "distal",
                "joint_names": ADAPTIVE_NOISE_DISTAL_JOINTS,
                "gain": self._vel_noise_gain_distal.value(),
                "vel_ref": self._vel_noise_ref_distal.value(),
            })
        vel_scopes_json = json.dumps(vel_scopes) if vel_scopes else None
        pose_reg_equal_split = self._pose_reg_equal_split.value()
        pose_reg_rest_pose = self._pose_reg_rest_pose.value()
        pose_reg_enabled = pose_reg_equal_split > 0.0 or pose_reg_rest_pose > 0.0
        pose_reg_joint_names_json = json.dumps(POSE_REG_SPINE_CHAIN) if pose_reg_enabled else None
        soft_limit_margin = self._soft_limit_margin.value()
        soft_limit_noise_std = self._soft_limit_noise_std.value()
        soft_limit_enabled = soft_limit_noise_std > 0.0
        soft_limit_joint_names_json = (
            json.dumps(SOFT_LIMIT_JOINT_NAMES) if soft_limit_enabled else None
        )
        nis_feedback_scopes_json = None
        if self._nis_feedback_enabled.isChecked():
            nis_feedback_scopes_json = json.dumps([
                {"name": "core", "joint_names": ADAPTIVE_NOISE_CORE_JOINTS},
                {"name": "limbs", "joint_names": NIS_FEEDBACK_LIMB_JOINTS},
            ])
        with self._conn:
            self._conn.execute(
                "INSERT INTO tracker_configs"
                " (id, name, parent_id, created_at,"
                "  process_noise_std, process_noise_vel_std, velocity_half_life_s,"
                "  measurement_noise_std, pose_noise_std, outlier_threshold, tracker_fps,"
                "  velocity_mode_camera_ids,"
                "  use_relative_observations, relative_min_confidence,"
                "  cross_pair_max_px, cross_pair_max_n,"
                "  cross_person_max_world_mm, cross_person_min_confidence, cross_person_max_n,"
                "  process_noise_vel_gain_joint, process_noise_vel_ref_joint,"
                "  process_noise_vel_gain_root, process_noise_vel_ref_root,"
                "  process_noise_vel_joint_names,"
                "  process_noise_vel_scopes,"
                "  pose_reg_joint_names, pose_reg_equal_split_noise_std, pose_reg_rest_pose_noise_std,"
                "  soft_limit_joint_names, soft_limit_margin_rad, soft_limit_noise_std,"
                "  nis_feedback_scopes, nis_feedback_threshold, nis_feedback_max_multiplier)"
                " VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,"
                "         ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    config_id, "ui-run", now,
                    self._proc_noise_std.value(),
                    self._proc_vel_noise.value(),
                    self._vel_half_life.value(),
                    self._calib_noise.value(),   # stored in legacy column for compat
                    self._pose_noise.value(),
                    self._outlier_thresh.value(),
                    self._tracker_fps.value(),
                    vel_ids_json,
                    use_rel,
                    rel_min_conf,
                    cross_px_val,
                    cross_n,
                    cross_person_mm_val,
                    cross_person_min_conf,
                    cross_person_n,
                    joint_gain,
                    self._vel_noise_ref_joint.value(),
                    self._vel_noise_gain_root.value(),
                    self._vel_noise_ref_root.value(),
                    joint_names_json,
                    vel_scopes_json,
                    pose_reg_joint_names_json,
                    pose_reg_equal_split,
                    pose_reg_rest_pose,
                    soft_limit_joint_names_json,
                    soft_limit_margin,
                    soft_limit_noise_std,
                    nis_feedback_scopes_json,
                    self._nis_feedback_threshold.value(),
                    self._nis_feedback_max_mult.value(),
                ),
            )
        return config_id

    def _on_output(self, line: str) -> None:
        m = re.match(r"\s*Progress:\s*(\d+)/(\d+)\s*\(([0-9.]+)%\)", line)
        if m:
            self._progress_bar.setValue(int(float(m.group(3))))
            self._status_label.setText(line)
        else:
            self._log.appendPlainText(line)

    def _on_finished(self, exit_code: int, run_id: str) -> None:
        self._run_id = run_id or None
        self._thread = None
        self._run_btn.setEnabled(True)

        if exit_code != 0:
            self._progress_bar.setValue(0)
            self._status_label.setText(f"Tracker exited with code {exit_code}.")
            detail = _describe_windows_exit_code(exit_code)
            if detail:
                QMessageBox.critical(self, "Tracker failed to start", detail)
            return

        self._progress_bar.setValue(100)
        self._status_label.setText("Tracking complete.")
        self._show_results()
        if self._run_id:
            self.run_finished.emit(self._run_id)

    def _on_multi_finished(self, exit_code: int, run_ids: list) -> None:
        self._thread = None
        self._run_btn.setEnabled(True)

        if exit_code != 0:
            self._progress_bar.setValue(0)
            self._status_label.setText(f"Tracker exited with code {exit_code}.")
            detail = _describe_windows_exit_code(exit_code)
            if detail:
                QMessageBox.critical(self, "Tracker failed to start", detail)
            return

        self._progress_bar.setValue(100)
        self._status_label.setText("Tracking complete.")
        self._show_multi_results(run_ids)
        # Emit run_finished for the primary person's run only -- e.g. so a
        # PersonPanel embedding this widget can select the new run for the
        # person it's showing. Other people's runs are separate tracking_runs
        # rows visible in the main tree.
        if run_ids and run_ids[0]:
            self.run_finished.emit(run_ids[0])

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _show_multi_results(self, run_ids: list) -> None:
        if self._conn is None or not self._multi_run_labels:
            return
        lines = []
        for label, run_id in zip(self._multi_run_labels, run_ids):
            if not run_id:
                lines.append(f"{label}: (no run_id — tracker exited early)")
                continue
            # person_id = 0 always -- every person lives in their own sequence.
            row = self._conn.execute(
                "SELECT COUNT(*) AS total,"
                "       SUM(CASE WHEN tracking_lost = 0 THEN 1 ELSE 0 END) AS tracked,"
                "       AVG(COALESCE(n_inlier_observations, 0)) AS avg_inliers"
                " FROM tracking_results"
                " WHERE run_id = ? AND person_id = 0 AND is_smoothed = 0",
                (run_id,),
            ).fetchone()
            if row and row["total"]:
                total = row["total"]
                tracked = row["tracked"] or 0
                pct = 100.0 * tracked / total
                avg = row["avg_inliers"] or 0.0
                lines.append(
                    f"{label} — run {run_id[:12]}…: "
                    f"{tracked}/{total} steps ({pct:.1f}%), avg inliers {avg:.1f}"
                )
            else:
                lines.append(f"{label} — run {run_id[:12]}…: (no per-frame stats)")
        self._results_label.setText("\n".join(lines))
        self._results_box.setVisible(True)
        # BVH export below only knows about a single run_id -- default it to
        # the primary person; export for the other people's runs from
        # wherever those tracking_runs rows surface in the main tree.
        self._run_id = run_ids[0] if run_ids else None

    def _show_results(self) -> None:
        if self._conn is None or self._run_id is None:
            return
        # person_id = 0 always -- every person lives in their own sequence.
        row = self._conn.execute(
            "SELECT COUNT(*) AS total,"
            "       SUM(CASE WHEN tracking_lost = 0 THEN 1 ELSE 0 END) AS tracked,"
            "       AVG(COALESCE(n_inlier_observations, 0)) AS avg_inliers"
            " FROM tracking_results"
            " WHERE run_id = ? AND person_id = 0 AND is_smoothed = 0",
            (self._run_id,),
        ).fetchone()
        if row and row["total"]:
            total = row["total"]
            tracked = row["tracked"] or 0
            pct = 100.0 * tracked / total
            avg = row["avg_inliers"] or 0.0
            text = (
                f"Run: {self._run_id[:16]}…\n"
                f"Tracked: {tracked}/{total} steps ({pct:.1f}%)\n"
                f"Average inliers per step: {avg:.1f}"
            )
        else:
            text = f"Run: {self._run_id[:16]}…\n(No per-frame stats available.)"
        self._results_label.setText(text)
        self._results_box.setVisible(True)

    # ------------------------------------------------------------------
    # BVH export
    # ------------------------------------------------------------------

    def _export_bvh(self) -> None:
        if self._run_id is None or self._session_path is None:
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save BVH file", "", "BVH files (*.bvh)")
        if not out_path:
            return

        export_script = _TOOLS_DIR / "export_bvh.py"
        self._status_label.setText("Exporting BVH…")
        self._export_bvh_btn.setEnabled(False)

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._bvh_process = proc

        def _done(code: int, _status) -> None:
            self._bvh_process = None
            self._export_bvh_btn.setEnabled(True)
            if code == 0:
                self._status_label.setText(f"BVH exported: {Path(out_path).name}")
                QMessageBox.information(self, "Export complete",
                                        f"BVH file written to:\n{out_path}")
            else:
                output = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
                self._status_label.setText("BVH export failed.")
                QMessageBox.critical(
                    self, "Export failed",
                    f"export_bvh.py exited with code {code}.\n\n{output[-800:]}",
                )

        proc.finished.connect(_done)
        proc.start(
            sys.executable,
            [
                str(export_script),
                "--session-db", self._session_path,
                "--run-id",     self._run_id,
                "--person-id",  "0",  # every person lives in their own sequence
                "--smoothed",
                "--output",     out_path,
            ],
        )


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class RunTrackerDialog(QDialog):
    """Standalone dialog for running the tracker (accessible from pose window)."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        session_path: str,
        sequence_id: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run Tracker")
        self.setMinimumWidth(640)
        self.setMinimumHeight(420)
        self.resize(700, 650)

        self._widget = RunTrackerWidget()
        self._widget.set_session(conn, session_path)
        if sequence_id is not None:
            self._widget.preselect_sequence(sequence_id)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._widget, 1)
        layout.addWidget(buttons)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _float_spin(default: float, mn: float, mx: float, decimals: int) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(mn, mx)
    spin.setDecimals(decimals)
    spin.setValue(default)
    return spin


# Windows NTSTATUS codes with the Error severity bits set (top two bits) are
# always > 2**31 -- a normal program's own exit code never reaches that range,
# so seeing one here means Windows killed the process before main() ran
# (missing DLL, bad image, etc.), not that the tracker itself failed.
_STATUS_DLL_NOT_FOUND = 0xC0000135


def _describe_windows_exit_code(exit_code: int) -> str | None:
    """Return a human-readable explanation for a Windows process-launch
    failure exit code, or None if *exit_code* looks like an ordinary exit
    code the tracker itself returned (nothing further to explain here).
    """
    if exit_code < 0x8000_0000:
        return None
    if exit_code == _STATUS_DLL_NOT_FOUND:
        return (
            "The tracker binary could not load a required DLL "
            "(boost_serialization.dll and/or yaml-cpp.dll).\n\n"
            "See CONTRIBUTING.md's \"Windows (native, MSVC)\" section — "
            "these need to be copied next to posetrak-tracker.exe, not just "
            "added to PATH (re-running setup-windows.ps1 does this)."
        )
    return (
        f"Windows terminated the tracker process before it could run "
        f"(NTSTATUS 0x{exit_code:08X}), rather than the tracker exiting "
        f"with an error of its own. This usually means a missing or "
        f"mismatched runtime dependency -- see CONTRIBUTING.md's "
        f"\"Windows (native, MSVC)\" section."
    )

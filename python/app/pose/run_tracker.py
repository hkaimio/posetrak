"""run_tracker.py — Widget and dialog for running the posetrak tracker binary."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import yaml
from PySide6.QtCore import QPoint, QProcess, QRect, QThread, Qt, Signal
from PySide6.QtGui import QDoubleValidator, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStyle,
    QStyleOptionTab,
    QStylePainter,
    QTabBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from posetrak.db.manage_config import (
    BASELINE_CONFIG_ID,
    edit_config,
    list_configs,
    resolve_default_tracker_config,
    set_default_tracker_config,
)
from posetrak.db.manage_person import list_persons
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

# Hierarchical solver (docs/roadmap/features/hierarchical-solver/
# hierarchical-solver-design.md) per-stage tuning overrides: tracker_config_stages
# columns that build_stage_tracker_config() (src/tracking/hierarchical_solver.cpp)
# actually applies. min_inliers_ratio/max_innovation_norm exist as DB columns but
# have no TrackerConfig field to receive them yet -- deliberately not exposed here
# so the UI never implies they do something.
_STAGE_OVERRIDE_COLUMNS: list[tuple[str, str]] = [
    ("process_noise_std", "Process σ"),
    ("process_noise_vel_std", "Proc-vel σ"),
    ("velocity_half_life_s", "Vel half-life"),
    ("pose_noise_std", "Pose σ"),
    ("calib_noise_std", "Calib σ"),
    ("outlier_threshold", "Outlier thr"),
    ("init_joint_std", "Init-joint σ"),
    ("init_velocity_std", "Init-vel σ"),
]


def discover_stage_groups(conn: sqlite3.Connection, skeleton_ids: list[str]) -> list[str]:
    """Group names with a freeflyer_joint declared, across the given skeletons'
    own groups: YAML sections -- the hierarchical-solver child-stage candidates.
    Union across skeletons, first-seen order; skeletons sharing the usual
    HandL/HandR naming convention just union to the same two names.
    """
    seen: dict[str, None] = {}
    for skel_id in skeleton_ids:
        row = conn.execute(
            "SELECT yaml_content FROM skeletons WHERE id=?", (skel_id,)
        ).fetchone()
        if row is None:
            continue
        try:
            skel = yaml.safe_load(row[0]) or {}
        except yaml.YAMLError:
            continue
        for group in skel.get("groups") or []:
            if group.get("freeflyer_joint"):
                seen.setdefault(group["name"], None)
    return list(seen.keys())

# Soft joint-limit repulsion Phase 1 scope -- prototyping only, see
# docs/roadmap/features/soft-joint-limits/soft-joint-limits-design.md. Scoped to the
# joints diagnosed as overshooting their own ball-joint limits during a fast bilateral
# motion; extend only if the same saturation pattern is confirmed on other joints.
SOFT_LIMIT_JOINT_NAMES: list[str] = ["upper_arm.L", "upper_arm.R"]


# ---------------------------------------------------------------------------
# Numeric field widget
# ---------------------------------------------------------------------------


class NumericLineEdit(QLineEdit):
    """Validated plain-text numeric field, replacing QDoubleSpinBox/QSpinBox
    for most tracker-config values.

    See docs/roadmap/features/configuration-improvements/config-improvements-design.md,
    B2: spin-box up/down arrows have no natural step size for a std-dev or a
    noise scale the user types a specific tuned number into, so a plain
    validated field is clearer for most of this dialog's fields. Small,
    genuinely-bounded integer *counts* (e.g. max cross-pair count) are a
    reasonable exception and stay QSpinBox elsewhere in this file.

    Exposes the same value()/setValue()/valueChanged surface QDoubleSpinBox
    has, so it drops into existing wiring code (``.valueChanged.connect``,
    ``.setEnabled``, ``.setToolTip``) unchanged.
    """

    valueChanged = Signal(float)

    def __init__(
        self, default: float, minimum: float, maximum: float, decimals: int = 4, parent=None
    ) -> None:
        super().__init__(parent)
        validator = QDoubleValidator(minimum, maximum, decimals, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.setValidator(validator)
        self._decimals = decimals
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.textChanged.connect(lambda _text: self.valueChanged.emit(self.value()))
        self.setValue(default)

    def value(self) -> float:
        text = self.text().strip()
        if not text or text in ("-", "."):
            return 0.0
        try:
            return float(text)
        except ValueError:
            return 0.0

    def setValue(self, value: float) -> None:  # noqa: N802 (matches QDoubleSpinBox's API)
        self.setText(f"{value:.{self._decimals}f}")


def _numeric(default: float, mn: float, mx: float, decimals: int) -> NumericLineEdit:
    return NumericLineEdit(default, mn, mx, decimals)


# ---------------------------------------------------------------------------
# Horizontal-text tab bar (for a West-positioned QTabWidget)
# ---------------------------------------------------------------------------


class _HorizontalTabBar(QTabBar):
    """QTabBar for a West-positioned QTabWidget whose tab labels still read
    left-to-right, instead of Qt's default of rotating tab text 90 degrees
    to fit a vertical strip.

    Plain ``setTabPosition(QTabWidget.West)`` gives a vertical strip of
    tabs, which is what was asked for (all tab names visible down the left
    side, like Blender's or Visual Studio's settings dialogs) -- but Qt
    also rotates each tab's *text* to run bottom-to-top by default, which
    means only a few characters of a longer tab name fit before the tab's
    (fixed, content-driven) height runs out, and most of a wide dialog's
    height goes unused. Blender/VS keep the tab *shape* vertical but the
    *label* horizontal. Achieving that isn't exposed as a QTabWidget/QTabBar
    property -- it needs overriding tabSizeHint() (report a shape as if this
    were a horizontal, North-positioned bar, so there's enough width for
    upright text) and paintEvent() (draw each tab's shape normally, but
    rotate the *painter* -90° only while drawing that tab's label, so the
    label paints upright inside the rotated-to-vertical tab). This is a
    well-known Qt/PySide recipe, not project-specific cleverness.
    """

    def tabSizeHint(self, index: int):
        size = super().tabSizeHint(index)
        size.transpose()
        return size

    def paintEvent(self, event) -> None:
        painter = QStylePainter(self)
        option = QStyleOptionTab()
        for index in range(self.count()):
            self.initStyleOption(option, index)
            painter.drawControl(QStyle.ControlElement.CE_TabBarTabShape, option)
            painter.save()

            size = option.rect.size()
            size.transpose()
            rect = QRect(QPoint(), size)
            rect.moveCenter(option.rect.center())
            option.rect = rect

            # Qt's own CE_TabBarTabLabel drawing already rotates the painter
            # by -90 for a West-shaped tab (see qcommonstyle.cpp) before
            # drawing the label -- rotating by -90 again here (the intuitive
            # choice, since West tabs read top-to-bottom) actually compounds
            # to -180, i.e. upside-down, right-to-left text. +90 cancels the
            # style's own -90 back to the identity, leaving the label
            # genuinely horizontal.
            center = self.tabRect(index).center()
            painter.translate(center)
            painter.rotate(90)
            painter.translate(-center)
            painter.drawControl(QStyle.ControlElement.CE_TabBarTabLabel, option)
            painter.restore()


# ---------------------------------------------------------------------------
# Tracker-config editor (tabs + load/save)
# ---------------------------------------------------------------------------


class TrackerConfigWidget(QWidget):
    """Vertical-tab tracker-config editor with a load/save bar.

    Extracted from RunTrackerWidget (config-improvements design doc, phase 3)
    so it can be reused standalone -- editing a session/capture/trial's
    *default* tracker config (see DefaultConfigDialog below) has nothing to
    do with people/detection runs/starting a tracking run, and embedding the
    whole of RunTrackerWidget just to reach its config tabs would drag all of
    that unrelated machinery along for the ride.

    The Hierarchical solver stages tab needs to know which skeletons are in
    play to discover eligible child-stage groups -- callers supply that via
    set_skeleton_ids() (RunTrackerWidget pushes its people table's current
    skeleton selection; DefaultConfigDialog isn't tied to any one person's
    skeleton choice, so it pushes every skeleton in the session instead).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._conn: sqlite3.Connection | None = None
        self._skeleton_ids: list[str] = []
        # Which named tracker_configs row (if any) this widget was last
        # loaded from / saved as -- see _open_load_config_dialog()/
        # _open_save_as_dialog()/load_config_row(). None means "factory
        # defaults" (manage_config.BASELINE_CONFIG_ID).
        self.loaded_config_id: str | None = None
        self.loaded_config_name: str | None = None
        # Snapshot of collect_config_overrides()/collect_stage_snapshot() at
        # the moment loaded_config_id was last set -- lets the status label
        # tell "loaded, untouched" from "loaded, then edited" (see
        # _is_dirty()/_update_config_status_label()) instead of always
        # saying "Based on X" regardless of whether X's values still match.
        self._loaded_snapshot: dict | None = None
        self._loaded_stage_snapshot: tuple = ()

        # ---- Load/save bar ------------------------------------------------
        self._config_status_label = QLabel()
        self._config_status_label.setToolTip(
            "Which saved configuration this dialog's current values are based on.\n"
            "Starting a run always saves a fresh, unnamed snapshot of the current\n"
            "values -- use \"Save as…\" to keep a named, reusable copy."
        )
        load_config_btn = QPushButton("Load…")
        load_config_btn.setToolTip("Load a previously saved, named tracker configuration.")
        load_config_btn.clicked.connect(self._open_load_config_dialog)
        save_as_config_btn = QPushButton("Save as…")
        save_as_config_btn.setToolTip(
            "Save the current tab values as a new named, reusable configuration."
        )
        save_as_config_btn.clicked.connect(self._open_save_as_dialog)

        config_header = QHBoxLayout()
        config_header.addWidget(QLabel("Configuration:"))
        config_header.addWidget(self._config_status_label, 1)
        config_header.addWidget(load_config_btn)
        config_header.addWidget(save_as_config_btn)

        # ---- Tabs -----------------------------------------------------
        self._proc_noise_std  = _numeric(0.1,   0.0, 1000.0, 4)
        self._proc_vel_noise  = _numeric(0.5,   0.0, 1000.0, 4)
        self._vel_half_life   = _numeric(0.25,  0.0,   10.0, 4)

        self._vel_noise_gain_joint = _numeric(0.0, 0.0, 100.0, 3)
        self._vel_noise_gain_joint.setToolTip(
            "Adaptive process noise (Phase 1): scales each joint DOF's own process noise\n"
            "by (1 + gain * |its velocity| / reference velocity). 0 = disabled (static noise,\n"
            "matches pre-Phase-1 behaviour)."
        )
        self._vel_noise_ref_joint = _numeric(1.0, 1.0e-3, 1000.0, 3)
        self._vel_noise_ref_joint.setToolTip("Reference velocity for the joint gain above (rad/s).")
        self._vel_noise_gain_root = _numeric(0.0, 0.0, 100.0, 3)
        self._vel_noise_gain_root.setToolTip(
            "Same as the joint gain above, but for the root's position/orientation DOFs\n"
            "(separate knob: root moves in metres/rad, joints in radians)."
        )
        self._vel_noise_ref_root = _numeric(1.0, 1.0e-3, 1000.0, 3)
        self._vel_noise_ref_root.setToolTip(
            "Reference velocity for the root gain above (m/s for position, rad/s for orientation)."
        )
        self._vel_noise_gain_proximal = _numeric(0.0, 0.0, 100.0, 3)
        self._vel_noise_gain_proximal.setToolTip(
            "Independent adaptive process noise gain for ADAPTIVE_NOISE_PROXIMAL_JOINTS\n"
            "(shoulder/upper_arm/forearm, hip/knee) -- excluded from the torso gain above\n"
            "to avoid over-loosening fast normal gestures, so give them their own,\n"
            "separately-tuned gain here instead of none at all. 0 = disabled."
        )
        self._vel_noise_ref_proximal = _numeric(1.0, 1.0e-3, 1000.0, 3)
        self._vel_noise_ref_proximal.setToolTip(
            "Reference velocity for the proximal-limb gain above (rad/s)."
        )
        self._vel_noise_gain_distal = _numeric(0.0, 0.0, 100.0, 3)
        self._vel_noise_gain_distal.setToolTip(
            "Independent adaptive process noise gain for ADAPTIVE_NOISE_DISTAL_JOINTS\n"
            "(wrist/hand/fingers, ankle/foot/toe) -- kept separate from the proximal scope\n"
            "above since distal joints are typically more accurate but move faster, so\n"
            "warrant their own (likely higher) reference velocity. 0 = disabled."
        )
        self._vel_noise_ref_distal = _numeric(1.0, 1.0e-3, 1000.0, 3)
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

        self._pose_reg_equal_split = _numeric(0.0, 0.0, 10.0, 4)
        self._pose_reg_equal_split.setToolTip(
            "Pose regularization: pseudo-measurement pulling POSE_REG_SPINE_CHAIN's joint\n"
            "angles toward each other, per axis (stiffness = this std, radians; smaller =\n"
            "stronger pull). 0 = disabled."
        )
        self._pose_reg_rest_pose = _numeric(0.0, 0.0, 10.0, 4)
        self._pose_reg_rest_pose.setToolTip(
            "Pose regularization: pseudo-measurement pulling POSE_REG_SPINE_CHAIN's joint\n"
            "angles toward zero, per axis (stiffness = this std, radians). 0 = disabled."
        )

        self._soft_limit_margin = _numeric(0.0, 0.0, 1.5, 4)
        self._soft_limit_margin.setToolTip(
            "Soft joint-limit repulsion: width (radians) of the soft zone just inside\n"
            "each SOFT_LIMIT_JOINT_NAMES axis's hard limit. Only matters if the noise std\n"
            "below is nonzero."
        )
        self._soft_limit_noise_std = _numeric(0.0, 0.0, 10.0, 4)
        self._soft_limit_noise_std.setToolTip(
            "Soft joint-limit repulsion: pseudo-measurement pulling SOFT_LIMIT_JOINT_NAMES's\n"
            "joint angles away from their own hard limits once inside the margin above\n"
            "(stiffness = this std, radians; smaller = stronger pull). 0 = disabled."
        )

        self._nis_feedback_threshold = _numeric(1.5, 0.1, 100.0, 2)
        self._nis_feedback_threshold.setToolTip(
            "NIS-feedback safety net (Mechanism B): windowed NIS/DOF for the 'core' and\n"
            "'limbs' scopes above this triggers a temporary process-noise multiplier.\n"
            "Only takes effect if 'Enable NIS feedback' is checked."
        )
        self._nis_feedback_max_mult = _numeric(10.0, 1.0, 1000.0, 1)
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

        self._pose_noise      = _numeric(0.0,   0.0, 1.0e6,  2)
        self._calib_noise     = _numeric(60.0,  0.0, 1.0e6,  2)
        self._outlier_thresh  = _numeric(4.0,   0.1,   50.0, 2)
        self._tracker_fps     = _numeric(120.0, 1.0,  500.0, 1)

        self._vel_cam_label = QLabel("None")
        vel_cam_edit_btn = QPushButton("Edit…")
        vel_cam_edit_btn.setFixedWidth(60)
        vel_cam_edit_btn.clicked.connect(self._edit_velocity_cameras)
        vel_cam_row = QHBoxLayout()
        vel_cam_row.addWidget(self._vel_cam_label, 1)
        vel_cam_row.addWidget(vel_cam_edit_btn)
        self._sequence_cameras: list[str] = []
        self._velocity_cam_indices: set[int] = set()

        self._use_relative = QCheckBox()
        self._use_relative.setChecked(False)
        self._use_relative.setToolTip(
            "Emit child-minus-parent pixel observations alongside absolute positions.\n"
            "Calibration error cancels in the difference; requires pose_noise_std > 0."
        )
        self._relative_min_conf = _numeric(0.5, 0.0, 1.0, 2)
        self._relative_min_conf.setToolTip(
            "Minimum keypoint confidence for both child and parent to form a relative pair."
        )
        self._use_relative.toggled.connect(self._relative_min_conf.setEnabled)
        self._relative_min_conf.setEnabled(False)

        self._cross_pair_max_px = _numeric(0.0, 0.0, 9999.0, 1)
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

        self._cross_person_max_world_mm = _numeric(0.0, 0.0, 99999.0, 1)
        self._cross_person_max_world_mm.setToolTip(
            "3D world-space marker-pair distance gate (mm) for cross-person\n"
            "PAIR_DIFF anchoring between people tracked together below\n"
            "(e.g. ukemi throws, handshakes). 0 = disabled (Phase 5)."
        )
        self._cross_person_min_conf = _numeric(0.5, 0.0, 1.0, 2)
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

        # ---- Hierarchical solver (child stages) --------------------------
        self._hierarchical_enabled = QCheckBox()
        self._hierarchical_enabled.setToolTip(
            "Run named skeleton groups (e.g. HandL/HandR) as separate fixed-root\n"
            "child stages after the main pass, merged into the same tracking_results\n"
            "rows -- see docs/roadmap/features/hierarchical-solver/\n"
            "hierarchical-solver-design.md. Only groups with a freeflyer_joint\n"
            "declared in the skeleton's groups: section are eligible; use Refresh\n"
            "after changing a person's skeleton."
        )
        self._hierarchical_enabled.toggled.connect(self._on_hierarchical_toggled)

        hier_refresh_btn = QPushButton("↻ Refresh stages")
        hier_refresh_btn.setToolTip(
            "Re-scan currently selected skeletons for eligible child-stage groups."
        )
        hier_refresh_btn.clicked.connect(self._refresh_stage_table)

        hier_row_widget = QWidget()
        hier_row = QHBoxLayout(hier_row_widget)
        hier_row.setContentsMargins(0, 0, 0, 0)
        hier_row.addWidget(self._hierarchical_enabled)
        hier_row.addWidget(hier_refresh_btn)
        hier_row.addStretch(1)

        self._stage_table = QTableWidget(0, 2 + len(_STAGE_OVERRIDE_COLUMNS))
        self._stage_table.setHorizontalHeaderLabels(
            ["On", "Group"] + [label for _, label in _STAGE_OVERRIDE_COLUMNS]
        )
        self._stage_table.verticalHeader().setVisible(False)
        self._stage_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._stage_table.setToolTip(
            "One row per eligible skeleton group. 'On' includes it as a child stage\n"
            "for this run. Numeric columns override this stage's own tuning\n"
            "(blank = inherit the main config's value for that field)."
        )
        self._stage_table.setMinimumHeight(90)

        # ---- Assemble tabs (vertical, left side) -------------------------
        self._summary_text = QPlainTextEdit()
        self._summary_text.setReadOnly(True)
        summary_mono = QFont("Monospace")
        summary_mono.setStyleHint(QFont.StyleHint.Monospace)
        self._summary_text.setFont(summary_mono)

        ukf_form = QFormLayout()
        ukf_form.addRow("Process noise std:", self._proc_noise_std)
        ukf_form.addRow("Velocity noise std:", self._proc_vel_noise)
        ukf_form.addRow("Velocity half-life (s):", self._vel_half_life)
        ukf_form.addRow("Tracker FPS:", self._tracker_fps)
        ukf_form.addRow("Velocity cameras:", vel_cam_row)

        obs_form = QFormLayout()
        obs_form.addRow("Pose noise std (px in model):", self._pose_noise)
        obs_form.addRow("Calib noise std (px in video):", self._calib_noise)
        obs_form.addRow("Outlier threshold:", self._outlier_thresh)
        obs_form.addRow("Relative observations:", self._use_relative)
        obs_form.addRow("Relative min confidence:", self._relative_min_conf)
        obs_form.addRow("Cross-pair radius (px):", self._cross_pair_max_px)
        obs_form.addRow("Cross-pair max count:", self._cross_pair_max_n)

        adaptive_form = QFormLayout()
        adaptive_form.addRow("Gain (joint/core):", self._vel_noise_gain_joint)
        adaptive_form.addRow("Reference vel (joint/core, rad/s):", self._vel_noise_ref_joint)
        adaptive_form.addRow("Gain (root):", self._vel_noise_gain_root)
        adaptive_form.addRow("Reference vel (root, m/s, rad/s):", self._vel_noise_ref_root)
        adaptive_form.addRow("Gain (proximal):", self._vel_noise_gain_proximal)
        adaptive_form.addRow("Reference vel (proximal, rad/s):", self._vel_noise_ref_proximal)
        adaptive_form.addRow("Gain (distal):", self._vel_noise_gain_distal)
        adaptive_form.addRow("Reference vel (distal, rad/s):", self._vel_noise_ref_distal)

        posereg_form = QFormLayout()
        posereg_form.addRow("Equal-split std (spine1/2, rad):", self._pose_reg_equal_split)
        posereg_form.addRow("Rest-pose std (spine1/2, rad):", self._pose_reg_rest_pose)
        posereg_form.addRow("Soft-limit margin (upper_arm, rad):", self._soft_limit_margin)
        posereg_form.addRow("Soft-limit noise std (upper_arm, rad):", self._soft_limit_noise_std)

        nis_form = QFormLayout()
        nis_form.addRow("Enable NIS feedback (core+limbs):", self._nis_feedback_enabled)
        nis_form.addRow("NIS feedback threshold:", self._nis_feedback_threshold)
        nis_form.addRow("NIS feedback max multiplier:", self._nis_feedback_max_mult)

        crossperson_form = QFormLayout()
        crossperson_form.addRow("Cross-person distance (mm):", self._cross_person_max_world_mm)
        crossperson_form.addRow("Cross-person min confidence:", self._cross_person_min_conf)
        crossperson_form.addRow("Cross-person max count:", self._cross_person_max_n)

        hierarchical_layout = QVBoxLayout()
        hierarchical_layout.addWidget(hier_row_widget)
        hierarchical_layout.addWidget(self._stage_table)

        self._config_tabs = QTabWidget()
        self._config_tabs.setTabBar(_HorizontalTabBar())
        self._config_tabs.setTabPosition(QTabWidget.TabPosition.West)
        # Without this, a too-narrow window silently collapses the tab strip
        # to scroll arrows (names hidden) instead of the tab bar honestly
        # reporting the width it needs -- defeating the point of West tabs
        # (all names visible at once, Blender/VS-style).
        self._config_tabs.setUsesScrollButtons(False)
        self._config_tabs.addTab(self._summary_text, "Summary")
        self._config_tabs.addTab(_tab_page(ukf_form), "UKF && process model")
        self._config_tabs.addTab(_tab_page(obs_form), "Observations && outliers")
        self._config_tabs.addTab(_tab_page(adaptive_form), "Adaptive process noise")
        self._config_tabs.addTab(_tab_page(posereg_form), "Pose reg. && joint limits")
        self._config_tabs.addTab(_tab_page(nis_form), "NIS feedback")
        self._config_tabs.addTab(_tab_page(crossperson_form), "Cross-person coupling")
        self._config_tabs.addTab(_tab_page(hierarchical_layout), "Hierarchical solver")
        self._config_tabs.currentChanged.connect(self._on_config_tab_changed)

        root = QVBoxLayout(self)
        root.addLayout(config_header)
        root.addWidget(self._config_tabs, 1)

        # Generic dirty-tracking: any tuning field changing should flip the
        # status label from "<name>" to "<name> (modified)" (see
        # _update_config_status_label()/_is_dirty()) -- connecting via
        # findChildren() rather than one-by-one avoids this list silently
        # going stale as fields are added above. Stage-table cell widgets
        # are created later by _refresh_stage_table() and wire themselves.
        for w in self.findChildren(NumericLineEdit):
            w.valueChanged.connect(self._update_config_status_label)
        for w in self.findChildren(QCheckBox):
            w.toggled.connect(self._update_config_status_label)
        for w in self.findChildren(QSpinBox):
            w.valueChanged.connect(self._update_config_status_label)

        self._update_config_status_label()
        self._refresh_summary()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_connection(self, conn: sqlite3.Connection | None) -> None:
        self._conn = conn

    def set_skeleton_ids(self, skeleton_ids: list[str]) -> None:
        """Skeletons currently in play, for Hierarchical solver stage
        discovery -- pushed by the caller (RunTrackerWidget's people table),
        not queried by this widget, since it has no notion of "people" of
        its own (DefaultConfigDialog never calls this)."""
        self._skeleton_ids = skeleton_ids

    def load_config_row(self, config_id: str, name: str | None, row: sqlite3.Row) -> None:
        """Apply an already-fetched tracker_configs row and record it as
        loaded -- for a caller (DefaultConfigDialog) that resolves which row
        to start from itself, rather than going through the "Load…" picker.
        """
        self._apply_config_row(row)
        self.loaded_config_id = config_id
        self.loaded_config_name = name
        self._load_stage_overrides(config_id)
        self._capture_loaded_snapshot()
        self._update_config_status_label()
        self._refresh_summary()

    def collect_overrides(self) -> dict:
        return self._collect_config_overrides()

    def sync_stage_overrides(self, config_id: str) -> None:
        self._sync_stage_overrides(config_id)

    def validate_stage_overrides(self) -> str | None:
        return self._validate_stage_overrides()

    def edit_velocity_cameras_context(self, sequence_cameras: list[str]) -> None:
        """Supply the current trial's camera labels, for the velocity-mode
        camera picker -- pushed by RunTrackerWidget whenever the selected
        trial changes, mirroring set_skeleton_ids() above."""
        self._sequence_cameras = sequence_cameras
        self._velocity_cam_indices = set()
        self._update_velocity_cam_label()

    @property
    def velocity_cam_indices(self) -> set[int]:
        return self._velocity_cam_indices

    # ------------------------------------------------------------------
    # Velocity-mode cameras
    # ------------------------------------------------------------------

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
    # Configuration load/save (config-improvements design doc, phase 2)
    # ------------------------------------------------------------------

    def _on_config_tab_changed(self, index: int) -> None:
        if index == 0:  # Summary
            self._refresh_summary()

    def _update_config_status_label(self) -> None:
        if self.loaded_config_id is None:
            self._config_status_label.setText("Based on factory defaults (unsaved)")
            return
        label = self.loaded_config_name or "(unnamed snapshot)"
        if self._is_dirty():
            label += " (modified)"
        self._config_status_label.setText(label)

    def _capture_loaded_snapshot(self) -> None:
        """Record the tab values as of the last load/save, for _is_dirty()."""
        self._loaded_snapshot = self._collect_config_overrides()
        self._loaded_stage_snapshot = self._collect_stage_snapshot()

    def _is_dirty(self) -> bool:
        """Whether the tabs' current values differ from loaded_config_id's,
        as of when it was loaded/saved -- see _capture_loaded_snapshot()."""
        if self._loaded_snapshot is None:
            return False
        return (
            self._collect_config_overrides() != self._loaded_snapshot
            or self._collect_stage_snapshot() != self._loaded_stage_snapshot
        )

    def _collect_stage_snapshot(self) -> tuple:
        """Serialize the stage table's current enabled rows + override text,
        for _is_dirty() -- deliberately independent of dict/JSON ordering
        concerns _collect_config_overrides() doesn't have to worry about."""
        if not self._hierarchical_enabled.isChecked():
            return ()
        rows = []
        for i in range(self._stage_table.rowCount()):
            if not self._stage_row_enabled(i):
                continue
            group = self._stage_table.item(i, 1).text()
            values = []
            for j, (_field, _label) in enumerate(_STAGE_OVERRIDE_COLUMNS):
                edit = self._stage_table.cellWidget(i, 2 + j)
                values.append(edit.text().strip() if edit else "")
            rows.append((group, tuple(values)))
        return tuple(sorted(rows))

    def _open_load_config_dialog(self) -> None:
        if self._conn is None:
            return
        rows = [r for r in list_configs(self._conn) if r["is_named"]]
        if not rows:
            QMessageBox.information(
                self, "No saved configs", "No named tracker configurations are saved yet."
            )
            return
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        labels = [f"{r['name']}  ({r['created_at'][:19]})" for r in rows]
        choice, ok = QInputDialog.getItem(
            self, "Load configuration", "Configuration:", labels, 0, False
        )
        if not ok:
            return
        row = rows[labels.index(choice)]
        self.load_config_row(row["id"], row["name"], row)

    def _open_save_as_dialog(self) -> None:
        if self._conn is None:
            return
        name, ok = QInputDialog.getText(self, "Save configuration", "Name:")
        name = name.strip()
        if not ok or not name:
            return
        base = self.loaded_config_id or BASELINE_CONFIG_ID
        new_id = edit_config(self._conn, base, is_named=True, name=name,
                             **self._collect_config_overrides())
        self._sync_stage_overrides(new_id)
        self.loaded_config_id = new_id
        self.loaded_config_name = name
        self._capture_loaded_snapshot()
        self._update_config_status_label()
        QMessageBox.information(self, "Saved", f'Configuration saved as "{name}".')

    def _apply_config_row(self, row: sqlite3.Row) -> None:
        """Populate every tab widget from a loaded tracker_configs row."""
        def g(col: str, default: object = None) -> object:
            try:
                value = row[col]
            except (IndexError, KeyError):
                return default
            return default if value is None else value

        self._proc_noise_std.setValue(float(g("process_noise_std", 0.0)))
        self._proc_vel_noise.setValue(float(g("process_noise_vel_std", 0.0)))
        self._vel_half_life.setValue(float(g("velocity_half_life_s", 0.0)))
        self._tracker_fps.setValue(float(g("tracker_fps", 120.0)))
        self._pose_noise.setValue(float(g("pose_noise_std", 0.0)))
        self._calib_noise.setValue(float(g("measurement_noise_std", 0.0)))
        self._outlier_thresh.setValue(float(g("outlier_threshold", 4.0)))

        self._use_relative.setChecked(bool(g("use_relative_observations", 0)))
        self._relative_min_conf.setValue(float(g("relative_min_confidence", 0.5)))

        self._cross_pair_max_px.setValue(float(g("cross_pair_max_px", 0.0)))
        self._cross_pair_max_n.setValue(int(g("cross_pair_max_n", 10)))

        self._cross_person_max_world_mm.setValue(float(g("cross_person_max_world_mm", 0.0)))
        self._cross_person_min_conf.setValue(float(g("cross_person_min_confidence", 0.5)))
        self._cross_person_max_n.setValue(int(g("cross_person_max_n", 10)))

        self._vel_noise_gain_joint.setValue(float(g("process_noise_vel_gain_joint", 0.0)))
        self._vel_noise_ref_joint.setValue(float(g("process_noise_vel_ref_joint", 1.0)))
        self._vel_noise_gain_root.setValue(float(g("process_noise_vel_gain_root", 0.0)))
        self._vel_noise_ref_root.setValue(float(g("process_noise_vel_ref_root", 1.0)))

        scopes_json = g("process_noise_vel_scopes")
        scopes = json.loads(scopes_json) if scopes_json else []
        proximal = next((s for s in scopes if s.get("name") == "proximal"), None)
        distal = next((s for s in scopes if s.get("name") == "distal"), None)
        self._vel_noise_gain_proximal.setValue(float(proximal["gain"]) if proximal else 0.0)
        self._vel_noise_ref_proximal.setValue(float(proximal["vel_ref"]) if proximal else 1.0)
        self._vel_noise_gain_distal.setValue(float(distal["gain"]) if distal else 0.0)
        self._vel_noise_ref_distal.setValue(float(distal["vel_ref"]) if distal else 1.0)

        self._pose_reg_equal_split.setValue(float(g("pose_reg_equal_split_noise_std", 0.0)))
        self._pose_reg_rest_pose.setValue(float(g("pose_reg_rest_pose_noise_std", 0.0)))

        self._soft_limit_margin.setValue(float(g("soft_limit_margin_rad", 0.0)))
        self._soft_limit_noise_std.setValue(float(g("soft_limit_noise_std", 0.0)))

        self._nis_feedback_enabled.setChecked(bool(g("nis_feedback_scopes")))
        self._nis_feedback_threshold.setValue(float(g("nis_feedback_threshold", 1.5)))
        self._nis_feedback_max_mult.setValue(float(g("nis_feedback_max_multiplier", 10.0)))

        vel_cam_json = g("velocity_mode_camera_ids")
        self._velocity_cam_indices = set(json.loads(vel_cam_json)) if vel_cam_json else set()
        self._update_velocity_cam_label()

    def _collect_config_overrides(self) -> dict:
        """Build the tracker_configs override dict from current tab values,
        for edit_config() -- shared by RunTrackerWidget's per-run snapshot
        and _open_save_as_dialog()'s named save. List/dict values pass
        through as plain Python objects; edit_config()'s own _encode() JSON-
        encodes them.
        """
        vel_ids = sorted(self._velocity_cam_indices) if self._velocity_cam_indices else None
        use_rel = 1 if self._use_relative.isChecked() else 0
        rel_min_conf = self._relative_min_conf.value() if use_rel else None

        cross_px = self._cross_pair_max_px.value()
        cross_px_val = cross_px if cross_px > 0.0 else None
        cross_n = self._cross_pair_max_n.value() if cross_px_val else None

        cross_person_mm = self._cross_person_max_world_mm.value()
        cross_person_mm_val = cross_person_mm if cross_person_mm > 0.0 else None
        cross_person_min_conf = self._cross_person_min_conf.value() if cross_person_mm_val else None
        cross_person_n = self._cross_person_max_n.value() if cross_person_mm_val else None

        joint_gain = self._vel_noise_gain_joint.value()
        joint_names = ADAPTIVE_NOISE_CORE_JOINTS if joint_gain > 0.0 else None

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

        pose_reg_equal_split = self._pose_reg_equal_split.value()
        pose_reg_rest_pose = self._pose_reg_rest_pose.value()
        pose_reg_enabled = pose_reg_equal_split > 0.0 or pose_reg_rest_pose > 0.0

        soft_limit_noise_std = self._soft_limit_noise_std.value()
        soft_limit_enabled = soft_limit_noise_std > 0.0

        nis_scopes = None
        if self._nis_feedback_enabled.isChecked():
            nis_scopes = [
                {"name": "core", "joint_names": ADAPTIVE_NOISE_CORE_JOINTS},
                {"name": "limbs", "joint_names": NIS_FEEDBACK_LIMB_JOINTS},
            ]

        return dict(
            process_noise_std=self._proc_noise_std.value(),
            process_noise_vel_std=self._proc_vel_noise.value(),
            velocity_half_life_s=self._vel_half_life.value(),
            measurement_noise_std=self._calib_noise.value(),
            pose_noise_std=self._pose_noise.value(),
            outlier_threshold=self._outlier_thresh.value(),
            tracker_fps=self._tracker_fps.value(),
            velocity_mode_camera_ids=vel_ids,
            use_relative_observations=use_rel,
            relative_min_confidence=rel_min_conf,
            cross_pair_max_px=cross_px_val,
            cross_pair_max_n=cross_n,
            cross_person_max_world_mm=cross_person_mm_val,
            cross_person_min_confidence=cross_person_min_conf,
            cross_person_max_n=cross_person_n,
            process_noise_vel_gain_joint=joint_gain,
            process_noise_vel_ref_joint=self._vel_noise_ref_joint.value(),
            process_noise_vel_gain_root=self._vel_noise_gain_root.value(),
            process_noise_vel_ref_root=self._vel_noise_ref_root.value(),
            process_noise_vel_joint_names=joint_names,
            process_noise_vel_scopes=vel_scopes or None,
            pose_reg_joint_names=POSE_REG_SPINE_CHAIN if pose_reg_enabled else None,
            pose_reg_equal_split_noise_std=pose_reg_equal_split,
            pose_reg_rest_pose_noise_std=pose_reg_rest_pose,
            soft_limit_joint_names=SOFT_LIMIT_JOINT_NAMES if soft_limit_enabled else None,
            soft_limit_margin_rad=self._soft_limit_margin.value(),
            soft_limit_noise_std=soft_limit_noise_std,
            nis_feedback_scopes=nis_scopes,
            nis_feedback_threshold=self._nis_feedback_threshold.value(),
            nis_feedback_max_multiplier=self._nis_feedback_max_mult.value(),
        )

    def _refresh_summary(self) -> None:
        lines = [self._config_status_label.text(), ""]
        lines.append(f"Process noise std: {self._proc_noise_std.value():g}")
        lines.append(f"Velocity noise std: {self._proc_vel_noise.value():g}")
        lines.append(f"Velocity half-life (s): {self._vel_half_life.value():g}")
        lines.append(f"Tracker FPS: {self._tracker_fps.value():g}")
        lines.append(f"Velocity-mode cameras: {self._vel_cam_label.text()}")
        lines.append("")
        lines.append(f"Pose noise std: {self._pose_noise.value():g}")
        lines.append(f"Calib noise std: {self._calib_noise.value():g}")
        lines.append(f"Outlier threshold: {self._outlier_thresh.value():g}")
        lines.append(
            f"Relative observations: {'on' if self._use_relative.isChecked() else 'off'}"
        )
        if self._use_relative.isChecked():
            lines.append(f"  min confidence: {self._relative_min_conf.value():g}")
        if self._cross_pair_max_px.value() > 0.0:
            lines.append(
                f"Cross-pair radius: {self._cross_pair_max_px.value():g}px, "
                f"max {self._cross_pair_max_n.value()}"
            )
        lines.append("")
        if self._vel_noise_gain_joint.value() > 0.0:
            lines.append(f"Adaptive noise (core): gain {self._vel_noise_gain_joint.value():g}")
        if self._vel_noise_gain_root.value() > 0.0:
            lines.append(f"Adaptive noise (root): gain {self._vel_noise_gain_root.value():g}")
        if self._vel_noise_gain_proximal.value() > 0.0:
            lines.append(
                f"Adaptive noise (proximal): gain {self._vel_noise_gain_proximal.value():g}"
            )
        if self._vel_noise_gain_distal.value() > 0.0:
            lines.append(
                f"Adaptive noise (distal): gain {self._vel_noise_gain_distal.value():g}"
            )
        if self._pose_reg_equal_split.value() > 0.0 or self._pose_reg_rest_pose.value() > 0.0:
            lines.append(
                f"Pose regularization: equal-split {self._pose_reg_equal_split.value():g}, "
                f"rest-pose {self._pose_reg_rest_pose.value():g}"
            )
        if self._soft_limit_noise_std.value() > 0.0:
            lines.append(
                f"Soft joint limits: margin {self._soft_limit_margin.value():g}, "
                f"noise {self._soft_limit_noise_std.value():g}"
            )
        if self._nis_feedback_enabled.isChecked():
            lines.append(
                f"NIS feedback: threshold {self._nis_feedback_threshold.value():g}, "
                f"max multiplier {self._nis_feedback_max_mult.value():g}"
            )
        if self._cross_person_max_world_mm.value() > 0.0:
            lines.append(
                f"Cross-person coupling: {self._cross_person_max_world_mm.value():g}mm, "
                f"min conf {self._cross_person_min_conf.value():g}, "
                f"max {self._cross_person_max_n.value()}"
            )
        if self._hierarchical_enabled.isChecked():
            enabled_groups = [
                self._stage_table.item(i, 1).text()
                for i in range(self._stage_table.rowCount())
                if self._stage_row_enabled(i)
            ]
            lines.append(
                f"Hierarchical solver stages: {', '.join(enabled_groups) or '(none enabled)'}"
            )
        self._summary_text.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # Hierarchical solver stages
    # ------------------------------------------------------------------

    def _on_hierarchical_toggled(self, checked: bool) -> None:
        if checked:
            self._refresh_stage_table()

    def _refresh_stage_table(self) -> None:
        if self._conn is None:
            return
        groups = discover_stage_groups(self._conn, self._skeleton_ids)
        self._stage_table.setRowCount(len(groups))
        for i, group in enumerate(groups):
            chk_container = QWidget()
            chk_layout = QHBoxLayout(chk_container)
            chk_layout.setContentsMargins(0, 0, 0, 0)
            chk_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chk = QCheckBox()
            chk.setChecked(True)
            chk.toggled.connect(self._update_config_status_label)
            chk_layout.addWidget(chk)
            self._stage_table.setCellWidget(i, 0, chk_container)

            name_item = QTableWidgetItem(group)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._stage_table.setItem(i, 1, name_item)

            for j, (_field, _label) in enumerate(_STAGE_OVERRIDE_COLUMNS):
                edit = QLineEdit()
                edit.setPlaceholderText("inherit")
                edit.textChanged.connect(lambda _text: self._update_config_status_label())
                self._stage_table.setCellWidget(i, 2 + j, edit)

    def _load_stage_overrides(self, config_id: str) -> None:
        """Populate the Hierarchical solver tab from *config_id*'s existing
        tracker_config_stages rows (if any), and set the enable checkbox and
        each discovered group's own checkbox to match.

        Without this, loading an already-hierarchical config (e.g. a
        trial's default that has stages configured) always showed an empty,
        disabled table -- indistinguishable from "no stages configured" --
        and saving from that state (_sync_stage_overrides() always rebuilds
        from the table, discarding whatever was in the DB) silently deleted
        the real stage selection.
        """
        if self._conn is None:
            return
        saved = {
            r["group_name"]: r
            for r in self._conn.execute(
                "SELECT * FROM tracker_config_stages WHERE tracker_config_id = ?",
                (config_id,),
            ).fetchall()
        }
        self._hierarchical_enabled.blockSignals(True)
        self._hierarchical_enabled.setChecked(bool(saved))
        self._hierarchical_enabled.blockSignals(False)
        self._refresh_stage_table()
        for i in range(self._stage_table.rowCount()):
            group = self._stage_table.item(i, 1).text()
            saved_row = saved.get(group)
            container = self._stage_table.cellWidget(i, 0)
            chk = container.findChild(QCheckBox) if container else None
            if chk is not None:
                chk.setChecked(saved_row is not None)
            if saved_row is None:
                continue
            for j, (field, _label) in enumerate(_STAGE_OVERRIDE_COLUMNS):
                edit = self._stage_table.cellWidget(i, 2 + j)
                value = saved_row[field]
                if edit is not None and value is not None:
                    edit.setText(f"{value:g}")

    def _stage_row_enabled(self, row: int) -> bool:
        container = self._stage_table.cellWidget(row, 0)
        chk = container.findChild(QCheckBox) if container else None
        return chk is not None and chk.isChecked()

    def _validate_stage_overrides(self) -> str | None:
        """Return an error message if any enabled stage's override field isn't a
        valid number (or blank, meaning inherit), else None."""
        if not self._hierarchical_enabled.isChecked():
            return None
        for i in range(self._stage_table.rowCount()):
            if not self._stage_row_enabled(i):
                continue
            group = self._stage_table.item(i, 1).text()
            for j, (_field, label) in enumerate(_STAGE_OVERRIDE_COLUMNS):
                edit = self._stage_table.cellWidget(i, 2 + j)
                text = edit.text().strip() if edit else ""
                if not text:
                    continue
                try:
                    float(text)
                except ValueError:
                    return f"Stage '{group}': '{label}' is not a number: {text!r}"
        return None

    def _sync_stage_overrides(self, config_id: str) -> None:
        """Replace *config_id*'s tracker_config_stages rows with the stage
        table's current values. Always deletes first: edit_config() already
        copied the base config's own stage rows forward (if any), and this
        run's stage selection may add, drop, or re-tune any of them.
        """
        with self._conn:
            self._conn.execute(
                "DELETE FROM tracker_config_stages WHERE tracker_config_id = ?", (config_id,)
            )
            if not self._hierarchical_enabled.isChecked():
                return
            for i in range(self._stage_table.rowCount()):
                if not self._stage_row_enabled(i):
                    continue
                group_name = self._stage_table.item(i, 1).text()
                overrides = []
                for j, (_field, _label) in enumerate(_STAGE_OVERRIDE_COLUMNS):
                    edit = self._stage_table.cellWidget(i, 2 + j)
                    text = edit.text().strip() if edit else ""
                    overrides.append(float(text) if text else None)
                self._conn.execute(
                    "INSERT INTO tracker_config_stages"
                    " (tracker_config_id, group_name, process_noise_std,"
                    "  process_noise_vel_std, velocity_half_life_s, pose_noise_std,"
                    "  calib_noise_std, outlier_threshold, init_joint_std,"
                    "  init_velocity_std)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (config_id, group_name, *overrides),
                )


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
        # (skeleton_id, display name), refreshed once per set_session() and
        # shared by every person row's skeleton combo.
        self._all_skeletons: list[tuple[str, str]] = []
        # Display label per person row, captured at run start (for
        # _show_multi_results() -- the table itself may change before the
        # run finishes if the user starts editing it again).
        self._multi_run_labels: list[str] | None = None

        self._config_widget = TrackerConfigWidget()

        # ---- People group -------------------------------------------------
        # Person names are only defined per detection run (sequence_persons is
        # a 1:1 name label per pose_observation_sequences row, itself one per
        # person per detection run -- see
        # docs/roadmap/features/error-improvements/phase5-cross-person-plan.md).
        # So: pick a Trial first (the natural "track together" scope --
        # cross-person anchoring needs shared cameras/world space, which every
        # detection run in one trial shares by construction), then one row per
        # person named within it; each row's Detection run choices are
        # whichever detection runs in that trial produced a sequence for that
        # name, and each row keeps its own Skeleton (people have different
        # body proportions).
        self._trial_combo = QComboBox()
        self._trial_combo.currentIndexChanged.connect(self._on_trial_changed)
        trial_form = QFormLayout()
        trial_form.addRow("Trial:", self._trial_combo)

        self._people_table = QTableWidget(0, 4)
        self._people_table.setHorizontalHeaderLabels(
            ["Person", "Detection run", "Skeleton", ""]
        )
        header = self._people_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._people_table.verticalHeader().setVisible(False)
        self._people_table.setMinimumHeight(90)
        self._people_table.setToolTip(
            "If this trial's capture has defined persons (CapturePanel's\n"
            "Persons section), one row per person is listed with a checkbox\n"
            "for whether to include them in this run. Otherwise, row 1 is\n"
            "the primary person and can't be removed; \"Add person…\" tracks\n"
            "another named person from the same trial alongside them.\n"
            "Either way, tracking multiple people together interleaves them\n"
            "frame-by-frame (enables cross-person anchoring if Cross-person\n"
            "distance above is set > 0), and each keeps their own detection\n"
            "run and skeleton."
        )

        self._add_person_btn = QPushButton("Add person…")
        self._add_person_btn.clicked.connect(self._add_person_row)

        people_layout = QVBoxLayout()
        people_layout.addLayout(trial_form)
        people_layout.addWidget(self._people_table)
        add_person_row = QHBoxLayout()
        add_person_row.addStretch()
        add_person_row.addWidget(self._add_person_btn)
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
        self._set_trial_default_chk = QCheckBox("Set as trial default")
        self._set_trial_default_chk.setToolTip(
            "After this run starts, make its tracker configuration this\n"
            "trial's default -- future runs in this trial will start from\n"
            "it (see the trial's \"Default tracker config\" row), instead of\n"
            "needing to be set up by hand each time."
        )

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
        # Only the tracker-configuration widget stretches to fill extra
        # space -- people, run controls, the Run button, and progress/
        # results always stay visible without needing to resize the window.
        # People first (who's being tracked), then the (usually much taller)
        # configuration tabs, then the run-mechanics controls (output dir,
        # binary path) immediately above the button that starts the run.
        root = QVBoxLayout(self)
        root.addWidget(people_box)
        root.addWidget(self._config_widget, 1)
        root.addWidget(run_box)
        root.addWidget(self._set_trial_default_chk)
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
        self._config_widget.set_connection(conn)
        self._refresh_skeletons()
        self._refresh_trials()
        self._update_run_btn()

    def preselect_sequence(self, seq_id: str) -> None:
        """Pre-select and lock the primary person's Trial/Person/Detection run
        to whichever of those *seq_id* resolves to.

        Call after set_session(). Those three combos are disabled so the user
        cannot change them when this widget is embedded in a PersonPanel;
        Skeleton stays editable.
        """
        row = self._conn.execute(
            "SELECT dr.trial_id, dr.id AS detection_run_id, sp.person_name"
            " FROM pose_observation_sequences pos"
            " JOIN detection_runs dr ON dr.id = pos.detection_run_id"
            " LEFT JOIN sequence_persons sp"
            "        ON sp.sequence_id = pos.id AND sp.person_id = 0"
            " WHERE pos.id = ?",
            (seq_id,),
        ).fetchone()
        if row is None or row["trial_id"] is None or row["person_name"] is None:
            return

        for i in range(self._trial_combo.count()):
            if self._trial_combo.itemData(i)["trial_id"] == row["trial_id"]:
                self._trial_combo.setCurrentIndex(i)
                break
        self._trial_combo.setEnabled(False)

        # Column 0 holds a QComboBox (legacy free-text-person mode, one row)
        # or a QCheckBox per row (capture_persons mode, one row per defined
        # person) -- see _row_included()/_row_person_name(). Find whichever
        # row matches this sequence's person.
        target_row: int | None = None
        for r in range(self._people_table.rowCount()):
            widget = self._people_table.cellWidget(r, 0)
            if isinstance(widget, QCheckBox):
                if widget.text() == row["person_name"]:
                    target_row = r
            elif widget is not None:
                for i in range(widget.count()):
                    if widget.itemData(i) == row["person_name"]:
                        widget.setCurrentIndex(i)
                        target_row = r
                        break
            if target_row is not None:
                break

        if target_row is not None:
            # This view locks onto a single sequence's person -- drop every
            # other row so only the matched one remains, then lock it.
            while self._people_table.rowCount() > 1:
                if target_row == 0:
                    self._people_table.removeRow(1)
                else:
                    self._people_table.removeRow(0)
                    target_row -= 1

            person_widget = self._people_table.cellWidget(target_row, 0)
            if isinstance(person_widget, QCheckBox):
                person_widget.setChecked(True)
            person_widget.setEnabled(False)

            dr_combo = self._people_table.cellWidget(target_row, 1)
            if dr_combo is not None:
                for i in range(dr_combo.count()):
                    if dr_combo.itemData(i)[3] == row["detection_run_id"]:
                        dr_combo.setCurrentIndex(i)
                        break
                dr_combo.setEnabled(False)

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

    def _refresh_trials(self) -> None:
        self._trial_combo.blockSignals(True)
        self._trial_combo.clear()
        if self._conn is not None:
            rows = self._conn.execute(
                # DISTINCT: a trial can have several detection runs (re-runs);
                # each just needs to have produced at least one sequence.
                "SELECT DISTINCT t.id AS trial_id, t.name AS trial_name,"
                "       t.time_start_s, t.time_end_s,"
                "       cap.label AS capture_label, cap.capture_number,"
                "       cap.extrinsic_calibration_id"
                " FROM trials t"
                " JOIN captures cap ON cap.id = t.capture_id"
                " JOIN detection_runs dr ON dr.trial_id = t.id"
                " JOIN pose_observation_sequences pos ON pos.detection_run_id = dr.id"
                " ORDER BY cap.capture_number, t.time_start_s"
            ).fetchall()
            for r in rows:
                self._trial_combo.addItem(self._trial_label(r), r)
        self._trial_combo.blockSignals(False)
        self._on_trial_changed()

    @staticmethod
    def _trial_label(r: sqlite3.Row) -> str:
        capture = r["capture_label"] or f"capture{r['capture_number']:03d}"
        trial_name = r["trial_name"] or "(unnamed trial)"
        duration = r["time_end_s"] - r["time_start_s"]
        label = (
            f"{capture} — {trial_name}"
            f"  [{r['time_start_s']:.1f}–{r['time_end_s']:.1f}s, {duration:.1f}s]"
        )
        if not r["extrinsic_calibration_id"]:
            label += "  ⚠ no extrinsics"
        return label

    def _current_trial_id(self) -> str | None:
        data = self._trial_combo.currentData()
        return data["trial_id"] if data is not None else None

    def _person_names_for_trial(self, trial_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT sp.person_name"
            " FROM sequence_persons sp"
            " JOIN pose_observation_sequences pos ON pos.id = sp.sequence_id"
            " JOIN detection_runs dr ON dr.id = pos.detection_run_id"
            " WHERE dr.trial_id = ? AND sp.person_id = 0"
            " ORDER BY sp.person_name",
            (trial_id,),
        ).fetchall()
        return [r["person_name"] for r in rows]

    def _detection_runs_for_person(self, trial_id: str, person_name: str) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT dr.id AS detection_run_id, dr.created_at, dr.detector_model,"
            "       dr.pose_model, dr.status,"
            "       pos.id AS seq_id, pos.time_start_s, pos.time_end_s"
            " FROM detection_runs dr"
            " JOIN pose_observation_sequences pos ON pos.detection_run_id = dr.id"
            " JOIN sequence_persons sp ON sp.sequence_id = pos.id"
            " WHERE dr.trial_id = ? AND sp.person_name = ? AND sp.person_id = 0"
            " ORDER BY dr.created_at DESC",
            (trial_id, person_name),
        ).fetchall())

    def _capture_id_for_trial(self, trial_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT capture_id FROM trials WHERE id = ?", (trial_id,)
        ).fetchone()
        return row["capture_id"] if row is not None else None

    def _detection_runs_for_capture_person(
        self, trial_id: str, capture_person_id: str, name: str
    ) -> list[sqlite3.Row]:
        """Detection runs in *trial_id* with observations for the capture
        person *capture_person_id* -- matches by that link where set,
        falling back to an exact person_name match for sequences written
        before capture_persons existed (capture_person_id IS NULL)."""
        return list(self._conn.execute(
            "SELECT dr.id AS detection_run_id, dr.created_at, dr.detector_model,"
            "       dr.pose_model, dr.status,"
            "       pos.id AS seq_id, pos.time_start_s, pos.time_end_s"
            " FROM detection_runs dr"
            " JOIN pose_observation_sequences pos ON pos.detection_run_id = dr.id"
            " JOIN sequence_persons sp ON sp.sequence_id = pos.id"
            " WHERE dr.trial_id = ? AND sp.person_id = 0"
            "   AND (sp.capture_person_id = ?"
            "        OR (sp.capture_person_id IS NULL AND sp.person_name = ?))"
            " ORDER BY dr.created_at DESC",
            (trial_id, capture_person_id, name),
        ).fetchall())

    @staticmethod
    def _detection_run_label(r: sqlite3.Row) -> str:
        label = f"{r['created_at'][:19]}  ({r['detector_model']}/{r['pose_model']})"
        if r["status"] != "complete":
            label += f"  ⚠ {r['status']}"
        return label

    def _cameras_for_trial(self, trial_id: str) -> list[str]:
        row = self._conn.execute(
            "SELECT sync_config_id FROM detection_runs WHERE trial_id = ? LIMIT 1",
            (trial_id,),
        ).fetchone()
        if not row or not row["sync_config_id"]:
            return []
        rows = self._conn.execute(
            "SELECT ci.label"
            " FROM capture_videos sv"
            " JOIN captures sh ON sh.id = sv.shot_id"
            " JOIN sync_configs scfg ON scfg.shot_id = sh.id"
            " JOIN camera_instances ci ON ci.id = sv.camera_instance_id"
            " WHERE scfg.id = ?"
            " ORDER BY ci.label ASC",
            (row["sync_config_id"],),
        ).fetchall()
        return [r["label"] for r in rows]

    def _make_skeleton_combo(self) -> QComboBox:
        combo = QComboBox()
        for skel_id, name in self._all_skeletons:
            combo.addItem(name, skel_id)
        return combo

    def _row_detection_run_data(self, row: int) -> tuple[str, float, float, str] | None:
        combo = self._people_table.cellWidget(row, 1)
        return combo.currentData() if combo is not None else None

    def _row_skeleton_id(self, row: int) -> str | None:
        combo = self._people_table.cellWidget(row, 2)
        return combo.currentData() if combo is not None else None

    def _row_included(self, row: int) -> bool:
        """Whether *row* should be tracked. Column 0 holds a QComboBox
        (person picker) in the legacy free-text-person mode -- always
        included, no per-row opt-out -- or a QCheckBox (person name as its
        label) in the capture_persons mode, where inclusion is the whole
        point of the checkbox. See _insert_capture_person_rows()."""
        widget = self._people_table.cellWidget(row, 0)
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        return True

    def _row_person_name(self, row: int) -> str:
        widget = self._people_table.cellWidget(row, 0)
        if widget is None:
            return ""
        return widget.text() if isinstance(widget, QCheckBox) else widget.currentText()

    def _insert_person_row(
        self, trial_id: str, used_names: set[str], *, removable: bool
    ) -> int | None:
        names = [n for n in self._person_names_for_trial(trial_id) if n not in used_names]
        if not names:
            return None

        row = self._people_table.rowCount()
        self._people_table.insertRow(row)

        person_combo = QComboBox()
        for n in names:
            person_combo.addItem(n, n)
        self._people_table.setCellWidget(row, 0, person_combo)

        dr_combo = QComboBox()
        self._people_table.setCellWidget(row, 1, dr_combo)
        person_combo.currentIndexChanged.connect(
            lambda _index=None, r=row: self._refresh_detection_run_combo(r)
        )

        skeleton_combo = self._make_skeleton_combo()
        skeleton_combo.currentIndexChanged.connect(self._update_run_btn)
        self._people_table.setCellWidget(row, 2, skeleton_combo)

        if removable:
            remove_btn = QPushButton("✕")
            remove_btn.setFixedWidth(28)
            remove_btn.setToolTip("Remove this person")
            remove_btn.clicked.connect(self._make_remove_handler(remove_btn))
            self._people_table.setCellWidget(row, 3, remove_btn)

        self._refresh_detection_run_combo(row)
        self._update_run_btn()
        return row

    def _refresh_detection_run_combo(self, row: int) -> None:
        person_combo = self._people_table.cellWidget(row, 0)
        dr_combo = self._people_table.cellWidget(row, 1)
        if person_combo is None or dr_combo is None:
            return
        trial_id = self._current_trial_id()
        person_name = person_combo.currentData()
        dr_combo.clear()
        if trial_id is not None and person_name is not None:
            for r in self._detection_runs_for_person(trial_id, person_name):
                dr_combo.addItem(
                    self._detection_run_label(r),
                    (r["seq_id"], r["time_start_s"], r["time_end_s"], r["detection_run_id"]),
                )
        self._update_run_btn()

    def _make_remove_handler(self, button: QPushButton):
        def _remove() -> None:
            for r in range(self._people_table.rowCount()):
                if self._people_table.cellWidget(r, 3) is button:
                    self._people_table.removeRow(r)
                    break
            self._update_run_btn()

        return _remove

    def _add_person_row(self) -> None:
        trial_id = self._current_trial_id()
        if trial_id is None:
            QMessageBox.information(
                self, "Select a trial first", "Choose a Trial before adding people."
            )
            return
        used_names = {
            self._people_table.cellWidget(r, 0).currentData()
            for r in range(self._people_table.rowCount())
            if self._people_table.cellWidget(r, 0) is not None
        }
        row = self._insert_person_row(trial_id, used_names, removable=True)
        if row is None:
            QMessageBox.information(
                self, "No other people", "No other person names found for this trial."
            )

    def _insert_capture_person_rows(self, trial_id: str, persons: list[sqlite3.Row]) -> None:
        """Populate the people table from *persons* (the trial's capture's
        defined capture_persons), one row per person with observations in
        this trial -- data-source switch described in the config-improvements
        design doc, "Person model", D3. Unlike the legacy free-text mode,
        the roster here is fixed (defined via CapturePanel's Persons
        section, not ad hoc per run): each row is a checkbox for whether to
        include that person in this run, a detection-run picker (only
        enabled when more than one exists for this person in this trial),
        and a skeleton combo pre-filled from the person's own default.
        """
        for person in persons:
            detection_runs = self._detection_runs_for_capture_person(
                trial_id, person["id"], person["name"]
            )
            if not detection_runs:
                continue  # nothing to track for this person in this trial

            row = self._people_table.rowCount()
            self._people_table.insertRow(row)

            include_chk = QCheckBox(person["name"])
            include_chk.setChecked(True)
            include_chk.toggled.connect(self._update_run_btn)
            self._people_table.setCellWidget(row, 0, include_chk)

            dr_combo = QComboBox()
            for r in detection_runs:
                dr_combo.addItem(
                    self._detection_run_label(r),
                    (r["seq_id"], r["time_start_s"], r["time_end_s"], r["detection_run_id"]),
                )
            dr_combo.setEnabled(len(detection_runs) > 1)
            dr_combo.currentIndexChanged.connect(self._update_run_btn)
            self._people_table.setCellWidget(row, 1, dr_combo)

            skeleton_combo = self._make_skeleton_combo()
            if person["default_skeleton_id"]:
                idx = skeleton_combo.findData(person["default_skeleton_id"])
                if idx >= 0:
                    skeleton_combo.setCurrentIndex(idx)
            skeleton_combo.currentIndexChanged.connect(self._update_run_btn)
            self._people_table.setCellWidget(row, 2, skeleton_combo)

    def _current_skeleton_ids(self) -> list[str]:
        return [
            sid for row in range(self._people_table.rowCount())
            if self._row_included(row) and (sid := self._row_skeleton_id(row)) is not None
        ]

    def _update_run_btn(self) -> None:
        ok = len(self._all_skeletons) > 0 and any(
            self._row_included(row) and self._row_detection_run_data(row) is not None
            for row in range(self._people_table.rowCount())
        )
        self._run_btn.setEnabled(ok)
        self._config_widget.set_skeleton_ids(self._current_skeleton_ids())

    def _on_trial_changed(self) -> None:
        trial_id = self._current_trial_id()
        self._people_table.setRowCount(0)
        if trial_id is not None:
            capture_id = self._capture_id_for_trial(trial_id)
            persons = list_persons(self._conn, capture_id) if capture_id else []
            if persons:
                self._insert_capture_person_rows(trial_id, persons)
                self._add_person_btn.setVisible(False)
            else:
                # No capture_persons defined for this capture yet (not
                # adopted, or a pre-existing capture) -- fall back to the
                # original free-text-name discovery so existing captures
                # keep working unchanged until someone defines persons for
                # them via CapturePanel's Persons section.
                self._insert_person_row(trial_id, used_names=set(), removable=False)
                self._add_person_btn.setVisible(True)
            self._sequence_cameras = self._cameras_for_trial(trial_id)
        else:
            self._sequence_cameras = []
        self._config_widget.edit_velocity_cameras_context(self._sequence_cameras)
        self._update_run_btn()
        self._load_trial_default_config(trial_id)

    def _load_trial_default_config(self, trial_id: str | None) -> None:
        """Load *trial_id*'s resolved default tracker config into the config
        widget, so a run started without touching Load…/a saved config
        starts from the trial's own tuned default rather than silently
        falling back to factory defaults -- see
        manage_config.resolve_default_tracker_config(). Called after
        _update_run_btn() has already pushed this trial's skeleton
        selection, so the Hierarchical solver tab can discover the right
        stage groups for the loaded default's stage rows (if any).
        """
        if self._conn is None or trial_id is None:
            return
        resolved_id = resolve_default_tracker_config(self._conn, trial_id=trial_id)
        row = self._conn.execute(
            "SELECT * FROM tracker_configs WHERE id = ?", (resolved_id,)
        ).fetchone()
        if row is None:
            return
        name = row["name"] if row["is_named"] else None
        self._config_widget.load_config_row(resolved_id, name, row)

    def _maybe_set_trial_default(self, config_id: str) -> None:
        """If "Set as trial default" is checked, repoint the current
        trial's default_tracker_config_id to *config_id* -- the snapshot
        just created for this run."""
        if not self._set_trial_default_chk.isChecked():
            return
        trial_id = self._current_trial_id()
        if trial_id is not None:
            set_default_tracker_config(self._conn, config_id, trial_id=trial_id)

    def _browse_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self._out_dir_edit.setText(path)

    def _browse_binary(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select posetrak binary", "", "All files (*)")
        if path:
            self._binary_edit.setText(path)

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
            if not self._row_included(row):
                continue
            data = self._row_detection_run_data(row)
            skel_id = self._row_skeleton_id(row)
            if data is None or skel_id is None:
                QMessageBox.critical(
                    self, "Cannot run tracker",
                    "Every included person needs both a detection run and a skeleton selected.",
                )
                return
            seq_id, t0, t1, _dr_id = data
            person_name = self._row_person_name(row)
            dr_label = self._people_table.cellWidget(row, 1).currentText()
            people.append((seq_id, skel_id, t0, t1, f"{person_name} — {dr_label}"))

        if not people:
            QMessageBox.critical(
                self, "Cannot run tracker", "Include at least one person to track."
            )
            return

        primary_seq_id, primary_skel_id, time_start_s, time_end_s, _ = people[0]

        err = self._check_sequence_ready(primary_seq_id)
        if err:
            QMessageBox.critical(self, "Cannot run tracker", err)
            return

        stage_err = self._config_widget.validate_stage_overrides()
        if stage_err:
            QMessageBox.critical(self, "Cannot run tracker", stage_err)
            return

        out_dir = self._resolve_out_dir(primary_seq_id, primary_skel_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        base = self._config_widget.loaded_config_id or BASELINE_CONFIG_ID
        config_id = edit_config(
            self._conn, base, is_named=False, **self._config_widget.collect_overrides()
        )
        self._config_widget.sync_stage_overrides(config_id)
        self._maybe_set_trial_default(config_id)
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
        # Wide enough for the widest Hierarchical-solver-tab-included config
        # tab strip (West-positioned, horizontal labels -- see
        # _HorizontalTabBar) to show every tab name at once instead of
        # falling back to scroll arrows; tall enough for its 8 tab rows plus
        # the People/Run groups above and the Run button below without
        # needing an immediate manual resize.
        self.setMinimumWidth(900)
        self.setMinimumHeight(700)
        self.resize(980, 820)

        self._widget = RunTrackerWidget()
        self._widget.set_session(conn, session_path)
        if sequence_id is not None:
            self._widget.preselect_sequence(sequence_id)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._widget, 1)
        layout.addWidget(buttons)


class DefaultConfigDialog(QDialog):
    """Edit the default tracker config for a capture or trial.

    Config-improvements design doc, phase 3, section C: loads the resolved
    effective default (this scope's own, else its capture's, else the
    checked-in baseline -- see manage_config.resolve_default_tracker_config())
    into a standalone TrackerConfigWidget, and on "Set as default" always
    produces a *new* tracker_configs row via edit_config() (copy-on-write,
    matching the design doc's "editing a default is always copy-on-write,
    never in-place mutation") and repoints *only* this scope's own
    default_tracker_config_id -- never the other level's. The embedded
    widget's own "Load…"/"Save as…" buttons still work as usual, for loading
    a totally different named template as the new default, or additionally
    saving the tweaked default under a name for reuse elsewhere; neither of
    those repoints this scope's default by itself -- only "Set as default"
    does that.

    Exactly one of *trial_id*/*capture_id* must be given.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        trial_id: str | None = None,
        capture_id: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if (trial_id is None) == (capture_id is None):
            raise ValueError("DefaultConfigDialog: supply exactly one of trial_id/capture_id")
        self._conn = conn
        self._trial_id = trial_id
        self._capture_id = capture_id

        self.setWindowTitle("Edit Default Tracker Configuration")
        self.setMinimumWidth(560)
        self.setMinimumHeight(420)
        self.resize(640, 600)

        from posetrak.db.manage_config import resolve_default_tracker_config

        resolved_id = resolve_default_tracker_config(
            conn, trial_id=trial_id, capture_id=capture_id
        )
        row = conn.execute(
            "SELECT * FROM tracker_configs WHERE id = ?", (resolved_id,)
        ).fetchone()

        self._config_widget = TrackerConfigWidget()
        self._config_widget.set_connection(conn)
        # A default isn't tied to any one person's skeleton choice (that's
        # picked per-run), so offer every skeleton's eligible stage groups
        # here rather than none -- without this, the Hierarchical solver tab
        # had nothing to discover and stages could never be set from this
        # dialog at all.
        all_skeleton_ids = [r["id"] for r in conn.execute("SELECT id FROM skeletons").fetchall()]
        self._config_widget.set_skeleton_ids(all_skeleton_ids)
        self._config_widget.load_config_row(
            resolved_id, row["name"] if row["is_named"] else None, row
        )

        set_default_btn = QPushButton("Set as default")
        set_default_btn.setToolTip(
            "Save the current tab values as a new configuration and make it this\n"
            "scope's default -- never mutates the configuration in place."
        )
        set_default_btn.clicked.connect(self._set_as_default)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(self.reject)
        buttons.addButton(set_default_btn, QDialogButtonBox.ButtonRole.AcceptRole)

        layout = QVBoxLayout(self)
        layout.addWidget(self._config_widget, 1)
        layout.addWidget(buttons)

    def _set_as_default(self) -> None:
        from posetrak.db.manage_config import set_default_tracker_config

        base = self._config_widget.loaded_config_id or BASELINE_CONFIG_ID
        new_id = edit_config(self._conn, base, is_named=False, **self._config_widget.collect_overrides())
        self._config_widget.sync_stage_overrides(new_id)
        set_default_tracker_config(
            self._conn, new_id, trial_id=self._trial_id, capture_id=self._capture_id
        )
        self.accept()


def build_default_config_row(
    conn: sqlite3.Connection,
    *,
    trial_id: str | None = None,
    capture_id: str | None = None,
    parent: QWidget | None = None,
) -> QWidget:
    """Build a "Default tracker config: ‹name› [Edit] [Change…]" row for a
    TrialPanel or CapturePanel (config-improvements design doc, phase 3,
    section C). Exactly one of *trial_id*/*capture_id* should be given.

    "Edit" opens DefaultConfigDialog (tweak values, "Set as default" saves a
    new copy-on-write row and repoints this scope). "Change…" repoints to an
    existing *named* configuration directly, with no new row created --
    distinct from "Edit", which always creates one even if nothing was
    changed (matching the design doc's "editing a default is always
    copy-on-write" rule -- "Change…" is not an edit, so it isn't bound by
    that rule).
    """
    from posetrak.db.manage_config import resolve_default_tracker_config, set_default_tracker_config

    row_widget = QWidget(parent)
    row = QHBoxLayout(row_widget)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(QLabel("Default tracker config:"))
    status_label = QLabel()
    row.addWidget(status_label, 1)
    edit_btn = QPushButton("Edit…")
    change_btn = QPushButton("Change…")
    row.addWidget(edit_btn)
    row.addWidget(change_btn)

    def _refresh_status() -> None:
        resolved_id = resolve_default_tracker_config(
            conn, trial_id=trial_id, capture_id=capture_id
        )
        cfg_row = conn.execute(
            "SELECT name, is_named FROM tracker_configs WHERE id = ?", (resolved_id,)
        ).fetchone()
        if cfg_row is None:
            status_label.setText("(unresolved)")
            return
        own = conn.execute(
            "SELECT default_tracker_config_id FROM "
            + ("trials" if trial_id is not None else "captures")
            + " WHERE id = ?",
            (trial_id if trial_id is not None else capture_id,),
        ).fetchone()
        source = "" if own and own["default_tracker_config_id"] else " (inherited)"
        name = cfg_row["name"] if cfg_row["is_named"] else "(unnamed snapshot)"
        status_label.setText(f"{name}{source}")

    def _on_edit() -> None:
        dlg = DefaultConfigDialog(
            conn, trial_id=trial_id, capture_id=capture_id, parent=row_widget
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            _refresh_status()

    def _on_change() -> None:
        rows = [r for r in list_configs(conn) if r["is_named"]]
        if not rows:
            QMessageBox.information(
                row_widget, "No saved configs", "No named tracker configurations are saved yet."
            )
            return
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        labels = [f"{r['name']}  ({r['created_at'][:19]})" for r in rows]
        choice, ok = QInputDialog.getItem(
            row_widget, "Change default configuration", "Configuration:", labels, 0, False
        )
        if not ok:
            return
        chosen = rows[labels.index(choice)]
        set_default_tracker_config(
            conn, chosen["id"], trial_id=trial_id, capture_id=capture_id
        )
        _refresh_status()

    edit_btn.clicked.connect(_on_edit)
    change_btn.clicked.connect(_on_change)
    _refresh_status()
    return row_widget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tab_page(content: QFormLayout | QVBoxLayout | QWidget) -> QScrollArea:
    """Wrap a layout or widget in a scrollable page for one QTabWidget tab."""
    if isinstance(content, QWidget):
        page = content
    else:
        page = QWidget()
        page.setLayout(content)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setWidget(page)
    return scroll


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

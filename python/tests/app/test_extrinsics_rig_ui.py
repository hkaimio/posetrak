# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Dialog-level tests for Phase 8's portable calibration rig detection +
anchoring UI in ExtrinsicsAutoCalibDialog.

See docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md, section 9 Tier A and section 10.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QDialog, QMessageBox

from app.setup.extrinsics_solver import (
    CalibResult,
    CamCalibState,
    ControlPoint,
    MarkerGroup,
    MarkerPoseResult,
)
from app.setup.fiducial_markers import ARUCO_DICTIONARIES
from app.setup.page_extrinsics import (
    ExtrinsicsAutoCalibDialog,
    _SceneMarkerGroupPickerDialog,
    _SceneMarkerManagerDialog,
)
from posetrak.db.manage_marker_body import import_marker_body, upsert_scene_marker_body

_ONE_MARKER_RIG_YAML = """\
name: test-rig
units: meters
markers:
  - name: front
    type: aruco
    dictionary: DICT_4X4_50
    id: "3"
    size: 0.1
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
"""


def _render_marker_image(marker_id: int, dictionary: str = "DICT_4X4_50") -> np.ndarray:
    """A real ArUco marker rendered to a BGR image, with a white border so
    detection actually succeeds (mirrors test_fiducial_markers.py's helper)."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary])
    gray = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)
    padded = cv2.copyMakeBorder(gray, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


def _make_state(label: str, image: np.ndarray | None = None) -> CamCalibState:
    K = np.array([[900.0, 0.0, 150.0], [0.0, 900.0, 150.0], [0.0, 0.0, 1.0]])
    return CamCalibState(
        video_id=label, label=label, K=K, K_orig=K.copy(),
        dist=np.zeros((1, 4)), fisheye=False, image=image,
    )


@pytest.fixture()
def fake_conn(tmp_path):
    from posetrak.db.db import create_session
    conn = create_session(tmp_path / "rig_ui_test.db")
    yield conn
    conn.close()


@pytest.fixture()
def rig_yaml_path(tmp_path) -> Path:
    p = tmp_path / "test_rig.yaml"
    p.write_text(_ONE_MARKER_RIG_YAML, encoding="utf-8")
    return p


def test_dialog_starts_with_no_rig_loaded(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        assert dlg._rig_config is None
        assert dlg._rig_detector is None
        assert dlg._rig_detections_by_camera == {}
        assert not dlg._rig_anchored
        assert dlg._rig_status_label.text() == "No rig config loaded."
    finally:
        dlg.done(0)


def test_load_rig_config_populates_state_and_imports_to_db(qapp, fake_conn, rig_yaml_path) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)  # test-only helper, see below

        assert dlg._rig_config is not None
        assert dlg._rig_config.rig_id == "test-rig"
        assert dlg._rig_detector is not None
        assert dlg._rig_definition_id is not None
        assert "test-rig" in dlg._rig_status_label.text()

        row = fake_conn.execute(
            "SELECT name FROM marker_body_definitions WHERE id = ?", (dlg._rig_definition_id,)
        ).fetchone()
        assert row is not None
        assert row[0] == "test-rig"
    finally:
        dlg.done(0)


def test_detect_rig_without_config_shows_warning(qapp, fake_conn, monkeypatch) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        warned = []
        monkeypatch.setattr(
            "app.setup.page_extrinsics.QMessageBox.warning",
            lambda *a, **kw: warned.append(a),
        )
        dlg._on_detect_rig_clicked("cam_A")
        assert len(warned) == 1
        assert dlg._rig_detections_by_camera == {}
    finally:
        dlg.done(0)


def test_detect_rig_finds_marker(qapp, fake_conn, rig_yaml_path) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)
        dlg._on_detect_rig_clicked("cam_A")

        assert "cam_A" in dlg._rig_detections_by_camera
        assert len(dlg._rig_detections_by_camera["cam_A"]) == 1
        assert dlg._rig_detections_by_camera["cam_A"][0].marker_id == "3"
        # Loading a rig config with the marker already visible now
        # auto-anchors immediately (see _apply_loaded_rig_config).
        assert "anchored" in dlg._rig_status_label.text()
    finally:
        dlg.done(0)


def test_detect_rig_with_no_image_shows_warning_not_crash(qapp, fake_conn, monkeypatch, rig_yaml_path) -> None:
    states = [_make_state("cam_A", image=None)]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)
        warned = []
        monkeypatch.setattr(
            "app.setup.page_extrinsics.QMessageBox.warning",
            lambda *a, **kw: warned.append(a),
        )
        dlg._on_detect_rig_clicked("cam_A")
        assert len(warned) == 1
        assert dlg._rig_detections_by_camera == {}
    finally:
        dlg.done(0)


def test_control_points_empty_before_anchor(qapp, fake_conn, rig_yaml_path) -> None:
    """Unlike ChArUco, a rig detection has no free/unanchored intermediate
    state -- see _build_rig_group's docstring for why. That intermediate
    state can still occur transiently: the rig isn't visible in any
    camera's frame at load time (so auto-anchor-on-load finds nothing),
    then the user redetects a single camera by hand without also
    re-running "Anchor Rig"."""
    states = [_make_state("cam_A", image=None)]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)  # no image yet -> nothing auto-detected
        assert not dlg._rig_anchored

        # Now the camera "scrubs" to a frame where the marker is visible,
        # and the user redetects just that camera.
        dlg._states_by_id["cam_A"].image = _render_marker_image(3)
        dlg._on_detect_rig_clicked("cam_A")

        assert dlg._rig_detections_by_camera["cam_A"]
        assert not dlg._rig_anchored
        assert dlg._rig_control_points() == []
    finally:
        dlg.done(0)


def test_anchor_without_detection_shows_warning(qapp, fake_conn, monkeypatch, rig_yaml_path) -> None:
    states = [_make_state("cam_A", image=None)]  # marker not visible -> nothing to detect
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)
        assert not dlg._rig_anchored
        warned = []
        monkeypatch.setattr(
            "app.setup.page_extrinsics.QMessageBox.warning",
            lambda *a, **kw: warned.append(a),
        )
        dlg._on_anchor_from_rig()
        assert len(warned) == 1
        assert not dlg._rig_anchored
    finally:
        dlg.done(0)


def test_anchor_fixes_world_xyz_on_all_corners(qapp, fake_conn, rig_yaml_path) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)
        dlg._on_detect_rig_clicked("cam_A")
        dlg._on_anchor_from_rig()

        assert dlg._rig_anchored
        cps = dlg._rig_control_points()
        assert len(cps) == 4  # one marker's 4 corners
        assert all(cp.world_xyz is not None for cp in cps)
        assert "anchored" in dlg._rig_status_label.text()
    finally:
        dlg.done(0)


def test_clear_rig_resets_detection_state_not_config(qapp, fake_conn, rig_yaml_path) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)
        dlg._on_detect_rig_clicked("cam_A")
        dlg._on_anchor_from_rig()

        dlg._on_clear_rig()

        assert dlg._rig_detections_by_camera == {}
        assert not dlg._rig_anchored
        assert dlg._rig_config is not None  # config itself stays loaded
        assert "No rig config loaded" not in dlg._rig_status_label.text()
    finally:
        dlg.done(0)


def test_rig_marker_drawn_as_overlay(qapp, fake_conn, rig_yaml_path) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)
        dlg._on_detect_rig_clicked("cam_A")
        widget = dlg._cam_widgets["cam_A"]
        assert len(widget._markers) == 4
    finally:
        dlg.done(0)


def test_solve_includes_rig_control_points(qapp, fake_conn, rig_yaml_path, monkeypatch) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)
        dlg._on_detect_rig_clicked("cam_A")
        dlg._on_anchor_from_rig()

        captured = {}
        from app.setup import page_extrinsics as module
        orig = module._SolveThread.__init__

        def spy_init(self, states_, control_points, *args, **kwargs):
            captured["control_points"] = control_points
            return orig(self, states_, control_points, *args, **kwargs)

        monkeypatch.setattr(module._SolveThread, "__init__", spy_init)
        dlg._sift_check.setChecked(False)
        dlg._on_solve()

        names = {cp.name for cp in captured["control_points"]}
        assert "rig_3_c0" in names
        assert len(captured["control_points"]) == 4
    finally:
        if dlg._solve_thread is not None:
            dlg._solve_thread.wait(2000)
        dlg.done(0)


def test_load_rig_config_auto_anchors_when_marker_visible(qapp, fake_conn, rig_yaml_path) -> None:
    """Loading a rig config now runs detect-across-all-cameras + anchor
    immediately, collapsing what used to be three separate clicks (load,
    per-camera detect, "Set origin & axes") into one -- per user feedback
    after the first manual GUI test pass."""
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)

        assert dlg._rig_anchored
        assert "cam_A" in dlg._rig_detections_by_camera
        cps = dlg._rig_control_points()
        assert len(cps) == 4
        assert all(cp.world_xyz is not None for cp in cps)
    finally:
        dlg.done(0)


def test_load_rig_config_no_marker_visible_stays_unanchored(qapp, fake_conn, rig_yaml_path) -> None:
    states = [_make_state("cam_A", image=None)]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)

        assert dlg._rig_config is not None  # config still loaded
        assert not dlg._rig_anchored
        assert dlg._rig_detections_by_camera == {}
    finally:
        dlg.done(0)


def test_anchor_rig_button_redetects_every_camera(qapp, fake_conn, rig_yaml_path) -> None:
    """"Anchor Rig" always redetects fresh across every camera -- not just
    an anchor of whatever was already detected -- so it also serves as
    the "redo everything" action after scrubbing several cameras to
    better frames, per user feedback."""
    states = [
        _make_state("cam_A", image=None),
        _make_state("cam_B", image=None),
    ]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)
        assert not dlg._rig_anchored

        # Both cameras "scrub" to a frame showing the marker.
        dlg._states_by_id["cam_A"].image = _render_marker_image(3)
        dlg._states_by_id["cam_B"].image = _render_marker_image(3)
        dlg._on_anchor_from_rig()

        assert dlg._rig_anchored
        assert "cam_A" in dlg._rig_detections_by_camera
        assert "cam_B" in dlg._rig_detections_by_camera
        assert "2/2" in dlg._rig_status_label.text()
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# Min-cameras-to-anchor guard (2026-08-12) -- a physical rig glimpsed by
# only one stray camera is often left-over clutter from an earlier capture,
# not this capture's intended anchor (Harri's "moved rig" report). Doesn't
# apply to a "Load Markers…" config, and is clamped to however many
# cameras a dialog actually has (see _detect_and_anchor_rig).
# ---------------------------------------------------------------------------


def test_rig_seen_by_only_one_of_several_cameras_does_not_auto_anchor(
    qapp, fake_conn, rig_yaml_path
) -> None:
    states = [
        _make_state("cam_A", _render_marker_image(3)),  # sees the (moved) rig
        _make_state("cam_B", image=None),
        _make_state("cam_C", image=None),
    ]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)

        assert not dlg._rig_anchored
        assert "1/3" in dlg._rig_status_label.text() or "1/3" in dlg._status_label.text()
        assert dlg._rig_control_points() == []
    finally:
        dlg.done(0)


def test_rig_seen_by_only_one_camera_can_be_force_anchored_by_lowering_minimum(
    qapp, fake_conn, rig_yaml_path
) -> None:
    states = [
        _make_state("cam_A", _render_marker_image(3)),
        _make_state("cam_B", image=None),
    ]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)
        assert not dlg._rig_anchored

        dlg._rig_min_cameras_spin.setValue(1)
        dlg._on_anchor_from_rig()

        assert dlg._rig_anchored
        assert len(dlg._rig_control_points()) == 4
    finally:
        dlg.done(0)


def test_min_cameras_guard_does_not_apply_to_scene_markers_source(qapp, fake_conn) -> None:
    _seed_scene_marker_tag(fake_conn, "sess1", group_name="grp")
    states = [
        _make_state("cam_A", _render_marker_image(3)),
        _make_state("cam_B", image=None),
    ]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_scene_marker_group("grp")

        assert dlg._rig_anchored
        assert len(dlg._rig_control_points()) == 4
    finally:
        dlg.done(0)


def test_min_cameras_guard_clamped_to_available_cameras(qapp, fake_conn, rig_yaml_path) -> None:
    """The default minimum (2) must not block a genuinely single-camera
    dialog -- there's no way to satisfy "seen by >= 2 cameras" when only
    one exists, so the guard clamps to the dialog's own camera count."""
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        assert dlg._rig_min_cameras_spin.value() == 2  # default, unchanged
        dlg._load_rig_config_from_path(rig_yaml_path)

        assert dlg._rig_anchored
        assert len(dlg._rig_control_points()) == 4
    finally:
        dlg.done(0)


def test_load_invalid_rig_yaml_shows_warning_not_crash(qapp, fake_conn, tmp_path, monkeypatch) -> None:
    bad_path = tmp_path / "bad_rig.yaml"
    bad_path.write_text("name: bad\nmarkers:\n  - name: a\n    type: aruco\n", encoding="utf-8")
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        warned = []
        monkeypatch.setattr(
            "app.setup.page_extrinsics.QMessageBox.warning",
            lambda *a, **kw: warned.append(a),
        )
        dlg._load_rig_config_from_path(bad_path)  # must not raise
        assert len(warned) == 1
        assert dlg._rig_config is None
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# "Load from scene markers…" (Phase 9, design doc section 9 Tier B) --
# re-anchoring from previously-persisted scattered tags, no physical rig.
# Same underlying anchor_from_marker_rig mechanism as the file-loaded rig
# above, just a different MarkerRigConfig source -- mirrors the CLI's
# `extrinsics reanchor` command's own DB query/construction exactly.
# ---------------------------------------------------------------------------


def _seed_scene_marker_tag(
    conn, session_id: str, marker_id: str = "3", dictionary: str = "DICT_4X4_50", size: float = 0.1,
    group_name: str | None = None,
) -> None:
    from app.setup.page_extrinsics import upsert_scene_marker_body
    conn.execute(
        "INSERT OR IGNORE INTO mocap_sessions (id, recorded_at) VALUES (?, '2026-01-01')",
        (session_id,),
    )
    conn.commit()
    upsert_scene_marker_body(
        conn, session_id, label=f"tag:{marker_id}",
        R=np.eye(3), t=np.array([0.5, 0.0, 2.0]), group_name=group_name,
        marker_type="aruco", dictionary=dictionary, marker_id=marker_id, marker_size=size,
    )


def test_load_from_scene_markers_with_none_persisted_shows_warning(qapp, fake_conn, monkeypatch) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        fake_conn.execute(
            "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
        )
        fake_conn.commit()
        warned = []
        monkeypatch.setattr(
            "app.setup.page_extrinsics.QMessageBox.warning",
            lambda *a, **kw: warned.append(a),
        )
        dlg._on_load_rig_from_scene_markers()
        assert len(warned) == 1
        assert dlg._rig_config is None
    finally:
        dlg.done(0)


def test_load_from_scene_markers_populates_config(qapp, fake_conn) -> None:
    _seed_scene_marker_tag(fake_conn, "sess1", group_name="grp")
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_scene_marker_group("grp")

        assert dlg._rig_config is not None
        assert dlg._rig_config.rig_id == "scene markers (grp)"
        assert set(dlg._rig_config.marker_corners) == {"3"}
        assert dlg._rig_config.marker_dictionaries["3"] == "DICT_4X4_50"
        assert dlg._rig_source == "scene_markers"
        assert dlg._rig_definition_id is None
        assert dlg._rig_detector is not None
    finally:
        dlg.done(0)


def test_load_from_scene_markers_then_detect_and_anchor(qapp, fake_conn) -> None:
    _seed_scene_marker_tag(fake_conn, "sess1", group_name="grp")
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_scene_marker_group("grp")
        dlg._on_detect_rig_clicked("cam_A")
        assert "cam_A" in dlg._rig_detections_by_camera
        assert len(dlg._rig_detections_by_camera["cam_A"]) == 1

        dlg._on_anchor_from_rig()
        assert dlg._rig_anchored
        cps = dlg._rig_control_points()
        assert len(cps) == 4
        assert all(cp.world_xyz is not None for cp in cps)
    finally:
        dlg.done(0)


def test_scene_markers_source_not_repersisted_on_accept(qapp, fake_conn) -> None:
    """A scene_markers-sourced rig config must not create a new
    scene_marker_bodies row on Accept -- its rows already exist and their
    own geometry hasn't changed by re-anchoring a different capture from
    them (only the cameras' poses changed)."""
    _seed_scene_marker_tag(fake_conn, "sess1", group_name="grp")
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        count_before = fake_conn.execute(
            "SELECT COUNT(*) FROM scene_marker_bodies WHERE session_id = 'sess1'"
        ).fetchone()[0]

        dlg._load_rig_config_from_scene_marker_group("grp")
        assert dlg._rig_source == "scene_markers"
        # Directly exercise the same guard _on_accept uses, without needing
        # a full solve/CalibResult round-trip through _SolveThread.
        assert not (dlg._rig_anchored and dlg._rig_config is not None and dlg._rig_source == "file")

        count_after = fake_conn.execute(
            "SELECT COUNT(*) FROM scene_marker_bodies WHERE session_id = 'sess1'"
        ).fetchone()[0]
        assert count_after == count_before
    finally:
        dlg.done(0)


def test_load_rig_config_from_file_sets_source_file(qapp, fake_conn, rig_yaml_path) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)
        assert dlg._rig_source == "file"
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# "From Registry…" -- pick an already-imported marker_body_definitions row
# without re-selecting its YAML file (the gap the user hit after a CLI
# `marker-body import`: the dialog's in-memory state had no way to see a
# rig that only existed in the DB).
# ---------------------------------------------------------------------------


def test_load_rig_config_from_registry_row_populates_state(qapp, fake_conn, rig_yaml_path) -> None:
    definition_id = import_marker_body(fake_conn, rig_yaml_path, name="test-rig")
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        row = fake_conn.execute(
            "SELECT yaml_content FROM marker_body_definitions WHERE id = ?", (definition_id,)
        ).fetchone()

        dlg._load_rig_config_from_registry_row(row[0], definition_id)

        assert dlg._rig_config is not None
        assert dlg._rig_config.rig_id == "test-rig"
        assert dlg._rig_definition_id == definition_id
        assert dlg._rig_source == "file"
        # Same auto-detect-and-anchor behaviour as the file/registry-picker
        # paths -- the marker is visible in cam_A's image.
        assert dlg._rig_anchored
    finally:
        dlg.done(0)


def test_load_rig_config_from_registry_row_invalid_yaml_shows_warning(
    qapp, fake_conn, monkeypatch
) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        warned = []
        monkeypatch.setattr(
            "app.setup.page_extrinsics.QMessageBox.warning",
            lambda *a, **kw: warned.append(a),
        )
        dlg._load_rig_config_from_registry_row("not: [valid, yaml", "some-id")
        assert len(warned) == 1
        assert dlg._rig_config is None
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# Persisting sized ArUco markers ("Detect ArUco" + a real size, not a rig)
# to scene_marker_bodies via the explicit "Save Markers…" action (UX Phase
# 5, see docs/roadmap/features/extrinsics-improvements/
# extrinsics-ux-redesign.md) -- the GUI's own route to what the CLI's
# `anchor-rig --tag-size` does, so a later capture can reuse them via "Load
# Markers…"/`reanchor --name`. Accept itself no longer persists anything;
# see _save_markers_items/_save_markers.
# ---------------------------------------------------------------------------


def _seed_session_and_camera(conn, session_id: str = "sess1", label: str = "cam_A") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO mocap_sessions (id, recorded_at) VALUES (?, '2026-01-01')",
        (session_id,),
    )
    conn.execute(
        "INSERT INTO camera_models (id, manufacturer, model_name) VALUES ('model1', 'Test', 'Cam')"
    )
    conn.execute(
        "INSERT INTO camera_instances (id, camera_model_id, label) VALUES ('inst1', 'model1', ?)",
        (label,),
    )
    conn.commit()


def _anchor_solved_sized_marker(dlg, states) -> None:
    """Wires a single sized ArUco marker ('tag:9') into a dialog's
    post-solve state, as if "Detect ArUco" + a real Solve had just run --
    shared setup for the Save Markers tests below."""
    dlg._marker_groups["9"] = MarkerGroup(marker_id="9", size=0.1, dictionary="DICT_5X5_50")
    rvec, _ = cv2.Rodrigues(np.eye(3))
    mp = MarkerPoseResult(
        rvec=rvec, tvec=np.array([1.0, 2.0, 3.0]), size=0.1, rms_reprojection_px=0.5
    )
    states[0].R = np.eye(3)
    states[0].t = np.zeros((3, 1))
    dlg._result = CalibResult(
        cameras={"cam_A": states[0]}, points_3d=[], reprojection_errors={},
        unsolved=[], pair_matches={}, marker_poses={"9": mp},
    )


def test_save_markers_items_empty_when_nothing_anchored(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(9))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        assert dlg._save_markers_items() == []
        assert not dlg._save_markers_btn.isEnabled()
    finally:
        dlg.done(0)


def test_save_markers_items_lists_sized_marker_after_solve(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(9))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        _anchor_solved_sized_marker(dlg, states)
        dlg._refresh_save_markers_button()

        assert any(label == "tag:9" for label, _ in dlg._save_markers_items())
        assert dlg._save_markers_btn.isEnabled()
    finally:
        dlg.done(0)


def test_save_markers_directly_persists_selected_sized_marker(qapp, fake_conn) -> None:
    _seed_session_and_camera(fake_conn)
    states = [_make_state("cam_A", _render_marker_image(9))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        _anchor_solved_sized_marker(dlg, states)

        dlg._save_markers("room7", {"tag:9"})

        row = fake_conn.execute(
            "SELECT dictionary, marker_id, marker_size, group_name FROM scene_marker_bodies "
            "WHERE session_id = 'sess1' AND label = 'tag:9'"
        ).fetchone()
        assert row is not None
        assert row[0] == "DICT_5X5_50"
        assert row[1] == "9"
        assert row[2] == pytest.approx(0.1)
        assert row[3] == "room7"
    finally:
        dlg.done(0)


def test_save_markers_only_persists_selected_labels(qapp, fake_conn, rig_yaml_path) -> None:
    """A file-sourced rig anchor is also eligible -- but only the labels
    the caller (the "Save Markers…" dialog's checklist) actually selected
    get written."""
    _seed_session_and_camera(fake_conn)
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)  # auto-anchors (marker "3" visible)
        assert dlg._rig_anchored
        rig_label = f"rig:{dlg._rig_config.rig_id}"
        assert (rig_label, "primary anchor") in dlg._save_markers_items()

        dlg._save_markers("room7", set())  # nothing selected

        count = fake_conn.execute(
            "SELECT COUNT(*) FROM scene_marker_bodies WHERE session_id = 'sess1'"
        ).fetchone()[0]
        assert count == 0

        dlg._save_markers("room7", {rig_label})

        row = fake_conn.execute(
            "SELECT group_name FROM scene_marker_bodies WHERE session_id = 'sess1' "
            f"AND label = '{rig_label}'"
        ).fetchone()
        assert row is not None
        assert row[0] == "room7"
    finally:
        dlg.done(0)


def test_on_save_markers_warns_when_nothing_eligible(qapp, fake_conn, monkeypatch) -> None:
    states = [_make_state("cam_A", _render_marker_image(9))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        warned = []
        monkeypatch.setattr(
            "app.setup.page_extrinsics.QMessageBox.warning",
            lambda *a, **kw: warned.append(a),
        )
        dlg._on_save_markers()
        assert len(warned) == 1
    finally:
        dlg.done(0)


class _FakeSaveMarkersDialog:
    """Stands in for _SaveMarkersDialog -- avoids constructing the real
    checklist/name-field widget just to drive _on_save_markers's wiring."""

    def __init__(self, items, parent=None) -> None:
        self.items = items

    def exec(self) -> int:
        return QDialog.DialogCode.Accepted

    def group_name(self) -> str:
        return "room7"

    def selected_labels(self) -> set[str]:
        return {"tag:9"}


def test_on_save_markers_opens_dialog_and_saves_selected(qapp, fake_conn, monkeypatch) -> None:
    _seed_session_and_camera(fake_conn)
    states = [_make_state("cam_A", _render_marker_image(9))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        _anchor_solved_sized_marker(dlg, states)
        monkeypatch.setattr(
            "app.setup.page_extrinsics._SaveMarkersDialog", _FakeSaveMarkersDialog
        )
        dlg._on_save_markers()

        row = fake_conn.execute(
            "SELECT group_name FROM scene_marker_bodies WHERE session_id = 'sess1' "
            "AND label = 'tag:9'"
        ).fetchone()
        assert row is not None
        assert row[0] == "room7"
    finally:
        dlg.done(0)


def test_on_save_markers_nothing_selected_does_not_save(qapp, fake_conn, monkeypatch) -> None:
    _seed_session_and_camera(fake_conn)
    states = [_make_state("cam_A", _render_marker_image(9))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        _anchor_solved_sized_marker(dlg, states)

        class _NothingSelectedDialog(_FakeSaveMarkersDialog):
            def selected_labels(self) -> set[str]:
                return set()

        monkeypatch.setattr(
            "app.setup.page_extrinsics._SaveMarkersDialog", _NothingSelectedDialog
        )
        dlg._on_save_markers()

        count = fake_conn.execute(
            "SELECT COUNT(*) FROM scene_marker_bodies WHERE session_id = 'sess1'"
        ).fetchone()[0]
        assert count == 0
    finally:
        dlg.done(0)


def test_on_solve_done_refreshes_save_markers_button(qapp, fake_conn) -> None:
    """_refresh_save_markers_button() must run after a solve too, not just
    after rig-anchor changes -- a sized marker only becomes eligible once
    _result is populated (see _save_markers_items)."""
    states = [_make_state("cam_A", _render_marker_image(9))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        assert not dlg._save_markers_btn.isEnabled()

        rvec, _ = cv2.Rodrigues(np.eye(3))
        mp = MarkerPoseResult(
            rvec=rvec, tvec=np.array([1.0, 2.0, 3.0]), size=0.1, rms_reprojection_px=0.5
        )
        states[0].R = np.eye(3)
        states[0].t = np.zeros((3, 1))
        result = CalibResult(
            cameras={"cam_A": states[0]}, points_3d=[], reprojection_errors={},
            unsolved=[], pair_matches={}, marker_poses={"9": mp},
        )
        dlg._on_solve_done(result)

        assert dlg._save_markers_btn.isEnabled()
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# Replacing an existing world-frame anchor (rig or manual control points)
# with a newly-loaded one asks first (UX Phase 5's
# _confirm_replace_existing_anchor -- CLAUDE.md's "automation vs. prior
# human edits" design principle applied to loading a rig/scene-marker
# config over something already anchored).
# ---------------------------------------------------------------------------


def test_confirm_replace_returns_true_when_nothing_anchored_yet(qapp, fake_conn) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        assert dlg._confirm_replace_existing_anchor() is True
    finally:
        dlg.done(0)


def test_loading_new_rig_over_existing_anchor_asks_and_replaces_on_yes(
    qapp, fake_conn, rig_yaml_path, monkeypatch
) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)  # auto-anchors
        assert dlg._rig_anchored

        asked = []
        monkeypatch.setattr(
            "app.setup.page_extrinsics.QMessageBox.question",
            lambda *a, **kw: (asked.append(a), QMessageBox.StandardButton.Yes)[1],
        )
        dlg._load_rig_config_from_path(rig_yaml_path)  # loading again should ask

        assert len(asked) == 1
        assert dlg._rig_anchored  # replaced, still anchored
    finally:
        dlg.done(0)


def test_loading_new_rig_over_existing_anchor_declines_leaves_state_unchanged(
    qapp, fake_conn, rig_yaml_path, monkeypatch
) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)
        assert dlg._rig_anchored
        original_config = dlg._rig_config

        monkeypatch.setattr(
            "app.setup.page_extrinsics.QMessageBox.question",
            lambda *a, **kw: QMessageBox.StandardButton.No,
        )
        result = dlg._apply_loaded_rig_config(
            original_config, definition_id=dlg._rig_definition_id, source="file",
        )

        assert result is False
        assert dlg._rig_config is original_config  # untouched
        assert dlg._rig_anchored
    finally:
        dlg.done(0)


def test_confirm_replace_triggers_on_manual_control_point_anchor(qapp, fake_conn, monkeypatch) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._control_points.append(
            ControlPoint(name="manual_cp", world_xyz=np.array([1.0, 2.0, 3.0]))
        )

        asked = []
        monkeypatch.setattr(
            "app.setup.page_extrinsics.QMessageBox.question",
            lambda *a, **kw: (asked.append(a), QMessageBox.StandardButton.No)[1],
        )
        result = dlg._confirm_replace_existing_anchor()

        assert len(asked) == 1
        assert result is False
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# Manage Scene Markers dialog (2026-08-12) -- view/delete stored scene
# markers, e.g. to prune a rig's stale anchor row once physically removed.
# ---------------------------------------------------------------------------


def test_manager_dialog_lists_stored_markers(qapp, fake_conn) -> None:
    _seed_session_and_camera(fake_conn)
    upsert_scene_marker_body(
        fake_conn, "sess1", label="rig:calib-box", R=np.eye(3), t=np.zeros(3),
        marker_body_definition_id="def1", is_primary_anchor=True,
    )
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:7", R=np.eye(3), t=np.array([1.0, 2.0, 3.0]),
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="7", marker_size=0.1,
    )

    dlg = _SceneMarkerManagerDialog(fake_conn, "sess1")
    try:
        assert dlg._table.rowCount() == 2
        labels = {dlg._table.item(i, 0).text() for i in range(dlg._table.rowCount())}
        assert labels == {"rig:calib-box", "tag:7"}
    finally:
        dlg.done(0)


def test_manager_dialog_delete_removes_row(qapp, fake_conn, monkeypatch) -> None:
    _seed_session_and_camera(fake_conn)
    upsert_scene_marker_body(
        fake_conn, "sess1", label="rig:calib-box", R=np.eye(3), t=np.zeros(3),
        marker_body_definition_id="def1", is_primary_anchor=True,
    )
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:7", R=np.eye(3), t=np.array([1.0, 2.0, 3.0]),
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="7", marker_size=0.1,
    )
    monkeypatch.setattr(
        "app.setup.page_extrinsics.QMessageBox.question",
        lambda *a, **kw: QMessageBox.StandardButton.Yes,
    )

    dlg = _SceneMarkerManagerDialog(fake_conn, "sess1")
    try:
        dlg._table.selectRow(
            next(i for i in range(dlg._table.rowCount()) if dlg._table.item(i, 0).text() == "rig:calib-box")
        )
        dlg._on_delete_selected()

        assert dlg._table.rowCount() == 1
        assert dlg._table.item(0, 0).text() == "tag:7"
        rows = fake_conn.execute(
            "SELECT label FROM scene_marker_bodies WHERE session_id = 'sess1'"
        ).fetchall()
        assert [r[0] for r in rows] == ["tag:7"]
    finally:
        dlg.done(0)


def test_manager_dialog_delete_cancelled_keeps_row(qapp, fake_conn, monkeypatch) -> None:
    _seed_session_and_camera(fake_conn)
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:7", R=np.eye(3), t=np.zeros(3),
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="7", marker_size=0.1,
    )
    monkeypatch.setattr(
        "app.setup.page_extrinsics.QMessageBox.question",
        lambda *a, **kw: QMessageBox.StandardButton.No,
    )

    dlg = _SceneMarkerManagerDialog(fake_conn, "sess1")
    try:
        dlg._table.selectRow(0)
        dlg._on_delete_selected()

        assert dlg._table.rowCount() == 1
        count = fake_conn.execute(
            "SELECT COUNT(*) FROM scene_marker_bodies WHERE session_id = 'sess1'"
        ).fetchone()[0]
        assert count == 1
    finally:
        dlg.done(0)


def test_on_manage_scene_markers_opens_dialog(qapp, fake_conn, monkeypatch) -> None:
    _seed_session_and_camera(fake_conn)
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        opened = []
        monkeypatch.setattr(
            "app.setup.page_extrinsics._SceneMarkerManagerDialog.exec",
            lambda self: opened.append(True),
        )
        dlg._on_manage_scene_markers()
        assert opened == [True]
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# Rig registry manager (view + delete + import marker_body_definitions
# rows, 2026-08-15 follow-up) -- "Calibration rig setup"'s counterpart to
# "Manage markers…"/_SceneMarkerManagerDialog above. See
# _RigRegistryManagerDialog/_on_manage_rigs.
# ---------------------------------------------------------------------------


def test_rig_registry_manager_lists_imported_rigs(qapp, fake_conn, rig_yaml_path) -> None:
    from app.setup.page_extrinsics import _RigRegistryManagerDialog

    import_marker_body(fake_conn, rig_yaml_path, name="test-rig")
    dlg = _RigRegistryManagerDialog(fake_conn)
    try:
        assert dlg._table.rowCount() == 1
        assert dlg._table.item(0, 0).text() == "test-rig"
    finally:
        dlg.done(0)


def test_rig_registry_manager_delete_removes_row(qapp, fake_conn, rig_yaml_path, monkeypatch) -> None:
    from PySide6.QtWidgets import QMessageBox
    from app.setup.page_extrinsics import _RigRegistryManagerDialog

    import_marker_body(fake_conn, rig_yaml_path, name="test-rig")
    monkeypatch.setattr(
        "app.setup.page_extrinsics.QMessageBox.question",
        lambda *a, **kw: QMessageBox.StandardButton.Yes,
    )

    dlg = _RigRegistryManagerDialog(fake_conn)
    try:
        dlg._table.selectRow(0)
        dlg._on_delete_selected()

        assert dlg._table.rowCount() == 0
        assert fake_conn.execute(
            "SELECT COUNT(*) FROM marker_body_definitions"
        ).fetchone()[0] == 0
    finally:
        dlg.done(0)


def test_rig_registry_manager_delete_cancelled_keeps_row(
    qapp, fake_conn, rig_yaml_path, monkeypatch,
) -> None:
    from PySide6.QtWidgets import QMessageBox
    from app.setup.page_extrinsics import _RigRegistryManagerDialog

    import_marker_body(fake_conn, rig_yaml_path, name="test-rig")
    monkeypatch.setattr(
        "app.setup.page_extrinsics.QMessageBox.question",
        lambda *a, **kw: QMessageBox.StandardButton.No,
    )

    dlg = _RigRegistryManagerDialog(fake_conn)
    try:
        dlg._table.selectRow(0)
        dlg._on_delete_selected()

        assert dlg._table.rowCount() == 1
    finally:
        dlg.done(0)


def test_rig_registry_manager_from_file_imports_without_detecting(
    qapp, fake_conn, rig_yaml_path, monkeypatch,
) -> None:
    from app.setup.page_extrinsics import _RigRegistryManagerDialog

    monkeypatch.setattr(
        "app.setup.page_extrinsics.QFileDialog.getOpenFileName",
        lambda *a, **kw: (str(rig_yaml_path), ""),
    )
    dlg = _RigRegistryManagerDialog(fake_conn)
    try:
        dlg._on_from_file()

        assert dlg._table.rowCount() == 1
        assert dlg._table.item(0, 0).text() == "test-rig"
    finally:
        dlg.done(0)


def test_rig_registry_manager_from_file_invalid_yaml_shows_warning(
    qapp, fake_conn, tmp_path, monkeypatch,
) -> None:
    from app.setup.page_extrinsics import _RigRegistryManagerDialog

    bad_path = tmp_path / "bad_rig.yaml"
    bad_path.write_text("name: bad\nmarkers:\n  - name: a\n    type: aruco\n", encoding="utf-8")
    monkeypatch.setattr(
        "app.setup.page_extrinsics.QFileDialog.getOpenFileName",
        lambda *a, **kw: (str(bad_path), ""),
    )
    warned = []
    monkeypatch.setattr(
        "app.setup.page_extrinsics.QMessageBox.warning",
        lambda *a, **kw: warned.append(a),
    )
    dlg = _RigRegistryManagerDialog(fake_conn)
    try:
        dlg._on_from_file()  # must not raise

        assert len(warned) == 1
        assert dlg._table.rowCount() == 0
    finally:
        dlg.done(0)


def test_on_manage_rigs_opens_dialog(qapp, fake_conn, monkeypatch) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        opened = []
        monkeypatch.setattr(
            "app.setup.page_extrinsics._RigRegistryManagerDialog.exec",
            lambda self: opened.append(True),
        )
        dlg._on_manage_rigs()
        assert opened == [True]
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# A loaded physical rig's own markers must never leak into "Detect ArUco"
# as ordinary scattered tags (2026-08-12) -- Harri's report: a deleted
# "rig:<id>" scene marker still "came back" because its individual
# corner markers had separately been saved as "tag:<id>" rows via Detect
# ArUco, which had no rig exclusion (unlike the ChArUco one it already had).
# ---------------------------------------------------------------------------


def test_aruco_marker_excluded_when_matches_loaded_rig(qapp, fake_conn, rig_yaml_path) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)  # rig's own marker id "3", DICT_4X4_50

        dlg._on_detect_aruco_clicked("cam_A")

        assert "3" not in dlg._marker_groups
        assert "belonging to the ChArUco board/rig excluded" in dlg._status_label.text()
    finally:
        dlg.done(0)


def test_aruco_marker_not_excluded_for_scene_markers_source(qapp, fake_conn) -> None:
    """A "Load Markers…" config's own marker ids ARE ordinary
    scattered tags -- they must stay detectable/refreshable via "Detect
    ArUco", unlike a genuine physical rig's markers."""
    _seed_scene_marker_tag(fake_conn, "sess1", marker_id="3", dictionary="DICT_4X4_50", group_name="grp")
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_scene_marker_group("grp")
        assert dlg._rig_source == "scene_markers"

        dlg._on_detect_aruco_clicked("cam_A")

        assert "3" in dlg._marker_groups
    finally:
        dlg.done(0)


def test_loading_rig_purges_matching_marker_groups_detected_earlier(
    qapp, fake_conn, rig_yaml_path
) -> None:
    """Click order shouldn't matter: if "Detect ArUco" ran before the rig
    was loaded (no exclusion could have applied yet), loading the rig
    retroactively purges any of its own markers already sitting in
    _marker_groups."""
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_detect_aruco_clicked("cam_A")
        assert "3" in dlg._marker_groups

        dlg._load_rig_config_from_path(rig_yaml_path)

        assert "3" not in dlg._marker_groups
    finally:
        dlg.done(0)


def test_manager_dialog_flags_row_matching_rig_geometry(qapp, fake_conn, rig_yaml_path) -> None:
    """A "tag:<id>" row whose (dictionary, marker_id) matches a known
    rig's own marker is flagged, even though nothing in the row itself
    says it came from a rig -- helps find stale leaked-in rows like
    Harri's report."""
    import_marker_body(fake_conn, rig_yaml_path, name="test-rig")
    fake_conn.execute(
        "INSERT OR IGNORE INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    fake_conn.commit()
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:3", R=np.eye(3), t=np.zeros(3),
        marker_type="aruco", dictionary="DICT_4X4_50", marker_id="3", marker_size=0.1,
    )
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:99", R=np.eye(3), t=np.zeros(3),
        marker_type="aruco", dictionary="DICT_5X5_50", marker_id="99", marker_size=0.1,
    )

    dlg = _SceneMarkerManagerDialog(fake_conn, "sess1")
    try:
        rows_by_label = {
            dlg._table.item(i, 0).text(): dlg._table.item(i, 8).text()
            for i in range(dlg._table.rowCount())
        }
        assert "test-rig" in rows_by_label["tag:3"]
        assert rows_by_label["tag:99"] == ""
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# Scene marker groups (group_name, 2026-08-12) -- named groups (e.g. one per
# room) so "Load Markers…" can load a specific room's markers instead
# of every stored marker in the session loading together indiscriminately.
# ---------------------------------------------------------------------------


def test_load_from_scene_markers_no_named_groups_shows_warning(qapp, fake_conn, monkeypatch) -> None:
    """Always-named as of UX Phase 5 (see docs/roadmap/features/
    extrinsics-improvements/extrinsics-ux-redesign.md) -- an ungrouped
    tag from before that requirement existed isn't a pickable config
    anymore, so this warns instead of silently loading it."""
    _seed_scene_marker_tag(fake_conn, "sess1")  # ungrouped
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        warned = []
        monkeypatch.setattr(
            "app.setup.page_extrinsics.QMessageBox.warning",
            lambda *a, **kw: warned.append(a),
        )
        dlg._on_load_rig_from_scene_markers()
        assert len(warned) == 1
        assert dlg._rig_config is None
    finally:
        dlg.done(0)


def test_load_from_scene_markers_with_named_groups_opens_picker(qapp, fake_conn, monkeypatch) -> None:
    _seed_scene_marker_tag(fake_conn, "sess1", marker_id="3")
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:7", R=np.eye(3), t=np.array([1.0, 0.0, 0.0]),
        group_name="room7",
        marker_type="aruco", dictionary="DICT_4X4_50", marker_id="7", marker_size=0.1,
    )
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        from PySide6.QtWidgets import QDialog as _QDialog

        opened = []

        def fake_exec(self):
            opened.append(self)
            return _QDialog.DialogCode.Rejected

        monkeypatch.setattr(_SceneMarkerGroupPickerDialog, "exec", fake_exec)
        dlg._on_load_rig_from_scene_markers()

        assert len(opened) == 1
        assert dlg._rig_config is None  # rejected -> nothing loaded
    finally:
        dlg.done(0)


def test_load_rig_config_from_scene_marker_group_loads_named_group(qapp, fake_conn) -> None:
    fake_conn.execute(
        "INSERT OR IGNORE INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    fake_conn.commit()
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:7", R=np.eye(3), t=np.zeros(3), group_name="room7",
        marker_type="aruco", dictionary="DICT_4X4_50", marker_id="7", marker_size=0.1,
    )
    states = [_make_state("cam_A", image=None)]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_scene_marker_group("room7")

        assert dlg._rig_config is not None
        assert set(dlg._rig_config.marker_corners) == {"7"}
        assert "room7" in dlg._status_label.text()
    finally:
        dlg.done(0)


def test_load_rig_config_from_scene_marker_group_different_groups_dont_mix(
    qapp, fake_conn,
) -> None:
    fake_conn.execute(
        "INSERT OR IGNORE INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    fake_conn.commit()
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:3", R=np.eye(3), t=np.zeros(3), group_name="room7",
        marker_type="aruco", dictionary="DICT_4X4_50", marker_id="3", marker_size=0.1,
    )
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:9", R=np.eye(3), t=np.zeros(3), group_name="room8",
        marker_type="aruco", dictionary="DICT_4X4_50", marker_id="9", marker_size=0.1,
    )
    states = [_make_state("cam_A", image=None)]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_scene_marker_group("room7")
        assert set(dlg._rig_config.marker_corners) == {"3"}
    finally:
        dlg.done(0)


def test_group_picker_dialog_lists_named_groups_only(qapp, fake_conn) -> None:
    """Always-named as of UX Phase 5 -- no more "(ungrouped)" row; an
    ungrouped tag from before that requirement existed just doesn't show
    up here (see test_load_from_scene_markers_no_named_groups_shows_warning
    for the "nothing to pick from" case)."""
    from posetrak.db.manage_marker_body import list_scene_marker_group_names
    fake_conn.execute(
        "INSERT OR IGNORE INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    fake_conn.commit()
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:3", R=np.eye(3), t=np.zeros(3), group_name="room7",
        marker_type="aruco", dictionary="DICT_4X4_50", marker_id="3", marker_size=0.1,
    )
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:9", R=np.eye(3), t=np.zeros(3),
        marker_type="aruco", dictionary="DICT_4X4_50", marker_id="9", marker_size=0.1,
    )  # ungrouped -- must not appear
    named = list_scene_marker_group_names(fake_conn, "sess1")

    dlg = _SceneMarkerGroupPickerDialog(named)
    try:
        names = {dlg._table.item(i, 0).text() for i in range(dlg._table.rowCount())}
        assert names == {"room7"}
    finally:
        dlg.done(0)


def test_group_picker_dialog_selecting_named_group_returns_name(qapp, fake_conn) -> None:
    from posetrak.db.manage_marker_body import list_scene_marker_group_names
    fake_conn.execute(
        "INSERT OR IGNORE INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    fake_conn.commit()
    upsert_scene_marker_body(
        fake_conn, "sess1", label="tag:3", R=np.eye(3), t=np.zeros(3), group_name="room7",
        marker_type="aruco", dictionary="DICT_4X4_50", marker_id="3", marker_size=0.1,
    )
    named = list_scene_marker_group_names(fake_conn, "sess1")
    dlg = _SceneMarkerGroupPickerDialog(named)
    try:
        dlg._table.selectRow(0)
        dlg.accept()
        assert dlg.selected_group_name() == "room7"
    finally:
        dlg.done(0)


def test_group_picker_dialog_no_selection_accept_is_noop(qapp, fake_conn) -> None:
    dlg = _SceneMarkerGroupPickerDialog([])
    try:
        dlg.accept()  # nothing to select -- must not raise or set a name
        assert dlg.selected_group_name() is None
        assert dlg.result() != QDialog.DialogCode.Accepted
    finally:
        dlg.done(0)


# ---------------------------------------------------------------------------
# "Calib rig…" button-bar dialog (2026-08-14 follow-up, Harri: "charuco
# board is closer to a calibration rig so I'd add charuco boards as an
# option to the rig dialog... maybe it could be another tab in the
# dialog"). Physical Rig tab tested here (see
# test_extrinsics_charuco_ui.py for the ChArUco Board tab). See
# _CalibRigDialog/_on_calib_rig_bulk.
# ---------------------------------------------------------------------------


def _default_charuco_settings() -> dict:
    return {
        "dictionary": "DICT_4X4_50", "squares_x": 5, "squares_y": 7,
        "square_length": 0.04, "marker_length": 0.02,
        "face_up": True, "legacy_pattern": False, "min_marker_pct": 1.0,
    }


def test_calib_rig_dialog_physical_tab_lists_registry_rigs(qapp, fake_conn, rig_yaml_path) -> None:
    from app.setup.page_extrinsics import _CalibRigDialog

    import_marker_body(fake_conn, rig_yaml_path, name="test-rig")
    dlg = _CalibRigDialog(fake_conn, _default_charuco_settings())
    try:
        assert dlg._registry_table.rowCount() == 1
        assert dlg._registry_table.item(0, 0).text() == "test-rig"
    finally:
        dlg.done(0)


def test_calib_rig_dialog_ok_with_no_selection_warns(qapp, fake_conn, monkeypatch) -> None:
    from app.setup.page_extrinsics import _CalibRigDialog

    dlg = _CalibRigDialog(fake_conn, _default_charuco_settings())
    try:
        warned = []
        monkeypatch.setattr(
            "app.setup.page_extrinsics.QMessageBox.warning",
            lambda *a, **kw: warned.append(a),
        )
        dlg._on_ok()
        assert len(warned) == 1
        assert dlg.result_kind() == _CalibRigDialog.RESULT_NONE
    finally:
        dlg.done(0)


def test_calib_rig_dialog_ok_with_registry_row_selected(qapp, fake_conn, rig_yaml_path) -> None:
    from app.setup.page_extrinsics import _CalibRigDialog

    import_marker_body(fake_conn, rig_yaml_path, name="test-rig")
    dlg = _CalibRigDialog(fake_conn, _default_charuco_settings())
    try:
        dlg._registry_table.selectRow(0)
        dlg._on_ok()
        assert dlg.result_kind() == _CalibRigDialog.RESULT_REGISTRY
        assert dlg.registry_yaml() is not None
    finally:
        dlg.done(0)


def test_calib_rig_dialog_double_click_registry_row_accepts(qapp, fake_conn, rig_yaml_path) -> None:
    from app.setup.page_extrinsics import _CalibRigDialog

    import_marker_body(fake_conn, rig_yaml_path, name="test-rig")
    dlg = _CalibRigDialog(fake_conn, _default_charuco_settings())
    try:
        dlg._registry_table.selectRow(0)
        dlg._registry_table.doubleClicked.emit(dlg._registry_table.model().index(0, 0))
        assert dlg.result_kind() == _CalibRigDialog.RESULT_REGISTRY
        assert dlg.result() == QDialog.DialogCode.Accepted
    finally:
        dlg.done(0)


def test_calib_rig_dialog_from_file_button_accepts_with_file_kind(
    qapp, fake_conn, rig_yaml_path, monkeypatch,
) -> None:
    from app.setup.page_extrinsics import _CalibRigDialog

    dlg = _CalibRigDialog(fake_conn, _default_charuco_settings())
    try:
        monkeypatch.setattr(
            "app.setup.page_extrinsics.QFileDialog.getOpenFileName",
            lambda *a, **kw: (str(rig_yaml_path), ""),
        )
        dlg._on_from_file()
        assert dlg.result_kind() == _CalibRigDialog.RESULT_FILE
        assert dlg.file_path() == str(rig_yaml_path)
        assert dlg.result() == QDialog.DialogCode.Accepted
    finally:
        dlg.done(0)


def test_calib_rig_bulk_registry_loads_detects_and_anchors(qapp, fake_conn, rig_yaml_path, monkeypatch) -> None:
    from app.setup.page_extrinsics import _CalibRigDialog

    definition_id = import_marker_body(fake_conn, rig_yaml_path, name="test-rig")
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        def fake_exec(self):
            self._registry_id = definition_id
            row = fake_conn.execute(
                "SELECT yaml_content FROM marker_body_definitions WHERE id = ?", (definition_id,)
            ).fetchone()
            self._registry_yaml = row[0]
            self._result_kind = _CalibRigDialog.RESULT_REGISTRY
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(_CalibRigDialog, "exec", fake_exec)
        dlg._on_calib_rig_bulk()

        assert dlg._rig_config is not None
        assert dlg._rig_config.rig_id == "test-rig"
        assert dlg._rig_anchored  # marker "3" visible in cam_A -> auto-anchors
    finally:
        dlg.done(0)


def test_calib_rig_bulk_file_loads_detects_and_anchors(qapp, fake_conn, rig_yaml_path, monkeypatch) -> None:
    from app.setup.page_extrinsics import _CalibRigDialog

    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        def fake_exec(self):
            self._file_path = str(rig_yaml_path)
            self._result_kind = _CalibRigDialog.RESULT_FILE
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(_CalibRigDialog, "exec", fake_exec)
        dlg._on_calib_rig_bulk()

        assert dlg._rig_config is not None
        assert dlg._rig_config.rig_id == "test-rig"
        assert dlg._rig_anchored
    finally:
        dlg.done(0)


def test_calib_rig_bulk_cancelled_changes_nothing(qapp, fake_conn, monkeypatch) -> None:
    from app.setup.page_extrinsics import _CalibRigDialog

    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        monkeypatch.setattr(_CalibRigDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
        dlg._on_calib_rig_bulk()

        assert dlg._rig_config is None
        assert not dlg._rig_anchored
    finally:
        dlg.done(0)

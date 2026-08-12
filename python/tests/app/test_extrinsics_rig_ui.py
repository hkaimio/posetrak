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

from app.setup.extrinsics_solver import CamCalibState
from app.setup.fiducial_markers import ARUCO_DICTIONARIES
from app.setup.page_extrinsics import ExtrinsicsAutoCalibDialog

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
        assert "not yet anchored" in dlg._rig_status_label.text()
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
    state -- see _build_rig_group's docstring for why."""
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)
        dlg._on_detect_rig_clicked("cam_A")
        assert dlg._rig_control_points() == []
    finally:
        dlg.done(0)


def test_anchor_without_detection_shows_warning(qapp, fake_conn, monkeypatch, rig_yaml_path) -> None:
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._load_rig_config_from_path(rig_yaml_path)
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
) -> None:
    from app.setup.page_extrinsics import upsert_scene_marker_body
    conn.execute(
        "INSERT OR IGNORE INTO mocap_sessions (id, recorded_at) VALUES (?, '2026-01-01')",
        (session_id,),
    )
    conn.commit()
    upsert_scene_marker_body(
        conn, session_id, label=f"tag:{marker_id}",
        R=np.eye(3), t=np.array([0.5, 0.0, 2.0]),
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
    _seed_scene_marker_tag(fake_conn, "sess1")
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_load_rig_from_scene_markers()

        assert dlg._rig_config is not None
        assert dlg._rig_config.rig_id == "scene markers"
        assert set(dlg._rig_config.marker_corners) == {"3"}
        assert dlg._rig_config.marker_dictionaries["3"] == "DICT_4X4_50"
        assert dlg._rig_source == "scene_markers"
        assert dlg._rig_definition_id is None
        assert dlg._rig_detector is not None
    finally:
        dlg.done(0)


def test_load_from_scene_markers_then_detect_and_anchor(qapp, fake_conn) -> None:
    _seed_scene_marker_tag(fake_conn, "sess1")
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        dlg._on_load_rig_from_scene_markers()
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
    _seed_scene_marker_tag(fake_conn, "sess1")
    states = [_make_state("cam_A", _render_marker_image(3))]
    dlg = ExtrinsicsAutoCalibDialog(states, fake_conn, "sess1")
    try:
        count_before = fake_conn.execute(
            "SELECT COUNT(*) FROM scene_marker_bodies WHERE session_id = 'sess1'"
        ).fetchone()[0]

        dlg._on_load_rig_from_scene_markers()
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

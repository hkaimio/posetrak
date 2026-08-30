# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for phase 3 of the config-improvements design doc: the tracker-
config-editor extraction (TrackerConfigWidget) and the trial/capture
"Default tracker config" row (build_default_config_row/DefaultConfigDialog).

Pure widget-construction and DB-state assertions -- no display needed
(qapp fixture forces the offscreen Qt platform, see conftest.py). The
interactive walkthrough CLAUDE.md requires for genuinely new UI is still
separate, out of scope for source-only work.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from posetrak.db.db import create_session
from posetrak.db.manage_config import (
    BASELINE_CONFIG_ID,
    create_config_from_toml,
    set_default_tracker_config,
)


def _make_capture_and_trial(conn: sqlite3.Connection) -> tuple[str, str]:
    conn.execute(
        "INSERT INTO mocap_sessions (id, recorded_at) VALUES ('sess1', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO captures (id, session_id, capture_number) VALUES ('cap1', 'sess1', 1)"
    )
    conn.execute(
        "INSERT INTO trials (id, capture_id, name) VALUES ('trial1', 'cap1', 'take 1')"
    )
    conn.commit()
    return "cap1", "trial1"


@pytest.fixture()
def session_db(tmp_path: Path):
    conn = create_session(tmp_path / "session.db")
    yield conn
    conn.close()


def _write_toml(tmp_path: Path, process_noise_std: float = 0.1) -> Path:
    p = tmp_path / "cfg.toml"
    p.write_text(f"[tracking]\nprocess_noise_std = {process_noise_std}\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# TrackerConfigWidget (extracted from RunTrackerWidget)
# ---------------------------------------------------------------------------


def test_tracker_config_widget_constructs_standalone(qapp) -> None:
    from app.pose.run_tracker import TrackerConfigWidget

    w = TrackerConfigWidget()
    assert w._config_tabs.count() == 8
    assert w.loaded_config_id is None


def test_tracker_config_widget_collect_apply_round_trip(qapp, session_db) -> None:
    from app.pose.run_tracker import TrackerConfigWidget
    from posetrak.db.manage_config import edit_config, seed_baseline_tracker_config

    # session DBs aren't seeded with the baseline config at creation (unlike
    # registries) -- see the design doc's self-containment note.
    seed_baseline_tracker_config(session_db)

    w = TrackerConfigWidget()
    w.set_connection(session_db)
    w._proc_noise_std.setValue(0.42)
    w._use_relative.setChecked(True)
    w._relative_min_conf.setValue(0.6)

    new_id = edit_config(
        session_db, BASELINE_CONFIG_ID, is_named=True, name="rt-test",
        **w.collect_overrides(),
    )
    row = session_db.execute(
        "SELECT * FROM tracker_configs WHERE id = ?", (new_id,)
    ).fetchone()
    assert row["process_noise_std"] == pytest.approx(0.42)
    assert row["use_relative_observations"] == 1

    w2 = TrackerConfigWidget()
    w2.load_config_row(new_id, "rt-test", row)
    assert w2._proc_noise_std.value() == pytest.approx(0.42)
    assert w2._use_relative.isChecked() is True
    assert w2.loaded_config_id == new_id
    assert w2.loaded_config_name == "rt-test"


def test_collect_overrides_velocity_cameras_explicitly_cleared(qapp, session_db) -> None:
    """Unchecking every velocity-mode camera must produce an explicit empty
    list override, not None -- edit_config() treats None as 'keep the source
    row's value', so a bare None here would silently keep inheriting whatever
    velocity_mode_camera_ids the *previous* config already had instead of
    clearing it. Confirmed live, 2026-08-23: Harri unchecked a camera's
    velocity-mode checkbox, re-ran tracking twice, and both runs still showed
    it enabled."""
    from app.pose.run_tracker import TrackerConfigWidget
    from posetrak.db.manage_config import edit_config, seed_baseline_tracker_config

    seed_baseline_tracker_config(session_db)

    # A parent config with velocity mode already on for camera index 2.
    parent_id = edit_config(
        session_db, BASELINE_CONFIG_ID, is_named=True, name="parent-with-velocity",
        velocity_mode_camera_ids=[2],
    )
    parent_row = session_db.execute(
        "SELECT * FROM tracker_configs WHERE id = ?", (parent_id,)
    ).fetchone()

    w = TrackerConfigWidget()
    w.set_connection(session_db)
    w.load_config_row(parent_id, "parent-with-velocity", parent_row)
    assert w._velocity_cam_indices == {2}

    # Simulate the user opening "Velocity mode cameras" and unchecking the
    # only selected one.
    w._velocity_cam_indices = set()

    new_id = edit_config(session_db, parent_id, **w.collect_overrides())
    row = session_db.execute(
        "SELECT velocity_mode_camera_ids FROM tracker_configs WHERE id = ?", (new_id,)
    ).fetchone()
    assert json.loads(row["velocity_mode_camera_ids"]) == []


def test_collect_overrides_nis_feedback_explicitly_disabled(qapp, session_db) -> None:
    """Same None-means-inherit trap as velocity cameras, for the other
    checkbox-gated list field with no separate always-explicit enable
    column."""
    from app.pose.run_tracker import TrackerConfigWidget
    from posetrak.db.manage_config import edit_config, seed_baseline_tracker_config

    seed_baseline_tracker_config(session_db)

    w = TrackerConfigWidget()
    w.set_connection(session_db)
    w._nis_feedback_enabled.setChecked(True)
    parent_id = edit_config(
        session_db, BASELINE_CONFIG_ID, is_named=True, name="parent-with-nis",
        **w.collect_overrides(),
    )
    parent_row = session_db.execute(
        "SELECT * FROM tracker_configs WHERE id = ?", (parent_id,)
    ).fetchone()
    assert json.loads(parent_row["nis_feedback_scopes"])  # non-empty

    w2 = TrackerConfigWidget()
    w2.set_connection(session_db)
    w2.load_config_row(parent_id, "parent-with-nis", parent_row)
    assert w2._nis_feedback_enabled.isChecked() is True
    w2._nis_feedback_enabled.setChecked(False)

    new_id = edit_config(session_db, parent_id, **w2.collect_overrides())
    row = session_db.execute(
        "SELECT nis_feedback_scopes FROM tracker_configs WHERE id = ?", (new_id,)
    ).fetchone()
    assert json.loads(row["nis_feedback_scopes"]) == []


# ---------------------------------------------------------------------------
# DefaultConfigDialog
# ---------------------------------------------------------------------------


def test_default_config_dialog_requires_exactly_one_scope(qapp, session_db) -> None:
    from app.pose.run_tracker import DefaultConfigDialog

    with pytest.raises(ValueError, match="exactly one"):
        DefaultConfigDialog(session_db)
    with pytest.raises(ValueError, match="exactly one"):
        DefaultConfigDialog(session_db, trial_id="t1", capture_id="c1")


def test_default_config_dialog_loads_resolved_baseline(qapp, session_db) -> None:
    from app.pose.run_tracker import DefaultConfigDialog

    _capture_id, trial_id = _make_capture_and_trial(session_db)
    dlg = DefaultConfigDialog(session_db, trial_id=trial_id)
    assert dlg._config_widget.loaded_config_id == BASELINE_CONFIG_ID


def test_default_config_dialog_set_as_default_repoints_only_its_own_scope(
    qapp, session_db, tmp_path: Path
) -> None:
    from app.pose.run_tracker import DefaultConfigDialog

    capture_id, trial_id = _make_capture_and_trial(session_db)
    dlg = DefaultConfigDialog(session_db, trial_id=trial_id)
    dlg._config_widget._proc_noise_std.setValue(0.77)
    dlg._set_as_default()

    trial_row = session_db.execute(
        "SELECT default_tracker_config_id FROM trials WHERE id = ?", (trial_id,)
    ).fetchone()
    capture_row = session_db.execute(
        "SELECT default_tracker_config_id FROM captures WHERE id = ?", (capture_id,)
    ).fetchone()
    assert trial_row["default_tracker_config_id"] is not None
    assert capture_row["default_tracker_config_id"] is None

    new_cfg = session_db.execute(
        "SELECT process_noise_std, is_named FROM tracker_configs WHERE id = ?",
        (trial_row["default_tracker_config_id"],),
    ).fetchone()
    assert new_cfg["process_noise_std"] == pytest.approx(0.77)
    assert new_cfg["is_named"] == 0  # "Set as default" is not a named save

    # Re-opening resolves to the newly-set config, not the baseline again.
    dlg2 = DefaultConfigDialog(session_db, trial_id=trial_id)
    assert dlg2._config_widget.loaded_config_id == trial_row["default_tracker_config_id"]
    assert dlg2._config_widget._proc_noise_std.value() == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# build_default_config_row
# ---------------------------------------------------------------------------


def test_build_default_config_row_shows_own_default(qapp, session_db, tmp_path: Path) -> None:
    from app.pose.run_tracker import build_default_config_row
    from PySide6.QtWidgets import QLabel

    capture_id, _trial_id = _make_capture_and_trial(session_db)
    toml_path = _write_toml(tmp_path)
    cfg_id = create_config_from_toml(session_db, "my-capture-config", toml_path)
    set_default_tracker_config(session_db, cfg_id, capture_id=capture_id)

    row_widget = build_default_config_row(session_db, capture_id=capture_id)
    status_label = row_widget.findChildren(QLabel)[1]
    assert status_label.text() == "my-capture-config"


def test_build_default_config_row_shows_inherited_from_capture(
    qapp, session_db, tmp_path: Path
) -> None:
    from app.pose.run_tracker import build_default_config_row
    from PySide6.QtWidgets import QLabel

    capture_id, trial_id = _make_capture_and_trial(session_db)
    toml_path = _write_toml(tmp_path)
    cfg_id = create_config_from_toml(session_db, "my-capture-config", toml_path)
    set_default_tracker_config(session_db, cfg_id, capture_id=capture_id)

    row_widget = build_default_config_row(session_db, trial_id=trial_id)
    status_label = row_widget.findChildren(QLabel)[1]
    assert status_label.text() == "my-capture-config (inherited)"


def test_build_default_config_row_falls_back_to_baseline(qapp, session_db) -> None:
    from app.pose.run_tracker import build_default_config_row
    from PySide6.QtWidgets import QLabel

    _capture_id, trial_id = _make_capture_and_trial(session_db)
    row_widget = build_default_config_row(session_db, trial_id=trial_id)
    status_label = row_widget.findChildren(QLabel)[1]
    assert status_label.text() == "(factory defaults) (inherited)"


# ---------------------------------------------------------------------------
# Live-review fixes, 2026-07-24: status-label dirty tracking, stage-override
# load/save round trip, DefaultConfigDialog stage discovery, RunTrackerWidget
# auto-loading a trial's default.
# ---------------------------------------------------------------------------

_SKELETON_WITH_HANDS = """
name: test
groups:
  - name: main
    joints: [hips]
    markers: []
  - name: HandL
    joints: [hand.L]
    markers: []
    freeflyer_joint: forearm.L
    ref_marker: MRK-wrist.L
"""


def test_status_label_shows_bare_name_when_unmodified(
    qapp, session_db, tmp_path: Path
) -> None:
    from app.pose.run_tracker import TrackerConfigWidget
    from posetrak.db.manage_config import create_config_from_toml, seed_baseline_tracker_config

    seed_baseline_tracker_config(session_db)
    cfg_id = create_config_from_toml(session_db, "my-cfg", _write_toml(tmp_path))
    row = session_db.execute("SELECT * FROM tracker_configs WHERE id = ?", (cfg_id,)).fetchone()

    w = TrackerConfigWidget()
    w.set_connection(session_db)
    w.load_config_row(cfg_id, "my-cfg", row)
    assert w._config_status_label.text() == "my-cfg"

    w._proc_noise_std.setValue(w._proc_noise_std.value() + 1.0)
    assert w._config_status_label.text() == "my-cfg (modified)"


def test_status_label_unnamed_snapshot_when_loaded_without_a_name(qapp, session_db) -> None:
    from app.pose.run_tracker import TrackerConfigWidget
    from posetrak.db.manage_config import edit_config, seed_baseline_tracker_config

    seed_baseline_tracker_config(session_db)
    new_id = edit_config(session_db, BASELINE_CONFIG_ID, is_named=False, process_noise_std=0.3)
    row = session_db.execute("SELECT * FROM tracker_configs WHERE id = ?", (new_id,)).fetchone()

    w = TrackerConfigWidget()
    w.load_config_row(new_id, None, row)
    assert w._config_status_label.text() == "(unnamed snapshot)"


def test_load_config_row_restores_stage_overrides(qapp, session_db) -> None:
    """A config with existing tracker_config_stages rows must show up as
    enabled, with its per-stage override values, when loaded back --
    otherwise editing (e.g.) a trial's already-hierarchical default starts
    from a blank table and silently drops the stage selection on save.
    """
    from app.pose.run_tracker import TrackerConfigWidget
    from posetrak.db.manage_config import edit_config, seed_baseline_tracker_config

    seed_baseline_tracker_config(session_db)
    session_db.execute(
        "INSERT INTO skeletons (id, name, yaml_content, created_at) "
        "VALUES ('skel1', 'test', ?, '2026-01-01')",
        (_SKELETON_WITH_HANDS,),
    )
    session_db.commit()

    cfg_id = edit_config(session_db, BASELINE_CONFIG_ID, is_named=True, name="hier-cfg")
    session_db.execute(
        "INSERT INTO tracker_config_stages (tracker_config_id, group_name, process_noise_std) "
        "VALUES (?, 'HandL', 0.25)",
        (cfg_id,),
    )
    session_db.commit()
    row = session_db.execute("SELECT * FROM tracker_configs WHERE id = ?", (cfg_id,)).fetchone()

    w = TrackerConfigWidget()
    w.set_connection(session_db)
    w.set_skeleton_ids(["skel1"])
    w.load_config_row(cfg_id, "hier-cfg", row)

    assert w._hierarchical_enabled.isChecked() is True
    assert w._stage_table.rowCount() == 1
    assert w._stage_table.item(0, 1).text() == "HandL"
    chk = w._stage_table.cellWidget(0, 0).findChild(type(w._hierarchical_enabled))
    assert chk.isChecked() is True
    proc_noise_edit = w._stage_table.cellWidget(0, 2)
    assert proc_noise_edit.text() == "0.25"

    # Loaded-and-untouched: label shows the bare name, not "(modified)",
    # and re-syncing (as _start_tracking()/_set_as_default() would) must
    # not have silently dropped the stage row.
    assert w._config_status_label.text() == "hier-cfg"
    w.sync_stage_overrides(cfg_id)
    stage_rows = session_db.execute(
        "SELECT group_name, process_noise_std FROM tracker_config_stages "
        "WHERE tracker_config_id = ?",
        (cfg_id,),
    ).fetchall()
    assert len(stage_rows) == 1
    assert stage_rows[0]["group_name"] == "HandL"
    assert stage_rows[0]["process_noise_std"] == pytest.approx(0.25)


def test_default_config_dialog_discovers_stages_from_all_session_skeletons(
    qapp, session_db
) -> None:
    """DefaultConfigDialog isn't tied to one person's skeleton choice, so it
    must offer every session skeleton's eligible stage groups -- otherwise
    the Hierarchical solver tab has nothing to discover and a trial/capture
    default can never have stages configured from this dialog at all.
    """
    from app.pose.run_tracker import DefaultConfigDialog

    _capture_id, trial_id = _make_capture_and_trial(session_db)
    session_db.execute(
        "INSERT INTO skeletons (id, name, yaml_content, created_at) "
        "VALUES ('skel1', 'test', ?, '2026-01-01')",
        (_SKELETON_WITH_HANDS,),
    )
    session_db.commit()

    dlg = DefaultConfigDialog(session_db, trial_id=trial_id)
    assert dlg._config_widget._skeleton_ids == ["skel1"]
    dlg._config_widget._hierarchical_enabled.setChecked(True)
    dlg._config_widget._refresh_stage_table()
    assert dlg._config_widget._stage_table.rowCount() == 1
    assert dlg._config_widget._stage_table.item(0, 1).text() == "HandL"


# ---------------------------------------------------------------------------
# RunTrackerWidget auto-loading a trial's default config
# ---------------------------------------------------------------------------


def test_run_tracker_widget_loads_trial_default_on_trial_change(qapp, session_db) -> None:
    from app.pose.run_tracker import RunTrackerWidget
    from posetrak.db.manage_config import edit_config, seed_baseline_tracker_config, \
        set_default_tracker_config

    seed_baseline_tracker_config(session_db)
    _capture_id, trial_id = _make_capture_and_trial(session_db)
    tuned_id = edit_config(
        session_db, BASELINE_CONFIG_ID, is_named=True, name="tuned", process_noise_std=0.55,
    )
    set_default_tracker_config(session_db, tuned_id, trial_id=trial_id)

    w = RunTrackerWidget()
    w.set_session(session_db, str(session_db.execute("PRAGMA database_list").fetchone()[2]))
    w._load_trial_default_config(trial_id)

    assert w._config_widget.loaded_config_id == tuned_id
    assert w._config_widget.loaded_config_name == "tuned"
    assert w._config_widget._proc_noise_std.value() == pytest.approx(0.55)


def test_run_tracker_widget_set_trial_default_checkbox_repoints_trial(qapp, session_db) -> None:
    """_maybe_set_trial_default() -- called by _start_tracking() right after
    a run's config snapshot is created -- should repoint the trial's
    default_tracker_config_id only when the checkbox is checked."""
    from app.pose.run_tracker import RunTrackerWidget
    from posetrak.db.manage_config import edit_config, seed_baseline_tracker_config

    seed_baseline_tracker_config(session_db)
    _capture_id, trial_id = _make_capture_and_trial(session_db)
    config_id = edit_config(session_db, BASELINE_CONFIG_ID, is_named=False)

    w = RunTrackerWidget()
    w.set_session(session_db, str(session_db.execute("PRAGMA database_list").fetchone()[2]))
    # _refresh_trials()'s own query requires a detection run + sequence to
    # list a trial (irrelevant here) -- add it directly instead so
    # _current_trial_id() resolves to trial_id.
    w._trial_combo.addItem("trial1", {"trial_id": trial_id})
    w._trial_combo.setCurrentIndex(w._trial_combo.count() - 1)

    # Unchecked: no-op.
    w._set_trial_default_chk.setChecked(False)
    w._maybe_set_trial_default(config_id)
    trial_row = session_db.execute(
        "SELECT default_tracker_config_id FROM trials WHERE id = ?", (trial_id,)
    ).fetchone()
    assert trial_row["default_tracker_config_id"] is None

    # Checked: repoints.
    w._set_trial_default_chk.setChecked(True)
    w._maybe_set_trial_default(config_id)
    trial_row = session_db.execute(
        "SELECT default_tracker_config_id FROM trials WHERE id = ?", (trial_id,)
    ).fetchone()
    assert trial_row["default_tracker_config_id"] == config_id


def test_person_row_skeleton_change_updates_config_widget_skeleton_ids(
    qapp, session_db
) -> None:
    """A person row's Skeleton combo previously had no change signal wired
    at all -- picking a different (e.g. hierarchical) skeleton never
    reached the config widget's set_skeleton_ids(), so "Refresh stages"
    kept discovering groups for whatever skeleton happened to be first in
    the combo at row-creation time, not the one actually selected.
    """
    from app.pose.run_tracker import RunTrackerWidget

    session_db.execute(
        "INSERT INTO skeletons (id, name, yaml_content, created_at) "
        "VALUES ('skel-plain', 'plain', 'name: plain\ngroups: []', '2026-01-01')"
    )
    session_db.execute(
        "INSERT INTO skeletons (id, name, yaml_content, created_at) "
        "VALUES ('skel-hier', 'hier', ?, '2026-01-01')",
        (_SKELETON_WITH_HANDS,),
    )
    session_db.commit()

    w = RunTrackerWidget()
    w.set_session(session_db, str(session_db.execute("PRAGMA database_list").fetchone()[2]))
    # Bypass the detection-run/sequence scaffolding _person_names_for_trial()
    # would otherwise require -- irrelevant to the combo-wiring bug itself.
    w._person_names_for_trial = lambda trial_id: ["alice"]
    row = w._insert_person_row("trial-x", used_names=set(), removable=False)
    assert row is not None

    combo = w._people_table.cellWidget(row, 2)
    # "hier" sorts before "plain" (_refresh_skeletons() orders by name), so
    # the combo already defaults to skel-hier -- switch to skel-plain first
    # so the later switch back to skel-hier is a real index change that
    # actually needs the currentIndexChanged wiring to propagate, not one
    # that would trivially "pass" from the row-creation default alone.
    plain_index = combo.findData("skel-plain")
    assert plain_index >= 0
    combo.setCurrentIndex(plain_index)
    assert w._config_widget._skeleton_ids == ["skel-plain"]

    hier_index = combo.findData("skel-hier")
    assert hier_index >= 0
    combo.setCurrentIndex(hier_index)
    assert w._config_widget._skeleton_ids == ["skel-hier"]

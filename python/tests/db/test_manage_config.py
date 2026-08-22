# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/db/manage_config.py."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


from posetrak.db.manage_config import (
    BASELINE_CONFIG_ID,
    BASELINE_CONFIG_NAME,
    create_config_from_toml,
    edit_config,
    list_configs,
    name_existing_config,
    resolve_default_tracker_config,
    seed_baseline_tracker_config,
    set_default_tracker_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_TOML = """\
[tracking]
process_noise_std = 0.15
measurement_noise_std = 20.0
outlier_threshold = 4.0

[tracking.ukf]
alpha = 0.1
beta = 2.0
kappa = 0.0

[tracking.initialization]
ik_max_iterations = 1000
ik_tolerance = 0.02
init_position_std = 1.0
init_orientation_std = 1.0
init_joint_std = 0.1
init_velocity_std = 1.0
min_cameras_for_init = 2

[processing]
tracker_fps = 120.0
"""


def _write_toml(tmp_path: Path, content: str = _SAMPLE_TOML) -> Path:
    p = tmp_path / "regress.toml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# create_config_from_toml
# ---------------------------------------------------------------------------


def test_create_config_from_toml_returns_id(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """create_config_from_toml() should return a non-empty UUID string."""
    toml_path = _write_toml(tmp_path)
    config_id = create_config_from_toml(registry_db, "test_config", toml_path)
    assert config_id
    row = registry_db.execute(
        "SELECT id FROM tracker_configs WHERE id = ?", (config_id,)
    ).fetchone()
    assert row is not None


def test_create_config_from_toml_reads_tracking_params(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """[tracking] parameters are read correctly."""
    toml_path = _write_toml(tmp_path)
    config_id = create_config_from_toml(registry_db, "test_config", toml_path)
    row = registry_db.execute(
        "SELECT process_noise_std, measurement_noise_std, outlier_threshold "
        "FROM tracker_configs WHERE id = ?",
        (config_id,),
    ).fetchone()
    assert row["process_noise_std"] == pytest.approx(0.15)
    assert row["measurement_noise_std"] == pytest.approx(20.0)
    assert row["outlier_threshold"] == pytest.approx(4.0)


def test_create_config_from_toml_reads_ukf_params(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """[tracking.ukf] parameters are read correctly."""
    toml_path = _write_toml(tmp_path)
    config_id = create_config_from_toml(registry_db, "test_config", toml_path)
    row = registry_db.execute(
        "SELECT alpha, beta, kappa FROM tracker_configs WHERE id = ?",
        (config_id,),
    ).fetchone()
    assert row["alpha"] == pytest.approx(0.1)
    assert row["beta"] == pytest.approx(2.0)
    assert row["kappa"] == pytest.approx(0.0)


def test_create_config_from_toml_reads_init_params(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """[tracking.initialization] and [processing] parameters are read correctly."""
    toml_path = _write_toml(tmp_path)
    config_id = create_config_from_toml(registry_db, "test_config", toml_path)
    row = registry_db.execute(
        "SELECT ik_max_iterations, ik_tolerance, init_position_std, "
        "init_orientation_std, init_joint_std, init_velocity_std, "
        "min_cameras_for_init, tracker_fps "
        "FROM tracker_configs WHERE id = ?",
        (config_id,),
    ).fetchone()
    assert row["ik_max_iterations"] == 1000
    assert row["ik_tolerance"] == pytest.approx(0.02)
    assert row["init_position_std"] == pytest.approx(1.0)
    assert row["init_orientation_std"] == pytest.approx(1.0)
    assert row["init_joint_std"] == pytest.approx(0.1)
    assert row["init_velocity_std"] == pytest.approx(1.0)
    assert row["min_cameras_for_init"] == 2
    assert row["tracker_fps"] == pytest.approx(120.0)


def test_create_config_from_toml_missing_section_stores_null(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Parameters absent from TOML are stored as NULL."""
    toml_path = tmp_path / "minimal.toml"
    toml_path.write_text("[tracking]\nprocess_noise_std = 0.1\n", encoding="utf-8")
    config_id = create_config_from_toml(registry_db, "minimal", toml_path)
    row = registry_db.execute(
        "SELECT alpha, tracker_fps FROM tracker_configs WHERE id = ?",
        (config_id,),
    ).fetchone()
    assert row["alpha"] is None
    assert row["tracker_fps"] is None


# ---------------------------------------------------------------------------
# edit_config
# ---------------------------------------------------------------------------


def test_edit_config_creates_new_row_with_parent_id(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """edit_config() creates a new row whose parent_id points to the source."""
    toml_path = _write_toml(tmp_path)
    count_before = registry_db.execute("SELECT COUNT(*) FROM tracker_configs").fetchone()[0]
    orig_id = create_config_from_toml(registry_db, "base", toml_path)
    new_id = edit_config(registry_db, orig_id, alpha=0.5)
    assert new_id != orig_id
    row = registry_db.execute(
        "SELECT parent_id FROM tracker_configs WHERE id = ?", (new_id,)
    ).fetchone()
    assert row["parent_id"] == orig_id
    count = registry_db.execute("SELECT COUNT(*) FROM tracker_configs").fetchone()[0]
    assert count == count_before + 2


def test_edit_config_carries_forward_unchanged_fields(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Fields not supplied to edit_config() are carried forward from the original row."""
    toml_path = _write_toml(tmp_path)
    orig_id = create_config_from_toml(registry_db, "base", toml_path)
    new_id = edit_config(registry_db, orig_id, alpha=0.5)
    row = registry_db.execute(
        "SELECT beta, tracker_fps, process_noise_std FROM tracker_configs WHERE id = ?",
        (new_id,),
    ).fetchone()
    assert row["beta"] == pytest.approx(2.0)
    assert row["tracker_fps"] == pytest.approx(120.0)
    assert row["process_noise_std"] == pytest.approx(0.15)


def test_edit_config_overrides_supplied_fields(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Supplied kwargs override the copied values in the new row."""
    toml_path = _write_toml(tmp_path)
    orig_id = create_config_from_toml(registry_db, "base", toml_path)
    new_id = edit_config(
        registry_db, orig_id,
        alpha=0.9,
        process_noise_std=0.05,
        min_cameras_for_init=3,
    )
    row = registry_db.execute(
        "SELECT alpha, process_noise_std, min_cameras_for_init "
        "FROM tracker_configs WHERE id = ?",
        (new_id,),
    ).fetchone()
    assert row["alpha"] == pytest.approx(0.9)
    assert row["process_noise_std"] == pytest.approx(0.05)
    assert row["min_cameras_for_init"] == 3


def test_edit_config_invalid_id_raises(registry_db: sqlite3.Connection) -> None:
    """edit_config() raises ValueError when the supplied config_id does not exist."""
    with pytest.raises(ValueError, match="tracker_configs"):
        edit_config(registry_db, "00000000-0000-0000-0000-000000000000", alpha=0.1)


def test_edit_config_defaults_to_unnamed_even_from_a_named_source(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """edit_config()'s copy is unnamed by default, regardless of whether the
    source row is a named template -- a name is never silently inherited."""
    toml_path = _write_toml(tmp_path)
    orig_id = create_config_from_toml(registry_db, "base", toml_path)  # is_named=True by default
    orig_row = registry_db.execute(
        "SELECT is_named FROM tracker_configs WHERE id = ?", (orig_id,)
    ).fetchone()
    assert orig_row["is_named"] == 1

    new_id = edit_config(registry_db, orig_id, alpha=0.5)
    new_row = registry_db.execute(
        "SELECT is_named FROM tracker_configs WHERE id = ?", (new_id,)
    ).fetchone()
    assert new_row["is_named"] == 0


def test_edit_config_is_named_true_produces_named_row(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Passing is_named=True explicitly (e.g. a "Save"/"Save as..." action)
    produces a named, browsable row."""
    toml_path = _write_toml(tmp_path)
    orig_id = create_config_from_toml(registry_db, "base", toml_path)
    new_id = edit_config(registry_db, orig_id, alpha=0.5, is_named=True)
    row = registry_db.execute(
        "SELECT is_named FROM tracker_configs WHERE id = ?", (new_id,)
    ).fetchone()
    assert row["is_named"] == 1


def test_edit_config_name_override_produces_save_as_semantics(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Passing name=... gives the new row a different name than the source
    -- the "Save as..." case, distinct from "Save" (no name kwarg, keeps
    the source's own name)."""
    toml_path = _write_toml(tmp_path)
    orig_id = create_config_from_toml(registry_db, "base", toml_path)

    same_name_id = edit_config(registry_db, orig_id, alpha=0.5, is_named=True)
    same_name_row = registry_db.execute(
        "SELECT name FROM tracker_configs WHERE id = ?", (same_name_id,)
    ).fetchone()
    assert same_name_row["name"] == "base"

    renamed_id = edit_config(registry_db, orig_id, alpha=0.5, is_named=True, name="tuned-base")
    renamed_row = registry_db.execute(
        "SELECT name FROM tracker_configs WHERE id = ?", (renamed_id,)
    ).fetchone()
    assert renamed_row["name"] == "tuned-base"


def test_create_config_from_toml_is_named_false_opt_out(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """create_config_from_toml(is_named=False) opts out of the default."""
    toml_path = _write_toml(tmp_path)
    config_id = create_config_from_toml(registry_db, "throwaway", toml_path, is_named=False)
    row = registry_db.execute(
        "SELECT is_named FROM tracker_configs WHERE id = ?", (config_id,)
    ).fetchone()
    assert row["is_named"] == 0


# ---------------------------------------------------------------------------
# seed_baseline_tracker_config
# ---------------------------------------------------------------------------


def test_seed_baseline_tracker_config_creates_named_row(
    registry_db: sqlite3.Connection,
) -> None:
    """create_registry() already seeds the baseline config (see conftest's
    registry_db fixture); confirm its shape directly."""
    row = registry_db.execute(
        "SELECT name, is_named, parent_id FROM tracker_configs WHERE id = ?",
        (BASELINE_CONFIG_ID,),
    ).fetchone()
    assert row is not None
    assert row["name"] == BASELINE_CONFIG_NAME
    assert row["is_named"] == 1
    assert row["parent_id"] is None


def test_seed_baseline_tracker_config_idempotent(registry_db: sqlite3.Connection) -> None:
    """Calling seed_baseline_tracker_config() again is a no-op, not a duplicate row."""
    seed_baseline_tracker_config(registry_db)
    seed_baseline_tracker_config(registry_db)
    count = registry_db.execute(
        "SELECT COUNT(*) FROM tracker_configs WHERE id = ?", (BASELINE_CONFIG_ID,)
    ).fetchone()[0]
    assert count == 1


def test_seed_baseline_tracker_config_has_real_values(
    registry_db: sqlite3.Connection,
) -> None:
    """The baseline row should be a usable starting point, not a wall of
    NULLs -- see BASELINE_CONFIG_VALUES for provenance."""
    row = registry_db.execute(
        "SELECT process_noise_std, measurement_noise_std, tracker_fps, "
        "pose_noise_std, use_relative_observations, velocity_mode_camera_ids "
        "FROM tracker_configs WHERE id = ?",
        (BASELINE_CONFIG_ID,),
    ).fetchone()
    assert row["process_noise_std"] == pytest.approx(0.3)
    assert row["measurement_noise_std"] == pytest.approx(25.0)
    assert row["tracker_fps"] == pytest.approx(120.0)
    assert row["pose_noise_std"] == pytest.approx(13.0)
    assert row["use_relative_observations"] == 1
    # Scene-specific (names a camera by index in one particular capture) --
    # deliberately not carried into the baseline. See BASELINE_CONFIG_VALUES.
    assert row["velocity_mode_camera_ids"] is None


def test_refresh_baseline_tracker_config_backfills_existing_null_row(
    registry_db: sqlite3.Connection,
) -> None:
    """A registry/session created before BASELINE_CONFIG_VALUES existed has
    an all-NULL baseline row forever (seed is INSERT OR IGNORE) unless
    explicitly refreshed."""
    from posetrak.db.manage_config import refresh_baseline_tracker_config

    registry_db.execute(
        "UPDATE tracker_configs SET process_noise_std = NULL, tracker_fps = NULL "
        "WHERE id = ?",
        (BASELINE_CONFIG_ID,),
    )
    registry_db.commit()

    refresh_baseline_tracker_config(registry_db)

    row = registry_db.execute(
        "SELECT process_noise_std, tracker_fps FROM tracker_configs WHERE id = ?",
        (BASELINE_CONFIG_ID,),
    ).fetchone()
    assert row["process_noise_std"] == pytest.approx(0.3)
    assert row["tracker_fps"] == pytest.approx(120.0)


def test_refresh_baseline_tracker_config_noop_if_missing(
    registry_db: sqlite3.Connection,
) -> None:
    """Safe to call even if the baseline row doesn't exist (matches seed's
    own idempotency), rather than raising."""
    from posetrak.db.manage_config import refresh_baseline_tracker_config

    registry_db.execute("DELETE FROM tracker_configs WHERE id = ?", (BASELINE_CONFIG_ID,))
    registry_db.commit()
    refresh_baseline_tracker_config(registry_db)  # must not raise
    count = registry_db.execute(
        "SELECT COUNT(*) FROM tracker_configs WHERE id = ?", (BASELINE_CONFIG_ID,)
    ).fetchone()[0]
    assert count == 0


# ---------------------------------------------------------------------------
# name_existing_config
# ---------------------------------------------------------------------------


def test_name_existing_config_names_in_place(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """name_existing_config() sets name/is_named on the SAME row -- no new
    row, no parent_id set, unlike edit_config()'s copy-on-write."""
    toml_path = _write_toml(tmp_path)
    config_id = create_config_from_toml(registry_db, "ui-run", toml_path, is_named=False)
    count_before = registry_db.execute("SELECT COUNT(*) FROM tracker_configs").fetchone()[0]

    name_existing_config(registry_db, config_id, "my-tuned-run")

    count_after = registry_db.execute("SELECT COUNT(*) FROM tracker_configs").fetchone()[0]
    assert count_after == count_before  # no new row created

    row = registry_db.execute(
        "SELECT name, is_named, parent_id FROM tracker_configs WHERE id = ?", (config_id,)
    ).fetchone()
    assert row["name"] == "my-tuned-run"
    assert row["is_named"] == 1
    assert row["parent_id"] is None


def test_name_existing_config_can_rename(registry_db: sqlite3.Connection, tmp_path: Path) -> None:
    """Calling it again overwrites the previous name."""
    toml_path = _write_toml(tmp_path)
    config_id = create_config_from_toml(registry_db, "base", toml_path, is_named=False)
    name_existing_config(registry_db, config_id, "first-name")
    name_existing_config(registry_db, config_id, "second-name")
    row = registry_db.execute(
        "SELECT name FROM tracker_configs WHERE id = ?", (config_id,)
    ).fetchone()
    assert row["name"] == "second-name"


def test_name_existing_config_invalid_id_raises(registry_db: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="tracker_configs"):
        name_existing_config(registry_db, "00000000-0000-0000-0000-000000000000", "x")


def test_edit_config_preserves_post_v21_columns(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Regression test for the column-set-completeness bug (config-improvements
    design doc, "Prerequisite fix"): edit_config() must carry forward columns
    added in migrations v22-v37, not just the original ~20-column subset it
    used to hardcode. Directly UPDATEs the source row (rather than routing
    scalar/JSON columns through create_config_from_toml(), which doesn't read
    these from TOML) so this test exercises edit_config() in isolation.
    """
    toml_path = _write_toml(tmp_path)
    orig_id = create_config_from_toml(registry_db, "base", toml_path)
    with registry_db:
        registry_db.execute(
            "UPDATE tracker_configs SET "
            "pose_noise_std = ?, use_relative_observations = ?, "
            "pose_reg_joint_names = ?, pose_reg_equal_split_noise_std = ?, "
            "cross_person_max_world_mm = ?, cross_person_max_n = ? "
            "WHERE id = ?",
            (12.5, 1, '["spine1", "spine2"]', 0.02, 150.0, 3, orig_id),
        )

    new_id = edit_config(registry_db, orig_id, alpha=0.9)  # unrelated override

    row = registry_db.execute(
        "SELECT pose_noise_std, use_relative_observations, pose_reg_joint_names, "
        "pose_reg_equal_split_noise_std, cross_person_max_world_mm, cross_person_max_n "
        "FROM tracker_configs WHERE id = ?",
        (new_id,),
    ).fetchone()
    assert row["pose_noise_std"] == pytest.approx(12.5)
    assert row["use_relative_observations"] == 1
    assert row["pose_reg_joint_names"] == '["spine1", "spine2"]'
    assert row["pose_reg_equal_split_noise_std"] == pytest.approx(0.02)
    assert row["cross_person_max_world_mm"] == pytest.approx(150.0)
    assert row["cross_person_max_n"] == 3


def test_edit_config_overrides_post_v21_column_with_list_value(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """A list/dict override value is JSON-encoded automatically, for any
    column -- not just the one (velocity_mode_camera_ids) the old hardcoded
    implementation special-cased."""
    toml_path = _write_toml(tmp_path)
    orig_id = create_config_from_toml(registry_db, "base", toml_path)
    new_id = edit_config(
        registry_db, orig_id,
        pose_reg_joint_names=["spine1", "spine2"],
        velocity_mode_camera_ids=[1, 2],
    )
    row = registry_db.execute(
        "SELECT pose_reg_joint_names, velocity_mode_camera_ids "
        "FROM tracker_configs WHERE id = ?",
        (new_id,),
    ).fetchone()
    assert json.loads(row["pose_reg_joint_names"]) == ["spine1", "spine2"]
    assert json.loads(row["velocity_mode_camera_ids"]) == [1, 2]


def test_edit_config_copies_stage_rows_forward(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """edit_config() must carry a hierarchical config's tracker_config_stages
    rows forward to the new row -- the old implementation didn't touch that
    table at all, silently dropping per-stage overrides on every edit."""
    toml_path = _write_toml(tmp_path)
    orig_id = create_config_from_toml(registry_db, "base", toml_path)
    with registry_db:
        registry_db.execute(
            "INSERT INTO tracker_config_stages "
            "(tracker_config_id, group_name, process_noise_std, init_joint_std) "
            "VALUES (?, ?, ?, ?)",
            (orig_id, "HandL", 0.3, 0.02),
        )
        registry_db.execute(
            "INSERT INTO tracker_config_stages "
            "(tracker_config_id, group_name, process_noise_std, init_joint_std) "
            "VALUES (?, ?, ?, ?)",
            (orig_id, "HandR", 0.3, 0.02),
        )

    new_id = edit_config(registry_db, orig_id, alpha=0.9)  # unrelated override

    stage_rows = registry_db.execute(
        "SELECT group_name, process_noise_std, init_joint_std "
        "FROM tracker_config_stages WHERE tracker_config_id = ? ORDER BY group_name",
        (new_id,),
    ).fetchall()
    assert len(stage_rows) == 2
    assert stage_rows[0]["group_name"] == "HandL"
    assert stage_rows[0]["process_noise_std"] == pytest.approx(0.3)
    assert stage_rows[0]["init_joint_std"] == pytest.approx(0.02)
    assert stage_rows[1]["group_name"] == "HandR"

    # Original config's own stage rows are untouched.
    orig_stage_rows = registry_db.execute(
        "SELECT group_name FROM tracker_config_stages WHERE tracker_config_id = ?",
        (orig_id,),
    ).fetchall()
    assert len(orig_stage_rows) == 2


def test_create_config_from_toml_reads_post_v21_field(
    registry_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """create_config_from_toml() reads tuning fields beyond the original
    ~20-column subset (here: pose_noise_std, added in migration v22) via the
    same generic PRAGMA-driven column list edit_config() uses."""
    toml_path = tmp_path / "with_pose_noise.toml"
    toml_path.write_text(
        "[tracking]\nprocess_noise_std = 0.1\npose_noise_std = 7.5\n",
        encoding="utf-8",
    )
    config_id = create_config_from_toml(registry_db, "test_config", toml_path)
    row = registry_db.execute(
        "SELECT pose_noise_std FROM tracker_configs WHERE id = ?", (config_id,)
    ).fetchone()
    assert row["pose_noise_std"] == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# list_configs
# ---------------------------------------------------------------------------


def test_list_configs_empty(registry_db: sqlite3.Connection) -> None:
    """A fresh registry contains only the seeded baseline config."""
    rows = list_configs(registry_db)
    assert len(rows) == 1
    assert rows[0]["name"] == BASELINE_CONFIG_NAME


def test_list_configs_returns_all(registry_db: sqlite3.Connection, tmp_path: Path) -> None:
    """list_configs() returns all rows when no name filter is given (plus the
    seeded baseline config)."""
    toml_path = _write_toml(tmp_path)
    create_config_from_toml(registry_db, "config_a", toml_path)
    create_config_from_toml(registry_db, "config_b", toml_path)
    rows = list_configs(registry_db)
    assert len(rows) == 3


def test_list_configs_filter_by_name(registry_db: sqlite3.Connection, tmp_path: Path) -> None:
    """list_configs(name=...) returns only rows matching that name exactly."""
    toml_path = _write_toml(tmp_path)
    create_config_from_toml(registry_db, "alpha_run", toml_path)
    create_config_from_toml(registry_db, "beta_run", toml_path)
    rows = list_configs(registry_db, name="alpha_run")
    assert len(rows) == 1
    assert rows[0]["name"] == "alpha_run"
    rows_no_match = list_configs(registry_db, name="nonexistent")
    assert rows_no_match == []


# ---------------------------------------------------------------------------
# resolve_default_tracker_config / set_default_tracker_config
# ---------------------------------------------------------------------------


def _make_capture_and_trial(conn: sqlite3.Connection) -> tuple[str, str]:
    """Insert a minimal mocap_sessions -> captures -> trials chain; return
    (capture_id, trial_id)."""
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


def test_resolve_default_tracker_config_falls_back_to_baseline(
    session_db: sqlite3.Connection,
) -> None:
    """With no default set at either level, resolves to the baseline config,
    seeding it into the session DB on demand (session DBs aren't seeded at
    creation, unlike registries)."""
    capture_id, trial_id = _make_capture_and_trial(session_db)
    assert session_db.execute(
        "SELECT COUNT(*) FROM tracker_configs WHERE id = ?", (BASELINE_CONFIG_ID,)
    ).fetchone()[0] == 0

    resolved = resolve_default_tracker_config(session_db, trial_id=trial_id)
    assert resolved == BASELINE_CONFIG_ID
    assert session_db.execute(
        "SELECT COUNT(*) FROM tracker_configs WHERE id = ?", (BASELINE_CONFIG_ID,)
    ).fetchone()[0] == 1


def test_resolve_default_tracker_config_uses_trial_default(
    session_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """A trial's own default_tracker_config_id wins over its capture's."""
    capture_id, trial_id = _make_capture_and_trial(session_db)
    toml_path = _write_toml(tmp_path)
    trial_cfg = create_config_from_toml(session_db, "trial-specific", toml_path)
    capture_cfg = create_config_from_toml(session_db, "capture-specific", toml_path)
    set_default_tracker_config(session_db, capture_cfg, capture_id=capture_id)
    set_default_tracker_config(session_db, trial_cfg, trial_id=trial_id)

    assert resolve_default_tracker_config(session_db, trial_id=trial_id) == trial_cfg


def test_resolve_default_tracker_config_falls_through_to_capture(
    session_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """No trial-level default -> falls through to the capture's default."""
    capture_id, trial_id = _make_capture_and_trial(session_db)
    toml_path = _write_toml(tmp_path)
    capture_cfg = create_config_from_toml(session_db, "capture-specific", toml_path)
    set_default_tracker_config(session_db, capture_cfg, capture_id=capture_id)

    assert resolve_default_tracker_config(session_db, trial_id=trial_id) == capture_cfg


def test_resolve_default_tracker_config_unknown_trial_raises(
    session_db: sqlite3.Connection,
) -> None:
    with pytest.raises(ValueError, match="trial"):
        resolve_default_tracker_config(session_db, trial_id="nonexistent")


def test_resolve_default_tracker_config_requires_a_scope(
    session_db: sqlite3.Connection,
) -> None:
    with pytest.raises(ValueError, match="trial_id or capture_id"):
        resolve_default_tracker_config(session_db)


def test_set_default_tracker_config_repoints_only_its_own_scope(
    session_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Setting a trial's default never touches its capture's, and vice versa."""
    capture_id, trial_id = _make_capture_and_trial(session_db)
    toml_path = _write_toml(tmp_path)
    trial_cfg = create_config_from_toml(session_db, "trial-cfg", toml_path)

    set_default_tracker_config(session_db, trial_cfg, trial_id=trial_id)

    trial_row = session_db.execute(
        "SELECT default_tracker_config_id FROM trials WHERE id = ?", (trial_id,)
    ).fetchone()
    capture_row = session_db.execute(
        "SELECT default_tracker_config_id FROM captures WHERE id = ?", (capture_id,)
    ).fetchone()
    assert trial_row["default_tracker_config_id"] == trial_cfg
    assert capture_row["default_tracker_config_id"] is None


def test_set_default_tracker_config_requires_exactly_one_scope(
    session_db: sqlite3.Connection, tmp_path: Path
) -> None:
    capture_id, trial_id = _make_capture_and_trial(session_db)
    toml_path = _write_toml(tmp_path)
    cfg = create_config_from_toml(session_db, "cfg", toml_path)
    with pytest.raises(ValueError, match="exactly one"):
        set_default_tracker_config(session_db, cfg)
    with pytest.raises(ValueError, match="exactly one"):
        set_default_tracker_config(session_db, cfg, trial_id=trial_id, capture_id=capture_id)

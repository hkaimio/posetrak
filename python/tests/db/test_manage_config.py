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
    seed_baseline_tracker_config,
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

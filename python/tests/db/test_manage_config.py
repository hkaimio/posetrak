"""Tests for scripts/db/manage_config.py."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


from posetrak.db.manage_config import create_config_from_toml, edit_config, list_configs


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
    orig_id = create_config_from_toml(registry_db, "base", toml_path)
    new_id = edit_config(registry_db, orig_id, alpha=0.5)
    assert new_id != orig_id
    row = registry_db.execute(
        "SELECT parent_id FROM tracker_configs WHERE id = ?", (new_id,)
    ).fetchone()
    assert row["parent_id"] == orig_id
    count = registry_db.execute("SELECT COUNT(*) FROM tracker_configs").fetchone()[0]
    assert count == 2


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


# ---------------------------------------------------------------------------
# list_configs
# ---------------------------------------------------------------------------


def test_list_configs_empty(registry_db: sqlite3.Connection) -> None:
    """list_configs() returns an empty list when no configs are registered."""
    assert list_configs(registry_db) == []


def test_list_configs_returns_all(registry_db: sqlite3.Connection, tmp_path: Path) -> None:
    """list_configs() returns all rows when no name filter is given."""
    toml_path = _write_toml(tmp_path)
    create_config_from_toml(registry_db, "config_a", toml_path)
    create_config_from_toml(registry_db, "config_b", toml_path)
    rows = list_configs(registry_db)
    assert len(rows) == 2


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

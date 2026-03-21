"""Tests for scripts/db/import_calib_toml.py."""

from __future__ import annotations

import datetime
import struct
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))  # project root

from scripts.db.import_calib_toml import CalibImportResult, import_calib_toml


# ---------------------------------------------------------------------------
# Basic import behaviour
# ---------------------------------------------------------------------------


def test_import_returns_result_with_two_cameras(
    registry_db: sqlite3.Connection,
    sample_calib_toml: Path,
) -> None:
    """import_calib_toml() should return a CalibImportResult with two cameras."""
    result = import_calib_toml(registry_db, sample_calib_toml)
    assert isinstance(result, CalibImportResult)
    assert len(result.camera_instance_ids) == 2
    assert len(result.camera_mode_ids) == 2
    assert len(result.intrinsics_ids) == 2


def test_import_creates_camera_model_row(
    registry_db: sqlite3.Connection,
    sample_calib_toml: Path,
) -> None:
    """One camera_models row should be created per import call."""
    result = import_calib_toml(registry_db, sample_calib_toml)
    row = registry_db.execute(
        "SELECT * FROM camera_models WHERE id = ?", (result.camera_model_id,)
    ).fetchone()
    assert row is not None
    assert sample_calib_toml.name in row["model_name"]


def test_import_creates_camera_instances(
    registry_db: sqlite3.Connection,
    sample_calib_toml: Path,
) -> None:
    """Two camera_instances rows should be created for a two-camera TOML."""
    result = import_calib_toml(registry_db, sample_calib_toml)
    count = registry_db.execute("SELECT COUNT(*) FROM camera_instances").fetchone()[0]
    assert count == 2
    labels = set(result.camera_instance_ids.keys())
    assert labels == {"Camera1", "Camera2"}


def test_import_creates_camera_modes(
    registry_db: sqlite3.Connection,
    sample_calib_toml: Path,
) -> None:
    """Two camera_modes rows should be created for a two-camera TOML."""
    import_calib_toml(registry_db, sample_calib_toml)
    count = registry_db.execute("SELECT COUNT(*) FROM camera_modes").fetchone()[0]
    assert count == 2


def test_import_creates_intrinsics(
    registry_db: sqlite3.Connection,
    sample_calib_toml: Path,
) -> None:
    """Two intrinsics_calibrations rows should be created for a two-camera TOML."""
    import_calib_toml(registry_db, sample_calib_toml)
    count = registry_db.execute(
        "SELECT COUNT(*) FROM intrinsics_calibrations"
    ).fetchone()[0]
    assert count == 2


# ---------------------------------------------------------------------------
# Intrinsic parameter values
# ---------------------------------------------------------------------------


def test_intrinsics_fx_fy_cx_cy_correct(
    registry_db: sqlite3.Connection,
    sample_calib_toml: Path,
) -> None:
    """Focal lengths and principal point for cam1 should match the TOML matrix."""
    result = import_calib_toml(registry_db, sample_calib_toml)
    intr_id = result.intrinsics_ids["Camera1"]
    row = registry_db.execute(
        "SELECT fx, fy, cx, cy FROM intrinsics_calibrations WHERE id = ?",
        (intr_id,),
    ).fetchone()
    assert row["fx"] == pytest.approx(800.0)
    assert row["fy"] == pytest.approx(800.0)
    assert row["cx"] == pytest.approx(640.0)
    assert row["cy"] == pytest.approx(360.0)


def test_intrinsics_dist_coeffs_blob(
    registry_db: sqlite3.Connection,
    sample_calib_toml: Path,
) -> None:
    """dist_coeffs blob should decode to the original distortion values for cam1."""
    result = import_calib_toml(registry_db, sample_calib_toml)
    intr_id = result.intrinsics_ids["Camera1"]
    row = registry_db.execute(
        "SELECT dist_coeffs FROM intrinsics_calibrations WHERE id = ?",
        (intr_id,),
    ).fetchone()
    blob: bytes = row["dist_coeffs"]
    n = len(blob) // 8  # 8 bytes per float64
    decoded = struct.unpack(f"<{n}d", blob)
    expected = [-0.1, 0.05, 0.001, -0.002]
    assert len(decoded) == len(expected)
    for got, exp in zip(decoded, expected):
        assert got == pytest.approx(exp, abs=1e-9)


# ---------------------------------------------------------------------------
# Optional parameters
# ---------------------------------------------------------------------------


def test_import_with_width_height_fps(
    registry_db: sqlite3.Connection,
    sample_calib_toml: Path,
) -> None:
    """Width, height, and fps passed to import_calib_toml should be stored in camera_modes."""
    import_calib_toml(
        registry_db, sample_calib_toml, width_px=1920, height_px=1080, nominal_fps=120.0
    )
    rows = registry_db.execute(
        "SELECT width_px, height_px, nominal_fps FROM camera_modes"
    ).fetchall()
    for row in rows:
        assert row["width_px"] == 1920
        assert row["height_px"] == 1080
        assert row["nominal_fps"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Idempotency / duplicate import
# ---------------------------------------------------------------------------


def test_import_idempotent_second_call_fails_or_succeeds(
    registry_db: sqlite3.Connection,
    sample_calib_toml: Path,
) -> None:
    """A second import of the same file should succeed (UUIDs are unique each call)."""
    import_calib_toml(registry_db, sample_calib_toml)
    import_calib_toml(registry_db, sample_calib_toml)
    # Each call creates new UUID-keyed rows; expect 4 intrinsics rows total.
    count = registry_db.execute(
        "SELECT COUNT(*) FROM intrinsics_calibrations"
    ).fetchone()[0]
    assert count == 4


# ---------------------------------------------------------------------------
# Missing distortions
# ---------------------------------------------------------------------------


def test_missing_distortions_defaults_to_zeros(
    registry_db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    """When distortions key is absent, dist_coeffs blob should decode to four zeros."""
    toml_path = tmp_path / "no_dist.toml"
    toml_path.write_text(
        "[cam1]\n"
        'name = "NoDist"\n'
        "matrix = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]\n"
        "rotation = [0.0, 0.0, 0.0]\n"
        "translation = [0.0, 0.0, 1.0]\n",
        encoding="utf-8",
    )
    result = import_calib_toml(registry_db, toml_path)
    intr_id = result.intrinsics_ids["NoDist"]
    row = registry_db.execute(
        "SELECT dist_coeffs FROM intrinsics_calibrations WHERE id = ?",
        (intr_id,),
    ).fetchone()
    blob: bytes = row["dist_coeffs"]
    n = len(blob) // 8
    decoded = struct.unpack(f"<{n}d", blob)
    assert decoded == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-12)


# ---------------------------------------------------------------------------
# calibrated_at
# ---------------------------------------------------------------------------


def test_calibrated_at_defaults_to_today(
    registry_db: sqlite3.Connection,
    sample_calib_toml: Path,
) -> None:
    """calibrated_at should default to today's ISO date string."""
    today = datetime.date.today().isoformat()
    import_calib_toml(registry_db, sample_calib_toml)
    rows = registry_db.execute(
        "SELECT calibrated_at FROM intrinsics_calibrations"
    ).fetchall()
    for row in rows:
        assert row["calibrated_at"] == today


def test_custom_calibrated_at(
    registry_db: sqlite3.Connection,
    sample_calib_toml: Path,
) -> None:
    """A custom calibrated_at string should be stored as given."""
    import_calib_toml(registry_db, sample_calib_toml, calibrated_at="2025-01-15")
    rows = registry_db.execute(
        "SELECT calibrated_at FROM intrinsics_calibrations"
    ).fetchall()
    for row in rows:
        assert row["calibrated_at"] == "2025-01-15"


# ---------------------------------------------------------------------------
# metadata section skipped
# ---------------------------------------------------------------------------


def test_metadata_section_skipped(
    registry_db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    """The [metadata] section in a TOML should be silently skipped."""
    toml_path = tmp_path / "with_meta.toml"
    toml_path.write_text(
        "[metadata]\n"
        'adjusted_by = "pose2sim"\n'
        "\n"
        "[cam1]\n"
        'name = "OnlyCam"\n'
        "matrix = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]\n"
        "rotation = [0.0, 0.0, 0.0]\n"
        "translation = [0.0, 0.0, 1.0]\n"
        "distortions = [0.0, 0.0, 0.0, 0.0]\n",
        encoding="utf-8",
    )
    result = import_calib_toml(registry_db, toml_path)
    assert len(result.camera_instance_ids) == 1
    assert "OnlyCam" in result.camera_instance_ids


# ---------------------------------------------------------------------------
# Foreign-key constraint satisfaction
# ---------------------------------------------------------------------------


def test_foreign_key_constraints_satisfied(
    registry_db: sqlite3.Connection,
    sample_calib_toml: Path,
) -> None:
    """All FK relationships within the registry should be satisfied after import."""
    result = import_calib_toml(registry_db, sample_calib_toml)

    # camera_modes.camera_model_id must refer to an existing camera_models row
    mode_rows = registry_db.execute("SELECT camera_model_id FROM camera_modes").fetchall()
    for row in mode_rows:
        exists = registry_db.execute(
            "SELECT 1 FROM camera_models WHERE id = ?", (row["camera_model_id"],)
        ).fetchone()
        assert exists is not None, "camera_modes.camera_model_id references missing camera_models row"

    # intrinsics_calibrations.camera_mode_id must refer to an existing camera_modes row
    intr_rows = registry_db.execute(
        "SELECT camera_mode_id FROM intrinsics_calibrations"
    ).fetchall()
    for row in intr_rows:
        exists = registry_db.execute(
            "SELECT 1 FROM camera_modes WHERE id = ?", (row["camera_mode_id"],)
        ).fetchone()
        assert exists is not None, (
            "intrinsics_calibrations.camera_mode_id references missing camera_modes row"
        )

    # camera_instances.camera_model_id must refer to an existing camera_models row
    inst_rows = registry_db.execute(
        "SELECT camera_model_id FROM camera_instances"
    ).fetchall()
    for row in inst_rows:
        exists = registry_db.execute(
            "SELECT 1 FROM camera_models WHERE id = ?", (row["camera_model_id"],)
        ).fetchone()
        assert exists is not None, (
            "camera_instances.camera_model_id references missing camera_models row"
        )

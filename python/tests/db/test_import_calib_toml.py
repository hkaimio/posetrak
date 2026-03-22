"""Tests for scripts/db/import_calib_toml.py."""

from __future__ import annotations

import datetime
import sqlite3
import struct
from pathlib import Path

import pytest


from posetrak.db.import_calib_toml import CalibImportResult, import_calib_toml


# ---------------------------------------------------------------------------
# Basic import behaviour
# ---------------------------------------------------------------------------


def test_import_returns_result_with_two_cameras(
    registry_db: sqlite3.Connection,
    camera_mode_id: str,
    sample_calib_toml: Path,
) -> None:
    """import_calib_toml() should return a CalibImportResult with two cameras."""
    result = import_calib_toml(registry_db, sample_calib_toml, camera_mode_id)
    assert isinstance(result, CalibImportResult)
    assert len(result.camera_instance_ids) == 2
    assert len(result.intrinsics_ids) == 2


def test_import_creates_camera_instances(
    registry_db: sqlite3.Connection,
    camera_mode_id: str,
    sample_calib_toml: Path,
) -> None:
    """Two camera_instances rows should be created for a two-camera TOML."""
    result = import_calib_toml(registry_db, sample_calib_toml, camera_mode_id)
    count = registry_db.execute("SELECT COUNT(*) FROM camera_instances").fetchone()[0]
    assert count == 2
    labels = set(result.camera_instance_ids.keys())
    assert labels == {"Camera1", "Camera2"}


def test_import_creates_intrinsics(
    registry_db: sqlite3.Connection,
    camera_mode_id: str,
    sample_calib_toml: Path,
) -> None:
    """Two intrinsics_calibrations rows should be created for a two-camera TOML."""
    import_calib_toml(registry_db, sample_calib_toml, camera_mode_id)
    count = registry_db.execute(
        "SELECT COUNT(*) FROM intrinsics_calibrations"
    ).fetchone()[0]
    assert count == 2


def test_import_does_not_create_extra_model_or_mode_rows(
    registry_db: sqlite3.Connection,
    camera_mode_id: str,
    sample_calib_toml: Path,
) -> None:
    """import_calib_toml() must not create additional camera_models or camera_modes rows."""
    n_models_before = registry_db.execute(
        "SELECT COUNT(*) FROM camera_models"
    ).fetchone()[0]
    n_modes_before = registry_db.execute(
        "SELECT COUNT(*) FROM camera_modes"
    ).fetchone()[0]

    import_calib_toml(registry_db, sample_calib_toml, camera_mode_id)

    assert registry_db.execute(
        "SELECT COUNT(*) FROM camera_models"
    ).fetchone()[0] == n_models_before
    assert registry_db.execute(
        "SELECT COUNT(*) FROM camera_modes"
    ).fetchone()[0] == n_modes_before


# ---------------------------------------------------------------------------
# Intrinsic parameter values
# ---------------------------------------------------------------------------


def test_intrinsics_fx_fy_cx_cy_correct(
    registry_db: sqlite3.Connection,
    camera_mode_id: str,
    sample_calib_toml: Path,
) -> None:
    """Focal lengths and principal point for cam1 should match the TOML matrix."""
    result = import_calib_toml(registry_db, sample_calib_toml, camera_mode_id)
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
    camera_mode_id: str,
    sample_calib_toml: Path,
) -> None:
    """dist_coeffs blob should decode to the original distortion values for cam1."""
    result = import_calib_toml(registry_db, sample_calib_toml, camera_mode_id)
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


def test_intrinsics_linked_to_supplied_camera_mode(
    registry_db: sqlite3.Connection,
    camera_mode_id: str,
    sample_calib_toml: Path,
) -> None:
    """All intrinsics_calibrations rows must reference the supplied camera_mode_id."""
    import_calib_toml(registry_db, sample_calib_toml, camera_mode_id)
    rows = registry_db.execute(
        "SELECT camera_mode_id FROM intrinsics_calibrations"
    ).fetchall()
    for row in rows:
        assert row["camera_mode_id"] == camera_mode_id


# ---------------------------------------------------------------------------
# Invalid camera mode ID(s)
# ---------------------------------------------------------------------------


def test_invalid_homogeneous_mode_id_raises(
    registry_db: sqlite3.Connection,
    sample_calib_toml: Path,
) -> None:
    """Passing a non-existent homogeneous camera_mode_id should raise ValueError."""
    with pytest.raises(ValueError, match="camera_mode_id"):
        import_calib_toml(registry_db, sample_calib_toml, "00000000-0000-0000-0000-000000000000")


def test_invalid_per_camera_mode_id_raises(
    registry_db: sqlite3.Connection,
    camera_mode_id: str,
    sample_calib_toml: Path,
) -> None:
    """Passing a non-existent mode ID in a per-camera dict should raise ValueError."""
    with pytest.raises(ValueError, match="camera_mode_id"):
        import_calib_toml(
            registry_db,
            sample_calib_toml,
            {"cam1": camera_mode_id, "cam2": "00000000-0000-0000-0000-000000000000"},
        )


# ---------------------------------------------------------------------------
# Per-camera mode mapping
# ---------------------------------------------------------------------------


def test_per_camera_mapping_imports_only_listed(
    registry_db: sqlite3.Connection,
    camera_mode_id: str,
    sample_calib_toml: Path,
) -> None:
    """Per-camera mapping should only import cameras whose section key is listed."""
    result = import_calib_toml(
        registry_db, sample_calib_toml, {"cam1": camera_mode_id}
    )
    assert len(result.camera_instance_ids) == 1
    assert "Camera1" in result.camera_instance_ids
    assert "Camera2" not in result.camera_instance_ids


def test_per_camera_mapping_skipped_set(
    registry_db: sqlite3.Connection,
    camera_mode_id: str,
    sample_calib_toml: Path,
) -> None:
    """Cameras not listed in the per-camera mapping should appear in result.skipped."""
    result = import_calib_toml(
        registry_db, sample_calib_toml, {"cam1": camera_mode_id}
    )
    assert "cam2" in result.skipped
    assert "cam1" not in result.skipped


def test_per_camera_mapping_different_modes(
    registry_db: sqlite3.Connection,
    sample_calib_toml: Path,
) -> None:
    """Each camera can be linked to a different camera mode."""
    from posetrak.db.db import create_camera_model, create_camera_mode

    model_id = create_camera_model(registry_db, model_name="Mixed")
    mode_4k = create_camera_mode(registry_db, model_id, width_px=3840, height_px=2160)
    mode_hd = create_camera_mode(registry_db, model_id, width_px=1920, height_px=1080)

    result = import_calib_toml(
        registry_db,
        sample_calib_toml,
        {"cam1": mode_4k, "cam2": mode_hd},
    )

    assert len(result.camera_instance_ids) == 2

    intr1 = registry_db.execute(
        "SELECT camera_mode_id FROM intrinsics_calibrations WHERE id = ?",
        (result.intrinsics_ids["Camera1"],),
    ).fetchone()
    intr2 = registry_db.execute(
        "SELECT camera_mode_id FROM intrinsics_calibrations WHERE id = ?",
        (result.intrinsics_ids["Camera2"],),
    ).fetchone()
    assert intr1["camera_mode_id"] == mode_4k
    assert intr2["camera_mode_id"] == mode_hd


def test_homogeneous_mode_skipped_set_is_empty(
    registry_db: sqlite3.Connection,
    camera_mode_id: str,
    sample_calib_toml: Path,
) -> None:
    """Homogeneous mode imports all cameras; skipped set should be empty."""
    result = import_calib_toml(registry_db, sample_calib_toml, camera_mode_id)
    assert result.skipped == set()


# ---------------------------------------------------------------------------
# Idempotency / duplicate import
# ---------------------------------------------------------------------------


def test_import_idempotent_second_call_fails_or_succeeds(
    registry_db: sqlite3.Connection,
    camera_mode_id: str,
    sample_calib_toml: Path,
) -> None:
    """A second import of the same file should succeed (UUIDs are unique each call)."""
    import_calib_toml(registry_db, sample_calib_toml, camera_mode_id)
    import_calib_toml(registry_db, sample_calib_toml, camera_mode_id)
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
    camera_mode_id: str,
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
    result = import_calib_toml(registry_db, toml_path, camera_mode_id)
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
    camera_mode_id: str,
    sample_calib_toml: Path,
) -> None:
    """calibrated_at should default to today's ISO date string."""
    today = datetime.date.today().isoformat()
    import_calib_toml(registry_db, sample_calib_toml, camera_mode_id)
    rows = registry_db.execute(
        "SELECT calibrated_at FROM intrinsics_calibrations"
    ).fetchall()
    for row in rows:
        assert row["calibrated_at"] == today


def test_custom_calibrated_at(
    registry_db: sqlite3.Connection,
    camera_mode_id: str,
    sample_calib_toml: Path,
) -> None:
    """A custom calibrated_at string should be stored as given."""
    import_calib_toml(
        registry_db, sample_calib_toml, camera_mode_id, calibrated_at="2025-01-15"
    )
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
    camera_mode_id: str,
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
    result = import_calib_toml(registry_db, toml_path, camera_mode_id)
    assert len(result.camera_instance_ids) == 1
    assert "OnlyCam" in result.camera_instance_ids


# ---------------------------------------------------------------------------
# Foreign-key constraint satisfaction
# ---------------------------------------------------------------------------


def test_foreign_key_constraints_satisfied(
    registry_db: sqlite3.Connection,
    camera_mode_id: str,
    sample_calib_toml: Path,
) -> None:
    """All FK relationships created by the importer should be satisfied."""
    import_calib_toml(registry_db, sample_calib_toml, camera_mode_id)

    # intrinsics_calibrations.camera_mode_id must refer to an existing camera_modes row
    for row in registry_db.execute(
        "SELECT camera_mode_id FROM intrinsics_calibrations"
    ).fetchall():
        exists = registry_db.execute(
            "SELECT 1 FROM camera_modes WHERE id = ?", (row["camera_mode_id"],)
        ).fetchone()
        assert exists is not None, (
            "intrinsics_calibrations.camera_mode_id references missing camera_modes row"
        )

    # camera_instances.camera_model_id must refer to an existing camera_models row
    for row in registry_db.execute(
        "SELECT camera_model_id FROM camera_instances"
    ).fetchall():
        exists = registry_db.execute(
            "SELECT 1 FROM camera_models WHERE id = ?", (row["camera_model_id"],)
        ).fetchone()
        assert exists is not None, (
            "camera_instances.camera_model_id references missing camera_models row"
        )

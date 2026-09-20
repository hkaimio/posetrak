# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for registry, camera-model, camera-mode CLI commands."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.db.db import create_session, generate_id
from posetrak.db.db import create_registry


class TestRegistryInit:
    def test_creates_registry(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db_path = tmp_path / "new_registry.db"
        result = cli_runner.invoke(
            main, ["--registry", str(db_path), "registry", "init"]
        )
        assert result.exit_code == 0, result.output
        assert db_path.exists()
        assert "Registry created" in result.output

    def test_fails_if_already_exists(
        self, cli_runner: CliRunner, registry_db_path: Path
    ) -> None:
        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "registry", "init"]
        )
        assert result.exit_code != 0

    def test_creates_parent_dirs(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        db_path = tmp_path / "sub" / "dir" / "registry.db"
        result = cli_runner.invoke(
            main, ["--registry", str(db_path), "registry", "init"]
        )
        assert result.exit_code == 0, result.output
        assert db_path.exists()


class TestRegistryInfo:
    def test_shows_info(self, cli_runner: CliRunner, registry_db_path: Path) -> None:
        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "registry", "info"]
        )
        assert result.exit_code == 0, result.output
        assert "schema version" in result.output
        assert "camera models" in result.output

    def test_json_mode(self, cli_runner: CliRunner, registry_db_path: Path) -> None:
        import json

        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "--json", "registry", "info"]
        )
        assert result.exit_code == 0, result.output
        obj = json.loads(result.output.strip())
        assert "schema_version" in obj
        assert "camera_models" in obj

    def test_fails_on_missing_registry(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        result = cli_runner.invoke(
            main,
            ["--registry", str(tmp_path / "no_such.db"), "registry", "info"],
        )
        assert result.exit_code != 0


class TestCameraModelAdd:
    def test_adds_model(self, cli_runner: CliRunner, registry_db_path: Path) -> None:
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "camera-model", "add",
                "--manufacturer", "GoPro",
                "--model-name", "HERO 11 Mini",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "camera_model_id:" in result.output

    def test_adds_model_minimal(
        self, cli_runner: CliRunner, registry_db_path: Path
    ) -> None:
        result = cli_runner.invoke(
            main,
            ["--registry", str(registry_db_path), "camera-model", "add"],
        )
        assert result.exit_code == 0, result.output


class TestCameraModelList:
    def test_empty_registry(
        self, cli_runner: CliRunner, registry_db_path: Path
    ) -> None:
        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "camera-model", "list"]
        )
        assert result.exit_code == 0, result.output
        assert "No camera models registered" in result.output

    def test_lists_models(self, cli_runner: CliRunner, registry_db_path: Path) -> None:
        # Add two models first.
        for mfr in ("Acme", "Beta"):
            cli_runner.invoke(
                main,
                [
                    "--registry", str(registry_db_path),
                    "camera-model", "add",
                    "--manufacturer", mfr,
                    "--model-name", "Cam",
                ],
            )
        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "camera-model", "list"]
        )
        assert result.exit_code == 0, result.output
        assert "Acme" in result.output
        assert "Beta" in result.output

    def test_json_mode(self, cli_runner: CliRunner, registry_db_path: Path) -> None:
        import json

        cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "camera-model", "add",
                "--manufacturer", "TestCo",
                "--model-name", "X1",
            ],
        )
        result = cli_runner.invoke(
            main,
            ["--registry", str(registry_db_path), "--json", "camera-model", "list"],
        )
        assert result.exit_code == 0, result.output
        obj = json.loads(result.output.strip())
        assert "id" in obj
        assert "manufacturer" in obj


class TestCameraModeAdd:
    def test_adds_mode(
        self, cli_runner: CliRunner, registry_db_path: Path, camera_model_id: str
    ) -> None:
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "camera-mode", "add",
                "--model-id", camera_model_id,
                "--width", "1920",
                "--height", "1080",
                "--fps", "60",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "camera_mode_id:" in result.output


class TestCameraModeList:
    def test_empty(self, cli_runner: CliRunner, registry_db_path: Path) -> None:
        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "camera-mode", "list"]
        )
        assert result.exit_code == 0
        assert "No camera modes" in result.output

    def test_lists_modes(
        self,
        cli_runner: CliRunner,
        registry_db_path: Path,
        camera_model_id: str,
    ) -> None:
        cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "camera-mode", "add",
                "--model-id", camera_model_id,
                "--width", "1280",
                "--height", "720",
                "--fps", "120",
            ],
        )
        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "camera-mode", "list"]
        )
        assert result.exit_code == 0
        assert "1280" in result.output
        assert "720" in result.output

    def test_filter_by_model(
        self,
        cli_runner: CliRunner,
        registry_db_path: Path,
        camera_model_id: str,
    ) -> None:
        cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "camera-mode", "add",
                "--model-id", camera_model_id,
                "--width", "640",
                "--height", "480",
            ],
        )
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "camera-mode", "list",
                "--model-id", camera_model_id,
            ],
        )
        assert result.exit_code == 0
        assert "640" in result.output


# ---------------------------------------------------------------------------
# Helpers for import-session tests
# ---------------------------------------------------------------------------


def _make_session_with_cameras(
    tmp_path: Path,
    *,
    n_modes: int = 1,
    n_calibrations: int = 1,
) -> tuple[Path, str, str, str, list[str]]:
    """Create a session DB with one camera model/instance, n_modes modes, n_calibrations calibs.

    Returns (db_path, model_id, instance_id, mode_id_first, calib_ids).
    """
    db_path = tmp_path / f"session_{generate_id()[:8]}.db"
    conn = create_session(db_path)

    model_id = generate_id()
    instance_id = generate_id()
    conn.execute(
        "INSERT INTO camera_models (id, manufacturer, model_name) VALUES (?,?,?)",
        (model_id, "AcmeCorp", "Cam X"),
    )
    conn.execute(
        "INSERT INTO camera_instances (id, camera_model_id, label) VALUES (?,?,?)",
        (instance_id, model_id, "cam-front"),
    )

    mode_ids = []
    for i in range(n_modes):
        mid = generate_id()
        mode_ids.append(mid)
        conn.execute(
            "INSERT INTO camera_modes "
            "(id, camera_model_id, width_px, height_px, nominal_fps) VALUES (?,?,?,?,?)",
            (mid, model_id, 1920 + i * 10, 1080, 60.0),
        )

    calib_ids = []
    for i in range(n_calibrations):
        cid = generate_id()
        calib_ids.append(cid)
        conn.execute(
            "INSERT INTO intrinsics_calibrations "
            "(id, camera_mode_id, calibrated_at, distortion_model,"
            " fx, fy, cx, cy) VALUES (?,?,?,?,?,?,?,?)",
            (cid, mode_ids[0], f"2026-01-0{i+1}", "radtan",
             1000.0 + i, 1001.0 + i, 960.0, 540.0),
        )

    conn.commit()
    conn.close()
    return db_path, model_id, instance_id, mode_ids[0], calib_ids


# ---------------------------------------------------------------------------
# camera import-session
# ---------------------------------------------------------------------------


class TestCameraImportSession:
    def _invoke(
        self,
        cli_runner: CliRunner,
        registry_db_path: Path,
        session_db_path: Path,
        extra_args: list[str] | None = None,
    ):
        args = [
            "--registry", str(registry_db_path),
            "--session", str(session_db_path),
            "camera", "import-session",
        ] + (extra_args or [])
        return cli_runner.invoke(main, args, catch_exceptions=False)

    def test_import_basic(
        self, cli_runner: CliRunner, registry_db_path: Path, tmp_path: Path
    ) -> None:
        session_path, model_id, instance_id, mode_id, calib_ids = (
            _make_session_with_cameras(tmp_path)
        )
        result = self._invoke(cli_runner, registry_db_path, session_path)
        assert result.exit_code == 0, result.output
        assert "1 imported" in result.output  # at minimum one table got a row

        reg = sqlite3.connect(str(registry_db_path))
        reg.row_factory = sqlite3.Row
        assert reg.execute(
            "SELECT id FROM camera_models WHERE id=?", (model_id,)
        ).fetchone() is not None
        assert reg.execute(
            "SELECT id FROM camera_instances WHERE id=?", (instance_id,)
        ).fetchone() is not None
        assert reg.execute(
            "SELECT id FROM camera_modes WHERE id=?", (mode_id,)
        ).fetchone() is not None
        assert reg.execute(
            "SELECT id FROM intrinsics_calibrations WHERE id=?", (calib_ids[0],)
        ).fetchone() is not None
        reg.close()

    def test_import_idempotent(
        self, cli_runner: CliRunner, registry_db_path: Path, tmp_path: Path
    ) -> None:
        session_path, *_ = _make_session_with_cameras(tmp_path)
        # First import
        r1 = self._invoke(cli_runner, registry_db_path, session_path)
        assert r1.exit_code == 0, r1.output
        # Second import — all rows already present
        r2 = self._invoke(cli_runner, registry_db_path, session_path)
        assert r2.exit_code == 0, r2.output
        assert "Nothing to import" in r2.output or "0 imported" in r2.output

    def test_import_partial_model_already_exists(
        self, cli_runner: CliRunner, registry_db_path: Path, tmp_path: Path
    ) -> None:
        """If model is in registry but mode and calibration are missing, they get imported."""
        session_path, model_id, instance_id, mode_id, calib_ids = (
            _make_session_with_cameras(tmp_path, n_calibrations=2)
        )
        # Pre-insert the model and instance into registry, but not the mode/calibration.
        reg = sqlite3.connect(str(registry_db_path))
        reg.execute(
            "INSERT INTO camera_models (id, manufacturer, model_name) VALUES (?,?,?)",
            (model_id, "AcmeCorp", "Cam X"),
        )
        reg.execute(
            "INSERT INTO camera_instances (id, camera_model_id, label) VALUES (?,?,?)",
            (instance_id, model_id, "cam-front"),
        )
        reg.commit()
        reg.close()

        result = self._invoke(cli_runner, registry_db_path, session_path)
        assert result.exit_code == 0, result.output

        reg = sqlite3.connect(str(registry_db_path))
        reg.row_factory = sqlite3.Row
        # Mode must now be present
        assert reg.execute(
            "SELECT id FROM camera_modes WHERE id=?", (mode_id,)
        ).fetchone() is not None
        # Both calibrations must be present
        for cid in calib_ids:
            assert reg.execute(
                "SELECT id FROM intrinsics_calibrations WHERE id=?", (cid,)
            ).fetchone() is not None
        reg.close()

        # Counts: 0 models, 0 instances, 1 mode, 2 calibrations imported
        assert "0 imported" in result.output   # models/instances skipped
        assert "1 imported" in result.output   # mode
        assert "2 imported" in result.output   # calibrations

    def test_import_dry_run(
        self, cli_runner: CliRunner, registry_db_path: Path, tmp_path: Path
    ) -> None:
        session_path, model_id, *_ = _make_session_with_cameras(tmp_path)
        result = self._invoke(cli_runner, registry_db_path, session_path, ["--dry-run"])
        assert result.exit_code == 0, result.output
        assert "Dry run" in result.output
        assert "would import" in result.output

        # Nothing should have been written
        reg = sqlite3.connect(str(registry_db_path))
        row = reg.execute(
            "SELECT id FROM camera_models WHERE id=?", (model_id,)
        ).fetchone()
        reg.close()
        assert row is None

    def test_import_no_session(
        self, cli_runner: CliRunner, registry_db_path: Path
    ) -> None:
        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "camera", "import-session"]
        )
        assert result.exit_code != 0

    def test_import_multiple_modes_and_calibs(
        self, cli_runner: CliRunner, registry_db_path: Path, tmp_path: Path
    ) -> None:
        session_path, _, _, _, calib_ids = _make_session_with_cameras(
            tmp_path, n_modes=3, n_calibrations=3
        )
        result = self._invoke(cli_runner, registry_db_path, session_path)
        assert result.exit_code == 0, result.output

        reg = sqlite3.connect(str(registry_db_path))
        n_modes = reg.execute("SELECT COUNT(*) FROM camera_modes").fetchone()[0]
        n_calibs = reg.execute(
            "SELECT COUNT(*) FROM intrinsics_calibrations"
        ).fetchone()[0]
        reg.close()
        assert n_modes == 3
        assert n_calibs == 3


# ---------------------------------------------------------------------------
# Session-aware camera list / camera-model list / camera-mode list
# ---------------------------------------------------------------------------


class TestSessionAwareCameraList:
    def test_camera_list_from_session(
        self, cli_runner: CliRunner, registry_db_path: Path, tmp_path: Path
    ) -> None:
        session_path, _, instance_id, _, _ = _make_session_with_cameras(tmp_path)
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "--session", str(session_path),
                "camera", "list",
            ],
        )
        assert result.exit_code == 0, result.output
        assert instance_id[:8] in result.output or "cam-front" in result.output

    def test_camera_model_list_from_session(
        self, cli_runner: CliRunner, registry_db_path: Path, tmp_path: Path
    ) -> None:
        session_path, model_id, *_ = _make_session_with_cameras(tmp_path)
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "--session", str(session_path),
                "camera-model", "list",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "AcmeCorp" in result.output

    def test_camera_mode_list_from_session(
        self, cli_runner: CliRunner, registry_db_path: Path, tmp_path: Path
    ) -> None:
        session_path, *_, mode_id, _ = _make_session_with_cameras(tmp_path, n_modes=2)
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "--session", str(session_path),
                "camera-mode", "list",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "1920" in result.output

    def test_camera_list_without_session_uses_registry(
        self, cli_runner: CliRunner, registry_db_path: Path
    ) -> None:
        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "camera", "list"]
        )
        assert result.exit_code == 0, result.output
        assert "No camera instances registered" in result.output


# ---------------------------------------------------------------------------
# calib show
# ---------------------------------------------------------------------------


def _calibration_row(
    db_path: Path, calib_id: str, *, original: bool = True, coefficients=(0.1, -0.2, 0.001, 0.002, 0.03),
    model: str = "radtan",
) -> None:
    """Fill in the stored OpenCV parameters of a calibration made by _make_session_with_cameras."""
    import struct

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "UPDATE intrinsics_calibrations SET distortion_model = ?, calibration_tool = 'auto-charuco', "
        "rms_error = 0.43, image_width = 1920, image_height = 1080, dist_coeffs = ?, matrix_original = ? "
        "WHERE id = ?",
        (model, struct.pack(f"<{len(coefficients)}d", *coefficients),
         struct.pack("<9d", 900.5, 0, 950.25, 0, 901.5, 530.75, 0, 0, 1) if original else None, calib_id),
    )
    conn.commit()
    conn.close()


def _show(db_path: Path, *args: str, json_mode: bool = False, use_registry: bool = False):
    base = ["--registry", str(db_path)] if use_registry else ["--session", str(db_path)]
    if json_mode:
        base = ["--json", *base]
    return CliRunner().invoke(main, [*base, "calib", "show", *args], catch_exceptions=False)


class TestCalibShow:
    def test_it_shows_both_matrices_and_the_distortion_coefficients_by_name(self, tmp_path: Path) -> None:
        db_path, _, _, _, (calib_id,) = _make_session_with_cameras(tmp_path)
        _calibration_row(db_path, calib_id)

        result = _show(db_path, calib_id[:8])

        assert result.exit_code == 0, result.output
        out = result.output
        assert f"id:               {calib_id}" in out
        assert "AcmeCorp Cam X" in out and "auto-charuco, rms 0.430 px" in out and "1920 x 1080" in out
        assert "camera_matrix (of the raw image: use with dist_coeffs)" in out
        assert "fx 900.500000  fy 901.500000  cx 950.250000  cy 530.750000" in out          # the original
        assert "dist_coeffs (k1, k2, p1, p2, k3)" in out and "0.100000000, -0.200000000" in out
        assert "new_camera_matrix" in out and "fx 1000.000000  fy 1001.000000  cx 960.000000  cy 540.000000" in out

    def test_json_is_one_object_with_opencv_arrays(self, tmp_path: Path) -> None:
        db_path, _, _, mode_id, (calib_id,) = _make_session_with_cameras(tmp_path)
        _calibration_row(db_path, calib_id)

        result = _show(db_path, calib_id, json_mode=True)

        record = json.loads(result.output)
        assert record["id"] == calib_id and record["camera_mode_id"] == mode_id
        assert record["camera_matrix"] == [[900.5, 0.0, 950.25], [0.0, 901.5, 530.75], [0.0, 0.0, 1.0]]
        assert record["dist_coeffs"] == [0.1, -0.2, 0.001, 0.002, 0.03]
        assert record["dist_coeff_names"] == ["k1", "k2", "p1", "p2", "k3"]
        assert record["new_camera_matrix"] == [[1000.0, 0.0, 960.0], [0.0, 1001.0, 540.0], [0.0, 0.0, 1.0]]
        assert (record["image_width"], record["image_height"], record["distortion_model"]) == (1920, 1080, "radtan")
        assert record["has_original_matrix"] is True and record["rms_error_px"] == 0.43

    def test_a_fisheye_calibration_names_its_four_coefficients(self, tmp_path: Path) -> None:
        db_path, _, _, _, (calib_id,) = _make_session_with_cameras(tmp_path)
        _calibration_row(db_path, calib_id, coefficients=(0.05, 0.01, -0.02, 0.003), model="fisheye")

        record = json.loads(_show(db_path, calib_id, json_mode=True).output)
        text = _show(db_path, calib_id).output

        assert record["dist_coeff_names"] == ["k1", "k2", "k3", "k4"]
        assert "OpenCV fisheye" in text and "dist_coeffs (k1, k2, k3, k4)" in text

    def test_without_a_stored_original_matrix_the_two_are_the_same_and_it_says_so(self, tmp_path: Path) -> None:
        db_path, _, _, _, (calib_id,) = _make_session_with_cameras(tmp_path)
        _calibration_row(db_path, calib_id, original=False)

        record = json.loads(_show(db_path, calib_id, json_mode=True).output)
        text = _show(db_path, calib_id).output

        assert record["camera_matrix"] == record["new_camera_matrix"] and record["has_original_matrix"] is False
        assert "no separate original matrix is stored" in text

    def test_a_calibration_without_distortion_coefficients_shows_zeros(self, tmp_path: Path) -> None:
        db_path, _, _, _, (calib_id,) = _make_session_with_cameras(tmp_path)

        record = json.loads(_show(db_path, calib_id, json_mode=True).output)

        assert record["dist_coeffs"] == [0.0, 0.0, 0.0, 0.0] and record["dist_coeff_names"] == ["k1", "k2", "p1", "p2"]

    def test_it_reads_the_registry_when_no_session_is_given(self, registry_db_path: Path, tmp_path: Path) -> None:
        source, _, _, _, (calib_id,) = _make_session_with_cameras(tmp_path)
        _calibration_row(source, calib_id)
        CliRunner().invoke(main, ["--registry", str(registry_db_path), "--session", str(source),
                                  "camera", "import-session"], catch_exceptions=False)

        result = _show(registry_db_path, calib_id[:8], use_registry=True)

        assert result.exit_code == 0, result.output
        assert "fx 900.500000" in result.output

    def test_unknown_and_ambiguous_ids_are_reported(self, tmp_path: Path) -> None:
        db_path, _, _, _, calib_ids = _make_session_with_cameras(tmp_path, n_calibrations=2)
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE intrinsics_calibrations SET id = 'aaaa-1' WHERE id = ?", (calib_ids[0],))
        conn.execute("UPDATE intrinsics_calibrations SET id = 'aaaa-2' WHERE id = ?", (calib_ids[1],))
        conn.commit()
        conn.close()

        unknown = _show(db_path, "zzzz")
        ambiguous = _show(db_path, "aaaa")

        assert unknown.exit_code != 0 and "No intrinsics calibration 'zzzz'" in unknown.output
        assert ambiguous.exit_code != 0 and "matches 2 calibrations" in ambiguous.output

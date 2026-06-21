"""Tests for registry, camera-model, camera-mode CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from posetrak.cli.main import main
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

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for marker-body CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from posetrak.cli.main import main


MINIMAL_MARKER_BODY_YAML = """\
name: test-rig
units: meters
markers:
  - name: top
    type: aruco
    dictionary: DICT_4X4_50
    id: "4"
    size: 0.145
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
"""

TWO_MARKER_BODY_YAML = """\
name: test-bokken
units: meters
markers:
  - name: hilt
    type: aruco
    dictionary: DICT_4X4_50
    id: "3"
    size: 0.05
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
  - name: tip
    type: aruco
    dictionary: DICT_4X4_50
    id: "7"
    size: 0.03
    center: [0.0, 0.9, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
"""


@pytest.fixture()
def marker_body_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "test_rig.yaml"
    path.write_text(MINIMAL_MARKER_BODY_YAML, encoding="utf-8")
    return path


@pytest.fixture()
def two_marker_body_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "test_bokken.yaml"
    path.write_text(TWO_MARKER_BODY_YAML, encoding="utf-8")
    return path


class TestMarkerBodyImport:
    def test_import_to_registry(
        self, cli_runner: CliRunner, registry_db_path: Path, marker_body_yaml: Path,
    ) -> None:
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "marker-body", "import",
                "--file", str(marker_body_yaml),
                "--global",
                "--name", "My Test Rig",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "marker_body_id:" in result.output

    def test_import_to_session(
        self, cli_runner: CliRunner, registry_db_path: Path, session_db_path: Path,
        marker_body_yaml: Path,
    ) -> None:
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "--session", str(session_db_path),
                "marker-body", "import",
                "--file", str(marker_body_yaml),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "marker_body_id:" in result.output

    def test_import_requires_target(
        self, cli_runner: CliRunner, registry_db_path: Path, marker_body_yaml: Path,
    ) -> None:
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "marker-body", "import",
                "--file", str(marker_body_yaml),
            ],
        )
        assert result.exit_code != 0

    def test_import_idempotent(
        self, cli_runner: CliRunner, registry_db_path: Path, marker_body_yaml: Path,
    ) -> None:
        args = [
            "--registry", str(registry_db_path),
            "marker-body", "import",
            "--file", str(marker_body_yaml),
            "--global",
        ]
        r1 = cli_runner.invoke(main, args)
        r2 = cli_runner.invoke(main, args)
        assert r1.exit_code == 0 and r2.exit_code == 0
        id1 = r1.output.split("marker_body_id:")[-1].strip()
        id2 = r2.output.split("marker_body_id:")[-1].strip()
        assert id1 == id2


class TestMarkerBodyList:
    def test_empty_registry(self, cli_runner: CliRunner, registry_db_path: Path) -> None:
        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "marker-body", "list"]
        )
        assert result.exit_code == 0
        assert "No marker body definitions" in result.output

    def test_lists_after_import(
        self, cli_runner: CliRunner, registry_db_path: Path, marker_body_yaml: Path,
    ) -> None:
        cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "marker-body", "import",
                "--file", str(marker_body_yaml),
                "--global",
                "--name", "My Rig",
            ],
        )
        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "marker-body", "list"]
        )
        assert result.exit_code == 0
        assert "My Rig" in result.output

    def test_json_mode(
        self, cli_runner: CliRunner, registry_db_path: Path, marker_body_yaml: Path,
    ) -> None:
        cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "marker-body", "import",
                "--file", str(marker_body_yaml),
                "--global",
                "--name", "JSON Rig",
            ],
        )
        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "--json", "marker-body", "list"],
        )
        assert result.exit_code == 0
        rows = [json.loads(line) for line in result.output.strip().splitlines()]
        assert all("id" in r and "name" in r for r in rows)
        assert any(r["name"] == "JSON Rig" for r in rows)


class TestMarkerBodyShow:
    def test_show_by_id_prefix(
        self, cli_runner: CliRunner, registry_db_path: Path, marker_body_yaml: Path,
    ) -> None:
        import_result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "marker-body", "import",
                "--file", str(marker_body_yaml),
                "--global",
                "--name", "Show Rig",
                "--source", "hand-measured",
            ],
        )
        body_id = import_result.output.split("marker_body_id:")[-1].strip()

        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "marker-body", "show", body_id[:8]],
        )
        assert result.exit_code == 0, result.output
        assert "Show Rig" in result.output
        assert "hand-measured" in result.output

    def test_show_not_found(self, cli_runner: CliRunner, registry_db_path: Path) -> None:
        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "marker-body", "show", "deadbeef"],
        )
        assert result.exit_code != 0


class TestMarkerBodyExport:
    def test_export_round_trips_yaml(
        self, cli_runner: CliRunner, registry_db_path: Path, marker_body_yaml: Path,
    ) -> None:
        import_result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "marker-body", "import",
                "--file", str(marker_body_yaml),
                "--global",
            ],
        )
        body_id = import_result.output.split("marker_body_id:")[-1].strip()

        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "marker-body", "export", body_id],
        )
        assert result.exit_code == 0, result.output
        assert result.output == MINIMAL_MARKER_BODY_YAML

    def test_export_to_file(
        self, cli_runner: CliRunner, registry_db_path: Path, marker_body_yaml: Path,
        tmp_path: Path,
    ) -> None:
        import_result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "marker-body", "import",
                "--file", str(marker_body_yaml),
                "--global",
            ],
        )
        body_id = import_result.output.split("marker_body_id:")[-1].strip()
        out_path = tmp_path / "exported.yaml"

        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "marker-body", "export", body_id, "--output", str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_path.read_text(encoding="utf-8") == MINIMAL_MARKER_BODY_YAML


class TestMarkerBodyToSkeleton:
    def _import(self, cli_runner, registry_db_path, yaml_path, name="") -> str:
        args = [
            "--registry", str(registry_db_path),
            "marker-body", "import",
            "--file", str(yaml_path),
            "--global",
        ]
        if name:
            args += ["--name", name]
        result = cli_runner.invoke(main, args)
        assert result.exit_code == 0, result.output
        return result.output.split("marker_body_id:")[-1].strip()

    def test_creates_skeleton_and_records_provenance(
        self, cli_runner: CliRunner, registry_db_path: Path, two_marker_body_yaml: Path,
    ) -> None:
        body_id = self._import(cli_runner, registry_db_path, two_marker_body_yaml)

        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "marker-body", "to-skeleton", body_id],
        )
        assert result.exit_code == 0, result.output
        assert "skeleton_id:" in result.output
        skeleton_id = result.output.split("skeleton_id:")[-1].strip()

        export_result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "skeleton", "export", skeleton_id],
        )
        assert export_result.exit_code == 0, export_result.output
        parsed = yaml.safe_load(export_result.output)
        assert parsed["name"] == "test-bokken"
        assert parsed["generated_from_marker_body"] == body_id
        assert len(parsed["markers"]) == 8  # 2 markers * 4 corners
        assert parsed["joints"][0]["name"] == "prop_root"

    def test_idempotent(
        self, cli_runner: CliRunner, registry_db_path: Path, two_marker_body_yaml: Path,
    ) -> None:
        body_id = self._import(cli_runner, registry_db_path, two_marker_body_yaml)
        args = ["--registry", str(registry_db_path), "marker-body", "to-skeleton", body_id]
        r1 = cli_runner.invoke(main, args)
        r2 = cli_runner.invoke(main, args)
        assert r1.exit_code == 0 and r2.exit_code == 0
        id1 = r1.output.split("skeleton_id:")[-1].strip()
        id2 = r2.output.split("skeleton_id:")[-1].strip()
        assert id1 == id2

    def test_name_override(
        self, cli_runner: CliRunner, registry_db_path: Path, two_marker_body_yaml: Path,
    ) -> None:
        body_id = self._import(cli_runner, registry_db_path, two_marker_body_yaml)
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "marker-body", "to-skeleton", body_id, "--name", "my-prop-skeleton",
            ],
        )
        assert result.exit_code == 0, result.output
        skeleton_id = result.output.split("skeleton_id:")[-1].strip()

        export_result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "skeleton", "export", skeleton_id],
        )
        assert yaml.safe_load(export_result.output)["name"] == "my-prop-skeleton"

    def test_output_file(
        self, cli_runner: CliRunner, registry_db_path: Path, two_marker_body_yaml: Path,
        tmp_path: Path,
    ) -> None:
        body_id = self._import(cli_runner, registry_db_path, two_marker_body_yaml)
        out_path = tmp_path / "generated_skeleton.yaml"
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "marker-body", "to-skeleton", body_id, "--output", str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        parsed = yaml.safe_load(out_path.read_text(encoding="utf-8"))
        assert parsed["name"] == "test-bokken"

    def test_not_found(self, cli_runner: CliRunner, registry_db_path: Path) -> None:
        result = cli_runner.invoke(
            main,
            ["--registry", str(registry_db_path), "marker-body", "to-skeleton", "deadbeef"],
        )
        assert result.exit_code != 0

    def test_empty_marker_body_errors(
        self, cli_runner: CliRunner, registry_db_path: Path, tmp_path: Path,
    ) -> None:
        empty_path = tmp_path / "empty.yaml"
        empty_path.write_text("name: empty-body\nunits: meters\nmarkers: []\n", encoding="utf-8")
        body_id = self._import(cli_runner, registry_db_path, empty_path)

        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "marker-body", "to-skeleton", body_id],
        )
        assert result.exit_code != 0
        assert "no markers" in result.output

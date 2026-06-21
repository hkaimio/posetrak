"""Tests for skeleton CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from posetrak.cli.main import main
from posetrak.db.db import create_registry, open_registry


# ---------------------------------------------------------------------------
# Minimal skeleton YAML fixture
# ---------------------------------------------------------------------------


MINIMAL_SKELETON_YAML = """\
name: test_skeleton
units: meters
joints:
- name: root
  type: root
  parent: null
  offset: [0.0, 0.0, 0.9]
  bone_tip_offset: [0.0, 0.1, 0.0]
- name: chest
  type: ball
  parent: root
  offset: [0.0, 0.1, 0.0]
  bone_tip_offset: [0.0, 0.2, 0.0]
  limits: [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
markers: []
"""

SCALABLE_SKELETON_YAML = """\
name: scalable_skeleton
units: meters
joints:
- name: hips
  type: root
  parent: null
  offset: [0.0, 0.0, 0.9]
  bone_tip_offset: [0.0, 0.1, 0.0]
- name: shoulder.L
  type: ball
  parent: hips
  offset: [-0.2, 0.4, 0.0]
  bone_tip_offset: [0.0, 0.0, 0.0]
  limits: [[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]]
- name: shoulder.R
  type: ball
  parent: hips
  offset: [0.2, 0.4, 0.0]
  bone_tip_offset: [0.0, 0.0, 0.0]
  limits: [[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]]
- name: thigh.L
  type: ball
  parent: hips
  offset: [-0.1, -0.1, 0.0]
  bone_tip_offset: [0.0, -0.42, 0.0]
  limits: [[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]]
- name: thigh.R
  type: ball
  parent: hips
  offset: [0.1, -0.1, 0.0]
  bone_tip_offset: [0.0, -0.42, 0.0]
  limits: [[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]]
- name: shin.L
  type: ball
  parent: thigh.L
  offset: [0.0, -0.42, 0.0]
  bone_tip_offset: [0.0, -0.40, 0.0]
  limits: [[-1.5, 0.1], [-0.3, 0.3], [-0.3, 0.3]]
- name: shin.R
  type: ball
  parent: thigh.R
  offset: [0.0, -0.42, 0.0]
  bone_tip_offset: [0.0, -0.40, 0.0]
  limits: [[-1.5, 0.1], [-0.3, 0.3], [-0.3, 0.3]]
- name: foot.L
  type: ball
  parent: shin.L
  offset: [0.0, -0.40, 0.0]
  bone_tip_offset: [0.0, 0.0, 0.1]
  limits: [[-0.5, 0.5], [-0.3, 0.3], [-0.5, 0.5]]
- name: foot.R
  type: ball
  parent: shin.R
  offset: [0.0, -0.40, 0.0]
  bone_tip_offset: [0.0, 0.0, 0.1]
  limits: [[-0.5, 0.5], [-0.3, 0.3], [-0.5, 0.5]]
- name: upper_arm.L
  type: ball
  parent: shoulder.L
  offset: [-0.15, 0.0, 0.0]
  bone_tip_offset: [0.0, -0.30, 0.0]
  limits: [[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]]
- name: upper_arm.R
  type: ball
  parent: shoulder.R
  offset: [0.15, 0.0, 0.0]
  bone_tip_offset: [0.0, -0.30, 0.0]
  limits: [[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]]
- name: forearm.L
  type: ball
  parent: upper_arm.L
  offset: [0.0, -0.30, 0.0]
  bone_tip_offset: [0.0, -0.28, 0.0]
  limits: [[-0.1, 1.5], [-0.5, 0.5], [-0.5, 0.5]]
- name: forearm.R
  type: ball
  parent: upper_arm.R
  offset: [0.0, -0.30, 0.0]
  bone_tip_offset: [0.0, -0.28, 0.0]
  limits: [[-0.1, 1.5], [-0.5, 0.5], [-0.5, 0.5]]
- name: hand.L
  type: ball
  parent: forearm.L
  offset: [0.0, -0.28, 0.0]
  bone_tip_offset: [0.0, -0.1, 0.0]
  limits: [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]
- name: hand.R
  type: ball
  parent: forearm.R
  offset: [0.0, -0.28, 0.0]
  bone_tip_offset: [0.0, -0.1, 0.0]
  limits: [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]
markers: []
"""


@pytest.fixture()
def skeleton_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "test_skeleton.yaml"
    path.write_text(MINIMAL_SKELETON_YAML, encoding="utf-8")
    return path


@pytest.fixture()
def scalable_skeleton_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "scalable_skeleton.yaml"
    path.write_text(SCALABLE_SKELETON_YAML, encoding="utf-8")
    return path


class TestSkeletonImport:
    def test_import_to_registry(
        self,
        cli_runner: CliRunner,
        registry_db_path: Path,
        skeleton_yaml: Path,
    ) -> None:
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "skeleton", "import",
                "--file", str(skeleton_yaml),
                "--global",
                "--name", "My Test Skeleton",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "skeleton_id:" in result.output

    def test_import_to_session(
        self,
        cli_runner: CliRunner,
        registry_db_path: Path,
        session_db_path: Path,
        skeleton_yaml: Path,
    ) -> None:
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "--session", str(session_db_path),
                "skeleton", "import",
                "--file", str(skeleton_yaml),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "skeleton_id:" in result.output

    def test_import_requires_target(
        self,
        cli_runner: CliRunner,
        registry_db_path: Path,
        skeleton_yaml: Path,
    ) -> None:
        # No --global and no --session — should fail.
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "skeleton", "import",
                "--file", str(skeleton_yaml),
            ],
        )
        assert result.exit_code != 0


class TestSkeletonList:
    def test_empty(
        self, cli_runner: CliRunner, registry_db_path: Path
    ) -> None:
        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "skeleton", "list"]
        )
        assert result.exit_code == 0
        assert "No skeletons" in result.output

    def test_lists_after_import(
        self,
        cli_runner: CliRunner,
        registry_db_path: Path,
        skeleton_yaml: Path,
    ) -> None:
        cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "skeleton", "import",
                "--file", str(skeleton_yaml),
                "--global",
                "--name", "My Skeleton",
            ],
        )
        result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "skeleton", "list"]
        )
        assert result.exit_code == 0
        assert "My Skeleton" in result.output

    def test_json_mode(
        self,
        cli_runner: CliRunner,
        registry_db_path: Path,
        skeleton_yaml: Path,
    ) -> None:
        cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "skeleton", "import",
                "--file", str(skeleton_yaml),
                "--global",
                "--name", "JSON Skel",
            ],
        )
        result = cli_runner.invoke(
            main,
            ["--registry", str(registry_db_path), "--json", "skeleton", "list"],
        )
        assert result.exit_code == 0
        obj = json.loads(result.output.strip())
        assert "id" in obj
        assert "name" in obj


class TestSkeletonScale:
    def test_scale_name_saves_to_db(
        self,
        cli_runner: CliRunner,
        registry_db_path: Path,
        scalable_skeleton_yaml: Path,
    ) -> None:
        # First import the skeleton to registry.
        import_result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "skeleton", "import",
                "--file", str(scalable_skeleton_yaml),
                "--global",
                "--name", "Scalable",
            ],
        )
        assert import_result.exit_code == 0, import_result.output
        skeleton_id = import_result.output.split("skeleton_id:")[-1].strip()

        # Scale it.
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "skeleton", "scale",
                skeleton_id,
                "--femur", "0.45",
                "--shin", "0.40",
                "--name", "Scalable-human",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "skeleton_id:" in result.output
        assert "Scalable-human" in result.output

        # Verify it shows up in list.
        list_result = cli_runner.invoke(
            main, ["--registry", str(registry_db_path), "skeleton", "list"]
        )
        assert "Scalable-human" in list_result.output

    def test_scale_output_stdout(
        self,
        cli_runner: CliRunner,
        registry_db_path: Path,
        scalable_skeleton_yaml: Path,
    ) -> None:
        import_result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "skeleton", "import",
                "--file", str(scalable_skeleton_yaml),
                "--global",
                "--name", "Scalable2",
            ],
        )
        assert import_result.exit_code == 0, import_result.output
        skeleton_id = import_result.output.split("skeleton_id:")[-1].strip()

        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "skeleton", "scale",
                skeleton_id,
                "--femur", "0.50",
                "--output", "-",
            ],
        )
        assert result.exit_code == 0, result.output
        # The YAML content should contain the skeleton name.
        assert "scalable_skeleton" in result.output
        # joints key should appear in YAML output.
        assert "joints:" in result.output

    def test_scale_requires_name_or_output(
        self,
        cli_runner: CliRunner,
        registry_db_path: Path,
        scalable_skeleton_yaml: Path,
    ) -> None:
        import_result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "skeleton", "import",
                "--file", str(scalable_skeleton_yaml),
                "--global",
            ],
        )
        skeleton_id = import_result.output.split("skeleton_id:")[-1].strip()
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "skeleton", "scale",
                skeleton_id,
                "--femur", "0.45",
            ],
        )
        assert result.exit_code != 0

    def test_scale_rejects_both_name_and_output(
        self,
        cli_runner: CliRunner,
        registry_db_path: Path,
        scalable_skeleton_yaml: Path,
    ) -> None:
        import_result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "skeleton", "import",
                "--file", str(scalable_skeleton_yaml),
                "--global",
            ],
        )
        skeleton_id = import_result.output.split("skeleton_id:")[-1].strip()
        result = cli_runner.invoke(
            main,
            [
                "--registry", str(registry_db_path),
                "skeleton", "scale",
                skeleton_id,
                "--femur", "0.45",
                "--name", "foo",
                "--output", "-",
            ],
        )
        assert result.exit_code != 0

"""Tests for python/tools/upgrade_skeleton_hand_groups.py -- the
hierarchical-solver skeleton groups: converter (see
docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md and
docs/skeleton-format.md's "Hierarchical solver fields" section).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
import sys
from pathlib import Path

import yaml

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "upgrade_skeleton_hand_groups.py"
)
_spec = importlib.util.spec_from_file_location("upgrade_skeleton_hand_groups", _MODULE_PATH)
upgrade_skeleton_hand_groups = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = upgrade_skeleton_hand_groups
_spec.loader.exec_module(upgrade_skeleton_hand_groups)

upgrade_groups = upgrade_skeleton_hand_groups.upgrade_groups
upgrade_yaml_text = upgrade_skeleton_hand_groups.upgrade_yaml_text
upgrade_db = upgrade_skeleton_hand_groups.upgrade_db


def _matching_skeleton_yaml() -> str:
    """A minimal skeleton matching the reallusion-style family this script
    targets: has main/HandL/HandR groups, hand.{L,R}/forearm.{L,R} joints,
    and the exact stale references the design doc flagged (palm.* joints,
    MRK-thumb2 markers)."""
    return """
name: test_skeleton
joints:
  - name: root
    type: root
    offset: [0, 0, 0]
  - name: forearm.L
    type: revolute
    parent: root
    offset: [0.3, 0, 0]
    axis: [1, 0, 0]
    limits: [-1.5, 1.5]
  - name: hand.L
    type: revolute
    parent: forearm.L
    offset: [0.1, 0, 0]
    axis: [1, 0, 0]
    limits: [-1.5, 1.5]
  - name: thumb.01.L
    type: revolute
    parent: hand.L
    offset: [0.02, 0, 0]
    axis: [1, 0, 0]
    limits: [-1.5, 1.5]
  - name: forearm.R
    type: revolute
    parent: root
    offset: [-0.3, 0, 0]
    axis: [1, 0, 0]
    limits: [-1.5, 1.5]
  - name: hand.R
    type: revolute
    parent: forearm.R
    offset: [-0.1, 0, 0]
    axis: [1, 0, 0]
    limits: [-1.5, 1.5]
  - name: thumb.01.R
    type: revolute
    parent: hand.R
    offset: [-0.02, 0, 0]
    axis: [1, 0, 0]
    limits: [-1.5, 1.5]
markers:
  - name: MRK-wrist.L
    parent: hand.L
    offset: [0, 0, 0]
  - name: MRK-thumb.L
    parent: thumb.01.L
    offset: [0, 0, 0]
  - name: MRK-wrist.R
    parent: hand.R
    offset: [0, 0, 0]
  - name: MRK-thumb.R
    parent: thumb.01.R
    offset: [0, 0, 0]
groups:
  - name: main
    joints: [root, forearm.L, hand.L, forearm.R, hand.R, palm.01.L, palm.04.L, palm.01.R, palm.04.R]
    markers: [MRK-wrist.L, MRK-wrist.R]
    optional: false
  - name: HandL
    depends_on: main
    joints: [palm.01.L, palm.02.L, palm.03.L, palm.04.L, thumb.01.L]
    markers: [MRK-thumb.L, MRK-thumb2.L]
  - name: HandR
    depends_on: main
    joints: [palm.01.R, palm.02.R, palm.03.R, palm.04.R, thumb.01.R]
    markers: [MRK-thumb.R, MRK-thumb2.R]
"""


def _real_palm_joints_skeleton_yaml() -> str:
    """A different topology from _matching_skeleton_yaml(): palm.01.{L,R} are
    REAL joints here (fingers attach to them, not directly to hand.{L,R}) --
    this is the exact shape of tests/data/Harri_skeleton-regress-test.yaml,
    which an earlier version of this script corrupted by assuming palm.*
    references are always phantom. main's reference to palm.01.L/palm.01.R
    is legitimate here and must survive unchanged; HandL/HandR must NOT gain
    hand.{L,R}, MRK-wrist.{side}, freeflyer_joint, or ref_marker, since the
    design doc never analyzed this topology."""
    return """
name: test_skeleton_real_palm
joints:
  - name: root
    type: root
    offset: [0, 0, 0]
  - name: forearm.L
    type: revolute
    parent: root
    offset: [0.3, 0, 0]
    axis: [1, 0, 0]
    limits: [-1.5, 1.5]
  - name: hand.L
    type: revolute
    parent: forearm.L
    offset: [0.1, 0, 0]
    axis: [1, 0, 0]
    limits: [-1.5, 1.5]
  - name: palm.01.L
    type: revolute
    parent: hand.L
    offset: [0.02, 0, 0]
    axis: [1, 0, 0]
    limits: [-1.5, 1.5]
  - name: f_index.01.L
    type: revolute
    parent: palm.01.L
    offset: [0.02, 0, 0]
    axis: [1, 0, 0]
    limits: [-1.5, 1.5]
  - name: forearm.R
    type: revolute
    parent: root
    offset: [-0.3, 0, 0]
    axis: [1, 0, 0]
    limits: [-1.5, 1.5]
  - name: hand.R
    type: revolute
    parent: forearm.R
    offset: [-0.1, 0, 0]
    axis: [1, 0, 0]
    limits: [-1.5, 1.5]
  - name: palm.01.R
    type: revolute
    parent: hand.R
    offset: [-0.02, 0, 0]
    axis: [1, 0, 0]
    limits: [-1.5, 1.5]
  - name: f_index.01.R
    type: revolute
    parent: palm.01.R
    offset: [-0.02, 0, 0]
    axis: [1, 0, 0]
    limits: [-1.5, 1.5]
markers:
  - name: MRK-index.L
    parent: f_index.01.L
    offset: [0, 0, 0]
  - name: MRK-index.R
    parent: f_index.01.R
    offset: [0, 0, 0]
groups:
  - name: main
    joints: [root, forearm.L, hand.L, palm.01.L, forearm.R, hand.R, palm.01.R, ghost_joint]
    markers: []
    optional: false
  - name: HandL
    depends_on: main
    joints: [palm.01.L, f_index.01.L]
    markers: [MRK-index.L]
  - name: HandR
    depends_on: main
    joints: [palm.01.R, f_index.01.R]
    markers: [MRK-index.R]
"""


def _non_matching_skeleton_yaml() -> str:
    """No hand.L/forearm.L -- the converter must leave this alone."""
    return """
name: unrelated_skeleton
joints:
  - name: root
    type: root
    offset: [0, 0, 0]
markers: []
groups:
  - name: main
    joints: [root]
    markers: []
  - name: HandL
    joints: []
    markers: []
  - name: HandR
    joints: []
    markers: []
"""


def test_upgrade_groups_corrects_a_matching_skeleton():
    skeleton = yaml.safe_load(_matching_skeleton_yaml())
    changes, warnings = upgrade_groups(skeleton)

    assert changes
    assert warnings == []

    groups = {g["name"]: g for g in skeleton["groups"]}
    assert "palm.01.L" not in groups["main"]["joints"]
    assert "palm.04.R" not in groups["main"]["joints"]

    handl = groups["HandL"]
    assert handl["joints"] == ["hand.L", "thumb.01.L"]
    assert "MRK-thumb2.L" not in handl["markers"]
    assert "MRK-wrist.L" in handl["markers"]
    assert handl["freeflyer_joint"] == "forearm.L"
    assert handl["ref_marker"] == "MRK-wrist.L"

    handr = groups["HandR"]
    assert handr["joints"] == ["hand.R", "thumb.01.R"]
    assert handr["freeflyer_joint"] == "forearm.R"
    assert handr["ref_marker"] == "MRK-wrist.R"


def test_upgrade_groups_leaves_hand_membership_alone_when_palm_joints_are_real():
    """Regression test for the exact bug this script shipped and then fixed:
    a skeleton whose fingers attach via real palm.0N.{side} joints (not
    directly to hand.{side}) must NOT have those joints stripped from
    HandL/HandR, must NOT gain hand.{side}/MRK-wrist.{side}/freeflyer_joint/
    ref_marker, but a genuinely stale reference elsewhere (ghost_joint) must
    still be removed."""
    skeleton = yaml.safe_load(_real_palm_joints_skeleton_yaml())
    changes, warnings = upgrade_groups(skeleton)

    assert any("ghost_joint" in c for c in changes)
    assert any("NOTE" in w and "palm.0N" in w for w in warnings)

    groups = {g["name"]: g for g in skeleton["groups"]}
    assert "ghost_joint" not in groups["main"]["joints"]
    assert "palm.01.L" in groups["main"]["joints"]  # legitimate, untouched

    handl = groups["HandL"]
    assert handl["joints"] == ["palm.01.L", "f_index.01.L"]  # unchanged
    assert "hand.L" not in handl["joints"]
    assert "MRK-wrist.L" not in handl["markers"]
    assert "freeflyer_joint" not in handl
    assert "ref_marker" not in handl

    handr = groups["HandR"]
    assert handr["joints"] == ["palm.01.R", "f_index.01.R"]
    assert "freeflyer_joint" not in handr


def test_upgrade_groups_skips_a_non_matching_skeleton():
    skeleton = yaml.safe_load(_non_matching_skeleton_yaml())
    original = yaml.safe_load(_non_matching_skeleton_yaml())

    changes, warnings = upgrade_groups(skeleton)

    assert changes == []
    assert warnings == []
    assert skeleton == original  # untouched


def test_upgrade_groups_is_idempotent():
    skeleton = yaml.safe_load(_matching_skeleton_yaml())
    upgrade_groups(skeleton)  # first pass: applies corrections in place

    changes, warnings = upgrade_groups(skeleton)  # second pass: nothing left to do
    assert changes == []
    assert warnings == []


def test_upgrade_yaml_text_only_rewrites_the_groups_section():
    original = _matching_skeleton_yaml()
    new_text, changes, warnings = upgrade_yaml_text(original)

    assert changes
    assert warnings == []
    # Every non-groups: line (joints:, markers:, comments) is untouched --
    # only the groups: block's own lines may differ.
    original_lines = original.splitlines()
    new_lines = new_text.splitlines()
    groups_idx_orig = next(i for i, l in enumerate(original_lines) if l.strip() == "groups:")
    assert original_lines[:groups_idx_orig] == new_lines[:groups_idx_orig]

    # Result still parses and round-trips through yaml.safe_load cleanly.
    reparsed = yaml.safe_load(new_text)
    assert reparsed["joints"][0]["name"] == "root"


def test_upgrade_yaml_text_second_pass_is_a_no_op():
    original = _matching_skeleton_yaml()
    upgraded_once, _, _ = upgrade_yaml_text(original)
    upgraded_twice, changes, warnings = upgrade_yaml_text(upgraded_once)

    assert changes == []
    assert upgraded_twice == upgraded_once  # byte-identical, no unnecessary write


def _make_registry_db(path: Path, skeleton_yaml: str) -> str:
    """Minimal DB with a single skeletons row. Returns its id (content hash,
    matching posetrak/db/manage_skeleton.py's convention)."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE skeletons (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, parent_id TEXT,
            person_label TEXT, source TEXT, yaml_content TEXT NOT NULL,
            created_at TEXT NOT NULL, notes TEXT
        );
    """)
    skeleton_id = hashlib.sha256(skeleton_yaml.encode("utf-8")).hexdigest()
    conn.execute(
        "INSERT INTO skeletons (id, name, yaml_content, created_at) VALUES (?, ?, ?, '2026-01-01')",
        (skeleton_id, "test_skeleton", skeleton_yaml),
    )
    conn.commit()
    conn.close()
    return skeleton_id


def test_upgrade_db_inserts_a_new_row_and_leaves_the_original_untouched(tmp_path):
    db_path = tmp_path / "registry.db"
    original_yaml = _matching_skeleton_yaml()
    original_id = _make_registry_db(db_path, original_yaml)

    upgrade_db(db_path, dry_run=False)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM skeletons").fetchall()
    conn.close()

    assert len(rows) == 2
    original_row = next(r for r in rows if r["id"] == original_id)
    assert original_row["yaml_content"] == original_yaml  # never mutated in place
    assert original_row["parent_id"] is None

    new_row = next(r for r in rows if r["id"] != original_id)
    assert new_row["parent_id"] == original_id
    assert new_row["id"] == hashlib.sha256(new_row["yaml_content"].encode("utf-8")).hexdigest()

    upgraded_groups = yaml.safe_load(new_row["yaml_content"])["groups"]
    handl = next(g for g in upgraded_groups if g["name"] == "HandL")
    assert handl["freeflyer_joint"] == "forearm.L"


def test_upgrade_db_dry_run_makes_no_changes(tmp_path):
    db_path = tmp_path / "registry.db"
    original_yaml = _matching_skeleton_yaml()
    _make_registry_db(db_path, original_yaml)

    upgrade_db(db_path, dry_run=True)

    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM skeletons").fetchone()[0]
    conn.close()
    assert count == 1


def test_upgrade_db_is_idempotent(tmp_path):
    db_path = tmp_path / "registry.db"
    original_yaml = _matching_skeleton_yaml()
    _make_registry_db(db_path, original_yaml)

    upgrade_db(db_path, dry_run=False)
    upgrade_db(db_path, dry_run=False)  # second pass: no new row

    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM skeletons").fetchone()[0]
    conn.close()
    assert count == 2

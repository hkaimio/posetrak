# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""
blender_align_skeleton.py — Align a template armature to a CC character armature in Blender.

Run this script from Blender's text editor (Text > Run Script) after importing the CC
character.  Both armatures must already be in the scene.

For each template joint listed in the bone map, the script finds the corresponding CC bone
and sets the template bone's head, tail and roll to match the CC bone's world-space
transforms.  Bones not in the bone map are left untouched.

Configuration:
    Edit the CONFIG block at the top of this file, or call align_skeleton() directly with
    keyword arguments if integrating into an extension later.

Bone map format  (JSON):
    {
        "<template_joint_name>": "<cc_bone_name>",
        "hips":       "CC_Base_Hip",
        "waist":      "CC_Base_Waist",
        ...
    }
"""

import json
from pathlib import Path

import bpy
import mathutils

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

CONFIG = dict(
    # Name of the template armature object in the Blender scene
    template_armature="Armature",

    # Name of the CC character armature object in the Blender scene
    cc_armature="CC_Base_Body",

    # Path to the bone map JSON file
    bone_map_path="/home/harri/projects/posetrak/scripts/reallusion_bone_map.json",

    # Whether to copy roll angle from the CC bone.
    # Keep False (default) so that template bone local frames are preserved
    # and existing DOF limits remain semantically correct.
    copy_roll=False,
)

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _collect_cc_transforms(cc_obj: bpy.types.Object) -> dict[str, dict]:
    """Switch cc_obj into edit mode and collect world-space head/tail/roll for all bones.

    Returns a dict  {bone_name: {"head": Vector, "tail": Vector, "roll": float}}.
    Leaves the object in OBJECT mode.
    """
    # Make cc_obj the active object and enter edit mode
    bpy.ops.object.select_all(action="DESELECT")
    cc_obj.select_set(True)
    bpy.context.view_layer.objects.active = cc_obj
    bpy.ops.object.mode_set(mode="EDIT")

    mat = cc_obj.matrix_world
    transforms = {}
    for eb in cc_obj.data.edit_bones:
        transforms[eb.name] = {
            "head": mat @ eb.head.copy(),
            "tail": mat @ eb.tail.copy(),
            "roll": eb.roll,
        }

    bpy.ops.object.mode_set(mode="OBJECT")
    return transforms


def _apply_to_template(
    template_obj: bpy.types.Object,
    cc_transforms: dict[str, dict],
    bone_map: dict[str, str],
    copy_roll: bool,
) -> tuple[list[str], list[str], list[str]]:
    """Enter template edit mode and reposition mapped bones.

    Returns (aligned, missing_cc, missing_template) name lists.
    Leaves the object in OBJECT mode.
    """
    bpy.ops.object.select_all(action="DESELECT")
    template_obj.select_set(True)
    bpy.context.view_layer.objects.active = template_obj
    bpy.ops.object.mode_set(mode="EDIT")

    mat_inv = template_obj.matrix_world.inverted()
    edit_bones = template_obj.data.edit_bones

    aligned: list[str] = []
    missing_cc: list[str] = []
    missing_template: list[str] = []

    for tmpl_name, cc_name in bone_map.items():
        if cc_name not in cc_transforms:
            missing_cc.append(f"{tmpl_name} → {cc_name}")
            continue

        if tmpl_name not in edit_bones:
            missing_template.append(tmpl_name)
            continue

        cc_t = cc_transforms[cc_name]
        eb = edit_bones[tmpl_name]

        # Disconnect from parent so we can set head freely
        was_connected = eb.use_connect
        eb.use_connect = False

        eb.head = mat_inv @ cc_t["head"]
        eb.tail = mat_inv @ cc_t["tail"]
        if copy_roll:
            eb.roll = cc_t["roll"]

        # Leave disconnected — the template hierarchy is maintained through
        # parent references, not connectivity constraints, since CC proportions
        # will differ from the original template spacing.
        _ = was_connected  # intentionally not restoring

        aligned.append(f"{tmpl_name} ← {cc_name}")

    bpy.ops.object.mode_set(mode="OBJECT")
    return aligned, missing_cc, missing_template


def align_skeleton(
    template_armature: str,
    cc_armature: str,
    bone_map_path: str,
    copy_roll: bool = False,
) -> None:
    """Main entry point.  All arguments correspond to CONFIG keys."""

    # --- Validate objects ---
    template_obj = bpy.data.objects.get(template_armature)
    cc_obj = bpy.data.objects.get(cc_armature)

    if template_obj is None:
        raise ValueError(f"Template armature not found in scene: '{template_armature}'")
    if cc_obj is None:
        raise ValueError(f"CC armature not found in scene: '{cc_armature}'")
    if template_obj.type != "ARMATURE":
        raise ValueError(f"'{template_armature}' is not an armature")
    if cc_obj.type != "ARMATURE":
        raise ValueError(f"'{cc_armature}' is not an armature")

    # --- Load bone map ---
    map_path = Path(bone_map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"Bone map not found: {map_path}")
    with open(map_path) as f:
        bone_map: dict[str, str] = json.load(f)
    print(f"Loaded bone map: {len(bone_map)} entries from {map_path}")

    # --- Collect CC transforms ---
    print(f"Collecting transforms from CC armature '{cc_armature}' ...")
    cc_transforms = _collect_cc_transforms(cc_obj)
    print(f"  Found {len(cc_transforms)} CC bones")

    # --- Apply to template ---
    print(f"Aligning template armature '{template_armature}' ...")
    aligned, missing_cc, missing_tmpl = _apply_to_template(
        template_obj, cc_transforms, bone_map, copy_roll
    )

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"Aligned {len(aligned)} / {len(bone_map)} bones")
    print(f"{'='*60}")
    for entry in aligned:
        print(f"  OK  {entry}")
    if missing_cc:
        print(f"\nCC bones not found ({len(missing_cc)}) — check CC armature name or bone map:")
        for entry in missing_cc:
            print(f"  MISSING-CC  {entry}")
    if missing_tmpl:
        print(f"\nTemplate bones not found ({len(missing_tmpl)}) — check template armature:")
        for entry in missing_tmpl:
            print(f"  MISSING-TMPL  {entry}")
    if not missing_cc and not missing_tmpl:
        print("\nAll bones aligned successfully.")

    # Restore a clean selection state
    bpy.ops.object.select_all(action="DESELECT")
    template_obj.select_set(True)
    bpy.context.view_layer.objects.active = template_obj


# ---------------------------------------------------------------------------
# Run when executed as a Blender text-editor script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    align_skeleton(**CONFIG)

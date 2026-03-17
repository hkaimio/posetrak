# Design: CC Character → Tracking YAML Export

## Overview

When a new subject is tracked, their skeleton proportions must be captured accurately so
the kinematic model matches the real person.  The current workflow produces a scaled
skeleton YAML by running an automated bone-length calibration pass over tracking data.
This works well for most joints but produces poor results for the spine (incorrect neutral
curvature when the chain is even slightly too long) and neck (systematically exaggerated).

The proposed workflow replaces the problematic calibration steps with geometry extracted
from a Character Creator (CC) character:

1. **Measure** — use the `key-measurements` Marimo notebook to read 7 key body dimensions
   from tracking data and export `measurements.json`.
2. **Build** — open the subject's CC character, use the measurements to set body proportions
   manually in Character Creator, and export the result to a `.blend` file.
3. **Align** — with both the template armature and CC armature in the Blender scene, run the
   **CC Align** operator (or CLI command) to copy bone head and tail positions from CC bones
   to the corresponding template bones, leaving roll angles unchanged.
4. **Export** — export the now-repositioned **template armature** to YAML using the existing
   export infrastructure.  DOF limits, joint types, marker definitions, and group structure
   are read directly from the template armature's existing config — no template YAML parsing
   is needed.

The result is a drop-in replacement for the calibrated YAML.  All tracker configuration
(limits, markers, groups) is preserved because the template armature's local bone frames
are preserved; only bone lengths and global positions change.

### Why copy head+tail but not roll

DOF limits in the existing tracker configs (e.g. `cc5.py`) are specified in the **local
coordinate frame of each bone**.  In Blender, a bone's local X and Z axes (around which
roll limits are measured) are determined by its roll angle.  The CC armature may use
different roll conventions for some bones.  If roll were copied from CC bones, the same
numerical limits would constrain rotation around physically different axes on the template
armature, silently producing wrong DOF boundaries.

By copying only head and tail (which set bone direction and length) and keeping the
template's original roll, the local frames — and therefore the semantics of all limits —
remain correct.

---

## Requirements

### Functional

- Given a Blender scene containing both a template armature and a CC character armature,
  and a bone-map JSON, reposition each mapped template bone so its head and tail match
  the world-space head and tail of the corresponding CC bone, leaving roll unchanged.
- Joints listed in `_no_cc_equivalent` in the bone map are silently skipped.
- After alignment, the template armature can be exported with the existing
  `export_armature_to_skeleton_yaml` pipeline without any modifications to `config.py`,
  `export.py`, or other exporter files.

### Interfaces

The align-and-export workflow must be runnable in two ways without duplicating logic:

**A. Blender operator (interactive)**
Invoked from the Blender UI or F3 search.  The user selects the template armature as the
active object, picks the CC armature from a dropdown, selects the bone-map JSON and output
path via file pickers, and clicks Run.  The operator aligns the template bones and then
immediately exports.

**B. Command-line script (batch / CI)**
Invoked as:
```
blender --background <scene.blend> \
    --python <path-to-extension>/skeleton_yaml/cc_align_cli.py -- \
    --template-armature  "Armature" \
    --cc-armature        "CC_Base_Body" \
    --bone-map           reallusion_bone_map.json \
    --output             harri-from-cc.yaml
```
The `.blend` file contains both armatures already in the scene.  All logic lives in
`cc_align.py`; the CLI script is a thin argument-parsing wrapper.

---

## Architecture

### Key insight

`get_bone_transform` in the existing exporter computes parent-relative offsets and
orientations from `pose_bone.matrix`, which reflects the bone's actual world-space
position after the alignment step.  No changes to the exporter are needed — aligning the
template bones in edit mode and then running the existing export is sufficient.

### New files

#### `skeleton_yaml/cc_align.py`

Core alignment logic.  No `bpy` type annotations in public signatures (for testability),
but does import `bpy` at runtime since it executes inside Blender.

```python
def align_template_to_cc(
    template_armature_name: str,
    cc_armature_name: str,
    bone_map_path: str,
) -> dict:
    """Copy head+tail world positions from CC bones to template bones.

    Roll angles are intentionally NOT copied, so that existing DOF limit
    axes (defined in template bone local frames) remain semantically correct.

    Returns a summary dict with keys 'aligned', 'missing_cc', 'missing_template'.
    """
    ...
```

Steps:

1. Look up both armature objects in `bpy.data.objects`; validate both are ARMATURE type.
2. Load `bone_map.json`; strip `_comment` and `_no_cc_equivalent` keys to build
   `{template_name: cc_name}`.
3. Enter edit mode on the CC armature; collect world-space `{cc_name: {"head": Vector, "tail": Vector}}`
   for all bones.  Leave CC armature in OBJECT mode.
4. Enter edit mode on the template armature.  For each `(template_name, cc_name)` in the map:
   - Set `eb.use_connect = False` (disconnect from parent so head can be set freely).
   - Set `eb.head = template_matrix_world_inv @ cc_transforms[cc_name]["head"]`.
   - Set `eb.tail = template_matrix_world_inv @ cc_transforms[cc_name]["tail"]`.
   - Do **not** set `eb.roll` — preserve existing value.
5. Leave template armature in OBJECT mode.
6. Return summary.

#### `skeleton_yaml/cc_align_cli.py`

Thin CLI wrapper — no alignment logic.

```python
def main() -> None:
    # Parse sys.argv after "--" (Blender passes extra args after --)
    # Call align_template_to_cc(...)
    # Determine export config from active template armature preset
    # Call export_armature_to_skeleton_yaml(template_armature_name, config, output)
    # Print progress; exit(1) on error

if __name__ == "__main__":
    main()
```

### Modified files

#### `skeleton_yaml/operators.py`

Add `PE_OT_AlignAndExportFromCC`:

```
bl_idname:  "pose_editor.align_and_export_from_cc"
bl_label:   "Align Template to CC and Export"

Properties:
  filepath            FILE_PATH  — output .yaml
  cc_armature_name    STRING     — name of CC armature object in scene
  bone_map            FILE_PATH  — reallusion_bone_map.json (default: bundled path)

invoke:  open file browser for filepath (pre-fill from active armature name)
execute:
  1. template_armature_name = context.active_object.name  (validate it's ARMATURE)
  2. align_template_to_cc(template_armature_name, cc_armature_name, bone_map)
  3. config = get_config_for_armature(template_armature_name)  [existing preset lookup]
  4. export_armature_to_skeleton_yaml(template_armature_name, config, filepath)
  5. self.report({'INFO'}, f"Exported {filepath}")
```

Register alongside the existing `PE_OT_ExportSkeletonYAML` in `register()` /
`unregister()`.

### Files not touched

`config.py`, `export.py`, `converters.py`, `hierarchy.py`, `yaml_builder.py`, `opensim/`,
existing configs (`cc5.py`, `simple.py`, `auto_config.py`).

The existing `blender_align_skeleton.py` (standalone text-editor script in the posetrak
repo) implements the alignment logic independently and can be used for validation until
the extension operator is integrated.

---

## Implementation order for the agent

1. Read `skeleton_yaml/operators.py`, `skeleton_yaml/export.py`, and one existing config
   (e.g. `configs/cc5.py`) to understand the preset lookup and export call signatures.
2. Write `skeleton_yaml/cc_align.py` — core alignment logic only, no export.
3. Write `skeleton_yaml/cc_align_cli.py` — thin wrapper.
4. Add `PE_OT_AlignAndExportFromCC` to `operators.py` and register it.
5. Smoke-test: run `cc_align_cli.py` against a `.blend` file that has both armatures,
   diff the output YAML against a reference to verify joint names, marker count, group
   structure, and that bone lengths match the CC character.

# Posetrak rig profile for BVH Retargeter (Blender)

Drop-in rig files for the [BVH and FBX Retargeter](https://github.com/Diffeomorphic/retarget_bvh/wiki)
Blender extension, so it recognizes Posetrak's exported skeleton and
applies a correct rest T-pose instead of guessing one.

See the full walkthrough: [Retargeting to Blender](../../../../docs/user-guide/retargeting-blender.md).

## What's here

| File | Goes into (inside the extension's install folder) | Purpose |
|---|---|---|
| `known_rigs/posetrak.json` | `known_rigs/posetrak.json` | Maps Posetrak's joint names to BVH Retargeter's canonical bone names, so source-rig identification (including the two-bone neck) works without manual bone-by-bone assignment. |
| `t_poses/posetrak.json` | `t_poses/posetrak.json` | The exact rest-pose rotation of every joint in Posetrak's T-pose, captured directly from a real export. Needed because BVH Retargeter's automatic T-pose guess doesn't reorient clavicle or foot bones, and Posetrak's BVH rest-pose bone directions don't happen to be anatomically correct for those joints. |

Both files use Posetrak's default skeleton (as produced by
`export_bvh.py` / `posetrak.export.bvh`) and its default joint names.
If you're tracking a custom skeleton YAML with renamed or restructured
joints, treat these as templates — see the full guide for how to
regenerate them for your own rig.

## Install

Copy both `known_rigs/posetrak.json` and `t_poses/posetrak.json` into
the matching subfolders of your BVH Retargeter installation, then run
**BVH Retargeter → Init Known Rigs** in Blender (or just restart
Blender). Details, including where to find the install folder, are in
the guide linked above.

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""blender_add_pen_pad_animation.py — PROTOTYPE: build animated "pen" and
"pad" rigid objects (plus a static "pen_tip" child empty) from
export_pen_pad_to_blender.py's keyframe JSON.

Run inside Blender (headless: `blender --background FILE.blend --python
blender_add_pen_pad_animation.py`), not with plain python -- needs bpy.
Same two-stage convention as blender_add_cameras.py: a plain-Python
script builds the JSON, this one only consumes it.

Deliberately exports "pen" and "pad" as full rigid objects with real
position + rotation animation (not a pre-computed world-space tip
trajectory) -- per Harri's own framing, Blender's own dynamic-paint
system should be able to detect the pen-tip-to-pad proximity itself once
both real rigid objects are present, the same way it would for any other
brush/canvas pair. "pen_tip" is parented to "pen" at a fixed local offset
(never keyframed itself) -- Blender's own parenting propagates the pen's
full animated pose to it automatically.

Reads scratch/blender_pen_pad_keyframes.json (built by
export_pen_pad_to_blender.py, marker-based-mocap session, 2026-09-13).
"""

import json
import sys
from pathlib import Path

import bpy

CONFIG_PATH = Path(__file__).resolve().parents[2] / "scratch" / "blender_pen_pad_keyframes.json"


def _make_empty(name: str, parent=None, empty_type: str = "PLAIN_AXES", size: float = 0.05):
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = empty_type
    obj.empty_display_size = size
    obj.rotation_mode = "QUATERNION"
    bpy.context.collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


def add_pen_pad_animation(config_path: Path) -> dict[str, "bpy.types.Object"]:
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    objects = {}
    for obj_name in ("pen", "pad"):
        obj = _make_empty(obj_name, size=0.1 if obj_name == "pad" else 0.05)
        for kf in config[obj_name]:
            obj.location = kf["location"]
            obj.rotation_quaternion = kf["rotation_quaternion"]
            obj.keyframe_insert(data_path="location", frame=kf["frame"])
            obj.keyframe_insert(data_path="rotation_quaternion", frame=kf["frame"])
        # Constant (not linear) interpolation between real solved frames
        # would misrepresent gaps as held poses; leave Blender's default
        # (Bezier) -- smooths across the gaps, which is the honest
        # representation of "we don't actually know the pose there",
        # closer to the real motion than a step function.
        objects[obj_name] = obj
        print(f"{obj_name}: {len(config[obj_name])} keyframes")

    tip = _make_empty("pen_tip", parent=objects["pen"], empty_type="SPHERE", size=0.01)
    tip.location = config["pen_tip_local_offset"]
    objects["pen_tip"] = tip
    print(f"pen_tip: static local offset {config['pen_tip_local_offset']}")

    scene = bpy.context.scene
    scene.render.fps = round(config["fps"])
    all_frames = [kf["frame"] for name in ("pen", "pad") for kf in config[name]]
    if all_frames:
        scene.frame_start = min(all_frames)
        scene.frame_end = max(all_frames)

    return objects


if __name__ == "__main__":
    add_pen_pad_animation(CONFIG_PATH)
    save_path = None
    for i, a in enumerate(sys.argv):
        if a == "--save":
            save_path = sys.argv[i + 1]
    if save_path:
        bpy.ops.wm.save_as_mainfile(filepath=save_path)
        print(f"Saved -> {save_path}")

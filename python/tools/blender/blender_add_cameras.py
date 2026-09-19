# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""blender_add_cameras.py — PROTOTYPE: add posetrak's solved real cameras
into a Blender scene, positioned/oriented from a solved extrinsic
calibration, with FOV matched to each camera's real (undistorted)
intrinsics.

Run inside Blender (headless: `blender --background FILE.blend --python
blender_add_cameras.py`), not with plain python -- needs bpy.

Reads scratch/blender_camera_setup.json (built by a plain-Python DB query
script -- see the marker-based-mocap session that produced this, 2026-09-07)
with one entry per camera: Blender-space location + rotation_quaternion
(already converted from the tracker's Z-up OpenCV-convention extrinsics via
posetrak.export.common's yup transform, verified against a numerical
round-trip check before trusting it), plus fx/fy/cx/cy/width/height for the
lens.

Video background is NOT added here -- see blender_add_video_backgrounds.py
for that, once undistorted/downscaled proxy videos exist.
"""

import json
import sys
from pathlib import Path

import bpy

CONFIG_PATH = Path(__file__).resolve().parents[3] / "scratch" / "blender_camera_setup.json"
SENSOR_WIDTH_MM = 36.0  # arbitrary reference (full-frame-equivalent); only the ratio to lens matters


def add_cameras(config_path: Path) -> list[str]:
    with open(config_path, encoding="utf-8") as f:
        cams = json.load(f)

    created = []
    for c in cams:
        name = f"cam_{c['label']}"
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
        cam_data_name = f"camdata_{c['label']}"
        if cam_data_name in bpy.data.cameras:
            bpy.data.cameras.remove(bpy.data.cameras[cam_data_name])

        cam_data = bpy.data.cameras.new(cam_data_name)
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.sensor_width = SENSOR_WIDTH_MM
        cam_data.lens = c["fx"] * SENSOR_WIDTH_MM / c["width"]
        # Principal-point offset -> Blender lens shift (fraction of the
        # sensor's larger dimension -- Blender's own documented convention).
        max_dim = max(c["width"], c["height"])
        cam_data.shift_x = (c["width"] / 2.0 - c["cx"]) / max_dim
        cam_data.shift_y = (c["cy"] - c["height"] / 2.0) / max_dim

        cam_obj = bpy.data.objects.new(name, cam_data)
        cam_obj.location = c["location"]
        cam_obj.rotation_mode = "QUATERNION"
        cam_obj.rotation_quaternion = c["rotation_quaternion"]
        bpy.context.scene.collection.objects.link(cam_obj)
        created.append(name)
        print(f"Added {name}: loc={c['location']} lens={cam_data.lens:.1f}mm "
              f"({c['width']}x{c['height']}, fisheye={c.get('fisheye')})")

    return created


def render_camera_check(cam_names: list[str], out_dir: Path) -> None:
    """Render one still per camera (character only, no video yet) for a
    quick visual sanity check of placement/framing before investing in
    video preprocessing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_set(200)  # a frame comfortably past the rest pose, mid-motion
    orig_res_x, orig_res_y = scene.render.resolution_x, scene.render.resolution_y
    orig_cam = scene.camera
    for name in cam_names:
        obj = bpy.data.objects[name]
        scene.camera = obj
        scene.render.resolution_x = 960
        scene.render.resolution_y = int(960 * obj.data.sensor_height / obj.data.sensor_width) \
            if obj.data.sensor_fit == "VERTICAL" else 540
        scene.render.filepath = str(out_dir / f"{name}_check.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {scene.render.filepath}")
    scene.render.resolution_x, scene.render.resolution_y = orig_res_x, orig_res_y
    scene.camera = orig_cam


if __name__ == "__main__":
    names = add_cameras(CONFIG_PATH)
    do_render = "--render-check" in sys.argv
    if do_render:
        out = Path(__file__).resolve().parents[3] / "scratch" / "blender_camera_check"
        render_camera_check(names, out)
    save_path = None
    for i, a in enumerate(sys.argv):
        if a == "--save":
            save_path = sys.argv[i + 1]
    if save_path:
        bpy.ops.wm.save_as_mainfile(filepath=save_path)
        print(f"Saved -> {save_path}")

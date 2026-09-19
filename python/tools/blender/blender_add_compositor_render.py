# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""blender_add_compositor_render.py — PROTOTYPE: per-camera render scenes
that composite the animated character over its real undistorted video,
for actually rendering an output file (not just the interactive viewport
check blender_add_video_backgrounds.py already covers).

Run inside Blender after both blender_add_cameras.py and
blender_add_video_backgrounds.py (needs the cam_<label> objects and the
loaded movie clips both already in place; headless:
`blender --background FILE.blend --python blender_add_compositor_render.py`).

One Scene per camera (`render_<label>`), all linked to the SAME objects as
the main scene (armature, meshes, lights -- not copied, so the baked
animation drives every render scene identically) but with:
  - scene.camera set to that camera only
  - render.film_transparent on, so the character render layer carries an
    alpha channel the compositor can key over the real video
  - a compositor node tree: Movie Clip (background) -> Alpha Over
    (character layer keyed over it, using its own alpha) -> Composite

Switch the active scene (top of Blender's window, or
`bpy.context.window.scene = bpy.data.scenes["render_<label>"]`) and render
(F12 for a still, Ctrl+F12 for the full animation) to get that camera's
composited output.
"""

import sys
from pathlib import Path

import bpy

MANIFEST_PATH = Path(__file__).resolve().parents[3] / "scratch" / "blender_proxy_videos" / "manifest.json"
FIRST_TRACKED_FRAME = 2  # see blender_add_video_backgrounds.py's docstring


def build_render_scene(label: str, clip_path: str, main_scene: bpy.types.Scene) -> bpy.types.Scene:
    cam_obj = bpy.data.objects.get(f"cam_{label}")
    if cam_obj is None:
        raise RuntimeError(f"No camera object cam_{label} -- run blender_add_cameras.py first")

    scene_name = f"render_{label}"
    if scene_name in bpy.data.scenes:
        bpy.data.scenes.remove(bpy.data.scenes[scene_name])
    scene = bpy.data.scenes.new(scene_name)

    for obj in main_scene.objects:
        if obj.name not in scene.collection.objects:
            scene.collection.objects.link(obj)
    scene.camera = cam_obj

    scene.frame_start = main_scene.frame_start
    scene.frame_end = main_scene.frame_end
    scene.render.fps = main_scene.render.fps
    scene.render.film_transparent = True

    clip = bpy.data.movieclips.get(Path(clip_path).name) or bpy.data.movieclips.load(str(Path(clip_path).resolve()))
    clip.frame_start = FIRST_TRACKED_FRAME  # datablock-level: applies to every user of this clip
    w, h = clip.size[0], clip.size[1]
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100

    # Blender 5.x moved the compositor tree off Scene.node_tree (still
    # settable via the now-deprecated Scene.use_nodes, but that no longer
    # populates node_tree at all in this version -- confirmed by
    # introspecting Scene's own RNA properties) onto a real, separately
    # assigned NodeTree datablock: Scene.compositing_node_group.
    tree = bpy.data.node_groups.new(f"Compositing_{label}", "CompositorNodeTree")
    scene.compositing_node_group = tree

    n_clip = tree.nodes.new("CompositorNodeMovieClip")
    n_clip.clip = clip
    n_clip.location = (-400, 200)

    n_render = tree.nodes.new("CompositorNodeRLayers")
    n_render.scene = scene
    n_render.location = (-400, -100)

    n_over = tree.nodes.new("CompositorNodeAlphaOver")
    n_over.location = (0, 100)

    # No standalone "Composite" output node anymore -- a compositing node
    # group's own group-output socket IS the scene's final render result
    # now (confirmed: CompositorNodeComposite no longer exists as a node
    # type in this Blender version; the compositor tree is a real NodeTree
    # datablock with the same Group Input/Output + interface-socket
    # mechanism Geometry Nodes already uses).
    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    n_output = tree.nodes.new("NodeGroupOutput")
    n_output.location = (300, 100)

    # Blender 5.x's AlphaOver renamed/reordered its sockets to
    # "Background" (0) / "Foreground" (1) / "Factor" (2) / "Type" (3) /
    # "Straight Alpha" (4) -- confirmed by introspection; the old
    # inputs[1]/inputs[2] convention silently wired the render layer into
    # the Factor float socket instead of an image slot, which is exactly
    # what produced the flat grey/black output this replaced.
    tree.links.new(n_clip.outputs["Image"], n_over.inputs["Background"])
    tree.links.new(n_render.outputs["Image"], n_over.inputs["Foreground"])  # keyed by its own alpha
    tree.links.new(n_over.outputs["Image"], n_output.inputs["Image"])

    out_dir = Path(__file__).resolve().parents[3] / "scratch" / "blender_render_out" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out_dir) + "/"

    print(f"Built render scene {scene_name}: camera={cam_obj.name} clip={clip_path} "
          f"res={w}x{h} frame_start(clip)={FIRST_TRACKED_FRAME}")
    return scene


def render_test_frame(scene: bpy.types.Scene, frame: int, out_path: Path) -> None:
    old_scene = bpy.context.window.scene
    bpy.context.window.scene = scene
    scene.frame_set(frame)
    scene.render.filepath = str(out_path)
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    bpy.context.window.scene = old_scene
    print(f"Rendered test frame -> {out_path}")


if __name__ == "__main__":
    import json

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    main_scene = bpy.context.scene
    scenes = []
    for entry in manifest:
        scenes.append(build_render_scene(entry["label"], entry["proxy_path"], main_scene))

    if "--render-check" in sys.argv:
        out_dir = Path(__file__).resolve().parents[3] / "scratch" / "blender_compositor_check"
        out_dir.mkdir(parents=True, exist_ok=True)
        for scene in scenes:
            render_test_frame(scene, 200, out_dir / f"{scene.name}_check.png")

    save_path = None
    for i, a in enumerate(sys.argv):
        if a == "--save":
            save_path = sys.argv[i + 1]
    if save_path:
        bpy.ops.wm.save_as_mainfile(filepath=save_path)
        print(f"Saved -> {save_path}")

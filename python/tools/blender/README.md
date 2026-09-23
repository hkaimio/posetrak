# Blender scene tools

Scripts that move data between posetrak and Blender: building a scene that
shows tracked results over the real footage, and exporting 2D tracks made in
Blender's movie clip editor. They are kept as tools and are not planned to
become `posetrak` commands
(`docs/roadmap/features/marker-based-mocap/productization-architecture-and-plan.md` §3.10).

Two kinds of script live here:

- **Run inside Blender** (`import bpy`):
  `blender --background FILE.blend --python <script> -- <args>`
- **Plain Python** (no `bpy`): `export_pen_pad_to_blender.py`,
  `make_blender_proxy_videos.py`.

Scripts that write scratch files resolve the repository's `scratch/`
directory relative to their own location, so they must stay at this depth
(`python/tools/blender/`).

## Building a scene over the real footage

1. `blender_add_cameras.py` — add the solved cameras, with FOV from each
   camera's undistorted intrinsics.
2. `make_blender_proxy_videos.py` — undistort and downscale the capture's
   videos into proxies (plain Python; writes a manifest).
3. `blender_add_video_backgrounds.py` — attach each proxy as that camera's
   background so playback shows the character over the real footage.
4. `blender_add_compositor_render.py` — one render scene per camera that
   composites the animated character over its footage.
5. `blender_render_scene_range.py` — render those scenes over a frame range
   to H.264, in one Blender process.

## Pen and pad objects

`export_pen_pad_to_blender.py` (plain Python) converts the pen and pad
trajectory CSV into a Blender-space keyframe JSON;
`blender_add_pen_pad_animation.py` builds the animated objects from it.
Both are specific to that capture. Generalizing them to "export an object
run to Blender" is worth doing only if it turns out to be cheap.

## Importing 2D tracks made in Blender

`blender_export_2d_tracks.py` exports each movie clip's tracking curves as
one CSV per clip and track, in posetrak's pixel convention (raw distorted
pixels, origin top-left, Y down). Import them with `posetrak detect import-2d`,
one `--camera LABEL CSV` per track; the CSVs need no conversion. The run is
bound to a capture object with `--object`, and `posetrak sequence
finalise-object` makes the object's sequence from it.

Note: `OUTPUT_DIR` at the top of `blender_export_2d_tracks.py` has no
default; set it before running interactively, or pass `--output-dir`
headless.

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""blender_add_video_backgrounds.py — PROTOTYPE: attach each camera's
undistorted proxy video as that camera's background image, so playing
the scene's animation shows the character over the real footage.

Run inside Blender after blender_add_cameras.py has already added the
`cam_<label>` objects (headless: `blender --background FILE.blend
--python blender_add_video_backgrounds.py`).

Timing: the proxy videos (make_blender_proxy_videos.py) were trimmed to
start exactly at the tracking run's own time_start_s, so proxy-video
frame 0 always corresponds to the same instant as the animation's own
first *tracked* frame. Per this project's BVH export convention
(posetrak.export.bvh, include_rest_frame=True default), that's animation
frame 2 (frame 1 is a prepended rest pose) -- confirmed for this specific
armature: frame 1's pose is a clear outlier next to frames 2 onward's
otherwise-continuous motion. If a future export changes that convention,
adjust FIRST_TRACKED_FRAME below.

Camera and video native fps both being 120 (this session's tracking run
matched the video's own frame rate almost exactly, to within 0.1%) means
a single constant frame offset stays in sync for the whole clip without
needing a speed-ramp effect -- verify this holds if reusing this script
against a differently-configured capture.
"""

import json
import sys
from pathlib import Path

import bpy

MANIFEST_PATH = Path(__file__).resolve().parents[3] / "scratch" / "blender_proxy_videos" / "manifest.json"
FIRST_TRACKED_FRAME = 2  # see module docstring


def add_video_backgrounds(manifest_path: Path) -> None:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    for entry in manifest:
        cam_name = f"cam_{entry['label']}"
        cam_obj = bpy.data.objects.get(cam_name)
        if cam_obj is None:
            print(f"WARNING: no camera object {cam_name!r} -- run blender_add_cameras.py first")
            continue

        clip_path = str(Path(entry["proxy_path"]).resolve())
        clip = bpy.data.movieclips.load(clip_path)
        # Datablock-level (Blender 5.x's MovieClipUser dropped frame_start/
        # frame_offset entirely -- confirmed by introspecting its actual RNA
        # properties, which are now just frame_current/proxy_render_size/
        # use_render_undistorted): "global scene frame at which this clip's
        # own frame 0 plays" -- applies uniformly to every user of this
        # clip, background image here and the compositor's Movie Clip node
        # alike.
        clip.frame_start = FIRST_TRACKED_FRAME
        clip.frame_offset = 0

        cam_data = cam_obj.data
        cam_data.show_background_images = True
        # Clear any previous background images on this camera (re-run safety).
        while cam_data.background_images:
            cam_data.background_images.remove(cam_data.background_images[0])
        bg = cam_data.background_images.new()
        bg.source = "MOVIE_CLIP"
        bg.clip = clip
        bg.clip_user.use_render_undistorted = False  # already undistorted by our own proxy step
        bg.alpha = 1.0
        bg.display_depth = "BACK"

        print(f"{cam_name}: background <- {clip_path}")


if __name__ == "__main__":
    add_video_backgrounds(MANIFEST_PATH)
    save_path = None
    for i, a in enumerate(sys.argv):
        if a == "--save":
            save_path = sys.argv[i + 1]
    if save_path:
        bpy.ops.wm.save_as_mainfile(filepath=save_path)
        print(f"Saved -> {save_path}")

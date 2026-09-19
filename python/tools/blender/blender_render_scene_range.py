# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""blender_render_scene_range.py — PROTOTYPE: render each render_<label>
scene's animation (built by blender_add_compositor_render.py) over a
given frame range to its own H.264 file, in one Blender process (avoids
per-invocation addon-load overhead 6x over).

--frame-step > 1 renders at a fraction of the scene's own fps (e.g. step=4
on a 120fps scene renders an effective 30fps) -- output PNG filenames
carry the real (non-consecutive) frame number, so combine them with
ffmpeg's glob pattern (`-pattern_type glob -i "frame_*.png"`), not a
`%04d` sequential pattern, and declare `-framerate <scene_fps>/<step>` on
the input to keep real-time playback speed correct.

Each render_<label> scene's valid image_settings.file_format turned out to
be a fixed, mutually-exclusive per-scene partition in one real file this
was built against -- some scenes only ever accept still-image formats,
one only ever accepted 'FFMPEG' (probably from earlier manual GUI
experimentation on that one scene specifically; switching either
direction via plain property assignment was rejected outright, not just
inconvenient). Rather than fight that, this renders whichever format each
scene actually allows: a PNG sequence normally, or a directly-encoded
.mp4 for a scene stuck on FFMPEG -- the manifest below records which, so
the grid-combine step can treat each camera accordingly.

Run headless:
    blender --background FILE.blend --python blender_render_scene_range.py \\
        -- --frame-start 200 --frame-end 560 --frame-step 4 \\
        --output-dir scratch/blender_render_out
"""

import sys
from pathlib import Path

import bpy


def render_scene_range(
    scene: bpy.types.Scene, frame_start: int, frame_end: int, frame_step: int, out_dir: Path,
) -> dict:
    """Returns {"kind": "png_dir"|"video", "path": str}.

    Which format a scene's image_settings.file_format actually accepts
    turned out to be knowable only by trying the assignment and catching
    the TypeError -- bl_rna.properties['file_format'].enum_items returns
    the same full static list for every scene regardless of the real,
    dynamically-enforced per-scene restriction (confirmed directly: a
    scene whose only real option was 'FFMPEG' still reported the full
    15-format static list). An earlier version of this function checked
    that static list first and was fooled by it into always attempting
    'PNG' and failing on the one FFMPEG-only scene.
    """
    bpy.context.window.scene = scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.frame_step = frame_step
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        scene.render.image_settings.file_format = "PNG"
    except TypeError:
        pass
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(out_dir) + "/frame_"
        print(f"Rendering {scene.name} (PNG sequence): frames [{frame_start}, {frame_end}] "
              f"step={frame_step} -> {out_dir}")
        bpy.ops.render.render(animation=True)
        return {"kind": "png_dir", "path": str(out_dir)}

    try:
        scene.render.image_settings.file_format = "FFMPEG"
    except TypeError as e:
        raise SystemExit(f"{scene.name}: neither 'PNG' nor 'FFMPEG' accepted here: {e}") from e
    video_path = out_dir.with_suffix(".mp4")
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = str(video_path)
    print(f"Rendering {scene.name} (direct video -- 'PNG' not accepted here): "
          f"frames [{frame_start}, {frame_end}] step={frame_step} -> {video_path}")
    bpy.ops.render.render(animation=True)
    return {"kind": "video", "path": str(video_path)}


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-start", type=int, required=True)
    ap.add_argument("--frame-end", type=int, required=True)
    ap.add_argument("--frame-step", type=int, default=1,
                     help="Render every Nth frame (default: 1, every frame). E.g. 4 on a "
                          "120fps scene renders at an effective 30fps.")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[3] / "scratch" / "blender_render_out"))
    ap.add_argument("--scene", nargs="*", default=None, help="Restrict to these render_<label> scenes")
    args = ap.parse_args(argv)

    # A relative path here resolves against Blender's own CWD (observed to
    # default to C:\, not the .blend file's directory or the shell's own
    # CWD) -- resolve to absolute explicitly rather than relying on that.
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = [s for s in bpy.data.scenes if s.name.startswith("render_")]
    if args.scene:
        scenes = [s for s in scenes if s.name in args.scene]

    manifest = {}
    for scene in scenes:
        label = scene.name[len("render_"):]
        manifest[label] = render_scene_range(scene, args.frame_start, args.frame_end, args.frame_step, out_dir / label)

    if scenes:
        effective_fps = scenes[0].render.fps / args.frame_step
        import json
        manifest_path = out_dir / "render_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"effective_fps": effective_fps, "frame_start": args.frame_start,
                       "cameras": manifest}, f, indent=2)
        print(f"\nDone. Rendered {len(scenes)} scene(s) -> {out_dir}")
        print(f"Effective output rate: {effective_fps:.4f}fps -- use this (not the scene's own "
              f"{scenes[0].render.fps}fps) as ffmpeg's -framerate when combining a png_dir camera "
              f"(-pattern_type glob, frame numbers are not consecutive when --frame-step > 1); a "
              f"video camera's own encoded fps is already correct. See {manifest_path}.")

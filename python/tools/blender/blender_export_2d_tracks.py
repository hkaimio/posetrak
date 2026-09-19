# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""blender_export_2d_tracks.py -- export every MovieClip's 2D tracking
curves to one CSV per (clip, track), in PoseTrak's own pixel convention
(raw/distorted pixels, origin top-left, Y-down), for import into a
session DB as anonymous 'dots'-style observations (see
finalize_ball_cutie_detection.py for the DB-writing half of this --
same wire format, just fed from Blender's tracker instead of Cutie's
segmentation-mask centroid).

Blender itself stores marker positions normalized to [0,1] with the
origin at the *bottom-left* (Y-up) -- the opposite of the pixel/OpenCV
convention (origin top-left, Y-down) used everywhere else in this
project. This script converts on the way out, so the resulting CSVs
are already in the right convention to compare against/import as
`pose_observations` alongside real reflective-dot or vitpose data.

A track's `.frame` values are *scene* frame numbers, not necessarily
the source video's own frame index -- offset by the clip's own
`frame_start`. This script reports frame_start and the clip's own
first/last frame per clip so you can confirm the alignment rather than
silently trusting it; video_frame in the CSV is computed as
`scene_frame - frame_start` (0-indexed into the clip's source video).
If your camera's real video_frame numbering doesn't start at 0 for the
same physical frame (e.g. the capture pipeline's own sync convention),
you'll need to apply that same offset when importing.

A track with a re-initialization partway through (2-3 manual re-seeds
per throw, as found necessary for this project's ball-tracking
experiment) shows up here as either one track with a real gap in its
markers, or as separate track objects (Track, Track.001, ...) if a
fresh track was started each time -- this script does not try to
stitch these together; it just exports whatever markers exist,
faithfully, gaps included. Handling multiple per-camera segments (with
real gaps between them) is already exactly what the tracker's own
anonymous-dot path expects, so no stitching should be needed on
import either.

Usage -- two ways, deliberately made to work the same either way (see
2026-09-17 note below on why this matters):

  1. Interactive (Blender's own Scripting workspace): edit OUTPUT_DIR
     below to a real path, paste this whole file into a new text block,
     Run Script. Uses OUTPUT_DIR and CLIP_NAME as configured below.

  2. Headless:
         blender --background your_file.blend --python blender_export_2d_tracks.py -- \\
             --output-dir /path/to/output

     A `--output-dir` given this way overrides OUTPUT_DIR below.

Must be run *inside* Blender (imports bpy) -- not part of PoseTrak's
own Python environment.

2026-09-17 (Harri): the first version of this script used a *required*
argparse argument for --output-dir. Run interactively (no command-line
args at all -- there's no way to supply one from the Scripting tab),
argparse's missing-required-argument path calls sys.exit(), which
Blender's Text Editor "Run Script" operator catches and reports via
BPy_errors_to_report() -- and hit a real Blender 5.2 bug where Python
garbage collection during that exact error-report path crashes while
deallocating a GPU index-buffer object (nothing to do with this script
itself; see the crash log's own stack: PyGC_Collect ->
GLIndexBuf::~GLIndexBuf). Fixed by never requiring a CLI arg (a plain
in-file variable works both ways) and by never letting an exception
escape main() at all -- caught and printed instead of raised, so this
can't trigger that Blender-side error-report path regardless of what
else might go wrong here.
"""
from __future__ import annotations

import csv
import sys
import traceback
from pathlib import Path

# Edit these two for interactive (Scripting-tab) use. Ignored if
# --output-dir is given on the command line (headless use).
OUTPUT_DIR = r"C:\Users\HarriKaimio\Desktop\blender_tracks"
CLIP_NAME = None  # None = export every loaded clip; or a specific clip's name


def _resolve_output_dir() -> str:
    # Blender puts its own args before "--"; only args after "--" are ours,
    # and there may be none at all (interactive Scripting-tab run).
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
        if "--output-dir" in argv:
            return argv[argv.index("--output-dir") + 1]
    return OUTPUT_DIR


def _run() -> None:
    import bpy  # noqa: PLC0415 -- only importable inside Blender's own interpreter

    out_dir = Path(_resolve_output_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = [c for c in bpy.data.movieclips if CLIP_NAME in (None, c.name)]
    if not clips:
        print(f"No matching movie clip found (CLIP_NAME={CLIP_NAME!r}); "
              f"loaded clips: {[c.name for c in bpy.data.movieclips]}")
        return

    for clip in clips:
        width, height = clip.size
        print(f"\n=== clip '{clip.name}' ===")
        print(f"  source: {clip.filepath}")
        print(f"  resolution: {width}x{height}")
        print(f"  frame_start={clip.frame_start}, frame_duration={clip.frame_duration}  "
              f"(video_frame = scene_frame - frame_start -- verify this matches your "
              f"capture pipeline's own frame-0 convention before trusting the export)")

        # A clip can have multiple "tracking objects" (rare -- usually just the
        # default camera object); export every track in every one of them.
        for tobj in clip.tracking.objects:
            for track in tobj.tracks:
                rows = []
                for marker in track.markers:
                    if marker.mute:
                        continue
                    px = marker.co.x * width
                    py = (1.0 - marker.co.y) * height
                    video_frame = marker.frame - clip.frame_start
                    rows.append((marker.frame, video_frame, px, py))
                if not rows:
                    continue
                safe_clip = "".join(c if c.isalnum() or c in "-_" else "_" for c in clip.name)
                safe_track = "".join(c if c.isalnum() or c in "-_" else "_" for c in track.name)
                out_path = out_dir / f"{safe_clip}__{safe_track}.csv"
                with open(out_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["scene_frame", "video_frame", "pixel_x", "pixel_y"])
                    w.writerows(rows)
                print(f"  track '{track.name}': {len(rows)} markers "
                      f"(scene frames {rows[0][0]}-{rows[-1][0]}) -> {out_path.name}")

    print(f"\nDone. Wrote CSVs to {out_dir}")


def main() -> None:
    # Never let anything escape to Blender's own script-runner error-report
    # path -- see this module's 2026-09-17 docstring note for why.
    try:
        _run()
    except Exception:
        print("blender_export_2d_tracks.py failed:")
        traceback.print_exc()


if __name__ == "__main__":
    main()

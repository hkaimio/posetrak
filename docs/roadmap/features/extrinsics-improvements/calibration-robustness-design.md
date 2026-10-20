# Calibration robustness: box tool failure modes, room tags, camera relabeling, intrinsics quality, refinement from keypoints

Companion to [extrinsics-improvements-design.md](extrinsics-improvements-design.md). Collects the
calibration problems found while tracking real multi-person captures, and says which of them belong
to the SOMA/GEM-X productization and which are separate work.

Calibration was the largest single source of projection error in the pose-model study: refining the
extrinsics of three cameras lowered the leave-one-out keypoint residual from 8-10% to 2-3% of the box
height, and a capture calibrated with the moving-box tool reached 1.0-2.4% without refinement. Every
tracking improvement is bounded by it.

## 1. Moving-box tool fails silently

The method behind the calibration box video anchors the world frame on the box's final resting position
and assumes it is at rest in view of every camera. On a capture where the box was set down only after
the stated range and seen by two of four cameras, it produced reprojection errors of 100-800 px and
printed the result as a normal solve.

It is `posetrak extrinsics box-video` (`posetrak/cli/extrinsics_rig.py`) and the dialog "Calibrate from
box video…" of the extrinsics status dialog, both running `app/setup/box_video.py`. The box is a marker
body definition, the videos are the capture's, the synchronisation is its first sync configuration.

- **Windows.** One reference camera is scanned for runs of frames where the box's detected corners
  barely move. Each window is looked up in every camera and detected there. A window with fewer than 8
  detected corners (all cameras together), or seen by fewer than two cameras (a camera sees a window
  with at least one whole marker), is dropped; the dropped ones are listed with the reason.
  `--exclude-window-ts` drops chosen windows.
- **Anchor.** The kept windows are ranked by the number of cameras that see them, then by being the
  final rest position, then by corner count. The best one gives the fixed control points; the others add
  their corners as free points, one set of points per window. When the final rest position is not the
  anchor, the solved scene is moved rigidly (a fit of the final window's triangulated corners to the box
  geometry) so that the world frame is the box at its final rest position. When
  that window has too few corners seen by two cameras, the world frame stays the anchor window's box and
  the report says so.
- **Verdict.** A per-camera table (windows seen, observed box corners, median, 90th-percentile, mean
  and maximum reprojection error) is always printed. The calibration fails, with a non-zero exit and
  nothing written, when a camera is unsolved, has no box observation at all (its error cannot be
  judged), has a median error above `--max-median-error` (default 5 px), or a 90th-percentile error
  above four times that: the median alone passes a camera whose view of one window or marker is far
  off, as long as most of its observations fit. The errors are computed before the alignment to the
  final rest position, which moves the cameras but not the anchor's fixed points (a rigid move of the
  whole scene leaves reprojection errors unchanged). The dialog shows the same table, enables saving
  only a passing result, and stops a run when it is closed.
- **Output.** Only with `--write` a new `extrinsic_calibrations` row (method `moving-box-anchor`) is
  stored and linked to the capture; the default is a dry run. `--residuals` writes every box-corner
  observation's error.
- **Options.** `--marker-body` (the box geometry; its id is the rig id), `--rig-dict` (ArUco
  dictionary for markers the body does not name), `--world-z-offset`, `--exclude-window-ts`,
  `--min-window-corners`, `--max-median-error`, `--residuals` (per-observation errors as CSV).

A box that ends at rest in view of some cameras only is no longer a failure by itself: if an earlier
window is seen by all cameras it anchors the solve. The error table is what fails a capture where no
such window exists or a camera's fit is poor.

### Run time

Measured on a 24-thread AMD desktop (Windows 11, PyAV/FFmpeg software HEVC decode, OpenCV 4.13), 4K 120 fps
videos with 120-frame GOPs (the Insta360 camera's GOP is 40 frames). Seconds per phase: *scan* is the
search for stationary windows in the reference camera, *lookups* is the detection of every window in every
camera, *solve* is `run_calibration` plus the verdict (and the re-read of the anchor window's frames).

| Case | Cameras | Range | Windows | Scan | Lookups | Solve | Total |
|---|---|---|---|---|---|---|---|
| Reference gopro13_02 (succeeding capture) | 6 | 6-33 s | 12 | 125.6 | 81.2 | 18.4 | 225 |
| Reference insta_ace2_pro, failing range | 4 | 63-104 s | 58 | 198.5 | 338.7 | 25.5 | 563 |
| Reference insta_ace2_pro, passing range | 4 | 100-115 s | 6 | 101.7 | 8.7 | 8.1 | 119 |
| Reference gopro13_02, same clip | 4 | 63-104 s | 36 | 189.2 | 211.3 | 9.1 | 410 |
| Reference gopro13_02, same clip | 4 | 100-115 s | 6 | 73.0 | 39.4 | 8.2 | 121 |

(rows: sequential scan, one open and seek per window and camera.) After the changes below:

| Case | Scan | Lookups | Solve | Total |
|---|---|---|---|---|
| 6 cameras, 6-33 s | 14.6 | 14.4 | 18.0 | 47 |
| 4 cameras, insta reference, 63-104 s | 17.6 | 30.7 | 24.7 | 73 |
| 4 cameras, insta reference, 100-115 s | 7.6 | 11.5 | 8.3 | 27 |
| 4 cameras, gopro13_02 reference, 63-104 s | 20.3 | 26.8 | 8.3 | 55 |
| 4 cameras, gopro13_02 reference, 100-115 s | 9.8 | 9.4 | 8.1 | 27 |

Each change on its own (total / the phase it changes):

- Frames were decoded at about 30 ms (decode plus BGR conversion) each and the scan decoded every frame
  of the range: 1,800 to 5,000 frames, decode 80-90 % of the scan; detection of the sampled frames
  (about 40 ms each) is the rest. Converting only the sampled frames saves 10 % of the scan (73 to 66 s on
  the 15 s range) and nothing more, so the decoder is the cost.
- Windows looked up through one open container per camera, in time order, one thread per camera:
  lookups 339 to 107 s (58 windows), 81 to 29 s (12 windows). The 6-window case got slower in this step
  alone (9 to 21 s): a seek costs the decode from the preceding keyframe (60 frames on average with
  120-frame GOPs), and with few windows the slowest camera bounds the phase.
- A camera's windows split into consecutive chunks, about one decoder per core: lookups 107 to 32 s and
  21 to 11.5 s.
- The scan split into one chunk per core, each decoder seeking to its chunk start: scan 195 to 18 s
  (58 windows), 107 to 15 s (6 cameras), 66 to 8 s (15 s range).

Decoding stays single-threaded per decoder; the parallelism is several decoders, one per core, which
puts the peak memory at about 3.8 GB on this 24-thread machine (15 s range, four cameras). Windows, per-window
corner counts, verdicts and extrinsics are identical to the sequential code on all five cases (window
times equal to the sample, extrinsics difference 0.0000 mm and 0.00000 degrees: detections are the same
frames, so the solver input is the same).
What is left on the 41 s range: the solve (25 s of 73 s on the failing range: 1,700 least-squares
iterations of the bundle adjustment, 8 s of feature matching between the anchor window's images), the
lookups (a seek per window and camera, about 1.5 s of decode each, divided over the cores) and the scan
(chunk seeks plus detection on a sixth of the frames). The scan was not made lower resolution: with the
parallel decoder it takes 8-20 s, and the detections that decide the windows keep full resolution.

## 2. Room tags as scene markers

Cameras moved 70-400 px between two captures of one session, so extrinsics cannot be reused across
captures. Fixed tags in the room give a common world frame.

This is mostly built: `scene_marker_bodies` stores solved poses of scattered tags with a `group_name`
per physical space, so a later capture in the same room can load them (extrinsics design, section 9,
Tier B). Missing is the workflow that re-solves a new capture's extrinsics against the stored room
tags without a calibration object in view.

**Separate feature** (extrinsics improvements): not needed for SOMA, and the storage exists.

## 3. Safe camera relabeling

Sync points belong to a capture video row (`sync_points.shot_video_id`), so fixing a wrong camera
label by swapping file paths between `capture_videos` rows silently breaks the synchronisation. The
"reassign camera" operation (`posetrak/db/reassign_camera.py`) changes `capture_videos.camera_instance_id`
(and, when the new camera is of another model, `camera_mode_id` and the intrinsics calibration that
follows the mode) and leaves file paths and sync points with their videos. Cameras of several videos
can be exchanged in one call, and no two videos of a capture may end with the same camera. It refuses
when detection runs, pose observation sequences or tracking runs reference the capture, unless the
caller confirms that they will be recomputed; they are not rewritten and keep the old camera ids. A
linked extrinsic calibration belongs to cameras by id, so the operation warns about it.

`sync_points.camera_instance_id` has two meanings in existing data: sync points written by the
synchronisation dialog hold the video id there, imported ones the camera id. Only the latter follow
the video to its new camera.

Where: `posetrak capture reassign-camera --capture ID --assign VIDEO=CAMERA ...` and the "Cameras…"
button of the capture panel in the main viewer.

**In the productization, first release**: small, and it prevents a silent corruption of the
synchronisation that has happened on real captures.

## 4. Intrinsics quality

Capture setup found no intrinsics for one capture's videos; one camera's stored distortion has k1
about -2.5; another camera's tags were 8-10 px off at the image edges; a wide-lens mode fits neither the
pinhole nor the fisheye model, and focal length and distance trade off when the subjects fill a small
part of the frame (a 15% focal change and 1.3 m camera shift gave a better residual near the subjects
and worse tracking).

- A check at capture setup and before tracking: every video has intrinsics; the distortion
  coefficients are within plausible ranges for the model; the calibration's own RMS and coverage of the
  image (fraction of the frame where board corners were observed) are reported; cameras below a
  threshold are flagged and default to velocity-only observations.
- Refits of the flagged cameras are a capture-side task, not code.

**In the productization, second release**: the check. Better lens models for wide modes are separate.

## 5. Extrinsics refinement from keypoints

Bundle adjustment on the pipeline's own detections (strongest candidate in the person's own mask,
peak at least 0.5, 15 Hz sampling, a robust loss and a pair gate) gave the same improvement as
refinement on operator labels, out of sample. Only the extrinsics should be refined that way: freeing
focal length and principal point added nothing on good cameras and fitted wrong geometry on the
wide-lens one.

**Separate feature**, available first as a documented command-line recipe using the existing extrinsics
solver; productized in the calibration dialog later. It must store the refined set as a new
`extrinsic_calibrations` row, never overwrite the old one.

## 6. Session database locking

Covered by [session DB concurrency](../session-db-concurrency/session-db-concurrency-design.md),
first release.

## 7. Joint limits of the default skeletons

Covered by [pose-model skeleton binding](../pose-model-skeleton-binding/pose-model-skeleton-binding-design.md),
first release.

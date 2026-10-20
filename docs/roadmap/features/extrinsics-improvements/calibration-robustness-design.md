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

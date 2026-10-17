# Calibration robustness: box tool failure modes, room tags, camera relabeling, intrinsics quality, refinement from keypoints

Companion to [extrinsics-improvements-design.md](extrinsics-improvements-design.md). Collects the
calibration problems found while tracking real multi-person captures, and says which of them belong
to the SOMA/GEM-X productization and which are separate work.

Calibration was the largest single source of projection error in the pose-model study: refining the
extrinsics of three cameras lowered the leave-one-out keypoint residual from 8-10% to 2-3% of the box
height, and a capture calibrated with the moving-box tool reached 1.0-2.4% without refinement. Every
tracking improvement is bounded by it.

## 1. Moving-box tool fails silently

`python/tools/calibrate_extrinsics_from_moving_box.py` (a prototype, the method behind the calibration
box video) anchors the world frame on the box's final resting position and assumes it is at rest in
view of every camera. On a capture where the box was set down only after the stated range and seen by
two of four cameras, it produced reprojection errors of 100-800 px and printed the result as a normal
solve.

Changes, as part of productizing the tool into the `extrinsics_rig` command it was written for:

- Rank stationary windows by camera coverage and corner count; drop windows with too few corners.
- Anchor on a window every camera sees; align to the final rest frame afterwards when it is seen by
  fewer cameras.
- Fail with a non-zero exit and a per-camera error table when any camera's median reprojection error
  exceeds a threshold (default 5 px), instead of printing a result.
- Expose it in the calibration dialog as "Calibrate from box video".

**In the productization, first release**: the failure modes produce wrong extrinsics without warning,
and every later result depends on them.

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
label by swapping file paths between `capture_videos` rows silently breaks the synchronisation. A
"reassign camera" operation changes `capture_videos.camera_instance_id` (and the intrinsics reference
that follows the camera mode) and leaves file paths and sync points with their videos; it refuses when
detection or tracking runs already reference the capture, unless the user confirms that they will be
recomputed.

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

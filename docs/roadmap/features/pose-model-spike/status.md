```toml
name = "SOMA/GEM-X Productization"
status = "in_progress"
progress_pct = 60
description = """
SOMA-77 (GEM-X) as the recommended pose model in the main code line, in three releases: SOMA \
recommended with the detection pipeline, person assignment, skeleton binding, tracker cost and \
calibration items; keypoint candidates and mask versioning; tracklet editing.
"""
categories = ["detection-pipeline", "tracker-core", "data-model"]
target_release = "TBD"
last_updated = 2026-10-20
```

# SOMA/GEM-X Productization — Implementation Status

See [the productization plan](soma-gemx-productization-plan.md): stages, release split, and each
stage's outcome. The spike that preceded it is [soma77-spike-report.md](soma77-spike-report.md).

## Current state

**First release (stages 0-7): merged to `main`, not yet packaged.** No prototype installer has been
built from it (the last is `v0.1.0-proto2`, see [packaging status](../packaging/status.md)); one waits
until the release has been tested on real captures, end to end from detection through assignment to
tracking. Main compared with `v0.1.0-proto2`:

- SOMA-77 as the recommended and preselected pose model, with the weights downloaded after accepting
  their licences; RTMPose-133 when they are declined.
- One detection pass for any box source (YOLOX, segmentation masks, an existing run) with the hand
  refinement inside it, on by default for SOMA runs.
- `posetrak detect assign`: automatic assignment of YOLOX track fragments to persons.
- Default skeletons composed from layers with COCO and SOMA marker sets and a new joint-limit table;
  `posetrak skeleton compose`.
- Hierarchical tracking by default for 77-keypoint skeletons, within 8 GB on the three-person trial.
- Session database robustness: busy timeouts, run status, interrupted runs.
- New: box-video calibration (`posetrak extrinsics box-video`, "Calibrate from box video" in the
  calibration dialog), which prints a per-camera error table and refuses to save a failed solve; the
  moving-box method was a prototype script before.
- New: camera reassignment of a capture's videos (`posetrak capture reassign-camera`, "Cameras…" in the
  capture panel).

| Stage | Status |
|---|---|
| 0 De-risking measurements | Done apart from M4 (question to NVIDIA drafted, not sent) |
| 1 Session database robustness | Done |
| 2 Pose model and skeleton binding | Done; [status](../pose-model-skeleton-binding/status.md) |
| 3 Detection pipeline | Done; decode soak open; [status](../detection-pipeline/status.md) |
| 4 Track to person assignment | Done; contact trial short of target; [status](../track-person-assignment/status.md) |
| 5 Hand refinement for SOMA runs | Done; default provisional; [status](../hand-detection-refinement/status.md) |
| 6 Tracker cost | Done; per-person threads not kept; [status](../tracker-performance/status.md) |
| 7 Calibration items | Done, with box-video run time cut from about 12 min to 73 s on a 41 s range; [status](../extrinsics-improvements/status.md) |
| 8 Mask versioning | Done (session schema 68); [status](../segmentation-versioning/status.md) |
| 9 Keypoint candidates | Next |
| 10-12 | Not started |

## Known issues

The plan's "Known limits of the first release" lists them. The ones to settle by testing:

- **Hand tracking.** The refinement default comes from one labelled trial, and the hand stages' tracker
  cost has never been measured (no baseline sequence had refined hands). A capture made for hand
  tracking is planned: single-hand and two-hand gestures with checkable contacts, prop handling, and the
  same with the hand occluded in one or two cameras.
- **First real run.** The stages were accepted one by one; detection, assignment and tracking have not
  run end to end on a new capture.
- **Existing persons' skeletons** have no SOMA markers; `skeleton compose --base` upgrades them.
- **Box-video reference camera.** On one capture the same range fails with one reference camera and
  passes with another; not investigated.

# Segmentation + Pose Experiment — Notes and Conclusions

**Date:** 2026-05-30
**Context:** Explored whether instance segmentation could improve pose estimation quality in
challenging capture frames involving close person-to-person contact (ukemi / judo throws).

## Motivation

The YOLO-based person detector crops a rectangular bounding box around each person.  In contact
scenes the bounding boxes of two persons overlap: a foot, hand, or arm from person B may appear
inside person A's crop and trick the pose estimator into attaching a keypoint to the wrong body.
These "close-range outliers" look plausible to the pose model (correct body part, just wrong
person) and survive the UKF's Mahalanobis gate, causing the filter to slowly drift or diverge.

## Experiment

**Script:** `segpose_test.py` (project root)
**Video:** `~/projects/mocap_videos/test.mp4` — Pixel 7, 120 fps capture / 30 fps container,
3-person frame (ukemi attempt 6, ~307–421 s into source video)
**Models:** `yolo11x-seg.pt` (segmentation) + `yolo11x-pose.pt` (pose)

### Runs
| Run | `--step` | `--style` | Images | Time |
|-----|----------|-----------|--------|------|
| 1   | 15       | hard      | 645    | 41 s |
| 2   | 5        | soft      | 1912   | 109 s |

### Hard-mask approach
- Background pixels replaced with gray (value 70).
- Result: pose estimates on masked crops were **worse** than the full-frame baseline, even on
  easy (no-contact) frames.
- Root cause: the pose detector (YOLO-pose, 17 kp) appears to expect photometric context around
  limbs.  Gray background confuses it into thinking legs are close together when they are spread
  (e.g. person taking a wide step forward).

### Soft-mask approach
- Background dimmed to 20% brightness.
- Result on easy frames: still worse than baseline (same leg-spread failure).
- Result on difficult frames: **introduced a new failure** — the dim background still shows the
  adjacent person, and the pose detector chose to estimate that person's pose inside the target
  person's bounding box.

### Segmentation quality on contact frames
- The segmentation model itself (YOLO-seg) failed on the most interesting frames — exactly the
  close-contact, occlusion frames where improvement was needed.
- Segmentation was reliable on clean (no-contact) frames where the baseline already works well.

## Conclusions

1. **Pre-hoc masking does not help.** Feeding a masked crop to the pose model degrades accuracy
   because modern pose models expect natural image context.

2. **Post-hoc filtering is the right integration point.** Run the pose model on the original
   (unmasked) crop, then use the segmentation mask only to *evaluate* whether each detected
   keypoint falls inside the person's silhouette.  Keypoints outside the mask get inflated noise
   in the UKF update; the filter naturally down-weights them without hard rejection.

3. **SAM2 preferred over SAM3 for this use case.** SAM2 has frame-to-frame memory that propagates
   a person's segmentation identity through transient occlusions.  SAM3 adds open-vocabulary text
   prompting but is stateless per-frame — it will fail on the same contact frames as YOLO-seg.

4. **Optical flow as a complementary signal.** Dense optical flow (RAFT or Lucas-Kanade on keypoint
   neighbourhoods) can provide predicted keypoint positions between frames.  Keypoints that deviate
   strongly from the flow prediction are likely from a neighbouring body and can be flagged for
   noise inflation alongside the segmentation check.  This is simpler to implement than full SAM2
   integration and may catch cases where segmentation itself is uncertain.

5. **The UKF infrastructure is ready for per-observation noise.** The existing
   `per-frame-measurement-noise-design.md` already proposes a per-observation noise vector for the
   UKF update.  Adding segmentation-based inflation is a natural extension of that mechanism rather
   than a new subsystem.

## Next Steps

See `segmentation-keypoint-weighting-design.md` for the full integration design.

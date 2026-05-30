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

---

## Phase 1 — Pre-hoc masking (YOLO-seg, May 2026)

**Script:** `segpose_test.py` (project root)
**Video:** `~/projects/mocap_videos/test.mp4` — Pixel 7, 120 fps capture / 30 fps container,
3-person frame (ukemi attempt 6, ~307–421 s into source video)
**Models:** `yolo11x-seg.pt` (segmentation) + `yolo11x-pose.pt` (pose)

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

**Conclusion:** Pre-hoc masking does not help.

---

## Phase 2 — Post-hoc confidence weighting (Cutie, May 2026)

**Tool:** `python/tools/add_seg_quality.py`
**Data:** "Ukemit yritys 6" trial, detection run `8bfded7f`, 5 cameras, 3 persons
(Timo, Tommi, Roosa), ~3400 frames each
**Segmentation model:** Cutie (XMem++) initialised per-camera with SAM2 bounding-box prompts

### Quality scoring
Each detection keypoint was scored against the Cutie segmentation mask:
- `1.0` — inside the eroded person mask
- `0.5` — in the boundary zone
- `0.0` — outside the mask
- `-1.0` — unavailable (no mask data for this frame)

Score distribution on pixel7 (representative): 43% inside / 35% boundary / 22% outside.

Scores written to `keypoint_obs_quality` table (new).
`python/tools/apply_seg_weighting.py` then cloned the `pose_observation_sequences`,
multiplying `kp_blob` confidence by quality score so that the existing C++ tracker
could be tested without code changes.

### Tracker comparison — Timo (sequence `438fc116` baseline vs `161781bc` weighted)

| Metric | Baseline | Seg-weighted |
|---|---|---|
| Steps tracked | 3375 | 3164 |
| Steps lost | 0 | 0 |
| Mean inlier observations/step | 154 | 93 |
| NIS/DoF median | **2.16** | **1.11** |
| NIS/DoF p90 | 3.73 | 2.85 |
| Mean root-pos divergence from baseline | — | 16 cm |
| Max root-pos divergence | — | 107 cm |

NIS/DoF ≈ 1.0 means the filter's measurement noise model matches actual innovation
statistics; values >> 1 indicate the filter is over-confident.  The improvement from
2.16 to 1.11 shows the quality weighting brings the filter into near-ideal calibration.

### Standout event: hip throw at t ≈ 42.6–42.9 s

At this moment the baseline tracker drops to **1 inlier** for ~14 consecutive steps
(~0.3 s) while the weighted tracker never falls below **32 inliers**:

| | Baseline (worst step) | Weighted (same step) |
|---|---|---|
| Inlier count | 1 | 61–70 |
| Root position drift vs weighted | — | ~100 cm |

Root cause: cross-person keypoints from the throw partner survived Mahalanobis rejection
in the baseline (they were geometrically inconsistent with each other, so the UKF rejected
nearly all of them — a death spiral).  Quality weighting zeroed those keypoints before they
reached the filter, keeping the tracker anchored through the throw.

The relevant video frames are approx:
- gopro-11_mini_01 frame 5106–5109
- gopro-11_mini_02 frame 13884–13887
- pixel9 frame 2564–2567

### Limitations observed

1. **Data quality dependency.** The source pose data came from YOLO detections that were
   manually stitched into person timelines and contain holes and tracking errors.  In some
   windows outside the obvious throw events, aggressive outside-mask rejection left the
   tracker with almost no inliers, causing secondary drift unrelated to contact.  The
   quality of this experiment is partially bottlenecked by the detection data rather than
   the segmentation approach itself.

2. **Fragmented seg_quality_runs.** All 5 cameras ended up in 4 separate `seg_quality_runs`
   due to a Hydra re-initialisation bug (fixed: model now loaded once before the camera
   loop).  Production runs should produce a single run per trial.

3. **Only Timo was compared.** Tommi and Roosa have no scaled skeleton in the DB yet, so
   the multi-person comparison is deferred.

---

## Conclusions

1. **Pre-hoc masking does not help.** Feeding a masked crop to the pose model degrades
   accuracy because modern pose models expect natural image context.

2. **Post-hoc confidence weighting works and is measurably better.** Running the pose model
   on the original crop and then using the segmentation mask to score keypoints produces
   near-ideal UKF calibration (NIS/DoF ≈ 1.1 vs 2.2) and prevents tracking collapse during
   contact events.

3. **Cutie (XMem++) is the right video segmentation backbone.** Produces consistent
   person-identity tracks through close contact and throws; errors appear only at the
   boundary frames of deep mutual occlusions.

4. **Preferred end-to-end architecture (not yet implemented):**
   Run Cutie first → derive tight mask-aligned bounding boxes → feed to RTMPose as
   detection ROI → score resulting keypoints against the mask.  This is better than the
   current approach (YOLO bbox → RTMPose → score against Cutie) because the ROI fed to the
   pose estimator already excludes the adjacent person's body, reducing the chance of
   cross-person keypoints being generated in the first place.  Post-hoc scoring then acts
   as a second line of defence for boundary cases.

5. **Optical flow as a complementary signal** (not tested).  Dense optical flow on keypoint
   neighbourhoods could flag keypoints that deviate strongly from predicted motion as likely
   cross-person contamination.  Simpler than full SAM2 integration and may catch cases where
   the segmentation mask itself is uncertain.

6. **The UKF infrastructure is ready for per-observation noise.** The correct long-term
   implementation is noise inflation (not hard zeroing): inflate measurement noise for
   outside-mask keypoints proportional to their distance from the mask, rather than zeroing
   confidence.  This is a natural extension of the per-observation noise mechanism already
   proposed in `per-frame-measurement-noise-design.md`.

## Next Steps

1. **RTMPose with Cutie ROI** — implement the preferred architecture (conclusion 4 above):
   Cutie masks → tight bounding boxes → RTMPose re-detection → score against masks.
2. **Per-observation noise inflation in C++** — modify `session_reader.cpp` to read
   `keypoint_obs_quality` and inflate `measurement_noise_std` per keypoint rather than
   zeroing confidence (smoother degradation, avoids the all-or-nothing problem seen here).
3. **Scale Tommi and Roosa skeletons** — needed to run the multi-person comparison.

See also `segmentation-keypoint-weighting-design.md` for the detailed integration design.

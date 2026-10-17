# Hand refinement for SOMA-77 runs and without segmentation

Companion to [hand-detection-refinement-design.md](hand-detection-refinement-design.md), which
describes the hand pass itself (wrist-centred crop, `rtmlib.Hand`, proximity gate, separate
`hand_l` / `hand_r` rows with their own noise scale).

## 1. Why it looks tied to segmentation

The hand pass is not tied to segmentation in its implementation:

- `HandRefinementPipeline` (`posetrak/detection/hand_refinement.py`) reads a finished run's
  `full_body` rows per (frame, track) and refines each row's wrists independently. It uses neither
  masks nor person identity; a fragmented YOLOX track id is as good as a mask label for it.
- The YOLOX path of the pose extraction tool (`app/pose/main.py`) and the older `posetrak-pose run`
  command (`app/pose/cli.py`) both run it after `DetectionPipeline` and default it on.
- The segmentation route runs it from `PoseWorker` after the pose pass.

The tie is only in `posetrak detect run`: its option rules reject `--refine-hands` without
`--segmentation` (`detect.py`, "only supported with --segmentation"), and its plain YOLOX path never
calls the pass. That is a gap of one command, not a design constraint.

The real limit for the new default is elsewhere: the pass runs only for a 133-keypoint pose model.
It checks that `"133"` is in the run's `pose_model` and reads the wrist and elbow at COCO-133 indices
9/10 and 7/8, and the tracker merges its rows into the COCO-133 index space at the fixed offsets 91
and 112. On a SOMA-77 run it silently does nothing.

## 2. Design

1. **Pose-model independent input.** The wrist and elbow positions come from the run's pose model
   through the pose model registry (SOMA `LeftHand`/`LeftForeArm`, COCO `left_wrist`/`left_elbow`),
   not fixed indices. The check on `"133"` is replaced by "the model names a wrist and an elbow".
2. **Own layout.** Refined rows are written with a `hand21` manifest under `obs_source = 'hand_l' |
   'hand_r'` (see [pose-model skeleton binding](../pose-model-skeleton-binding/pose-model-skeleton-binding-design.md)),
   so the tracker binds them to `hand21:` landmark markers whatever the body model. The fixed offsets
   stay only for old sequences without manifests.
3. **Same frame loop.** In the single-pass detection pipeline the hand pass runs on the frame already
   decoded, after the body model, instead of decoding the video again.
4. **Every box source.** `detect run` allows `--refine-hands` for every box source. With YOLOX
   fragments each fragment's wrists are refined; the person assignment step later carries the hand
   rows with the body rows, as finalisation already does.
5. **Model choice.** `rtmlib.Hand` (the existing pass) is the default. The spike also ran the SOMA
   model itself on a hand crop: it agreed with the hand pass to a median of 0.05-0.07 of the hand
   size, so it is a possible second backend that needs no extra model download, but it adds 77-channel
   inference per hand. Not built unless the measurement below prefers it.

## 3. Open question: is it worth it on SOMA runs

The spike's hand-crop test gave no tracker-level gain, but it judged body joints and its finger labels
were unreviewed. What is measured:

- On the labelled trial, SOMA runs with and without refined hands, hierarchical tracker configuration
  (hands in their own stages): wrist and hand-stage marker distance to the label-fed run, the
  failure-event table, and visual review of hand pose on the rendered video.
- Pass cost per frame from the detection profiling.

If refined hands do not improve the hands, the pass stays available and off by default for SOMA runs;
it stays on by default for COCO-133 runs, where it was validated.

## 4. Acceptance

- `detect run` refines hands for YOLOX, mask and run box sources, for COCO-133 and SOMA-77 runs.
- A SOMA run with refined hands tracks with the default skeleton's `hand21:` markers in the hand
  stages; a COCO-133 run's hand rows bind exactly as before.
- The default for SOMA runs is set from the measurement in section 3.

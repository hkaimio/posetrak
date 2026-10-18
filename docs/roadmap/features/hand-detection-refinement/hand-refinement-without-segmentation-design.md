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
   `PoseModelSpec.wrists` / `elbows` hold the keypoint names per side and `arm_indices()` resolves them;
   `HandPass(pose_model=...)`, `HandPass.supports(pose_model)` and `HandRefinementPipeline` use them. The
   older COCO-133-only module constants (`_WRIST_IDX`, `_ELBOW_IDX`, `_HAND_BASE_IDX`) remain for the hand
   redetection of the keypoint editor, which works on COCO-133 sequences.
2. **Own layout.** Refined rows are written with a `hand21` manifest under `obs_source = 'hand_l' |
   'hand_r'` (see [pose-model skeleton binding](../pose-model-skeleton-binding/pose-model-skeleton-binding-design.md)),
   so the tracker binds them to `hand21:` landmark markers whatever the body model. The fixed offsets
   stay only for old sequences without manifests. Finalisation already writes this manifest for every
   sequence that has hand rows, so nothing changed in the stored layout: a COCO-133 and a SOMA-77 run bind
   the same 20 `hand21:` markers of the default skeleton (the hand root has none; the wrist marker of the
   body layer serves).
3. **Same frame loop.** In the single-pass detection pipeline the hand pass runs on the frame already
   decoded, after the body model, instead of decoding the video again.
4. **Every box source.** `detect run` allows `--refine-hands` for every box source. With YOLOX
   fragments each fragment's wrists are refined; the person assignment step later carries the hand
   rows with the body rows, as finalisation already does.
5. **Model choice.** `rtmlib.Hand` (the existing pass) is the default. The spike also ran the SOMA
   model itself on a hand crop: it agreed with the hand pass to a median of 0.05-0.07 of the hand
   size, so it is a possible second backend that needs no extra model download, but it adds 77-channel
   inference per hand. Not built unless the measurement below prefers it.

## 3. Is it worth it on SOMA runs

### Method

Labelled aikido trial (three people, five cameras), Roosa over 51.7-56.8 s, the range with the operator's
labels. Reference: the label-fed run of that range. Input: SOMA-77 detection with mask boxes, with and
without the hand pass on the same boxes. Tracker: the hierarchical two-hand configuration, optimised
build, the reference's calibration. Distances are mean / maximum over 52.5-56.8 s of the joint origins
(wrist = hand joint, fingers = finger and thumb joints) to the reference; twist is the difference of the
hand's rotation about the forearm axis to the reference, orientation the geodesic angle of the hand frame.

Skeleton configurations (all SOMA body markers, the hand21 markers added with `skeleton compose`):

- *base*: spike skeleton, SOMA finger markers in the hand stages, no hand21 markers, sequence without hands.
- *both*: SOMA finger markers and hand21 markers in the hand stages (the hand rows bind the hand21 markers,
  the body row binds the 44 SOMA finger markers too).
- *hand21 only*: the layout of the default skeleton: wrist and the index and little-finger base joints of
  SOMA in the main stage, the hand stages hold the hand21 markers only.

### Results

| Run | wrist cm | fingers cm | body cm | forearm twist L / R deg | hand orientation L / R deg |
|---|---|---|---|---|---|
| base, no hands | 23.4 / 72.4 | 26.0 / 81.9 | 13.6 / 55.3 | 25.8 / 23.6 | 49.0 / 95.0 |
| hands, both | 25.9 / 73.7 | 28.6 / 83.3 | 14.9 / 56.5 | 42.5 / 24.6 | 68.1 / 104.2 |
| hands, hand21 only | 24.5 / 70.3 | 27.6 / 82.4 | 14.7 / 54.6 | 33.6 / 20.4 | 60.0 / 72.4 |
| no hands, hand21 only layout | 23.8 / 73.0 | 28.9 / 84.2 | 14.8 / 55.3 | 38.3 / 33.2 | 106.7 / 94.2 |

Outside the four failure events (calm windows 52.5-53.1, 53.7-53.8, 54.5-55.3, 56.4-56.8 s), mean wrist
distance: base 12.0, hands both 14.1, hands hand21 only 11.6, no hands hand21 only 13.6 cm; fingers 13.0,
14.4, 12.6, 19.2 cm. In the four failure events (right arm lost twice, left leg, fall) every
configuration is 66-126 cm from the reference, with and without hands.

- The hands do not move the failures: the strongest-peak body errors of the falls dominate every hand
  figure, and the wrist distance is the body's.
- With the default skeleton layout the refined hands help the fingers (12.6 against 19.2 cm without hands,
  calm windows) and the forearm twist of the right hand (15 against 35 deg), and are neutral on the wrist
  (11.6 against 13.6 cm). Against the spike skeleton, which keeps the SOMA fingers in the hand stages, they
  are no better (wrist 12.0 base, 11.6 hand21 only) and worse with both finger sets (14.1, twist 46 deg);
  the likely cause is two observations of the same knuckles with different noise scales, not tested separately.
- Over the whole range the refined hands are not better than the body model's own fingers (wrist 24.5
  against 23.4 cm, twist L 34 against 26 deg). One person and one 4-second range is a small sample.
- The 'both' configuration is the worst; the hand stages of the default skeleton therefore carry hand21
  markers only, and the SOMA finger markers beyond the two base joints stay out of the skeleton.

Pass cost: 3058 camera-frames (5 cameras, 5.1 s) took 259 s without and 463 s with the hand pass, which is
67 ms per camera-frame of wall time (the hand detector at batch size 1 per crop, 42 ms per camera-frame,
dominates the hand model stage; the hand pose model is batched). Detection of this range is 1.8 times as
slow with the pass.

### Decision

No consistent gain in tracker-level hand accuracy on SOMA runs at 1.8 times the detection time, so the
pass is **off by default for SOMA runs** and available (`--refine-hands`, the Refine hands checkbox). It
stays on by default with mask boxes for COCO-133 runs, where it was validated. The default is fixed after
the operator's review of the hand pose on rendered video.

## 4. Acceptance

- `detect run` refines hands for YOLOX, mask and run box sources, for COCO-133 and SOMA-77 runs.
- A SOMA run with refined hands tracks with the default skeleton's `hand21:` markers in the hand
  stages; a COCO-133 run's hand rows bind exactly as before.
- The default for SOMA runs is set from the measurement in section 3.

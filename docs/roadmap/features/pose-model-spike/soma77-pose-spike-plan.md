# Spike: a DINOv3 / SOMA-77 2D pose model as posetrak's pose detector

Status: plan proposal for a prototyping spike. Nothing in it is built yet.

## Why

posetrak's markerless path uses a ViTPose-L whole-body model (COCO-WholeBody, 133 keypoints) on
person crops. Its keypoint set has no point on the trunk between the hips and the neck, the face
takes 68 of the 133 keypoints, and its accuracy limits the tracker wherever the person is far from
a standard pose (rolls, throws, partly hidden limbs).

NVIDIA has released a 2D pose model with a different design, as part of GEM-X
(`github.com/NVlabs/GEM-X`; weights `nvidia/GEM-X` on Hugging Face under the NVIDIA Open Model
License, `vitpose.pth` and `onnx/vitpose.onnx` of about 3.4 GB each):

- the backbone is DINOv3 (ViT-H: patch 16, width 1280, depth 32) in a top-down ViTPose arrangement
  with a heatmap head;
- the person crop is warped to 256 x 192 and the model returns 77 heatmaps of 64 x 48, decoded by
  argmax with sub-pixel refinement and flip averaging, as (x, y, confidence) per keypoint;
- the keypoints are the SOMA-77 skeleton's joint centres: Hips, Spine1, Spine2, Chest, Neck1,
  Neck2, Head, HeadEnd, Jaw, the eyes, shoulders, arms with finger joint chains, legs with toe
  chains. There are keypoints on the spine and chest, and on the finger joints.

The same release contains a video 3D model (SOMA body parameters per frame from the 2D keypoints
and image features). It is out of scope of this spike except as a later option (section Phase 4).

Questions the spike answers:

1. Does the model detect the body more accurately and more stably than the current ViTPose-L, on the
   joints they share and on the ones only SOMA-77 has (spine, chest, finger joints)?
2. Does a tracker fed with its keypoints produce better skeleton poses than one fed with COCO-133,
   measured against an independent reference?
3. Does it hold up on multi-person footage with people occluding each other, and on complex poses?
4. What would it cost to integrate for real: keypoint-to-skeleton mapping, storage, editing, hands?

## Scope

In scope: run the released 2D model on posetrak's existing person boxes, store its output as a
pose sequence, view it, compare it with the current detector and with an independent reference,
track with it, and decide.

Out of scope: training or fine-tuning a model, the NIM container, the 3D model, any change to the
existing COCO-133 path, replacing the detector for production use.

## Inputs

- The reference capture: a person wearing reflective markers filmed by four usable cameras, with
  two trials that are already processed: a calibration clip (standing and squatting movements,
  starting in a T-pose) and a clip of rolling falls with fast movement. For both there are
  person masks and boxes (segmentation-driven detection), a markerless tracking run with the
  current detector, and marker-based tracking runs with human-reviewed dot assignments.
  `python/posetrak/markers/` and the run register of that capture describe them.
- Multi-person aikido footage with people occluding each other (Phase 3); needs to be chosen and
  its cameras calibrated if not already.

## Phase 0: model bring-up and joint names (about half a day)

1. Download `onnx/vitpose.onnx` (and its `.data` file) to the model cache; record sizes, file
   hashes and the license text next to them.
2. Load it with onnxruntime-gpu (CUDA provider, TensorRT provider as a second try). Check input
   and output tensor names and shapes against `VitPoseExtractor` in GEM-X: 256 x 192 input after
   the affine warp, 77 x 64 x 48 heatmaps.
3. Write the pre- and post-processing (box to affine warp with the same centre/scale convention,
   ImageNet normalisation, UDP sub-pixel decoding, flip test with the left-right joint map) and
   check it against the PyTorch `VitPoseExtractor` on a few frames: outputs must agree to well under
   a pixel. If the ONNX file is not usable, fall back to the PyTorch class (needs the DINOv3
   network definition from `facebookresearch/dinov3`).
4. Take the 77 joint names and parents in order from the SOMA-X repository or its technical report
   (arXiv 2603.16858) and write them to a table in the spike notes. Mark which are joint centres of
   the same bones as posetrak's skeleton joints (hips, spine1, spine2, neck1, neck2, head,
   shoulder/upper_arm/forearm/hand, finger chains, thigh/shin/foot/toe) and which have no
   counterpart (Chest, Jaw, eyes, HeadEnd).
5. Measure speed per crop (batch 1 and 16) on the GPU against the current ViTPose-L model.

Exit: the model runs in posetrak's environment and agrees with the reference implementation; the
joint-name table exists; speed is known.

## Phase 1: estimator, storage and viewing (about 2 days)

1. A `PoseEstimator` in `python/posetrak/detection/` (registered under a name such as
   `soma77-dinov3-vitpose`) that wraps the model: `estimate(frame, detections)` returns
   `PoseResult` objects with float32 `[77, 3]` keypoints in distorted pixels, like the existing
   estimators. The person boxes come from the existing segmentation-driven detection, so the
   estimator does no person detection.
2. Store the output as a pose sequence whose `pose_model` is the new name, with a
   `pose_sequence_keypoints` manifest of the 77 joint names. The tracker's session reader binds a
   manifest name to a skeleton marker through `Marker::landmark`; the plan for the spike is to use
   that mechanism. Verify early that a person sequence with a manifest loads: the reader's hand
   handling assumes the full COCO-133 layout and may need a bypass for a manifest sequence.
3. A skeleton variant for the spike: the current person skeleton plus one marker per SOMA joint on the
   matching posetrak bone, named with the SOMA joint name as `landmark`. The offsets are not zero: the
   Phase 0 comparison with reflective dots showed that SOMA's knee and ankle joints lie several
   centimetres above the anatomical joints and that the displacement is stable (spread about 1 cm), so the
   offsets are calibrated per bone against the marker-based reference, like dot placements. A SOMA joint
   with no counterpart (Chest) is attached to the nearest bone with an offset along it, or left out of the
   tracker and kept for viewing.
4. A `PoseModel` entry for the new layout in `python/app/pose/kp_models.py` (names, groups, limb chains)
   so the existing person view draws and edits the keypoints.
5. Run it on both trials of the reference capture, all four cameras, and look at it in the viewer.

Exit: both trials have a SOMA-77 pose sequence; the viewer shows the keypoints on the video.

## Phase 2: measurement on the reference capture (about 3 days)

Reference: the skeleton joint trajectories of the marker-based tracking runs. They come from a
different measurement (reflective dots and reviewed assignments), so they are an independent
estimate of where the joints are; where they are wrong they are wrong in different ways from a
detector's.

2D comparison. Project the reference skeleton's joints into every camera to get a reference pixel for
each joint with a counterpart. For each detector (current ViTPose-L, the new model) report, per joint
and camera, the error in pixels and as a share of the person's size, the share of frames in which the
confidence is above a threshold, the frame-to-frame jitter of the detected keypoint, and the
confidence of detections that are far wrong (is the confidence usable as a measurement weight).
Report the joints both models share, and separately the SOMA-only ones (spine, chest, finger joints
and toes) against the reference where it has them. Split the numbers by pose: the T-pose start,
squats, rolling falls.

3D comparison. Run the markerless tracker on each detector's keypoints with the same settings,
and compare the joint positions and angles of the result with the reference over time: error per
joint, time to a valid start without a T-pose, behaviour during the rolls, and the hips and back.
A tracker run needs the spike skeleton variant of Phase 1 for the new detector.

Hands. The new model's input is a full-body crop of 256 x 192, so a hand covers few pixels and finger
joint detections may be coarse; the same model also accepts a crop that contains only a hand, which
would make a separate hand model unnecessary. Measure both separately: the dot markers on the index and pinky
side of the hands give a reference for the knuckle region, and the current hand refinement pass
(a second pass on hand crops) is the comparison for finger quality. The question is whether the
finger joints of the new model are good enough to use directly, to seed the hand pass, or to
ignore.

Gate 1. Continue to Phase 3 only if the new model, on the reference capture, is clearly better
somewhere that matters (trunk, hips, rolls) or equal where it is the same and cheaper to use, and
the integration problems found in Phase 1 are tractable. Otherwise write up the result and stop.

## Phase 3: multi-person footage (only after Gate 1; about 3 days)

1. Choose and prepare the aikido footage: cameras calibrated and synchronised, person masks or boxes
   per person (the existing multi-person segmentation path), the people's identities across frames.
2. Run both detectors on it. Count identity-related failures: keypoints of one person landing on
   another, boxes swapping, missing detections while a person is occluded, and how quickly the
   detection recovers.
3. Where possible, add a small hand-labelled set: a few dozen frames per camera with the occluded or
   complex poses, labelled for a subset of joints, to measure error directly where the marker-based
   reference does not exist (no markers on this footage).
4. Track with both detectors in the multi-person tracker and review the results visually on the
   hardest passages; report the same measures as in Phase 2 where the labelled set allows.

Gate 2: decide whether to integrate, and in which form (replacement, a second detector fused
with the first, or only for the trunk keypoints).

## Phase 4: optional follow-ons (not planned in detail)

- The 3D model of the release (SOMA parameters from video, per camera): measure its depth and global
  position error against the multi-camera reference, and test it as a tracker initialiser and as a
  weak per-camera 3D observation. It needs the SAM 3D Body image features (a further 2.1 GB model with
  its own license) and one person per call.
- Fine-tuning the 2D model on pseudo-labels made by projecting the multi-camera result into each
  camera, if the base model is promising but wrong in ways specific to the people or cameras of
  interest.

## Deliverables

- the estimator and the spike skeleton on this branch, kept out of production paths;
- a notes file with the joint-name table, speed numbers, the 2D, 3D and hand measurements, and the
  multi-person findings, kept outside the repository until a decision is made;
- a recommendation at each gate.

## Risks and open points

- The ONNX file may not run under onnxruntime-gpu, or may need an operator the CUDA provider lacks;
  the PyTorch path is the fallback and needs network access to the DINOv3 code.
- The released 2D weights are described only through code and file listings; the training data is
  not documented in the sources read. A model trained on synthetic renders may behave differently on
  real footage than on the reference capture's clean studio-like scenes.
- Licensing: the weights are under the NVIDIA Open Model License; the DINOv3 code and weights are
  under Meta's DINOv3 license; the 3D route adds the SAM license. Read all three before any
  distribution, not only for the spike.
- The hand layout of the session reader and of the hand-refinement code is specific to COCO-133;
  using SOMA hand keypoints beyond the viewer may need that code generalised.
- Joint-centre keypoints depend on the model's idea of the person's proportions, not on a visible
  point; comparing them with a reference needs a joint's definition to match, which is judged joint by
  joint in Phase 0.

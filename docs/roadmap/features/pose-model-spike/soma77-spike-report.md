# Spike report: GEM-X / SOMA-77 pose model and candidate keypoints

Status: report on the finished spike described in `soma77-pose-spike-plan.md`. The code lives on the
local branch `feature/soma77-pose-spike` (not pushed); the data and most scripts live outside the
repository (section 12).

## 1. Summary

- The released GEM-X 2D model (DINOv3 ViT-H, top-down heatmaps, 77 SOMA joint centres) runs inside
  posetrak, is fast enough (about 12 ms per crop at batch 16 on an RTX 4080 SUPER) and its keypoint
  set suits the tracker better than COCO-WholeBody on the trunk, the head and the hips. Measured
  against reference data it is not more accurate than the current ViTPose-L (worse on the rolling-fall clip);
  the operator judged the retargeted result visibly better, mainly through the trunk and head.
- Out of the box it needed three fixes before it was usable: heatmap gating with the person mask
  (otherwise detections land on room features), a larger crop (1.6 times the longer box side; the
  default produces half-size skeletons on some frames), and per-bone offsets between the model's
  joint centres and the skeleton's joints.
- In crowded multi-person footage (three aikido practitioners, five cameras) the strongest heatmap
  peak is right on about 94% of the visible keypoints, but a correct candidate exists among the
  several maxima of a keypoint on 99% of them. Linking the maxima into tracklets and choosing among
  them with local features (peak, tracklet length, agreement of the cameras, mask) raises the
  correct share by 1.5 points over the strongest peak, and ten more signals tried on top added
  between nothing and three points each.
- The errors that matter are short. Failures of the tracked skeleton last typically less than 0.5 s
  and sit in the falls and throws, where the visible-keypoint accuracy of the picks is lowest (93% in
  the hard windows against 99% elsewhere) and where about half of all camera-keypoint-frames are
  invisible, yet 86-92% of the invisible ones still get a pick. A range-wide accuracy figure hides
  them.
- Nothing tried at the level of choosing a candidate (before the tracker) removed those failures.
  Two independent reviews (section 9) added that the error sources other than the detector (calibration,
  the measurement model, correlated observations) were not separated. The next steps (section 11) start
  with controls that locate the error source, then short-window robust estimation and 3D correction;
  gap filling with a motion model comes later. A new GEM-X release that NVIDIA has announced may change the candidate statistics.

## 2. Goal and questions

From the plan: replace or complement the whole-body ViTPose-L detector with the GEM-X 2D model and
answer (1) is it more accurate and stable, (2) does a tracker fed with it give better skeleton poses
against an independent reference, (3) does it hold up with people occluding each other, and (4) what
would integration cost.

During the multi-person tests the goal widened: a heatmap has several local maxima, and with two
people close together the strongest one is often the neighbour's joint. The question became whether
keeping every maximum as a candidate, linking the candidates into tracklets, and assigning tracklets
to each person's keypoint slots (like reflective dots) gives better and more automatic tracking, and
whether tracklet-level editing could replace the per-frame keypoint editing of the application.

## 3. Material and method

- Reference capture with reflective markers (calibration clip with squats and a T-pose start, and a
  clip of rolling falls), used for phases 0-2. The marker-based tracking runs share the COCO keypoint
  markers of the base skeleton, so they carry ViTPose-L's errors; a dots-only run is an independent
  but sparse reference.
- Multi-person trial: five cameras, three practitioners (mask labels 1, 2, 3), 38.08-66.44 s,
  tracker rate 120 Hz. Segmentation masks and boxes come from the existing segmentation-driven
  detection; the masks were edited by the operator during the spike.
- Ground truth: the operator labelled with a purpose-built tool (section 4) two 4-5 s ranges: 43.5-47.5 s
  (labels 1 and 3) and 51.7-56.8 s (labels 1 and 2), body keypoints in four cameras (the fifth has an
  unreliable segmentation and calibration), plus not-visible marks. Finger keypoints were not
  labelled individually.
- Metrics: a pick is correct when it lies within 3% of the person's box height of the label; per
  frame, camera and keypoint the outcomes are correct, wrong, not picked, and (for invisible keypoints)
  false pick. At the tracker level the 3D joint distance to a run fed with the operator's labels, and
  the operator's review of rendered skeleton videos, which produced a list of 17 failure events.

## 4. What was built

- Model bring-up and estimator: ONNX Runtime estimator with the GEM-X pre- and post-processing, flip
  test, parabolic sub-pixel refinement, person-mask gating of the heatmaps, 1.6 crop.
- Storage and tracking: SOMA output as a pose sequence with a keypoint manifest of the 77 joint names;
  the tracker reads such sequences through the skeleton markers' `landmark` names; a skeleton variant
  with one marker per SOMA joint and per-bone offsets calibrated against a reference run; a SOMA
  keypoint model entry for the viewer (with its own bone pairs).
- Candidate pipeline: extraction of all heatmap maxima (local maxima of 5x5 cells, at least 0.05, at
  most six per keypoint), multi-camera agreement flag by triangulation, constant-velocity tracklet
  linking (gaps of up to three frames), a proposal per keypoint (agreeing candidate, else the
  continuation, else the strongest), and incremental re-extraction when masks change.
- Ground-truth labeller (PySide6): five camera views with candidates and trails, per-camera timelines,
  edits as frame spans with modifier-key scopes, assignment of a tracklet to another person, rejection
  of tracklets and rectangles of them, not-visible marks per keypoint or body-part group, suggested
  not-visible frames from a visibility model, label conversion onto re-extracted candidates.
- Tracker fixes found on the way: IK initialisation triangulated pair-difference and velocity
  observations as positions (only position observations are triangulated now); the hierarchical hand
  stage paired smoothed frames with steps by count instead of by timestamp. Tracking runs got names.
- Supporting scripts: tracker-run builders that feed the tracker labels, proposals or filtered
  proposals, event scoring, rendering of grid videos with the skeletons, BVH export of the runs.

## 5. Findings by topic

### 5.1 The model and its output

- Speed: about 23 ms for one crop, 12 ms per crop at batch 16, 34 ms with the flip test.
  Extracting one person's candidates over the whole 28 s trial of five cameras takes about 15 min; refreshing all three
  persons after mask edits, re-running the model only for frames whose crop changed, about 12 min.
- Parity: the ONNX model reproduces the PyTorch model (heatmaps within 0.012), except for isolated
  keypoints that flip by one heatmap cell where two peaks are nearly equal.
- Joint placement: the hips are centred on the pelvis (the midpoint of the two hip joints lies 0.5 cm
  from the midpoint of the ASIS dots; ViTPose-L is 2.2 cm off). Lower-limb joints are displaced toward
  the joint above by a stable amount in upright poses (ankle about 21% of the segment, 6 cm above the
  malleolus dots) but not in rolls, so a fixed offset per bone is a calibration, not a constant.
- Decoding: the model's heatmap cell convention (i/W) differs from the unbiased convention
  (i/(W-1)); the difference is of the order of half a cell and cannot be separated from the offset
  calibration by the available 3D checks.
- Hands: the same model on a hand-only crop agrees with the hand pass to a median of 0.05-0.07 of the
  hand size, against 0.20 when the fingers are read from the body crop. Finger keypoints from the body
  crop stay unreliable even when confident.
- Failures: (a) without gating, detections land on room features on some cameras and frames;
  (b) with a crop of 1.2 times the box, one camera gives a second heatmap peak at about half scale
  around the crop centre in 14-18% of frames; the frequency falls to 0% at 1.6. A reproducible case set was
  prepared for NVIDIA (section 12). (c) the model was trained on synthetic data with every joint
  labelled, so it guesses hidden joints.
- Confidence: stored confidence below 0.5 is mostly wrong, but between 0.5 and 1 it does not rank
  keypoints; unclipped peak and second-peak ratio are the best single signals (AUC 0.64 overall; the second-peak ratio 0.58-0.73 per body group)
  and none separates good from bad.

### 5.2 Accuracy against ViTPose-L (single person, reference capture)

- Calibration clip, held-out window, leg / arm / head error against the reference run: SOMA with
  calibrated offsets and gating 2.4 / 4.0 / 2.6 cm, ViTPose-L 1.5 / 3.4 / 2.9 cm (the reference uses
  ViTPose-L's keypoints, which favours it). The slow start seen with zero offsets (a thigh 16-22 cm off
  for the first 11 s) disappears with calibrated offsets, gating and the 1.6 crop.
- Rolling-fall clip against the dots-only reference: legs 6.2 / 5.3 cm (before / after 229 s into the clip), arms 9.5 / 8.5, head 8.6 / 6.5 for SOMA, against 5.9 / 3.9, 6.9 / 6.2, 7.4 / 5.6 for
  ViTPose-L. SOMA is worse in all six figures (the dots-only reference is independent of both
  detectors, but it is itself a tracker output). On the calibration clip it is mixed (worse on legs and arms, better on head).
  There is no measurement here that shows the new model more accurate in 3D.
- The operator retargeted the SOMA tracking to a character rig, saw a clear difference from the
  ViTPose-based tracking, and judges the torso and head pose and the hip positions to be better. The
  measurements above support the hips; the rest rests on visual judgement.

### 5.3 Multi-person footage and candidate keypoints

- First single-person SOMA run on the aikido trial: usable for most of the clip, with a failure at a
  high fall (about 52 s) and a start that was wrong because of the initialisation bug (fixed).
- Candidates: among samples where at least three cameras agree, the agreeing candidate is the
  strongest peak in 86-96% (trunk 96, head 89, wrists 89, fingers 87, legs 86), the second in 4-12%.
  Mixed tracklets (right for part of their length) are common for two of the three persons: 18-25% of
  tracklets of at least 15 detections agree with other cameras for only part of their detections.
- Fragmentation depends on the body part: median tracklet length 287 frames for head and neck, 227 for
  the torso, 186 for the arms, 105 for legs and feet, 56 for the wrist and 45 for the fingers.
- Against the operator's labels (range 1, 107,589 camera-keypoint-frames with a labelled position):
  strongest peak 94.2%, proposal 95.7%; a candidate within 3% of the label exists on 99%. Extending
  to more ranges, the proposal is right on 96% of visible keypoints. These figures measure
  association, not localisation: the labels were made by accepting, rejecting or reassigning
  candidates (about 89% of labelled frames are accepted proposals), so a candidate near the label
  exists largely by construction, and a visible keypoint with no usable candidate can only have been
  marked not visible. The 3% tolerance is also a share of the box height, which shrinks when a person
  is on the mat.
- The dominant error is not a wrong position but a pick on an invisible keypoint: 13% of the picks
  against 2.1% wrong positions (over the three labelled person-ranges); 86-100% of the keypoints the
  operator marked invisible still get a pick.
- Visibility is predictable once the marks are complete: a six-feature logistic model (heatmap peak,
  tracklet mean peak, tracklet length, agreement, own-mask, other-mask) reaches an AUC of 0.89-0.92
  (0.83-0.95 for a held-out camera) and transfers across ranges. Dropping the picks it calls invisible,
  or weighting them down, did not bring the tracker closer to the label-fed run.
- Pick quality in the hard windows (53.0-54.5 and 54.9-56.4 s of range 3) against elsewhere: visible
  keypoints correct 93.2% / 92.0% (two persons) against 99.4% / 98.9%; wrong picks 6.8% / 7.8% against
  0.5% / 1.0%; about half of all camera-keypoint-frames are invisible there (14,000 against 3,500).

### 5.4 Tracker level

- In range 1 the tracker run fed with the operator's labels was no closer to the labels than runs fed
  with the strongest peak or the proposal (median 0.05 of the box height in all three); better picks
  changed the 3D result by a median of 1-3 cm, with 90th percentiles of 4-35 cm.
- In range 3, mean joint distance to the label-fed run over 52-56.8 s: strongest peak 21.3 cm and
  proposal 10.9 cm for one person; 10.3 and 6.8 cm for the other. The proposal therefore helps clearly
  where the strongest peak fails, but in the hard windows the proposal's runs still differ from the
  label-fed run by 20-70 cm in the same places where the operator's review found the skeleton wrong.
- Where the proposal's errors hurt (range 3, hard windows, joint distance to the label-fed run, median /
  90th percentile): proposal 4.1 / 25.2 cm; operator positions on the visible keypoints with the
  proposal's picks kept on the invisible ones 3.5 / 19.1 cm; proposal positions kept but the picks on
  the keypoints the operator marked invisible dropped 1.7 / 15.9 cm. Removing invisible picks is worth
  several times more than correcting the positions. (Figures from the working notes of the spike.)
- The label-fed run is a controlled comparison, not a 3D reference: it shares the tracker's skeleton,
  calibration and noise model, and it was not reviewed in video like the other variants.
- A three-person proposal run over 43.5-66.4 s against the strongest-peak run differs by a mean of
  4.6-6.4 cm, concentrated in a few windows (a high fall 51-53 s: 31 cm; throw 53-56 s: 14-23 cm).
- The operator's review of the rendered 120 fps videos of both variants listed 17 failure events,
  usually shorter than 0.5 s, in the falls and throws; neither variant was clearly better. A cheap
  failure flag exists: the baseline and proposal runs differ by at least 40 cm in 12 of the 15 events
  that could be scored.

### 5.5 What was tried to improve the choice of candidate, and how it went

| Idea | Result |
|---|---|
| Neighbour grouping (tight keypoint groups follow an edited neighbour) | Right in 82-91% of ambiguous cases on agreement data (fingers 94%), but no gain in labelling or against labels |
| Own-mask / other-mask share as features | Adds 0.0-0.1 points |
| Cross-person pooling (candidates of the other persons as extra candidates) | The label is reachable only through another person's candidate in 0.2-0.3% of frames; +0.3-0.4 points |
| Visibility model as hard filter or soft weight | Predicts visibility (AUC 0.9), but no consistent tracker gain; soft weight improved some forearms and a median |
| Hand-crop disagreement as an occlusion signal | No signal beyond the existing ones |
| Detecting identity jumps inside tracklets from local signals | Not possible: the switches happen along smooth tracklets |
| Dropping picks that collide with another person's pick of the same keypoint after approaching it | Drops bad and good picks at nearly the base rate (16% bad against 15%); the tracker got further from the label-fed run (15.4 against 10.9 cm) |
| Same, judged by whether the pick moves like the other person's mask | AUC 0.62 for bad picks; too weak alone |
| Pose-guided reassignment (candidate nearest the tracked pose) | Works only as a gate: invisible picks fall from 86-92% to 33-43%, wrong picks from 6.8-7.8% to 4.7 / 3.1%, but 8-27% of the visible keypoints are lost; the label-fed pose gives the same result, because the skeleton projects a keypoint within 3% of the box height only about a quarter of the time |
| Finger grouping (finger within 0.15 box heights of its hand pick) | 17.8% of finger picks violate in the hard windows (3.1% elsewhere), but the tracked fingers, hands and forearms are no closer to the reference; hand error drives finger error |
| Re-extraction with boxes from the edited masks | Candidates reach the labels less often in cameras where the boxes changed, and the labels were made on the older candidates, so the comparison is not conclusive |

### 5.6 Why the failures happen (observed, partly explained)

- Identity: a long, good tracklet of one person jumps onto the neighbour's keypoint when the two
  overlap (for example a head directly in front of the other person's head while turning away from
  the camera). The jump is inside a smooth tracklet, so the tracklet-level and pick-level features
  cannot see it; the information that she does not stop there is in the mask and the body state.
- Visibility: in the falls and throws half of the keypoint observations are hidden, the model guesses
  them, and the guesses are confident; limbs behind the body or the other person are the worst.
- Model: the tracked skeleton explains the keypoints only to within 5-10% of the box height, which
  limits what any gate on the tracked pose can do; after a fall the constant-velocity prediction is
  wrong for a while.

## 6. What worked

- The model integrates cleanly (manifest sequences, landmark-bound markers, the tracker unchanged).
- Mask gating, the 1.6 crop and calibrated offsets made it usable; the hips and trunk are better placed.
- Keeping several maxima: the correct keypoint is among the candidates on 99% of visible keypoints,
  and the proposal beats the strongest peak where the strongest peak fails.
- The labeller, with tracklets and trails, makes the failures inspectable; visibility can be predicted.
- Incremental extraction and label conversion keep ground truth usable when masks or boxes change.
- The failure-event list and a baseline-versus-variant disagreement are workable evaluation tools.

## 7. What did not work

- Local features and heuristics for choosing the candidate saturate at about 96% (visible) and do not
  touch the short failures that matter.
- Hand-crop refinement, mask features, neighbour grouping, pooling, collision rules, finger grouping
  and visibility filters gave no tracker-level gain.
- Single-keypoint confidence does not separate guessed keypoints from observed ones.
- Per-keypoint labelling is slow; two 5 s ranges for two persons each took most of the spike's
  labelling effort.
- A dots-only reference, and any reference that shares the COCO keypoint markers, is biased or too
  sparse to judge the SOMA model on the hard motion.

## 8. What was learned

1. Range-wide accuracy hides the failures. Evaluate windows around known failure events, with the
   number of events avoided and their duration.
2. The gap is invisible keypoints, not wrong positions: guessed, confident and often placed on the
   right body part of the wrong person. A tracker needs a visibility estimate from geometry (the
   tracked body, the other persons), not from the detector.
3. Selecting a candidate without the body state saturates. The tracker prediction, the other cameras
   and the other persons' poses are the missing information; a gate on a coarse projection is not
   enough, so selection should happen inside the filter with the covariance.
4. Evaluation references must be independent of the detector under test; operator labels and operator
   review of rendered videos were the only trustworthy ones.
5. Mask quality is a shared dependency: box changes from mask edits changed candidates in the cameras
   with poor segmentation, and nothing downstream is better than the masks and calibration of the
   worst camera.
6. Identical uniforms (aikido) make leg and foot detection hard in the wild; capture-side aids
   (colour-coded clothing, dots) help only in controlled captures.

## 9. Independent reviews

Two independent reviews of this report are kept beside it:
[soma77-independent-assessment.md](soma77-independent-assessment.md) and
[soma77-independent-assessment-opus55.md](soma77-independent-assessment-opus55.md). Their points that
changed the conclusions:

- The accuracy evidence does not support replacing ViTPose-L (section 5.2). SOMA is worse on the
  rolling-fall clip in every listed group. The trunk, head and hip advantage rests on retargeting
  impressions that were not blind and on one hip measurement. A hybrid (SOMA trunk, head and hips,
  ViTPose limbs) is better justified than replacement.
- The labels measure association, not localisation (section 5.3). A small set placed freely with the
  candidates hidden would calibrate both the coverage figure and the "invisible" class.
- The label-fed run is not a 3D reference (section 5.4). The skeleton explains the labelled keypoints
  to within the tolerance only about a quarter of the time, which is a finding about the measurement
  model (joint offsets, landmark definitions, calibration), not only a limit on gates.
- The ablation in section 5.4 says removing invisible picks matters more than the positions. The
  pose-guided gate cut invisible picks the most but was never run through the tracker.
- Error sources other than the detector were not separated: calibration (one camera has poor
  intrinsics, and two cameras needed agreement radii up to 60 px against 12 px), correlated
  observations treated as independent by a filter with a hard Mahalanobis gate, and the shrinking
  tolerance in falls. Synchronisation is expected to be good (blinking-LED sync, under 10 ms).
- Forty-eight of 77 keypoints are fingers that the body crop does not localise; body and hands
  should be scored separately and the fingers dropped from the body pass.
- The failures are few and short with good frames on both sides, which suits short-window batch
  optimisation (candidates as a mixture, robust loss) better than a forward filter, and a single 3D
  keyframe inside a failure may be a cheaper correction than 2D keypoint editing.
- Not taken from the reviews: the claim that the fifth camera can be fixed in a day (its masks are
  unedited too) and capture-side fixes as the answer (the aim is in-the-wild material).

## 10. Follow-up checks after the reviews

First three items of the revised directions (section 11). Same material as before: range 3, two persons,
body keypoints, the fifth camera left out.

- Label triangulation. Each camera's label against the point triangulated from the other cameras'
  labels: median residual 4-5% of the box height for the best camera, 6-7% and 10-11% for the others. It
  is flat in keypoint speed up to 4 box heights per second, the same in head, trunk, arms and legs, and a
  constant offset per camera (median 3-9% of the box height). Shifting a camera's labels in time gives the
  minimum at zero (one camera at +8 ms). The dominant error is therefore calibration, or a constant
  offset of the model's heatmap decoding, not synchronisation and not occlusion. Checking the heatmap
  decoding and flip-test alignment against the camera offsets, and refining the extrinsics and intrinsics
  from the labels, comes before any further work on selection.
- Calibration. The extrinsics of the capture come from an automatic image-feature calibration; two of the
  cameras share one intrinsics calibration and one has imported intrinsics. Refining the extrinsics of
  three cameras by bundle adjustment on the labels (15 Hz, 2,931 points, robust loss) lowers the
  leave-one-out residual of held-out times from 8-10% to 2-3% of the box height. Trained outside the
  hard windows, the held-out hard-window medians fall from 5-13% to 3-7% (the 90th percentile stays at
  17-37%: there the labels and picks are wrong). Focal length and principal point add nothing. The fitted
  corrections are small (0.1-2.3 degrees, 7-33 cm), differ between the two halves and are coupled, so only
  their combined effect is trusted. A single offset shared by all cameras (proportional to the box height)
  explains 40-50% of the gain, per-camera offsets all of it; a decoding offset therefore cannot be ruled
  out by this test, but the decoder's half-cell convention bounds it at about 1.25% of the box height
  (4-6 px) against 11-46 px observed. With the refined cameras the label-fed skeleton projects a visible
  labelled keypoint within 3% of the box height 76-82% of the time outside the hard windows (27-28%
  before; in-sample) and 34% / 64% inside them (20% / 22% before; held-out), and within 6% 92-97% outside.
  The proposal run gets closer to the label-fed run for one of two persons (9.4 to 7.7 cm; 3.8 cm
  unchanged). Calibration is thus the largest single source of the projection error, and it makes the
  pose-guided gate (which was limited by that error) worth repeating.
- Self-calibration. The same bundle adjustment using only the pipeline's own detections (strongest
  candidate in the person's own mask, peak at least 0.5, 15 Hz, 17 s outside the labelled range, robust
  pair gate) gives the same result on the labels, which it never saw: leave-one-out median residual 1.3-2.4%
  of the box height outside the hard windows (4-11% original) and 2.4-5.1% inside them (5-13% original);
  the label-based refinement gives 1.1-2.0% and 2.5-6.8%. The label-fed skeleton then projects within 3% of a
  visible label 74-82% of the time outside the hard windows and 32% / 65% inside, all out of sample. The
  calibration can therefore be refined without labels. Tracker-level distances move less consistently than
  the 2D metrics (the proposal run of one person is 14.6 cm from the label-fed run with these cameras, 9.4
  with the original, 7.7 with the label-refined ones), because the failure events dominate them.
- Fifth camera. Fitting its extrinsics, focal length, principal point and first two fisheye coefficients to
  its own detections against the four-camera points lowers its residual from 17% to 1.9% of the box height
  on the labelled range (1% to 85% of keypoints within 6%). The changes are large and coupled (5 degrees,
  15% focal length, 124 px principal point) and it has not been run through the tracker.
- Reference capture (moving-box extrinsics). The same leave-one-out measure on SOMA detections gives
  1.0-2.4% median per camera (6 cameras, one person, 10% pair gate): as good as the refined aikido
  calibration. The aikido capture's original extrinsics (from scene features, many on the planar mat) were
  clearly worse.
- Decoding shift. The decoder maps a heatmap peak to the left edge of its cell, while the mask gate uses cell
  centres. On the reference capture (good extrinsics, fixed) one shift shared by all detections has a sharp
  minimum at half a cell to the right in x (+0.52 cells; the vertical shift is not constrained): the
  reprojection residual falls from 1.33% to 0.76% of the box side. On the aikido capture the shift cannot be
  separated from the cameras (with the original cameras the fit gives -1.5 cells, with cameras refined from
  detections 0): applying half a cell lowers the original cameras' residual by 20-30% and leaves the refined
  ones unchanged. The shift is real but small (about 1.3% of the box height, 4-6 px) and the large offsets are
  the cameras. A decode fix (peak index plus 0.5) changes stored sequences and the offsets fitted under the old
  decode, so it belongs with the next re-extraction. The shift is the same (0.49-0.52 cells in x) for crop
  enlargements 1.2, 1.6 and 2.0 and for parabolic and quarter-cell sign refinement, so it scales with the
  crop, is present with the settings GEM-X ships and does not come from how this project calls the model;
  its decoder (`keypoints_from_heatmaps`) maps the index to the cell's left edge as ours does. Whether it is a
  defect or a training convention is not known; the vertical shift is weakly constrained (about 0.5). The 1.3-cell difference between SOMA and ViTPose-L in an
  earlier comparison is a difference between the two extractors, not a measure of error.
- Observability. In the hard windows 11-20% of the body keypoint-times are seen by fewer than two of the
  four cameras (under 3% elsewhere), 45-54% by two or fewer; those cannot be recovered by any selection.
- Filter. NIS per degree of freedom stays at or below 1 in the failure events (0.9-1.0 against 0.6-0.8
  elsewhere), so the filter is not overconfident. The gate rejects 5% of visible-labelled observations in
  events against 3% elsewhere and only 0-8% of the invisible-labelled ones. The failures are coherent
  wrong observations that the innovation statistics do not reveal.
- Pose-guided gate through the tracker. Mean body-joint distance to the label-fed run, proposal against
  gate (and gate exclusive between persons), over 52-56.8 s: 9.4 / 3.8 cm against 7.6 / 3.4 (13.5 / 3.2);
  hard windows 12.9 / 4.8 against 10.1 / 4.0 (16.8 / 3.8). A modest gain, 10-20%, and exclusivity hurts
  in one of the two persons.
- Failure flag. A baseline-versus-proposal difference of 40 cm or more in any joint catches 13 of the 15
  events with 22 windows (19 per minute of person-footage, about 16% of the time flagged); half of the
  windows contain a listed event, the others are unverified. Restricted to body joints it catches 9
  (12 at 25 cm, with 18 windows per minute).
- Edited ViTPose-L runs (human-edited, so a quality reference, not a raw-detector control). In 13 of the
  15 events both SOMA variants differ from it by 40 cm or more, which confirms them independently. Mean
  body-joint distance to it is 9-13 cm outside the events (landmark and skeleton differences) and
  17-20 cm inside for two persons.
- Pipeline rerun with the decoder fix and the cameras refined from the pipeline's own detections (new
  extraction, linking, selection and tracker runs; the fifth camera stays out; per-bone skeleton offsets
  were not refitted). Mean body-joint distance to the label-fed run, range 3, proposal: 9.4 / 3.3 cm
  (two persons) against 14.6 / 3.4 with the old decoder and the same cameras, and 9.4 / 3.8 with the
  old decoder and the original cameras; hard windows 13.1 / 4.6. The pose-guided gate gives 8.6 / 3.8,
  its exclusive variant 12.9 / 4.0; baseline 16.3 / 3.8. So the fix and the refined cameras together
  recover what the refined cameras alone lost in one person, but do not improve on the original
  cameras with the old decoder: the tracker-level gain is small. The label-fed skeleton explains the
  labels within 3% of the box height for 70% / 78% outside the hard windows and 34% / 59% inside.
  Failure flag over the whole trial: unchanged (13 of 15 events at 40 cm, 18 windows per minute, 10 of
  21 windows contain an event). Distance to the edited ViTPose-L runs: 11-15 cm outside the events,
  19-24 cm inside, no change from before. The decoder fix removes a systematic multi-camera
  inconsistency, but the failures that matter (wrong person or limb picked in contact) are unaffected.
- Fifth camera refined (wide "mega" lens mode). Fitting its pose, focal length, principal point and two
  fisheye coefficients to the other four cameras' points gave a 1.4% median residual (13.5% before) but
  changed the focal length by -15% and the camera position by 1.3 m: focal length and distance trade off
  when the subjects fill a small part of the frame, so the fit is not the real geometry. Fitting the pose
  alone gave 1.8% (camera moved about 0.9 m and turned about 4.5 degrees). Tracking with the camera as a
  position camera was worse in both cases (mean body-joint distance to the label-fed run, range 3, two
  persons: 15.0 / 5.0 cm against 9.4 / 3.3 with the original camera used as velocity-only); as a
  velocity-only camera the refined pose changes nothing (10.0 / 3.3). The lens in this mode fits neither
  the pinhole nor the fisheye model, so a low residual near the subjects does not mean correct geometry
  elsewhere in the frame. Using the camera for position needs an intrinsics calibration recorded in that
  mode with a model that can describe it; until then it stays velocity-only.
- Manual correction (simulated with the operator's labels). Windows come from the failure flag (baseline
  against proposal, 40 cm): three for one person, two for the other, 0.3-1.8 s each. Correcting the picks
  of all cameras within 0.1 s of a keyframe, with the proposal elsewhere, barely helps: mean body-joint
  distance to the label-fed run 9.4 -> 9.4 (3 keyframes), 9.2 (4), 8.5 (6) cm for one person, 3.3 -> 2.8-2.9
  for the other with 2-3 keyframes. The filter does not hold the correction, and the wrong picks after the
  keyframe pull the skeleton back. Correcting the whole flagged windows reaches 5.8 / 2.4 cm, and the
  windows with 0.3 s margin 3.2 / 1.2 cm (hard windows 3.3 / 1.7), i.e. the label-fed level. In those
  windows only 7-9% of the label-visible keypoint-frames have a wrong pick, but removing those takes about
  1800 and 670 corrections at the level of one keypoint, camera and tracklet segment. So a correction must
  cover the whole failure window; single-frame keyframes are not enough, and per-keypoint tracklet
  decisions are too many to be a small manual load. A cheaper correction unit would have to act on a
  person or a body part for the whole window (for example "this limb is the person's, for this
  interval" applied to every keypoint and camera), or the tracker would have to keep a keyframe's
  constraint over a longer interval.
- Grouping of the wrong picks. In the same windows the wrong picks (more than 3% of the box height from
  the label) form 452 and 245 continuous segments per camera and keypoint (the earlier figure of about
  1800 and 670 counted a segment again at every tracklet change and overstated the number of decisions).
  Grouping segments of one camera that overlap in time by at least half and lie within two joints of each
  other in the skeleton gives 223 and 161 groups, 149 and 122 when merged over cameras; the 9 and 5
  largest groups hold half of the wrong frames and 27 and 25 hold 80%. The wrong frames are mostly feet,
  toes and hands; knee keypoints form 12-13 groups of their own per person. A grouped correction is
  therefore one to two orders of magnitude fewer decisions than per-keypoint ones, but its effect on the
  tracker was not measured.
- Time-reversed tracker. The tracker was run unchanged on the whole trial with the observation times
  mirrored (no code change; the tracker reads the timestamps of the observation table), for the
  strongest-peak and the proposal sets, and the forward and backward runs compared. Failure flag at 40 cm
  (any joint): forward against backward on the strongest-peak set catches 14 of 15 events with 21
  windows per minute of person-footage, 11 of 24 windows containing an event; baseline against proposal
  (forward) 13 of 15 with 18 windows per minute, 10 of 21; the maximum of the two 14 of 15, 9 of 22.
  The backward flag is therefore about as good as the baseline-against-proposal one and not
  independent of it, which is expected since both directions are fed the same observations. Quality:
  the backward run is worse than the forward one against the label-fed run (18.4 / 6.4 cm against 9.7 /
  3.5 cm, two persons, range 3) and the average of the two is worse than forward alone; against the
  edited ViTPose-L runs the two directions are level and the average is slightly closer inside the
  events (17.6 against 19.8 cm for one person, 19.0 against 19.3 for the other). The large gap in the
  label-fed comparison has not been explained and may include an artefact of the mirroring.
- Grouped assignment prototype. For every keypoint the candidate is chosen by the support of its whole
  tracklet: the share of frames (window of one second each way) in which the tracklet lies within the
  plausible distance of the current picks of the neighbouring keypoints (limits: 98th percentile of the
  label distances outside the hard windows, times 1.15), plus small terms for multi-camera agreement and
  peak strength; iterated three times from the proposal. Share of label-visible keypoints picked within 3%
  of the box height in the hard windows, proposal against grouped: body 87.9 -> 89.9% and 94.5 -> 97.0%
  (two persons), fingers 81.4 -> 83.7% and 82.8 -> 84.4%; outside the hard windows 98.7 -> 98.9% and
  99.4 -> 99.5%. Through the tracker (mean body-joint distance to the label-fed run, range 3): 9.4 -> 8.8
  and 3.3 -> 3.0 cm, hard windows 13.1 -> 12.1 and 4.6 -> 4.2, about the same as the pose-guided gate
  (8.6 / 3.8). A small gain: the wrong picks that remain are mostly ones that are neighbour-consistent
  (the whole limb or person is wrong together), which local coherence cannot detect.
- Simulated editing with propagation. The labels play the operator, who fixes the largest remaining
  wrong-pick segment of a camera to the labelled candidate, and the grouped assignment re-decides the
  other keypoints around the fixed ones. Clicks needed (summed over the four cameras) to leave 20% of the
  wrong frames: 204 without propagation against 154 (one person) and 114 against 57 (the other); for 50%
  of them 77 against 55 and 44 against 19. Propagation saves a quarter to a half of the clicks, but the
  absolute number stays high because every camera is edited on its own; a decision shared by all cameras
  (a tracklet belongs to the person, seen through the triangulated agreement) was not simulated.
- Cross-camera grouped assignment. The grouped assignment gets a second support term: the distance of the
  other cameras' current picks of the same keypoint (nearest frame within 8 ms) from the candidate's epipolar
  line, smoothed along the tracklet like the neighbour support and iterated jointly over the four cameras.
  Hard-window body picks correct: 87.9 -> 89.6 (neighbour support only) -> 91.0% (cross-camera, weight 1) and
  94.5 -> 96.9 -> 97.7% for the other person; fingers 83.7 -> 84.7 and 84.3 -> 85.4%; a larger weight gives
  a little more on the body. Simulated editing over all four cameras at once (batches of five segments
  between re-decisions, which understates propagation): clicks to leave 50% / 20% of the wrong frames:
  no propagation 75 / >200 and 40 / 105; neighbour propagation 40 / 140 and 5 / 30; with cross-camera
  propagation 30 / 115 and 0 / 20 (the automatic pass alone halves the wrong frames). Cross-camera
  The clicks stay high for the first person; the reason was not analysed.
  The epipolar test is weak along the line, so a triangulated three-camera test might give more.
- Person mask as a candidate-selection term. In the hard windows the labelled correct point lies on another
  person's mask for 10-21% of body points and 34-45% of hand and finger points (outside them 5-12%), so the
  masks are least reliable where the choice is hardest. A bonus for the candidate lying on the person's own
  mask (penalty on another person's) in the cross-camera assignment gives body picks +0.1 to +0.8 points at
  best and costs fingers 1-9 points, more with a larger weight. After a window's masks were corrected by
  hand (a realistic level for the segmentation algorithm; 10-15% of the stored mask labels changed): in
  that window body picks 90.6 -> 91.6% and 97.9 -> 97.9% (best weight, against 91.2 / 97.8 with the old masks),
  fingers 90.6 -> 88.9% and 87.9 -> 86.4% (old masks 86.8 / 86.0). Correct masks recover about half of the
  finger loss but the term still does not pay for itself. Not tried: restricting the epipolar search to
  the mask, and a mask term for the body only.
- Mask term per tracklet, body joints only. Splitting the hard-window picks by part (body = not hand, foot or
  finger; picks within 3% of box height of the label; corrected masks), the per-candidate mask bonus (weight
  0.3) raises body picks but lowers hand, foot and finger picks (hands and feet often leave the mask in fast
  falls): for one person body 91.9 -> 94.3% and 93.0 -> 95.0% (corrected window and later hard window),
  hands 91.2 -> 88.6% and 83.9 -> 78.7%, fingers 79.8 -> 75.8% in the later window. Taking the mask support
  as the mean over the candidate's tracklet within +-2 s softens the loss; applying it to body keypoints only
  removes it: body 91.9 -> 94.2% and 93.0 -> 95.4% (other person 98.2 -> 98.3% and 98.2 -> 99.4%), while
  hands, feet and fingers are unchanged. Weights 0.6 and 1 give no more. The weight was chosen on the same
  windows, and the tracker was not run with it.
- Reading the finger figures. The labels were reviewed for body, hand and foot keypoints only; finger keypoints
  keep the automatic proposal where nothing was edited. Finger "pick accuracy" in the tables above is
  therefore agreement with the unreviewed proposal and is not a correctness measure, and the finger losses
  of the mask term mean "differs from the proposal". The tracker comparisons use body joints and are not
  affected. Pick accuracy is also reported as the share of picks that are the same candidate the label
  holds (equal to the within-3%-of-box-height figure in all measurements so far).
- Tracker with the body mask term. Cross-camera selection plus the per-tracklet mask term on body keypoints
  (corrected masks), range 3: mean body-joint distance to the label-fed run 8.4 cm (all) and 11.3 cm (hard
  windows) for one person, 2.8 and 3.7 cm for the other, the same as without the mask term (8.4 / 11.3 and
  2.8 / 3.8). The roughly two-point gain in body picks does not reach the skeleton.
- Picks on other persons' masks. In the corrected window, 37.7% of one person's body picks from the
  cross-camera selection lie on another person's mask (33.9% with the body mask term), while 22.3% of the
  labelled body points do (the person is partly behind the other, so correct joints sit under the other's
  mask). The selection variants never decide "no pick": a candidate is picked wherever one exists, including
  75% of the body keypoint-frames the labels mark not visible, and the tracker's gate has to reject them.
  This is the invisible-pick error of section 5; a mask score cannot remove it because occluded joints are
  legitimately on another person's mask.
- Gaps in the observation sequences, and the rerun with complete data. Every tracker run of the spike up to
  this point lacked all observations of a person in some intervals (for one person: all four position cameras
  between roughly 53.8 and 54.1 s, one camera for 2.3 s; for the other person the intervals differ and the
  wide-lens camera had a long gap too). The sequences are built from the per-camera track-to-person
  assignments, and those have gaps where the track identity was uncertain, although the keypoint rows and
  boxes of the track exist there and the candidate file (boxes from the person masks) covers the frames.
  The label-fed reference runs had the same gaps, so the hard-window distances to them in the earlier
  paragraphs are distances between runs that had no data from those cameras, not errors against
  labels. The sequences are now built over every frame of the candidate file (boxes from the candidate crops
  where no track box exists, strongest candidates as the base keypoints, one assignment per camera over the
  range). Range 3 (51.7-56.8 s), corrected masks, mean body-joint distance (cm) to the label-fed run built from
  the same complete data, all frames / hard windows, one person then the other: automatic proposal 12.9 / 15.6
  and 4.1 / 5.4; neighbour grouping 10.2 / 11.6 and 3.6 / 4.4; cross-camera 10.1 / 11.3 and 3.2 / 3.9;
  cross-camera with the per-tracklet body mask term 9.9 / 10.7 and 3.0 / 3.6. With complete data the
  selection logic matters clearly more than in the runs with gaps, and the mask term now helps a little.
- What the keypoints in the label-fed runs are. The labels were carried over to the re-extracted candidates by
  matching each labelled position to the nearest new candidate within 6% of the box height (positions with no
  candidate that close fall back to the automatic proposal), and where nothing was edited the automatic
  proposal stands as accepted. Against the frozen label export, 99.2-99.3% of the body, hand and foot
  positions are reproduced (about 350-440 positions per person differ in range 3, about 45-80 are visible in the
  export and unlabelled in the run); the differences are listed per camera and frame by the conversion check.
  The mask is not enforced in any run except as the soft term of the last variant. The wide-lens camera is
  never labelled: its keypoints are the strongest candidates of the person's crop, and 24.5% / 27.1% of them
  lie on another person's mask, against 8-17% of the label-fed picks of the labelled cameras.
- Tooling findings. The person view picks per camera the segmentation run with the most masks; two runs with
  equal counts made it show a copy without the later edits (ties now go to the newest run). Segmentation masks
  are edited in place, so nothing records which masks a derived result saw; this is logged in docs/open-issues.md.
- Strict person-mask filter on candidates. A candidate is kept only if at least half of the detections of its
  tracklet within +-10 frames of its frame lie on the person's own mask; the survivors go through the existing
  selection (proposal, cross-camera grouping or pose-guided gate). Mask labels were refreshed from the current masks;
  the masks had been reviewed for 52.5-57 s only. In the reviewed range the filter removes 41-44% of the candidate
  rows of two persons (12% for the third), and for the whole trial 29-33%. Of the labelled points of range 3,
  the share that still has a candidate within 3% of the box height afterwards: body 77% / 90% in the hard windows
  (one person / the other) and 90-93% elsewhere, hands 78% / 49% in the hard windows, feet about 70-74%. Range 3,
  mean body-joint distance (cm) to the label-fed run, all frames / hard windows, one person then the other:
  proposal 12.9 / 15.6 and 4.1 / 5.4 without the filter, 8.2 / 11.5 and 4.4 / 5.4 with it; cross-camera 10.1 / 11.3
  and 3.2 / 3.9 without, 8.3 / 11.7 and 4.3 / 5.2 with; pose-guided gate after the filter 10.1 / 14.0 and 4.6 / 5.9.
  The filter helps a lot where the person is often hidden behind the other (the proposal improves by a third) and is
  neutral or slightly harmful for the other person and for the better selection logic: it discards correct
  candidates of hidden joints that lie under the other person's mask.
- Wide-lens intrinsics from the calibration board inside the action area. The trial's keypoints cover x 1500-2450,
  y 380-820 px of the 3840x2160 image. 84 of 348 detected board views have enough corners in a margin around that
  area. Fitted to those corners only (a quarter of the views held out): the original 14-coefficient calibration
  gives 1.4 px RMS in the area, pinhole k1,k2 0.9 px, pinhole k1..k3 with tangential terms 0.8 px, rational
  model 0.8 px. The focal lengths differ by 25% between the models (the original 2370 px per radian near the area,
  the fits 1460-1800), so a board in a small area does not fix the scale. Judged by the geometry of the other four
  cameras (extrinsics fitted to triangulated keypoints, half of the points held out): the tracker's fisheye
  calibration 2.3% of box height median, the 14-coefficient one 2.7%, the k1,k2 fit 1.9% and 1.8%, the larger fits 2.3%;
  freeing the focal scale as well moves the other models to a local scale of about 1470-1500 px per radian, the
  value the k1,k2 fit already has. That fit is used for the tracker test.

## 11. Possible directions

In rough order, after the reviews and the follow-up checks of section 10 (which did the first run of items
1-3 below: the error source is mostly calibration or a constant offset per camera, the filter is not overconfident,
the pose-guided gate gives a modest gain and the failure flag costs about one window per three seconds).

0. Calibration (first pass done, section 10): turn the label-based refinement into a refinement from the
   system's own detections (cameras agreeing on candidates, no labels), calibrate the intrinsics of the cameras
   that share one calibration, then repeat the pose-guided gate and the tracker comparisons with the refined
   cameras. The decoding offset stays an open question (a test with a known pixel truth is needed).

1. Locate the error source: triangulate the labels and plot each camera's residual against
   keypoint speed and image position; count the cameras that see each joint in the failure windows;
   read the filter diagnostics (NIS, outliers, gate decisions) in the failure windows to see whether
   the filter is overconfident before a failure and rejects correct observations after it.
2. Run the tracker on the pose-guided gate picks.
3. Failure flag precision: false alarms per minute of the baseline-versus-proposal disagreement
   over the whole trial. Score the ViTPose-L runs (including the edited COCO-133 ones, mapped to
   shared joints) on the failure events. Report body and hands separately; normalise the tolerance by
   body scale.
4. Robust short-window estimation on the failure windows: candidates as a mixture with an explicit
   "none of these" option, a heavy-tailed loss, frames before and after, bone lengths in 3D;
   compare event by event with the filter. Alternatively, adaptive process noise after innovation
   spikes.
5. A posed 3D keyframe inside each failure as a correction: feed the label-fed pose at the midpoint of
   each event to the proposal run and count the events that disappear. If most do, build 3D
   correction and triage in the application, not per-camera keypoint editing.
6. Required visibility accuracy: corrupt the operator's visibility marks at controlled error rates and
   rerun the tracker.
7. Measurement model: per-subject attachment points and bone lengths from the subject's own
   triangulated joint centres including non-upright poses; observation noise from the unclipped peak.
8. Hybrid detector (SOMA trunk, head, hips with ViTPose limbs), drop body-crop fingers, read the
   licences, and rerun the failure-event table when the next GEM-X release arrives.
9. Later: gap filling with a motion prior (reported Apache-2.0 with partial joint constraints,
   neither verified; a solo prior does not know the partner, and it must be evaluated separately from
   faithful capture), mask-conditioned or visibility-aware detectors (unverified leads), a
   time-reversed pass, humanoid rendering for occlusion, recalibrating the camera with poor
   intrinsics.
Stop: further local 2D selection heuristics and extending per-keypoint 2D labelling.

## 12. Pointers

### Code (branch `feature/soma77-pose-spike`, local)

- Plan: `docs/roadmap/features/pose-model-spike/soma77-pose-spike-plan.md` (7be1267, 00c02bb).
- Estimator and options: `python/posetrak/detection/backends_soma77.py` (c396f38 prototype, d472b3e
  sub-pixel, 9432f25 mask gate, 0271c73 crop 1.6); extraction of a pose sequence
  `python/tools/prototypes/soma77_extract.py` (6a238c1 for one track of a multi-person run); spike
  skeleton `python/tools/prototypes/build_soma77_skeleton.py`; viewer model `python/app/pose/kp_models.py`
  (abd6762, 72bd055); offset calibration (e190f37).
- Tracker fixes: 57acfe5 (initialisation triangulates positions only; also on branch
  `fix/init-triangulation-position-only`), 6c9e004 (smoothing by timestamp; also on branch
  `fix/hierarchical-smoothing-lost-steps`), 5c06f76 (names for tracking runs).
- Candidates and labelling: `python/tools/prototypes/kp_labeling/` with `extract_candidates.py` (5f0a3d7),
  `link_candidates.py`, `build_frame_cache.py`, `labeler.py` (47dee32 and the commits up to 1ce24e0),
  `convert_labels.py` (fcf6b07), `train_visibility.py` (2f9f6e3).

### Documentation and notes

- The memory notes `posetrak-pose-model-study.md` (model, phases 0-2, aikido first runs) and
  `posetrak-candidate-keypoints-test-plan.md` (candidates, labelling, all measurements above with the
  scripts that produced them) in the agent memory repository `memory/projects/`.
- Failure events of the operator's review and the event scoring: `ukemi_events.json`,
  `event_scores.py` in the spike data directory.

### Data

- Models, GEM-X sources, phase 0-2 scripts, annotated frames and the case set for NVIDIA:
  `D:\mocap\models\gem-x\` (`spike\`, `bug_report\`).
- Multi-person session: original `E:\mocap\vanhaa\ukemi-tommi-20260509.db` (never written); work copy
  `D:\mocap\vanhaa-spike\session.db` (segmentation run `e862c9af`, edited by the operator; snapshot of
  the masks in `mask_snapshots\`).
- Candidate sets and caches: `D:\mocap\vanhaa-spike\gt5` (current, with labels in `gt5\labels`),
  `gt4`, `gt_frozen\` (frozen range-1 labels); visibility models in `gt5\visibility_model.json`.
- Tracking runs (named in the session database): per-person strongest-peak runs over 38-66 s, proposal
  runs over 43.5-66.4 s, label-fed and variant runs for range 3, listed in the memory notes; BVH files
  in `D:\mocap\vanhaa-spike\bvh\` and grid videos in `D:\mocap\vanhaa-spike\` (`out_grid_*`), copies in
  the OneDrive folder `posetrak\misc`.
- Scripts of section 5.5: `pose_reassign.py`, `finger_check.py`, `mask_motion.py`, `collide.py`,
  `make_pick_run3.py` (modes gt, proposal, vis, xp, xpa, fg), `run_seq.py`, `event_xp.py`,
  `event_timo.py`, `swap_check.py`, `render_grid.py`, all in `D:\mocap\vanhaa-spike\`.

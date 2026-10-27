<!--
SPDX-FileCopyrightText: 2026 Harri Kaimio

SPDX-License-Identifier: Apache-2.0
-->

# GEM-X release spike: keypoint decoder convention and Body Pose

Offline measurements on the multi-person aikido trial (43.5-66.44 s, five cameras, three subjects) of the
keypoint-candidates work (`keypoint-candidates-design.md` section 10.10). Nothing here changes product behaviour. The
scripts are in `python/tools/prototypes/gemx/`.

Terms used below:

- **Window.** The part of the person's square crop the 2D model sees: the central 192 of 256 columns, 0.75 x the
  crop side wide and one crop side high. The 2D model gives 77 heatmaps of 48 x 64 cells over it.
- **Decoder convention.** Where heatmap cell `i` of `W` lies in the window. *Left edge*: `i / W`. *Cell centre*:
  `(i + 0.5) / W` (the product's decoder). *UDP edge*: `i / (W - 1)` (the new official GEM-X decoder). Left edge and
  cell centre differ by a constant half cell; cell centre and UDP agree in the window centre and differ by up to half a
  cell at its edges, which shows as a scale difference (a slope).
- **Cell.** One heatmap cell: about 9 px of a 4K frame at the crops of this trial. Half a cell is 1.25% of the box
  height, about 2 cm on a person.
- **Candidate, same candidate, strongest peak, windowed choice.** As in the keypoint-candidates design: a candidate is a
  local maximum of a keypoint's heatmap; a pick is "the same candidate" as a label when it lies within 0.5 px of the
  candidate the label holds; the strongest peak is the rank-0 candidate; the windowed choice is the tracker's choice of
  a candidate per keypoint over a time window (basis: `'window'`, ace2 pro outside the choice, body-only strict
  filter).
- **Subjects.** The segmentation mask labels 1, 2, 3 (range-3 labels exist for 1 and 2).
- **Body Pose.** The GEM-X single-image 3D SOMA body model (`body_s240000.ckpt`): from a crop, a square box and the
  camera intrinsics it returns SOMA body parameters and 77 joints in camera coordinates (metres) and projected into the
  image.

## 1. Summary

| Question | Result |
|---|---|
| Which decoder convention fits? (2D) | The frozen evaluation files are left-edge decoded (the product's candidates are exactly the frozen ones plus half a cell). Two references prefer left edge over the product's cell centre by 0.4-0.5 cell (section 3). UDP adds a slope that one reference disfavours. The earlier dot-based check in the pose-model study pointed the other way, so the question is open. |
| 3D runs for the convention? | Not run (decision for the operator, section 3.4). |
| Body Pose as a candidate cue (A) | Gate **not met**: the nearest candidate to the Body Pose keypoint is correct in 53.6% of the cases where the strongest peak is wrong (gate 75%; the tracker's windowed choice 63.8%, the tracker prior 64.8%). |
| Body Pose as an occlusion cue (B) | Gate **not met**: AUC 0.61 for picks more than 0.10 box heights off (gate 0.75); the heatmap peak has 0.88. |
| 3D pose and initialisation (C) | Fused over cameras the mean joint error is 10-20 cm; the tracker's own start reaches about 3 cm. Not a replacement; a fallback prior at most. |
| Body shape (D) | Bone lengths within 1-4 cm of the edited runs' skeletons, but nearly the same for all three subjects. |
| Throughput | 47 crops/s in fp32 (98 in bf16, which moves joints by 1 cm at the median) on the RTX 4080 SUPER; the whole trial (41,061 crops) takes 23 minutes. |
| Score term in the windowed choice (2.3) | Not built: neither gate passed. |

## 2. Setup

- **Branch and data.** Branch `spike/gemx-body-pose` from `main`. Data work area `D:/mocap/gemx-work` with copies of
  the session and registry databases of `D:/mocap/jitter-work`; the operator's databases were not written.
- **Basis.** Five cameras, calibration `b416cd21`, the ace2 configuration, boxes from the product's mask box path
  (detection run `8e2147a0`, tracks 1-3 are mask labels 1-3), the frozen evaluation set `multi-person-ukemi-v1`.
- **References.** The label-fed runs of range 3 (`855e1e19`, `bc3886f9`), the labels of `kp_eval_set.toml`, and the
  operator-edited runs `6972748b` (subject 1), `47cfeb6c` (subject 2) and `80d7fab3` (subject 3). The subject of each
  edited run was found from the position of its hips: the right subject's hips lie 0.17-0.24 m from the Body Pose
  hips, the others 1.6-2.5 m.
- **Body Pose environment.** `D:/mocap/models/gem-x-repo/.venv`: PyTorch 2.10.0+cu128, the GEM package at upstream
  commit `20555bef`, `py-soma-x` 0.3.3 with the SOMA v0.3.3 assets. Weights `body_s240000.ckpt` and
  `body_config.json` of `nvidia/GEM-X` at revision `4fce057963671fd95cd4154297944397e7b5a9a0`. The Posetrak
  environment was not touched.
- **Input.** Per video frame of the trial, all five cameras, the three subjects' boxes together in one call. Each frame
  is undistorted with the calibration's intrinsics and distortion (the same undistorted matrix the tracker uses); the
  boxes are mapped into that image (the box edge points are undistorted and enclosed), and made square as GEM does:
  `1.2 x max(height, width / 0.75)`. The intrinsics of the undistorted image are passed to the model. Results are in
  undistorted pixels and camera coordinates. One file per camera in `D:/mocap/gemx-work/bodypose/out`: frame, time,
  subject, box, `joints_camera` (77 x 3 m), `keypoints_2d` (77 x 2, undistorted pixels), `projection_valid`, body
  pose, global orientation, identity (128), scales (57), camera translation. Body pose rotations are stored in
  float16.
- **Throughput.** Model load 1.5 s (cached files). Model alone with SOMA forward kinematics and the vertices, batch 16:
  47 crops/s in fp32 (5.5 GB peak), 98 crops/s in bf16 (7.3 GB). In bf16 the joints differ from fp32 by 0.7-1.0 cm
  median, 2-3 cm 99th percentile, up to 6 cm, so the run used fp32. Complete runs including 4K decoding and
  undistortion: 38 crops/s for the Pixel cameras, 25-26 for the two GoPros, 25 for the ace2 pro; 41,061 crops in 23.4
  minutes.

## 3. Decoder convention (2D)

### 3.1 What was measured, and a correction

The frozen candidate files store, for each candidate, its position in the distorted frame and, in a frame table, the
crop (centre and side). The cell coordinate of every candidate can therefore be recovered exactly and re-mapped to
another convention (the mapping is affine in the cell coordinate), so no model re-run was needed.

An earlier version of this section treated the frozen files as cell-centre decoded. That was wrong: the candidates of
the product path (cell centre) minus those of the frozen files are exactly +0.5 cell in x and y (10th-90th percentile
0.49999-0.50001 over 1,200 candidates), so **the frozen files are left-edge decoded**, as the task description says.
All results below use that.

The task asked for a comparison with the human labels, which "do not depend on any convention". In the range-3 files
that is true for only 0.2% of the labels (772 and 961 free-hand positions); the other labels hold a candidate and carry
its decoded position. Two references were used instead, and both were chosen to be free of the decoder under test:

- **A. Multi-camera consistency.** For each labelled visible keypoint that holds a candidate, the point is triangulated
  from the three other labelled cameras (the same convention applied in all cameras) and the difference between the
  keypoint and the reprojection in the target camera's distorted pixels is taken, in cells. A decoding bias of the
  target camera is independent of the other cameras' errors, so it shows as a constant (offset) or a trend with the
  keypoint's position in the window (slope). About 590,000 and 670,000 samples; samples farther than 0.08 box heights
  are dropped (62-68% remain). Uncertainties are bootstrap standard errors over 1 s blocks.
- **B. Free-hand labels.** The free-hand labels have no candidate at their position. Each is paired with the nearest
  candidate of the same camera, frame and keypoint within 0.05 box heights (210 and 283 pairs); the pairs are made once
  and the three conventions are applied to the same pairs. The difference label minus candidate is fitted against the
  label's position in the window.

### 3.2 Results

Offset in cells (positive: the keypoint lies right of / below the reference), x only for A because a common vertical
shift of all upright cameras is indistinguishable from a vertical move of the 3D point.

| Convention | A: x offset, subject 1 / 2 | A: median residual (box heights) | B: x offset, subject 1 / 2 | B: y offset, subject 1 / 2 |
|---|---|---|---|---|
| Left edge (frozen files) | -0.04 +- 0.03 / -0.07 +- 0.03 | 0.0265 / 0.0226 | +0.08 +- 0.08 / +0.01 +- 0.07 | +0.03 +- 0.08 / +0.08 +- 0.06 |
| Cell centre (product) | +0.37 +- 0.03 / +0.32 +- 0.05 | 0.0299 / 0.0261 | -0.42 +- 0.08 / -0.49 +- 0.07 | -0.47 +- 0.08 / -0.42 +- 0.06 |
| UDP edge | +0.37 +- 0.03 / +0.31 +- 0.05 | 0.0300 / 0.0262 | -0.43 +- 0.08 / -0.50 +- 0.07 | -0.47 +- 0.08 / -0.43 +- 0.06 |

Slope (cells per window width) in B, x / y, the two subjects pooled by inverse variance. Left edge and cell centre
have the same slope by construction; UDP is 1.0 lower:

| Convention | x slope | y slope |
|---|---|---|
| Left edge and cell centre | -0.05 +- 0.34 | +0.46 +- 0.26 |
| UDP edge | -1.07 +- 0.35 | -0.55 +- 0.27 |

Per camera and keypoint group (A, x offset in cells, left edge / cell centre, subjects 1 and 2 averaged):

| | Offset, left edge | Offset, cell centre |
|---|---|---|
| Cameras (pixel9, gopro02, gopro01, ace2, pixel7) | -0.20, -0.02, -0.20, +0.50, -0.31 | +0.38, +0.72, +0.13, +0.60, -0.08 |
| Trunk, head, arm, wrist, fingers, leg | 0.00, 0.04, 0.03, -0.11, -0.12, 0.02 | 0.46, 0.47, 0.45, 0.29, 0.27, 0.43 |

B has too few pairs to split by camera or group, so it is reported pooled.

### 3.3 What this shows, and what it does not

- Both references prefer left edge over the product's cell centre by 0.4-0.5 cell in x, and B also in y. For A the
  median residual is 11-13% smaller with left edge. The shift is 1.25% of the box height, about 2 cm on a person.
- The per-camera offsets of A scatter by +-0.4 cell around their mean (the ace2 pro is the outlier), more than the
  bootstrap error of 0.03; a common offset of 0.4 cell is therefore only about 2 sigma against that scatter. B's
  offset is the cleaner evidence (about 6 sigma), but it rests on 493 pairs from keypoints where the operator found
  no correct candidate, paired by nearest candidate, which pulls offsets towards zero rather than away from it.
- UDP has the same offset as cell centre in the window centre; B favours a slope of zero over -1.0 at about 3.7 sigma
  (x and y combined). A cannot judge the slope: every camera applies the same scale about a crop centred on the person,
  which a 1% rescaling of the 3D skeleton absorbs (cell centre and UDP slopes differ by 0.04-0.10 in A, against a
  bootstrap error of 0.15-0.19).
- These results disagree with the earlier check in the pose-model study (hip-joint midpoint against the two ASIS dots in
  the front camera, which was 0.56 cell left of the dots with `i / W` and 0.08 with `i / (W - 1)`). That check used
  one camera, one joint, and a joint whose definition differs from the dots'. The two cannot be reconciled here.

### 3.4 What follows

The task runs the 3D comparison only if the *slope* differs between conventions by more than its uncertainty. It does
(UDP against the others, 1.0 against 0.35), but UDP is the convention the references disfavour. The more relevant
difference is the constant half cell between the product's decoder and the frozen files, which is not a slope. The 3D
runs were not made; the decision is the operator's. If run, the comparison is the product decoder (cell centre)
against left edge, tracked at process noise 0.3 and 0.15 over range 3 and the whole trial, with the marker offsets
re-calibrated under each. The product's decoder was not changed.

## 4. Body Pose

### 4.1 A. Candidate cue

For every labelled visible keypoint of range 3 whose label holds a candidate (reviewed units, 51.7-56.8 s, the four
labelled cameras; 308,924 keypoint-frames, 140,705 for subject 1 and 168,219 for subject 2), the candidates were ranked by
their distance to Body Pose's `keypoint_2d` of the same keypoint (both in undistorted pixels). The cue is "correct" when
the nearest candidate is the one the label holds. The comparison picks are made on the same keypoint-frames:

- **Strongest peak**: the rank-0 candidate.
- **Windowed choice**: the candidate the basis tracking runs `36afbdaf` (subject 1) and `de1e686b` (subject 2) took
  (`tracking_kp_picks`; positions shifted by the half-cell difference between the product's and the frozen candidates).
- **Prior**: the candidate nearest to the marker position those runs predicted (the forward filter's prior; it includes
  the influence of the run's own earlier picks, so it flatters the prior slightly).

| Share where the pick is the labelled candidate | Strongest | Windowed choice | Prior | Body Pose cue |
|---|---|---|---|---|
| All (308,924) | 88.4% | 91.8% | 92.1% | 90.3% |
| Two or more candidates (96,592) | 62.8% | 74.0% | 75.1% | 68.9% |
| **Strongest peak wrong (35,909)** | 0% | 63.8% | 64.8% | **53.6%** |
| Windowed choice wrong (24,629) | 48.5% | 0% | 34.7% | 46.0% |
| Strongest wrong, body keypoints without fingers (5,854) | 0% | 70.9% | 68.3% | 54.9% |

Where the strongest peak is wrong, by subject, the cue is correct in 48.0% (subject 1) and 59.3% (subject 2); by
group in 61.8% (trunk), 53.0% (head), 49.6% (arm), 51.9% (wrist), 53.4% (fingers), 56.6% (leg).

Distance of the correct candidate to the Body Pose keypoint, in box heights: median 0.031 and 90th percentile 0.299 over
all keypoint-frames; 0.185 and 0.488 where the strongest peak is wrong (0.149 and 0.514 for body keypoints). A score
term `exp(-d^2 / 2 sigma^2)` would need sigma of about 0.2 box heights (2-3 times the 0.03 tolerance of a correct pick),
and so cannot separate candidates that lie closer together than that.

Gate: 75% where the strongest peak is wrong. **Not met** (53.6%, best group 61.8%). Body Pose finds the right candidate
less often than the tracker's own prior and the windowed choice, and in the cases where the windowed choice is wrong it
does not do better than the strongest peak (46.0% against 48.5%).

### 4.2 B. Occlusion

The run stored joints and body parameters, not meshes, so surfaces are **capsules** (a bone from the parent joint to the
joint, radius by body part: torso 0.14 m, upper leg 0.075-0.08, shin 0.05, upper arm 0.05, forearm and hand 0.04,
fingers 0.012, neck 0.05, head 0.10, plus a pelvis bar and a shoulder bar) built from the Body Pose joints of all three
subjects of the frame. A keypoint is *hidden* if, along the camera ray to its joint, a capsule lies in front of the joint
(its closest approach at least half a radius in front); for the keypoint's own body the bones touching the joint are
excluded. The continuous score is the largest coverage `1 - d / r` over the capsules in front. The radii are rough
anthropometric values, not fitted; a mesh depth buffer was not used.

Targets, over all labelled keypoint-frames of range 3 (361,052 keypoint-frames with candidates for the not-visible target):
(i) the label marks the keypoint not visible, scored on keypoint-frames that have candidates; (ii) the windowed choice's
pick is more than 0.10 box heights from the label (301,999 picks, 24,818 of them off).

| Target | Occlusion score AUC | Heatmap-peak AUC | Hidden / target confusion (TP, FP, FN, TN) |
|---|---|---|---|
| Label not visible, all | 0.59 | 0.78 | 35,044 / 161,998 / 16,084 / 147,926 |
| Label not visible, body without fingers | 0.68 | 0.87 | 18,736 / 31,692 / 10,856 / 74,698 |
| **Pick > 0.10 off, all** | **0.61** | 0.88 | 17,253 / 139,897 / 7,565 / 137,284 |
| Pick > 0.10 off, body without fingers | 0.56 | 0.91 | 1,107 / 25,875 / 1,802 / 69,945 |

Pick > 0.10 off by group (occlusion AUC / peak AUC): trunk 0.55 / 0.91, head 0.62 / 0.91, arm 0.56 / 0.96, wrist 0.62 /
0.88, fingers 0.57 / 0.86, leg 0.53 / 0.86. Not-visible by group: trunk 0.74 / 0.88, head 0.74 / 0.94, arm 0.65 / 0.86,
wrist 0.68 / 0.84, fingers 0.57 / 0.76, leg 0.63 / 0.82. The capsule model calls 52-55% of all keypoints hidden, so most
of its hidden calls are wrong.

Gate: AUC 0.75 for picks more than 0.10 box heights off. **Not met** (0.61). The heatmap peak of the pick is far better
here (0.88) than the 0.60-0.72 of the earlier spike, because the picks scored here already passed the mask gate and the
windowed choice, which leaves the weak peaks as the failures. The result for occlusion is limited by the capsule
approximation; a mesh depth buffer would be tighter, but the gap to the gate (0.14) is large.

### 4.3 C. 3D pose and initialisation

Body Pose joints are moved to the world with the extrinsics of `b416cd21` and compared with the same joints of the
edited runs: Hips, Spine1, Chest (against `spine2`), Neck1, Neck2, Head, both shoulders, upper arms, forearms and
hands, both thighs, shins, feet and toe bases (22 joints, no fingers). The skeleton's joints are defined differently
from SOMA's (for example the ankle), so every distance is also given with a constant offset per subject and joint
removed (the median difference in the skeleton joint's own frame over the trial, what the tracker's calibrated marker
offsets do). Run `6972748b` has no smoothed states, so its forward-pass states are used (3,194 steps). 40,049 of the
41,061 crops had an edited state within 20 ms. Distances are mean joint distances per crop, in cm.

Three ways to use the cameras: **per camera** (each crop alone), **median** (per-joint median of the 3D joints over the
cameras of one frame time, at least three), and **triangulated** (the Body Pose 2D keypoints of at least three cameras,
triangulated by DLT; 8,056 fused poses).

| Mean joint distance, whole trial (cm) | Per camera | Median over cameras | Triangulated 2D |
|---|---|---|---|
| Absolute | 46.0 (median 28.0) | 20.0 | 13.8 (median 10.4) |
| Absolute, offsets removed | 44.8 | 18.0 | 13.0 |
| Root-aligned | 13.7 | 11.7 | 11.8 |
| Root-aligned, offsets removed | 11.9 | 9.8 | 10.1 |
| Hips (root) alone, absolute | 44.8 | 18.7 | 12.4 |

Per camera, absolute (pixel9, gopro02, gopro01, ace2, pixel7): 56.2, 33.5, 38.1, 77.8, 24.8 cm. Root-aligned: 14.5, 13.1,
13.2, 15.2, 12.7. The root error is mostly depth along the camera ray, which one image gives poorly; the two cameras
with the largest absolute error are the ace2 pro and the Pixel 9.

Inside and outside the ukemi events (the 17 events of the evaluation set, per subject), root-aligned with offsets removed:

| Mean joint distance (cm) | Per camera | Median over cameras | Triangulated 2D |
|---|---|---|---|
| Whole trial, inside events / outside | 17.7 / 11.0 | 13.3 / 9.3 | 14.3 / 9.5 |
| Range 1 (43.5-47.5 s), inside / outside | 15.3 / 11.2 | 11.3 / 9.0 | 11.2 / 9.8 |
| Range 3 (51.7-56.8 s), inside / outside | 21.2 / 12.0 | 16.3 / 9.9 | 16.9 / 10.3 |

Absolute, triangulated: 19.9 inside and 12.9 outside the events in the whole trial; 22.6 and 13.7 in range 3. Per part,
root-aligned with offsets removed, triangulated: trunk 1.8-2.2 cm, head 9.5-11.1, arms 12.5-13.9, legs 9.9-13.4. Share of
poses with a mean joint distance under 5 cm: 17% (whole trial, root-aligned, offsets removed), 0% absolute; under 10
cm: 67% and 46-53%.

**Usable as a tracker start?** Not as a replacement. The tracker's current start (triangulated keypoints, then inverse
kinematics) reaches about 3 cm RMS when it converges; the best Body Pose pose is 10-14 cm from the edited runs, with
the root 10-12 cm off even when triangulated from the cameras' 2D keypoints, and the limbs 12-14 cm off. Only the trunk
is close (2-4 cm). Some of that is the edited runs' own error and joint definitions, which the offset removal reduces
only by 0.8-1.7 cm. Whether such a pose would let the filter converge where the triangulation start fails (the
failure of the SOMA runs' start in the pose-model study) was not tried; it is the one case where it could help, as a
fallback prior to be checked in the tracker.

### 4.4 D. Body shape

Bone lengths of the Body Pose joints, median over all frames and cameras of each subject, against the same bones in the
edited runs' skeletons (cm; Body Pose / skeleton; interquartile range of Body Pose over frames and cameras):

| Bone | Subject 1 | Subject 2 | Subject 3 |
|---|---|---|---|
| Upper arm | 25.9 / 26.4 (1.6) | 26.5 / 26.4 (1.8) | 27.0 / 30.2 (1.8) |
| Forearm | 22.9 / 25.0 (2.4) | 23.4 / 25.0 (2.6) | 23.6 / 25.0 (2.3) |
| Thigh | 41.5 / 39.2 (2.3) | 41.8 / 40.6 (1.5) | 42.1 / 40.6 (1.3) |
| Shin | 37.1 / 41.0 (0.9) | 37.7 / 38.4 (0.8) | 38.2 / 43.4 (0.8) |
| Foot | 12.8 / 14.6 (0.3) | 13.0 / 14.6 (0.4) | 13.1 / 14.6 (0.5) |
| Hip width | 23.4 / 19.1 (1.0) | 23.4 / 19.1 (0.9) | 23.3 / 19.1 (0.9) |
| Shoulder width | 32.1 / 33.3 (1.5) | 33.0 / 33.2 (2.5) | 33.0 / 34.0 (2.6) |

The medians per camera lie within 1-2 cm of each other. The limb bones agree with the skeletons to within 1-4 cm, but the
three subjects come out nearly identical in Body Pose (thigh 41.5-42.1 cm, upper arm 25.9-27.0 cm), while the skeletons
differ by up to 5 cm (shin 38.4-43.4 cm, upper arm 26.4-30.2 cm). Trunk bones differ by definition (the chest-to-neck bone is
24 cm in Body Pose and 32 cm in the skeletons). The predicted shape is therefore close to
an average body here and cannot replace the bone-length calibration.

## 5. Not done

- Section 2.3 (a Body Pose score term in the windowed choice, and the hidden-keypoint option): neither gate passed.
- The 3D tracker runs of section 3 (a decision for the operator, section 3.4).

## 6. Product work that would follow, and its cost

None is recommended by these results. If Body Pose were wanted anyway (for example as an initialisation fallback):

- **Export.** There is no ONNX export of Body Pose; the model is a DINOv3 ViT-H/16+ backbone with a transformer head,
  884 M parameters, run in PyTorch. An export would be new work (days, with parity checks against the PyTorch model).
- **Runtime dependencies.** PyTorch 2.10, the GEM package, `py-soma-x` 0.3.3 with the SOMA v0.3.3 assets (the
  corrective weights are needed), a 3.5 GB checkpoint, and a 5.5 GB GPU memory budget at batch 16.
- **Run time.** 47 crops/s on the RTX 4080 SUPER in fp32 (the model with FK and vertices); 23 minutes for 41,061 crops
  of a 23 s, five-camera, three-subject trial, in about the same time as the 2D pass over it. bf16 doubles the speed but
  moves joints by about 1 cm.

## 7. Reproducing

All scripts take the work-area copy of the session database and the frozen evaluation set.

- `decoder_convention_2d.py`, `decoder_convention_handplaced.py`: section 3.
- `bodypose_jobs.py` (Posetrak environment) writes the box and calibration jobs; `bodypose_run.py` (GEM environment)
  runs the model; `bodypose_smoke.py` times it.
- `bodypose_cue.py` (A), `bodypose_occlusion.py` (B), `bodypose_pose3d.py` (C and D); `bodypose_common.py` is shared.

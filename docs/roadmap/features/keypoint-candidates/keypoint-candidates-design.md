# Keypoint candidates: several maxima per keypoint, tracklets, and assignment in the tracker

## 1. Goal and evidence

A heatmap pose model has several local maxima per keypoint. In crowded footage the strongest one
is right on about 94% of visible keypoints, while a correct candidate exists among the maxima on
99%. Keeping the maxima, linking them into tracklets, and letting the tracker choose among them
with its prediction, the other cameras and the person masks gave the better tracker results of the
pose-model spike (mean body-joint distance to the label-fed run, labelled range, two persons: proposal
12.9 / 4.1 cm, cross-camera grouped choice 10.1 / 3.2 cm, with the per-tracklet body mask term
9.9 / 3.0 cm). It is not better everywhere, and it multiplies the stored data, so it is optional:
switched on per detection run, and usable or not per tracking run.

The same problem is solved for reflective dots: anonymous candidates per frame, tracklets, a
shared assignment of candidates to marker slots across all subjects with a mask gate and Hungarian
assignment, and tracklet decisions made outside the tracker. This design reuses that machinery for
keypoints rather than building a second one.

## 2. The existing dot machinery

| Piece | Where | What it does |
|---|---|---|
| Candidate rows | `detection_keypoints`, `region_type = 'dots'`; blob `int32 n` + `float32[n, 9]` (`encode_dot_candidates`) | one row per (run, video, frame); tracklet id per candidate |
| Copy into a sequence | `pose_observations`, `source = 'dots'` (`sequence_dots.add_dots_to_sequence`) | candidates travel with the subject's sequence |
| Loader | `SessionReader::load_unlabeled_candidates` -> `UnlabeledCandidate` | undistorts, keeps tracklet id, `subject_mask` |
| Shared assignment | `resolve_shared_dot_assignment` / `resolve_dot_assignment` (`dot_assignment.hpp`) | per camera, Hungarian over (candidate, subject slot) with squared Mahalanobis cost, gate 9.21, cost modifiers |
| Cost modifiers | `TrackletContinuityModifier`, `CandidateOwnershipModifier`, `ReacquisitionGateModifier`, `CrossViewCorroborationModifier`, `SegmentationMaskModifier` (with `TrackletMaskAccumulator`), `TrackletAssignmentModifier` | evidence that changes or excludes a pairing |
| Tracklet decisions | `dot_assignment_edits` (human), `dot_assignment_proposals` (automatic), `TrackletAssignments` | assign / forbid / reject / other / suggest per tracklet and frame range |
| Cross-camera grouping | `posetrak/markers/grouping.py` | groups tracklets of several cameras whose rays meet over their shared lifetime |

The two tracking-time options for keypoint candidates already exist as settings of this phase:

- **Strict mask adherence** (a candidate is used only if its tracklet lies on the person's own mask
  for enough of a time window) is `SegmentationMaskModifier` with the tracklet window and minimum
  in-mask fraction (`dot_seg_mask_enabled`, `dot_seg_mask_tracklet_window`,
  `dot_seg_mask_min_tracklet_fraction`, `dot_seg_mask_hard_distance_px`).
- **Multi-view agreement** (a candidate is preferred when another camera's candidate for the same
  slot lies on its epipolar line) is `CrossViewCorroborationModifier`
  (`dot_cross_view_corroboration_px`, `dot_corroboration_gate_multiplier`).

## 3. What is different for keypoints

1. **A candidate has a class.** A dot can be any marker; a keypoint maximum belongs to one keypoint
   of the model (heatmap channel). A new `KeypointClassModifier` excludes a pair unless the slot's
   marker landmark is the candidate's keypoint. Left-right swaps stay possible through the flip
   test's own averaging, not through assignment.
2. **A candidate comes from one person's crop.** It carries the person (track or mask label) whose
   crop produced it, and by default only that subject may claim it (`subject_mask`, the existing
   `CandidateOwnershipModifier`). Where two persons' crops overlap the same image point gives a
   maximum in both crops. The loader merges candidates of the same keypoint, camera and frame that
   lie within a small radius (default 3 px, or 0.5% of the box height if larger) into one candidate
   whose `subject_mask` is the union. This is how detections of overlapping persons are shared, and
   the Hungarian solve then lets only one subject claim it. Merging is done at load time, so
   candidates from several detection runs in one sequence merge the same way.
3. **The strongest pick still exists.** Every candidate run also writes the ordinary keypoint row
   (strongest maximum per keypoint, `region_type = 'full_body'`), so every existing reader, the
   viewer and a tracking run without candidates work unchanged. With candidates in use, the tracker
   uses the candidates instead of the fixed body observations for the markers bound to that model.
4. **Confidence.** Heatmap peak height does not rank keypoints above 0.5 (spike section 5.1); it is
   carried as `peak` for the noise model and for diagnostics, not as a gate.
5. **Hand stages.** Hierarchical child stages do not run the shared assignment. Candidates are used
   for the `main` stage's markers only; hands use refined-hand observations.

## 4. Storage

### 4.1 Candidate rows

Written by the detection run when `config_json.candidates` is present:

```json
"candidates": {"max_per_keypoint": 6, "min_peak": 0.05, "nms_cells": 5,
               "link": {"max_gap_frames": 3, "gate_box_frac": 0.04}}
```

- `detection_keypoints` row per (run, video, frame, track id) with `region_type = 'kp_candidates'`.
- Blob: `int32 n`, `int32 n_fields` (= 6), then `float32[n, n_fields]` with fields
  `keypoint_idx, x, y, peak, rank, tracklet_id`. `x, y` in distorted pixels like every keypoint row;
  `rank` 0 is the strongest maximum of its keypoint; `tracklet_id` is -1 when unlinked. The field
  count makes the format extensible without the "re-run detection" failure the dot blob has.
- Finalisation copies the rows of assigned tracks into `pose_observations` with
  `source = 'kp_candidates'` and the person id of the sequence, exactly as body rows are copied.

### 4.2 Tracklets

- Linking is done in the detection run, online, per (camera, track, keypoint): constant-velocity
  prediction, Hungarian matching against the frame's candidates, a gate of `gate_box_frac` of the box
  height, gaps of up to `max_gap_frames`. It is causal, so ids are known when a row is written and
  rows are never rewritten.
- Tracklet ids are unique per (detection run, camera) across keypoints and tracks, so the existing
  edit and proposal tables, keyed (detection run, camera, tracklet id), address them without change.
- There is no tracklet table. A tracklet's extent and statistics are computed from the rows when
  needed (one trial is tens of megabytes of candidate rows). A summary table is added only if the
  editor measures this as too slow.

### 4.3 Volume

Up to six maxima per keypoint; the spike averaged about two above the threshold. For 77 keypoints that
is roughly 150 candidates, 3.6 kB per person, camera and frame, against 0.9 kB for the ordinary row.
Measured on the evaluation trial: 91 candidates, 2.4 kB per row (section 9.4).
Dropping body-crop fingers from extraction is not done: the stored maxima are cheap, and the
decision which keypoints to track is the skeleton's.

## 5. Tracker

- `UnlabeledCandidate` gains `keypoint_idx` (-1 for dots). The loader reads `kp_candidates` rows
  (manifest gives the keypoint name per index), undistorts, merges duplicates (section 3).
- The single-subject loop (`track.cpp`) and `MultiPersonTracker` already call the shared assignment;
  they pass keypoint candidates alongside dot candidates. `KeypointClassModifier` is registered when
  any candidate has a class.
- `tracker_configs.kp_candidates_enabled` (0/1) selects candidate use for a run; with 0 the fixed
  observations are used even if the sequence holds candidates. Recorded with the run, so two runs on
  the same sequence differ in their configuration only.
- Gate, tracklet continuity, reacquisition, multi-view agreement and the mask settings are the
  existing `dot_*` settings: one set for every anonymous candidate. If a subject wears dots and
  is tracked from keypoint candidates and the two need different values, separate settings are added
  then; nothing measured so far asks for it.
- An optional `suggest` rule per candidate tracklet can come from an automatic grouped choice (the
  spike's cross-camera grouped assignment) written into `dot_assignment_proposals`. Not part of the
  first release: the tracker's own assignment is measured first.

## 6. Marker and markerless side by side

| | Reflective dots | Keypoint candidates |
|---|---|---|
| Detection | `dot_blob_detector.py`, scene-wide | heatmap maxima inside a person's crop |
| Owner at detection | none (scene) | the crop's person |
| Class | none: any dot marker | the heatmap channel |
| Tracklet linking | `dot_tracklet.MotionGatedLinker` | the same linker, per keypoint, with a box-relative gate |
| Storage | `region_type = 'dots'`, 9 fields | `region_type = 'kp_candidates'`, 6 fields + header |
| Assignment | shared phase | same shared phase + `KeypointClassModifier` |
| Decisions | `dot_assignment_edits` | the same table |

## 7. Measurements before and with the build

1. **Why the two mask-gated variants were nearly identical.** With the strict tracklet-mask filter on,
   the proposal and the cross-camera choice gave 8.2 / 11.5 and 8.3 / 11.7 cm. Hypothesis: the filter
   leaves one candidate for most keypoint-frames, so there is nothing left for multi-view agreement to
   choose. Measured from the existing candidate files: the share of keypoint-frames with two or more
   candidates before and after the filter, in the hard windows and elsewhere. If the hypothesis holds,
   the two options are complementary, not redundant, and both stay as options.
   **Result.** Share of keypoint-frames (a camera, frame and keypoint triple that has at least one
   candidate) with two or more candidates, before and after the strict tracklet-mask filter, per person.
   Capture: the five-camera, three-person trial, 38-66 s, candidates of the SOMA-77 model (at most six
   maxima per keypoint, at least 0.05); unfiltered file with mask labels refreshed from the current masks,
   filtered file = same with a candidate kept only if at least half of the detections of its tracklet within
   +-10 frames lie on the person's own mask. Hard windows: 53.0-54.5 and 54.9-56.4 s; reviewed: the rest of
   the 52.5-57 s span, the only span whose masks were reviewed; rest: the remainder of the trial (unreviewed
   masks). Command: `python python/tools/prototypes/kp_labeling/candidate_counts.py <unfiltered> <filtered>`.

   | Person | Window | Keypoint-frames before | with 2+ before | with 2+ after | emptied by the filter | agreement can change the pick, before / after |
   |---|---|---|---|---|---|---|
   | A | hard | 131,420 | 51.6% | 25.3% | 27.0% | 15.7% / 11.2% |
   | A | reviewed | 64,099 | 18.5% | 8.1% | 21.6% | 4.3% / 3.6% |
   | A | rest | 840,871 | 17.6% | 7.4% | 22.1% | 4.3% / 2.7% |
   | B | hard | 135,255 | 45.5% | 20.5% | 33.9% | 15.5% / 11.4% |
   | B | reviewed | 67,051 | 16.3% | 7.2% | 20.2% | 4.5% / 2.3% |
   | B | rest | 841,420 | 18.2% | 6.7% | 19.1% | 4.9% / 2.7% |
   | C | hard | 138,633 | 3.5% | 1.6% | 8.9% | 0.3% / 0.2% |
   | C | reviewed | 69,347 | 3.5% | 1.5% | 11.9% | 0.9% / 0.6% |
   | C | rest | 838,934 | 17.1% | 6.6% | 24.5% | 4.5% / 2.8% |

   "Agreement can change the pick" is the share of keypoint-frames where a candidate other than the
   strongest agrees with the other cameras while the strongest does not, i.e. where multi-view agreement
   would choose differently from the strongest peak. Person C is not in contact in the hard windows (the
   windows belong to the other two); his rows are a calm reference. The agreement flags in the file were
   computed before the filter.

   The hypothesis holds in outline: after the filter 75-80% of the keypoint-frames of persons A and B in the
   hard windows and about 93% elsewhere have exactly one candidate, from 48-55% and 82-84% before. It does not
   hold completely. In the hard windows 11% of the keypoint-frames still offer multi-view agreement a
   different choice from the strongest peak after the filter (16% before), so the filter removes about a
   quarter of the cases where agreement matters, not all. It also empties 27-34% of the keypoint-frames of
   the two persons in contact there, which is where the choice is hardest. The two options therefore stay
   as separate options. Agreement has nothing to change on 89% of the hard-window keypoint-frames and on
   about 97% elsewhere after the filter, which accounts for much of the near-identical tracker figures of
   the two variants but not for all of it; the remaining difference is not explained by these counts.
2. **Acceptance metric before the build.**
   The ground-truth harness (`python/tools/gt/`, see its README) gets the
   "same candidate" metric (share of picks that are the same candidate the label holds) next to the
   within-3%-of-box-height metric, and the label-fed tracker comparison, reading productized storage.
   The labelled data stays outside the repository; the harness is in it.
   **Result.** The harness (`eval_kp_candidates.py`, evaluation set `kp_eval_set.toml`) reads candidates
   through `read_candidates()`, which accepts the `kp_candidates` rows of a detection run (blob format of
   §4.1, unit-tested on fabricated rows) and the prototype's candidate files; body picks come from the
   strongest maxima or from a pose observation sequence, tracking results from `tracking_results`. It
   reproduces the spike's figures from the frozen evaluation set (labels of two ranges, four
   person-ranges; build: the commit that adds the harness; session database: the work copy of the
   spike):

   | Figure | Spike | Harness |
   |---|---|---|
   | Strongest peak within 3% of box height, range 1, two persons (about 106,000 labelled camera-keypoint-frames) | 94.2% (107,589 frames, labels of an earlier day) | 94.4% (106,278 frames) |
   | Same candidate, same range | equal to the 3% figure in all measurements | 94.3%, 0.1 points below the 3% figure |
   | Mean body-joint distance of the proposal runs to the label-fed run, range 3, complete data, all frames (cm) | 12.9 and 4.1 | 12.7 and 4.4 |
   | Failure events flagged at 40 cm by baseline against proposal (any joint) | 13 of 15 | 13 of 15 |

   The distance figures differ by up to 0.3 cm from the spike's. The likely cause is the harness'
   definition of body joints (every joint except finger joints, sampled every 0.05 s from 52.0 s), which was
   reconstructed from the spike's event script and not copied from the script that produced those
   figures; it is fixed here so that later runs are comparable with each other. `kp_eval_set.toml` names
   the files, their checksums and the scoring script, and `eval_kp_candidates.py verify` checks them.
3. **Reproduction.** The productized path must reproduce the prototype's tracker numbers on the
   labelled range within 0.5 cm mean body-joint distance before any default changes.

## 8. Acceptance

- A candidate run, finalised and tracked with candidates on, gives mean body-joint distances to the
  label-fed run no worse than the prototype's cross-camera choice plus 0.5 cm, both scored by the
  evaluation harness (`eval_kp_candidates.py`, range 3, complete data). The harness's definition of
  body joints differs from the one behind the 10.1 / 3.2 cm figures by up to 0.3 cm, so the reference is
  the prototype's run scored by the harness, not the figures quoted in section 1.
- Tracking the same sequence with candidates off gives the result of a run without candidates.
- With candidates on and no masks, no multi-view agreement, the result equals the tracker's own
  per-frame choice; each option changes only its own modifier (unit tests per modifier).

## 9. As built and acceptance results

### 9.1 Where the build differs from sections 3 to 5

- **Linker.** The tracklets are linked by `kp_candidates.PersonLinker`, not by `dot_tracklet.MotionGatedLinker`.
  That linker gates in absolute pixels (a 3 px measurement standard deviation, a 25 px birth gate) and refuses
  to commit when the two best candidates of a tracklet are nearly tied. The first does not follow a person's
  apparent size; the second is the normal situation for the neighbouring maxima of one keypoint. The new linker
  is the prototype's: per (camera, person, keypoint), constant-velocity prediction (the velocity is the average of the old one and the last step), Hungarian
  matching, a gate of `gate_box_frac` of the **longer side** of the person's box (the
  prototype's box height; for an upright person the same), widened by half of itself for each frame of gap,
  and a gap of at most `max_gap_frames`. Tracklet ids are unique per (detection run, camera) across keypoints
  and persons; a run written in several passes continues the ids after the ones it already holds.
- **Confidence.** Finalisation multiplies the candidate `peak` by the pose model's confidence scale, as it
  does the confidence of the body row. The tracker's measurement noise of a keypoint is divided by the
  confidence, so unscaled peaks would have given every candidate five times the noise of the fixed observation
  of its keypoint.
- **Duplicate merge.** The loader gives each candidate the radius `max(3 px, 0.5% of the box height)`; the box
  height follows from the row's `noise_scale` (160 px of box per unit), because candidate rows do not carry
  the box. Two candidates merge when they are the same keypoint in the same camera and frame and lie within the
  smaller of their two radii; the pool keeps the first position and the union of the owners. Merging is in
  `append_unique_candidates`, i.e. when the subjects of a multi-person run share a pool; a single-subject run
  has one crop per keypoint and nothing to merge.
- **Slot covariance.** The shared assignment costs a pair by the squared Mahalanobis distance under the
  prediction's pixel covariance, which for dots is the state's uncertainty alone. A keypoint candidate adds
  its detection noise, `(pose_noise_std * crop_scale + calib_noise_std) / max(confidence, 0.1)`, squared, the
  way the filter's innovation covariance adds it for a fixed observation. Without it the gate would reject
  correct candidates whenever the state is confident.
- **Routing.** A subject takes part in the shared assignment when its skeleton has dot markers or its config
  enables candidates and the sequence holds some for markers of the skeleton (enabled with none is an error
  that names `detect run --candidates`). The markers bound to the candidates' manifest get slot predictions
  (`Tracker::set_candidate_slot_markers`); `KeypointClassModifier` pairs a candidate only with the marker bound
  to its keypoint and keeps dots and keypoints apart. On steps with candidates the fixed POSITION observations
  of those markers are dropped and the resolved ones used; initialisation, steps without candidates, the
  relative and cross-person observations and the hand stages still use the fixed observations.
- **Options.** Strict mask adherence and multi-view agreement are the `dot_*` settings of
  `SegmentationMaskModifier` and `CrossViewCorroborationModifier`; `track run --strict-mask` and
  `--multi-view-agreement` set them (with `--mask-min-fraction`, `--mask-window`, `--agreement-px`,
  `--agreement-factor`). The mask modifier needs the subject's mask and per-tracklet history, which only the
  single-subject loop of `posetrak-tracker track` provides; the multi-person loop passes neither, as it does
  not for dots.
- **Not built.** A candidate-row `box_height` field, a tracklet table, candidates for RTMPose runs, the
  `suggest` rule and the tracklet editor (stage 10).

### 9.2 Acceptance run

Capture and range as in section 7 (the five-camera, three-person trial, range 3 = 51.7-56.8 s, subjects 1 and 2),
on a work copy of the spike's session database. Boxes: `detect run --from-run 69c1d770 --candidates` (the
spike's hybrid box run), finalised over the whole span of every track (a mask label is a stable identity; the
source run's assignments have gaps where the track identity was uncertain, which the prototype's sequences do
not have). Tracking: the tracker config and skeletons of the label-fed reference runs (`hier-dualhand`,
skeletons with `soma77:`-qualified landmarks), 51.7-56.8 s, optimized build. Scored by
`eval_kp_candidates.py tracker`: mean body-joint distance (cm) to the label-fed reference run from 52.0 s, over
the range and inside the hard windows. Target: no worse than the prototype's cross-camera run plus 0.5 cm,
i.e. at most 10.7 / 3.9 cm over the range.

| Run | Subject 1 range / hard | Subject 2 range / hard |
|---|---|---|
| prototype, automatic proposal (`908e75e4`, `eb7ade35`) | 12.7 / 14.8 | 4.4 / 5.7 |
| **prototype, cross-camera (`db888f74`, `c30c6322`)** | **10.2 / 10.9** | **3.4 / 4.2** |
| prototype, cross-camera + body mask term (`47613f1f`, `01aa291b`) | 10.0 / 10.2 | 3.3 / 4.0 |
| prototype, strict mask filter + proposal (`b3c714df`, `d8b198ed`) | 8.7 / 11.9 | 4.8 / 6.2 |
| prototype, strict mask filter + cross-camera (`bea2b0fd`, `150da557`) | 8.8 / 12.0 | 4.7 / 5.9 |
| productized, candidates off (sequence without candidate rows: `bdc8c963`, `22673d9e`) | 18.0 / 21.1 | 5.3 / 6.8 |
| productized, candidates off (sequence with candidate rows: `2a51e2fa`, `5d577395`) | 18.0 / 21.1 | 5.3 / 6.8 |
| productized, candidates on (`e9000211`, `44fb7af3`) | 18.7 / 21.8 | 7.6 / 9.6 |
| productized, candidates + strict mask (`4e219f3d`, `c8a24ead`) | 14.6 / 16.0 | 7.9 / 10.2 |
| productized, candidates + multi-view agreement (`71f9d80c`, `b5bc77da`) | 14.4 / 16.0 | 6.9 / 8.8 |
| productized, candidates, gate 36 (diagnosis only, `a3b11abd`, `aff37cdd`) | 17.5 / 20.4 | 6.4 / 8.3 |

(Run ids are those of the work copy of the spike database, which holds the prototype's runs too.)

**Result: the acceptance bar is not met.** With candidates on and no options the tracker's own per-step choice is
8.0 cm (subject 1) and 4.2 cm (subject 2) behind the prototype's cross-camera run, and 0.7 and 2.3 cm behind its own
run on the strongest maxima of the same sequence. Nothing was retuned and the `suggest` rule was not built. The
figures and their sources:

- **Identity with candidates off.** A tracking run of the sequence that holds the candidate rows with
  `--no-candidates`, a run of a copy of that sequence without the rows, and a repeat of the first give bit-identical
  state vectors, for both subjects. Two detection runs over the same frames are not bit-identical (105 of 3,007
  body rows of subject 1 differ, median 0.03 px, maximum 4 px, from the GPU), so comparing runs on sequences of two
  detection runs compares different data.
- **Where the difference comes from.** Not only the hard windows: for subject 1 the mean joint distance in the frames
  outside them is 12.3 cm with the strongest maxima, 13.2 cm with candidates and 8.0 cm for the prototype's
  cross-camera run (inside them 21.4, 22.1 and 11.5 cm). By joint, candidates are worse than the strongest maxima
  on the hands (right hand 11.7 to 18.0 cm outside the hard windows, 22.7 to 36.7 inside) and the left leg in the hard
  windows (left shin 11.4 to 17.2 cm), and better on the feet (right foot 29.9 to 23.9 cm in the hard windows); the
  prototype's choice is better than both on nearly every joint. That choice used the other cameras' picks and the
  tracklet over the whole range; the tracker's has the prediction of one step. Where the prediction is off (fast
  limbs) the candidate nearest to it keeps the error, where the strongest maximum does not depend on it. The options
  help in the direction the prototype's did: with strict mask adherence 14.6 cm and with multi-view agreement 14.4 cm
  against 18.7 for subject 1; for subject 2 agreement gives 6.9 and the mask 7.9 against 7.6, all still behind the
  strongest maxima (5.3). The assignment gate is not the cause: at 36 (the 6 sigma of the filter's own outlier test)
  the figures are 17.5 and 6.4 cm.
- **Next step, not taken here.** The prototype's gain came from a choice made per tracklet and over time. The
  `suggest` rule from an automatic grouped choice (section 5) is what would carry it into the tracker; its
  decision, like any change of the `dot_*` defaults, is the operator's.

### 9.3 Like for like

- **Candidates.** `eval_kp_candidates.py compare` against the prototype's candidate file: on frames whose crop is
  the same (2,200 of 3,007 for subject 1, 2,256 of 3,051 for subject 2; 73-74%) the run reproduces 100.0% of the
  prototype's candidates within 0.5 px, and the prototype's 100.0% of the run's, once the file's positions are moved
  by half a heatmap cell: the frozen file stores the cell index at the cell's left edge, the decoder places a
  position at the cell centre (4 px at a crop of 513 px, 1.3% of the person's height), and nothing else differs. On
  the other frames (the crops differ by more than 1 px) 2-5% are reproduced.
- **Boxes.** The run uses `69c1d770` as box source, which holds the prototype's crop on 73% of the frames of
  subject 1 in the range (2,204 of 3,017); the other 27% are concentrated at 53-56 s, the hard windows (112, 24, 19,
  120, 120, 166, 126, 75 and 51 frames in the bins 51.7-52.5, 52.5-53, 53-53.5, ... 56-56.8 s). Another existing run
  (`fcb6b961`, boxes from the masks) holds 82%; 18% (547 frames) belong to no run: the prototype re-extracted them with
  boxes from the masks as edited later. In the hard windows about a third of the frames are therefore not on the prototype's
  crops. Not fixed: every existing run covers only part of the frames, and a synthetic run is
  not an existing one. `--from-run fcb6b961` is the closer choice for a repeat.
- **Frames.** The sequences cover 3,007 (subject 1) and 3,051 (subject 2) frames of 5 cameras x 612; the prototype's
  candidate file has 3,017 frames of subject 1 in the range. A frame without a box is absent in both.
- **Other differences of the pipeline.** The prototype's tracker sequences were built from the candidate file's
  picks (positions in the left-edge convention, so a systematic half-cell offset in the label-fed reference too, one
  camera left on the strongest maxima), with keypoints it judged invisible removed; the productized sequence holds
  every strongest maximum. The strongest-maxima run of the productized pipeline is at the level of the prototype's
  own (subject 1: 18.0 cm against 14.7 and 18.5; subject 2: 5.3 against 6.1 and 10.2, for its range-3 and whole-trial
  strongest-peak runs).

### 9.4 Cost

Optimized build, the acceptance runs, one machine, other work idle except where noted.

| | Candidates off | Candidates on |
|---|---|---|
| Detection, 51.7-56.8 s, 5 cameras, 3 persons, boxes from a run (wall) | 6 min 23 s | 7 min 24 s (+16%; a few short test runs overlapped it) |
| Detection rows of the run | `full_body` 8.15 MB, hands 2.9 MB, previews 48 MB | + `kp_candidates` 21.0 MB: 2.4 kB per person, camera and frame (91 maxima), 2.6 times the ordinary row, +26% of the run's rows |
| Sequence rows after finalisation | | the same 21.0 MB again in `pose_observations` |
| Tracking, subject 1 (wall, peak memory of the process tree) | 100.5 s, 4,863 MB | 110.5 s (+10%), 4,904 MB (+0.8%) |
| Tracking, subject 2 | 109.9 s, 4,883 MB | 114.2 s (+4%), 4,930 MB (+1.0%) |
| Shared assignment per step with candidates (about 90 per camera and person) | | 13.8 ms (subject 1), 13.3 ms (subject 2), of 181 ms and 187 ms per step; the prediction of the 77 slots in 5 cameras is on top (inside the wall figures) |
| Strict mask option | | 130.8 s and 139.8 s (+23% and +25% over the plain candidate runs of the first pass, 106.6 s and 112.2 s: one mask decode per camera and step) |

### 9.5 Strict mask option

Compared with the prototype's strict person-mask filter (a candidate kept only if at least half the detections of
its tracklet within +-10 frames lie on the person's mask; masks reviewed for 52.5-57 s only), scored by the harness:
the filter with the proposal gave 8.7 / 11.9 and 4.8 / 6.2 cm, with the cross-camera choice 8.8 / 12.0 and 4.7 / 5.9
(range / hard windows, subject 1 and 2); the tracker's mask modifier with candidates gives 14.6 / 16.0 and
7.9 / 10.2. The modifier is a different test: it judges a tracklet from the steps in which one of its candidates
won, up to the current step, not from its detections around the frame, and it has no effect on a tracklet that has
not yet won; the prototype's filter looks at all detections, before and after. The share of keypoint-frames the
filter empties for the persons in contact (27-34% in the hard windows, section 7) is not reproduced by the
modifier: it excludes a pairing, and only after the tracklet has a history of at least five steps. Multi-view
agreement gave the same improvement on subject 1 (14.4 cm) and more on subject 2 (6.9 against 7.9 cm).

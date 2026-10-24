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
Measured on the evaluation trial: 91 candidates, 2.4 kB per row (section 9.5).
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
- Gate, reacquisition, multi-view agreement and the mask settings are the existing `dot_*` settings: one set
  for every anonymous candidate. If a subject wears dots and is tracked from keypoint candidates and the two
  need different values, separate settings are added then; nothing measured so far asks for it. The exception is
  what only a keypoint candidate has: the cost is a negative log-likelihood with a peak term
  (`kp_candidate_peak_weight`) and tracklet continuity has its own multiplier
  (`kp_candidate_tracklet_multiplier`), because the dots' is 1.0 by default (section 9.2).
- The per-step assignment above does not reproduce the prototype's choice (section 9). Section 10 describes
  a choice over a time window, computed in the tracking run before the filter steps.

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

Dots may later get a soft class from their tracklet, for example the average tone of the skin or clothing around
the dot. Such a class differs from a keypoint's: it is more often wrong and can be shared by many marker slots of
the skeleton, so it would weight a candidate's cost rather than restrict it to one slot as
`KeypointClassModifier` does. Neither the per-step assignment nor the windowed choice of section 10 should
assume that a class names exactly one slot.

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
- **Slot covariance.** The shared assignment gates a pair on the squared Mahalanobis distance under the
  prediction's pixel covariance, which for dots is the state's uncertainty alone (the cost of a keypoint candidate
  has two more terms, section 9.2). A keypoint candidate adds
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

### 9.2 Cost and continuity of the assignment

The dots' cost, the squared Mahalanobis distance alone, does not suit keypoint candidates, and the dots' tracklet
continuity is off by default (`dot_tracklet_gate_multiplier` 1.0; the modifier is registered only above 1.0):

- A candidate's detection noise is added to the slot covariance and grows as the peak falls (section 9.1). With the
  distance alone, at equal distance a weak maximum has the wider covariance and so the **lower** cost, and nothing
  rewards peak height. Dots all have one noise, so this does not show for them. Measured with that cost (boxes
  `69c1d770`), the tracker following the candidates was worse than following the strongest maxima (subject 1: 18.7
  against 18.0 cm, subject 2: 7.6 against 5.3).
- Without continuity the tracklets linked at detection take no part in the choice, which is where the prototype's
  gain came from.

As built, the cost of a keypoint candidate is a negative log-likelihood of the detection under the slot's
distribution:

    cost = d^2 + ln det S - w_peak * ln(max(peak, 0.01))

with `d^2` the squared Mahalanobis distance under `S`, the slot covariance with the detection noise, and `w_peak`
= `tracker_configs.kp_candidate_peak_weight` (0 leaves the peak out; `ln det S` is always in, for candidates only;
a dot's cost is unchanged). The gate tests `d^2` alone, after the modifiers, as before. The terms can be negative
and the solver takes non-negative costs below one gate: every pair that passed the gate is shifted by the smallest
extra term of the camera's matrix and the solver's gate is widened by the spread of the extra terms, so passing
pairs stay below it and the hopeless and the excluded ones (all one cost, above) stay above. With no keypoint
candidate in the camera both are zero and the matrix is exactly the dots' (unit-tested: a dot-only assignment is
the same whatever the keypoint settings; at equal distance the stronger of two keypoint candidates wins).

`TrackletContinuityModifier` divides `d^2` by `tracker_configs.kp_candidate_tracklet_multiplier` for a keypoint
candidate that continues the tracklet resolved into the same (subject, camera, marker) slot on the previous step; a
dot keeps `dot_tracklet_gate_multiplier`. A marker is bound to one keypoint and a tracklet id is unique per camera
across keypoints, so continuation is per slot and keypoint without further bookkeeping (unit-tested next to
`KeypointClassModifier`). The multiplier is a setting of its own because the dots' is 1.0 by default and
`dot_*` values are not to be retuned for this stage. `track run --kp-peak-weight W --kp-tracklet-multiplier M`
sets them. Defaults: 0 and 1.0 (see the grid below for why they stay).

**Tuning on range 1** (43.5-47.5 s, subjects 1 and 3, no hard window and no label-fed tracker run). Each setting is
tracked on the finalised candidates of boxes `fcb6b961` (detection run `b3390a8b`, sequences `c740eb22` and
`95a56392`) and scored by `eval_kp_candidates.py track-picks`: each used POSITION observation of the run
(`obs_blob`, mode 0, inlier) is traced to the candidate of its keypoint, in the sequence row nearest in time,
whose undistorted position is the nearest to it (97-98% of the used observations are traced; the rest are
candidates merged from two crops, which keep one of the two positions). The score is the share of labelled
keypoint-frames, among those the run took a candidate for, where that candidate lies within 3% of the box height of
the label, next to the strongest maximum's share on the same keypoint-frames. Share of the picks within 3%,
subject 1 / subject 3 (the strongest maximum: 90.7-90.8% / 94.4-94.7%):

| `w_peak` \ multiplier | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 0 | 91.7 / 96.4 | 91.7 / 96.4 | 91.9 / 96.4 | |
| 0.5 | 91.7 / 96.4 | 91.8 / 96.4 | 91.8 / 96.3 | |
| 1 | 91.7 / 96.4 | 91.7 / 96.4 | 91.8 / 96.3 | |
| 2 | 91.7 / 96.3 | 91.7 / 96.3 | **92.7 / 96.3** | 92.6 / 96.3 |
| 4 | | | 92.6 / 96.2 | |
| 8 | | | | 91.6 / 96.2 |

The grid is flat: the settings move the share by less than 0.2 points, apart from a point at multiplier 4 or 8 with
`w_peak` 2 or more for subject 1, where the relaxed gate also keeps about 500 more keypoint-frames. Every setting
is 1-2 points above the strongest maximum on its own picks. Setting chosen: `w_peak` 2, multiplier 4: the best
of subject 1, equal for subject 3 within 0.1, in the middle of the plateau (its neighbours 92.6) with the grid's
corner (8, 8) lower. The evidence is thin; the labelled frames lack the contact windows where the picks fail
(any candidate is within 3% of the label for 99.7% of the labelled keypoint-frames, the strongest one for 94.3-94.9%).
The defaults stay at 0 and 1.0 (off): a default on needs a measured gain, and range 3 gives the chosen setting none over the cost
term alone (section 9.3). Tracklets exist for every keypoint candidate by construction, which is why continuity is a
setting here and not a precondition; whether it should be on follows from the `suggest` decision.

### 9.3 Acceptance run

Capture and range as in section 7 (the five-camera, three-person trial, range 3 = 51.7-56.8 s, subjects 1 and 2).
Boxes: `detect run --from-run fcb6b961 --candidates` (boxes from the masks; holds the prototype's crop on 82-83% of
the frames, section 9.4), run `b56f18cd`, finalised over the whole span of every track (a mask label is a stable
identity; the source run's assignments have gaps where the track identity was uncertain, which the prototype's
sequences do not have): subject 1 `9bfda1ba`, subject 2 `40fd1be1`. Tracking: the tracker config and skeletons of the
label-fed reference runs (`hier-dualhand`, skeletons with `soma77:`-qualified landmarks), 51.7-56.8 s, optimized
build. Scored by `eval_kp_candidates.py tracker`: mean body-joint distance (cm) to the label-fed reference run from
52.0 s, over the range and inside the hard windows. The setting is the one chosen on range 1, not tuned on this range.

| Run | Subject 1 range / hard | Subject 2 range / hard |
|---|---|---|
| prototype, automatic proposal (`908e75e4`, `eb7ade35`) | 12.7 / 14.8 | 4.4 / 5.7 |
| **prototype, cross-camera (`db888f74`, `c30c6322`)** | **10.2 / 10.9** | **3.4 / 4.2** |
| prototype, cross-camera + body mask term (`47613f1f`, `01aa291b`) | 10.0 / 10.2 | 3.3 / 4.0 |
| prototype, strict mask filter + proposal (`b3c714df`, `d8b198ed`) | 8.7 / 11.9 | 4.8 / 6.2 |
| prototype, strict mask filter + cross-camera (`bea2b0fd`, `150da557`) | 8.8 / 12.0 | 4.7 / 5.9 |
| productized, candidates off (`f751edad`, `1d44260e`) | 17.3 / 20.2 | 5.4 / 6.9 |
| productized, candidates on, `w_peak` 2, multiplier 4 (`ed62ae0c`, `b3e3ebf0`) | 16.8 / 19.0 | 5.6 / 7.2 |
| productized, candidates on + multi-view agreement (`8a495422`, `dabf4fab`) | 16.5 / 18.9 | 5.8 / 7.3 |
| productized, candidates on, cost only: `w_peak` 0, multiplier 1 (`380c2f18`, `763b375c`) | 15.3 / 17.3 | 5.7 / 7.3 |

(Run ids are those of the work copy of the spike database, which holds the prototype's runs too. The productized rows
were tracked with the capture's current extrinsic calibration, the prototype's runs and the reference with another: section 10.7
repeats the comparison on the reference's calibration. Target: no worse
than the prototype's cross-camera run plus 0.5 cm, at most 10.7 / 3.9 cm over the range: **not met**.)

**Result: candidates on do not clearly beat candidates off.** The gain over the run's own strongest maxima, which
leaves out the pipeline differences the candidates are not responsible for, is 0.5 cm (subject 1) and -0.2 cm
(subject 2) with the chosen setting, against 4.5 and 2.7 cm for the prototype's cross-camera choice over its
strongest-peak run; with the cost alone 2.0 and -0.3 cm. The correct candidate is chosen more often (range 3 picks
within 3% of the label: 88.2% against 86.9% for the strongest maximum on the same keypoint-frames for subject 1,
93.2 against 92.5 for subject 2), but the joint error does not follow:

- The likelihood cost turns the loss of the distance-only cost into a small gain (subject 2: 7.6 to 5.6 cm; the two
  measurements are on different boxes, `69c1d770` and `fcb6b961`). Subject 2 is still not better than candidates off.
- The continuity multiplier and the peak weight, as chosen on range 1, do not add to the cost term here: the cost
  alone is 1.5 cm better for subject 1 and equal for subject 2. This is reported, not used: range 3 is not for choosing.
- By joint (subject 1, cm, inside / outside the hard windows, off / chosen / cost only / prototype): the chosen setting
  improves the arms, shoulders, neck and head (forearm L 27.4 to 22.0 inside, head 31.7 to 28.5, upper arm L 33.3 to
  27.9) and worsens the legs (foot L 13.5 to 18.5, shin L 10.8 to 15.4, hips 11.0 to 12.5, toe L 13.4 to 17.7). The
  cost alone keeps the left foot and toe better than off (11.5 and 11.7 cm), the shin and hips a little worse (12.1,
  13.1), and the arms by more (forearm L 20.8, head 24.6). The right foot and toe are 18-30 cm off in every
  productized run, with or without candidates, and 13-16 cm in the prototype's.
  The prototype's choice (other cameras' picks and the tracklet over the whole range) is better than all of them on
  nearly every joint.
- Multi-view agreement on top: 0.3 cm better for subject 1, 0.2 cm worse for subject 2.

The prototype's gain came from a choice made per tracklet and over time, which the tracker's per-step assignment
does not reproduce. Section 10 describes the windowed choice that does the same inside the tracking run.

### 9.4 Like for like

- **Candidates.** `eval_kp_candidates.py compare` against the prototype's candidate file, boxes `fcb6b961`
  (run `b56f18cd`): on the frames whose crop is the same (2,466 of 3,013 for subject 1, 2,522 of 3,051 for subject 2;
  82-83%) the run reproduces 100.0% of the prototype's candidates within 0.5 px, and the prototype's 100.0% of the
  run's, once the file's positions are moved by half a heatmap cell: the frozen file stores the cell index at the
  cell's left edge, the decoder places a position at the cell centre (4 px at a crop of 513 px, 1.3% of the
  person's height), and nothing else differs. Over all common frames 82.2% of the prototype's candidates are
  reproduced (subject 1; 83.0% the other way) and 84.7% (subject 2; 83.0%). On the other frames (the crops differ
  by more than 1 px; 547 and 529 frames) 4-6% are reproduced; those frames lie on masks that differ between the state the
  box run was built from and the one the prototype's crops come from (section 10.7), and boxes rebuilt from the latter
  reproduce all the crops. The earlier box run `69c1d770` held the crop on 73-74%.
- **Frames.** The run's sequences cover 3,013 and 3,051 frames of the 5 cameras x 612 (the prototype's candidate
  file: 3,017 and 3,051 in the range); a frame without a box is absent in both.
- **Other differences of the pipeline.** The prototype's tracker sequences were built from the candidate file's
  picks (positions in the centre convention, like the label-fed reference's input; the frozen evaluation files are the
  left-edge export; one camera left on the strongest maxima), with keypoints it judged invisible removed; the productized sequence holds
  every strongest maximum. The strongest-maxima run of the productized pipeline is at the level of the prototype's
  own (subject 1: 17.3 cm against 14.7 and 18.5; subject 2: 5.4 against 6.1 and 10.2, for its range-3 and
  whole-trial strongest-peak runs).
- **Identity with candidates off.** A tracking run with `--no-candidates` of a sequence that holds the candidate
  rows, a run of a copy without the rows and a repeat are bit-identical (measured with the distance-only cost; a
  run without candidates does not reach the cost code). Two detection runs over the same frames are not bit-identical (105 of 3,007
  body rows differ, median 0.03 px, maximum 4 px, from the GPU), so comparing runs on sequences of two detection
  runs compares different data.

### 9.5 Cost

Optimized build, the acceptance runs one after the other, one machine, other work idle.

| | Candidates off | Candidates on (`w_peak` 2, multiplier 4) |
|---|---|---|
| Detection, 51.7-56.8 s, 5 cameras, 3 persons, boxes from a run (wall) | 6 min 23 s | 7 min 24 s (+16%; boxes `69c1d770`) and 8 min 11 s with `fcb6b961` as box source |
| Detection rows of the run | `full_body` 8.15 MB, hands 2.9 MB, previews 48 MB | + `kp_candidates` 21.0 MB: 2.4 kB per person, camera and frame (91 maxima), 2.6 times the ordinary row, +26% of the run's rows |
| Sequence rows after finalisation | | the same 21.0 MB again in `pose_observations` |
| Tracking, subject 1 (wall, peak memory of the process tree) | 97.4 s, 4,867 MB | 103.9 s (+6.7%), 4,915 MB (+1.0%); with multi-view agreement 107.3 s |
| Tracking, subject 2 | 105.6 s, 4,882 MB | 111.2 s (+5.3%), 4,927 MB (+0.9%); with multi-view agreement 112.1 s |
| Shared assignment per step with candidates (about 90 per camera and person) | | 13.8 ms (subject 1), 13.3 ms (subject 2), of 181 ms and 187 ms per step (distance-only cost, boxes `69c1d770`) |
| Strict mask option | | 130.8 s and 139.8 s (+30% and +22% over the plain candidate run with the distance-only cost: one mask decode per camera and step) |

The cost term adds one logarithm per pair and the continuity test one lookup. The +6.7% and +5.3% of tracking
time are of the size of the +10% and +4% measured with the distance-only cost.

### 9.6 Strict mask option

Measured with the distance-only cost, no continuity and boxes `69c1d770`, not with the likelihood cost; the
modifier is the same in both. Compared with the prototype's strict person-mask filter (a candidate kept only if at least half the detections of
its tracklet within +-10 frames lie on the person's mask; masks reviewed for 52.5-57 s only), scored by the harness:
the filter with the proposal gave 8.7 / 11.9 and 4.8 / 6.2 cm, with the cross-camera choice 8.8 / 12.0 and 4.7 / 5.9
(range / hard windows, subject 1 and 2); the tracker's mask modifier with candidates gives 14.6 / 16.0 and
7.9 / 10.2. The modifier is a different test: it judges a tracklet from the steps in which one of its candidates
won, up to the current step, not from its detections around the frame, and it has no effect on a tracklet that has
not yet won; the prototype's filter looks at all detections, before and after. The share of keypoint-frames the
filter empties for the persons in contact (27-34% in the hard windows, section 7) is not reproduced by the
modifier: it excludes a pairing, and only after the tracklet has a history of at least five steps. Multi-view
agreement gave the same improvement on subject 1 (14.4 cm) and more on subject 2 (6.9 against 7.9 cm).

## 10. Choice of candidates over a time window

Section 9 shows that the tracker's per-step assignment does not reproduce the prototype's results. The prototype
chose its candidates offline, per camera, keypoint and frame, from the candidates, the other cameras and the skeleton
alone. This section describes that choice as it runs in the tracker, as a phase of the tracking run before the first
filter step (`tracker_configs.kp_candidate_choice = 'window'`), and how it was measured.

### 10.1 The choice

The choice is made per person. Each person's candidates come from its own crops, and an exclusive assignment of
candidates between persons was measured as harmful and is not part of it (`WindowChoice`,
`keypoint_window_choice.hpp`; it has no database or tracker dependency).

1. **Consensus flag.** For each time bucket (a frame of the fastest camera) and keypoint, every pair of candidates
   from two cameras is triangulated, the other cameras' candidates within an inlier radius of the projection are
   counted, and the best set of at least three cameras is flagged `consistent`. The radius of a camera is 2.5 times
   the 30th percentile of the residuals of a first pass over every ninth bucket with a radius of 60 px, at least 12 px
   and at most 60 px. All cameras vote, the ones that take no part in the choice included. The flags are computed over
   the loaded window, after tracklet decisions and before the strict filter.
2. **Strict mask filter** (optional). A candidate stays only if at least a minimum share (0.5) of the detections of its
   tracklet within +-10 video frames lie on the person's own mask. The window looks backwards and forwards. Detections
   in a frame without a mask are not counted; a candidate with no counted detection stays. Removed candidates take no
   part in anything after it. A keypoint-frame left without a candidate has no pick and gives no observation: the
   tracker predicts that keypoint, which is better than following a wrong candidate.
3. **Starting pick** of every keypoint-frame: the strongest candidate that is `consistent`, else the candidate
   continuing the previous frame's tracklet, else the first (rank 0).
4. **Score of every candidate, then the best per frame, repeated for three rounds** from the previous round's picks of
   all cameras:

       score = A + delta * X + beta * consistent + gamma * peak [+ m * M]

   - `A`, skeleton-neighbour support: for each neighbouring keypoint with a pick in that frame, 1 if the candidate is
     within the pair's distance limit of that pick (in box heights, distorted pixels), decaying exponentially beyond it
     (scale 0.05 box heights); the mean over the neighbours. A pair without a limit gives no support.
   - `X`, cross-camera support: for each other camera that takes part and has a pick of the same keypoint at the
     nearest frame (within 8 ms), 1 if that pick lies within 0.04 box heights of the candidate's epipolar line
     (undistorted pixels), decaying beyond it; the mean over the cameras.
   - `M`, the mask term: +1 on the person's own mask, -1 on another person's, 0 on the background, for the keypoint
     classes the setting names (default: the body without hands and feet).
   - `A`, `X` and `M` are averaged over the candidate's tracklet within a window around the frame (+-1.0 s for `A` and
     `X`, +-2.0 s for `M`; prefix sums per tracklet, truncated, not padded, at the ends of the loaded window). A tracklet
     that jumps to another person's keypoint loses support around the jump only. Where the tracklet has no value in the
     window the average is 0.5 (0 for `M`). A candidate without a tracklet is a tracklet of its own.
   - `peak` is the heatmap peak, the stored peak over the pose model's confidence scale (5.0 for SOMA-77).
   - Ties go to the candidate of the lowest rank. Defaults: `delta` 1.0, `beta` 0.3, `gamma` 0.2, `m` 0.3 when on.
5. **Decisions.** A tracklet range rejected, marked as another subject's or forbidden for the keypoint's marker is
   removed; one assigned to the keypoint's marker is the pick of every frame of its range whatever its score. Hand-placed
   keypoint edits stay.

### 10.2 Against the tracker's per-step assignment

| | Windowed choice | Per-step assignment (section 9) |
|---|---|---|
| Time used | +-1 s around the frame, both directions, three rounds | the current step only |
| Main evidence | skeleton neighbours' picks, other cameras' picks | distance to the filter's one-step prediction |
| Tracklet | support averaged within the window | continuity with the previous step's choice, which follows a jump |
| Other cameras | epipolar distance to their settled picks; triangulation consensus | epipolar distance to their candidates at the step (option) |
| Mask | point lookup of the candidate's tracklet in both directions: removes candidates; soft term | tracklet history of won steps only, a neighbourhood of the position, excludes after five steps (option) |
| Nothing fits | no observation | the nearest candidate within the gate |

The per-step assignment depends on the prediction, so it keeps an error once the prediction carries it. The windowed
choice uses only 2D detections, the calibration and the masks, not the filter's state, which is why it can run before
the filter. The per-step assignment stays as the other mode of `kp_candidate_choice` (`'step'`, the default).

### 10.3 Where it runs: a phase of the tracking run

The choice runs inside `posetrak-tracker track` (`keypoint_choice_phase.cpp`), between loading and the first filter
step, and writes nothing but the run's own results:

1. load the candidates of the tracked window plus a margin on both sides (the support window, or the mask window if the
   mask term is on), every keypoint of the sequence's candidate manifest, any peak
   (`SessionReader::load_keypoint_choice_data`);
2. apply the tracklet decisions (`--tracklet-assignments`, exported from `dot_assignment_edits`);
3. read the mask label at every candidate at the run's pinned segmentation revision when the strict filter or the mask
   term is on (`mask_label_at`, with the decoder and the scaling of the per-step option; one decode per camera and
   frame of the window);
4. flag the consensus, apply the strict filter, compute the picks;
5. **replace the stored body row** of every (camera, frame) of the cameras that take part by the picks
   (`load_observations(..., row_overrides)`) and build the observations from them. The picked candidate's position and
   confidence take the place of the stored keypoint; a keypoint without a pick has confidence 0. Replacing only the
   positions of the bound markers is not enough: the child-minus-parent pairs of a configuration with relative
   observations are built from the same row and would carry the strongest maxima into the filter next to the picks.
   A camera that takes part in no choice (`kp_choice_skip_camera_ids`, a wide-lens camera, say) keeps its stored row
   and does not support the picks of the others.

This gives one command and one run record, and no stored intermediate result to keep consistent with masks, edits or
detection. Re-tracking after an edit is the same command. `run-persons` runs the phase for each person without masks.

**Stored picks.** The run records its picks in `tracking_kp_picks`, one row per (run, person, camera): little-endian
blobs for the video frames, `int32[frame, keypoint]` tracklet ids (-1 no pick, -2 picked without a tracklet) and
`int8[frame, keypoint]` ranks, with the keypoint-frames that had a candidate, the ones the strict filter emptied and
the ones picked. The editor of tracklet decisions reads what the run used from this table without recomputing it. A
blob per camera (about 400 bytes per frame and camera for 77 keypoints) is read whole and needs no join for the
editor's view, a chain of picks per keypoint over time; `tracking_assignment_trace` has a row per candidate and slot
for a requested range of steps only, is opt-in, and would need a row per keypoint-frame and camera for every run. The share of keypoint-frames the filter emptied is printed per person in the
run's log and sums from the rows.

`track run --kp-choice-dump FILE` writes every candidate with its consensus flag, whether it was removed and whether it
was picked, for comparing the choice with another implementation.

### 10.4 Neighbour distance limits without labels

The limit of a pair of neighbouring keypoints is a property of the keypoint layout and of the box, not of a person.

- **A table shipped with the pose model** (`posetrak/detection/soma77_neighbour_limits.py`, written into a choice model
  file for the tracker by `kp_choice_model.py`): 1.15 times the 98th percentile of the labelled distance between the
  keypoints of the pair in box heights, pooled over the persons and cameras of a labelled range, for the pairs with more
  than 50 samples. `python/tools/gt/derive_neighbour_limits.py` derives it. The shipped table comes from range 1 of the
  evaluation set only (28 pairs, 9,300-10,200 samples each: the labelled units cover the body, not the fingers; limits
  0.06-0.58 box heights, median 0.30) and is evaluated on range 3. A pair outside the table gives no support.
- **Estimated in the run** (`estimate_neighbour_limits.py`): the same percentile over the candidates the consensus
  flagged. Over range 3 it covers all 76 pairs, but it is looser than the table where picks agree on the wrong
  candidate: on the 28 pairs of the table the estimate is 0.96-5.3 times the table (median 1.60) for subject 1 and
  0.48-2.1 times (median 0.95) for subject 2, with the largest ratios on the jaw, neck, toes and head. The tracker
  results with it are not better (section 10.7), so the table stays. `track run --kp-choice-model FILE` takes another
  file.

### 10.5 Settings

In `tracker_configs`, each recorded with the run and set by a `track run` option; the defaults are the prototype's
values. A NULL column means the default.

| Column | Option | Default |
|---|---|---|
| `kp_candidate_choice` (`'window'` \| `'step'`) | `--kp-choice` | `'step'` |
| `kp_choice_support_window_s` | `--kp-support-window` | 1.0 |
| `kp_choice_delta`, `kp_choice_beta`, `kp_choice_gamma` | `--kp-delta`, `--kp-beta`, `--kp-gamma` | 1.0, 0.3, 0.2 |
| `kp_choice_rounds` | `--kp-rounds` | 3 |
| `kp_choice_skip_camera_ids` (JSON array of camera indices) | `--kp-skip-camera LABEL` (repeatable) | none |
| `kp_strict_filter`, `kp_strict_filter_window_frames`, `kp_strict_filter_min_fraction`, `kp_strict_filter_classes` | `--kp-strict-filter`, `--kp-strict-window`, `--kp-strict-min-fraction`, `--kp-strict-classes` | off, 10, 0.5, `body` |
| `kp_mask_term_weight`, `kp_mask_term_window_s`, `kp_mask_term_classes` | `--kp-mask-weight`, `--kp-mask-window`, `--kp-mask-classes` | 0 (off), 2.0, `body` |

The mask options need `--seg-run` (the label of the person is read from the subject registry, or given with
`--seg-label`). The strict filter needs reviewed masks around the contact windows: on unreviewed masks it removes
correct candidates of hidden keypoints. The run reports the share of keypoint-frames the filter emptied, per person.
The choice is used only with `--candidates`.

Two settings outside `tracker_configs` decide what the choice starts from. `detect run --segmentation` multiplies the
SOMA heatmaps with the person's mask grown by `--mask-gate-margin` pixels (default 40; 0 turns the gate off). The
candidates are the maxima of the heatmap before the gate; the stored keypoint row, which `'step'`, candidates off and
the cameras outside the choice use, is the strongest maximum after it. `--kp-skip-camera` takes a camera out of the
choice while its stored row stays an observation: a camera whose lens model fits only part of the image (the ace2 pro in
its wide mode) is better left out of the choice (section 10.7).

### 10.6 Cost

Optimized build, range 3 (51.7-56.8 s), five cameras, one person (332,000 candidates, 176,000 keypoint-frames of the
four cameras that take part), the whole run about 107 s:

| | Measured, 5 s | Extrapolated to an 85 s trial |
|---|---|---|
| Phase wall time | 2.5 s: load 0.1, consensus 1.7, choice 0.1, the rest (conversion of the candidates, reload of the observations) 0.6 | about 43 s (linear in the candidates; the consensus is the largest part) |
| With the strict filter or the mask term | + 14.6 s of mask decoding (3,013 camera-frames, 4.9 ms each) | + about 250 s per person |
| Peak memory of the phase | 331 MB at the end of the phase, 216 MB before it (+115 MB) | + about 2 GB if it grows linearly; the filter run peaks at about 4.7 GB for 5 s and 14-20 GB for an 85 s trial |
| Whole run | 98 s with candidates off, 107-115 s with the choice (the spread between runs is about 10 s); the process peaks at 4.72 GB either way | |

The consensus triangulates every pair of candidates of two cameras for each keypoint and bucket, which the first
estimate of this section left out; the scoring itself is about 0.1 s.

### 10.7 Acceptance results

Range 3 (51.7-56.8 s), subjects 1 and 2, scored by `eval_kp_candidates.py tracker` (mean body-joint distance in cm to the
label-fed reference run from 52.0 s, over the range / inside the hard windows), optimized build, tracker configuration
and skeletons of the label-fed reference runs, extrinsic calibration `5351f35e` of the reference and of the
prototype's runs (the productized rows of section 9.3 used the capture's current one, `b416cd21`).

**Boxes and masks.** The box run `fcb6b961` was built from the masks of segmentation `e862c9af` as the operator's
database held them; the prototype's crops, and so its candidate files (`gt6`, `gt6m`), come from the masks of a later
state (the prototype's snapshot of 10 October). The two differ in 1,132 masks, and the work copy's own masks, which
equal the prototype's current database, are a third state. All 547 and 529 frames whose crop differs from the prototype's
lie on frames where the person's mask differs between the first two states (576 frames, all in two cameras, `pixel9`
and `gopro-11_mini_02`); 29 and 47 frames with a different mask have the same tight box. The stored boxes of
`fcb6b961` equal the tight boxes of the first state on all 3,017 and 3,055 range-3 frames, and the prototype's crops
equal those of the second on all of them. The masks of the segmentation run were edited in place without a new
revision, so neither the box run nor a tracking run records which version of the masks it used (the box run's
`config_json` names the run, not a revision).

The figures of the next table are on boxes rebuilt from the second state with the script that built `fcb6b961`, on candidates detected without the mask gate (the product's path with the gate is below)
(`make_boxes_run.py`, a prototype script; run `1defb4f3`): the prototype's crop is
reproduced on all 3,013 and 3,051 frames, and 99.8% and 99.2% of its candidates within 0.5 px (the rest are strong
maxima, 73% of them on fingers and hands, that moved by 1-2 px, which GPU non-determinism and the sub-cell refinement
explain). Detection run `cf7b5581`, sequences `4d68198d` (subject 1) and `401c8527` (subject 2). The strict filter and the
mask term read the same masks.

| Run | Subject 1 range / hard | Subject 2 range / hard |
|---|---|---|
| prototype, cross-camera (`db888f74`, `c30c6322`) | 10.2 / 10.9 | 3.4 / 4.2 |
| prototype, cross-camera + body mask term (`47613f1f`, `01aa291b`) | 10.0 / 10.2 | 3.3 / 4.0 |
| prototype, strict mask filter + cross-camera (`bea2b0fd`, `150da557`) | 8.8 / 12.0 | 4.7 / 5.9 |
| the prototype's input sequences of the cross-camera runs, productized tracker (`5304c2ae`, `d7df80f0`) | 10.9 / 11.4 | 3.3 / 4.2 |
| candidates off (`66819882`, `1f9e3d05`) | 14.3 / 16.7 | 4.7 / 6.1 |
| `'step'`, `w_peak` 2, multiplier 4 (`730814f7`, `89c175e8`) | 13.9 / 16.0 | 5.4 / 7.2 |
| **`'window'`** (`131e538e`, `f75ec6dc`) | **13.5 / 15.4** | **4.0 / 5.1** |
| `'window'` + strict filter (`a7e8bbf3`, `4d9029c2`) | 9.4 / 11.6 | 6.0 / 7.5 |
| `'window'` + body mask term (`007da5e4`, `93dfe252`) | 12.9 / 14.7 | 3.9 / 4.9 |
| **`'window'` + strict filter on the body keypoints** (`03ddcd74`, `63e2a2c1`) | **9.4 / 10.7** | **3.7 / 4.6** |
| `'window'` + strict filter, tracklets of 100 or more detections spared (`i-strict-x100`) | 13.3 / 15.5 | 3.3 / 4.1 |
| `'window'` + strict filter on the body, tracklets of 100 or more spared | 12.9 / 14.7 | 3.5 / 4.3 |
| `'window'` + tracklet-length term, weight 1 and 3 | 13.3 / 15.0 and 13.6 / 15.8 | 4.0 / 4.9 and 4.0 / 5.0 |

(Target at most 10.7 / 3.9 over the range: `'window'` alone **does not meet it**, by 2.8 cm for subject 1 and 0.1 cm for
subject 2; `'window'` with the strict filter restricted to the body keypoints met it on these ungated candidates, 9.4 and
3.7, but not on the product path below, 10.73 and 3.48.)
The strict filter emptied 39,123 of 177,483 keypoint-frames (22.0%) of subject 1 and 49,792 of 183,110 (27.2%) of
subject 2. On the boxes of `fcb6b961`, which hold 82% of the prototype's crops, the same runs gave 17.8 / 5.2, 17.1 / 5.3,
16.0 / 4.6, 15.9 / 4.5 and 15.7 / 4.6 cm (subjects 1 and 2, range) for candidates off, `'step'`, `'window'`, `'window'` +
strict filter and `'window'` + mask term; the corrected boxes lower the candidates-off figure of subject 1 by 3.5 cm and
the windowed one by 2.5 cm.

On the corrected boxes the windowed choice is better than candidates off and the per-step assignment for subject 1
(0.8 and 0.4 cm) and for subject 2 (0.7 cm better than off, 1.4 cm better than `'step'`, which is worse than off for
subject 2). The strict filter changes subject 1 by 4.1 cm (13.5 to 9.4, as the prototype's filter did) and makes
subject 2 worse by 2.0 cm: it removes the correct candidates of keypoints hidden under the other person's mask, which
section 10.5 warns about. The mask term gains 0.6 and 0.1 cm.

**Why the filter is restricted to the body keypoints.** Against the labels of range 3, the accuracy of the picks
(within 3% of the box height) is 83% when the pick lies on another person's mask in its own frame and 94% on the
person's own mask, but 81% when the pick's tracklet is never on the person's own mask within +-10 frames and 94.5% when
it always is: the window separates far better than the frame. The filter's rule (own share below 0.5) removes 58% of
the wrong body picks and loses 14% of the correct ones, 47% and 29% for the feet, 47% and 46% for the hands: the masks
of the hands and feet are thin and wrong in contact, so there the rule discards as many correct picks as wrong ones. With
the body keypoints alone the filter empties 6.3% and 4.8% of the keypoint-frames instead of 22.0% and 27.2% and
keeps the gain of subject 1 without the loss of subject 2. The tracklet length is the strongest single signal of a
wrong pick (body keypoints, tracklets of at most 10 detections: 73-76% correct on or off the person's mask; of more
than 100: 99.5% on and 93% off it), but a score term for it did not help (weights 1 and 3), and sparing the long
tracklets from the filter gave back the filter's gain in subject 1 (12.9 against 9.4): the wrong picks of subject 1 sit
on long tracklets of the other person's keypoints. The class restriction was chosen after seeing the range 3 figures
of six variants; the pick accuracy of range 1 (43.5-47.5 s, subjects 1 and 3, strict filter over all classes against
none: 92.9 to 93.5 and 96.5 to 97.5%) cannot separate them, because it leaves out the keypoint-frames a filter empties.

**Product path: the mask box source and the gate.** `detect run --segmentation` on the same masks (run `78b3486c`, sequences
`2bfa4f66`, `94447f4d`, `eab70a9f`; the work copy holds exactly the mask state of the prototype's crops) gives person boxes
identical to run `1defb4f3` on all 9,122 boxes of range 3 (largest difference 0.0 px) and reproduces the prototype's
crops on all frames and its candidates in 99.8% and 99.2%, as the run on the script's boxes does: the box path
contributes nothing. The gate leaves the candidates alone and changes the stored keypoint row. Same variants on it
(calibration `5351f35e`, ace2 pro outside the choice, subjects 1 and 2, range / hard windows; the ungated figures are those
of the table above):

| Variant | Gated (product): subject 1 | subject 2 | Ungated: subject 1 | subject 2 |
|---|---|---|---|---|
| candidates off | 11.4 / 12.3 | 4.6 / 6.0 | 14.3 / 16.7 | 4.7 / 6.1 |
| `'step'` | 11.9 / 13.0 | 4.6 / 6.0 | 13.9 / 16.0 | 5.4 / 7.2 |
| `'window'` | 13.1 / 15.0 | 3.8 / 4.9 | 13.5 / 15.4 | 4.0 / 5.1 |
| `'window'` + strict filter on the body | 10.7 / 11.9 | 3.5 / 4.5 | 9.4 / 10.7 | 3.7 / 4.6 |
| `'window'` + body mask term | 12.9 / 14.6 | 3.7 / 4.8 | 12.9 / 14.7 | 3.9 / 4.9 |
| prototype, cross-camera | 10.2 / 10.9 | 3.4 / 4.2 | | |

(gated runs `e20d9f1a`/`0d4d997f`, `8742539f`/`388d0909`, `ad73c888`/`b79acc74`, `2bccac80`/`1e5c2c8e`,
`adcbd2ea`/`e23aafe3`.) The gate lowers candidates off by 2.9 cm for subject 1 and the per-step assignment by 2.0 cm,
because it removes the maxima on the other person from the stored row; the windowed choice replaces that row and gains
nothing from it (0.4 cm), and the filter's result for subject 1 is 1.3 cm worse than without the gate (10.73 against the
bar of 10.7; subject 2 3.48 against 3.9). With the gate the windowed choice without the filter is worse than candidates
off for subject 1 (13.1 against 11.4) and better for subject 2 (3.8 against 4.6).

**The ace2 pro camera.** The ace2 pro's wide mode leaves the centre of the image almost undistorted and the edge
strongly distorted, which the radial-tangential model fits only inside the area where the people move. The capture's
refined extrinsics `b416cd21` with the camera's intrinsics fitted to that area (`2a5ad9cf`: pinhole with k1 and k2, 3840 x
2160) and the tracker configuration with the camera as a position camera give a reference and runs on a far better
basis than `5351f35e`. The product reads those intrinsics as the prototype did (the original camera matrix, the five
radial-tangential coefficients, the image size of the row, the undistorted matrix equal to the original); its inversion of
the distortion differs from an exact one by at most 0.02 px up to a distorted radius of 1.25 focal lengths, beyond which
the fitted model, whose forward map stops being monotone at a radius of 1.36, has no inverse (10% of the image area, the
corners). Tracker distance to the label-fed reference on the same calibration with the camera as a position camera
(`855e1e19`, `bc3886f9`), product-path sequences, range / hard windows:

| Variant | Subject 1 | Subject 2 |
|---|---|---|
| candidates off (`085f3fb8`, `c94125e0`) | 8.1 / 11.2 | 4.0 / 5.3 |
| `'step'` (`f0f0ef8b`, `73d44276`) | 6.6 / 8.9 | 4.0 / 5.3 |
| `'window'`, ace2 in the choice (`d6aaefd0`, `e6b7fd32`) | 6.5 / 8.7 | 3.6 / 4.7 |
| `'window'`, ace2 outside the choice (`7ab6edba`, `d96476fc`) | 6.0 / 8.0 | 3.6 / 4.6 |
| `'window'` + strict filter on the body, ace2 in (`34920a1a`, `bd60e04c`) | 7.5 / 10.4 | 3.8 / 4.8 |
| `'window'` + strict filter on the body, ace2 outside (`36afbdaf`, `de1e686b`) | 6.8 / 9.2 | 3.5 / 4.5 |
| `'window'` + body mask term, ace2 in (`46f90c67`, `67e06422`) | 6.5 / 8.6 | 3.6 / 4.7 |
| `'window'` + body mask term, ace2 outside (`ab889359`, `b8b95c4e`) | 6.0 / 8.0 | 3.5 / 4.5 |
| prototype, cross-camera + mask term, ace2 position (`841ec44f`, `0a61936d`) | 5.5 / 7.3 | 2.5 / 3.5 |
| prototype, ace2 velocity-only (`ae5a7bd4`, `55314e45`) | 12.4 / 15.5 | 5.0 / 6.3 |

The best product setting here is `'window'` with the ace2 outside the choice, 0.5 and 1.1 cm behind the prototype. The
strict filter on the body keypoints no longer helps: it makes subject 1 worse by 0.8 cm and improves subject 2 by 0.1
cm. Under `b416cd21` the consensus inlier radius of the ace2 pro is 12 px, the floor and the same as the other cameras
(12-13 px), against 60 px under `5351f35e`; 41.6% of its candidates and 62.4% of its picks are in a consistent set,
against 31-46% and 46-65% for the other four.

What was checked:

- **The choice itself is the prototype's.** Given the same candidates, the same tracklets and the consensus flags of the
  prototype's own code, the tracker's choice and `xcam.XAssign.run` pick the same candidate for 176,112 of 176,113
  keypoint-frames of subject 1 and 184,520 of 184,523 of subject 2 (`eval_kp_candidates.py window-picks` against the
  prototype's picks of the run's own candidates). The consensus flags equal those of the prototype's code on the same
  candidates in 99.994% of the candidates.
- **The tracker is the prototype's.** The prototype's own input sequence of the cross-camera run, tracked by the
  productized tracker, gives 10.9 / 11.4 and 3.3 / 4.2 cm.
- **Picks against the prototype's on the same crops.** Against the picks of the prototype's choice on `gt6m`, the file
  its tracker runs used, with the same neighbour limits, the run picks the same candidate for 96.2% (subject 1) and
  96.1% (subject 2) of the keypoint-frames with a pick: 99.5% and 99.2% outside the hard windows, 93.8% and 93.9%
  inside them (bar: 99%; 6,761 and 7,122 keypoint-frames differ, a prototype candidate is missing from the run's in 2
  and 22 of them). The remaining difference is the consensus flags: the file's stored flags cannot be reproduced (below),
  and on the old boxes, against the picks on the frozen evaluation file, giving the run's candidates that file's flags raised
  the agreement from 93.2% to 97.3%. The prototype's choice does not depend on which of two tracklet partitions it uses.

**Half-cell convention.** The frozen evaluation files store a heatmap maximum at the cell's left edge; `gt6` and `gt6m`
are those files plus exactly half a cell (4.2 px at a 513 px crop: 1.00 of a half cell, in x and y), so they, the
prototype's input sequences, the label-fed reference's input sequence and the productized pipeline are all in the centre
convention, and the scoring has no convention bias. The evaluation files are the left-edge export, used for the pick
scores against the labels. A sequence moved to the left edge (the prototype's cross-camera input minus half a cell) gives
13.7 / 15.0 and 4.9 / 5.6 cm against 10.9 / 11.4 and 3.3 / 4.2, so the scores are sensitive to the convention, 2.8 and
1.6 cm. Which convention is right was not decidable on this capture: the cross-camera reprojection residual of the
consensus sets is 17.6 px (left edge) and 17.4 px (centre) for subject 1 and 18.0 and 18.4 px for subject 2, a
difference of 0.3-0.4 px under a floor set by the calibration (the wide-lens camera is at 30 px), and a scan of the
shift has its minimum at +0.5 cells for subject 1 and at 0 to +0.25 for subject 2. The spike's measurement on the moving-box reference capture, with
good extrinsics, has a sharp minimum at +0.52 cells in x (residual from 1.33% to 0.76% of the box side), which is the
evidence for the centre convention; the decoder keeps it.

**Consensus flags of the frozen files.** Recomputed with the prototype's `link_candidates.py` on the candidates of the
frozen file under each of the extrinsic calibrations `5351f35e`, `b416cd21` and `9abfc616` and both conventions, and on
`gt6` under `9abfc616`, `8c53e26a` and `5d2295a7`, the flags disagree with the stored ones in 16-25% of the candidates;
the best combination (`b416cd21`, centre) still disagrees in 15.9% (with `gt6`) and 25.9% (with the frozen file). The
stored flags of `gt6` and `gt6m` are identical and differ from those of the frozen file in 23.9%, so the files hold the
output of different computations whose inputs are not recoverable: the spike repository has one version of the script,
unchanged since 6 October, and the files date from 11-17 October. The productized consensus equals the prototype's code
on the same input, so there is no bug to correct. The consensus does depend on the calibration: under `5351f35e` the
wide-lens camera's residuals are about 24 px and its inlier radius becomes 60 px, against the 12 px floor under
`b416cd21`, which has that camera's corrected intrinsics.

The windowed choice is not made the default, and the strict filter and the mask term stay off.

**Review material** for the keypoint-frames where the run's picks still differ from the prototype's: contact sheets
outside the repository (`D:/mocap/stage9w-work/review/`, an `index.html` and a folder per subject and camera).

### 10.8 What the first description got wrong

- The phase needs the consensus (section 10.1, item 1), which is the largest part of its cost, and the picks have to
  replace the stored keypoint row, not only the fixed observations of the bound markers (section 10.3).
- The pair limits are 1.15 times the 98th percentile and need more than 50 samples; the labelled units cover the body
  only, so the table covers 28 of the 76 pairs.
- The peak weight applies to the heatmap peak, not to the stored confidence.
- The prototype's runs, and the reference they are scored against, were tracked with another extrinsic calibration than
  the capture's current one.
- The estimate of the cost (section 10.6) left out the consensus and the mask decoding.
- The crops that differed from the prototype's were not a missing run but a mask version: the box run was built from
  one state of the segmentation's masks and the prototype's candidates from another. A box run and a tracking run need
  to record which version of the masks they read (a revision, or a hash of the masks of the frames used).
- The frozen evaluation files are the left-edge export of the candidates; the prototype's tracker runs and the label-fed
  reference were fed the centre convention, so the scoring has no convention bias.
- The prototype's results were not reproducible from the productized candidates mainly because of the box masks, not
  because of the choice: on boxes from the same masks the windowed choice is within 0.1 cm of the bar for subject 2 and
  2.8 cm from it for subject 1.
- The figures with the filter on the body keypoints were measured on candidates without the mask gate, which the product
  applies by default; they were also selected among six variants on the one range. With the gate the filter's result for
  subject 1 is 10.73 cm (bar 10.7), and on the better basis (fitted ace2 intrinsics, `b416cd21`) the filter makes subject 1
  worse. The restriction to the body keypoints stands as the less harmful form of the filter; that the filter helps at all
  is not established outside range 3.
- The mask box source reproduces the boxes of the prototype's script exactly; the difference between the product's figures
  and the earlier ones is the gate, which changes the stored keypoint row.

### 10.9 Foot orientation with the SOMA-77 keypoints

SOMA-77 has three keypoints on each foot: the ankle (`Foot`), the ball of the foot (`ToeBase`) and the toe tip
(`ToeEnd`). They lie almost on one line along the foot, so they fix the foot's direction but leave its twist about that
line poorly determined; the COCO-133 foot keypoints (big toe, small toe and, if present, heel) form a triangle and fix
it. How the markers are bound to the joints decides how much of this the tracker can recover.

In the composed skeleton the chain is hips - thigh - shin - foot - toe, every joint a three-axis ball joint. With
`ToeBase` and `ToeEnd` both bound to the toe joint, the foot has only the ankle keypoint of its own: the two toe markers
depend on the foot and the toe rotation together, and the toe joint (limits of 45 and 90 degrees) takes up the foot's
orientation. Tracking runs over a 28 s trial of three people showed the toe bent 30-58 degrees almost permanently, the foot
twist at its limit of 20 degrees, and the foot axis 30-40 degrees away from the foot axis of runs tracked from edited COCO-133
keypoints, in which the toe markers are bound to the foot and the toe joint stays at rest. The shin twist (limit 90
degrees) is also hardly observable from the keypoints; after initialisation it stayed at its limit for about 3 s, until the
person took a step.

Changes tried on the same trial, each against the edited runs:

| Change | Result |
|---|---|
| `ToeBase` bound to the foot (its offset moved into the foot frame), toe joint locked | the foot has three rigid markers; the foot-axis difference falls to 20-29 degrees; the toe stays at its rest angle |
| shin twist limit 35 degrees | the shin twist no longer sits at its limit after initialisation |
| toe joint limited to 15 degrees (6 degrees of twist) instead of locked, `ToeBase` on the foot | foot-axis difference 21-30 degrees, within 1.5 degrees of the locked toe; the angle between foot and toe axes has a median of 11-20 degrees (6 degrees of it is the rest angle) and a 90th percentile of 16-26 degrees |
| foot twist and swing limits widened to 40 degrees | no improvement; the foot twists up to 50-68 degrees, i.e. the keypoints and their offsets disagree with the foot frame |

**Toe flexion range.** The toe joint's rest frame is turned 34 degrees about the bone against the foot's, so the toe bends
about a diagonal of the x-z plane of its frame, not about x or z alone: in that frame the shank points along (0.87, 0, -0.49)
on the left and (-0.84, 0, -0.55) on the right (median over the trial, all three subjects), which gives the flexion axes
(-0.49, 0, -0.87) on the left and (-0.55, 0, 0.84) on the right, positive towards dorsiflexion. The bundled limits give the
toe about 90 degrees of dorsiflexion and 40 of plantar flexion about that axis, 15 degrees of side swing and 6 degrees of
twist: x from -57 to 33 degrees and z from -86 to 43 degrees on the left, x from -63 to 35 and z from -42 to 84 degrees on the
right. Limits of 15 degrees on x and z, as in the table above, hold the toe near its rest angle and cannot follow a bent toe.

With these limits on the trial (28 s, windowed choice, ace2 pro outside the choice, three subjects):

- Standing (hips above 0.8 m) the toe dorsiflexion has a median of 2 and 3 degrees (left, right) for subject 1, 2 and -6
  for subject 2 and -8 and -21 for subject 3; the 5th to 95th percentile spans 40-50 degrees.
- Subject 1 kneels at 65-66.4 s with the toes tucked: the median dorsiflexion is 67 degrees on the left and 46 on the right,
  and the largest is 87. The frames at 46.9-47.3 s and 56.8-57.9 s have the knees low but not the toes tucked, and the toe stays
  at -9 to 14 degrees.
- The foot-axis difference to the edited runs is 20 / 25, 25 / 19 and 30 / 20 degrees (left / right, subject 1 to 3), within 4
  degrees of the 15 degree limits and of the locked toe.
- On the labelled range the tracker distance to the label-fed reference (subject 1 / subject 2, range / hard windows) is
  6.65 / 8.66 and 3.72 / 4.83, against 6.5 / 8.4 and 3.60 / 4.84 with the toe locked and 6.0 / 8.0 and 3.59 / 4.60 with the toe on
  the original binding. Body-joint distance to the edited runs and the jerk of the joint positions did not change.

The remaining difference of 20-30 degrees in the foot axis follows from the marker offsets, which were fitted for upright
and squatting poses.

The bundled skeletons follow these results: `ToeBase` is bound to the foot in the SOMA-77 marker set
(`catalog/pose-models/soma77.marker-offsets.yaml`), and the toe limits are those above (`joint_limits.yaml`). The shin twist is
already locked in the bundled limits (the 90 degree twist above belongs to skeletons built outside the bundled layers), so it
needs no change there. The ids of the bundled default skeletons change with their content; skeletons already stored in a session
or registry keep the earlier binding and limits and have to be rebuilt from the marker set to get the new ones.

**Open design question: the twist of the foot.** The twist of the foot about its long axis cannot be observed from the three
nearly collinear SOMA-77 foot keypoints, whatever the binding: it stays at what the limits and the marker offsets allow.
Options, none chosen:

- a foot-contact or ground prior, which would also help against foot sliding;
- the foot keypoints of COCO-133 as additional measurements;
- markers on both sides of the foot in production captures;
- the foot orientation of the per-camera body model fit.

### 10.10 Jitter of the tracked motion

The tracked joint positions carry high-frequency noise that shows in a rendered clip. `python/tools/gt/jitter.py` measures
it and relates it to the events of the choice and the filter (README of `python/tools/gt/`).

**Metric.** The 3D position of each body joint (joint origins, finger joints left out) is sampled at 60 Hz. The jitter is
the residual from a Savitzky-Golay fit of order 3 over 0.2 s, divided by `sqrt(1 - h0)` (`h0` the centre coefficient of the
filter) so that white noise of a given RMS gives that RMS back; a smooth motion plus white noise of 4 mm per axis returns
6.94 mm in 3D against 6.93 expected (`jitter.py synthetic`). The jerk is the third derivative of a fit of order 4 over the same
span. A time series is the RMS of the squared residual over a sliding 0.5 s (30 samples), advanced one sample at a time. An
episode of a joint group is a run of consecutive samples above that group's threshold. The threshold is the 99th percentile of
the series of the reference runs, pooled over the subjects: the tracker's own noise on good input then crosses it on one
sample in a hundred by construction, and seldom for long. Fast genuine motion raises the residual of the reference runs as
well, so an episode that the reference run of the same subject also has in the same group, within 0.25 s, is counted as
motion and not as jitter of the run under test.

**Runs compared** (43.5-66.4 s, three subjects): the windowed choice (ace2 pro outside the choice, no filter, calibration
`b416cd21`, the ace2 configuration, the product's box path; runs `e8e1bb80`, `5dc83449`, `ad3708a3`), the prototype's
`ace5` runs and the operator-edited runs. All use the same filter settings (process noise 0.3, measurement noise 25 px,
outlier threshold 6, RTS smoothing on). The edited run of subject 1 has no smoothed states, so its figures are those of the
filtered states and are higher than they would be smoothed. The thresholds come from the edited runs of subjects 2 and 3.
Thresholds of the sliding RMS (mm): torso 31, arms 35, hands 50, legs 35, feet 67.

RMS residual (mm) / RMS jerk (m/s^3) over the trial, smoothed states:

| Run | Torso | Arms | Hands | Legs | Feet |
|---|---|---|---|---|---|
| windowed, subject 1 | 14.7 / 248 | 18.1 / 334 | 28.1 / 551 | 14.1 / 288 | 28.1 / 659 |
| windowed, subject 2 | 11.9 / 197 | 19.5 / 407 | 37.1 / 777 | 12.3 / 230 | 21.2 / 458 |
| windowed, subject 3 | 14.2 / 282 | 18.2 / 349 | 25.0 / 474 | 11.9 / 250 | 21.8 / 502 |
| prototype `ace5`, subject 1 | 15.4 / 254 | 18.8 / 345 | 28.9 / 579 | 15.4 / 305 | 33.3 / 791 |
| prototype `ace5`, subject 2 | 11.5 / 192 | 18.7 / 368 | 34.5 / 712 | 12.7 / 238 | 25.1 / 532 |
| prototype `ace5`, subject 3 | 16.6 / 350 | 19.4 / 398 | 28.0 / 529 | 16.0 / 356 | 28.0 / 616 |
| edited, subject 1 (filtered) | 18.1 / 370 | 22.1 / 451 | 31.5 / 615 | 20.2 / 395 | 32.0 / 654 |
| edited, subject 2 | 9.0 / 167 | 11.1 / 217 | 18.2 / 383 | 13.2 / 264 | 27.2 / 502 |
| edited, subject 3 | 11.3 / 229 | 13.4 / 279 | 20.2 / 431 | 14.2 / 305 | 24.2 / 536 |

- The windowed run jitters as much as the prototype's run (differences of 1-4 mm) and, on the torso, legs and feet, about as
  much as the tracker does on edited keypoints. The tracker itself on good input leaves 9-14 mm on the torso and legs and 24-27
  mm on the feet. The arms and hands are where keypoint input matters: 1.5 to 2 times the edited runs for subjects 2 and 3 (arms
  18-20 against 11-13 mm, hands 25-37 against 18-20 mm).
- Over the trial the windowed run has 18 of 27 episodes that the edited run does not have. The hands of subject 2 have six
  (4.1 s in total) and the arms of subject 2 five. The prototype's runs have the same episodes at the same places.

**The reported moments.** Subject 1 at 53.2-54.4 s and 55.35-56.25 s (the two hard windows) has episodes in the torso, arms and
hands, and at 53.8-54.3 s in the legs; the edited run has episodes at the same times, so they are not specific to the choice.
At 42-44 s (4-6 s into the trial) subject 1's jitter stays below the thresholds. Subject 2 has no episode at 58 s (20 s into
the trial); the windowed run, the prototype's and the edited run agree on the legs there within 6-7 cm, and the observed
ankle positions of subject 2 stay nearly still in several cameras at 57-58.25 s, so the stall in the walk is in the data
and not a loss of tracking.

**Where it comes from.** The rate of each event inside the episodes against its rate over the trial (windowed run, over the
markers of the joint group), and the share of the variance of the smoothed jitter that the events explain. Fast motion raises
both, so the regression has the group's speed as its first regressor: the speed alone explains 0.45 of the variance on
average over the subjects and groups (0.06-0.79), the speed and all events 0.85. The gain in R^2 when one event is added to
the speed (mean over 3 subjects x 5 groups): jump of an observation from the fit of its own track (more than 12 px from the
fit over 0.2 s) 0.31, tracklet switch of the pick 0.23, change in the number of cameras observing the marker 0.18, outlier
rejection 0.12, NIS spike 0.04, missing observation 0.02, run of five missing observations 0.01. Of the 18 episodes the
edited run does not have, the event rate inside the episode is at least twice the rate over the trial for jumps in 13, tracklet
switches in 15, changes of the camera count in 12, NIS spikes in 13, outlier rejections in 10, missing observations in 5 and
runs of missing observations in 4; every episode has at least one. The events overlap (a switch to another tracklet usually
is a jump, and changes the camera count when a camera loses the keypoint), so the shares do not add up. Missing observations
explain the episodes in the legs and torso of subject 1 at 53.8-54.3 s and in the hands and arms of subject 2 at 62.7-63.4 s,
where a keypoint is absent for several frames, and little elsewhere.

- The picks are smoother than the strongest maxima of the same keypoint-frames: 2D residual (all / keypoint still) 12.5 / 4.4 px
  against 18.2 / 8.9 px for subject 1, 12.7 / 3.4 against 17.6 / 4.5 for subject 2, 9.8 / 2.2 against 14.9 / 2.8 for subject
  3. The windowed run's observations are as noisy as the prototype's (12.9 / 4.2 against 12.5 / 4.6 px for subject 1). The
  choice does not add noise; what remains is the noise of the keypoints themselves, a few pixels per frame, and jumps.
- The filter settings are the same in every run. Halving the process noise (0.3 to 0.15) for subject 1 lowers the RMS residual
  by 8-10% on the filtered states and by 7-27% on the smoothed states (hands 28.1 to 24.2 mm, feet 28.1 to 20.6 mm), leaves the
  number of episodes about the same and the distance to the edited run at 14.6 and 14.7 cm. The smoother lowers the residual
  of the windowed run by 7-15% (subject 1, hands 33.3 to 28.1 mm).

**Candidate fixes**, none chosen and none tried beyond the checks above:

- A sticky choice, which keeps the previous tracklet unless the score difference exceeds a margin. Evidence: jumps and tracklet
  switches have the largest gain in R^2 and are in most episodes; the choice has no term for continuity apart from the
  averaging of the support over the tracklet.
- Lower process noise, or a noise that depends on the joint group (hands and feet are the noisiest). Evidence: the check above.
- Smoothing of the keypoint tracks before they enter the filter, or an observation noise that grows with the distance of the
  observation from the track's own fit. Evidence: the jump event.
- Handling of the camera count: a keypoint that enters or leaves a camera changes the solution. Evidence: 12 of 18 episodes.
- For runs of missing observations (62.7-63.4 s of subject 2), prediction that holds the joint's velocity damped. Evidence:
  four episodes.
- Better hand input. Evidence: the hands jitter 1.5 to 2 times the edited runs.

**Five worst episodes of the windowed run that the edited run does not have** (ranked by the sum over the episode of the
series' ratio to the threshold minus one; groups at the same time are one row):

| Time (s) | Subject | Joints | Cause the measurement points to |
|---|---|---|---|
| 51.1-51.9 | 3 | forearm.L, upper_arm.L/R, head, neck, hands | jumps (7.7-8.7 times the trial rate), tracklet switches (6-7.6), outliers (4-6) |
| 55.0-56.1 | 2 | hand.L/R, forearm.R/L | missing observations (3.4), tracklet switches (3.2-5.2), jumps (2.9-4.9) |
| 62.7-63.4 | 2 | hand.R/L, forearm.R/L, upper_arm.R | runs of missing observations (6.2, 4.5 in the arms), missing observations (4.1) |
| 50.4-51.4 | 2 | hand.L/R, forearm.L, upper_arm.L/R | NIS spikes (2.7), outlier rejections (2.5), jumps (3.4) |
| 47.8-48.4 | 2 | hand.L/R | jumps (2.6), tracklet switches (2.5), outliers (2.4) |

The scores of the filter classes over the whole trial, and the jitter of the runs without candidates and with the per-step
assignment, need runs that are made separately (`trial_runs.py`); they are not part of this subsection until they exist.

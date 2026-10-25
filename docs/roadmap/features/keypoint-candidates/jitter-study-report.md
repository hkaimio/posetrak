# Arm and Hand Jitter Study

Can a "sticky" choice of keypoint candidates and lower filter process noise bring the arms and hands of the windowed
keypoint choice down to the smoothness of the per-step assignment, without losing its accuracy in the hard aikido falls?

The design is in [keypoint-candidates-design.md](keypoint-candidates-design.md) §10.1 (the sticky chain) and §10.10
(jitter, the grid and the remaining episodes). This report explains the study for a reader who knows Posetrak but not
this feature.

## Short answer

The smoothness and accuracy targets were met. The episode target was missed, and most of the gain comes from the
process noise, not from the sticky choice.

- **The sticky choice is built and correct.** It is a Viterbi chain over each keypoint's candidates. It is unit-tested
  and costs nothing measurable.
- **The jitter target was met.** The best setting brings the arms to 14.7 mm and the hands to 22.6 mm. The per-step
  assignment has 14.4 / 23.1 mm, and the baseline had 18.7 / 31.7 mm. Accuracy against the operator-edited runs is
  unchanged.
- **The episode target was missed.** 15 jitter episodes remain that the edited runs do not have. The target was at most
  10, and the per-step assignment has 8.
- **Process noise is the lever.** Lowering it from 0.3 to 0.15 alone takes arms plus hands from 50.3 to 40.2 mm and the
  feet from 23.0 to 15.6 mm. The sticky choice adds only about 3 mm more, and only at the low noise.
- **Defaults are unchanged.** The sticky choice ships off (`--kp-switch-penalty 0`). Process noise 0.15 is a
  recommendation, and the tracker default stays 0.3.
- **The remaining jitter is mostly jumps within a tracklet.** A penalty on switches between tracklets cannot remove them.

## 1. Context

Posetrak tracks a skeleton from multi-camera video in three steps:

1. A pose model (here SOMA-77) detects 2D body keypoints in each camera.
2. An Unscented Kalman Filter (UKF) in joint space fits the skeleton to those detections frame by frame.
3. An RTS smoother runs a backward pass over the result.

On hard footage the pose model's heatmap often has more than one plausible maximum for a keypoint. For example, it may
peak on the left wrist of one person and on the right wrist of their partner. The keypoint candidates feature keeps
several maxima per keypoint instead of only the strongest. It then decides which one is right before the filter sees it.

The windowed choice makes that decision over a time window. It was more accurate than the alternatives in the hard parts
of the trial, but its output jittered: the arms and hands had 1.5-2 times the frame-to-frame noise of tracking from
hand-corrected keypoints.

## 2. Terms

**Keypoint candidate.** One local maximum of the pose model's heatmap for a keypoint in one camera frame. A keypoint can
have several candidates per frame, ranked by heatmap peak.

**Tracklet.** A chain of candidates of the same keypoint in the same camera, linked across consecutive frames because
they are close to each other. It is a short 2D track, and it is the unit a person reviewing the data accepts or rejects.

**Per-step assignment (`'step'`, the default).** Inside the filter, at each frame, it picks the candidate closest to
where the filter predicts the keypoint (Mahalanobis distance). It follows the filter's own track, so it is smooth. When
the track has already drifted onto the wrong person, it follows the error.

**Windowed choice (`'window'`).** A phase that runs before the filter and picks one candidate per keypoint, camera and
frame. Each candidate is scored by:

- cross-camera support: how well it agrees geometrically with the other cameras' picks, averaged over its tracklet
  within ±1 s;
- a multi-camera consensus flag;
- its heatmap peak.

The choice iterates in three rounds and can use the future as well as the past. Without the sticky chain, each frame's
pick is the best-scoring candidate of that frame on its own. Nothing then discourages switching between neighbouring
candidates from frame to frame.

**Strict filter (body only).** It uses the person's segmentation masks. It removes a candidate whose tracklet, within
±10 frames, has fewer than half of its detections on that person's mask. It applies to the body keypoints only, because
the hand and foot masks are thin and often wrong in contact. Option: `--kp-strict-filter`.

**The ace2 pro outside the choice.** One of the five cameras has a less precise calibration (a 12 px consensus radius).
It takes no part in the windowed choice and keeps its strongest detection. Option: `--kp-skip-camera`.

**Sticky choice.** For each person, camera and keypoint, it takes the chain of picks over the frames that maximises the
sum of the scores, minus a switch penalty for every change between consecutive frames.

- A Viterbi search finds the chain exactly. It sees the past and the future alike, so the result does not depend on
  where the window starts.
- A change is either a different tracklet, or a move of more than the jump threshold between frames. The threshold
  defaults to 0.04 box heights.
- A frame with no candidate breaks the chain at no cost.
- A tracklet the operator has assigned stays the pick, whatever the penalty.
- The chain is applied in the last round only. The earlier rounds' picks are the evidence for the next round's support
  scores, so they are left as they are.

**Switch penalty.** The price of a change, in score units (scores are roughly 0-2). 0 turns the sticky choice off.
Options: `--kp-switch-penalty`, `--kp-jump-box`.

**Process noise.** The filter's assumed random acceleration of the pose (`--process-noise-std`, default 0.3). A lower
value trusts the motion model more and the per-frame observations less, which gives a smoother output. The risk is lag
in fast motion.

**Ukemi events.** The aikido falls and rolls in the trial: fast motion, unusual poses, and people occluding each other.
This is where the candidates matter and where the trackers differ most.

**Operator-edited runs.** Tracker runs on keypoints that an operator corrected by hand over the whole trial, one per
subject. They are the best available reference for the whole trial, but not ground truth.

**Ranges, range 3, hard windows.** The trial (43.5-66.44 s) is split into ranges. Range 3 (51.7-56.8 s, subjects 1 and
2) is the hardest. The hard windows are its two shortest and most occluded stretches.

**Label-fed reference (range 3).** A tracker run fed with hand-labelled keypoints over range 3, the closest thing to
ground truth. The label-fed figures are the mean body-joint distance to it, given as a pair: over range 3 from 52.0 s /
inside the hard windows.

## 3. Metrics

**Jitter RMS (mm), per joint group.**

1. Each joint's 3D position is sampled at 60 Hz.
2. A Savitzky-Golay fit (order 3, 0.2 s) is subtracted. The high-frequency residual that remains is the noise.
3. The residual is corrected for the part of white noise that the fit absorbs, so its RMS equals the noise level.

The figure is reported per joint group (torso, arms, hands, legs, feet) and averaged over the three subjects. The sum
"arms + hands" is the ranking figure.

**Jitter episodes not in the edited runs.**

1. A 0.5 s sliding RMS is computed per joint group.
2. An episode is a stretch where that RMS stays above the 99th percentile of the same series in the edited runs.
3. Fast genuine motion also crosses the threshold. An episode that the edited run also has, in the same subject and
   group, is counted as motion. Only the remaining episodes are counted as jitter.

The episode count measures visible glitches, and the RMS measures the overall level.

**Distance to the edited runs (cm).** The mean 3D distance of the body joints to the edited run, pooled over the
subjects. It is reported over the whole trial, inside and outside the ukemi events, and per range.

**Event enrichment (causes).** For each episode, the rate of each event inside it is divided by that event's rate over
the whole trial. A ratio of 3 means the event is three times as common inside the episode. The events are:

- a tracklet switch of the pick;
- a jump of the observation away from its own track's fit;
- a change in the number of cameras that see the keypoint;
- an outlier rejection;
- a missing observation, or a run of missing observations;
- a NIS spike.

**Selection rule.** The rule was fixed before the grid ran.

- A setting is admissible if its distance to the edited runs is no more than 0.2 cm worse than the filtered baseline's,
  both over the whole trial and inside the ukemi events.
- Among the admissible settings, the lowest arms + hands jitter wins.

**Target.**

- Arms at or below 14-16 mm and hands at or below 21-25 mm, the per-step assignment's level.
- At most 10 extra episodes.
- Accuracy within the rule's tolerance.

## 4. Setup

- **Capture.** Five cameras and three people practising aikido, 43.5-66.44 s, all three subjects.
- **Basis.**
  - calibration `b416cd21` and the ace2 tracker configuration;
  - boxes from the segmentation masks, with the mask gate;
  - the windowed choice, with the ace2 pro outside it;
  - the optimized build, with RTS smoothing on.
- **Grid.** The switch penalty (0, 0.2, 0.5 and 1.0) crossed with the process noise (0.3, 0.2 and 0.15), all with the
  body-only strict filter. For comparison, the chosen setting was also run without the filter, and the per-step
  assignment was run as well.

## 5. Results

### Jitter

RMS of the high-frequency residual in mm, mean of three subjects, and the number of jitter episodes not in the edited
runs. Lower is smoother.

| Penalty / noise | Torso | Arms | Hands | Legs | Feet | Arms + hands | Episodes |
|---|---|---|---|---|---|---|---|
| 0 / 0.3 (filtered baseline) | 12.9 | 18.7 | 31.7 | 12.5 | 23.0 | 50.3 | 18 |
| 0.2 / 0.3 | 12.7 | 18.4 | 29.4 | 12.3 | 22.5 | 47.8 | 16 |
| 0.5 / 0.3 | 13.2 | 18.9 | 29.9 | 12.7 | 22.9 | 48.8 | 18 |
| 1.0 / 0.3 | 13.2 | 19.2 | 31.9 | 12.7 | 23.0 | 51.1 | 20 |
| 0 / 0.2 | 11.7 | 16.0 | 24.3 | 11.2 | 18.3 | 40.4 | 20 |
| 0.2 / 0.2 | 11.5 | 15.9 | 24.6 | 10.9 | 18.2 | 40.5 | 16 |
| 0.5 / 0.2 | 11.7 | 16.1 | 25.9 | 11.1 | 18.0 | 42.0 | 15 |
| 1.0 / 0.2 | 12.1 | 16.9 | 26.8 | 11.3 | 18.3 | 43.8 | 12 |
| 0 / 0.15 | 11.3 | 15.5 | 24.8 | 10.4 | 15.6 | 40.2 | 14 |
| 0.2 / 0.15 | 11.1 | 14.9 | 23.5 | 10.1 | 15.5 | 38.5 | 12 |
| 0.5 / 0.15 | 11.5 | 15.3 | 22.6 | 10.3 | 15.5 | 37.9 | 17 |
| **1.0 / 0.15 (chosen)** | **11.4** | **14.7** | **22.6** | **10.4** | **15.5** | **37.3** | **15** |
| 1.0 / 0.15, no strict filter | 12.5 | 15.5 | 23.1 | 11.3 | 17.4 | 38.6 | 14 |
| Unfiltered baseline, 0 / 0.3 | 13.6 | 18.6 | 30.1 | 12.8 | 23.7 | 48.7 | 18 |
| Per-step assignment | 10.5 | 14.4 | 23.1 | 10.5 | 19.5 | 37.6 | 8 |

### Accuracy

Mean body-joint distance in cm. The first four columns are against the operator-edited runs. The label-fed columns are
range 3 / hard windows. Lower is better. All twelve filtered cells pass the rule's 0.2 cm tolerance: 12.46-12.56 cm over
the whole trial and 16.68-17.24 cm inside the ukemi events.

| Penalty / noise | Whole trial | Range 3 | Inside ukemi | Outside ukemi | Label-fed S1 | Label-fed S2 |
|---|---|---|---|---|---|---|
| 0 / 0.3 (filtered baseline) | 12.53 | 13.86 | 17.06 | 11.87 | 7.61 / 9.94 | 3.43 / 4.63 |
| 0 / 0.15 | 12.50 | 13.73 | 16.80 | 11.88 | 7.68 / 9.85 | 3.69 / 4.75 |
| 0.2 / 0.15 | 12.46 | 13.80 | 16.68 | 11.85 | 7.42 / 9.48 | 3.80 / 4.83 |
| 0.5 / 0.15 | 12.48 | 13.65 | 16.76 | 11.86 | 7.24 / 9.30 | 3.91 / 4.93 |
| **1.0 / 0.15 (chosen)** | **12.50** | **13.74** | **16.97** | **11.85** | **7.28 / 9.36** | **3.98 / 5.13** |
| 1.0 / 0.15, no strict filter | 12.99 | 15.13 | 19.49 | 12.06 | 6.61 / 8.08 | 3.74 / 4.77 |
| Unfiltered baseline, 0 / 0.3 | 12.71 | 14.29 | 19.17 | 11.78 | 6.75 / 8.54 | 3.73 / 4.82 |
| Per-step assignment | 12.69 | 15.00 | 18.25 | 11.89 | 8.25 / 10.88 | 4.16 / 5.77 |

### What the numbers say

- **Process noise does most of the work.** At penalty 0, going from 0.3 to 0.15:
  - lowers arms + hands from 50.3 to 40.2 mm and the feet from 23.0 to 15.6 mm;
  - lowers the torso and legs by 1.6-2.1 mm;
  - leaves the accuracy against the edited runs the same (12.50 against 12.53 cm), and slightly better inside the falls
    (16.80 against 17.06 cm).

  There is no sign of lag on this trial.
- **The sticky choice helps a little, and only at low noise.**
  - At noise 0.15 it takes arms + hands from 40.2 to 37.3 mm (penalty 1.0).
  - At noise 0.3 it does nothing consistent (47.8-51.1 mm).
  - It does what it is meant to do: the share of keypoint-frames whose pick switches tracklet falls by a fifth to a
    third (subject 1: 9.96% to 7.40%). Most of the jitter is not tracklet switches, though.
- **The jitter is mostly jumps inside a tracklet.** In a pilot on subject 2, the 2D residual stayed at 16.5 px for every
  penalty, and changing the jump threshold (0.02-0.1 box heights) made no difference. A tracklet can link two
  neighbouring heatmap maxima and hop between them, and a penalty on tracklet switches does not see that.
- **Episode counts are noisy.** At noise 0.15 the four penalties give 14, 12, 17 and 15 episodes, in no order.
  Differences of 2-3 episodes between cells are within the spread. No cell reaches 10.
- **The label-fed range 3 does not improve.** At the chosen setting, subject 1 is 0.2-0.4 cm better and subject 2 is
  0.3-0.5 cm worse (5.13 against 4.63 cm in the hard windows). Both stay within the spread of the baselines. The
  selection rule does not look at these figures.

## 6. Remaining episodes with the chosen setting

The chosen setting leaves 15 episodes:

- by joint group: hands 6, arms 4, torso 3, legs 2;
- by subject: subject 1 4, subject 2 8, subject 3 3.

Over the 14 episodes that are longer than one sample, the events are enriched as follows:

| Event | Mean enrichment | Episodes at 1.5× or more |
|---|---|---|
| Tracklet switch of the pick | 3.9 | 9 |
| Jump of the observation from its track | 3.4 | 12 |
| Change in camera count | 2.4 | 12 |
| Outlier rejection | 2.2 | 8 |
| Missing observation / run of missing | 2.0 / 2.0 | 7 / 7 |
| NIS spike | 1.4 | 5 |

- **Jumps and camera-count changes are the common companions.** Each is enriched in 12 of the 14 episodes.
- **Tracklet switches are the sharpest where they occur.**
  - 9.5-12.5× the trial rate in subject 3's torso and arms at 51.3-51.8 s;
  - 5-6.6× in subject 2's arms at 55.1-55.5 s and 62.8-63.2 s.
- **Subject 2's five hand episodes** combine jumps and outliers.

Camera-count changes do not dominate, so no fix for them is built. A possible fix would ramp the measurement weight of a
newly visible keypoint in over about five frames, and out again before the keypoint disappears.

## 7. Other findings

- **Per-group process noise is not available for the arms and hands.**
  - The hierarchical configuration has per-stage noise overrides. In this configuration, though, the child stages are
    only the two hands' fingers. The arms, wrists and torso share the parent stage.
  - A per-joint multiplier of the base noise in the process model would need one configuration field, its database
    columns, and a change to `ConstantVelocityModel::get_process_noise`.
- **The strict filter is a trade-off between the two references.** Without the filter, the chosen setting is:
  - 2.5 cm worse inside the ukemi events against the edited runs (19.49 against 16.97 cm);
  - 0.7-1.3 cm better on subject 1's label-fed range 3.
- **The sticky chain costs nothing measurable.**
  - On 5 s with five cameras, the choice itself takes 0.3 s with the penalty off and 0.4 s with it on. The whole phase
    takes 26 s, mostly mask decoding.
  - Whole-trial runs took 600 s and 591 s, the same within the spread between runs.

## 8. Verification and caveats

- **Code.**
  - The Viterbi search segments at frames without a candidate and backtracks per segment.
  - An assigned tracklet is the only option of its frame.
  - The settings flow through `tracker_configs`, `SessionReader` and `track run`.
  - Five unit tests cover the four required cases, plus gaps and jumps.
- **Tests.** C++: 482 test cases, 473 passed and 9 failed as expected. Python: 2,708 passed, 20 skipped, 3 xfailed.
- **Caveats.**
  - The figures come from one trial with three subjects.
  - The jitter metric rewards smoothing by construction, so lag would show only as lost accuracy. On this trial
    accuracy was not lost, even inside the falls.
  - The edited runs themselves were tracked at noise 0.3.
- **Visual check.** The recommended setting was exported and retargeted in iClone. The motion looks clearly better than
  in earlier runs on this capture.

## 9. Open decisions and next steps

- **Default process noise.** Changing the tracker default from 0.3 to 0.15 for SOMA runs is the cheapest large gain
  measured. It stays an open item until more captures have been tracked and checked.
- **Jumps within a tracklet** are the main remaining cause. Candidate fixes:
  - split tracklets where they hop between neighbouring maxima;
  - make the observation noise grow with the distance from the tracklet's own fit;
  - smooth the keypoint tracks before the filter.
- **Camera-count changes.** The weight ramp of section 6, if they turn out to matter on other footage.

## Appendix: run ids

| Cell | Subject 1 | Subject 2 | Subject 3 |
|---|---|---|---|
| Unfiltered 0 / 0.3 | `e8e1bb80` | `5dc83449` | `ad3708a3` |
| Filtered 0 / 0.3 | `a5b0cbad` | `f92edc37` | `c912d125` |
| 0 / 0.2 | `8039e21c` | `4f7b0714` | `9c2052c9` |
| 0 / 0.15 | `fd7a4d5c` | `853c345d` | `80c35358` |
| 0.2 / 0.3 | `3427fcec` | `5df35710` | `89ba75c9` |
| 0.2 / 0.2 | `80ed63f8` | `ffe8d839` | `01c0f09f` |
| 0.2 / 0.15 | `3acb0560` | `badc8ece` | `b25f1e50` |
| 0.5 / 0.3 | `9ffa6af7` | `c87fbd44` | `fbe857c1` |
| 0.5 / 0.2 | `4fe058d0` | `bad210b0` | `1363e26e` |
| 0.5 / 0.15 | `98655c37` | `f7313d12` | `7be9d7b6` |
| 1.0 / 0.3 | `663b2a56` | `a5838097` | `a7bf107e` |
| 1.0 / 0.2 | `9ab6b6d0` | `8f43f525` | `8cd1097d` |
| **1.0 / 0.15 (chosen)** | `0384a260` | `2c23a871` | `fcd04cc2` |
| Unfiltered 1.0 / 0.15 | `43b5fa24` | `5d41b840` | `46eb79e9` |
| Per-step assignment | `581f60b7` | `ae9a2ac2` | `4cd78935` |
| Operator-edited (reference) | `6972748b` | `47cfeb6c` | `80d7fab3` |
| Label-fed reference, range 3 | `855e1e19` | `bc3886f9` | - |

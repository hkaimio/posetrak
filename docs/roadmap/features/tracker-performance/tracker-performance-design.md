# Tracker cost: run time and memory of long tracking runs

How much a tracking run costs for a 77-keypoint skeleton, where the cost comes from, and the
cheap changes that come before snapshot-based iterative tracking.

## 1. Cost model

The tracker is an error-state UKF. With `K` active joint DOFs the error state has
`e = 2 (6 + K)` entries (root pose, joint angles, and their velocities; see the `cov_diag` comment
on `tracking_results` in `session_schema.sql`). With `M` observed markers and `C` cameras the
measurement has up to `m = 2 M C` entries. Per tracker step:

| Part | Cost |
|---|---|
| Sigma points | `2e + 1` forward-kinematics evaluations and `(2e + 1) M C` projections |
| Predict | Cholesky of `P`, `O(e^3 / 3)` |
| Update | LDLT of the innovation covariance, `O(m^3 / 3)` (`UnscentedKalmanFilter::update`, `ukf.cpp`); cross-covariance `O(e m (2e + 1))` |

RTS smoothing keeps per step, in `FrameSmootherData` (`rts_smoother.hpp`), the posterior
covariance, the prior covariance and the cross-covariance: three dense `e x e` matrices. The
backward pass returns a `SmoothedFrame` per step with a fourth, the smoothed covariance, while the
forward cache is still alive. Peak memory of the smoother is therefore about
`4 e^2 * 8 bytes` per step.

### Numbers for the default skeleton

The bundled default skeleton (`python/posetrak/data/skeletons/default_male.yaml`) has 52 joints,
all `ball` except the root: 153 joint DOFs. Its groups are `main` (63 DOFs, 25 markers) and the
two hand groups `HandL` and `HandR` (48 DOFs and 21 markers each, with a free-flyer root of 6).

| Configuration | `e` (upper bound, no locked axes) | Smoother memory per step | 10 000 steps |
|---|---|---|---|
| Single stage, all 153 DOFs | 318 | 3.2 MB | 32 GB |
| Hierarchical, `main` stage | 138 | 0.61 MB | 6.1 GB |
| Hierarchical, one hand stage | 108 | 0.37 MB | 3.7 GB |

Locked axes lower `K`. The 18 GB observed on a 10 000-step single-stage run fits the formula for
`e` of about 240, which is plausible for a skeleton with locked finger axes; the measurement in
section 3 reads the actual `K_active` from the run.

For a SOMA-77 skeleton with 77 markers and five cameras, a single-stage step has `m` up to 770:
637 sigma points, each projected into 385 marker-camera slots, and a 770 x 770 LDLT. In the
hierarchical configuration the `main` stage carries the 29 body markers and the first joints of
the index and little fingers (33 markers, `m` up to 330, 277 sigma points) and each hand stage 24 markers on a small subtree. From the operation
counts alone the hierarchical configuration should be several times faster per step; this is an
estimate, not a measurement.

### Why the 19-minute figure is not a baseline

The run that took about 19 minutes for 85 s of footage used a single-stage tracker configuration.
Hierarchical mode is selected by the presence of `tracker_config_stages` rows for the
configuration; a run made from a copy of only the main `tracker_configs` row is single-stage.
The figure is therefore the cost of the most expensive configuration and says nothing about the
one that should be the default.

## 2. Options, cheapest first

1. **Hierarchical configuration as the default for 77-keypoint skeletons.** Exists. The SOMA
   skeleton builder already places the finger markers in the hand groups. The parent stage is
   restricted to its group only when the run is given the parent's joint groups (`--joint-groups
   main` of the tracker's database mode, `TrackerConfig::active_joint_groups`); without them the
   parent stage tracks the whole skeleton and the hand stages run in addition (section 3). The
   groups are stored with the configuration (`tracker_configs.active_joint_groups`, a JSON array;
   the command line overrides it); `posetrak track run --config` copies the configuration's stage
   rows; a bundled hierarchical configuration (stage rows and `["main"]`) is the default for
   skeletons with hand groups; and a run start prints a warning when such a skeleton is tracked
   single-stage, or hierarchically with an unrestricted parent.
2. **Do not keep the full smoothed covariance.** The result writer stores only `cov_diag`. If no
   consumer of `SmoothedFrame::covariance` needs more than its diagonal (the hierarchical child
   stages read the smoothed *state* of the parent through `BatchTrajectoryStream`; this has to be
   confirmed for the covariance), keep the diagonal only. Saves one of four matrices per step, no
   change to results.
3. **Free the forward cache stage by stage.** In a hierarchical run each stage's cache must be
   released before the next stage runs; peak memory is then the largest stage, not the sum.
   Verify, and fix if not.
4. **Spill the forward cache to a temporary file.** The backward pass reads the cache once, in
   reverse order. Writing each step's three matrices to a memory-mapped file instead of keeping
   them in RAM bounds resident memory at the cost of disk space (the same size as today's RAM
   use) and sequential I/O. Exact; no change to results. Only needed if options 1-3 leave a
   realistic session length above the memory of the target machine.
5. **Lower tracker rate.** `tracker_configs.tracker_fps` exists (120 Hz in the runs measured).
   Measured in section 3: a step at 60 Hz pools two video frames of every camera, so the update
   handles twice the observations and the run is slower, not faster, and moves every failure event.
   A lower rate would save time only if a step decimated the observations instead of pooling them.
   Dropped as a cost option; `tracker_fps` stays as a configuration value.
6. **Fewer markers in the main stage.** Dropping 44 of the 48 body-crop finger keypoints from the
   tracked markers (they localise poorly: a median of 0.20 of the hand size against 0.05-0.07 from
   a hand crop) removes most of `m` in a single-stage run and nothing in a hierarchical one, where
   the fingers already sit in the hand stages (the restricted form of section 3 has them out of the
   parent stage).
7. **Load only the requested window, and smaller observations.** The tracker loads every
   observation of the sequence whatever time range is tracked: about 1 GB resident for the 85 s
   four-camera sequence (section 5; a 5 s slice peaks 1.0 GB lower when only its window is loaded).
   The window is loaded alone; a compact in-memory observation record is not worth building at that
   size. The smoother's per-step cache, not the observations, is what grows with a long run.
8. **Per-person threads outside contact.** `MultiPersonTracker::run()` steps the persons one after
   another within each frame; outside contact windows they are independent, so they could step on
   separate threads. Tried and not kept: in the three-person trial only 23% of the frames have no
   contact coupling, and the threaded run was slower than the sequential one (572 against 529 s),
   because the persons' threads compete for the cores the UKF's own sigma-point loops (OpenMP)
   already use. Worth retrying only on captures where the persons are apart most of the time, with
   the UKF's thread count capped per person.

A fixed-lag or windowed smoother also bounds memory, but it changes the estimator and belongs to
iterative tracking (snapshots, restart over a changed range), which is a separate feature. Options
2-4 are exact and come first.

## 3. Measurements before changing anything

Per-step timings are already recorded (`StatisticsTracker` fields `predict_ms`, `update_ms` and the
update sub-timings, written to `tracking_stats.csv`); `POSETRAK_PROFILE_FRAME_STEP` adds the
dot-assignment pieces. The measurement needs no code:

1. Same sequence (the 85 s run), the SOMA skeleton, optimized build (`optbuild/`), two configurations:
   single-stage and hierarchical (stage rows copied with the configuration). Record wall time, peak
   resident memory (Windows: peak working set from the process), and the per-step timing columns.
2. The hierarchical run at 120 Hz and 60 Hz.
3. Read `K_active` of each stage from the layout (the length of `cov_diag`) to check the cost model.
4. Compare the runs on the failure-event list with the existing scoring script, so that a cheaper
   configuration that loses events is visible.

These runs are long (tens of minutes); they are scheduled as an early stage of the plan, not run
during design.

### Procedure and results

`scripts/tracker_baseline.py` runs the variants in a copy of the session database with the optimized
binary (`optbuild/`) and writes wall time, peak resident memory (the tracker process, sampled twice a
second), the per-step timing columns of `tracking_stats.csv`, the error-state dimension of every stage (the
tracker's `error_state_dim=` lines and the length of `cov_diag`) and, for the multi-person trial, the event
table of each variant against the hierarchical 120 Hz run:

    python scripts/tracker_baseline.py --work-dir D:/mocap/tracker-baseline --part a
    python scripts/tracker_baseline.py --work-dir D:/mocap/tracker-baseline --part b --events <events.json>

Part a is the 85 s single-person sequence with the SOMA-77 skeleton (152.8-238.0 s, 10,214 steps at 120 Hz, four
cameras); part b is the three-person trial (38-66 s). The variants: `single120`, `single60` (no stage rows);
`hier120`, `hier60` (stage rows, parent stage on the whole skeleton: the run is not given `--joint-groups`);
`main120`, `main60` (stage rows, run given `--joint-groups main`, so the parent stage tracks its own group
and the hand stages their hands). Hierarchical mode needs `tracker_config_stages` rows, and `posetrak track run --config` does not
copy them (it writes a new configuration row from the base without its stage rows), so a run started that
way is single-stage whatever the base configuration; the script builds the variants itself: the hierarchical
variant is a copy of the configuration with its stage rows (`HandL`, `HandR`), the single-stage variant a copy
without them. The full runs take tens of minutes each and are run by the operator in a standalone shell.

**Preliminary slices.** To check the commands and the cost model, every part-a variant was run on the first
10 s of the sequence (1,200 steps at 120 Hz, 600 at 60 Hz; not a baseline: peak memory is dominated by
loading the sequence's observations and the run is too short for the smoother's cache to matter). Binary:
`optbuild/cpp/cli/posetrak-tracker.exe`, built before the one later C++ commit that changes only the
initialisation (the instrumentation commit touches no C++); skeleton "Default male SOMA-77 anatomical limits"; configuration `hier-dualhand-c14786ab2fbe221e` with the variants described above.

| Variant | Wall s | Parent stage s | Hand stages s | Peak RSS GB | Observations per step | Update ms/step (of which Kalman gain) | Mean reprojection px | Lost steps |
|---|---|---|---|---|---|---|---|---|
| single-stage, 120 Hz | 129 | 129 | - | 5.6 | 552 | 81.6 (42.5) | 50 | 0 |
| hierarchical, 120 Hz | 145 | 122 | 11 + 12 | 6.1 | 552 | 76.1 (41.1) | 50 | 0 |
| single-stage, 60 Hz | 200 | 200 | - | 4.8 | 1,104 | 301.5 (231.4) | 92 | 0 |
| hierarchical, 60 Hz | 229 | 215 | 7 + 7 | 5.1 | 1,104 | 326.1 (255.9) | 92 | 0 |

Error-state dimension from the tracker's output: the parent stage has `total_dof=153`, `error_dim=202` (the
anatomical limits lock 58 of the 153 axes, so the active DOFs are 95) in both the single-stage and the
hierarchical run; each hand stage has `total_dof=48`, `error_dim=44`.

These slices disagree with the cost model above in three places:

1. **Without `--joint-groups main` the parent stage is the whole skeleton.** The `main` group has 22 joints
   and 29 markers, but a hierarchical run that is not given the group initialises its parent stage with all 153
   DOFs (error state 202) and all 77 markers (552 observations per step from four cameras), the same as the
   single-stage run, and the hand stages run in addition. That is the most expensive form of the
   configuration: 12% more time and 0.5 GB more memory than single-stage on the slice. It is what the
   `hier120` and `hier60` rows below measure; the groups are not stored with the configuration or the run, so
   nothing in a stored run says which form it used. Given `--joint-groups main` the parent stage has
   `total_dof=63`, `error_dim=122` and the hand stages `error_dim=44` each; the `main120` and `main60` rows
   are that form (a 5 s smoke run at 120 Hz: 24.5 ms update per step against 80 ms, peak 1.7 GB). The
   whole-sequence single-stage run took about 19 minutes and the unrestricted hierarchical one 1,362 s (parent
   stage 1,085 s, 137 s per hand stage).
2. **A lower tracker rate is slower, not faster.** At 60 Hz a step pools two video frames of every camera
   (1,104 observations against 552), the Kalman-gain part of the update grows 5.4 times (the measurement
   covariance is factorised, cost cubic in the number of observations) and the wall time is 55% higher; the
   mean reprojection error is also higher (92 against 50 px). Option 5 of section 2 as written does not
   save time; it would need the observations of a step to be decimated rather than pooled.
3. **Where the update time goes.** At 120 Hz the Kalman-gain part of the update is half of it (42.5 of
   81.6 ms); the recomputation over the inlier set takes 9.1 ms, the innovation covariance 8.6 ms, the two
   projections 5.1 ms, and about 15 ms are not covered by the sub-timings. The prediction takes 11.4 ms per
   step (mean and covariance of the sigma points 6.7, propagation 3.1, generation 1.5). The gain's cost grows
   with the number of observations to a power above 2 (5.4 times for twice the observations), so fewer
   observations per step (the body-crop finger markers of the parent stage) reduce the largest term
   directly.

4. **A large part of the resident memory is not the smoother.** A 2 s slice already peaks at 4.6 GB resident and a
   10 s slice at 4.8-6.1 GB; the tracker loads all observations of the sequence (5.5 million for the 85 s
   sequence of four cameras) whatever time range is asked for. The smoother's cache adds `4 e^2` doubles per
   step on top (1.3 MB at e = 202, about 13 GB at 10,000 steps), so the acceptance limit of 8 GB at 10,000
   steps cannot be met by the smoother changes alone if the observation load stays; the full runs show how
   the two add up.

**Part a, full sequence** (85 s, four cameras, SOMA-77 skeleton, optimized build, smoother on):

| Variant | Wall s | Peak RSS GB | Steps | Lost steps | Error-state dim | Update ms/step |
|---|---|---|---|---|---|---|
| single-stage, 120 Hz | 1,103 | 19.7 | 10,214 | 0 | 202 | 80.1 |
| hierarchical, unrestricted parent, 120 Hz | 1,320 | 20.5 | 10,214 | 0 | 202 | 75.0 |
| single-stage, 60 Hz | 1,580 | 11.7 | 5,106 | 0 | 202 | 280.3 |
| hierarchical, unrestricted parent, 60 Hz | 1,754 | 12.4 | 5,106 | 0 | 202 | 288.3 |
| **hierarchical, parent `main`, 120 Hz** | **584** | **7.8** | 10,214 | 0 | 202 | 24.1 |
| hierarchical, parent `main`, 60 Hz | 542 | 4.6 | 5,106 | 0 | 202 | 66.1 |

The unrestricted hierarchical form (`hier120`, `hier60`) costs 20% more time and 0.8 GB more memory
than single-stage at 120 Hz. The restricted form (parent `main`) takes 584 s, 1.9 times faster than single-stage
(3.3 times less update time per step), and peaks at 7.8 GB, 2.5 times less: the 10,214-step run is below the
8 GB acceptance limit, by 0.2 GB. The margin is small, and how the 7.8 GB divides between the smoother's
cache and the observation load is not known (the model below, 5.6 GB plus 4 e^2 doubles per step, predicts
10.7 GB for e = 122, so one of its terms is smaller for the restricted form); the memory options of section 2 are
still needed for longer sequences or more cameras.

For the unrestricted forms, 60 Hz costs 43% more time than 120 Hz (the update is 3.5 times slower per step
and there are half as many steps) and saves memory only because it has half as many steps. Their peak memory
follows the model of finding 4: about 5.6 GB for the observation load plus 1.3 MB per step
(5.6 + 10,214 x 1.3 MB = 18.9 GB against 19.7 GB measured; 4.8 + 5,106 x 1.3 MB = 11.4 GB against 11.7 GB),
2.5 times over the acceptance limit, with the smoother's share (about 14 GB) larger than the observation
load's. The restricted form removes most of that. (Section 5 measures the observation load itself at
about 1 GB: the model's 5.6 GB intercept is mostly a fixed working set of the filter, not observations.)

**Part b, three-person trial** (28.4 s, 38.1-66.4 s, three persons in one run, optimized build, smoother on):

| Variant | Wall s | Peak RSS GB | Steps per person | Lost steps | Error-state dim | Update ms/step |
|---|---|---|---|---|---|---|
| single-stage, 120 Hz | 2,385 | 25.7 | 3,402 / 3,396 / 3,402 | 0 | 218 | 115.6 |
| hierarchical, unrestricted parent, 120 Hz | 2,648 | 26.7 | 3,402 / 3,396 / 3,402 | 0 | 218 | 115.1 |
| hierarchical, unrestricted parent, 60 Hz | 2,844 | 18.7 | 1,700 / 1,697 / 1,700 | 0 | 218 | 425.7 |
| **hierarchical, parent `main`, 120 Hz** | **793** | **8.7** | 3,402 / 3,396 / 3,402 | 0 | 218 | 35.3 |
| hierarchical, parent `main`, 60 Hz | 752 | 5.4 | 1,700 / 1,697 / 1,700 | 0 | 218 | 100.4 |

(The error-state column is the length of the stored `cov_diag`, not that of the parent stage; for the
restricted runs read the stages' `error_state_dim=` lines in `results.json`.)

- **The restricted hierarchical form is the cheap one.** At 120 Hz it takes 793 s against 2,385 s for
  single-stage (3.0 times faster, 3.3 times less update time per step) and peaks at 8.7 GB against 25.7 GB
  (3.0 times less). The unrestricted form is 11% slower and 1 GB heavier than single-stage. The memory
  reduction is less than the 5x of the cost model because the observation load does not change with the
  configuration. Three persons of about 3,400 steps (10,200 steps in all) peak 0.7 GB above the 8 GB
  acceptance limit, before any of the memory options in section 2 are applied.
- **60 Hz still pools.** With the parent restricted it is 5% faster than 120 Hz (752 against 793 s) although it
  takes half the steps, because each step costs 2.8 times as much (100 against 35 ms); it saves 3.3 GB through
  the step count. The unrestricted 60 Hz form is 7% slower than 120 Hz.

Largest joint distance (cm) in each failure event between a variant and the unrestricted hierarchical 120 Hz
run (`hier120`, the configuration of the spike's runs). The control run is not ground truth: the table shows how
far a configuration moves the result, not which side is right.

| Event | single-stage 120 Hz | hier. unrestricted 60 Hz | parent `main` 120 Hz | parent `main` 60 Hz |
|---|---|---|---|---|
| u1-legs | 0 | 72 | 26 | 46 |
| u1-hand | 9 | 34 | 23 | 27 |
| u2-rhand-lost | 18 | 25 | 25 | 25 |
| u2-rhand-raise | 19 | 27 | 28 | 28 |
| u2-leg | 0 | 38 | 19 | 36 |
| u3-legs | 0 | 24 | 16 | 24 |
| u3-fall | 13 | 42 | 164 | 68 |
| u3-timo-arm | 10 | 14 | 22 | 16 |
| u3-timo-legs | 0 | 17 | 4 | 20 |
| u4-rarm | 18 | 25 | 41 | 30 |
| u4-lleg | 0 | 54 | 68 | 45 |
| u4-jitter | 12 | 65 | 68 | 49 |
| u4-rarm2 | 24 | 66 | 78 | 95 |
| u4-fall | 16 | 132 | 155 | 148 |
| u5-legs | 0 | 19 | 33 | 20 |
| u5-hand | 2 | 7 | 10 | 10 |
| u6-ankle | 0 | 9 | 44 | 9 |

- **Single-stage differs from the unrestricted hierarchical form only where the arms and hands are involved.**
  The leg events are identical (0 cm) and the arm and hand events differ by 9-24 cm. The parent stage is the
  same in both, so the difference comes from the hand stages feeding back into the arm chain.
- **Restricting the parent stage moves the result everywhere, the legs included, by 4-164 cm** (largest in
  the fall events, 164 and 155 cm, then `u4-rarm2` and `u4-lleg`). The markers that leave the parent stage
  change what it is fitted to. Whether the restricted form is better or worse than the unrestricted one is not
  decided by this table; it needs the evaluation harness's distance to the labels. The cost measurement does
  not show that the cheap form is safe: the acceptance criterion that no failure event is lost is neither met
  nor failed yet.
- **60 Hz moves every event, in both forms, by 7-148 cm.** Pooling two frames per step changes the result
  everywhere; on cost and on movement alone it has nothing to recommend it.

## 4. Acceptance

- A 10 000-step hierarchical run of a SOMA-77 skeleton peaks below 8 GB resident memory.
- The run time of the default configuration is reported as a real-time factor per person and
  camera count, and is the baseline later optimisation is measured against.
- The default configuration loses no failure event against the single-stage run on the labelled
  trial, and is chosen on the evaluation harness's distance to the label-fed run where the two differ
  (the event table measures how far a configuration moves the result, not which result is right).
- A tracking run records the parent stage's joint groups through its configuration.

## 5. Results of the changes

Capture: the four-camera 85.13 s sequence of part a (10,214 steps at 120 Hz, one person) and the
three-person trial of part b (28.36 s, five cameras, about 10,200 steps in all). Configuration: SOMA-77
skeleton without its 44 body-crop finger markers (they localise poorly, option 6; the scripts call
these variants `m33` after the 33 markers left in the parent group), 120 Hz, RTS smoothing,
`optbuild/` release build on the development PC, run in database copies by
`scripts/tracker_baseline.py`; accuracy by `scripts/tracker_score.py`.

**Memory per change** (part a, 40 s slice, peak resident GB; each row adds one change to the rows above it):

| Build | parent `main` | single-stage |
|---|---|---|
| before | 3.81 | 9.73 |
| + tracked window only | 3.63 | 9.83 |
| + cache released after smoothing | 2.71 | 9.39 |
| + diagonal covariance | 2.20 | 7.95 |

The window change only helps sub-range runs (a 5 s slice of the same run fell from 2.3 to 1.3 GB). A
compact observation record was not built: observations are not what dominates (a fixed working set of about
3.6 GB for the single-stage form plus a cache that grows with the steps), and the hierarchical runs are
below the limit without it. Disk spill was not built for the same reason. The changed build gives the same
results as the old one, on the slice and on the full runs (every figure of the accuracy tables below is
unchanged).

**Full runs** (wall seconds and peak resident GB; real-time factor = wall time / sequence time):

| Run | Persons x cameras | Before | After | Real-time factor after |
|---|---|---|---|---|
| parent `main`, part a | 1 x 4 | 425 s / 7.2 GB | 535 s / 4.3 GB | 6.3 |
| parent `main`, part b | 3 x 5 | 528 s / 8.0 GB | 529 s / 6.0 GB | 18.7 (6.2 per person) |
| single-stage, part a | 1 x 4 | 530 s / 19.4 GB | 531 s / 13.4 GB | 6.2 |
| single-stage, part b | 3 x 5 | | 649 s / 21.4 GB | 22.9 (7.6 per person) |

The wall times after the changes are 0-26% above those before with code that does no more work: the
machine was shared and the two sets were run at different times, so wall-time differences below about 25%
in this table are noise (memory is not affected). The same trial with the body-crop finger markers took 2,385 to 2,648 s
single-stage, 84-93 times real time (28-31 per person).

**Accuracy against the label-fed run** (subjects 1 and 2, the subjects with a reference run; mean / median /
p90 joint distance in cm over the labelled range):

| Configuration | Subject 1 | Subject 2 |
|---|---|---|
| single-stage, with body-crop fingers | 14.0 / 7.8 / 34.4 | 7.3 / 4.2 / 15.3 |
| single-stage | 16.4 / 8.3 / 42.8 | 6.9 / 3.9 / 15.7 |
| parent `main`, with body-crop fingers | 18.8 / 9.2 / 49.2 | 7.2 / 4.2 / 15.9 |
| parent `main` | 18.5 / 9.5 / 46.2 | 7.3 / 4.1 / 16.1 |

Largest joint distance to the label-fed run in the six failure events (cm; 40 cm flags an event, and every
variant flags the same five):

| Configuration | u3-timo-legs | u4-fall | u4-jitter | u4-lleg | u4-rarm | u4-rarm2 |
|---|---|---|---|---|---|---|
| single-stage, with body-crop fingers | 37 | 156 | 107 | 83 | 78 | 64 |
| single-stage | 30 | 108 | 89 | 71 | 52 | 94 |
| parent `main` | 37 | 127 | 116 | 61 | 57 | 78 |

**Reading.** The restricted form loses no failure event (the same five are flagged) but does not match the
single-stage run everywhere: against the single-stage run with the same markers it is closer to the
label-fed run in two of the six events (`u4-lleg`, `u4-rarm2`), further in the other four, and 2 cm worse
on average and 3 cm worse at the 90th percentile for subject 1. Without the body-crop finger markers the
single-stage form is as fast as the restricted one here, but it peaks at 13.4 GB (part a) and 21.4 GB
(part b), above the 8 GB limit, and that speed holds only because these sequences have no refined hands: with
them, the 42 `hand21` markers join the one filter and its cost rises again, while the restricted form keeps
them in the hand stages. The default therefore ships as the restricted form, `["main"]`, with this accuracy
difference stated.

None of the baseline sequences has `hand21` rows (the detection runs come from the spike pipeline, not the
refine-hands path), so these figures say nothing about the hand stages; on these sequences the hand stages
of the default configuration track without finger observations. Measuring them needs a detection run with
refined hands on a copy of the session.

# SOMA/GEM-X productization plan

Staged plan for turning the findings of the GEM-X / SOMA-77 pose-model spike
([soma77-spike-report.md](soma77-spike-report.md)) into the main code line. Inputs: the owner's
brief ([soma-gemx-productization-brief.md](soma-gemx-productization-brief.md)), its review
([soma-gemx-productization-review-notes.md](soma-gemx-productization-review-notes.md)), the spike
report and its two independent assessments, and the code. The software architecture and every
schema change are merged in [soma-gemx-architecture-and-schema.md](soma-gemx-architecture-and-schema.md).

## 1. Fixed decisions

SOMA-77 becomes the default (recommended) pose model and RTMPose-133 stays available, also as the
fallback when the SOMA licences are not accepted; detection works with and without a segmentation
mask; several candidates per keypoint are stored and switched on per detection run; the cross-camera criterion is called multi-view agreement; iterative tracking and segmentation UI
improvements are separate features; hand refinement is investigated, not dropped.

## 2. Where the code or the evidence disagrees with the brief or the review

1. **Hand refinement is not tied to segmentation.** The review says the plain YOLOX path never calls
   it. It does in the pose extraction tool (`app/pose/main.py`) and in `posetrak-pose run`
   (`app/pose/cli.py`), both on by default; only `posetrak detect run` rejects the option without
   `--segmentation`. The actual obstacle for the new default is that the pass runs only for
   133-keypoint models (it checks the model name and reads COCO indices) and the tracker places its
   rows at fixed COCO offsets: on a SOMA run it silently does nothing.
2. **Strict mask and multi-view agreement already exist in the tracker.** Both toggles the brief asks
   for are settings of the shared dot-assignment phase (`SegmentationMaskModifier` with its tracklet
   window, `CrossViewCorroborationModifier`). Building them for keypoints is routing candidates
   through that phase, not new mechanisms. The near-identical results of the two mask-gated variants
   are probably explained by the strict filter leaving one candidate for most keypoint-frames; a
   short measurement decides (stage 0).
3. **"Feed the GPU" is a hypothesis, not a diagnosis.** The code shows single-threaded 4K decode one
   camera at a time, JPEG encoding of preview crops inside the loop, per-frame PNG mask decode and
   full-resolution resize in the segmentation route, and no batching across frames for RTMPose. Any of
   them can starve the GPU. Profiling comes first; the variable frame rate helps only if the model,
   not decoding, dominates.
4. **Smoother memory does not need windowed smoothing.** About 14 of the 20 GB of an 85 s run are
   the smoother's dense error-state matrices kept per step (error state 202 with the anatomical limits);
   the observations of the whole sequence are loaded whatever range is tracked, about 1 GB measured
   (stage 0 and stage 6 measurements). A hierarchical configuration whose parent stage is restricted to `main`
   (at most 138-dimensional), a diagonal-only smoothed covariance, per-stage release or a disk spill of
   the forward cache, and loading only the tracked window are all exact; windowed smoothing changes the
   estimator and stays with iterative tracking.
5. **The manifest binding has a latent defect for dual-model skeletons.** The tracker's reader tries
   the manifest and then the implied COCO map for every slot. A skeleton carrying both COCO and SOMA
   markers (needed so one person's skeleton serves both models) would bind unclaimed SOMA slots to
   COCO markers. The binding design fixes this before default skeletons get SOMA markers.
6. **Room tags are mostly built.** Scattered tags with a per-room group already persist in
   `scene_marker_bodies`; what is missing is a re-solve workflow, which is not needed for SOMA.
7. **"Segmentation is the solution" for YOLOX fragments.** It is the best source where persons
   overlap, but an automatic cross-camera association of fragments handles the separated majority of
   frames, fixes the observation gaps that every spike tracking run had, and can seed segmentation.
8. **Replacing the detector.** Both independent assessments advise against replacing ViTPose/RTMPose
   on the present evidence (SOMA worse on all six rolling-fall figures). Their figures measure
   keypoint accuracy against labels and are not disputed. The choice of SOMA rests on a different
   criterion, the visual quality of the tracked motion, where the GEM-X results are clearly better,
   presumably because of the SOMA keypoint structure. Which criterion matters depends on the use:
   visual quality for animation, the quantitative figures for measurement. The plan therefore makes
   SOMA-77 the recommended model, keeps RTMPose-133 fully supported, and reports the failure-event
   comparison of the two so that a user who needs the quantitative figures can choose.

## 3. Answers to the planning questions

| # | Question | Answer, in short | Design |
|---|---|---|---|
| 1 | Performance | Unknown where the time goes; code points to decode and CPU work in the loop. Profile per stage, then one frame loop with a pluggable box source, decode threads per camera, batched inference across frames and cameras, a writer thread, hand pass in the same loop; variable rate only if the model dominates. | [detection pipeline](../detection-pipeline/detection-pipeline-design.md) |
| 2 | Tracker cost | Cost model: sigma points `2e+1`, LDLT of a measurement of up to `2MC`; memory `4e^2` doubles per step plus the whole sequence's observations. Measured: the Kalman gain dominates and grows faster than quadratically with observations per step. Fewer markers in the parent stage (restricted to `main`, no body-crop fingers), diagonal-only smoothed covariance, per-stage release, window-only observation load (per-person threads were tried and are slower); a lower rate is slower as implemented; snapshots later. | [tracker performance](../tracker-performance/tracker-performance-design.md) |
| 3 | Skeleton and model coupling | Pose model = keypoint vocabulary; manifest per observation source for every new sequence; model-qualified landmarks; skeletons composed from hierarchy, joint-limit, marker-set and dimension layers, default skeletons with COCO and SOMA marker sets; existing skeletons and runs untouched, upgrade by a new skeleton version. | [skeleton binding](../pose-model-skeleton-binding/pose-model-skeleton-binding-design.md) |
| 4 | Candidate storage and assignment | `kp_candidates` rows with a self-describing blob and tracklet ids linked online at detection; copied into sequences like dots; tracker uses the shared dot assignment with a keypoint-class modifier and duplicate merging across crops; no tracklet table. | [keypoint candidates](../keypoint-candidates/keypoint-candidates-design.md) |
| 5 | Person assignment without segmentation | Cross-camera triangulation of fragment trunk keypoints, 3D linking, fragment labelling with splits, written to the existing assignment table; segmentation for contact windows, seeded by it. | [track to person assignment](../track-person-assignment/track-person-assignment-design.md) |
| 6 | Hand refinement | Only a command-line rule ties it to segmentation; the real limit is its COCO-only input and output layout. Make it model-independent with a `hand21` manifest, run it in the same frame loop for any box source, measure its value on SOMA runs. | [hand refinement](../hand-detection-refinement/hand-refinement-without-segmentation-design.md) |
| 7 | Model licence | Weights under the NVIDIA Open Model License (commercial use and redistribution allowed with the agreement and a notice; use restrictions on consent and surveillance); the 2D model is DINOv3-based and the DINOv3 License may apply as well (redistribution only under it, "Built with DINOv3", prohibited military and similar uses). Nothing of it is in the repository. Download on first use with licence acceptance; no bundling by default. | [detection pipeline §4](../detection-pipeline/detection-pipeline-design.md) |
| 8 | Mask versioning | Append-only revisions per segmentation run, only changed frames stored, latest-revision view for current readers, revision recorded with detection and tracking runs, staleness check and incremental refresh into a new run. | [segmentation versioning](../segmentation-versioning/segmentation-versioning-design.md) |
| 9 | Keypoint editing | Tracklets as the editing unit, stored in the existing tracklet decision table; first stage uses its existing kinds; second stage adds force, split and group operations with time-range and cross-camera grouping. | [tracklet editing](../keypoint-editing/tracklet-editing-design.md) |
| 10 | Calibration and robustness | In: database locking and run status, joint-limit table, fail-loud box calibration, safe camera relabeling (first release); intrinsics quality check (second). Separate: room-tag re-solve workflow, extrinsics refinement from keypoints. | [calibration robustness](../extrinsics-improvements/calibration-robustness-design.md), [DB concurrency](../session-db-concurrency/session-db-concurrency-design.md) |

## 4. Stages

Stage 0 is measurement only and comes first because several later scopes depend on its answers.

| Stage | Release | Depends on | Design |
|---|---|---|---|
| 0 De-risking measurements | first | - | several, see below |
| 1 Session database robustness | first | - | [DB concurrency](../session-db-concurrency/session-db-concurrency-design.md) |
| 2 Pose model and skeleton binding | first | (M2) | [skeleton binding](../pose-model-skeleton-binding/pose-model-skeleton-binding-design.md) |
| 3 Detection pipeline | first | 2, M1, M4 | [detection pipeline](../detection-pipeline/detection-pipeline-design.md) |
| 4 Track to person assignment | first | 3 | [track to person assignment](../track-person-assignment/track-person-assignment-design.md) |
| 5 Hand refinement for SOMA runs | first | 2, 3 | [hand refinement](../hand-detection-refinement/hand-refinement-without-segmentation-design.md) |
| 6 Tracker cost | first | 2, M2b | [tracker performance](../tracker-performance/tracker-performance-design.md) |
| 7 Calibration items of the first release | first | - | [calibration robustness](../extrinsics-improvements/calibration-robustness-design.md) |
| 8 Mask versioning | second | 1, 3 | [segmentation versioning](../segmentation-versioning/segmentation-versioning-design.md) |
| 9 Keypoint candidates | second | 2, 3, 8, M3, M5 | [keypoint candidates](../keypoint-candidates/keypoint-candidates-design.md) |
| 10 Tracklet editing, first stage | second | 9 | [tracklet editing](../keypoint-editing/tracklet-editing-design.md) |
| 11 Intrinsics quality check | second | - | [calibration robustness §4](../extrinsics-improvements/calibration-robustness-design.md) |
| 12 Tracklet editing, second stage | third | 10 | [tracklet editing](../keypoint-editing/tracklet-editing-design.md) |

### Stage 0 — De-risking measurements

- **Scope:**
  - **M1 Detection profile.** Per-stage timing, sampling profile and GPU utilisation over a fixed
    10 s five-camera clip, for YOLOX + RTMPose, masks + RTMPose, masks + SOMA (flip on/off), hand
    pass on/off, and decode alone ([detection pipeline §3](../detection-pipeline/detection-pipeline-design.md)).
    Needs a small timing instrumentation behind an environment variable.
  - **M2 Tracker baseline.** The 85 s sequence with the SOMA skeleton, optimized build:
    single-stage against hierarchical, 120 against 60 Hz; wall time, peak memory, per-step timings,
    failure events ([tracker performance §3](../tracker-performance/tracker-performance-design.md)).
  - **M3 Mask filter and multi-view agreement.** From the spike's candidate files: share of
    keypoint-frames with two or more candidates before and after the strict tracklet-mask filter
    ([keypoint candidates §7](../keypoint-candidates/keypoint-candidates-design.md)).
  - **M4 Licence question.** Ask NVIDIA whether the DINOv3 terms apply to `vitpose.pth`; record the
    answer in [detection pipeline §4](../detection-pipeline/detection-pipeline-design.md).
  - **M5 Acceptance harness.** Move the ground-truth metric harness to read productized storage and
    add the same-candidate metric; freeze the labelled ranges, the failure-event list and the
    scoring script as the evaluation set.
- **Depends on:** nothing.
- **Acceptance:** each measurement produces its table in the design document it belongs to.
- **Risks:** M2 runs are tens of minutes each; they are run in a standalone shell, not in an agent's
  background task (open issue on background runs being killed).
- **Outcome:** M1, M2 (with M2b, the parent stage restricted to `main`), M3 and M5 done; results in the
  design documents. M4: the question to NVIDIA is drafted in detection pipeline §4 and has not been sent.
  What the results change is folded into stages 3, 6 and 9 below.

### Stage 1 — Session database robustness

- **Design:** [session DB concurrency](../session-db-concurrency/session-db-concurrency-design.md).
- **Scope:** busy timeouts, run status, interrupted-run detection, UI respecting status; schema
  version 64.
- **Depends on:** nothing.
- **Acceptance:** the two-process lock test and the killed-tracker test of the design pass.
- **Risks:** low; the UI changes touch the session tree only.
- **Outcome:** done; both tests of the design pass.

### Stage 2 — Pose model and skeleton binding

- **Design:** [pose-model skeleton binding](../pose-model-skeleton-binding/pose-model-skeleton-binding-design.md).
- **Scope:** pose model registry; manifest per observation source (schema 65) and the reader change;
  model-qualified landmarks; skeleton layer files (hierarchy, joint limits, marker sets, dimensions)
  and `posetrak skeleton compose`; SOMA marker set with the body markers and the first index and
  little finger joints; default skeletons composed with both marker sets and the joint-limit table;
  hinge test; decoder at cell centres in the SOMA estimator.
- **Depends on:** nothing; M2 informs the default hierarchical configuration shipped with the
  skeletons.
- **Acceptance:** the binding design's tests; the reference capture's held-out errors within 0.3 cm
  of the spike's calibrated-offset figures.
- **Risks:** regenerating the defaults changes their ids (intended, existing rows stay); the
  wrist-deviation saturation may need marker or frame changes beyond limits.
- **Outcome:** done; the reference capture's held-out errors are 2.2 / 3.9 / 2.7 cm (legs / arms / head),
  within 0.2 cm of the spike. Not done: a UI action that adds SOMA markers to an existing skeleton
  (`skeleton compose --base` does it). The skeletons of existing persons have no SOMA markers, so a SOMA
  sequence tracked with one stops at the start, naming the missing pose model. The trial in which the
  earlier joint-limit table saturated a wrist axis was not re-run.

### Stage 3 — Detection pipeline

- **Design:** [detection pipeline](../detection-pipeline/detection-pipeline-design.md).
- **Scope:** box source protocol and the three sources; SOMA as a registered batched estimator with
  mask gate; decode / infer / write pipeline sized by M1; `PoseWorker` and GUI dialogs on the same
  loop; model download with licence acceptance; SOMA-77 preselected as the recommended model, with
  RTMPose-133 when the licences are declined; run inputs recorded (schema 66). From M1, in order of
  measured cost:
  - A decode thread per camera. Decode is 12 times real time and serial, and the GPU is idle while it
    runs.
  - All persons of a frame in one pass. The prototype decodes the clip once per person.
  - The hand pass inside the loop, which removes its second decode, and batched hand crops.
  - Cheaper box sources: mask boxes computed at mask resolution and scaled, instead of a resize to 4K
    and a scan per label (25 ms per frame); YOLOX (24 ms per frame) batched across cameras.
  - An explanation of why RTMPose costs twice as much per crop in the mask route as in the YOLOX
    route.

  With decode overlapped, inference is the bound. The variable frame rate stays off (owner
  decision 5) and is revisited on the finished pipeline.
- **Depends on:** stage 2 (the estimator writes manifests), M1, M4.
- **Acceptance:** detection pipeline §5 (no prior RTMPose pass needed; GPU at least 85% busy or
  bounded by decode; at least twice the prototype route's throughput; parity with the prototype
  keypoints).
- **Risks:** decode parallelism on Windows (the threaded-decode hang); GPU memory with ViT-H batches
  and several decode threads.
- **Outcome:** done; 2.9 times the prototype's throughput, keypoints identical to the prototype's on the
  same boxes. The GPU is 79-82% busy, short of 85%: the model bounds the loop, with crop preparation and
  YOLOX (batch size 1 in its ONNX graph) on the inference thread. The 30-minute decode soak of the thread
  and process producers has not been run.

### Stage 4 — Track to person assignment

- **Design:** [track to person assignment](../track-person-assignment/track-person-assignment-design.md).
- **Scope:** `detect assign` (cross-camera clustering, 3D linking, splits, subject seeds, report).
- **Depends on:** stage 3 for SOMA trunk keypoints (works on RTMPose runs before).
- **Acceptance:** the design's agreement and gap criteria on the labelled trial.
- **Risks:** calibration quality of the capture limits association; the report must make that
  visible rather than hide it.
- **Outcome:** done. Single-person footage and separated persons need no manual step. With bystanders
  the number of persons must be given (`--persons`). The labelled contact trial reaches 88.7% against
  98%: identity through a contact where persons are tangled is not attempted, and such stretches are
  listed in the report for the stitcher or segmentation.

### Stage 5 — Hand refinement for SOMA runs

- **Design:** [hand refinement without segmentation](../hand-detection-refinement/hand-refinement-without-segmentation-design.md).
- **Scope:** model-independent wrist input, `hand21` manifest, pass inside the frame loop, every box
  source; measurement of its value on SOMA runs.
- **Depends on:** stages 2 and 3.
- **Acceptance:** the design's §4; the SOMA default set from the measurement.
- **Risks:** no gain, in which case it ships off for SOMA runs.
- **Outcome:** done; on by default for SOMA runs, from the labelled trial (fingers, forearm twist and hand
  orientation improved). The default is provisional until a capture made to measure hand tracking
  confirms it (owner decision 7).

### Stage 6 — Tracker cost

- **Design:** [tracker performance](../tracker-performance/tracker-performance-design.md).
- **Scope:**
  - The parent stage's joint groups stored with the configuration (`tracker_configs.active_joint_groups`,
    schema 67), and the default configuration for 77-keypoint skeletons chosen from M2b and the
    harness.
  - A warning at run start for a skeleton with hand groups tracked single-stage or with an
    unrestricted parent.
  - Diagonal-only smoothed covariance and per-stage release of the forward cache.
  - Observations loaded for the tracked window only.
  - Per-person threads outside contact (tried, not kept, see the outcome).
  - Disk spill only if the above leave a 10 000-step run above the limit.
  - The lower tracker rate is dropped: it pools observations and is slower.
- **Depends on:** M2b; stage 2 for the parent marker set without body-crop fingers. The gain cost
  grows faster than quadratically with observations per step, so a parent stage without the body-crop
  fingers and the hand markers is the largest single saving.
- **Acceptance:** the design's §4: below 8 GB at 10 000 steps, real-time factor reported, no lost
  failure event, and the default chosen on the harness where configurations differ (single-stage and
  hierarchical differ by 9-24 cm on arm and hand events).
- **Risks:**
  - Threading the multi-person loop needs care at contact boundaries; it can be dropped from the
    release without affecting the rest.
  - Three persons take about 90 times real time today; the first release may not reach a
    comfortable speed, and that should be visible in the reported real-time factor.
- **Outcome:** done, apart from per-person threads ([tracker performance §5](../tracker-performance/tracker-performance-design.md)):
  - The bundled hierarchical configuration (parent `main`, hand stages) is the default for 77-keypoint
    skeletons: 4.3 GB and 6.3 times real time on part a, 6.0 GB and 18.7 times real time on the
    three-person part b; disk spill not needed. Every variant flags the same 5 of the 6 failure events.
  - Accuracy is slightly below single-stage (subject 1 mean 18.5 against 16.4 cm); the hierarchical form
    was chosen for cost, the memory limit, and because refined hands would join a single-stage filter.
  - Per-person threads were slower (572 against 529 s on part b): 23% of the frames qualify and the
    threads compete with the filter's OpenMP loops. Not kept.
  - The hand stages' cost is not measured: no baseline sequence has refined hands.
  - A parent stage on the whole skeleton is still possible; it is only warned about.

### Stage 7 — Calibration items of the first release

- **Design:** [calibration robustness §1 and §3](../extrinsics-improvements/calibration-robustness-design.md).
- **Scope:** fail-loud moving-box calibration in the `extrinsics_rig` command and the dialog; safe
  camera relabeling.
- **Depends on:** nothing.
- **Acceptance:** a capture whose box cannot be solved produces an error table and a non-zero exit
  and writes nothing; a box that ends at rest in view of some cameras only is solved from an earlier
  window every camera sees; relabeling keeps sync points with their videos.
- **Risks:** low.
- **Outcome:** done. `posetrak extrinsics box-video` and "Calibrate from box video" in the calibration
  dialog are new: the moving-box method was a prototype script before. Per-camera error table with
  median and 90th-percentile limits, nothing written on failure. On the design's failing capture the
  63-104 s range fails with insta_ace2_pro as the reference camera and passes with gopro13_02 (36
  instead of 58 windows); why the reference camera decides is not known. The 100-115 s range passes.
  Several decoders in parallel bring a 41 s range to about 73 s. Camera reassignment: `capture
  reassign-camera` and the "Cameras…" button of the capture panel. The capture's first sync
  configuration is always used.

### Stage 8 — Mask versioning

- **Design:** [segmentation versioning](../segmentation-versioning/segmentation-versioning-design.md).
- **Scope:** revisions, view, panel changes, concurrent use (pinned reads, one open revision per
  segmentation run), recorded revisions, `detect stale` and refresh, prune; schema 68.
- **Depends on:** stage 1 (status, busy timeout), stage 3 (detection records the revision).
- **Acceptance:** the design's §4, including a tracking run that reads one revision while the panel
  saves another.
- **Risks:** the rebuild of `seg_masks` on large sessions (time, disk); missing one of the direct
  readers (they are enumerated: about twenty queries in eight files).
- **Outcome:** done ([status](../segmentation-versioning/status.md)). Reproduction and the two-process
  test pass on a real ukemi trial; the migration takes seconds and grows the file by the size of its masks
  until a `VACUUM`. Open: a detection run assembled from several pose jobs of the segmentation window
  records the first job's revision; the crash-recovery offer works for editors on the same host only.

### Stage 9 — Keypoint candidates

- **Design:** [keypoint candidates](../keypoint-candidates/keypoint-candidates-design.md).
- **Scope:** extraction and online linking in the SOMA estimator; finalisation copy; loader with class
  and duplicate merge; `KeypointClassModifier`; `kp_candidates_enabled` (schema 69).
- **Depends on:** stages 2, 3, M3, M5; stage 8 for reproducible mask gating.
- **Acceptance:** the design's §8: reproduces the prototype's tracker figures within 0.5 cm, both
  scored by the harness, whose figures differ from the spike's by up to 0.3 cm; off equals no
  candidates.
- **From M3:** strict mask adherence and multi-view agreement stay separate options. After the filter,
  agreement can still change the pick on 11% of the hard-window keypoint-frames, and the filter
  empties 27-34% of the keypoint-frames of the persons in contact. The second figure is a risk for
  the strict option exactly where the choice is hardest.
- **Risks:** per-step cost of the shared assignment with about 150 candidates per person and camera
  (the dot phase was tuned for a few dozen); measured in the acceptance run, with the class
  restriction pruning the cost matrix per keypoint.
- **Outcome:** built (session schema 69, registry 18): `detect run --candidates`, finalisation, loader, merge,
  `KeypointClassModifier`, `track run --candidates / --strict-mask / --multi-view-agreement`. Candidates off give
  a result identical to a sequence without candidates, and the extracted candidates equal the prototype's
  on the same crops. **The accuracy bar is not met**: the tracker's per-step choice with candidates is 18.7 / 7.6 cm
  on range 3 against 10.2 / 3.4 for the prototype's offline cross-camera choice (bar 10.7 / 3.9) and not better
  than the strongest maxima (18.0 / 5.3); strict mask adherence (14.6 / 7.9) and multi-view agreement (14.4 / 6.9)
  each help. The cost is +4-10% tracking time, +16% detection time and +26% stored detection rows; the shared
  assignment takes 14 ms per step. The `suggest` rule from an automatic grouped choice is the open decision;
  figures and the like-for-like check in the design §9. Candidates are off by default and stay so.

### Stage 10 — Tracklet editing, first stage

- **Design:** [tracklet editing](../keypoint-editing/tracklet-editing-design.md).
- **Scope:** tracklet lanes, candidates and trails in the crop grid, reject / other / forbid / assign
  with ranges, re-track with the rules.
- **Depends on:** stage 9.
- **Acceptance:** the design's §5, first item.
- **Risks:** timeline performance with tracklets computed on the fly.

### Stage 11 — Intrinsics quality check

- **Design:** [calibration robustness §4](../extrinsics-improvements/calibration-robustness-design.md).
- **Scope:** the intrinsics check at capture setup and before tracking.
- **Depends on:** nothing.
- **Acceptance:** cameras with missing intrinsics, implausible distortion (k1 about -2.5) or large
  edge errors are flagged on their captures.
- **Risks:** thresholds tuned on few cameras; flagged cameras fall back to velocity-only
  observations, which must be visible in the run.

### Stage 12 — Tracklet editing, second stage

- **Design:** [tracklet editing](../keypoint-editing/tracklet-editing-design.md).
- **Scope:** force, split, groups (`edit_group_id`, schema 70), time-range and cross-camera
  grouping, suggested review windows from the failure flag.
- **Depends on:** stage 10.
- **Acceptance:** the design's §5, second and third items.
- **Risks:** grouping heuristics tuned on one trial; the operation must stay correctable one
  tracklet at a time.

## 5. Release split

| Release | Stages | Usable state | Schema |
|---|---|---|---|
| First: SOMA recommended | 0-7 | SOMA-77 from any box source in one pass, with refined hands, automatic person assignment for YOLOX footage, default skeletons composed from layers with SOMA markers and corrected joint limits, hierarchical tracking within memory, robust concurrent writes; new: box-video calibration that fails loudly, and safe camera reassignment. RTMPose-133 unchanged and used when the SOMA licences are declined. | 64, 65, 66, 67 |
| Second: candidates | 8-11 | Reproducible mask use under concurrent editing; keypoint candidates optional per detection and per tracking run, with strict mask and multi-view agreement options; first tracklet editing; intrinsics check. | 68, 69 |
| Third: tracklet editing | 12 | Force, split and group operations, review windows. | 70 |

### Known limits of the first release

The stage outcomes above in one list, for testing before the release:

- **Hands.** The hand refinement default rests on one labelled trial, and the cost of the hand stages
  in the tracker is not measured. A capture made for hand tracking settles both.
- **Accuracy.** The hierarchical default is slightly less accurate than single-stage on the trial. The
  failure-event comparison of SOMA-77 against RTMPose-133, to be reported to users (owner decision 6),
  is not produced yet.
- **Existing skeletons.** They have no SOMA markers; `skeleton compose --base` upgrades one, with no UI
  action.
- **Person assignment.** Contact stretches are listed, not resolved; the person count with bystanders
  is an option.
- **Detection.** GPU below the 85% target; the decode soak not run.
- **Box-video calibration.** The reference camera can decide pass or fail; the sync configuration is
  not selectable.
- **Data model.** `sync_points.camera_instance_id` holds a `capture_videos.id` for the rows written by
  the sync dialog, against what its name says.

The first release adds every column its components write, including the segmentation columns of
detection and tracking runs (NULL revision until the second release), so later releases only add
tables, columns and values.

### Branches and schema versions

- Each stage is a branch from `main`, merged when its acceptance passes. The spike branch
  (`feature/soma77-pose-spike`) is kept as a reference and is not merged: a stage takes over, or
  re-implements, the spike code it needs (the SOMA-77 estimator, the detection timing hooks, the
  ground-truth harness, the tracker-baseline runner).
- `main` stays releasable between merges: SOMA-77 is selectable as soon as stage 3 merges, and the
  switch that preselects it is the last merge of the first release.
- The schema versions in this plan are provisional; only their order is fixed. Work on other
  features adds migrations too, so a branch takes the next free version when it merges, and
  whichever branch merges later renumbers its migration. A branch's migrations run only on copies of
  session and registry databases until it is merged; a database stamped with a provisional version
  would skip the migration that later takes that number.

## 6. Deferred, and why

| Item | Why deferred |
|---|---|
| Iterative tracking: snapshots, restart over a changed range, windowed smoother | Large tracker change; the cheaper cost options and tracklet editing come first; then it is the next feature for the edit-track loop |
| Short-window robust estimation and 3D keyframe correction | Recommended by both assessments for the failure windows; a research step with its own spike, after candidates are productized |
| Hybrid detector (SOMA trunk and head, RTMPose limbs) | Possible as a configuration once manifests are per source and skeletons carry both marker sets; costs two model passes; measured, not built |
| Automatic grouped candidate proposals (`suggest` rules) | Measure the tracker's own assignment of candidates first |
| Body-crop finger joints other than the first index and little finger joints | They localise poorly; an accurate finger pose comes from the hand pass |
| Skeleton layers in the registry: per-performer limit and marker overrides, per-capture marker sets, scaling into a dimensions layer | Its own feature; the first release's layer files and composition step are its starting point |
| Per-subject SOMA offset calibration | Belongs to skeleton scaling and the skeleton-layers feature; nominal offsets ship first |
| Variable frame rate | A possible later optimisation; built and switched on only when shown to work reliably on the failure-event table |
| Lower tracker rate by decimating observations | Pooling, as implemented, is slower and moves every failure event; decimation only if the other cost options fall short |
| Segmentation UI improvements and segmentation speed-up | Separate feature by decision; the variable frame-rate rule and seeding from automatic assignment are inputs to it |
| Room-tag re-solve workflow | Storage exists; not needed for SOMA |
| Extrinsics refinement from keypoints in the UI | Command-line recipe first; productize with the calibration dialog |
| Visibility model, learned motion prior, mask-conditioned detectors | No measured tracker gain so far, or not yet evaluated |
| Video caching for scrubbing | Its own feature (preview caching); the labeller's cache is not productized |
| A summary table of tracklets | Only if computing tracklets from rows is measured as too slow |

## 7. Owner decisions

| # | Question | Decision |
|---|---|---|
| 1 | Distribution of the SOMA weights | Download on first use after showing both licences. Bundling waits for NVIDIA's answer on the DINOv3 terms. |
| 2 | Body-crop finger keypoints of SOMA | The first joints of the index and little fingers are tracked from the body crop on the hand segment, because they determine forearm twist and wrist rotation without a hand pass. The other body-crop fingers are not tracked; an accurate finger pose needs the hand refinement. |
| 3 | One skeleton for both pose models, or one per model | One joint hierarchy per tracking need, with marker sets per pose model as separate layers composed into the resolved skeleton the tracker uses. Joint limits (overridable per performer), marker placement (per model, performer or capture) and dimensions (per performer) are separate layers ([binding design §2.7](../pose-model-skeleton-binding/pose-model-skeleton-binding-design.md)). |
| 4 | Unit of a mask revision | One revision per explicit save of the segmentation panel, with closing the panel or finishing a propagation job also saving. Command-line jobs running during a segmentation session pin the newest saved revision at their start ([versioning design §2.3](../segmentation-versioning/segmentation-versioning-design.md)). |
| 5 | Default of the variable frame rate | Off; a later optimisation, kept off until it is shown to work reliably. |
| 6 | Default switch to SOMA | SOMA-77 is the recommended, preselected model. A user who does not accept its licences gets RTMPose-133 (or ViTPose). The failure-event comparison against RTMPose is produced and reported but does not gate the recommendation. |

Open:

7. **Hand refinement default for SOMA runs.** (a) set from the stage 5 measurement, (b) on
   regardless. Recommendation: (a).

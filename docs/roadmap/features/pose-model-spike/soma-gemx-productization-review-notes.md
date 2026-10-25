# Review notes on the SOMA/GEM-X productization brief

Companion to `soma-gemx-productization-brief.md`. Marks where the evidence supports the brief,
where it argues for a different priority, and what the brief leaves out. Numbers come from the
spike runs and from two further real captures that had no ground truth.

Caveats: the evidence that SOMA beats the existing detector comes from one labelled capture. The
license of the GEM-X model weights has not been checked.

## Where the evidence argues for something different

### 1. "Make SOMA the default"

The tracker-level evidence supports it, but the brief omits what has to come with it.

- The skeleton is tied to the detector. The SOMA-77 skeleton and its marker offsets come from a
  prototype script and a single calibration. The pose model and the skeleton's markers must be
  one concept in the data model (the sequence keypoint manifest is the start of it).
- The default skeletons must carry the SOMA markers, and the offsets ship as catalog data.
- The default skeleton is already close to SOMA in structure: it has spine and neck joints, and has
  to. COCO-133 has no spine or neck keypoints, so the tracker must fit the spine from thigh,
  shoulder and head markers, which do not constrain the actual pose well. This is probably the
  main reason COCO-133 poses are often flawed, and it is the strongest argument for SOMA as the
  default: it supplies keypoints for exactly the joints COCO leaves unobserved.
- The model license must allow redistribution before it becomes a default, because the repository
  is Apache/MIT and REUSE-clean.

### 2. Hand refinement pass

The spike's hand-crop test gave no tracker-level gain, so its value is unproven. It still has to be
investigated during productization, not dropped.

Today `--refine-hands` fails without a segmentation run. That is a limit of the code path, not a
known design reason: `detect run` refines hands only inside the segmentation-sourced pipeline, and
the plain YOLOX path never calls the refinement. Whether refinement can work per frame on
fragmented YOLOX identities is open and belongs in the design study.

### 3. "Keep the GPU at 100%"

Performance is not good enough and must be profiled and optimized as part of productization. The
available data is thin: single-sample GPU readings of about 96 % during the SOMA pass and 45-50 %
during the RTMPose pass, and about 35 minutes per trial for an RTMPose pass run only to obtain
boxes. These do not establish where the time goes, so no claim is made that the second pass is the
main cost.

Productize one `detect run` with a pluggable box source:

- segmentation masks (boxes derived from the masks),
- the YOLOX detector,
- an existing detection run.

SOMA becomes a native pose model of that command instead of a prototype that needs a source run.

As part of the optimization, consider a variable frame rate: process fewer frames while the
person boxes are far apart or do not overlap, and the full rate when they overlap.

### 4. Two toggles: strict mask and cross-camera corroboration

The mask-gated runs with and without the cross-camera criterion gave nearly identical results, and
the reason has not been explained. Explain it before building the second toggle; its measured
gain is about zero.

Name for the cross-camera criterion: "multi-view agreement".

### 5. "Candidates on" should depend on stored data, not on the detection model

Store the candidate count in the detection run's configuration, so any model can use it.

Define the acceptance metric before building: the "same-candidate" metric next to the within-3 %
metric on the labelled data. Keep that harness in the repository.

### 6. Iterative tracking with snapshots and a windowed smoother

This is a large C++ change and likely a separate feature; the tracker has already been optimized
several times. Profile first. A tracking run took about 19 minutes for 85 s of footage (roughly 13
times slower than real time) with a single-stage tracker configuration (153 degrees of freedom).
The same sequence with the hierarchical configuration (two hand stages) took 1362 s: the parent
stage again ran with 153 degrees of freedom and took 1085 s, and each hand stage (48 degrees of
freedom) added about 137 s. So for this skeleton the hierarchical configuration did not lower the
cost; it added the hand stages on top of an unchanged body stage, and the body and wrist joint
angles came out identical in the two runs. Why the parent stage keeps the finger degrees of
freedom is not established and belongs in the design study. Cheaper options come before snapshots:
a lower tracker rate (the setting exists, currently 120 fps) or a smaller state.

## Additions

- **Tracker memory.** A 10 000-step run reached about 18 GB resident, probably from RTS smoothing
  keeping per-step covariances. It triggered low-memory process kills. Windowed smoothing fixes
  this together with item 6.
- **Database locking.** The tracker's result writer opens the session database without a busy
  timeout, so a concurrent writer (a detection job on the same database, the UI, a segmentation
  edit) makes it fail at once and leaves a half-written run. Failed jobs also leave unfinished
  run rows. Fix: busy timeout, short transactions, a status per run (running, complete, failed)
  that the UI respects. Recorded in `docs/open-issues.md`.
- **Track-to-person assignment for YOLOX.** A YOLOX run produces hundreds of short track fragments
  per camera, so no single track id covers a person. The SOMA prototype needs a one-person
  workaround. Segmentation is the recommended solution and works well, but it is slow, so the
  variable frame rate idea above applies to segmentation as well. Productize a real assignment step.
- **Anonymous marker and keypoint assignment.** The brief is right that they share a framework.
  The dot-assignment machinery in the tracker (tracklets, mask gate, Hungarian assignment) already
  solves the same problem; reuse it. The design study should cover the markerless implementation
  as well as the marker one.
- **Joint limits.** Keep one table of anatomical limits for the default skeletons, with a test
  that hinge joints have zero off-axis limits. Found in the current defaults:
  - knee: shin side-bend limits of +-90 degrees,
  - elbow: sideways limits of +-90 degrees on the forearm joint,
  - wrist: twist limit of +-30 degrees and deviation limits of +-90 degrees.

  After locking these, the tracked wrist deviation still saturated at its new limit for much of a
  trial, so the cause (marker placement, hand-frame axes, forearm twist range) needs a look.
- **Mask versioning.** Store the segmentation version with every detection and tracking run, in the
  same append-only style as detection runs.

## Extrinsics and calibration

- **Calibration-box tool.** It assumes the box ends at rest in view of every camera. A capture
  where the box was only set down after the stated range, seen by two of four cameras, produced
  errors of 100-800 px and the tool still printed the result as a normal solve. Rank windows by
  camera coverage, anchor on one that all cameras see, align to the rest frame afterwards, filter
  windows by corner count, and fail loudly on large errors.
- **Room tags.** Cameras moved by 70-400 px between two captures of one session, so extrinsics
  cannot be reused. Fixed room tags saved as scene markers give a common world frame.
- **Relabeling cameras.** Sync points belong to a capture-video row, so fixing a camera label by
  swapping file paths breaks the sync. Provide a safe "reassign camera label" operation.
- **Intrinsics.** Capture setup found no intrinsics for one capture's videos. Separately, one
  camera's stored distortion (k1 about -2.5) and another camera's edge accuracy (tags 8-10 px off
  at the image edges) need a quality check and a refit.

## Keep from the spike

The flip test, the parabolic refinement, the mask gate as an option, the corrected joint limits.

## Drop for now

The SOMA hand-crop refinement, the velocity-camera setting from the old capture, the experiment
scripts for the individual spike variants.

## Pointers for the planning round

Read first:

- `soma77-spike-report.md` in this directory: method, findings, what worked and did not, follow-up
  checks (section 10), directions (section 11), pointers (section 12). The two independent
  assessments (`soma77-independent-assessment*.md`) and `soma77-pose-spike-plan.md` sit beside it.
- `docs/open-issues.md`: mask versioning, session database locking and other open problems.
- `docs/data-model-and-storage.md` section 3: data-model invariants. Detection runs are append-only;
  a sequence whose keypoint layout is not a person model's carries a `pose_sequence_keypoints`
  manifest.

Code that exists today:

- `python/posetrak/detection/backends_soma77.py`: the SOMA-77 model backend and keypoint names.
- `python/tools/prototypes/soma77_extract.py`: SOMA extraction over an existing detection run's
  person boxes, with the heatmap mask gate and a one-person mode. Prototype, not part of
  `posetrak detect`.
- `python/tools/prototypes/build_soma77_skeleton.py` and `calibrate_soma77_offsets.py`: SOMA-77
  skeleton from a base skeleton, and calibration of the marker offsets.
- `python/tools/prototypes/kp_labeling/`: the labeling tool the brief prefers for editing (candidate
  extraction, linking into tracklets, labeler UI, frame cache). `python/tools/gt/` holds the
  ground-truth metric harness. `python/tools/prototypes/README.md` indexes what absorbed what.
- `python/posetrak/cli/detect.py`: the `detect run` command and its option rules, including the
  segmentation-only hand refinement.
- `python/posetrak/detection/hand_refinement.py` and
  `docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md`.
- `python/tools/calibrate_extrinsics_from_moving_box.py`,
  `python/app/setup/extrinsics_solver.py` and
  `docs/roadmap/features/extrinsics-improvements/extrinsics-improvements-design.md`: calibration
  from a box video.

Related designs:

- Detection from segmentation: `docs/roadmap/features/segmentation-reuse/segmentation-reuse-design.md`,
  `segmentation-ui-improvements/`, and `subject-identity/subject-identity-design.md` (who is who
  across captures).
- Tracking: `docs/cpp-architecture-overview.md`, the hierarchical solver
  (`docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md`,
  `docs/hierarchical-ukf-design.md`, `docs/hierarchical-tracker-redesign.md`). Hierarchical mode is
  switched on by the presence of per-group stage rows in the tracker configuration
  (`tracker_config_stages`), so copying only the main configuration row silently gives a
  single-stage run.
- Joint limits: `docs/skeleton-format.md`, `docs/roadmap/features/swing-twist-joint-limits/`,
  `docs/roadmap/features/soft-joint-limits/`, and `docs/skeleton-scaling-calibration-design.md`.
- Keypoint editing: `docs/roadmap/features/keypoint-editing/` (brief, design, improvements brief,
  user guide).
- Marker and dot assignment, which the markerless candidate assignment should share a framework
  with: `docs/roadmap/features/marker-based-mocap/productization-architecture-and-plan.md` and
  `docs/roadmap/features/pose-detect-improvements/`.
- Result semantics: `docs/roadmap/features/observation-results-semantics.md` (what the stored
  observation fields mean per measurement mode).
- Preview caching, for the scrubbing item: `docs/roadmap/features/preview-caching/`.

Data and tooling that are not in the repository:

- The spike's work copy of the labelled capture and its scripts live outside the repository, beside
  the session database. The labelled data is the only ground truth for the candidate-keypoint
  questions; the two later real captures have none.
- The MCP diagnostic server (`python/app/mcp/`, described in `CLAUDE.md`) gives filter statistics
  and camera coverage for any tracking run without hand-written queries; use it when profiling or
  comparing tracker configurations.

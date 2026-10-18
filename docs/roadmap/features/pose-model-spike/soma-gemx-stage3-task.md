# Task: stage 3 of the SOMA/GEM-X productization — detection pipeline

Implement stage 3 of [soma-gemx-productization-plan.md](soma-gemx-productization-plan.md) (section 4)
as designed in [detection-pipeline-design.md](../detection-pipeline/detection-pipeline-design.md).
Work in a separate git worktree on a new branch made from `main`. The spike branch
`feature/soma77-pose-spike` is a reference and is not merged (plan section 5): take over, or
re-implement, the spike code this stage needs. Other work uses the main work area, so do not touch
its build directories or its databases.

## Read first

- `CLAUDE.md`, all of it, including the invariant that detection runs are append-only.
- The detection pipeline design, all of it, and its `status.md`. Section 3 holds the measured costs
  this stage is sized by; section 4 the licence handling.
- Plan section 4, stages 3 and 5 (the hand pass's split between them), and section 5.
- [soma-gemx-architecture-and-schema.md](soma-gemx-architecture-and-schema.md) §3.3
  (`detection_runs`) and the migration table of section 4.
- [pose-model-skeleton-binding-design.md](../pose-model-skeleton-binding/pose-model-skeleton-binding-design.md)
  §2.1-§2.2: the pose model registry and the manifests finalisation writes, both on `main`.
- The session-db concurrency design: detection writers update `run_progress` per batch and set
  `detection_runs.status`; the new loop keeps doing both.
- On `main`: `posetrak/detection/pipeline.py`, `frame_source.py` (the note on threaded decode
  hanging on close), `backends.py`, `backends_rtmpose.py`, `backends_soma77.py`, `hand_refinement.py`,
  `app/pose/pose_worker.py`, `app/pose/db_cache.py` and the detection dialogs.
- On the spike branch, read with `git show feature/soma77-pose-spike:<path>`:
  `python/posetrak/detection/stage_timing.py` with its test,
  `python/tools/prototypes/profile_detection_routes.py` and `python/tools/prototypes/soma77_extract.py`
  (the main work area has uncommitted edits of the latter; use the committed version).

## Scope

1. **Stage timing:** take over `stage_timing.py` and its test (`POSETRAK_PROFILE_DETECTION`), and
   time the stages of the new loop with it.
2. **Box sources** (design §2): `BoxSource` with `YoloxBoxSource`, `MaskBoxSource` and
   `RunBoxSource`. The mask source computes boxes at mask resolution and scales them, instead of
   resizing the mask to native size and scanning per label, and exposes the mask for the heatmap gate.
3. **Batched estimators:** `estimate_batch` for RTMPose and SOMA-77; SOMA-77 registered as
   `soma77-vitpose-h` with flip test, crop 1.6, parabolic refinement, cell-centre decoding and the
   mask gate. Every option goes into `config_json`.
4. **One frame loop** (design §2, feeding the GPU): decode producers per camera, one inference
   consumer batching across frames, cameras and persons, a writer thread. All persons of a frame in
   one pass. YOLOX batched across cameras. `detect run`, `PoseWorker` and the GUI dialogs use it, and
   `pose_worker.py`'s own loop goes.
5. **Decode producers, threads or processes:** the design leaves this open, and the soak test that
   was to decide it has not run. Build the producer so either form can be chosen, and write a soak
   test (prototype under `python/tools/prototypes/`): the five videos of the profiling clip opened,
   decoded in random 10 s windows and closed in parallel, with a watchdog that counts hangs on open,
   decode or close (no progress for 60 s) per mode. Run it briefly yourself; the long run goes to the
   operator. Choose the default from its result and record it in design §3.
6. **Hand pass in the loop:** the existing COCO-133 hand refinement runs on the frame already
   decoded, with batched hand crops, instead of a second decode. Making it model-independent
   (`hand21` input from any body model) is stage 5.
7. **Run inputs (schema):** `detection_runs.seg_quality_run_id`, `seg_revision` (NULL until mask
   versioning) and `source_detection_run_id`. The plan calls this version 65, which stage 2 has taken;
   use the next free session version on `main` (66 now), provisional. Document every column of
   `detection_runs` and add it to `DOCUMENTED_TABLES`.
8. **Weights and licences** (design §4): download on first use from `nvidia/GEM-X` into the model
   cache after showing both licence texts (the NVIDIA Open Model License and DINOv3, which is assumed
   to apply until NVIDIA answers); the accepted licence version stored with the file; the notice line
   and "Built with DINOv3" in the about box and the user documentation; no weights in the repository
   or the installer.
9. **Default model:** SOMA-77 preselected in `detect run` and the dialogs. A user who declines the
   licences or has no weights gets RTMPose-133, and the pipeline says why.
10. **The RTMPose cost gap:** explain why RTMPose takes twice as long per crop in the mask route, or
    show that the unified loop removes it (design §3, "Unexplained").

Out of scope: the variable frame rate (stays off, owner decision 5), keypoint candidates (stage 9),
track to person assignment (stage 4), a model-independent hand pass (stage 5), segmentation
versioning beyond the NULL revision column.

Commit in reviewable steps that each leave `main`-quality code. If the stage does not fit, deliver
the loop, the box sources and the SOMA estimator first, and report what is left.

## Acceptance

Design §5:
- One `detect run` produces a SOMA-77 run from YOLOX boxes, from masks and from an existing run,
  without a prior RTMPose pass.
- On the profiling clip (design §3 material), the SOMA route's mean GPU utilisation is at least 85%,
  or the profile shows which non-GPU stage bounds it and that stage is at its decode floor.
- The SOMA route from masks has at least twice the throughput of the prototype route (source run
  plus `soma77_extract.py`) on the same clip.
- Keypoints of the native SOMA run equal those of the prototype on the same boxes and options to
  within 0.5 px.

Also:
- The migration runs on a copy of a real session database; existing detection runs get NULL in the
  new columns, and a spike prototype run whose `config_json` names its source run gets
  `source_detection_run_id`. Nothing else changes (row counts and checksums of the other tables).
- The results table of design §3 gets rows for the new loop, with the same columns, material and
  commands.
- C++ tests in the worktree's own build directory (if C++ is touched) and `pytest python/tests/`
  pass.

## Rules

- **Before the first build in the new worktree**, ask the operator to exclude the worktree directory
  from F-Secure scanning, and wait until they confirm. F-Secure blocks freshly linked executables
  (the build then fails with `LNK1104`), and every blocked path has to be unblocked by hand; an
  exclusion added afterwards does not release them. This applies to the Meson build (C++ and its
  tests) and to anything else that writes executables or DLLs into the worktree.
- Never write to a real session or registry database. Every run writes into a copy made outside the
  repository.
- Agent background tasks are killed after about 20 minutes. The acceptance measurements need a quiet
  machine, and their timings are compared with design §3: hand them, and the long soak run, to the
  operator as ready commands with the expected run time. Short smoke tests (a second or two of
  video) you run yourself.
- GPU memory: ViT-H batches next to several decode producers; check the peak and say what it is.
- Commits: cohesive, titled `comp: short description` (`detection:`, `pose:`, `db:`, `ui:`, `cli:`,
  `docs:`), no Co-authored-by line, pre-commit installed in the worktree, tests passing before each
  commit. Stage paths explicitly; never `git commit -a`.
- Update the detection pipeline's `status.md`. Where the code disagrees with the design, change the
  design document to match what was built and say so in your report.

## Report back

What was built, the test results (with output for any failure), the decode-producer choice and its
soak result, the migration check, the commands left for the operator with expected run times, any
deviation from the design, what is left if the stage did not fit, and the branch name.

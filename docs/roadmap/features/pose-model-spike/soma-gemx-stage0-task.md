# Task: stage 0 of the SOMA/GEM-X productization — de-risking measurements

You are working in the posetrak repository on branch `feature/soma77-pose-spike`. Carry out stage 0
of [soma-gemx-productization-plan.md](soma-gemx-productization-plan.md) (section 4, "Stage 0 —
De-risking measurements"): the five measurements M1-M5. Stage 0 measures; it does not build any later
stage. Do not start stage 1 or later work, even where a fix looks small. Record what you find
instead.

## Read first

- `CLAUDE.md`, all of it, especially the data-model invariants, comment rules and git conventions.
- The plan, sections 2-4, and the parts of the designs that each measurement belongs to:
  - M1: [detection pipeline §2-§3](../detection-pipeline/detection-pipeline-design.md)
  - M2: [tracker performance §1-§4](../tracker-performance/tracker-performance-design.md)
  - M3: [keypoint candidates §2 and §7](../keypoint-candidates/keypoint-candidates-design.md)
  - M4: [detection pipeline §4](../detection-pipeline/detection-pipeline-design.md)
  - M5: [keypoint candidates §7-§8](../keypoint-candidates/keypoint-candidates-design.md) and
    `python/tools/gt/`
- [soma77-spike-report.md](soma77-spike-report.md) to find the captures, trials, labelled ranges,
  tracking runs, candidate files and the failure-event table the measurements reuse. Capture data and
  session databases live on the data drive, outside the repository; check the machine's local setup
  notes for the path, and do not assume one.

## The measurements

**M1 Detection profile.** Add per-stage timing (decode, box source, crop and preprocessing,
inference, decode of heatmaps, hand pass, preview encoding, database write) to the detection loop,
behind an environment variable in the style of `POSETRAK_PROFILE_FRAME_STEP`. With it off, the code
path and output must be unchanged. Run a fixed 10 s clip of a five-camera capture through YOLOX +
RTMPose, masks + RTMPose, masks + SOMA (flip test on and off), with the hand pass on and off, and
decode alone. Record per-stage time, a sampling profile of the slowest configuration (py-spy or
equivalent), and GPU utilisation (`nvidia-smi` sampling). Each run writes a new detection run, so run
against a copy of the session database, not the original.

**M2 Tracker baseline.** The 85 s sequence of the spike with the SOMA skeleton, optimized build in
`optbuild/`: single-stage against hierarchical, 120 against 60 Hz. Record wall time, peak resident
memory, per-step timings (`POSETRAK_PROFILE_FRAME_STEP`), `K_active` and the error-state dimension of
each stage, and the failure events of the spike's table. Hierarchical mode needs
`tracker_config_stages` rows (tracker performance §1); state which configuration each run used. Run
against a copy of the session database. These runs take tens of minutes. Do not run them as agent
background tasks, which are killed after about 20 minutes (see the open issue). Write the exact
commands into one script and ask the operator to run it in a standalone shell. You may run short
slices yourself to check the commands.

**M3 Mask filter and multi-view agreement.** From the spike's existing candidate files, compute the
share of keypoint-frames with two or more candidates, before and after the strict tracklet-mask
filter, in the hard windows and elsewhere, per person. Say whether the hypothesis of keypoint
candidates §7 item 1 holds. This is a read-only analysis: prototype scripts are fine, under
`python/tools/prototypes/`.

**M4 Licence question.** Draft the question to NVIDIA: does the DINOv3 License apply to `vitpose.pth`
in `nvidia/GEM-X`, in addition to the NVIDIA Open Model License, for redistribution and for use? Add
the draft to detection pipeline §4 as an open item. Do not contact anyone. The operator sends it and
records the answer.

**M5 Acceptance harness.** Make the ground-truth metric harness in `python/tools/gt/` read
productized storage, and add the same-candidate metric next to the within-3%-of-box-height metric and
the label-fed tracker comparison. Freeze the evaluation set: labelled ranges, failure-event list and
scoring script, all named in one place. The labels stay outside the repository; the harness only
refers to them.

The `kp_candidates` rows do not exist yet; they come in stage 9. The harness reads candidates through
one reader function. That function implements the blob format of keypoint candidates §4.1, is
unit-tested on fabricated rows, and also accepts the prototype's candidate files, so the harness
works now and later without change. Body picks and tracking results are read from the existing
tables.

## Rules

- Open session databases read-only (`mode=ro`). Any run that writes (detection, tracking) goes to a
  copy of the database, made outside the repository; say where the copy is.
- Results go into the design document each measurement belongs to, as a table with the capture,
  configuration, build and command that produced it. Update each touched feature's `status.md`.
  Design documents contain no dates, people's names, planning step identifiers or references to
  chats or reviews.
- Instrumentation and harness code follow `CLAUDE.md`'s comment rules and leave one runnable check
  behind: a unit test for the timing switch's off path, and one for the candidate reader.
- Challenge the plan where a measurement disagrees with it. Section 2 of the plan and the cost model
  in tracker performance §1 are estimates. When a number contradicts them, say so in the design and
  in your report. Do not bend the measurement to fit.
- Commits: cohesive, titled `comp: short description` (`docs:`, `python:`, `tools:`), no
  Co-authored-by line, pre-commit installed, unit tests passing (`pytest python/tests/`; and
  `./run_tests.sh` if C++ changed). Stage paths explicitly. Leave unrelated changes alone: the staged
  `docs/open-issues.md`, the modified prototype scripts, `cohn_db.json`, and `python/tools/iclone/`
  (locally excluded).

## Report back

- One table per measurement, with the main number and where it is recorded.
- For each measurement, what it changes in the plan: scope of stages 2, 3, 6 and 9, the default
  hierarchical configuration, and whether the variable frame rate is worth building.
- What you could not measure and why.
- The script for the M2 runs and the M4 draft, ready for the operator.

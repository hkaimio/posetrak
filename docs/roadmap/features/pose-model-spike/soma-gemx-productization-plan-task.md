# Task: plan and design the productization of the GEM-X/SOMA detector

You are running a planning and design round. You produce documents, not code. Implementation is a
later, separate step that will work from your output, so the documents must stand on their own for
a reader who has not seen this round.

## Goal

Turn the findings of the GEM-X/SOMA-77 pose-model spike into a staged, reviewable plan with a design
for each feature, ready to be implemented in the main code line.

## Read first, in this order

1. `soma-gemx-productization-brief.md` — the owner's view of the outcome: what to build.
2. `soma-gemx-productization-review-notes.md` — a critical review of the brief with additions, and
   a pointers section listing the code and design documents to read. Treat the review's
   disagreements as open questions, not as decisions.
3. `soma77-spike-report.md` (especially sections 5, 10, 11, 12) and the two independent assessments
   beside it.
4. `docs/open-issues.md`, `docs/data-model-and-storage.md` section 3, and `CLAUDE.md`.

Then read the code and designs the pointers section names, as far as each feature needs.

## Decisions already made (do not reopen)

- SOMA-77 keypoints become the default pose model. The existing detector stays available.
- It must work with and without a segmentation mask.
- Several candidates per keypoint are supported in the database and switched on per detection run.
  The brief says the candidate approach is not universally proven better, so it is optional.
- The cross-camera criterion is named "multi-view agreement".
- Iterative tracking (snapshots, restart over a changed range, windowed smoothing) is likely a
  separate feature, not part of the first release.
- Segmentation UI improvements are a separate feature.
- The hand refinement pass is to be investigated during productization, not dropped.

## Questions you must answer with evidence from the code or from measurement

Where a question needs a measurement you cannot take, say what would be measured and how, and plan
it as an early step.

1. **Performance.** Where does the detection time go, for both pose models and for the segmentation
   route? Propose a profiling plan, and the pipeline changes it implies (single pass with a
   pluggable box source, feeding the GPU, variable frame rate when person boxes are far apart).
2. **Tracker cost.** What does a tracking run cost with the hierarchical configuration against a
   single-stage one for a 77-keypoint skeleton, and which cheaper options (lower tracker rate,
   smaller state) come before snapshot-based iterative tracking? Memory use of the smoothing step
   is part of this (a 10 000-step run reached about 18 GB).
3. **Skeleton and model coupling.** How are the SOMA marker offsets stored and shipped, how are
   they bound to a pose model, and what happens to existing skeletons and tracking runs?
4. **Candidate storage and assignment.** A schema for several candidates per keypoint and for
   tracklets, and how assignment reuses the tracker's existing dot and tracklet assignment
   machinery. Cover both the markerless and the marker implementations.
5. **Person assignment without segmentation.** The plain detector gives hundreds of short track
   fragments per camera. What assignment step is needed, and how does it relate to segmentation?
6. **Hand refinement.** Why is it tied to the segmentation pipeline, and can it work without?
7. **Model license.** What do the terms of the model weights allow, and what follows for shipping
   it as a default? Report what you find in the repository and the model's published terms; do not
   guess.
8. **Mask versioning.** A design that fits the append-only data model.
9. **Keypoint editing.** Tracklets as first-class objects: reject, force, split, group operations,
   visualization. Stage it so that the first stage needs only the storage decided in question 4.
10. **Calibration and robustness items** from the review (calibration-box tool failure modes, room
    tags as scene markers, safe camera relabeling, intrinsics quality, database locking, joint
    limit table for the default skeletons). Decide which belong to this productization and which
    are separate features.

## Deliverables

- `soma-gemx-productization-plan.md` in this directory: the staged plan. For each stage: scope,
  dependencies, acceptance criteria, measurable where possible, and risks. Put cheap, de-risking
  measurements first. Include an explicit list of what is deferred and why.
- One design document per feature under `docs/roadmap/features/<feature>/`, following the layout of
  the existing feature directories (design document plus status document). Reuse an existing
  directory where the feature already has one.
- A combined architecture and schema summary, in this directory: the final software architecture
  (components, data flow from detection through assignment and tracking to editing) and every
  database schema change, merged across all proposed features so that overlaps and conflicts
  between the per-feature designs are resolved in one place. Each changed table lists all its
  columns, as `CLAUDE.md` requires, and the migration order is stated.
- A release split: the implementation may be delivered in phased releases. Say which stages form
  each release, what each release leaves in a usable state, and which schema changes each needs, so
  that an early release does not force a later schema rewrite.
- A short list of decisions that need the owner, each with options and your recommendation.

## Constraints

- Follow `CLAUDE.md`: schema changes document every column of the touched tables; source and design
  documents contain no dates, people's names, planning step identifiers or references to chats or
  review discussions; development history belongs in commit messages.
- Do not modify data under any session database, and do not run long jobs. Read databases
  read-only. If a measurement needs a long job, plan it instead of running it.
- Do not write implementation code. Small throwaway scripts for reading code or data are fine and
  are not deliverables.
- Challenge the brief and the review notes where the code or the evidence disagrees, and say so in
  the plan.
- Commit documents in cohesive commits using the title format `docs: short description`.

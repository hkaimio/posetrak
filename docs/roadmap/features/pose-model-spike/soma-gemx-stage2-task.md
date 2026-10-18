# Task: stage 2 of the SOMA/GEM-X productization — pose model and skeleton binding

Implement stage 2 of [soma-gemx-productization-plan.md](soma-gemx-productization-plan.md) (section 4)
as designed in
[pose-model-skeleton-binding-design.md](../pose-model-skeleton-binding/pose-model-skeleton-binding-design.md).
Work in a separate git worktree on a new branch made from `main`. The spike branch
`feature/soma77-pose-spike` is a reference and is not merged (plan section 5): take over, or
re-implement, the spike code this stage needs. Other work uses the main work area, so do not touch
its build directories or its databases.

## Read first

- `CLAUDE.md`, all of it: build and test commands, schema documentation rules, comment rules and git
  conventions.
- The binding design, all of it, and its `status.md`.
- Plan section 4, stages 2, 3 and 6 (what stage 2 hands to them), and section 5.
- [soma-gemx-architecture-and-schema.md](soma-gemx-architecture-and-schema.md): the
  `pose_sequence_keypoints` table and the migration table of section 4.
- [soma77-spike-report.md](soma77-spike-report.md) sections 5.1 and 5.2: the offsets, the decoding
  convention and the held-out figures this stage must reproduce.
- The spike code, read with `git show feature/soma77-pose-spike:<path>`:
  `python/posetrak/detection/backends_soma77.py`, `python/tests/test_soma77_backend.py`,
  `python/tools/prototypes/build_soma77_skeleton.py`, `calibrate_soma77_offsets.py` and
  `soma77_extract.py`. The main work area has uncommitted edits of `soma77_extract.py`; use the
  committed version.

## Scope

1. **Pose model registry** (design §2.1): `posetrak/detection/pose_models.py` with `PoseModelSpec`
   for `soma77`, `coco133` and `hand21`; the viewer and finalisation read it instead of their own
   copies of the names.
2. **SOMA-77 estimator on main:** take over `backends_soma77.py` and its test from the spike, using
   the registry's names. It already decodes at cell centres (index + 0.5); keep that and the test that
   covers it. Registering it as a batched estimator and the frame loop belong to stage 3.
3. **Manifest per observation source** (design §2.2): `pose_sequence_keypoints.obs_source` and the
   new primary key, finalisation writing manifest rows for every new sequence (SOMA body, COCO-133
   body, hand-refinement rows), and the reader binding by manifest per source with the implied COCO
   map only for sources without manifest rows. Model-qualified landmarks (design §2.3), with bare
   names still matching for spike data.
4. **Schema:** the plan calls this change version 66; take the next free session version on `main`
   (65 now) and treat it as provisional. Only the `pose_sequence_keypoints` change belongs here;
   `tracker_configs.active_joint_groups` is stage 6's. Document every column of
   `pose_sequence_keypoints` and add it to `DOCUMENTED_TABLES`.
5. **Layered skeleton** (design §2.6, §2.7): hierarchy, `joint_limits.yaml`, marker-set files
   (`coco133`, `soma77`, `hand21`) and dimensions as files; `posetrak skeleton compose`, which also
   covers `add-pose-model` for a user's skeleton; the SOMA marker set with the 29 body markers plus
   `*HandIndex1` / `*HandPinky1` on the hand segment in the `main` group (design §2.4), offsets from
   the spike's calibration as `catalog/pose-models/soma77.marker-offsets.yaml`; the default skeletons
   regenerated from the layers; the hinge test.
6. **Wrist deviation:** examine why the tracked wrist deviation sat at its limit in the spike (design
   §2.6, last paragraph), and report the cause you find. Do not loosen the limit to hide it.
7. A tracking run whose skeleton binds no marker of the sequence's manifest fails with a message
   naming the missing pose model (design §2.5).

Out of scope: the UI action "add SOMA-77 markers" (record it as not done in `status.md`), skeleton
layers in the registry, per-subject offset calibration, the hierarchical default configuration
(stage 6), and anything of stage 3.

## Acceptance

- The design's section 3 tests pass: dual-set reader test with unclaimed finger slots, hand rows
  bound through their own manifest, hinge test failing on the old limits and passing on the new ones,
  composition with the old limits reproducing today's default apart from key order.
- The reference capture's held-out window, tracked with the regenerated default skeleton, gives leg /
  arm / head errors within 0.3 cm of the spike's 2.4 / 4.0 / 2.6 cm. Find the capture, the SOMA
  detection run, the reference run and the window in the spike report and the spike's status files.
  Track on a copy of its database with an optimized build in the worktree's own build directory.
- The migration runs on a copy of a real session database: existing manifest rows get `obs_source =
  'body'`, and nothing else changes (row counts and checksums of the other tables).
- C++ tests in the worktree's own build directory and `pytest python/tests/` pass.

## Rules

- Never write to a real session or registry database. Test against copies made outside the
  repository.
- Agent background tasks are killed after about 20 minutes. Any run longer than about 15 minutes goes
  to the operator as a ready command, with the expected run time. The operator has a quiet window of
  a few hours for those runs.
- Keep C++ builds and test runs short where possible; the machine is shared.
- Commits: cohesive, titled `comp: short description` (`pose:`, `db:`, `cpp:`, `cli:`, `catalog:`,
  `docs:`), no Co-authored-by line, pre-commit installed in the worktree, tests passing before each
  commit. Stage paths explicitly; never `git commit -a`.
- Update the binding design's `status.md`. Where the code disagrees with the design, change the
  design document to match what was built and say so in your report.

## Report back

What was built, the test results (with output for any failure), the held-out errors against the
spike's figures, the migration check, the wrist-deviation finding, any deviation from the design,
commands left for the operator, and the branch name.

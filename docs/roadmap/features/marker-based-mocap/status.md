# Marker-based mocap — status

- **2026-08-31** — First real GUI-launched tracking run for an object threw
  `apply_keypoint_edits: edit blob has 20 keypoints, expected 133`. Root
  cause turned out to be an installed binary, not a live code bug: the
  tracker binary `run_tracker()` actually invokes (`~/.posetrak/
  posetrak-tracker.exe`, preferred over `optbuild/` per
  `default_binary_path()`) was dated 2026-08-23 -- a week before *any* of
  this feature's C++ work landed, so it still had the pre-fix
  `SessionReader::load_observations()` that matched a group's base row by
  literal `source == "body"`. Since an object's real source is `'markers'`,
  every single frame's base row failed that check, so every frame fell
  into the "no base row for this group" placeholder branch -- silently
  producing an all-zero-confidence, 133-wide (`kFullBodyNKp`, a COCO-133
  literal) array for every frame; the one frame with a real keypoint edit
  (correctly sized to this object's own 20-keypoint width) then failed the
  width check against that wrong placeholder. Rebuilt `optbuild` and copied
  the fresh binary over the stale installed one to fix Harri's immediate
  block.

  While diagnosing, generalized that placeholder branch anyway (defensive,
  not the actual trigger here since the *current* code's body-row match
  already works correctly for a real 'markers' row): it still hardcoded
  `kFullBodyNKp` regardless of sequence, so a genuine object ghost-frame
  (edit-only, e.g. a hand-overlay-shaped row from some future feature)
  would hit the same wrong-width bug the stale binary did, just via a
  different path to the same "no base row" branch. Now uses the sequence's
  own manifest width (`pose_sequence_keypoints` row count) when it has one,
  falling back to `kFullBodyNKp` only for a person sequence (no manifest at
  all) -- exactly the same "generalize the assumption, don't add a second
  hardcoded case" pattern as every other 'body'-literal fix in this
  feature. New regression test in `test_session_reader.cpp` forces the
  null-base-row branch for an object sequence (synthetic `hand_l`-sourced
  row) and confirms the manifest width applies instead of 133.

  **Lesson for later real-data testing**: `~/.posetrak/posetrak-tracker.exe`
  is a separate, manually-installed copy that silently wins over both
  `optbuild/` and `builddir/` (see `default_binary_path()`) and has no
  install script to keep it current -- worth checking (or just re-copying)
  after any C++ tracker change, not just rebuilding `optbuild`.

- **2026-08-31** — Added the GUI entry point 1f's own validation left
  missing: `ObjectPanel` now has a "Tracking runs" list and a "Run
  tracker…" button, opening new `ObjectRunTrackerDialog`
  (`app/pose/run_tracker.py`). Deliberately not `RunTrackerWidget`/
  `RunTrackerDialog` extended to cover objects too: that widget is built
  entirely around multi-person tracking (a trial → people table,
  cross-person coupling, hierarchical child stages), none of which applies
  to a rigid prop -- one object is always exactly one sequence, one
  skeleton, one track (`person_id=0`). The new dialog is the same idea
  stripped to what an object needs: skeleton picker, the existing generic
  `TrackerConfigWidget`, time range, Run -- reusing the exact same
  execution path (`run_tracker()`/`_TrackerThread`) the person single-
  sequence case already uses, so no backend or CLI changes were needed.
  Results are ordinary `tracking_runs` rows, so the session tree's existing
  generic `TRACKING_RUN` nodes (already wired under `_add_object_tracks`)
  and `TrackingRunPanel` (already generic, keyed only by `run_id`) picked
  them up with no changes there either. New tests in
  `test_object_run_tracker_dialog.py` (new) and `test_object_panel.py`.

  Harri also asked whether a per-detection-run "Finalise" step should be
  needed at all, and floated a bigger direction: a "Run tracker" button on
  the Trial page eventually becoming the standard place to launch tracking
  across a chosen set of detection runs (mixed person/object), rather than
  per-sequence panels. Both are answered/noted, not built: finalisation for
  new marker runs is already automatic (previous entry); the trial-level
  launcher is a real, separate design question (which detection runs to
  include, how mixed person+object runs interleave) worth its own pass
  once more than one or two object types exist in practice. The
  `ObjectRunTrackerDialog` execution plumbing added here is not throwaway
  either way -- a future trial-level launcher would call into the same
  `run_tracker()` path per selected sequence.

- **2026-08-31** — Fixed a gap Harri hit doing his own manual pass over 1c:
  running marker detection for an object left it stuck. `finalise_object_to_db`
  (1d) existed but nothing ever called it -- `RunDetectionDialog._on_finished`
  linked the run to a trial and stopped there for both person and object runs,
  and the session tree's detection-run node (`StandaloneRunPanel`) always
  built a `StitcherPanel`, the person track-to-person assignment UI, which has
  no person tracks to show for an object run and no way to proceed. Fixed
  two ways: (1) `RunDetectionDialog._on_finished` now auto-finalises a marker
  run the moment its job completes -- consistent with §7.1's 1d/1e ordering
  note that an object has no stitching decision to make, so finalising is the
  only remaining step; (2) `StandaloneRunPanel` now branches on
  `detector_type`, showing a small object summary (a "Finalise" button if a
  run somehow reaches it unfinalised -- an older run, or a failed
  auto-finalise -- otherwise "Review corners…" straight into `ObjectPanel`)
  instead of falling through to `StitcherPanel`. New tests in
  `test_run_detection_dialog.py` and `test_standalone_run_panel.py` (new).

  While verifying this, found `python/tests/db/test_observation_edits.py::
  test_edit_marks_outlier_zeroes_confidence` fails on its own, independent of
  this fix (confirmed against the pre-fix commit too): `db_cache.py`'s
  `_apply_edit` unconditionally overwrites a slot's x/y from the edit blob
  even when the edit only marks the slot an outlier, instead of leaving the
  original position untouched as the test (and `read_observations_with_edits`'s
  own docstring) expect. That overwrite logic dates to the June "Phase 7 —
  ghost-frame keypoint placement" commit, well before marker-based-mocap;
  the regression test exposing it was added in this feature's own 1d work
  but evidently never actually verified failing at the time. Not fixed here
  -- unrelated to this fix and to marker-based-mocap generally (it would
  affect person keypoint editing the same way) -- flagged to Harri to
  prioritise separately.

- **2026-08-30** — 1f (tracker: multi-source load + rigid init) built and
  validated end-to-end on real data — **phase 1's finish line reached**.
  `Skeleton` gained `input_tracks_` (`InputTrack{id, type}`) and `Marker`
  gained `track`/`landmark` fields (design §5.1), parsed from `input_tracks:`/
  `track:`/`landmark:` in skeleton YAML. `SessionReader::load_observations()`
  now resolves keypoint-blob slots via a `pose_sequence_keypoints` manifest
  (landmark name → marker index) before falling back to the legacy COCO-id
  map, so an object skeleton with no `coco_id` at all still loads. Doing
  this surfaced the same bug class as the Python-side fix above, in C++:
  the group's base/primary row was picked by literal `source == "body"`,
  so a `'markers'`-source object sequence's own row was never recognised
  as primary and every keypoint was silently dropped. Fixed the same way —
  generalised to "whichever row isn't a recognized overlay" via the
  existing `hand_base_idx`/`split_source` helpers, rather than adding a
  second hardcoded literal; behaviour for existing person data (source
  always `'body'`) is unchanged. Kept as one commit with the manifest-load
  feature rather than split out like the Python fix: unlike that case, this
  bug has no test or manifestation independent of the new manifest-resolution
  path (object skeletons carry no `coco_id`, so there was no data shape that
  could exercise `load_observations`'s base-row selection over a non-'body'
  source before this feature existed).

  Added `Tracker::initialize_rigid_body()` (algorithms §4.2): for a
  root-only skeleton (no active joint below the root), computes rest-pose
  FK marker positions, matches them against the frame's triangulated world
  positions, rejects a collinear marker layout (SVD second singular value
  ≤ 1e-4 m — matches the design doc's own deferral of that case), fits a
  closed-form Kabsch/Umeyama rigid transform (`Eigen::umeyama`,
  `with_scaling=false`), and rejects if RMS residual exceeds the new
  `rigid_init_max_residual_m` config field (default 0.02 m). `initialize()`
  routes to it automatically when the skeleton has no non-root active DOF.

  Validated against real capture data (`ukemi-tommi-20260509.db`, capture
  `ecf8c983-2e0a-4906-96ca-73207a71ad7c`, the ArUco-marked calibration box):
  real per-camera marker detections are sparse and independently timed, so
  the CLI's default init search (a narrow ~1-tracker-frame window at
  `start_time`) rarely lands on an instant with ≥3 markers seen by ≥2
  cameras — needed a `--start-time` scan to find one (`t=21.1s` worked: 4
  markers, RMS residual 0.0086 m). Once initialized, a full tracking run
  (`--start-time 21.1 --end-time 28.3`) tracked 344/863 steps with 0 lost,
  writing a smooth, bounded 6-DOF trajectory to `root_pose.csv` (no
  NaN/Inf, quaternions normalized, ~0.4 m of real object motion over the
  window, covariance condition number 9–409). Mean reprojection error
  (~29 px) is higher than ideal and worth a closer look before relying on
  this for accuracy-sensitive work, but is not a phase-1 blocker — the
  validation criterion here is a plausible tracked trajectory, not
  calibration/detection accuracy. The narrow-init-window characteristic is
  a real UX gap for sparse marker data (worth a follow-up to widen or
  auto-scan the CLI's init search) but is out of scope for this phase.

- **2026-08-30** — 1d (finalisation) and 1e (ObjectPanel review) built and
  merged. Building 1e surfaced a real bug in shared code, not specific to
  markers: `merge_observation_sources`/`infer_body_width`/
  `update_single_keypoint_edit` hardcoded `source='body'` throughout. For
  any sequence whose real source is never 'body' (an object's 'markers'
  source), the moment even one edit existed anywhere in the camera,
  `merge_observation_sources`'s "ghost frame → synthesize zero body"
  fallback fired regardless, silently discarding every real, untouched
  keypoint slot's data. Fixed by generalising all three functions to a
  `primary_source`/`source` parameter (default unchanged, so every
  existing person-panel call site is unaffected) rather than working
  around it in ObjectPanel — the whole point of finalising before
  reviewing (previous entry) was reusing this machinery genuinely, not
  papering over its person-only assumption.

- **2026-08-30** — Implementation progress: 1a (ArUco detection layer),
  1b (skeleton generator), and 1c (capture-object plumbing + GUI
  marker-detection run mode) built, tested against synthetic fixtures and
  real capture data (calibration box, `ukemi-tommi-20260509.db`), and
  merged. While starting 1d, found that review-before-finalisation (the
  original 1d→1e order) doesn't fit the codebase: pre-finalisation review
  for a person is track-to-person *stitching*, a real decision with no
  per-frame correction path anywhere in the project; per-frame correction
  only exists post-finalisation, via `pose_observation_edits`. Since an
  object's own phasing already established "no stitching step," swapped
  the order (§7.1 now runs 1d finalisation, then 1e review) so ObjectPanel
  reuses that existing mechanism directly instead of building a parallel
  one for raw `detection_keypoints`. Confirmed with Harri before proceeding.

- **2026-08-30** — Phase 1 broken into six independently-buildable
  sub-phases (design §7.1), each with its own validation check: detection
  layer (1a), skeleton generator (1b, parallel to 1a), capture-object
  plumbing (1c), ObjectPanel review (1d), finalisation + manifest (1e),
  tracker multi-source load + rigid init (1f, phase 1's actual finish
  line). Requested because phase 1 as originally scoped bundled DB schema,
  Python detection, Python finalisation, C++ tracker, and two GUIs into one
  slab with a single end-to-end validation criterion.

- **2026-08-30** — Second review round: UC1 phasing restructured so
  anonymous/reflective dots on props are pulled into the first iteration
  alongside ArUco, instead of waiting for UC2 (Harri: real props already
  combine ArUco + reflective dots, and a dots-only prop is also a valid
  configuration). Design §7 now runs seven phases — ArUco prop (1),
  dot-only prop (2), person + prop together (3), multiple mixed props +
  person — UC1 complete (4) — before UC2's identified (5) and anonymous
  (6) person markers, then moving camera (7). Split driven by labeling
  difficulty, not marker type: rigid-prop dot labeling is a single-body
  problem (algorithms §3.4 tier 1), so it doesn't need the cross-subject
  `MarkerAssociator` machinery UC2 requires until multiple marked bodies
  can compete in phase 4/6. Added the previously-missing cold-start
  procedure for a body with no coded anchor: unlabeled rigid-template
  registration by pairwise-distance RANSAC (algorithms §4.1). UC2 (person
  markers) is confirmed as the next project after UC1, not interleaved
  with it.

- **2026-08-19** — First review round: Harri's inline comments on the
  design addressed. Main outcome: new design §5.2 splits session-scoped
  marker attachments out of the skeleton into a composed *marker
  attachment set* document (so person-scale improvements from marker
  sessions propagate to markerless skeletons), plus clarifications on
  definition/capture-object/skeleton roles (§4.2), symmetry-axis marking
  (§6.1), and global cross-subject assignment for uncoded markers
  (§6.2, algorithms §3.3).

- **2026-08-19** — Design written from the brief + codebase analysis:
  [marker-mocap-design.md](marker-mocap-design.md) (requirements, data
  model, architecture, UX, phasing) and
  [marker-mocap-algorithms.md](marker-mocap-algorithms.md) (detection,
  measurement model, anonymous-marker association, rigid init, offset
  calibration, camera-drift monitoring). Consolidates and supersedes
  `docs/aruco-prop-tracking-design.md` where they conflict; builds on
  `pose-detect-improvements/marker-detection-analysis.md` and
  extrinsics-improvements §10 marker-body infrastructure. Not yet
  reviewed; no implementation started.

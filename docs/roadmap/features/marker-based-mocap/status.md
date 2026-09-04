# Marker-based mocap — status

- **2026-09-04** — Real end-to-end validation: dot-assisted tracking closes
  the fast-motion gap that motivated this whole feature. Ran the tracker
  against the real sword capture three ways, all on
  `ukemi-tommi-20260509.db`:

  1. Original ArUco-only baseline (older detection run,
     `tracking_runs.id` `319ebb30...`): 52.3% tracked.
  2. A fresh detection run with dot detection enabled on the two ring-lit
     cameras, tracked with the new ArUco+dots skeleton (`ef3d451b...`,
     `tracking_runs.id` `88b86bc0...`): **73.3% tracked** (4849/6618
     steps), 1466 explicitly lost.
  3. Control: the *same* fresh detection run's sequence, tracked with the
     old dots-free skeleton (`2c93603c...`, `tracking_runs.id`
     `c8da1eac...`) to rule out the improvement being an artifact of
     re-running detection rather than of the dots themselves: 52.3%
     tracked (3458/6618) -- matches the original baseline almost exactly,
     confirming runs 1 and 3 are a clean apples-to-apples pair and the
     ~21-point gain in run 2 is attributable to the dots.

  Checked the specific 53.6-55.1s window previously found to be a
  complete tracker freeze (zero `tracking_results` rows, not
  constant-velocity coasting as the design docs had assumed): the control
  run confirms that finding again (only 6 of ~150 possible steps in that
  window have any row at all). The dots-enabled run has a row for every
  one of the ~150 steps in that same window -- 108 with real inlier
  observations (avg. ~2 per step, consistent with only a couple of dots
  being visible at a time during a fast swing) and 40 honestly reported
  as lost rather than silently skipped. This is the direct answer to the
  question that started this phase: dots keep the tracker alive and
  producing real updates specifically during the gap ArUco alone
  couldn't cover, not just improving some aggregate statistic elsewhere
  in the capture.

- **2026-09-03/04** — Wired the shared dot-assignment phase into both real
  per-frame tracking loops (`run_track_from_db()`'s raw loop,
  `MultiPersonTracker::run()`) -- the one piece that was still missing
  between "the resolution math works" and "the tracker can actually use
  dots at all" (see
  [dot-assignment-architecture-design.md](dot-assignment-architecture-design.md)
  §5.2). Preceded and motivated by a real manual dot-geometry calibration
  session for the sword (7 dots, all real, cross-checked -- see the
  session log for the full back-and-forth) and registering it properly:
  new `marker_body_definitions` row (id `a1e503f0...`, ArUco + all 7 dots),
  sword-bokken's `capture_objects` row repointed at it, and a real
  generated skeleton (id `ef3d451b...`) confirmed to route dots onto their
  own `unlabeled_points` track correctly (the fix from the entry below).

  New `Skeleton::has_unlabeled_points_track()` -- the caller-facing
  yes/no a subject-building step needs before ever constructing a
  `Tracker`, mirroring the per-marker check
  `Tracker::predict_dot_slot_predictions()` already did internally.

  `PersonContext` gained `has_dot_track` and `unlabeled_candidates`
  (loaded once via `SessionReader::load_unlabeled_candidates()`, only
  when `has_dot_track` is true -- every existing person/object sequence
  pays nothing here). Two new pure helpers,
  `person_context_step_window()` and `bucket_candidates_by_camera()`,
  factor out the per-step time window and per-camera bucketing so neither
  call site duplicates them.

  New `step_person_context_predict()`/`step_person_context_update()`
  sibling pair to `step_person_context()` itself -- **not** a refactor of
  it into a predict+update wrapper, deliberately: `step_person_context()`
  still checks "any observations this step?" *before* deciding whether to
  predict at all (skip entirely, no write, exactly as today), which a
  dot-bearing step can't do -- predicting is required just to *find out*
  whether a dot will resolve. So `step_person_context_update()` always
  calls `Tracker::update_step()` and lets its own "insufficient
  observations" handling cover the empty case safely, rather than trying
  to replicate the early-return optimization on top of an already-run
  predict. This is a deliberate, narrow, documented difference scoped to
  only the steps that actually go through it -- every dot-free step, and
  every dot-bearing subject's step with nothing queued that particular
  frame, keeps calling the untouched `step_person_context()` exactly as
  before.

  Both real call sites now do the three-pass shape (predict every
  dot-bearing-this-step subject, resolve jointly across all of them via
  `resolve_shared_dot_assignment()`, then update each with its own
  resolved share appended to its existing observations/anchors) *only*
  when at least one participating subject actually has candidates queued
  that step; every other subject and every other step is untouched.
  `MultiPersonTracker::run()`'s version merges every participating
  subject's own candidate list into one combined pool per camera before
  resolving -- the known, explicitly-deferred limitation from §5.4 (no
  real de-duplication across subjects' own detection runs) still applies,
  unchanged, since no real multi-subject-with-dots capture exists yet to
  need it.

  Test coverage: the full existing C++ suite (including
  `test_multi_person_tracker.cpp`'s bitwise-identical-to-single-person
  regression tests) passed **unchanged** after this wiring -- real
  confirmation that no dot-free call path was disturbed. Added direct
  unit tests for the two new pure helpers. Did not build a new synthetic
  multi-person-with-dots DB fixture for the three-pass shape itself
  (the design doc's own suggested next test) -- the underlying resolution
  math and the Tracker-level integration are already covered elsewhere
  (`test_dot_assignment.cpp`, the `test_tracker_integration.cpp`
  posterior-equivalence test), and a real end-to-end validation against
  the actual sword capture was imminent and clearly more valuable than a
  synthetic fixture built just for this; revisit if real testing surfaces
  something that needs isolating.

- **2026-09-02** — Started calibration-time dot geometry (see
  [reflective-dot-detection-design.md](reflective-dot-detection-design.md)
  §3.1), then pivoted from automatic to manual annotation after real-data
  testing surfaced a real correctness gap the automatic approach couldn't
  cheaply clear. Also fixed a real, separate bug found along the way: the
  skeleton generator was routing reflective dots onto the *same* input
  track as coded-marker corners.

  **Bug found and fixed**: `posetrak.skeleton.marker_body_to_skeleton
  .generate_prop_skeleton()` bound `reflective_dot` entries to the same
  `labeled_points`-type track as ArUco corners (`prop_markers`) --
  predating the dot-assignment design, never updated once that design
  settled on a separate `unlabeled_points` track type. A skeleton
  generated from *any* existing dot-bearing marker body would never
  actually have engaged the shared dot-assignment machinery built earlier
  today, since nothing checks for an `unlabeled_points` track on
  `prop_markers`. Fixed: dots now get their own `prop_dots`
  (`unlabeled_points`) track, only emitted when the body actually has
  dots (same for `prop_markers` and coded markers). Updated the one
  existing test that had encoded the old (wrong) behavior as expected,
  and added a mixed-body test covering both tracks at once.

  **Automatic calibration attempt**: extended
  `tools/calibrate_rigid_marker_body.py` with `--detect-dots`, reusing
  the same reference-marker co-occurrence mechanism Phase A/B already
  validated for ArUco corners, restricted to instants where >=2 cameras
  each saw exactly one dot candidate (avoiding the harder general
  multi-view correspondence problem -- real GoPro footage has ~39% of
  frames with exactly one candidate per camera vs. ~11% with more than
  one, so this restriction still looked viable on paper). Running it
  against the real "Weapon test 2026-08-20" capture (full ~66s range, all
  6 cameras, 20 minutes of real decode) caught a real bug before it could
  reach production: "each camera saw exactly one candidate" does not mean
  those candidates are the *same physical point* -- two unrelated bright
  spots can each be their own camera's only candidate. The first real run
  triangulated a "dot" over 3 meters from the sword. Added a reprojection-
  error check (reject a triangulation that doesn't reproject correctly
  into every contributing view) and re-ran; the result was still
  implausible, confirming `marker-detection-analysis.md`'s own original
  recommendation (verify against a third view, not just two) was right
  and the two-view-plus-reprojection shortcut this round tried isn't
  strong enough. On top of that, genuine reference+dot co-occurrence in
  this specific capture is already sparse (41 of 680 buckets across the
  full real range), so a properly-strict (3-view) requirement would
  likely yield close to zero usable samples here regardless.

  **Decided (Harri): manual annotation for this prop now, automatic
  calibration remains a real longer-term goal.** New
  `tools/annotate_dots_manually.py`: for a handful of human-picked
  timestamps where a dot is known to be visible in >=2 cameras, solves
  the reference marker's pose from that instant's own ArUco detections
  (same mechanism as the automatic path), shows each camera's frame via
  OpenCV for the user to click the dot (or skip), and triangulates with
  the identical reprojection-checked `triangulate_point_multi_view()` the
  automatic path uses -- human-confirmed correspondence sidesteps the
  correctness problem entirely, so only a handful of good instants are
  needed per dot rather than the many samples an automatic approach needs
  to average out false positives. Reads an existing (ArUco-calibrated)
  marker body YAML and writes a new one with the manually-triangulated
  dots appended, same output format either way.

  Test coverage: `triangulate_point_multi_view()` (recovers a known point
  from synthetic multi-view observations, rejects a simulated
  cross-camera false match, needs >=2 views, ignores unknown camera ids)
  and `cluster_dot_samples()` (separates distinct dots, merges within
  tolerance, empty/single-sample edge cases) in
  `test_calibrate_rigid_marker_body.py`; `write_marker_body_yaml()`'s
  round-trip in `test_annotate_dots_manually.py`. The interactive
  click-and-triangulate workflow itself needs a real person clicking a
  real window -- not something this session's own tools can drive, so
  that part is unvalidated pending Harri actually running it.

- **2026-09-02** — Closed the loop on the central claim the whole
  dot-assignment design rests on (see
  [dot-assignment-architecture-design.md](dot-assignment-architecture-design.md)
  §1: `UnscentedKalmanFilter::update()` never changes, because it only ever
  consumes already-labeled `Observation`s regardless of how they got
  labeled). New integration test in `test_tracker_integration.cpp`, built on
  the existing rigid-prop fixture: adds one `unlabeled_points` marker
  directly to the loaded skeleton (no fixture-file change needed --
  `Marker::track` was already an opaque, unvalidated string at load time),
  drives one real `Tracker` through `predict_step()` ->
  `resolve_shared_dot_assignment()` -> `update_step()` for a frame with both
  ordinary labeled markers and one anonymous dot candidate, and checks the
  resulting posterior state and covariance against a second `Tracker` fed
  the identical data as a single pre-labeled `Observation` list through the
  ordinary `track_frame()`. Bit-identical (1e-9 tolerance) on both.

  This is deliberately scoped narrower than "wire the shared phase into the
  real per-frame loops" (`run_track_from_db()`'s raw loop,
  `MultiPersonTracker::run()`) -- that's substantial, separate plumbing work
  the design doc itself calls out as such (§5.2/§9/§11), and it can only be
  exercised synthetically right now anyway: no real dot-bearing capture
  exists yet (gated on the separate calibration-time phase, C1, which
  hasn't been built). This test confirms the piece that actually matters
  today -- the resolution math and the Tracker seam it plugs into behave
  correctly together -- without touching the orchestration loops every
  current real tracking run (dot-free) already depends on.

- **2026-09-02** — Built the shared dot-assignment phase itself: one
  combined Hungarian solve per camera across every participating tracked
  subject's candidate reflective-dot predictions, so a candidate can only
  ever be claimed by one subject -- the actual double-claim problem the
  whole shared-phase redesign (see
  [dot-assignment-architecture-design.md](dot-assignment-architecture-design.md))
  exists to fix. Not yet wired into either real per-frame loop
  (`run_track_from_db()`'s raw loop, `MultiPersonTracker::run()`) -- that's
  separate, later work; this is the resolution logic itself, callable and
  fully tested on its own.

  New `cpp/{include,src}/posetrak/tracking/dot_assignment.{hpp,cpp}`, split
  into two layers mirroring this codebase's existing
  `update_contact_pairs()`/`build_cross_person_anchors()` vs.
  `MultiPersonTracker::update_contact_gate()`/`build_anchor_observations()`
  split for the structurally analogous cross-person case:

  - `resolve_dot_assignment()` -- the pure resolution core. No
    Tracker/skeleton/camera access at all: takes each subject's
    already-computed `MarkerPrediction`s (camera → marker → prediction) and
    the frame's candidate pool (camera → candidates), builds one cost
    matrix per camera (rows = that camera's candidates, columns = the union
    of every subject's dot-slot predictions for that camera), solves via
    the existing Hungarian solver with the configured Mahalanobis gate, and
    returns each subject's own resolved `Observation`s. Being Tracker-free
    is what makes it directly testable against fabricated predictions and
    candidates, the same reason the cross-person functions above are pure.
  - `resolve_shared_dot_assignment()` -- a thin wrapper that calls
    `Tracker::predict_dot_slot_predictions()` per subject per camera and
    delegates to the pure core. This is the shape a future orchestrator
    actually calls; deviated from the design doc's original suggestion to
    put it in `multi_person_tracker.hpp`/`.cpp` -- neither function needs
    `PersonContext`/`MultiPersonTracker` machinery, and a dedicated file
    matches this codebase's existing one-concept-per-file pattern for the
    assignment/prediction seams (`assignment.hpp`, `marker_prediction.hpp`)
    this builds directly on top of.

  Resolved `Observation`s reuse the same near-zero `crop_scale` convention
  already established for ArUco corners and dot candidates at write
  time -- a candidate's centroid comes from thresholding the full-resolution
  frame directly, not a fixed-input-resolution network, so calibration
  error alone should dominate the noise model.

  Test coverage (`test_dot_assignment.cpp`, new, 10 cases): straightforward
  single-subject matches and gate rejection first, then the two cases that
  actually matter -- a clearly-closer-fit scenario confirming the losing
  subject gets nothing (not a forced worse pairing), and a genuinely
  equidistant/ambiguous candidate confirming exactly one subject wins, via
  logical XOR, never both and never neither. Plus multi-marker and
  multi-camera independence checks, and one test wiring
  `resolve_shared_dot_assignment()` against two real rigid-body `Tracker`s
  end to end (not fabricated data) to confirm the Tracker-calling glue
  itself, not just the math. Full C++ suite green afterward.

- **2026-09-02** — Built the detection-time write path for anonymous
  reflective-dot candidates, and finalisation support for them, closing the
  loop with the read side added earlier the same day (this file's next
  entry): a marker detection run can now produce dot candidates a session DB
  round-trips end to end, from detection through `pose_observations`.

  Promoted `detect_blobs()`/`BlobCandidate` out of the throwaway
  `tools/prototype_dot_blob_detector.py` script into a real module,
  `posetrak/detection/dot_blob_detector.py` -- the script itself now imports
  from it rather than duplicating the detection logic, keeping its role to
  what it's actually for (eyeballing detector behavior against a real
  capture, dumping annotated frames), not holding the production copy.

  New `DotCandidateWriter` (`db_cache.py`), mirroring the existing coded-
  marker corner writer's shape but for a variable candidate count per frame
  (`float32[N,4]`: px, py, area, compactness) rather than a fixed slot
  layout -- always writes a row even when a frame has zero candidates, so
  "processed, saw nothing" stays distinguishable from "never processed".
  Reuses the same near-zero `noise_scale` reasoning the marker writer
  established: a dot centroid comes from thresholding the full-resolution
  frame directly, not a fixed-input-resolution network, so there's no
  crop_scale-scaled detection error to describe.

  Wired into `MarkerDetectionPipeline` as an opt-in add-on
  (`detect_dots_for_cameras`, a set of camera instance ids) alongside
  whichever coded-marker detector is already running -- same frame, same
  loop, a second writer. Per-camera rather than a single on/off switch
  because dot visibility depends on the physical rig (a ring light on the
  GoPros used to validate this detector, not necessarily on every camera in
  the same capture); the caller decides, the pipeline doesn't guess from a
  camera label. Defaults to disabled everywhere, so every existing caller
  is unaffected. The GUI's run-detection dialog doesn't expose this yet --
  deliberately deferred; a caller building the pipeline directly can
  already use it.

  Extended `finalise_object_to_db()`'s existing marker-corner copy into a
  small shared helper parameterized by (track id, region type, source), and
  called it a second time for dots -- the exact "one more parameter value,
  no dots-specific code path" the design called for. Dots get no
  `pose_sequence_keypoints` manifest entries (they're anonymous, not named
  landmarks); everything else about the copy is identical to markers.

  Full targeted test coverage green: new `test_dot_blob_detector.py` (the
  detector itself against synthetic frames), `DotCandidateWriter` round-trip
  and noise-scale tests plus a pipeline end-to-end test gated on
  `detect_dots_for_cameras` (`test_marker_pipeline.py`), and finalisation
  copying dots alongside markers, and correctly writing nothing when a run
  never had dot detection enabled (`test_finalise_object.py`). Also ran the
  full non-GUI Python test suite (`tests/db`, `tests/detection`,
  `tests/cli`, `tests/tracker`, `tests/skeleton`, `tests/tools`,
  `tests/test_segmentation.py`) plus every touched `tests/app` file: 671
  passed, 2 failed -- both pre-existing and unrelated to this work
  (a Windows-path-absoluteness assumption in `test_posetrak_db.py`, and an
  observation-edit outlier test in `test_observation_edits.py`). The full
  suite including every GUI test could not be run to completion in this
  environment -- an unrelated pre-existing crash partway through the Qt
  widget tests (confirmed reproducible in isolation, e.g.
  `test_page_sync_led.py`'s `_build_combined_observations` being passed the
  wrong result type) kills the whole test process before it reaches a
  summary; flagged, not investigated further here since it long predates
  and is unrelated to this feature.

- **2026-09-02** — Added `SessionReader::load_unlabeled_candidates()`: reads
  anonymous reflective-dot candidate detections back out of the DB for a
  sequence, decoding each `pose_observations` row with `source='dots'`
  through the existing variable-length blob decoder and undistorting
  positions the same way labeled keypoints already are. New
  `UnlabeledCandidate` struct (camera, frame, timestamp, undistorted +
  distorted position, area, compactness) -- deliberately not an
  `Observation`, since there is no marker identity yet; resolving that
  identity is exactly what the shared dot-assignment phase does later (see
  [dot-assignment-architecture-design.md](dot-assignment-architecture-design.md)).

  Chose a new sibling method over changing `load_observations()`'s own
  signature -- `SessionReader` already has many single-purpose `load_*`
  methods (config, cameras, sequence info, ...), and dot candidates aren't
  filtered by `person_id` at all (they're scene-wide, not tied to any one
  tracked subject), so folding them into the person-scoped observation
  loader would have been the more awkward shape, not the simpler one.
  Factored the small "resolve camera_instance_id → Camera" and "read the
  pixels_are_undistorted flag" lookups `load_observations()` already did
  inline into two private helpers shared by both methods, rather than
  duplicating them.

  Found and fixed a real bug while adding the test fixture, not just a test
  artifact: `load_observations()`'s own query had no filter excluding
  `source='dots'` rows, so a dots row sharing an existing row's
  `person_id` would get pulled into that row's group and fail decoding as
  a labeled keypoint blob (wrong element count). Added an explicit
  `source != 'dots'` exclusion -- dots candidates are scene-wide and have
  no natural `person_id` of their own, so nothing guarantees a future
  write path picks one that never collides.

  Test coverage extends the existing manifest-bound object-sequence
  fixture (`test_session_reader.cpp`) with two `source='dots'` frames of
  different candidate counts (3, then 1), confirming per-frame decoding,
  camera resolution, and the confirmed-empty case for a sequence with no
  dots rows at all (every sequence before the dot-detection write path
  exists). Full C++ suite green afterward.

- **2026-09-02** — Split `Tracker`'s per-frame predict/update cycle into
  two public methods, and added a query for where each unlabeled ("dot")
  marker on a rigid-body skeleton is expected to project this frame. This
  is the foundation the shared dot-assignment phase (see
  [dot-assignment-architecture-design.md](dot-assignment-architecture-design.md))
  is built on top of: an orchestrator resolving competing dot candidates
  across several tracked subjects needs every subject's live prediction
  for the *same* instant, which requires calling predict() on all of them
  before any one commits its update.

  `Tracker::run_parent_step()` removed; its predict half is now public
  `predict_step(dt)`, its update half public
  `update_step(observations, timestamp)` (also absorbing `track_frame()`'s
  own post-step bookkeeping — `last_timestamp_`/`frame_count_`/
  `prev_observations_`/`frame_callback_` — since `update_step()` is now
  the terminal call for a frame either way). `track_frame()` itself is a
  two-line wrapper. The original design sketch assumed only
  `prior_state`/`prior_cov` needed to survive the split; tracing
  `run_parent_step()`'s actual body first found that the RTS smoother
  needs the *resolved* `PredictResult::cross_cov_future` too — an async
  computation deliberately resolved late (in `update_step()`) so its work
  overlaps `ukf_->update()`'s — so the whole `PredictResult` is stashed as
  a `Tracker` member across the split, guarded by a `predict_pending_`
  bool that makes calling `update_step()` without a preceding
  `predict_step()` throw rather than read stale/absent state.

  New `predict_dot_slot_predictions(camera_id)` — one camera at a time
  (matches `marker_projection_std(camera_id, ...)`'s own existing
  per-camera signature convention), returns every unlabeled marker's
  predicted pixel position and covariance for that camera. Rejects a
  non-rigid-body skeleton (the general/articulated case is still deferred)
  and an unknown camera id. Body-local marker geometry is recomputed fresh
  from rest-pose FK on every call rather than cached at init time — cheap
  for a rigid body (no articulation to run FK over), and avoids the
  fragility of tying correctness to which of `Tracker`'s several init
  paths happened to run (`initialize_rigid_body()` is only one of them;
  `initialize_from_state()` and `initialize_with_fixed_root()` also
  produce an initialized `Tracker` but were never going to populate a
  rigid-body-specific cache).

  Test coverage (`test_tracker_predict_update_split.cpp`, new): a
  frame-by-frame regression against the same articulated fixture
  `test_marker_projection_std.cpp` uses, confirming `predict_step()` +
  `update_step()` reproduce `track_frame()`'s `TrackingResult` field-for-
  field (state, covariance, per-observation diagnostics) to within
  floating-point tolerance every frame, not just at the end — proving the
  single-subject case is unaffected by the split existing at all. Plus the
  throw-without-predict, unknown-camera, non-rigid-skeleton, and a
  hand-computed-position rigid-body prediction case. Full C++ suite green
  afterward (posetrak_tests: all passing, 3986 assertions in 325 test
  cases).

- **2026-09-02** — Built the four dot-assignment pieces with no
  dependencies on anything else (see
  [dot-assignment-architecture-design.md](dot-assignment-architecture-design.md)):
  the Hungarian assignment solver, the rigid-body closed-form marker-
  position/covariance prediction, the DB blob-decoding convention for
  anonymous dot candidates, and the tracker's dot-assignment gating-
  threshold config field. All four unit-tested in isolation; full C++
  suite green afterward (319 test cases, 3272+ assertions).

  - **Hungarian assignment solver**
    (`cpp/include/posetrak/tracking/assignment.hpp`, header-only, matches
    `blob_codec.hpp`'s own convention): classic O(n³)
    Jonker-Volgenant-potentials Hungarian, gated per-pair (a pairing above
    the gate is dropped, not forced) rather than all-or-nothing. Tested
    against a genuinely adversarial case (globally-optimal vs.
    greedy-first-match disagree) and at 40×40 scale, comfortably past the
    "several tens per scene" target.
  - **Rigid closed-form marker prediction** (new
    `cpp/{include,src}/posetrak/tracking/marker_prediction.{hpp,cpp}`):
    predicts a marker's pixel position and covariance directly from the
    root pose and its covariance, with no FK/Pinocchio call, for a
    rigid-body (prop) skeleton. Cross-checked the Jacobian two ways before
    trusting it: algebraically against
    `Tracker::marker_projection_std()`'s independently-derived (general,
    FK-based) version of the same quantity — both reduce to the same
    formula via a standard rotation-of-skew identity — and empirically via
    a hand-computed lever-arm test that a sign error couldn't have passed
    by accident (nonzero u-variance, exactly-zero v-variance from a
    rotation that only moves the marker in depth).
  - **DB blob-decoding convention for dot candidates**:
    `decode_dot_candidates()` added to `blob_codec.hpp` as a
    `decode_keypoints()` sibling (float32[N,4]: px, py, area,
    compactness) — confirms the design's "no new tables needed" call holds
    in practice, not just on paper: the existing
    `detection_keypoints`/`pose_observations` tables already fit a
    variable-N blob directly.
  - **Config surface**: found and fixed a real inaccuracy in the design
    doc itself before implementing — it had cited
    `rigid_init_max_residual_m` as a "DB column +
    `load_tracker_config()` wiring" precedent, but checking the actual
    code (`session_reader.cpp`'s real column list, `config.cpp`) showed
    it's TOML-only, no DB column at all — exactly the
    `init_search_window_s` (2026-08-31) pattern instead. Corrected the
    doc, then implemented against the corrected, verified precedent: the
    new `dot_assignment_gate_mahalanobis` field is TOML-parsed/validated
    on `TrackerAppConfig` only.

  Also fixed, unrelated to the new code's correctness: the full test
  suite's own wall-clock time (mostly binary-load/process-startup
  overhead, not CPU -- observed ~25s real vs. ~0.03s user) had crept
  close enough to meson's 30s default test timeout that adding these four
  small files pushed it over, causing an intermittent spurious timeout
  with all 3272 assertions actually passing when run directly. Widened
  `tests/meson.build`'s timeout to 120s rather than leave the suite
  looking flaky as more tests get added.

- **2026-09-02** — Broke the live dot-labeling design
  ([dot-assignment-architecture-design.md](dot-assignment-architecture-design.md))
  into 12 independently buildable/testable pieces of work (the doc's own
  implementation-phasing section), same discipline as phase 1's own
  six-sub-phase breakdown. Four have no dependencies and can start
  immediately/in parallel: the Hungarian solver, the rigid closed-form
  marker-prediction math, the DB schema/blob-codec convention, and the
  config surface. The rest chain through the `Tracker` predict/update
  split and the shared dot-assignment orchestrator (tested synthetically
  against the actual double-claim scenario the whole redesign exists to
  fix) before wiring into both tracking call paths and finally real-data
  validation — which is explicitly gated on the separate calibration-time
  dot-geometry phase existing too, not just this phase's own pieces. The
  scene-wide-detection/de-dup bridge a real second dot-bearing subject
  would need stays explicitly out of this phasing, unchanged from the
  prior entry.

- **2026-09-02** — Answered a direct question (Harri: "the tracked
  subjects already influence each other via the cross-subject relative
  observation mechanism -- why does dot assignment need a new layer?") by
  tracing the existing cross-person coupling mechanism precisely and
  confirming why it can't be reused. It never splits predict from update
  because it doesn't need to: `build_anchor_observations()` (verified,
  `multi_person_tracker.cpp`) borrows another subject's *own*
  already-computed posterior state (this frame's if they already stepped,
  else last frame's extrapolated) as a soft reference value for an
  already-known marker correspondence -- correspondence is never in
  question, and one frame of staleness is an accepted, designed-for
  approximation. Dot assignment has neither property: it must resolve a
  genuine identity ambiguity across subjects sharing one candidate pool,
  and needs every competitor's prediction for the identical instant, not
  a stale one, to actually prevent double-claiming. Confirmed
  `UnscentedKalmanFilter::predict()` mutates state in place (not a
  peekable dry-run), which is what makes `Tracker`'s predict/update split
  unavoidable rather than a design preference. Also surfaced and resolved
  a second, more discretionary choice bundled into the design: joint
  (one combined cost matrix across subjects, globally optimal) vs.
  sequential greedy against a shrinking pool (closer to the existing
  anchor mechanism's own shape, less new machinery, but order-dependent).
  Harri confirmed joint, consistent with this design's other
  production-quality-over-cheaper-path calls. Written up in
  [dot-assignment-architecture-design.md](dot-assignment-architecture-design.md)
  §5.3 (new).

- **2026-09-02** — Made dot assignment a shared phase across every
  tracked subject, not a per-subject step -- Harri's explicit call, not
  needed for the sword alone but wanted "very soon." Substantially
  revised `dot-assignment-architecture-design.md`'s §5 (previously "put
  assignment inside `Tracker::run_parent_step()`," which structurally
  cannot prevent two subjects from claiming the same candidate, since
  each `Tracker` only ever sees its own prediction). New design: split
  `Tracker::run_parent_step()` (verified private/atomic today) into
  public `predict_step()`/`update_step()`, so an orchestrator can call
  `predict_step()` on every dot-bearing subject first, run **one**
  Hungarian solve per camera across the *union* of every participating
  subject's dot slots (not one solve per subject), then call each
  subject's own `update_step()` with its resolved share -- a sibling
  phase to `MultiPersonTracker`'s existing `update_contact_gate()`
  cross-person orchestration, not a new kind of thing that orchestrator
  hasn't done before. `Tracker::track_frame()` itself is unchanged for
  every dot-free subject (every existing person, and the sword's own
  ArUco corners).

  Working through this surfaced a real, previously-unstated dependency:
  joint arbitration only works if the *candidates* are a single
  de-duplicated pool per (camera, frame) -- today's per-capture_object
  detection runs (§3) would each independently detect the same physical
  dot as a separate row, so a combined cost matrix wouldn't actually
  arbitrate anything between two subjects' own redundant detections.
  This confirms the already-logged shared-scene-detection item
  (2026-08-31 entry below) is a hard *correctness* dependency now, not
  merely a performance one -- flagged as the real remaining gap (§5.3/§9),
  not designed or built this round; no real multi-subject-with-dots
  capture exists yet to design or validate either the full fix or an
  interim de-dup bridge against.

- **2026-09-02** — Revised `dot-assignment-architecture-design.md`
  against Harri's inline review comments -- both landed on real gaps in
  the first draft, not just clarifications:
  - **Storage**: dropped both proposed new tables entirely.
    `detection_keypoints`/`pose_observations` were already
    one-row-per-(frame,camera,source) with an arbitrary-length blob --
    ArUco's fixed corner count was a property of what ArUco stores, not a
    schema constraint. Dots reuse the same tables with a new
    `region_type`/`source='dots'` value and a variable-N blob layout, and
    finalisation reuses the *existing* generic copy loop unmodified (just
    one more parameter value) rather than new machinery -- directly fixes
    the row-count-at-scale concern too (blob size grows with candidate
    count, row count doesn't).
  - **Generalizes beyond one rigid prop**: named the actual seam
    (`MarkerPrediction`: predicted position + pixel covariance for one
    marker slot) that assignment consumes, and confirmed the rigid
    closed-form math is only one implementation of it -- the general/
    articulated implementation isn't hypothetical, it's a documented reuse
    of `UKF::predict_measurements()`'s existing sigma-point machinery
    (already marker_id/FK-generic, verified against the code), deferred
    only because no articulated dot-augmented capture exists yet to build
    or test against. Corrected the assignment solver's expected scale
    (Harri: "several tens per scene", not "a dozen") with real numbers
    (O(n^3) at n=50 is sub-millisecond, nowhere near the bottleneck).
  - **New, explicitly acknowledged gap**: multi-subject candidate-pool
    arbitration (two dot-bearing subjects drawing from the same shared
    detection pass) isn't designed -- flagged as a real hole (§9.1) tied
    to the already-logged shared-scene-detection item (2026-08-31 entry
    below), not silently assumed away.

- **2026-09-01** — Full design round for dot assignment (Option A),
  requested straight after the detection prototype confirmed the approach
  was worth designing properly. See
  [dot-assignment-architecture-design.md](dot-assignment-architecture-design.md).
  Highlights: no skeleton format change needed (a new `unlabeled_points`
  `input_tracks:` type slots in for free -- the loader already parses
  `type` as an opaque, unvalidated string); `UKF::update()` needs zero
  changes, since assignment resolves candidates to ordinary `Observation`s
  *before* it's ever called; the exact integration point is
  `Tracker::run_parent_step()` between `predict()` and `update()`, reusing
  `prior_state`/`prior_cov` it already computes rather than any new
  work; and the Mahalanobis cost function has an exact closed form for a
  rigid-body skeleton (no sigma points, no Pinocchio call -- a 3x6
  Jacobian from the state's own error-state convention, verified against
  `State::apply_error_update()`), which is both cheap and exact rather
  than an approximation-on-an-approximation. New DB tables needed
  (`detection_dot_candidates`/`pose_observation_dot_candidates`, variable
  row count per frame -- fixed-width blobs don't fit an anonymous
  candidate set) and a small hand-written Hungarian solver (no existing
  dependency covers this; problem sizes are small enough not to need one).
  RANSAC cold-start and the non-rigid (person-marker) case are explicitly
  out of scope for this round. Not built yet.

- **2026-09-01** — Prototyped reflective-dot detection against real
  footage before committing to the bigger architecture (Harri's call:
  de-risk the detector's real unknowns first, given a schema/skeleton/
  tracker design round is real work not worth doing on assumptions).
  Visually confirmed the dots are retroreflective (bright blown-out white
  points, ring-lit) rather than colored; per Harri, `pixel7` has no ring
  light and other Android phones' tone-mapping may complicate detection,
  so started with the two GoPro cameras. `python/tools/
  prototype_dot_blob_detector.py` (threshold + connected components +
  compactness filter, throwaway spike) ran against the baseline
  tracking run's single biggest observation gap (53.6-55.1s, where the
  ArUco-only pipeline produced zero observations for the whole 6-camera
  tracker): median 4 candidates/frame, only 14% of frames empty, and a
  clean 3-way area separation (real dots 30-70px vs. a tiny recurring
  floor-glint false positive vs. large stationary LED-ring-light blobs
  from other tripods) that a simple area+compactness filter already
  handles. Individual zero-candidate frames checked by eye were genuine
  benign per-camera misses (dot face turned away from that one camera),
  exactly what the multi-camera rig exists to cover.

  Also visually re-examined why ArUco actually misses observations: not
  purely motion blur as first framed -- one checked instant showed the
  marker plate genuinely edge-on to the camera (viewing-angle failure, not
  blur), another showed a clean sharp marker that still produced no
  tracked observation (likely a decode-margin cutoff invisible by eye).
  Broadens the case for dots (robust to viewing angle and decode margins,
  not just blur) without changing the conclusion.

  Decided: dot labeling during live tracking goes with Option A
  (Hungarian algorithm over a Mahalanobis-distance cost matrix, inside the
  C++ tracker using its own live pose prediction) -- Harri's call,
  explicitly for production quality over a quick demo, and explicitly not
  assuming a fixed dot count (rules out any count-specific shortcut).
  This is a real design round (schema, skeleton representation,
  `SessionReader`/`ObservationSet` changes, where in the pipeline
  assignment happens) that hasn't been done yet -- a first-cut agenda for
  it is in
  [reflective-dot-detection-design.md](reflective-dot-detection-design.md)
  §7, not the round itself.

- **2026-09-01** — Scoped (not built)
  [reflective-dot-detection-design.md](reflective-dot-detection-design.md),
  the follow-up Harri asked for after reading the sword baseline below:
  fast motion frequently leaves ArUco undetectable (blur kills the bit-
  pattern decode well before the marker looks unrecognizable), so the
  tracker coasts on its constant-velocity model through the gap and
  visibly snaps once a fresh, distant observation arrives -- his own
  diagnosis, matching the run's actual numbers (19% outlier rate, only 5.5
  mean inliers). A dot's blob detection should be far more blur-tolerant,
  a specific and testable hypothesis (compare gap duration during fast
  segments, not just aggregate tracked-step-fraction, once dots exist).
  Detection method itself was already resolved generically in
  `pose-detect-improvements/marker-detection-analysis.md` (threshold +
  connected components + centroid); calibration-time dot geometry is a
  straightforward Phase A/B extension (no cold-start needed, restrict to
  ArUco-anchored frames). The real open piece is live per-frame labeling
  of anonymous dots during an actual tracking run, where a genuine
  architecture fork exists between prediction-gated assignment inside the
  C++ tracker (matches what marker-detection-analysis.md already designed
  generically for anonymous person markers, reusable there later, real new
  tracker-side work) vs. resolving to fixed slots in the Python detection
  pipeline before the C++ side ever sees them (no tracker changes, but
  still needs the already-designed pairwise-distance RANSAC cold-start
  registration for frames with no ArUco anchor -- exactly the fast-motion
  case this exists to fix, so it doesn't avoid the hard part either).
  Neither decided yet; see the design doc's §3.2 and §6.

- **2026-09-01** — First end-to-end real tracking run of a prop calibrated
  entirely by Phase A (previous entries), no manual measurement anywhere in
  the chain. Full pipeline against the real "Weapon test 2026-08-20"/"Harri
  bokken" capture: `calibrate_rigid_marker_body.py` (54 co-occurrence
  samples, marker size 9.5cm measured by Harri) → `marker-body import` →
  new `capture_objects` row → `to-skeleton` → `MarkerDetectionPipeline` over
  the full 66s trial, 6 cameras, `frame_step=1` (47,688 observations) →
  `finalise_object_to_db` → `posetrak-tracker track`.

  Result: **0 steps lost** across the whole trial, 3458/6618 steps tracked
  (52.3%), mean reprojection error **8.4px** -- notably better than the
  calibration box's ~29px, the first real confirmation the marker-noise fix
  (`crop_scale=0` for markers, giving ArUco corners credit for their own
  sub-pixel precision, 2026-08-31 entry below) actually helps as predicted.
  The rigid init-search fix (also 2026-08-31) worked exactly as designed on
  this brand-new sequence too: no valid window at the exact requested
  start, found one 0.06s later, 4.6mm RMS Kabsch/Umeyama fit.

  Honest caveat, not a blocker: frame-to-frame position deltas show real
  jitter at several points -- implied instantaneous speeds up to ~160 m/s
  between consecutive 10ms steps, physically impossible for a hand-held
  sword. Lines up with `mean_num_inliers` of only 5.5 (out of 8 possible
  marker-corners across 6 cameras) and a 19% outlier rate: with only 2
  markers (no dots yet), fast motion during the more energetic parts of the
  kata leaves many steps weakly constrained by a single marker, and motion
  blur degrades detection further right when it matters most. Never full
  track loss, but real per-step noise -- exactly the gap Phase C
  (reflective dots) should measurably close. This run is deliberately kept
  as the "before" baseline (Harri's own plan) to compare against once dots
  are added: same trial, same skeleton family, re-run and compare tracked-
  step fraction / reprojection error / jitter directly.

  Real DB rows created (`E:\mocap\vanhaa\ukemi-tommi-20260509.db`, all
  additive, nothing mutated): marker body definition
  `c5a55f312cb9c972d935332820c38260d7d7b5c18a9e7cafbcd9d96a7ee51121`,
  capture object `b0ac2784-53c8-4ad0-88d8-9fe94a0fd020`, skeleton
  `2c93603ced9b3facce19c3376b5ead2dc2953f5969799cc866caf6e14e80c862`,
  detection run `3f35c79e-6289-4556-81af-563b65d4c654`, sequence
  `98ec877c-3740-497e-9a98-f69bbca04a62`, tracking run
  `319ebb30-49a9-4ee1-b2ad-2874b1495a2f`.

- **2026-09-01** — Visually confirmed (annotated frame dumps) the two
  stray marker IDs found while validating co-occurrence (previous entry)
  are both harmless: `17` is a genuine false-positive decode (drawn quad
  on bare rock wall, no real marker there); `10` is real but is one of the
  room's own fixed floor-mounted calibration markers, seen repeatedly at
  its one stationary location, unrelated to the sword. Neither affects the
  calibration tool, which only ever considers the explicit marker IDs it's
  given. Built (not yet run against real data) Phase A of
  [rigid-marker-body-calibration-design.md](rigid-marker-body-calibration-design.md):
  `python/tools/calibrate_rigid_marker_body.py`, a standalone script
  (design doc §8) that solves each visible ArUco's own rigid pose per
  frame (reusing `extrinsics_solver.solve_marker_pose()` unmodified) and
  robust-averages every other marker's corners in the reference marker's
  frame across the whole capture. Needs the sword's physical marker size
  to actually run -- not yet recorded anywhere in the DB.

- **2026-09-01** — Validated the co-occurrence assumption
  [rigid-marker-body-calibration-design.md](rigid-marker-body-calibration-design.md)'s
  §2 rests on, against the real "Weapon test 2026-08-20"/"Harri bokken"
  capture (read-only ArUco scan across all 6 cameras, no DB writes):
  confirmed, and abundantly so, not marginally -- the sword's two ArUco IDs
  (`2`, `3`, one per face) co-occur from different cameras in 227 distinct
  0.05s time-buckets across the trial, starting immediately and repeating
  on nearly every sample for stretches at a time. Phase A of that design
  has real data to build against. Full numbers and a couple of stray IDs
  worth a quick sanity check (likely noise, not a third marker) are in the
  design doc's §6.

- **2026-08-31** — Scoped (not built) a new capability triggered by a real
  capture that the existing turn-around-video calibration method can't
  handle: "Weapon test 2026-08-20"'s sword prop has ArUco markers on both
  flat faces with no marker ever visible from both, which a single-camera
  turn-around can't link (no shared correspondence across the transition).
  See
  [rigid-marker-body-calibration-design.md](rigid-marker-body-calibration-design.md):
  a calibrated multi-camera rig sidesteps this because the two faces' anti-
  parallel normals mean opposite-side cameras typically see both faces in
  the same synchronized frame, and each ArUco marker is a self-contained
  pose reference (known planar corners), so no cross-face co-occurrence is
  even required -- just some frame where each marker co-occurs with *any*
  decodable ArUco. Large fraction of the numerics already exist and are
  production-used by the (structurally identical) extrinsics-calibration
  path (`solve_marker_pose()` et al. in `app/setup/extrinsics_solver.py`);
  genuinely new pieces are a reflective-dot blob detector (phase 2 scope,
  not yet built at all), offset extraction/aggregation, an optional joint
  least-squares refine, and the orchestrating CLI. Phased A (ArUco-only,
  de-risks the core co-occurrence assumption first) / B (joint refine) / C
  (reflective dots).

- **2026-08-31** — An external code review (`gpt-sol-review-20260831.md`,
  no code changes) found two real issues worth acting on immediately, both
  confirmed against the actual code and fixed (Harri agreed on these two;
  the rest of the review's findings are either already-known/tracked
  (the outlier-edit bug, #8) or correctly scoped to later phases already
  named in the design doc (symmetric/locked-root DOFs, multi-track loading,
  dot/mixed finalisation) rather than phase-1 defects):

  - **Finding #1 (High) — failed rigid init silently fell back to a
    meaningless rest pose.** The CLI tried exactly one window at
    `start_time`, and on failure printed a warning and called
    `initialize_from_rest_pose()` -- fine for an articulated person (still
    anchored, if imprecise), meaningless for a free-floating rigid prop at
    the world origin. This is exactly the failure this phase's own 1f
    validation hit and worked around by hand (brute-force `--start-time`
    scanning) rather than fixing. Also found while reading the code: a
    comment in both `config.hpp` and `tracker.hpp` already described "the
    existing retry-on-a-later-frame loop" as if it existed -- it never did,
    for persons or objects, in either of the CLI's two separate init call
    sites (`track.cpp`'s TOML-config-file path, `run_track()`; and the
    DB-driven path every GUI-launched run actually uses,
    `multi_person_tracker.cpp`'s `build_person_context()` -- initially
    fixed only the former and real-data-validated against the latter by
    accident of it still printing the *old* message verbatim, catching the
    miss before calling it done). Fixed in both: search forward from
    `start_time` (a new `init_search_window_s`, 2.0s default -- a
    `[tracking.initialization]` TOML field in `run_track()`, a local
    constant in `build_person_context()` pending a real need to tune it
    per-capture), trying `tracker.initialize()` at each candidate window
    and shifting `start_time` (and `num_steps`) to the first one that
    succeeds. If the whole window fails, a rigid-body skeleton (new
    `Skeleton::is_rigid_body()`, replacing a duplicated local check in
    `Tracker::initialize()`) now throws with a clear message instead of
    silently proceeding; an articulated skeleton keeps the original
    rest-pose-fallback-with-warning behaviour unchanged, since that wasn't
    reported broken and rest pose is at least a defensible guess there.
    Real-data validated on `ukemi-tommi-20260509.db`'s calibration box:
    from `--start-time 14.93` (the object's sequence start, sparse
    1-camera-only coverage per Harri's own account of the capture) the
    2s window isn't enough and it now fails loudly with a clear message
    instead of silently tracking from the origin; from `--start-time 19.5`
    it searches forward 0.942s, finds a valid window, and tracks
    547/547 steps (100%) -- no more manual `--start-time` scanning needed
    for a reasonably-close guess. Not yet covered by an automated test
    (review's own finding #9 flags this gap correctly): the fixture needed
    -- a rigid skeleton with real per-camera timing gaps large enough to
    exercise both the search-succeeds and search-exhausted paths -- is
    substantially bigger than a quick addition; deferred rather than
    rushed, real-data validation stands in for now.

  - **Finding #4 (High) — marker corners weren't getting credit for their
    own precision.** The tracker's measurement noise splits into two
    pieces (`Observation::measurement_noise_std(ep, ec) = ep*crop_scale +
    ec`): `ep` (`pose_noise_std`) is the detection algorithm's own
    localization error, scaled by `crop_scale` (how much the algorithm's
    fixed input resolution was stretched to cover the real-world crop);
    `ec` (`calib_noise_std`) is camera-specific error (extrinsics
    inaccuracy, autofocus drift affecting intrinsics) that applies
    regardless of detection method (Harri's framing, confirmed against the
    C++ formula). `MarkerKeypointWriter` wrote `noise_scale=NULL`, which
    `SessionReader` defaults to the person pipeline's `crop_scale=1.0` --
    giving marker corners the *full* `ep` contribution meant for a
    markerless pose network's fixed-input-resolution error, when an ArUco
    corner is found by direct sub-pixel refinement on the full-resolution
    frame with no such resolution-scaling error to describe. Likely a real
    contributor to the ~29px mean reprojection error 1f's real-data
    validation flagged as "worth a closer look." Fixed: `MarkerKeypointWriter`
    now writes `noise_scale=0.0` instead of `NULL`, letting `ec` alone
    dominate for markers -- no schema change, no C++ change, since the
    existing formula already does the right thing once `crop_scale` is
    correct. Old runs (`noise_scale=NULL`) are unaffected (`COALESCE`
    still gives them 1.0); only new marker runs get the fix.

- **2026-08-31** — First real end-to-end GUI pass feedback (Harri), logged
  for design rather than acted on now -- all genuinely need more thought
  than a quick patch, and naturally land around phase 2 (dot markers, so
  multiple detector types) / phase 4 (multiple mixed props + person, UC1
  complete):
  - **Marker detection is single-threaded and sequential across cameras**
    (~13-20% CPU observed). `MarkerDetectionPipeline.run()` processes
    cameras one at a time; `iter_frames()`'s per-video decode is
    *deliberately* pinned to `thread_type="NONE"` (a documented past
    FFmpeg hang with threaded decode + this code's early-close pattern),
    so the fix isn't intra-video frame parallelism -- it's running
    independent cameras in separate processes (multiprocessing, not
    threads, since decode is CPU-bound and the GIL blocks real threaded
    speedup).
  - **No preview crops during/after marker detection** -- `ObjectPanel`'s
    crop grid falls back to on-demand `FrameReader` decoding (a deliberate
    1e choice: "no crop-caching infrastructure needed for objects, unlike
    persons"), which is slow in practice. Persons get crops for free as a
    byproduct of the same decode pass detection already does
    (`_encode_crop()`, into an already-generic crop table); replicating
    that for markers (union bbox of a frame's detected corners) looks
    straightforward on its own. The part that needs real design, per
    Harri: (1) today's one-full-decode-per-registered-subject architecture
    means a scene with several props + performers re-decodes the same
    footage once per subject -- a per-run crop fix doesn't worsen that,
    but doesn't fix it either, and a shared single-pass scene-wide
    detection (matching UC1 phase 4 / UC2's later multi-subject phasing)
    is the real fix; (2) preview *bounding boxes* need the same
    overlap-merging logic multi-person preview already uses -- a person
    holding a tracked prop (e.g. a sword) must not naively get two
    separate, nonsensical previews (one for the person's bbox, one for the
    sword's) when they're the same physical region.

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

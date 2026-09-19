# Marker-based mocap — status

- **2026-09-19** (productization plan, review round 1) — Harri's inline
  review folded into
  [productization-architecture-and-plan.md](productization-architecture-and-plan.md)
  (§8 has the summary table). **One comment found a real design flaw**:
  the draft's "compose a subject sequence by copying detection rows into
  it" would have copied a scene-wide reflective-dot run's candidates into
  every participating subject's sequence — and since
  `MultiPersonTracker::run()` concatenates each subject's own candidate
  list into the shared cost matrix, one physical dot would have appeared
  as two distinct candidates and could be claimed by two subjects at
  once, silently defeating the mutual exclusion the shared assignment
  phase exists to provide. This is the prerequisite
  dot-assignment-architecture-design.md §5.4 already flagged; copying
  would have turned it from a latent limitation into a guaranteed bug on
  the first two-dot-bearing-subject capture. Corrected: labeled sources
  are copied into the subject's sequence, anonymous candidate pools are
  *referenced* and loaded once per detection run, shared across subjects,
  with per-subject camera filters as an eligibility view. That also
  delivers §5.4's real fix at the data-model level instead of needing a
  de-dup bridge. Other resolutions: `rel-proto` branched from current
  main with main returning to trunk; external-2D import gains a labeled
  layout with an external-label→PoseTrak-label map; tracking runs are per
  trial with a subject roster, so segments are per (run, subject); WS4
  (decode performance) moves ahead of the torso+arm module's processing
  since that capture lands in 1-2 weeks; DaVinci Resolve joins Blender in
  the BVH-gap spike. **Skeleton document split**: recommended as its own
  feature rather than folded in (§9) — the full split touches loading,
  hashing, scaling, provenance, export and every existing session DB, and
  three of this plan's workstreams will generate the requirements it
  still lacks. Pulled forward instead: a topology name plus a structural
  hash that catalog modules and attachment sets declare and the composer
  validates, which bounds the duck-typing risk for ~a day of work and is
  forward-compatible with the eventual split. Four evidence triggers
  recorded for when to open that feature.

- **2026-09-18** (productization architecture and plan drafted) —
  [productization-architecture-and-plan.md](productization-architecture-and-plan.md)
  proposes the target architecture for landing this branch on `main`
  and the workstream order to get there. Headline calls, all open for
  review: merge the foundation now and finish on `main` (the old merge
  bar's schema condition is met by the 2026-09-14 person+pen+pad run);
  keep one sequence per subject and make sequence *composition* a
  first-class, provenance-recording operation (`pose_sequence_sources`)
  in place of the three bespoke finalise scripts; external 2D tracks as
  an `external_2d` detector type; attachment sets as stored registry
  documents materialized into generated skeletons; a cost-modifier
  pipeline plus a reacquisition-corroboration gate for tracking-time dot
  assignment (the twice-observed confidently-wrong-candidate failure);
  segments as the unit of re-initialization; and a full triage of the
  ~60 tools into package/CLI, prototypes, GT harness, or archive.

- **2026-09-17** (throw3/4's remaining divergence explained; productization
  thoughts from the ball-tracking investigation) — Harri asked whether
  throw3/4's still-visible divergence (vs. throw1/2 "looking pretty
  convincing") traces to camera count: confirmed yes.
  `gopro13_01`'s own Blender track only spans t=56.74-64.66s (throws 1-2
  only -- it wasn't hand-tracked further), so throw3/4 are triangulated
  from just 2 cameras (`insta_ace2_pro`, `gopro-11_mini_01`) instead of
  3, with no third view to cross-check or average against -- matching
  the already-measured numeric gap (throw3/4's `gopro-11_mini_01`
  reprojection median 72-96px vs. 12-57px when a third camera was
  available). Not a new bug -- an expected consequence of losing a
  camera, and a real, direct illustration of why a UI eventually needs
  to surface per-segment camera-coverage/confidence rather than a
  single trajectory that looks uniformly trustworthy.

  **Conclusions from the whole investigation** (reflective-dot attempt,
  Cutie/SAM2 segmentation attempt, Blender external-tracker experiment)
  towards eventually discussing productization -- Harri's own, plus a
  few more prompted by what this session actually hit:

  - **External track import (and export) as a real feature**, not just
    a one-off script: useful beyond the ball case, for debugging and
    for any object/person a user chooses to track with an external
    tool. `finalize_ball_blender_detection.py` is a first, narrowly-
    scoped instance of the import half; no export-for-external-editing
    path exists yet.
  - **Semi-automatic reinitialization for an object/person that
    disappears and reappears** (the ball going in and out of the hand
    is the general case: any tracked subject can be fully occluded for
    a while). PoseTrak should *detect* a clear gap in tracking data
    itself, but the user needs to be able to edit/force a reinit
    rather than the system silently guessing. This also needs a way to
    represent a gap in exported BVH, which has no native concept of a
    timeline gap -- candidate approaches (holding the last valid pose,
    snapping to a defined default/rest pose for the gap's duration, or
    exporting separate per-segment BVH files plus a manifest describing
    the gaps between them) haven't been evaluated against what real
    downstream tools (Blender, Unreal, ...) actually do with each.
  - **Prop segmentation is still worth having even though it didn't
    solve the ball**: with many props in frame, a mask (even an
    imprecise one) is a useful hint for assigning anonymous dots to the
    right object, separate from the "is the mask alone accurate enough
    to BE the position" question this session answered no to.
  - **Soft classification hints for reflective dots** (Harri's
    recurring idea: store the average hue/color of a dot's immediate
    surroundings) -- worth an actual test. This session's own
    `gopro13_02` clutter (a near-static candidate that repeatedly won
    assignment over the real, moving ball) is a ready-made real-world
    test case: if that clutter's surrounding hue differs detectably
    from the ball's, this would have caught it independently of the
    tracklet-motion mechanism below.
  - **Marker-assignment-for-skeletons is entirely script-based today**
    (every skeleton/marker-body YAML this session touched was
    hand-authored or generated by a one-off tool) -- needs a real UI,
    and ideally automation, before this is usable by anyone other than
    someone comfortable editing YAML and querying the DB directly.
  - **Multi-detection-source UI and DB schema**: different tracked
    subjects legitimately need different combinations of detection
    sources feeding one tracker -- fully markerless, markerless
    filtered by segmentation, ArUco + reflective dots + segmentation,
    or (the ball) a single reflective dot with no orientation
    observability and no usable segmentation at all. Today each
    combination is wired up by a bespoke script
    (`finalize_ball_*_detection.py`, `setup_pen_pad_capture_objects.py`,
    ...); there's no schema or UI concept of "this object's tracker
    input is sourced from these N detectors, combined by these rules."

  **A few more, prompted directly by this investigation's own root
  causes, not yet discussed with Harri**:

  - The `gopro13_02` failure is fundamentally a **data-association**
    problem, not a threshold-tuning one: a single wrong candidate can
    be sharp and confident (low measurement noise) while still being
    the wrong physical point, and no per-frame Mahalanobis gate
    tuned for genuine detection noise can distinguish "confidently
    wrong" from "correctly right" using geometry from one camera alone.
    This is exactly the class of problem the parallel, not-yet-built
    tracklet-aware assignment gating (design already sketched:
    dot candidates get a `tracklet_id`, per-frame assignment relaxes
    the gate for a candidate whose tracklet matches the previous
    frame's resolved one) targets -- today's finding is a second,
    independent real-world case motivating it, on top of whatever
    prompted the original design.
  - **Segmentation might combine better as an ROI filter than as a
    position source**: this session only tried "mask centroid = 3D
    position input," which failed once the mask itself drifted onto
    the wrong object. Using the mask instead to *restrict which raw
    dot candidates are even considered* for assignment (a much weaker,
    more robust use of an imprecise mask) hasn't been tried, and the
    ball's own Cutie mask + dot data already exist to test it against
    cheaply.
  - **Per-source trust, not just per-marker**: the multi-source
    combination rules Harri describes above will need each source to
    carry its own trust/measurement-noise characteristics (a
    human-verified external track vs. a raw automatic dot candidate
    vs. a segmentation-derived position aren't equally reliable) --
    today's config model (`confidence_threshold_marker_names`, etc.)
    is per-marker, not per-source-per-marker.
  - **A reusable cold-start assistant**: this session's own
    triangulate-a-seed-position-and-trim-to-real-coverage work was all
    ad hoc Python run interactively. If external-track import becomes
    a real feature, seeding a dots-only object's tracker from whatever
    multi-camera data exists (rather than requiring `--seed-position`
    to be computed and passed by hand each time) should become a real,
    reusable tool.

  **Worth prototyping before a productization design discussion**
  (small, cheap validations rather than committing to a schema/UI
  design first):

  1. Cross-camera geometric consistency as an assignment *pre-filter*
     (the multi-view RANSAC approach `prototype_ball_tracking.py`
     already used, but run ahead of the UKF instead of as a standalone
     script) -- directly targets today's "confidently wrong single
     candidate" failure mode, and is a smaller, more isolated change
     to validate than the full tracklet-gating design.
  2. Segmentation-mask-as-ROI-filter, tested directly against the
     ball's own existing Cutie mask + dot data (cheap: the data already
     exists, no new capture needed).
  3. The hue/color soft-classification idea, tested against the same
     `gopro13_02` clutter case identified today.
  4. A small spike on the 3 candidate BVH-gap representations against
     real downstream tools, before committing to one in an export
     design.
  5. Hand-build the fusion logic for 1-2 concrete multi-source cases
     (e.g. a person using markerless + segmentation-filtered dots, a
     prop using ArUco + dots + segmentation) before designing the
     general schema/UI -- matching how this project's own person/prop
     work always started from a concrete script before a schema.

- **2026-09-17** (single-marker rigid-body tracking: architecture fixes,
  and an external-tracker experiment for the "3rd case" ball) — Three
  real C++ gaps found and fixed while trying to track the reflective
  ball (a rigid body whose only marker is a single anonymous
  `unlabeled_points` dot, so it has no coded anchor and no cold-start
  observation at all):

  1. `Tracker::initialize()` rejected any rigid body with fewer than 3
     markers before even checking `is_rigid_body()`. Fixed by checking
     rigid-body-ness first (only requires a non-empty marker set) and
     adding a genuine single-marker branch to `initialize_rigid_body()`:
     with one point there's no orientation to fit (Kabsch/Umeyama need
     >=2), so it places `root_position = world_point - body_local_offset`
     directly and leaves `root_orientation` at identity -- correct here
     since nothing else in a single-marker skeleton reads orientation.
  2. A pure-dot skeleton has zero `Observation`s ever (dots aren't
     resolved to a labeled marker at load time), which two separate
     fatal checks assumed impossible: `session_reader.cpp`'s
     `load_observations()` (`rows_total == 0`) and
     `multi_person_tracker.cpp`'s `build_person_context()`
     (`ctx->observations.empty()`). Both relaxed to tolerate zero
     labeled observations specifically when the skeleton has no labeled
     marker at all / the context has a dot track, rather than loosening
     the check for every skeleton.
  3. Added a `--seed-position X Y Z` CLI escape hatch
     (`BuildPersonContextOptions::seed_position`, DB mode, single-subject
     only): a dots-only skeleton's per-frame candidate assignment needs
     an *existing* predicted position to gate candidates against, but
     there's no labeled observation to cold-start from -- chicken and
     egg. When given, `build_person_context()` skips the normal
     observation-search init entirely and calls
     `initialize_from_state()` with a hand-built state at the given
     position, identity orientation, zero velocity.

  All three verified via `./run_tests.sh` (still green) and by actually
  tracking the ball's own anonymous-dot data end to end -- previously
  impossible; `Tracker::initialize()` failed unconditionally before
  reaching any of this. Not yet committed (see below for why).

  **External-tracker experiment**: earlier 2026-09-15/16 attempts to
  position the ball from data already in this project (reflective-dot
  detection, then Cutie/SAM2 mask-centroid segmentation, both below)
  didn't hold up. Harri tested Blender's own Movie Clip Editor 2D
  tracker interactively instead, manually re-seeding 2-3 times per
  camera to survive occlusion, and it tracked every throw cleanly in
  3 of the 6 cameras (`insta_ace2_pro`, `gopro-11_mini_01`, `gopro13_01`
  -- the other 3 were skipped: `gopro13_02` already has good
  reflective-dot detections, `oneplus9pro-01`/`pixel9` have too much
  occlusion to be worth hand-tracking). New
  `python/tools/blender_export_2d_tracks.py` exports each track's 2D
  curve to CSV in PoseTrak's own pixel convention (Blender stores
  markers normalized bottom-left/Y-up; converted to top-left/Y-down on
  export) -- verified the exported `video_frame` numbering needs no
  correction by cross-checking against real `pose_observations` rows
  for known frames. New `python/tools/finalize_ball_blender_detection.py`
  writes these 3 cameras' tracks plus `gopro13_02`'s existing raw
  reflective-dot candidates (copied unfiltered for the same span -- the
  tracker's own Mahalanobis-gated assignment already separates real
  ball candidates from body-marker clutter, the same mechanism the
  leg markers rely on) into a fresh `capture_objects` /
  `detection_runs` / `pose_observation_sequences` /
  `pose_observations` set, reusing the `dots` wire format and the
  existing single-reflective-dot marker body from the Cutie attempt.

  First tracking attempt (one continuous UKF run across all 4 throws,
  `--seed-position` at throw 1's start) diverged badly after throw 1
  (root position drifting to y=-18m, z=-8m by throw 4 -- checked via
  `tracking_obs_results`, not just trusted the summary numbers, per
  this project's own established practice). Root cause: a Blender track
  only covers the frames where a human actually re-seeded it, so there
  are real multi-second gaps between throws where none of the 3
  hand-tracked cameras have any data at all; during those gaps the only
  observation left was `gopro13_02`'s raw dot feed, which includes real
  clutter (a near-static candidate, almost certainly a body marker or
  a reflection). A single camera's 2D point can't constrain 3D position
  along its own viewing ray, so locking onto that one clutter candidate
  for an extended gap let the filter drift freely, and it never
  recovered even once real multi-camera data resumed later.

  Fixed by tracking each throw as its own separate, freshly-triangulated
  run (DLT from whichever cameras have a real, time-synced sample near
  the throw's start) instead of one continuous pass bridging the
  inter-throw dead zones, with the run's own `--start-time`/`--end-time`
  trimmed to where at least 2 of the 3 hand-tracked cameras actually
  have data (found via time-binned coverage counting) -- this keeps
  `gopro13_02`'s clutter-prone single-camera tail out of every run
  entirely rather than just reducing its impact. All 4 resulting
  trajectories are now physically plausible (root position within
  roughly a 1m x 1m x 1m box per throw, z tracing a fall-and-bounce
  shape consistent with "thrown down, caught on the bounce") --
  a real improvement over both prior automated approaches, though not
  yet confirmed by visual reprojection-video inspection the way this
  project's own practice requires before trusting numeric plausibility
  alone.

  **Visual verification, round 1** (reprojection video, throw1, all 3
  hand-tracked cameras -- tracked 3D position reprojected in red, the
  camera's own Blender 2D track in green): cropped/upscaled tightly
  around the *tracked* trajectory's own median position. Looked mostly
  good (red/green coincide through most of the sampled frames), with
  large error only right at the very start/end of the window -- read at
  the time as just the expected re-seed-convergence and occlusion
  tails. Wrong: Harri caught that the tight crop was itself hiding the
  interesting (diverging) frames, since a badly-off tracked point can
  fall outside a crop window sized only around the tracked trajectory.

  **Visual verification, round 2** (same video, crop resized to the
  full extent of *both* red and green across the whole throw): revealed
  a real, substantial divergence during the post-bounce rise -- e.g.
  at t=57.558s (well before any occlusion, ball clearly visible in all
  3 cameras) the tracked position was still down near the floor while
  the real ball was already back up near the catching hand, confirmed
  visually in all 3 camera views at once (not a single camera's own
  calibration/reprojection quirk -- the fused 3D estimate itself was
  wrong). Root-caused via `tracking_obs_results`: `gopro13_02`'s raw,
  *unfiltered* dot feed (folded in earlier on the assumption that the
  tracker's own Mahalanobis gate would sort real ball candidates from
  clutter, the same mechanism the leg markers rely on) contains at
  least two different near-static clutter candidates, and the per-frame
  assignment kept flip-flopping between them and the real moving ball
  candidate. Each individual jump's Mahalanobis distance stayed under
  this tracker config's `outlier_threshold=6.0` even though it was
  visibly a different physical point, so nothing ever got flagged as an
  outlier -- and because 3 real cameras plus 1 wrong one still looked
  like 4 "agreeing" observations, the fused state got pulled measurably
  off course for several tens of frames at a time before the real
  cameras pulled it back. The Mahalanobis-gate reasoning that justified
  including unfiltered `gopro13_02` data does hold for genuinely-labeled
  markers (many real candidates, one true position, gate rejects the
  rest) -- it broke down here specifically because a totally different,
  also-near-static point could itself look locally consistent with the
  (already slightly off) predicted trajectory for a while.

  Fixed by dropping `gopro13_02` from the sequence entirely and
  re-tracking all 4 throws from just the 3 hand-tracked cameras.
  Reprojection error across the whole of throw 1 dropped from a max of
  500+px to median 16-57px / p90 <=67px per camera (all 4 throws
  checked numerically, medians 12-96px, no more multi-hundred-pixel
  divergence anywhere) -- re-verified visually in the wide-crop video
  too, not just by the summary numbers. What error remains (mostly at
  the very start/end of each throw's window) is the re-seed-convergence
  and occlusion-tail pattern described in round 1, now genuinely the
  dominant remaining source of error rather than masking a bigger bug.
  A production version of this workflow should (a) never fold in a
  PoseTrak-internal automated detection feed as if it were as trustworthy
  as a human-verified external track without checking for exactly this
  kind of static-clutter contamination, and (b) trim each throw's
  reported range to its own confidently-tracked core rather than
  exporting the full re-seed-to-catch window uncritically.

  Cleanup done: the two now-superseded Cutie-based ball
  `capture_objects` (`efb975b0-...`, `540b0bf1-...`) and all dependent
  `detection_runs`/sequences/`pose_observations`/tracking rows deleted.

  Still open: a decision on whether this result is good enough to
  justify committing the three C++ fixes above and/or building native
  external-track import support into PoseTrak (Harri's own framing:
  decide after seeing this result).

- **2026-09-16** (segmentation targets extended to non-person capture
  objects; Cutie/SAM2 mask-centroid tracking tried and rejected for the
  ball) — Harri's idea: since the Cutie/SAM2 segmentation pass already
  exists for persons, add non-person `capture_objects` (the ball) as
  selectable segmentation targets too, then see whether a mask centroid
  alone is enough to position the ball. Traced the actual data flow
  first rather than assuming scope: `cutie_init_panel.py`'s
  `self._persons` is used everywhere purely as a list of opaque label
  strings (button creation, SAM object-id ordinals, the
  `seg_quality_runs.persons_json` snapshot), and `capture_objects` is
  structurally the same shape as `capture_persons` (id, capture_id,
  name, a body-definition FK, notes) -- so the whole fix was widening
  one query, unioning `capture_objects.name` into the existing
  `capture_persons` / `detection_track_assignments.person_name` set.
  Committed (`a8218f7`), with a new
  `test_persons_union_capture_objects` test.

  Then tried the ball itself: `python/tools/
  cutie_segment_ball_all_cameras.py` ran Cutie bidirectionally from a
  SAM2 box-prompt seed (from a prior constant-velocity UKF run's 3D
  position, projected into each camera) across all 6 cameras x 4
  throws, and `python/tools/finalize_ball_cutie_detection.py` wrote the
  resulting mask centroids into the DB the same way the Blender-track
  data above was later written. First tracking result looked
  numerically plausible (real z motion, 1.0m&rarr;0.57m) after fixing
  a bad seed-frame choice (was picking the median index of an already
  near-rest stretch of the seeding run; fixed to pick the max-z point
  within the throw's own core window) -- but Harri's own visual review
  of the rendered segmentation debug video showed the mask landing on
  body parts in every camera, not the ball, despite the plausible
  numbers. A repeated lesson this project keeps re-learning: trust
  direct visual inspection over aggregate numeric plausibility. Cutie
  (a general video-object-segmentation model, no person-specific logic
  -- confirmed by explaining SAM2's actual mechanics: a general
  promptable segmenter with no class labels, reliable on people only
  because full-body silhouettes are extremely well-represented in its
  SA-1B training data) drifts onto a nearby larger/higher-contrast
  region when the true target is small and fast-moving, a known
  general failure mode rather than a pipeline bug. Abandoned as a ball
  detector; the two resulting `capture_objects` DB entries
  (`efb975b0-...`, `540b0bf1-...`) are stale and still present,
  cleanup not yet done.

- **2026-09-14** (combined-fix skeleton's full-trial run completed:
  100% tracked, ankle_lat_L regression confirmed fixed, latest) — The
  full-trial run flagged as in-progress in the previous entry
  (`056b41dd-4efc-4d61-b786-163d62529666`, combined shoulder-width +
  ankle-fix skeleton) finished: 11588/11588 steps tracked (100%), 0 lost,
  152/11588 frames needed the PSD-eigensolver fallback. Comparing the
  same full trial against the shoulder-fix-only run that carried the
  ankle regression (`d4f78ce6...`): `ankle_lat_L` median reprojection
  error improved 37.6px&rarr;27.9px, `ankle_med_L` 35.7px&rarr;25.3px,
  `ankle_med_R` 37.5px&rarr;25.6px -- confirming the lineage-based fix
  actually resolved the regression, not just the skeleton's static joint
  offsets. Head/hip/shoulder markers are, as expected, unaffected by this
  fix (nose/ear/hip/shoulder medians match the previous run to within
  0.5px) -- the ankle fix only ever touched leg joints, and the
  still-elevated head-marker error remains the open item from the
  confidence-threshold entry above.

- **2026-09-14** (skeleton scaling: added a hip-to-ear derived neck+head
  measurement) — Follow-up to the head-marker jitter discussion:
  Harri asked for a "hips to ears" measurement in the skeleton-scaling
  UI, preferring it over "shoulders to ears" because shoulders move a lot
  with arm motion, so a time-range average anchored there carries more
  variation than one anchored at the (largely static) hips. Scaling
  itself is still spine-first: hip-to-shoulder continues to define the
  spine segment scale (`torso_height`, unchanged), and
  `(hip_to_ear - torso_height)` defines the neck+head chain's own scale
  independently.

  `scale_skeleton.py`: `template_measurements()` gains a derived `"head"`
  key (`hip_to_head_joint - torso_height`, using the `head` *joint*, not
  the ear markers, as `_fk_rest_pose()` only resolves joints); guarded to
  0.0 when `head`/`thigh.L`/`thigh.R` aren't present (e.g. a bare prop
  skeleton). `scale_skeleton_yaml()` gains a matching block: given a
  `hip_to_ear` measurement, it derives the head-chain ratio the same way
  and scales only the Y-component of `neck1`/`neck2`/`head`'s offsets by
  it -- mirroring the existing `_TORSO_HEIGHT_JOINTS` Y-only pattern, and
  leaving X/Z (and every other joint) untouched. `skeleton_scaling_panel.py`
  (the real PySide6 scaling GUI, not the read-only Marimo notebook) adds
  `"hip_to_ear"` to its already fully data-driven `MEAS_KEYS` list and the
  corresponding `_MeasWorker` midpoint-to-midpoint computation
  (`MRK-ear.L`/`MRK-ear.R` vs `MRK-hip.L`/`MRK-hip.R`) -- no other UI
  wiring was needed, confirming the panel's existing key-driven design
  paid off here.

  **Known imprecision, documented rather than chased further**: like the
  existing `torso_height` measurement, this is a Euclidean-distance
  *difference*, not a direct segment length -- a synthetic stress test
  showed a real ~15% relative discrepancy between the requested and
  re-measured head-chain length, proportionally larger than
  `torso_height`'s own error on the same class of approximation because
  the head segment is shorter. Not a logic bug (verified the ratio is
  computed and applied exactly as intended); left as a documented
  limitation rather than over-fit to one synthetic case.

  Added `python/tests/db/test_scale_skeleton.py` -- no prior test
  coverage existed for `scale_skeleton.py` at all. Covers the pre-existing
  limb/torso/shoulder-width scaling plus the new hip-to-ear path on a
  small synthetic skeleton (all offsets along Y, so expected results are
  exact rather than approximate). `pytest python/tests/db/
  test_scale_skeleton.py python/tests/app/test_skeleton_scaling_panel.py`
  passes (14 tests, no regression in the pre-existing panel tests).

- **2026-09-14** (shoulder-width skeleton fix confirmed on the full trial,
  but reintroduced an old ankle-lateral-offset regression; root-caused
  and fixed via skeleton lineage) — Ran the full ~97s trial
  (11588 steps, not the ~1799-step windows used for the earlier A/B
  tests) with "Nelli scale attempt 2026-09-14" (shoulder width corrected
  from the default female skeleton's 35cm to the measured 30.4cm; see
  the previous entry's re-examination of the head-marker asymmetry for
  why this was suspected). Comparing the same step range (0-1798) against
  the pre-fix skeleton's own run, torso markers genuinely improved
  (`MRK-hip.L` 33.8px&rarr;27.3px median, `MRK-hip.R` 38.4px&rarr;21.5px,
  `MRK-shoulder.L` 23.9px&rarr;20.7px, `MRK-shoulder.R` 27.2px&rarr;19.6px)
  while the head markers did not improve on this same slice
  (`MRK-nose` 44.9px&rarr;53.0px, `MRK-ear.L` 24.9px&rarr;34.3px,
  `MRK-ear.R` 44.7px&rarr;36.7px) -- consistent with Harri's own visual
  read (torso clearly better, head markers "still too high" and jittery)
  and with the previous entry's finding that the head-marker error isn't
  simple skeleton scaling. No true full-trial baseline exists at the
  pre-fix skeleton to compare against directly (the pre-fix runs are all
  ~1799-2399 step diagnostic windows), so this comparison is scoped to
  the overlapping step range only, not the whole trial.

  **Regression found on review**: Harri noticed `ankle_lat_L`'s lateral
  offset was wrong again in the shoulder-fix skeleton -- a problem
  already fixed once earlier this project (leg markers, 2026-09-0x).
  Root-caused via skeleton lineage (`skeletons.parent_id` chain, not by
  re-diagnosing the marker calibration itself): "Nelli scale attempt
  2026-09-14" descended from a *stale* pre-ankle-fix base
  (`leg.calibrated.2026-09-06-kare-tests`, created 07:59 that day), while
  every other skeleton in active use descended from the correct
  post-fix base (`v3, fully refit single-camera`, created 16:08 the same
  day -- the shoulder-fix scaling pass had simply been run from an old
  working copy). Fixed by diffing the shoulder-fix skeleton's joint
  offsets against its own (wrong) base to isolate exactly what the
  scaling pass itself changed (12 joints: spine1, spine2, shoulder.L/R,
  upper_arm.L/R, forearm.L/R, hand.L/R, foot.L/R; confirmed zero other
  marker-level diffs, so no unrelated regressions), then reapplying that
  same set of offset deltas onto the *correct* base -- producing a new
  combined-fix skeleton with both the shoulder-width correction and the
  ankle fix intact. Verified directly: `ankle_lat_L`'s marker offset is
  now `[0.0411, 0.3868, 0.0061]`, matching the correct base, not the
  wrong skeleton's `[0.1341, 0.2975, -0.0391]`. A full-trial run with
  this combined-fix skeleton is in progress; results to follow in a
  later entry.

- **2026-09-14** (confidence-vs-head-orientation analysis validates a
  generic confidence-threshold fix for head-marker jitter; per-camera
  hand-detection quality gap does not trace to pipeline stage or noise
  model) — Two follow-up investigations from Harri's head-marker/finger-
  quality observations, both requested with an explicit preference for a
  generic fix over head-orientation-specific tracker logic.

  **Head marker occlusion**: built a "visibility proxy" per (frame,
  camera, marker) -- `dot(normalize(marker_3d - head_center),
  normalize(camera_world_pos - head_center))`, where `head_center` is the
  per-frame mean of nose+ear.L+ear.R 3D positions (from
  `tracking_results.csv`) and `camera_world_pos = -R^T @ t` from the
  camera's own extrinsics -- and correlated it against ViTPose's raw
  (uncalibrated, not [0,1]) confidence score from `observations.csv`.
  Found a real positive correlation for all three head markers (Pearson
  r=0.55-0.62): binned confidence sits around 2.8-3.9 when the proxy says
  the marker is on the occluded/back side of the head (visibility &lt;
  -0.2), plateauing at 4.4-4.7 once the marker faces the camera
  (visibility &gt; 0). This confirms Harri's hypothesis (occlusion
  measurably lowers detector confidence) without needing any head-
  orientation-specific exclusion logic in the tracker, matching his
  stated preference -- **recommended fix: raise the confidence threshold
  for `nose`/`ear.L`/`ear.R` to roughly 3.5-4.0** (not yet implemented;
  needs checking against the rest of the confidence distribution for
  false-reject risk before picking a final number).

  **Per-camera hand-detection quality**: Harri observed gopro13_02 and
  insta_ace2_pro fingers were visibly jitterier/less plausible than
  gopro-11_mini_01 and oneplus9pro-01's. Checked whether all 6 cameras
  actually ran the dedicated hand-keypoint pass rather than falling back
  to full-body ViTPose for fingers: confirmed via `pose_observations`
  grouped by camera + source that every camera has substantial
  `hand_l`/`hand_r` row counts (6259-10227 range) -- no camera is
  missing the dedicated pass. Checked two plausible pipeline-level
  explanations and rejected both: mean `noise_scale` (crop_scale) for
  gopro13_02 (0.859) and insta_ace2_pro (0.811) is actually *lower*
  (better) than gopro-11_mini_01 (1.321) and oneplus9pro-01 (1.197), the
  opposite of what would explain worse detections; and mean/median raw
  confidence shows no such pattern either (insta_ace2_pro has the
  *highest* mean confidence, 1.972, despite being visually the worse of
  the two). Conclusion: the quality gap is most likely a genuine
  per-camera 2D detection/viewing-angle/lens characteristic, not
  something explainable (or fixable) at the noise-model or pipeline-
  stage level.

- **2026-09-14** (fixed the `marker_projections.csv` mode-ambiguity bug;
  head-marker reprojection error is real but asymmetric, not simple
  skeleton scaling) — Follow-up to the previous entry's finding
  #3. Fixed `TrackingExporter::write_frame()` (`cpp/src/io/
  tracking_export.cpp`): the per-(marker,camera) loop now checks
  `obs.mode` before building the exported "error" -- `VELOCITY`
  observations are skipped (would need the previous frame's projection,
  not available in this per-frame call); `PAIR_DIFF` observations are
  handled properly by projecting the *parent* marker too (via
  `ref_marker_id`, or the fixed cross-person `anchor_position` when
  that's what set the observation) and comparing
  `(child_proj - parent_proj)` against the already-relative `obs.position`
  -- apples to apples, instead of an absolute prediction against a
  relative observation. `POSITION` mode is unchanged. Rebuilt
  (`meson compile -C optbuild`), `./run_tests.sh --test-args="[export]"`
  passes (the existing `TrackingExporter basic functionality` test,
  `[export][io]` -- no dedicated regression test for the PAIR_DIFF case
  itself yet, a real gap; validated instead by re-running Nelli's own
  sequence against real data and checking the numbers directly, matching
  this project's own established practice for this kind of fix).

  **Verified against a fresh tracking run** (same sequence/skeleton/
  config as the original-noise baseline): every previously-corrupted
  marker's median "error" dropped from the 1800-2600px artifact range
  into a sane 5-50px range (e.g. `MRK-thumb3.R` 5px, `MRK-wrist.L` 20px,
  `MRK-Ankle.L` 9.5px -- all now physically plausible for a run that
  tracked 100%).

  **Re-examined the head-marker question on the now-correct data --
  real, but not simple skeleton scaling**: `MRK-nose` (49px) and
  `MRK-ear.R` (46px) are genuinely elevated vs. `MRK-elbow.L/R` (23px,
  22px) and `MRK-wrist.L/R` (20px, 22px) -- and elbow/wrist's own ~22px
  sits close to what the configured noise model itself predicts
  (`pose_noise_std`=13px * ~1.6 average `crop_scale` &asymp; 21px),
  suggesting elbow/wrist are behaving close to nominal while nose/ear
  are not. But **`MRK-ear.L` (26px) is barely elevated at all -- nearly
  half `MRK-ear.R`'s 46px** -- and a pure skeleton-scale error (a wrong
  head-segment size or offset) would affect both ears symmetrically,
  since the geometry is mirror-symmetric by construction. `MRK-hip.L/R`
  (32/38px) and `MRK-shoulder.L/R` (24/29px) show the same right-higher-
  than-left pattern, more mildly. This argues *against* Harri's original
  skeleton-scaling hypothesis as the primary explanation and *for*
  something more specific to viewing angle or camera calibration on her
  right side/face -- not yet root-caused. **Open next steps**: (1) check
  whether nose/ear-R's error correlates with a specific camera or a
  specific head-orientation range (e.g. worse only when she's turned
  away from certain cameras) rather than being uniform across the whole
  run; (2) hip/shoulder's own moderate (not dramatic) elevation over
  elbow/wrist's noise-model-consistent baseline is still worth a look --
  possibly a torso-segment scale/offset issue distinct from the more
  striking head-asymmetry finding, not necessarily the same root cause.

- **2026-09-14** (person+prop coupling A/B test, plus three real findings
  it led to) — Tried the person+pen/pad coupled tracking run
  Phase 3 flagged as unexercised (previous entry): `--person ec1b3e2f...
  119a24b5... b0eff656...` (Nelli + pen + pad) via `MultiPersonTracker`.
  It ran without crashing, and a direct C++ source trace (not just
  behavioral testing) confirmed `update_contact_pairs()`/
  `build_cross_person_anchors()` are genuinely wired into every subject's
  per-step update, contradicting the CLI's own stale `--person` help text
  ("No cross-person coupling yet (Stage 1)"). Both configs used had
  `cross_person_max_world_mm=400` (not disabled).

  **A/B result (coupled vs. `cross_person_max_world_mm=0` on Nelli's own
  config): coupling does not help, and introduces localized jitter.**
  Wrist-to-pen distance is essentially unchanged (median 17.60cm coupled
  vs. 17.00cm without -- no measurable improvement in hand/prop
  agreement). Frame-to-frame wrist jitter in one specific window (frames
  1218-1224) is 3-4x higher *with* coupling (e.g. 1.10-1.28cm/frame vs.
  0.15-0.44cm/frame) while the *overall* jitter distribution across the
  whole run is nearly identical between the two runs -- consistent with
  a contact pair activating/deactivating right around there and
  perturbing the fit transiently, not a general problem. **Open
  question, not yet root-caused**: why does that specific window trigger
  a transient contact-pair change, and can the anchor noise floor/gating
  be tuned to avoid the perturbation without losing whatever benefit
  coupling is meant to provide elsewhere.

  **Real finding #1 -- root-caused a reported tracking artifact**: Harri
  reported the left shin "turns upward" around frames 1015-1020 in the
  no-coupling run specifically (not visible in the coupled run). Checked
  first against raw per-camera 2D hand-keypoint detections around the
  *hand* jitter window (a different, unrelated hypothesis Harri floated
  about the pen's ArUco marker confusing the hand-keypoint detector on
  gopro13_02) -- all 5 cameras' real 2D detections there are smooth and
  continuous, ruling that out directly. The shin issue itself traces to
  something else entirely: **`ankle_lat_L` and `ankle_med_L` both have a
  genuine 2-frame detection gap at frames 1018-1019, then reacquire on
  the same cameras at wildly discontinuous pixel positions** (e.g.
  camera 1's `ankle_lat_L` jumps from a stable ~(2100,1870) to
  (2315,1297) the instant it resumes) -- every marker in the shin/foot
  group (ankle, heel, toe) swings together by up to ~70cm in Z while the
  knee markers barely move, confirming the error is localized at/below
  the ankle joint, not a whole-leg problem. This is the exact
  reacquire-onto-the-wrong-candidate-after-occlusion failure mode this
  session already characterized and fixed for the pen's reflective bands
  (`prototype_pen_band_tracking.py`'s bootstrap+gate design) -- it just
  hadn't been noticed in the production leg-dot path before. Matches
  Harri's own recollection that the same class of problem existed in
  markerless-only runs and was fixed by adding leg markers -- this
  particular occurrence is that same underlying gap resurfacing when
  reacquisition itself briefly fails, not a general regression. **Open
  idea for a real fix**: apply the same tracking-gate discipline (never
  reseed from a bad guess after a gap, verify consistency before
  trusting a resumed detection) to leg dot-marker reacquisition
  specifically, not just prop-dot tracking.

  **Real finding #2 -- reducing measurement noise makes jitter *worse*,
  not better**: Harri's hypothesis was that the configured noise values
  (`calib_noise_std`=25px, `pose_noise_std`=13px -- confirmed via direct
  code trace: DB column `measurement_noise_std` legacy-maps to
  `calib_noise_std`; `session_reader.cpp` line ~255) are too large given
  this capture's actual (small) calibration error, and that tightening
  them should reduce hand-tracking jitter. Tested directly (new config,
  `measurement_noise_std`=10, `pose_noise_std`=6, full single-person
  rerun): **median wrist jitter went from ~0.51-0.52cm to 1.27cm (p95
  1.90-1.95cm -> 9.49cm, max ~17cm -> 38.6cm)** -- 2-5x worse, the
  opposite of the hypothesis, despite the tighter-noise run reporting
  *better* numerical conditioning (0 PSD-eigensolver fixes vs. 18-21 in
  the looser-noise runs). Kalman-filter-theoretic explanation: telling
  the filter the sensor is more precise than it really is makes it trust
  *every* observation more strongly, amplifying the effect of any single
  bad/outlier detection into a larger state correction instead of
  damping it -- the original 13/25px values are very plausibly providing
  real, load-bearing robustness against exactly the kind of bad
  reacquisition found in finding #1, not simply padding out an
  overcautious noise budget. **Open implication**: the real fix for hand
  jitter is very likely at the *detection/gating* layer (reject or gate
  bad observations before they reach the filter), not the noise-trust
  knob -- tightening noise without first fixing the underlying bad-data
  problem looks actively counterproductive on this evidence.

  **Real finding #3 -- a third instance of the mode-ambiguity export
  bug**: investigating Harri's separate question ("is head-marker
  [nose/ear] reprojection error big because of a skeleton-scaling
  issue?") via `marker_projections.csv` surfaced a real bug in that
  export, not (yet) a skeleton-scaling answer: many markers --
  `MRK-wrist.*`, `MRK-Ankle.*`, `MRK-heel.*`, `MRK-*Toe.*`,
  `MRK-elbow.*`, `MRK-knee.*`, plus the hand/finger set -- show median
  "reprojection error" of **1800-2600px**, physically nonsensical for a
  run that tracked 100%. Root cause: the export subtracts `obs_x/y` from
  `proj_x/y` unconditionally, without checking measurement mode first --
  for any observation using `PAIR_DIFF` (relative, child-minus-parent)
  encoding, `obs_x/y` is a small relative offset, not an absolute pixel
  position, so comparing it against an absolute `proj_x/y` produces a
  huge, meaningless "error". This is the exact same mode-ambiguity bug
  class `docs/roadmap/features/observation-results-semantics.md` already
  documents as having caused two real bugs before -- a third instance,
  in the CSV diagnostic export this time. **Not yet fixed** (next up).
  After manually filtering out the obviously-corrupted rows, a
  preliminary (still bug-tainted) look shows `MRK-nose` (49px) and
  `MRK-ear.R` (46px) moderately -- not dramatically -- higher than
  `MRK-hip.L/R` (32-38px) and `MRK-shoulder.R` (29px), a real but modest
  signal partially supporting the skeleton-scaling hypothesis. Harri's
  own reaction: the hip/shoulder numbers themselves look higher than
  they should be too -- needs the export bug fixed and a clean re-pull
  before trusting any of these numbers, head or otherwise.

  **Open idea -- a better jitter metric**: raw frame-to-frame Euclidean
  distance (used throughout this investigation so far) doesn't
  distinguish real fast movement from noise. Literature-standard
  alternative: an acceleration-based smoothness metric (mean magnitude
  of the 2nd derivative of joint position -- reported as "Accel" in
  VIBE and similar pose-estimation-smoothness literature; grounded in
  the biomechanics finding that real voluntary human movement is
  empirically jerk-minimizing, Flash & Hogan 1985, so elevated
  acceleration/jerk flags non-physiological noise specifically). Harri's
  own "distance to a 5-frame median/mean" idea is in the same family;
  a Savitzky-Golay-smoothed local polynomial fit would refine it further
  by not penalizing genuine acceleration/deceleration the way a flat
  windowed mean or median would. Not yet implemented.

- **2026-09-13** (validated the production tracker on the pen+pad
  objects end-to-end) — Phase 3's own "validate the mechanical
  path first" item (`marker-mocap-productization-plan.md` §4): registered
  the pen and pad as real `capture_objects` and ran the actual
  `posetrak-tracker` UKF pipeline against them, not a standalone script.

  The missing piece turned out to already exist and be tested:
  `app.pose.finalise.finalise_object_to_db()` reads an ArUco detection
  run bound to a `capture_object_id` and writes one
  `pose_observation_sequence` automatically (no stitching needed -- one
  prop is always one track). New `setup_pen_pad_capture_objects.py`
  wires this up end-to-end per CLAUDE.md's append-only convention:
  imports each object's marker body YAML, inserts a `capture_objects`
  row, then -- rather than mutating the existing multi-purpose ArUco run
  (`75cbf678...`) -- creates a fresh, object-bound `detection_runs` row
  and *re-encodes* (not byte-copies) the relevant `detection_keypoints`
  rows onto it, so the new run's `config_json.marker_ids` is a clean,
  minimal list (`["2","3"]` for the pen, `["1"]` for the pad) rather than
  inheriting the original 8-marker wire format. Generates + imports each
  object's prop skeleton (`marker_body_to_skeleton`) to finish.

  Pad uses marker "1" only -- not "0": the calibration-box ID collision
  (previous entry) means "0" sightings are frequently the box, not the
  pad, and a single ArUco marker already fully determines a rigid body's
  6-DOF pose, so dropping "0" avoids relying on the tracker's own outlier
  rejection to save us from a problem this script can just sidestep.
  New `catalog/pen.calibrated.2026-09-06-kare-tests.real-capture.yaml`
  and `catalog/pad.calibrated.2026-09-06-kare-tests.marker1-only.yaml`
  (the pad one built from the calibration-video's validated relative
  pose, since marker "1" alone needs no cross-marker geometry at all --
  its own flat template is the whole rigid body).

  **Ran the real `posetrak-tracker track` CLI** (`optbuild/`, not the
  stale `~/.posetrak` copy -- `default_binary_path()`'s own documented
  trap) against both new sequences, `--smooth` on, factory-defaults
  tracker config (`seed_baseline_tracker_config()`). **Pen: 97.2% tracked
  (1747/1798 steps), 0 lost, rigid-body init RMS 1.8mm. Pad: 94.9% tracked
  (1708/1800), 0 lost, RMS 3.6mm.** Cross-validated the smoothed output
  against the standalone trajectory script's own numbers (previous
  entries): applying the same calibrated tip offset to the tracker's own
  `smoothed_root_pose.csv` reproduces the identical physical story --
  ~10-15cm tip-to-pad distance through the writing period, rising to
  35-41cm as the pen is lifted away -- from a completely independent
  computation path (real UKF + RTS smoothing, not a bespoke script).

  Not yet done: a coupled person+pen/pad run using `MultiPersonTracker`'s
  `--person`-mode cross-subject anchors (`build_cross_person_anchors()`)
  to minimize hand-to-pen/pad error jointly -- the next item on Phase 3's
  own "validate the mechanical path" list, still unexercised for a
  rigid-body (non-person) subject.

- **2026-09-13** (exported pen+pad trajectory into Blender) —
  Built a two-stage Blender export for the pen+pad trajectory (same
  split as `blender_add_cameras.py`: a plain-Python stage does the real
  work, a `bpy`-only stage consumes its JSON, since Blender's bundled
  interpreter has none of this project's own dependencies). Exports
  "pen" and "pad" as full animated rigid objects (position + rotation
  keyframes from their own solved poses), with a static "pen_tip" empty
  parented to "pen" at its calibrated local offset -- deliberately not a
  pre-computed world-space tip trajectory: per Harri's own framing,
  Blender's dynamic-paint system should be able to detect the pen-tip-
  to-pad proximity itself once both real rigid objects are present, the
  same way it would for any other brush/canvas pair, and Blender's own
  parenting propagates the pen's full animated pose to the tip
  automatically (verified algebraically before relying on it: a
  *local*, parent-relative offset needs no Z-up/Y-up conversion at all,
  only the parent's own world position/rotation do).

  Reuses `posetrak.export.common`'s already-established coordinate
  conversion (`_coord_matrices("yup")`, `quat_to_matrix`,
  `matrix_to_quat_components`) rather than re-deriving axis conventions.

  **Real bug found and fixed before trusting the result**: the first
  version computed Blender frame numbers directly from the capture's own
  global timestamp (`round(time_s * fps)`) -- since this clip is
  98-113s into a much longer recording, every keyframe landed around
  frame 1960-2260. It "worked" (Blender happily stored the keyframes)
  but the object would render as completely frozen for the entire
  default 1-250 timeline, silently. Fixed by normalizing to the clip's
  own start (`(time_s - time_s.min()) * fps`) before trusting the
  output.

  **Verified in a real headless Blender 4.5 run**, not just by reading
  the exporter's own code: queried the saved `.blend`'s evaluated
  `matrix_world` (i.e. asked Blender itself to resolve the parented
  transform, the same mechanism its dynamic-paint proximity detection
  would use) at several frames. Tip-to-pad distance came back 59.9cm
  (frame 0, handoff) -> 13.6-18.8cm (frames 100-200, writing) -> 40.1cm
  (frame 300, pen lifted away) -- matches the earlier NumPy-only
  analysis almost exactly. Harri independently confirmed the pen motion
  "looks reasonable" opening the file directly.

  New files: `export_pen_pad_to_blender.py` (plain Python),
  `blender_add_pen_pad_animation.py` (`bpy`). The saved `.blend` itself
  is a validation artifact only, not committed.

- **2026-09-13** (validated real-capture pen+pad prop tracking end-to-end) —
  Continuation of the pen+pad prop-tracking pivot: resolved the
  pen's calibration-video tag-3 failure by switching to the real multi-
  camera capture's own cross-camera co-occurrence, then built a full
  pen+pad trajectory across the actual 98-113s writing scene.

  **Pen ArUco body, real capture vs. calibration video**: re-ran
  `calibrate_rigid_marker_body.py --marker-ids 2 3 --reference-id 2`
  directly against the real 6-camera writing-scene footage (not the
  separate calibration video that had produced garbage tag-3 data,
  175-1200px reprojection error) -- confirmed first that markers 2 and 3
  are genuinely co-visible from different cameras at the same
  synchronized instant in 268 of 451 time-bins (59%) across the window,
  the actual precondition for this mechanism. Result: tag 3 solved from
  26 co-occurrence samples, corners forming a near-perfect square
  (9.476-9.478cm edges against a 9.5cm marker) and a normal 178.6° from
  tag 2's own -- genuinely opposite-facing, matching the physical prop --
  with **no twist ambiguity**, unlike the calibration-video's 2-point
  Kabsch fit. This fully supersedes the earlier calibration-video tag-3
  result.

  **Reflective-band (dot) calibration attempt, `--detect-dots`**: failed
  outright first try (0 usable samples, every bucket "ambiguous").
  Diagnosed via direct visual inspection (not more parameter tuning):
  the ~150-candidates-per-frame reading that first suggested "the whole
  scene is this cluttered" was dominated by the two-person pen-handoff
  transition at the very start of the window (98-99s); restricting to
  the calm one-person writing period (101-111.5s) alone dropped per-frame
  candidates from ~150 to a 14-30 plateau. A second fix -- explicitly
  masking each visible marker's own (dilated) quad, since a marker's own
  high-contrast edges leak through as background-subtraction residual
  whenever it moves slightly between the background's own sample frames
  -- didn't reach the "exactly 1 candidate" this script's correspondence
  mechanism needs, but did clean up the *remaining* clutter enough that,
  zoomed into one calm frame, the real band candidates were identifiable
  among a short list rather than buried in generic noise. New
  `--dot-quad-exclude-scale` and a `--camera-labels` filter (see pad
  section below) added to `calibrate_rigid_marker_body.py`.

  **Reflective-band tracking, `prototype_pen_band_tracking.py` (new)**:
  since the script's own "exactly 1 unambiguous candidate per camera"
  correspondence can't resolve a 14-30-candidate field, built a dedicated
  temporal tracker instead -- bootstrap each band from one manually-
  confirmed (camera, frame, pixel) seed (automatic bootstrap heuristics
  were tried and shown unreliable in this cluttered scene), then track
  forward frame-to-frame via nearest-candidate-within-gate_px (same
  policy already validated for the calibration video's tag-2 dot),
  independently per camera. Combined across cameras *and* time via a
  "virtual projection" DLT triangulation: each tracked (camera, frame)
  sighting's projection matrix is composed with that instant's own
  already-solved marker-2 world pose, so sightings never need to be
  simultaneous -- the same cross-time/cross-camera co-occurrence bridging
  `calibrate_rigid_marker_body.py`'s own ArUco corner registration
  already relies on, just applied to an unknown point instead of a known
  corner offset. Tracked the pen-tip band (the one relevant to the
  pen-tip-to-pad accuracy goal) across 395 frames on gopro13_02 and 118
  on pixel9; 405 of 513 pooled sightings had a bucket with a solved
  marker-2 pose. **Result: local-frame offset [0.306, 0.033, -0.005]m,
  reprojection error median 2.33px / p90 8.53px** (381/405 rows under a
  15px inlier threshold) -- comparable to the ArUco tag-2 fit's own
  1.62px, nowhere near the broken tag-3-from-calibration-video attempt's
  175-311px. The "top" band (near the tags) was not seeded this pass --
  not cleanly separable from marker-edge residue in the one frame
  checked, and the tip band was the priority.

  **Full pen trajectory, `prototype_track_pen_trajectory.py` (new)**:
  standalone per-instant 6-DOF trajectory across the whole 98-113s window
  (NOT the production object-tracking path -- see caveat below), reusing
  `solve_marker_pose()`: solves marker 2 directly when visible, falls
  back to solving marker 3 and rigid-registering its already-known local
  corners against marker 2's frame when only marker 3 is visible (roughly
  doubling coverage since the two tags rarely both hide at once). **244
  of 300 buckets solved (223 direct + 21 via the marker-3 fallback),
  spanning the full window with only 9 gaps >0.15s (largest 0.80s)**.
  Derived the tip's world trajectory from the calibrated local offset
  above. Found (via a rolling-median outlier check, not by eye alone) 24
  isolated single-bucket glitches -- classic single-frame bad-solve
  artifacts a real outlier-rejecting tracker (Mahalanobis gating, RTS
  smoothing) would filter automatically; this standalone script has none,
  so they were removed by a simple rolling-median filter instead. The
  cleaned trajectory is smooth and physically coherent end to end:
  standing handoff (z~1.4m) -> drop to writing height (~101s,
  z~0.6-0.9m) -> stable writing motion (101-110.5s) -> brief
  repositioning (~111s) -> settled continuation.

  **Pad calibration, real capture -- a second real ID-collision bug
  found**: naively assumed markers 0/1 (the pad's own IDs, confirmed by
  Harri) would calibrate the same way as the pen's 2/3. First attempt
  against all 6 cameras gave a garbage marker-1 corner std of
  [7,19,22]cm (vs. sub-1.3cm for every other marker calibrated this
  session). Root cause (confirmed by Harri, not just inferred): **the
  extrinsics calibration box mistakenly carries a same-ID (DICT_4X4_50,
  id "0") ArUco tag on one of its faces** -- meant to be a DICT_5X5 tag,
  wrong marker attached by mistake; also confirmed any marker ID >10 in
  this capture's detection run is a false positive. Restricting to 3
  cameras whose marker-"1" sightings showed clear real motion improved
  but didn't fix it (std still [6,3,2]cm) -- direct inspection of
  marker-0's own raw pixel trajectory per camera showed it's **frozen at
  one fixed screen position for nearly the entire window in every camera
  checked**, with only rare, isolated deviations -- the box's mistaken
  tag dominates "0" sightings almost everywhere, not just occasionally.
  Fix: dropped marker "0" from the real-capture solve entirely; used
  marker "1" (no known ID conflict, confirmed clear real motion in every
  camera) as the pad's sole real-capture anchor, reusing the
  *already-validated* calibration-video pad geometry (median translation
  36.93cm, rotation 4.04°, std <=2.7cm, from the earlier
  `prototype_calibrate_pen_and_pad.py` run) for marker 0's known-but-
  unneeded offset. Added `--camera-labels` to `calibrate_rigid_marker_
  body.py` to support this kind of camera-subset restriction generically.

  **Combined pen+pad trajectory**: extended `prototype_track_pen_
  trajectory.py` to also solve the pad's pose (via marker 1 alone) in
  the same per-frame pass -- zero extra decode cost, since ArUco
  detection already finds every marker in frame regardless of which IDs
  are used downstream. **Pad solved in 226 of 244 buckets (92.6%), zero
  outliers** (vs. 24 for the pen) -- confirms the marker-1-only fix is
  clean. Pad position is essentially static after ~100.5s (she sets it
  on her lap and only the pen moves), matching physical expectation
  exactly. **Tip-to-pad distance**: 61cm during the handoff -> drops
  sharply as the pen approaches -> stable 10-20cm band throughout the
  main writing period (101-110.5s, fluctuations matching real
  stroke-to-stroke motion) -> rises to ~40cm and stays there from ~111s
  (pen lifted away). This is the pen-tip-relative-to-pad motion Harri
  asked for; caveat -- it's distance to marker 1's own mounting point,
  not the paper surface itself, which would need its own (likely small,
  fixed) offset calibration if exact contact timing matters.

  **Explicit scope caveat**: everything in this entry is a standalone
  Python trajectory-export prototype (`prototype_pen_band_tracking.py`,
  `prototype_track_pen_trajectory.py`), not the production C++ UKF
  tracker -- no smoothing, no Mahalanobis outlier gating beyond the
  rolling-median bolt-on described above. The production single-object
  tracking path (`ObjectPanel`/`ObjectRunTrackerDialog`, same UKF
  pipeline used for people) already exists and works, but wiring this
  specific pen/pad object into it needs the ArUco/dot observations
  finalized into a `pose_observation_sequences` row and a
  `capture_objects` entry, neither of which exist yet for this capture --
  a real integration gap (`marker-mocap-productization-plan.md` §3.4/§4
  already flags multi-subject/object-tracking-launch validation as "not
  yet exercised"), not something this investigation closes.

- **2026-09-13** (fixed `ObservationSequence::get_in_range()`) —
  Harri: "profile `observations.get_all_in_range()` next -- or maybe
  worth a check of the code, this sounds like it is doing something
  stupid that might be visible with just inspection." It was:
  `ObservationSequence::get_in_range()` did a full linear scan over the
  *entire* per-camera observation history on every single call, when
  `observations` is already guaranteed sorted by timestamp (verified
  before trusting it, not assumed -- the DB loader's own query is
  `ORDER BY camera_instance_id, video_frame` (`session_reader.cpp`),
  the legacy JSON loader explicitly sorts by frame number before
  appending (`observation_loader.cpp`), and nothing anywhere mutates an
  already-built sequence's `observations` afterward). Replaced the scan
  with `std::lower_bound` binary search on that invariant. Added a
  regression test specifically for the tie-handling this depends on
  (several observations sharing one timestamp per frame, queried at
  boundaries that land inside vs. between tied runs) -- the existing
  `get_in_range()` tests already used strictly-increasing timestamps, so
  wouldn't have caught a tie-handling bug in a naive rewrite.
  `./run_tests.sh`: all 372 cases pass (371 + this one).

  Re-profiled the same 2399-frame window: `observations.get_all_in_range()`
  dropped **33.89 -> 0.044 ms/frame (770x)**, exactly the O(n) -> O(log n)
  result predicted. Real observed throughput: **~5.45 -> ~6.74 fps
  (+24%)**, precisely measured from file timestamps. (First wall-clock
  sample during verification looked unchanged -- a noisy early read from
  a 20-second partial window, not trusted once the full run's real
  numbers came back; flagging the correction rather than the wrong
  number.) Frame-time reconciliation still closes to ~2.5% residual.

  This fix is not dot-marker-specific -- `get_in_range()` is called by
  every tracking run in this project, marker-augmented or not, so this
  benefits markerless tracking too, not just today's leg-module work.

- **2026-09-13** (closed the frame-time accounting) — Harri
  caught a real error in the previous entry's own numbers: "if update()
  is 61ms and predict_marker_slots() is 4.13 it does not add up" against
  the ~180ms/frame real total. Correct: `u_kalman_ms` (61ms) is a
  *sub-step inside* `update_ms` (107ms total), not a sibling of
  `predict_marker_slots()` -- and even `predict_ms + update_ms +
  predict_marker_slots()` (123ms) left 64ms/frame, over a third of the
  whole budget, completely unmeasured by anything.

  Built `frame_step_profile` (env-var-gated,
  `POSETRAK_PROFILE_FRAME_STEP`) instrumenting every previously-untimed
  piece of a dot-augmented step in `track.cpp`/`multi_person_tracker.cpp`:
  `bucket_candidates_by_camera()`, `resolve_shared_dot_assignment()`
  (assignment-only, with the nested `predict_marker_slots()` cost
  subtracted via a `dot_predict_profile::snapshot()` delta so it isn't
  double-counted), `observations.get_all_in_range()`, predicted-
  observations export, state-vector export, the posterior-state FK
  call, and both output writers (CSV exporter, DB result writer).
  `./run_tests.sh`: all 371 cases still pass (no behavior change, timing
  only).

  **Residual closed from 64ms/frame (35% unaccounted) to ~3.5ms/frame
  (~2%)** on the same 2399-frame window:

  | piece | ms/frame | % of real total |
  |---|---|---|
  | `update()` (incl. `u_kalman_ms` ~61ms) | 108.11 | 58.9% |
  | `observations.get_all_in_range()` | 33.89 | 18.5% |
  | `predict()` | 11.10 | 6.0% |
  | `resolve_shared_dot_assignment()` (assignment only) | 10.93 | 6.0% |
  | `bucket_candidates_by_camera()` | 5.17 | 2.8% |
  | `predict_marker_slots()` | 4.05 | 2.2% |
  | everything else (predicted-obs export, both writers, posterior FK) | 6.78 | 3.7% |
  | **real observed** | **183.55** | (5.45 fps) |

  **Real, surprising finding**: `observations.get_all_in_range()` --
  just fetching this frame's real pose-keypoint observations, before
  doing anything with them -- costs 33.9 ms/frame, **more than 8x**
  today's whole `predict_marker_slots()` optimization target, and is
  not dot-marker-specific at all (every tracking run calls this, dot-
  augmented or not). Smells like a linear scan or unindexed lookup
  against a large, session-wide observation collection, re-run every
  frame. Flagged as the next concrete profiling/fix target -- likely
  higher-value and easier than anything left on the dot-prediction
  side, and its fix would help every tracking run, not just marker-
  augmented ones. Not investigated further this pass.

- **2026-09-13** (batched dot-slot prediction across cameras) —
  Harri's direct question ("why don't we batch all cameras for marker
  slot matching -- rerunning FK 6 times sounds self-evident") had no
  real answer beyond "not built yet": the per-camera API shape
  (`Tracker::predict_dot_slot_predictions(camera_id)`) was inherited
  from the rigid-body path, where per-camera cost is genuinely
  negligible (closed-form, no sigma points) -- nothing about the
  articulated case actually required it. Built
  `predict_dot_slot_predictions_all_cameras()` (`Tracker`) and
  `UnscentedKalmanFilter::predict_marker_slots_all_cameras()`: batch
  every (marker, camera) pair the caller needs into one
  `predict_measurements()` call per sigma point, so sigma-point
  generation and the FK sweep each run once per *frame* instead of once
  per camera -- `predict_measurements()` needed no change at all, since
  it already reads `camera_id` independently off each observation.
  `resolve_shared_dot_assignment()` (`dot_assignment.cpp`) now calls
  this once per subject instead of looping per camera. Added a real
  regression test asserting the batched result matches calling the old
  per-camera path once per camera, bit-for-bit
  (`isApprox(..., 1e-9)`), not just "looks reasonable" -- on a fixture
  with 2 dot markers on different joints so the (marker, camera)
  indexing this change introduced is actually exercised.
  `./run_tests.sh`: all 371 cases pass (370 + this one).

  Re-profiled the same 2399-frame window: `calls: 2399` confirms the
  batching worked (was 14385 = once per camera). `predict_marker_slots()`'s
  own total dropped **19.66 -> 4.13 ms/frame (4.8x further)** -- sigma
  generation 12.13 -> 2.05 ms/frame, the FK+projection loop 6.79 -> 1.26
  ms/frame. `update()`'s `u_kalman_ms` again measured unchanged (60.14
  -> 61.28 ms/frame, within run-to-run noise across all three profiling
  passes today). Real observed throughput: **~4.96 -> ~5.35 fps**.

  Combined across both of today's dot-slot-prediction fixes:
  `predict_marker_slots()` 61.49 -> 4.13 ms/frame (**14.9x**), real
  throughput ~4.1 -> ~5.35 fps (+30%). `update()`'s own Kalman-gain
  cost (~61 ms/frame) is now clearly the largest remaining piece of the
  whole frame, well ahead of dot-slot prediction (~4 ms/frame) -- the
  one still-open item from the productization plan's marker-count-
  scaling question, and a materially different, bigger piece of work
  (inside the shared body+hand update, not the dot-specific path),
  not started here.

- **2026-09-13** (parallelized `predict_marker_slots()`) —
  Harri's own question after the profiling entry below ("we already
  parallelize FK for markerless keypoints -- why not for marker slots
  too?") was right: `update()`'s equivalent per-sigma-point loop
  (`ukf.cpp` Step 2) already runs under `#pragma omp parallel for` with
  a per-thread `ForwardKinematics` from the existing `data_pool_`;
  `predict_marker_slots()`'s loop was plain sequential. Gave it the
  identical treatment (same `ensure_data_pool()`/`data_pool_` machinery,
  already `const`-compatible). `./run_tests.sh`: all 370 cases still
  pass.

  Re-profiled the same 2399-frame window: the `predict_measurements()`
  loop (FK+projection) dropped **49.07 -> 6.79 ms/frame (7.2x)**;
  `predict_marker_slots()`'s own total dropped **61.49 -> 19.66 ms/frame
  (3.1x)**. `update()`'s own `u_kalman_ms` measured unchanged (58.68 ->
  60.14 ms/frame, within run-to-run noise) -- confirms this fix didn't
  touch that separate, larger cost center, and rules out a confound in
  the comparison. Real observed throughput (from file birth/modify
  timestamps, not estimated): **~4.1 -> ~4.96 fps (+21%)**, for a five-
  line change reusing infrastructure that already existed.

  **New bottleneck, and it points straight at the next fix.** With the
  loop no longer dominant, sigma-point generation is now 61.7% of
  `predict_marker_slots()`'s own cost (12.13 ms/frame) -- and it is
  called once *per camera* (6x/frame) from identical inputs (the same
  prior `state`/`covariance` every time, `camera_id` never enters sigma
  generation at all). Same redundancy shape as the marker-batching fix
  from the previous entry, one level up: cameras, not markers. Harri's
  direct question ("why don't we batch all cameras... if we rerun FK
  unnecessarily 6 times that sounds self-evident") is correct and not
  yet built -- the per-camera API shape
  (`Tracker::predict_dot_slot_predictions(int camera_id)`) was inherited
  from the rigid-body path, where per-camera cost was genuinely
  negligible (`predict_rigid_marker()` is closed-form, no sigma points
  at all), not a deliberate choice for the articulated case. No
  technical obstacle found to fixing it; scoped as the next concrete
  piece of work, restructuring so sigma generation and FK both run once
  per frame and only the (cheap) camera projection step repeats per
  camera.

- **2026-09-13** (dot-slot prediction profiled for real) — Per
  Harri's own priority ("do the profiling first"), built real
  instrumentation instead of guessing further: new
  `dot_predict_profile` (env-var-gated, `POSETRAK_PROFILE_DOT_PREDICT`)
  splits `UnscentedKalmanFilter::predict_marker_slots()`'s own cost into
  sigma-point generation, the per-sigma-point `predict_measurements()`
  loop (FK + projection combined), and mean/covariance aggregation.
  Ran a 20 s / 2399-frame window (`--start-time 40 --end-time 60`) with
  the final dot-augmented skeleton, and a matched dot-free baseline on
  the same window for direct comparison. `./run_tests.sh`: all 370
  cases still pass.

  **Real numbers, not estimates:**
  | | baseline (no dots) | dot-augmented (16 markers) |
  |---|---|---|
  | `predict()` + `update()` (shared UKF step) | 113.08 ms/frame | 117.52 ms/frame (+3.9%) |
  | `predict_marker_slots()` (separate, dot-only) | n/a | 61.49 ms/frame |

  **This overturns part of last night's own assumption.** The shared
  UKF `update()` step's Kalman-gain computation (`u_kalman_ms`) is the
  single largest cost in the whole frame (~58-62 ms/frame) — but it
  barely changes with 16 dot markers added (58.68 -> 62.11 ms/frame).
  It is a large, *pre-existing* cost of this skeleton's whole-body+hand
  UKF (~366 pose observations across 6 cameras already, before any dot
  markers), not something the dot-marker work introduced. All of the
  real added cost from dot markers is isolated in the separate
  `predict_marker_slots()` calls, which run *before* assignment, outside
  `update()` entirely.

  Within `predict_marker_slots()` itself (14385 calls over 2399 frames,
  ~6/frame matching 6 cameras, 437 sigma points, 16 markers/call):
  sigma-point generation is 19.2% (11.78 ms/frame), the
  `predict_measurements()` loop (forward kinematics + camera projection,
  combined — not split further since that function is also on the main
  `update()` hot path and splitting it there would conflate the two
  callers) is **79.8%** (49.07 ms/frame), and mean/covariance
  aggregation is 1.0% (0.64 ms/frame).

  Reconciling with the whole frame: baseline + `predict_marker_slots()`
  sums to 179.0 ms/frame (5.6 fps) but the real, previously-measured
  full-run throughput was ~4.1 fps (244 ms/frame) — leaving ~65 ms/frame
  not accounted for by either measured piece. Honestly unattributed:
  most likely `resolve_dot_assignment()`'s cost-matrix/Hungarian-solve
  step and/or the larger CSV write volume (77 vs. 61 markers/frame
  across `observations.csv`/`predicted_observations.csv`/
  `marker_projections.csv`), neither profiled this pass.

  **What this means for the productization plan's marker-count-scaling
  open question**: two separate cost centers, not one, and they don't
  necessarily scale the same way.
  1. The `predict_measurements()` loop's 79.8% share is exactly what
     the *already-identified* cross-camera batching idea (one call per
     frame instead of once per camera — the redundant-FK-across-cameras
     analogue of the redundant-FK-across-markers fix from the previous
     entry) would target — real, well-scoped follow-on work.
  2. The update()'s own Kalman-gain cost is large but *unaffected* by
     today's 16 dot markers specifically because 16 is small next to
     the ~366 pose observations already there — this should **not** be
     read as "dot marker count doesn't affect update() cost." At a
     Vicon-scale marker set (~53+24, more than doubling the observation
     count), and with Kalman-gain computation typically scaling worse
     than linearly in observation count, this could become a real,
     separate bottleneck that today's 16-marker data point cannot
     predict. Needs its own profiling once a materially larger module
     (torso+arms, per the productization plan's own sequencing) exists
     — not extrapolated from here.

- **2026-09-13** (productization draft revised after Harri's review,
  latest) — Four corrections/additions from Harri's review of the
  productization plan draft: (1) `label_tracklet_groups_gui.py` isn't
  "already general" in a shippable sense — it needs main-GUI
  integration, a real performance check at scale, and a DB-backed data
  model instead of by-convention JSON files; folded into the
  finalisation-path work item since they're the same underlying gap.
  (2) Multi-person dot disambiguation has a concrete solution path
  after all: Cutie segmentation masks (already built) combined with
  tracklets, the same mechanism the prop-side plan already proposes for
  object-init cueing — downgraded from "deepest, riskiest item" to "a
  real integration task with a known shape." (3) The next module to
  build is torso+arms, not hands — and hands specifically may end up
  markerless entirely, seeded from a good wrist position, worth checking
  before assuming finger markers are needed at all. (4) A real,
  unanswered performance concern: markerless tracking runs ~10fps on
  this capture; the 16-marker leg module (even after tonight's perf fix)
  brought that to ~4-5fps — a full-body/hand marker set (Vicon-scale:
  ~53+24 markers) may need real algorithmic work beyond today's
  per-camera batching, not just more of the same fix. Flagged as an
  open question needing real measurement against a larger module, not
  answered here.

- **2026-09-13** (skeleton-scaling draft revised after Harri's review,
  latest) — Harri's review of the scaling design draft caught a real
  error: a rigid marker-cluster fit gives a segment's own local marker
  geometry, not bone length — that needs relating the segment's frame to
  a neighbour's, which the doc's own §2.3 already required but §2.2 had
  wrongly claimed wasn't necessary. Corrected, and reframed §2.1's real
  benefit as jointly-fit, more robust marker offsets within a cluster
  (which would plausibly have solved `ankle_lat_L` without the
  mirrored-guess workaround). Also added, per Harri's two follow-on
  questions: §2.5, a kinematic-chain calibration approach that can
  recover an *uninstrumented* middle link's length and joint locations
  from two solved neighbouring segments alone (the same problem as robot/
  exoskeleton kinematic calibration from end-effector poses, needing only
  a trusted joint type, not markers on every link); and §5, using
  marker-derived joint ground truth to detect and correct systematic,
  viewpoint-dependent bias in the markerless keypoint detector itself —
  a payoff that would improve every future markerless-only run, not just
  marker-augmented ones.

- **2026-09-13** (two design drafts, overnight) — Per Harri's request
  after the skeleton-scaling discussion and the productization ask,
  wrote two first drafts (no implementation, no review from Harri yet):
  [skeleton-scaling-and-marker-calibration-design.md](skeleton-scaling-and-marker-calibration-design.md)
  (rigid marker-cluster fits per segment, functional joint-center
  estimation, and an anchored small regional bundle adjustment for
  under-instrumented segments — reusing `calibrate_rigid_marker_body.py`'s
  already-solved rigid-geometry-from-capture math rather than a single
  global joint optimization, which risks the same circularity the
  `ankle_lat_L` fix diagnosed) and
  [person-marker-mocap-productization-plan.md](person-marker-mocap-productization-plan.md)
  (generalizing this session's one-off B1-B5/refit/skeleton-build script
  chain into a real multi-module, multi-person-aware pipeline, with a
  calibration-trial concept as the onboarding workflow for a new module
  or subject). Both cross-reference each other and the existing
  `marker-mocap-productization-plan.md` (prop case). Reviewed and
  revised per Harri's own feedback (see the two entries above) before
  committing.

- **2026-09-12** (fixed `ankle_lat_L`'s calibration, and a real perf bug
  along the way, latest) — Harri asked for a single-camera reprojection
  refit of the calibrated attachment set (previous entry's B4 fit needed
  2+ cameras simultaneously triangulating the same dot, which starved
  `ankle_lat_L` to 28 usable samples). Built `refit_attachment_set_
  single_camera.py`: uses a dot-augmented tracking run's own tracked
  parent-joint transform `T(t)` (mostly driven by body/hand keypoints,
  largely independent of the dot markers) plus every inlier single-
  camera dot observation to solve each slot's local offset via nonlinear
  least-squares reprojection -- no triangulation needed. 14/16 slots
  refit cleanly with small, sensible corrections.

  **`ankle_lat_L` did not fix itself, and the reason why is worth
  recording.** The refit converged right back to the same wrong offset
  (along 0.657, lateral +0.134) regardless of starting point -- verified
  by re-running the least-squares from four very different initial
  guesses (the wrong B4 value, a mirrored ankle_lat_R, zero, and a random
  far-off point); all four landed on the identical answer, ruling out a
  local-minimum artifact. Harri correctly diagnosed the real mechanism:
  the assignment gate that selected which observations to refit against
  used `predict_marker_slot()`'s covariance directly with **no added
  measurement noise** (confirmed in `dot_assignment.cpp` -- gating uses
  `pred.covariance` alone), so an observation only survives when either
  the wrong offset's induced pixel bias happens to project small for
  that camera/pose (a geometric coincidence, not confirmation), or the
  predicted covariance is simply wide enough to absorb the bias -- both
  situations select *for* agreement with the wrong offset, not against
  it. The quantitative signature matched: even against the wrong offset,
  median residual was 40px with a 80px p90 -- a wide spread consistent
  with both mechanisms firing together, and large enough (this marker's
  offset was off by ~10cm, an order of magnitude more than the other 15
  slots' ~1-2cm B4 errors) to bias which candidates got assigned in the
  first place. This is why the other 15 slots refit cleanly and this one
  didn't: their B4 errors were small enough to stay within normal
  detection noise, so gating on them barely affects which candidates
  get accepted.

  **Fix**: patch `ankle_lat_L` with a mirrored `ankle_lat_R` estimate
  (along/lateral/anterior mirrored across the sagittal plane -- lateral
  sign flips, along/anterior don't), rebuild the dot-augmented skeleton
  (skeleton_id `5dbdd921f7e8b43f299d08ae8ac75aa23603cc96ecdcfc43dd0f45a5b96b3153`),
  re-track, then refit again from that run's own (now much-less-biased)
  observations. Verified the same four-starting-point robustness check
  on the new run: all four (including the old wrong offset and a random
  guess) converged to the *same* answer this time -- along 0.854,
  lateral +0.041, anterior +0.006 -- which now sits right in the ~0.85-
  0.87 along / 0.03-0.07 lateral cluster the other three independently-
  fit ankle markers agree on, using 13,440 samples (vs. 28 originally).
  Overall body tracking was unaffected either way (100% tracked, mean
  NIS/dof 1.921 vs the unpatched run's 1.920) -- one marker's local
  offset doesn't move the aggregate state-wide metric much, which is why
  this needed the per-slot reprojection check rather than relying on
  NIS alone.

  **Real performance bug found and fixed along the way**: re-tracking
  after the patch was needed to validate the fix, and it revealed that
  the whole dot-augmented pipeline was running at ~1.2 tracked-fps (a
  2.6-hour run for a 97s capture) -- never noticed before because the
  first validation pass (previous entry) only checked correctness, not
  timing. Root cause: `UnscentedKalmanFilter::predict_marker_slot()`
  (previous entry) called `predict_measurements()` once per sigma point
  *per dot marker* (16 markers x 6 cameras = 96 calls/frame), and
  `predict_measurements()` runs a full-skeleton forward-kinematics pass
  every single call regardless of how many observations it's given --
  so the same ~437-sigma-point FK sweep was being redundantly repeated
  16 times per camera per frame. Replaced it with `predict_marker_
  slots()` (plural): batches every dot-track marker on one camera into
  a single `predict_measurements()` call per sigma point, so the FK
  sweep runs once per camera per frame and every marker's own mean/
  covariance is read off its own 2x2 block of the shared result --
  identical math, ~437 FK evaluations/frame instead of ~42,000. Measured
  speedup: ~1.2 -> ~4.1 tracked-fps (a ~3.4x reduction in wall time, not
  the ~16x the raw FK-call-count would suggest -- per-sigma-point camera
  projection across 16 markers isn't free either, and wasn't reduced by
  this change, so FK cost and projection cost are evidently closer in
  magnitude than assumed; not profiled further). `./run_tests.sh`: all
  370 cases still pass, confirming the batched result matches the
  per-marker one exactly.

  Deliberately not investigated further here: why FK-cost vs. projection-
  cost split doesn't give the full theoretical speedup (would need real
  profiling, not guesswork); whether a similar cross-camera batching
  (one `predict_measurements()` call per frame instead of per camera)
  is worth the larger API change to `Tracker::predict_dot_slot_
  predictions()`'s per-camera signature.

- **2026-09-12** (first real end-to-end dot-augmented tracking run)
  — Completed the 3-step plan from the previous entry (build the
  articulated `MarkerPrediction` UKF path, convert the calibrated
  attachment set into a real skeleton, run it end-to-end and compare
  against baseline). All three done and verified with real data; this
  is the first time this project has tracked a person using leg dot
  slots through the full production tracker.

  **Step 1 — skeleton conversion.** New `build_dot_augmented_skeleton.py`
  merges `catalog/leg.calibrated.2026-09-06-kare-tests.yaml`'s 16 fitted
  markers into a copy of the base `reallusion-no-waist` skeleton
  (`track: dots` / `landmark: <name>` / `normal:`, plus an
  `input_tracks: [{id: dots, type: unlabeled_points}]` entry) and
  registers it via `import_skeleton_str()` with `parent_id` set to the
  base skeleton for lineage. Every `parent_joint` resolved against a
  real joint in the base skeleton (checked before writing); field names
  (`track`/`landmark`/`normal`) were verified directly against
  `skeleton_loader.cpp`'s parser and `cpp/tests/data/rigid_prop.yaml`
  before use, not assumed. New skeleton_id:
  `723cf161f2527cd89f94a25eaeaebe5291f33b32f95c54e32bf9c35ebc7c55d8`,
  written to
  `catalog/reallusion-no-waist.dot-augmented.2026-09-06-kare-tests.yaml`.

  **Real gap found while wiring up validation**: the dot-detection run
  (`75cbf678-2066-4a58-ab81-d27ea4c58d02`, ArUco+dots, `capture_object_id`
  NULL) had no path into `pose_observations` at all — the only existing
  finalisation function, `finalise_object_to_db()`, is scoped to rigid
  capture-objects and refuses any run without one. Rather than
  generalising that function to a person/no-object case (real design
  work, not something to invent solo mid-validation), built a narrow
  additive tool, `copy_dot_candidates_to_sequence.py`, that copies the
  same `detection_keypoints` rows (`region_type='dots'`) into an
  *existing* person sequence's `pose_observations` instead of minting a
  new object sequence — `person_id` is already documented as an ignored
  placeholder for `source='dots'` rows, so `sequence_id` is the only
  thing that actually matters to `SessionReader::load_unlabeled_
  candidates()`. Verified the 9-floats-per-candidate wire format
  (Phase B's tracklet-id bump) matches on both the Python encode and
  C++ decode side before trusting it, and that `sync_config_id`/
  `shot_id`/time range all matched the target sequence. Copied all
  69,486 rows into sequence `ec1b3e2f-1ef8-4e31-806c-33102a969ecd`
  (the same sequence the `990fb01a` baseline run used) with zero skips.

  **Step 2 — articulated `MarkerPrediction` (built the same day as the
  previous entry, verified again here in a real run, not just unit
  tests)**: `UnscentedKalmanFilter::predict_marker_slot()` (new,
  `ukf.cpp`/`ukf.hpp`) projects sigma points through the existing
  private `predict_measurements()` and reads off each dot slot's own
  2×2 covariance block, exactly mirroring `dot-assignment-architecture-
  design.md` §6's design and matching `predict_rigid_marker()`'s
  state-uncertainty-only convention (no measurement noise added).
  `Tracker::predict_dot_slot_predictions()` now branches on
  `skeleton_->is_rigid_body()` instead of unconditionally throwing for
  the articulated case. `./run_tests.sh`: all 370 cases pass, including
  a rewritten `test_tracker_predict_update_split.cpp` case that
  exercises the new path on a real articulated fixture with a dot
  marker on a non-root joint.

  **Step 3 — real end-to-end run.** Rebuilt `optbuild` (release) with
  the new UKF code, then ran `posetrak-tracker track` against the
  dot-augmented skeleton, reusing the exact same tracker_config as the
  `990fb01a` baseline (factory defaults, untouched) so the comparison
  isolates just the dot-tracking addition. New tracking_run:
  `d096d14f-3e66-4875-a843-b3ddf9aab263`.

  Overall body/hand tracking, compared to baseline:
  | | baseline (`990fb01a`, no dots) | new (`d096d14f`, +16 dot slots) |
  |---|---|---|
  | tracked steps | 11587/11587 (100.0%) | 11587/11587 (100.0%) |
  | mean NIS/dof | 1.959 | 1.920 |
  | median NIS/dof | 1.814 | 1.778 |

  No regression — essentially identical, if marginally better. One
  thing worth noting honestly rather than glossing over: IK
  initialization printed `IK residual 1.414 m > 0.50 m — using analytic
  root estimate with zero joint angles` (RMS across the 61 openpose
  markers, which is all IK ever fits — dots play no part in
  initialization). Checked whether this hurt anything: NIS/dof for the
  first 5 s (599 steps) averaged 1.830, *not* elevated relative to the
  1.925 average over the rest of the run — the UKF's designed "refine
  over first frames" recovery behaved exactly as intended, and this
  reads as a property of this capture's frame-0 keypoints rather than
  anything introduced by the dot-skeleton work (dots aren't in the IK
  objective at all).

  Dot-slot-specific results (the actual point of the exercise) — every
  one of the 16 slots picked up real, low-outlier-rate observations
  across the whole ~97 s capture via the new sigma-point assignment
  path, using a tracker_config that was never tuned for dot markers
  (still `measurement_noise_std=25`, factory default):

  | slot | frame coverage | median reproj. error | inliers / outliers |
  |---|---|---|---|
  | ankle_lat_L | 53.7% | 40.4 px | 7658 / 0 |
  | ankle_lat_R | 82.0% | 16.0 px | 17423 / 52 |
  | ankle_med_L | 87.3% | 35.0 px | 18770 / 61 |
  | ankle_med_R | 82.1% | 37.4 px | 16230 / 43 |
  | heel_L | 78.9% | 23.5 px | 15718 / 26 |
  | heel_R | 70.0% | 18.0 px | 13101 / 18 |
  | hip_L | 46.8% | 11.8 px | 6852 / 1 |
  | hip_R | 58.9% | 25.7 px | 9903 / 1 |
  | knee_front_L | 64.5% | 29.6 px | 9615 / 15 |
  | knee_front_R | 55.1% | 20.2 px | 9575 / 15 |
  | knee_lat_L | 73.5% | 16.0 px | 12008 / 12 |
  | knee_lat_R | 61.3% | 15.7 px | 10590 / 12 |
  | knee_med_L | 49.0% | 23.8 px | 7576 / 12 |
  | knee_med_R | 49.2% | 29.5 px | 8155 / 15 |
  | toe_L | 82.1% | 19.9 px | 17017 / 47 |
  | toe_R | 82.1% | 27.0 px | 16973 / 129 |

  `ankle_lat_L` — flagged all session as the one low-confidence slot
  (only 28 cross-camera calibration samples, from the reconciliation
  entry above) — still tracks coherently with zero outliers, just at
  lower frame coverage and higher median error than its peers, which is
  consistent with a marker whose fitted offset came from thin data
  rather than with anything actually broken.

  Deliberately not done in this pass (real, open items for a follow-on):
  tracker_config isn't tuned for dot markers at all yet (still using
  the openpose-keypoint factory defaults); no dedicated GUI/finalize
  path for a person-worn (non-rigid-object) dot-detection run —
  `copy_dot_candidates_to_sequence.py` is a validation-scoped tool, not
  a production pipeline entry point; normal-direction fitting still
  defaults to radial-perpendicular rather than being fit against real
  camera visibility; no BVH/visual review of the resulting dot-driven
  joint angles has been done yet.

- **2026-09-12** (reconciliation + re-calibration) — Built
  `reconcile_conflict_review.py`: folds the conflict-review pass's
  resolved assignments back into the main
  `tracklet_group_assignments_full_v2.json`.

  **Real bug found and fixed before trusting the output**: a naive
  first version (remove every resolved member from wherever it was,
  re-add each resolved conflict-review group as a new one) tore apart
  the `gopro-11_mini_01#66` + `gopro13_02#79` pairing fixed by hand the
  same day -- `#66` also correctly appeared (via "keep own slot") in an
  unrelated conflict chain, and blindly rebuilding its grouping from
  that chain's own narrow view split the pair back into two single-
  camera groups, silently re-introducing the exact `ankle_lat_L`
  coverage gap this whole thread started from. Fixed by only touching a
  member when its resolution is a *real* correction (different slot/
  rejection than what the main file already has, checked at the exact
  key) -- 119 of 134 resolved members turned out to be genuine no-ops
  (the conflict review confirmed an already-correct slot) and were left
  completely alone; only 15 real corrections were applied. Range-
  overlap matching (not exact-key) still correctly handles a whole
  main-file tracklet superseded by conflict-review split fragments
  (`oneplus9pro-01#1235`, split into two at frame 10560 during review).
  Verified thoroughly before replacing the main file: all 134 resolved
  members' final status matches expectation, zero cross-group overlaps.
  Original pre-reconciliation file kept as `tracklet_group_assignments_
  full_v2.pre-reconcile.bak.json`.

  **Re-ran B4** on the reconciled data: `ankle_lat_L` now gets real
  cross-camera samples (n=28, residual median 0.7cm -- confirms the
  original coverage-gap diagnosis was right and is now genuinely fixed).
  Honest caveat, not hidden: re-validating against GT found `ankle_lat_L`'s
  own purity actually *dipped slightly* (0.62 -> 0.56, still confused
  with `ankle_med_L`) -- its lateral offset (13.4cm) is implausibly
  large next to its own right-side counterpart (4.1cm) and every other
  ankle slot (3-4cm), most likely because n=28 from a single pair over a
  limited time window isn't enough data to average out whatever local
  tracking bias exists there, unlike the thousands of samples backing
  every other slot. The *coverage* problem (zero data at all) is solved;
  the *confidence* of this one slot's fit is not yet, and needs more
  assigned cross-camera groups for it in a future capture before it's
  trustworthy. Every other slot's purity held steady or improved
  slightly from the 15 real corrections.

- **2026-09-12** (conflict review complete + marker rendering fix,
  latest) — Harri finished the conflict review pass ("checked all") --
  most groups' slots were already correct, some real misclassifications
  fixed via the new accept-hint action and manual review together.

  UI feedback for next time: the green cross drawn over each marked dot
  covered exactly the pixel content (real dot vs. noise/misdetection) a
  reviewer needs to see there. Fixed in `_bgr_to_qpixmap()`: dropped
  `cv2.drawMarker`'s cross, kept only the circle (an outline ring, which
  doesn't cover the center). Confirmed visually on a real frame before
  and after.

  Cleared the 302MB on-disk thumbnail cache (`scratch/dot_ground_truth/
  thumb_cache/`) -- it holds pre-rendered PNGs from before this fix, so
  anything already viewed would otherwise keep showing the old
  cross-obscured version until re-decoded anyway. Revisiting previously-
  seen frames will re-decode from video once more (the ~190x cache
  speedup resets), acceptable now that both review passes are done and
  future use should be spot-checks, not another full sweep.

- **2026-09-12** (accept-hint action) — "The 'was: xxx' comment
  is quite often correct -- can I just select the correct ones and keep
  that slot for those?" Added "Keep checked members' own 'was: ...'
  slot" to `label_tracklet_groups_gui.py`: unlike "Assign checked as new
  group" (one Slot combo value applied to every checked member), this
  finalizes each checked member *individually* to whatever its own hint
  says -- necessary because a conflict group's checked members
  routinely have *different* correct previous slots (confirmed: of a
  17-member chain's members, 3 shared `ankle_lat_L` while others had
  their own distinct hints). Members sharing the same hinted slot are
  still consolidated into one new group together, not one singleton
  group each.

  Refactored `_finalize_checked` to share a new `_finalize_members`
  helper (takes an explicit member list instead of always reading the
  checkboxes) so the batch action can make several such mutations before
  a single `_save()`/`_populate_list()` refresh, not one per member.

  Verified against the real conflict-review data: accepting 2 members
  with different hints assigns each to its own correct slot; accepting
  3 members sharing one hint consolidates them into a single new group
  (not three); checking a member with no hint warns and mutates
  nothing (verified by patching `QMessageBox.warning` -- the real
  modal dialog blocks headless test automation the same way it would
  wait for a real click, not a bug, just untestable un-patched).
  Full crash sweep re-run clean on the original 24-group dataset too
  (no regression from the `_finalize_checked` refactor).

- **2026-09-12** (conflict review workflow) — "How can we review
  it?" -- rather than build a new review UI, converted the audit's
  output into `build_tracklet_groups.py`'s own group-file shape
  (`prepare_conflict_review.py`, new): every conflict chain and
  high-confidence mismatch becomes one synthetic "group" (already
  flagged contradiction+ambiguous), reviewable with the *exact* existing
  `label_tracklet_groups_gui.py` -- same thumbnails, suggestions,
  Assign/Split/Discard/scrub-split, zero new UI code. Added one small,
  additive enhancement to that GUI: an optional `member_hints` block in
  the groups file (`"label#tid": "was: <slot>"`) shown alongside each
  member's checkbox, so a conflict between two previously-different
  slot assignments is visible at a glance without cross-referencing
  `cross_slot_conflicts.json` separately. Low-confidence pairs excluded
  by default (`--include-low-confidence` to add them).

  Verified against the real data: 30 conflict groups generated (21
  chains + 9 high-confidence mismatches), all render cleanly in a
  headless smoke test, hints display correctly (e.g. `gopro-11_mini_01
  #1423` shows "was: toe_R").

  Review with:
  ```
  python python/tools/label_tracklet_groups_gui.py \
      --groups scratch/dot_ground_truth/tracklet_groups_conflicts.json \
      --output scratch/dot_ground_truth/tracklet_group_assignments_conflicts.json
  ```
  Not yet built: folding the corrected assignments back into the main
  `tracklet_group_assignments_full_v2.json` once this review pass is
  done (a real, needed follow-up step, deliberately not built ahead of
  having real corrected data to reconcile against).

- **2026-09-12** (cross-slot consistency audit) — Harri checked
  the raw video and confirmed `ankle_lat_L` definitely has good
  multi-camera coverage (40-45s, visible in both `gopro-11_mini_01` and
  `gopro13_02`) -- contradicting B4's "single-camera-only" finding for
  that slot. Root-caused, not a coverage gap: `gopro-11_mini_01#66`
  (assigned `ankle_lat_L`) and `gopro13_02#79` (assigned `toe_L`)
  triangulate together at **median 0.14px** (about as perfect as this
  data gets) -- the same physical marker, split into two different B3
  groups from the same original contaminated B2 component
  (`group_108`, 12 members, `[CONTRADICTION][ambiguous]`) and assigned
  *different* slots because the splitting process (necessarily, per
  finding #3) loses the "these were originally linked" information once
  pieces are pulled apart. Fixed directly: merged the `toe_L` entry into
  the `ankle_lat_L` one in `tracklet_group_assignments_full_v2.json`.

  **Generalized into a systematic audit**, since this is exactly finding
  #3's proposed mechanism in action: for every pair of *differently*
  -slotted tracklets that came from the same original B2 component,
  re-ran the real `pairwise_stats()` reprojection check. Result: **124 of
  744 checked pairs (17%)** show strong agreement (<3px median, <6px
  p90) despite disagreeing final slots. Not all real errors -- many
  (~89) are low-sample or anatomically implausible pairs (e.g.
  `hip_L`/`toe_R`), most likely 2-camera epipolar-line coincidences
  over a short window, not real mixups. Filtered to **35 high-confidence
  cases** (n>=30, median<2px, anatomically adjacent same-side pair --
  lat/med, knee variants, heel/ankle/toe -- or a same-slot left/right
  swap) as the actionable subset.

  Found at least one genuine 3-way conflict needing visual judgment, not
  just automated merging: in `group_1`, `oneplus9pro-01#15312` (n=1771,
  very strong evidence either way) triangulates consistently with *both*
  `gopro13_02#2803` (labelled `ankle_med_L`) and `gopro-11_mini_01#2693`
  (labelled `ankle_lat_R`) -- at least two of the three labels on this
  one real physical marker are wrong, but geometry alone can't say which
  is *right* (lat vs. med, or which leg) without looking at the image.
  Deliberately did not auto-apply the other 34 high-confidence fixes for
  this reason -- flagged to Harri to work through rather than guessed.

  **Formalized into `audit_cross_slot_consistency.py`**, per Harri's
  request. Real improvement over the ad hoc version: connected-
  components clustering over every accepted cross-camera edge within
  an original B2 component (not just isolated pairwise checks), and
  properly includes split-range (4-element) members this time, not
  only whole tracklets. Re-running the clustered version on the full
  dataset changed the picture substantially: most of what looked like
  simple 2-way mismatches turned out to be fragments of larger tangles.
  Final, accurate result: **9 simple (2-tracklet, 2-slot) high-
  confidence mismatches** (cleanly actionable without a visual check)
  and **21 conflict chains** (3-17 tracklets each, 2-6 distinct slots)
  that genuinely need a human to look at the actual images -- geometry
  proves several labels in each chain are wrong but not which one(s)
  are right. Two chains stand out in size: `group_43` (17 tracklets, 6
  slots) and `group_6` (13 tracklets, 3 slots -- mostly a toe_L/toe_R
  tangle). Full results (including every chain's own members and
  pairwise stats, not just what's printed) written to
  `scratch/dot_ground_truth/cross_slot_conflicts.json` for reference
  while working through them.

  Not yet done: applying the 9 high-confidence fixes or working through
  the 21 chains (needs Harri's visual judgment, not automation) --
  B4 should be re-run once real corrections land, given `toe_L`/`toe_R`
  (already the two worst-residual slots) are implicated in several of
  the largest chains.

- **2026-09-12** (B5 output) — Completed B4's output to match the
  redesign doc §1.9 file shape exactly (`module`, `skeleton_topology`,
  `requires_joints` headers, alongside the `calibration:` block and
  per-marker fields already produced by B4): `skeleton_topology` pulled
  from the skeleton YAML's own top-level `name` (`reallusion-no-waist` --
  distinct from the DB row's per-person instance name, "Default female");
  `requires_joints` derived from the parent joints actually used
  (`foot, shin, thigh`), not assumed; added `date` and `unfit_slots`
  (explicit, e.g. `ankle_lat_L`) to the `calibration:` block so a
  consumer sees the coverage gap without re-deriving it.

  Written to the real project tree for the first time, not scratch:
  `catalog/leg.calibrated.2026-09-06-kare-tests.yaml` -- the project's
  first calibrated (or nominal) marker catalog file. Re-verified the
  finished file loads and validates identically to B4's own numbers
  (same purities, same residuals) -- the header additions don't disturb
  anything `_load_attachment_set()`/`eval_fk_prediction.py` reads.

  P-B (B1-B5) is now a complete, working pipeline end to end: per-camera
  motion-gated tracklet linking -> cross-camera grouping -> manual
  slot assignment -> calibration fit -> a real, validated marker
  attachment set ready for FK-based prediction. Not yet wired into the
  production tracker config (a separate integration step) or into
  `posetrak-tracker`'s own marker-observation pipeline.

- **2026-09-12** (B4 calibration fit) — Built
  `fit_calibrated_attachment_set.py` per the redesign doc §4's own B4
  design: for each assigned group, per-frame `offset_local = T(t)^T @
  (p(t) - T(t)_translation)` against the parent joint's FK transform,
  median over every frame of every group assigned to that slot (not just
  one group -- robust to a last-frame outlier or a single bad group).
  Parent-joint mapping matches `_RIGHT_TRIAL`'s established convention
  (hip->thigh, knee_*/ankle_*->shin, heel/toe->foot); confirmed both
  sides' local +Y is down-bone for this skeleton, so local X/Z read
  directly as lateral/anterior with no extra axis-building. Fits both
  sides independently from real data (not mirror-generated) -- a
  calibrated set should reflect whatever real (a)symmetry the session
  shows, not force symmetry.

  Run against the completed 2026-09-12 B3 review (395 entries, 294
  assigned): 109 usable groups (many assigned groups turned out
  single-camera-only -- no simultaneous 2-different-camera view to
  triangulate from, a real finding in its own right, e.g. 28/34 `toe_L`
  groups and 20/29 `toe_R` groups contributed nothing, consistent with
  finding #1 below) across 15 of 16 slots. `ankle_lat_L` got zero
  samples (all 9 of its assigned groups were single-camera-only) --
  surfaced as an explicit warning in the tool's own output, not silently
  dropped; a real data gap needing better multi-camera coverage of that
  slot in a future capture, not a bug.

  Fitted values pass a clean anatomical sanity check on their own: lat/
  med offsets are opposite-signed on the same leg and mirrored between
  legs (fully self-consistent); along-bone fractions land exactly where
  expected (hip ~0.02-0.04 near the hip joint, knee_* ~0.00-0.02 near
  the knee, ankle_* ~0.81-0.84 near the distal shin, toe ~0.80-0.81 near
  the foot's far end, heel *negative* ~(-0.32)-(-0.41), i.e. behind the
  ankle -- opposite direction from the toe along the same bone, correct).
  Residuals (predicted vs. actual, using the fitted offset) median
  1.2-4.4cm, p90 2.3-9.1cm -- plausible given tracking + triangulation
  noise, per the design doc's own expectation.

  **Validated exactly per the design doc's own plan**: added
  `--attachment-set` to `eval_fk_prediction.py` (was hard-coded to P-A's
  exploratory `_DEFAULT_TRIAL` probes) and re-ran the same GT metric.
  Every single slot's purity improved, several by 30-50 points (e.g.
  ankle_med_R 0.35->0.88, knee_front_L 0.32->0.72, knee_front_R
  0.41->0.85, toe_L 0.50->0.85, toe_R 0.49->0.96, knee_lat_R
  0.42->0.83, knee_med_L 0.71->1.00) -- and every probe now maps to
  *itself* by name, not a shared generic anchor, directly confirming
  the 2026-09-11/12 lat/med-confusion diagnosis was a catalog-
  completeness problem (no distinct offset per real slot), not a
  missing-normals problem. `ankle_lat_L` stayed weak (0.62, confused
  with `ankle_med_L`) exactly as predicted by its own zero-sample
  warning above -- consistent, not a surprise.

  Not done in this pass (matches the design doc's own explicit scope
  cut): normal direction still defaults to the offset's own radial
  perpendicular, not fitted against observed camera visibility --
  revisit only if the visibility prior needs more accuracy than that
  gives. `ankle_lat_L`'s data gap needs either better multi-camera
  coverage in a future capture or a single-camera-only fitting
  fallback (not built).

- **2026-09-12** (B3 review complete) — **All 156 B2 groups
  reviewed and assigned** (`tracklet_group_assignments_full_v2.json`:
  395 final entries after splits, 294 assigned to a real slot, 101
  rejected -- 91 noise, 10 other-subject/prop, using the reject-reason
  distinction added 2026-09-11). Slot distribution ranges 9-34 groups
  per slot; a real, complete labeled dataset now exists for B4.

  Three concluding findings from the full pass:

  1. **"Most common misdetection: non-existing dots in the foot near the
     toes."** Confirmed quantitatively, not just impression: `toe_L`
     (34) and `toe_R` (29) together account for 63 of the 294 assigned
     groups -- 2-3x a typical slot's count (9-27 elsewhere). Consistent
     with phantom/false dot detections near the feet (likely ground
     reflection or shoe highlights passing the brightness+shape gate)
     fragmenting into many separate tracklet groups needing individual
     review. A detection-tuning target (per-camera or per-region
     threshold near the foot/ground), not investigated further tonight --
     a separate track of work from the linker/grouping issues above.

  2. **"Normal-based culling for initial candidates would have saved a
     lot of time."** Confirms and extends 2026-09-11's lat/med finding --
     not just ankle lat/med (literally position-indistinguishable today)
     but suggestions generally would benefit from normal-direction
     filtering, not just position-distance ranking. Reinforces that B4
     (a real per-slot marker catalog with distinct offsets *and*
     normals, replacing the exploratory `_DEFAULT_TRIAL` probes) is the
     right next investment, not a smaller patch.

  3. **"Very large groups were confusing -- hard to tell what
     commonality caused tracklets to end up together. Need tighter
     grouping criteria: covered time ranges should overlap *enough*, and
     during that overlap the dots should triangulate consistently near
     the slot's own predicted position."** A concrete design proposal
     for the 2026-09-11 "B2/linker tension" problem (more numerous
     fragments from the fixed linker give B2's pairwise reprojection
     test more chances to chain a wrong link into a large blob). Two
     specific, actionable criteria: (a) a *relative* overlap requirement
     (overlap duration relative to each tracklet's own span, not just an
     absolute `min-shared-frames` count -- a tiny sliver of overlap
     between two long, otherwise-unrelated tracklets is currently
     sufficient "evidence"), and (b) fold the FK-slot-consistency check
     directly into the *grouping* decision itself (today it only ranks
     suggestions after grouping, so a pairwise link can be accepted even
     if the resulting 3D trajectory doesn't correspond to any real
     slot's predicted position at all). Not built yet -- a real,
     scoped follow-up to design properly, not a quick patch.

  **Open decision, not yet made**: whether to re-run full-capture
  detection with 2026-09-12's `accel_std_px=2` fix now that the review
  is complete -- doing so would invalidate every completed group_id
  (fresh tracklet ids again, same as the 2026-09-11 re-run did), forcing
  a full B3 re-review on top of the one just finished. Leaning toward
  *not* redoing it now (the fix matters most for future captures, which
  already get it via the changed default) and instead proceeding to B4
  with today's completed data -- but this is Harri's call, not decided
  unilaterally.

- **2026-09-12** (accel_std_px fix) — Harri's evening feedback on
  the full-capture B3 review (paused for the night, resuming tomorrow):
  "much better than earlier"; but "quite many tracklets that are
  otherwise OK but last frame is clearly outlier"; and a design question
  -- "should normals be in use? There are lot of cases where e.g.
  lat/med is wrong."

  **Last-frame-outlier pattern, diagnosed and fixed.** Confirmed real and
  systematic (not a fluke) by scanning every reviewed member's own
  frame-to-frame step sizes: e.g. `oneplus9pro-01#3645`, 153 frames of
  near-zero motion (median step 0.10px), then a 137px jump on its very
  last recorded frame. Mechanism: even after the 2026-09-11 coasting-bug
  fix (so `dt` is correctly 1 per real elapsed frame, no compounding),
  the Kalman process noise Q still grows with `dt**4` *per accumulated
  missed frame* -- three consecutive misses (our `max_missed=3`) alone
  grow the position std to ~30px (95% radius ~60px) even from a
  converged, confident steady state. Since that's the gate's widest
  point in a track's whole life, and it occurs on exactly the frame
  before `max_missed` would kill the track anyway, a track that's about
  to legitimately die is also the one most likely to snap onto a stray
  candidate right then -- becoming its erroneous last point before
  actually dying.

  First tried capping the Kalman noise-growth's own `dt` directly
  (`max_noise_dt`, new optional parameter) -- added, tested, found to be
  a no-op on real data: `dt` passed to `_predict()` is already always 1
  per call for every actual detection run so far (frame_step=1), so
  capping it at >=1 changes nothing. Kept in `dot_tracklet.py` as a
  real, general (if currently inert) parameter -- would matter for a
  frame_step > 1 run -- clearly documented as such, not removed.

  The actual lever: `accel_std_px` itself (10 was simply too generous
  once compounded). Swept 1/2/3/5/7/10 against every mid-tracklet split
  Harri had made in today's full-capture review (17 real cases by
  evening): monotonic, 10 -> 0/17, 7 -> 9/17, 5 -> 12/17, 3 -> 14/17,
  2 -> 17/17, 1 -> 17/17 (2 has better purity than 1: 396/429 vs
  390/429 clean tracklets >= 0.90 purity). Also re-checked the
  2026-09-11 B2 cross-camera precision issue at the same settings:
  0.39 -> 0.44 on the 42-48s GT window -- a real but partial
  improvement, the larger B2/linker tension from that entry is not
  resolved by this alone. **Set `accel_std_px=2.0` as the new default**
  in `dot_tracklet.py` (was 10.0) and in both validation tools'
  own CLI defaults. All 66 detection/marker-pipeline tests still pass.

  **Not yet run**: a fresh full-capture detection with this new default
  -- needs Harri's go-ahead (each of today's two re-detection runs was
  done only after his explicit confirmation; holding to that same
  pattern overnight). Once confirmed: new detection run (append-only,
  `75cbf678` untouched), fresh full-capture B2, and Harri's review
  continues on that data instead.

  **Normals question, answered, no code change**: confirmed via
  `_SLOT_TO_PROBE` directly that `ankle_lat_{L,R}` and `ankle_med_{L,R}`
  map to the *exact same* FK probe -- literally position-indistinguishable
  in the current suggestion mechanism (knee lat/med already use
  different, if crude, probes -- a separate, subtler mis-mapping issue,
  not a missing-feature one). Normals (marker-catalog redesign doc
  §1.6) are the right long-term answer, but not the immediate fix:
  `_DEFAULT_TRIAL` is still P-A's exploratory probe catalog, never a
  real per-slot marker catalog, and simply has no distinct offset for
  ankle lat vs. med to begin with -- there's nothing for a normal
  comparison to disambiguate yet. Right sequence: give every one of the
  16 real slots its own distinct offset first (B4's job per the redesign
  doc, not built yet -- likely fixes most lat/med confusion on its own,
  since B3 compares the fused *3D* position, and real markers are
  physically separated by a few cm even where single-camera 2D
  projections look close), *then* add normal-direction scoring for
  whatever's still ambiguous after that. Flagged as a real, bounded
  follow-up task (even a quick per-slot offset fit from the existing GT
  labels would likely help) -- not started, deferred to Harri's own
  scoping call rather than built unilaterally overnight.

- **2026-09-11** (B2/linker tension found) — Re-ran full-capture
  detection with the fixed linker + `max_missed=3`
  (`detection_run_id 75cbf678`, 47m40s -- faster than the buggy run's
  59m12s, plausibly because shorter-lived tracks shrink the Hungarian
  cost matrix on noisy cameras). Ran full-capture B2 on it
  (`tracklet_groups_full_v2.json`): **156 groups, 44 flagged
  contradiction (28.2%), 44 ambiguous (28.2%)** -- measurably *worse*
  than the buggy run's 143/27(18.9%)/42(29.4%), and with new,
  much-larger contaminated groups (top sizes 42/33/28/21 vs. the buggy
  run's 28/17/12/11). Investigated the 42-member outlier directly:
  confirmed via frame-range overlap (two same-camera tracklets active
  *simultaneously*) that it's a real B2 transitivity-chaining mistake --
  several genuinely different real markers wrongly linked into one blob
  -- not simply "one marker fragmented a lot," so this is a real, not
  cosmetic, regression.

  Confirmed with real GT on the 42-48s window: recall improved (0.47 ->
  0.65) but **precision collapsed 0.61 -> 0.39** (23 false groupings vs.
  9). Re-tuning B2's own thresholds against this population
  (`min-shared-frames` up to 20, tighter `median`/`p90-thresh-px` down to
  1.5/3px) recovered at best 0.56 -- still below the old baseline, and
  only by giving up most of the recall gain.

  Built `sweep_linker_and_b2.py` (re-links raw candidates in-memory for
  an arbitrary `MotionGatedLinker` config and feeds the result directly
  into B2's own prefilter/build_groups/validate, bypassing the DB --
  turns an N x M linker-param x B2-param sweep into N re-link passes
  instead of N x M full detection runs) to sweep `max_missed` at the
  *group* level without new detection runs each time. Cross-checked
  against the real CLI-measured number for `max_missed=3` first
  (precision matched exactly: 0.39, 15/38, 23 false) to trust the
  harness, then swept `max_missed` in {2,3,4,5,6,8,10,15}: **precision
  never recovers past 0.50 at any value** (best case), nowhere near the
  buggy linker's 0.61 -- so this is not a `max_missed` tuning question at
  all. Even `max_missed=6` (the *same* setting the buggy run used, only
  with the coasting bug now fixed) gives precision 0.41, still well
  below the buggy run's 0.61.

  **Working explanation, not yet further verified**: the buggy linker's
  runaway gate greedily vacuumed many real detections into *fewer,
  longer* (internally contaminated) tracklet IDs. B2's cross-camera
  pairwise test only gets a chance to err once per *pair of tracklet
  fragments* it examines -- fewer, bigger fragments mechanically means
  fewer pairs, hence fewer chances for B2 itself to make a wrong link,
  even though each fragment is individually worse. The fix produces
  more, shorter, individually *purer* fragments (validated, wanted), but
  that gives B2's own pairwise reprojection test more opportunities to
  slip up, and no linker parameter or B2 threshold tested closes that
  gap. This reframes the problem: it's a real tension between
  per-tracklet purity (now fixed) and B2's own grouping algorithm's
  robustness to a more-fragmented population, not a tuning question --
  likely needs a change to B2's own matching logic (e.g. requiring
  consistency across more than one pairwise test before accepting a
  link, or being more conservative about chaining through a low-evidence
  edge), a larger task than today's fixes.

  **Decision (Harri)**: review anyway with the current tools rather than
  build B2 improvements first -- the flagged-contradiction/ambiguous
  groups (even the 42-member outlier) ARE still correctly flagged, and
  this session's split/merge/scrub/assign-as-new-group tooling should
  handle them; the "big jump within one tracklet" failure that stopped
  the previous review pass should now be gone, even if it's replaced by
  "more groups need splitting." B2's own grouping-robustness improvement
  stays a real, identified follow-up, not started. Resuming B3 review on
  `tracklet_groups_full_v2.json` (a fresh output file, e.g.
  `tracklet_group_assignments_full_v2.json` -- the previous partial
  review in `tracklet_group_assignments_full.json` was built against the
  old, buggy-linker tracklet population and doesn't correspond to this
  run's tracklet ids, so it can't be reused or merged in).

- **2026-09-11** (max_missed tuning) — Harri's diagnosis of the
  remaining failure pattern: most wrong-identity jumps happen when a
  marker is briefly occluded by another body part while a *different*
  marker's trajectory crosses nearby, so coasting through the whole
  occlusion risks the crossing marker looking like the better match once
  real detections resume -- proposed cutting tracklets more aggressively
  on occlusion instead of trying to coast through it.

  Swept `max_missed` (1/2/3/4/6) against the same real full-capture
  human-split regression + purity sets. Result: **3 is the actual
  measured optimum**, not just "lower is more aggressive" -- it wins on
  *both* metrics simultaneously (12/12 real cuts correctly separated, up
  from 11/12 at the old default of 6 -- notably, this also fixes the one
  previously-unresolved case, `pixel9#201 @ 14425`, the accelerating-
  marker-through-a-long-gap case: cutting sooner means the track dies
  before the gap gets long enough for the coast to run out of road, a
  different mechanism than the earlier compounding-dt bug fix but the
  same practical effect; best purity in the sweep too, 25/40 >= 0.90
  vs. 22/40 at the old default). Set as the new default
  (`max_missed=3`, both in production `dot_tracklet.py` and the
  prototype's own CLI default). Full detection/marker-pipeline test
  suite (66 tests) still passes.

  Re-running full-capture detection with the fixed linker + this tuning
  now; see the next entry for the result.

- **2026-09-11** (linker bug fix) — Harri stopped the full-capture
  B3 review: still too many splits needed, often from "really big
  jumps" -- and correctly guessed this needed fixing before continuing,
  offering the splits already made as test material. Asked how the
  algorithm works, which led straight to the real cause.

  **Found a genuine implementation bug, not a tuning problem** (Harri's
  own guess was tuning -- it wasn't): in `MotionGatedLinker.link_frame()`,
  `tr.last_frame` was only ever updated on a real match, never during a
  coast (missed-frame) step. So the *next* predict() computed `dt` as
  time-since-last-*match* and reapplied it on top of state that had
  already been advanced by the previous coast -- the position/covariance
  compounds quadratically across consecutive missed frames instead of
  linearly. Quantified numerically: a marker at a steady 10px/frame,
  occluded 6 frames, should predict to ~160px with a ~85px gate std-dev;
  the buggy version predicted **310px with a ~957px std-dev** -- a gate
  wide enough after just a handful of missed frames to accept nearly any
  nearby candidate, including a distant, unrelated real marker. Exactly
  the "big jump" symptom. Fixed with a one-line change (advance
  `last_frame` on every coast step too); added
  `test_coasting_does_not_compound_across_multiple_missed_frames`, first
  confirmed it actually fails against the pre-fix logic (assigns the far
  distractor the real tracklet's id, the true continuation a new one),
  then confirmed it passes with the fix. All 10 `test_dot_tracklet.py`
  tests and all 66 detection/marker-pipeline tests pass.

  **Re-validated using Harri's own full-capture splits as the test
  material**, per his suggestion: extracted 12 real human-determined cut
  points from the partially-completed `tracklet_group_assignments_full.
  json` (up from the original 11-cut small-window set) and re-ran them
  through the *fixed* linker directly against the existing `98d59082`
  run's raw candidates (no new detection run needed -- the bug is
  linking-only, raw per-frame positions are unaffected by it). Result:
  **11/12 correctly separated** -- a dramatic improvement consistent with
  what was driving Harri to stop reviewing. The one miss
  (`pixel9#201 @ 14425`) is a different, genuinely harder case: a real
  7-frame occlusion gap right after the marker was visibly *accelerating*
  (velocity climbing 1.1->2.7px/frame in the preceding 8 frames) -- a
  constant-velocity coast can't extrapolate real acceleration through a
  blind spot that long, and the re-observed position is consistent with
  continued acceleration, not a different marker (not visually confirmed
  either way). A constant-acceleration model or particle filter (Harri's
  own original suggestion) would likely help this specific residual case;
  not fixed now, flagged as a possible follow-up, not blocking.

  **Next**: re-run full-capture detection with the fixed linker (a fresh
  `detection_runs` row -- append-only, `98d59082` untouched), then
  full-capture B2, then resume the B3 review Harri paused.

- **2026-09-11** (B3) — Four more real fixes from Harri's
  full-capture (143-group) review:

  1. **"Assign a single tracklet without split-then-reselect-then-assign
     as three steps"**: added "Assign checked as new group" -- cuts the
     checked members into their own group (or, if every member in the
     current group is checked, just finalizes the current group directly
     -- no pointless split) and immediately applies the Slot combo's
     current selection, in one click.

  2. **"The view jumps to another group after discard/split, re-sorting
     makes it hard to keep working on one group"**: `_populate_list()`
     now takes a `keep_group_id` and re-selects that exact group after
     rebuilding the (re-sorted) list, instead of leaving selection at
     whatever numeric row it lands on. Applied to every action that
     changes the current group's own state (Assign, Reject, Discard,
     Split, and the new Assign-as-new-group) -- the view now always
     stays on the group being worked on, showing it with the acted-on
     members already gone, until the operator explicitly picks another
     row. Merge is the one exception by design: it advances to the
     *target* group, since that's the group the operator just asked to
     see the merged result of.

  3. **"Filter to show only unassigned groups"**: added a "Show only
     unassigned" checkbox above the list. Interacts correctly with fix
     #2's `keep_group_id`: assigning/rejecting the current group while
     the filter is on makes it drop out of view (it's no longer
     unassigned) and the view auto-advances to another unassigned one --
     confirmed via a scripted test, not just visually.

  4. **"Noise vs. a valid marker on another subject/prop -- worth
     distinguishing?"**: yes -- added `reject: noise` and `reject: other
     subject / prop` as two pseudo-entries in the Slot combo, recorded as
     `reject_reason` in the output JSON (`tracklet_group_assignments.json`'s
     schema extended accordingly). Also fixed a real, pre-existing gap
     this surfaced: "Discard checked (noise)" previously just deleted the
     members with *no record at all*; it now goes through the same
     finalize-with-reason mechanism as everything else, so every
     rejection -- noise or other-subject -- leaves a real audit trail
     instead of silently vanishing. The plain "Reject" button (whole
     group, no reason) stays available for the genuinely-unclear case,
     relabelled "Reject (unspecified)" to make that distinction visible
     in the UI itself.

  All four verified via scripted headless tests against the real
  42-48s/5-camera review data (not just read -- actually exercised: check
  a member, click the new button, assert the resulting `self.groups` /
  `self.assignments` state and that the view didn't jump) plus a full
  24-group open-every-row sweep against the real saved assignments file,
  confirming no regressions.

- **2026-09-11** (linker port) — Ported the validated
  `MotionGatedLinker` prototype into production, per Harri's explicit
  go-ahead. `posetrak/detection/dot_tracklet.py`'s `DotTrackletLinker`
  replaced outright by `MotionGatedLinker` (Kalman-filter motion gating +
  per-frame disambiguation margin, `accel_std_px=10`,
  `disambiguation_margin=2.0` as the validated defaults); `link_frame()`
  now takes an explicit `video_frame` (needed for a real `dt` in the
  Kalman predict step -- `marker_pipeline.py`'s call site updated to pass
  it, confirmed every call site already has the real frame index
  available). `prototype_motion_gated_linker.py` now imports the real
  production class instead of keeping a parallel copy, so its
  regression/purity checks double as a standing validation harness for
  any future gate-tuning, not just this one-off port.

  One real bug found by writing the new unit tests before trusting the
  port: the per-frame disambiguation margin was rejecting a *brand-new*
  track's second observation whenever another track happened to be born
  nearby, since a birth-phase row's raw pixel-distance costs are on the
  same small scale as genuine ambiguity and tripped the margin check
  even with zero real motion evidence yet. Fixed by only applying the
  disambiguation check to motion-established tracks (`n_obs >= 2`,
  Mahalanobis-gated) -- caught by
  `test_two_close_simultaneous_tracks_do_not_swap_identity` before it
  ever reached real data. Re-ran the prototype's own validation against
  the real 42-48s ground truth with the actual ported class: 11/11 real
  cuts still correctly separated, confirming the port behaves as
  designed, not just the standalone copy.

  Rewrote `tests/detection/test_dot_tracklet.py` for the new class (9
  tests, including two new ones for the two failure modes this linker
  specifically targets: coasting past a stale distractor during
  occlusion, and not swapping identity between two close simultaneous
  tracks). Full `pytest python/tests/` run (2156 tests): all detection/
  marker-pipeline tests affected by this change pass (65/65); two
  pre-existing, unrelated failures found elsewhere (a Windows-path-
  absoluteness test and an unrelated observation-edits test, neither
  touching detection code) and two Qt GUI test files
  (`test_page_sync_led.py`, `test_pair_scrubber.py`) crash the interpreter
  in this environment regardless of this change -- confirmed pre-existing
  by reproducing the same crash point before touching any detection code;
  out of scope for this change, not investigated further here.

  **Launched a fresh full-capture detection run** (append-only, new
  `detection_runs` row, `6abcba67` untouched) reproducing `6abcba67`'s
  exact recorded `config_json` (dictionary DICT_4X4_50, marker ids 0 1 2
  3 16 17 34 37, all 6 cameras dot-enabled, same per-camera threshold/
  max-saturation/blacklist-frac overrides, same 33.62-130.185s range,
  `run_parallel()`) with only the linker swapped -- the only variable
  changed is `MotionGatedLinker` vs. the old fixed-radius linker, so any
  difference in the new run's tracklet quality is attributable to that
  change alone.

  **Result** (`detection_run_id 98d59082`): complete, 69,486 frames --
  same total as `6abcba67`. **Real, measured perf regression**: 59m12s
  vs. the old run's 35m19s (~1.7x). Diagnosed live while it ran (Harri
  flagged unusually low CPU/disk I/O partway through): 5/6 cameras
  finished promptly, but `gopro13_01` -- the one camera with
  `blacklist_frac=0.95` (its glare veto almost fully disabled, so a much
  higher per-frame candidate count than any other camera) -- ran solo for
  a long tail after the rest finished, at a measured ~10 fps. Consistent
  with `MotionGatedLinker`'s Hungarian-assignment cost matrix (built via
  a nested Python loop over every open track x every candidate, then a
  scipy `linear_sum_assignment` call) costing meaningfully more per frame
  than the old greedy nearest-neighbor once candidate counts are large --
  not investigated further now (Harri's explicit call: detector/linker
  performance work is real and high priority, but after the tracking
  pipeline itself is further along, not blocking it now). Worth revisiting
  before this linker is asked to run on a noisier or longer capture --
  likely fixes: cap simultaneously-open tracks, vectorize the cost-matrix
  build, or a cheap pre-filter before the full Hungarian solve.

  **Full-capture B2 run** (`tracklet_groups_full.json`, 4 cameras
  excluding gopro13_01 as before, 33.62-130.185s): 143 groups, 27
  flagged contradiction (18.9%), 42 ambiguous (29.4%) -- plus ~27% more
  raw/pre-filtered tracklets per camera than an identical run against the
  *old* linker's candidates (`tracklet_groups_full_oldlinker.json`, same
  detector output, `6abcba67`, only the linker differs: 141 groups, 33
  contradiction (23.4%), 37 ambiguous (26.2%)). The B2-internal
  contradiction/ambiguous rate alone reads as roughly a wash (down on
  contradiction, up on ambiguous, "either" rate ~unchanged) -- more
  tracklet fragments from the new linker apparently gives B2's
  cross-camera pairing more chances to find an ambiguous partial match,
  partially offsetting the per-tracklet cleanliness gain.

  **Re-validated against real GT** (Harri's call, given that wash) on
  the same 42-48s window that has labelled ground truth, sweeping
  `min-lifetime`, `min-shared-frames`, `disambiguation-margin`, and the
  reprojection thresholds against the new linker's tracklets: zero
  sensitivity to the first three (the bottleneck isn't the prefilter or
  the per-camera-pair disambiguation stage), and loosening the
  reprojection threshold reproduces the exact same "knee of the curve"
  collapse found tuning the old linker (loosening median 3->5px:
  recall 0.47->0.53 but precision 0.61->0.34). **Conclusion: the existing
  defaults (3px/6px median/p90, min-lifetime=15, min-shared-frames=5,
  disambiguation-margin=1.5) are already the right operating point for
  the new linker too -- no retuning needed.** And by the metric that
  actually matters (recall/precision against real GT, not the cruder
  internal flags), the new linker vs. the old at the same window and
  thresholds: precision 0.61 vs. 0.56 (+0.05), recall 0.47 vs. 0.52
  (-0.05) -- a real, modest trade exactly in the intended direction
  (favor low false-link rate).

  Next: full B3 review of the 143 groups in `tracklet_groups_full.json`
  (already generated with the confirmed-best default thresholds -- no
  need to regenerate).

- **2026-09-11** (linker prototype) — All 24 B2 groups in the
  42-48s/5-camera test window reviewed and assigned (25 final entries,
  12/16 real slots covered in this window). Generic observation from the
  review: ~14/25 final members needed a manual mid-tracklet split, and
  Harri asked for a next step given how common this was, with a specific
  proposal (motion-consistency penalty -- Kalman/particle filter) and
  offered the splits made during review as a ready-made test set. Agreed
  path (Harri's choice over running full-capture B2 immediately):
  prototype and validate a fix before touching production or spending a
  full-capture detection run on it.

  Built `prototype_motion_gated_linker.py` (Python-only, no production/
  detection changes): `MotionGatedLinker`, a per-tracklet constant-
  velocity Kalman filter replacing `DotTrackletLinker`'s fixed 60px
  nearest-neighbor gate. Gates on Mahalanobis distance of the KF
  innovation instead of raw pixel distance, so an occluded track *coasts*
  forward on its own last known velocity rather than freezing at its last
  seen pixel -- directly targeting the diagnosed mechanism (a different
  real marker later drifting near that frozen, stale point gets adopted).

  Validated against real, human-produced ground truth -- exactly the test
  set Harri suggested: every 4-element (sub-range) member in
  `tracklet_group_assignments.json` records a real human-determined
  identity-switch frame. Extracting both bounds of every such fragment
  gave **11 real cut points** across 4 cameras.

  Result: **11/11 correctly separated** -- the new linker independently
  recovers every single manual split Harri made, with zero manual input,
  using only the raw already-detected candidate positions. A purity
  check on the 31 *clean* (non-split) tracklets found the default
  process-noise setting too tight for a smooth deceleration (a real
  foot-plant, not an occlusion -- confirmed by hand-tracing
  `gopro13_02#275`'s own frame-by-frame positions: no gap, just slowing
  motion the constant-velocity model's default noise couldn't track
  without accumulating bias); loosening `accel_std_px` 6->10 fixed it
  with no cost to the 11/11 result (loosening further, 15+, started
  losing real separations -- not worth it for a purity gain elsewhere).

  A second, different failure mode surfaced and was investigated by
  hand: `gopro13_02#283` showed near-frame-by-frame flip-flopping between
  two output ids -- not an occlusion artifact but two *simultaneously
  visible*, closely-spaced real markers (almost certainly adjacent
  knee/ankle markers) both sitting inside one track's gate every frame,
  with the Hungarian solver's global-optimum pick flipping which one
  "wins" on tiny cost differences. This is the same close-marker-
  projection ambiguity that originally motivated the whole marker-catalog
  -and-assignment redesign (§0) -- a motion-only gate doesn't fully solve
  it on its own. Added a per-frame disambiguation margin (B2's own
  "runner-up must be clearly worse" idea, applied per frame: refuse to
  commit a match when the best and second-best candidates are near-tied,
  coast instead of risking a wrong pick) -- fixed `#283` dramatically
  (0.59 -> 0.99 purity) with the 11/11 regression result unaffected, at
  the cost of mild new fragmentation on a couple of already-marginal
  short tracklets. Not fully solved for every close-marker case, and not
  expected to be by a single-camera 2D fix alone -- this is exactly what
  B2's cross-camera reprojection test (and, failing that, a human via the
  now-fast scrub tool) exists to catch.

  **Recommendation, not yet actioned**: port `MotionGatedLinker` (with
  the disambiguation margin) into production `dot_tracklet.py`
  (`accel_std_px=10`, `disambiguation_margin=2.0` as the validated
  starting point), re-run detection on the full capture, then proceed to
  the full-capture B2/B3 pass -- on data needing substantially less
  manual identity-switch cleanup than an unmodified linker would produce.

- **2026-09-11** (B3) — Replaced the blind typed-frame-number split
  dialog with a real scrub UI, per Harri's feedback after using it: the
  main grid shows timestamps, not frame numbers, and there was no way to
  actually *look* at the candidate frames before committing to a split
  point. Added `_ScrubSplitDialog`: steps one at a time through a
  member's own real frames (not the grid's handful of coarse sample-time
  columns), rendering the actual cropped/marked image at each step via a
  new `GroupAnalyzer.frame_image()` (a direct specific-frame render,
  factored out of `sample_image()`, reused so scrub steps hit the same
  disk cache) plus a slider and ±1/±10 step buttons, showing both the
  video_frame index and the real time at every position. "Split checked
  member's tracklet (scrub)..." opens it seeded at the auto-suggested
  transition frame; right-clicking any thumbnail now offers "Scrub &
  split near here..." (seeded at that exact frame) alongside the existing
  immediate "Split before this frame". Verified against `gopro13_02#287`:
  dialog opens at the suggested frame 6948 with a real decoded image,
  scrub steps render correctly, and the returned cut frame is always one
  the tracklet actually has. Full `_on_select` render sweep (the exact
  code path building the new thumbnail widgets) re-run clean across all
  24 groups.

- **2026-09-11** (B3) — Four more fixes from Harri's continued
  review of the 24-group test run:

  1. **"No frame shown, can't evaluate" + "how are time columns chosen"**:
     the old sample-instant selection only ever picked instants where 2+
     members' triangulation succeeded -- a member with no such overlap
     (e.g. only briefly co-visible with the rest) could go through every
     rendered column with nothing to show, indistinguishable from a bug.
     Redesigned `GroupAnalyzer.analyze()`'s selection: one instant at
     *every* member's own median real time (guarantees each is shown at
     least once) plus an even 5-point spread across the group's whole
     real time span, merged/deduped and capped at 8 columns. Verified on
     `group_4`: went from 3 fixed columns to 8 spanning 42.0-48.0s, with
     every member's own median time represented.

  2. **Root cause confirmed for a within-tracklet identity drift**: Harri
     spotted `gopro13_02#287` showing `heel_L` at t=44.74/46.18 but
     `heel_R` at t=48.00 -- one *single* tracklet silently switching which
     real marker it's tracking partway through -- and correctly diagnosed
     the cause before I checked: `DotTrackletLinker`
     (`posetrak/detection/dot_tracklet.py`) links purely by a fixed
     60px nearest-neighbor gate (`max_link_px`) with **no
     appearance/identity check at all**, so it can hop from one real dot
     onto a different nearby one (e.g. during a foot-crossing moment) and
     never know it happened. Confirmed exactly as described, filed as a
     real production-pipeline follow-up (tightening/adding an identity
     check to the linker, or a confidence-based gate) -- not fixed today,
     since it needs a full re-detection run to validate, out of scope for
     this B3 review pass.
     For the immediate, practical need ("how do I split this, and how do
     I know the timestamp"): added `GroupAnalyzer.classify_member_frames`
     (per-frame, single-camera nearest-real-slot classification via the
     FK-predicted 2D projection -- coarse and known-approximate, since 2D
     proximity is exactly the ambiguity source this whole redesign exists
     to move away from, but good enough to localize *where* a tracklet
     destabilizes) plus `_suggest_split_frame` (best two-segment
     identity-change point). New GUI action "Split checked member's
     tracklet at frame..." (dialog, pre-filled with the suggestion) and a
     right-click "Split before this frame" on any shown thumbnail cut one
     member's own tracklet into two frame-range-restricted fragments of
     the same tracklet_id, which can then be routed to different groups
     via the existing "Split checked into new group" action. Verified end
     -to-end on `gopro13_02#287`: suggested cut at frame 6948 (t=46.11s)
     lands right at the boundary between a long stable classification run
     and an unstable multi-slot-confusion run -- matching Harri's own
     visually-identified transition window -- and the split action itself
     cleanly produces 164 + 227 frame fragments from the original 391.
     Member JSON format extended: a 4-element
     `[camera_label, tracklet_id, frame_lo, frame_hi]` entry alongside the
     existing 2-element (whole-tracklet) form.

  3. **"Switching between groups is slow"**: `sample_image()` re-decoded
     a video seek from scratch on every single thumbnail, every group
     render, every time. Added an on-disk crop cache (PNG files under
     `<groups-file-dir>/thumb_cache/<svid>/<tracklet_id>_<frame>.png`,
     keyed on exactly what determines a rendered crop) -- measured ~190x
     speedup on a cache hit (0.48s cold decode vs. 0.0025s cached) on a
     real frame from this run. Persists across GUI relaunches on the same
     groups file, not just within one session.

  Verified individually against the real 24-group run: `group_4`'s
  sample-time spread, the `gopro13_02#287` split (164+227 frames from
  391), and cache hit/miss both returning identical correct pixmaps. A
  full 24-group re-render sweep (every group, every member, every
  sample-time column) came back clean: 0 exceptions.

- **2026-09-11** (B3) — **Fixed a column-misalignment bug** found by
  Harri on his first pass with the reworked grid UI: `group_4`
  (`gopro13_02#287`, `pixel9#443`, `pixel9#1313`, `gopro-11_mini_01#360`,
  `gopro-11_mini_01#427` -- a flagged-contradiction group) showed, for a
  single "t=..." column, two `gopro-11_mini_01` images (#360 and #427)
  that were visibly different real moments (right-foot vs. left-foot
  stance).

  Confirmed against real data: `#360` and `#427` are temporally disjoint
  fragments (frames 8127-8330 -> t=44.65-46.35s, and frames 8375-8645 ->
  t=46.72-48.97s -- a real gap, not an off-by-one). `GroupAnalyzer.
  sample_image()`'s "nearest frame this tracklet actually has" fallback
  had no bound on how far away that nearest frame could be, so for a
  column whose real time fell inside `#360`'s span, `#427`'s row still
  rendered its own single nearest frame (up to ~2s away) as if it were at
  that instant.

  Fixed by rejecting the fallback when the candidate frame's own real
  time (via `sync_table.frame_to_global_time`) is more than
  `_MAX_SAMPLE_GAP_S` (0.15s) from the column's requested instant --
  shows `(no frame here, ±Ns away)` instead of a mismatched frame.
  Verified against `group_4` directly: `#427` and `pixel9#1313` (whose
  spans likewise don't cover this window) now correctly return no image
  with the real gap (0.6-2.3s) at each of the group's three sample
  times, where they previously would have rendered a stale frame.

  This is a UI-honesty fix, not a grouping fix -- `group_4` staying
  flagged `[CONTRADICTION]` is still the right call for Harri to resolve
  via split, same workflow as before.

- **2026-09-11** (B3) — **Built** `label_tracklet_groups_gui.py`
  (PySide6): consumes B2's `tracklet_groups.json`, independently
  re-derives per group (not stored in B2's own small JSON) a fused 3D
  trajectory (N-view triangulation at shared instants), ranked real-slot
  suggestions (mean 3D distance to each slot's FK-predicted trajectory),
  and a few sample frames per member camera with the tracklet's own dot
  circled. Group list (sorted by size) on the left; selecting one shows
  thumbnails + a ranked-suggestion table; Assign / Reject / Remove-member
  (split) / Merge-into (target group_id) actions; saves incrementally to
  `tracklet_group_assignments.json`.

  Also extended `build_tracklet_groups.py`'s `--output` with a `meta`
  block (session/shot/tracking-run/detection-run/time-window) and stable
  `group_id`s, so B3 doesn't need those repeated on its own command line.

  Two real bugs found and fixed via a headless PySide6 smoke test before
  handing it off: (1) some groups (the flagged contaminated/ambiguous
  ones) have *multiple tracklets from the same camera* -- the anchor-vs-
  other-member comparison only checked `camera_id`, not the full
  `(camera_id, tracklet_id)` pair, so a same-camera sibling tracklet got
  mistaken for the anchor itself and crashed on a `KeyError`. (2) the
  suggestion mechanism compared the fused trajectory against
  `_DEFAULT_TRIAL`'s own probe names (`a_hip_R`, `k_Xp_R`, ... -- P-A's
  unlabelled exploratory offsets), not the real anatomical slot names the
  assignment combo box uses, so a suggestion could never actually be
  picked. Fixed by adding `_SLOT_TO_PROBE`, the empirical mapping
  `eval_fk_prediction.py` already derived (real slot -> best-matching
  probe) -- several real slots still share one probe (the trial catalog
  doesn't yet cleanly separate every same-joint offset), an honest
  reflection of that known limitation, not a new bug.

  Smoke-tested (offscreen QPA) against the validated 5-camera, 42-48s
  run: loads 24 groups; a clean 2-member group shows a confident
  suggestion (`toe_L` at 5.7cm, runner-up 19cm); the flagged 6-member
  contaminated group's suggestions all bunch around ~50cm -- itself a
  useful signal to a reviewer that something's off before they even look
  at the images. Remove-member and Reject both verified to save correctly
  and leave the tool in a consistent state after.

  **Reworked after Harri's first real try** -- two real usability gaps,
  not just polish: (1) sample frames were a single long strip with no
  structure, hard to compare across cameras -- changed to a grid, one
  row per (camera, tracklet), one column per shared timestamp, so the
  same instant lines up across cameras. (2) a concrete real case (one
  flagged-contradiction group actually contained *three* different
  physical markers -- hip_L, ankle_lat_L, ankle_lat_R -- chained together
  by transitivity) exposed that "Remove member" only discarded one
  member with nowhere to go -- no way to actually *form the correct
  group* from what's removed. Replaced with checkboxes per member row +
  "Discard checked" (noise, belongs nowhere) / "Split checked into new
  group" (this subset is a different real marker -- creates a proper new
  group_id from just those members). Verified on Harri's exact example:
  checking the pixel9+oneplus pair and splitting produced a clean new
  group whose top suggestion is `ankle_lat_R` at 4.8cm -- matching what
  he'd identified visually.

  Next: Harri reviews the 24 groups from the 42-48s run interactively
  with the reworked UI; then a full-capture B2 run (33.62-130.185s) for
  real calibration input.

- **2026-09-11** (B2) — **Built and validated** `build_tracklet_
  groups.py`, the redesign doc §4 algorithm (pre-filter by lifetime + FK
  proximity, pairwise reprojection consistency over shared frames,
  disambiguation margin over the runner-up, connected components,
  contradiction flagging), plus `--validate-against-gt` (matches
  labelled GT points back to their real `tracklet_id` -- free
  cross-camera tracklet-group ground truth, no extra labeling).

  Two real bugs in the *validation*, not the algorithm, found and fixed
  before trusting any number: (1) GT tracklet identity was derived from
  the full 40-60s labelled set but tested against a narrower window --
  most GT tracklet_ids for a slot simply don't exist in the shorter
  window (the same physical marker gets a fresh id after every
  occlusion), so recall's denominator was wrong (0.02 -> meaningless).
  Fixed by restricting to nodes whose tracklet actually exists in the
  tested window. (2) Even fixed, many "same-slot GT pairs" are
  *temporally non-overlapping fragments* of one physical marker (zero
  shared frames) -- not something a simultaneous-reprojection test can
  ever link, a different problem (fragment-stitching) entirely out of
  B2's scope. Fixed by only counting a GT pair in the recall denominator
  if it actually has >= min-shared-frames -- confirmed both fixes by
  hand-tracing one real pair (gopro-11#125 / gopro13_02#88, ankle_med_R):
  median reprojection error 0.94px, correctly grouped.

  **Validated baseline** (42-48s, 4 cameras excluding the noise-flooded
  gopro13_01 -- see below): recall 0.52, precision 0.56 (median<=3px,
  p90<=6px thresholds). Looked at every miss/false-positive by hand: on
  the two clean cameras (gopro-11, gopro13_02) matches are excellent
  (0.9-2.5px) except `hip_L`, sitting at a genuine 4.3-4.6px near-miss;
  weak-camera pairs (gopro13_01, oneplus, one pixel9 tracklet) show real
  inconsistency, sometimes matching beautifully, sometimes wildly wrong
  (40-200px) -- looks like genuine tracklet-quality noise on those
  cameras, not an algorithm bug. Tried loosening to median<=5px/p90<=8px
  to catch `hip_L`: recall rose to 0.63 but **precision collapsed to
  0.23** -- 3px/6px is close to the right knee of the curve, not too
  tight; a global threshold bump trades away far more than it buys.
  Reverted to 3px/6px.

  `gopro13_01` alone produced **14,204 raw tracklets in this 6s window**
  (vs 300-2300 for other cameras) -- its near-disabled glare veto
  (P-D's frac=0.95) floods it with short, noisy tracklets that poisoned
  early grouping runs with spurious chance-agreement links; excluded
  from validation via a new `--camera-label` filter. A real,
  not-yet-solved question for using gopro13_01 in B2 at all: does its
  genuine marker keep a stable tracklet_id under that much candidate
  density, or does the linker itself hop between the real dot and
  nearby glare? Worth checking before including it in a real B3 pass.

  Accepted, not chased further this pass: B2 doesn't need to be perfect
  -- it feeds B3's manual review, where a person confirms, splits, or
  merges. A near-miss like `hip_L` still surfaces as a top-ranked
  suggestion for its own ungrouped tracklet; the remaining gap is B3's
  job, not a reason to keep tuning B2 in isolation.

  **Follow-up, same day**: checked whether gopro13_01 (failing ring
  light, per Harri, same as insta_ace2_pro) could be recovered rather
  than excluded. Its *individual* tracklets turned out genuinely stable
  once locked on (median 0.25-0.31px frame-to-frame jitter over 500+
  frames on three checked examples) -- the earlier 40-200px reprojection
  "failures" involving it were traced to the free GT-derivation trick
  itself: the sparse 25-timepoints-over-20s GT set can't resolve fast
  real inter-sample motion, so its majority vote sometimes attributes
  one slot name to several different, simultaneously-active real
  tracklets. So gopro13_01's *tracklets* are trustworthy; its *raw
  detection* is the noisy part (14,204 tracklets in this 6s window vs
  300-2300 elsewhere, from P-D's near-disabled glare veto). The existing
  FK-proximity pre-filter alone brings that down to 81 -- in line with
  other cameras -- so tried including it in the full B2 run: **recall
  and precision both got strictly worse** (0.52/0.56 -> 0.40/0.38), with
  the *same* 14 true positives either time -- it added zero genuine new
  links, only false ones, and chained 5-6 of its tracklets into single
  contaminated groups via transitivity. Tried tightening the pre-filter
  to require 60% of a tracklet's own frames near a prediction (not just
  one) -- **no change at all**, identical numbers: its false-linking
  tracklets are near a prediction persistently, not by a lucky single
  frame, so this filter axis doesn't discriminate them. Concluded:
  excluding gopro13_01 from B2/B3 is the right call for now -- two
  reasonable attempts to recover it both made things strictly worse.
  Not pursued further (identifying what those specific tracklets
  actually are would need real visual investigation, for a
  hardware-limited camera -- diminishing returns unless this capture
  specifically needs the coverage later).

  Next: build B3 (manual tracklet-group -> slot assignment GUI, on the
  5 cameras excluding gopro13_01), or run B2 across the full capture
  range for real calibration input first.

- **2026-09-10** (P-C annotation tooling) — First tried extending
  the cv2 `label_dot_ground_truth.py` with a `--slots` mode +
  `--seed-detection-run`; Harri found it too clunky (no context menus,
  every left-click adds a point, `unlabeled` text clutter). Replaced with
  **`label_marker_slots_gui.py`** (PySide6): frame opens with the
  detector's candidates pre-placed (grey, no clutter); **right-click a
  dot → menu → pick slot** / `unlabeled` / Delete; **Ctrl+left-click
  empty → add a missed dot**; left-click selects, `1`-`9`/`0` set the
  selected dot's slot, Delete removes. Previous same-camera frame's
  tagged dots show as ghosts; "Carry from prev" snaps each to the
  nearest untagged current dot. **Track id derived** (`= slot_palette
  .index(slot)`, one physical marker per slot) rather than entered.
  Output JSON identical shape (`points` + parallel `point_meta`
  `{slot,track}`), so `refine_dot_ground_truth.py` /
  `validate_dot_detector.py` are untouched. Added
  `build_gt_frame_manifest.py` (global time -> per-camera video_frame via
  the sync table). Built a tiny 3-time x 3-good-camera manifest
  (`scratch/dot_ground_truth/pc_tiny_manifest.json`, 9 frame-views) to
  prove the tool -> metric loop before scaling to the full 25 x 6.

  **Tool -> metric loop proven end to end.** Harri labelled the 9-frame
  tiny set (`scratch/dot_ground_truth/pc_labels.json`), adding a few
  detector misses by hand. First metric script `eval_dot_detection.py`
  (Stage A, redesign doc §2.2 -- per-camera recall/precision via
  Hungarian match at a radius gate) against detection run `01c2e3c1`:

  | camera | GT slot-dots | recall | precision | precision* | #det |
  |---|---|---|---|---|---|
  | gopro-11_mini_01 | 18 | 1.00 | 0.43 | 1.00 | 42 |
  | gopro13_02 | 19 | 1.00 | 0.49 | 1.00 | 39 |
  | pixel9 | 14 | 0.64 | 0.53 | 1.00 | 17 |

  `precision*` (detection matched *any* real GT dot incl. `unlabeled`)
  = 1.00 everywhere -> blacklist-mode detector produces no junk; ~half
  its detections are real reflective things that just aren't leg markers
  (prop dots, the other person, glare). Recall 1.0 on the two 4K
  GoPro-family cameras; **pixel9 0.64** -- misses `hip_R` in all three
  frames plus `ankle_lat_R`/`toe_L` once (1080p, dimmer/smaller
  markers). No `knee_front_*` labelled in any frame (check whether those
  markers were placed).

  **Fixed `refine_dot_ground_truth.py`**: it snapped GT points to
  *subtract-mode residual* centroids, but the production detector uses
  *blacklist-mode raw-brightness* centroids -- so refining pulled GT
  points a few px *off* the detector's own centroids (`precision*`
  1.00->0.7 after refine, spurious misses). Added `--centroid-mode`
  (default `blacklist`, matches production: raw-threshold at
  `--blacklist-threshold`, contour-moments centroid; `subtract-ladder`
  keeps the original for subtract-mode captures). Blacklist mode needs
  no background decode -> 7 s instead of 5+ min. Re-refined; metric at
  radius **4 px** now clean: gopro-11 / gopro13_02 recall 1.00,
  precision* 1.00, median match distance 0.0 px; pixel9 recall 0.64,
  precision* 0.88 (2 faint top-of-frame `unlabeled` points refine
  couldn't place well). Loop **label -> refine -> metric** is solid.

  **Full 25 x 6 GT labelled** (`pc_labels.json`, 126 frame entries).
  `insta_ace2_pro` left entirely unlabelled -- its ring light was
  failing during the take, markers look too different to be worth
  extending bright-dot detection to (a capture-hardware issue, not a
  tuning one). Stage A metric on the full refined GT (blacklist run
  `01c2e3c1`, threshold 200, radius 4 px):

  | camera | GT markers | recall | precision | precision* |
  |---|---|---|---|---|
  | gopro-11_mini_01 | 133 | 0.98 | 0.48 | 1.00 |
  | gopro13_02 | 140 | 0.96 | 0.59 | 1.00 |
  | pixel9 | 82 | 0.70 | 0.48 | 0.91 |
  | gopro13_01 | 80 | **0.12** | 0.09 | 1.00 |
  | oneplus9pro-01 | 103 | **0.01** | 0.02 | 1.00 |

  gopro13_01 detects plenty (113 candidates) but they're prop/glare, not
  markers (precision* 1.0, recall 0.12) -- its markers sit at brightness
  ~200 / saturation 50-80, clipped by threshold 200 and rejected by
  max_saturation 45. oneplus threshold 200 far above its tone-mapped
  marker brightness. `precision` ~0.5 on the good cameras is expected
  (half the detections are real non-body dots -> assignment's problem).

  **P-D underway** -- `sweep_dot_detection.py` (new): runs `detect_blobs`
  directly on the labelled frames over a threshold x max_saturation
  grid, per camera. First sweep (blacklist veto, short-span background)
  was misleading -- the background was subject-contaminated so the veto
  over-killed dim markers. Re-run with `--no-blacklist` (pure
  threshold/saturation/shape) is clear:

  | camera | marker brightness peak | recall @ prod (thr200/sat45) | best no-veto recall |
  |---|---|---|---|
  | gopro-11_mini_01 | >210 | 0.97 | 0.97 (thr>=170) |
  | gopro13_01 | **~130** | 0.12 | **0.72** (thr 120, sat 90) |
  | oneplus9pro-01 | **~175** | 0.01 | **0.77** (thr 170, sat 45) |
  | pixel9 | ~200 | 0.70 | **0.89** (thr 180-190, sat 120) |

  gopro13_01's markers are genuinely dim (~130) -- threshold 200 clips
  them all; ~120 recovers 72%. oneplus markers sit right at ~175 (170
  recovers 77%, collapses again at 190). pixel9 gains from a lower
  threshold *and* a looser `max_saturation` (its and gopro13_01's
  markers are ~50-90 saturated). No-veto precision is ~0 (all glare
  passes) -- the definitive per-camera numbers need the veto with a
  *subject-free* background; that sweep is running (45k-frame bg span).
  **Done**: added `dot_max_saturation_by_camera` and
  `dot_blacklist_frac_by_camera` to `MarkerDetectionPipeline` /
  `load_pipeline_for_capture_object` (same `.get(camera_instance_id,
  scalar_default)` pattern as the existing `dot_threshold_by_camera`),
  wired `_process_camera_core` to use both, and matching
  `--dot-max-saturation-by-camera` / `--dot-blacklist-frac-by-camera` CLI
  flags on `run_standalone_marker_detection.py`. New test
  (`test_pipeline_passes_per_camera_max_saturation_and_blacklist_frac`)
  confirms the plumbing at the `detect_blobs()` call site. All 26
  `test_marker_pipeline.py` tests pass.

  Also re-ran the veto-frac sweep (§ above found frac=0.7 too tight):
  0.85 recovers gopro-11's veto-lost recall (0.92->0.98) at a modest
  precision cost; gopro13_01 needs 0.95 to get anywhere (0.45->0.63 at
  thr 130). **Launched a fresh full-capture detection run** (append-only,
  new `detection_runs` row, `01c2e3c1` untouched) with: global
  threshold=200/max_saturation=45/blacklist_frac=0.85, per-camera
  overrides gopro13_01={130,90,0.95}, oneplus9pro-01={150,45,0.85},
  pixel9={180,90,0.85} (gopro-11_mini_01/gopro13_02 keep the global
  defaults; insta_ace2_pro still included at defaults despite its known
  ring-light failure, for a uniform run). `run_parallel()`, same
  33.62-130.185s range as `01c2e3c1`.

  **Confirmed on the real run** (`detection_run_id 6abcba67`, 69,486
  frames): recall gopro13_01 0.12->**0.65**, oneplus 0.01->**0.62**,
  pixel9 0.70->**0.80** (gopro-11/gopro13_02 unchanged at 0.98/0.96, as
  expected -- their settings didn't change). Real cost, as expected from
  "recall first, let downstream filter": precision on gopro13_01 fell to
  **0.01** (frac 0.95 nearly disables its glare veto, ~4270 detections
  for ~80 real markers) and dropped somewhat on gopro-11/gopro13_02 too
  (the global frac 0.7->0.85 bump). Accepted -- a missed detection is
  unrecoverable, a false one is the assignment layer's problem.

- **2026-09-11** (later) — The real **P-A metric**: FK-predicted marker
  position vs. the actual slot-labelled GT (not "nearest raw detection",
  P-A's original proxy before GT existed). New
  `eval_fk_prediction.py`; also finds each GT slot's empirical best-
  matching probe from `prototype_fk_marker_prediction.py`'s `_DEFAULT_
  TRIAL` (majority vote), resolving what P-A left open (which
  unlabelled `k_Xp`/`k_Zm`/etc. probe direction is which real
  anatomical slot).

  First run surfaced a real bug, not noise: left-side GT slots showed
  60-200px "error" against right-side slots' 11-40px. `_DEFAULT_TRIAL`
  only ever had **right-leg** entries -- left-leg GT was being compared
  against right-leg predictions. Confirmed the fix empirically against
  the skeleton's own *rest pose* (zero joint angles -- true bind-pose
  symmetry, not confounded by the current animated pose putting each leg
  somewhere different): thigh.L/shin.L/foot.L's local axes are
  thigh.R/shin.R/foot.R's under reflection across the sagittal plane
  composed with a per-side local-frame correction -- net rule, a local
  offset/normal (x,y,z) on the right mirrors to (-x,y,z) on the left
  (only the medial/lateral local-X component flips; down-bone Y and Z
  unchanged). Implemented as `_mirror_to_left()`; `_DEFAULT_TRIAL` is now
  `_RIGHT_TRIAL` + its mirror, 32 entries total.

  **Re-run, clean**: median prediction error 10-40px across all 16
  slots, consistent between left and right (e.g. knee_lat_L 12.7px /
  knee_lat_R 14.0px; heel_L 12.9px / heel_R 14.0px) -- confirms the P-A
  keystone premise with a real number against real ground truth, not
  just "looks about right" from the earlier probe video. Side finding
  for the next catalog-authoring pass: `ankle_lat_R` and `ankle_med_R`
  both pick the plain joint-center anchor as their closest probe, not
  any of the offset probes -- the generic +-4.5cm ring doesn't actually
  bracket where the real ankle markers sit; needs real tuned offsets,
  not guessed ones.

  Next: author real named medial/lateral/anterior offsets (replacing the
  unlabelled probes) using this diagnostic, or move to P-B (cross-camera
  tracklet grouping + manual tracklet->slot assignment).

- **2026-09-11** (later) — Wrote the detailed **P-B design** (redesign
  doc §4, replacing its earlier sketch): B1 per-camera tracklets (no new
  code, `DotTrackletLinker`'s existing id); B2 cross-camera tracklet
  grouping (`build_tracklet_groups.py` -- pre-filter by lifetime + FK
  proximity, pairwise reprojection-consistency test over *shared frames*
  -- the real difference from per-frame fusion, a sustained agreement not
  one lucky frame -- disambiguation margin over the runner-up, connected
  components, contradictory triples flagged not forced); B3 manual
  tracklet-group -> slot assignment GUI (`label_tracklet_groups_gui.py`,
  new tool sharing `label_marker_slots_gui.py`'s image helpers -- group
  list + trajectory view, not a frame canvas, different enough to want
  its own tool); B4 calibration fit (closed-form least squares per slot
  from the group's fused 3D path vs. the parent joint's FK transform,
  scale-robust `along`/`lateral`/`anterior`, `normal` defaulted to the
  radial direction, residuals reported); B5 output (calibrated
  attachment set, same file shape as the nominal catalog + a
  `calibration:` provenance block). Validation: extend
  `eval_fk_prediction.py` with `--attachment-set` to check the fitted set
  against GT (purity should jump, error should drop toward tracking-noise
  level) -- directly tests whether the `ankle_lat_R`/`ankle_med_R`
  "closer to the plain anchor" finding was a probe-placement problem.
  Real, cheap validation for B2 itself noted: the existing slot-labelled
  GT's own detections can be matched back to their `tracklet_id`,
  yielding real cross-camera tracklet-group ground truth for free to tune
  B2's thresholds against, rather than guessing them. Not yet
  implemented -- design only.

- **2026-09-10** (P-A) — Ran the keystone experiment (redesign
  doc §8): `python/tools/prototype_fk_marker_prediction.py` reads the
  existing markerless `tracking_results` (smoothed state) + skeleton
  YAML, does FK via `posetrak.db.skeleton_layout.SkeletonLayout`,
  resolves a hand-authored trial right-leg attachment set (real skeleton
  anchors + `±X`/`±Z` probe markers around the shin) to world position +
  FK-carried normal, projects into all 6 cameras, computes the facing
  prior, and renders a 6-camera grid video
  (`scratch/dot_ground_truth/fk_prediction.mp4`).
  **Result: the premise holds.** On the 3 cameras that calibrate well
  *and* detect the leg dots (gopro13_02, gopro-11_mini_01, pixel9),
  nominal FK prediction lands median ~10-40 px (~2-4 cm) from a real
  detection, on the right body part -- inside a usable clustering radius
  (gopro13_02: 90-98% of frames matched within 60 px). Extrinsics are
  fine on all 6 (a tracker-trusted FK 3D knee reprojected vs the vitpose
  knee = ~15-30 px on every camera); the earlier "3 cameras 200-1500 px
  off" was a metric artefact of measuring "distance to nearest
  *detected* dot" on cameras that detect almost none. So **Stage A
  per-camera detection (P-D) is the binding constraint**, not prediction:
  gopro13_01 / insta_ace2_pro / oneplus9pro-01 match 0-15% of predictions
  purely from under-detection. The FK-carried normal / facing prior works
  (paired probes show consistent opposite facing signs; filled-vs-hollow
  rendering separates front/back correctly). D6 resolved: leg joints'
  `bone_tip_offset = [0,+L,0]` is usable as `e_long`. Follow-ups: author
  a real medial/lateral/anterior leg set (this pass used unlabelled
  probes to find the local-axis mapping); consider running P-D before
  P-B.

- **2026-09-10** (later) — Harri review of the fusion rewrite + current-
  state doc. Substantive pushback, not tuning: (1) marker grouping by one
  shared joint anchor is wrong (markers can be mid-bone / near the child
  joint; markers close in 3D project close in one view -- the main
  mislabel cause in the latest video); (2) the pelvis-proxy for
  anatomical direction is too weak -- normals should be defined in the
  *parent joint frame* and taken to world via FK from the markerless
  result / tracker predict; (3) tracklets should probably be the primary
  assignment unit (match two tracklets in 3D if per-frame errors over
  shared frames are under threshold and better than alternatives; assign
  a whole tracklet to one slot if all its frames match); (4) the metrics
  are ad-hoc and need agreed definitions + ground truth, separately per
  stage; (5) Stage A: gopro13_01 / oneplus / insta_ace2_pro miss most
  visibly-present dots. Also a strategic fork raised: enrich the catalog
  enough to predict a marker's projected position + visibility from the
  markerless result *before* per-person calibration, OR do manual
  tracklet->slot assignment for the calibration capture first, get
  marker-augmented tracking working from per-person-calibrated markers,
  and solve auto-assignment later.

  Delivered per request: two diagnostic grid videos (all 6 cameras, 40-50s):
  `scratch/dot_ground_truth/all_dots_overview.mp4` (every raw candidate,
  no assignment) and `all_tracklets_overview.mp4` (candidates coloured +
  numbered by DotTrackletLinker tracklet_id) --
  `python/tools/render_dot_detection_overview_video.py`.

  Stage A diagnosis (real data, frame at t=42.925): the markers gopro-11
  renders at brightness 254 / saturation 2 (saturated pure white, ideal
  for threshold 200 + max_saturation 45), gopro13_01 renders at ~198-209
  / saturation 27-79, and oneplus at ~76-228 / saturation up to 164 on
  the leg. threshold=200 and max_saturation=45.0 were effectively tuned
  for the GoPro Mini alone; the other cameras render the same physical
  markers dimmer and more chroma-tinted, so both filters reject real
  markers there. Relaxing to threshold 150 roughly doubles gopro13_01's
  and oneplus's candidate counts. `dot_threshold_by_camera` already
  exists for exactly this but was only ever set for pixel9; a per-camera
  `max_saturation` override does not exist yet. Not re-run -- feeds the
  design phase.

  Next: a scoped design phase before more implementation.

- **2026-09-10** (later still, after Harri's inline comments) — Revised
  the redesign doc §1.3–1.4 per those comments: no "link" concept
  (markers keep a plain parent joint, per the existing schema); the
  bone-local frame is defined **canonically for the right side** with a
  `mirror: true` flag (avoids the left-side handedness/anatomy
  inconsistency); `e_long` uses the schema's existing `bone_tip_offset`
  instead of an undefined "primary child joint". Added §1.11 recording
  the skeleton-schema redesign (decouple topology / markers / metrics /
  placement; explicit link+twist concepts) as **deferred** — Harri:
  "let's not do it yet... don't understand all schema requirements yet".
  Added **§8, a concrete prototyping plan** answering "how does
  prototyping continue": P-A FK marker-slot prediction from the existing
  markerless `tracking_results` + a hand-authored trial right-leg
  attachment set (the keystone experiment — does nominal prediction +
  markerless pose land close enough to detections to drive clustering,
  and get visibility sign right); P-B cross-camera tracklet grouping +
  manual tracklet→slot assignment → calibrated attachment set (the
  manual-first deliverable); P-C minimal slot-labelled GT + per-stage
  metric harness (parallel with P-A); P-D per-camera detection
  calibration; P-E body-model spike. Order: P-A + P-C now in parallel,
  P-B after P-A, reconvene before committing to §3 auto-assignment.

- **2026-09-10** (earlier) — Wrote
  [marker-catalog-and-assignment-redesign.md](marker-catalog-and-assignment-redesign.md),
  the design doc for that phase. Deep sections, per Harri's priority: (1)
  **catalog spec** -- markers attach to a *link* (identified by its
  proximal joint), not a joint anchor; offset stored scale-robustly as
  `along` (fraction of bone length) + `lateral`/`anterior` (metres) in a
  link-local frame `(e_long, e_lat, e_ant)` built from the skeleton rest
  pose without depending on its Euler `orientation`; explicit surface
  `normal` in the same frame for FK-carried backface culling; one-side
  authoring + sagittal mirror; modules composed per capture; nominal vs.
  per-person-calibrated attachment sets, same file format. (2)
  **metrics** -- two GT captures (CAL slow / HARD fast), multi-camera
  hand-labeled dot GT with slot names + per-camera track ids + triangul-
  ated 3D slot positions; a distinct metric per stage (detection
  recall/precision per camera; tracklet fragmentation/purity/coverage/ID-
  switch vs GT tracks -- the principled version of the ad-hoc "purity";
  fusion 3D error / recall / false-merge / view-completeness; assignment
  per-slot precision/recall split by confidence and by near-neighbour
  presence, plus temporal consistency). Sketched: FK-based slot
  prediction replacing the pelvis proxy; per-camera projected-cluster +
  cross-camera-3D + tracklet-native assignment replacing the old
  Stages C-E; tracklet->slot manual assignment for calibration captures.
  Recommends manual-first (calibrate markers by manual tracklet
  assignment, get marker-augmented tracking working, build auto-assignment
  against the resulting GT + calibration afterward). Includes a §6 on
  whether to align with a permissive body model (SOMA-X / ANNY both
  Apache; SMPL rejected on licence + joint quality) -- recommends a
  1-2 day spike (licence audit incl. model assets, pick a reference
  skeleton topology so catalog files are forward-compatible, assess
  SOMA's labeling approach as an architectural alternative), not adoption
  now.

- **2026-09-10** — Rewrote `fuse_frame()`'s view-membership logic per
  Harri's explicit request ("fusion should group all detections that
  reproject close, it does not make sense to just apply to the winning
  pair"): a fused point now reprojects into *every* camera with
  candidates (not only the pair that seeded it) and claims each camera's
  best-matching candidate via global greedy best-first resolution, sorted
  by reprojection error -- one raw candidate can now only ever support
  one fused point. This structurally eliminates the shared-raw-pixel bug
  from the previous entry (no longer needs the local
  `_dedup_shared_view_fused_points` workaround in
  `prototype_marker_normal_assignment.py`, which was removed) and expands
  coverage to cameras that weren't part of the winning triangulating
  pair. Measured, honest trade-off on the same validation clip: purity
  improved further (0.814 -> 0.821) but ambiguous-group coverage actually
  *dropped* (7,436 -> 4,734 assignments) rather than increasing -- the
  more aggressive global consensus also eliminates more low-quality
  duplicate hypotheses, not just recovers extra views; net effect is
  fewer but higher-confidence fused points, not a straightforward
  coverage win. Re-rendered and re-verified the previously-buggy frame
  (8679) on both a single-camera video and a new 6-camera grid video
  (`scratch/dot_ground_truth/normal_aware_check_v4.mp4`,
  `normal_aware_grid_check.mp4`).

  Also wrote
  [person-marker-assignment-current-state.md](person-marker-assignment-current-state.md)
  -- a from-scratch (non-chronological) description of the pipeline as it
  actually stands today, requested by Harri after this session's chain of
  fixes made the design doc's original phased plan hard to reconcile with
  the real, current behavior ("I have a feeling that there are now so
  many changes that we do not see the forest from the trees").

- **2026-09-09** (later still) — Built marker-normal disambiguation for
  the ambiguous same-joint groups (`prototype_marker_normal_assignment.py`),
  next per Harri's own sequencing ("tracklets first, then normals").
  Confirmed a real structural gap this closes: every marker in an
  ambiguous group (e.g. `knee_L_medial`/`knee_L_lateral`/`knee_L_front`)
  shares one vitpose anchor index, so their triangulated 3D anchor in
  `hybrid_assign_frame` was numerically *identical* for all group members
  -- the existing Hungarian match had zero information to tell them apart
  and was tie-breaking arbitrarily, not deciding. This is almost
  certainly the real mechanism behind the medial/lateral/front flicker
  Harri reported.

  Approach: derive medial/lateral/anterior directions *anatomically* each
  frame from the skeleton's own triangulated pose (pelvis-width vector
  for medial/lateral, `cross(world_up, pelvis_vector)` for anterior),
  rather than requiring per-marker 3D geometric offsets baked into the
  catalog ahead of time. Real bug caught before trusting it: an earlier
  version derived anterior from each limb's own instantaneous thigh/shank
  direction, which swings with hip/knee *flexion* (raising a leg) even
  though the knee's true anterior-facing direction doesn't rotate with
  flexion, only with hip axial rotation -- caught by rendering the
  computed frame as arrows on a real image
  (`render_marker_normal_debug_frame.py`) and checking anterior against
  which way the subject was visibly facing: the limb-direction version
  pointed backwards on the raised (bent) leg while looking correct on the
  straight standing leg; the pelvis-anchored version matches on both.

  Validated on real data (10s/6-camera window, same clip as the tracklet-
  smoothing work): per-tracklet name-purity within ambiguous groups (how
  consistently a physical dot gets the same name) improved from a 71.9%
  baseline (`hybrid_assign_frame`) to 75.6% with directional preference
  alone, up to 82.1% when actively rejecting poorly-aligned pairings
  (`min_direction_score=0.6`) -- a real, expected coverage/purity
  trade-off (ambiguous-group assignment count drops from 13,325 to 8,582
  at that threshold). Spot-checking rendered frames shows the previously-
  impossible case working: `knee_L_lateral` and `knee_R_medial` correctly
  distinguished on two simultaneously-visible knees in the same frame.
  Individual-frame semantic correctness beyond that spot check not yet
  independently verified against ground truth -- video handed to Harri
  for the real review, matching how every other real bug this session
  found (the coordinate bias, the fusion-stage framing, the frame-7920
  swap, the render-pipeline blur) was ultimately caught by watching, not
  by a script-level check alone.

  That review immediately caught another real bug: `knee_L` was
  confidently mislabeled `medial` (should be `lateral`) for the first
  ~350 frames of the clip, correcting itself around frame 8130. Traced to
  real per-frame data, not a smoothing artifact: the 3D pass (available
  once cross-camera triangulation of the candidate kicks in around frame
  7900) was correct and stable throughout at a strong ~0.9 cosine score;
  the bug was in the **2D-only fallback pass**, active during the earlier
  single-camera-only stretch. At frame 7600 the one nearby candidate's
  real projected pixel offset was ~32 degrees off *every* modeled
  direction (medial/lateral/front) -- "medial" won only for being the
  least-wrong of three bad options (cosine 0.85), not for being right. A
  plain score threshold can't catch this: 0.85 looks like a confident
  match, it's simply the wrong one, because a single camera's projected-
  offset geometry is far more sensitive to joint-position triangulation
  noise and imperfect real marker placement than working directly in 3D.
  Fixed by disabling ambiguous-group directional disambiguation in the 2D
  fallback pass by default (`disambiguate_2d_fallback=False`) -- drop
  rather than confidently guess wrong, matching this project's existing
  precedent. Re-measured: purity rises further to 0.814 at
  `min_direction_score=0.4` (vs 0.719 baseline, vs 0.797 with the buggy
  2D fallback active), at a real coverage cost (246 -> 151 ambiguous-
  group tracklets get any name at all in the 10s/1-camera check).
  Re-verified on the actual rendered video: `knee_L` now stays correctly
  unassigned through the previously-wrong stretch and shows `lateral`
  from its first appearance, no transition.

  Harri's follow-up review found two more real, distinct issues, both
  root-caused with real data:
  - **Fusion-stage duplicate-pixel bug** (frame 8679, `knee_L_lateral`
    shown as `ankle_L_medial`): two `FusedPoint`s shared the exact same
    raw pixel in one camera, paired with two *different* candidates in
    two other cameras, producing two distinct 3D positions from one 2D
    dot -- the per-group direction Hungarian had no way to know they
    shared a raw pixel and assigned different names to both. Fixed with
    `_dedup_shared_view_fused_points()`: before assignment, drop the
    lower-quality (higher reprojection error) of any two FusedPoints that
    share a raw (camera, candidate) view. Likely also affects
    `hybrid_assign_frame`'s own unambiguous-group 3D pass (a single
    Hungarian call doesn't prevent two different FusedPoint columns from
    sharing a raw pixel either) -- not fixed there yet, wasn't the
    reported symptom.
  - **Knee/toe often undetected despite a visible dot** -- measured, not
    guessed, over the full 10s/1-camera window: toe's keypoint is visible
    1192/1199 frames but a raw candidate is only *nearby* in 381 of those
    (32%) -- toe's real bottleneck is upstream detection/proximity, not
    assignment (when a candidate is nearby, 93% get assigned). Knee's
    candidate is nearby 1106/1192 times (93%) but only 529 (48%) get
    assigned, split into two distinct causes: 459 frames (78% of the gap)
    have no cross-camera-triangulated candidate near the joint at all --
    the direct, deliberate cost of disabling the 2D single-camera
    fallback for ambiguous groups; 118 frames (20% of the gap) have a
    fused candidate that *was* found and correctly named, but using two
    *other* cameras' pixels -- confirmed the camera under review has its
    own separate raw candidate only 14.8px from its own knee keypoint at
    a sampled instance, almost certainly the same physical dot, that
    never inherits the name because `fuse_frame`'s `views` only records
    the winning triangulation pair, not every camera that can also see
    the point. Not fixed -- flagged as a real, well-understood follow-on
    (back-fill a confirmed name to any other camera's own unconsumed
    candidate that reprojects close to it) pending Harri's go-ahead.

- **2026-09-09** (later still) — Two real bugs found reviewing the
  tracklet-smoothed video with Harri, both now fixed and re-verified
  against the actual rendered frames (not just Python-level data checks,
  which had missed both):
  - **Tracklet-smoothing rewrite pass had a mutual-swap clobber bug**
    (`prototype_tracklet_smoothed_assignment.py`): when two tracklets swap
    names within the *same* frame -- exactly the frame-7920 heel_L/ankle_L
    case -- correcting them one at a time via `dict.pop(old_name)` /
    `dict[new_name] = ...` on the shared per-frame dict let the second
    tracklet's correction clobber the first's, since both fight over the
    same two dict keys (a "swap two variables without a temp" bug).
    Confirmed with real per-frame tracklet votes: tracklet 31 (physical
    ankle_L dot, 392/446 votes) and tracklet 110 (physical heel_L dot)
    swap names for frames 7920-7922; the old rewrite left `heel_L` on
    tracklet 31's position and dropped `ankle_L_medial` entirely -- this
    is why the rendered video still showed the swap even though a
    Python-side dict inspection said it was fixed (the inspection never
    replayed the actual multi-tracklet rewrite order). Fixed by rebuilding
    each frame's assignment dict from scratch (`tracklet_id ->
    majority_name`) in one pass instead of mutating the shared dict
    in place per tracklet. Re-verified against the actual rendered
    frames around 7920: heel_L and ankle_L_medial each now stay pinned to
    the same physical spot throughout.
  - **`render_hybrid_assignment_video.py` drew annotations before the
    grid-cell resize**, then squashed the whole image (non-aspect-
    preserving) down to a fixed cell size -- blurred small circles/text
    away, and on a camera whose crop window's aspect ratio differs a lot
    from the cell's (`oneplus9pro-01`: crop 1298x1772, cell 640x480), the
    anisotropic squash (~3.7x on the tall axis) shrank 5px-radius
    unassigned-candidate circles into near-invisible slivers -- this, not
    an actual absence of detections, is why that camera appeared to have
    none. Confirmed candidates were present all along by reproducing the
    render script's own per-frame DB lookups directly. Fixed by cropping
    then resizing (one uniform scale factor for both axes, aspect ratio
    preserved, letterboxed into the cell) *before* drawing, so every
    circle/label is drawn at its real final-output pixel size. Also
    bumped label/timestamp font scale per Harri's request.
  - Also added multi-camera grid rendering to the same script (one cell
    per `--camera-label` value) so a same-frame issue can be checked
    across every camera's own view at once.

- **2026-09-09** (later still) — Checked Harri's question about
  `oneplus9pro-01`'s crop appearing to drift slightly during a clip
  (hypothesis: phone autofocus). The render script's own crop window is
  computed once and is static in code, so any visible drift has to be
  real. Confirmed with ORB feature matching on the static background
  (furniture, the calibration box) between the start and end of a 10s
  window: a single consistent affine transform fit all 200 matched
  points with zero RANSAC outliers -- a genuine, systematic ~0.2% scale
  change plus ~4px translation over just 10 seconds, not noise. This is a
  real concern for this camera's calibration: if continuous autofocus
  changes effective focal length mid-capture, a single fixed-intrinsics
  calibration for the whole take will accumulate real reprojection error
  over time. Not yet investigated further (whether this is present on
  other phone cameras in the rig, whether it's severe enough to need
  per-segment recalibration, or a capture-setup fix like locking AE/AF) --
  flagging as an open question for the productization plan, not fixed
  here.

- **2026-09-09** — Person-marker-assignment-design.md's phases P1-P7
  prototyped and validated (real video review each step, not just
  aggregate stats) against the real person-marker capture:
  - **P1**: one-to-one (Hungarian) assignment of the full 16-marker leg
    catalog against pose-keypoint anchors, single camera. Real limitation
    quantified: full coverage of a multi-marker joint (3 at a knee, 2 at
    an ankle) from one camera alone is rare (0-25% of frames).
  - **P7** (promoted ahead of P2 per discussion): multi-camera
    triangulation-consistency fusion (exactly the "epipolar-consistency
    correspondence matching" `calibrate_rigid_marker_body.py`'s own
    docstring names and defers). A pure 3D-primary assignment genuinely
    improved multi-marker-joint coverage but collapsed single-marker-slot
    coverage (hip: 53% -> under 2% of frames) -- requiring *both* anchor
    keypoint and candidate to independently reach 2+-camera agreement is
    much stricter than either alone. Fixed with a **hybrid**: 3D-primary
    pass where possible, per-camera 2D fallback for the leftovers.
  - **Real, diagnosed bug** (not fixed by the hybrid alone): `heel_L` and
    `ankle_L`'s anchor keypoints came within 2.6cm of each other for a few
    frames -- close enough, given the bootstrap-stage 15cm match radius,
    that they briefly competed for the same candidates and the per-frame
    Hungarian solver flipped which name got which candidate for 3 frames,
    then flipped back. Root-caused with real per-frame anchor/fusion data,
    not guessed -- the real vitpose keypoints were rock-stable throughout;
    the instability was purely in per-frame-independent triangulation
    competing for a shared candidate pool.
  - **Tracklet-majority-vote smoothing** (prototype_tracklet_smoothed_
    assignment.py): using `DotCandidateWriter`'s own already-computed
    per-camera tracklet_id (previously ignored by every assignment
    prototype this session, despite being stored on every candidate) to
    overwrite a tracklet's minority-frame name flips with its majority
    name across the whole tracklet. Fixed the diagnosed heel_L/ankle_L
    case correctly: the wrongly-swapped frames now go unlabeled rather
    than mislabeled ("drop, don't guess", R2.3) rather than either the
    original wrong label or a guessed-correct one. ~73% of corrections
    landed in the expected ambiguous (same-joint multi-marker) groups.
  - Still open, by design, not yet built: marker-normal-based
    disambiguation for same-joint markers (medial/lateral/front all
    currently collapse to one anchor with no way to tell them apart) --
    next up per discussion, needs the marker layout catalog extended with
    rough per-marker geometric offset/normal info first (not every marker
    sits exactly at its parent joint -- e.g. forearm-twist or torso
    markers), which needs its own design pass before implementation.

- **2026-09-08** (later still, after the full re-run) — Full standalone
  detection re-run on the person-worn-marker capture with the validated
  `background_mode='blacklist'` + per-camera thresholds + `run_parallel()`:
  `detection_run_id a6431d58...` (old, `subtract` mode) superseded by
  `01c2e3c1...`, same 69,486 frames/6 cameras, well under an hour instead
  of overnight. Average raw candidates/frame dropped sharply on every
  camera (e.g. `oneplus9pro-01` 125.2 -> 1.1, `gopro13_01` 77.6 -> 5.9,
  `gopro-11_mini_01` 33.2 -> 11.1) -- consistent with, though not a
  substitute for, the ground-truth-measured precision gain, since this is
  a raw-count comparison on fresh full-scale data, not a re-measured
  precision figure on it specifically.

  Re-checking `prototype_marker_person_filter.py`'s own person-marker
  filter against this fresh data surfaced a second real bug in the
  *filter*, not the detector: it gave each named leg keypoint its own
  independent nearest-candidate search, with no check for whether another
  keypoint had already claimed the same candidate. Quantified at scale
  (386 sampled frames, one camera): **97.3% of frames with any match at
  all had at least one candidate claimed by 2+ different keypoint names
  simultaneously**, most commonly 2-4 -- this is the mechanism behind the
  single-frame "false positive at every toe" Harri flagged earlier,
  generalized. Fixed by replacing the independent per-keypoint search with
  a real one-to-one match (`_assign_keypoints_to_dots`, scipy
  `linear_sum_assignment` over the whole frame's keypoints x candidates
  cost matrix, infeasible/too-far pairs cost-gated to never be proposed) --
  a multi-claim is structurally impossible once assignment is solved
  jointly rather than per-keypoint. Confirmed on the same sample: 0
  multi-claims (by construction) once assigned jointly, average 3.88 of 10
  leg keypoints matched per frame on this camera. Still deliberately crude
  beyond that fix -- doesn't yet use the marker layout's own known
  multiplicity (3 markers at the knee, 2 at the ankle; this only considers
  one candidate per named keypoint) -- a real limitation, not this pass's
  scope.

- **2026-09-08** (later still) — Added `MarkerDetectionPipeline.run_parallel()`:
  camera-level parallelism via `ProcessPoolExecutor`, one process per camera,
  per the productization plan's own "profile before parallelizing, cameras
  are the natural first axis" guidance (§2). Profiled first, on a real
  camera: ArUco detection (~35ms/frame) is actually the single biggest
  per-frame cost, ahead of video decode (~26ms/frame) and dot detection
  (~13ms/frame) -- both genuinely per-camera-independent CPU work, exactly
  what this parallelizes. Frame-chunk parallelism *within* one camera (the
  plan's harder, second axis -- `DotTrackletLinker` is inherently
  sequential) was deliberately not built this pass: with 6 cameras on a
  24-core machine there's real headroom left on the table, but building
  the more invasive axis speculatively, before measuring whether the
  simpler one is actually insufficient, isn't worth it yet.

  Real-data validation (not just the synthetic pipeline tests): an 8s/6-
  camera window that took 456.5s sequentially took 163.0s with
  `run_parallel()` -- a real 2.8x, not the full 6x camera-count ceiling,
  most likely disk I/O contention decoding six separate 4K files
  concurrently off one drive (a bottleneck already documented elsewhere in
  this project) rather than CPU contention. Confirmed identical output
  between the two runs (`frames_processed`, `detection_keypoints` row
  counts for both region types all matched exactly).

  Two real trade-offs against `run()`, both because a worker process can't
  share the pipeline instance's live state: no live cancellation
  (`stop_event` doesn't cross a process boundary) and coarser, per-camera-
  not-per-frame progress reporting. Requires a file-backed session (each
  worker opens its own connection to the same file; WAL mode -- already
  set once by `create_session`, persisted in the file itself -- plus a
  per-connection `busy_timeout` lets concurrent writers retry briefly on
  lock contention instead of failing immediately).

- **2026-09-08** (later) — Ported `background_mode='blacklist'` into
  production (`dot_blob_detector.detect_blobs()`, `MarkerDetectionPipeline`,
  `run_standalone_marker_detection.py`), following up the same day's earlier
  entry's ground-truth-driven diagnosis that background subtraction is
  structurally the wrong tool for a person-worn marker on a subject in an
  atypical pose. Threshold raw brightness directly (shape classification
  never sees more than a marker's own local contour) and use the
  background only to veto a spot that's already nearly as bright with no
  subject present -- the thing background subtraction was actually trying
  to suppress. `'subtract'` stays the unchanged default; `'blacklist'` is
  opt-in. Also added `dot_threshold_by_camera` (camera_instance_id ->
  threshold, overriding `dot_threshold` for that camera): two cameras on
  the same rig can cap a real marker's peak brightness at very different
  absolute levels through their own sensor/tone-mapping (confirmed: one
  camera's real markers saturate at 251-254, another's -- running local
  tone-mapping that compresses highlights -- cap out at 194-232), and
  picking one global threshold against the dimmer camera's floor lets real
  markers on the brighter camera start fusing with nearby moderately-
  bright skin/fabric into non-round blobs, the same fusion failure as
  `'subtract'` mode just triggered by too low a threshold rather than
  subtraction. Validated end-to-end against the same hand-labeled ground
  truth: recall 65.1% -> 80.7%, precision 50.0% -> 75.3% (per-camera
  threshold, `'blacklist'` mode) -- both metrics up together on the real
  capture that motivated this, not a recall/precision trade-off.

  New tests at both layers: `dot_blob_detector.py`'s own unit tests (a
  synthetic fused-blob case 'subtract' loses and 'blacklist' recovers; a
  fixed-glare-source case 'blacklist' correctly still vetoes) and
  `MarkerDetectionPipeline`-level tests confirming `background_mode` and
  `dot_threshold_by_camera` are actually threaded through the real pipeline
  wiring, not just correct in the underlying function.

- **2026-09-08** — Ran standalone (no `capture_objects`/`marker_body_definitions`
  registered yet) ArUco + dot detection overnight on a second real capture
  (`nelli-defaults` trial: person-worn leg markers -- hips, knees, ankles,
  heels, big toes -- plus ArUco on both props and persons; detection_run
  `a6431d58-...`, 69,486 frames across 6 cameras). ArUco yield: ids
  `0`/`1`/`2`/`3`/`17`/`37` solidly seen (1.3-27.7% of frames); `16`/`34`
  vanishingly rare (<0.05%), consistent with the sparse full-trial id scan
  done before launching the run.

  Prototyped (`python/tools/prototype_marker_person_filter.py`, scratch
  tool, not production) filtering the huge raw dot-candidate pool (20-125
  candidates/frame/camera, almost all background noise -- no
  `capture_objects` exist yet to assign against) down to one specific
  person's own markers, using that person's *already-finalized* vitpose
  pose sequence as a spatial cue instead of segmentation: nearest raw
  candidate within a radius of each leg keypoint (hip/knee/ankle/big-toe/
  heel -- HALPE/COCO-WholeBody already carries foot keypoints at the same
  locations two of these physical markers sit). Confirmed on real footage:
  clean matches land within 4-30px of genuinely visible sewn-in reflective
  markers on a "good" camera; the same technique on a camera with a
  malfunctioning ring light instead surfaced background-wide false-positive
  noise, and one sun-glared camera (`oneplus9pro-01`, 290 candidates/frame)
  produced zero matches -- both camera-quality problems, not a filtering
  logic problem.

  **Real detector bug found and fixed**: a correctly round, visibly genuine
  marker (compactness 0.78) was rejected by `max_saturation`'s chroma check
  at a mean saturation of 46.2 against a 45.0 cutoff -- just over. Root
  cause: a small candidate's filled contour mask includes its anti-aliased/
  chroma-subsampled boundary pixels, which blend with whatever sits
  directly behind it; on this capture's brightly patterned red/orange
  leggings (unlike the sword capture's backdrop) that pulls the mean up
  regardless of the marker's own true near-neutral color. Fixed in
  `dot_blob_detector.detect_blobs()` by eroding the contour mask one pixel
  before averaging saturation (falling back to the un-eroded mask if
  erosion empties it) -- confirmed on the same real frame: the miss's mean
  dropped 46.2 -> 28.0 (now passes) while a genuine same-frame false
  positive (actual fabric-pattern texture) stayed rejected, 116.6 -> 105.1.
  New regression test added (`test_detect_blobs_max_saturation_survives_a_
  saturated_backdrops_edge_bleed`); real-footage before/after check on a
  short window showed the fix recovers a small, real handful of near-
  keypoint misses (35/60 -> matched at 80px after the fix, up from 30/60)
  without measurably raising the false-candidate rate (50.8 -> 51.7
  candidates/frame average) -- targeted, not a blanket loosening.

  Most of the *remaining* gap after the fix is not a detector bug: hip
  markers went unmatched on every checked frame from one camera with
  steadily growing "nearest candidate" distance, and a directly-inspected
  missed ankle marker turned out to render as a plain dark dot, not a
  bright highlight, in that frame -- both consistent with the already-
  documented (2026-09-06 entry below) angle-dependent retroreflection
  limitation: a marker only throws a strong return toward a camera near
  its own light source's axis, so per-camera misses on off-axis views are
  expected and need multi-camera coverage to resolve, not a per-camera
  detector fix.

- **2026-09-06** (later still) — Wrote
  [marker-mocap-productization-plan.md](marker-mocap-productization-plan.md):
  the merge-to-`main` bar (schema validated for multi-object/multi-person,
  CLI-complete + GUI-minimal, detection performance in the few-minutes
  range) plus a phase-3 (person+prop) design and an object-initialization
  design (ArUco-anchored init stays primary; Cutie segmentation — reusing
  the existing person-init infrastructure via manual click-seeding, no new
  segmentation code needed — as a secondary init signal and as a new
  dot-assignment cost-relaxation cue, mirroring `dot_tracklet_gate_multiplier`).
  Grounded in a direct code read (not just design docs): `ObjectCropGridWidget`
  only ever loads the `'markers'` (ArUco) source, never `'dots'`; no GUI
  anywhere authors a `marker_body_definitions` row; `PersonSpec`/
  `cross_person_*` look subject-kind-agnostic already (promising for phase
  3) but have never actually been run with an object in the mix.

- **2026-09-06** — Dot detector improvements (background subtraction +
  chroma filter + a real shape-classification bug fix) prototyped, iterated
  with Harri against real footage, then integrated into production and
  validated end-to-end on the real sword capture -- **91.4% tracked
  (6057/6626)** vs. the prior best baseline's 83.5% (5526/6618), a real
  +7.9pp production improvement, not just a promising-looking prototype.

  **Investigation** (`python/tools/prototype_streak_detector.py`, never
  touching production code until validated): continuing the 2026-09-05
  finding that ArUco *and* dot detection both go completely dark for
  ~400ms during the fastest part of a swing, checked whether a lower
  brightness threshold could recover a dimmed streak -- it floods with the
  room's own wall-texture noise long before any real signal emerges
  cleanly. Background subtraction (per-camera median frame, subtract,
  threshold the residual) works instead: it surfaces real streak signal a
  fixed threshold misses, but also the swinger's own moving skin, a false
  positive a **chroma (saturation) filter** cleanly rejects (dots are
  white/near-neutral; skin isn't). A *third*, different false-positive
  class survived both of those -- static residual leaks (light-fixture
  flicker, high-contrast edges the background model doesn't fully cancel)
  that pass every shape and chroma check but never move, unlike a real dot
  on a swinging blade -- caught by a **tracklet/motion-consistency filter**
  (nearest-neighbor linking + a minimum total displacement over a
  minimum track length).

  Two follow-up ideas Harri proposed were tested and came back **negative**,
  worth recording so they aren't retried blind: **local contrast** (a
  candidate's own brightness against a surrounding ring, on the
  un-subtracted frame) doesn't discriminate real dots from the static-noise
  class -- a light fixture is a genuinely strong local peak too, static or
  not; only motion tells them apart. **Digital downsampling** (tried after
  noticing one lower-resolution camera, pixel9, uniquely caught a streak
  during the worst window) doesn't help either -- a resize just re-averages
  already-dim captured pixels, it can't recover sensor sensitivity lost at
  capture time. The likely real explanation for pixel9's own success:
  retroreflective material throws its return preferentially back toward
  the light source, so whichever camera sits closest to the illuminator's
  own axis at a given blade orientation gets the strong hit -- angle-
  dependent, not resolution-dependent, and not fixable in software (worth
  remembering for future rig/light placement).

  **Production integration (Phase A, built and validated this entry)**:
  `dot_blob_detector.detect_blobs()` now shape-classifies (round vs. streak
  vs. reject) *before* any area-based rejection -- the real bug: raw pixel
  `contourArea` was checked unconditionally first, so a legitimately
  dot-width streak whose area exceeds the round-dot-sized `max_area=400`
  (area scales with length, not just width) was discarded before its shape
  was ever considered, for round candidates too, not just streaks.
  `max_streak_length_px` raised from an unvalidated 40px guess to a more
  modest, evidence-based 60px (the only *confirmed* real streak found was
  ~20-22px; several longer-looking candidates turned out to be skin or
  noise on inspection, so this stays deliberately conservative). Background
  subtraction (`background=`) and the chroma filter (`bgr=`/
  `max_saturation=`) are new opt-in parameters on `detect_blobs()` itself
  (`background=None`/`max_saturation=255.0` reproduce prior behavior
  exactly) and wired into `MarkerDetectionPipeline` as
  `dot_bg_subtract`/`dot_threshold`/`dot_max_saturation`/
  `dot_bg_sample_count`, off by default so every existing/other capture is
  unaffected until validated more broadly -- recorded into
  `detection_runs.config_json["dot_detection"]` when used, the same
  precedent already used for the ArUco side. No wire-format change in this
  phase -- `BlobCandidate`'s existing 8 fields already cover it.

  **Real-data validation**: fresh detection run
  (`471dfa17-06a5-46e4-b71c-4516b6b4cf63`, ~49 minutes wall time for 6
  cameras/47,729 frames -- background subtraction roughly doubles decode
  for the 5 dot-enabled cameras, a real, accepted cost for an opt-in
  feature) with `dot_bg_subtract=True, dot_threshold=60,
  dot_max_saturation=45.0`, finalised to sequence
  `72830564-8489-495c-81ec-93850e655ab4`, tracked with the same
  streak-velocity + `outlier_threshold=20` config that produced the prior
  83.5% baseline (`751aa5e3-...`) -- new run `648c76e0-0943-4572-bbf5-
  48025b8ae8d2`, **91.4% (6057/6626)**. Sanity-checked before trusting it,
  same two-sided check as every other tuning change this session: whole-run
  reprojection error stayed essentially flat (median 7.95px -> 8.96px, p90
  19.3px -> 21.0px, p99 and max both slightly *better*) despite inliers
  growing 38,249 -> 46,107 (+21%) -- the outlier count also grew (39 ->
  1,328) but proportionally to a much larger candidate pool (dot detection
  now runs on 5 cameras instead of the original 2), not a quality
  regression: the UKF's own outlier rejection scaling up to filter a bigger
  pool is the mechanism working correctly, not contamination.

  **Phase B (designed, not built this round)**: tracklet construction +
  tracklet-aware assignment gating, so a candidate that's part of the same
  moving tracklet as what resolved into a given (subject, camera, marker)
  slot *last* frame can pass `resolve_dot_assignment()`'s gate more easily
  than a genuinely new, unrelated candidate. Key design points, resolved
  during planning: (1) the tracklet linker does **not** need to be causal --
  this whole detection pipeline is an offline batch pass over already-
  recorded video with the full frame sequence available before the tracker
  ever runs, so it can look ahead across a camera's whole footage exactly
  like `build_tracklets()` already does in the prototype, and exactly like
  this project's own RTS smoother already does on the tracking side; (2)
  the concrete gate-relaxation mechanism, confirmed against the real
  `dot_assignment.cpp` cost-matrix code: rather than changing
  `solve_assignment()`'s single global `gate_mahalanobis` scalar (a bigger,
  more invasive change), divide a same-tracklet pair's own `mahal_sq` by a
  new `dot_tracklet_gate_multiplier` before it goes into the cost matrix --
  makes the *existing* gate implicitly looser for that one pairing only,
  no `assignment.hpp` changes needed; (3) needs a 4th wire-format bump
  (`float32[N,8]` -> `float32[N,9]`, adding `tracklet_id`) following the
  established fail-loud-and-rerun convention, plus a new persistent
  `prev_dot_tracklet_ids_` map in `Tracker` parallel to the existing
  `prev_observations_`; (4) found in passing and worth fixing alongside it:
  `dot_assignment_gate_mahalanobis` itself currently has **no DB wiring at
  all** (TOML-only), so it isn't actually settable from a production
  (DB-driven) `tracker_config` row today.

- **2026-09-06** (later same day) — Phase B (tracklet-aware assignment gate
  relaxation) built and validated end-to-end on the real sword capture.
  Confirms the pattern from the original dot-detector prototype work: raw
  detection recall alone wasn't the bottleneck once Phase A landed --
  reviewing Phase A's own production tracking run showed "very few samples
  accepted as inliers, especially during the interesting motions," matching
  exactly what the prototype saw before tracklets were added there too.

  **C++ side**: bumped the dot-candidate blob wire format to `float32[N,9]`
  (adding `tracklet_id`, produced by the Python-side `DotTrackletLinker`,
  Phase B's earlier Python-only commit); `resolve_dot_assignment()` now
  divides a candidate's squared-Mahalanobis assignment cost by
  `dot_tracklet_gate_multiplier` when its `tracklet_id` matches the one that
  resolved into the same `(subject, camera, marker)` slot on the previous
  frame (`Tracker::prev_dot_tracklet_ids()`, parallel to the existing
  `prev_observations_`). `dot_assignment_gate_mahalanobis` (previously
  TOML-only) and the new multiplier both got real DB columns this pass
  (session schema v51->v52, registry v10->v11) -- the gap flagged in the
  design entry above.

  **Real-data validation**: fresh detection run on the sword capture (same
  bg-subtract/chroma settings as the Phase A validation, tracklet linker now
  live) -- `detection_run_id eae4abcd-2f0b-4990-a9aa-7cae46c6934b`,
  finalised to sequence `ac055305-36ba-4da4-b4ce-33f9ee26982e`. Ran that
  *same* sequence through tracking four times, varying only
  `dot_tracklet_gate_multiplier` (cloned from the Phase A baseline config
  `75299631-...`, `outlier_threshold=20` + streak velocity) -- confirming,
  per Harri's question, that the multiplier can be swept without re-running
  detection at all: `tracklet_id` lives in the detection-time blob,
  the multiplier is purely a tracking-time `tracker_configs` field.

  | `dot_tracklet_gate_multiplier` | tracked | dot median/p90/p99/max reproj. error |
  |---|---|---|
  | 1.0 (no relaxation, control) | 90.7% (6002/6617) | 8.97 / 20.64 / 36.52 / 144px |
  | 2.0 | 93.3% (6172/6617) | 10.64 / 22.67 / 36.97 / 183px |
  | **4.0** | **93.5% (6189/6617)** | 11.95 / 24.90 / 43.31 / 205px |
  | 8.0 | 93.1% (6158/6617) | 12.07 / 25.24 / 45.97 / 204px |

  4.0 is the best of this coarse sweep (8.0 is very slightly worse, so
  higher isn't unconditionally better -- consistent with "relax only what
  has independent identity evidence," not "widen the gate more"). The
  control run's 90.7% closely reproduces Phase A's own 91.4% on the
  original detection run, confirming the fresh run behaves consistently.
  The real cost of the relaxation is visible in the error columns above --
  more marginal correspondences now survive assignment, so the whole
  distribution's error is measurably higher, not just its tail. Per this
  project's own "never trust tracked% alone" precedent (an earlier gate
  change made a known failure window worse while looking like a win),
  checked real frames before accepting the number: extracted stills on
  `gopro-11_mini_02` at the known 63.6-64.5s and 69.5-70.2s fast-swing
  windows for the control vs. `mult=4.0` runs. At 69.6s the control
  resolves *zero* dots (predicted markers just drift, candidates sit
  unmatched nearby); `mult=4.0` correctly locks dot3/dot4/dot5/dot6 onto
  the real, visible markers. At 64.2s, mid-swing, `mult=4.0` correctly
  resolves `dot6` right at the moving blade tip -- the exact case this
  feature was built for -- while the control only resolves the
  slower-moving markers near the grip. No wrong-looking correspondence
  found in any of the six checked frames. **Recommendation: `4.0`** for
  this capture; not yet swept finer or validated on a second capture.

- **2026-09-05** (even later still) — Multi-camera grid-video review of the
  sword capture (see the 2026-09-05 "later still" entry below for the
  render tool itself) surfaced two real findings, both from Harri's own
  frame-by-frame viewing, not a metric:

  **Capture design limit, not a tracking bug.** In the known gap window,
  the predicted pose drifts specifically in the direction roughly
  orthogonal to `gopro-11_mini_02` -- and that's also the direction none
  of this capture's markers are visible from any camera at all (every
  marker sits on one face of the blade). No amount of filter tuning
  recovers information no camera captured. Practical conclusion for
  future captures: a marker body needs markers/tags visible from more
  than one side (e.g. the sword's front face too), not just opposite
  edges of the same face.

  **Real detections lost at the dot-assignment gate during fast swings,
  not at the UKF's outlier check.** Checked directly: of 14,212 dot
  observations that made it into `tracking_obs_results` for the
  streak-velocity run, zero were ever flagged outlier by
  `UnscentedKalmanFilter::reject_outliers()`. The loss Harri saw (real,
  visibly-streaked dots in gopro-11_mini_02/insta_ace2_pro/oneplus9pro-01/
  pixel9 producing zero accepted detections during the 70-71s cut) happens
  one stage earlier, in `resolve_dot_assignment()`'s Hungarian-solver gate,
  which only pairs a candidate with a predicted marker if their squared
  Mahalanobis distance (using the *predicted* marker covariance) is within
  `dot_assignment_gate_mahalanobis` (9.21). That covariance is supposed to
  grow with the root's own current velocity (adaptive process noise,
  Mechanism A -- already active here at gain_root=4/ref_root=2), but the
  variance-domain multiplier was hard-capped at a fixed
  `kMaxVelocityNoiseMultiplier = 10.0` UKF constant. At this tuning, any
  root DOF's velocity above ~1.08 rad/s already saturates that cap.
  Checked against real state data: the known 63.6-64.3s regression window
  sits at a mean 1.77 rad/s and is **100% saturated** for its entire
  duration, while the (already near-perfect) 53.6-55.1s ArUco gap window
  never saturates at all (max 1.06 rad/s) -- strong correlation between
  "known-bad window" and "noise-scaling pinned at its ceiling."

  Deliberately did *not* reach for widening `dot_assignment_gate_mahalanobis`
  itself -- this project already has a direct precedent for that making
  things worse (see the "Constant-velocity lag during fast cuts" entry
  below: widening the search gate before fixing correspondence let
  wrong-face candidates get matched more easily), and baseline already has
  one confirmed bad ~934px match in this exact regression window
  attributed to the existing adaptive-noise permissiveness already being
  too loose at times. Instead, exposed the previously-hardcoded cap as a
  new tunable, `process_noise_vel_max_multiplier` (config field + new
  `UnscentedKalmanFilter::set_velocity_noise_max_multiplier()`, default
  10.0 so every existing config is unaffected; session schema v51/registry
  v10).

  **That first experiment (raised cap) didn't work -- and the per-step data
  shows exactly why.** Raising the cap to 50 barely moved the aggregate
  number (80.1% vs baseline 80.5%) and, checked frame-by-frame against
  Harri's own named 70-71s window, changed *nothing* about the actual
  symptom: in both the old and raised-cap runs, root angular velocity
  freezes at a constant value the instant dot resolution hits zero, and
  stays frozen (identical value, to 2 decimals) for the next 700+ ms while
  zero dots resolve in either run. The velocity-scaling formula depends on
  the *current velocity estimate* -- but once no observation is being
  accepted, there's nothing left to correct that estimate, so it stops
  evolving entirely regardless of how high the cap goes. Raising the
  ceiling on a covariance that's supposed to widen around a moving mean
  does nothing once the mean itself has stopped moving.

  Checking *why* even ArUco corners (deterministic per-tag correspondence,
  not Hungarian-matched like dots, so none of the wrong-face risk applies)
  were producing zero accepted detections in this same window found the
  real mechanism: real, correctly-detected corners *were* present every
  step, sitting at real, correct pixel positions, but rejected by
  `UnscentedKalmanFilter::reject_outliers()` at Mahalanobis-squared values
  of 10-19 against the effective `outlier_threshold` of 5.991 (this
  config's `tracker_configs` row left it NULL, so it fell through to
  `TrackerConfig`'s own struct default rather than the separate
  `TrackerAppConfig` default of 4.0 used by the TOML/CLI path -- worth
  remembering that these two structs' defaults for the same-named field
  differ). Reprojection error at those same instants was 400-1100+ px --
  the *predicted mean* had diverged hugely (the frozen-velocity coast
  above), not the detection. No amount of covariance widening fixes a
  wrong mean; it only widens the ellipse drawn around it.

  Raising `outlier_threshold` to 20 (a real config field already wired
  through `session_reader.cpp`, no code change needed) let those large-but-
  genuine innovations back in as inliers, snapping the mean back instead of
  waiting for it to organically re-align. Validated on the same
  sequence/skeleton: baseline 80.5% -> **84.2%** tracked (5325 -> 5574/6618),
  streak-velocity 79.1% -> **83.5%** (5234 -> 5526/6618) -- the
  streak-velocity/baseline gap (the still-unexplained net regression from
  the 2026-09-05 entries above) shrinks from -91 steps to -48, so a real
  chunk of that regression turns out to be this same outlier-rejection
  deadlock, not something specific to streak velocity. Sanity-checked
  before trusting it: whole-run median/p90 reprojection error barely moved
  (7.70px -> 8.14px median, 18.28 -> 19.85 p90), and a scan for sustained
  (>=5 consecutive steps, >30px) bad dot matches anywhere in the run found
  none of concern -- this isn't "the gate stopped filtering," it's
  specifically recovering genuine detections that a too-tight post-hoc
  check was discarding during real, temporary divergence. A small sweep
  (15/20/30, and 30+the raised cap combined) found 20 already captures
  essentially all of the gain (20 and 30 are statistically indistinguishable
  here) with a smaller cut into the outlier-rejection safety margin, and
  combining with the raised cap added nothing (if anything, very slightly
  worse: 83.4% vs 84.2%) -- so the cap change, while real and now available,
  isn't the fix for *this* failure mode. Re-rendered the regression and
  70-71s grid videos against the fixed streak-velocity run
  (`751aa5e3-2bb7-4cce-af93-5783cc2030db`) alongside the original
  (pre-fix) ones already sent, plus the remaining 73.8-100.5s of the
  full-sequence request on the same improved run. See
  `LOCAL-TEST-DATA-NOTES.md` for every ID and file name.

  Open risk, not yet investigated: `outlier_threshold=20` is a single
  global scalar applied identically to ArUco corners (safe to trust even
  at huge Mahalanobis distance, since their correspondence is tag-ID-based,
  not nearest-neighbor) and dots (Hungarian-matched, so a wrong-but-nearest
  assignment *can* now survive this check almost unchallenged -- only 43
  outliers survived out of ~38,000 dot+ArUco observations at threshold 30,
  down from 3,982 at the default). The whole-run sanity checks above found
  no evidence of this happening on this particular capture, but a
  marker-type-specific threshold (looser for ArUco, tighter for dots)
  would close this gap properly rather than relying on this capture
  happening not to trigger it.

- **2026-09-05** (later still) — A real bug in the streak-velocity
  regression videos, caught by Harri's own review ("actual" dot markers
  floating over blank wall, nowhere near any possible detection): fixed a
  `tracking_obs_results` diagnostic-write collision that streak velocity's
  dual-Observation-per-marker design exposed, plus two related `ukf.cpp`
  bugs the same root assumption ("at most one Observation per (camera,
  marker) per step") had caused, one of which could affect real outlier
  decisions, not just diagnostics. Full account:
  [observation-results-semantics.md](../observation-results-semantics.md).
  Re-tracked both configs fresh -- identical numeric results to before
  the fix -- and re-scanned both runs' videos with corrected diagnostics:
  streak velocity's own "hundreds of pixels" is now fully explained and
  gone; a *separate*, real ~934px bad match remains in the baseline run's
  regression window, unrelated to streak velocity, most likely the
  adaptive-root-noise tuning's gate becoming too permissive during high
  uncertainty (not investigated further). See
  [streak-velocity-design.md](streak-velocity-design.md) §4 and
  `LOCAL-TEST-DATA-NOTES.md`'s 2026-09-05 entries for real IDs and
  numbers.

  Harri then correctly refused to accept that fix on the numbers alone --
  the re-rendered videos still showed the identical symptom. Right call:
  a *second, separate, pre-existing* bug in `render_tracking_debug_frames.py`
  itself (not the mode-mixing one), only found by actually extracting and
  viewing PNG frames rather than trusting a DB query: RTS-smoothed
  `tracking_results` rows use a different `tracker_step` numbering than
  raw ones, and the tool's nearest-timestamp lookup didn't filter
  `is_smoothed`, so it could return a smoothed row's step number and reuse
  it to key into `tracking_obs_results` (raw-indexed only) -- silently
  fetching a real, valid, *different* instant's data onto the wrong video
  frame. One-line fix (`AND is_smoothed = 0`); added the module's first
  test coverage. Re-rendered all four videos a third time, this time
  visually confirmed correct.

- **2026-09-05** (later same day) — Streak-velocity phase 3 (tracker
  integration) built and wired into the real per-frame loop -- see
  [streak-velocity-design.md](streak-velocity-design.md) §4 for the full
  account. First real-data validation was a genuine, diagnosed **net
  regression**: 79.1% tracked vs. the 80.5% baseline, despite the mechanism
  recovering 77 previously-lost steps (it also newly lost 168 others). Most
  likely cause: `Tracker::prev_observations_` isn't filtered by the UKF's
  own outlier verdict, unlike the offline validation script that first
  confirmed the mechanism's real signal -- a bad match during fast,
  ambiguous motion can contaminate the next frame's streak-velocity
  reference position. Not yet fixed; not yet confirmed as *the* cause
  either, just the most likely one given the code. `dot_streak_velocity_enabled`
  defaults to false, so this doesn't affect any existing run.

- **2026-09-05** — Streak-velocity design (still-visible constant-velocity
  lag during the fastest sword cuts, see the entry below): phases 1-2 done
  and validated against real data, phase 3 (tracker integration) not
  started -- see
  [streak-velocity-design.md](streak-velocity-design.md) for the full
  mechanism. `dot_blob_detector.py` now reports a canonicalized streak
  axis (`dir_x`/`dir_y`) alongside its existing length; the 'dots' wire
  format bumped again to carry it (`float32[N,6]` -> `float32[N,8]`, same
  versioned-count-prefix scheme as the previous bump). A new standalone
  script, `estimate_streak_exposure_ratio.py`, estimates `k =
  exposure_time/frame_time` per camera as a ratio of sums over real,
  resolved frame-to-frame displacements (not a mean of per-sample ratios,
  which is unstable near zero displacement) -- `k` is provably
  speed-independent (the object's own velocity cancels out of the ratio),
  so a real per-camera constant is exactly what a stable estimate across
  different motion speeds should look like. Re-ran detection + tracking
  on the real sword capture to get real streak data (same 80.5% tracked
  result as before, confirming the format bump changed nothing else):
  `gopro-11_mini_02` gave 195 samples, `k=0.65` (stable across a
  first/second-half split), direction cosine 0.99 against real
  displacement -- a real, positive validation that both streak length and
  direction track real motion well enough to be worth building the
  tracker-integrated version. Full IDs and numbers in
  `LOCAL-TEST-DATA-NOTES.md`'s 2026-09-05 entry.

- **2026-09-05** — Real visual review of the corrected-calibration
  tracking video (previous entry) surfaced two further, distinct
  problems, both now fixed and validated together:

  **Cross-face dot confusion.** The video showed a real dot mislabeled
  as a different, wrong-face dot for a stretch (~60-80px error
  'converging' onto a self-consistent but wrong fit) -- a flat prop with
  markers on both faces means a candidate near a *near-face* marker's
  prediction can be mistakenly assigned to a *far-face* marker's slot
  instead, since nothing before now excluded a physically-impossible
  (facing away from the camera) marker from the assignment pool at all.
  Fixed with a new `Marker::normal` (`cpp/include/posetrak/core/
  skeleton.hpp`): `predict_rigid_marker()` now excludes a marker from a
  camera's prediction entirely when its current world-frame normal (the
  object's own orientation applied fresh every call) faces away from
  that camera -- same `std::nullopt` convention already used for
  "behind the camera". An ArUco tag's normal comes free from its own
  corner geometry; a dot has no inherent orientation, so it's either
  inferred (nearest face-plane by signed distance) or, better, declared
  explicitly via a new `same_face_as: <marker name>` marker body YAML
  field. The explicit path turned out necessary, not optional, on the
  real sword body: geometric inference put two real dots on the
  geometrically "nearest" plane, which turned out to be the *wrong*
  face once Harri checked the real footage directly -- not a
  calibration error, just confirmation that a real prop isn't the flat
  two-plane shape inference assumes closely enough to rely on for this.
  Worth remembering for next time: Harri's *first* verbal description of
  which dots share which face was also wrong (corrected once actually
  checking the video) -- this is exactly the kind of fact that needs
  direct physical/video confirmation before use, not memory alone, no
  matter how confident it sounds either time.

  **Constant-velocity lag during fast cuts.** Separately, the tracker
  visibly failed to react fast enough at the start of a fast sword cut,
  losing several real, correctly-identified dots as outliers because
  the constant-velocity process model's predicted uncertainty didn't
  grow fast enough to keep pace with the real acceleration. The fix
  already existed and was already validated on human motion, just
  switched off by default: `TrackerConfig::process_noise_vel_gain_root`/
  `_ref_root` (`adaptive-process-noise-design.md`'s Mechanism A, done
  2026-07-06/07) scales root process noise with the root's own current
  velocity estimate. A first tuning attempt (gain=2, ref=3), tested
  *before* the face-culling fix above, made a known failure window
  *worse* -- confirmed why: widening the prediction's search gate
  before fixing correspondence just gives wrong-face candidates more
  room to be mistakenly matched, trading one failure mode for another.
  Retested after the face-culling fix with a more aggressive tuning
  (gain=4, ref=2): a clean, substantial win, this time with nothing left
  to interact with it badly.

  **Combined, final result**: 80.5% tracked (5325/6618) on the real
  sword capture, and the known 53.6-55.1s ArUco gap window down to only
  17/300 lost steps -- fewer than even the *original, mislabeling-prone*
  run's 53/300, while now actually being correct (reprojection error
  stayed in the same healthy 5.8-10.9px median range throughout, so this
  isn't just looser gating accepting sloppier matches). Visually
  reconfirmed at the exact previously-worst instant: dots cluster
  tightly and correctly on the real, visible object, no scattering.
  Full IDs, config values, and every intermediate (including the
  since-superseded first noise tuning) in `LOCAL-TEST-DATA-NOTES.md`'s
  2026-09-05 entries.

- **2026-09-05** — Fixed the marker-body calibration bug the previous
  entry found (`aruco_2`/`aruco_3` 5.6cm in-plane bias), confirmed both
  numerically and visually. The original multi-camera rig that shot the
  sword performance is gone (two of its six cameras have since broken),
  but the two ArUco tags' shared rigid harness survived, separated from
  the sword -- along with 4 of the sword's 7 reflective dots (the other 3
  were mounted elsewhere on the sword itself). Recalibrated it from a
  single camera slowly orbiting the stationary harness, anchored by two
  extra static ArUco markers placed in view (`calibrate_harness_from_orbit.py`,
  new): every frame gets its own camera-pose solve from whichever anchor
  is visible, turning each into an independent viewpoint onto the harness
  -- no two harness markers need to be seen in the same instant, unlike
  the original method, and a single ~74s orbit produced thousands of
  samples per marker instead of a few dozen.

  Result: `aruco_2`/`aruco_3` in-plane offset 5.64cm -> 0.10cm (through-
  thickness 1.98cm -> 2.93cm) -- matching "opposite faces, same point
  along the blade" exactly, as Harri had described from direct physical
  inspection. The 4 harness dots shifted only 0.2-0.8cm from their old,
  manually-annotated positions (those were computed differently -- a
  fresh reference-pose solve per instant, not an average across many --
  and so were already largely unaffected by the ArUco bug); the 3 dots
  not on the harness were confirmed safe to carry through unchanged
  (every manual annotation session used `--reference-id 2`, so they were
  never exposed to the `aruco_2`/`aruco_3` relative-pose bug in the first
  place -- worth checking explicitly rather than assuming, since the
  wrong assumption either way would have silently introduced or missed a
  real few-cm error).

  Re-ran detection fresh (the prior run's 'dots' data predates the same
  day's later blob-format change) and tracking with the corrected body:
  74.9% tracked (4957/6618), close to the previous (miscalibrated) run's
  73.3-74.8% -- the headline number barely moved, but visual inspection
  via `render_tracking_debug_frames.py` against real frames shows what
  actually changed: the "predicted markers scattered tens-to-100+px into
  empty space" incoherence the miscalibrated run showed is gone, in both
  a normal segment and inside the known 53.6-55.1s ArUco gap. The
  aggregate tracked-fraction alone couldn't distinguish "fitting a
  plausible pose" from "fitting an implausible one" -- this is the
  distinction that matters, and it's now the right one.

  One honest, unresolved caveat: `aruco_2`'s own corner reprojection error
  in the final run (median ~8.5-11.5px) is still somewhat higher than
  `aruco_3`'s (~5.4-7px) in the same run or the original ArUco-only
  baseline's (~5-6px). Not the bug just fixed (numerically and visually
  confirmed separately) and not investigated further -- plausibly
  `aruco_2`'s hard-coded perfect-square template (it defines the body's
  own local origin/orientation by construction, unlike every other
  marker's empirically-fitted geometry) not perfectly matching some real
  imperfection in the physical tag or its mount.

- **2026-09-04** — Opening a marker-based-mocap object's tracking run in
  the GUI turned out to have real, pre-existing gaps once the two dots-
  enabled runs above were actually opened: the crop-preview pipeline
  (`frame_cache_entries`, the wide-crop cache, `CropBackfillWorker`'s
  bbox-aware fast path) is entirely keyed off `person_name` +
  `detection_track_assignments` -- concepts `finalise_object_to_db()`
  never populates (an object has no track-to-person stitching step, see
  the 2026-08-30 entry below) -- so every frame fell through to a
  full-resolution, no-bbox decode every time. Not fixed yet (real feature
  work: needs an object-appropriate crop-window source); documented so
  the next person touching ObjectPanel doesn't have to re-discover it.

  Built `python/tools/render_tracking_debug_frames.py` instead, an
  offline diagnostic (still PNGs or a slow-motion MP4 over a time range)
  overlaying real video frames with actual ArUco/dot detections, the raw
  (including unresolved) dot candidates, and the tracker's own current
  pose estimate -- re-projected via forward kinematics from
  `tracking_results.state` and the real camera calibration, not read from
  `tracking_obs_results.obs_blob` (which only ever carries a slot for a
  marker that had an actual observation that step -- an undetected
  marker's predicted position was invisible until this tool computed it
  independently). Iterated on real user feedback against real footage:
  fixed a genuine coordinate-space bug in the first version (obs_blob's
  actual_x/y are in the tracker's *undistorted* pixel space --
  `Camera::project_undistorted()`, "for UKF" -- while the raw video frame
  is distorted; drawing one directly onto the other put every obs_blob-
  sourced overlay a few pixels off, worse near the frame edges where
  distortion is larger -- confirmed by redistorting a real point and
  getting a bit-identical raw candidate pixel back), switched from a
  hollow ring to a filled center dot + thin ring (a hollow ring's true
  center is hard to judge once the ring's own radius is a real fraction
  of the zoomed view), and switched unassigned raw candidates back to a
  ring (a filled gray dot disappears against a similarly gray/textured
  background).

  This tool is what surfaced the actual calibration problem behind the
  "tracking is badly off" report: `aruco_2`'s own four corners don't fit
  one consistent rigid pose equally well even in an easy frame (two
  corners within ~5px, two off by ~30px); computed directly from the
  calibrated skeleton geometry, `aruco_2` and `aruco_3`'s tag centers are
  5.6cm apart in-plane (only ~2cm through the blade's thickness) when the
  physical tags sit opposite each other at nearly the same point along
  the blade -- confirming Harri's own visual read of the same offset; and
  the debug videos show several markers' predicted positions landing
  tens-to-100+px from the real, visible sword in the middle of the known
  ArUco gap while others (right on a tag) fit fine -- consistent with a
  real dot-to-wrong-slot mismatch (e.g. dot0/dot1, physically on the
  sword's far face and out of view for this camera, resolving instead to
  real dot4/dot5 detections because the miscalibrated body's predicted
  dot0/dot1 positions land closer to them than to nothing). Conclusion,
  stated plainly rather than hedged: this capture's marker-body
  calibration needs to be redone (starting with the `aruco_2`/`aruco_3`
  relative pose, since everything else -- dots calibrated relative to
  whichever tag was visible -- likely inherits its error) before its
  tracking results can be trusted; not yet done.

- **2026-09-04** — Separately, fixed `detect_blobs()`'s
  (dot_blob_detector.py) real structural weakness: its compactness
  filter, needed to reject glare
  streaks, was also rejecting a genuinely blurred dot during fast
  swings -- exactly the moments this feature exists to help with most.
  Now accepts an elongated blob whose *width* still matches a round dot's
  diameter range even when its *length* doesn't, capped by
  `max_streak_length_px` (a first cut, not yet checked against a real
  blurred-dot example). Feeds the streak's length into the resolved
  Observation's noise (`resolve_dot_assignment()`) rather than trusting a
  streaked centroid as tightly as a round dot's -- required a versioned
  wire-format change (`pose_observations`/`detection_keypoints`'s 'dots'
  blob: int32 candidate count + float32[N,6], not the inferred-from-byte-
  length float32[N,4] the format had been since first written) to carry
  the streak's axes through to the C++ side at all; see
  [reflective-dot-detection-design.md](reflective-dot-detection-design.md)'s
  "Open questions" section for the full rationale, the two future ideas
  (velocity-from-streak, blinking-LED sub-frame timing) this also logs,
  and why old 'dots' data (the one real detection run recorded under the
  old format) needs re-running rather than migrating -- moot anyway,
  since that run's tracking needs redoing once the calibration above is
  fixed.

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

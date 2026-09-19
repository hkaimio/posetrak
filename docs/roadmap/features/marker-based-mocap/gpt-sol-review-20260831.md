# Marker-based motion capture code review

  Scope: main...hkaimio/marker-based-mocap at 6b4697a — 47 files, approximately 7,200 added lines. No code changes were made.

  ## Feature summary

  Phase 1 introduces end-to-end tracking of rigid props carrying coded ArUco markers:

  - Marker detection runs with fixed corner-slot blobs.
  - Capture-scoped objects linked to reusable marker-body definitions.
  - Generation of root-only prop skeletons.
  - Finalized object observation sequences with landmark manifests.
  - Object review/edit UI.
  - Manifest-based C++ observation loading.
  - Closed-form Kabsch/Umeyama rigid-body initialization.
  - GUI launch of single-object tracking runs.

  The design ultimately targets anonymous dots, mixed coded/dot bodies, person-attached markers, person–prop coupling, and camera movement recovery.

  ## General verdict

  The phase-1 vertical slice is thoughtfully structured, heavily documented, and has unusually good unit coverage. Reusing ordinary observations, skeletons, tracking runs, and editing infrastructure is the right architectural direction.

  I would not consider it production-ready yet. The main blocker is initialization behavior: sparse marker data commonly causes rigid initialization to fail, after which the CLI silently starts the prop at the world origin. Several interfaces described as foundations for later phases are currently syntax-only or single-source implementations, particularly track
  bindings, marker-specific noise, locked root DOFs, and object provenance.

  No direct security vulnerability was identified. SQL values are parameterized, YAML uses safe loading, and subprocess execution appears argument-based. The main risks are correctness, provenance, extensibility, and performance.

  ## Prioritized findings

  1. High — Failed rigid initialization silently falls back to a meaningless rest pose.
     The CLI only tries observations from the first tracker interval. If fewer than three markers triangulate, the fit is rejected, or the layout is collinear, it initializes the prop at the world origin and continues as if successful (/D:/mocap/posetrak/cpp/cli/track.cpp:470, /D:/mocap/posetrak/cpp/cli/track.cpp:501). For a free-moving prop, unlike a human
     skeleton, rest pose has no useful world-space meaning. Real-data notes already report that a usable frame had to be found manually. The CLI should search a configurable initialization window and fail clearly if no valid rigid pose is found.

  2. High — Symmetry/locked-root-DOF support is emitted but not implemented.
     The generator writes locked_dofs.axis (/D:/mocap/posetrak/python/posetrak/skeleton/marker_body_to_skeleton.py:90), but neither SkeletonLoader, SkeletonLayout, nor the root state consumes it. The test explicitly acknowledges this (/D:/mocap/posetrak/python/tests/skeleton/test_marker_body_to_skeleton.py:207). Consequently, symmetric and collinear props promised
     by R1.4 cannot be initialized or prevented from drifting; collinear layouts are simply rejected (/D:/mocap/posetrak/cpp/src/tracking/tracker.cpp:423).

  3. High — The advertised multi-source input-track model is not implemented.
     load_observations() still accepts exactly one sequence and builds one manifest map. It matches all skeleton landmarks by name without filtering on Marker.track (/D:/mocap/posetrak/cpp/src/db/session_reader.cpp:700, /D:/mocap/posetrak/cpp/src/db/session_reader.cpp:769). There is no invocation-level track_id → sequence_id binding or PersonSpec track map as
     specified. Duplicate landmark names on different tracks can also bind incorrectly. Phase 3 person–prop coupling and later marker-augmented person tracking will require a significant loader/CLI/provenance redesign.

  4. High — Marker-specific measurement noise is absent.
     Marker gained track and landmark, but not the design’s noise_std; loaded marker observations never receive noise_std_override. They therefore use the ordinary run-wide pose/calibration noise model. This defeats the accuracy advantage of ArUco corners and prevents phase 3+ from safely mixing precise dots with noisy bands or cloth-mounted markers (/D:/mocap/
     posetrak/cpp/include/posetrak/core/skeleton.hpp:71, /D:/mocap/posetrak/cpp/src/db/session_reader.cpp:798).

  5. Medium — Object identity is not written into tracking-run provenance.
     The schema adds tracking_run_persons.capture_object_id, but the single-object launcher only invokes the legacy single-sequence tracker, and ResultWriter creates only tracking_runs (/D:/mocap/posetrak/python/app/pose/run_tracker.py:2484, /D:/mocap/posetrak/cpp/src/db/result_writer.cpp:109). No tracking_run_persons object row is inserted. Tracking results cannot
     reliably answer which physical capture object they represent, contrary to the design and future mixed-subject requirements.

  6. Medium — Finalization is hard-coded to ArUco and four-corner landmarks.
     finalise_object_to_db() rejects every detector type except "aruco" and always creates four landmarks per configured marker (/D:/mocap/posetrak/python/app/pose/finalise.py:366, /D:/mocap/posetrak/python/app/pose/finalise.py:399). Phase 2’s blob detector and phase 3 mixed coded/dot bodies cannot pass through this supposedly generic finalized-observation seam
     without refactoring. Detector output should carry or resolve a generic landmark manifest.

  7. Medium — Detection does not meet the design’s per-camera parallelism requirement.
     Cameras are processed serially (/D:/mocap/posetrak/python/posetrak/detection/marker_pipeline.py:165). Additionally, frame_step skips detection only after every intervening frame has already been decoded (/D:/mocap/posetrak/python/posetrak/detection/marker_pipeline.py:294). Full-resolution multi-camera jobs will scale poorly; frame seeking/decimation and bounded
     per-camera workers should be benchmarked.

  8. Medium — Marker edits inherit a confirmed broken outlier-edit path.
     Marking a point as an outlier replaces its coordinates with the edit blob’s placeholder coordinates instead of preserving the original position. The focused suite fails test_edit_marks_outlier_zeroes_confidence. Object review directly reuses this path, so edited marker data and UI display can disagree with the documented semantics.

  9. Medium — Tests stop short of the principal end-to-end acceptance criterion.
     The C++ rigid test validates noiseless single-frame initialization, residual rejection, and collinearity, but not a moving trajectory with noise/dropout, initialization search, tracking RMSE, or loss/reacquisition (/D:/mocap/posetrak/cpp/tests/test_tracker_integration.cpp:724). There are also no tests for two bound input tracks, marker-specific noise, locked
     root DOFs, or persisted object provenance.

  10. Low — Multi-dictionary identity uses bare IDs rather than composite identities.
     The loader rejects the same numeric ID appearing in two different dictionaries because downstream filtering uses the bare ID (/D:/mocap/posetrak/python/app/setup/fiducial_markers.py:706, /D:/mocap/posetrak/python/app/setup/fiducial_markers.py:841). This is safe for phase 1 but unnecessarily limits marker-body definitions and complicates future AprilTag/ArUco
     coexistence. (family/dictionary, id) should be the stable identity.

  ## All findings by category and severity

  ### Correctness

  High

  - Failed rigid initialization falls back to origin/rest pose rather than searching or failing.
  - Symmetry-axis root locking is not implemented; collinear props cannot be tracked.
  - Marker-specific measurement noise is not propagated, producing an inappropriate noise model.

  Medium

  - Outlier-only edits overwrite coordinates; one committed test currently fails.
  - Rigid initialization uses an unweighted, non-robust fit. One bad triangulated marker rejects the entire frame rather than fitting a consensus subset.
  - The finalization path silently skips detection rows whose video or sync mapping is missing instead of reporting incomplete camera coverage (/D:/mocap/posetrak/python/app/pose/finalise.py:493).

  ### Architecture and future compatibility

  High

  - Input-track declarations are not backed by invocation-time track bindings or multi-sequence loading.
  - Manifest resolution ignores Marker.track, so (track, landmark) is effectively reduced to just landmark.

  Medium

  - Tracking-run object provenance is not persisted in tracking_run_persons.
  - Object finalization is coupled to detector_type == "aruco" and four-corner layouts.
  - Generated skeletons include reflective dots even though current finalization emits only coded-corner slots. The effective skeleton and observation manifest can therefore diverge until later detector work lands.
  - Bare marker IDs prevent reuse of the same numeric ID across dictionaries.
  - The new object launcher is intentionally single-subject and cannot feed phase 3’s person–prop MultiPersonTracker workflow. Its execution helper is reusable, but the run/provenance data model should be resolved before extending the UI.

  ### Performance

  Medium

  - Cameras are processed sequentially despite the explicit parallelism requirement.
  - frame_step does not avoid decoding skipped frames.
  - Every processed frame writes a full dense blob including all absent markers. This is simple and compatible with current readers, but dot-heavy future phases should benchmark database size and write throughput before adopting the same layout unchanged.

  ### Code quality and best practices

  Medium

  - Detection pipeline camera/sync loading is duplicated from the person pipeline, increasing the chance that sync, camera filtering, or boundary behavior diverges.
  - The “primary source” loader chooses the first non-hand source rather than validating that exactly one exists (/D:/mocap/posetrak/cpp/src/db/session_reader.cpp:1001). Future overlay/detector sources could be silently ignored or selected nondeterministically.
  - Several schema relationships important to provenance are represented only by convention or JSON duplication, particularly detector configuration versus marker-body definition and tracking run versus capture object.

  Low

  - dictionary in marker-run metadata is described as a best-effort value for multi-dictionary bodies. A nullable or explicitly multi-valued representation would be clearer than storing a misleading single family.
  - Cancellation records a run as "failed" and preserves partial rows. A distinct cancelled status would improve diagnostics and retry UX.

  ### Security

  No actionable security vulnerability was found in the reviewed feature diff.

  ### Missing or insufficient tests

  High

  - No test for initialization failure through the real CLI, including verification that tracking does not proceed from an origin rest pose.
  - No test for locked/symmetric root behavior.
  - No two-track/two-sequence binding test.

  Medium

  - No synthetic moving rigid-body trajectory test with noise, dropout, occlusion, and pose RMSE.
  - No test that object tracking creates tracking_run_persons.capture_object_id.
  - No test that marker-specific noise reaches Observation::noise_std_override.
  - No generic blob/dot or mixed coded-plus-dot finalization test.
  - No performance test for multi-camera full-resolution detection.
  - The focused Python suite is currently red due to the outlier-edit failure.

  ## Verification performed

  - Focused Python tests: 59 passed, 1 failed.
  - C++ [rigid_init]: passed, 19 assertions.
  - C++ manifest-filtered test: passed, 12 assertions.
  - Workspace remained unchanged.

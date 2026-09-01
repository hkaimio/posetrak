# Dot assignment architecture (Phase C2, Option A)

Design round for
[reflective-dot-detection-design.md](reflective-dot-detection-design.md)
§3.2/§7: tracker-side Hungarian/Mahalanobis assignment of anonymous
reflective-dot candidates to named, calibrated dot slots, for a rigid
prop. Grounded directly in the current codebase at each step below, not
sketched in the abstract. Not built yet — design only.

## 1. What doesn't need to change

**Skeleton format: nothing.** A calibrated dot is an ordinary named
`Marker` with a known local offset (from Phase A/B/C1 calibration) —
identical in kind to an ArUco corner, just without a deterministic
per-frame manifest mapping. `Skeleton::input_tracks()`
(`skeleton.hpp`/`skeleton_loader.cpp`) already carries a per-track `type`
string that the loader never validates or restricts to a known set
(`track_type = track_node["type"].as<std::string>("")`, stored as-is) —
the existing `type: labeled_points` (ArUco) and a new `type:
unlabeled_points` (dots) both work with zero parser changes. The
distinction the tracker needs — "this track's markers resolve via the
manifest" vs. "this track's markers resolve via runtime assignment" —
already has a natural home.

**`UKF::update()`: nothing.** It takes `std::vector<Observation> const&`
with every observation already carrying a resolved `marker_id`
(`observation.hpp`). If assignment happens *before* `update()` is called
and produces ordinary `Observation`s, the entire joint sigma-point
machinery (predicted-measurement batching, innovation covariance, outlier
gating, NIS feedback) runs completely unchanged on the combined ArUco +
resolved-dot observation set. This was the explicit design goal in the
scoping doc and it holds exactly as hoped.

## 2. Data flow

```
detection (Python)          storage (DB)                    load+track (C++)
──────────────────         ──────────────                  ─────────────────
blob detector    ──write──▶ detection_dot_candidates  ──┐
(per camera/frame,          (variable rows/frame,        │  finalise (copy+rekey,
 N candidates)               detection_run-scoped)       │  no assembly needed --
                                                          ▼  unlike ArUco's fixed
                             pose_observation_dot_    ◀───┘  corner-slot blob)
                             candidates (sequence-scoped)
                                    │
                                    ▼
                        SessionReader::load_observations()
                        (new: also returns per-camera/frame
                         UnlabeledCandidate lists, keyed by
                         "unlabeled_points" input tracks)
                                    │
                                    ▼
                    Tracker::run_parent_step(), between
                    predict() and update() -- §5: build cost
                    matrix from prior_state_/prior_cov_ (already
                    computed by predict(), no extra work),
                    solve Hungarian, gate, emit Observations
                                    │
                                    ▼
                         ukf_->update(observations, ...)  <- UNCHANGED
```

## 3. Detection-time storage

Fixed-width blobs (`detection_keypoints.keypoints`, matching
`marker_id → fixed slot` for ArUco) don't fit a variable-count candidate
list per frame. New table, one row per candidate — SQL's native row
model already handles "variable count", no blob-packing scheme needed:

```sql
CREATE TABLE detection_dot_candidates (
    detection_run_id TEXT NOT NULL REFERENCES detection_runs(id),
    shot_video_id    TEXT NOT NULL,
    video_frame      INTEGER NOT NULL,
    candidate_idx    INTEGER NOT NULL,  -- 0..N-1 within this (run, camera, frame)
    px               REAL NOT NULL,     -- distorted pixel x, matches every other
    py               REAL NOT NULL,     -- detection_keypoints convention
    area             REAL NOT NULL,
    compactness       REAL NOT NULL,
    confidence       REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (detection_run_id, shot_video_id, video_frame, candidate_idx)
);
```

Written by a new `DotCandidateWriter` (Python, mirrors
`MarkerKeypointWriter`'s batching pattern in `db_cache.py`) from
`prototype_dot_blob_detector.py`'s `detect_blobs()` once it's promoted out
of throwaway-script status. `area`/`compactness` ride along for
diagnostics and possible future cost-function refinement (e.g.
down-weighting marginal candidates), not required by the Hungarian solver
itself.

Sequence-scoped mirror, populated at finalisation:

```sql
CREATE TABLE pose_observation_dot_candidates (
    sequence_id         TEXT NOT NULL REFERENCES pose_observation_sequences(id),
    camera_instance_id  TEXT NOT NULL,
    video_frame         INTEGER NOT NULL,
    timestamp_s         REAL NOT NULL,
    candidate_idx       INTEGER NOT NULL,
    px                  REAL NOT NULL,
    py                  REAL NOT NULL,
    area                REAL NOT NULL,
    compactness          REAL NOT NULL,
    confidence          REAL NOT NULL DEFAULT 1.0,
    detection_run_id    TEXT NOT NULL,
    PRIMARY KEY (sequence_id, camera_instance_id, video_frame, candidate_idx)
);
```

Unlike ArUco's finalisation (`finalise_object_to_db`, which assembles
per-marker corners into one fixed-width blob per frame), this is a
straight copy + timestamp resolution — no assembly step, since there's no
fixed layout to assemble into. Small, mechanical addition to
`finalise_object_to_db`.

**No `pose_sequence_keypoints` manifest rows for dot landmarks.** That
table's whole purpose is a deterministic `keypoint_idx → name` mapping,
which doesn't exist for an anonymous candidate. The dot markers' *names*
still live on the skeleton (§1); `SessionReader` recognizes them by
belonging to an `unlabeled_points` input track, not by manifest lookup.

## 4. `SessionReader`/`ObservationSet` changes

New struct, deliberately not `Observation` (no `marker_id` yet):

```cpp
struct UnlabeledCandidate {
    int camera_id;
    int frame_idx;
    double timestamp;
    Eigen::Vector2d position;             // undistorted, matches Observation::position
    Eigen::Vector2d position_distorted;
    double confidence;
    double area;
    double compactness;
};
```

`SessionReader::load_observations()` gains a second output alongside the
existing `ObservationSet`: a per-camera, per-frame
`std::vector<UnlabeledCandidate>` read from
`pose_observation_dot_candidates`, undistorted the same way ArUco corners
already are. Whether this becomes a second return value, an out-parameter,
or a field folded into `ObservationSet` is an implementation-detail
decision for whoever builds this, not architecturally significant either
way — the load-time work (query, undistort, bucket by frame) is the same
shape as what already happens for labeled observations.

## 5. Where assignment happens — exact integration point

`Tracker::run_parent_step()` (`tracker.cpp:867`), verified against the
current code:

```cpp
auto predict_result = ukf_->predict(dt);          // Step 1 (unchanged)
State const prior_state = ukf_->state();           // ALREADY computed
Eigen::MatrixXd const prior_cov = ukf_->covariance(); // ALREADY computed
// ... debug print (unchanged) ...

// NEW Step 2: dot assignment, only if this skeleton has an
// unlabeled_points input track AND this frame has candidates for it.
// Uses prior_state/prior_cov directly -- no extra predict() or FK call
// beyond what already runs today.
std::vector<Observation> augmented = observations;  // was: used directly
assign_dot_observations(augmented, unlabeled_candidates_this_frame,
                        prior_state, prior_cov, *skeleton_, cameras_, config_);

if (!has_sufficient_observations(augmented)) { ... }   // Step 2 (was: observations)
auto update_info = ukf_->update(augmented, cameras_, *fk_, ...);  // Step 3 (was: observations)
```

`track_frame()` needs to thread the frame's `unlabeled_candidates`
through from its own caller (ultimately `step_person_context()` in
`multi_person_tracker.cpp`, which already looks up `frame_obs` from
`ObservationSet` the same way) down to `run_parent_step()` — mechanical,
not architecturally interesting.

This lands assignment exactly where the scoping doc's §3.2 asked for it:
downstream of `update()`'s own math needing zero awareness that
assignment happened, and upstream reusing state that's already being
computed regardless.

## 6. Cost function — closed form for a rigid body, not sigma points

Running the full joint sigma-point UKF machinery just to get a gating
covariance *before* knowing which candidates to include would be
expensive (an extra ~`n_sigma`≈25 FK evaluations per frame) and somewhat
circular (that machinery's whole design assumes the observation set is
already known). For a rigid-body skeleton specifically
(`Skeleton::is_rigid_body()`, exactly the case this applies to today), the
covariance propagation has an exact closed form — no FK, no Pinocchio
call, no linearization approximation beyond the same EKF-style Jacobian
this class of problem always needs:

For marker `m` with calibrated local offset `p_local(m)`, world position
`p_world(m) = R·p_local(m) + t`. Verified against
`State::apply_error_update()`'s exact convention (`state.cpp:66`): the
error state is `[position(3), axis-angle(3), joint_angles, velocity(3),
angular_velocity(3), joint_velocities]`, with orientation perturbed as
`R_new = R · Exp(δθ)` (**right**, body-frame multiplicative). First-order
expansion:

```
p_world_perturbed(m) ≈ p_world(m) + δt − R·skew(p_local(m))·δθ
```

So the 3×6 Jacobian of `p_world(m)` w.r.t. the error state's
`[position, axis-angle]` block is exactly `J_m = [I₃ | −R·skew(p_local(m))]`
— a skew-symmetric-matrix construction and one 3×3 multiply, nothing
else. The needed 6×6 covariance block is `prior_cov.block<6,6>(0,0)`
directly (root position + orientation are always the first 6 error-state
dimensions, joint angles come after — true for any skeleton, but for a
rigid body there are zero joint angles so this *is* the whole pose
covariance). Then:

```
Cov_world(m) = J_m · prior_cov.block<6,6>(0,0) · J_mᵀ            (3×3)
Cov_pixel(m, cam) = J_cam · Cov_world(m) · J_camᵀ                 (2×2)
```

`J_cam` is the standard pinhole projection Jacobian (2×3, camera-frame
derivative composed with the camera's known world→camera rotation) —
no analytic camera Jacobian exists in this codebase yet
(`camera.cpp` only has a *numerical* one used elsewhere), but the pinhole
form is textbook and cheap to add. `Cov_pixel` plus the predicted 2D
position (`fk` at `prior_state`, or the same closed-form transform since
there's no articulation to run FK over) is exactly what a Mahalanobis
cost entry needs: `cost(slot, candidate) = (candidate − predicted)ᵀ ·
Cov_pixel⁻¹ · (candidate − predicted)`.

This is exact for a rigid body (not an approximation layered on top of an
approximation) and reuses state already computed by `predict()` — cheap
enough to run every frame without materially affecting the ~500+
steps/s the tracker already achieves (1f/baseline real-data measurements).
**Explicitly rigid-body-only**: a future non-rigid extension (person
markers, UC2) would need the general sigma-point-based version this
sidesteps — out of scope here, and should be flagged as such wherever this
gets implemented so nobody assumes it generalizes for free.

## 7. Assignment solver

No existing dependency provides this (checked: Eigen, Pinocchio, fmt,
Catch2, CLI11, toml++, nlohmann::json, sqlite3, yaml-cpp — none). Given
realistic problem sizes (single digits to perhaps a dozen candidates per
camera per frame, per §2.1's real measurements), a hand-written O(n³)
Hungarian algorithm is the right scope — no new subproject/wrap needed.
New small header, e.g. `cpp/include/posetrak/tracking/assignment.hpp`,
independently unit-testable against synthetic cost matrices with no
tracker/skeleton/camera involved at all.

**Gating** (`marker-detection-analysis.md`'s "ambiguity policy — drop,
don't guess", carried over unchanged): entries above the configured
Mahalanobis threshold are not valid assignments. Standard approach:
pad the cost matrix with dummy rows/columns at a fixed high cost (or run
Hungarian on the raw rectangular matrix, then discard any resulting pair
whose cost exceeds the gate) so a genuinely unmatched slot or candidate
doesn't force a bad pairing just because the solver must produce a
complete assignment. An unmatched slot contributes no `Observation` for
that camera this step — the same "missing observation costs a little
covariance growth, a wrong one injects a confident lie" philosophy this
codebase already applies everywhere else.

Assignment runs **per camera independently** (candidates are inherently
per-camera 2D detections, exactly like every other marker observation
already works — the same physical dot seen by two cameras already
produces two independent `Observation`s with the same `marker_id` today).

## 8. Config surface

New `TrackerConfig` fields (mirroring `rigid_init_max_residual_m`'s own
addition pattern from phase 1f — struct field, TOML parsing, validation,
DB column + `SessionReader::load_tracker_config()` wiring):

- `dot_assignment_gate_mahalanobis` (double) — separate from
  `outlier_threshold` (which gates already-labeled observations after the
  fact); this gates candidate-to-slot *assignment* before an Observation
  even exists. Likely wants its own, probably looser, threshold — an
  unresolved slot costs nothing, but an artificially tight gate here
  would waste real candidates the outlier-rejection stage could otherwise
  have used.
- Enable/disable is implicit: an `unlabeled_points` input track present
  in the skeleton *and* candidates present for a frame is sufficient
  trigger; no separate toggle needed.

## 9. Explicitly out of scope for this round

- **RANSAC cold-start** (marker-mocap-algorithms.md §4.1, "unlabeled
  rigid-template registration by pairwise-distance RANSAC") for
  initializing a track with *no* ArUco anchor at all. Not needed today:
  the validated init-search fix (status.md, 2026-08-31) already gets a
  reliable ArUco-based rigid init in practice, and dots only need to
  *join* an already-initialized track (§5 runs after the tracker already
  has a `prior_state` to predict from). Worth building if a capture ever
  needs dots-only init, not before.
- **Non-rigid extension** (§6) — the closed-form covariance is specific
  to `is_rigid_body()` skeletons; person markers (UC2) would need the
  general case.

## 10. Testing strategy

- **Assignment solver** (§7): pure unit tests, synthetic cost matrices,
  no tracker/skeleton/camera dependency — verify correct recovery
  including non-square matrices (more candidates than slots or vice
  versa), gating (a cost above threshold must not produce a pairing), and
  a tie-breaking case.
- **Cost/gating math** (§6): unit test against a synthetic rigid skeleton
  (reuse `tests/data/rigid_prop.yaml` from phase 1f) with a known
  `prior_state`/`prior_cov` and hand-computed expected `Cov_pixel` for at
  least one marker/camera pair — catches a sign error in the Jacobian
  (the right-vs-left perturbation convention in §6 is exactly the kind of
  thing that's easy to get backwards) before it reaches integration
  testing.
- **Integration**: extend `test_tracker_integration.cpp`'s rigid-init
  fixture (phase 1f) with a synthetic frame carrying both a labeled ArUco
  observation and unlabeled dot candidates at known positions, verify the
  resulting posterior matches what feeding the same data in as
  pre-labeled `Observation`s directly would produce — confirms `update()`
  really is unaffected by how observations got resolved, closing the loop
  on §1's central claim.
- **Real data**: re-run the "Harri bokken" baseline (status.md,
  2026-09-01) once C1 (calibration) and this design are both built —
  Phase C3's own stated purpose, already planned.

## 11. Open questions

- **`ObservationSet` API shape** (§4) — second return value, out-param,
  or folded field; a real but low-stakes implementation decision, not
  picked here.
- **Per-dot `noise_std`** — same open item Phase A/B/C1 already flagged,
  now also feeds `Cov_pixel`'s candidate-side uncertainty (currently
  implicit in the Mahalanobis distance's *predicted*-side covariance
  only, §6); worth deciding whether candidate detection noise should also
  contribute to the gate once real residuals exist to measure it from.
- **`area`/`compactness` as soft cost terms** — currently along for
  diagnostics only (§3); worth revisiting once real assignment behavior
  on the GoPro-only data (§2.1's 3-way area separation) shows whether the
  hard area/compactness filter at detection time is sufficient on its own
  or whether marginal candidates would benefit from a soft cost penalty
  instead of a hard reject.

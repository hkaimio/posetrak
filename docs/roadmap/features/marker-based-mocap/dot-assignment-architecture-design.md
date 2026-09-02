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
detection (Python)              storage (DB, existing tables)         load+track (C++)
──────────────────             ─────────────────────────────         ─────────────────
blob detector    ──write──▶ detection_keypoints                  ──┐
(per camera/frame,          region_type='dots', variable-N          │  finalise (same generic
 N candidates)               blob, detection_run-scoped)            │  copy loop as markers,
                                                                     ▼  region_type='dots' --
                             pose_observations                  ◀───┘  no new mechanism, §3)
                             source='dots', same blob layout
                             (sequence-scoped)
                                    │
                                    ▼
                        SessionReader::load_observations()
                        (new: also returns per-camera/frame
                         UnlabeledCandidate lists, decoded from
                         source='dots' rows, keyed by
                         "unlabeled_points" input tracks)
                                    │
                                    ▼
                    Tracker::run_parent_step(), between
                    predict() and update() -- §5: get each dot
                    slot's MarkerPrediction (§6 -- closed form
                    today) from prior_state_/prior_cov_ (already
                    computed by predict()), solve Hungarian (§7),
                    gate, emit Observations
                                    │
                                    ▼
                         ukf_->update(observations, ...)  <- UNCHANGED
```

## 3. Detection-time storage

**Revised 2026-09-01 in response to review** (Harri: *"I believe the
keypoint detections are stored as a single blob per camera frame. Is
there a reason for using row per detected point candidate for dots? And
is there need for a separate copy after finalization as it is a
no-op?"*). Checked directly against the schema
(`session_schema.sql`) — both questions land on a real overreach in the
first draft:

**No new tables at all.** `detection_keypoints` (`detection_run_id,
shot_video_id, video_frame, track_id, region_type, keypoints BLOB,
noise_scale`) and `pose_observations` (`sequence_id, camera_instance_id,
video_frame, timestamp_s, person_id, source, kp_blob, noise_scale`) are
*already* one-row-per-(frame, camera, source/region_type) with an
arbitrary-length blob — ArUco's fixed 4-corners-per-marker layout is a
property of what ArUco happens to store, not a constraint the schema
imposes. Dots fit the existing tables directly: a new
`region_type='dots'` (detection-time) / `source='dots'` (sequence-scoped)
value, with a documented variable-length blob layout —
`float32[N, 4]` per (frame, camera): `(px, py, area, compactness)` for
each of the frame's `N` candidates, `N` recovered from `byte_count /
16` exactly the way `decode_keypoints()` already recovers its own N from
`byte_count / 12` (`blob_codec.hpp`). A new `decode_dot_candidates()`
sibling function handles the 4-floats-per-row layout; everything else
(one row per frame/camera, `PRIMARY KEY` shape, `noise_scale` column)
is unchanged. This directly fixes the row-count scaling concern too
(§7): going from a handful to "several tens" of candidates per frame
changes blob *size*, not row *count* — the write/query volume this
schema was already sized for doesn't change at all as candidate count
grows.

Written by a new `DotCandidateWriter` (Python, mirrors
`MarkerKeypointWriter`'s batching pattern in `db_cache.py`, `INSERT`ing
into `detection_keypoints` with `region_type='dots'`) from
`prototype_dot_blob_detector.py`'s `detect_blobs()` once it's promoted out
of throwaway-script status. `area`/`compactness` ride along per candidate
for diagnostics and possible future cost-function refinement (e.g.
down-weighting marginal candidates), not required by the Hungarian solver
itself.

**The finalisation copy is still needed, but it's not new machinery —
it's the existing generic copy loop, one more parameter value.**
`finalise_object_to_db`'s existing marker-copy query is already
parameterized by `region_type`/`source`:

```python
kp_rows = session.execute(
    "SELECT shot_video_id, video_frame, keypoints, noise_scale FROM detection_keypoints "
    "WHERE detection_run_id = ? AND track_id = ? AND region_type = ? "
    ...
    (detection_run_id, MARKER_TRACK_ID, MARKER_REGION_TYPE),
)
```

Dots need the identical call shape with `region_type='dots'` — not a
second code path. What that copy actually *does* (and why it's not a
literal no-op, even though no cross-camera assembly happens): resolves
`shot_video_id → camera_instance_id` and `video_frame → timestamp_s` (via
the trial's `SyncTable`, exactly as markers' own copy already does), and
crosses the same append-only-detection-run → stable-sequence boundary
every other observation source in this schema crosses — `pose_observations`
rows are what tracking runs, review, and `pose_observation_edits` all key
off, and are immutable once tracked (the existing re-finalisation guard);
`detection_keypoints` rows are provisional until finalised. That boundary
is what the copy buys, for dots exactly as much as for markers — the
"assembly" framing in the original draft was the wrong justification
(dots indeed need none), but the boundary itself isn't optional.

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
`std::vector<UnlabeledCandidate>`, decoded via the new
`decode_dot_candidates()` (§3) from `pose_observations` rows with
`source='dots'`, undistorted the same way ArUco corners already are.
Whether this becomes a second return value, an out-parameter, or a field
folded into `ObservationSet` is an implementation-detail decision for
whoever builds this, not architecturally significant either way — the
load-time work (query, decode, undistort, bucket by frame) is the same
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

## 6. Cost function — a predicted-position-and-covariance seam, closed form for rigid bodies now

**Revised 2026-09-01 in response to review** (Harri: *"the architecture
should scale to the generic case of having [dozens of unlabeled markers]
in articulated bodies, e.g. as augmentation in performers"*). The first
draft framed the rigid closed-form math as *the* cost function and waved
at "a future non-rigid extension" as someone else's problem. Checked
against the code and that framing undersold what already exists: the
assignment/Hungarian layer (§7) and the `Tracker`/`SessionReader`
integration points (§2, §4, §5) never assumed rigidity in the first
place — only the covariance computation is rigid-specific. So the actual
seam is narrow, and worth naming explicitly rather than leaving implicit:

> **`MarkerPrediction { Eigen::Vector2d position; Eigen::Matrix2d covariance; }`
> for one named marker slot, in one camera, at the tracker's current
> `predict()`-step state.** Assignment (§7) only ever consumes this —
> it has no idea whether the number behind it came from a closed form or
> from sigma points.

Two implementations of that seam:

**Rigid closed form (this round; §6.1 below).** For
`Skeleton::is_rigid_body()`, cheap and exact — no FK, no Pinocchio call.

**General/articulated (deferred to whenever UC2's person-marker
augmentation needs it; not hypothetical, already exists in kind).**
`UnscentedKalmanFilter::predict_measurements()` (`ukf.cpp:1798`) —
verified directly, not assumed — takes an arbitrary
`std::vector<Observation>` keyed only by `marker_id` (any named marker on
the skeleton, rigid or articulated) and a `State`, runs FK once, and
projects into cameras. This is *already* the general, articulation-aware
version of exactly what §6.1's closed form computes for the rigid case.
Getting a `MarkerPrediction` for an articulated skeleton's dot slot is
running this same function — across all sigma points, exactly as
`update()`'s own Step 2 already does every frame for labeled observations
— against a synthetic one-pseudo-observation-per-dot-slot vector, then
reading off each slot's own diagonal 2×2 block of the resulting
innovation covariance (`ukf.cpp`'s existing `innovation_cov` computation,
same formula, computed one step earlier and on placeholder observations).
The real cost is an extra `n_sigma`≈25 FK evaluations per frame for the
dot slots specifically — exactly the expense §6.1's shortcut exists to
avoid for the common single-rigid-prop case — not new numerical method
design. **Not built in this round**: no articulated capture with dot
augmentation exists yet to design or test against (same
prototype-before-architecture discipline this whole feature has followed
throughout), and the rigid case covers the sword. But the seam above is
real and both implementations plug into the identical §5 integration
point and §7 solver unchanged, so building the general one later is
additive, not a rewrite.

### 6.1 Rigid closed form

For a rigid-body skeleton specifically (`Skeleton::is_rigid_body()`,
exactly the case this applies to today), the covariance propagation has
an exact closed form — no FK, no Pinocchio call, no linearization
approximation beyond the same EKF-style Jacobian this class of problem
always needs:

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
It implements the `MarkerPrediction` seam above; the general/articulated
implementation described there is deferred, not designed away.

## 7. Assignment solver

No existing dependency provides this (checked: Eigen, Pinocchio, fmt,
Catch2, CLI11, toml++, nlohmann::json, sqlite3, yaml-cpp — none). A
hand-written O(n³) Hungarian algorithm is the right scope — no new
subproject/wrap needed. New small header, e.g.
`cpp/include/posetrak/tracking/assignment.hpp`, independently
unit-testable against synthetic cost matrices with no
tracker/skeleton/camera involved at all.

**Revised 2026-09-01**: the first draft under-scoped the expected
candidate count ("single digits to a dozen") against Harri's own expected
scale ("several tens per scene"). The solver itself doesn't need to
change for that: O(n³) at n=50 is 125,000 operations, sub-millisecond —
several orders of magnitude below the tracker's existing per-step budget
(~2ms at the ~500 steps/s the 1f baseline already measures). The real
scaling questions this raises are elsewhere, already addressed above: row
volume is §3's blob-vs-row fix (flat regardless of N), and the
predicted-position cost per slot is §6's `MarkerPrediction` seam (closed
form is O(1) per slot for a rigid body; the deferred articulated
implementation is the one that actually costs more at higher N, via its
extra sigma-point FK pass — a real cost, but the solver isn't where it
lands).

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

Assignment runs **per camera, per tracked subject (skeleton), independently**
(candidates are inherently per-camera 2D detections, exactly like every
other marker observation already works — the same physical dot seen by
two cameras already produces two independent `Observation`s with the same
`marker_id` today). "Per subject" is doing real work in that sentence —
see §9's new subsection on what this assumes and doesn't yet handle.

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
- **The general/articulated `MarkerPrediction` implementation** (§6) —
  seam is designed, implementation deferred; no articulated
  dot-augmented capture exists yet to build or validate it against.

### 9.1 Scene-wide / multi-subject candidate pools — a real gap, not decided here

§7's "per subject, independently" is only correct when each tracked
subject's candidates are already known to be its own — true for the
sword today (one object, its own detection run, its own candidate rows).
It stops being true the moment two or more dot-bearing subjects (a prop
*and* a performer wearing augmentation markers, or two props) are tracked
from the *same* shared candidate pool in the same scene: nothing in this
design arbitrates a candidate that gates acceptably against slots on two
different skeletons' independent `MultiPersonTracker`-owned `Tracker`
instances, so both could independently "claim" it in the same step.

Not designed here, deliberately: no real multi-subject-with-dots capture
exists yet to design or validate against — the same discipline this
whole feature has followed throughout (prototype/validate against real
data before committing architecture, most recently for the rigid
closed-form math itself, §2.1's blob detector, and Phase A's
co-occurrence check). This also directly connects to the still-open
scene-wide-detection item already logged (status.md, 2026-08-31: "a scene
with several props + performers re-decodes the same footage once per
subject... a shared single-pass scene-wide detection... is the real fix")
— multi-subject candidate arbitration and shared-pass detection are
naturally the same future piece of work, not two separate ones. When it's
tackled, the likely shape is orchestration one level up
(`MultiPersonTracker`, which already runs its owned `Tracker` instances in
a defined order per step and already has precedent for cross-instance
coordination via cross-person `PAIR_DIFF` anchor observations) rather
than anything inside a single `Tracker`'s own assignment call — flagged
so the eventual design starts from the right layer, not decided further
than that here.

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
- **Blob codec + finalisation reuse** (§3): unit test
  `decode_dot_candidates()` round-trips a variable-N blob correctly
  (including N=0 and a large N in the "several tens" range, §7); a
  `finalise_object_to_db` test with a `region_type='dots'` detection run
  confirms the existing copy loop produces correct
  `pose_observations`/`source='dots'` rows with no dots-specific code
  path.

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
- **Multi-subject candidate arbitration** (§9.1) — real gap, explicitly
  not designed this round; needs a real multi-subject-with-dots capture
  to design against.
- **When to build the general `MarkerPrediction` implementation** (§6) —
  gated on a real articulated-body dot-augmentation use case existing,
  not decided here.

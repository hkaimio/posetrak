# Dot assignment architecture (Phase C2, Option A)

Design round for
[reflective-dot-detection-design.md](reflective-dot-detection-design.md)
§3.2/§7: tracker-side Hungarian/Mahalanobis assignment of anonymous
reflective-dot candidates to named, calibrated dot slots. Grounded
directly in the current codebase at each step below, not sketched in the
abstract. Not built yet — design only.

**Target architecture, decided 2026-09-02**: assignment is **one shared
phase across every tracked subject in the scene**, not a per-subject
step — Harri's explicit call, because the alternative (each subject
independently resolving its own candidates) cannot prevent two subjects'
skeletons from both claiming the same physical candidate, and that
conflict is exactly the kind of thing this project designs against up
front rather than working around later. Not needed for the very first
real use (the sword, alone in its own trial) — a single subject is the
N=1 degenerate case of the shared design, not a separate implementation
that needs replacing once a second subject shows up. §5 below designs the
shared phase directly; nothing here builds a single-subject shortcut.

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
              ── per frame, orchestrator level (§5) ──
              1. predict_step() on every dot-bearing subject
                 (Tracker split: predict, not yet update)
              2. every subject's MarkerPrediction (§6) for its
                 own dot slots, gathered into ONE cost matrix
                 per camera across ALL participating subjects
              3. one Hungarian solve (§7) per camera, gated,
                 resolved candidates split back out per subject
              4. update_step(own_obs + resolved_dots) per subject
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

## 5. Where assignment happens — a shared phase, requires splitting `Tracker`'s predict from its update

**Revised 2026-09-02** (Harri: *"dot assignment must be a common phase
for all tracked subjects... not needed for this first demo but very
soon"*). The original draft put assignment *inside* `Tracker::
run_parent_step()`, between its own already-called `predict()` and
`update()` — correct for exactly one subject, but structurally unable to
arbitrate a candidate two different subjects' skeletons could both claim,
since each `Tracker` only ever sees its own predicted positions. A shared
phase needs every participating subject's *prediction* available before
*any* subject's *update* runs — which today's `Tracker` can't expose,
because `run_parent_step()` (verified: `tracker.cpp:867`, `private`) does
both atomically, and `track_frame()` (the only `public` per-frame entry
point today) calls it as one unit.

### 5.1 Split `Tracker`'s predict from its update

`run_parent_step()`'s own body already divides cleanly at exactly this
boundary — verified line by line, not assumed:

```cpp
// Everything through here only needs dt -- no observations yet:
auto predict_result = ukf_->predict(dt);
State const prior_state = ukf_->state();
Eigen::MatrixXd const prior_cov = ukf_->covariance();
if (frame_count_ < config_.debug_init_frames) { print_init_debug(prior_state, "PRIOR "); }

// Everything from here on is the update half:
if (!has_sufficient_observations(observations)) { return {...lost...}; }
auto update_info = ukf_->update(observations, cameras_, *fk_, ...);
// ... NIS feedback, debug print, TrackingResult construction ...
```

Two new **public** methods replace the current private/atomic split:

- **`Tracker::predict_step(double dt)`** — runs the predict half above,
  stores `prior_state_`/`prior_cov_` as new private members (today they're
  locals). Callable independently of any observations.
- **`Tracker::update_step(std::vector<Observation> const& observations, double timestamp) -> TrackingResult`**
  — runs the update half, using the `prior_state_`/`prior_cov_` stored by
  the most recent `predict_step()` call, plus the existing bookkeeping
  `track_frame()` currently does after `run_parent_step()` returns
  (`last_timestamp_`, `frame_count_`, `prev_observations_`,
  `frame_callback_`).
- **`Tracker::predict_dot_slot_predictions() -> std::vector<std::pair<int, MarkerPrediction>>`**
  (marker_id → `MarkerPrediction`, §6) — callable only after
  `predict_step()`, for every marker belonging to this skeleton's
  `unlabeled_points` input track (§1). This is the query surface the
  orchestrator (§5.2) actually uses; it's what makes §6's rigid-vs-general
  implementation choice a `Tracker`-internal decision the orchestrator
  never needs to know about.
- **`Tracker::track_frame()` is unchanged in behavior and signature** —
  becomes `{ predict_step(dt); return update_step(annotated, timestamp); }`
  internally, still the right call for every subject that has no
  `unlabeled_points` track (every existing person and the sword's own
  ArUco corners) or no candidates this particular frame. The split only
  matters to a subject actually participating in shared dot assignment
  this frame.

### 5.2 Orchestrator-level shared resolution

The split needs a caller above `Tracker` that can see every participating
subject at once. Neither of today's two call paths is that caller today,
but both already reuse the same free-function layer for the ordinary
per-frame step (`step_person_context()`, `multi_person_tracker.hpp`) —
the single-subject CLI path (`run_track_from_db()`, a raw loop calling
`step_person_context()` directly) and the actual multi-subject path
(`MultiPersonTracker::run()`, `multi_person_tracker.cpp:1111`, already
literally shaped "for frame: for subject: step subject", already
running one shared per-frame phase — `update_contact_gate()` — before
each subject's own step, for the structurally analogous cross-person
contact-anchor case). A new shared-dot-assignment phase is a sibling to
`update_contact_gate()`/`build_anchor_observations()`, not a new kind of
thing this orchestrator hasn't done before:

```cpp
// New free function, multi_person_tracker.hpp/.cpp -- one call per frame,
// given every subject (PersonContext or object-equivalent) with an
// unlabeled_points track and candidates this frame.
struct SubjectDotAssignment {
    std::vector<Observation> resolved;  // this subject's newly-labeled dot Observations
};
std::unordered_map<int, SubjectDotAssignment>
resolve_shared_dot_assignment(
    std::vector<DotAssignmentSubject> const& subjects,  // tracker*, skeleton, cameras -- enough to call predict_dot_slot_predictions()
    std::unordered_map<int, std::vector<UnlabeledCandidate>> const& candidates_by_camera,
    TrackerConfig const& config, int frame_idx, double timestamp);
```

Per frame, for dot-bearing subjects only:

```cpp
for (auto* s : dot_bearing_subjects) s->tracker->predict_step(dt);   // 1. everyone predicts first

auto assignment = resolve_shared_dot_assignment(dot_bearing_subjects,   // 2. ONE combined
                                                candidates_this_frame,  //    resolution, §7.1
                                                config, step, timestamp);

for (auto* s : dot_bearing_subjects) {                                // 3. then everyone updates
    auto obs = s->own_labeled_observations_this_frame;                //    with their own share
    auto const& resolved = assignment[s->idx].resolved;
    obs.insert(obs.end(), resolved.begin(), resolved.end());
    s->tracker->update_step(obs, timestamp);
}
// every other subject (no unlabeled_points track) keeps calling
// track_frame() exactly as today, untouched by any of the above.
```

`run_track_from_db()`'s raw loop and `MultiPersonTracker::run()`'s
`for step: for person` loop both need this three-pass shape *only* when
at least one participating subject has dots; the ordinary
`step_person_context()` call stays exactly as-is for everyone else,
including the sword's own ArUco corners (which have no
`unlabeled_points` track and never enter this path at all). Threading
this through both call paths cleanly (vs. only wiring it into
`MultiPersonTracker`, which the sword's single-subject CLI path doesn't
use today) is real, mechanical work — probably a new sibling pair to
`step_person_context()` itself (e.g. `step_person_context_predict()`/
`step_person_context_update()`, mirroring the existing
`step_person_context_frame0()`/`step_person_context()` split-by-purpose
pattern already in this file), not something to improvise per call site.

### 5.3 Why this can't reuse the existing cross-person coupling mechanism, and joint vs. sequential resolution

**Added 2026-09-02** in response to a direct question (Harri: *"the
tracked subjects already influence each other via the cross-subject
relative observation mechanism... How is that feature implemented now,
and what new requirements does dot assignment bring?"*). Worth writing
down precisely, since the answer explains why §5.1's `Tracker` split is
genuinely unavoidable rather than a design preference.

**Cross-person coupling (`error-improvements/phase5-cross-person-plan.md`,
implemented) never splits predict from update.** Traced directly in
`multi_person_tracker.cpp`: `MultiPersonTracker::run()` still calls each
subject's plain, atomic `Tracker::track_frame()`. Before subject A's
turn, `build_anchor_observations()` constructs a `PAIR_DIFF` anchor
`Observation` using subject B's **own already-computed state**
(`other_ctx.tracker->state()`) — B's current-frame posterior if B already
took its turn this frame (processing order rotates every frame), or a
one-frame constant-velocity extrapolation of B's *last* frame's posterior
if B hasn't gone yet (`build_anchor_observations()`'s own "anchor
freshness" comment, verbatim: *"current-frame posterior if already
stepped this frame, else a one-frame constant-velocity extrapolation of
their frame-(t-1) posterior"*). That anchor is appended to A's own
observation list; A's single unsplit `track_frame()` call runs unchanged.

This works without ever needing a live, same-instant peek at another
subject's mid-step prediction, because of two properties specific to what
it does that dot assignment doesn't share:

- **Correspondence is already known.** Contact-gating decides *which*
  marker on A pairs with *which* marker on B (a 3D proximity computation
  between two already-named, already-labeled points) — there is no
  identity ambiguity to resolve, only a reference *value* to borrow.
- **Staleness is tolerable.** A one-frame-old extrapolated reference is
  an accepted design point, not a shortcut — it's a soft `PAIR_DIFF`
  residual carrying its own noise term, and the extrapolation exists
  specifically to absorb that lag.

Dot assignment has neither property: it has to resolve *which
candidate is which named marker* — a genuine combinatorial identity
problem the anchor mechanism was never built to solve — and it needs
every competing subject's prediction for the *identical instant*, not a
stale one, because the correctness property actually wanted (never let
two subjects claim the same candidate) is a hard constraint on a discrete
decision. A subject deciding from its own last-frame extrapolation, while
another decides from its own, doesn't prevent both concluding "candidate
#7 is mine" in the same frame — staleness doesn't fix a double-claim, it
only hides it.

That forces the one piece of new capability the anchor mechanism never
needed: a way to get a subject's genuinely-current prediction to an
outside caller before that subject's own update commits. Checked whether
this could be avoided by "peeking" without a real split —
`UnscentedKalmanFilter::predict()` (`ukf.cpp:620`) writes `state_`/
`covariance_` in place (confirmed by reading through to its final
assignment, `state_ = compute_state_mean(...)`); it is a real state
transition, not a pure/dry-run computation callable speculatively and
discarded. So §5.1's split (or some equivalent) is not a stylistic
choice — it's what "get a live prediction without committing to it
first" actually requires, given predict() is exactly as stateful as it
looks.

**Joint vs. sequential resolution — a separate, more discretionary
choice bundled into the design above.** Given the split is required
either way, two shapes both fully prevent double-claiming:

- **Joint (what §5.2 designs, confirmed 2026-09-02)**: `predict_step()`
  on every dot-bearing subject first, then one combined cost matrix
  across all of them, solved together — globally optimal, immune to
  processing-order artifacts. A candidate always goes to whichever
  subject it genuinely fits best, never to whoever happened to ask first.
- **Sequential greedy (considered, not chosen)**: process subjects one
  at a time in the same rotated order the anchor mechanism already uses;
  each subject's own local Hungarian solve runs against whatever
  candidates earlier subjects this frame haven't already claimed,
  removing them from the pool as it goes. Needs the identical
  per-subject `predict_step()`/`update_step()` split (predict() is
  exactly as stateful here as in the joint case), but never constructs a
  cross-subject cost matrix and doesn't need a full predict-everyone pass
  before any subject updates — closer to the existing anchor mechanism's
  own shape. Trade-off: order-dependent — a candidate can go to a
  merely-adequate match for whoever processes first, when a later subject
  in the same frame would have been the better fit, and that failure mode
  gets more likely as candidate density grows (exactly the "several tens
  per scene" / person-picks-up-a-marked-prop case this is being designed
  for).

Decided: **joint**, for the same reason the earlier per-count and
per-scale decisions in this doc went the more rigorous way — production
quality over the cheaper path, and the actual cost difference is small
(§7.1's combined-matrix complexity is still comfortably sub-millisecond
at the scales discussed).

### 5.4 The real prerequisite this exposes: candidates must be a single shared pool, not N redundant per-subject lists

A genuinely joint resolution requires that the *candidates themselves* —
not just the resolution step — be a single authoritative list per
(camera, frame), not one independently-detected list per subject's own
detection run. Today's detection model (§3, unchanged from ArUco's own
convention) runs one detection pass **per capture_object**, each
re-decoding the same footage independently. If two subjects' own
passes each happen to detect the *same* physical dot, that's two
separate, pixel-coincident-but-distinct candidate rows (one per
detection run) — feeding both into one cost matrix doesn't arbitrate
anything, since a Hungarian solver sees two different candidates, not
one contested one, and could still assign both to different subjects.

This is a real, currently-unresolved prerequisite the shared-phase design
surfaces, not a detail: **either detection itself needs to become a
single shared scene-wide pass** (already logged as a separate, bigger,
deferred item — status.md, 2026-08-31, "a scene with several props +
performers re-decodes the same footage once per subject... a shared
single-pass scene-wide detection... is the real fix" — and now confirmed
to be a hard *dependency* of correct shared assignment, not merely a
performance nicety), **or** the orchestrator needs an interim
de-duplication step: cluster candidates from different subjects'
detection runs that are within some small pixel tolerance at the same
(camera, frame) into one physical candidate before building the cost
matrix. The de-dup bridge is the pragmatic near-term option (scene-wide
detection is bigger, later work), but isn't designed or built here — see
§9.1.

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

Exposed via `Tracker::predict_dot_slot_predictions()` (§5.1) — the
orchestrator (§5.2) calls this on every dot-bearing subject after their
shared `predict_step()` pass, gathers the results per camera, and never
touches `Skeleton`/`State`/FK itself. Which implementation runs is
entirely a per-`Tracker` decision driven by its own `Skeleton::
is_rigid_body()`, invisible to the orchestrator either way.

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

**Revised 2026-09-01 (scale) and 2026-09-02 (shared pool)**: the first
draft under-scoped the expected candidate count ("single digits to a
dozen") against Harri's own expected scale ("several tens per scene").
The solver itself doesn't need to change for that: O(n³) at n=50 is
125,000 operations, sub-millisecond — several orders of magnitude below
the tracker's existing per-step budget (~2ms at the ~500 steps/s the 1f
baseline already measures), and this holds even summed across several
subjects' slots at once (§7.1) — still comfortably two-digit-microsecond
territory at n in the low hundreds. The real scaling questions this
raises are elsewhere, already addressed above: row volume is §3's
blob-vs-row fix (flat regardless of N), and the predicted-position cost
per slot is §6's `MarkerPrediction` seam (closed form is O(1) per slot for
a rigid body; the deferred articulated implementation is the one that
actually costs more at higher N, via its extra sigma-point FK pass — a
real cost, but the solver isn't where it lands).

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

### 7.1 One combined cost matrix per camera, across every participating subject

**Revised 2026-09-02** (Harri: *"dot assignment must be a common phase
for all tracked subjects"*): the first draft ran one independent Hungarian
solve per (camera, subject) — exactly the shape that can't prevent two
subjects from both claiming the same candidate, since neither subject's
solve knows the other's even ran. The fix isn't a different algorithm,
it's a differently-shaped input: **one Hungarian solve per (camera,
frame), with columns = the union of every participating subject's dot
slots that frame, rows = that camera's (de-duplicated, §5.4) candidate
list.** Gating, the dummy-row/column padding, and the "unmatched costs
nothing" philosophy above are unchanged — they already operate on an
arbitrary rectangular cost matrix, which a multi-subject matrix still is,
just wider. This is the shape `resolve_shared_dot_assignment()` (§5.2)
actually builds and solves, splitting the result back out per subject
afterward.

## 8. Config surface

**Corrected 2026-09-02**: checked `rigid_init_max_residual_m`'s actual
wiring before reusing it as precedent, and it is *not* DB-wired —
verified against both `SessionReader::load_tracker_config()`'s real
column list (`session_reader.cpp:157`, no such column selected) and
`config.cpp` (parsed from TOML only). It's exactly the same situation
`init_search_window_s` (2026-08-31) already named explicitly: a
TOML-config-file field for the `run_track()` CLI path, and a plain
`TrackerConfig` struct default for the DB-driven path every real capture
(the sword included) actually uses, "pending a real need to tune it
per-capture." `dot_assignment_gate_mahalanobis` follows that same,
correctly-verified pattern, not the DB-column one this section
originally (incorrectly) cited:

- New `TrackerConfig`/`TrackerAppConfig` struct field, `TrackerAppConfig`
  TOML parsing + validation (`config.cpp`, matching
  `rigid_init_max_residual_m`'s real shape) — usable from `run_track()`'s
  CLI config-file path. No `tracker_configs` DB column, no
  `SessionReader::load_tracker_config()` change, no Python/GUI wiring —
  add these only if real per-capture tuning need shows up, same
  threshold `init_search_window_s` was held to.
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
- **Scene-wide shared detection, and its interim de-dup bridge** (§5.4)
  — the shared assignment *phase* (§5) is now designed and targeted for
  the near future per Harri's direction, but it's only *correct* once
  candidates are a single de-duplicated pool per (camera, frame) rather
  than one independently-detected list per subject's own detection run.
  Neither the full fix (a real shared-pass detector) nor the interim
  bridge (clustering near-coincident candidates across subjects' existing
  per-object detection runs) is designed or built in this round — no real
  multi-subject-with-dots capture exists yet to design or validate either
  against, the same discipline this whole feature has followed throughout
  (prototype/validate against real data before committing architecture,
  most recently for the rigid closed-form math itself, §2.1's blob
  detector, and Phase A's co-occurrence check).
- **`resolve_shared_dot_assignment()`'s exact signature, and threading
  the predict/update split through both call paths** (§5.2) — the shape
  is designed; wiring it into `run_track_from_db()`'s raw loop and
  `MultiPersonTracker::run()` without disturbing every dot-free subject's
  existing behavior is real implementation work, not sketched to the
  line-of-code level here.

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
- **`predict_step()`/`update_step()` split** (§5.1): a direct unit test
  that calling them in sequence produces byte-identical `TrackingResult`s
  to calling `track_frame()` — the single-subject case must be exactly
  unaffected by the split existing at all.
- **`resolve_shared_dot_assignment()` — the actual multi-subject
  arbitration** (§5.2/§7.1): synthetic two-subject test with two
  skeletons whose dot slots predict to nearby-but-distinct pixel
  positions and a candidate pool containing one ambiguous point gating
  against both — verify it resolves to exactly one subject (not both,
  not neither when the gate should admit one), and a second case with the
  candidate genuinely equidistant/ambiguous verifying the loser gets no
  `Observation` rather than a forced wrong pairing. This is the test that
  actually exercises the problem Harri's review comment identified;
  everything else here can pass while that bug still exists.
- **Real data**: re-run the "Harri bokken" baseline (status.md,
  2026-09-01) once C1 (calibration) and this design are both built —
  Phase C3's own stated purpose, already planned. Single-subject only
  until a second dot-bearing subject exists in a real capture.
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
- **When to build scene-wide shared detection (or its de-dup bridge)**
  (§5.4/§9) — a real, currently-unresolved dependency of *correct* shared
  assignment with more than one subject's own detection run in play; not
  designed here.
- **`resolve_shared_dot_assignment()`'s exact call-site wiring** (§5.2) —
  the free-function shape is designed; how it plugs into
  `run_track_from_db()`'s raw loop vs. `MultiPersonTracker::run()`
  without disturbing dot-free subjects is real implementation work.
- **When to build the general `MarkerPrediction` implementation** (§6) —
  gated on a real articulated-body dot-augmentation use case existing,
  not decided here.

## 12. Implementation phasing

Scope for Phase C2 only (live per-frame labeling) — not C1 (calibration-
time dot geometry, `reflective-dot-detection-design.md` §5) or C3
(re-run and compare), each its own phase with its own scope. C2.11's own
real-data validation is gated on C1 existing (a skeleton with calibrated
`unlabeled_points` markers to track against) even though nothing else
below needs it. Each sub-task independently buildable/testable, same
discipline as phase 1's own 1a–1f breakdown
(`marker-mocap-design.md` §7.1).

| Sub-task | Delivers | Depends on | Validation |
|---|---|---|---|
| **C2.1** — Hungarian solver | `cpp/include/posetrak/tracking/assignment.hpp`: hand-written O(n³) Hungarian over an arbitrary rectangular cost matrix, with gating (§7). | None. | Pure unit tests, synthetic cost matrices (§10): correct recovery on square and non-square matrices, gating rejects above-threshold pairs without forcing a match, a tie-breaking case. No tracker/skeleton/camera involved. |
| **C2.2** — Rigid closed-form `MarkerPrediction` | The §6.1 math: `Cov_pixel`/predicted-position computation for one marker on a rigid-body skeleton, from `prior_state`/`prior_cov`. | None. | Unit test against a synthetic rigid skeleton (reuse `tests/data/rigid_prop.yaml`) with a known `prior_state`/`prior_cov` and a hand-computed expected `Cov_pixel` for at least one marker/camera pair — catches a Jacobian sign error before integration. |
| **C2.3** — DB schema + blob codec | `region_type`/`source='dots'` convention on the existing `detection_keypoints`/`pose_observations` tables (§3); `decode_dot_candidates()` sibling to `decode_keypoints()` in `blob_codec.hpp`. | None. | Unit test: round-trips a variable-N blob correctly, including N=0 and N in the "several tens" range (§7.1). |
| **C2.4** — `Tracker` predict/update split | New public `Tracker::predict_step(dt)` / `update_step(observations, timestamp)`, replacing the private atomic `run_parent_step()` (§5.1); `track_frame()` becomes a thin wrapper over both. | None (can stub `predict_dot_slot_predictions()` until C2.2 lands). | Direct unit test: calling `predict_step()` then `update_step()` in sequence produces byte-identical `TrackingResult`s to calling `track_frame()` — the single-subject case must be provably unaffected by the split existing at all. |
| **C2.5** — `predict_dot_slot_predictions()` | `Tracker` method exposing every `unlabeled_points`-track marker's `MarkerPrediction` after a `predict_step()` call (§5.1, §6). | C2.2, C2.4. | Unit test against the same rigid fixture as C2.2, called through the real `Tracker` (not the standalone math) — confirms the wiring, not just the formula. |
| **C2.6** — `SessionReader`/`ObservationSet` loading | New `UnlabeledCandidate` struct; `load_observations()` gains the per-camera/frame candidate output, decoded via C2.3 (§4). | C2.3. | Extend `test_session_reader.cpp`'s existing manifest-bound-sequence fixture (1f) with a `source='dots'` row; confirm decoded candidates match expected positions/counts. |
| **C2.7** — Detection-time write path | Promote `prototype_dot_blob_detector.py`'s `detect_blobs()` out of throwaway-script status into a real `DotCandidateWriter` (mirrors `MarkerKeypointWriter`), wired into the marker detection pipeline for GoPro cameras first (§2.1's validated regime). | C2.3. | Unit test on synthetic frames (known blob positions) — same style as existing `MarkerKeypointWriter` tests. Real-footage validation reuses §2.1's already-confirmed area/compactness parameters, not new detector tuning. |
| **C2.8** — Finalisation | Extend `finalise_object_to_db`'s existing generic copy loop to also handle `region_type='dots'` → `pose_observations`/`source='dots'` (§3) — the identical call shape as markers, one more parameter value. | C2.7. | `finalise_object_to_db` test with a `region_type='dots'` detection run confirms correct `pose_observations` rows with no dots-specific code path (already specified in §10). |
| **C2.9** — Config surface | `dot_assignment_gate_mahalanobis`: `TrackerConfig`/`TrackerAppConfig` struct field, TOML parsing, validation (§8) — verified as `rigid_init_max_residual_m`'s *actual* shape (TOML-only, no DB column), not the DB-wired version originally assumed. | None. | Mirrors phase 1f's own config-field test coverage (parse + validate), no DB round-trip needed. |
| **C2.10** — `resolve_shared_dot_assignment()` | The orchestrator-level free function (§5.2): builds one combined cost matrix per camera across every participating subject's dot slots (§7.1), solves via C2.1, splits results back out per subject. | C2.1, C2.5, C2.9. | Synthetic multi-subject test (§10, already specified): two skeletons whose slots predict near a shared ambiguous candidate — verify it resolves to exactly one subject, never both, and the loser gets no `Observation` rather than a forced pairing. This is the test that actually exercises the double-claim problem the whole shared-phase redesign exists to fix. |
| **C2.11** — Wiring into both call paths | Thread the predict-all/resolve/update-all shape (§5.2) through `run_track_from_db()`'s raw loop and `MultiPersonTracker::run()`, likely via new `step_person_context_predict()`/`step_person_context_update()` siblings to the existing `step_person_context()` (§5.2) — both call paths already share that free-function layer, so this is one piece of new plumbing, not two. Every dot-free subject's call path (every existing person, the sword's own ArUco corners) is untouched. | C2.6, C2.10. | Regression: every existing test involving `step_person_context()`/`track_frame()` for a dot-free subject must be unaffected. New: a single-subject (N=1) synthetic case confirms the three-pass shape degenerates correctly to "just track normally" when there's nothing to arbitrate. |
| **C2.12** — Real end-to-end validation | Re-run against real footage once a dot-calibrated skeleton exists. | C2.7, C2.8, C2.11, **and Phase C1** (external to this phasing). | Phase C3's own stated purpose (§5 of the scoping doc): re-run the exact "Harri bokken" baseline (status.md, 2026-09-01) with dots included; compare tracked-step fraction, reprojection error, and specifically gap duration during fast segments against the ArUco-only numbers already recorded. Single-subject only — no real multi-subject-with-dots capture exists yet to validate C2.10's actual arbitration behavior against real data, only synthetically (C2.10's own test). |

**Parallelizable**: C2.1, C2.2, C2.3, C2.9 have no dependencies on each
other or on anything else in this list — natural first batch. C2.7 (once
C2.3 lands) and C2.4 can also proceed independently of each other.

**Explicitly not in this phasing** (§9, §5.4): the scene-wide-detection/
de-dup bridge that a *second* real dot-bearing subject would need for
`resolve_shared_dot_assignment()`'s joint arbitration to be correct
against real (not synthetic) data — the sword alone never exercises it,
and no real multi-subject-with-dots capture exists yet to design or
validate it against.

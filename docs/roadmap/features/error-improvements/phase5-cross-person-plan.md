# Cross-person relative observations (Phase 5, error-improvements) — implementation plan

## Context

`docs/roadmap/features/error-improvements/implementation-plan.md` Phases
1/3/4 (split pose/calib noise, within-person `PAIR_DIFF` relative
measurements, spatial cross-pairs) are already implemented and tested.
Phase 5 — cross-*person* relative observations, for multi-person contact
and close interaction (ukemi throws, handshakes, assisted movements) — is
the one remaining item Harri has "high hopes for" on tracking-quality
impact, and is the priority pick from the first-release backlog
(`docs/roadmap/first-release-backlog.md`).

**The original Phase 5 sketch needs correcting.** It assumed a Python
orchestrator (`posetrak/tracker/multi_person.py`, `MultiPersonTracker`)
holding live C++ `Tracker` objects across Gauss-Seidel iterations. That's
not possible today: there is no Python↔C++ binding (confirmed —
`meson_options.txt`'s `enable_python` is explicitly marked "future", no
`subdir('python')` in the real build, no pybind11/nanobind/ctypes anywhere
in the tree). The only existing interface is the CLI executable run as a
subprocess (`python/posetrak/tracker/runner.py::run_tracker()`), with
results read back from CSVs/the session DB afterward.

**Settled architecture (discussed with Harri)**: build the multi-person
orchestrator **in C++, in-process**, not as a Python subprocess
orchestrator, and not as a new pybind11 binding (explicitly rejected — the
tracker should stay a separate process for stability). Two things drove
this, both near-term (not speculative) future needs Harri flagged:

1. **Collision detection** is planned as a follow-on improvement to the
   multi-person solver, and needs "fine-grained iteration between the
   state of the persons being solved" — i.e. per-frame or near-per-frame
   coupling, not a coarse few-passes-over-a-whole-window scheme. A
   subprocess-per-pass design (spawn the CLI, wait, read CSVs, repeat)
   cannot get anywhere near that granularity — process-spawn overhead alone
   rules it out. An in-process C++ orchestrator managing multiple `Tracker`
   instances directly can iterate as tightly as needed.
2. **Combining multiple detection sequences** (markerless pose + physical
   motion-capture markers, potentially spanning multiple people in one
   marker stream) is a planned future capability. That's a data-loading
   concern more than an orchestration one, but it reinforces that the
   tracker itself — not a Python layer bolted on the side — is where
   multi-input, multi-person capability belongs going forward.

This makes Phase 5 bigger than the doc's original 3-5 day estimate (more
like C++ `feature/measurement-error-model`-scale work), but it's the right
foundation instead of something that gets thrown away once collision
detection is tackled.

The plan supports **N persons**, not just two — the motivating sequences
are two-person martial arts, but nothing in the design may assume exactly
two (team-sports captures are a plausible later workload). Everything
below is written per *ordered person pair*; with N persons there are
N·(N−1) ordered pairs, gated cheaply (see contact gating).

## The measurement model (pinned down)

This section is normative — the rest of the plan refers back to it.

For an (anchored person **B**, anchoring person **A**) ordered pair, a
marker pair (m_B, m_A), and a camera **c** in which **both markers have a
real detection this frame**:

- **Measured value**: `z = det_c(m_B) − det_c(m_A)` — the pixel difference
  of the two *detections*, exactly analogous to within-person `PAIR_DIFF`
  (`observation.hpp`: "obs.position = child_pixel − parent_pixel").
  Same-camera calibration error cancels in the difference.
- **Prediction**: `h(x_B) = project_c(m_B, x_B) − anchor_position`, where
  `anchor_position = project_c(m_A, x̂_A)` is a **constant across B's sigma
  points**, computed from A's best available state estimate (see anchor
  freshness below).
- **Coupling signal**: the innovation contains `(anchor − det_c(m_A))` —
  A's own reprojection residual — which is what pulls B's estimate into
  consistency with A's.

Two properties worth stating explicitly because they shape the gating
design:

1. **The model is exact regardless of actual contact.** It predicts the
   pixel difference correctly at *any* separation between the persons —
   there is no "the markers coincide" assumption anywhere. The world-space
   distance threshold below is purely a **relevance/cost gate** (nearby
   pairs are where the coupling carries signal worth paying for), not a
   correctness condition. Consequently, gate flicker cannot inject a wrong
   constraint; it only changes which valid observations are included.
2. **Both detections must exist in the same camera in the same frame.**
   No detection → no cross-person observation for that camera. (A
   contact *pseudo-measurement* that works without detections was
   considered and rejected: real contacts don't occur at predefined marker
   locations, and no detector currently reports contact points. Revisit
   only if such a detector materialises.)

**Noise composition**: because A's state is frozen in the prediction, the
innovation attributes *all* error to B, so the observation noise must
carry every term:

```
σ² = σ_pose,B² + σ_pose,A² + σ_anchor²
```

where σ_pose,{A,B} are the usual per-detection pose-noise terms
(`pose_noise_std × crop_scale`, per detection) and σ_anchor is A's
projected-marker uncertainty from the anchor-uncertainty machinery (Stage
3). Following the Phase 4 convention (`session_reader.cpp:1036-1051`):
bake the composed σ into `noise_std_override`, set
`confidence = min(conf_A, conf_B)` so the standard confidence scaling in
`Observation::measurement_noise_std()` applies once and only once, and
apply `cross_person_min_confidence` to both detections.

Additionally, **floor and mildly inflate σ_anchor** (implementation
constant, revisit empirically). This is the cheap guard against
*data incest*: the per-pair filters exchange information every frame with
no cross-covariance bookkeeping, which is the textbook decentralized-
fusion failure mode — mutual overconfidence, covariance collapse, and
then the Mahalanobis gate rejecting real detections. Stage 4's acceptance
criteria monitor for this explicitly (NIS and covariance condition number
inside contact windows — the MCP `get_filter_stats` tool already surfaces
both). Escalate to covariance-intersection-style conservative weighting
only if monitoring shows collapse.

## Key design decisions

**Orchestration**: new C++ class, in-process, owning N `Tracker`
instances (one per person) and driving them frame-synchronously
(Gauss-Seidel over persons within each frame). Not a Python orchestrator,
not a new binding.

**Frame-synchronous coupling without splitting `predict()`/`update()`**:
`Tracker::track_frame()` (`include/posetrak/tracking/tracker.hpp:167`) is
currently a single atomic call (predict+update combined) — splitting it
would be invasive. Instead, the orchestrator processes persons **in
sequence within each frame**: a person's `track_frame()` uses the
just-computed **current-frame** posteriors of every person processed
before it in this frame (available via `Tracker::state()`/`covariance()`,
`tracker.hpp:175-184`), and one-frame-old (velocity-extrapolated, see
below) estimates of persons processed after it. Revisit with a real
predict/update split only if this proves insufficient.

**Rotating processing order**: rotate the person order every frame
(with N persons, rotate the sequence by one each frame). This converts
the who-processes-first asymmetry from a persistent one-sided bias into a
zero-mean alternation, and slightly weakens the incest loop. Costs
nothing.

**Anchor freshness — velocity extrapolation for the stale direction**:
when person P's anchor comes from a person processed *after* P this frame,
the freshest available estimate is that person's frame-(t−1) posterior —
one frame stale, and the target sequences (throws) are fast. The state
already contains velocities: extrapolate the stale posterior one frame
forward with the same constant-velocity model the process model uses,
then FK-project the extrapolated state for the anchor. Nearly free,
removes most of the lag bias. (Fallback if this ever proves inadequate:
additionally inflate σ_anchor by an estimate of ‖marker velocity‖·dt in
pixels.)

**Anchor mechanism reuses `PAIR_DIFF`, not a new `MeasurementMode`**:
`PAIR_DIFF`'s existing prediction code (`src/filters/ukf.cpp:1862-1883`)
subtracts a *reference marker's* per-sigma-point reprojection from the
child's. A cross-person anchor is structurally identical except the
reference is a **fixed external pixel value** (constant across sigma
points) rather than another marker reprojected fresh per sigma point. Add
an optional `anchor_position` field to `Observation` (alongside the
existing `ref_marker_id`, `include/posetrak/core/observation.hpp:40`) and
branch on it inside the existing `PAIR_DIFF` handling: if
`anchor_position` is set, use it directly as the (sigma-point-constant)
reference instead of reprojecting `ref_marker_id`. No new enum value.
Noise via `noise_std_override` per the measurement-model section above.

Unlike Phase 3/4 relative observations (built at load time in
`SessionReader`), cross-person observations are built **at runtime by the
orchestrator**, because they depend on the evolving state estimates. The
orchestrator therefore keeps each person's raw per-camera detections for
the current frame at hand (it already has them — it feeds them to
`track_frame()`).

**Contact gating — three levels, coarse to fine** (each frame, per
person pair):

1. **Bounding-box pre-gate**: per person, an axis-aligned bounding box
   over its current FK marker positions (already computed each frame),
   inflated by `cross_person_max_world_mm`. Only person pairs whose boxes
   intersect proceed. With ~60 markers/person this is what keeps the
   common no-contact case at O(N²) box checks instead of O(N²·M²)
   distance computations.
2. **Marker-pair distance gate**: for surviving person pairs, pairwise 3D
   world distances between the two persons' FK marker positions; keep
   pairs under `cross_person_max_world_mm`. Apply **hysteresis** so the
   active set doesn't flicker frame-to-frame (enter at d < T, exit at
   d > ~1.2·T — implementation constant, not a config knob). Per the
   measurement-model section, flicker is an NIS-continuity/cost concern,
   not a correctness one — hysteresis keeps the observation-set
   composition stable.
3. **Candidate cap** (mirrors Phase 4's pattern at
   `session_reader.cpp:1031-1034` exactly): with ~60 markers per person
   (hands/fingers included) the under-threshold pair count can still be
   large during close contact — thousands of candidate pairs are possible
   across cameras. The cap operates on **per-camera candidate lists**: for
   each camera, a marker pair from gate 2 is a candidate only if **both
   markers have a detection in that camera this frame with confidence ≥
   `cross_person_min_confidence`** (the measurement-model section's
   both-detections-required condition — pairs occluded or low-confidence
   in a given camera never enter that camera's list). Sort each camera's
   candidates by 3D world distance and keep at most `cross_person_max_n`
   **per person pair per camera per frame** (default 10, same as
   `cross_pair_max_n`).

**Identity-switch guard**: contact frames are exactly where 2D detectors
swap or merge keypoints between persons; a cross-person anchor can make a
swapped detection look *consistent* and lock the error in. Cross-person
observations are therefore always fully subject to the standard
Mahalanobis outlier gate — **never** `force_inlier`. (Optional refinement
if v1 shows swap-induced failures: skip the anchor when B's detection
lies closer to A's predicted marker than to B's own prediction — the
signature of a swap.)

**Per-marker anchor uncertainty — Jacobian in production, sigma-point as
test oracle** (revised from the earlier sigma-point-in-production choice
after costing it): nothing today computes per-marker positional
covariance (only whole-state `cov_condition_number`/`nis_value` exist,
`src/io/statistics_tracker.cpp:44-52`). The sigma-point route —
regenerate posterior sigma points, reproject all 2n+1 through FK +
`Camera::project_undistorted()` — costs a fresh Cholesky plus ~2n+1 full
FK evaluations per person per frame (n ≈ 200 error DOFs), i.e. roughly
another full update's FK bill; FK is the dominant per-frame cost (that's
why `u_fk1_ms`/`u_fk2_ms` instrumentation exists). Instead:

- **Production**: linearized propagation. Per contact marker, pixel
  covariance ≈ `J P Jᵀ` with `J` the 2×n Jacobian of the
  FK-then-project map (Pinocchio frame Jacobians are cheap once
  `computeJointJacobians` has run; chain with the camera projection
  Jacobian). Cost: one small GEMM per contact marker. Computed
  **lazily** — only for markers in currently-active contact pairs, only
  for cameras that gate through.
- **Test oracle**: the sigma-point computation, implemented once in test
  code, validates the Jacobian version (projection is near-linear at
  post-convergence covariance scales, so they should agree closely).

**World-space contact detection inputs**: the per-frame 3D world marker
positions come from the same FK the tracker already runs per frame
(`run_parent_step()` refreshes `fk_` post-update). `Tracker` needs a
small public accessor for current marker world positions — new API,
listed under files touched.

**RTS smoothing**: run independently per person, after the full
frame-synchronous coupled forward pass completes for all persons — i.e.
the "smooth-after-all-iterations" option from the original doc, not
"smooth-then-re-solve". Simpler, and the per-frame coupling during the
forward pass already gets most of the accuracy benefit; revisit
smooth-then-re-solve only if contact-window residual error after v1
doesn't look good enough. Note the smoother cache is O(frames·edim²)
*per person* (~1 MB/frame at edim ≈ 200), so memory scales linearly with
person count — acceptable for 2-3 persons, and the long-term answer for
team-sized N is a fixed-lag (sliding-window) smoother (deferred, see
below).

**No `cross_person_max_iter` in v1**: the original doc's bounded
window-re-solve loop requires snapshot/restore of full tracker state
(state + covariance + timestamp + `prev_observations_` + NIS-feedback
windows + smoother-cache truncation), which `Tracker` does not have
(`initialize_from_state()` takes only a `State`; covariance is
re-initialized from config). A **`Tracker` checkpoint/restore API is a
separate, already-planned work item** with independent use cases
(re-running part of a sequence after time-local edits); window re-solve
becomes feasible once that lands and is deferred until then. Do not ship
a config knob that has no backing implementation.

**CLI — no TOML**: the TOML config mechanism is legacy (pre-session-DB)
and Harri wants it phased out, so the multi-person mode must not extend
it. Use CLI11's fixed-arity multi-value options:
`app.add_option("--person")->expected(4)` giving repeatable
`--person <sequence> <skeleton> <tracker_config> <person_id>` as four
space-separated values — no separator character at all, so no conflict
with Windows drive-letter colons or quoting. Longer term the natural
end-state is DB-driven (`--db <path> --person-id N --person-id M ...`,
everything else resolved from the session DB, which already stores
tracker config — see `session_reader.cpp:164-265`); that belongs with the
multi-input-sequences data-loading work, not this phase.

## New config fields

All on `TrackerConfig` (`include/posetrak/core/config.hpp`, alongside
`cross_pair_max_px`/`cross_pair_max_n`), all following the established
"0 = disabled" convention, and all wired through the session-DB config
path (`session_reader.cpp` column mapping) as well:

- `double cross_person_max_world_mm = 0.0` — marker-pair 3D distance
  gate; 0 = feature disabled.
- `double cross_person_min_confidence = 0.5` — both detections must pass.
- `int cross_person_max_n = 10` — max cross-person observations per
  person pair per camera per frame, closest-first (mirrors
  `cross_pair_max_n`).

(Hysteresis exit factor and σ_anchor floor/inflation are implementation
constants in v1, not config — avoid knob bloat until tuning shows they
need exposure.)

## Scope for this phase (staged)

**Stage 1 — `MultiPersonTracker` C++ orchestrator harness, no coupling yet.**
New `include/posetrak/tracking/multi_person_tracker.hpp` +
`src/tracking/multi_person_tracker.cpp`: owns N `Tracker` instances (one
per person, each already fully self-contained per today's `Tracker`
design — own UKF/FK/IK/Pinocchio model, per
`docs/cpp-architecture-overview.md`), drives them frame-synchronously
(`for frame: for person: person.track_frame(...)`), writes per-person
results (reuse existing `ResultWriter`/`TrackingExporter`, one instance
per person, unchanged). New CLI mode in `cli/track.cpp` with the repeated
4-value `--person` flag described above. Structure the orchestrator's
inner loop so per-person work is a self-contained callable — persons are
fully independent outside contact windows (each owns its Pinocchio
model/data, so per-instance thread safety is clean), and running them on
threads for non-contact frames is a near-free ~N× later; don't bake the
sequential interleave into deep call structure.

Before this stage: **audit `Tracker`/UKF for shared statics/globals** —
they were written assuming one instance per process. One already found:
`static int call_count` in `ukf.cpp` (~line 1888, debug-only; will
interleave across instances and confuse debug output, though not
numerics). The bitwise-identical check below catches numeric shared
state; logging/debug statics need the audit.

*Verification*: run two people's sequences through the new mode with
contact detection disabled; confirm output is bitwise-identical to
running each person through today's single-person `track` command
separately.

**Stage 2 — Contact gating + anchor injection.**
Add the three config fields. Implement the three-level gate (bbox
pre-gate → distance gate with hysteresis → per-camera closest-first cap)
and runtime anchor-observation construction per the measurement-model
section: composed noise in `noise_std_override`, `confidence = min(pair)`,
never `force_inlier`. Injection timing: anchor observations for person P
at frame t are built **immediately before P's `track_frame(t)` call**,
using current-frame posteriors for persons already processed this frame
and velocity-extrapolated frame-(t−1) posteriors for the rest. Contact-set
maintenance (gate levels 1-2 with hysteresis) runs once per frame after
all persons complete, from their frame-t FK outputs, and determines the
active pairs for frame t+1. Rotate the person processing order each
frame.

*Verification*: unit tests — (a) two synthetic skeletons out of range:
zero cross-person observations built; (b) two skeletons within threshold:
each person receives `anchor_position`-populated observations citing the
other's markers, with correctly composed noise and confidence; (c) more
candidate pairs than `cross_person_max_n`: closest-first cap enforced per
camera; (d) three persons: all gated ordered pairs produce observations
(no hardcoded two-person assumptions); (e) hysteresis: distance
oscillating around T does not toggle the active set every frame.

**Stage 3 — Per-marker anchor uncertainty.**
New method on `Tracker` (e.g. `marker_projection_std(camera_id, marker_ids)`
— exact shape TBD during implementation) computing the Jacobian-based
pixel std described above, lazily for the requested markers only. Feed
into Stage 2's noise composition as σ_anchor. Also the new
current-marker-world-positions accessor if not already added in Stage 2.

*Verification*: unit test with a small, known covariance — the Jacobian
std must closely match a sigma-point-reprojection std computed
independently in the test (the test-oracle role), and match an analytic
value for a hand-constructed near-linear case.

**Stage 4 — Config wiring + RTS smoothing + surfacing.**
Wire `--smooth` to run each person's independent RTS pass after the full
coupled forward pass. Session-DB config columns + `session_reader.cpp`
mapping for the three new fields. Python UI/MCP surfacing
(`run_tracker.py` multi-person controls, `content_panels.py`
contact-window summary, `app/mcp/tools/runs.py` `describe_config`) — same
shape as every other phase's UI wiring, done last once the C++ side is
solid.

## Explicitly out of scope for this phase (flagged, not solved)

- **Collision detection / response**: this architecture is built to
  accommodate it (fine-grained in-process iteration), but actually
  detecting and responding to interpenetration is separate follow-on work.
- **`Tracker` checkpoint/restore API** (and with it, bounded contact-window
  re-solve / `cross_person_max_iter`): separate planned work item with
  independent use cases (re-running part of a tracking sequence after
  time-local edits). Window re-solve is deferred until it exists.
- **Combining multiple detection sequences per tracker run** (markerless +
  marker-based, potentially multi-person marker streams): a data-loading
  concern (`SessionReader`/`ObservationSet`), related but distinct from
  this phase's orchestration work. The DB-driven multi-person CLI
  end-state belongs with this work too.
- **True same-frame symmetric coupling** (splitting `predict()`/`update()`):
  deferred per the rotating-order Gauss-Seidel approximation above;
  revisit only if it proves insufficient.
- **Joint-state (merged) UKF**: still deferred, same reasoning as the
  original doc (O(n²) sigma-point cost, complex merged-state RTS
  smoothing).
- **Fixed-lag (sliding-window) RTS smoother**: the answer to smoother-cache
  memory when person count grows (team sports); makes memory
  O(lag·edim²) per person regardless of sequence length. Later
  enhancement.
- **Parallel per-person tracking on non-contact frames**: deferred, but
  Stage 1's loop structure must keep it easy (see above).
- **Covariance-intersection weighting for the anchor exchange**: only if
  Stage 4 monitoring shows covariance collapse in contact windows.

## Files likely touched

- `include/posetrak/tracking/multi_person_tracker.hpp` (new),
  `src/tracking/multi_person_tracker.cpp` (new)
- `include/posetrak/core/observation.hpp` (`anchor_position` field)
- `src/filters/ukf.cpp` (~line 1862-1883, extend `PAIR_DIFF` prediction
  branch for fixed external anchors)
- `include/posetrak/tracking/tracker.hpp` / `src/tracking/tracker.cpp`
  (marker-projection-std accessor + current-marker-world-positions
  accessor for Stages 2-3)
- `include/posetrak/core/config.hpp` + `src/core/config.cpp`
  (`cross_person_max_world_mm`, `cross_person_min_confidence`,
  `cross_person_max_n`)
- `src/db/session_reader.cpp` (+ session-DB schema migration) — new
  tracker-config columns
- `cli/track.cpp` (multi-person mode, repeated 4-value `--person` flag)
- New C++ tests: multi-person harness bitwise-identical check, contact
  gating unit tests (incl. N>2, cap, hysteresis), anchor-uncertainty
  Jacobian-vs-sigma-point check
- Python/UI wiring last (`run_tracker.py`, `content_panels.py`,
  `app/mcp/tools/runs.py`) — Stage 4

## Verification approach

- Stage 1: bitwise-identical output vs. today's single-person runs (no
  contact detection enabled) — regression safety net for everything after.
- Stages 2-3: new C++ unit tests (`./run_tests.sh`), following this
  codebase's existing Catch2 conventions (see `tests/test_ukf_update.cpp`
  for `PAIR_DIFF` test precedent, `tests/test_skeleton_layout.cpp` for
  `hierarchy_distance` precedent).
- Stage 4 integration: a real two-person contact sequence (per the
  original doc's acceptance criteria):
  - mean 3D wrist distance during contact lower than today's independent
    per-person tracking;
  - no NIS regression on non-contact frames;
  - **no covariance collapse inside contact windows** — watch NIS/DOF and
    covariance condition number over the contact window via the MCP
    `get_filter_stats` tool (guards against the data-incest failure mode);
    if NIS drops well below 1 and stays there while the condition number
    climbs, the anchor-noise inflation constant needs raising.
  - Use `optbuild/` for this (per CLAUDE.md, debug build is too slow for
    real timing/tuning work).

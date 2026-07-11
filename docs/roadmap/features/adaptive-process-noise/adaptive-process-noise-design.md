# Adaptive process noise — design note

> **Status (2026-07-11)**: A full-trial A/B comparison (Mechanisms A+B
> enabled vs. both disabled, all other config identical) across three people
> found **on is worse than off for every person, on every conditioning
> metric, no exceptions** — avg NIS/DOF, % steps overconfident, average and
> worst-case covariance condition number all regress, most severely for the
> least-manually-cleaned data (Timo: 11x worse avg condition number; one
> single-step spike 5,593x worse). Not yet isolated whether Mechanism A,
> Mechanism B, or their interaction is responsible. No decision made yet on
> whether to revert, retune, or keep either mechanism. Full writeup, tables,
> and a per-mechanism keep/kill initial lean:
> `docs/roadmap/features/tracking-crisis-debugging-log.md`, sections
> "Adaptive process noise (Mechanisms A+B) on/off comparison" and "Mechanism
> inventory — keep/kill initial lean." This significantly changes the
> confidence level behind the *Phasing* section below — treat Phase 1/2 as
> "implemented but now under review," not "done and validated," until that
> follow-up work lands.

> **Status (2026-07-07)**: Phase 1 (Mechanism A) is implemented and validated
> on Case 1. Validating it surfaced an unplanned regression (arms/hands losing
> tracking under a body-wide gain) and, after fixing that by scoping the gain
> to specific joints, a new failure case (Case 3) on the now-excluded joints.
> This revision documents both and revises Mechanism B's design in response —
> see the *Status* callouts inline below. No code changes yet for the Mechanism
> B revision; this is the sketch to review before implementing it.

## Goals

Reduce the two related UKF failure modes already observed in real captures, where
a fast, large-amplitude motion (a throw, a deep forward bend) either gets rejected
outright by the outlier gate or destabilises the filter's covariance, because
`process_noise_std` / `process_noise_vel_std` are single, static values chosen as
a compromise between "tight enough to track calm motion precisely" and "loose
enough to keep up with a throw." No single static value is both.

Non-goals for the first pass:
* Replacing the outlier gate itself (Mahalanobis threshold / `outlier_threshold`)
  — that's a separate, already-tracked concern (see *Motivation* below).
* A full multi-model filter (IMM). Considered, deliberately deferred — see
  *Literature background*.
* Changing sigma-point generation or the UKF's core predict/update structure.

---

## Motivation: two real diagnosed failures

**Case 1 — forward bend not tracked** (run `61ab65e7-c436-44f0-ab43-d4f6c97c9aab`,
t≈59.03s, person bends ~90° at the hip). `get_filter_stats` showed `NIS/DOF`
spiking to 2.5–7.5 while `cov_condition_number` stayed flat at its normal
baseline (~221k) — the filter is *overconfident*: its own uncertainty estimate
never grew to accommodate the actual motion, so real, correct, high-residual
observations from three cameras were rejected as outliers (`n_inlier_observations`
collapsed from ~450 to as low as 136) and the tracked torso stayed near its
prior (upright) estimate through the whole bend.

**Case 2 — hip throw, right leg not following** (run
`84b141ce-35c0-42b3-88ee-1c9337c9578a`, t≈45.6s). Here `cov_condition_number`
itself blew up to as high as 3.8×10⁷ during the fast leg-swing motion, which
defeated the Mahalanobis gate in the *other* direction: wildly wrong detections
got accepted as inliers (small Mahalanobis distance despite 2000+px pixel gaps)
while genuinely corrected observations were rejected.

These are different symptoms (overconfident-and-static vs. ill-conditioned) but
the same underlying cause: one static `process_noise_std`, chosen for "typical"
motion, is a bad fit for the actual range of dynamics in a mocap capture —
standing still, walking, and a fast martial-arts throw or bend all appear in the
same trial.

**Case 3 — bilateral hand-raise not tracked, on a joint deliberately excluded
from Mechanism A** (run `5dff7e33-feba-4164-929e-cd629912a45a`,
`ukemi-tommi-20260509.db`, t≈59.06s, both hands raise). Found while validating
Phase 1, in two steps:

1. An early Phase 1 config applied the joint gain body-wide (no scoping).
   This *did* fix Case 1, but visual review found a new regression it
   introduced: arm/hand tracking lost during ordinary fast gesture, confirmed
   at three separate timestamps across two body-wide-gain runs (e.g. right
   arm lost at t≈59.08s, left arm at t≈60.0s). Root cause: `vel_ref_joint` was
   tuned around the torso/hip's slow typical velocity; arms routinely move
   faster than that even during mundane motion, so the same reference engaged
   far more readily for arms than intended, over-loosening their process
   noise until tracking lock was lost.
2. Fixed by scoping the joint gain to a literal joint-name list
   (`process_noise_vel_joint_names`, empty = all joints = original
   behaviour) — deliberately name-based rather than skeleton-group-based,
   since existing skeleton YAMLs don't define groups fine-grained enough (one
   "main" group spans the whole body) and adding a finer split would mean
   editing every person's skeleton file. Excluding arms preserved the Case 1
   improvement in full (NIS/DOF 2.68 vs. 2.65 unscoped) and even further
   improved covariance conditioning — no tradeoff on the original problem.

But excluding arms reopened, on arms specifically, exactly the failure
Mechanism A exists to fix: as both hands rise, wrist Mahalanobis distance
climbs steadily from ~1.5 (t≈58.98s) to over 6 across roughly 20 steps
(~0.17s), crosses `outlier_threshold=6.0` at step ~500-501, and the wrist
observations get rejected. From that point the filter has no correction
signal for the wrists at all and cannot recover in the forward pass, however
correct the detections remain.

Critically, the whole-skeleton `NIS/DOF` aggregate does **not** surface this:
across the 2-second window containing this exact rejection, aggregate
NIS/DOF for this run was 2.68 — unremarkable, since 2 markers' worth of
residual is diluted across the other ~59. This is the direct motivation for
scoping Mechanism B by joint-group rather than computing it whole-skeleton
(see *Mechanism B*, revised below).

Tracing Case 3 further back found a proximate trigger one step upstream of
everything above: `spine1` alone absorbs the forward-bend rotation until it
hits its own configured joint limit, at which point the fit degrades faster
than root rotation (correctly, but not fast enough) can compensate. That's a
kinematic-redundancy problem — root, `spine1`, and `spine2` all rotate the
same torso markers, and nothing currently prefers a sensible split among
solutions that fit about equally well — not a process-noise problem per se,
though it's what triggers the process-noise failure above. See
`docs/roadmap/features/pose-regularization/pose-regularization-design.md`
for a separate, complementary mechanism targeting that trigger directly.

---

## Current implementation (baseline)

`UnscentedKalmanFilter::process_noise_` (`src/filters/ukf.cpp:69`) is a single
`error_dim() × error_dim()` diagonal matrix, rebuilt only by
`rebuild_process_noise()` (`ukf.cpp:83-112`):

- **Global**: `pos_var = base_noise_std_²` and `vel_var = vel_noise_std_²` are
  each a single scalar applied as `scalar * Identity(active_dof, active_dof)`
  across *every* active position/velocity DOF (`ukf.cpp:94-98`) — there is no
  per-joint distinction today, only a special-cased override that freezes
  prismatic DOFs to zero unless calibration mode is active (`ukf.cpp:100-111`,
  the one place today that already loops `layout_->joints()` and writes
  individual DOF entries — the pattern any per-DOF work would extend).
- **Static**: `rebuild_process_noise()` only re-runs when `set_vel_noise_std()`
  or `enable_calibration_mode()` is called — not once per `predict()`. For a
  normal tracking run it is computed once from `TrackerConfig::process_noise_std`
  / `process_noise_vel_std` (`include/posetrak/core/config.hpp:81-83`, loaded in
  `src/db/session_reader.cpp`) and never changes again.
- Added into the covariance every step at `covariance_ += process_noise_ * dt;`
  (`ukf.cpp:513`) — the actual injection point any time-varying scheme has to
  feed into, every `predict()` call.

A related but distinct existing mechanism: exponential velocity damping
(`vel_half_life_s_`, applied at `ukf.cpp:284-294`) decays sigma points'
velocity after propagation, to bound *unobserved* velocity-covariance growth.
It already runs every `predict()` step and is a useful reference for "a
per-step adjustment keyed off `dt`," but it doesn't address the static-Q
problem — it damps velocity uncertainty growth, it doesn't widen position/angle
process noise in response to actual motion.

`State` already exposes per-DOF velocities (`joint_velocities()`,
`root_velocity()`, `root_angular_velocity()`), and `JointDesc` already maps each
joint's DOFs to both a `State::joint_velocities()` index (`state_index`) and an
error-state index (`error_index`, `include/posetrak/core/skeleton_layout.hpp:33-58`)
— the two pieces a velocity-driven per-DOF scheme needs are both already in
place, just not connected to `process_noise_`.

---

## Literature background

- **Singer model / "Current Statistical Model"** (Singer 1970; Zhou & Kumar
  1984, maneuvering-target radar tracking) — models each DOF's acceleration as
  a correlated random process whose variance scales with the *currently
  estimated* velocity/acceleration magnitude. Naturally per-DOF, and
  *proactive*: it widens uncertainty as soon as motion picks up, without
  waiting for a bad residual first. Closest fit to what's proposed below.
- **Sage-Husa adaptive filtering** — recursively re-estimates Q (and R) from a
  moving window of innovation residuals. Reactive, well-studied pitfall: an
  unconstrained recursive estimate can drift toward a non-positive-definite Q
  and needs explicit clamping.
- **Strong Tracking Filter** (Zhou & Frank, 1996) — a per-step "fading factor"
  that inflates the *predicted* covariance based on how far the actual
  innovation covariance departs from its theoretical value. A principled,
  provably-stable version of "scale process noise up when NIS/DOF is high" —
  directly matches the symptom in Case 1 above, and reuses data the project
  already computes (`nis_value`, `nis_dof` per step).
- **Interacting Multiple Model (IMM)** — run parallel filters (e.g. "calm" vs
  "maneuvering" process models) and blend by mode probability. The most
  capable option and the standard answer to exactly this calm-vs-fast-motion
  problem in the tracking literature, but a materially bigger change (parallel
  sigma-point sets, mode mixing across two full filters). Deliberately deferred
  — see Phase 3.

---

## Proposed design

### Mechanism A — velocity-driven per-DOF process noise (primary, proactive)

At each `predict()` call, before the process-noise block is added to
`covariance_`, compute a per-DOF scale factor from that DOF's *current*
velocity estimate (the `posterior_state` already saved at `ukf.cpp:156-157`,
before prediction runs), Singer-model style:

```
noise_std_dof = base_noise_std * (1 + k * |velocity_dof| / velocity_ref)
```

(exact functional form — linear vs. saturating — is an open question, see
below) applied independently to:
- each joint's active DOFs, via the same `layout_->joints()` / `error_index`
  loop already used to freeze prismatic DOFs (`ukf.cpp:100-111`), reading
  velocity from `State::joint_velocities()[state_index]`;
- the root's position/orientation DOFs, from `root_velocity()` /
  `root_angular_velocity()`, likely with its own gain — root moves in world
  units (metres), joints in radians, so a shared gain would conflate two
  different natural scales (see *Open questions*).

This turns `rebuild_process_noise()` from "rebuild on config change" into
"rebuild every `predict()` from the current velocity state" — cheap, since it's
only per-DOF diagonal writes, not a propagation-cost change.

New `TrackerConfig` fields (`config.hpp`, loaded in `session_reader.cpp` next to
the existing `process_noise_std` fields): a velocity gain `k` and a reference
velocity scale, at minimum; possibly separate root/joint gains per the open
question above.

### Mechanism B — regional NIS-feedback fading safety net (secondary, reactive)

> **Status (2026-07-07), revised from the original whole-skeleton design**:
> Case 3 above shows the original design (one NIS/DOF number for the entire
> skeleton, one fading multiplier applied everywhere) has a dilution problem
> — confirmed empirically, not just in theory: aggregate NIS/DOF across the
> Case 3 window was 2.68, unremarkable, despite a real, severe rejection
> confined to two wrist markers happening inside that exact window. A
> whole-skeleton Mechanism B would very likely have missed Case 3 for the
> same reason the aggregate metric did when we first went looking for it.
> Not yet implemented — this is the revised design to review before coding.

A Strong-Tracking-Filter-style catch-all for whatever Mechanism A's velocity
heuristic doesn't anticipate (a sudden contact/collision event that isn't a
smooth velocity ramp, an occlusion-driven jump, or — Case 3 — a body region
Mechanism A doesn't cover at all because it's scoped out).

**Revised design: track NIS/DOF per noise scope, not whole-skeleton.** A
*scope* is the same kind of literal joint-name list Mechanism A already uses
(`process_noise_vel_joint_names`) — the simplest starting point is to reuse
whatever scopes Mechanism A is already configured with (e.g. "core" and
"arms"), so a residual spike confined to the arms shows up in the arms' own
windowed NIS/DOF, undiluted by an unaffected torso/legs.

For each scope: track a short moving window (e.g. last 5-10 steps, needs
tuning — see *Open questions*) of NIS/DOF computed only from observations
whose marker's parent joint falls in that scope. This requires attributing
each step's per-observation Mahalanobis contribution to a scope — a
bookkeeping addition over `ObservationResult` data the update step already
computes, not a change to the UKF's predict/update math itself (consistent
with this note's non-goals). If a scope's windowed average exceeds threshold
— reuse the `1.5` `_NIS_HIGH` value the MCP diagnostic server already uses
(`app/mcp/tools/diagnostics.py`), keeping "is this filter healthy" consistent
between offline diagnosis and the tracker's own runtime behaviour — apply a
temporary multiplier λ > 1 to *that scope's* process noise for the next
`predict()`, decaying back to 1 as its windowed NIS/DOF returns to nominal.

This targets the actual pathology directly (model overconfident relative to
real motion → good observations rejected → no recovery) rather than a proxy
for it (raw velocity), so unlike Mechanism A it doesn't need a body-part-
specific velocity reference tuned at all — which is exactly what got Mechanism
A into trouble on arms in the first place. The natural allocation this
suggests: joints excluded from Mechanism A (arms) get Mechanism B only;
joints Mechanism A already handles well (core) keep both, with B as the
safety net for whatever A's velocity heuristic doesn't anticipate. Worth
confirming once B actually exists to test against, rather than assuming it
now (see *Open questions*).

---

## Validation plan

- `tests/regress.toml` plus the existing per-step `nis_value` / `nis_dof` /
  `cov_condition_number` logging is the harness — no new instrumentation needed
  to *measure* the effect, only to compute the new scale factors.
- Re-run tracking on the two segments this design note is motivated by (the
  forward bend in `61ab65e7…` at t≈59.03s, the hip throw in `84b141ce…` at
  t≈45.6s) before/after, comparing NIS/DOF distribution and inlier percentage
  in the affected windows specifically.
- Equally important: verify a *calm* segment (standing, walking) does **not**
  regress — this is the concrete reason Mechanism A must be per-DOF rather than
  a single global scalar: a global scale-up during the bend would also loosen
  the standing leg's tracking precision for no benefit, since it never
  accelerated.
- Add a debug export for the computed per-DOF scale factors (mirroring the
  existing `debug_enabled_` / `get_frame_number()` CSV dumps already in
  `ukf.cpp`), so the tuning process doesn't require rebuilding to inspect what
  the gain is doing frame-to-frame. **Done** — `process_noise_velocity_scale.csv`,
  per predict() call, one column per active DOF.
- **Case 3's segment is now the primary regression target for Mechanism B**:
  run `5dff7e33-feba-4164-929e-cd629912a45a`, `MRK-wrist.L`/`MRK-wrist.R`,
  t≈58.98-59.2s. Mahalanobis distance for both wrists must stay under
  `outlier_threshold` through the hand-raise *without* widening
  `vel_ref_joint` for arms in Mechanism A (that reintroduces the original
  over-loosening regression Case 3's own history started from).

---

## Phasing

**Phase 1 — velocity-driven per-DOF process noise (Mechanism A). Done
(2026-07-06/07).** `rebuild_process_noise()` takes the current velocity state
and computes a true per-DOF diagonal instead of one scalar per block;
`process_noise_vel_gain_joint`/`_gain_root` and `_ref_joint`/`_ref_root`
config fields. *Validated* on Case 1: inlier count and covariance
conditioning improved monotonically as gain went 0→1→2 (NIS/DOF avg
3.41→3.09→2.65), with zero regressions against a stashed pre-change baseline
across the full C++ test suite.

**Unplanned extension — joint-name-list scoping for Mechanism A. Done
(2026-07-07).** Validating Phase 1 with a body-wide gain surfaced a new
regression (arm/hand tracking lost during ordinary fast gesture — see Case 3)
before Case 3's own failure was even found. Fixed by adding
`process_noise_vel_joint_names` (empty = all joints, unchanged behaviour) so
the joint gain can be scoped to specific joints by literal name. Excluding
arms fully preserved the Case 1 improvement and further improved covariance
conditioning, confirming the scoping itself costs nothing on the original
problem — the tradeoff only shows up on the excluded joints (Case 3).

**Phase 2 — regional NIS-feedback fading safety net (Mechanism B). Revised
design below, not yet implemented.** Reuses the joint-name-list scopes from
Phase 1's extension instead of a whole-skeleton NIS/DOF aggregate — see
*Mechanism B* above for why the original whole-skeleton design would likely
have missed Case 3. *Validation*: on Case 3 (wrist rejection during a
bilateral hand-raise), confirm the "arms" scope's windowed NIS/DOF triggers
the fading factor before Mahalanobis crosses `outlier_threshold`, and that
the wrist stays an inlier through the motion; confirm the fading factor stays
≈1 in nominal segments; re-verify Case 1 and Case 2 don't regress; re-verify
a calm segment doesn't regress.

**Phase 3 (only if 1+2 measured insufficient) — chain-scoped / IMM-lite.** If
per-DOF scaling under-reacts for a motion that's coupled across several joints
at once (a whole-arm swing, say), group DOFs by kinematic chain and drive
Mechanism A/B off chain-level statistics instead of strictly per-DOF. Largely
subsumed by Phase 2's scope mechanism already borrowing this same
chain/group-scoping idea (applied to Mechanism B rather than a full IMM) —
revisit only if Phase 2's scopes turn out too coarse even at joint-name-list
granularity. Explicitly gated on Phase 1+2 turning out insufficient in
practice; not worth building ahead of that evidence.

---

## Open questions

1. **Linear vs. saturating velocity→noise mapping.** A pure linear scale risks
   runaway noise if a velocity estimate itself is briefly wrong (e.g. right
   after an outlier-heavy frame). Needs empirical tuning against the
   regression harness rather than a decision up front.
2. **Root vs. joint gain.** Should the root (floating base, world-unit metres)
   share a gain with joint angles (radians), or does that need two independent
   knobs given the different natural scales?
3. **Interaction with `vel_half_life_s_`.** Should velocity-driven process
   noise decay on the same half-life as the existing velocity damping, or are
   the two mechanisms better kept orthogonal (one bounds unobserved velocity
   covariance growth, the other reacts to observed motion)?
4. **Interaction with the outlier gate itself.** This design note treats
   `outlier_threshold` / the Mahalanobis gate as out of scope, but Case 2 above
   shows process-noise tuning and gate behaviour are coupled (an
   ill-conditioned covariance defeats the gate regardless of Q). Worth
   revisiting once Phase 1/2 are measured — a better Q might reduce how often
   the gate's own failure mode is even reached. Case 3 is a second, direct
   example of this coupling: the wrist rejection *is* the gate's failure mode
   being reached, not a separate issue.
5. **Scope granularity for Mechanism B.** One scope per Mechanism-A-excluded
   joint list (e.g. a single "arms" scope covering both sides), or finer
   (left/right arm tracked separately, so a one-sided event isn't diluted by
   an unaffected other side)? Case 3 was bilateral, so a single "arms" scope
   would have caught it — but a one-sided reach might not, at that
   granularity. Needs more real examples before deciding either way.
6. **Does Mechanism A still apply to excluded joints at all, or does
   Mechanism B fully replace it there?** Current thinking (see *Phasing*) is
   B-only for arms, A+B for core — worth confirming this is actually the
   right allocation once B exists to test against, rather than assuming it
   now.
7. **Implementation cost of per-scope NIS.** Today `nis_value`/`nis_dof` are
   single whole-body-per-step scalars (`UpdateResult`). Per-scope tracking
   needs attributing each observation's Mahalanobis contribution to a scope
   via its marker's parent joint, then windowing/decaying per scope somewhere
   above the UKF (`Tracker`, most likely, since scopes are a Mechanism-A/B
   config concept, not intrinsic to the UKF's own state). Real but bounded
   work — a bookkeeping restructure, not a new algorithm.

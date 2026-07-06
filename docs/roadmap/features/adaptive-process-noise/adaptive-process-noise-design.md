# Adaptive process noise — design note

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

### Mechanism B — NIS-feedback fading safety net (secondary, reactive)

A Strong-Tracking-Filter-style catch-all for whatever Mechanism A's velocity
heuristic doesn't anticipate (a sudden contact/collision event that isn't a
smooth velocity ramp, an occlusion-driven jump). Track a short moving window
(e.g. last 5-10 steps) of `NIS/DOF` — already computed per step
(`UpdateResult`, surfaced today via `tracking_results.nis_value`/`nis_dof` and
the MCP diagnostic server's `_NIS_HIGH = 1.5` threshold,
`app/mcp/tools/diagnostics.py`). If the windowed average exceeds threshold,
apply a temporary multiplier λ > 1 on top of Mechanism A's per-DOF Q for the
next `predict()`, decaying back to 1 as NIS/DOF returns to nominal.

Reusing the same `1.5` threshold the diagnostic tooling already uses would keep
the "is this filter healthy" definition consistent between offline diagnosis
and the tracker's own runtime behaviour.

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
  the gain is doing frame-to-frame.

---

## Phasing

**Phase 1 — velocity-driven per-DOF process noise (Mechanism A).**
`rebuild_process_noise()` takes the current velocity state and computes a true
per-DOF diagonal instead of one scalar per block; new `process_noise_velocity_gain`
(and reference-velocity) config fields. *Validation*: on the forward-bend and
hip-throw segments, verify the spine/leg DOFs' own noise visibly grows while
uninvolved DOFs (the other side's standing leg, e.g.) stay near baseline;
verify NIS/DOF and inlier% improve in the affected window without moving in the
calm baseline segment.

**Phase 2 — NIS-feedback fading safety net (Mechanism B).** Track a short
NIS/DOF window in `Tracker`, compute a fading multiplier against the `1.5`
threshold already used by the MCP diagnostics, apply on top of Phase 1's
per-DOF Q. *Validation*: fading factor stays ≈1 in nominal segments (no
behavioural change when nothing is wrong); rises and decays smoothly around a
real anomaly window without oscillating or itself causing `cov_condition_number`
to blow up.

**Phase 3 (only if 1+2 measured insufficient) — chain-scoped / IMM-lite.** If
per-DOF scaling under-reacts for a motion that's coupled across several joints
at once (a whole-arm swing, say), group DOFs by kinematic chain — reusing the
`SkeletonLayout::from_groups()` machinery already built for child filters —
and drive Mechanism A/B off chain-level statistics instead of strictly
per-DOF. Explicitly gated on Phase 1+2 turning out insufficient in practice;
not worth building ahead of that evidence.

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
   the gate's own failure mode is even reached.

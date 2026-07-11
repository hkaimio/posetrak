# Tracking crisis debugging log — ukemi-tommi trial

> Working notes, not a design doc. Persisted so findings survive context
> resets across this multi-session debugging effort. Update as investigation
> continues; prune stale/superseded entries rather than letting it grow
> unbounded.

**Trial**: `D:\mocap\vanhaa\ukemi-tommi-20260509.db`, sequence
`a5da88ea-f7ba-4e0e-bbd4-43c68205dcf6`, skeleton
`ec1e64533914a48c3a4abeb17383f5d2bf68af89b88696fa9f589ff8c5579200`, person 0.
Full observation range: t=38.079-66.433s. Motion is a martial-arts
ukemi (breakfall/roll) with fast bilateral arm-raise moments.

**Original motivating case**: bilateral hand-raise not tracked well —
started the whole investigation arc (pose regularization →
adaptive-process-noise Mechanisms A/B → arm-scoped/proximal-distal gain
splits → soft joint limits).

---

## Config lineage (tracker_configs, chronological)

| config id | what changed |
|---|---|
| `5e3208d8-a7f0-4d5d-b67f-d8e0622f2086` | Best config prior to this arc's work: pose-reg (spine1/spine2) + Mechanism B (NIS-feedback, max_mult=3) + single "arms" adaptive-noise scope (gain=1.0, ref=3.0) covering all arm+finger joints |
| `02ec1166-e338-41cf-859e-fc46e319a0e8` | First attempt at splitting the "arms" scope: proximal (shoulder/upper_arm/forearm/thigh/shin) ref=1.5, distal (hand/foot/toe/fingers) ref=3.0. **Regressed** — see below |
| `e7bb4c0b-6473-4f6c-a0f9-acc699177ba0` | Fixed split: proximal ref=3.0 (unchanged from old single-scope value), distal ref=5.0 |
| `dc1be02f-ed4f-4b29-b6c4-441ab457987e` | e7bb4c0b + soft joint-limit repulsion on `upper_arm.L`/`upper_arm.R`, margin=0.1222 rad (~7°), noise_std=0.03, all 3 axes (x/y/z, wherever that joint has a configured limit) |

## Run lineage (tracking_runs, chronological, only ones referenced in findings below)

| run id | config | window | notes |
|---|---|---|---|
| `1437b02d-c4a2-42bb-a477-d6c92822d154` | 5e3208d8 | full trial (t=38.0-66.5s) | Used `--start-time 38.0`, slightly before actual data (38.079s) → bad IK init, first ~2.8s corrupted. Source of the step-1631 trace below |
| `9c7a5323-3078-4079-a6cb-92e4ac5eb4ca` | 5e3208d8 | 10s window | Baseline for crisis-window comparisons |
| `193a80a3-2c61-4c09-8139-c19c972d8bb8` | 02ec1166 | 10s window | Regressed run — wrist.R max_mahal 3.59→14.48, 42/148 outlier obs |
| `824154de-1e8c-499a-9a5c-e544c9e148b3` | e7bb4c0b | full trial, `--start-time 38.1` | "Before soft-limit" baseline. PSD fired 603/3389 (17.8%) |
| `f6dc7126-a700-4ff2-8dfa-0b775f56e8e0` | dc1be02f | full trial, `--start-time 38.1` | "After soft-limit" run. PSD fired 470/3389 (13.9%). Being visually reviewed now (see *New observations*) |

All full-trial runs since `824154de` use `tracker_step` at 120Hz with
`t = 38.1 + tracker_step/120`.

---

## Diagnosed root causes (confirmed)

### 1. Bad IK initialization from boundary start-time
`--start-time 38.0` is before the sequence's actual first observation
(38.07946s) → first frame has zero triangulable observations → catastrophic
NIS explosion in the first ~2.8s. **Fix**: use `--start-time 38.1` or later
(anything safely past 38.08s). Not a tracker bug, a CLI usage error.

### 2. Step-1631 event (t≈51.59s, run `1437b02d`) — rejection cascade
`MRK-wrist.L`/`MRK-elbow.L` go from healthy (mahal 1-5) to badly rejected
(mahal 22-36) starting step 1635, recover by step 1710. `MRK-shoulder.L`
stays healthy — the cascade is specifically forearm/wrist. Root-arm marker
(elbow.R in a later run's analogous event) is consistently the first to
degrade. **Not fully root-caused** — covariance conditioning didn't show an
obvious single trigger in this instance (oscillates 5×10⁵-3×10⁶ both before
and after).

### 3. Missing CSV frames 1945-1951 (run `1437b02d`)
Genuinely absent from `tracking_stats.csv` (confirmed via set-difference, not
a UI artifact). Near-total occlusion in that window (~3 observations at
frames 1940-1944) but raw `pose_observations` table has 573 rows in that
window — so it's an upstream confidence/visibility filter dropping
observations before they reach the tracker, not literally no camera data.
**Not root-caused** — would need to trace `has_sufficient_observations()`
and the CLI's stats-export path, not yet examined.

### 4. Crisis B (t≈58-61s / step 2388-2748 in `824154de`) — "arms completely lost"
The severe, sustained crisis this whole mechanism-tuning arc targeted.
Traced in detail:
- Camera coverage is **good** throughout (4.0-4.4 of 5 cameras average,
  never below 2) — ruled out occlusion.
- Raw 2D detections stay smooth/continuous frame-to-frame (checked
  `MRK-wrist.R` actual_x/actual_y, steps 2448-2568) — ruled out bad/noisy
  observations.
- **Covariance condition number jumps 5.3× at step 2517→2518** (508K→2.71M,
  min eigenvalue 2.50e-6→9.99e-7), one full frame *before* `upper_arm.R`
  first touches its limit at step 2519, and before the inlier count crashes
  at step 2520 (397→254).
- `MRK-elbow.R` (attached to `upper_arm.R`/`forearm.R`) is the first marker
  to show elevated mahalanobis (3.87 at step 2518), consistent with
  `upper_arm.R` being the joint destabilizing first.
- Both `upper_arm.L` and `upper_arm.R` hit their limits within the same
  1-2 frame window — consistent with a genuinely fast bilateral motion.
- **Adaptive process noise (Mechanism A) was making the overshoot worse, not
  better**: scaling noise *up* with velocity right as the joint approaches
  its hard wall widens the sigma-point spread exactly when a narrow one
  would behave better. Explains why no Mechanism A tuning variant
  (arms-scope, proximal/distal split, distal ref 3.0→5.0) ever touched this
  crisis.
- `enforce_joint_limits()`'s hard clamp has a real inconsistency:
  `damp_velocity_covariance_at_limits()` only shrinks *velocity* covariance
  for a clamped DOF, leaves *position* covariance untouched despite a
  potentially multi-radian deterministic override — a plausible contributor
  to the *oscillatory* recovery pattern (repeated clamp/release bursts
  through t≈66s) rather than a clean one-time correction.

### 5. Soft joint-limit repulsion (Phase 1, implemented and tested) — see `docs/roadmap/features/soft-joint-limits/soft-joint-limits-design.md`
Built and validated against crisis B specifically to test whether the
joint-limit clamp was *causing* or merely a *symptom* of the crisis.
**Result: symptom, confirmed.** With soft limits enabled on
`upper_arm.L`/`upper_arm.R` (margin 7°, noise_std=0.03):
- `upper_arm.R` never hits its hard limit at all across the whole run (was:
  recurring multi-frame bursts). `upper_arm.L` down to 8 isolated single-to-
  4-frame instances total, none near the crisis window.
- PSD-eigensolver firing dropped 603/3389 (17.8%) → 470/3389 (13.9%) —
  real, general improvement in covariance health.
- **But crisis B's mahalanobis/outlier stats are essentially unchanged**:
  max mahalanobis still 115-221σ across wrist/elbow markers (was 115-219σ),
  and wrist.R/elbow.R outlier counts got *worse* (165→324, 133→282).
- **Conclusion**: the joint-limit clamp was a downstream symptom of crisis
  B, not its trigger. The actual trigger — whatever destabilizes covariance
  at step 2517→2518, before anything hits a limit — is still unidentified.
  This was the live open thread — **see the multi-axis-corner finding
  under *New observations* below, now the leading candidate.**

---

## New observations (visual QC pass, run `f6dc7126`, soft-limit enabled) — 2026-07-08

User watched the full-trial soft-limit run in the UI and flagged (not yet
investigated unless noted):

- **~step 200**: weird divergence — init looks good, then shortly after the
  *whole skeleton* goes wrong, including root position. **Checked,
  root-caused.** `upper_arm.L` jumps discontinuously on **all three axes
  simultaneously** between frames 227→228 (x: 2.36→-0.42, y: -0.43→0.70,
  z: -1.86→0.24 rad — up to 151° in a single 8.3ms step), landing within
  5-7° of all three of its configured limits at once — the *same*
  multi-axis-corner signature as crisis B. `MRK-wrist.L`/`MRK-elbow.L`
  mahalanobis explode immediately after (7-10 → 25-38 by frame 230); root
  position/orientation then show large single-frame deltas (up to 0.23m /
  23°) escalating from frame ~235, peaking ~297-298. Confirms the
  multi-axis-corner pathology isn't specific to the "arms raised in front"
  motion — it recurs whenever the Kalman update lands `upper_arm` near a
  box corner, in a completely different part of the motion (t≈39.9s vs
  crisis B's t≈58-61s). Strengthens the case for fixing the corner problem
  generally (see *Proposals* below) rather than treating crisis B as an
  isolated case.
- **~step 800**: both arms/hands incorrect. User hasn't seen this specific
  failure in earlier runs — possibly new, or newly visible because an
  earlier fix removed something that was masking it. Not yet checked.
- **steps 1925-1939**: tracking solution appears to freeze — no visible
  change in the pose for ~14 frames — then snaps to a correct-looking pose
  at step 1940. **Checked, root-caused — confirmed to be the same
  underlying event as the missing-CSV-frames finding from run `1437b02d`**
  (frames 1945-1951 there vs 1933-1939 here — exactly a 12-frame /
  0.1s offset, matching the two runs' `--start-time` difference, 38.0 vs
  38.1). Observation count collapses to 3 (from ~14-25) at frame 1923,
  stays at exactly 3 through frame 1932, then **frames 1933-1939 are
  entirely absent from `tracking_stats.csv`** (confirmed via presence
  check, not a UI artifact), then recovers to 118 observations at frame
  1940 (NIS=331, a real but non-catastrophic one-time correction as the
  filter "catches up" after ~15 frames of unconstrained coasting). This is
  a genuine near-total-occlusion event in the source video (same moment,
  both runs) — the "freeze" is the filter correctly free-running on
  constant-velocity extrapolation with almost no correcting data, not a
  bug in the UKF itself. The *CSV export gap* (why rows 1933-1939 don't
  appear at all, vs. just having very few observations like 1923-1932
  do) is still unexplained — would need
  `has_sufficient_observations()`/the CLI's stats-export path, not yet
  examined.
- **Crisis B, refined blow-by-blow** (matches the already-diagnosed
  t=58-61s window, now with more granular sub-events from visual review):
  - step 2476: left hand raises up incorrectly
  - step 2483: right hand does the same
  - step 2516: right leg swings backward incorrectly
  - step 2534: everything suddenly converges
  - step 2570: right hand diverges again — user's own read: "looks like an
    incorrect measurement getting through the outlier gate" (a genuine
    bad-detection hypothesis, distinct from the internal-divergence
    mechanism diagnosed above — worth checking directly against raw
    detections for this specific sub-event)
  - step 2657: converges
- **steps 2680-2707, and again 2750-2827**: the "bend forward" artifact from
  earlier sessions is back (this is the spine1/spine2 kinematic-redundancy
  symptom pose-regularization was originally built for — worth checking
  whether pose-reg is actually engaged/effective here, or whether this is a
  different recurrence). The second occurrence (2750-2827) is also where
  arm tracking is lost badly a second time.
- **steps 3000-3040**: hips and legs unstable. User's own assessment:
  "likely not related" to the arm crisis, lower priority.
- **~step 3120**: arms lost again when the person raises them, converges
  soon after. Same "arms raised" motion pattern as crisis B.

**User's hypothesis on the "arms raised in front of body" pattern**
(crisis B, step 3120, possibly others): could this be a *different* axis's
joint limit — x or z on `upper_arm.L/R`, not y (which the soft-limit
mechanism already covers and has been shown not to be the driver of crisis
B specifically)? Local axis orientation in the skeleton file isn't
confirmed by inspection alone; worth checking whether these events show the
joint sitting *near* (not necessarily *at*) its x/z limit while rotating
far on the other axis — a near-limit conditioning effect distinct from
literally hitting the wall, which the soft-limit mechanism's margin (7°)
may or may not be wide enough to catch.
**Status: checked, confirmed and stronger than hypothesized.** Computed
per-axis distance-to-limit (not just boolean at-limit) for `upper_arm.L/R`
across crisis B. Important correctness note: `upper_arm.L`'s z-axis limits
are **mirrored** relative to `upper_arm.R`'s (`L: z∈[-2.618, 0.349]` vs
`R: z∈[-0.349, 2.618]`, confirmed by reading the skeleton YAML directly) —
x and y are the same for both sides. Using per-side-correct limits:

- **Crisis B, frames 2470-2540**: `upper_arm.L` is within 5.7° (x), 4.0°
  (y), and 6.2° (z) of its limits — **all three axes simultaneously** near
  their bounds. `upper_arm.R` the same shape: 5.9°/5.1°/6.2°.
- **Crisis B, frames 2560-2660**: `upper_arm.L` x gets to within **2.2°**
  of its limit (frame 2566) — inside the soft-limit margin, so the
  mechanism should be actively engaging there too, not just on y.
  `upper_arm.R` x/y both within ~7°; z is farther out (37°) at its closest.
- **Step-3120 event**: does *not* show the same pattern — only
  `upper_arm.R`'s y gets close (6.6°), x and z are both 45-60°+ away on
  both sides. Likely a different or milder mechanism than crisis B, not
  the same triple-axis-corner effect.

**Interpretation**: during crisis B specifically, both shoulders are being
pushed toward a *corner* of their box-constrained rotation limits (2-3
axes near their bounds simultaneously), not just one wall. Box-constrained
per-axis Euler-style limits are known to behave badly near corners —
small changes in the *actually achievable* rotation there require large,
correlated changes across multiple axes, which plausibly breaks the UKF's
local-linearity assumption for the sigma-point cloud region-wide, not just
along one axis. This is a stronger, more specific candidate for crisis B's
still-unidentified trigger (the covariance-conditioning jump at step
2517→2518) than a single-axis limit ever was — and explains why the
soft-limit mechanism (independent per-axis pulls) didn't fix it: pulling
each axis back independently doesn't resolve a corner where the
*combination* is the problem, and a per-axis-independent margin can also
have its own multi-axis interaction effects that were never tested (all
three axes engaging at once, potentially fighting each other or the real
observations in ways an independent-per-axis design doesn't account for).
**Not yet fixed** — would need either a genuinely joint (multi-axis)
near-limit penalty, or reconsidering whether box-constrained per-axis
limits are the right representation for this ball joint at all (a
conical/spherical soft region might avoid the corner pathology
entirely — ties into the existing "not anatomical accuracy" caveat, since
real shoulder range-of-motion isn't box-shaped either).

---

## Proposal 1 (near-limit process-noise damping) — implemented and tested, net negative

Implemented `UnscentedKalmanFilter::set_near_limit_damping()` /
`apply_near_limit_damping()`: shrinks process-noise variance for a DOF
whose current covariance-implied spread (`mean ± spread_sigma·sqrt(var)`,
not just the mean) already reaches close to a configured hard limit —
composes with Mechanism A/B in `predict()`. Schema v33 (additive,
`near_limit_damping_joint_names`/`margin_rad`/`spread_sigma`/
`damping_factor`). 5 new unit tests, full suite otherwise unchanged (see
below for a false-start and its lesson).

Tested on top of the soft-limit config (`dc1be02f`), `upper_arm.L/R`,
margin=0.1222 rad, spread_sigma=3.0, damping_factor=0.3, full trial.
**First attempt (`e69fa339`, run `0e297b47`) was silently a no-op**: the
config row was cloned *before* the v33 migration was applied to the live
DB, so its `near_limit_*` columns were `NULL` and the mechanism never
engaged — results were bit-for-bit identical to the soft-limit-only run.
Caught because *everything* matched exactly (PSD count, the frame 227/228
jump values, all crisis A/B/C mahalanobis/outlier numbers) — a
suspiciously perfect match is itself a signal to check whether the
mechanism actually ran. Fixing this by `UPDATE`-ing the existing config
row was correctly blocked by the auto-mode safety classifier (that row
was already referenced by a completed, analyzed run — editing it in
place would have corrupted that run's historical parameter record).
Created a fresh config (`2f49c6b1`) instead, verified the columns were
actually populated before relaunching. **Lesson**: after any DB schema
migration, verify a just-created config row's new columns directly before
relying on them, especially when cloning from an older row — the ADD
COLUMN's NULL-fill-for-existing-rows silently propagates through a
`SELECT *`-based clone if the clone happens before the ALTER TABLE.

**Corrected run** (`c04401c8`, config `2f49c6b1`) vs. the soft-limit-only
baseline (`f6dc7126`, PSD 470/3389):
- **Hard-limit clamp events: fully eliminated.** `upper_arm.L` down to 0
  (from 8), `upper_arm.R` stays at 0. Damping achieves its narrow goal
  more completely than soft limits alone.
- **The frame 227→228 corner jump still happens, essentially unchanged.**
  Same magnitude (~150° across x/y/z simultaneously), same frame, lands
  in the same corner region (within a few degrees of the earlier landing
  point). Confirms the hypothesis from the design discussion: this jump
  is the Kalman *mean* snapping between two locally-competitive solutions
  for a genuinely ambiguous 3-DOF fit, not a wide-sigma-spread/
  linearization artifact — so shrinking process noise doesn't touch it.
- **Crisis B (t=58-61s) is a mixed bag, not an improvement.** Max
  mahalanobis is unchanged within noise across all 4 markers (still
  115-220σ). Outliers: `elbow.R` clearly improves (282→177), `elbow.L`
  roughly flat (204→212), but `wrist.L`/`wrist.R` clearly *worsen*
  (226→335, 324→374). Crisis A and C are flat/noise-level in both
  directions.
- **Overall covariance conditioning got modestly worse**: PSD-eigensolver
  fired 506/3389 (14.9%) vs. 470/3389 (13.9%) for soft-limit-only — a
  ~7.7% increase, the opposite of the intended effect. Plausible
  explanation: shrinking process noise for a joint already under stress
  makes the filter overconfident in that DOF, so real fast motion there
  produces larger effective innovations relative to the (now
  underestimated) uncertainty, which the Joseph-form update has to
  reconcile — plausibly increasing how often the covariance needs the
  eigenvalue-floor repair elsewhere in the state too.

**Conclusion: net negative as configured — not adopted.** Eliminates hard
clamps essentially completely, but doesn't move the crisis it was aimed
at, makes a mixed and slightly negative net difference to per-marker
outliers, and makes overall covariance conditioning modestly worse. The
underlying theory (wide sigma spread near a corner breaks local
linearity) may still be partially true, but it isn't the dominant
mechanism behind crisis B or the frame-227/228 jump — those look driven
by genuine solution ambiguity in the Kalman *mean* update itself, which
no process-noise-domain mechanism (wider *or* narrower) can address.
Reinforces that the multi-axis-corner problem needs a fix at the geometry
level (Proposal 2 or 3 from the earlier discussion — parent-joint
redistribution or a cone/swing-twist limit representation), not further
tuning of the sigma-point spread.

---

## Correction: skeleton limits must be read from the DB, not a worktree YAML

Discovered while checking `shoulder.L/R` slack (as a diagnostic for "would
parent-joint redistribution help crisis B / the step-200 event"): the
skeleton actually used by this trial (`skeleton_id`
`ec1e64533914a48c3a4abeb17383f5d2bf68af89b88696fa9f589ff8c5579200`) is
stored in the DB's `skeletons.yaml_content` column, and it's a
**per-person scaled skeleton** ("Scaled from run 7f93ddf3" per its
`source` column) — not necessarily identical to any on-disk YAML in a
worktree. Verified: `upper_arm.L/R`'s limits in the DB skeleton do match
what was used throughout this whole log's analysis (x/y/z all confirmed
identical) — the crisis-B and step-200 corner findings above stand
unchanged. But **`shoulder.L/R`'s actual limits are much tighter than the
worktree file's** and are *not* mirrored L/R (both sides identical):
`x=[-0.12, 0.6]` (-6.9° to 34.4°), `y=[-0.4, 0.4]` (±22.9°),
`z=[-0.6, 0.6]` (±34.4°) rad.

**Finding**: `shoulder.L/R` sits *exactly* at these hard limits repeatedly
and extensively during both crisis B and the step-200 event (values like
x=0.600, y=-0.400, z=±0.600 recur across many frames in both windows —
bit-for-bit matches to the configured bounds). **Shoulder has no slack to
redistribute to in many of the frames where `upper_arm` is near its own
corner** — this weakens the case for a naive parent-redistribution fix
(Proposal 2 below) as originally framed, and raises a different
possibility: since this is a *scaled*, person-specific skeleton, these
limits may simply be too tight for this person's real range of motion in
this specific gesture (e.g. a calibration pose set that didn't include a
full overhead/forward arm raise) — a data/calibration problem, not
something any Kalman-filter-side mechanism (soft limits, damping,
redistribution) can fully compensate for. **Not yet investigated further**
— would need comparing these limits against the person's actual
observed range of motion elsewhere in the trial, or against how the
scaling pipeline derived them.

**Methodology fix for future checks**: always pull joint limits via
`sqlite3` on the `skeletons` table (`SELECT yaml_content FROM skeletons
WHERE id=?`, then parse as YAML) for the specific `skeleton_id` in use,
never assume an on-disk file matches without verifying.

---

## Karcher-mean convergence investigation — 2026-07-09

Following the swing-twist design sketch (see
`docs/roadmap/features/swing-twist-joint-limits/swing-twist-joint-limits-design.md`),
tested a specific hypothesis: that the frame-227/228 jump (and possibly
crisis B) is caused by `compute_state_mean()`'s hard-capped 10-iteration
Karcher/geodesic mean failing to converge robustly once sigma points'
raw axis-angle representations start wrapping around the π-radian
topological boundary.

**Instrumentation**: added per-frame, per-spherical-joint diagnostic
logging to `compute_state_mean()`'s SPHERICAL branch (`karcher_mean_diagnostics.csv`
under `<output_dir>/debug/<run_id>/`, gated on `debug_enabled_`):
`nominal_angle_deg` (pre-iteration mean), `sigma_angle_min/max_deg` (the
range of *actual* rotation angles across all sigma points for that
joint), `sigma_rawvec_dist_max_rad` (max Euclidean distance between any
sigma point's raw stored vector and the nominal — the direct measure of
representation-level spread, as distinct from spread in the actual
rotation), `iterations_used`, `final_error_deg` (Karcher-mean convergence
quality), and `mean_angle_deg` (the resulting prior mean). 5 new lines,
compiles clean, full UKF test suite (49 cases) still passes. Ran the
full trial (config `dc1be02f`, soft-limit only, `--debug`) —
`tracking_run_id fd702b17-4356-4c74-a9af-1c81d788b667`, 3389/3389 tracked,
PSD fired 470/3389 (identical to the earlier non-debug run of the same
config — confirms the instrumentation is side-effect-free).

**Result: the specific non-convergence hypothesis is refuted, but a
related and more precisely-located finding replaces it.**

1. **`iterations_used` never exceeds 2** across the entire trial,
   including frame 227/228 and crisis B. **`final_error_deg` is
   effectively 0** (≤6×10⁻⁵°) every single frame. The Karcher-mean
   iteration converges essentially immediately and cleanly everywhere
   tested — it is *not* the site of the corruption.
2. **`sigma_rawvec_dist_max_rad` spikes dramatically and specifically at
   frame 228**: 6.089 rad (349°) — versus 1.46 rad at frame 227 (the
   frame before), and versus a *typical baseline range of 0.4-1.4 rad*
   measured at quiet frames (~100, ~1000) and throughout crisis B
   (0.42-0.61 rad, frames 2505-2525). This is a clean, unambiguous, ~4-15×
   outlier, occurring exactly when `nominal_angle_deg` (162.65°→174.55°)
   and `sigma_angle_max_deg` (176.79°→179.95°, i.e. sigma points reaching
   essentially all the way to π) put the sigma cloud right at the
   representation's topological boundary. This closely matches the
   magnitude predicted by the earlier synthetic experiment (~353° for a
   similar-magnitude nominal). **Crisis B shows no elevated
   `sigma_rawvec_dist_max_rad` at any point** (stays in the unremarkable
   0.4-0.6 range) — direct, measured confirmation (not just inference
   from rotation angles staying low) that crisis B is a genuinely
   different mechanism from the frame-227/228 event.
3. **The *prior* (predicted, pre-measurement-update) mean stays smooth
   through frame 228**: `mean_angle_deg` = 174.55° at frame 228, a
   perfectly continuous extension of the climb from 227 (162.65°). The
   discontinuity to ~48.7° is *not present yet* at this point in the
   pipeline. It only shows up as the *next* frame's (229) `nominal_angle_deg`
   (48.72°) — which is derived from frame 228's *posterior* state, i.e.
   the state *after* the camera-measurement Kalman update. **The actual
   jump happens during `update()`, not `predict()`/`compute_state_mean()`.**
4. Checked whether `update()`'s cross-covariance computation could be
   the corruption site instead: it uses the same `compute_state_error()`
   function as `compute_state_mean()`/`compute_state_covariance()` (confirmed
   at `ukf.cpp:467` and `:546`), which was already verified to do proper
   SO(3) relative-rotation extraction (`R_rel = R_ref^T · R_state` then
   `quaternion_to_axis_angle`), not raw vector subtraction. So the
   cross-covariance/Kalman-gain math is not naively corrupted by the raw
   representation spread either, at least not through an obviously wrong
   computation.

**Revised interpretation**: this isn't necessarily a numerical bug in
the Kalman math (predict, mean, covariance, and cross-covariance are all
consistently SO(3)-aware wherever checked). What's more likely is that
the *raw representation spread is a symptom of, not a cause of, a real
loss of observability*: at frame 227/228, `sigma_angle_min/max_deg`
shows the sigma cloud's *actual* (not just raw-vector) rotation-angle
hypotheses genuinely spanning ~148-180° — a wide range of real
alternative poses the filter considers plausible for that DOF, reflecting
that 2D marker observations underconstrain a joint whose rotation is
already near π (a form of observability degradation inherent to a nearly
half-turn rotation, not merely representational noise). With the prior
that uncertain, a modest shift in which hypothesis the frame's camera
data best fits can produce a large, but not obviously *incorrect*, one-step
Kalman correction. If so, the fix is still the swing-twist proposal, but
via a different logical path than originally framed: not "stop a
numerically-corrupted mean/covariance from misbehaving," but "keep the
joint's operating range away from a region where genuine observability
legitimately degrades, by construction." A swing-twist-limited joint,
capped well below π by its own configured `max_swing_angle`, would never
let the filter's prior grow this uncertain in the first place.

**Not yet done / possible follow-up**: didn't instrument the `update()`
step's actual Kalman gain / innovation covariance (`S`) computation
directly to watch the correction happen frame-by-frame, nor checked
whether the *raw* observations at frame 228 specifically look anomalous
(vs. a genuinely poorly-observable-but-not-anomalous set of pixels). Would
be the natural next empirical step if this needs to be pinned down
further before committing to the swing-twist implementation.

---

## User-driven observation-data cleanup — crisis B — 2026-07-09/10

User manually reviewed raw per-camera footage for the crisis B window and
found (and disabled, via observation edits in the DB) three separate,
independent bad-observation-data issues, retested after each:

1. **Camera "pixel9", ~frame 2450**: incorrect right-hand/arm
   measurements — located above the person's head, wrongly classified as
   inliers.
2. **Camera "gopro02"**: when the person's right side faces the camera,
   the pose detector mixes up left/right hands (assigns right-hand
   detections to *both* the left and right hand labels) — disabled the
   left-hand detections from that camera.
3. **Camera "insta ace2", from ~frame 2370**: a second, untracked person
   walks between the subject and the camera; many of *that* person's
   detections were wrongly classified as inliers for the tracked person.

**Result across all three fixes, individually and cumulatively: crisis
B's core severity did not move.** Max mahalanobis for
wrist.L/wrist.R/elbow.L/elbow.R stayed pinned at ~115-220σ through every
retest — remarkably consistent (e.g. wrist.L: 116.58 → 115.93 → 117.88;
elbow.R: 221.37 → 216.93 → 215.08 — never more than a few percent of
movement). Outlier *counts* did shift per-marker (e.g. elbow.R's outlier
count dropped sharply, 282→75→82, while wrist.L/wrist.R stayed flat or
got slightly worse), and total observation count fell as expected
(1423829 → 1396876 → 1388662) as bad detections were removed — so the
fixes are doing something real to the input data, just not to the
crisis's core severity.

**But re-examining the corrected trajectory changed the diagnosis.**
Checked `upper_arm.L/R`'s rotation-vector angle (norm of the stored
axis-angle vector) across the crisis B window in the run with all three
fixes applied (`c64c50a1`, run over config `dc1be02f`): `upper_arm.L`
now climbs smoothly to **178.86° at frame 2500** (hovering 176-179° for
frames 2498-2503) before descending — this pattern is **not present** in
any of the earlier (uncorrected) runs, where the same window stayed in
the unremarkable 20-45° range. The covariance-conditioning jump
previously traced at step 2517-2518 sits just a few frames after this
peak. Unlike the frame-227/228 event, there's no discontinuous snap here
— the angle rises and falls smoothly, spending several frames right at
the boundary rather than jumping across it — but the proximity to π
itself is unmistakable and wasn't visible until the bad observations
were removed.

**Revised conclusion, superseding the "Karcher-mean convergence
investigation" section's claim that crisis B shows no π-proximity
signature**: that earlier check was run against a trajectory now known
to be distorted by three uncorrected bad-observation problems pulling
the state away from the person's actual motion. With those removed, the
real motion visibly brings `upper_arm.L/R` to the edge of the axis-angle
representation's topological boundary right before the crisis begins —
the same architectural fragility identified for frame-227/228 and
targeted by the swing-twist design
(`docs/roadmap/features/swing-twist-joint-limits/swing-twist-joint-limits-design.md`).
Current best read: crisis B and the frame-227/228 event are likely the
*same* underlying mechanism, not two separate problems as previously
concluded — the bad observations were a red herring for explaining the
crisis itself (even though fixing them was legitimate and worth doing on
its own merits), not a masking layer that happened to also obscure the
real cause underneath.

**Resolved.** User found and fixed a fourth round of bad observations in
the crisis B segment. Retest (run `79855e29`, config `dc1be02f`, full
trial): **crisis B's max mahalanobis dropped from ~115-220σ to 17-25σ**
— the same mild range as crisis A and C, down from catastrophic:

| | wrist.L | wrist.R | elbow.L | elbow.R |
|---|---|---|---|---|
| Original baseline (`f6dc7126`) | 116.58σ | 121.05σ | 152.99σ | 221.37σ |
| Fixes 1-3 (`c64c50a1`) | 117.88σ | 123.86σ | 155.41σ | 215.08σ |
| **Fixes 1-4 (`79855e29`)** | **24.76σ** | **20.87σ** | **20.03σ** | **17.04σ** |

**The π-proximity excursion found in the fixes-1-3 run is also gone**:
`upper_arm.L/R`'s max rotation-vector angle in the crisis window dropped
from 178.86°/177.76° back to 109.88°/114.29° — comfortably normal.

**This supersedes the "revised conclusion" above** — the near-π
excursion after fixes 1-3 was itself an artifact of the *remaining*
uncleaned observations (from whatever the fourth round fixed) pulling
the state toward an extreme pose, not evidence of the same architectural
mechanism as frame-227/228. Crisis B and the frame-227/228 event are
most likely unrelated after all, back to the original assessment before
the fixes-1-3 detour: crisis B was a (multi-part, four-fixes-deep) bad
observation-data problem, full stop. The swing-twist proposal remains
motivated by frame-227/228 specifically, not by crisis B.

**Methodology note**: an intermediate, partially-cleaned dataset can look
architectural (matching a real, independently-diagnosed failure
signature) purely by coincidence of *which* bad data happens to remain.
Don't treat a partial fix's residual symptom as confirming a deeper
mechanism until the data cleanup is actually exhausted — keep going
until a fix stops changing the result, not just until a fix produces a
result that superficially matches another known pattern.

---

## Adaptive process noise (Mechanisms A+B) on/off comparison, all three people — 2026-07-10/11

Prompted by a qualitative observation while reviewing other people's data:
the adaptive gain appeared to be widening process noise enough that the
Mahalanobis outlier gate was letting clear misses through, causing visible
limb "reactions" to bad observations. Tested directly: full-trial reruns for
Roosa, Tommi, and Timo, each with the current shared config (`dc1be02f…`,
Mechanism A velocity gain + Mechanism B NIS-feedback both enabled, same as
their most recent prior runs) and a cloned config
(`b257daa3906f44038f5eba24639b1267`, `parent_id`→`dc1be02f…`) with **both**
mechanisms disabled (`process_noise_vel_gain_joint`/`_root` = 0,
`process_noise_vel_scopes` = null, `nis_feedback_scopes` = null) — everything
else (base process noise, outlier threshold, pose-reg, soft joint limits)
identical. Both mechanisms were disabled together rather than isolated,
since both widen process noise and either could produce the symptom; not yet
determined which (or both) is the actual driver.

Rebuilt `optbuild` first to confirm today's AVX2 flag was included. 6 runs,
~590-620s each:

| run_id | person | config | | run_id | person | config |
|---|---|---|---|---|---|---|
| `8851a531-9534-4008-82ce-86eb4b633832` | Roosa | on | | `a7e317f4-c5e7-4be1-8024-b11b3d2fe29a` | Roosa | off |
| `cc9cff6a-e16f-4054-bd82-c7774c65221f` | Tommi | on | | `4f6daf6a-f448-4233-b748-c9842bc9575a` | Tommi | off |
| `396295ee-2bc6-4515-b6b0-032825842151` | Timo | on | | `5be2f1f4-1f6d-48b9-8093-54f0326bff25` | Timo | off |

BVH exports for all 6: `<scratchpad>/adaptive-gain-comparison/{Person}-{on,off}.bvh`
(session-scratchpad path, not repo-permanent — re-export from the run_ids
above if these have been cleaned up).

**Result: on is worse than off, for every person, on every metric, no
exceptions.**

| metric | Roosa on | Roosa off | Tommi on | Tommi off | Timo on | Timo off |
|---|---|---|---|---|---|---|
| avg NIS/DOF | 2.11 | **1.52** | 3.48 | **1.79** | 2.79 | **2.05** |
| % steps NIS/DOF > 1.5 | 47.0% | **42.7%** | 67.3% | **58.1%** | 76.0% | **71.3%** |
| avg cov condition # | 699K | **368K** | 1.09M | **406K** | 5.02M | **452K** |
| % steps ill-conditioned (>1e6) | 11.8% | **5.4%** | 22.3% | **5.9%** | 18.9% | **7.5%** |
| avg n_inlier_observations | 352.9 | 350.2 | 393.9 | 392.9 | 402.3 | 398.1 |
| tracking_lost | 0% | 0% | 0% | 0% | 0% | 0% |

Effect size scales inversely with how much manual observation cleanup each
person's data has had (least cleaned → largest effect): Timo (least) 11x
worse avg condition number, Tommi (partial) 2.7x, Roosa (most, including all
four rounds of crisis-B cleanup documented above) still ~1.9x.

**Concrete single-step spikes**, same timestamp compared directly:

| person | t | cov_cond ON | cov_cond OFF | ratio |
|---|---|---|---|---|
| Timo | 63.05s | 1.73×10⁹ | 309,298 | **5,593x** |
| Tommi | 50.66s | 1.16×10⁸ | 887,007 | **131x** |
| Roosa | 45.37s | 1.15×10⁷ | 244,774 | **47x** |

All three people's ill-conditioned (>1e6) spans under the "on" config cluster
in roughly the same t≈39-65s window — the trial's full observation range is
t=38.08-66.43s (per the *Trial* header above), so this is essentially "most
of the fast/bilateral-motion portion of the trial," not an isolated moment.
Given Roosa's sequence (`a5da88ea…`) is the exact trial this whole crisis
log is about, and her data has already had crisis B cleaned to 17-25σ
mahalanobis (see "User-driven observation-data cleanup" above) — the fact
that her "on" run *still* shows meaningfully worse conditioning than "off"
suggests this is at least partly a mechanism-level effect, not purely
residual bad data. Not proven to be the *only* driver of the noisier-looking
data the investigation started from, but a real, reproducible, cross-person
effect.

**Methodology note — a metric that didn't work**: also tried a direct
"gate-fooled" check (pixel gap between actual and predicted marker position
for observations the gate accepted as inliers). Abandoned it — distal limb
markers (wrist/elbow/ankle/knee/toe) showed enormous nominal gaps (1000px+)
in *both* configs, apparently dominated by a lens-distortion/coordinate-space
effect specific to certain cameras (GoPro views far worse than
`insta_ace2_pro`), unrelated to adaptive gain. NIS/DOF and covariance
condition number (both already-established diagnostics from earlier in this
log) were the reliable signal; raw actual-vs-predicted pixel gap was not,
at least not without an undistortion correction this check didn't do.

**Not yet decided**: whether to disable Mechanism A, Mechanism B, or both,
permanently — see the mechanism inventory below. This run only shows both
disabled together is better than both enabled; it doesn't isolate which one
(or whether it's their interaction, e.g. Mechanism B's NIS-feedback reacting
to noise Mechanism A already introduced) is responsible.

---

## Mechanism inventory — keep/kill initial lean, not decided — 2026-07-11

Every process-noise/covariance-shaping mechanism implemented during this
investigation arc, with an initial lean based on evidence gathered so far.
**None of these are final decisions** — flagging for discussion.

| mechanism | what it does | evidence | initial lean |
|---|---|---|---|
| **Mechanism A** (`process_noise_vel_gain_joint/_root`, `process_noise_vel_scopes`) | Scales process noise up proportionally to a DOF's current velocity (Singer-model style) | Validated as a net *improvement* in isolation on Case 1 (forward bend) back in Phase 1. But the on/off comparison above shows it — bundled with Mechanism B — makes full-trial conditioning meaningfully worse for all three people. Not tested in isolation from Mechanism B since Phase 1. | **Lean: reconsider / re-validate in isolation.** The Case 1 win may still be real but be outweighed by fast-motion-segment harm; or Mechanism B may be the actual culprit and A is fine. Needs an A-only vs. B-only split run before deciding either way. |
| **Mechanism B** (`nis_feedback_scopes`, NIS-feedback fading) | Reactively multiplies process noise when a scope's windowed NIS/DOF exceeds threshold | Built specifically as the Case 3 fix (arms losing tracking) and never independently re-validated against a full trial since. Plausible feedback-loop risk: high NIS → widens noise → worse conditioning → more bad observations accepted → higher NIS next step — exactly the shape of the spikes found above (up to 5,593x at a single step). Not proven, but is the more likely of the two mechanisms to runaway like this given its reactive design. | **Lean: prime suspect, test in isolation first.** |
| **Pose regularization** (`pose_reg_joint_names` = spine1/spine2) | Equal-split + rest-pose soft constraint on the spine1/spine2 kinematic-redundancy pair | Independent of this A/B test — left enabled in both "on" and "off" configs above, so this test says nothing about it either way. Built for the "bend forward" artifact (steps 2680-2707/2750-2827 in the visual QC pass), not yet confirmed whether it's actually engaging there. | **Lean: keep, orthogonal to this finding** — but still has its own open question (does it actually fire where intended?) from the visual QC pass, item 3 in *Open threads*. |
| **Soft joint-limit repulsion** (`soft_limit_joint_names` = upper_arm.L/R) | Soft repulsive penalty as a joint approaches its hard limit | Also left enabled in both configs above. Independently validated earlier in this log: real, general conditioning improvement (PSD fired 17.8%→13.9%) and eliminated `upper_arm.R`'s hard-limit clamps entirely — though it didn't fix crisis B (which turned out to be pure bad data, not a limit problem). | **Lean: keep** — real, if modest, benefit; no negative evidence found. |
| **Near-limit process-noise damping** | Shrinks process noise for a DOF whose covariance-implied spread already nears a hard limit | Already implemented, tested, and conclusively evaluated as **net negative** earlier in this log ("Proposal 1" — eliminates hard clamps but makes overall conditioning *worse*, doesn't touch crisis B). **Already not deployed**: the shared config used for the runs above (`dc1be02f…`) has `near_limit_damping_factor` unset/disabled. | **Lean: kill, already effectively done** — just needs the code path formally removed or left permanently off if there's any reason to keep it available for future experiments. |
| **Swing-twist joint limits** | Design sketch only (cone/twist limit representation instead of box-constrained per-axis limits), not implemented | Motivated by the frame-227/228 event's π-proximity signature, which is still open (not re-verified against a data-cleanup pass the way crisis B was). No new evidence from this A/B test either way. | **Lean: still open, unaffected by this finding** — see *Open threads* item 2. |

**Suggested next step** (not started): an A-only and a B-only variant (2 more
configs, 6 more runs across the three people) to isolate which mechanism —
or their interaction — is actually driving the conditioning regression,
before deciding whether to revert, retune, or keep either one.

---

## Visual QC of the on/off BVH exports + two concrete false-inlier root causes — 2026-07-11

User visually reviewed the Roosa and Tommi on/off BVH pairs. **Tommi: off
(adaptive disabled) is clearly better.** **Roosa: a tie** — consistent with
the hunch that residual data-quality issues in her sequence are muddying
the comparison, confirmed below.

User flagged two specific moments where a far-away observation appeared to
be accepted as an inlier and destabilize tracking, both in the Roosa "off"
run (`a7e317f4-c5e7-4be1-8024-b11b3d2fe29a`): `gopro-11_mini_01` step 911
(t=45.672s) and `gopro-11_mini_02` step 1884 (t=53.780s). **Root-caused,
and it's not an adaptive-noise or gate problem at all**: both are literal
`(0.0, 0.0)` pixel coordinates with confidence **1.0** (fully trusted),
written via `pose_observation_edits` on ghost frames (no backing raw
detection), `created_at` 2026-07-03/04 — predating this week's marker/
chain-placement/interpolate-missing UI work, so not caused by that. A
follow-up hygiene scan (read-only, all 10,640 edit rows in the DB, all
sequences) found exactly **3** such rows total, **all in Roosa's sequence,
none in Tommi's or Timo's**, and no other degenerate patterns (`NaN`,
extreme-magnitude coordinates) anywhere else in the trial. Narrow and fully
enumerated — see
`docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md`
for the scan details and a design sketch for related ideas (an
outlier-gate bypass for *human-verified* edited keypoints, gated on this
staying clean; plus two hand/finger-tracking quality ideas raised in the
same discussion: a dedicated hand-detection pass in the pipeline, and
automated post-edit hand redetection).

**Revises the adaptive-gain conclusion slightly**: Tommi's clean-cut result
(no known data-quality confound) is the more reliable signal of the two
BVH comparisons done so far — the earlier full-trial NIS/cov-condition
numbers stand regardless (that evidence didn't depend on these three edit
rows), but Roosa's *visual* tie is now explained by a real, separate
data-quality issue rather than adaptive gain being a wash for her too.

**Also still open, not yet investigated**: fast bilateral hand-raises still
lose tracking lock at the outlier gate — once the state has drifted, the
*correct* observations get rejected as outliers, so the arm stays stuck
rather than recovering (the original Case 1/Case 3 failure mode, still
present). Hand/finger keypoints are also frequently wrong for two other,
distinct reasons: incorrect identity assignment during grabs (common in
this aikido footage), and self-occlusion where the pose model still emits
a confident-looking guess. See the new design doc for candidate mitigations
for both — none implemented yet.

---

## Recurring pitfalls / methodology notes

- **`marker_projections.csv`'s `proj_x/proj_y` vs `obs_x/obs_y` are in
  different, non-comparable coordinate spaces.** Always use
  `tracking_obs_results.obs_blob` (`float32[n_cam, n_mrk, 8]`:
  `[actual_x, actual_y, pred_x, pred_y, mahal_dist, used, is_outlier, pad]`)
  for per-marker mahalanobis/outlier/position analysis, decoded via
  `tracking_runs.marker_names`.
- **Even within `obs_blob`, raw `pred_x/pred_y` vs `actual_x/actual_y`
  magnitude comparisons aren't always trustworthy** — hit this once and
  couldn't fully explain it; stick to the `mahal_dist`/`is_outlier` columns
  themselves, which are internally consistent.
- **`tail`/bash on an actively-written detached-process log file shows
  stale/buffered content.** Use the `Read` tool (not `tail`) to monitor
  long-running tracker run logs.
- **Detached background tracker processes in this environment can run at
  wildly variable wall-clock throughput** (observed 0.04-0.2 steps/s some
  sessions vs the process's own reported ~5 steps/s) despite the process's
  own internal timer reporting normal speed. Full-trial runs (~3400 steps)
  have taken anywhere from ~10 minutes to ~14 minutes wall-clock in
  practice. Not yet understood; just budget for it.
- **Skeleton joint limits are per-axis, stored in the YAML** (e.g.
  `.claude/worktrees/agent-a249f470222fbe397/harri-skeleton-bisect-testing-only-main-group-markers.yaml`
  for `upper_arm.R`: x=[-30°,160°], y=[-45°,45°], z=[-20°,150°]). Clamped
  values match configured bounds exactly bit-for-bit when a hard-limit
  clamp fires — a reliable diagnostic signature.
- **`enforce_joint_limits()` clamps each joint's own angle independently —
  no redistribution to parent joints.** If `upper_arm.R` maxes out, nothing
  pushes the "leftover" rotation to `shoulder.R` (the clavicle). Real
  scapulohumeral-rhythm-style compensation isn't modeled at all. Noted as a
  possible future mechanism, not attempted.

---

## Open threads (as of 2026-07-11)

0. **Adaptive process noise (Mechanisms A+B) — full-trial on/off comparison
   done, showing net-negative conditioning across all three people (Roosa,
   Tommi, Timo) — see the new section above. Not yet isolated which
   mechanism (A, B, or their interaction) is responsible, and no decision
   made on whether to revert/retune/keep either one — see the mechanism
   inventory above for an initial (non-final) lean per mechanism. Visual
   QC of the BVH exports confirms Tommi clearly favors adaptive-off;
   Roosa's tie is explained by three unrelated corrupted edit rows (found
   and fully enumerated via hygiene scan, see below), not by adaptive gain
   being a wash for her too — Tommi's result is the more reliable signal.
   The A-only/B-only isolation run is still the suggested next step.
0.5. **Hand/finger tracking quality — design sketch written, nothing
   implemented.** Fast bilateral hand-raises still lose lock at the outlier
   gate and don't recover once the state drifts (Case 1/3's original
   failure mode, still present). Hand keypoints are frequently wrong from
   identity mixup during grabs or self-occlusion. See
   `docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md`
   for three related ideas (trusted-edit gate bypass, pipeline hand-detection
   pass, automated post-edit hand redetection) and their open questions.

1. **Crisis B — resolved.** Four rounds of camera-by-camera bad-observation
   fixes (pixel9, gopro02 hand mixup, insta-ace2 second-person
   contamination, and a fourth unspecified round) brought max mahalanobis
   down from ~115-220σ to 17-25σ (see "User-driven observation-data
   cleanup" above). It was a data problem the whole time, not an
   architectural one — the mid-investigation finding that it showed the
   same π-proximity signature as frame-227/228 turned out to be an
   artifact of still-incomplete data cleanup, not a shared mechanism.
   No further tracker-side work needed for crisis B specifically.
2. **The frame-227/228 event is still open** and, as far as tested,
   remains architectural rather than a data problem (not re-verified
   against a data-cleanup pass the way crisis B was — worth a similar
   per-camera review around t≈39.9s before fully committing to the
   swing-twist implementation, given crisis B's lesson that an
   architectural-looking symptom can still turn out to be bad data).
   Near-limit process-noise damping (Proposal 1) doesn't touch it.
   Karcher-mean non-convergence is ruled out (converges cleanly always).
   What's confirmed: the sigma cloud's raw axis-angle representation
   spikes to ~4-15× its normal spread right at the π-radian topological
   boundary, and the sigma cloud's *actual* rotation-angle hypotheses
   genuinely span a wide ~148-180° range there — consistent with real
   observability degradation near a half-turn rotation. Swing-twist
   (Proposal 3, design sketch written, not implemented — see
   `docs/roadmap/features/swing-twist-joint-limits/swing-twist-joint-limits-design.md`)
   remains the best-supported fix *if* a data-cleanup pass rules out the
   same red herring that delayed the crisis B diagnosis.
3. New visual observations from the original visual QC pass, not yet
   investigated: step 800 arm/hand error, step 2570 possible
   bad-detection-through-gate, steps 2680-2707/2750-2827 "bend forward"
   (spine) recurrence, steps 3000-3040 hip/leg instability (user thinks
   unrelated, lower priority). Given crisis B turned out to be pure data
   corruption, these are worth the same per-camera-review treatment
   before assuming any of them need a tracker-side fix.
4. Missing CSV frames (root-cause item 3 above) — not traced past
   "something upstream filters near-zero observations before the tracker
   sees them."
5. Whether `shoulder.L/R`'s tight scaled-skeleton limits (see the
   correction above) are themselves miscalibrated for this person's real
   range of motion — not yet checked against other parts of the trial or
   the scaling pipeline.

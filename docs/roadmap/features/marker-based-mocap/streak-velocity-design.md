# Streak-derived dot velocity

Scope note (2026-09-05): follow-up to
[reflective-dot-detection-design.md](reflective-dot-detection-design.md)'s
"Open questions" section, which first raised a motion-blur streak's own
vector as an unused velocity signal, and to status.md's 2026-09-05 entry
(adaptive process noise closed most, not all, of the fast-swing lag gap).
Phase 1 (wire format) and phase 2 (standalone calibration/validation
script) done, validated against real data. Phase 3 (tracker integration)
built and wired into the real per-frame loop, but its first real-data
validation was net negative (§4) -- a real, diagnosed bug, not yet fixed.

## 1. Why a streak is a velocity measurement

A motion-blur streak's length is the marker's own image-plane displacement
*during the exposure window*, not the full inter-frame interval — a camera's
shutter is open for less time than a frame lasts. If `v` is the (roughly
locally constant) image-plane speed:

```
streak_length              ≈ v * exposure_time
frame-to-frame_displacement ≈ v * frame_time
```

Their ratio, `k = exposure_time / frame_time`, is independent of `v` — a
property of the camera's shutter/frame-rate settings alone, not of how fast
the object happens to be moving at any given instant. That means `k` can be
estimated once (or as a slowly-drifting running average) per camera from
*any* real, resolved motion, then applied to convert a single frame's own
streak length into a same-frame speed estimate: `v = streak_length / (k *
frame_time)`. This is valuable specifically because it needs no adjacent
frame to be reliably tracked, unlike the existing frame-to-frame `VELOCITY`
measurement mode — exactly the situation during a fast cut, where the
previous frame may itself be blurred past detection or occluded.

## 2. Wire format (done)

`dot_blob_detector.BlobCandidate` gained `dir_x`/`dir_y`: a unit vector
along a streak's own axis, derived from `cv2.boxPoints()` on the accepted
candidate's minimum-area rectangle (not from `cv2.minAreaRect`'s angle
field directly — its convention/range differs across OpenCV versions).
`(0.0, 0.0)` for a round dot.

A blur streak is a *line*, not an arrow: there's no way to recover which
end the dot started from out of the blob shape alone. `dir_x`/`dir_y` is
therefore canonicalized to a fixed half-plane (`dy >= 0`, or `dx >= 0` when
`dy == 0`) rather than left arbitrary — deterministic, but still carrying a
180-degree sign ambiguity that only gets resolved downstream, against a
predicted or previously-resolved velocity (§4).

The 'dots' blob format bumped again: `float32[N,6]` (px, py, area,
compactness, major_axis, minor_axis) → `float32[N,8]` (+ dir_x, dir_y),
same versioned count-prefix scheme as the 2026-09-04 bump
(`db_cache.encode_dot_candidates`/`decode_dot_candidates`,
`posetrak::db::decode_dot_candidates` in blob_codec.hpp). Pure blob-format
change — the `pose_observations`/`detection_keypoints` schema itself is
untouched (`keypoints`/`kp_blob` is a plain `BLOB` column with no
candidate-count or per-candidate-width constraint), so there is no DB
migration to revert if this doesn't pan out, only the codec functions.
`UnlabeledCandidate` (session_reader.hpp) carries `dir_x`/`dir_y` through
to the tracker, left in *distorted*-pixel space unlike `position` (a
direction at a point needs the local undistort Jacobian to transform
correctly, not just re-running `undistort()` on a second point — not done,
a first cut; lens distortion should be mild across a dot's own small streak
extent regardless).

## 3. Calibration: estimating k per camera (done, standalone)

`k` is estimated as a **ratio of sums**, not a mean of per-sample ratios:
`k = Σ streak_length / Σ frame_displacement` over a window of real,
resolved samples. A mean of per-sample ratios blows up as an individual
sample's displacement approaches zero — exactly the case a "some real
movement" admission gate exists to guard against; summing first is the more
robust form of the same idea (a near-zero-displacement sample still
contributes to both sums, but can't dominate the ratio on its own).

`tools/estimate_streak_exposure_ratio.py` (standalone, not yet wired into
the tracker) validates this against real data before building the
tracker-integrated version: for a resolved tracking run, it walks
consecutive `tracking_obs_results` steps per camera, and for every marker
resolved (non-outlier, non-NaN `actual_x`/`y`) in two consecutive steps,
computes the undistorted-space frame-to-frame displacement — exactly the
same quantity the existing `VELOCITY` measurement mode already uses — and
looks up that step's own raw dot candidate (matched by redistorting the
resolved position and finding the nearest raw candidate) to read its
`major_axis - minor_axis` (the streak's length beyond the dot's own
footprint — the same `elongation` quantity `resolve_dot_assignment()`
already uses to inflate measurement noise) and `dir_x`/`dir_y`. Samples
below `--min-displacement-px` are dropped before summing (the movement
gate); the rest feed the per-camera sums, plus a direction-consistency
check (mean `|cos|` between the streak's sign-resolved axis and the actual
displacement direction) as an independent sanity check that the stored
direction really does track real motion, not just that the length ratio is
plausible.

**Validated against the real sword capture** (2026-09-05, re-detected with
direction capture, tracking_run_id `86c35ea2-...` -- same 80.5% tracked
result as the pre-direction run, confirming the format bump changed
nothing else): `gopro-11_mini_02` (the more fast-motion-exposed of the two
dot-detection cameras) gave 195 real samples, `k = 0.65` (first/second-half
split 0.66/0.64 -- stable), mean direction cosine 0.99 (the stored streak
axis tracks real displacement almost exactly). `k = 0.65` at 120fps implies
a ~5.4ms exposure -- physically plausible for an indoor, ring-lit capture
pushing toward a fairly open shutter for light. `gopro-11_mini_02`'s
sibling camera only produced 7 samples (noisier, 0.61/0.47 half-split) --
not concerning on its own, but not yet explained why one ring-lit camera
saw so much less matchable fast motion than the other pointed at the same
scene; worth another look before trusting a per-camera `k` for it
specifically.

## 4. Tracker integration (built, wired in, net negative on first real run)

No new UKF residual math was needed: `MeasurementMode::VELOCITY` already
computes `h(x_t) = project(x_t) - project(x_{t-1})` from the saved
posterior state alone (`ukf.cpp`'s `prev_projections` — confirmed it does
*not* require an actual t-1 *observation*, just a projectable posterior),
exactly the shape a streak-derived measurement needs. Built as:

- `StreakKAccumulator` (`streak_k_accumulator.hpp`/`.cpp`): the same
  ratio-of-sums windowed estimator as §3's standalone script, as plain,
  Tracker-agnostic data (mirroring `resolve_dot_assignment()`'s own "pure
  core, directly testable" split). `Tracker` owns one map of these, keyed
  by camera_id, alongside `prev_observations_` (`streak_k_accumulators()`,
  mutable accessor).
- `resolve_dot_assignment()` takes an optional `StreakVelocityConfig` (a
  `TrackerConfig::dot_streak_*` bundle — disabled by default) plus a
  `PrevDotPositions` (each subject's own `Tracker::prev_observations()`,
  gathered by `resolve_shared_dot_assignment()`). For a resolved streaked
  candidate with a real previous position for that exact (subject, camera,
  marker) slot: computes the real frame-to-frame displacement, admits
  `(elongation, disp_px)` into that camera's `StreakKAccumulator` once past
  the movement gate, and — once `k` is trusted — emits a second,
  `VELOCITY`-mode `Observation` alongside (not instead of) the usual
  `POSITION` one, with measured value `streak_length / k` along the
  streak's own axis.
- Sign ambiguity resolved against the real measured displacement just
  computed (`streak_dir` flipped if it points the wrong way), not a
  predicted velocity — simpler than the original plan sketch, and
  consistent with what the offline validation actually checked (§3's
  direction-cosine numbers are against real displacement too, not a
  prediction).
- `TrackerConfig`/`TrackerAppConfig` gained six `dot_streak_*` fields;
  `tracker_configs` (session schema v50 / registry schema v9) gained the
  matching columns, same `ALTER TABLE`-per-migration pattern as every
  earlier tunable (e.g. `process_noise_vel_gain_root`).

**Real validation (2026-09-05, tracking_run_id `a5ce3a4c-...`, config
`sword-streak-velocity-v1` — gain=4/ref=2 adaptive root noise + face
culling, same as the validated `86c35ea2-...` baseline, plus
`dot_streak_velocity_enabled=1` with every other `dot_streak_*` field at
its default): 79.1% tracked (5234/6618) vs. baseline's 80.5% (5325/6618) —
**a net regression**, not the hoped-for improvement, despite the
mechanism's own real signal being genuine: comparing the two runs
step-by-step, streak velocity newly recovered 77 previously-lost steps
*and* newly lost 168 previously-good ones (net -91, matching the aggregate
delta). The known 53.6–55.1s ArUco gap window itself was unaffected either
way (0/150 lost in both runs — already perfect at this baseline, so not
where the difference shows up).

**First cause theory (`prev_observations_` contamination) — retracted.**
Originally hypothesized `Tracker::prev_observations_` records every
observation's position unconditionally regardless of the UKF's own outlier
verdict, contaminating the next frame's streak-velocity reference. Real
attempted evidence for this (rendering debug videos of the regression
window and reading `tracking_obs_results.obs_blob` to look for bad
matches) turned out to be looking at a **separate, confirmed bug** instead
(caught by Harri reviewing the videos directly — predicted markers tracked
the real sword correctly while "actual" markers were rendered floating
over blank wall, nowhere near any possible detection, which is the tell
that it's a data bug rather than a real-but-wrong correspondence): this
feature is the first thing to ever give one (camera, marker) two
Observations in the same step, and that broke three places in the
codebase that assumed at most one, including one able to affect the real
outlier-rejection decision. Full account and fix in
`docs/roadmap/features/observation-results-semantics.md`'s "Two
observations per slot" section. The original `prev_observations_` theory is neither confirmed nor
ruled out yet — it needs re-investigating with the now-fixed diagnostics,
since every video and query used to support it was reading corrupted data.

**Re-checked after the fix (2026-09-05, re-tracked runs `b4328ce1-...`
baseline / `17456a44-...` streak, same 80.5%/79.1% numbers as before —
confirming the obs_blob fix didn't change either run's real outcome):
scanning every dot observation's nearest real raw candidate distance,
filtered to `mode==POSITION` only, in both windows for both runs --

| window | baseline max | streak max |
|---|---|---|
| gap (53.6–55.1s) | 61px | 61px (was 1135px pre-fix — artifact, gone) |
| regression (63.6–64.3s) | **934px** | 22px (was 1139px pre-fix — artifact, gone) |

The streak run's own "hundreds of pixels" is now fully explained as the
obs_blob bug — nothing left to investigate there. But **baseline's own
934px anomaly in the regression window is real and unrelated to streak
velocity or this bug** (every entry checked was genuinely `mode==POSITION`
already) — a real bad match, most likely the originally-suspected
Mahalanobis-gate-too-loose mechanism: the aggressive adaptive-root-noise
tuning (gain=4) inflates covariance enough during extreme uncertainty that
the same fixed chi-squared gate becomes very permissive in raw pixels,
letting a statistically-plausible-but-visually-wrong match through. This
is a pre-existing characteristic of the already-validated baseline
tuning, exposed by looking here, not something streak velocity caused —
worth its own look someday, but out of this feature's scope.**

## 5. Open questions

- **(top priority, partially resolved 2026-09-06)** Re-diagnose the net
  regression (77 recovered / 168 newly lost). A real chunk of it turned out
  to be a confound unrelated to streak velocity itself: `outlier_threshold`
  left at its default (5.991) was rejecting large-but-genuine innovations
  during fast swings as outliers, faster than the filter could otherwise
  recover (see status.md's 2026-09-06 entry for the full account). Raising
  it to 20 shrinks the baseline/streak-velocity gap from -91 steps to -48
  (80.5%->84.2% baseline, 79.1%->83.5% streak-velocity) -- real, validated
  on the same sequence, but streak velocity still trails baseline by 48
  steps even with this fixed, so it isn't the whole story. The
  `prev_observations_` contamination theory (§4) remains open and
  unconfirmed for whatever gap remains; needs a fresh look, ideally
  re-run with `outlier_threshold=20` on both configs from the start so the
  now-known confound isn't mixed into the remaining diagnosis.
- Movement-gate threshold (`--min-displacement-px`) and minimum sample
  count before trusting `k` are both first cuts, not yet tuned against how
  quickly `k` actually converges/stays stable on real footage.
- Whether `k` needs to vary over a capture (auto-exposure changing with
  scene brightness) or is safely constant for a whole session — the
  windowed/running-average design assumes the latter is at least
  approximately true; not yet checked against a real capture with lighting
  changes.
- The streak-derived `VELOCITY` observation's own noise model (distinct
  from the elongation-based inflation already applied to the `POSITION`
  observation) isn't designed yet.

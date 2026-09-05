# Streak-derived dot velocity

Scope note (2026-09-05): follow-up to
[reflective-dot-detection-design.md](reflective-dot-detection-design.md)'s
"Open questions" section, which first raised a motion-blur streak's own
vector as an unused velocity signal, and to status.md's 2026-09-05 entry
(adaptive process noise closed most, not all, of the fast-swing lag gap).
Phase 1 (wire format) and phase 2 (standalone calibration/validation
script) done, validated against real data; phase 3 (tracker integration)
not started.

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

## 4. Tracker integration (not started)

No new UKF residual math is needed: `MeasurementMode::VELOCITY` already
computes `h(x_t) = project(x_t) - project(x_{t-1})` from the saved
posterior state alone (`ukf.cpp`'s `prev_projections` — confirmed it does
*not* require an actual t-1 *observation*, just a projectable posterior),
exactly the shape a streak-derived measurement needs. The plan:

- A per-camera windowed accumulator on `Tracker`, alongside
  `prev_observations_`/`nis_feedback_windows_`, holding the running
  `Σ streak_length`/`Σ frame_displacement` sums (§3), gated on a minimum
  sample count before `k` is trusted for that camera.
- When `resolve_dot_assignment()` resolves a streaked candidate with a
  trusted `k`, additionally construct a `VELOCITY`-mode `Observation` with
  measured value `streak_vector_px / k` (rescaling the sub-exposure
  displacement up to a full-frame-equivalent one), alongside (not instead
  of) the existing `POSITION` observation with its elongation-inflated
  noise.
- Sign ambiguity resolved by projecting the candidate's own predicted
  velocity onto the streak's canonicalized axis and taking whichever of the
  two directions agrees.

## 5. Open questions

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

# RTS Smoother Failure in Calibration Mode

**Date:** 2026-03-04
**Status:** Unresolved — parked for focus on real-world dataset testing
**Repro config:** `tracking_tests/ikkyo-harri-scale.toml`
**Output dir:** `tracking_tests/ikkyo-harri-calib/`
**Diagnostic CSV:** `tracking_tests/ikkyo-harri-calib/rts_smoother_diag.csv`

---

## 1. Symptom

Running `posetrak track --smooth` with calibration mode enabled (scalable skeleton,
prismatic DOFs) produces a `smoothed_state_vectors.csv` that is all-NaN for the first
~1618 frames (≈ 13.5 s) and wildly wrong values (positions × 10¹⁵³) for frames up to
~2128, with only the last ~18 frames (k=2128…2147) being near-correct.

Without calibration (no prismatic/scale DOFs), smoothing works correctly on the same
sequence.

---

## 2. Diagnostic instrumentation

A per-step CSV (`rts_smoother_diag.csv`) was added to `RTSSmoother::smooth()`.
Fields logged at each backward step k:

| Column | Meaning |
|---|---|
| `prior_min_eig` | Minimum eigenvalue of P_{k+1\|k} (prior covariance) |
| `prior_max_eig` | Maximum eigenvalue |
| `prior_condition` | Max/min ratio |
| `G_spectral_norm_raw` | Spectral norm of gain before clamping |
| `G_spectral_norm_clamped` | After SVD clamping to ≤ 1 |
| `delta_norm` | ‖ x_{k+1\|N} ⊖ x_{k+1\|k} ‖ (tangent-space correction driver) |
| `correction_norm` | ‖ G · delta ‖ |

---

## 3. Root cause analysis

### 3.1 Ill-conditioned forward-pass covariance

At the end of the sequence the posterior covariance statistics (from `tracking_stats.csv`) are:

```
cov_min_eigenvalue ≈ 1e-8   (clamped floor from PSD projection)
cov_max_eigenvalue ≈ 3.6e10 (condition number × min_eig)
condition number   ≈ 3.6e10
```

The prior covariance (posterior + process noise) has `prior_max_eig ≈ 364` throughout
the sequence. This comes from **unconstrained velocity DOFs**: the constant-velocity
process model adds noise to velocities every step, but the measurement update provides
very little velocity observability. After 2147 × (1/120 s) = 17.9 s, some velocity
variances accumulate to ~360 (rad/s)² or (m/s)².

The large spread between min (1e-8) and max (364) is driven by:

- **Velocity-limit zeroing**: after `enforce_joint_limits()`, joints at limits have
  velocities clamped to zero. `damp_velocity_covariance_at_limits()` zeros the
  corresponding covariance row/col and sets the diagonal to 1e-8. This creates a severe
  gap between those velocity directions (1e-8) and other directions (360).

- **Prismatic DOFs with near-zero process noise** (0.0001): scale-group DOFs barely
  move in the prior, causing tiny entries in the cross-covariance D for those rows.

The condition number 3.6×10¹⁰ makes the inversion `P_prior⁻¹` unreliable: small
numerical errors in D are amplified ~10¹⁰× in the gain computation.

### 3.2 Smoother gain spectral norm > 1 (always)

With this covariance, the computed smoother gain G = D · P_prior⁻¹ consistently has
spectral norm **2.5 – 20** (highest spikes at k≈7 (G=18.2) and k≈3 (G=19.7), which
are the earliest frames with very poorly conditioned initial covariance).

An RTS smoother is only stable when G_spec ≤ 1. A value of 2.67 at the final frame
means errors are amplified 2.67× per backward step — exponential divergence.

### 3.3 SVD clamping does not fix the accumulation

**Fix attempted**: clamp each singular value of G to [0, 1] before applying the
correction:

```cpp
G_clamped = U · diag(min(σᵢ, 1)) · V^T
```

After clamping, `G_spectral_norm_clamped = 1` for all steps (verified). However the
backward sweep still fails because:

1. Even with G_spec = 1, `correction_norm ≤ delta_norm`.
2. `delta = state_error(sm_next.state, fwd_next.prior_state)` grows each step because
   the *accumulated* correction has pushed `sm_next.state` away from the forward
   trajectory.
3. At k=2145: delta=0.068, correction=0.062
   At k=2141 (4 steps later): delta=0.334, correction=0.305
   At k=2140: delta=0.429, correction=0.411
4. Only **527 of 2146** backward steps produce a finite delta. At k=1618 (timestamp
   13.49 s) delta becomes `inf` because the accumulated state has overflowed floating
   point. All earlier frames then compute `state_error(inf_state, prior)` = inf / NaN.

The divergence is not a spike at one bad frame — it is fundamentally unstable from step
1 of the backward sweep due to G_spec being exactly 1 (the boundary of stability).

### 3.4 UKF `state_error` / smoother `state_error` mismatch (minor)

The UKF's `compute_state_error()` has:

```cpp
if ((j.type == JointType::REVOLUTE) ||
    (j.type == JointType::PRISMATIC && !j.is_scale_follower)) {
```

The smoother's `state_error()` uses:

```cpp
if (j.type == JointType::REVOLUTE || j.type == JointType::PRISMATIC) {
```

Follower prismatic DOFs (which share `state_index` with the leader) are included in the
smoother's error computation but skipped in the UKF's. The cross-covariance D was
computed with the UKF convention (zero rows for followers), so the G matrix has
near-zero rows for those directions; the smoother still computes non-zero delta entries
there, but they don't materially contribute to the correction. This mismatch is harmless
but should be fixed for correctness.

---

## 4. Things we tried

| Attempt | Result |
|---|---|
| Wire `export_debug` and add `--debug` CLI flag | ✅ Fixed in commit `8d29e05`. Unrelated to smoother. |
| Fix `damp_velocity_covariance_at_limits`: replace `row *= factor` / `col *= factor` with `row.setZero()` / `col.setZero()` to prevent non-PSD posterior | Posterior stays PSD; smoother still diverges (condition number unchanged). |
| Add PSD re-projection after velocity damping | Same result. |
| Add LLT integrity check with Tikhonov regularisation fallback in the smoother | Same result (prior_cov was already PSD, it's the conditioning not sign). |
| SVD-clamp G singular values to [0, 1] | G_spec = 1 at all steps ✓, but accumulation still diverges in ~500 backward steps. |

---

## 5. Known-good vs failing configurations

| Config | Smoothing |
|---|---|
| No calibration (no prismatic DOFs) | ✅ Works |
| Calibration enabled (scalable skeleton, prismatic DOFs) | ❌ Diverges as described |

The determining factor is most likely the **extreme condition number** caused by
velocity-limit covariance zeroing combined with unconstrained velocity variance growth.
Prismatic DOFs with tiny process noise exacerbate this by adding near-zero rows to D.

---

## 6. Candidate real fixes (not yet attempted)

These require more investigation and should be revisited when smoother quality matters:

### A. Velocity-only prior covariance for the smoother gain
Replace the full prior P with a block-structured approximation that uses a moderate
floor for velocity directions instead of 1e-8. E.g. keep velocity-diagonal entries
≥ `process_noise_std²` (0.0225) instead of 1e-8. This would reduce condition number
from ~3.6×10¹⁰ to ~16 and should bring G_spec << 1.

### B. Don't smooth velocity DOFs at all
The velocity state is an internal filter device (constant-velocity kinematics). It has
no physical meaning at fixed points in time — only the position/orientation DOFs matter
for the output. One option: project the smoother to work only on the position half of
the error state, making G a `pos_dim × pos_dim` matrix with guaranteed spectral
norm ≤ 1.

### C. Separate treatment of calibration (prismatic) DOFs
Since prismatic DOFs have process_noise_std = 0.0001 (essentially frozen), the RTS
smoother provides almost zero additional information for them (G≈0 in scale rows). They
could simply keep their forward-filter values while position/orientation DOFs are
smoothed normally. This also sidesteps the cross-contamination between
poorly-conditioned scale directions and well-conditioned position directions.

### D. Fix the `state_error` mismatch for follower prismatic DOFs
Smoother `state_error()` should skip `j.is_scale_follower` the same way
`compute_state_error()` does. Low priority (doesn't cause the divergence).

---

## 7. Repro configuration

```toml
# tracking_tests/ikkyo-harri-scale.toml
[data]
skeleton = "harri-skeleton-bisect-testing-scalable.yaml"
cameras = "/mnt/d/mocap/2026-01-11-kotegaesh-joint-space-test/Calib_scene.toml"
observations_dir = "/mnt/c/temp/aikido-2025-11-15-all/Harri_aihanmi_katatedori_ikkyo/pose"
sync = "/mnt/c/temp/aikido-2025-11-15-all/Harri_aihanmi_katatedori_ikkyo/sync_data.json"
person_id = 0
active_joint_groups = ["main", "HandL", "HandR"]

[tracking]
process_noise_std = 0.15
measurement_noise_std = 20.0
outlier_threshold = 4.0

[tracking.initialization]
ik_max_iterations = 1000
ik_tolerance = 0.02
init_position_std = 1.0
init_orientation_std = 1.0
init_joint_std = 0.1
init_velocity_std = 1.0
min_cameras_for_init = 2

[tracking.ukf]
alpha = 0.1
beta = 2.0
kappa = 0.0

[calibration]
enabled = true
prismatic_process_noise_std = 0.0001

[output]
directory = "/home/harri/projects/posetrak/tracking_tests/ikkyo-harri-calib"
export_tracking_results = true
export_statistics = true

[processing]
start_time = 0.0
end_time = 17.9
tracker_fps = 120.0
```

Run command:
```
optbuild/cli/posetrak track --smooth tracking_tests/ikkyo-harri-scale.toml
```

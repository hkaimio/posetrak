# Filter inconsistency diagnosis — 20260322-teacup-exc2

**Session**: `/mnt/d/mocap/20260322-teacup-exc2/session.db`
**Run analysed**: `186b3c71-87bc-4031-96ff-44de3eb123e0`
**Comparison run** (loosened wrist limits): `2c3bcc80-a047-4686-b36f-760eb9ff703e`
**Date**: 2026-03-26

## Observed symptoms

- Right hand and forearm diverge around frame ~348, recover around frame ~470
- NIS/dof ≈ 10 throughout (ideal = 1.0), average 10.13 over the full run
- Covariance condition number baseline ~1e6–1e7, spikes to ~1e11 during divergence
- Loosening `hand.L/R` x-limit from ±30° (±0.524 rad) to ±45.8° (±0.8 rad) caused complete tracking divergence

## Tracker config

| Parameter | Value |
|---|---|
| `measurement_noise_std` | 20.0 px |
| `process_noise_std` | 0.15 |
| `outlier_threshold` | 4.0 (Mahalanobis) |
| `tracker_fps` | 120.0 Hz |

## Root causes

### 1. `measurement_noise_std` too small — primary issue

NIS/dof ≈ 10 **from frame 1**, before any divergence. The filter is persistently over-confident throughout the entire run.

NIS = νᵀ S⁻¹ ν, and S = H P Hᵀ + R. For NIS/dof ≈ 10:
- R = (20 px)² = 400 px² is ~10× too small
- Actual effective measurement noise in this session ≈ √(10 × 400) ≈ **63 px σ**

With R too small:
- Mahalanobis gate (threshold 4.0) is measured against a 20 px distribution; actual noise is 63 px, so the gate rejects real observations at modest state errors
- Any perturbation (arm near a limit, brief occlusion) triggers mass outlier rejection and covariance divergence
- Filter becomes brittle despite never triggering `tracking_lost`

**Fix**: increase `measurement_noise_std` to **~60–70 px** (start with 60).

### 2. Wrist sideways-flexion limit ±30° may be correct but is sensitive to sigma-point clamping

`hand.L/R` x-axis is sideways flexion (ulnar/radial deviation), for which ±30° is a reasonable anatomical range. However the exercise involves repeated extremes of this motion.

When `hand.R.x` reaches ±0.524 rad, UKF sigma points (perturbed in active error-state dimensions) are clamped by `enforce_joint_limits()`. The resulting sigma point set becomes asymmetric around the mean. FK then produces a biased wrist/finger marker cloud with artificially small spread, inflating Mahalanobis distances for actual wrist observations → outlier rejection → no correction → arm drifts until motion returns inside the limit range.

**Why loosening to ±45.8° made it worse**: with NIS/dof already ~10 (Issue 1), the filter is systematically over-confident. The ±30° limit was accidentally acting as a regularizer. Without it, the inconsistent filter is free to drift further into implausible states and never recovers.

**Fix order matters**: fix Issue 1 first. With a consistent filter (NIS/dof ≈ 1), the gate works correctly even at the limit boundary. If divergence still occurs at ±30°, the limits may need small adjustments, but the measurement noise is the primary lever.

### 3. Q matrix applies uniform noise to positions and velocities

`get_process_noise()` returns `Q = (σ · dt)² · I` for all state dimensions. With dt = 1/120 s:

```
Q_variance = (0.15 / 120)² = 1.56×10⁻⁶  per dimension per step
```

This applies the same process noise to joint angles (rad) and joint angular velocities (rad/s). In a proper double-integrator (CWNA) noise model:
- Position/angle noise should scale as `σ² · dt⁴/4`
- Velocity noise should scale as `σ² · dt²`

The current model gives both `dt²` scaling. Velocity DOFs accumulate uncertainty faster than measurements can correct (velocities are only indirectly observed), so velocity variances inflate over time. This contributes to the condition number growing from ~1e6 to ~1e11, and makes process noise hard to tune correctly.

The condition number metric is also partly misleading because it mixes angle variances (rad²) with velocity variances (rad²/s²) — their ratio has no single physical interpretation.

**Fix**: use separate process noise for position and velocity dimensions. Practically: add `process_noise_vel_std` parameter with a larger value (e.g. 5–10× `process_noise_pos_std` at 120fps), or implement the proper double-integrator Q.

## Recommended action sequence

1. **Increase `measurement_noise_std` to 60 px** and re-run. Check NIS/dof — should drop from ~10 toward ~1. This is the primary fix.

2. **Re-test with the original ±30° wrist limits** on the same session. With a consistent filter, the sigma-point clamping at the limit should no longer cause sustained divergence.

3. **If arm divergence persists** after step 1, consider small limit adjustments (e.g. ±35–40°) rather than the large jump to ±45.8°.

4. **Longer term**: fix Q matrix to use proper position/velocity scaling. Monitor NIS/dof rather than condition number as the primary filter health indicator.

## Supporting data

NIS/dof by 100-frame bins (run `186b3c71`):

| Frames | NIS/dof mean | Condition number mean | Inliers mean |
|---|---|---|---|
| 0–99 | 11.6 | 1.1e7 | 217 |
| 100–199 | 9.5 | 8.0e6 | 213 |
| 200–299 | 10.1 | 1.4e7 | 219 |
| 300–399 | 10.4 | 1.0e7 | 188 |
| 400–499 | 9.7 | **1.7e9** | 189 |
| 500–599 | 7.4 | 2.5e8 | 178 |
| 600–699 | 9.7 | 2.0e8 | 135 |
| 700–799 | 11.2 | 8.5e7 | 208 |
| 800–899 | 9.6 | 1.1e7 | 203 |
| 900–999 | 11.2 | 4.0e7 | 196 |
| 1000–1099 | 9.4 | 2.7e8 | 166 |
| 1100–1199 | 11.8 | 5.8e7 | 139 |

Note: NIS/dof never drops below 5.8 at any individual frame. Filter is inconsistent for the entire duration.

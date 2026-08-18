# UKF algorithm

Posetrak estimates skeletal pose using an **error-state Unscented Kalman Filter (UKF)** in joint space.  This page explains what that means and how to interpret the tuning parameters.

---

## Why a Kalman filter?

The straightforward approach to multi-camera pose estimation is to triangulate each body landmark into 3-D and then solve inverse kinematics each frame independently.  In practice this breaks down quickly:

- **Noise and flickering** — pose detectors fire keypoints at slightly wrong positions every frame.  Frame-independent IK turns that noise directly into visible jitter in the animation.
- **Missing detections** — when a limb is occluded or the detector fails, there is no observation to triangulate.  IK has no way to estimate where the limb went.
- **Temporal inconsistency** — two consecutive frames that look visually identical can produce very different IK solutions when the problem is underconstrained (e.g. multiple plausible elbow positions for a given wrist position).

A Kalman filter solves all three by maintaining a **probability distribution** over the full body state and propagating it forward in time.  The current estimate is not just a point — it is a mean and a covariance that encode uncertainty.  Each frame, the filter predicts where the body should be given its recent motion, then corrects that prediction using the new camera observations.  Missing observations leave the prediction unchanged but increase uncertainty; noisy observations are weighted against the prediction by how surprised the filter is.

## Why UKF rather than EKF?

The measurement model — the function from joint angles to 2-D pixel observations — is nonlinear: joint angles → forward kinematics → 3-D marker positions → camera projection → pixels.

An Extended Kalman Filter would linearise this chain by computing its Jacobian analytically.  This is possible but brittle: the Jacobian is non-trivial to implement correctly, and the linear approximation degrades for larger rotations.

The **Unscented Kalman Filter** sidesteps linearisation entirely.  Instead of a Jacobian, it uses **sigma points**: a small set of carefully chosen sample states that together represent the mean and covariance of the distribution.  Each sigma point is passed through the full nonlinear chain — FK, then projection — and the results are recombined.  No approximation is required; only forward evaluations of functions that already exist.

---

## What the filter estimates

The filter maintains a probability distribution over the full body state:

- **Root position** — 3 numbers (x, y, z in world space, metres)
- **Root orientation** — represented as a quaternion, with 3 effective DOFs in the error-state (see below)
- **Joint angles** — one number per revolute DOF, three per spherical DOF (radians)
- **All of the above, differentiated** — the corresponding velocity for each

For a full-body skeleton with hands, the error-state dimension is approximately 230 (108 joint DOFs × 2 for position and velocity, plus 12 for root), yielding around 460 sigma points per frame.  A simpler skeleton without hands is roughly 60–70 DOF and ~130 sigma points.  The exact number depends on the skeleton definition.

The filter expresses uncertainty as a covariance matrix of that dimension.

---

## Predict step (process model)

Each frame, the filter **predicts** how the state will change before it has seen new measurements.

The process model is **constant velocity**: each position/angle advances by `velocity × dt`.  Velocities decay exponentially toward zero with a half-life controlled by `velocity_half_life_s` — this encodes the expectation that a person will slow down and stop rather than continue indefinitely.

Process noise (controlled by `process_noise_std` and `process_noise_vel_std`) adds uncertainty to the prediction at each step, reflecting the fact that joint accelerations are unknown.  Too small → the filter becomes overconfident and rejects valid measurements as outliers.  Too large → the filter becomes sluggish and the covariance grows unboundedly.

---

## Update step (measurement model)

After predicting, the filter **updates** its state using the 2-D keypoint observations from the cameras.

The measurement model is:

1. Take the predicted state.
2. Run **forward kinematics** (Pinocchio): state → 3-D marker positions in world space.
3. **Project** each marker through each camera's intrinsics to get expected pixel positions.
4. Compare expected pixel positions to observed keypoint positions.
5. Correct the state toward the observations; reduce covariance.

**Outlier rejection** — Before applying an observation, the filter computes its **Mahalanobis distance**: how far the observation is from the prediction in units of standard deviations.  Observations beyond the `outlier_threshold` (in σ units) are discarded.  This is what allows the filter to tolerate occasional bad detections without being corrupted by them.

---

## Error-state formulation

Orientations — for the root joint and for spherical joints — cannot be treated as ordinary Euclidean numbers.  The fundamental issue is that rotation is a curved space (a manifold): if you add two rotation vectors together you do not generally get a valid rotation, and the shortest path between two rotations wraps around rather than going in a straight line.  Standard Kalman filter arithmetic assumes flat Euclidean space, so it cannot be applied directly to quaternions or rotation matrices.

Posetrak uses an **error-state formulation** to work around this:

- The filter stores the current best estimate of each orientation as a **full quaternion** (the *nominal state*).  This quaternion always represents a valid rotation.
- The **covariance matrix** is maintained over a small 3-D **axis-angle perturbation** (the *error state*) around that quaternion.  The error state lives in the tangent space of the rotation manifold at the current quaternion — a flat Euclidean space where ordinary Kalman arithmetic works.
- When sigma points are generated, each one is obtained by applying a small axis-angle perturbation to the current quaternion.  When the update step computes a correction, that correction is also expressed as an axis-angle, then composed onto the nominal quaternion via the exponential map.

The effect is that all the Kalman filter mathematics operates in flat space on small perturbations, while the stored state always stays on the rotation manifold.  `State::apply_error_update()` performs the retraction step at the end of each predict/update cycle.

---

## Joint limits

Joint limits define the allowable range of motion for each joint axis.  They appear in the skeleton YAML as `limits: {x: [min, max], y: [...], z: [...]}` (radians).

### How limits are applied

After sigma point propagation, the filter computes the **predicted mean state** from the weighted average of propagated sigma points.  Joint limits are applied to that mean state immediately, before the covariance is recomputed.  The implementation is in `ConstantVelocityModel::enforce_joint_limits()` (`cpp/src/filters/process_model.cpp`):

- **Revolute joints** (1 DOF): the single angle is clamped with `std::clamp(angle, min, max)`.
- **Spherical joints** (3 DOF): each axis is clamped independently.  Axes where `min == max` are treated as **locked DOFs** and are set to exactly the limit value at every step.  Active axes (where `min < max`) are clamped to their range.

Limits are **not** applied to individual sigma points — only to the mean after propagation.  This means the covariance can still represent uncertainty beyond the limits for a single predict step; the clamp takes effect before the update and before the next predict.

### Why limits matter for tracking

Joint limits serve two purposes:
- **Physical plausibility** — prevents the filter from finding unrealistic poses (elbow bending backwards, etc.)
- **Disambiguation** — for joints with symmetric measurement models, limits prevent the filter from converging to the anatomically wrong solution

### Common failure mode

If a technique involves a large range of motion and the skeleton limits do not allow it, the predicted markers freeze at the boundary while the actual person keeps moving.  This widens the gap between prediction and observation, raising Mahalanobis distances and triggering outlier rejection of the (correct) observations — which in turn prevents any correction, making the divergence worse.

If tracking degrades at specific joints, check whether the limits in the skeleton YAML are wide enough for the activity being captured.  The joint_angles CSV output shows which DOFs are hitting their limits.

---

## Why sigma points?

For a 230-dimensional state the UKF generates $2 \times 230 + 1 = 461$ sigma points.  Each requires a full forward kinematics pass, which is the dominant per-frame cost.  This is the main reason an optimised build (`optbuild/`) matters for real tracking runs — the debug build is too slow.

---

## UKF hyperparameters

The three spread parameters `alpha`, `beta`, `kappa` control sigma point placement:

| Parameter | Typical value | Effect |
|---|---|---|
| `alpha` | 0.5 | Spread of sigma points around the mean.  **Must be ≥ 0.5 for typical state dimensions (~60–230 DOF)** — smaller values cause negative weights on the central sigma point, which breaks the covariance update. |
| `beta` | 2.0 | Encodes prior knowledge about the distribution (Gaussian → β = 2 is optimal). |
| `kappa` | 0.0 | Secondary scaling; usually left at 0. |

---

## Noise parameters

| Parameter | Unit | Meaning |
|---|---|---|
| `process_noise_std` | rad/s² (joint), m/s² (root) | Expected standard deviation of joint/position acceleration per second.  Controls how quickly the filter trusts new measurements over its own prediction. |
| `process_noise_vel_std` | rad/s² | Same, applied to velocity DOFs. |
| `velocity_half_life_s` | seconds | Exponential damping on velocities.  Shorter → velocities decay faster; keeps covariance bounded during static poses. |
| `measurement_noise_std` | pixels | Expected standard deviation of a single keypoint detection in pixel space.  Larger → filter trusts detections less and relies more on prediction. |
| `outlier_threshold` | σ | Mahalanobis distance threshold for accepting an observation.  Observations beyond this are discarded.  Typical value: 4. |

### Tuning guidance

**NIS / dof** (Normalised Innovation Squared divided by degrees of freedom) is the primary consistency check for the filter.  A well-tuned filter has $\text{NIS}/\text{dof} \approx 1$.

- $\text{NIS}/\text{dof} > 1$ → filter is **overconfident** — measurements consistently surprise it.  Increase `measurement_noise_std` or `process_noise_std`.
- $\text{NIS}/\text{dof} < 1$ → filter is **underconfident** — covariance is too large.  Decrease noise parameters.
- High `cov_condition_number` ($> 10^6$) → covariance is becoming ill-conditioned.  Try reducing `velocity_half_life_s` (shorter damping keeps condition numbers lower).

The parameter sweep tool (`python/tools/param_sweep.py`) automates a grid search over these parameters and ranks configurations by a score that combines NIS consistency, tracking-lost rate, and covariance condition number.  See the tool's docstring for usage.

---

## Covariance update

The filter uses the **Joseph form** of the covariance update:

$$P_{k+1|k+1} = (I - K H)\, P_{k+1|k}\, (I - K H)^T + K R K^T$$

where:

| Symbol | Meaning |
|---|---|
| $P_{k+1\|k+1}$ | Updated (posterior) covariance — uncertainty after incorporating the new observations |
| $P_{k+1\|k}$ | Predicted (prior) covariance — uncertainty before the update, propagated from the previous step |
| $K$ | Kalman gain — $n_\text{state} \times n_\text{obs}$ matrix that weights how much to trust each observation vs. the prediction |
| $H$ | Linearised measurement matrix — maps state perturbations to expected measurement changes (approximated from sigma points in the UKF) |
| $R$ | Measurement noise covariance — diagonal matrix with $\sigma_\text{measurement}^2$ on each observed pixel coordinate |
| $I$ | Identity matrix |

The standard form is $P = (I - KH)P_\text{prior}$, which is equivalent algebraically.  The Joseph form adds the $KRK^T$ term and the symmetric product structure, making it self-evidently symmetric and positive-semidefinite even when the Kalman gain is computed with floating-point rounding errors.  For a 230-dimensional state this numerical stability matters.

---

## Initialisation

The filter is initialised from the first frame where enough cameras have detections:

1. **DLT triangulation** — each detected marker is triangulated across cameras using the Direct Linear Transform to get a 3-D world position.
2. **Inverse kinematics** — damped least-squares IK maps the triangulated marker positions to joint angles, giving the initial state.
3. **Initial covariance** — set from `init_position_std`, `init_orientation_std`, `init_joint_std`, `init_velocity_std` config parameters.

---

## Optional RTS smoothing

After a tracking pass, **Rauch-Tung-Striebel (RTS) smoothing** can be applied to refine the estimates using future observations.  Enable in the config before running; call the `smooth()` pass after the forward tracking loop completes.  Smoothed results are stored alongside unsmoothed results (distinguished by `is_smoothed` in `tracking_results`).

# Hierarchical UKF for Decoupled Body-Hand Tracking

## Problem Statement

### Observed Symptoms

**Frame 685 Analysis (process_noise_std=0.5):**
- Elbow (forearm.R) covariance: 626° standard deviation
- Fingertip predictions spread over meters (500+ pixel residuals)
- Paradox: Elbow predictions remain accurate (15-27 pixel residuals, Mahalanobis 0.5-1.1)
- Filter is **inconsistent**: accurate estimates but overconfident (huge uncertainty)

**Frame 920 Divergence (process_noise_std=0.2):**
- Covariance explosion: 130° → 266° in 5 frames
- Vicious cycle:
  1. Few observations → covariance grows
  2. Large covariance → large innovation covariance
  3. Large innovation covariance → small Mahalanobis distance
  4. All observations accepted as inliers (5276 pixel residuals, Mahalanobis 0.61)
  5. Large innovation covariance → small Kalman gain
  6. Measurements don't reduce covariance effectively

**Covariance Growth Pattern:**
```
Frame 100: 254° (already elevated)
Frame 200: 400° (accelerating)
Frame 300: 336° (temporary improvement)
Frame 400: 323° (stable)
Frame 500: 461° (growing again)
Frame 600: 632° (rapid jump)
Frame 650: 640° (plateaued but very high)
Frame 685: 626° (sustained high uncertainty)
```

## Root Cause Analysis

### Weak Observability Problem

**Kinematic Chain Structure:**
```
shoulder.R (3 DOF) → upper_arm.R (3 DOF) → forearm.R (3 DOF) → hand.R (3 DOF)
                                                                      ↓
                                                            [20+ finger markers]
```

**Marker Distribution:**
- **Arm segments**: 2-4 markers per segment (shoulder, elbow, wrist)
- **Hand/fingers**: 20+ markers on palms and fingers
- **Observation imbalance**: 5-10× more finger markers than arm markers

**Weak Observability Mechanism:**

1. **Short bone lengths**: Finger bones are ~2-4 cm long
   - Small absolute displacement per degree of rotation
   - ±30° finger joint rotation → only ~1-2 cm tip displacement
   - In pixel space: ~10-20 pixels at typical camera distance

2. **Weak constraint on elbow**:
   - Finger marker positions depend on entire kinematic chain
   - ~15 joints between elbow and fingertip (shoulder → upper_arm → forearm → hand → palm → finger joints)
   - Small finger bone displacements provide weak information about elbow state
   - Large elbow uncertainty → wide sigma point spread → dominates finger marker predictions

3. **Information content**:
   - Arm markers directly constrain arm angles (strong observability)
   - Finger markers primarily constrain finger angles (strong local observability)
   - Finger markers weakly constrain elbow through long kinematic chain (weak remote observability)

### Cross-Covariance Coupling in UKF

**UKF Measurement Update Equations:**

$$
\begin{align}
\text{Predicted observations: } & \quad \mathbf{z}_i = h(\boldsymbol{\chi}_i) \\
\text{Mean predicted observation: } & \quad \bar{\mathbf{z}} = \sum w_i \mathbf{z}_i \\
\text{Innovation covariance: } & \quad P_{zz} = \sum w_i (\mathbf{z}_i - \bar{\mathbf{z}})(\mathbf{z}_i - \bar{\mathbf{z}})^T + R \\
\text{Cross-covariance: } & \quad P_{xz} = \sum w_i (\boldsymbol{\chi}_i - \bar{\mathbf{x}})(\mathbf{z}_i - \bar{\mathbf{z}})^T \\
\text{Kalman gain: } & \quad K = P_{xz} P_{zz}^{-1} \\
\text{State update: } & \quad \mathbf{x}^+ = \mathbf{x}^- + K(\mathbf{y} - \bar{\mathbf{z}})
\end{align}
$$

**Spurious Coupling Mechanism:**

1. **Large elbow uncertainty** creates wide sigma point spread:
   - Sigma points $\boldsymbol{\chi}_i$ cover ±626° range for elbow
   - Each sigma point propagates through full forward kinematics

2. **Forward kinematics amplification**:
   - Long kinematic chain: elbow → wrist → palm → finger joints (6-10 joints)
   - Small elbow angle variation → large cumulative displacement at fingertip
   - Example: 10° elbow variation × 60 cm arm length → 10 cm fingertip displacement

3. **Cross-covariance $P_{xz}$ computation**:
   - Correlates state deviations $(\boldsymbol{\chi}_i - \bar{\mathbf{x}})$ with observation deviations $(\mathbf{z}_i - \bar{\mathbf{z}})$
   - Elbow sigma point spread → correlated finger marker prediction spread
   - Creates **spurious correlation** between elbow state and finger observations

4. **Kalman gain coupling**:
   - $K = P_{xz} P_{zz}^{-1}$ couples finger innovations to elbow corrections
   - Finger measurement innovation $(\mathbf{y}_\text{finger} - \bar{\mathbf{z}}_\text{finger})$ updates elbow state
   - But finger measurements contain **weak information** about elbow angles

5. **Numerical amplification**:
   - 20+ finger markers vs 2-4 arm markers
   - Finger observations dominate measurement vector (5-10× larger)
   - Even weak correlations in $P_{xz}$ create significant Kalman gain components
   - Noisy finger-to-elbow updates accumulate over time

### Experimental Validation

**Test**: Disabled most palm/finger markers, kept only minimal set for wrist orientation.

**Result**: Divergence significantly reduced.

**Interpretation**:
- Removing abundant finger observations eliminated spurious coupling
- Filter focused on directly observable arm markers
- Arm covariance remained bounded without finger observation interference
- **Confirms theory**: Information imbalance (abundant but weakly informative observations) causes coupling

## Theoretical Framework

### Information Geometry Perspective

**Fisher Information Matrix** for measurement $\mathbf{y}$ and parameter $\theta$:

$$
I(\theta) = \mathbb{E}\left[\left(\frac{\partial \log p(\mathbf{y}|\theta)}{\partial \theta}\right)^2\right]
$$

**Observability Strength:**
- **Direct observation**: Elbow marker observes elbow angle directly → high $I(\theta_\text{elbow})$
- **Indirect observation**: Finger marker observes elbow angle through FK → low $I(\theta_\text{elbow})$

**Information Content Paradox:**
- 20 finger markers provide **high total information** about finger DOFs
- Each finger marker provides **low individual information** about elbow DOF
- But UKF couples all observations through cross-covariance
- Net effect: Many weak constraints can dominate few strong constraints

### Why Process Noise Tuning Failed

**Process noise reduction** (0.5 → 0.2 → 0.15):
- **Helps**: Slows covariance growth rate in predict step
- **Doesn't solve**: Structural coupling problem in BOTH predict and update
- **Risk**: Too low process noise → divergence (observed at 0.1)

**Fundamental issues:**

1. **Predict phase**: If elbow has 626° uncertainty in covariance matrix:
   - Sigma points spread ±626° in elbow angle
   - This spread persists through process model propagation
   - Forward kinematics amplifies spread to fingertips (meter-scale prediction spread)
   - Process noise only affects rate of growth, not the existing large uncertainty

2. **Update phase**: Spurious cross-covariance coupling (detailed in earlier section)
   - Finger observations incorrectly update elbow state
   - Process noise doesn't address observation space coupling

Process noise is the wrong tool for this problem - it's a **structural coupling issue**, not a temporal tuning parameter.

## Proposed Solutions

### Option 1: Hierarchical/Split UKF (Recommended)

**Approach**: Decouple body and hand tracking into sequential passes.

**Architecture:**

```
Pass 1: Body-Level Tracking
├─ Active DOFs: Main skeleton (no hand/finger joints)
├─ Observations: Arm/body markers + 2-3 palm markers for wrist orientation
├─ Process noise: 0.15 rad/√s (current tuned value)
├─ Measurement noise: 20.0 pixels (current value)
├─ Outlier threshold: 4.0 (Mahalanobis, current value)
└─ Output: Updated body state + covariance

Pass 2: Hand-Level Tracking (per hand)
├─ Fixed DOFs: Body skeleton (locked from Pass 1)
├─ Active DOFs: Hand and finger joints only
├─ Observations: All palm and finger markers
├─ Process noise: 0.25-0.30 rad/√s (higher for agile fingers)
├─ Measurement noise: 10.0-15.0 pixels (lower for dense markers)
├─ Outlier threshold: 3.0-3.5 (stricter for abundant observations)
└─ Output: Updated hand state + covariance
```

**Theoretical Justification:**

1. **Eliminates spurious coupling**:
   - Pass 1: No finger observations → no cross-covariance from fingers to elbow
   - Pass 2: Body is fixed → no coupling from finger uncertainty to body

2. **Preserves information flow**:
   - Wrist markers in Pass 1 provide body-to-hand connection
   - Pass 2 operates in accurate body reference frame

3. **Parameter specialization**:
   - Body: Lower process noise (large inertia, slow dynamics)
   - Fingers: Higher process noise (small inertia, fast dynamics)
   - Fingers: Lower measurement noise (dense marker coverage, high confidence)
   - Fingers: Stricter outlier rejection (abundant observations tolerate rejection)

4. **Established approach**:
   - Used in humanoid robotics (whole-body control + dexterous manipulation)
   - Hierarchical state estimation common in mobile manipulation
   - Similar to "coarse-to-fine" refinement in vision

**Implementation Strategy:**

**CRITICAL: Both predict AND update are hierarchical** - This is not just hierarchical update, but two complete independent UKF cycles.

```cpp
// Full hierarchical UKF cycle (predict + update)
void UKF::step(const std::vector<Observation>& observations) {
    // ========================================
    // Pass 1: Body UKF Cycle (complete)
    // ========================================

    // 1a. Body predict
    //   - Extract body DOFs from full state
    //   - Generate sigma points for body DOFs ONLY (no finger uncertainty!)
    //   - Propagate through process model: q_body[t+1] = q_body[t] + dt*q̇_body[t]
    //   - Reconstruct body prediction and covariance
    auto body_state = extract_body_state(state_);
    auto body_cov = extract_body_covariance(covariance_);

    auto [body_pred, body_pred_cov] = ukf_predict(
        body_state, body_cov, body_dof_mask, dt_
    );

    // 1b. Body update
    //   - Use arm/torso markers + 2-3 palm markers for wrist constraint
    //   - No finger markers → no spurious coupling
    auto body_observations = filter_observations_by_group(observations, {"main"});
    body_observations += select_palm_markers(observations, 2);  // Minimal wrist info

    auto [body_post, body_post_cov] = ukf_update(
        body_pred, body_pred_cov, body_observations, body_dof_mask
    );

    // ========================================
    // Pass 2: Hand UKF Cycles (per hand)
    // ========================================

    for (hand in ["HandL", "HandR"]) {
        // 2a. Hand predict
        //   - Body state is FIXED (from Pass 1)
        //   - Extract hand DOFs only
        //   - Generate sigma points for hand DOFs ONLY
        //   - Key: Wrist position from body_post is deterministic
        //   - No elbow uncertainty in sigma points!
        auto hand_state = extract_hand_state(state_, hand);
        auto hand_cov = extract_hand_covariance(covariance_, hand);

        auto [hand_pred, hand_pred_cov] = ukf_predict(
            hand_state, hand_cov, hand_dof_mask(hand), dt_
        );

        // 2b. Hand update
        //   - All palm + finger markers for this hand
        //   - Hand FK uses FIXED body state from Pass 1
        //   - Covariance is local to hand (no body cross-terms)
        auto hand_observations = filter_observations_by_group(
            observations, {hand}
        );

        auto [hand_post, hand_post_cov] = ukf_update(
            hand_pred, hand_pred_cov, hand_observations,
            hand_dof_mask(hand), body_post  // Pass fixed body state for FK
        );

        // Update hand portion of full state
        update_hand_state(state_, hand, hand_post);
        update_hand_covariance(covariance_, hand, hand_post_cov);
    }

    // Update body portion of full state
    update_body_state(state_, body_post);
    update_body_covariance(covariance_, body_post_cov);

    // NOTE: Cross-covariances between body and hands are assumed zero
    // This is the independence assumption that breaks the spurious coupling
}
```

**Expected Benefits:**

1. **Bounded elbow covariance**: <150° (vs 626° current)
   - Body pass focuses on directly observable arm markers
   - No spurious updates from weakly-informative finger observations

2. **Eliminates finger prediction spread**:
   - Hand predict uses fixed body state from Pass 1
   - Elbow uncertainty does NOT propagate into hand sigma points
   - Finger predictions based on deterministic wrist position (no meter-scale spread!)

3. **Maintained (or improved) finger tracking accuracy**:
   - Dense markers + specialized parameters
   - Lower measurement noise (10-15 vs 20 pixels)
   - Stricter outlier rejection (threshold 3.0-3.5 vs 4.0)
   - Higher process noise (0.25-0.3 vs 0.15) matches agile finger dynamics

4. **Robustness**: Body tracking unaffected by finger noise/occlusions

5. **Physical consistency**: Respects dynamic differences (massive body vs agile fingers)

**Challenges:**

1. **State consistency**:
   - Wrist position determined in Pass 1
   - Finger base starts from Pass 1 wrist
   - Need careful FK evaluation at boundary

2. **Covariance splitting**:
   - Full covariance 210×210 split into body block + hand blocks
   - Cross-covariance between body/hands discarded
   - Assumes independence (valid given decoupling goal)

3. **Computational cost**:
   - Two UKF passes instead of one
   - But each pass is smaller (fewer DOFs)
   - Overall similar or slightly higher cost

### Option 2: Per-Joint Measurement Noise Scaling

**Approach**: Increase measurement noise for finger markers to reduce their influence.

**Configuration:**
```toml
[tracking.measurement_noise]
# Default for most markers
default_std = 20.0

# Finger markers get higher noise
finger_std = 40.0  # or 50.0

# Implementation: lookup by marker name or parent joint
```

**Pros:**
- Simple implementation (one config parameter)
- No algorithmic changes
- Gradual tuning knob

**Cons:**
- **Doesn't address root cause**: Coupling still exists, just weighted differently
- **Reduces finger tracking fidelity**: Higher noise → less measurement influence
- **Arbitrary tuning**: No principled basis for noise ratio
- **May not fully solve**: Abundance effect (20 markers × 40 pixels) still large influence

**User Concern (validated):**
> "We do need accurate tracking for fingers as well, and if we increase measurement noise for those markers it could reduce finger tracking fidelity."

**Verdict**: Not recommended as primary solution. Treats symptom, not cause.

### Option 3: Adaptive Covariance Inflation

**Approach**: Detect weakly-observed DOFs and inflate their covariance adaptively.

**Algorithm:**
1. Monitor residual patterns per DOF
2. Identify DOFs with high uncertainty but accurate predictions (inconsistent filter)
3. Inflate process noise for those DOFs only
4. Prevents runaway growth while maintaining sensitivity

**Example:**
```cpp
// After predict step, before update
for (dof in skeleton.dofs) {
    if (is_weakly_observed(dof, observations)) {
        // Inflate covariance for this DOF
        covariance_[dof, dof] *= inflation_factor;  // e.g., 1.2
    }
}
```

**Pros:**
- Handles dynamic scenarios (different frames have different observation patterns)
- No architectural changes needed
- Principled approach based on observability

**Cons:**
- Complex implementation (need observability metric)
- Tuning required (inflation factor, detection threshold)
- Doesn't eliminate coupling, just manages symptoms
- May introduce new instabilities

**Verdict**: Interesting research direction, but more complex than hierarchical approach.

### Option 4: Observation Grouping with Sequential Updates

**Approach**: Process observations in groups, updating state incrementally.

**Algorithm:**
1. Group 1: Arm markers → update body state
2. Group 2: Wrist markers → update wrist state
3. Group 3: Finger markers → update finger state

**Difference from Split-UKF**: All DOFs active in each group, but observations processed sequentially.

**Pros:**
- Cross-covariances between groups still computed
- Information flows in directed manner (arm → wrist → fingers)
- Single filter, just different observation ordering

**Cons:**
- Order dependence: Results depend on observation processing order
- Doesn't fully eliminate coupling (all DOFs still active)
- Less principled than full hierarchy

**Verdict**: Middle ground between monolithic and hierarchical. Worth considering if Split-UKF too complex.

## Recommended Path Forward

### Phase 1: Implement Split-UKF (2-3 days)

**Tasks:**
1. Refactor `UKF::predict()` and `UKF::update()` to support DOF masking
   - Extract body/hand state subsets from full state vector
   - Generate sigma points for active DOFs only
   - Propagate with masked covariance matrices
2. Implement hierarchical `UKF::step()` function:
   - Pass 1: Full body UKF cycle (predict + update)
   - Pass 2: Full hand UKF cycles (predict + update, per hand)
   - Body state fixed during hand passes
3. Create marker grouping configuration (body vs hands)
4. Add configuration parameters for per-pass settings
5. Implement state/covariance extraction and merging utilities

**Testing:**
- Frame 685 single-frame test: Check elbow covariance stays <200° AND finger predictions reasonable
- Frame 600-700 sequence: Verify no runaway growth in either body or hand covariances
- Frame 915-920 divergence region: Verify stability through challenging period
- Full dataset: Compare tracking quality and divergence frequency

**Success Metrics:**
- Elbow covariance: <150° throughout sequence
- Finger tracking RMS error: Similar or better than current
- Divergence events: Zero (vs current occasional events)
- Computational overhead: <2× current runtime

### Phase 2: Parameter Tuning (1-2 days)

**Body Pass Parameters:**
- process_noise_std: Start with 0.15 (current tuned value)
- measurement_noise_std: 20.0 (current)
- outlier_threshold: 4.0 (current)
- alpha: 0.1 (current sigma point spread)

**Hand Pass Parameters:**
- process_noise_std: 0.25-0.30 (higher for agile fingers)
- measurement_noise_std: 10.0-15.0 (lower for dense markers, high confidence)
- outlier_threshold: 3.0-3.5 (stricter rejection, tolerate losing some fingers)
- alpha: 0.1 (keep same)

**Tuning Strategy:**
1. Start conservative (closer to current values)
2. Gradually increase finger process noise if hand motion too sluggish
3. Gradually decrease finger measurement noise if tracking seems too cautious
4. Adjust outlier threshold if too many false positives/negatives

### Phase 3: Validation & Analysis (1 day)

**Quantitative Metrics:**
- Covariance trajectories: Plot elbow, wrist, finger covariances over time
- Tracking RMS error: Per-marker, per-frame, compare to baseline
- Outlier rejection rates: Per-pass, ensure not over-rejecting
- Computation time: Compare to baseline

**Qualitative Checks:**
- Rerun visualization: Check for artifacts, unnatural motion
- Divergence events: Manual inspection of challenging frames
- Finger dexterity: Verify fine finger tracking maintained
- Edge cases: Occlusions, rapid motion, self-similarity

**Fallback Plan:**
If Split-UKF shows issues:
1. Try Option 4 (Sequential grouping) as simpler alternative
2. Combine with adaptive covariance inflation (Option 3)
3. As last resort, use per-joint noise scaling (Option 2) as temporary fix

## Implementation Notes

### DOF Masking for Selective Updates

**State Vector Structure:**
```
state = [q_body, q_left_hand, q_right_hand, q̇_body, q̇_left_hand, q̇_right_hand]
        |<------- positions ------->| |<-------- velocities -------->|
```

**Mask Examples:**
```cpp
// Body pass: Update body DOFs only
bool body_mask[2*num_dofs];
for (int i = 0; i < num_dofs; i++) {
    bool is_body = !is_hand_dof(i);
    body_mask[i] = is_body;              // Position
    body_mask[i + num_dofs] = is_body;   // Velocity
}

// Hand pass: Update hand DOFs only
bool hand_mask[2*num_dofs];
for (int i = 0; i < num_dofs; i++) {
    bool is_hand = is_hand_dof(i);
    hand_mask[i] = is_hand;
    hand_mask[i + num_dofs] = is_hand;
}
```

**Selective Kalman Gain:**
```cpp
// Standard: K P_{xz} P_{zz}^{-1}
// Masked: K_masked = K ⊙ mask (element-wise)
// Or: Only compute K rows for active DOFs
```

### Marker Grouping Configuration

**Extension to skeleton YAML:**
```yaml
joints:
  - name: forearm.R
    parent: upper_arm.R
    group: main  # or "body"

  - name: f_index.01.R
    parent: hand.R
    group: HandR

markers:
  - name: RElbow
    parent: forearm.R
    # Inherits group from parent joint
    # Override with explicit group: main
```

**Use groups to partition DOFs and observations:**
- `main` / `body`: Skeleton excluding hands (Pass 1)
- `HandL`: Left hand and fingers (Pass 2)
- `HandR`: Right hand and fingers (Pass 2)

**Special handling for wrist/palm markers:**
- Include 2-3 palm markers in body pass for wrist orientation constraint
- Also include in hand pass for finger base reference
- No double-counting issue: Different DOF subsets active in each pass

## Hierarchical RTS Smoothing

### Overview

RTS (Rauch-Tung-Striebel) smoothing for hierarchical UKF is straightforward because **body and hand dynamics are decoupled in state space**.

**Key insight:**
- Body dynamics: $\mathbf{x}_b[t+1] = f_b(\mathbf{x}_b[t], \mathbf{w}_b[t])$
- Hand dynamics: $\mathbf{x}_h[t+1] = f_h(\mathbf{x}_h[t], \mathbf{w}_h[t])$
- **No cross-terms!** Coupling exists only in observation space, not process model.

### Standard RTS Review

**Forward pass (filtering):**
```
For t = 1 to T:
  Predict:  x⁻[t], P⁻[t] = predict(x⁺[t-1], P⁺[t-1])
  Update:   x⁺[t], P⁺[t] = update(x⁻[t], P⁻[t], z[t])
  Store: x⁺[t], P⁺[t], x⁻[t], P⁻[t]
```

**Backward pass (smoothing):**
```
Initialize: x^s[T] = x⁺[T], P^s[T] = P⁺[T]

For t = T-1 down to 1:
  Smoother gain: C[t] = P⁺[t] F^T (P⁻[t+1])⁻¹
  State:         x^s[t] = x⁺[t] + C[t](x^s[t+1] - x⁻[t+1])
  Covariance:    P^s[t] = P⁺[t] + C[t](P^s[t+1] - P⁻[t+1])C[t]^T
```

Where $F = \frac{\partial f}{\partial \mathbf{x}}$ is the state transition Jacobian.

### Hierarchical RTS Algorithm

**Forward pass: Hierarchical filtering (already described)**

```
For t = 1 to T:
  # Pass 1: Body UKF
  x_b⁻[t], P_b⁻[t] = body_predict(x_b⁺[t-1], P_b⁺[t-1])
  x_b⁺[t], P_b⁺[t] = body_update(x_b⁻[t], P_b⁻[t], z_body[t])

  # Pass 2: Hand UKF (per hand)
  For each hand h:
    x_h⁻[t], P_h⁻[t] = hand_predict(x_h⁺[t-1], P_h⁺[t-1])
    x_h⁺[t], P_h⁺[t] = hand_update(x_h⁻[t], P_h⁻[t], z_h[t], x_b⁺[t])

  # Store all predictions and posteriors for smoothing
  Store: x_b⁺[t], P_b⁺[t], x_b⁻[t], P_b⁻[t]
         x_h⁺[t], P_h⁺[t], x_h⁻[t], P_h⁻[t]  (for each hand)
```

**Backward pass: Independent smoothers**

```
# Body smoother
x_b^s[T] = x_b⁺[T]
P_b^s[T] = P_b⁺[T]

For t = T-1 down to 1:
  # Body smoother gain
  C_b[t] = P_b⁺[t] F_b^T (P_b⁻[t+1])⁻¹

  # Body smoothed state
  x_b^s[t] = x_b⁺[t] + C_b[t](x_b^s[t+1] - x_b⁻[t+1])
  P_b^s[t] = P_b⁺[t] + C_b[t](P_b^s[t+1] - P_b⁻[t+1])C_b[t]^T

# Hand smoother (per hand, independent)
For each hand h:
  x_h^s[T] = x_h⁺[T]
  P_h^s[T] = P_h⁺[T]

  For t = T-1 down to 1:
    # Hand smoother gain
    C_h[t] = P_h⁺[t] F_h^T (P_h⁻[t+1])⁻¹

    # Hand smoothed state
    x_h^s[t] = x_h⁺[t] + C_h[t](x_h^s[t+1] - x_h⁻[t+1])
    P_h^s[t] = P_h⁺[t] + C_h[t](P_h^s[t+1] - P_h⁻[t+1])C_h[t]^T
```

### UKF Variant (Sigma Point Smoothing)

For UKF, the smoother gain uses sigma points instead of Jacobian:

```cpp
// During forward pass, save predicted sigma points for each pass
χ_b⁻[t+1] = predict_sigma_points(x_b⁺[t], P_b⁺[t])
χ_h⁻[t+1] = predict_sigma_points(x_h⁺[t], P_h⁺[t])  // per hand

// Backward pass: compute cross-covariance for smoother gain
// Body smoother gain
P_xx'_b[t] = 0
For each sigma point i:
  P_xx'_b[t] += w_i (χ_b[t] - x_b⁺[t])(χ_b⁻[t+1] - x_b⁻[t+1])^T
C_b[t] = P_xx'_b[t] (P_b⁻[t+1])⁻¹

// Hand smoother gain (same structure, per hand)
P_xx'_h[t] = 0
For each sigma point i:
  P_xx'_h[t] += w_i (χ_h[t] - x_h⁺[t])(χ_h⁻[t+1] - x_h⁻[t+1])^T
C_h[t] = P_xx'_h[t] (P_h⁻[t+1])⁻¹
```

### Why This Works

1. **No dynamic coupling**: Body motion doesn't depend on hand state, hands don't depend on arm state (in joint angle space)

2. **Observation coupling irrelevant for smoothing**: RTS operates in state space, uses only process model. Measurement coupling doesn't affect backward pass.

3. **Information flow**:
   - Body smoother: Propagates body trajectory constraints backward in time
   - Hand smoother: Propagates hand trajectory constraints backward in time
   - No cross-contamination needed

4. **Consistency with filtering**: We assumed body-hand independence during filtering (discarded cross-covariances). Maintaining this in smoothing is consistent.

### Computational Benefits

1. **Efficiency**: Two small smoothers instead of one huge smoother
   - Body: ~60 DOFs → 120×120 covariances
   - Hands: ~20 DOFs each → 40×40 covariances each
   - vs Monolithic: ~100 DOFs → 200×200 covariances
   - Matrix inversions scale as O(n³) → massive savings

2. **Numerical stability**: Smaller matrices → better conditioning, fewer numerical errors

3. **Modularity**: Same smoother code for body and hands, just different dimensions

4. **Parallelization**: Hand smoothers can run in parallel (independent)

### Implementation Notes

**Storage requirements:**
```cpp
// Per frame, store:
struct FrameData {
    // Body
    VectorXd x_body_prior, x_body_posterior;
    MatrixXd P_body_prior, P_body_posterior;
    std::vector<VectorXd> body_sigma_points;  // For UKF smoother gain

    // Per hand
    std::map<std::string, VectorXd> x_hand_prior, x_hand_posterior;
    std::map<std::string, MatrixXd> P_hand_prior, P_hand_posterior;
    std::map<std::string, std::vector<VectorXd>> hand_sigma_points;
};

std::vector<FrameData> trajectory(num_frames);  // Store full trajectory
```

**State transition Jacobian (for linear process model):**
```cpp
// Jacobian is simple for q[t+1] = q[t] + dt*q̇[t]
// Body and hand blocks are independent
F_body = [I_n×n, dt*I_n×n]
         [0_n×n, I_n×n    ]

F_hand = [I_m×m, dt*I_m×m]
         [0_m×m, I_m×m    ]
```

### Validation Strategy

1. **Consistency check**: Smoothed trajectories should be smoother than filtered (duh, but verify!)

2. **Coordinate consistency**: Body-hand attachment (wrist) should remain consistent after smoothing

3. **Temporal consistency**: No sudden jumps or discontinuities

4. **Physical plausibility**: Velocity/acceleration profiles should be reasonable

5. **Compare to ground truth**: If available, measure improvement in accuracy

## Architectural Design Details

### Shared DOFs Between Hierarchical Levels

**Problem**: Some joints participate in multiple filter levels. For body-hand case:
- Wrist joint: Needed in body pass (arm endpoint) AND hand pass (hand base reference)
- Palm joints: Used in body pass for wrist constraint AND hand pass for finger tracking

**Three architectural options:**

#### Option A: Parent Overwrites Child (Original Implicit Assumption)

```
1. Body pass: Estimate body + wrist DOFs (using palm markers for constraint)
2. Hand pass: Estimate hand + fingers, body+wrist FIXED from body pass
3. Merge: Keep body estimate, DISCARD hand's opinion of shared DOFs
```

**Pros:**
- Simple: Parent is authoritative
- Clear hierarchy: Information flows parent → child only

**Cons:**
- **Wastes information**: Hand pass sees dense finger markers that inform wrist orientation, but this is ignored
- Suboptimal: Child filter may have better estimate of shared DOFs due to more observations

#### Option B: Child Overwrites Parent (User's Intuition) ⭐ **RECOMMENDED**

```
1. Body pass: Estimate body + wrist DOFs (palm markers as weak constraint)
2. Hand pass: Re-estimate wrist + hand + fingers (dense observations)
3. Merge: Keep body estimate, hand OVERWRITES shared DOFs (wrist/palm)
```

**Rationale:**
- Palm markers in body pass: **Weak constraint** (3-6 markers, competing with 40+ body markers)
- Palm markers in hand pass: **Strong constraint** (same 3-6 markers, but primary role is orienting hand)
- Child filter has more *relative* information about boundary DOFs

**Pros:**
- **Better estimates**: Uses all available information optimally
- Physically motivated: Hand observations are more informative about wrist than body observations are

**Cons:**
- Slightly more complex: Need careful covariance handling
- Risk: If hand pass fails (occlusion), lose wrist estimate from body

**Implementation:**
```cpp
// After both passes, merge with child priority
for (dof in shared_dofs) {
    state_[dof] = hand_state[dof];           // Child overwrites
    covariance_.block(dof, dof) = hand_cov.block(dof, dof);  // Child uncertainty
}
```

#### Option C: Completely Separate States (No Shared DOFs)

```
1. Body pass: Estimate body UP TO wrist (wrist is BOUNDARY, not estimated)
2. Hand pass: TAKES wrist as input (fixed reference frame), estimates hand/fingers relative to wrist
3. No overlap: Body and hand have distinct DOF sets
```

**How it works:**
- Body estimates: root, spine, shoulders, elbows, **wrist position/orientation**
- Hand receives wrist transform as input (not state variable)
- Hand estimates: palm angles, finger joints (all relative to wrist frame)
- Palm markers: Split assignment
  - 2-3 markers: Body pass (to constrain wrist endpoint)
  - Remaining markers: Hand pass (to anchor hand base)

**Pros:**
- **Cleanest separation**: No ambiguity about DOF ownership
- No covariance merging issues
- Clear interface: Wrist transform is the contract between levels

**Cons:**
- More rigid: Wrist cannot be adjusted by hand observations
- Requires careful marker assignment (which palms to body, which to hand?)
- May lose some information if split is suboptimal

**Implementation:**
```cpp
// After body pass
Eigen::Isometry3d wrist_transform = compute_wrist_transform(body_state);

// Hand pass uses wrist as fixed reference
hand_state = estimate_hand(hand_observations, wrist_transform);

// No merging needed - distinct state spaces
```

#### Option D: Independent States with Synchronization ⭐ **RECOMMENDED**

```
1. Parent filter: Has its own state [body, arms, hand.*, palm.01.*, palm.04.*]
   - Runs full predict + update cycle
   - Output: parent_state, parent_cov

2. Child filter: Has its own state [hand.*, palm.*, fingers.*]
   - Uses parent's body/arm DOFs as FIXED in FK (read-only)
   - Runs full predict + update cycle
   - Output: child_state, child_cov

3. Merge: Combine for output (child wins overlap)
   - output_state = parent_state + child_state (child overwrites shared DOFs)

4. Synchronize: Update parent with child's shared DOFs for next frame
   - parent_state[shared] = child_state[shared]
   - Maintains temporal consistency
```

**Rationale for overlap (hand.* + palm.01.* + palm.04.*):**

1. **Minimum viable constraint**: Need ≥2 palm markers to determine hand pose
2. **Joint lock handling**: When elbow is locked (fully extended), shoulder rotation is ambiguous without knowing hand orientation
3. **Cross-covariances matter**: palm-shoulder and palm-arm correlations affect body estimate
4. **"As small as possible, but not smaller"**: This is the minimal set that provides necessary constraints

**Why synchronization is critical:**

Without sync:
```
Frame t:   Parent believes palm.01.R = 0.5, Child believes 0.6 → Output 0.6
Frame t+1: Parent predicts from 0.5 (stale!), Child predicts from 0.6 ✓
           → Parent's innovation is wrong, corrupts body estimate via cross-cov
```

With sync:
```
Frame t:   Output 0.6, then sync parent_state[palm] = 0.6
Frame t+1: Parent predicts from 0.6 ✓, Child predicts from 0.6 ✓
           → Temporal consistency maintained
```

**Implementation:**

```cpp
class HierarchicalUKF {
public:
    void step(const std::vector<Observation>& obs) {
        // 1. Parent filter (completely independent)
        parent_.predict(dt_);
        parent_.update(filter_observations(obs, parent_config_.groups));

        // 2. Child filters (independent, use parent DOFs as fixed)
        for (auto& child : children_) {
            // Extract fixed DOFs from parent for FK
            auto fixed_dofs = extract_dofs(parent_.state(), child.fixed_dof_names);

            child.predict(dt_);
            child.update(filter_observations(obs, child.config.groups), fixed_dofs);
        }

        // 3. Merge for output (child overwrites shared DOFs)
        output_state_ = parent_.state();
        for (auto& child : children_) {
            for (auto& dof_name : child.config.shared_dofs) {
                output_state_[dof_name] = child.state()[dof_name];
            }
        }

        // 4. Synchronize shared DOFs back to parent (for temporal consistency)
        if (config_.enable_sync) {
            for (auto& child : children_) {
                for (auto& dof_name : child.config.shared_dofs) {
                    parent_.state()[dof_name] = child.state()[dof_name];

                    // Optional: Also sync covariance (inflate to be conservative)
                    if (config_.sync_covariance) {
                        auto parent_var = parent_.covariance()(dof_idx, dof_idx);
                        auto child_var = child.covariance()(dof_idx, dof_idx);
                        parent_.covariance()(dof_idx, dof_idx) = std::max(parent_var, child_var);
                    }
                }
            }
        }
    }

private:
    UKF parent_;
    std::vector<UKF> children_;
    HierarchyConfig config_;
    VectorXd output_state_;
};
```

**Pros:**
- ✅ **Simplest architecture**: Truly independent filters with standard UKF interface
- ✅ **Clean separation**: No complex state extraction during filtering
- ✅ **Temporal consistency**: Synchronization prevents belief divergence
- ✅ **Robust**: Parent cannot be corrupted during child filtering
- ✅ **Natural constraints**: Parent includes palm joints → proper cross-covariances for joint locks
- ✅ **Optional sync**: Can disable to compare approaches

**Cons:**
- Slight memory overhead: Overlapping DOFs stored in both parent and child
- Post-facto sync: Parent state modified after its filter step (but clean+explicit)

**Fallback handling:**

```cpp
// Only sync if child converged successfully
if (child.num_inliers() > min_inliers && child.innovation_norm() < max_innov) {
    // Sync parent with child's better estimate
    sync_shared_dofs(parent_, child);
} else {
    // Child failed, parent's estimate stands
    // Don't sync, parent will use its own prediction next frame
}
```

### Simplified Configurable Hierarchy System

**Design goal**: Simple configuration in tracking TOML, referencing skeleton groups, no hard-coding.

**Configuration in tracking TOML file:**

```toml
[tracking]
# Standard UKF parameters (used as defaults)
process_noise_std = 0.15
measurement_noise_std = 20.0
outlier_threshold = 4.0
alpha = 0.1

[tracking.hierarchical]
enabled = true
enable_sync = true  # Synchronize shared DOFs after merge
sync_covariance = true  # Also sync covariance (conservative inflation)

# Parent filter definition
[tracking.hierarchical.parent]
# Reference groups defined in skeleton YAML
joint_groups = ["main", "HandR", "HandL"]  # Includes hand.* + selected palm joints

# Which groups to use for observations
observation_groups = ["body", "arms", "HandR", "HandL"]  # Uses markers from these groups

# Optional: Override UKF parameters for parent
process_noise_std = 0.15
measurement_noise_std = 20.0
outlier_threshold = 4.0

# Child filters (one per limb/appendage)
[[tracking.hierarchical.children]]
name = "hand_right"

# Joint groups for this child's state
joint_groups = ["HandR"]  # Full right hand (hand.R + palm.* + fingers.*)

# Observation groups
observation_groups = ["HandR"]  # All right hand markers

# DOFs shared with parent (child will overwrite these in output)
shared_dofs = ["hand.R", "palm.01.R", "palm.04.R"]

# DOFs from parent that are fixed (used in FK, not estimated)
# Can use wildcards: all joints not in joint_groups or shared_dofs
fixed_parent_dofs = "auto"  # or explicit: ["root", "spine.*", "shoulder.R", ...]

# Child-specific UKF parameters
process_noise_std = 0.25  # Higher for agile fingers
measurement_noise_std = 12.0  # Lower for dense markers
outlier_threshold = 3.5  # Stricter

# Convergence check for sync fallback
min_inliers_ratio = 0.3
max_innovation_norm = 500.0

[[tracking.hierarchical.children]]
name = "hand_left"
joint_groups = ["HandL"]
observation_groups = ["HandL"]
shared_dofs = ["hand.L", "palm.01.L", "palm.04.L"]
fixed_parent_dofs = "auto"
process_noise_std = 0.25
measurement_noise_std = 12.0
outlier_threshold = 3.5
min_inliers_ratio = 0.3
max_innovation_norm = 500.0
```

**Skeleton YAML structure (existing, no changes needed):**

```yaml
joints:
  - name: forearm.R
    parent: upper_arm.R
    group: main  # Body/arm group

  - name: hand.R
    parent: forearm.R
    group: HandR  # Right hand group (shared with parent!)

  - name: palm.01.R
    parent: hand.R
    group: HandR

  - name: f_index.01.R
    parent: palm.01.R
    group: HandR

markers:
  - name: RWrist1
    parent: hand.R
    group: HandR  # Inherits from parent joint
```

**Why this is simple:**

1. **References existing skeleton groups**: No duplication, just point to groups
2. **Parent = superset**: Parent's joint_groups can include child groups (natural for overlap)
3. **Explicit shared DOFs**: Clear declaration of what's shared
4. **Auto fixed DOFs**: Don't need to list every body joint
5. **Per-child parameters**: Easy to tune each limb independently
6. **Enable/disable sync**: Experiment with temporal consistency easily

**Alternative hierarchy example (body + feet):**

```toml
[tracking.hierarchical.parent]
joint_groups = ["main", "FootR", "FootL"]  # Include ankle + base toe joints
observation_groups = ["body", "legs"]

[[tracking.hierarchical.children]]
name = "foot_right"
joint_groups = ["FootR"]
shared_dofs = ["foot.R", "toes.01.R"]
# ... parameters ...
```

**No code changes needed** - just configuration!
## Open Questions

1. **Covariance initialization for child levels:**
   - Start with zero covariance (overconfident)?
   - Or use inflated initial covariance (conservative)?
   - Or propagate from parent (subset extraction)?

2. **Shared DOF covariance:**
   - When child overwrites shared DOFs, should covariance reflect:
     - Only child's local uncertainty (current approach)?
     - Combination of parent + child uncertainty?
     - Parent uncertainty propagated through child estimate?

3. **Temporal consistency:**
   - Should child covariance persist across frames?
   - Or reset per frame (if dynamics very different)?

4. **Failure modes:**
   - What if parent fails (no observations)?
   - Skip entire hierarchy? Use prediction only?

5. **Extension to 3+ levels:**
   - Current design assumes 2 levels
   - Generalize to N levels? (recursive implementation)
   - Use cases for 3+ levels?

6. **Cross-covariance in smoothing:**
   - Maintain parent-child cross-covariances during smoothing?
   - Or continue assumption of independence?

## References

- **Multi-level state estimation**: Koolen et al., "Design of a Momentum-Based Control Framework and Application to the Humanoid Robot Atlas", IJRR 2016
- **Hierarchical filtering**: Thrun et al., "Probabilistic Robotics", Chapter 3.4 (Hierarchical filtering)
- **Coarse-to-fine tracking**: Gordon et al., "Depth-Based Object Tracking Using a Robust Gaussian Filter", ICRA 2012
- **Information-theoretic sensor selection**: Krause et al., "Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies", JMLR 2008

## Appendix: Covariance Coupling Mathematics

### Detailed Derivation of Spurious Coupling

**Setup:**
- State: $\mathbf{x} = [q_\text{elbow}, q_\text{finger}, \dot{q}_\text{elbow}, \dot{q}_\text{finger}]^T$
- Observation: $\mathbf{y}_\text{finger}$ (2D marker position on finger)
- Forward kinematics: $\mathbf{z}_\text{finger} = h(q_\text{elbow}, q_\text{forearm}, q_\text{wrist}, q_\text{hand}, q_\text{finger})$

**Sigma Points with Large Elbow Uncertainty:**
$$
\boldsymbol{\chi}^{(i)} = \bar{\mathbf{x}} + \sqrt{(\lambda + n_\alpha)P} \cdot \mathbf{e}_i
$$

Where $P$ has large elbow component: $P_\text{elbow,elbow} = (626°)^2 \approx 119 \text{ rad}^2$

**Sigma Point Spread in Observation Space:**
$$
\mathbf{z}^{(i)}_\text{finger} = h(\boldsymbol{\chi}^{(i)})
$$

Due to long kinematic chain:
$$
\left\|\frac{\partial h}{\partial q_\text{elbow}}\right\| \approx L_\text{arm} + L_\text{forearm} + L_\text{hand} \approx 0.6 \text{ m}
$$

So:
$$
\Delta \mathbf{z}_\text{finger} \approx 0.6 \text{ m} \times 10° \approx 0.1 \text{ m} = 100 \text{ mm}
$$

In pixel space (~2 pixels/mm): 200 pixels spread.

**Cross-Covariance:**
$$
P_{xz}[\text{elbow}, \text{finger}] = \sum_i w_i \cdot \underbrace{(\chi^{(i)}_\text{elbow} - \bar{q}_\text{elbow})}_{\text{large}} \cdot \underbrace{(z^{(i)}_\text{finger} - \bar{z}_\text{finger})}_{\text{large due to FK}}
$$

Result: **Strong spurious correlation** despite weak actual observability.

**Kalman Gain Effect:**
$$
K_\text{elbow,finger} = P_{xz}[\text{elbow}, \text{finger}] \cdot P_{zz}^{-1}[\text{finger}, \text{finger}]
$$

This is **large** because:
- Numerator $P_{xz}$: Large due to sigma point spread
- Denominator $P_{zz}$: Moderate (innovation covariance)

Result: Finger observations **strongly update elbow state**, despite weak information content.

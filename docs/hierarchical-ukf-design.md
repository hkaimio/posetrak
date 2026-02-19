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
- **Helps**: Slows covariance growth rate
- **Doesn't solve**: Structural coupling problem remains
- **Risk**: Too low process noise → divergence (observed at 0.1)

**Fundamental issue**: Process noise controls temporal growth, not cross-DOF coupling.

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

```cpp
// UKF::update() modification
void UKF::update(const std::vector<Observation>& observations) {
    // Pass 1: Body tracking
    auto body_observations = filter_observations_by_group(observations, {"main"});
    auto body_state = state_;  // Copy state
    // Include minimal palm markers for wrist constraint
    body_observations += select_palm_markers(observations, 2);  // e.g., 2 per hand

    // Run UKF update on body DOFs only
    ukf_update_selective(body_state, body_observations, body_dof_mask);

    // Pass 2: Hand tracking (per hand)
    for (hand in ["HandL", "HandR"]) {
        auto hand_observations = filter_observations_by_group(observations, {hand});
        auto hand_state = body_state;  // Start from updated body

        // Lock body DOFs, only update hand DOFs
        ukf_update_selective(hand_state, hand_observations, hand_dof_mask);
    }

    state_ = hand_state;  // Final combined state
}
```

**Expected Benefits:**

1. **Bounded elbow covariance**: <150° (vs 626° current)
2. **Maintained finger tracking accuracy**: Dense markers + strict outlier rejection
3. **Robustness**: Body tracking unaffected by finger noise
4. **Physical consistency**: Respects dynamic differences (body vs fingers)

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
1. Refactor `UKF::update()` to support DOF masking
2. Create marker grouping configuration (body vs hands)
3. Implement Pass 1 (body) + Pass 2 (hands) sequential updates
4. Add configuration parameters for per-pass settings

**Testing:**
- Frame 685 single-frame test: Check elbow covariance stays <200°
- Frame 600-700 sequence: Verify no runaway growth
- Frame 915-920 divergence region: Verify stability
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

## Open Questions

1. **Covariance initialization for hands:**
   - Start Pass 2 with zero hand covariance (overconfident)?
   - Or use inflated initial covariance (conservative)?
   - Or propagate from Pass 1 (subset extraction)?

2. **Wrist constraint strength:**
   - How many palm markers needed in body pass?
   - Trade-off: More markers → better wrist estimate, but more coupling?

3. **Temporal consistency:**
   - Should hand covariance persist across frames?
   - Or reset per frame (hands are fast, covariance grows rapidly)?

4. **Failure modes:**
   - What if body pass fails (no arm markers visible)?
   - Skip hand pass? Use prediction only?

5. **Extension to feet:**
   - Apply same hierarchy to feet? (Usually fewer markers, less critical tracking)

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

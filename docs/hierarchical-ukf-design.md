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

## Implementation Plan

### Existing Infrastructure (Already Available!)

The codebase **already has** skeleton group infrastructure:

- **Skeleton groups schema**: `groups:` section in YAML lists joints/markers per group (see `Harri_skeleton-finger-group.yaml` lines 1947-2100)
- **Group parsing**: `skeleton_loader.cpp` reads `groups:` section and assigns each `joint.group` field
- **Group filtering**: `Skeleton::set_active_groups(groups)` activates joints by group names

However, analysis of `ukf.cpp`, `process_model.cpp`, and `sigma_points.cpp` reveals that DOF index arithmetic is **duplicated in at least 5 places** and all components hard-code the assumption that the root joint is always floating. This must be fixed before hierarchical tracking is possible.

---

### Phase 0a: `SkeletonLayout` — Single Source of DOF Arithmetic (2 days)

**Goal**: Create one canonical, precomputed, immutable description of how a set of joints maps to state vector indices. Eliminate all ad-hoc joint iteration loops.

**Core type: `JointDesc`** — all information a hot loop needs, computed once:

```cpp
struct JointDesc {
    std::string name;
    JointType   type;

    uint32_t storage_dof_count; // Elements in State::joint_angles (1/3/0)
    uint32_t active_dof_count;  // Free DOFs after accounting for locked axes
    uint32_t state_index;       // Start index in State::joint_angles / velocities
    uint32_t error_index;       // Start index in the joint portion of error-state
                                // (after root's 6: pos+ori, or vel+angvel)

    bool is_floating_root;      // True only for the kinematic root of a free body

    std::array<Eigen::Vector2d, 3> limits;
    uint32_t limit_count;
    std::array<bool, 3> active_dof_mask; // Which axes are free (SPHERICAL)
};
```

**`SkeletonLayout` class:**

```cpp
class SkeletonLayout {
public:
    // Pure query — does NOT mutate skeleton. Builds from group membership.
    static std::shared_ptr<const SkeletonLayout>
    from_groups(Skeleton const& skeleton, std::vector<std::string> const& group_names);

    static std::shared_ptr<const SkeletonLayout>
    from_full_skeleton(Skeleton const& skeleton);

    // O(1) accessors — all values precomputed at construction
    std::vector<JointDesc> const& joints() const;  // In state-vector order
    JointDesc const* get_joint(std::string const& name) const;  // O(1) via unordered_map

    uint32_t total_storage_dof_count() const;   // Size of State::joint_angles
    uint32_t joint_active_dof_count() const;    // Sum of active_dof_count across non-root joints
    uint32_t root_error_dof_count() const;      // 6 if has_floating_root(), else 0
    int      error_state_dim() const;           // 2 * (root_error_dof_count() + joint_active_dof_count())
    bool     has_floating_root() const;         // False for child filters (e.g. hand)

    // Build a merge index map: for each DOF in `subset`, the corresponding index
    // in THIS layout's state vector. Called ONCE at SubsetUKF construction, cached.
    // Throws if subset contains joints not present in this layout.
    std::vector<uint32_t> build_index_map_from(SkeletonLayout const& subset) const;

private:
    std::vector<JointDesc> joints_;
    std::unordered_map<std::string, uint32_t> name_to_idx_;  // For O(1) get_joint()
    uint32_t total_storage_dof_count_ = 0;
    uint32_t joint_active_dof_count_  = 0;
    bool has_floating_root_ = false;
};
```

**Tasks:**
1. Implement `JointDesc` and `SkeletonLayout` in `include/posetrak/core/skeleton_layout.hpp` and `src/core/skeleton_layout.cpp`
2. `from_groups()`: iterates skeleton joints, filters by group name, computes all indices in one pass
3. `from_full_skeleton()`: same as `from_groups()` but with no filter
4. `build_index_map_from()`: one-time O(N) name-matching, returns cached `vector<uint32_t>`

**Verification:**
- [ ] Unit test: `from_full_skeleton()` — total DOF matches `skeleton.total_dof_count()`
- [ ] Unit test: `from_groups({"main"})` — DOF count and `state_index` fields correct
- [ ] Unit test: `from_groups({"HandR"})` — indices start where "main" ends
- [ ] Unit test: `build_index_map_from()` — correct indices, throws on unknown joints
- [ ] Unit test: `has_floating_root()` — true for full skeleton, false for hands-only

**Success Criteria**: All DOF index arithmetic lives here; no other file computes joint→index mapping

---

### Phase 0b: Migrate `UnscentedKalmanFilter`, `ConstantVelocityModel`, `SigmaPointGenerator` (2-3 days)

**Goal**: Replace all ad-hoc joint iteration loops with `SkeletonLayout::joints()` iteration.

**Pattern**: every loop of the form `for (auto const& joint : skeleton_.get_joints_ordered())` with manual `angle_idx` tracking becomes:

```cpp
for (JointDesc const& j : layout_->joints()) {
    if (j.is_floating_root) { /* handle root */ continue; }
    if (j.type == JointType::REVOLUTE) {
        // j.state_index is the exact index — no arithmetic
        new_angles[j.state_index] += velocities[j.state_index] * dt;
    } else if (j.type == JointType::SPHERICAL) {
        auto aa = angles.segment<3>(j.state_index);  // j.state_index precomputed
        // ... SO(3) integration ...
    }
}
```

**Child filter root handling**: `has_floating_root() = false` → process model skips free body integration. Before each frame, the caller sets the full parent state:

```cpp
// In HierarchicalUKF::step():
hand_filter.set_parent_state(parent_.skeleton_state().state());
hand_filter.predict(dt);  // Root not integrated; only finger DOFs propagated
```

`set_parent_state` stores a full-skeleton-sized `State`. Each sigma point is expanded
by copying this background state and overwriting the child DOF positions (via merge_map_)
before calling FK. This gives FK the correct world-space joint chain even though the
child UKF only optimizes the finger DOFs.

**Files to modify:**
- `include/posetrak/filters/ukf.hpp` / `src/filters/ukf.cpp` — replace `Skeleton const&` with `shared_ptr<const SkeletonLayout>`, replace all joint loops
- `include/posetrak/filters/process_model.hpp` / `src/filters/process_model.cpp` — same
- `include/posetrak/filters/sigma_points.hpp` / `src/filters/sigma_points.cpp` — same

**Verification:**
- [ ] All existing UKF tests pass unchanged (full skeleton = identical behaviour)
- [ ] All existing process model tests pass
- [ ] Frame-by-frame numerical comparison: monolithic result identical before/after

**Success Criteria**: Zero behaviour change for existing code; DOF index arithmetic eliminated from these files

---

### Phase 0c: `SkeletonState` — State with Context (1 day)

**Goal**: Replace anonymous `State + vector<int>` with a self-describing type. Enables type-safe merge.

```cpp
class SkeletonState {
public:
    static SkeletonState create(std::shared_ptr<const SkeletonLayout> layout, State state);

    std::shared_ptr<const SkeletonLayout> const& layout() const;
    State const& state() const;
    State&       state();

    // Merge this state's DOFs into target using a precomputed index map.
    // merge_map built once via target.layout()->build_index_map_from(*this->layout())
    void merge_into(SkeletonState& target, std::vector<uint32_t> const& merge_map) const;

    // Extract subset covariance (for synchronization step)
    Eigen::MatrixXd extract_covariance_subset(
        Eigen::MatrixXd const& full_cov,
        std::vector<uint32_t> const& index_map) const;

private:
    std::shared_ptr<const SkeletonLayout> layout_;
    State state_;
};
```

**Delete** `include/posetrak/filters/subset_utils.hpp` and `src/filters/subset_utils.cpp` — functionality absorbed into `SkeletonState` and `SkeletonLayout`.

**Verification:**
- [ ] Unit test: round-trip extract/merge via `merge_into()` preserves all DOF values
- [ ] Unit test: merge of `{"HandR"}` subset into full skeleton overwrites correct indices only
- [ ] Unit test: covariance extraction symmetric

**Success Criteria**: No raw `vector<int>` index maps visible outside of `SkeletonLayout`/`SkeletonState`

---

### Phase 1: Configuration System (1 day)

**Goal**: Load hierarchical tracking configuration from TOML file.

**Tasks:**
1. **Define configuration structures**
   ```cpp
   struct ChildFilterConfig {
       std::string name;
       std::vector<std::string> joint_groups;
       std::vector<std::string> observation_groups;
       double process_noise_std, measurement_noise_std, outlier_threshold;
       double min_inliers_ratio, max_innovation_norm;
   };

   struct HierarchicalConfig {
       bool enabled = false;
       bool enable_sync = true;
       bool sync_covariance = true;
       std::vector<std::string> parent_joint_groups;
       std::vector<std::string> parent_observation_groups;
       // Parent UKF parameters...
       std::vector<ChildFilterConfig> children;
   };
   ```

2. **TOML parser integration**
3. **Create test configuration** `cpp_test_config_hierarchical.toml`

**Verification:**
- [ ] Unit test: Parse sample TOML, verify all fields loaded
- [ ] Config validation catches invalid group names

**Success Criteria**: Configuration loads without errors, validation works

---

### Phase 2: `SubsetUKF` (2 days)

**Goal**: A filter operating on a named subset of DOFs, using the foundation from Phase 0.

```cpp
class SubsetUKF {
public:
    SubsetUKF(Skeleton const& skeleton,
              std::shared_ptr<const SkeletonLayout> layout,  // from from_groups()
              std::vector<int> const& merge_map,            // full_layout->build_index_map_from(*layout)
              double process_noise_std,
              double alpha = 0.001, double beta = 2.0, double kappa = 0.0);

    void predict(double dt);
    UpdateResult update(std::vector<Observation> const& observations,
                       std::unordered_map<int, Camera> const& cameras,
                       ForwardKinematics& fk,
                       double measurement_noise_std,
                       double outlier_threshold);

    // Provide the full-skeleton state from the parent filter.
    // Must be called each frame before predict()/update().
    // During update, each sigma point is expanded from compact to full-skeleton
    // size by overwriting child DOFs in this background state before calling FK.
    // Child filters that are not floating-root use the background root pose too.
    void set_parent_state(State const& full_parent_state);

    SkeletonState skeleton_state() const;

private:
    // Expand compact child state into a full-skeleton State for FK.
    // Writes compact DOFs into background_state_ at merge_map_ positions.
    State expand_state(State const& compact) const;

    std::shared_ptr<const SkeletonLayout> layout_;
    std::vector<int> merge_map_;  // compact DOF i → index in full-skeleton State
    State compact_state_;         // layout-relative, compact-sized
    State background_state_;      // full-skeleton-sized, set by set_parent_state()
};
```

**Observation filtering**: pass only observations whose marker's `group` field is in `active_groups`.

**Verification:**
- [ ] Unit test: `SubsetUKF({"main"})` + body markers → matches monolithic UKF numerically
- [ ] Unit test: `SubsetUKF({"HandR"})` + hand markers → valid finger angle output
- [ ] Frame-by-frame comparison on tracking_tests sequence

**Success Criteria**: SubsetUKF produces identical results to monolithic when using all groups

---

### Phase 3: Hierarchical Execution & Merging (2 days)

**Goal**: Orchestrate parent and child filters, merge results via `SkeletonState::merge_into()`.

```cpp
class HierarchicalUKF {
public:
    void step(std::vector<Observation> const& observations, double dt);
    SkeletonState const& output() const;

private:
    void run_parent(std::vector<Observation> const& obs, double dt);
    void run_children(std::vector<Observation> const& obs, double dt);
    void merge_results();  // Child overwrites shared DOFs via precomputed merge_map_

    SubsetUKF parent_;
    std::vector<SubsetUKF> children_;
    SkeletonState output_;
    std::vector<std::vector<uint32_t>> child_merge_maps_;  // Built once at construction
};
```

**Verification:**
- [ ] Frame 685 test: Elbow covariance < 200° (vs 626° baseline)
- [ ] Frame 685 test: Finger predictions reasonable (< 100 pixels)
- [ ] Visualization in Rerun

**Success Criteria**: Hierarchical produces valid output, elbow covariance significantly reduced

---

### Phase 4: Synchronization & Robustness (1-2 days)

**Goal**: Implement temporal consistency and fallback strategies.

**Tasks:**
1. Post-merge: sync parent shared DOFs from child result (optional, config-controlled)
2. Convergence guard: skip sync if child has low inlier ratio or high innovation norm
3. Fallback: if child diverges, keep parent estimate for shared DOFs

**Verification:**
- [ ] Test: Enable sync, verify parent/child stay consistent
- [ ] Occlusion test: Remove all hand markers, verify parent-only fallback
- [ ] Frames 915-920: Verify convergence checks work

**Success Criteria**: Synchronization prevents divergence, occlusion handled gracefully

---

### Phase 5: Full Sequence Validation (2-3 days)

**Goal**: Validate on complete 959-frame dataset, compare to baseline.

**Quantitative targets:**
- Elbow covariance < 150° throughout (vs 626° peak)
- Zero divergence events
- Tracking RMSE similar or better to monolithic
- Runtime < 2× baseline

**Qualitative checks in Rerun:**
- Frame 685: Finger spread eliminated?
- Frames 915-920: Stable through divergence?

---

### Phase 6: Parameter Tuning (1-2 days)

**Goal**: Optimize UKF parameters separately for parent and child.

**Tasks:**
1. Body/parent: process_noise ~0.15 (current), measurement_noise ~20.0
2. Hand/child: process_noise ~0.25-0.30 (more agile), measurement_noise ~12.0-15.0 (denser markers)
3. Document final choices with ablation rationale

---

### Phase 7: Smoothing Integration (Future Work)

**Goal**: Extend to hierarchical RTS smoothing.

Independent smoothers for body and hands, storage management for trajectories.
**Note**: Lower priority, focus on filtering first.

---

## Success Metrics Summary

**Phase Gates:**
- ✅ Phase 0a: `SkeletonLayout` built, all DOF arithmetic centralised, tests pass
- ✅ Phase 0b: `UnscentedKalmanFilter`, `ConstantVelocityModel`, `SigmaPointGenerator` migrated — existing tests pass unchanged
- ✅ Phase 0c: `SkeletonState` merge/extract correct, `subset_utils` deleted
- ✅ Phase 1: Config loads correctly
- ✅ Phase 2: `SubsetUKF` matches monolithic
- ✅ Phase 3: Elbow covariance < 200° at frame 685
- ✅ Phase 4: Synchronization prevents divergence
- ✅ Phase 5: Full sequence clean, zero divergence
- ✅ Phase 6: Parameters tuned

**Primary Goals:**
1. Elbow covariance < 150° throughout (vs 626°)
2. No finger prediction spread (< 100 pixel residuals)
3. Zero divergence events in full sequence

**Secondary Goals:**
4. Tracking quality maintained (similar RMSE)
5. Acceptable overhead (< 2× runtime)
6. Extensible design (easy to add feet, etc.)

**Total Estimate**: 13-18 days (3-4 weeks) excluding smoothing

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

**Note**: Group names reference the `groups:` section in skeleton YAML (e.g., `Harri_skeleton-finger-group.yaml` lines 1947-2100), which lists joints and markers per group.

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
# Reference groups defined in skeleton YAML's groups: section
# Example from Harri_skeleton-finger-group.yaml:
#   groups:
#     - name: "main"
#       joints: [hips, spine1, ..., palm.01.L, palm.04.L, ...]
#     - name: "HandL"
#       joints: [palm.01.L, palm.02.L, ..., f_pinky.03.L]
joint_groups = ["main"]  # Uses joints from "main" group

# Which marker groups to use for observations
observation_groups = ["main"]  # Uses markers from "main" group

# Optional: Override UKF parameters for parent
process_noise_std = 0.15
measurement_noise_std = 20.0
outlier_threshold = 4.0

# Child filters (one per limb/appendage)

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

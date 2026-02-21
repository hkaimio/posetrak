# Hierarchical Tracker Redesign

*Working design doc — iterate before coding.*

## Problems with the Current Phase 2 Implementation

The in-progress `SubsetUKF` has three fundamental flaws that make it untenable:

1. **Full State inside child filter.** Every child UKF carries a 24-element State for a 2-DOF
   problem. The extra elements must be filled from the parent via `sync_from_background()` before
   every predict/update, which is fragile and breaks the independence assumption.

2. **Root handling baked into UKF.** `UnscentedKalmanFilter` has hardcoded root-joint logic
   (floating free-flyer always at index 0). There is no clean path to disable this for child
   filters whose root pose is external.

3. **Debug/stats output not filter-scoped.** All UKF instances write to the same directory and
   the statistics tracker is wired only to the outer `Tracker`.

The redesign below eliminates these problems by making each filter genuinely compact and
delegating coordination to a new `HierarchicalTracker`.

---

## Design Goals

- Each filter's `State` is **compact**: only the DOFs it owns.
- No filter needs to know about joints it does not track.
- FK for a child filter needs only the child subtree + one externally-supplied root transform.
- `Skeleton` is immutable; all structural reasoning goes through `SkeletonLayout`.
- `UnscentedKalmanFilter`, `ProcessModel`, `SigmaPointGenerator` take only a
  `shared_ptr<const SkeletonLayout>` — no separate `Skeleton` argument.
- Output, debug, and statistics are scoped per named filter.

---

## 1. Skeleton & SkeletonLayout Refactoring

### 1.1 Skeleton becomes shared and immutable

The application constructs one `Skeleton` at startup and wraps it immediately:

```cpp
auto skeleton = std::make_shared<const Skeleton>(load_skeleton(yaml_path));
```

`Skeleton` gets no new public mutating methods after this point. `set_active_groups()` is removed
from `Skeleton` entirely — group filtering moves into `SkeletonLayout` factory arguments.

> **Challenge:** `set_active_groups()` is currently called inside `UnscentedKalmanFilter`,
> `Tracker`, and the in-progress `SubsetUKF`. All callers must be updated to pass group lists to
> the layout factory instead.

### 1.2 SkeletonLayout owns the Skeleton pointer

```cpp
class SkeletonLayout {
public:
    // All factories take shared_ptr<const Skeleton>
    static std::shared_ptr<const SkeletonLayout>
    from_groups(std::shared_ptr<const Skeleton> skeleton,
                std::vector<std::string> const& group_names);

    static std::shared_ptr<const SkeletonLayout>
    from_full_skeleton(std::shared_ptr<const Skeleton> skeleton);

    std::shared_ptr<const Skeleton> const& skeleton() const { return skeleton_; }

private:
    std::shared_ptr<const Skeleton> skeleton_;  // kept alive by the layout
    // ... existing fields ...
};
```

Every object that needs skeleton data for computation reads it from the layout. No object receives
a bare `Skeleton const&` in its constructor.

### 1.3 UKF, ProcessModel, SigmaPointGenerator take only SkeletonLayout

```cpp
// Before
UnscentedKalmanFilter(Skeleton const& skeleton,
                      std::shared_ptr<const SkeletonLayout> layout, ...);

// After
UnscentedKalmanFilter(std::shared_ptr<const SkeletonLayout> layout, ...);
```

The `Skeleton` needed for FK inside UKF's `update()` is obtained from `layout->skeleton()`.

---

## 2. Subtree ForwardKinematics

### 2.1 Enhanced PinocchioModelBuilder — no sub-Skeleton needed

Rather than building a new `Skeleton` object for the child subtree, we extend
`PinocchioModelBuilder` with a new factory:

```cpp
// Build a pinocchio model covering only the joints in group_names, treating
// freeflyer_joint_name as the world-attached root (6-DOF free-flyer).
// All joints in group_names that are descendants of freeflyer_joint_name are included.
// Markers belonging to those joints are attached as operational frames.
// freeflyer_joint_name itself is NOT in group_names; it is the external anchor.
static void build_subtree_model(
    Skeleton const& skeleton,
    std::string const& freeflyer_joint_name,   // e.g. "wrist.R"
    std::vector<std::string> const& group_names, // e.g. {"HandR"}
    pinocchio::Model& model);
```

This reads joint offsets, types, and marker positions directly from the original `Skeleton`
— no data duplication, no new data structure. The builder walks descendants of
`freeflyer_joint_name` and includes those whose group is in `group_names`.

> **Challenge — connectivity assertion:** The builder should assert (or the coordinator's
> constructor should verify) that all joints in `group_names` are descendants of
> `freeflyer_joint_name` in the skeleton tree, forming a single connected subtree.
> For finger groups this is trivially true; config validation should enforce it.

### 2.2 Which joint is the free-flyer?

The free-flyer root for a child tracker is **the skeleton parent of the shallowest joint in
the child's groups** — not the shallowest joint itself.

**Why:** If the shallowest child joint (e.g., `palm.R`) were the free-flyer root, then its
joint angles would need to be folded into the free-flyer's 6-DOF pose on every sigma point,
producing complex and error-prone math. Instead:

- `wrist.R` (palm.R's parent) becomes the pinocchio free-flyer. It has **no DOFs in the child
  state** — its world transform is set once per frame by the coordinator.
- `palm.R` is then a regular spherical/revolute pinocchio joint with its own angles, estimated
  normally by the child UKF alongside the finger joints.

Result: the child tracker's state for HandR is:
```
State(N):  joint_angles = [palm.R(3), finger1.R(1), finger2.R(1), ...]
           root_position / root_orientation = wrist.R world transform (external, fixed per frame)
```

The coordinator extracts `wrist.R`'s world transform from the parent FK after the parent
update step.

> **Shared markers:** A palm marker appearing in both parent and child observation groups is
> fine. Each FK instance is independent; the same marker is observed through two different
> kinematic models and contributes to two separate UKF updates. No special handling needed.

### 2.3 Root transform injection

`ForwardKinematics::compute(State const& state)` already reads root position and orientation
from `state.root_position()` / `state.root_orientation()`. For child trackers, the coordinator
calls `child_ukf.set_root_transform(wrist_R_world)` once per frame before predict/update.
The child UKF stores this and injects it into every sigma point's `State` before FK evaluation.

Two cases in the UKF's sigma-point handling:

- **Parent filter** (`layout->has_floating_root() == true`): root pose is part of the error
  state and is perturbed by sigma points. Existing behaviour, unchanged.
- **Child filter** (`layout->has_floating_root() == false`): root pose is **not** varied by
  sigma points. The UKF injects the stored `root_world_transform_` into each sigma point
  before passing it to FK. This is one extra branch in `SigmaPointGenerator::generate()`
  and `UnscentedKalmanFilter::compute_measurement_sigma_points()`.

This is a small, localised change that does not alter the parent filter's behaviour at all.

---

## 3. Child Tracker State

A child filter tracking HandR (palm.R + 2 finger joints) has:

```
State(7):  joint_angles = [palm.R(3), finger1.R(1), finger2.R(1)]
           root_position / root_orientation = wrist.R world transform (external, not estimated)
```

`SkeletonLayout::from_groups(skeleton, {"HandR"}, /*has_root=*/false)` produces:

```
error_dim = 2*(3+1+1) = 10   (no root contribution; 5 storage DOFs, 5 active DOFs)
total_storage_dof_count = 5
```

The coordinator calls `child_ukf.set_root_transform(wrist_R_world)` before each predict/update,
where `wrist_R_world` comes from `parent_fk_.world_transform("wrist.R")` after the parent
update. The child UKF propagates palm.R angles and finger angles through sigma points as
ordinary joint DOFs; the free-flyer root is held constant within each predict/update cycle.

---

## 4. HierarchicalTracker

```cpp
class HierarchicalTracker {
public:
    HierarchicalTracker(std::shared_ptr<const Skeleton> skeleton,
                        HierarchicalConfig const& config);

    bool initialize(std::vector<Observation> const& obs,
                    std::unordered_map<int, Camera> const& cameras,
                    double timestamp);

    void step(std::vector<Observation> const& obs,
              std::unordered_map<int, Camera> const& cameras,
              double timestamp);

    // Full assembled state for output
    SkeletonState const& full_state() const;

    // Per-filter statistics (parent + each child by name)
    std::unordered_map<std::string, FilterStats> const& stats() const;

private:
    // Per-frame sequencing (see §5)
    void run_parent(std::vector<Observation> const&, cameras, double dt);
    void run_children(std::vector<Observation> const&, cameras, double dt);
    void merge_states();
    void sync_to_parent();

    std::unique_ptr<UnscentedKalmanFilter> parent_ukf_;
    std::unique_ptr<ForwardKinematics>     parent_fk_;

    struct ChildFilter {
        std::string name;
        std::unique_ptr<UnscentedKalmanFilter> ukf;
        std::unique_ptr<ForwardKinematics>     fk;
        std::shared_ptr<const SkeletonLayout>  layout;
        std::vector<int>                       merge_map;   // child compact DOF → full state idx
        std::string                            freeflyer_joint_name;  // skeleton parent of child root
        //   e.g. "wrist.R" for HandR — set from parent tracker, not estimated by child
        ChildFilterConfig                      config;
    };
    std::vector<ChildFilter> children_;

    std::shared_ptr<const Skeleton>       skeleton_;
    std::shared_ptr<const SkeletonLayout> full_layout_;   // from_full_skeleton, for merge_map
    SkeletonState                         full_state_;
    HierarchicalConfig                    config_;
    double                                last_timestamp_ = -1.0;
};
```

### 4.1 Per-frame sequence

```
given: obs (all observations), cameras, timestamp
dt = timestamp - last_timestamp_

1. parent_ukf_.predict(dt)
2. parent_ukf_.update(filter_obs(obs, parent_config.obs_groups), cameras, parent_fk_)
   → parent now has best estimate of body + boundary joints

2b. parent_fk_.compute(parent_ukf_.state())
    → refreshes pinocchio cache to posterior state so world_transform() is correct

3. for each child c:
   a. freeflyer_world = parent_fk_.world_transform(c.freeflyer_joint_name)
      // c.freeflyer_joint_name = skeleton parent of the shallowest joint in child's groups
      // e.g. "wrist.R" for HandR group (palm.R's parent)
   b. c.ukf.set_root_transform(freeflyer_world)
   c. c.ukf.predict(dt)
   d. c.ukf.update(filter_obs(obs, c.config.obs_groups), cameras, c.fk)

4. merge_states():
   full_state = parent full state (compact via full_layout_)
   for each child c:
     for i in 0..compact_dof_count(c):
       full_state.joint_angles[ c.merge_map[i] ] = child.state().joint_angles[i]
   (child always wins on overlapping DOFs)

5. if config_.enable_sync:
   sync_to_parent():
     parent_ukf_.set_state(scatter child DOFs into parent state)
     if config_.sync_covariance:
       inflate parent cov for those DOFs (take max of parent var, child var)

6. last_timestamp_ = timestamp
```

`ForwardKinematics::world_transform(joint_name, state)` is a new method needed on FK — returns
the 4×4 world transform of a named joint frame (distinct from `compute()` which returns marker
positions). Pinocchio caches these after `forwardKinematics()`, so it is cheap.

> **Open question:** Should `world_transform()` be added to `ForwardKinematics`, or should the
> coordinator call `compute()` and then run its own subtree FK? Adding it to FK is cleaner.

---

## 5. Merge Map Construction

Each child needs a precomputed `merge_map` that says "compact child DOF i → index in the full
assembled state". This is built at construction time by matching joint names between the child
layout and `full_layout_`.

`SkeletonLayout::build_index_map_from()` already does this. The full assembled state uses
`from_full_skeleton()` layout offsets.

---

## 6. Output, Debug, and Statistics

### 6.1 Tracking output

`HierarchicalTracker::full_state()` returns the merged `SkeletonState`. The existing
`TrackingExporter` receives this and writes joint angles, velocities, etc. No change needed there.

### 6.2 Debug output

Each `UnscentedKalmanFilter` accepts a `debug_dir` string at construction:

```cpp
parent_ukf_ = make_ukf(parent_layout, config, "tracking_output/debug/parent");
child_ukf   = make_ukf(child_layout,  config, "tracking_output/debug/HandR");
```

Sub-paths are derived from the filter's name field in config. This replaces the current
single-hardcoded-path problem.

### 6.3 Statistics

`FilterStats` (innovation norms, inlier counts, covariance trace) is collected inside each
`UnscentedKalmanFilter::update()` and returned in `UpdateResult`. `HierarchicalTracker::stats()`
aggregates `UpdateResult` from all filters per frame and exposes them keyed by filter name.
`StatisticsTracker` wraps `HierarchicalTracker` instead of raw `Tracker`.

---

## 7. Tracker Becomes HierarchicalTracker

`Tracker` is extended in-place to become `HierarchicalTracker`. With an empty `children_` list
its behaviour is bit-identical to the current single-filter path. There is no separate code path.

Migration steps:

1. Rename `Tracker` → `HierarchicalTracker` (typedef `Tracker = HierarchicalTracker` for a
   transitional period if the CLI or other callers need it).
2. Add `children_` vector (empty by default).
3. `step()` runs the existing UKF predict/update, then for each child runs the child sequence
   described in §4.1.
4. The zero-children integration test **must pass before** any child-related code is exercised.
   This acts as a regression guard: if the refactor breaks existing behaviour, the single-filter
   test catches it immediately.

The existing `Tracker` unit tests and integration tests become the zero-children
`HierarchicalTracker` tests with no changes needed.

---

## 8. Open Questions

**Q1: Free-flyer joint identity — RESOLVED**

The free-flyer root for a child tracker is the **skeleton parent of the shallowest joint in
the child's groups**, not the shallowest joint itself. For HandR: free-flyer = `wrist.R`,
child state starts at `palm.R`. This avoids folding palm.R's estimated angles into the
free-flyer 6-DOF pose on every sigma point. The coordinator extracts `wrist.R`'s world
transform from the parent FK after the parent update, sets it on the child UKF once, and
the child then evaluates `palm.R` and fingers as ordinary pinocchio joints.

**Q2: Child initialisation — RESOLVED**

The global state is initialised once via the existing method (IK or rest pose), exactly as today.
The coordinator then slices the relevant DOFs out of that global state to seed each child filter's
initial `State`. Child covariance is initialised to the same diagonal as today's single-filter
initialisation. This requires no new initialisation logic — child filters are just views into the
same initial global state.

**Q3: Child predict without parent update**

If the parent update fails (too few inliers), should children still run? The boundary joint
transform would come from a stale or prediction-only parent state. This is probably acceptable
— fingers can still track relative to a predicted wrist position.

*Recommendation: run children regardless; they will see a covariance-inflated root.*

**Q4: Covariance sync direction**

Option D from the existing doc syncs child→parent for shared DOFs after merge, taking `max` of
the two variances. This ensures the parent does not become overconfident about DOFs the child
estimated better. However: if the child has very low covariance (many finger markers, dense
observations) but the parent has high covariance for those same DOFs (few palm markers among many
body markers), then `max` is conservative but still correct. Is there a case where we'd want to
take the child's covariance directly (not max)? Probably worth experimenting; the sync is
optional and controlled by `sync_covariance` flag.

**Q5: `ForwardKinematics::world_transform()` — RESOLVED**

Add `world_transform(std::string const& joint_name) const` to `ForwardKinematics`, returning
the 4×4 homogeneous world transform of the named joint frame. Pinocchio caches these in
`data.oMi[]` (indexed by pinocchio joint index) after `forwardKinematics()` is called, so the
method is a trivial lookup — one map access to get the pinocchio joint index, one array access
for the transform.

Ordering contract: the coordinator calls `parent_ukf_.update()`, which internally calls FK on
each sigma point. After `update()` returns, the parent's posterior state is set but the FK
cache holds transforms for the last sigma point evaluated — **not** the posterior. The
coordinator therefore calls `parent_fk_.compute(parent_ukf_.state())` once after update
before querying `world_transform()`. This one extra FK evaluation is cheap (it is a single
full-skeleton FK pass, same cost as one sigma point).

**Q6: Two-level vs recursive**

The design above is naturally recursive: a `ChildFilter` could itself be a
`HierarchicalTracker`. For now, restrict to two levels by making `ChildFilter` hold a
`UnscentedKalmanFilter` directly rather than a nested `HierarchicalTracker`. If a third level is
ever needed, the type can be generalised.

---

## 9. Phase 3d — Detailed Design and Changes

Phase 3d establishes the building blocks needed by child filters: a subtree Pinocchio model,
a layout-aware config converter, and the first end-to-end child FK test. No runtime behaviour
changes; only new code with new tests.

### 9.1 `PinocchioModelBuilder::build_subtree_model()`

#### What it does

Builds a self-contained Pinocchio model for a subtree of the skeleton, suitable for a child
filter whose root pose is externally supplied each frame.

```cpp
// NEW declaration in pinocchio_model_builder.hpp

/// @brief Build a Pinocchio model for a subtree rooted at @p freeflyer_joint_name.
///
/// @param skeleton  Full skeleton (read-only; not mutated).
/// @param freeflyer_joint_name  Skeleton joint that becomes the Pinocchio free-flyer.
///        This joint itself contributes NO DOFs to the child state — only its world
///        transform is ever set on the child UKF.  The joint must exist in skeleton.
/// @param group_names  Groups whose joints form the child subtree.  Every joint in these
///        groups must be a descendant of freeflyer_joint_name (connectivity assertion).
///        FIXED joints in these groups are silently skipped (no pinocchio joint added).
/// @param[out] model  Cleared and rebuilt by this call.
///
/// Pinocchio joint order in the resulting model:
///   universe → freeflyer_joint_name (FreeFlyer) → child joints in skeleton insertion order
///
/// @throws std::invalid_argument if freeflyer_joint_name not in skeleton, or if any
///         joint in group_names is not a descendant of freeflyer_joint_name.
static void build_subtree_model(Skeleton const& skeleton,
                                std::string const& freeflyer_joint_name,
                                std::vector<std::string> const& group_names,
                                pinocchio::Model& model);

/// @brief Build marker frame map for a subtree model.
/// Only markers whose parent joint is included in the subtree are returned.
static std::map<std::string, pinocchio::FrameIndex>
build_subtree_marker_frame_map(pinocchio::Model const& model,
                               Skeleton const& skeleton,
                               std::string const& freeflyer_joint_name,
                               std::vector<std::string> const& group_names);
```

#### Algorithm

```
build_subtree_model(skeleton, freeflyer_joint_name, group_names, model):

  1. Validate: freeflyer_joint_name exists in skeleton.
  2. Build group_set = {group_names}.
  3. Find freeflyer_idx = index of freeflyer_joint_name in skeleton.joints().
  4. Connectivity assertion:
       For every joint j in skeleton whose group ∈ group_set:
           walk j's ancestor chain upward until hitting freeflyer_idx or the root.
           If freeflyer_idx is not encountered → throw (disconnected group).
  5. Add freeflyer_joint_name as FreeFlyer at universe (pinocchio parent = 0,
     placement = SE3::Identity()).  Call it freeflyer_pin_id.
  6. add_subtree_joints_recursive(model, skeleton, freeflyer_idx,
                                  freeflyer_pin_id, group_set, joint_to_id)
  7. add_marker_frames_for_groups(model, skeleton, group_set, joint_to_id)
```

`add_subtree_joints_recursive(model, skel, parent_skel_idx, parent_pin_id, group_set, joint_to_id)`:

```
  for each child_joint in skel.joints() where child_joint.parent_index == parent_skel_idx:
    child_skel_idx = index of child_joint
    in_group = (group_set.count(child_joint.group) > 0)

    if child_joint.type == FIXED:
      // Map to parent so markers can attach
      joint_to_id[child_joint.name] = parent_pin_id
      // Still recurse — a descendant might be in-group
      add_subtree_joints_recursive(model, skel, child_skel_idx, parent_pin_id, group_set, joint_to_id)

    else if in_group:
      // Add as normal joint (using same logic as add_joint_recursive, but always using offset)
      pin_id = model.addJoint(parent_pin_id, joint_type, SE3(offset, rest_rotation), child_joint.name)
      model.appendBodyToJoint(pin_id, Inertia::Identity())
      joint_to_id[child_joint.name] = pin_id
      add_subtree_joints_recursive(model, skel, child_skel_idx, pin_id, group_set, joint_to_id)

    else:
      // Joint is a non-group non-fixed joint in the subtree — should not happen if
      // connectivity assertion passes, because all descendants of freeflyer must be
      // in-group or fixed.  Log a warning and skip (do NOT recurse).
      // (The connectivity assertion in step 4 already prevents in-group joints being
      // children of skipped non-group joints.)
```

#### Key difference from `build_model()`

| | `build_model()` | `build_subtree_model()` |
|---|---|---|
| Free-flyer joint | Skeleton root (no parent) | Named interior joint |
| Root offset | Ignored (placed at SE3::Identity) | Named joint also placed at Identity |
| Joints added | All non-fixed joints | Only joints whose group ∈ group_names |
| Marker frames | All markers | Only markers on included joints |

The root-offset-ignored rule carries over unchanged: the freeflyer joint in the subtree model
is placed at identity, because its actual world transform is set externally on the child UKF
every frame (it is the "wrist.R world transform" injected by the coordinator).

---

### 9.2 `ForwardKinematics::state_to_config()` — layout-aware overload

The existing `state_to_config(State const& state, Skeleton const& skeleton)` iterates over
`skeleton.joints()` and maps each non-root, non-fixed joint in skeleton order to the pinocchio
`q` vector. This relies on the pinocchio joint ordering matching the skeleton joint ordering,
which holds for the full-skeleton case.

For the child FK, the pinocchio model only contains the subtree joints in skeleton insertion
order. The child `State` stores joint angles in layout order (same as skeleton insertion order
filtered to the group). A layout-aware overload can build `q` directly from the layout without
iterating the full skeleton:

```cpp
// NEW static method in forward_kinematics.hpp / forward_kinematics.cpp

/// @brief Convert a compact child-filter State to a Pinocchio config vector.
///
/// For child filters (has_floating_root == false in theory, but here the config vector
/// still starts with [pos(3), quat_xyzw(4)] because the pinocchio model has a free-flyer;
/// those 7 values come from state.root_position() / state.root_orientation()).
///
/// @param state Compact state — root_position/orientation = injected freeflyer transform;
///              joint_angles = child joints in layout order.
/// @param layout Layout that was used to build the child pinocchio model.  Its joints()
///               must be in the same order as the pinocchio model joints (after the
///               free-flyer), which is guaranteed when both are built from the same
///               skeleton in insertion order.
/// @param skeleton Full skeleton — needed to look up joint types (REVOLUTE/SPHERICAL)
///                 and compute quaternion representation for spherical joints.
static Eigen::VectorXd state_to_config(State const& state,
                                       SkeletonLayout const& layout,
                                       Skeleton const& skeleton);
```

Implementation sketch:
```cpp
Eigen::VectorXd ForwardKinematics::state_to_config(
    State const& state, SkeletonLayout const& layout, Skeleton const& skeleton) {

    // Child pinocchio model nq = 7 (freeflyer) + sum of config DOFs per layout joint
    // Compute nq from layout
    int nq = 7;  // root: pos(3) + quat(4)
    for (auto const& desc : layout.joints()) {
        nq += (desc.type == JointType::SPHERICAL ? 4 : 1);
    }

    Eigen::VectorXd q(nq);

    // Root: position + quaternion (xyzw) — injected freeflyer world transform
    q.segment<3>(0) = state.root_position();
    Eigen::Quaterniond const& ori = state.root_orientation();
    q[3] = ori.x(); q[4] = ori.y(); q[5] = ori.z(); q[6] = ori.w();

    int q_offset = 7;
    for (auto const& desc : layout.joints()) {
        if (desc.type == JointType::SPHERICAL) {
            // Convert 3-axis angles from state to quaternion for pinocchio
            // (same logic as existing state_to_config for spherical joints)
            Joint const* j = skeleton.get_joint(desc.name);
            // ... build quaternion from state.joint_angles().segment<3>(desc.state_index)
            q.segment<4>(q_offset) = /* quaternion xyzw */;
            q_offset += 4;
        } else {  // REVOLUTE
            q[q_offset++] = state.joint_angles()[desc.state_index];
        }
    }
    return q;
}
```

> **Implemented:** `ForwardKinematics` now takes a single `shared_ptr<const SkeletonLayout>`
> constructor argument. The skeleton is accessed via `layout->skeleton()`. The unified
> `state_to_config(State, SkeletonLayout)` handles both full-skeleton and child-subtree
> models — see §9.5.

---

### 9.3 Files Changed in Phase 3d

| File | Change |
|------|--------|
| `include/posetrak/kinematics/pinocchio_model_builder.hpp` | Declare `build_subtree_model()`, `build_subtree_marker_frame_map()` |
| `src/kinematics/pinocchio_model_builder.cpp` | Implement both; extract `add_subtree_joints_recursive()` and `add_marker_frames_for_groups()` as private helpers |
| `include/posetrak/kinematics/forward_kinematics.hpp` | Declare `state_to_config(State, SkeletonLayout, Skeleton)` overload; add optional `layout_` member |
| `src/kinematics/forward_kinematics.cpp` | Implement the overload; branch in `compute(State const&)` |
| `tests/test_pinocchio_integration.cpp` | Add subtree model tests (see §9.4) |

No other files change. In particular, `SkeletonLayout`, `UKF`, `Tracker`, and CLI are
untouched — Phase 3d is purely additive.

---

### 9.4 Tests for Phase 3d

All in `tests/test_pinocchio_integration.cpp` (extend existing file) under `[subtree_model]` tag.

**Skeleton fixture** (shared):
```
pelvis (root, FIXED offset=(0,0,0))
  └── spine (SPHERICAL, group="main")
  └── upper_arm.R (SPHERICAL, group="main")
       └── forearm.R (REVOLUTE, group="main")
            └── wrist.R (FIXED, group="main")   ← freeflyer for HandR
                 └── palm.R (SPHERICAL, group="HandR")
                      └── finger1.R (REVOLUTE, group="HandR")
                      └── finger2.R (REVOLUTE, group="HandR")
markers: MRK-palm (on palm.R), MRK-tip1 (on finger1.R), MRK-body (on spine)
```

**Test cases:**

1. **Subtree joint count**: `build_subtree_model(skel, "wrist.R", {"HandR"}, model)` →
   `model.njoints == 4` (universe + wrist.R freeflyer + palm.R + finger1.R + finger2.R = 5, but universe is 0 so njoints=5) and `model.nv == 6 + 3 + 1 + 1 = 11`.

2. **Freeflyer at origin**: `model.jointPlacements[1]` (wrist.R in pinocchio) == SE3::Identity().

3. **Child joint offsets preserved**: palm.R's placement in model has the offset from skeleton.

4. **Only child markers included**: `build_subtree_marker_frame_map(...)` returns
   {MRK-palm, MRK-tip1} — MRK-body (on spine, group="main") is absent.

5. **FK gives correct marker positions at identity state**:
   Construct a `State` with root at origin and zero angles. Compute FK with subtree model.
   Compare MRK-palm position to expected offset (palm.R offset relative to wrist.R world origin).

6. **FK with injected root transform**:
   Give state a non-zero root position/orientation. Verify MRK-palm moves accordingly.

7. **Connectivity assertion fails**: `build_subtree_model(skel, "forearm.R", {"HandR"}, model)`
   should throw (palm.R is a descendant of wrist.R, not forearm.R directly — actually it is
   a descendant of forearm.R, so this should NOT throw; revisit with a genuinely disconnected
   case: `build_subtree_model(skel, "upper_arm.R", {"HandR"}, model)` where HandR is not
   directly under upper_arm.R's subtree children... wait, it IS under there. Use group="main"
   joints — those are not descendants of wrist.R → should throw).
   `build_subtree_model(skel, "wrist.R", {"main"}, model)` → throws (spine is not under wrist.R).

8. **`state_to_config` layout-aware overload**: build layout with `from_groups(skel_ptr, {"HandR"})`,
   check q-vector length == 11, root portion matches state.root_*, joint portion matches angles.

---

### 9.5 Implementation Notes (Phase 3d — completed)

**P1: `ForwardKinematics` layout member ownership — RESOLVED**

The final design went further than either option considered here. `ForwardKinematics` now has
a **single constructor** taking `shared_ptr<const SkeletonLayout>`. The skeleton is obtained
via `layout->skeleton()`; no separate `Skeleton const&` parameter exists. The `layout_` member
is always non-null.

```cpp
// Single constructor — works for both full-skeleton and child-subtree FK
ForwardKinematics(pinocchio::Model const& model, pinocchio::Data& data,
                  std::map<std::string, pinocchio::FrameIndex> const& marker_frame_map,
                  std::shared_ptr<const SkeletonLayout> layout);
```

Callers that previously passed a bare `Skeleton const&` now build a layout first:
```cpp
auto layout = SkeletonLayout::from_full_skeleton(skeleton_ptr);
ForwardKinematics fk(model, data, marker_map, layout);
```

The unified `state_to_config(State const&, SkeletonLayout const&)` works for both full-skeleton
and child-subtree models without any branching — both pinocchio models start with a 7-DOF
freeflyer followed by layout joints in insertion order. The legacy
`state_to_config(State const&, Skeleton const&)` is kept for test call sites that use it directly.

**P2: FIXED joints in group with non-FIXED parent in group**

The connectivity check in step 4 walks ancestor chains. If a joint in the group has an
ancestor that is also in the group (but a different group string — e.g., a fixed connector
joint), the traversal is still correct because fixed joints are mapped to their parent's
pinocchio id in `add_subtree_joints_recursive`. Marker attachment to FIXED joints naturally
goes to the parent pinocchio joint frame, matching the existing `build_model()` behaviour.

**P3: Revolute axis determination for subtree joints**

`get_revolute_axis()` is a private static method already shared by `build_model()`. It is
reused unchanged by `add_subtree_joints_recursive()`.

---

## 10. Phase 3e — Detailed Design and Changes

Phase 3e makes `UnscentedKalmanFilter` usable as a child filter by adding a fixed-root
injection path. The parent filter's predict/update cycle is entirely unchanged — the new code
only activates when `layout_->has_floating_root() == false`.

### 10.1 What changes

Two new responsibilities for `UnscentedKalmanFilter` when acting as a child filter:

1. **Accept an externally-supplied root transform** (`set_root_transform()`). The coordinator
   calls this once per frame before `predict()`/`update()`.
2. **Hold the root fixed throughout predict/update**. The process model would otherwise
   integrate root velocity and advance root position. This integration is discarded for child
   filters: the root in every propagated sigma point is overwritten with the stored transform
   after process model propagation.

### 10.2 New API on `UnscentedKalmanFilter`

```cpp
/// @brief Inject the externally-known root transform for child-filter mode.
///
/// Must be called once per frame by the coordinator before predict()/update().
/// Has no effect if layout_->has_floating_root() == true (safety no-op).
///
/// Also updates state_.root_position() / root_orientation() immediately so that
/// the stored nominal state is consistent before sigma generation.
void set_root_transform(Eigen::Vector3d const& position,
                        Eigen::Quaterniond const& orientation);
```

No other public API changes. `predict()` and `update()` signatures are unchanged.

New private members:

```cpp
// Only meaningful when !layout_->has_floating_root()
Eigen::Vector3d  fixed_root_pos_;   ///< Injected root position (child filter)
Eigen::Quaterniond fixed_root_ori_; ///< Injected root orientation (child filter)
```

### 10.3 Root injection in `predict()`

```cpp
void UnscentedKalmanFilter::predict(double dt) {
    // [CHILD FILTER] Root is already correct in state_ from set_root_transform().
    // Generate sigma points — root is NOT perturbed because root_error_dof_count() == 0.
    auto sigma_points = sigma_gen_.generate_sigma_points(state_, covariance_);

    // Propagate each sigma point through the process model
    std::vector<State> propagated;
    propagated.reserve(sigma_points.size());
    for (auto const& sp : sigma_points) {
        State prop = process_model_.propagate(sp, dt);

        // [CHILD FILTER] Undo root velocity integration — root is externally set.
        if (!layout_->has_floating_root()) {
            prop = State(fixed_root_pos_, fixed_root_ori_,
                         prop.joint_angles(),
                         prop.root_velocity(), prop.root_angular_velocity(),
                         prop.joint_velocities());
        }
        propagated.push_back(std::move(prop));
    }

    // compute_state_mean / compute_state_covariance unchanged
    // ...
}
```

Key points:

- `SigmaPointGenerator::generate_sigma_points()` already does the right thing: when
  `has_floating_root() == false`, `error_dim_` has no root component (`root_error_dof_count()`
  returns 0), so `apply_error_to_state()` never touches root fields. Sigma-point roots carry
  through from the nominal state unchanged.
- The process model propagates the full `State` including root; we selectively overwrite root
  after propagation. Joint angles and velocities propagate normally.
- Root velocities (`prop.root_velocity()`, `prop.root_angular_velocity()`) are kept from the
  propagated state so they stay consistent — they are ignored by FK but the state struct holds
  them.

### 10.4 Root injection in `update()`

No changes needed. Sigma points are generated from the current `state_` (root already set by
`predict()` via `set_root_transform()`). `predict_measurements()` calls FK on each sigma point;
FK reads `state.root_position()` / `state.root_orientation()`, which are correct. The
innovation, Kalman gain, and state update all operate on joint DOFs only (root is not in the
error state for child filters, so `apply_error_to_state()` leaves root untouched during the
mean/covariance update as well).

### 10.5 `SigmaPointGenerator` and `apply_error_to_state` — no changes

`has_floating_root() == false` is already handled by both:

- `SigmaPointGenerator` constructor: `error_dim_ = layout_->error_state_dim()`, which uses
  `root_error_dof_count() == 0`, so error vectors have no root slots at all.
- `apply_error_to_state()`: applies `error_vec[0..5]` to root position/orientation only when
  root DOFs exist in the error vector; with `has_floating_root() == false`, those slots aren't
  present and root is never touched.

No test changes. No `SigmaPointGenerator` changes. No `ProcessModel` changes.

### 10.6 Files Changed in Phase 3e

| File | Change |
|------|--------|
| `include/posetrak/filters/ukf.hpp` | Declare `set_root_transform()`; add `fixed_root_pos_`, `fixed_root_ori_` private members |
| `src/filters/ukf.cpp` | Implement `set_root_transform()` (updates `state_` immediately + stores members); add root overwrite in `predict()` after process model propagation |
| `tests/test_ukf_update.cpp` or new `tests/test_child_ukf.cpp` | Test: child-layout UKF with fixed root — after `set_root_transform()` + `predict(dt)`, root in `state()` must equal the injected transform regardless of prior velocity |

### 10.7 Coordinator usage pattern (preview)

```cpp
// Per frame, inside HierarchicalTracker::step():
parent_ukf_->predict(dt);
parent_ukf_->update(parent_obs, cameras, *parent_fk_);
parent_fk_->compute(parent_ukf_->state());         // refresh pinocchio cache to posterior

for (auto& child : children_) {
    // Extract boundary joint world transform from parent FK
    auto const& wrist_pos = ...;   // from parent_fk_->world_transform("wrist.R")
    auto const& wrist_ori = ...;
    child.ukf->set_root_transform(wrist_pos, wrist_ori);  // Phase 3e
    child.ukf->predict(dt);
    child.ukf->update(child_obs, cameras, *child.fk);
}
```

`ForwardKinematics::world_transform()` is the missing piece delivered in Phase 3f.

---

### 10.8 Implementation Notes (Phase 3e — completed)

**Design vs reality — sigma-point root guards:**

Section §10.5 stated that `apply_error_to_state()` already handles
`has_floating_root() == false` correctly. This turned out to be wrong: both
`apply_error_to_state()` (sigma_points.cpp) and `compute_state_error()` (ukf.cpp)
unconditionally accessed `error_vec.segment<3>(active_dof)` and
`error_vec.segment<3>(active_dof + 3)` for root velocity and angular velocity.
For a child filter with `root_n = 0` and `active_dof = 4` (e.g. palm.R(3) +
finger1.R(1)), those accesses require indices {7,8,9} from an 8-element vector —
an out-of-bounds Eigen assertion (SIGABRT).

Fix: wrapped all four root-section reads/writes in both functions with
`if (root_n > 0) { ... }`. Joint sections were already correct because
they use `root_n + j.error_index` (= 0 + ... for child filters).

**Actual files changed vs design:**

| File | Planned | Actual |
|------|---------|--------|
| `include/posetrak/filters/ukf.hpp` | Declare `set_root_transform()`; add private members | ✅ as planned |
| `src/filters/ukf.cpp` | Implement `set_root_transform()`; root overwrite in `predict()` | ✅ as planned |
| `src/filters/sigma_points.cpp` | No changes planned | **Added** `if (root_n > 0)` guard in `apply_error_to_state()` |
| `src/filters/ukf.cpp` | — | **Added** `if (root_n > 0)` guard in `compute_state_error()` |
| `tests/test_ukf_update.cpp` | One new test | **Three** tests added under `[ukf][child_filter]` |

All three `[ukf][child_filter]` tests pass. No regressions against HEAD^.

---

## 11. Phase 3f — Detailed Design and Changes

Phase 3f adds `ForwardKinematics::world_transform(joint_name)`, which lets the
coordinator extract a skeleton joint's world-frame pose from the parent FK cache
after each update. This is the last missing piece before Phase 3g can wire up
`HierarchicalTracker`.

### 11.1 What it does

```cpp
/// @brief Get world-frame pose of a named skeleton joint after compute().
///
/// Reads from the Pinocchio joint cache (data_.oMi[]).  Must be called after
/// compute() has been called at least once for the current configuration.
///
/// @param joint_name  Name of a non-fixed skeleton joint (ball or revolute).
/// @return {position, orientation} of the joint frame in world coordinates.
/// @throws std::out_of_range if joint_name is not a known pinocchio joint.
std::pair<Eigen::Vector3d, Eigen::Quaterniond>
world_transform(std::string const& joint_name) const;
```

The coordinator usage (from §10.7) becomes:

```cpp
parent_fk_->compute(parent_ukf_->state());  // refresh pinocchio cache to posterior

for (auto& child : children_) {
    auto [pos, ori] = parent_fk_->world_transform(child.freeflyer_joint_name);
    child.ukf->set_root_transform(pos, ori);
    child.ukf->predict(dt);
    child.ukf->update(child_obs, cameras, *child.fk);
}
```

### 11.2 Why this is simple

Real skeletons contain no FIXED joints in the interior of the hierarchy — only
`root`, `ball`, and `revolute` types are used. (The `fixed` type in the skeleton
format doc is a synonym for `root` and only applies to the top-level root joint.)
Therefore every joint that could serve as a child filter's freeflyer is a real
pinocchio joint with its own slot in `model_.joints` and a corresponding world
transform in `data_.oMi[]` after `forwardKinematics()`.

The Phase 3d test fixture artificially introduced a FIXED `wrist.R` mid-chain,
which is not representative of real skeletons. In practice the freeflyer boundary
joint (e.g. `hand.L`, a ball joint) is always non-FIXED, and no special frame
plumbing is needed.

### 11.3 Implementation

A new private member in `ForwardKinematics`:

```cpp
std::unordered_map<std::string, pinocchio::JointIndex> joint_id_map_;
```

Populated in the constructor body by iterating `model_.names` (pinocchio's
joint name array, indexed 0 = universe, 1..njoints-1 = actual joints):

```cpp
// In ForwardKinematics constructor (after existing init-list)
for (pinocchio::JointIndex i = 1; i < static_cast<pinocchio::JointIndex>(model_.njoints); ++i) {
    joint_id_map_[model_.names[i]] = i;
}
```

`world_transform()` implementation — reads `data_.oMi[]` which `forwardKinematics()`
already populates (no extra pinocchio call needed):

```cpp
std::pair<Eigen::Vector3d, Eigen::Quaterniond>
ForwardKinematics::world_transform(std::string const& joint_name) const {
    auto it = joint_id_map_.find(joint_name);
    if (it == joint_id_map_.end()) {
        throw std::out_of_range("world_transform: unknown joint '" + joint_name + "'");
    }
    pinocchio::SE3 const& T = data_.oMi[it->second];
    return {T.translation(), Eigen::Quaterniond(T.rotation())};
}
```

No changes to `PinocchioModelBuilder`. No changes to `model_.nframes`. Existing
`[subtree_model]` tests are unaffected.

### 11.4 Files Changed in Phase 3f

| File | Change |
|------|---------|
| `include/posetrak/kinematics/forward_kinematics.hpp` | Declare `world_transform()`; add `joint_id_map_` private member |
| `src/kinematics/forward_kinematics.cpp` | Populate `joint_id_map_` in constructor; implement `world_transform()` |
| `tests/test_pinocchio_integration.cpp` | Tests under `[world_transform]` tag (see §11.5) |

### 11.5 Tests

All in `tests/test_pinocchio_integration.cpp` under `[world_transform]` tag.
Use the existing §9.4 skeleton fixture but query only non-FIXED joints
(`palm.R`, `forearm.R`), matching real-skeleton constraints.

1. **Ball joint at identity root, zero angles**: `world_transform("palm.R")`
   position equals the composed offset chain (forearm.R offset + wrist.R offset +
   palm.R offset) with identity root.

2. **Non-identity root shifts all joints**: inject root position `{1, 0, 0}`;
   verify `world_transform("palm.R").first` shifts by exactly `{1, 0, 0}`.

3. **Non-zero joint angles rotate child joints**: set `palm.R` to a known
   rotation; verify `world_transform("forearm.R")` is unchanged but a marker on
   palm moves as expected (cross-check against `compute()` output).

4. **Unknown joint throws**: `world_transform("nonexistent")` throws
   `std::out_of_range`.

5. **Subtree FK**: build child FK for HandR (subtree model) with freeflyer =
   `wrist.R`; the subtree pinocchio model contains `wrist.R` as a real pinocchio
   joint (the FreeFlyer), so `world_transform("wrist.R")` returns the injected
   root transform unchanged; `world_transform("palm.R")` gives the correct
   world position.

6. **Stale data before first compute**: calling `world_transform()` before any
   `compute()` must not crash (pinocchio zero-initialises `data_.oMi`). The
   result value is unspecified but safe.

---

## 12. Implementation Phases

Approximate ordering (each independently committable):

| Phase | Status | Work |
|-------|--------|------|
| 3a | ✅ done | Remove `Skeleton` arg from `UKF`/`ProcessModel`/`SigmaPointGenerator`; read from layout |
| 3b | ✅ done | `SkeletonLayout` factory functions take `shared_ptr<const Skeleton>` |
| 3c | ✅ done | Remove `set_active_groups()` from `Skeleton`; pass groups to layout factory |
| 3d | ✅ done | `PinocchioModelBuilder::build_subtree_model()` + unified `ForwardKinematics` API (`shared_ptr<const SkeletonLayout>` constructor + `state_to_config(State, SkeletonLayout)`) + 9 passing `[subtree_model]` tests |
| 3e | ✅ done | `UnscentedKalmanFilter::set_root_transform()` + fixed-root sigma point path |
| 3f | **next** | `ForwardKinematics::world_transform(joint_name)` |
| 3g | | Rename `Tracker` → `HierarchicalTracker`; add empty `children_`; zero-children test |
| 3h | | Child filter construction, per-frame sequencing, state merge |
| 3i | | Debug dir scoping, stats aggregation, statistics tracker wiring |
| 3j | | CLI / config plumbing; `HierarchicalConfig` drives child construction |
| 3k | | Integration test: parent+child on recorded data, compare to single-filter baseline |

Phases 3a–3c are pure refactors with no behaviour change; existing tests must stay green.
Phases 3d–3f add new functionality with new tests.
Phase 3g is the first end-to-end hierarchical run (zero-children parity test).

---

## 13. What Gets Deleted

- `SubsetUKF` (header + implementation + tests) — replaced by the coordinator pattern.
- `sync_from_background()` — the concept disappears.
- `set_active_groups()` on `Skeleton` — filtering is a layout concern.
- `background_state_` inside SubsetUKF — state ownership is unambiguous in the new design.
- `extract_subtree()` — was considered but not implemented; replaced by `build_subtree_model()`.

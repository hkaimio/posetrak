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

## 9. Implementation Phases

Approximate ordering (each independently committable):

| Phase | Work |
|-------|------|
| 3a | Remove `Skeleton` arg from `UKF`/`ProcessModel`/`SigmaPointGenerator`; read from layout |
| 3b | `SkeletonLayout` factory functions take `shared_ptr<const Skeleton>` |
| 3c | Remove `set_active_groups()` from `Skeleton`; pass groups to layout factory |
| 3d | `PinocchioModelBuilder::build_subtree_model()` + tests |
| 3e | `UnscentedKalmanFilter::set_root_transform()` + fixed-root sigma point path |
| 3f | `ForwardKinematics::world_transform(joint_name)` |
| 3g | Rename `Tracker` → `HierarchicalTracker`; add empty `children_`; zero-children test |
| 3h | Child filter construction, per-frame sequencing, state merge |
| 3i | Debug dir scoping, stats aggregation, statistics tracker wiring |
| 3j | CLI / config plumbing; `HierarchicalConfig` drives child construction |
| 3k | Integration test: parent+child on recorded data, compare to single-filter baseline |

Phases 3a–3c are pure refactors with no behaviour change; existing tests must stay green.
Phases 3d–3f add new functionality with new tests.
Phase 3g is the first end-to-end hierarchical run.

---

## 10. What Gets Deleted

- `SubsetUKF` (header + implementation + tests) — replaced by the coordinator pattern.
- `sync_from_background()` — the concept disappears.
- `set_active_groups()` on `Skeleton` — filtering is a layout concern.
- `background_state_` inside SubsetUKF — state ownership is unambiguous in the new design.
- `extract_subtree()` — was considered but not implemented; replaced by `build_subtree_model()`.

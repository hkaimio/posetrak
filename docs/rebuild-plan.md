# Hierarchical Tracker Rebuild Plan

## Status

| Step | Status | Commit | Notes |
|------|--------|--------|-------|
| Step 0 | ✅ Complete | `cb50617` | Branch created, baseline config committed |
| Step 1 | 🔲 Not started | — | Layout-relative UKF indexing |
| Step 2 | 🔲 Not started | — | M:N joint/marker group membership |
| Step 3 | 🔲 Not started | — | Phase 3h: child filter construction & sequencing |
| Step 4 | 🔲 Not started | — | Phase 3i: debug scoping & per-filter statistics |

---

## Context

The `hkaimio/hierarch-track-3h-debug` branch accumulated several changes on top
of the Phase 3g baseline (`26835ee`) that interact in ways that broke the
parent-only tracking path.  Rather than patching forward, we rebuild 3h and 3i
step-by-step on top of 3g, with regression verification after every step.

**Regression reference:** commit `26835ee` (Phase 3g).
**Reference config:** `tests/harri-no-palms-head-debug.toml` with
`active_joint_groups = ["main"]`, no `child_filters`, palm joints **not** in
the "main" group.

---

## Guiding principle

> After every step: rebuild → run reference config → verify tracking stats
> match 26835ee exactly (average inliers, residuals, state trajectory).

A numerical diff of `tracking_stats.csv` is the acceptance criterion.

---

## Step 0 — New branch from 3g

```bash
git checkout -b hkaimio/hierarch-rebuild 26835ee
```

- Copy `tests/harri-no-palms-head-debug.toml` from the debug branch (or
  recreate it).  Point its output to a dedicated directory, e.g.
  `tracking_tests/rebuild-stepN/`, separate from the preserved 26835ee
  reference results in `tracking_tests/harri-no-palms-26835ee/`.
- Commit the config file as the only change so the baseline is clean.

---

## Step 1 — Layout-relative UKF indexing

### Background: the `state_index` contract break (root cause of leg-zeros regression)

This step fixes a class of bugs that was introduced in commit `a50b4af` (Phase
3c: "remove `set_active_groups` from Skeleton; all filtering via SkeletonLayout")
and was present in every commit from `a50b4af` through `9fdcf74`, causing
**all leg-joint state-vector entries to read as zero** in the exported CSV.

**The two contracts that were mixed:**

| Contract | State size | `JointDesc::state_index` meaning | Used since |
|---|---|---|---|
| **A — full-skeleton** | `skeleton->total_dof_count()` | Global offset into the full-skeleton state vector | Before `a50b4af` |
| **B — layout-local** | `layout->total_storage_dof_count()` | Local offset within the layout-only state vector | Intended by `cd41ff4` |

At `cb637b8` (last known-good before regression) the code used Contract A
throughout. `from_active_skeleton()` explicitly documented:
```
// IMPORTANT: state_index uses the full-skeleton State vector offset
// (advances over ALL non-fixed non-root joints, including inactive ones)
```
So `thigh.L` at full-skeleton index 51 → `state.joint_angles()[51]` → correct.

In `a50b4af`, `initialize_ukf` switched from `from_active_skeleton()` to
`from_full_skeleton()` / `from_groups()`. The new `build()` function assigns
**layout-local** `state_index` values (0-based, counting only joints in the
layout). But the UKF state remained full-skeleton-sized
(`skeleton->total_dof_count()`). Now `thigh.L` got `state_index = 39` (its
layout-local position) but the full-skeleton angle lived at index 51.
`compute_state_mean`, `write_sigma_points_csv`, and every loop that reads
`state.joint_angles()[j.state_index]` silently read from the wrong element.
The exported CSV showed zeros for leg joints because `compute_state_mean`
allocated a layout-sized output `State` and wrote into it via layout-local
offsets, leaving the trailing positions (legs) at their default of zero.

`ForwardKinematics::state_to_config` still used the old `state_to_config(State,
Skeleton)` overload that walked all skeleton joints sequentially — so FK itself
was temporarily unaffected, explaining why tracking didn't instantly collapse
but the exported state was wrong.

The `e101fd0` state-slice fix in `initialize_ukf` happened to restore correct
behaviour because slicing made the state layout-local-sized, which matched
the layout-local `state_index` values — both sides of the contract became
consistent again.

**The rebuild commits fully to Contract B (layout-local) from this step
forward.** The `state_to_config(State, Skeleton)` overload (sequential
`joint_angle_idx++`) must be **removed** to prevent future divergence; only
the layout-indexed `state_to_config(State, SkeletonLayout)` overload survives.

### Goal

In 26835ee the UKF `state_` is always full-skeleton-sized (Contract A) and all
loops iterate all skeleton joints.  When `active_joint_groups` restricts the
layout, joints outside the group are silently skipped but still carried in the
state vector.  This step migrates fully to Contract B: the UKF state is exactly
as wide as the layout requires, and every read/write uses layout-local indices.

### Changes

**`src/filters/ukf.cpp`**
- Size `state_` from `layout->total_storage_dof_count()` instead of
  `skeleton()->total_dof_count()`.
- Rewrite `compute_state_mean`, `write_sigma_points_csv`,
  `export_state_vector`, `generate_state_header` to iterate
  `layout_->joints()` instead of all skeleton joints.

**`src/tracking/tracker.cpp`**
- `initialize_ukf`: after creating the `UnscentedKalmanFilter`, slice
  `initial_state` to layout dimensions before calling `set_state`:
  ```cpp
  if (raw.num_dof() != layout->total_storage_dof_count()) {
      auto full_layout = SkeletonLayout::from_full_skeleton(skeleton_);
      auto index_map   = full_layout->build_index_map_from(*layout);
      // copy angles/velocities via index_map → State sliced; ukf_->set_state(sliced);
  }
  ```
- `initialize_ukf`: when `!groups.empty()`, rebuild the pinocchio model/data/FK
  scoped to those groups (so `state_to_config` and FK work on the layout-sized
  state, not the full skeleton).
- `initialize_from_rest_pose`: size the zero vector from
  `layout->total_storage_dof_count()`.
- Add `UKF::layout()` and `Tracker::fk()` accessors (needed by child filter
  construction in Step 3).

**`include/posetrak/core/skeleton_layout.hpp`** (optional but recommended)
- Add a `slice_state(State const& full_skeleton_state) -> State` helper method
  so every future call site has a safe, named operation instead of an inline
  if-branch.

### Why this is numerically identical to 26835ee

With `active_joint_groups = ["main"]` and no palm joints in the group, the
layout has 21 joints / 55 storage DOFs.  These are the same 21 joints the
26835ee UKF implicitly iterated.  The state-slice strips the trailing zeros
(unused joints) from the full-skeleton initial_state.csv.  The pinocchio model
rebuild produces the same subtree.  Result: identical numerics.

---

## Step 2 — M:N joint/marker group membership

### Goal

Replace `Joint::group` (single string, last-write-wins during YAML load) with
`Skeleton::joint_in_groups()` / `marker_in_groups()` lookups backed by a
set-of-sets structure.  This is necessary for joints that legitimately belong
to multiple groups (e.g. `palm.01.L` in both "main" and "HandL").

### Changes

**`include/posetrak/core/skeleton.hpp` / `src/core/skeleton.cpp`**
- Remove `Joint::group` field.
- Add `Skeleton::register_group(name, joint_names, marker_names)`.
- Add `Skeleton::joint_in_groups(joint_name, group_set) -> bool`.
- Add `Skeleton::marker_in_groups(marker_name, group_set) -> bool`.

**`src/io/skeleton_loader.cpp`**
- Replace `joint.group = …` with `skeleton->register_group(…)` calls.

**`src/core/skeleton_layout.cpp`**
- `build()`: use `skeleton->joint_in_groups(joint.name, group_set)` instead of
  `group_set.count(joint.group)`.

**`src/kinematics/pinocchio_model_builder.cpp`**
- `build_subtree_model` / `add_subtree_joints_recursive`: use
  `joint_in_groups()` for filtering.

**`src/filters/subset_ukf.cpp`**
- `filter_observations`: use `skeleton_.marker_in_groups(name, obs_group_set_)`
  instead of `obs_group_set_.count(marker.group)`.

**Tests:** update all tests that set `Joint::group` directly.

### YAML stays unchanged

Palm joints are **not** added to "main" at this point.  The M:N capability
simply makes the data model correct; it will be exercised when child filters
are configured explicitly in Step 3.

### Verification

Tracking output numerically identical to 26835ee.

---

## Step 3 — Phase 3h: child filter construction & per-frame sequencing

### Goal

Add the hierarchical feature.  The parent filter (e.g. "main") runs first;
then per-frame child filters (e.g. "HandL", "HandR") are constructed, injected
with the parent FK world-transform for their anchor joint, run their own
predict/update cycle, and their joint angles are merged back into the parent
state.

### Changes

**`include/posetrak/core/config.hpp`**
- `ChildFilterConfig` struct: `groups`, `anchor_joint`, process/measurement
  noise overrides.
- Add `child_filters` vector to `TrackerConfig`.

**`include/posetrak/tracking/tracker.hpp` / `src/tracking/tracker.cpp`**
- `ChildFilter` struct: owns a `UnscentedKalmanFilter`, `ForwardKinematics`,
  `SkeletonLayout`, anchor joint name, and a merge-index map.
- `build_children(initial_state)`: constructs one `ChildFilter` per
  `config_.child_filters`; returns immediately if the vector is empty so the
  parent-only path is entirely untouched.
- `run_child_step(child, observations, dt)`: inject root transform from parent
  FK → child UKF predict → child UKF update → merge child angles into parent
  state.
- `slice_state_for_child(global_state, child_layout) -> State`: extract child
  DOFs from parent state via index map.

**Tests**
- Child filter construction.
- Per-frame sequencing (anchor injection, predict, update, merge).
- State merge correctness.

### Verification (two passes)

1. Config with no `child_filters` → numerically identical to 26835ee.
2. Config with `child_filters = [{groups=["HandL","HandR"], anchor="hand.L"/
   "hand.R", …}]` → human review of hand/finger tracking quality in
   `tracking_tests/rebuild-step3/`.

---

## Step 4 — Phase 3i: debug scoping & per-filter statistics

### Goal

Operational quality-of-life: separate debug output directories per child
filter, per-filter inlier/residual statistics in the tracking stats CSV.

No tracking behaviour change.  Verification: parent-only config still
matches 26835ee.

---

## Key lesson integrated throughout

> Every site that feeds a `State` into a layout-scoped UKF (IK result, CSV
> load, `slice_state_for_child`) **must** verify
> `state.num_dof() == layout->total_storage_dof_count()` and slice if not.

The recommended guard is the `SkeletonLayout::slice_state()` helper proposed
in Step 1.  After that helper exists, all future call sites use it and the
mismatch that caused the original 26835ee→HEAD regression becomes a compile-
time-obvious naming choice rather than a silent size mismatch.

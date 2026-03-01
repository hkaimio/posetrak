# Bisect Regression Findings

## Setup

Three builds were compared using `scripts/bisect_run.sh` against the same input data
(`tests/harri-no-palms.toml`, `active_joint_groups = ["main"]`, 8 s at 120 Hz = 959 frames):

| Label | Commit | Description |
|-------|--------|-------------|
| `baseline` | `a50b4af` | Last commit before hierarchical-tracker work began |
| `post-pinocchio` | `f7cecd1` | First commit that rebuilds the pinocchio model for active groups |
| `head` | `7c8ff69` | Current HEAD (multi-group refactor + fk_layout CLI fix) |

The reference "regression" run `harri-no-fingers-refactored` (133 columns) was confirmed to
be identical to `harri-no-palms-head` (same TOML, same binary).

---

## Observations

### State vector shape

| Run | State vector columns |
|-----|---------------------|
| `baseline` | 241 |
| `post-pinocchio` | 125 |
| `head` | 133 |

The column count change between `baseline` and `post-pinocchio` coincides exactly with the
introduction of `f7cecd1`.  The 8-column difference between `post-pinocchio` and `head`
corresponds to the 4 palm joints (`palm.01.L`, `palm.04.L`, `palm.01.R`, `palm.04.R`) that
were added to the `main` group by the multi-group refactor in `7c8ff69`.

### Inlier counts

| Run | Avg inliers (all frames) | Inliers at frame 700 | Zero-inlier frames |
|-----|--------------------------|----------------------|--------------------|
| `baseline` | 301.31 | ~175 | 0 |
| `post-pinocchio` | 301.31† | ~10 | 0 |
| `head` | 301.31† | 0 | ≥ 250 |

†The overall average is identical because the stat is dominated by the majority of frames where
tracking is nominally working; the per-frame breakdown reveals the divergence.

`baseline` holds ~175 inliers at frame 700.  `post-pinocchio` retains ~10.  `head` reaches
zero inliers by frame 700 and stays there for the remainder of the run.

### Predicted observations

At frame 700:

- `baseline` predicts positions for ~180 distinct marker names.
- `head` predicts positions for ~108 distinct marker names.

`baseline` marks 175 out of 236 raw observations as `used_in_tracking = true`.
`head` marks all 236 observations as `used_in_tracking = false`.

### Residuals at initialization (frame 2)

In `head`, the predicted-observation residuals are already >1 500 px at frame 2 (the first
tracking frame after initialization).  In `baseline` they start within a few tens of pixels.
This indicates the divergence is present from the very first frame, not an accumulation over
700 frames.

### Observation coverage

All three runs receive exactly 236 raw 2D observations per frame (61 unique markers projected
across ~4 cameras).  The difference in tracking quality is therefore not caused by a change in
the input data.

### IK and `config_to_state` at init time

`InverseKinematics::config_to_state` iterates `skeleton.joints()` (all joints in the full
skeleton) to map the pinocchio configuration vector `q` back into a `State`.  When the
pinocchio model contains only a joint subset, `q` is shorter than the full-skeleton nq, so
joints beyond the end of `q` silently read zero from the guard `if (q_idx < model_.nq)`.
The returned `State::joint_angles()` is sized by the count of all non-root, non-fixed skeleton
joints.

`load_python_state` in `cli/track.cpp` also sizes `joint_angles` using
`skeleton.total_dof_count()` (full skeleton) and iterates `skeleton.joints()`.

The UKF is constructed with a layout sized to the active joint groups only.  Its
`error_dim()` and sigma-point machinery are consistent with that compact layout, but
`ukf_->set_state()` receives a `State` whose `joint_angles().size()` equals the full-skeleton
DOF count rather than the layout DOF count.

---

## Hypothesis

The frame-2 residual explosion and all downstream inlier degradation are explained by a mismatch
between the `State::joint_angles()` size and the UKF layout's expected size at the point where
`ukf_->set_state(initial_state)` is called in `initialize_ukf`.

Specifically:

1. **`load_python_state` (CLI)** and **`InverseKinematics::config_to_state`** both return a
   `State` with `joint_angles` sized to the **full skeleton** (all non-root, non-fixed joints).
2. When `active_joint_groups` is non-empty, `initialize_ukf` constructs a UKF layout with
   fewer DOFs (only the joints in those groups).
3. `ukf_->set_state()` stores the oversized `State` verbatim.  Because `State::to_error_vector`
   and `State::apply_error_update` use `num_dof()` = `joint_angles_.size()`, the UKF's
   sigma-point and covariance operations operate on a vector that is much larger than the layout
   expects, placing joint angles at wrong indices from the perspective of `state_to_config`.
4. The result is that FK evaluations during the first predict/update step receive garbage joint
   angles → predicted marker positions are far from observed positions → all observations are
   rejected as outliers.

The fix is to slice `initial_state` down to the layout's DOF count before calling
`set_state`, analogously to what `slice_state_for_child` already does for child filters.
This can be done inside `initialize_ukf` whenever
`initial_state.joint_angles().size() != layout->total_storage_dof_count()`.

/**
 * @file hierarchical_solver.hpp
 * @brief CLI/config plumbing for the hierarchical solver's child stages --
 * PR 6 of docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md.
 *
 * A "stage" is a hierarchical solver child group (e.g. "HandL"): a filter
 * that tracks a named subset of joints/markers with its root held fixed at
 * another group's already-solved joint (the "freeflyer_joint"), per-frame,
 * sourced from that parent group's *smoothed* trajectory. See
 * SkeletonGroup (skeleton.hpp) for where freeflyer_joint/ref_marker live,
 * and Tracker::initialize_with_fixed_root()/set_external_root_transform()
 * for the underlying fixed-root machinery (PR 2/3).
 *
 * Existence-based hierarchical-mode toggle: a tracker_config_id with any
 * tracker_config_stages rows (SessionReader::load_tracker_config_stages())
 * runs hierarchically; one without runs monolithic, unchanged -- see the
 * design doc's "gap 2" resolution.
 */
#pragma once

#include "posetrak/core/skeleton.hpp"
#include "posetrak/db/session_reader.hpp"
#include "posetrak/tracking/multi_person_tracker.hpp"
#include <string>
#include <vector>

namespace posetrak {

/// @brief Build a child stage's effective TrackerConfig.
///
/// Starts from a copy of the parent's own TrackerConfig (so every field the
/// stage doesn't explicitly override -- including ones with no
/// tracker_config_stages column, e.g. ukf_alpha -- is inherited), applies
/// each non-nullopt field in @p overrides, then sets active_joint_groups to
/// {group.name} and fixed_root_joint_name to group.freeflyer_joint.
///
/// @param parent_config The parent stage's own (already-resolved) TrackerConfig.
/// @param overrides This stage's tracker_config_stages row.
/// @param group This stage's SkeletonGroup metadata (must be group.name ==
///        overrides.group_name; not checked here).
/// @return The child's effective TrackerConfig.
TrackerConfig build_stage_tracker_config(TrackerConfig const& parent_config,
                                         StageConfigOverrides const& overrides,
                                         SkeletonGroup const& group);

/// @brief Expand a compact-layout State to full-skeleton width for DB storage.
///
/// The parent stage's own Tracker is scoped to its own group (e.g. "main"),
/// so its State only has DOFs for that group's joints -- but tracking_results
/// rows must stay full-skeleton-width so a child stage's own merge
/// (SkeletonLayout::build_index_map_from(child_layout), which requires the
/// receiving layout to be a superset of the child's joints) can reach every
/// stage's DOFs, not just the ones the parent's own group covers. Every DOF
/// @p full_layout has that @p compact_layout doesn't is filled with rest-pose
/// defaults (0 angle, 0 velocity -- see State::State(int); this is the same
/// convention Tracker::initialize_from_rest_pose() uses). Root pose/velocity
/// are copied through unchanged (both layouts share the same root).
///
/// @param compact State produced by a Tracker scoped to @p compact_layout.
/// @param compact_layout The Tracker's own (possibly group-scoped) layout.
/// @param full_layout Full-skeleton layout; must be a superset of compact_layout
///        (true for any group vs. SkeletonLayout::from_full_skeleton() of the
///        same skeleton).
/// @return A State sized for full_layout.total_storage_dof_count().
State expand_state_to_full_layout(State const& compact, SkeletonLayout const& compact_layout,
                                  SkeletonLayout const& full_layout);

/// @brief Expand a compact-layout covariance diagonal to full-skeleton
/// error-state width for DB storage, the cov_diag analogue of
/// expand_state_to_full_layout().
///
/// Unlike state (which is storage-indexed and expanded via
/// SkeletonLayout::build_index_map_from()), a covariance diagonal is
/// error-state-indexed -- narrower than storage whenever a joint has a
/// locked axis (see SkeletonLayout::build_error_index_map_from()'s doc
/// comment) -- so it needs the error_index-based sibling mapping, not
/// build_index_map_from()'s map. Every full-layout error-state slot that
/// @p compact_layout doesn't own (i.e. every not-yet-solved child stage's
/// DOFs) is filled with a placeholder variance derived from the tracker
/// config's own init_joint_std/init_velocity_std, matching the convention
/// expand_state_to_full_layout() uses for state (rest-pose defaults).
/// Both layouts must have a floating root (this expands a *parent* stage's
/// own cov_diag; only the skeleton's true root owner calls this).
///
/// @param compact_diag Covariance diagonal produced by a Tracker scoped to
///        @p compact_layout (i.e. compact_layout.error_state_dim() wide).
/// @param compact_layout The Tracker's own (possibly group-scoped) layout.
/// @param full_layout Full-skeleton layout; must be a superset of compact_layout.
/// @param placeholder_pos_variance Variance placeholder for unsolved position DOFs
///        (typically init_joint_std² from the run's TrackerConfig).
/// @param placeholder_vel_variance Variance placeholder for unsolved velocity DOFs
///        (typically init_velocity_std² from the run's TrackerConfig).
/// @return A vector sized for full_layout.error_state_dim().
/// @throws std::invalid_argument if either layout lacks a floating root.
Eigen::VectorXd expand_cov_diag_to_full_layout(Eigen::VectorXd const& compact_diag,
                                               SkeletonLayout const& compact_layout,
                                               SkeletonLayout const& full_layout,
                                               double placeholder_pos_variance,
                                               double placeholder_vel_variance);

/// @brief Run every hierarchical-solver child stage for one person, after
/// their parent (main) forward pass + RTS smoothing has completed.
///
/// No-op if @p stage_overrides is empty (the existence-based hierarchical-
/// mode toggle -- see load_tracker_config_stages()). Each stage:
///  1. Builds its own fixed-root Tracker (active_joint_groups={group.name},
///     fixed_root_joint_name=group.freeflyer_joint) from the effective
///     TrackerConfig build_stage_tracker_config() computes.
///  2. Streams the parent's smoothed freeflyer_joint trajectory
///     (BatchTrajectoryStream) and, per frame, builds this stage's own
///     observations: the reference marker's own POSITION detections plus
///     build_ref_marker_pair_observations()'s PAIR_DIFF pairs for every
///     other stage marker.
///  3. Runs a full forward pass (initialize_with_fixed_root() on the first
///     frame, set_external_root_transform()+track_frame() thereafter) and
///     its own RTS smoothing pass.
///  4. Merges every frame's state AND cov_diag into the SAME tracking_results
///     rows the parent's own ResultWriter already wrote (both is_smoothed
///     families), via a new attach-mode ResultWriter. State uses
///     SkeletonLayout::build_index_map_from() (storage_index-based) to
///     translate the child's compact joint_angles/joint_velocities indices
///     into the parent layout's index space; cov_diag uses the separate
///     build_error_index_map_from() (error_index-based -- diverges from the
///     state map whenever a joint has a locked axis, see that method's doc
///     comment), replacing the placeholder variance
///     expand_cov_diag_to_full_layout() wrote at the parent's own write time
///     with this stage's own real per-DOF confidence. Per-observation
///     results are merged into obs_blob via
///     reconstruct_pair_diff_absolute() + ResultWriter::patch_obs_results(),
///     with parent_owned_markers set to the markers shared between the
///     parent's active group(s) and this stage (parent-wins).
///  5. Tracks progress in tracking_run_stages via ResultWriter::set_stage_status().
///
/// @param parent_ctx The person's already-finalized (tracked + smoothed) context.
///        parent_ctx.result_writer must still be open (attach-mode ResultWriter
///        instances are constructed against the same db_path/run_id).
/// @param stage_overrides This person's tracker_config_id's tracker_config_stages rows.
/// @param parent_smoothed The parent's RTS-smoothed trajectory, in the same
///        (already tracker_step-aligned) order Tracker::smooth() returns --
///        parent_smoothed[i] corresponds to tracker_step (i+1).
/// @param db_path Path to the session database (for attach-mode ResultWriter).
/// @throws std::runtime_error if a stage's group_name has no SkeletonGroup
///         metadata, or is missing freeflyer_joint/ref_marker, or if
///         parent_smoothed is empty while stage_overrides is non-empty.
void run_hierarchical_child_stages(PersonContext& parent_ctx,
                                   std::vector<StageConfigOverrides> const& stage_overrides,
                                   std::vector<SmoothedFrame> const& parent_smoothed,
                                   std::string const& db_path, bool verbose, bool quiet);

}  // namespace posetrak

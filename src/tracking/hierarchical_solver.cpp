#include "posetrak/tracking/hierarchical_solver.hpp"

#include <fmt/core.h>

#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/tracking/relative_observations.hpp"
#include "posetrak/tracking/trajectory_stream.hpp"
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace posetrak {

TrackerConfig build_stage_tracker_config(TrackerConfig const& parent_config,
                                         StageConfigOverrides const& overrides,
                                         SkeletonGroup const& group) {
    TrackerConfig child = parent_config;

    if (overrides.process_noise_std)
        child.process_noise_std = *overrides.process_noise_std;
    if (overrides.process_noise_vel_std)
        child.process_noise_vel_std = *overrides.process_noise_vel_std;
    if (overrides.velocity_half_life_s)
        child.velocity_half_life_s = *overrides.velocity_half_life_s;
    if (overrides.pose_noise_std)
        child.pose_noise_std = *overrides.pose_noise_std;
    if (overrides.calib_noise_std)
        child.calib_noise_std = *overrides.calib_noise_std;
    if (overrides.outlier_threshold)
        child.outlier_threshold = *overrides.outlier_threshold;
    if (overrides.init_joint_std)
        child.init_joint_std = *overrides.init_joint_std;
    if (overrides.init_velocity_std)
        child.init_velocity_std = *overrides.init_velocity_std;
    // min_inliers_ratio / max_innovation_norm: schema has them (tracker_config_stages),
    // but no TrackerConfig field exists yet to receive them -- not applied.

    child.active_joint_groups = {group.name};
    child.fixed_root_joint_name = group.freeflyer_joint;

    return child;
}

State expand_state_to_full_layout(State const& compact, SkeletonLayout const& compact_layout,
                                  SkeletonLayout const& full_layout) {
    State full(full_layout.total_storage_dof_count());
    full.set_root_position(compact.root_position());
    full.set_root_orientation(compact.root_orientation());
    full.set_root_velocity(compact.root_velocity());
    full.set_root_angular_velocity(compact.root_angular_velocity());

    Eigen::VectorXd angles = full.joint_angles();
    Eigen::VectorXd vels = full.joint_velocities();

    auto merge_map = full_layout.build_index_map_from(compact_layout);
    Eigen::VectorXd const& compact_angles = compact.joint_angles();
    Eigen::VectorXd const& compact_vels = compact.joint_velocities();
    for (size_t i = 0; i < merge_map.size(); ++i) {
        angles[merge_map[i]] = compact_angles[static_cast<Eigen::Index>(i)];
        vels[merge_map[i]] = compact_vels[static_cast<Eigen::Index>(i)];
    }
    full.set_joint_angles(angles);
    full.set_joint_velocities(vels);
    return full;
}

namespace {

/// @brief Markers declared in both the parent's active group(s) and this
/// stage's own group -- the "parent-wins" set for patch_obs_results().
std::vector<std::string> compute_parent_owned_markers(Skeleton const& skeleton,
                                                      std::vector<std::string> const& parent_groups,
                                                      SkeletonGroup const& stage_group) {
    std::unordered_set<std::string> parent_markers;
    for (auto const& parent_group_name : parent_groups) {
        SkeletonGroup const* parent_group = skeleton.get_group(parent_group_name);
        if (parent_group == nullptr)
            continue;
        for (auto const& m : parent_group->markers)
            parent_markers.insert(m);
    }

    std::vector<std::string> shared;
    for (auto const& m : stage_group.markers) {
        if (parent_markers.count(m) > 0)
            shared.push_back(m);
    }
    return shared;
}

/// @brief Run one stage's forward pass + smoothing + DB merge.
void run_one_stage(PersonContext& parent_ctx, StageConfigOverrides const& overrides,
                   std::vector<SmoothedFrame> const& parent_smoothed, std::string const& db_path,
                   bool verbose, bool quiet) {
    Skeleton const& skeleton = parent_ctx.skeleton;
    SkeletonGroup const* group = skeleton.get_group(overrides.group_name);
    if (group == nullptr || group->freeflyer_joint.empty() || group->ref_marker.empty()) {
        throw std::runtime_error(fmt::format(
            "run_hierarchical_child_stages: group '{}' has no freeflyer_joint/ref_marker "
            "metadata (tracker_config_stages references it, but the skeleton's groups: "
            "section doesn't declare it as a child stage)",
            overrides.group_name));
    }

    if (!quiet) {
        fmt::print("\nHierarchical stage '{}': freeflyer={} ref_marker={}\n", group->name,
                   group->freeflyer_joint, group->ref_marker);
    }

    if (!parent_ctx.full_layout) {
        throw std::runtime_error(
            "run_hierarchical_child_stages: parent_ctx.full_layout is null -- "
            "build_person_context() should have built it whenever this person's tracker_config "
            "has hierarchical-solver stages");
    }
    SkeletonLayout const& parent_layout = *parent_ctx.full_layout;

    ResultWriter stage_writer(db_path, parent_ctx.result_writer->run_id(),
                              parent_ctx.spec.person_id);
    stage_writer.set_stage_status(group->name, "running", /*set_started=*/true);

    // ---- Marker index lookups (marker_id == index into skeleton.markers(),
    //      same indexing the parent's own observations already use). ----
    std::unordered_map<std::string, int> marker_name_to_id;
    for (int i = 0; i < static_cast<int>(skeleton.markers().size()); ++i)
        marker_name_to_id[skeleton.markers()[i].name] = i;

    auto ref_it = marker_name_to_id.find(group->ref_marker);
    if (ref_it == marker_name_to_id.end()) {
        throw std::runtime_error(fmt::format(
            "run_hierarchical_child_stages: group '{}' ref_marker '{}' is not a marker in "
            "this skeleton",
            group->name, group->ref_marker));
    }
    int const ref_marker_id = ref_it->second;

    std::unordered_set<int> stage_marker_ids;
    for (auto const& name : group->markers) {
        auto it = marker_name_to_id.find(name);
        if (it != marker_name_to_id.end())
            stage_marker_ids.insert(it->second);
    }

    // ---- Build the child Tracker. ----
    auto skeleton_ptr = std::make_shared<const Skeleton>(skeleton);
    TrackerConfig child_config =
        build_stage_tracker_config(parent_ctx.tracker_config, overrides, *group);
    auto child_layout = SkeletonLayout::from_groups(skeleton_ptr, {group->name});

    Tracker child_tracker(skeleton_ptr, parent_ctx.cameras_by_id, child_config);
    child_tracker.enable_smoothing(true);

    BatchTrajectoryStream traj_stream(parent_smoothed, *parent_ctx.fk, group->freeflyer_joint);

    // ---- Forward pass: one iteration per parent_smoothed frame, matching
    //      tracker_step = i + 1 exactly (see hierarchical_solver.hpp doc comment). ----
    std::vector<std::string> const parent_owned_markers = compute_parent_owned_markers(
        skeleton, parent_ctx.tracker_config.active_joint_groups, *group);

    // child_tracker's RTS smoother cache only accumulates successful track_frame()
    // calls -- i==0 below uses initialize_with_fixed_root() instead (no track_frame
    // call at all, see its comment), and any tracking_lost step is `continue`d before
    // reaching track_frame's smoother push. So child_smoothed[k] (after the loop)
    // corresponds to child_tracked_steps[k], NOT tracker_step (k+1) -- tracking a
    // step's own the tracker_step whenever a track_frame() call actually succeeds is
    // the only way to keep the smoothing-pass merge below aligned when one or more
    // steps are skipped.
    std::vector<int> child_tracked_steps;
    child_tracked_steps.reserve(parent_smoothed.size());

    for (size_t i = 0; i < parent_smoothed.size(); ++i) {
        auto pose_opt = traj_stream.next();
        if (!pose_opt.has_value()) {
            throw std::runtime_error(fmt::format(
                "run_hierarchical_child_stages: group '{}' trajectory stream exhausted early "
                "at frame {} of {}",
                group->name, i, parent_smoothed.size()));
        }

        int const tracker_step = static_cast<int>(i) + 1;
        double const t_start =
            parent_ctx.start_time + tracker_step * parent_ctx.dt - parent_ctx.dt / 2.0;
        double const t_end = t_start + parent_ctx.dt;
        double const t_effective = t_start + parent_ctx.dt / 2.0;

        auto all_raw_obs = parent_ctx.observations.get_all_in_range(t_start, t_end);
        std::vector<Observation> raw_for_stage;
        for (auto const& obs : all_raw_obs) {
            if (obs.marker_id == ref_marker_id || stage_marker_ids.count(obs.marker_id) > 0)
                raw_for_stage.push_back(obs);
        }

        auto stage_obs = build_ref_marker_pair_observations(
            raw_for_stage, ref_marker_id, child_config.pose_noise_std,
            parent_ctx.tracker_config.relative_min_confidence);
        for (auto const& obs : raw_for_stage) {
            if (obs.marker_id == ref_marker_id)
                stage_obs.push_back(obs);
        }

        State const* merged_state = nullptr;

        if (i == 0) {
            child_tracker.initialize_with_fixed_root(stage_obs, pose_opt->position,
                                                     pose_opt->orientation, t_effective);
            // initialize_with_fixed_root() doesn't produce a TrackingResult (no update
            // ran -- there is no prior to predict from at step 0); read the
            // freshly-initialized state directly instead of calling track_frame() again
            // for the same frame. No observations were run through an update either, so
            // there's nothing to merge into obs_blob for this step.
            merged_state = &child_tracker.state();
        } else {
            child_tracker.set_external_root_transform(pose_opt->position, pose_opt->orientation);
            auto result = child_tracker.track_frame(stage_obs, t_effective);
            if (result.tracking_lost) {
                if (!quiet) {
                    fmt::print(stderr, "  Hierarchical stage '{}': tracking lost at step {}\n",
                               group->name, tracker_step);
                }
                continue;  // leave this step's slots as the parent left them (placeholder/NaN)
            }
            merged_state = &child_tracker.state();
            child_tracked_steps.push_back(tracker_step);

            // ---- Merge obs_blob (forward pass only -- tracking_obs_results has no
            //      is_smoothed dimension). ----
            if (!result.update_info.observations.empty()) {
                auto [absolute_results, reconstructed] = reconstruct_pair_diff_absolute(
                    result.update_info.observations, group->ref_marker);
                stage_writer.patch_obs_results(tracker_step, absolute_results, reconstructed,
                                               parent_owned_markers);
            }
        }

        // ---- Merge state into the parent's tracking_results row (is_smoothed=0). ----
        auto merge_map = parent_layout.build_index_map_from(*child_layout);
        int const n_dof_full = parent_layout.total_storage_dof_count();
        Eigen::VectorXd const& child_angles = merged_state->joint_angles();
        Eigen::VectorXd const& child_vels = merged_state->joint_velocities();

        std::vector<int> state_indices;
        std::vector<double> state_values;
        state_indices.reserve(2 * merge_map.size());
        state_values.reserve(2 * merge_map.size());
        for (size_t j = 0; j < merge_map.size(); ++j) {
            int const full_idx = merge_map[j];
            state_indices.push_back(6 + full_idx);
            state_values.push_back(child_angles[static_cast<Eigen::Index>(j)]);
            state_indices.push_back(12 + n_dof_full + full_idx);
            state_values.push_back(child_vels[static_cast<Eigen::Index>(j)]);
        }
        stage_writer.patch_frame(tracker_step, /*is_smoothed=*/false, state_indices, state_values);
    }

    // ---- Smoothing pass: merge into is_smoothed=1 rows. ----
    auto child_smoothed = child_tracker.smooth();
    if (child_smoothed.size() != child_tracked_steps.size()) {
        throw std::runtime_error(fmt::format(
            "run_hierarchical_child_stages: group '{}' smoother returned {} frames but {} "
            "track_frame() calls succeeded -- child_tracked_steps bookkeeping is out of sync",
            group->name, child_smoothed.size(), child_tracked_steps.size()));
    }
    auto merge_map = parent_layout.build_index_map_from(*child_layout);
    int const n_dof_full = parent_layout.total_storage_dof_count();
    for (size_t i = 0; i < child_smoothed.size(); ++i) {
        int const tracker_step = child_tracked_steps[i];
        Eigen::VectorXd const& child_angles = child_smoothed[i].state.joint_angles();
        Eigen::VectorXd const& child_vels = child_smoothed[i].state.joint_velocities();

        std::vector<int> state_indices;
        std::vector<double> state_values;
        state_indices.reserve(2 * merge_map.size());
        state_values.reserve(2 * merge_map.size());
        for (size_t j = 0; j < merge_map.size(); ++j) {
            int const full_idx = merge_map[j];
            state_indices.push_back(6 + full_idx);
            state_values.push_back(child_angles[static_cast<Eigen::Index>(j)]);
            state_indices.push_back(12 + n_dof_full + full_idx);
            state_values.push_back(child_vels[static_cast<Eigen::Index>(j)]);
        }
        stage_writer.patch_frame(tracker_step, /*is_smoothed=*/true, state_indices, state_values);
    }

    stage_writer.set_stage_status(group->name, "complete", /*set_started=*/false,
                                  /*set_completed=*/true);
    stage_writer.flush();

    if (!quiet) {
        fmt::print("  Hierarchical stage '{}' complete ({} frames)\n", group->name,
                   parent_smoothed.size());
    }
    (void)verbose;
}

}  // namespace

void run_hierarchical_child_stages(PersonContext& parent_ctx,
                                   std::vector<StageConfigOverrides> const& stage_overrides,
                                   std::vector<SmoothedFrame> const& parent_smoothed,
                                   std::string const& db_path, bool verbose, bool quiet) {
    if (stage_overrides.empty())
        return;

    if (parent_smoothed.empty()) {
        throw std::runtime_error(
            "run_hierarchical_child_stages: parent has no smoothed frames -- hierarchical "
            "mode requires --smooth");
    }

    for (auto const& stage : stage_overrides) {
        run_one_stage(parent_ctx, stage, parent_smoothed, db_path, verbose, quiet);
    }
}

}  // namespace posetrak

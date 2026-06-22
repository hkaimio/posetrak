#include "posetrak/filters/subset_ukf.hpp"

#include <fmt/format.h>

#include "posetrak/core/camera.hpp"
#include "posetrak/core/observation.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include <stdexcept>

namespace posetrak {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

SubsetUKF::SubsetUKF(Skeleton const& skeleton, std::vector<std::string> const& joint_groups,
                     std::vector<std::string> const& obs_groups, double process_noise_std,
                     double alpha, double beta, double kappa)
    : skeleton_(skeleton),
      joint_groups_(joint_groups),
      obs_groups_(obs_groups),
      obs_group_set_(obs_groups.begin(), obs_groups.end()),
      background_state_(skeleton.total_dof_count()) {
    // Activate the requested joint groups on the skeleton copy.
    skeleton_.set_active_groups(joint_groups);

    // active_layout_: full-skeleton state_index offsets, child DOFs only in error state.
    active_layout_ =
        SkeletonLayout::from_active_skeleton(std::make_shared<const Skeleton>(skeleton_));

    // compact_layout_: 0-based state_index for SkeletonState output.
    // from_groups() needs a skeleton without an active filter so it can see all joints.
    Skeleton skel_unfiltered = skeleton;
    compact_layout_ = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel_unfiltered),
                                                  joint_groups);

    // Build merge_map_: compact DOF i → full-skeleton state_index.
    merge_map_.reserve(static_cast<size_t>(compact_layout_->total_storage_dof_count()));
    for (auto const& jdesc : compact_layout_->joints()) {
        JointDesc const* active_jdesc = active_layout_->get_joint(jdesc.name);
        if (!active_jdesc) {
            throw std::invalid_argument(
                fmt::format("SubsetUKF: joint '{}' in compact_layout but not in active_layout — "
                            "check that joint_groups are consistent",
                            jdesc.name));
        }
        for (int d = 0; d < static_cast<int>(jdesc.storage_dof_count); ++d) {
            merge_map_.push_back(active_jdesc->state_index + d);
        }
    }

    // Construct the inner UKF now that layouts are ready.
    // skeleton_ (member) outlives ukf_ so the reference is safe.
    ukf_ = std::make_unique<UnscentedKalmanFilter>(active_layout_, process_noise_std, alpha, beta,
                                                   kappa);
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

void SubsetUKF::initialize(State const& initial_state, Eigen::MatrixXd const& initial_cov) {
    ukf_->set_state(initial_state);
    ukf_->set_covariance(initial_cov);
    background_state_ = initial_state;
}

// ---------------------------------------------------------------------------
// set_parent_state
// ---------------------------------------------------------------------------

void SubsetUKF::set_parent_state(State const& full_parent_state) {
    background_state_ = full_parent_state;
}

// ---------------------------------------------------------------------------
// sync_from_background (private)
// ---------------------------------------------------------------------------

void SubsetUKF::sync_from_background() {
    // Start from background (parent's full-skeleton estimate).
    State merged = background_state_;

    // Overwrite child DOFs with the most recent child estimates from the UKF.
    State const& child = ukf_->state();
    Eigen::VectorXd angles = merged.joint_angles();
    Eigen::VectorXd vels = merged.joint_velocities();

    for (auto const& jdesc : active_layout_->joints()) {
        int const si = jdesc.state_index;
        for (int d = 0; d < jdesc.storage_dof_count; ++d) {
            angles(si + d) = child.joint_angles()(si + d);
            vels(si + d) = child.joint_velocities()(si + d);
        }
    }

    merged.set_joint_angles(angles);
    merged.set_joint_velocities(vels);

    // Root: carry the child's root only if it tracks a floating root;
    // otherwise use the background root (parent's world-space pose).
    if (active_layout_->has_floating_root()) {
        merged.set_root_position(child.root_position());
        merged.set_root_orientation(child.root_orientation());
        merged.set_root_velocity(child.root_velocity());
        merged.set_root_angular_velocity(child.root_angular_velocity());
    }

    ukf_->set_state(merged);
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

void SubsetUKF::predict(double dt) {
    sync_from_background();
    ukf_->predict(dt);
}

// ---------------------------------------------------------------------------
// update
// ---------------------------------------------------------------------------

UpdateResult SubsetUKF::update(std::vector<Observation> const& observations,
                               std::unordered_map<int, Camera> const& cameras,
                               ForwardKinematics& fk, double pose_noise_std, double calib_noise_std,
                               double outlier_threshold) {
    sync_from_background();

    auto filtered = filter_observations(observations);
    if (filtered.empty()) {
        return UpdateResult{};
    }

    return ukf_->update(filtered, cameras, fk, pose_noise_std, calib_noise_std, outlier_threshold);
}

// ---------------------------------------------------------------------------
// skeleton_state
// ---------------------------------------------------------------------------

SkeletonState SubsetUKF::skeleton_state() const {
    State const& full = ukf_->state();
    int const n = compact_layout_->total_storage_dof_count();

    Eigen::VectorXd angles(n), vels(n);
    for (int i = 0; i < n; ++i) {
        angles(i) = full.joint_angles()(merge_map_[i]);
        vels(i) = full.joint_velocities()(merge_map_[i]);
    }

    State compact(n);
    compact.set_joint_angles(angles);
    compact.set_joint_velocities(vels);

    if (compact_layout_->has_floating_root()) {
        compact.set_root_position(full.root_position());
        compact.set_root_orientation(full.root_orientation());
        compact.set_root_velocity(full.root_velocity());
        compact.set_root_angular_velocity(full.root_angular_velocity());
    }

    return SkeletonState::create(compact_layout_, std::move(compact));
}

// ---------------------------------------------------------------------------
// filter_observations
// ---------------------------------------------------------------------------

std::vector<Observation>
SubsetUKF::filter_observations(std::vector<Observation> const& observations) const {
    std::vector<Observation> result;
    result.reserve(observations.size());

    auto const& markers = skeleton_.markers();
    for (auto const& obs : observations) {
        if (obs.marker_id < 0 || static_cast<size_t>(obs.marker_id) >= markers.size()) {
            continue;
        }
        if (obs_group_set_.count(markers[static_cast<size_t>(obs.marker_id)].group) > 0) {
            result.push_back(obs);
        }
    }
    return result;
}

}  // namespace posetrak

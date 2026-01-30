/**
 * @file tracker.cpp
 * @brief Implementation of main tracking orchestration
 */

#include "posetrak/tracking/tracker.hpp"

#include <fmt/core.h>

#include "posetrak/kinematics/pinocchio_model_builder.hpp"

namespace posetrak {

Tracker::Tracker(Skeleton const& skeleton, std::unordered_map<int, Camera> const& cameras,
                 TrackerConfig const& config)
    : skeleton_(skeleton), cameras_(cameras), config_(config) {
    // Build Pinocchio model for FK/IK
    model_ = std::make_unique<pinocchio::Model>();
    data_ = std::make_unique<pinocchio::Data>();
    PinocchioModelBuilder::build_model_and_data(skeleton_, *model_, *data_);
    marker_frame_map_ = PinocchioModelBuilder::build_marker_frame_map(*model_, skeleton_);

    // Create FK computer
    fk_ = std::make_unique<ForwardKinematics>(*model_, *data_, marker_frame_map_, skeleton_);

    // Create triangulator
    triangulator_ = std::make_unique<Triangulator>(Triangulator::Method::DLT);

    // Create IK solver
    ik_solver_ = std::make_unique<InverseKinematics>(*model_, *data_, *fk_, marker_frame_map_);
}

bool Tracker::initialize(std::vector<Observation> const& observations, double timestamp) {
    if (observations.empty()) {
        return false;
    }

    // Step 1: Triangulate marker positions
    // Group observations by marker
    std::map<int, std::vector<Observation>> obs_by_marker;
    for (auto const& obs : observations) {
        obs_by_marker[obs.marker_id].push_back(obs);
    }

    // Triangulate each marker
    std::map<std::string, Eigen::Vector3d> marker_positions;
    for (auto const& [marker_id, marker_obs] : obs_by_marker) {
        if (marker_obs.size() < static_cast<size_t>(config_.min_cameras_for_init)) {
            continue;  // Need at least N cameras
        }

        // Get marker name
        if (marker_id >= static_cast<int>(skeleton_.markers().size())) {
            continue;
        }
        std::string marker_name = skeleton_.markers()[marker_id].name;

        // Prepare for triangulation
        std::vector<Eigen::Vector2d> pixel_coords;
        std::vector<Camera const*> marker_cameras;
        std::vector<double> confidences;

        for (auto const& obs : marker_obs) {
            auto it = cameras_.find(obs.camera_id);
            if (it == cameras_.end()) {
                continue;
            }
            pixel_coords.push_back(obs.position);
            marker_cameras.push_back(&it->second);
            confidences.push_back(obs.confidence);
        }

        // Triangulate
        auto result = triangulator_->triangulate(pixel_coords, marker_cameras, confidences);
        if (result.success) {
            marker_positions[marker_name] = result.position;
        }
    }

    // Check if we have enough markers
    if (marker_positions.size() < 3) {
        return false;  // Need at least 3 markers for reasonable initialization
    }

    // Step 2: Solve IK to get initial joint configuration
    auto ik_result = ik_solver_->solve(marker_positions, skeleton_, std::nullopt,
                                       config_.ik_max_iterations, config_.ik_tolerance);

    if (!ik_result.converged) {
        return false;  // IK failed
    }

    // Step 3: Initialize UKF
    initialize_ukf(ik_result.state, timestamp);

    initialized_ = true;
    last_timestamp_ = timestamp;

    return true;
}

void Tracker::initialize_ukf(State const& initial_state, double timestamp) {
    // Create UKF
    ukf_ = std::make_unique<UnscentedKalmanFilter>(skeleton_, config_.process_noise_std);

    // Set initial state
    ukf_->set_state(initial_state);

    // Set initial covariance
    int const error_dim = initial_state.error_state_dim();
    int const pos_dim = error_dim / 2;

    Eigen::MatrixXd initial_cov = Eigen::MatrixXd::Zero(error_dim, error_dim);

    // Position uncertainties
    initial_cov.block(0, 0, 3, 3) =
        Eigen::Matrix3d::Identity() * (config_.init_position_std * config_.init_position_std);
    initial_cov.block(3, 3, 3, 3) =
        Eigen::Matrix3d::Identity() * (config_.init_orientation_std * config_.init_orientation_std);

    // Joint angle uncertainties
    int joint_dof = pos_dim - 6;
    if (joint_dof > 0) {
        initial_cov.block(6, 6, joint_dof, joint_dof) =
            Eigen::MatrixXd::Identity(joint_dof, joint_dof) *
            (config_.init_joint_std * config_.init_joint_std);
    }

    // Velocity uncertainties (all velocities)
    initial_cov.block(pos_dim, pos_dim, pos_dim, pos_dim) =
        Eigen::MatrixXd::Identity(pos_dim, pos_dim) *
        (config_.init_velocity_std * config_.init_velocity_std);

    ukf_->set_covariance(initial_cov);

    last_timestamp_ = timestamp;
}

TrackingResult Tracker::track_frame(std::vector<Observation> const& observations,
                                    double timestamp) {
    if (!initialized_) {
        throw std::runtime_error("Tracker::track_frame() called before initialization");
    }

    // Compute dt
    double dt = timestamp - last_timestamp_;
    if (dt < 0.0) {
        // Return failure result
        TrackingResult result(timestamp, ukf_->state(), ukf_->covariance(), {}, true,
                              "Negative dt: timestamps out of order");
        return result;
    }

    // Step 1: Predict
    ukf_->predict(dt);

    // Step 2: Check if we have observations
    if (!has_sufficient_observations(observations)) {
        TrackingResult result(timestamp, ukf_->state(), ukf_->covariance(), {}, true,
                              "Insufficient observations");
        return result;
    }

    // Step 3: Update
    auto update_info = ukf_->update(observations, cameras_, *fk_, config_.measurement_noise_std,
                                    config_.outlier_threshold);

    // Step 4: Create result
    TrackingResult result(timestamp, ukf_->state(), ukf_->covariance(), update_info, false, "");
    result.num_observations_used = update_info.num_inliers;

    // Update last timestamp
    last_timestamp_ = timestamp;

    // Call frame callback if set
    if (frame_callback_) {
        frame_callback_(result);
    }

    return result;
}

bool Tracker::has_sufficient_observations(std::vector<Observation> const& observations) const {
    // For now, just check if we have any observations
    // Could add more sophisticated checks (e.g., need observations of specific markers)
    return !observations.empty();
}

void Tracker::reset() {
    initialized_ = false;
    last_timestamp_ = 0.0;
    ukf_.reset();
}

}  // namespace posetrak

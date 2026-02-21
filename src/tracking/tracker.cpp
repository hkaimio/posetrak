/**
 * @file tracker.cpp
 * @brief Implementation of main tracking orchestration
 */

#include "posetrak/tracking/tracker.hpp"

#include <fmt/core.h>

#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/kinematics/pinocchio_model_builder.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace posetrak {

Tracker::Tracker(std::shared_ptr<const Skeleton> skeleton,
                 std::unordered_map<int, Camera> const& cameras, TrackerConfig const& config)
    : skeleton_(std::move(skeleton)), cameras_(cameras), config_(config) {
    // Build Pinocchio model for FK/IK
    model_ = std::make_unique<pinocchio::Model>();
    data_ = std::make_unique<pinocchio::Data>();
    PinocchioModelBuilder::build_model_and_data(*skeleton_, *model_, *data_);
    marker_frame_map_ = PinocchioModelBuilder::build_marker_frame_map(*model_, *skeleton_);

    // Create FK computer (full-skeleton layout; UKF may use a subset layout from initialize_ukf)
    fk_ = std::make_unique<ForwardKinematics>(*model_, *data_, marker_frame_map_,
                                              SkeletonLayout::from_full_skeleton(skeleton_));

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
        if (marker_id >= static_cast<int>(skeleton_->markers().size())) {
            continue;
        }
        std::string marker_name = skeleton_->markers()[marker_id].name;

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
    auto ik_result = ik_solver_->solve(marker_positions, *skeleton_, std::nullopt,
                                       config_.ik_max_iterations, config_.ik_tolerance);

    if (!ik_result.converged) {
        // Accept non-converged solution if error is reasonable (< 50cm RMS)
        // The UKF may be able to refine it over subsequent frames
        if (ik_result.residual > 0.5) {
            fmt::print("IK failed badly (RMS: {:.3f}m) - cannot initialize\n", ik_result.residual);
            return false;
        }
        fmt::print("IK didn't fully converge (RMS: {:.3f}m), but proceeding with initialization\n",
                   ik_result.residual);
    }

    // Step 3: Initialize UKF
    initialize_ukf(ik_result.state, timestamp);

    initialized_ = true;
    last_timestamp_ = timestamp;

    return true;
}

void Tracker::initialize_from_rest_pose(double timestamp) {
    // Create state with all zeros (rest pose)
    int num_dof = skeleton_->total_dof_count();

    Eigen::Vector3d root_position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond root_orientation = Eigen::Quaterniond::Identity();
    Eigen::VectorXd joint_angles = Eigen::VectorXd::Zero(num_dof);
    Eigen::Vector3d root_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d root_angular_velocity = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_velocities = Eigen::VectorXd::Zero(num_dof);

    State rest_state(root_position, root_orientation, joint_angles, root_velocity,
                     root_angular_velocity, joint_velocities);

    // Initialize UKF with rest pose
    initialize_ukf(rest_state, timestamp);

    initialized_ = true;
    last_timestamp_ = timestamp;

    fmt::print("Initialized from rest pose (all zeros, bypassing IK)\n");
}

void Tracker::initialize_from_state(State const& initial_state, double timestamp) {
    // Initialize UKF with provided state
    initialize_ukf(initial_state, timestamp);

    initialized_ = true;
    last_timestamp_ = timestamp;

    fmt::print("Initialized from provided state\n");
}

void Tracker::initialize_ukf(State const& initial_state, double timestamp) {
    // Create UKF using config parameters (must match Python exactly)
    // Note: Small alpha (0.001) can cause numerical issues but is what Python uses
    double alpha = config_.ukf_alpha;  // Use config value (typically 0.001 for Python comparison)
    double beta = config_.ukf_beta;    // Gaussian distribution parameter
    double kappa = config_.ukf_kappa;  // Secondary scaling

    auto const& groups = config_.active_joint_groups;
    auto layout = groups.empty() ? SkeletonLayout::from_full_skeleton(skeleton_)
                                 : SkeletonLayout::from_groups(skeleton_, groups);
    ukf_ = std::make_unique<UnscentedKalmanFilter>(layout, config_.process_noise_std, alpha, beta,
                                                   kappa);

    // Set initial state
    ukf_->set_state(initial_state);

    // Set initial covariance — size from the UKF (driven by the layout, not skeleton.active_dof())
    int const error_dim = ukf_->error_dim();
    int const pos_dim = error_dim / 2;  // position/orientation half

    fmt::print("\n=== TRACKER INITIALIZATION DEBUG ===\n");
    fmt::print("total_dof (storage)={}, error_dim={}\n", skeleton_->total_dof_count(), error_dim);
    fmt::print("Covariance will be {}x{}\n", error_dim, error_dim);
    fmt::print(
        "init_position_std={}, init_orientation_std={}, init_joint_std={}, init_velocity_std={}\n",
        config_.init_position_std, config_.init_orientation_std, config_.init_joint_std,
        config_.init_velocity_std);

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

    fmt::print("Initial covariance diagonal values:\n");
    fmt::print("  Position (0:3): {}, {}, {}\n", initial_cov(0, 0), initial_cov(1, 1),
               initial_cov(2, 2));
    fmt::print("  Orientation (3:6): {}, {}, {}\n", initial_cov(3, 3), initial_cov(4, 4),
               initial_cov(5, 5));
    fmt::print("  Joint[0] (6): {}\n", initial_cov(6, 6));
    fmt::print("  Velocity pos ({}:{}): {}, {}, {}\n", pos_dim, pos_dim + 3,
               initial_cov(pos_dim, pos_dim), initial_cov(pos_dim + 1, pos_dim + 1),
               initial_cov(pos_dim + 2, pos_dim + 2));
    fmt::print("  Velocity orient ({}:{}): {}, {}, {}\n", pos_dim + 3, pos_dim + 6,
               initial_cov(pos_dim + 3, pos_dim + 3), initial_cov(pos_dim + 4, pos_dim + 4),
               initial_cov(pos_dim + 5, pos_dim + 5));
    fmt::print("===================================\n\n");

    ukf_->set_covariance(initial_cov);

    last_timestamp_ = timestamp;
}

TrackingResult Tracker::track_frame(std::vector<Observation> const& observations,
                                    double timestamp) {
    if (!initialized_) {
        throw std::runtime_error("Tracker::track_frame() called before initialization");
    }

    // Compute dt; reject out-of-order timestamps before doing any work
    double dt = timestamp - last_timestamp_;
    if (dt < 0.0) {
        return TrackingResult{timestamp,
                              ukf_->state(),
                              ukf_->covariance(),
                              {},
                              0,
                              true,
                              "Negative dt: timestamps out of order"};
    }

    auto result = run_parent_step(observations, dt, timestamp);

    if (!result.tracking_lost) {
        for (auto& child : children_) {
            run_child_step(child, observations, dt);
        }
        last_timestamp_ = timestamp;
        if (frame_callback_) {
            frame_callback_(result);
        }
    }

    return result;
}

TrackingResult Tracker::run_parent_step(std::vector<Observation> const& observations, double dt,
                                        double timestamp) {
    fmt::print("\n=== Tracking frame at timestamp {:.6f} ===\n", timestamp);

    // Step 1: Predict
    ukf_->predict(dt);

    // Step 2: Check if we have observations
    if (!has_sufficient_observations(observations)) {
        return TrackingResult{timestamp, ukf_->state(), ukf_->covariance(),         {},
                              0,         true,          "Insufficient observations"};
    }

    // Step 3: Update
    auto update_info = ukf_->update(observations, cameras_, *fk_, config_.measurement_noise_std,
                                    config_.outlier_threshold);

    // Debug: Export observation results (all frames) — runs even when all observations are outliers
    if (ukf_->is_debug_enabled()) {
        std::string const& debug_dir = ukf_->get_debug_dir();
        int frame_number = ukf_->get_frame_number();
        std::string frame_dir =
            debug_dir + "/frame_" +
            std::string(4 - std::min(4, static_cast<int>(std::to_string(frame_number).length())),
                        '0') +
            std::to_string(frame_number);
        std::filesystem::create_directories(frame_dir);
        std::ofstream f(frame_dir + "/all_observations.csv");
        f << std::setprecision(15);

        // Write header matching Python format (simplified)
        f << "marker_name,camera_id,frame_idx,observed_u,observed_v,predicted_u,predicted_v,"
          << "residual_u,residual_v,residual_norm,mahalanobis_distance,is_outlier\n";

        for (auto const& obs_result : update_info.observations) {
            f << obs_result.marker_name << "," << obs_result.camera_id << ","
              << obs_result.camera_frame_idx << "," << obs_result.actual.x() << ","
              << obs_result.actual.y() << "," << obs_result.predicted.x() << ","
              << obs_result.predicted.y() << "," << obs_result.innovation.x() << ","
              << obs_result.innovation.y() << "," << obs_result.innovation.norm() << ","
              << obs_result.mahalanobis_distance << ","
              << (obs_result.is_outlier ? "True" : "False") << "\n";
        }
    }

    // Step 4: Refresh FK on posterior state so children can call world_transform()
    fk_->compute(ukf_->state());

    return TrackingResult{timestamp,   ukf_->state(),           ukf_->covariance(),
                          update_info, update_info.num_inliers, false,
                          ""};
}

void Tracker::run_child_step(ChildFilter& /*child*/,
                             std::vector<Observation> const& /*observations*/, double /*dt*/) {
    // Phase 3h: inject root from parent FK, run child UKF predict+update, merge state
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

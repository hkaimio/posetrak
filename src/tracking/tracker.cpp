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
    // Build the same layout initialize_ukf will use, so we get the correct
    // joint_angles size (excludes root floating body DOFs).
    auto const& groups = config_.active_joint_groups;
    auto layout = groups.empty() ? SkeletonLayout::from_full_skeleton(skeleton_)
                                 : SkeletonLayout::from_groups(skeleton_, groups);
    int num_dof = layout->total_storage_dof_count();

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

    // When active_joint_groups restricts the layout, rebuild the pinocchio model/data/FK
    // scoped to those joints so that state_to_config works against the layout-sized state.
    // (The constructor built everything for the full skeleton; IK init already ran.)
    if (!groups.empty()) {
        // Find skeleton root joint (no parent) to use as free-flyer anchor.
        std::string root_name;
        for (auto const& joint : skeleton_->joints()) {
            if (!joint.parent_index.has_value()) {
                root_name = joint.name;
                break;
            }
        }
        if (root_name.empty()) {
            throw std::runtime_error("initialize_ukf: skeleton has no root joint");
        }

        model_ = std::make_unique<pinocchio::Model>();
        PinocchioModelBuilder::build_subtree_model(*skeleton_, root_name, groups, *model_);
        data_ = std::make_unique<pinocchio::Data>(*model_);
        marker_frame_map_ = PinocchioModelBuilder::build_subtree_marker_frame_map(
            *model_, *skeleton_, root_name, groups);
        fk_ = std::make_unique<ForwardKinematics>(*model_, *data_, marker_frame_map_, layout);
    }

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

    // Build child filters from config (must come after parent UKF is ready)
    build_children(initial_state);

    last_timestamp_ = timestamp;
}

void Tracker::build_children(State const& global_initial_state) {
    children_.clear();  // reset in case of re-initialization
    if (config_.child_filters.empty()) {
        return;
    }

    // Full-skeleton layout is needed to build merge maps from child layouts.
    auto full_layout = SkeletonLayout::from_full_skeleton(skeleton_);

    for (auto const& ccfg : config_.child_filters) {
        ChildFilter child;
        child.anchor_joint_name = ccfg.anchor_joint_name;
        child.measurement_noise_std = ccfg.measurement_noise_std;
        child.outlier_threshold = ccfg.outlier_threshold;

        // 1. Build subtree pinocchio model + data
        child.layout = SkeletonLayout::from_groups(skeleton_, ccfg.joint_groups);
        child.model = std::make_unique<pinocchio::Model>();
        PinocchioModelBuilder::build_subtree_model(*skeleton_, child.anchor_joint_name,
                                                   ccfg.joint_groups, *child.model);
        child.data = std::make_unique<pinocchio::Data>(*child.model);
        child.marker_frame_map = PinocchioModelBuilder::build_subtree_marker_frame_map(
            *child.model, *skeleton_, child.anchor_joint_name, ccfg.joint_groups);

        // 2. Build child FK
        child.fk = std::make_unique<ForwardKinematics>(*child.model, *child.data,
                                                       child.marker_frame_map, child.layout);

        // 3. Build child UKF
        child.ukf = std::make_unique<UnscentedKalmanFilter>(child.layout, ccfg.process_noise_std,
                                                            config_.ukf_alpha, config_.ukf_beta,
                                                            config_.ukf_kappa);

        // 4. Seed child state from the GLOBAL initial state (full-skeleton).
        //    ukf_->state() must NOT be used here — the parent UKF only holds
        //    parent-group DOFs and has no slots for child joints.
        child.ukf->set_state(slice_state_for_child(global_initial_state, *child.layout));

        // 5. Initial covariance: diagonal joint uncertainty (no floating root for child)
        int const child_error_dim = child.ukf->error_dim();
        child.ukf->set_covariance(Eigen::MatrixXd::Identity(child_error_dim, child_error_dim) *
                                  (config_.init_joint_std * config_.init_joint_std));

        // 6. Build merge map: child DOF i → state_index in full-skeleton layout
        child.merge_map = full_layout->build_index_map_from(*child.layout);

        // 7. Build marker_id remap: full-skeleton marker index → child-skeleton marker index.
        //    The child UKF indexes markers() by obs.marker_id; observations use full-skeleton IDs.
        {
            auto const& full_markers = skeleton_->markers();
            auto const& child_markers = child.layout->skeleton()->markers();
            for (int child_mid = 0; child_mid < static_cast<int>(child_markers.size());
                 ++child_mid) {
                std::string const& child_name = child_markers[child_mid].name;
                for (int full_mid = 0; full_mid < static_cast<int>(full_markers.size());
                     ++full_mid) {
                    if (full_markers[full_mid].name == child_name) {
                        child.marker_id_remap[full_mid] = child_mid;
                        break;
                    }
                }
            }
        }

        fmt::print("Built child filter '{}': {} joints, {} child markers, merge_map size {}\n",
                   ccfg.name, child.layout->joints().size(), child.marker_frame_map.size(),
                   child.merge_map.size());

        children_.push_back(std::move(child));
    }
}

State Tracker::slice_state_for_child(State const& global_state,
                                     SkeletonLayout const& child_layout) const {
    auto full_layout = SkeletonLayout::from_full_skeleton(skeleton_);
    auto const index_map = full_layout->build_index_map_from(child_layout);

    int const n = child_layout.total_storage_dof_count();
    Eigen::VectorXd child_angles(n);
    for (int i = 0; i < n; ++i) {
        child_angles[i] = global_state.joint_angles()[index_map[i]];
    }

    // Root will be overwritten by set_root_transform() before the first predict;
    // use identity here so the State is well-formed.
    State s(n);
    s.set_joint_angles(child_angles);
    return s;
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

void Tracker::run_child_step(ChildFilter& child, std::vector<Observation> const& obs, double dt) {
    // 1. Inject the anchor joint's world-transform (already refreshed by run_parent_step).
    auto [root_pos, root_ori] = fk_->world_transform(child.anchor_joint_name);
    child.ukf->set_root_transform(root_pos, root_ori);

    // 2. Predict (root stays fixed; process model integration is discarded for root)
    child.ukf->predict(dt);

    // 3. Remap observation marker_ids from full-skeleton to child-skeleton indexing,
    //    dropping any observations not relevant to this child filter.
    std::vector<Observation> child_obs;
    child_obs.reserve(obs.size());
    for (Observation const& o : obs) {
        auto it = child.marker_id_remap.find(o.marker_id);
        if (it != child.marker_id_remap.end()) {
            Observation remapped = o;
            remapped.marker_id = it->second;
            child_obs.push_back(remapped);
        }
    }

    // 4. Update — child FK silently ignores markers it doesn't know
    child.ukf->update(child_obs, cameras_, *child.fk, child.measurement_noise_std,
                      child.outlier_threshold);

    // 5. FK refresh (enables grandchild support in the future)
    child.fk->compute(child.ukf->state());

    // 6. Merge child joint angles back into parent UKF state
    State parent_state = ukf_->state();
    auto parent_angles = parent_state.joint_angles();
    auto const& child_angles = child.ukf->state().joint_angles();
    for (int i = 0; i < static_cast<int>(child.merge_map.size()); ++i) {
        parent_angles[child.merge_map[i]] = child_angles[i];
    }
    parent_state.set_joint_angles(parent_angles);
    ukf_->set_state(parent_state);
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

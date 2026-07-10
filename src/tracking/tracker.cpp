/**
 * @file tracker.cpp
 * @brief Implementation of main tracking orchestration
 */

#include "posetrak/tracking/tracker.hpp"

#include <Eigen/Dense>

#include <fmt/core.h>

#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/kinematics/pinocchio_model_builder.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <unordered_set>

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

    // Mechanism B (NIS-feedback regional fading) bookkeeping -- see
    // update_nis_feedback_scopes() doc comment. Built once here since it only depends
    // on skeleton_/config_, both fixed for the lifetime of this Tracker.
    for (Marker const& marker : skeleton_->markers()) {
        marker_to_joint_name_[marker.name] = skeleton_->joints()[marker.joint_index].name;
    }
    for (NisFeedbackScope const& scope : config_.nis_feedback_scopes) {
        NisFeedbackScopeWindow window;
        window.name = scope.name;
        window.joint_names =
            std::unordered_set<std::string>(scope.joint_names.begin(), scope.joint_names.end());
        nis_feedback_windows_.push_back(std::move(window));
    }
}

bool Tracker::initialize(std::vector<Observation> const& observations, double timestamp) {
    if (observations.empty()) {
        return false;
    }

    // Step 1: Triangulate marker positions
    std::map<int, std::vector<Observation>> obs_by_marker;
    for (auto const& obs : observations) {
        obs_by_marker[obs.marker_id].push_back(obs);
    }

    std::map<std::string, Eigen::Vector3d> marker_positions;
    for (auto const& [marker_id, marker_obs] : obs_by_marker) {
        if (marker_obs.size() < static_cast<size_t>(config_.min_cameras_for_init)) {
            continue;
        }
        if (marker_id >= static_cast<int>(skeleton_->markers().size())) {
            continue;
        }
        std::string marker_name = skeleton_->markers()[marker_id].name;

        std::vector<Eigen::Vector2d> pixel_coords;
        std::vector<Camera const*> marker_cameras;
        std::vector<double> confidences;
        for (auto const& obs : marker_obs) {
            auto it = cameras_.find(obs.camera_id);
            if (it == cameras_.end())
                continue;
            pixel_coords.push_back(obs.position);
            marker_cameras.push_back(&it->second);
            confidences.push_back(obs.confidence);
        }

        auto result = triangulator_->triangulate(pixel_coords, marker_cameras, confidences);
        if (result.success) {
            marker_positions[marker_name] = result.position;
        }
    }

    if (marker_positions.size() < 3) {
        return false;
    }
    init_marker_positions_ = marker_positions;

    // Step 2: Analytically estimate root position + orientation from observed markers.
    // This gives us a good global pose even before IK runs.
    auto estimate_analytic_state = [&]() -> State {
        // Always use the full skeleton DOF count: the IK works in full-skeleton
        // space regardless of active_joint_groups, so the initial_guess State
        // must be sized for all joints (including any inserted prismatic DOFs).
        int num_dof = skeleton_->total_dof_count();

        // Root position: hip midpoint if visible, else all-marker centroid.
        Eigen::Vector3d root_pos = Eigen::Vector3d::Zero();
        {
            auto hip_L = marker_positions.find("MRK-hip.L");
            auto hip_R = marker_positions.find("MRK-hip.R");
            if (hip_L != marker_positions.end() && hip_R != marker_positions.end()) {
                root_pos = (hip_L->second + hip_R->second) / 2.0;
            } else {
                Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
                for (auto const& [n, p] : marker_positions)
                    centroid += p;
                root_pos = centroid / static_cast<double>(marker_positions.size());
            }
        }

        // Root orientation: Procrustes alignment using FK rest-pose body axes vs observed axes.
        Eigen::Quaterniond root_ori = Eigen::Quaterniond::Identity();
        {
            // Rest-pose FK (root at origin, all joints zero)
            Eigen::VectorXd q_rest = Eigen::VectorXd::Zero(model_->nq);
            if (model_->nq >= 7)
                q_rest[6] = 1.0;
            auto rest_mkrs = fk_->compute(q_rest);

            auto get = [](auto const& m, std::string const& k) -> std::optional<Eigen::Vector3d> {
                auto it = m.find(k);
                if (it == m.end())
                    return std::nullopt;
                return it->second;
            };

            auto hL_r = get(rest_mkrs, "MRK-hip.L"), hR_r = get(rest_mkrs, "MRK-hip.R");
            auto sL_r = get(rest_mkrs, "MRK-shoulder.L"), sR_r = get(rest_mkrs, "MRK-shoulder.R");
            auto hL_o = get(marker_positions, "MRK-hip.L"),
                 hR_o = get(marker_positions, "MRK-hip.R");
            auto sL_o = get(marker_positions, "MRK-shoulder.L"),
                 sR_o = get(marker_positions, "MRK-shoulder.R");

            if (hL_r && hR_r && hL_o && hR_o && sL_r && sR_r && sL_o && sR_o) {
                Eigen::Vector3d hip_ctr_r = (*hL_r + *hR_r) / 2.0;
                Eigen::Vector3d hip_ctr_o = (*hL_o + *hR_o) / 2.0;
                Eigen::Vector3d spine_r = (*sL_r + *sR_r) / 2.0 - hip_ctr_r;
                Eigen::Vector3d spine_o = (*sL_o + *sR_o) / 2.0 - hip_ctr_o;
                Eigen::Vector3d lat_r = *hL_r - *hR_r;
                Eigen::Vector3d lat_o = *hL_o - *hR_o;

                if (spine_r.norm() > 0.05 && spine_o.norm() > 0.05 && lat_r.norm() > 0.05 &&
                    lat_o.norm() > 0.05) {
                    // Build orthonormal body frames: columns = [lateral, spine, forward]
                    auto make_frame = [](Eigen::Vector3d up, Eigen::Vector3d lat) {
                        up = up.normalized();
                        lat = (lat - lat.dot(up) * up).normalized();
                        Eigen::Matrix3d F;
                        F.col(0) = lat;
                        F.col(1) = up;
                        F.col(2) = lat.cross(up).normalized();
                        return F;
                    };
                    Eigen::Matrix3d F_r = make_frame(spine_r, lat_r);
                    Eigen::Matrix3d F_o = make_frame(spine_o, lat_o);

                    // R maps rest body frame to observed body frame: F_o = R * F_r
                    Eigen::Matrix3d R = F_o * F_r.transpose();
                    // Project onto SO(3)
                    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU |
                                                                 Eigen::ComputeFullV);
                    double det_sign = svd.matrixU().determinant() * svd.matrixV().determinant();
                    R = svd.matrixU() * Eigen::DiagonalMatrix<double, 3>(1.0, 1.0, det_sign) *
                        svd.matrixV().transpose();

                    root_ori = Eigen::Quaterniond(R);
                    root_ori.normalize();
                    fmt::print("  Analytic root orientation estimated from hip/shoulder markers\n");
                }
            }
        }

        // Build a rest-pose state with the analytically estimated root transform.
        // Scale-group DOFs (stored as proportional scale factors) must start at 1.0,
        // not zero, so that bone lengths equal the reference skeleton at t=0.
        Eigen::VectorXd init_joint_angles = Eigen::VectorXd::Zero(num_dof);
        {
            int si = 0;
            for (auto const& j : skeleton_->joints()) {
                if (!j.parent_index.has_value() || j.type == JointType::FIXED)
                    continue;
                if (j.is_scale_follower)
                    continue;  // shares leader's slot; no separate index
                if (j.type == JointType::PRISMATIC) {
                    init_joint_angles[si] = 1.0;  // neutral scale: q = 1.0 * nominal_length
                }
                si += j.dof;
            }
        }
        fmt::print("  Analytic root position: ({:.3f}, {:.3f}, {:.3f})\n", root_pos.x(),
                   root_pos.y(), root_pos.z());
        return State(root_pos, root_ori, init_joint_angles, Eigen::Vector3d::Zero(),
                     Eigen::Vector3d::Zero(), Eigen::VectorXd::Zero(num_dof));
    };

    State analytic_state = estimate_analytic_state();

    // Step 2.5: Analytical limb warm-start.
    // Before running DLS IK (which finds local minima), analytically orient each major
    // limb segment to point from its proximal to its distal triangulated marker. This
    // replaces the T-pose starting guess with a geometry-derived pose that is much closer
    // to the true configuration, allowing the IK to converge to the correct solution.
    {
        auto full_layout = SkeletonLayout::from_full_skeleton(skeleton_);
        Eigen::VectorXd warm_angles = analytic_state.joint_angles();
        int n = static_cast<int>(warm_angles.size());

        // Compute the rotation (as an axis-angle in the joint's LOCAL frame) that aligns
        // the bone direction from prox_mkr to dist_mkr with the triangulated target.
        // joint_name  – spherical joint whose angle to warm-start.
        // prox/dist   – marker names bracketing the bone.
        // fk_markers  – current FK result (used for rest/warm direction).
        auto align_joint = [&](std::string const& joint_name, std::string const& prox_mkr,
                               std::string const& dist_mkr,
                               std::unordered_map<std::string, Eigen::Vector3d> const& fk_markers) {
            auto it_prox_fk = fk_markers.find(prox_mkr);
            auto it_dist_fk = fk_markers.find(dist_mkr);
            auto it_prox_tgt = marker_positions.find(prox_mkr);
            auto it_dist_tgt = marker_positions.find(dist_mkr);
            if (it_prox_fk == fk_markers.end() || it_dist_fk == fk_markers.end() ||
                it_prox_tgt == marker_positions.end() || it_dist_tgt == marker_positions.end())
                return;

            Eigen::Vector3d rest_dir = (it_dist_fk->second - it_prox_fk->second).normalized();
            Eigen::Vector3d tgt_dir = (it_dist_tgt->second - it_prox_tgt->second).normalized();

            double cos_a = std::clamp(rest_dir.dot(tgt_dir), -1.0, 1.0);
            double angle = std::acos(cos_a);
            if (angle < 1e-4)
                return;

            Eigen::Vector3d axis_world = rest_dir.cross(tgt_dir);
            double axis_n = axis_world.norm();
            if (axis_n < 1e-8) {
                axis_world = rest_dir.unitOrthogonal();
            } else {
                axis_world /= axis_n;
            }

            // Convert world-frame axis to joint's local frame. The joint angle is
            // defined in the frame of the joint's fixed placement, which equals
            // data_.oMi[jidx] evaluated with zero joint angles.
            Eigen::Matrix3d R_joint_frame;
            try {
                auto [jpos, jrot] = fk_->world_transform(joint_name);
                R_joint_frame = jrot.toRotationMatrix();
            } catch (std::exception const&) {
                return;  // FIXED or unknown joint – skip
            }

            Eigen::Vector3d axis_local = R_joint_frame.transpose() * axis_world;
            Eigen::Vector3d aa_local = axis_local * angle;

            auto const* desc = full_layout->get_joint(joint_name);
            if (desc && desc->storage_dof_count == 3) {
                int idx = static_cast<int>(desc->state_index);
                if (idx + 2 < n) {
                    for (int i = 0; i < 3; ++i) {
                        if (desc->active_dof_mask[i]) {
                            double val = aa_local[i];
                            if (i < desc->limit_count) {
                                val = std::clamp(val, desc->limits[i].x(), desc->limits[i].y());
                            }
                            warm_angles[idx + i] = val;
                        }
                    }
                }
            }
        };

        auto make_state = [&]() {
            Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
            return State(analytic_state.root_position(), analytic_state.root_orientation(),
                         warm_angles, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), z);
        };

        // Phase 1: upper arms and thighs (from T-pose rest FK).
        auto rest_fk = fk_->compute(analytic_state);
        align_joint("upper_arm.L", "MRK-shoulder.L", "MRK-elbow.L", rest_fk);
        align_joint("upper_arm.R", "MRK-shoulder.R", "MRK-elbow.R", rest_fk);
        align_joint("thigh.L", "MRK-hip.L", "MRK-knee.L", rest_fk);
        align_joint("thigh.R", "MRK-hip.R", "MRK-knee.R", rest_fk);

        // Phase 2: forearms and shins (after upper-arm/thigh angles are propagated).
        auto warm1_fk = fk_->compute(make_state());
        align_joint("forearm.L", "MRK-elbow.L", "MRK-wrist.L", warm1_fk);
        align_joint("forearm.R", "MRK-elbow.R", "MRK-wrist.R", warm1_fk);
        align_joint("shin.L", "MRK-knee.L", "MRK-Ankle.L", warm1_fk);
        align_joint("shin.R", "MRK-knee.R", "MRK-Ankle.R", warm1_fk);

        analytic_state = make_state();
        fmt::print("  Warm-start applied: joint_angles L2 = {:.4f}\n",
                   analytic_state.joint_angles().norm());
    }

    // Step 3: Run IK from the analytic starting state to refine joint angles.
    // Pass it as initial_guess so IK uses our root estimate instead of re-computing from scratch.
    fmt::print("  Running IK: max_iter={} tol={:.4f}m\n", config_.ik_max_iterations,
               config_.ik_tolerance);
    auto ik_result = ik_solver_->solve(marker_positions, *skeleton_, analytic_state,
                                       config_.ik_max_iterations, config_.ik_tolerance);

    // Step 4: Choose the best available state for UKF initialisation.
    // IK may not fully converge (joint angles are hard), but as long as the root is
    // in the right place the UKF will fix the pose within a handful of frames.
    //
    // Sanity check: if the IK displaced the root by more than max_root_drift from
    // the analytically estimated position, the IK found a degenerate local minimum
    // (e.g. root 1 m too high with joints compensating).  In that case keep the IK
    // joint angles but restore the analytic root pose, which is the Procrustes
    // estimate from triangulated hip/shoulder markers and is usually more reliable.
    constexpr double max_root_drift = 0.50;  // metres
    State init_state = analytic_state;       // baseline: correct root, zero joints
    if (ik_result.residual < 0.5) {
        double root_drift =
            (ik_result.state.root_position() - analytic_state.root_position()).norm();
        if (root_drift <= max_root_drift) {
            // IK converged and root stayed close — trust the full IK result.
            init_state = ik_result.state;
            fmt::print("  Using IK result (RMS: {:.3f} m, root drift: {:.3f} m)\n",
                       ik_result.residual, root_drift);
        } else {
            // IK moved the root too far — use IK joint angles but analytic root pose.
            init_state = ik_result.state;
            init_state.set_root_position(analytic_state.root_position());
            init_state.set_root_orientation(analytic_state.root_orientation());
            fmt::print(
                "  IK root drifted {:.3f} m > {:.2f} m limit — using IK joints + "
                "analytic root (RMS: {:.3f} m)\n",
                root_drift, max_root_drift, ik_result.residual);
        }
    } else {
        fmt::print(
            "  IK residual {:.3f} m > 0.50 m — using analytic root estimate with zero "
            "joint angles (UKF will refine over first frames)\n",
            ik_result.residual);
    }

    // Step 5: Initialize UKF
    initialize_ukf(init_state, timestamp);
    initialized_ = true;
    last_timestamp_ = timestamp;
    return true;
}

void Tracker::initialize_from_rest_pose(double timestamp) {
    // Create state with all zeros (rest pose)
    // Use layout that will be created by initialize_ukf to size the state correctly
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
    ukf_ = std::make_unique<UnscentedKalmanFilter>(layout, config_.process_noise_std, alpha, beta,
                                                   kappa);
    if (config_.process_noise_vel_std)
        ukf_->set_vel_noise_std(*config_.process_noise_vel_std);
    if (config_.velocity_half_life_s)
        ukf_->set_vel_half_life(*config_.velocity_half_life_s);
    ukf_->set_velocity_noise_gain(
        config_.process_noise_vel_gain_joint, config_.process_noise_vel_ref_joint,
        config_.process_noise_vel_gain_root, config_.process_noise_vel_ref_root,
        config_.process_noise_vel_joint_names);
    {
        std::vector<UnscentedKalmanFilter::VelocityNoiseScope> scopes;
        for (VelocityNoiseScope const& scope : config_.process_noise_vel_scopes) {
            scopes.push_back({scope.name, scope.joint_names, scope.gain, scope.vel_ref});
        }
        ukf_->set_velocity_noise_gain_scopes(scopes);
    }
    ukf_->set_pose_regularization(config_.pose_reg_joint_names,
                                  config_.pose_reg_equal_split_noise_std,
                                  config_.pose_reg_rest_pose_noise_std);
    ukf_->set_soft_joint_limits(config_.soft_limit_joint_names, config_.soft_limit_margin_rad,
                                config_.soft_limit_noise_std);
    ukf_->set_near_limit_damping(config_.near_limit_damping_joint_names,
                                 config_.near_limit_margin_rad, config_.near_limit_spread_sigma,
                                 config_.near_limit_damping_factor);
    {
        std::vector<std::pair<std::string, std::vector<std::string>>> scopes;
        for (NisFeedbackScope const& scope : config_.nis_feedback_scopes)
            scopes.emplace_back(scope.name, scope.joint_names);
        ukf_->set_nis_feedback_scopes(scopes);
    }

    // Enable calibration mode if requested (prismatic DOFs get small process noise)
    fmt::print("calibration_mode={}, prismatic_process_noise_std={}\n", config_.calibration_mode,
               config_.prismatic_process_noise_std);
    if (config_.calibration_mode) {
        ukf_->enable_calibration_mode(config_.prismatic_process_noise_std);
        fmt::print("  Calibration mode enabled: prismatic DOFs will drift during tracking\n");
    } else {
        fmt::print("  Calibration mode OFF: prismatic DOFs are frozen\n");
    }

    // Slice initial state to match layout dimensions if needed
    State sliced_state = layout->slice_state(initial_state);

    // Set initial state (now correctly sized for the layout)
    ukf_->set_state(sliced_state);

    // Rebuild FK if using a subset layout (so state_to_config and FK work on layout-sized state)
    if (!groups.empty()) {
        // Find skeleton root joint (the one with no parent)
        std::string root_joint_name;
        for (auto const& joint : skeleton_->joints()) {
            if (!joint.parent_index.has_value()) {
                root_joint_name = joint.name;
                break;
            }
        }
        if (root_joint_name.empty()) {
            throw std::runtime_error("Tracker::initialize_ukf: No root joint found in skeleton");
        }

        // Rebuild pinocchio model/data/FK scoped to the layout groups
        // Use build_subtree_model with the skeleton root as the freeflyer
        model_ = std::make_unique<pinocchio::Model>();
        PinocchioModelBuilder::build_subtree_model(*skeleton_, root_joint_name, groups, *model_);
        data_ = std::make_unique<pinocchio::Data>(*model_);
        marker_frame_map_ = PinocchioModelBuilder::build_subtree_marker_frame_map(
            *model_, *skeleton_, root_joint_name, groups);
        fk_ = std::make_unique<ForwardKinematics>(*model_, *data_, marker_frame_map_, layout);
    }

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

std::vector<Observation>
Tracker::build_annotated_observations(std::vector<Observation> const& observations) const {
    if (config_.velocity_mode_camera_ids.empty())
        return observations;

    std::unordered_set<int> vel_cams(config_.velocity_mode_camera_ids.begin(),
                                     config_.velocity_mode_camera_ids.end());
    std::vector<Observation> annotated = observations;
    for (Observation& obs : annotated) {
        if (!vel_cams.count(obs.camera_id))
            continue;
        auto cam_it = prev_observations_.find(obs.camera_id);
        if (cam_it == prev_observations_.end())
            continue;  // first frame for this camera — stay POSITION
        auto marker_it = cam_it->second.find(obs.marker_id);
        if (marker_it == cam_it->second.end())
            continue;  // marker not seen last frame — stay POSITION
        obs.mode = MeasurementMode::VELOCITY;
        obs.prev_position = marker_it->second;
        if (config_.velocity_measurement_noise_std.has_value())
            obs.noise_std_override = *config_.velocity_measurement_noise_std;
    }
    return annotated;
}

void Tracker::update_nis_feedback_scopes(std::vector<ObservationResult> const& observations) {
    if (nis_feedback_windows_.empty())
        return;

    for (NisFeedbackScopeWindow& window : nis_feedback_windows_) {
        double step_sum_sq = 0.0;
        int step_dof = 0;
        for (ObservationResult const& obs : observations) {
            auto joint_it = marker_to_joint_name_.find(obs.marker_name);
            if (joint_it == marker_to_joint_name_.end())
                continue;
            if (!window.joint_names.count(joint_it->second))
                continue;
            step_sum_sq += obs.mahalanobis_distance * obs.mahalanobis_distance;
            step_dof += 2;  // each observation is a 2-DOF (u,v) pixel residual
        }

        window.step_sum_mahal_sq.push_back(step_sum_sq);
        window.step_dof_count.push_back(step_dof);
        window.running_sum_mahal_sq += step_sum_sq;
        window.running_dof_count += step_dof;
        if (static_cast<int>(window.step_sum_mahal_sq.size()) > config_.nis_feedback_window) {
            window.running_sum_mahal_sq -= window.step_sum_mahal_sq.front();
            window.running_dof_count -= window.step_dof_count.front();
            window.step_sum_mahal_sq.pop_front();
            window.step_dof_count.pop_front();
        }

        double const windowed_nis_dof = (window.running_dof_count > 0)
                                            ? window.running_sum_mahal_sq / window.running_dof_count
                                            : 0.0;
        double multiplier = 1.0;
        if (windowed_nis_dof > config_.nis_feedback_threshold) {
            multiplier = std::min(config_.nis_feedback_max_multiplier,
                                  windowed_nis_dof / config_.nis_feedback_threshold);
        }
        ukf_->set_scope_noise_multiplier(window.name, multiplier);
    }
}

void Tracker::print_init_debug(State const& state, std::string const& label) const {
    auto fk_markers = fk_->compute(state);

    Eigen::Quaterniond const& q = state.root_orientation();
    fmt::print("\n=== INIT DEBUG [{}] frame={} ===\n", label, frame_count_);
    fmt::print("  root_pos = ({:.3f}, {:.3f}, {:.3f})\n", state.root_position().x(),
               state.root_position().y(), state.root_position().z());
    fmt::print("  root_quat = ({:.3f}, {:.3f}, {:.3f}, {:.3f})  [x,y,z,w]\n", q.x(), q.y(), q.z(),
               q.w());

    struct MarkerError {
        std::string name;
        double error;
        Eigen::Vector3d fk_pos;
        Eigen::Vector3d tgt_pos;
    };
    std::vector<MarkerError> errors;
    double rms_sq = 0.0;

    for (auto const& [name, fk_pos] : fk_markers) {
        auto it = init_marker_positions_.find(name);
        if (it == init_marker_positions_.end())
            continue;
        double err = (fk_pos - it->second).norm();
        rms_sq += err * err;
        errors.push_back({name, err, fk_pos, it->second});
    }

    std::sort(errors.begin(), errors.end(),
              [](MarkerError const& a, MarkerError const& b) { return a.error > b.error; });

    int n = static_cast<int>(errors.size());
    double rms = n > 0 ? std::sqrt(rms_sq / n) : 0.0;
    int gt5cm = static_cast<int>(
        std::count_if(errors.begin(), errors.end(), [](auto& e) { return e.error > 0.05; }));
    fmt::print("  n={} markers  RMS={:.4f}m  >{:.0f}cm: {}\n", n, rms, 5.0, gt5cm);
    fmt::print("  {:35s}  {:>7}  {:>25}  {:>25}\n", "marker", "err(m)", "fk(x,y,z)", "tgt(x,y,z)");
    for (auto const& me : errors) {
        fmt::print("  {:35s}  {:.4f}  ({:6.3f},{:6.3f},{:6.3f})  ({:6.3f},{:6.3f},{:6.3f})\n",
                   me.name, me.error, me.fk_pos.x(), me.fk_pos.y(), me.fk_pos.z(), me.tgt_pos.x(),
                   me.tgt_pos.y(), me.tgt_pos.z());
    }
    fmt::print("\n");
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

    std::vector<Observation> const annotated = build_annotated_observations(observations);
    auto result = run_parent_step(annotated, dt, timestamp);

    if (!result.tracking_lost) {
        for (auto& child : children_) {
            run_child_step(child, annotated, dt);
        }
        last_timestamp_ = timestamp;
        ++frame_count_;
        // Store raw pixel positions for next frame's velocity-mode annotation
        for (Observation const& obs : observations) {
            prev_observations_[obs.camera_id][obs.marker_id] = obs.position;
        }
        if (frame_callback_) {
            frame_callback_(result);
        }
    }

    return result;
}

TrackingResult Tracker::run_parent_step(std::vector<Observation> const& observations, double dt,
                                        double timestamp) {
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    // Step 1: Predict
    auto t0 = Clock::now();
    auto predict_result = ukf_->predict(dt);
    double const predict_ms = Ms(Clock::now() - t0).count();

    State const prior_state = ukf_->state();
    Eigen::MatrixXd const prior_cov = ukf_->covariance();

    if (frame_count_ < config_.debug_init_frames) {
        print_init_debug(prior_state, "PRIOR ");
    }

    // Step 2: Check if we have observations
    if (!has_sufficient_observations(observations)) {
        return TrackingResult{timestamp, ukf_->state(), ukf_->covariance(),         {},
                              0,         true,          "Insufficient observations"};
    }

    // Step 3: Update
    auto t1 = Clock::now();
    auto update_info = ukf_->update(observations, cameras_, *fk_, config_.pose_noise_std,
                                    config_.calib_noise_std, config_.outlier_threshold);
    double const update_ms = Ms(Clock::now() - t1).count();

    // Mechanism B: feed this step's per-observation Mahalanobis distances into each
    // configured scope's windowed NIS/DOF, and push the resulting multiplier into the
    // UKF so it's in effect for the next predict() call.
    update_nis_feedback_scopes(update_info.observations);

    if (frame_count_ < config_.debug_init_frames) {
        fmt::print(
            "\n=== UPDATE OBS DEBUG frame={} n_obs={} n_inliers={} n_outliers={} "
            "NIS={:.1f}/dof={} meas_noise_std={:.1f} outlier_thr={:.2f} ===\n",
            frame_count_, update_info.num_observations, update_info.num_inliers,
            update_info.num_outliers, update_info.nis, update_info.nis_dof, config_.calib_noise_std,
            config_.outlier_threshold);
        fmt::print("  {:35s}  {:>4}  {:>7}  {:>14}  {:>14}  {:>14}  {:>7}\n", "marker", "cam",
                   "inlier?", "pred(u,v)", "obs(u,v)", "innov(u,v)", "mahal");
        for (auto const& r : update_info.observations) {
            fmt::print(
                "  {:35s}  {:>4}  {:>7}  ({:6.1f},{:6.1f})  ({:6.1f},{:6.1f})  "
                "({:+6.1f},{:+6.1f})  {:6.2f}\n",
                r.marker_name, r.camera_id, r.is_outlier ? "OUT" : "in", r.predicted.x(),
                r.predicted.y(), r.actual.x(), r.actual.y(), r.innovation.x(), r.innovation.y(),
                r.mahalanobis_distance);
        }
        fmt::print("\n");
        print_init_debug(ukf_->state(), "POSTER");
    }

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

    // Accumulate RTS smoother data if enabled (only for successful frames).
    // Resolve the async cross-covariance future here: update() (~56ms) has fully overlapped
    // with the async computation (~16ms), so get() should return immediately.
    if (smoothing_enabled_) {
        smoother_cache_.push_back(FrameSmootherData{
            timestamp,
            ukf_->state(),
            ukf_->covariance(),  // posterior
            prior_state,
            prior_cov,                              // prior
            predict_result.cross_cov_future.get(),  // D_{k-1,k}: already done
        });
    } else {
        // Future destructor blocks until complete; get() here keeps it explicit and cheap
        // since the async work finished during update().
        predict_result.cross_cov_future.get();
    }

    return TrackingResult{
        timestamp,
        ukf_->state(),
        ukf_->covariance(),
        update_info,
        update_info.num_inliers,
        false,
        "",
        predict_ms,
        update_ms,
        predict_result.sigma_gen_ms,
        predict_result.propagate_ms,
        predict_result.mean_cov_ms,
        predict_result.rts_ms,
        update_info.fk1_ms,
        update_info.s_ms,
        update_info.outlier_ms,
        update_info.fk2_ms,
        update_info.inlier_ms,
        update_info.kalman_ms,
        update_info.cov_update_ms,
    };
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
    smoother_cache_.clear();
    prev_observations_.clear();
}

// ─── RTS smoothing ────────────────────────────────────────────────────────────

void Tracker::enable_smoothing(bool enable) {
    smoothing_enabled_ = enable;
    if (!enable) {
        smoother_cache_.clear();
    }
}

std::vector<SmoothedFrame> Tracker::smooth() const {
    if (!smoothing_enabled_) {
        throw std::runtime_error(
            "Tracker::smooth(): smoothing was not enabled. "
            "Call enable_smoothing(true) before track_frame().");
    }
    if (smoother_cache_.empty()) {
        throw std::runtime_error("Tracker::smooth(): no frames tracked yet.");
    }
    RTSSmoother smoother(ukf_->layout());
    return smoother.smooth(smoother_cache_);
}

}  // namespace posetrak

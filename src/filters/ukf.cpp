/**
 * @file ukf.cpp
 * @brief Implementation of Unscented Kalman Filter
 */

#include "posetrak/filters/ukf.hpp"

#include <fmt/core.h>

#include <omp.h>

#include "posetrak/core/skeleton_layout.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace posetrak {

// static void debug_print_state(std::ostream& os, State const& s,
//                               Skeleton const& skeleton, std::string const& line_prefix = "") {
//     // Write propagated sigma points (active joints only)
//     os << line_prefix << "Root pos:" << s.root_position().x() << "," << s.root_position().y() <<
//     ","
//        << s.root_position().z() << std::endl;
//     os << line_prefix << "Root orientation:" << s.root_orientation().w() << ","
//        << s.root_orientation().x() << "," << s.root_orientation().y() << ","
//        << s.root_orientation().z() << std::endl;
//     os << line_prefix << "Root velocity:" << s.root_velocity().x() << "," <<
//     s.root_velocity().y()
//        << "," << s.root_velocity().z() << std::endl;
//     os << line_prefix << "Root angular velocity:" << s.root_angular_velocity().x() << ","
//        << s.root_angular_velocity().y() << "," << s.root_angular_velocity().z() << std::endl;
//     int idx = 0;
//     for (auto& joint : skeleton.joints()) {
//         if (!joint.parent_index.has_value()) {
//             continue;  // Skip root (handled separately)
//         }
//         bool is_active = skeleton.is_joint_active(joint.name);
//         if (is_active) {
//             os << line_prefix << joint.name << ":";
//             if (joint.type == JointType::REVOLUTE) {
//                 os << s.joint_angles()(idx)
//                    << " (vel: " << s.joint_velocities()(idx) << ")";
//             } else if (joint.type == JointType::SPHERICAL) {
//                 os << s.joint_angles().segment<3>(idx).transpose()
//                    << " (vel: " << s.joint_velocities().segment<3>(idx).transpose()
//                    << ")";
//             }
//         }
//         os << std::endl;
//         int const dof_count =
//             (joint.type == JointType::SPHERICAL) ? 3 : (joint.type == JointType::REVOLUTE ? 1 :
//             0);
//         idx += dof_count;
//     }
// }

UnscentedKalmanFilter::UnscentedKalmanFilter(std::shared_ptr<const SkeletonLayout> layout,
                                             double process_noise_std, double alpha, double beta,
                                             double kappa)
    : layout_(layout),
      state_(layout->skeleton()->total_dof_count()),
      covariance_(Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim())),
      process_noise_(
          Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim())),
      sigma_gen_(layout, alpha, beta, kappa),
      process_model_(layout),
      base_noise_std_(process_noise_std),
      vel_noise_std_(process_noise_std) {
    // Build per-DOF process noise (prismatic DOFs start frozen; calibration mode activates them)
    rebuild_process_noise();

    // Debug: Print DOF counts
    fmt::print("UKF initialized: total_dof={}, error_dim={}\n",
               layout->skeleton()->total_dof_count(), layout->error_state_dim());
}

void UnscentedKalmanFilter::rebuild_process_noise() {
    int const n = error_dim();
    int const root_n = layout_->root_error_dof_count();
    int const jac = layout_->joint_active_dof_count();
    int const active_dof = root_n + jac;
    double const pos_var = base_noise_std_ * base_noise_std_;
    double const vel_var = vel_noise_std_ * vel_noise_std_;
    double const pris_var = calibration_mode_ ? prismatic_noise_std_ * prismatic_noise_std_ : 0.0;

    process_noise_.setZero(n, n);
    // Position/angle block (first active_dof dimensions)
    process_noise_.topLeftCorner(active_dof, active_dof) =
        pos_var * Eigen::MatrixXd::Identity(active_dof, active_dof);
    // Velocity block (second active_dof dimensions)
    process_noise_.bottomRightCorner(active_dof, active_dof) =
        vel_var * Eigen::MatrixXd::Identity(active_dof, active_dof);

    // Override prismatic DOF diagonals (frozen at zero unless calibration mode)
    for (JointDesc const& j : layout_->joints()) {
        if (j.type != JointType::PRISMATIC)
            continue;
        for (uint32_t d = 0; d < j.active_dof_count; ++d) {
            int const pos_idx = root_n + static_cast<int>(j.error_index) + static_cast<int>(d);
            int const vel_idx =
                active_dof + root_n + static_cast<int>(j.error_index) + static_cast<int>(d);
            process_noise_(pos_idx, pos_idx) = pris_var;
            process_noise_(vel_idx, vel_idx) = pris_var;
        }
    }
}

void UnscentedKalmanFilter::set_vel_noise_std(double vel_noise_std) {
    vel_noise_std_ = vel_noise_std;
    rebuild_process_noise();
}

void UnscentedKalmanFilter::set_vel_half_life(double half_life_s) {
    vel_half_life_s_ = half_life_s;
}

void UnscentedKalmanFilter::enable_calibration_mode(double prismatic_noise_std) {
    calibration_mode_ = true;
    prismatic_noise_std_ = prismatic_noise_std;
    rebuild_process_noise();
}

void UnscentedKalmanFilter::set_covariance(Eigen::MatrixXd const& covariance) {
    int const expected_dim = error_dim();
    if (covariance.rows() != expected_dim || covariance.cols() != expected_dim) {
        throw std::invalid_argument("Covariance size must match error dimension");
    }
    covariance_ = covariance;
}

void UnscentedKalmanFilter::set_root_transform(Eigen::Vector3d const& position,
                                               Eigen::Quaterniond const& orientation) {
    if (layout_->has_floating_root()) {
        return;  // Safety no-op: parent filters manage their own root.
    }
    fixed_root_pos_ = position;
    fixed_root_ori_ = orientation.normalized();
    // Update nominal state immediately so sigma generation starts from the correct root.
    state_.set_root_position(fixed_root_pos_);
    state_.set_root_orientation(fixed_root_ori_);
}

PredictResult UnscentedKalmanFilter::predict(double dt) {
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;
    PredictResult result;

    // Save posterior state x_{k|k} before anything modifies state_.
    // Needed both for the RTS smoother cross-covariance and for velocity-mode measurements.
    State const posterior_state = state_;
    prev_posterior_state_ = state_;

    // Generate sigma points
    auto t0 = Clock::now();
    auto sigma_points = sigma_gen_.generate_sigma_points(state_, covariance_);
    result.sigma_gen_ms = Ms(Clock::now() - t0).count();

    // Debug: Export generated sigma points (frame 0 - Python matching format)
    if (debug_enabled_ && frame_number_ == 0) {
        std::filesystem::create_directories(debug_dir_ + "/frame_0000");
        std::ofstream f(debug_dir_ + "/frame_0000/predict_sigma_points_generated.csv");
        f << std::setprecision(15);

        // Build list of active joints from layout (layout only contains active joints)
        std::vector<std::pair<std::string, int>> active_joint_info;  // (joint_name, dof_start_idx)
        for (auto const& jdesc : layout_->joints()) {
            if (!jdesc.is_floating_root) {
                active_joint_info.push_back({jdesc.name, jdesc.state_index});
            }
        }

        // Write header matching Python format (named joints, active only)
        f << "sigma_idx,root_pos_x,root_pos_y,root_pos_z,"
          << "root_quat_w,root_quat_x,root_quat_y,root_quat_z,"
          << "root_vel_x,root_vel_y,root_vel_z,"
          << "root_angvel_x,root_angvel_y,root_angvel_z";
        for (auto const& [joint_name, dof_idx] : active_joint_info) {
            // Determine DOF count for this joint
            auto const* joint = layout_->skeleton()->get_joint(joint_name);
            int num_dof = (joint->type == JointType::SPHERICAL) ? 3 : 1;
            for (int i = 0; i < num_dof; ++i) {
                f << "," << joint_name << "_angle_" << i;
            }
        }
        for (auto const& [joint_name, dof_idx] : active_joint_info) {
            auto const* joint = layout_->skeleton()->get_joint(joint_name);
            int num_dof = (joint->type == JointType::SPHERICAL) ? 3 : 1;
            for (int i = 0; i < num_dof; ++i) {
                f << "," << joint_name << "_vel_" << i;
            }
        }
        f << "\n";

        // Write sigma points (active joints only)
        for (size_t i = 0; i < sigma_points.size(); ++i) {
            auto const& s = sigma_points[i];
            f << i;
            f << "," << s.root_position().x() << "," << s.root_position().y() << ","
              << s.root_position().z();
            f << "," << s.root_orientation().w() << "," << s.root_orientation().x() << ","
              << s.root_orientation().y() << "," << s.root_orientation().z();
            f << "," << s.root_velocity().x() << "," << s.root_velocity().y() << ","
              << s.root_velocity().z();
            f << "," << s.root_angular_velocity().x() << "," << s.root_angular_velocity().y() << ","
              << s.root_angular_velocity().z();
            // Write only active joint angles
            for (auto const& [joint_name, dof_idx] : active_joint_info) {
                auto const* joint = layout_->skeleton()->get_joint(joint_name);
                int num_dof = (joint->type == JointType::SPHERICAL) ? 3 : 1;
                for (int j = 0; j < num_dof; ++j) {
                    f << "," << s.joint_angles()(dof_idx + j);
                }
            }
            // Write only active joint velocities
            for (auto const& [joint_name, dof_idx] : active_joint_info) {
                auto const* joint = layout_->skeleton()->get_joint(joint_name);
                int num_dof = (joint->type == JointType::SPHERICAL) ? 3 : 1;
                for (int j = 0; j < num_dof; ++j) {
                    f << "," << s.joint_velocities()(dof_idx + j);
                }
            }
            f << "\n";
        }
        std::cout << "DEBUG: Exported generated sigma points (" << active_joint_info.size()
                  << " active joints)\n";
    }

    // Debug: Export sigma points before propagation (frame 1 only)
    if (debug_enabled_ && frame_number_ == 1) {
        std::filesystem::create_directories(debug_dir_ + "/frame_0001");
        std::ofstream f(debug_dir_ + "/frame_0001/sigma_points_before.csv");
        f << std::setprecision(15);

        // Write header
        f << "sigma_idx,root_x,root_y,root_z,root_qw,root_qx,root_qy,root_qz";
        for (int i = 0; i < layout_->skeleton()->total_dof_count(); ++i) {
            f << ",joint_" << i;
        }
        f << ",root_vx,root_vy,root_vz,root_wx,root_wy,root_wz";
        for (int i = 0; i < layout_->skeleton()->total_dof_count(); ++i) {
            f << ",joint_vel_" << i;
        }
        f << "\n";

        // Write sigma points
        for (size_t i = 0; i < sigma_points.size(); ++i) {
            auto const& s = sigma_points[i];
            f << i;
            f << "," << s.root_position().x() << "," << s.root_position().y() << ","
              << s.root_position().z();
            f << "," << s.root_orientation().w() << "," << s.root_orientation().x() << ","
              << s.root_orientation().y() << "," << s.root_orientation().z();
            for (int j = 0; j < s.num_dof(); ++j) {
                f << "," << s.joint_angles()(j);
            }
            f << "," << s.root_velocity().x() << "," << s.root_velocity().y() << ","
              << s.root_velocity().z();
            f << "," << s.root_angular_velocity().x() << "," << s.root_angular_velocity().y() << ","
              << s.root_angular_velocity().z();
            for (int j = 0; j < s.num_dof(); ++j) {
                f << "," << s.joint_velocities()(j);
            }
            f << "\n";
        }
        std::cout << "DEBUG: Exported sigma points before propagation to " << debug_dir_
                  << "/frame_0001/sigma_points_before.csv\n";
    }

    // Propagate sigma points through process model
    auto t1 = Clock::now();
    std::vector<State> propagated_points;
    propagated_points.reserve(sigma_points.size());

    for (auto const& sigma_state : sigma_points) {
        propagated_points.push_back(process_model_.propagate(sigma_state, dt));
    }

    // Apply exponential velocity decay to prevent unbounded covariance growth.
    // Velocities are not directly observable (H depends only on positions/angles),
    // so without damping, velocity variance accumulates every predict() step.
    if (vel_half_life_s_ > 0.0) {
        double const alpha = std::pow(0.5, dt / vel_half_life_s_);
        for (auto& sp : propagated_points) {
            sp.set_root_velocity(alpha * sp.root_velocity());
            sp.set_root_angular_velocity(alpha * sp.root_angular_velocity());
            sp.set_joint_velocities(alpha * sp.joint_velocities());
        }
    }

    // [Child filter] Root pose is externally supplied — overwrite process-model root drift.
    if (!layout_->has_floating_root()) {
        for (auto& sp : propagated_points) {
            sp.set_root_position(fixed_root_pos_);
            sp.set_root_orientation(fixed_root_ori_);
        }
    }

    // Debug: Export propagated sigma points (frame 0 - Python matching format)
    if (debug_enabled_ && frame_number_ == 0) {
        std::ofstream f(debug_dir_ + "/frame_0000/predict_sigma_points_propagated.csv");
        f << std::setprecision(15);

        // Build list of active joints from layout (layout only contains active joints)
        std::vector<std::pair<std::string, int>> active_joint_info;
        for (auto const& jdesc : layout_->joints()) {
            if (!jdesc.is_floating_root) {
                active_joint_info.push_back({jdesc.name, jdesc.state_index});
            }
        }

        // Write header matching Python format (named joints, active only)
        f << "sigma_idx,root_pos_x,root_pos_y,root_pos_z,"
          << "root_quat_w,root_quat_x,root_quat_y,root_quat_z,"
          << "root_vel_x,root_vel_y,root_vel_z,"
          << "root_angvel_x,root_angvel_y,root_angvel_z";
        for (auto const& [joint_name, dof_idx] : active_joint_info) {
            auto const* joint = layout_->skeleton()->get_joint(joint_name);
            int num_dof = (joint->type == JointType::SPHERICAL) ? 3 : 1;
            for (int i = 0; i < num_dof; ++i) {
                f << "," << joint_name << "_angle_" << i;
            }
        }
        for (auto const& [joint_name, dof_idx] : active_joint_info) {
            auto const* joint = layout_->skeleton()->get_joint(joint_name);
            int num_dof = (joint->type == JointType::SPHERICAL) ? 3 : 1;
            for (int i = 0; i < num_dof; ++i) {
                f << "," << joint_name << "_vel_" << i;
            }
        }
        f << "\n";

        // Write propagated sigma points (active joints only)
        for (size_t i = 0; i < propagated_points.size(); ++i) {
            auto const& s = propagated_points[i];
            f << i;
            f << "," << s.root_position().x() << "," << s.root_position().y() << ","
              << s.root_position().z();
            f << "," << s.root_orientation().w() << "," << s.root_orientation().x() << ","
              << s.root_orientation().y() << "," << s.root_orientation().z();
            f << "," << s.root_velocity().x() << "," << s.root_velocity().y() << ","
              << s.root_velocity().z();
            f << "," << s.root_angular_velocity().x() << "," << s.root_angular_velocity().y() << ","
              << s.root_angular_velocity().z();
            // Write only active joint angles
            for (auto const& [joint_name, dof_idx] : active_joint_info) {
                auto const* joint = layout_->skeleton()->get_joint(joint_name);
                int num_dof = (joint->type == JointType::SPHERICAL) ? 3 : 1;
                for (int j = 0; j < num_dof; ++j) {
                    f << "," << s.joint_angles()(dof_idx + j);
                }
            }
            // Write only active joint velocities
            for (auto const& [joint_name, dof_idx] : active_joint_info) {
                auto const* joint = layout_->skeleton()->get_joint(joint_name);
                int num_dof = (joint->type == JointType::SPHERICAL) ? 3 : 1;
                for (int j = 0; j < num_dof; ++j) {
                    f << "," << s.joint_velocities()(dof_idx + j);
                }
            }
            f << "\n";
        }
        std::cout << "DEBUG: Exported propagated sigma points (" << active_joint_info.size()
                  << " active joints)\n";
    }

    // Debug: Export sigma points after propagation (frame 1 only)
    if (debug_enabled_ && frame_number_ == 1) {
        std::ofstream f(debug_dir_ + "/frame_0001/sigma_points_after.csv");
        f << std::setprecision(15);

        // Write header
        f << "sigma_idx,root_x,root_y,root_z,root_qw,root_qx,root_qy,root_qz";
        for (int i = 0; i < layout_->skeleton()->total_dof_count(); ++i) {
            f << ",joint_" << i;
        }
        f << ",root_vx,root_vy,root_vz,root_wx,root_wy,root_wz";
        for (int i = 0; i < layout_->skeleton()->total_dof_count(); ++i) {
            f << ",joint_vel_" << i;
        }
        f << "\n";

        // Write propagated sigma points
        for (size_t i = 0; i < propagated_points.size(); ++i) {
            auto const& s = propagated_points[i];
            f << i;
            f << "," << s.root_position().x() << "," << s.root_position().y() << ","
              << s.root_position().z();
            f << "," << s.root_orientation().w() << "," << s.root_orientation().x() << ","
              << s.root_orientation().y() << "," << s.root_orientation().z();
            for (int j = 0; j < s.num_dof(); ++j) {
                f << "," << s.joint_angles()(j);
            }
            f << "," << s.root_velocity().x() << "," << s.root_velocity().y() << ","
              << s.root_velocity().z();
            f << "," << s.root_angular_velocity().x() << "," << s.root_angular_velocity().y() << ","
              << s.root_angular_velocity().z();
            for (int j = 0; j < s.num_dof(); ++j) {
                f << "," << s.joint_velocities()(j);
            }
            f << "\n";
        }
        std::cout << "DEBUG: Exported sigma points after propagation to " << debug_dir_
                  << "/frame_0001/sigma_points_after.csv\n";
    }

    result.propagate_ms = Ms(Clock::now() - t1).count();

    // Compute predicted mean
    auto t2 = Clock::now();
    state_ = compute_state_mean(propagated_points, sigma_gen_.get_mean_weights());

    // Enforce joint limits on mean state (CRITICAL: must be done before computing covariance!)
    // This resets locked DOFs to their fixed values, ensuring error vectors are near-zero
    // for those dimensions. Without this, locked DOF errors can be huge (~157) and corrupt
    // covariance when weighted by large negative weight wc[0].
    enforce_joint_limits();

    // Debug: Export predicted mean state (frame 0 - JSON format)
    if (debug_enabled_ && frame_number_ == 0) {
        std::ofstream f(debug_dir_ + "/frame_0000/predict_state_mean.json");
        f << std::setprecision(15);
        f << "{\n";
        f << "  \"root_position\": [" << state_.root_position().x() << ", "
          << state_.root_position().y() << ", " << state_.root_position().z() << "],\n";
        f << "  \"root_quaternion\": [" << state_.root_orientation().w() << ", "
          << state_.root_orientation().x() << ", " << state_.root_orientation().y() << ", "
          << state_.root_orientation().z() << "],\n";
        f << "  \"root_velocity\": [" << state_.root_velocity().x() << ", "
          << state_.root_velocity().y() << ", " << state_.root_velocity().z() << "],\n";
        f << "  \"root_angular_velocity\": [" << state_.root_angular_velocity().x() << ", "
          << state_.root_angular_velocity().y() << ", " << state_.root_angular_velocity().z()
          << "],\n";
        f << "  \"joint_angles\": [";
        for (int i = 0; i < state_.num_dof(); ++i) {
            if (i > 0)
                f << ", ";
            f << state_.joint_angles()(i);
        }
        f << "],\n";
        f << "  \"joint_velocities\": [";
        for (int i = 0; i < state_.num_dof(); ++i) {
            if (i > 0)
                f << ", ";
            f << state_.joint_velocities()(i);
        }
        f << "]\n}\n";
        std::cout << "DEBUG: Exported predicted mean state\n";
    }

    // Debug: Export predicted mean state (prior state for frame 1)
    if (debug_enabled_ && frame_number_ == 1) {
        std::ofstream f(debug_dir_ + "/frame_0001/prior_state_computed.csv");
        f << std::setprecision(15);
        f << "root_x,root_y,root_z,root_qw,root_qx,root_qy,root_qz";
        for (int i = 0; i < layout_->skeleton()->total_dof_count(); ++i) {
            f << ",joint_" << i;
        }
        f << ",root_vx,root_vy,root_vz,root_wx,root_wy,root_wz";
        for (int i = 0; i < layout_->skeleton()->total_dof_count(); ++i) {
            f << ",joint_vel_" << i;
        }
        f << "\n";

        f << state_.root_position().x() << "," << state_.root_position().y() << ","
          << state_.root_position().z();
        f << "," << state_.root_orientation().w() << "," << state_.root_orientation().x() << ","
          << state_.root_orientation().y() << "," << state_.root_orientation().z();
        for (int j = 0; j < state_.num_dof(); ++j) {
            f << "," << state_.joint_angles()(j);
        }
        f << "," << state_.root_velocity().x() << "," << state_.root_velocity().y() << ","
          << state_.root_velocity().z();
        f << "," << state_.root_angular_velocity().x() << "," << state_.root_angular_velocity().y()
          << "," << state_.root_angular_velocity().z();
        for (int j = 0; j < state_.num_dof(); ++j) {
            f << "," << state_.joint_velocities()(j);
        }
        f << "\n";
        std::cout << "DEBUG: Exported computed prior state to " << debug_dir_
                  << "/frame_0001/prior_state_computed.csv\n";
    }

    // Compute predicted covariance
    covariance_ =
        compute_state_covariance(propagated_points, state_, sigma_gen_.get_covariance_weights());

    // Debug: Export covariance before process noise (frame 0 - Python matching format)
    if (debug_enabled_ && frame_number_ == 0) {
        std::ofstream f(debug_dir_ + "/frame_0000/predict_covariance_before_process_noise.csv");
        f << std::setprecision(15);

        // No header, just data (matching Python format)
        for (int i = 0; i < covariance_.rows(); ++i) {
            for (int j = 0; j < covariance_.cols(); ++j) {
                if (j > 0)
                    f << ",";
                f << covariance_(i, j);
            }
            f << "\n";
        }
        std::cout << "DEBUG: Exported covariance before process noise\n";
    }

    result.mean_cov_ms = Ms(Clock::now() - t2).count();

    // Add process noise
    covariance_ += process_noise_ * dt;

    // Debug: Export covariance after process noise (frame 0 - Python matching format)
    if (debug_enabled_ && frame_number_ == 0) {
        std::ofstream f(debug_dir_ + "/frame_0000/predict_covariance_after_process_noise.csv");
        f << std::setprecision(15);

        // No header, just data (matching Python format)
        for (int i = 0; i < covariance_.rows(); ++i) {
            for (int j = 0; j < covariance_.cols(); ++j) {
                if (j > 0)
                    f << ",";
                f << covariance_(i, j);
            }
            f << "\n";
        }
        std::cout << "DEBUG: Exported covariance after process noise\n";
    }

    // Debug: Export prior covariance if debug mode enabled
    if (debug_enabled_) {
        write_matrix_csv(covariance_, "prior_covariance.csv");
    }

    // ── RTS smoother cross-covariance (async) ────────────────────────────────
    // D = Σ_i W_c^i * e_pre_i * e_prop_i^T
    // e_pre_i  = tangent error of sigma_points[i]     wrt posterior x_{k|k}
    // e_prop_i = tangent error of propagated_points[i] wrt predicted mean x_{k+1|k}
    //
    // Launched as std::async so that this ~16ms computation runs in parallel with
    // the caller's update() step (~56ms).  sigma_points and propagated_points are
    // moved into the lambda; compute_state_error only reads layout_ (const) so
    // there is no data race with concurrent update() access to state_/covariance_.
    {
        auto t3 = Clock::now();
        Eigen::VectorXd wc = sigma_gen_.get_covariance_weights();
        int const edim = error_dim();
        result.cross_cov_future =
            std::async(std::launch::async,
                       [this, sp = std::move(sigma_points), pp = std::move(propagated_points),
                        post = posterior_state, pred = state_, wc = std::move(wc), edim]() mutable {
                           int const n_sigma = static_cast<int>(sp.size());
                           Eigen::MatrixXd E_pre(edim, n_sigma), E_prop(edim, n_sigma);
                           for (int i = 0; i < n_sigma; ++i) {
                               E_pre.col(i) = compute_state_error(sp[i], post);
                               E_prop.col(i) = compute_state_error(pp[i], pred);
                           }
                           Eigen::MatrixXd EW_pre = E_pre;
                           for (int i = 0; i < n_sigma; ++i)
                               EW_pre.col(i) *= wc(i);
                           Eigen::MatrixXd cross_cov(edim, edim);
                           cross_cov.noalias() = EW_pre * E_prop.transpose();
                           return cross_cov;
                       });
        result.rts_ms = Ms(Clock::now() - t3).count();
        return result;
    }
}

State UnscentedKalmanFilter::compute_state_mean(std::vector<State> const& states,
                                                Eigen::VectorXd const& weights) const {
    // Create mean state
    State mean_state(layout_->skeleton()->total_dof_count());

    // Mean position (simple weighted average)
    Eigen::Vector3d pos_mean = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < states.size(); ++i) {
        pos_mean += weights(i) * states[i].root_position();
    }
    mean_state.set_root_position(pos_mean);

    // Mean quaternion (iterative on manifold)
    Eigen::Quaterniond q_mean = states[0].root_orientation();
    for (int iter = 0; iter < 5; ++iter) {
        Eigen::Vector3d error_sum = Eigen::Vector3d::Zero();

        for (size_t i = 0; i < states.size(); ++i) {
            Eigen::Quaterniond const& q_i = states[i].root_orientation();
            // Compute quaternion difference: q_mean^-1 * q_i
            Eigen::Quaterniond q_diff = q_mean.conjugate() * q_i;

            // Ensure shortest rotation (w >= 0)
            if (q_diff.w() < 0.0) {
                q_diff.w() = -q_diff.w();
                q_diff.x() = -q_diff.x();
                q_diff.y() = -q_diff.y();
                q_diff.z() = -q_diff.z();
            }

            // Convert to axis-angle (error space)
            double const angle = 2.0 * std::atan2(q_diff.vec().norm(), q_diff.w());
            if (angle > 1e-8) {
                Eigen::Vector3d const axis = q_diff.vec().normalized();
                error_sum += weights(i) * angle * axis;
            }
        }

        // Update mean quaternion
        Eigen::Quaterniond q_error = State::axis_angle_to_quaternion(error_sum);
        q_mean = (q_mean * q_error).normalized();

        if (error_sum.norm() < 1e-6) {
            break;
        }
    }
    mean_state.set_root_orientation(q_mean);

    // Mean velocities (simple weighted average)
    Eigen::Vector3d vel_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel_mean = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < states.size(); ++i) {
        vel_mean += weights(i) * states[i].root_velocity();
        angvel_mean += weights(i) * states[i].root_angular_velocity();
    }
    mean_state.set_root_velocity(vel_mean);
    mean_state.set_root_angular_velocity(angvel_mean);

    // Mean joint angles and velocities
    Eigen::VectorXd angles_mean = Eigen::VectorXd::Zero(layout_->skeleton()->total_dof_count());
    Eigen::VectorXd velocities_mean = Eigen::VectorXd::Zero(layout_->skeleton()->total_dof_count());

    for (JointDesc const& j : layout_->joints()) {
        int const si = j.state_index;

        if ((j.type == JointType::REVOLUTE) ||
            (j.type == JointType::PRISMATIC && !j.is_scale_follower)) {
            // Simple weighted average (followers share the leader's slot; skip to avoid
            // double-accumulation on the same state_index).
            for (size_t i = 0; i < states.size(); ++i) {
                angles_mean(si) += weights(i) * states[i].joint_angles()(si);
                velocities_mean(si) += weights(i) * states[i].joint_velocities()(si);
            }

        } else if (j.type == JointType::SPHERICAL) {
            // Always 3 DOFs: iterative mean on SO(3) manifold
            Eigen::Vector3d const initial_aa = states[0].joint_angles().segment<3>(si);
            Eigen::Matrix3d R_mean = State::axis_angle_to_quaternion(initial_aa).toRotationMatrix();

            for (int iter = 0; iter < 10; ++iter) {
                Eigen::Vector3d error_sum = Eigen::Vector3d::Zero();

                for (size_t i = 0; i < states.size(); ++i) {
                    Eigen::Vector3d const aa_i = states[i].joint_angles().segment<3>(si);
                    Eigen::Matrix3d const R_i =
                        State::axis_angle_to_quaternion(aa_i).toRotationMatrix();

                    // Relative rotation: R_mean^T * R_i
                    Eigen::Matrix3d const R_rel = R_mean.transpose() * R_i;
                    Eigen::Quaterniond q_rel(R_rel);

                    // Ensure quaternion uses shortest rotation (w >= 0)
                    if (q_rel.w() < 0.0) {
                        q_rel.w() = -q_rel.w();
                        q_rel.x() = -q_rel.x();
                        q_rel.y() = -q_rel.y();
                        q_rel.z() = -q_rel.z();
                    }

                    Eigen::Vector3d const error_i = State::quaternion_to_axis_angle(q_rel);

                    error_sum += weights(i) * error_i;
                }

                // Update mean
                Eigen::Matrix3d const R_delta =
                    State::axis_angle_to_quaternion(error_sum).toRotationMatrix();
                R_mean = R_mean * R_delta;

                if (error_sum.norm() < 1e-6) {
                    break;
                }
            }

            // Convert back to axis-angle
            Eigen::Quaterniond const q_mean(R_mean);
            angles_mean.segment<3>(si) = State::quaternion_to_axis_angle(q_mean);

            // Velocities: simple weighted average
            for (size_t i = 0; i < states.size(); ++i) {
                velocities_mean.segment<3>(si) +=
                    weights(i) * states[i].joint_velocities().segment<3>(si);
            }
        }
    }
    Eigen::Index i;
    mean_state.set_joint_angles(angles_mean);
    mean_state.set_joint_velocities(velocities_mean);

    return mean_state;
}

Eigen::MatrixXd
UnscentedKalmanFilter::compute_state_covariance(std::vector<State> const& states,
                                                State const& mean_state,
                                                Eigen::VectorXd const& weights) const {
    int const n = error_dim();
    int const n_sigma = states.size();

    // Compute all error vectors
    Eigen::MatrixXd error_vectors(n_sigma, n);
    for (int i = 0; i < n_sigma; ++i) {
        error_vectors.row(i) = compute_state_error(states[i], mean_state);
    }

    // Debug: Export error vectors (frame 0 - Python matching format)
    if (debug_enabled_ && frame_number_ == 0) {
        std::ofstream f(debug_dir_ + "/frame_0000/predict_error_vectors.csv");
        f << std::setprecision(15);

        // No header, just data (matching Python format)
        for (int i = 0; i < n_sigma; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j > 0)
                    f << ",";
                f << error_vectors(i, j);
            }
            f << "\n";
        }
        std::cout << "DEBUG: Exported error vectors\n";
    }

    // Weighted covariance via DGEMM: P = error_vectors^T * EW
    // where EW = diag(weights) * error_vectors. Same pattern as S/Pxy refactor.
    Eigen::MatrixXd EW = error_vectors;
    for (int i = 0; i < n_sigma; ++i)
        EW.row(i) *= weights(i);
    Eigen::MatrixXd cov(n, n);
    cov.noalias() = error_vectors.transpose() * EW;

    return cov;
}

Eigen::VectorXd UnscentedKalmanFilter::compute_state_error(State const& state,
                                                           State const& reference) const {
    int const root_n = layout_->root_error_dof_count();  // 6
    int const jac = layout_->joint_active_dof_count();
    int const active_dof = root_n + jac;  // == error_dim() / 2
    Eigen::VectorXd error = Eigen::VectorXd::Zero(error_dim());

    if (root_n > 0) {
        // Position error
        error.segment<3>(0) = state.root_position() - reference.root_position();

        // Rotation error (in tangent space)
        Eigen::Quaterniond const& q_ref = reference.root_orientation();
        Eigen::Quaterniond const& q_state = state.root_orientation();
        Eigen::Quaterniond const q_diff = q_ref.conjugate() * q_state;

        // Convert to axis-angle
        double const angle = 2.0 * std::atan2(q_diff.vec().norm(), q_diff.w());
        if (angle > 1e-8) {
            Eigen::Vector3d const axis = q_diff.vec().normalized();
            error.segment<3>(3) = angle * axis;
        }

        // Root velocity errors (in velocity section of error vector)
        error.segment<3>(active_dof) = state.root_velocity() - reference.root_velocity();
        error.segment<3>(active_dof + 3) =
            state.root_angular_velocity() - reference.root_angular_velocity();
    }

    // Joint angle and velocity errors

    for (JointDesc const& j : layout_->joints()) {
        int const si = j.state_index;
        int const pos_base = root_n + j.error_index;
        int const vel_base = active_dof + root_n + j.error_index;

        if ((j.type == JointType::REVOLUTE) ||
            (j.type == JointType::PRISMATIC && !j.is_scale_follower)) {
            // Followers share state_index with leader; skip to avoid a second (wrong)
            // error entry for the same logical DOF.
            error(pos_base) = state.joint_angles()(si) - reference.joint_angles()(si);
            error(vel_base) = state.joint_velocities()(si) - reference.joint_velocities()(si);

        } else if (j.type == JointType::SPHERICAL) {
            if (j.active_dof_count == 3) {
                // All 3 DOFs active: use SO(3) manifold error
                Eigen::Vector3d const aa_ref = reference.joint_angles().segment<3>(si);
                Eigen::Vector3d const aa_state = state.joint_angles().segment<3>(si);

                Eigen::Matrix3d const R_ref =
                    State::axis_angle_to_quaternion(aa_ref).toRotationMatrix();
                Eigen::Matrix3d const R_state =
                    State::axis_angle_to_quaternion(aa_state).toRotationMatrix();

                // Relative rotation: R_ref^T * R_state
                Eigen::Matrix3d const R_rel = R_ref.transpose() * R_state;
                Eigen::Quaterniond const q_rel(R_rel);
                error.segment<3>(pos_base) = State::quaternion_to_axis_angle(q_rel);

                // Velocity error
                error.segment<3>(vel_base) = state.joint_velocities().segment<3>(si) -
                                             reference.joint_velocities().segment<3>(si);

            } else {
                // Some DOFs locked: Euclidean error only for active DOFs
                int partial = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    if (j.active_dof_mask[axis]) {
                        error(pos_base + partial) =
                            state.joint_angles()(si + axis) - reference.joint_angles()(si + axis);
                        error(vel_base + partial) = state.joint_velocities()(si + axis) -
                                                    reference.joint_velocities()(si + axis);
                        partial++;
                    }
                }
            }
        }
        // FIXED joints have 0 DOF
    }

    return error;
}

void UnscentedKalmanFilter::ensure_data_pool(ForwardKinematics const& fk) const {
    int const n = omp_get_max_threads();
    if (static_cast<int>(data_pool_.size()) < n) {
        data_pool_.clear();
        data_pool_.reserve(n);
        for (int t = 0; t < n; ++t)
            data_pool_.emplace_back(fk.model());
    }
}

UpdateResult UnscentedKalmanFilter::update(std::vector<Observation> const& observations,
                                           std::unordered_map<int, Camera> const& cameras,
                                           ForwardKinematics& fk, double measurement_noise_std,
                                           double outlier_threshold_mahalanobis) {
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;
    UpdateResult result;

    if (observations.empty()) {
        return result;  // No observations to process
    }

    int const n_obs = static_cast<int>(observations.size());
    int const measurement_dim = 2 * n_obs;  // 2D pixel per observation

    // Step 1: Generate sigma points from current state and covariance
    auto sigma_points = sigma_gen_.generate_sigma_points(state_, covariance_);
    int const n_sigma = static_cast<int>(sigma_points.size());

    // Debug: Export sigma points if debug mode enabled
    if (debug_enabled_) {
        write_sigma_points_csv(sigma_points);
    }

    // Precompute previous-frame projections for velocity-mode observations.
    // One FK evaluation on the saved posterior; result is read-only in the parallel loops below.
    std::unordered_map<int, std::unordered_map<int, Eigen::Vector2d>> prev_projections;
    {
        bool has_velocity =
            std::any_of(observations.begin(), observations.end(),
                        [](Observation const& o) { return o.mode == MeasurementMode::VELOCITY; });
        if (has_velocity) {
            auto prev_marker_pos = fk.compute(prev_posterior_state_);
            for (Observation const& obs : observations) {
                if (obs.mode != MeasurementMode::VELOCITY)
                    continue;
                auto const& marker = layout_->skeleton()->markers()[obs.marker_id];
                auto pos_it = prev_marker_pos.find(marker.name);
                if (pos_it == prev_marker_pos.end())
                    continue;
                auto cam_it = cameras.find(obs.camera_id);
                if (cam_it == cameras.end())
                    continue;
                auto proj = cam_it->second.project_undistorted(pos_it->second, false);
                if (proj.has_value())
                    prev_projections[obs.camera_id][obs.marker_id] = *proj;
            }
        }
    }

    // Step 2: Predict measurements for each sigma point (parallelized over sigma points)
    auto t_fk1 = Clock::now();
    Eigen::MatrixXd predicted_measurements(measurement_dim, n_sigma);
    ensure_data_pool(fk);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_sigma; ++i) {
        ForwardKinematics fk_local(fk.model(), data_pool_[omp_get_thread_num()],
                                   fk.marker_frame_map(), fk.fk_layout());
        predicted_measurements.col(i) = predict_measurements(sigma_points[i], observations, cameras,
                                                             fk_local, prev_projections);
    }
    result.fk1_ms = Ms(Clock::now() - t_fk1).count();

    // Step 3: Compute mean predicted measurement (using nanmean to ignore NaN)
    // For each measurement dimension, compute weighted mean of non-NaN values
    Eigen::VectorXd const weights_mean = sigma_gen_.get_mean_weights();
    Eigen::VectorXd measurement_mean = Eigen::VectorXd::Zero(measurement_dim);

    for (int dim = 0; dim < measurement_dim; ++dim) {
        double sum = 0.0;
        double weight_sum = 0.0;

        for (int i = 0; i < n_sigma; ++i) {
            double val = predicted_measurements(dim, i);
            if (std::isfinite(val)) {
                sum += weights_mean(i) * val;
                weight_sum += weights_mean(i);
            }
        }

        if (weight_sum > 0.0) {
            measurement_mean(dim) = sum / weight_sum;
        } else {
            // All sigma points have NaN for this dimension
            measurement_mean(dim) = std::numeric_limits<double>::quiet_NaN();
        }
    }

    // Step 4: Compute innovation covariance S = Pyy + R via batched DGEMM.
    // Build Z_full: NaN-safe centered innovations matrix (measurement_dim × n_sigma).
    // NaN predictions (marker behind camera) → 0, so they contribute nothing to S.
    auto t_s = Clock::now();
    Eigen::VectorXd const weights_cov = sigma_gen_.get_covariance_weights();
    Eigen::MatrixXd Z_full(measurement_dim, n_sigma);
    for (int i = 0; i < n_sigma; ++i) {
        Z_full.col(i) = predicted_measurements.col(i) - measurement_mean;
        for (int d = 0; d < measurement_dim; ++d)
            if (!std::isfinite(Z_full(d, i)))
                Z_full(d, i) = 0.0;
    }
    // ZW_full[:,i] = weights_cov[i] * Z_full[:,i]. S = ZW_full * Z_full^T (DGEMM).
    Eigen::MatrixXd ZW_full = Z_full;
    for (int i = 0; i < n_sigma; ++i)
        ZW_full.col(i) *= weights_cov(i);
    Eigen::MatrixXd innovation_cov(measurement_dim, measurement_dim);
    innovation_cov.noalias() = ZW_full * Z_full.transpose();

    // Add measurement noise R (diagonal)
    for (int i = 0; i < n_obs; ++i) {
        double noise_std = observations[i].measurement_noise_std(measurement_noise_std);
        double variance = noise_std * noise_std;
        innovation_cov(2 * i, 2 * i) += variance;
        innovation_cov(2 * i + 1, 2 * i + 1) += variance;
    }

    // Precompute state error matrix E (error_dim × n_sigma) used in Pxy.
    // compute_state_error is a pure const function — safe to run in parallel.
    Eigen::MatrixXd E(error_dim(), n_sigma);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_sigma; ++i)
        E.col(i) = compute_state_error(sigma_points[i], state_);
    Eigen::MatrixXd EW = E;
    for (int i = 0; i < n_sigma; ++i)
        EW.col(i) *= weights_cov(i);
    result.s_ms = Ms(Clock::now() - t_s).count();

    // Step 4.5: Perform outlier rejection if enabled
    std::vector<Observation> inlier_observations;
    std::vector<ObservationResult> observation_results;
    Eigen::MatrixXd cross_cov;
    Eigen::VectorXd observed;

    if (outlier_threshold_mahalanobis > 0.0) {
        // Perform outlier rejection
        auto t_outlier = Clock::now();
        auto [inliers, results] =
            reject_outliers(observations, predicted_measurements, measurement_mean, innovation_cov,
                            outlier_threshold_mahalanobis);
        inlier_observations = inliers;
        observation_results = results;
        result.outlier_ms = Ms(Clock::now() - t_outlier).count();

        // Debug: log outlier counts
        if (debug_enabled_ && frame_number_ == 0) {
            size_t nan_count = 0;
            size_t mahal_count = 0;
            for (auto const& r : results) {
                if (r.is_outlier) {
                    if (r.mahalanobis_distance == 0.0) {
                        nan_count++;
                    } else {
                        mahal_count++;
                    }
                }
            }
            std::cout << "  [DEBUG] Frame " << frame_number_ << ": " << observations.size()
                      << " total obs, " << inliers.size() << " inliers, "
                      << (results.size() - inliers.size()) << " outliers (" << nan_count
                      << " NaN proj, " << mahal_count << " Mahalanobis)\n";
        }

        // If all observations rejected, return early
        if (inlier_observations.empty()) {
            result.num_observations = static_cast<int>(observations.size());
            result.num_outliers = static_cast<int>(observations.size());
            result.num_inliers = 0;
            result.observations = observation_results;
            return result;
        }

        // Recompute predictions with only inliers (parallel FK)
        auto t_fk2 = Clock::now();
        int const n_inliers = static_cast<int>(inlier_observations.size());
        int const inlier_dim = 2 * n_inliers;

        Eigen::MatrixXd inlier_predictions(inlier_dim, n_sigma);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_sigma; ++i) {
            ForwardKinematics fk_local(fk.model(), data_pool_[omp_get_thread_num()],
                                       fk.marker_frame_map(), fk.fk_layout());
            inlier_predictions.col(i) = predict_measurements(sigma_points[i], inlier_observations,
                                                             cameras, fk_local, prev_projections);
        }
        result.fk2_ms = Ms(Clock::now() - t_fk2).count();

        // Recompute mean predicted measurement for inliers (NaN-safe)
        measurement_mean = Eigen::VectorXd::Zero(inlier_dim);
        for (int dim = 0; dim < inlier_dim; ++dim) {
            double sum = 0.0;
            double weight_sum = 0.0;

            for (int i = 0; i < n_sigma; ++i) {
                double val = inlier_predictions(dim, i);
                if (std::isfinite(val)) {
                    sum += weights_mean(i) * val;
                    weight_sum += weights_mean(i);
                }
            }

            if (weight_sum > 0.0) {
                measurement_mean(dim) = sum / weight_sum;
            } else {
                // All sigma points have NaN - should not happen for inliers
                measurement_mean(dim) = std::numeric_limits<double>::quiet_NaN();
            }
        }

        // Build Z_inlier and recompute S for inliers via DGEMM
        auto t_inlier = Clock::now();
        Eigen::MatrixXd Z_inlier(inlier_dim, n_sigma);
        for (int i = 0; i < n_sigma; ++i) {
            Z_inlier.col(i) = inlier_predictions.col(i) - measurement_mean;
            for (int d = 0; d < inlier_dim; ++d)
                if (!std::isfinite(Z_inlier(d, i)))
                    Z_inlier(d, i) = 0.0;
        }
        Eigen::MatrixXd ZW_inlier = Z_inlier;
        for (int i = 0; i < n_sigma; ++i)
            ZW_inlier.col(i) *= weights_cov(i);
        innovation_cov.resize(inlier_dim, inlier_dim);
        innovation_cov.noalias() = ZW_inlier * Z_inlier.transpose();

        // Add measurement noise for inliers
        for (int i = 0; i < n_inliers; ++i) {
            double noise_std = inlier_observations[i].measurement_noise_std(measurement_noise_std);
            double variance = noise_std * noise_std;
            innovation_cov(2 * i, 2 * i) += variance;
            innovation_cov(2 * i + 1, 2 * i + 1) += variance;
        }

        // Cross-covariance Pxy = EW * Z_inlier^T (DGEMM)
        cross_cov.resize(error_dim(), inlier_dim);
        cross_cov.noalias() = EW * Z_inlier.transpose();
        result.inlier_ms = Ms(Clock::now() - t_inlier).count();

        // Update observed vector to use inliers
        observed = observations_to_vector(inlier_observations);

        // ===============================================================================
        // CRITICAL FIX FOR REPROJECTION ERROR REPORTING BUG
        // ===============================================================================
        // Problem: The observation_results computed by reject_outliers() contain
        // innovation values based on the OLD measurement_mean (computed with ALL
        // observations including outliers). After outlier rejection, we recomputed
        // measurement_mean using ONLY inliers. The innovations in observation_results
        // are now stale and don't match the actual innovations used in the filter update.
        //
        // Impact: StatisticsTracker computes reprojection errors from these stale
        // innovations, resulting in ZERO or incorrect error values being exported.
        //
        // Solution: Recompute the innovation field in observation_results to use the
        // updated measurement_mean. This ensures the reported reprojection errors
        // match what's actually happening in the filter.
        // ===============================================================================
        for (size_t i = 0; i < observations.size(); ++i) {
            auto& obs_result = observation_results[i];

            // Skip outliers - their innovations are not used in the update
            if (obs_result.is_outlier) {
                continue;
            }

            // Find this observation in the inlier list to get its index in measurement_mean
            int inlier_idx = -1;
            for (size_t j = 0; j < inlier_observations.size(); ++j) {
                // Match by marker_id, camera_id, and frame_idx (uniquely identifies obs)
                if (inlier_observations[j].marker_id == observations[i].marker_id &&
                    inlier_observations[j].camera_id == observations[i].camera_id &&
                    inlier_observations[j].frame_idx == observations[i].frame_idx) {
                    inlier_idx = static_cast<int>(j);
                    break;
                }
            }

            if (inlier_idx >= 0) {
                // Recompute innovation using the updated measurement_mean for inliers
                Eigen::Vector2d updated_predicted = measurement_mean.segment<2>(2 * inlier_idx);
                Eigen::Vector2d actual = observed.segment<2>(2 * inlier_idx);
                obs_result.innovation = actual - updated_predicted;
                obs_result.predicted = updated_predicted;
            }
        }
    } else {
        // No outlier rejection - compute diagnostics for all observations
        observation_results =
            compute_observation_diagnostics(observations, measurement_mean, innovation_cov);
        inlier_observations = observations;
        observed = observations_to_vector(observations);

        // If every measurement dimension has a NaN mean (i.e. all sigma-point projections
        // failed — e.g. marker behind camera), there is no usable information to integrate.
        // Return early to keep the state and covariance untouched, mirroring what the
        // outlier-rejection path does when every observation is rejected.
        bool all_projections_failed = true;
        for (int dim = 0; dim < measurement_dim; ++dim) {
            if (std::isfinite(measurement_mean(dim))) {
                all_projections_failed = false;
                break;
            }
        }
        if (all_projections_failed) {
            result.num_observations = n_obs;
            result.num_outliers = n_obs;  // treat failed projections as outliers for accounting
            result.num_inliers = 0;
            result.observations = observation_results;
            return result;
        }
    }

    // Debug export moved to Tracker::track_frame to ensure it runs even when all observations are
    // outliers

    // Step 5: Compute cross-covariance Pxy = EW * Z^T (DGEMM).
    // If outlier rejection ran, Z_inlier and cross_cov are already set above.
    if (outlier_threshold_mahalanobis <= 0.0) {
        cross_cov.resize(error_dim(), measurement_dim);
        cross_cov.noalias() = EW * Z_full.transpose();
    }

    // Step 6–9: Factorize S (innovation_cov) once via LDLT; reuse for Kalman gain,
    // covariance update, and NIS — replacing three O(n³) inversions with one factorization.
    //
    // K = Pxy * S^-1  →  K^T = S^-1 * Pxy^T  →  solve S * K_T = Pxy^T
    // State correction: K * z = K_T^T * z
    // Covariance update: P - K*S*K^T = P - Pxy * S^-1 * Pxy^T = P - Pxy * K_T
    auto t_kalman = Clock::now();
    Eigen::LDLT<Eigen::MatrixXd> const S_ldlt(innovation_cov);

    // Step 7: Compute innovation (observed - predicted)
    Eigen::VectorXd innovation = observed - measurement_mean;

    // Step 8: K_T = S^-1 * Pxy^T; state_correction = K * innovation = K_T^T * innovation
    Eigen::MatrixXd const K_T = S_ldlt.solve(cross_cov.transpose());
    Eigen::VectorXd const state_correction = K_T.transpose() * innovation;

    state_ = sigma_gen_.apply_error_to_state(state_, state_correction);

    // Step 8b: Enforce joint limits and zero velocities for constrained joints
    State prev_state = state_;  // Save state before limit enforcement
    enforce_joint_limits();
    result.kalman_ms = Ms(Clock::now() - t_kalman).count();

    // Step 9: Covariance update P' = P - Pxy * K_T  (equivalent to P - K*S*K^T).
    auto t_cov = Clock::now();
    covariance_.noalias() -= cross_cov * K_T;

    // Enforce symmetry (floating-point arithmetic can break it slightly)
    covariance_ = 0.5 * (covariance_ + covariance_.transpose());

    // Step 10: Project covariance to nearest PSD matrix by clipping negative eigenvalues.
    //
    // Fast path: attempt Cholesky (LLT). After a few frames the covariance converges and
    // stays positive-definite, so LLT almost always succeeds — O(n³/3) ≈ 1ms vs 24ms for
    // SelfAdjointEigenSolver. Only fall through to the eigensolver when LLT fails (i.e.,
    // when a direction genuinely went negative).
    {
        Eigen::LLT<Eigen::MatrixXd> llt_check(covariance_);
        if (llt_check.info() != Eigen::Success) {
            ++psd_fix_count_;
            // Covariance is not PD — clip negative eigenvalues to floor.
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(covariance_);
            if (eigen_solver.info() != Eigen::Success) {
                throw std::runtime_error(
                    "Failed to compute eigenvalues for covariance conditioning");
            }
            const double min_eig_floor = 1e-6;
            Eigen::VectorXd eigenvalues = eigen_solver.eigenvalues().cwiseMax(min_eig_floor);
            covariance_ = eigen_solver.eigenvectors() * eigenvalues.asDiagonal() *
                          eigen_solver.eigenvectors().transpose();
            covariance_ = 0.5 * (covariance_ + covariance_.transpose());
        }
    }

    // Step 11: Damp velocity covariance for joints that hit limits
    damp_velocity_covariance_at_limits(prev_state, state_);
    result.cov_update_ms = Ms(Clock::now() - t_cov).count();

    // Step 12: NIS = innovation^T * S^-1 * innovation — reuse LDLT factorization (vector solve)
    double nis = 0.0;
    if (S_ldlt.info() == Eigen::Success) {
        nis = innovation.dot(S_ldlt.solve(innovation));
    }

    // Fill in result
    result.num_observations = static_cast<int>(observations.size());
    result.num_inliers = static_cast<int>(inlier_observations.size());
    result.num_outliers = result.num_observations - result.num_inliers;
    result.observations = observation_results;
    result.nis = nis;
    result.nis_dof = static_cast<int>(innovation.size());

    // Debug: Export posterior covariance if debug mode enabled
    if (debug_enabled_) {
        write_matrix_csv(covariance_, "posterior_covariance.csv");
    }

    return result;
}

Eigen::VectorXd UnscentedKalmanFilter::predict_measurements(
    State const& state, std::vector<Observation> const& observations,
    std::unordered_map<int, Camera> const& cameras, ForwardKinematics& fk,
    std::unordered_map<int, std::unordered_map<int, Eigen::Vector2d>> const& prev_projections)
    const {
    // Compute forward kinematics to get marker positions
    auto marker_positions = fk.compute(state);

    // Project each marker to its camera
    int const n_obs = static_cast<int>(observations.size());
    Eigen::VectorXd predictions(2 * n_obs);

    int nan_count = 0;  // Debug: count NaN projections

    for (int i = 0; i < n_obs; ++i) {
        Observation const& obs = observations[i];

        // Get marker position (3D world)
        auto const& marker = layout_->skeleton()->markers()[obs.marker_id];
        std::string const& marker_name = marker.name;

        auto it = marker_positions.find(marker_name);
        if (it == marker_positions.end()) {
            // Marker not found in FK result (FK failed) - use NaN to mark as failed
            predictions(2 * i) = std::numeric_limits<double>::quiet_NaN();
            predictions(2 * i + 1) = std::numeric_limits<double>::quiet_NaN();
            nan_count++;
            continue;
        }

        Eigen::Vector3d const& marker_pos_world = it->second;

        // Project to camera (undistorted coordinates for UKF).
        // clip_to_bounds=false: allow out-of-bounds projections so that sigma points
        // which move a joint outside the camera view still produce a real pixel value
        // and contribute non-zero cross-covariance. Without this, out-of-bounds sigma
        // points get replaced by the measurement mean, zeroing the cross-covariance and
        // making the corresponding state dimensions (e.g. root_pos_z) unobservable.
        Camera const& camera = cameras.at(obs.camera_id);
        auto projected_opt = camera.project_undistorted(marker_pos_world, /*clip_to_bounds=*/false);

        // if (marker.name == "MRK-hip.R" && debug_enabled_) {
        //     std::cout << "  [DEBUG] predict_measurements for marker '" << marker.name
        //               << " on camera " << camera.name() << " world pos (" << marker_pos_world.x()
        //               << ", " << marker_pos_world.y() << ", " << marker_pos_world.z()
        //               << ") projected to (";
        //     if (projected_opt.has_value()) {
        //         std::cout << projected_opt->x() << ", " << projected_opt->y();
        //     } else {
        //         std::cout << "NaN";
        //     }
        //     std::cout << ")\n";
        // }

        // Check if projection succeeded
        if (!projected_opt.has_value()) {
            predictions(2 * i) = std::numeric_limits<double>::quiet_NaN();
            predictions(2 * i + 1) = std::numeric_limits<double>::quiet_NaN();
            nan_count++;
        } else {
            Eigen::Vector2d const& projected = *projected_opt;
            predictions(2 * i) = projected.x();
            predictions(2 * i + 1) = projected.y();

            // For velocity-mode observations, subtract the previous-frame projection.
            // If the prev projection is missing (occluded last frame) the prediction becomes NaN
            // and the observation is treated as an outlier for this frame.
            if (obs.mode == MeasurementMode::VELOCITY) {
                auto cam_it = prev_projections.find(obs.camera_id);
                if (cam_it != prev_projections.end()) {
                    auto marker_it = cam_it->second.find(obs.marker_id);
                    if (marker_it != cam_it->second.end()) {
                        predictions(2 * i) -= marker_it->second.x();
                        predictions(2 * i + 1) -= marker_it->second.y();
                    } else {
                        predictions(2 * i) = std::numeric_limits<double>::quiet_NaN();
                        predictions(2 * i + 1) = std::numeric_limits<double>::quiet_NaN();
                        nan_count++;
                    }
                } else {
                    predictions(2 * i) = std::numeric_limits<double>::quiet_NaN();
                    predictions(2 * i + 1) = std::numeric_limits<double>::quiet_NaN();
                    nan_count++;
                }
            }
        }
    }

    // Debug: log NaN count for first sigma point (mean state)
    static int call_count = 0;
    if (debug_enabled_ && frame_number_ == 0 && call_count == 0) {
        std::cout << "  [DEBUG] predict_measurements: " << nan_count << " NaN projections out of "
                  << n_obs << "\n";
    }
    call_count++;

    return predictions;
}

Eigen::VectorXd
UnscentedKalmanFilter::observations_to_vector(std::vector<Observation> const& observations) const {
    int const n_obs = static_cast<int>(observations.size());
    Eigen::VectorXd measurements(2 * n_obs);

    for (int i = 0; i < n_obs; ++i) {
        if (observations[i].mode == MeasurementMode::VELOCITY) {
            measurements(2 * i) = observations[i].position.x() - observations[i].prev_position.x();
            measurements(2 * i + 1) =
                observations[i].position.y() - observations[i].prev_position.y();
        } else {
            measurements(2 * i) = observations[i].position.x();
            measurements(2 * i + 1) = observations[i].position.y();
        }
    }

    return measurements;
}

double
UnscentedKalmanFilter::compute_mahalanobis_distance(Eigen::Vector2d const& innovation,
                                                    Eigen::Matrix2d const& covariance) const {
    // Mahalanobis distance: sqrt(innovation^T * cov^-1 * innovation)
    Eigen::Matrix2d cov_inv = covariance.inverse();
    double distance_squared = innovation.transpose() * cov_inv * innovation;
    return std::sqrt(distance_squared);
}

std::pair<std::vector<Observation>, std::vector<ObservationResult>>
UnscentedKalmanFilter::reject_outliers(std::vector<Observation> const& observations,
                                       Eigen::MatrixXd const& predicted_measurements,
                                       Eigen::VectorXd const& measurement_mean,
                                       Eigen::MatrixXd const& innovation_cov,
                                       double threshold) const {
    std::vector<Observation> inliers;
    std::vector<ObservationResult> results;

    Eigen::VectorXd observed = observations_to_vector(observations);

    // First pass: compute Mahalanobis distances and validity for all observations
    struct ObsData {
        bool is_valid;
        double mahalanobis_distance;
        Eigen::Vector2d innovation;
        Eigen::Vector2d predicted;
        Eigen::Vector2d actual;
    };

    std::vector<ObsData> obs_data(observations.size());

    for (size_t i = 0; i < observations.size(); ++i) {
        Observation const& obs = observations[i];
        ObsData& data = obs_data[i];

        // Extract predicted and actual measurements for this observation
        data.predicted = measurement_mean.segment<2>(2 * i);
        data.actual = observed.segment<2>(2 * i);

        // Check: if mean prediction is NaN (all sigma points failed), mark as invalid
        if (!std::isfinite(data.predicted.x()) || !std::isfinite(data.predicted.y())) {
            data.is_valid = false;
            data.mahalanobis_distance = 0.0;
            data.innovation = Eigen::Vector2d::Zero();
            continue;
        }

        // Check if ALL sigma points failed projection
        bool all_sigma_points_nan = true;
        for (int sigma_idx = 0; sigma_idx < predicted_measurements.cols(); ++sigma_idx) {
            double u = predicted_measurements(2 * i, sigma_idx);
            double v = predicted_measurements(2 * i + 1, sigma_idx);
            if (std::isfinite(u) && std::isfinite(v)) {
                all_sigma_points_nan = false;
                break;
            }
        }

        if (all_sigma_points_nan) {
            data.is_valid = false;
            data.mahalanobis_distance = 0.0;
            data.innovation = Eigen::Vector2d::Zero();
            continue;
        }

        // Extract 2x2 covariance for this observation
        Eigen::Matrix2d cov_2x2 = innovation_cov.block<2, 2>(2 * i, 2 * i);

        // Compute innovation
        data.innovation = data.actual - data.predicted;

        // Compute Mahalanobis distance
        data.mahalanobis_distance = compute_mahalanobis_distance(data.innovation, cov_2x2);
        data.is_valid = true;
    }

    // Second pass: cross-camera outlier detection
    // Group valid observations by marker_id
    std::map<int, std::vector<size_t>> marker_to_obs_indices;
    for (size_t i = 0; i < observations.size(); ++i) {
        if (obs_data[i].is_valid) {
            marker_to_obs_indices[observations[i].marker_id].push_back(i);
        }
    }

    // For each marker with multiple valid observations, apply cross-camera consistency check
    std::vector<bool> is_cross_camera_outlier(observations.size(), false);
    double const cross_camera_multiplier = 3.0;  // Tunable: reject if distance > k * median
    double const min_median_threshold = 0.5;     // Skip cross-camera check if median is too small

    for (auto const& [marker_id, obs_indices] : marker_to_obs_indices) {
        if (obs_indices.size() < 2) {
            continue;  // Need at least 2 cameras to compare
        }

        // Collect Mahalanobis distances for this marker across cameras
        std::vector<double> distances;
        for (size_t idx : obs_indices) {
            distances.push_back(obs_data[idx].mahalanobis_distance);
        }

        // Compute median distance
        std::vector<double> sorted_distances = distances;
        std::sort(sorted_distances.begin(), sorted_distances.end());
        double median;
        size_t n = sorted_distances.size();
        if (n % 2 == 0) {
            median = (sorted_distances[n / 2 - 1] + sorted_distances[n / 2]) / 2.0;
        } else {
            median = sorted_distances[n / 2];
        }

        // Apply cross-camera threshold: reject if distance > multiplier * median
        // Only apply if median is above minimum threshold (avoids rejecting everything when all are
        // good)
        if (median > min_median_threshold) {
            double cross_camera_threshold = cross_camera_multiplier * median;

            for (size_t idx : obs_indices) {
                if (obs_data[idx].mahalanobis_distance > cross_camera_threshold) {
                    is_cross_camera_outlier[idx] = true;
                }
            }
        }
    }

    // Third pass: combine standard threshold and cross-camera check to generate final results
    for (size_t i = 0; i < observations.size(); ++i) {
        Observation const& obs = observations[i];
        ObsData const& data = obs_data[i];

        ObservationResult obs_result;
        obs_result.marker_name = layout_->skeleton()->markers()[obs.marker_id].name;
        obs_result.camera_id = obs.camera_id;
        obs_result.camera_frame_idx = obs.frame_idx;
        obs_result.predicted = data.predicted;
        obs_result.actual = data.actual;
        obs_result.innovation = data.innovation;
        obs_result.mahalanobis_distance = data.mahalanobis_distance;

        if (!data.is_valid) {
            // Invalid observation (NaN projection)
            obs_result.is_outlier = true;
        } else {
            // Valid observation - check both standard and cross-camera thresholds
            bool standard_outlier = data.mahalanobis_distance > threshold;
            bool cross_camera_outlier = is_cross_camera_outlier[i];

            obs_result.is_outlier = standard_outlier || cross_camera_outlier;

            // Keep observation only if it passes both checks
            if (!obs_result.is_outlier) {
                inliers.push_back(obs);
            }
        }

        results.push_back(obs_result);
    }

    return {inliers, results};
}

std::vector<ObservationResult> UnscentedKalmanFilter::compute_observation_diagnostics(
    std::vector<Observation> const& observations, Eigen::VectorXd const& measurement_mean,
    Eigen::MatrixXd const& innovation_cov) const {
    std::vector<ObservationResult> results;

    Eigen::VectorXd observed = observations_to_vector(observations);

    for (size_t i = 0; i < observations.size(); ++i) {
        Observation const& obs = observations[i];

        // Extract predicted and actual measurements
        Eigen::Vector2d predicted = measurement_mean.segment<2>(2 * i);
        Eigen::Vector2d actual = observed.segment<2>(2 * i);

        // Check for NaN in predicted (failed projection)
        if (!std::isfinite(predicted.x()) || !std::isfinite(predicted.y())) {
            ObservationResult obs_result;
            obs_result.marker_name = layout_->skeleton()->markers()[obs.marker_id].name;
            obs_result.camera_id = obs.camera_id;
            obs_result.camera_frame_idx = obs.frame_idx;
            obs_result.is_outlier = true;  // Mark as outlier for diagnostics
            obs_result.mahalanobis_distance = 0.0;
            obs_result.innovation = Eigen::Vector2d::Zero();
            obs_result.predicted = predicted;
            obs_result.actual = actual;
            results.push_back(obs_result);
            continue;
        }

        // Extract 2x2 covariance for this observation
        Eigen::Matrix2d cov_2x2 = innovation_cov.block<2, 2>(2 * i, 2 * i);

        // Compute innovation
        Eigen::Vector2d innovation = actual - predicted;

        // Compute Mahalanobis distance
        double mahal_dist = compute_mahalanobis_distance(innovation, cov_2x2);

        // Create result (not an outlier - just diagnostics)
        ObservationResult obs_result;
        obs_result.marker_name = layout_->skeleton()->markers()[obs.marker_id].name;
        obs_result.camera_id = obs.camera_id;
        obs_result.camera_frame_idx = obs.frame_idx;
        obs_result.is_outlier = false;
        obs_result.mahalanobis_distance = mahal_dist;
        obs_result.innovation = innovation;
        obs_result.predicted = predicted;
        obs_result.actual = actual;
        results.push_back(obs_result);
    }

    return results;
}

void UnscentedKalmanFilter::enforce_joint_limits() {
    // Clamp joint angles to their limits, then zero velocities for joints at limits.
    //
    // IMPORTANT: iterate layout_->joints() (active-group joints in state-vector order)
    // rather than skeleton()->joints() (all joints in YAML order).  Using skeleton
    // joints with a running counter shifts every index after any non-active-group joint
    // (e.g. heel.02.L/R), causing limits to be applied to wrong state slots and leaving
    // the actual joint angles unclamped.  The same class of bug was fixed in
    // TrackingExporter::write_frame (March 2026).
    Eigen::VectorXd angles = state_.joint_angles();
    Eigen::VectorXd velocities = state_.joint_velocities();

    for (auto const& desc : layout_->joints()) {
        if (desc.is_scale_follower)
            continue;

        uint32_t ai = desc.state_index;  // angle index in state vector
        uint32_t vi = desc.state_index;  // velocity index (same layout for angles + vels)

        if (desc.type == JointType::REVOLUTE || desc.type == JointType::PRISMATIC) {
            if (static_cast<size_t>(desc.limit_count) > 0 &&
                ai < static_cast<uint32_t>(angles.size())) {
                double lo = desc.limits[0].x(), hi = desc.limits[0].y();
                angles[ai] = std::clamp(angles[ai], lo, hi);
                if (vi < static_cast<uint32_t>(velocities.size())) {
                    if (std::abs(angles[ai] - lo) < 1e-6 || std::abs(angles[ai] - hi) < 1e-6) {
                        velocities[vi] = 0.0;
                    }
                }
            }

        } else if (desc.type == JointType::SPHERICAL) {
            auto active_mask = desc.active_dof_mask;
            if (ai + 2 < static_cast<uint32_t>(angles.size())) {
                for (int i = 0; i < 3; ++i) {
                    if (!active_mask[i]) {
                        angles[ai + i] =
                            (static_cast<size_t>(desc.limit_count) > static_cast<size_t>(i))
                                ? desc.limits[i].x()
                                : 0.0;
                        if (vi + i < static_cast<uint32_t>(velocities.size()))
                            velocities[vi + i] = 0.0;
                    } else if (static_cast<size_t>(desc.limit_count) > static_cast<size_t>(i)) {
                        double lo = desc.limits[i].x(), hi = desc.limits[i].y();
                        angles[ai + i] = std::clamp(angles[ai + i], lo, hi);
                        if (vi + i < static_cast<uint32_t>(velocities.size())) {
                            if (std::abs(angles[ai + i] - lo) < 1e-6 ||
                                std::abs(angles[ai + i] - hi) < 1e-6) {
                                velocities[vi + i] = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }

    state_.set_joint_angles(angles);
    state_.set_joint_velocities(velocities);
}

void UnscentedKalmanFilter::damp_velocity_covariance_at_limits(State const& prev_state,
                                                               State const& current_state,
                                                               double damping_factor) {
    // Compare velocities before and after limit enforcement
    // Damp covariance for velocities that changed

    Eigen::VectorXd const& prev_velocities = prev_state.joint_velocities();
    Eigen::VectorXd const& curr_velocities = current_state.joint_velocities();

    if (prev_velocities.size() != curr_velocities.size()) {
        return;
    }

    // Find velocity indices that were modified
    int const error_pos_dim = error_dim() / 2;

    // Check root velocities (always first 6 in velocity state)
    Eigen::Vector3d prev_root_vel = prev_state.root_velocity();
    Eigen::Vector3d curr_root_vel = current_state.root_velocity();
    Eigen::Vector3d prev_root_angvel = prev_state.root_angular_velocity();
    Eigen::Vector3d curr_root_angvel = current_state.root_angular_velocity();

    for (int i = 0; i < 3; ++i) {
        if (std::abs(prev_root_vel(i) - curr_root_vel(i)) > 1e-9) {
            int vel_idx = error_pos_dim + i;
            covariance_.row(vel_idx) *= damping_factor;
            covariance_.col(vel_idx) *= damping_factor;
            covariance_(vel_idx, vel_idx) = std::max(covariance_(vel_idx, vel_idx), 1e-8);
        }
        if (std::abs(prev_root_angvel(i) - curr_root_angvel(i)) > 1e-9) {
            int vel_idx = error_pos_dim + 3 + i;
            covariance_.row(vel_idx) *= damping_factor;
            covariance_.col(vel_idx) *= damping_factor;
            covariance_(vel_idx, vel_idx) = std::max(covariance_(vel_idx, vel_idx), 1e-8);
        }
    }

    // Check joint velocities
    for (int i = 0; i < prev_velocities.size(); ++i) {
        if (std::abs(prev_velocities(i) - curr_velocities(i)) > 1e-9) {
            int vel_idx = error_pos_dim + 6 + i;
            if (vel_idx < error_dim()) {
                covariance_.row(vel_idx) *= damping_factor;
                covariance_.col(vel_idx) *= damping_factor;
                covariance_(vel_idx, vel_idx) = std::max(covariance_(vel_idx, vel_idx), 1e-8);
            }
        }
    }
}

void UnscentedKalmanFilter::enable_debug(bool enable, std::string const& debug_dir) {
    debug_enabled_ = enable;
    debug_dir_ = debug_dir;
    if (enable) {
        std::filesystem::create_directories(debug_dir);
        std::cout << "DEBUG: UKF debug mode enabled, writing to " << debug_dir << "\n";
    }
}

void UnscentedKalmanFilter::write_matrix_csv(Eigen::MatrixXd const& matrix,
                                             std::string const& filename) const {
    // Create frame directory
    std::string frame_dir =
        debug_dir_ + "/frame_" +
        std::string(4 - std::min(4, static_cast<int>(std::to_string(frame_number_).length())),
                    '0') +
        std::to_string(frame_number_);
    std::filesystem::create_directories(frame_dir);

    std::string filepath = frame_dir + "/" + filename;
    std::ofstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "Failed to open " << filepath << " for writing\n";
        return;
    }

    f << std::setprecision(15);

    // Write matrix rows
    for (int row = 0; row < matrix.rows(); ++row) {
        for (int col = 0; col < matrix.cols(); ++col) {
            if (col > 0) {
                f << ",";
            }
            f << matrix(row, col);
        }
        f << "\n";
    }
}

void UnscentedKalmanFilter::write_sigma_points_csv(std::vector<State> const& sigma_points) const {
    // Create frame directory
    std::string frame_dir =
        debug_dir_ + "/frame_" +
        std::string(4 - std::min(4, static_cast<int>(std::to_string(frame_number_).length())),
                    '0') +
        std::to_string(frame_number_);
    std::filesystem::create_directories(frame_dir);

    std::string filepath = frame_dir + "/sigma_points.csv";
    std::ofstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "Failed to open " << filepath << " for writing\n";
        return;
    }

    f << std::setprecision(15);

    // Write header
    f << "sigma_idx,root_pos_x,root_pos_y,root_pos_z,"
      << "root_quat_w,root_quat_x,root_quat_y,root_quat_z,"
      << "root_vel_x,root_vel_y,root_vel_z,"
      << "root_angvel_x,root_angvel_y,root_angvel_z";

    // Add joint angle columns (in skeleton order). Skip scale-group followers
    // (they share the leader's state slot; no independent column in the state vector).
    auto const joints_ordered = layout_->skeleton()->get_joints_ordered();
    for (auto const& joint : joints_ordered) {
        if (!joint.parent_index.has_value())
            continue;  // Skip root
        if (joint.is_scale_follower)
            continue;  // Shares leader's slot

        int dof = 0;
        if (joint.type == JointType::REVOLUTE || joint.type == JointType::PRISMATIC) {
            dof = 1;
        } else if (joint.type == JointType::SPHERICAL) {
            dof = 3;
        }

        for (int i = 0; i < dof; ++i) {
            f << "," << joint.name << "_angle_" << i;
        }
    }

    // Add joint velocity columns (in skeleton order)
    for (auto const& joint : joints_ordered) {
        if (!joint.parent_index.has_value())
            continue;  // Skip root
        if (joint.is_scale_follower)
            continue;  // Shares leader's slot

        int dof = 0;
        if (joint.type == JointType::REVOLUTE || joint.type == JointType::PRISMATIC) {
            dof = 1;
        } else if (joint.type == JointType::SPHERICAL) {
            dof = 3;
        }

        for (int i = 0; i < dof; ++i) {
            f << "," << joint.name << "_vel_" << i;
        }
    }
    f << "\n";

    // Write sigma point data
    for (size_t sigma_idx = 0; sigma_idx < sigma_points.size(); ++sigma_idx) {
        State const& state = sigma_points[sigma_idx];

        // Sigma index
        f << sigma_idx;

        // Root position
        f << "," << state.root_position().x() << "," << state.root_position().y() << ","
          << state.root_position().z();

        // Root quaternion (w,x,y,z)
        f << "," << state.root_orientation().w() << "," << state.root_orientation().x() << ","
          << state.root_orientation().y() << "," << state.root_orientation().z();

        // Root velocity
        f << "," << state.root_velocity().x() << "," << state.root_velocity().y() << ","
          << state.root_velocity().z();

        // Root angular velocity
        f << "," << state.root_angular_velocity().x() << "," << state.root_angular_velocity().y()
          << "," << state.root_angular_velocity().z();

        // Joint angles (in skeleton order)
        int joint_angle_idx = 0;
        for (auto const& joint : joints_ordered) {
            if (!joint.parent_index.has_value())
                continue;  // Skip root
            if (joint.is_scale_follower)
                continue;  // Shares leader's slot

            if (joint.type == JointType::REVOLUTE || joint.type == JointType::PRISMATIC) {
                f << "," << state.joint_angles()(joint_angle_idx);
                joint_angle_idx += 1;
            } else if (joint.type == JointType::SPHERICAL) {
                // Ball joint: 3 DOF (axis-angle representation in joint_angles)
                f << "," << state.joint_angles()(joint_angle_idx) << ","
                  << state.joint_angles()(joint_angle_idx + 1) << ","
                  << state.joint_angles()(joint_angle_idx + 2);
                joint_angle_idx += 3;
            }
        }

        // Joint velocities (in skeleton order)
        int joint_vel_idx = 0;
        for (auto const& joint : joints_ordered) {
            if (!joint.parent_index.has_value())
                continue;  // Skip root
            if (joint.is_scale_follower)
                continue;  // Shares leader's slot

            if (joint.type == JointType::REVOLUTE || joint.type == JointType::PRISMATIC) {
                f << "," << state.joint_velocities()(joint_vel_idx);
                joint_vel_idx += 1;
            } else if (joint.type == JointType::SPHERICAL) {
                // Ball joint: 3 DOF velocities
                f << "," << state.joint_velocities()(joint_vel_idx) << ","
                  << state.joint_velocities()(joint_vel_idx + 1) << ","
                  << state.joint_velocities()(joint_vel_idx + 2);
                joint_vel_idx += 3;
            }
        }

        f << "\n";
    }
}

}  // namespace posetrak

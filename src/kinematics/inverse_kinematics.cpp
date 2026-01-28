/**
 * @file inverse_kinematics.cpp
 * @brief Implementation of damped least squares IK solver
 */

#include "posetrak/kinematics/inverse_kinematics.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <iostream>

namespace posetrak {

InverseKinematics::InverseKinematics(
    pinocchio::Model const& model, pinocchio::Data& data, ForwardKinematics const& fk,
    std::map<std::string, pinocchio::FrameIndex> const& marker_frame_map)
    : model_(model), data_(data), fk_(fk), marker_frame_map_(marker_frame_map) {}

IKResult InverseKinematics::solve(std::map<std::string, Eigen::Vector3d> const& target_markers,
                                  Skeleton const& skeleton,
                                  std::optional<State> const& initial_guess, int max_iterations,
                                  double tolerance, double damping) {
    // Convert initial guess to configuration vector
    Eigen::VectorXd q;
    if (!initial_guess.has_value() || initial_guess->joint_angles().size() == 0) {
        // Default: zero configuration
        q = Eigen::VectorXd::Zero(model_.nq);
        // Set root quaternion to identity [x,y,z,w] = [0,0,0,1]
        if (model_.nq >= 7) {  // Has root (free-flyer)
            q[6] = 1.0;        // w component
        }
    } else {
        q = ForwardKinematics::state_to_config(*initial_guess, skeleton);
    }

    // Extract marker names (maintain consistent ordering)
    std::vector<std::string> marker_names;
    for (auto const& [name, pos] : target_markers) {
        if (marker_frame_map_.count(name) > 0) {
            marker_names.push_back(name);
        }
    }

    if (marker_names.empty()) {
        return IKResult::failure();
    }

    // Damped least squares iteration
    double prev_error = std::numeric_limits<double>::infinity();
    int iter = 0;

    for (; iter < max_iterations; ++iter) {
        // Compute current error
        Eigen::VectorXd error = compute_error(q, target_markers);
        double rms_error = error.norm() / std::sqrt(marker_names.size());

        // Check convergence
        if (rms_error < tolerance) {
            // Convert back to State
            // TODO: Implement config_to_state properly
            // For now, create minimal state
            int num_joints = skeleton.joints().size() - 1;  // Exclude root
            State final_state(num_joints > 0 ? num_joints : 0);
            return IKResult{final_state, rms_error, iter + 1, true};
        }

        // Check for divergence (but allow first few iterations to have high error)
        if (iter > 3 && rms_error > prev_error * 1.5) {
            // Diverging - stop
            break;
        }
        prev_error = rms_error;

        // Compute Jacobian
        Eigen::MatrixXd J = compute_jacobian(q, marker_names);

        // Damped least squares: Δq = J^T(JJ^T + λI)^(-1) * error
        Eigen::MatrixXd JJT = J * J.transpose();
        Eigen::MatrixXd damped = JJT + damping * Eigen::MatrixXd::Identity(JJT.rows(), JJT.cols());

        // Solve: damped * y = error, then Δq = J^T * y
        Eigen::VectorXd y = damped.ldlt().solve(error);
        Eigen::VectorXd delta_q = J.transpose() * y;

        // Scale step to avoid too large updates
        double max_step = 0.3;  // Max 0.3 rad or 0.3m per iteration
        double delta_norm = delta_q.norm();
        if (delta_norm > max_step) {
            delta_q *= max_step / delta_norm;
        }

        // Update configuration
        // Root position (indices 0-2)
        q.head(3) += delta_q.head(3);

        if (model_.nq >= 7) {
            // Root quaternion (indices 3-6): Simple integration for now
            // Extract quaternion update from delta_q (which is in velocity space, nv)
            // For free-flyer, the velocity has 6 DOF: 3 linear + 3 angular
            Eigen::Vector3d omega = delta_q.segment<3>(3);  // Angular velocity

            // Convert angular velocity to quaternion update (small angle approximation)
            Eigen::Quaterniond q_current(q[6], q[3], q[4], q[5]);  // [w, x, y, z]
            Eigen::Quaterniond q_delta;
            double angle = omega.norm();
            if (angle > 1e-8) {
                Eigen::AngleAxisd aa(angle, omega.normalized());
                q_delta = Eigen::Quaterniond(aa);
            } else {
                q_delta = Eigen::Quaterniond::Identity();
            }

            Eigen::Quaterniond q_new = q_delta * q_current;
            q_new.normalize();

            // Store back [x, y, z, w]
            q[3] = q_new.x();
            q[4] = q_new.y();
            q[5] = q_new.z();
            q[6] = q_new.w();
        }

        // Update other DOFs (revolute joints)
        // Velocity space for free-flyer is 6 DOF, so joint velocities start at index 6
        if (model_.nv > 6 && model_.nq > 7) {
            int joint_dof = model_.nv - 6;
            q.tail(model_.nq - 7) += delta_q.tail(joint_dof);
        }

        // Enforce joint limits
        enforce_joint_limits(q, skeleton);
    }

    // Failed to converge
    Eigen::VectorXd final_error = compute_error(q, target_markers);
    double rms_error = final_error.norm() / std::sqrt(marker_names.size());

    int num_joints = skeleton.joints().size() - 1;  // Exclude root
    State final_state(num_joints > 0 ? num_joints : 0);
    return IKResult{final_state, rms_error, iter, false};
}

Eigen::VectorXd
InverseKinematics::compute_error(Eigen::VectorXd const& q,
                                 std::map<std::string, Eigen::Vector3d> const& target_markers) {
    // Compute FK
    auto current_markers = fk_.compute(q);

    // Build error vector (3 * num_markers)
    std::vector<double> errors;
    errors.reserve(target_markers.size() * 3);

    for (auto const& [name, target_pos] : target_markers) {
        if (marker_frame_map_.count(name) == 0) {
            continue;  // Skip markers not in model
        }

        auto it = current_markers.find(name);
        if (it == current_markers.end()) {
            continue;  // Skip if FK didn't compute this marker
        }

        Eigen::Vector3d const& current_pos = it->second;
        Eigen::Vector3d error = target_pos - current_pos;

        errors.push_back(error.x());
        errors.push_back(error.y());
        errors.push_back(error.z());
    }

    return Eigen::Map<Eigen::VectorXd>(errors.data(), errors.size());
}

Eigen::MatrixXd InverseKinematics::compute_jacobian(Eigen::VectorXd const& q,
                                                    std::vector<std::string> const& marker_names) {
    // Update kinematics for current q
    pinocchio::forwardKinematics(model_, data_, q);
    pinocchio::updateFramePlacements(model_, data_);

    // Allocate stacked Jacobian (3 * num_markers × nv)
    int num_markers = marker_names.size();
    Eigen::MatrixXd J_stacked = Eigen::MatrixXd::Zero(3 * num_markers, model_.nv);

    // Compute Jacobian for each marker
    for (size_t i = 0; i < marker_names.size(); ++i) {
        std::string const& name = marker_names[i];
        auto it = marker_frame_map_.find(name);
        if (it == marker_frame_map_.end()) {
            continue;
        }

        pinocchio::FrameIndex frame_id = it->second;

        // Compute 6D Jacobian (spatial velocity) in WORLD frame
        Eigen::Matrix<double, 6, Eigen::Dynamic> J_frame(6, model_.nv);
        J_frame.setZero();

        pinocchio::computeFrameJacobian(model_, data_, q, frame_id, pinocchio::LOCAL_WORLD_ALIGNED,
                                        J_frame);

        // Extract linear velocity part (first 3 rows)
        // Pinocchio stores [linear; angular] in 6D Jacobian
        J_stacked.block(3 * i, 0, 3, model_.nv) = J_frame.topRows(3);
    }

    return J_stacked;
}

void InverseKinematics::enforce_joint_limits(Eigen::VectorXd& q, Skeleton const& skeleton) {
    // Root position - no limits for now
    // Root quaternion - should be normalized but skip for now

    // Joint angles
    if (model_.nq <= 7) {
        return;  // Only root, no joints to limit
    }

    // Start after root (7 DOF)
    int q_idx = 7;

    for (auto const& joint : skeleton.joints()) {
        if (joint.parent_index == std::nullopt) {
            continue;  // Root joint
        }

        if (joint.type == JointType::REVOLUTE) {
            // Single DOF - apply limits
            if (q_idx < model_.nq && joint.num_limits > 0) {
                double min_limit = joint.limits[0].x();  // limits is array of Vector2d
                double max_limit = joint.limits[0].y();

                q[q_idx] = std::clamp(q[q_idx], min_limit, max_limit);
                q_idx++;
            } else if (q_idx < model_.nq) {
                // No limits specified
                q_idx++;
            }
        } else if (joint.type == JointType::SPHERICAL) {
            // 4 DOF quaternion - skip for now (would need proper quaternion clamping)
            q_idx += 4;
        }
    }
}

}  // namespace posetrak

/**
 * @file process_model.cpp
 * @brief Implementation of constant velocity process model
 */

#include "posetrak/filters/process_model.hpp"

#include <cmath>

namespace posetrak {

ConstantVelocityModel::ConstantVelocityModel(Skeleton const& skeleton, double process_noise_std)
    : skeleton_(skeleton), process_noise_std_(process_noise_std) {}

State ConstantVelocityModel::propagate(State const& state, double dt) const {
    // Create a mutable copy to modify
    State next_state = state;

    // 1. Root position: p' = p + v * dt
    Eigen::Vector3d new_pos = state.root_position() + state.root_velocity() * dt;
    next_state.set_root_position(new_pos);

    // 2. Root orientation: q' = q ⊗ exp(ω * dt / 2)
    // Use exponential map to integrate angular velocity
    Eigen::Vector3d const& angular_vel = state.root_angular_velocity();
    Eigen::Vector3d axis_angle = angular_vel * dt;
    Eigen::Quaterniond delta_q = State::axis_angle_to_quaternion(axis_angle);
    Eigen::Quaterniond new_orientation = (state.root_orientation() * delta_q).normalized();
    next_state.set_root_orientation(new_orientation);

    // 3. Joint angles: propagate based on joint type
    // - Revolute: simple addition θ' = θ + ω * dt
    // - Spherical: manifold composition using rotation matrices
    Eigen::VectorXd new_angles = state.joint_angles();
    auto joints_ordered = skeleton_.get_joints_ordered();

    int angle_idx = 0;  // Index in joint_angles vector
    int vel_idx = 0;    // Index in joint_velocities vector

    for (auto const& joint : joints_ordered) {
        // Skip root joint (handled above)
        if (!joint.parent_index.has_value()) {
            continue;
        }

        if (joint.type == JointType::REVOLUTE) {
            // Simple integration for revolute joints
            new_angles[angle_idx] += state.joint_velocities()[vel_idx] * dt;
            angle_idx++;
            vel_idx++;

        } else if (joint.type == JointType::SPHERICAL) {
            // Manifold integration for spherical joints (SO(3))
            // Check for locked DOFs
            auto active_mask = joint.get_active_dof_mask();
            int num_active = joint.active_dof();

            if (num_active == 3) {
                // All DOFs active - use full rotation composition
                // Current axis-angle representation
                Eigen::Vector3d current_axis_angle = state.joint_angles().segment<3>(angle_idx);
                Eigen::Vector3d angular_velocity = state.joint_velocities().segment<3>(vel_idx);

                // Convert current state to rotation matrix
                Eigen::Quaterniond current_q = State::axis_angle_to_quaternion(current_axis_angle);
                Eigen::Matrix3d R_current = current_q.toRotationMatrix();

                // Compute delta rotation from angular velocity
                Eigen::Vector3d delta_axis_angle = angular_velocity * dt;
                Eigen::Quaterniond delta_q_joint =
                    State::axis_angle_to_quaternion(delta_axis_angle);
                Eigen::Matrix3d R_delta = delta_q_joint.toRotationMatrix();

                // Compose: R_new = R_current * R_delta
                Eigen::Matrix3d R_new = R_current * R_delta;

                // Convert back to axis-angle
                Eigen::Quaterniond new_q(R_new);
                Eigen::Vector3d new_axis_angle = State::quaternion_to_axis_angle(new_q);

                new_angles.segment<3>(angle_idx) = new_axis_angle;
            } else {
                // Some DOFs locked - only propagate active ones using simple integration
                Eigen::Vector3d current_axis_angle = state.joint_angles().segment<3>(angle_idx);
                Eigen::Vector3d angular_velocity = state.joint_velocities().segment<3>(vel_idx);

                for (int i = 0; i < 3; ++i) {
                    if (active_mask[i]) {
                        // Active DOF: integrate
                        new_angles[angle_idx + i] =
                            current_axis_angle[i] + angular_velocity[i] * dt;
                    } else {
                        // Locked DOF: reset to fixed value (min limit)
                        if (joint.num_limits > static_cast<size_t>(i)) {
                            new_angles[angle_idx + i] = joint.limits[i].x();
                        }
                    }
                }
            }

            angle_idx += 3;
            vel_idx += 3;
        }
        // FIXED joints have 0 DOF, nothing to update
    }

    next_state.set_joint_angles(new_angles);

    // 4. Velocities remain constant (process noise added by UKF)
    // (already copied in next_state)

    // 5. Enforce joint limits
    enforce_joint_limits(next_state);

    return next_state;
}

Eigen::MatrixXd ConstantVelocityModel::get_process_noise(double dt, int state_dim) const {
    // Process noise covariance Q
    // For constant velocity model, noise is typically proportional to dt² for positions
    // and dt for velocities. We use a simplified model where all error-state dimensions
    // get the same noise scaled by dt².

    // Variance = (std * dt)²
    double variance = process_noise_std_ * process_noise_std_ * dt * dt;

    // Diagonal covariance (assumes independence between dimensions)
    return variance * Eigen::MatrixXd::Identity(state_dim, state_dim);
}

void ConstantVelocityModel::set_process_noise_std(double std_dev) {
    process_noise_std_ = std_dev;
}

void ConstantVelocityModel::enforce_joint_limits(State& state) const {
    auto joints_ordered = skeleton_.get_joints_ordered();

    int joint_angle_idx = 0;
    Eigen::VectorXd angles = state.joint_angles();  // Get mutable copy

    for (auto const& joint : joints_ordered) {
        // Skip root joint (no limits on root)
        if (!joint.parent_index.has_value()) {
            continue;
        }

        if (joint.type == JointType::REVOLUTE) {
            // Single DOF - enforce limits
            if (joint.num_limits > 0 && joint_angle_idx < angles.size()) {
                double min_limit = joint.limits[0].x();
                double max_limit = joint.limits[0].y();

                // Clamp angle to limits
                angles[joint_angle_idx] = std::clamp(angles[joint_angle_idx], min_limit, max_limit);
            }
            joint_angle_idx++;

        } else if (joint.type == JointType::SPHERICAL) {
            // Spherical joint: 3 DOF in error space
            // Limits are harder to enforce for spherical joints
            // For now, just skip (quaternion normalization is done elsewhere)
            joint_angle_idx += 3;
        }
        // FIXED joints have 0 DOF, no angles to enforce
    }

    state.set_joint_angles(angles);
}

}  // namespace posetrak

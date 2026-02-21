/**
 * @file process_model.cpp
 * @brief Implementation of constant velocity process model
 */

#include "posetrak/filters/process_model.hpp"

#include "posetrak/core/skeleton_layout.hpp"
#include <cmath>

namespace posetrak {

ConstantVelocityModel::ConstantVelocityModel(std::shared_ptr<const SkeletonLayout> layout,
                                             double process_noise_std)
    : layout_(std::move(layout)), process_noise_std_(process_noise_std) {}

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

    for (JointDesc const& j : layout_->joints()) {
        if (j.type == JointType::REVOLUTE) {
            // Simple integration for revolute joints
            new_angles[j.state_index] += state.joint_velocities()[j.state_index] * dt;

        } else if (j.type == JointType::SPHERICAL) {
            // Manifold integration for spherical joints (SO(3))
            // Always use 3 DOFs in state storage (locked DOFs enforced in limits)

            // Current axis-angle representation
            Eigen::Vector3d current_axis_angle = state.joint_angles().segment<3>(j.state_index);
            Eigen::Vector3d angular_velocity = state.joint_velocities().segment<3>(j.state_index);

            // Convert current state to rotation matrix
            Eigen::Quaterniond current_q = State::axis_angle_to_quaternion(current_axis_angle);
            Eigen::Matrix3d R_current = current_q.toRotationMatrix();

            // Compute delta rotation from angular velocity
            Eigen::Vector3d delta_axis_angle = angular_velocity * dt;
            Eigen::Quaterniond delta_q_joint = State::axis_angle_to_quaternion(delta_axis_angle);
            Eigen::Matrix3d R_delta = delta_q_joint.toRotationMatrix();

            // Compose: R_new = R_current * R_delta
            Eigen::Matrix3d R_new = R_current * R_delta;

            // Convert back to axis-angle
            Eigen::Quaterniond new_q(R_new);
            Eigen::Vector3d new_axis_angle = State::quaternion_to_axis_angle(new_q);

            new_angles.segment<3>(j.state_index) = new_axis_angle;
        }
        // FIXED joints have 0 DOF, nothing to update
    }

    next_state.set_joint_angles(new_angles);

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
    Eigen::VectorXd angles = state.joint_angles();  // Get mutable copy

    for (JointDesc const& j : layout_->joints()) {
        if (j.type == JointType::REVOLUTE) {
            // Single DOF — enforce limit if present
            if (j.limit_count > 0) {
                angles[j.state_index] =
                    std::clamp(angles[j.state_index], j.limits[0].x(), j.limits[0].y());
            }

        } else if (j.type == JointType::SPHERICAL) {
            // Spherical joint: always 3 DOFs in storage
            for (int i = 0; i < 3; ++i) {
                if (!j.active_dof_mask[i]) {
                    // Locked DOF: set to limit value (min == max)
                    angles[j.state_index + i] = (j.limit_count > i) ? j.limits[i].x() : 0.0;
                } else if (j.limit_count > i) {
                    // Active DOF with limits: clamp to range
                    angles[j.state_index + i] =
                        std::clamp(angles[j.state_index + i], j.limits[i].x(), j.limits[i].y());
                }
            }
        }
        // FIXED joints have 0 DOF, no angles to enforce
    }

    state.set_joint_angles(angles);
}

}  // namespace posetrak

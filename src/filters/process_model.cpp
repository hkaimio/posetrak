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

    // 2. Root orientation: For now, keep constant (no angular velocity in State)
    // TODO: When State includes root angular velocity, implement:
    // q' = q ⊗ exp(ω * dt / 2)
    // For now, orientation remains unchanged

    // 3. Joint angles: θ' = θ + ω * dt
    Eigen::VectorXd new_angles = state.joint_angles() + state.joint_velocities() * dt;
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

/**
 * @file sigma_points.cpp
 * @brief Implementation of sigma point generation
 */

#include "posetrak/filters/sigma_points.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <stdexcept>

namespace posetrak {

SigmaPointGenerator::SigmaPointGenerator(Skeleton const& skeleton, double alpha, double beta,
                                         double kappa)
    : skeleton_(skeleton),
      error_dim_(2 * (6 + skeleton.active_dof())),  // 2*(3 pos + 3 rot + ndof)
      alpha_(alpha),
      beta_(beta),
      kappa_(kappa) {
    // Compute lambda parameter
    int const n = error_dim_;
    double const lambda = alpha * alpha * (n + kappa) - n;

    // Compute weights for mean
    wm_.resize(2 * n + 1);
    wm_(0) = lambda / (n + lambda);
    for (int i = 1; i <= 2 * n; ++i) {
        wm_(i) = 0.5 / (n + lambda);
    }

    // Weights for covariance
    wc_ = wm_;
    wc_(0) += (1.0 - alpha * alpha + beta);

    // Scaling factor for sigma points
    gamma_ = std::sqrt(n + lambda);
}

std::vector<State>
SigmaPointGenerator::generate_sigma_points(State const& nominal_state,
                                           Eigen::MatrixXd const& covariance) const {
    int const n = error_dim_;

    // Validate covariance dimension
    if (covariance.rows() != n || covariance.cols() != n) {
        throw std::invalid_argument("Covariance matrix size must match error dimension");
    }

    // Compute matrix square root using Cholesky decomposition
    Eigen::MatrixXd L;
    Eigen::LLT<Eigen::MatrixXd> llt(covariance);
    if (llt.info() == Eigen::Success) {
        L = llt.matrixL();
    } else {
        // If Cholesky fails, use eigenvalue decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(covariance);
        if (eigensolver.info() != Eigen::Success) {
            throw std::runtime_error("Failed to compute matrix square root");
        }

        // Ensure positive eigenvalues
        Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
        for (int i = 0; i < eigenvalues.size(); ++i) {
            eigenvalues(i) = std::max(eigenvalues(i), 1e-10);
        }

        L = eigensolver.eigenvectors() * eigenvalues.cwiseSqrt().asDiagonal();
    }

    // Generate error vectors in error space
    std::vector<Eigen::VectorXd> error_vectors;
    error_vectors.reserve(2 * n + 1);

    // Central sigma point (zero error)
    error_vectors.push_back(Eigen::VectorXd::Zero(n));

    // Positive sigma points (indices 1 to n)
    for (int i = 0; i < n; ++i) {
        error_vectors.push_back(gamma_ * L.col(i));
    }

    // Negative sigma points (indices n+1 to 2n)
    for (int i = 0; i < n; ++i) {
        error_vectors.push_back(-gamma_ * L.col(i));
    }

    // Convert error vectors to state space
    std::vector<State> sigma_states;
    sigma_states.reserve(2 * n + 1);

    for (auto const& error_vec : error_vectors) {
        sigma_states.push_back(apply_error_to_state(nominal_state, error_vec));
    }

    return sigma_states;
}

State SigmaPointGenerator::apply_error_to_state(State const& nominal_state,
                                                Eigen::VectorXd const& error_vec) const {
    // Start with a copy of nominal state
    State new_state = nominal_state;

    int const dof = skeleton_.active_dof();

    // Apply position error (additive)
    Eigen::Vector3d new_pos = nominal_state.root_position() + error_vec.segment<3>(0);
    new_state.set_root_position(new_pos);

    // Apply rotation error (multiplicative on manifold)
    // q_new = q_nominal ⊗ exp(error_rotation)
    Eigen::Vector3d rot_error = error_vec.segment<3>(3);
    Eigen::Quaterniond q_error = State::axis_angle_to_quaternion(rot_error);
    Eigen::Quaterniond q_nominal = nominal_state.root_orientation();
    Eigen::Quaterniond q_new = (q_nominal * q_error).normalized();
    new_state.set_root_orientation(q_new);

    // Apply velocity errors (additive)
    Eigen::Vector3d new_vel = nominal_state.root_velocity() + error_vec.segment<3>(6 + dof);
    Eigen::Vector3d new_angvel =
        nominal_state.root_angular_velocity() + error_vec.segment<3>(9 + dof);
    new_state.set_root_velocity(new_vel);
    new_state.set_root_angular_velocity(new_angvel);

    // Apply joint angle errors
    auto joints_ordered = skeleton_.get_joints_ordered();
    Eigen::VectorXd new_angles = nominal_state.joint_angles();
    Eigen::VectorXd new_joint_vels = nominal_state.joint_velocities();

    int error_pos_idx = 6;  // Start after root position/rotation in error vector
    int angle_idx = 0;      // Index in joint_angles
    int vel_idx = 0;        // Index in joint_velocities

    for (auto const& joint : joints_ordered) {
        // Skip root joint
        if (!joint.parent_index.has_value()) {
            continue;
        }

        if (joint.type == JointType::REVOLUTE) {
            // Apply joint angle error
            new_angles(angle_idx) += error_vec(error_pos_idx);
            // Apply joint velocity error (offset by 6+dof from position error)
            new_joint_vels(vel_idx) += error_vec(error_pos_idx + 6 + dof);

            error_pos_idx += 1;
            angle_idx += 1;
            vel_idx += 1;

        } else if (joint.type == JointType::SPHERICAL) {
            auto active_mask = joint.get_active_dof_mask();
            int num_active = joint.active_dof();

            if (num_active == 3) {
                // All DOFs active - use manifold composition
                Eigen::Vector3d nominal_axis_angle =
                    nominal_state.joint_angles().segment<3>(angle_idx);
                Eigen::Matrix3d R_nominal =
                    State::axis_angle_to_quaternion(nominal_axis_angle).toRotationMatrix();

                // Error in tangent space
                Eigen::Vector3d error_axis_angle = error_vec.segment<3>(error_pos_idx);
                Eigen::Matrix3d R_error =
                    State::axis_angle_to_quaternion(error_axis_angle).toRotationMatrix();

                // Compose: R_new = R_nominal * R_error
                Eigen::Matrix3d R_new = R_nominal * R_error;
                Eigen::Quaterniond q_new_joint(R_new);
                Eigen::Vector3d new_axis_angle = State::quaternion_to_axis_angle(q_new_joint);

                new_angles.segment<3>(angle_idx) = new_axis_angle;

                // Apply velocity error (offset by 6+dof from position error)
                new_joint_vels.segment<3>(vel_idx) += error_vec.segment<3>(error_pos_idx + 6 + dof);

                error_pos_idx += 3;
            } else {
                // Some DOFs locked - apply error only to active DOFs
                for (int i = 0; i < 3; ++i) {
                    if (active_mask[i]) {
                        new_angles(angle_idx + i) += error_vec(error_pos_idx);
                        new_joint_vels(vel_idx + i) += error_vec(error_pos_idx + 6 + dof);
                        error_pos_idx++;
                    }
                }
            }

            angle_idx += 3;
            vel_idx += 3;
        }
        // FIXED joints have 0 DOF, nothing to update
    }

    new_state.set_joint_angles(new_angles);
    new_state.set_joint_velocities(new_joint_vels);

    return new_state;
}

}  // namespace posetrak

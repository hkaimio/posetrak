/**
 * @file sigma_points.cpp
 * @brief Implementation of sigma point generation
 */

#include "posetrak/filters/sigma_points.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <fmt/core.h>

#include "posetrak/core/skeleton_layout.hpp"
#include <cmath>
#include <stdexcept>

namespace posetrak {

SigmaPointGenerator::SigmaPointGenerator(std::shared_ptr<const SkeletonLayout> layout, double alpha,
                                         double beta, double kappa)
    : layout_(std::move(layout)),
      error_dim_(layout_->error_state_dim()),
      alpha_(alpha),
      beta_(beta),
      kappa_(kappa) {
    fmt::print("\n=== SIGMA POINT GENERATOR INIT ===\n");
    fmt::print("error_state_dim={}, n_sigma={}\n", error_dim_, 2 * error_dim_ + 1);
    fmt::print("alpha={}, beta={}, kappa={}\n", alpha, beta, kappa);
    fmt::print("==================================\n\n");

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

    int const root_n = layout_->root_error_dof_count();  // 6 for floating root
    int const jac = layout_->joint_active_dof_count();
    int const active_dof = root_n + jac;  // == error_dim_ / 2

    // Error vector structure:
    //   [0..root_n-1]               = root position (3) + orientation (3)
    //   [root_n..root_n+jac-1]      = joint position/rotation errors
    //   [active_dof..active_dof+root_n-1] = root velocity (3) + angular velocity (3)
    //   [active_dof+root_n..]       = joint velocity errors

    // Apply root position error (first 3 elements)
    Eigen::Vector3d new_pos = nominal_state.root_position() + error_vec.segment<3>(0);
    new_state.set_root_position(new_pos);

    // Apply root rotation error (next 3 elements, multiplicative on manifold)
    Eigen::Vector3d rot_error = error_vec.segment<3>(3);
    Eigen::Quaterniond q_error = State::axis_angle_to_quaternion(rot_error);
    Eigen::Quaterniond q_nominal = nominal_state.root_orientation();
    Eigen::Quaterniond q_new = (q_nominal * q_error).normalized();
    new_state.set_root_orientation(q_new);

    // Apply root velocity errors (first 6 elements of velocity section)
    Eigen::Vector3d new_vel = nominal_state.root_velocity() + error_vec.segment<3>(active_dof);
    Eigen::Vector3d new_angvel =
        nominal_state.root_angular_velocity() + error_vec.segment<3>(active_dof + 3);
    new_state.set_root_velocity(new_vel);
    new_state.set_root_angular_velocity(new_angvel);

    // Apply joint angle and velocity errors using precomputed layout indices
    Eigen::VectorXd new_angles = nominal_state.joint_angles();
    Eigen::VectorXd new_joint_vels = nominal_state.joint_velocities();

    for (JointDesc const& j : layout_->joints()) {
        int const si = j.state_index;
        int const pos_base = root_n + j.error_index;
        int const vel_base = active_dof + root_n + j.error_index;

        if (j.type == JointType::REVOLUTE) {
            // REVOLUTE: always 1 active DOF
            new_angles(si) += error_vec(pos_base);
            new_joint_vels(si) += error_vec(vel_base);

        } else if (j.type == JointType::SPHERICAL) {
            if (j.active_dof_count == 3) {
                // All 3 DOFs active: use full SO(3) manifold composition
                Eigen::Vector3d nominal_axis_angle = nominal_state.joint_angles().segment<3>(si);
                Eigen::Matrix3d R_nominal =
                    State::axis_angle_to_quaternion(nominal_axis_angle).toRotationMatrix();

                // Error in tangent space
                Eigen::Vector3d error_axis_angle = error_vec.segment<3>(pos_base);
                Eigen::Matrix3d R_error =
                    State::axis_angle_to_quaternion(error_axis_angle).toRotationMatrix();

                // Compose: R_new = R_nominal * R_error (right multiplication for body frame)
                Eigen::Matrix3d R_new = R_nominal * R_error;
                Eigen::Quaterniond q_new_joint(R_new);
                Eigen::Vector3d new_axis_angle = State::quaternion_to_axis_angle(q_new_joint);

                new_angles.segment<3>(si) = new_axis_angle;
                new_joint_vels.segment<3>(si) += error_vec.segment<3>(vel_base);

            } else {
                // Some DOFs locked: only apply error to active axes
                int partial = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    if (j.active_dof_mask[axis]) {
                        new_angles(si + axis) += error_vec(pos_base + partial);
                        new_joint_vels(si + axis) += error_vec(vel_base + partial);
                        partial++;
                    }
                }
            }
        }
        // FIXED joints have 0 DOF, nothing to update
    }

    new_state.set_joint_angles(new_angles);
    new_state.set_joint_velocities(new_joint_vels);

    return new_state;
}

}  // namespace posetrak

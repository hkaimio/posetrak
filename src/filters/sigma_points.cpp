/**
 * @file sigma_points.cpp
 * @brief Implementation of sigma point generation
 */

#include "posetrak/filters/sigma_points.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <fmt/core.h>

#include <cmath>
#include <stdexcept>

namespace posetrak {

SigmaPointGenerator::SigmaPointGenerator(Skeleton const& skeleton, double alpha, double beta,
                                         double kappa)
    : skeleton_(skeleton),
      error_dim_(2 * skeleton.active_dof()),  // active_dof now includes root's 6 DOFs
      alpha_(alpha),
      beta_(beta),
      kappa_(kappa) {
    fmt::print("\n=== SIGMA POINT GENERATOR INIT ===\n");
    fmt::print("active_dof={}, error_dim={}, n_sigma={}\n", skeleton.active_dof(), error_dim_,
               2 * error_dim_ + 1);
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

    int const active_dof = skeleton_.active_dof();  // Includes root's 6 DOFs

    // Error vector structure (Python convention):
    // error[0:active_dof] = rotation/position errors (root 6 + body joints)
    // error[active_dof:2*active_dof] = velocity errors (root 6 + body joints)

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

    // Apply joint angle errors (handle locked DOFs)
    // Filter to only active joints to avoid processing inactive groups
    auto all_joints = skeleton_.get_joints_ordered();
    std::vector<Joint> active_joints;
    active_joints.reserve(all_joints.size());

    // Build list of active joints and compute their storage indices
    std::vector<int> storage_indices;  // Index in joint_angles storage for each active joint
    storage_indices.reserve(all_joints.size());

    int storage_idx = 0;
    for (auto const& joint : all_joints) {
        // Skip root joint
        if (!joint.parent_index.has_value()) {
            continue;
        }

        if (skeleton_.is_joint_active(joint.name)) {
            active_joints.push_back(joint);
            storage_indices.push_back(storage_idx);
        }

        // Advance storage index for all joints (active or not)
        if (joint.type == JointType::SPHERICAL) {
            storage_idx += 3;
        } else if (joint.type == JointType::REVOLUTE) {
            storage_idx += 1;
        }
    }

    Eigen::VectorXd new_angles = nominal_state.joint_angles();
    Eigen::VectorXd new_joint_vels = nominal_state.joint_velocities();

    int error_pos_idx = 6;               // Start after root's 6 DOFs in rotation section
    int error_vel_idx = active_dof + 6;  // Start after root's 6 DOFs in velocity section

    // Process only active joints
    for (size_t i = 0; i < active_joints.size(); ++i) {
        auto const& joint = active_joints[i];
        int const joint_angles_idx = storage_indices[i];  // Index in full storage

        if (joint.type == JointType::REVOLUTE) {
            // REVOLUTE: always 1 active DOF
            if (error_pos_idx >= active_dof || error_vel_idx >= 2 * active_dof) {
                throw std::runtime_error(
                    fmt::format("Error index out of bounds for joint '{}': pos_idx={}, vel_idx={}, "
                                "active_dof={}",
                                joint.name, error_pos_idx, error_vel_idx, active_dof));
            }
            new_angles(joint_angles_idx) += error_vec(error_pos_idx);
            new_joint_vels(joint_angles_idx) += error_vec(error_vel_idx);

            error_pos_idx += 1;
            error_vel_idx += 1;

        } else if (joint.type == JointType::SPHERICAL) {
            // SPHERICAL: check how many DOFs are active
            std::array<bool, 3> const active_mask = joint.get_active_dof_mask();
            int const num_active = joint.active_dof();

            if (num_active == 3) {
                // All 3 DOFs active: use full SO(3) manifold composition
                if (error_pos_idx + 2 >= active_dof || error_vel_idx + 2 >= 2 * active_dof) {
                    throw std::runtime_error(
                        fmt::format("Error index out of bounds for joint '{}': pos_idx={}, "
                                    "vel_idx={}, active_dof={}",
                                    joint.name, error_pos_idx, error_vel_idx, active_dof));
                }
                Eigen::Vector3d nominal_axis_angle =
                    nominal_state.joint_angles().segment<3>(joint_angles_idx);
                Eigen::Matrix3d R_nominal =
                    State::axis_angle_to_quaternion(nominal_axis_angle).toRotationMatrix();

                // Error in tangent space
                Eigen::Vector3d error_axis_angle = error_vec.segment<3>(error_pos_idx);
                Eigen::Matrix3d R_error =
                    State::axis_angle_to_quaternion(error_axis_angle).toRotationMatrix();

                // Compose: R_new = R_nominal * R_error (right multiplication for body frame)
                Eigen::Matrix3d R_new = R_nominal * R_error;
                Eigen::Quaterniond q_new_joint(R_new);
                Eigen::Vector3d new_axis_angle = State::quaternion_to_axis_angle(q_new_joint);

                new_angles.segment<3>(joint_angles_idx) = new_axis_angle;
                new_joint_vels.segment<3>(joint_angles_idx) += error_vec.segment<3>(error_vel_idx);

                error_pos_idx += 3;
                error_vel_idx += 3;
            } else {
                // Some DOFs locked: only apply error to active axes
                for (int axis = 0; axis < 3; ++axis) {
                    if (active_mask[axis]) {
                        if (error_pos_idx >= active_dof || error_vel_idx >= 2 * active_dof) {
                            throw std::runtime_error(fmt::format(
                                "Error index out of bounds for joint '{}' axis {}: pos_idx={}, "
                                "vel_idx={}, active_dof={}",
                                joint.name, axis, error_pos_idx, error_vel_idx, active_dof));
                        }
                        new_angles(joint_angles_idx + axis) += error_vec(error_pos_idx);
                        new_joint_vels(joint_angles_idx + axis) += error_vec(error_vel_idx);

                        error_pos_idx += 1;
                        error_vel_idx += 1;
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

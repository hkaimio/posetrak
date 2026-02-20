/**
 * @file subset_utils.cpp
 * @brief Implementation of DOF subset extraction and merging utilities
 */

#include "posetrak/filters/subset_utils.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <stdexcept>

namespace posetrak {

std::vector<int> get_active_dof_indices(Skeleton const& skeleton) {
    std::vector<int> indices;

    // Iterate through joints in order and accumulate DOF indices
    int current_dof = 0;
    for (auto const& joint : skeleton.joints()) {
        // Skip root joint (has no parent)
        if (!joint.parent_index.has_value()) {
            continue;
        }

        // Check if this joint is active
        bool is_active = skeleton.is_joint_active(joint.name);

        // Add indices for this joint's DOFs
        if (joint.type == JointType::REVOLUTE) {
            if (is_active) {
                indices.push_back(current_dof);
            }
            current_dof += 1;
        } else if (joint.type == JointType::SPHERICAL) {
            if (is_active) {
                // Always use 3 DOFs for spherical joints (storage invariant)
                indices.push_back(current_dof);
                indices.push_back(current_dof + 1);
                indices.push_back(current_dof + 2);
            }
            current_dof += 3;
        }
        // FIXED joints contribute 0 DOFs
    }

    return indices;
}

State extract_subset_state(State const& full_state, std::vector<int> const& dof_indices) {
    // Validate indices
    int const max_dof = full_state.joint_angles().size();
    for (int idx : dof_indices) {
        if (idx < 0 || idx >= max_dof) {
            throw std::invalid_argument(
                fmt::format("DOF index {} out of bounds [0, {})", idx, max_dof));
        }
    }

    // Create subset state with reduced DOFs
    State subset(dof_indices.size());

    // Copy root pose and velocities as-is
    subset.set_root_position(full_state.root_position());
    subset.set_root_orientation(full_state.root_orientation());
    subset.set_root_velocity(full_state.root_velocity());
    subset.set_root_angular_velocity(full_state.root_angular_velocity());

    // Extract subset of joint angles and velocities
    Eigen::VectorXd subset_angles(dof_indices.size());
    Eigen::VectorXd subset_velocities(dof_indices.size());

    for (size_t i = 0; i < dof_indices.size(); ++i) {
        int idx = dof_indices[i];
        subset_angles(i) = full_state.joint_angles()(idx);
        subset_velocities(i) = full_state.joint_velocities()(idx);
    }

    subset.set_joint_angles(subset_angles);
    subset.set_joint_velocities(subset_velocities);

    return subset;
}

void merge_subset_state(State& full_state, State const& subset_state,
                        std::vector<int> const& dof_indices) {
    // Validate sizes
    if (subset_state.joint_angles().size() != static_cast<int>(dof_indices.size())) {
        throw std::invalid_argument(fmt::format("Subset state has {} DOFs but {} indices provided",
                                                subset_state.joint_angles().size(),
                                                dof_indices.size()));
    }

    int const max_dof = full_state.joint_angles().size();
    for (int idx : dof_indices) {
        if (idx < 0 || idx >= max_dof) {
            throw std::invalid_argument(
                fmt::format("DOF index {} out of bounds [0, {})", idx, max_dof));
        }
    }

    // Copy subset joint angles and velocities into full state at specified indices
    // Note: We need to modify the vectors, so we'll use const_cast to get mutable access
    // or create new vectors. Let's create new vectors to be safe.
    Eigen::VectorXd full_angles = full_state.joint_angles();
    Eigen::VectorXd full_velocities = full_state.joint_velocities();

    for (size_t i = 0; i < dof_indices.size(); ++i) {
        int idx = dof_indices[i];
        full_angles(idx) = subset_state.joint_angles()(i);
        full_velocities(idx) = subset_state.joint_velocities()(i);
    }

    full_state.set_joint_angles(full_angles);
    full_state.set_joint_velocities(full_velocities);

    // Note: Root pose is NOT modified (as per spec)
}

Eigen::MatrixXd extract_subset_covariance(Eigen::MatrixXd const& full_cov,
                                          std::vector<int> const& dof_indices) {
    // Error-state ordering: [root_pos(3), root_ori(3), joint_angles(N),
    //                        root_vel(3), root_angvel(3), joint_vels(N)]
    // Total dimension: 2 * (6 + N)

    int const n_dof_full = (full_cov.rows() / 2) - 6;
    if (n_dof_full < 0 || full_cov.rows() != 2 * (6 + n_dof_full)) {
        throw std::invalid_argument(
            fmt::format("Invalid covariance dimension {} (expected 2*(6+N))", full_cov.rows()));
    }

    if (full_cov.rows() != full_cov.cols()) {
        throw std::invalid_argument(fmt::format("Covariance matrix is not square: {}x{}",
                                                full_cov.rows(), full_cov.cols()));
    }

    // Validate DOF indices
    for (int idx : dof_indices) {
        if (idx < 0 || idx >= n_dof_full) {
            throw std::invalid_argument(
                fmt::format("DOF index {} out of bounds [0, {})", idx, n_dof_full));
        }
    }

    int const n_dof_subset = dof_indices.size();
    int const subset_dim = 2 * (6 + n_dof_subset);

    // Build list of all error-state indices to extract
    // Root components (always included): [0..5] (pos + ori)
    // Subset joint angles: [6 + idx for idx in dof_indices]
    // Root velocities (always included): [6 + n_dof_full .. 6 + n_dof_full + 5]
    // Subset joint velocities: [6 + n_dof_full + 6 + idx for idx in dof_indices]

    std::vector<int> error_indices;
    error_indices.reserve(subset_dim);

    // Root position and orientation (indices 0-5)
    for (int i = 0; i < 6; ++i) {
        error_indices.push_back(i);
    }

    // Subset joint angles (indices 6 + idx)
    for (int idx : dof_indices) {
        error_indices.push_back(6 + idx);
    }

    int const velocity_offset = 6 + n_dof_full;

    // Root velocities (indices velocity_offset .. velocity_offset+5)
    for (int i = 0; i < 6; ++i) {
        error_indices.push_back(velocity_offset + i);
    }

    // Subset joint velocities (indices velocity_offset + 6 + idx)
    for (int idx : dof_indices) {
        error_indices.push_back(velocity_offset + 6 + idx);
    }

    // Extract rows and columns
    Eigen::MatrixXd subset_cov(subset_dim, subset_dim);

    for (int i = 0; i < subset_dim; ++i) {
        for (int j = 0; j < subset_dim; ++j) {
            subset_cov(i, j) = full_cov(error_indices[i], error_indices[j]);
        }
    }

    return subset_cov;
}

}  // namespace posetrak

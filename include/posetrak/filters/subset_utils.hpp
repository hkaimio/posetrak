/**
 * @file subset_utils.hpp
 * @brief Utilities for extracting and merging DOF subsets for hierarchical filtering
 *
 * Provides functions to:
 * - Query DOF indices for active joints in a skeleton
 * - Extract subset states and covariances
 * - Merge subset states back into full states
 *
 * These utilities enable hierarchical UKF implementation by operating on
 * independent state subsets (e.g., body vs hands).
 */

#pragma once

#include <Eigen/Core>

#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/state.hpp"
#include <vector>

namespace posetrak {

/**
 * @brief Get DOF indices for active joints in skeleton
 *
 * Queries which DOF indices (in the full skeleton's joint_angles vector)
 * correspond to active joints after set_active_groups() has been called.
 *
 * Example:
 *   skeleton.set_active_groups({"HandR"});
 *   auto indices = get_active_dof_indices(skeleton);
 *   // indices contains DOF positions for all joints in HandR group
 *
 * @param skeleton Skeleton with active filter applied via set_active_groups()
 * @return Vector of DOF indices (0-based positions in joint_angles vector)
 *
 * @note DOF indexing accounts for SPHERICAL joints (3 DOFs each)
 * @note Returns indices in ascending order
 */
std::vector<int> get_active_dof_indices(Skeleton const& skeleton);

/**
 * @brief Extract subset of state corresponding to DOF indices
 *
 * Creates a new state containing only the specified DOFs. Root pose is preserved
 * as-is (not subset), only joint_angles and joint_velocities are subset.
 *
 * Example:
 *   State full_state(skeleton.total_dof_count());
 *   auto dof_indices = get_active_dof_indices(active_skeleton);
 *   State subset = extract_subset_state(full_state, dof_indices);
 *   // subset.joint_angles().size() == dof_indices.size()
 *
 * @param full_state Full state with all DOFs
 * @param dof_indices Indices of DOFs to extract (0-based in joint_angles)
 * @return New state with subset DOFs
 *
 * @throws std::invalid_argument if any index is out of bounds
 *
 * @note Root position/orientation and velocities are copied as-is
 * @note Only joint_angles and joint_velocities are subset
 */
State extract_subset_state(State const& full_state, std::vector<int> const& dof_indices);

/**
 * @brief Merge subset state back into full state
 *
 * Overwrites DOFs in full_state at the specified indices with values from
 * subset_state. Root pose from subset is ignored (full_state root unchanged).
 *
 * Example:
 *   State full_state = ...;
 *   State hand_state = extract_subset_state(full_state, hand_indices);
 *   // ... run hand UKF filter on hand_state ...
 *   merge_subset_state(full_state, hand_state, hand_indices);
 *   // full_state now has updated hand DOFs, body DOFs unchanged
 *
 * @param full_state Full state to update (modified in-place)
 * @param subset_state Subset state with updated DOF values
 * @param dof_indices Indices where to write subset values
 *
 * @throws std::invalid_argument if sizes mismatch or indices out of bounds
 *
 * @note Root pose in full_state is NOT modified
 * @note Only updates joint_angles and joint_velocities at specified indices
 */
void merge_subset_state(State& full_state, State const& subset_state,
                        std::vector<int> const& dof_indices);

/**
 * @brief Extract subset of covariance matrix
 *
 * Extracts rows and columns corresponding to specified DOF indices from the
 * full covariance matrix in error-state space.
 *
 * Error-state ordering: [root_pos (3), root_ori (3), joint_angles (N),
 *                        root_vel (3), root_angvel (3), joint_vels (N)]
 *
 * This function handles the mapping from joint DOF indices to error-state
 * indices, accounting for root pose components.
 *
 * Example:
 *   MatrixXd full_cov = ...; // Size: (2*(6+N)) x (2*(6+N))
 *   auto hand_indices = get_active_dof_indices(hand_skeleton); // M indices
 *   MatrixXd hand_cov = extract_subset_covariance(full_cov, hand_indices);
 *   // hand_cov.size() == (2*(6+M)) x (2*(6+M))
 *
 * @param full_cov Full covariance in error-state space
 * @param dof_indices Indices of DOFs to extract
 * @return Subset covariance matrix with root + selected DOFs
 *
 * @throws std::invalid_argument if dimensions mismatch or indices out of bounds
 *
 * @note Always includes root components (pos, ori, vel, angvel) in output
 * @note Output has dimensions: (2 * (6 + dof_indices.size())) x (2 * (6 + dof_indices.size()))
 */
Eigen::MatrixXd extract_subset_covariance(Eigen::MatrixXd const& full_cov,
                                          std::vector<int> const& dof_indices);

}  // namespace posetrak

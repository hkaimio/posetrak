// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file skeleton_state.hpp
 * @brief SkeletonState: compact State paired with a SkeletonLayout.
 *
 * SkeletonState is the unit of exchange between hierarchical UKF filters.
 * Each filter owns a SkeletonState whose State is compact-sized
 * (joint_angles.size() == layout->total_storage_dof_count()), not
 * full-skeleton-sized. This avoids padding the joint vector with zeros for
 * joints that a child filter does not track.
 *
 * The two key operations are:
 *  - merge_into(): scatter this filter's DOFs back into a parent/full state.
 *  - extract_covariance(): pull the relevant rows/cols from a larger covariance.
 */

#pragma once

#include <Eigen/Core>

#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/core/state.hpp"
#include <memory>
#include <vector>

namespace posetrak {

/// @brief Compact state for a (sub)set of skeleton joints, paired with its layout.
///
/// The State held here is compact: joint_angles().size() equals
/// layout()->total_storage_dof_count(). It is NOT full-skeleton-sized.
///
/// Use merge_into() to scatter DOFs into a parent SkeletonState, and
/// extract_covariance() to pull the matching slice out of a full covariance.
class SkeletonState {
   public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /// @brief Create a SkeletonState from a precomputed layout and a compact state.
    ///
    /// @p state must have joint_angles().size() == layout->total_storage_dof_count().
    /// @throws std::invalid_argument if the sizes are inconsistent.
    static SkeletonState create(std::shared_ptr<const SkeletonLayout> layout, State state);

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    std::shared_ptr<const SkeletonLayout> const& layout() const { return layout_; }

    State const& state() const { return state_; }
    State& state() { return state_; }

    // -------------------------------------------------------------------------
    // Merge / extract operations
    // -------------------------------------------------------------------------

    /// @brief Scatter this state's DOFs into @p target at positions from @p merge_map.
    ///
    /// merge_map must have one entry per storage DOF in *this (size ==
    /// layout()->total_storage_dof_count()). merge_map[i] is the index inside
    /// target.state().joint_angles() / joint_velocities() where compact DOF i should land.
    ///
    /// Build merge_map once at construction time:
    /// @code
    ///   auto merge_map = full_ss.layout()->build_index_map_from(*subset_ss.layout());
    /// @endcode
    ///
    /// Root pose, velocity, and angular velocity are NOT transferred; only joint
    /// angles and joint velocities.
    ///
    /// @throws std::invalid_argument if merge_map.size() != layout()->total_storage_dof_count().
    void merge_into(SkeletonState& target, std::vector<int> const& merge_map) const;

    /// @brief Extract the rows/cols of @p full_cov that correspond to this layout.
    ///
    /// Returns a square matrix of size layout()->error_state_dim()².
    /// Rows/cols are selected by mapping each joint's layout-relative error_index
    /// to the corresponding joint's error_index in @p full_layout (matched by name).
    ///
    /// The extraction handles both position and velocity halves of the error state.
    /// Root error DOFs are included at the front if layout()->has_floating_root().
    ///
    /// @param full_cov    Square covariance with rows == full_layout.error_state_dim().
    /// @param full_layout Layout that @p full_cov was built against.
    /// @throws std::invalid_argument if dimensions mismatch or a joint is missing.
    Eigen::MatrixXd extract_covariance(Eigen::MatrixXd const& full_cov,
                                       SkeletonLayout const& full_layout) const;

   private:
    SkeletonState(std::shared_ptr<const SkeletonLayout> layout, State state);

    std::shared_ptr<const SkeletonLayout> layout_;
    State state_;
};

}  // namespace posetrak

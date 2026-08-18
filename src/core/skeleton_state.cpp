// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/core/skeleton_state.hpp"

#include <fmt/format.h>

#include <cassert>
#include <stdexcept>

namespace posetrak {

// ---------------------------------------------------------------------------
// Private constructor
// ---------------------------------------------------------------------------

SkeletonState::SkeletonState(std::shared_ptr<const SkeletonLayout> layout, State state)
    : layout_(std::move(layout)), state_(std::move(state)) {}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SkeletonState SkeletonState::create(std::shared_ptr<const SkeletonLayout> layout, State state) {
    if (!layout) {
        throw std::invalid_argument("SkeletonState::create: layout must not be null");
    }
    if (state.joint_angles().size() != layout->total_storage_dof_count()) {
        throw std::invalid_argument(
            fmt::format("SkeletonState::create: state has {} joint DOFs but layout expects {}",
                        state.joint_angles().size(), layout->total_storage_dof_count()));
    }
    return SkeletonState(std::move(layout), std::move(state));
}

// ---------------------------------------------------------------------------
// merge_into
// ---------------------------------------------------------------------------

void SkeletonState::merge_into(SkeletonState& target, std::vector<int> const& merge_map) const {
    int const expected_size = layout_->total_storage_dof_count();
    if (static_cast<int>(merge_map.size()) != expected_size) {
        throw std::invalid_argument(
            fmt::format("SkeletonState::merge_into: merge_map has {} entries but layout has {} "
                        "storage DOFs",
                        merge_map.size(), expected_size));
    }

    // Copy target's angle/velocity vectors so we can do indexed scatter.
    Eigen::VectorXd angles = target.state_.joint_angles();
    Eigen::VectorXd vels = target.state_.joint_velocities();

    for (int i = 0; i < expected_size; ++i) {
        angles(merge_map[i]) = state_.joint_angles()(i);
        vels(merge_map[i]) = state_.joint_velocities()(i);
    }

    target.state_.set_joint_angles(angles);
    target.state_.set_joint_velocities(vels);
    // Root pose, velocity, and angular velocity are NOT transferred.
}

// ---------------------------------------------------------------------------
// extract_covariance
// ---------------------------------------------------------------------------

Eigen::MatrixXd SkeletonState::extract_covariance(Eigen::MatrixXd const& full_cov,
                                                  SkeletonLayout const& full_layout) const {
    // Validate that full_cov is square and matches full_layout.
    if (full_cov.rows() != full_cov.cols()) {
        throw std::invalid_argument(
            "SkeletonState::extract_covariance: covariance matrix must be square");
    }
    if (full_cov.rows() != full_layout.error_state_dim()) {
        throw std::invalid_argument(
            fmt::format("SkeletonState::extract_covariance: full_cov has {} rows but "
                        "full_layout.error_state_dim() is {}",
                        full_cov.rows(), full_layout.error_state_dim()));
    }

    int const sub_dim = layout_->error_state_dim();
    int const sub_root = layout_->root_error_dof_count();   // 6 or 0
    int const sub_jac = layout_->joint_active_dof_count();  // M free joint DOFs

    int const full_root = full_layout.root_error_dof_count();   // 6 or 0
    int const full_jac = full_layout.joint_active_dof_count();  // N free joint DOFs

    // Build the index mapping: sub_error_idx → full_error_idx.
    // Error-state layout (for any layout L):
    //   [0 .. L.root_error_dof_count()-1]                     root position+orientation
    //   [L.root_error_dof_count() .. +L.joint_active_dof_count()-1] joint angle errors
    //   [half .. half + L.root_error_dof_count()-1]           root velocity errors
    //   [half + L.root_error_dof_count() .. sub_dim-1]        joint velocity errors
    // where half = L.root_error_dof_count() + L.joint_active_dof_count().
    std::vector<int> idx_map;
    idx_map.reserve(static_cast<size_t>(sub_dim));

    // --- Root position/orientation (first half) ---
    for (int i = 0; i < sub_root; ++i) {
        idx_map.push_back(i);  // full layout's root block also starts at 0
    }

    // --- Joint angle errors (first half) ---
    for (auto const& jdesc : layout_->joints()) {
        JointDesc const* full_jdesc = full_layout.get_joint(jdesc.name);
        if (!full_jdesc) {
            throw std::invalid_argument(
                fmt::format("SkeletonState::extract_covariance: joint '{}' not found in "
                            "full_layout",
                            jdesc.name));
        }
        for (int d = 0; d < static_cast<int>(jdesc.active_dof_count); ++d) {
            idx_map.push_back(full_root + static_cast<int>(full_jdesc->error_index) + d);
        }
    }

    // Half-point offsets for the velocity block.
    int const full_half = full_root + full_jac;

    // --- Root velocity/angular velocity (second half) ---
    for (int i = 0; i < sub_root; ++i) {
        idx_map.push_back(full_half + i);
    }

    // --- Joint velocity errors (second half) ---
    for (auto const& jdesc : layout_->joints()) {
        JointDesc const* full_jdesc = full_layout.get_joint(jdesc.name);
        // Already validated above; full_jdesc is never null here.
        for (int d = 0; d < static_cast<int>(jdesc.active_dof_count); ++d) {
            idx_map.push_back(full_half + full_root + static_cast<int>(full_jdesc->error_index) +
                              d);
        }
    }

    assert(static_cast<int>(idx_map.size()) == sub_dim);

    // Extract the sub-matrix.
    Eigen::MatrixXd result(sub_dim, sub_dim);
    for (int i = 0; i < sub_dim; ++i) {
        for (int j = 0; j < sub_dim; ++j) {
            result(i, j) = full_cov(idx_map[i], idx_map[j]);
        }
    }
    return result;
}

}  // namespace posetrak

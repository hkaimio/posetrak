/**
 * @file skeleton_layout.hpp
 * @brief Precomputed, immutable DOF layout for a subset of skeleton joints.
 *
 * SkeletonLayout is the single source of truth for all DOF index arithmetic.
 * It replaces ad-hoc joint iteration loops that previously appeared independently
 * in UnscentedKalmanFilter, ConstantVelocityModel, SigmaPointGenerator, etc.
 *
 * Key design decisions:
 * - Immutable after construction (shared_ptr<const SkeletonLayout>)
 * - All indices precomputed in constructor — O(1) hot-path access
 * - state_index and error_index are layout-relative (not full-skeleton relative)
 * - build_index_map_from() translates subset→full indices, called once and cached
 */

#pragma once

#include <Eigen/Core>

#include "posetrak/core/skeleton.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace posetrak {

/// @brief Per-joint descriptor with all DOF information precomputed.
///
/// Populated once at SkeletonLayout construction. Every field needed by
/// hot-path loops (process model, sigma points, UKF) is available at O(1).
struct JointDesc {
    std::string name;  ///< Joint name (matches Skeleton::Joint::name)
    JointType type;    ///< REVOLUTE, SPHERICAL, or FIXED

    /// @brief Elements occupied in State::joint_angles / joint_velocities (1/3/0)
    uint32_t storage_dof_count;

    /// @brief Actually-free DOFs after accounting for locked axes
    /// For REVOLUTE: 1. For SPHERICAL: 1-3 (some axes may be locked by equal limits).
    /// For FIXED: 0.
    uint32_t active_dof_count;

    /// @brief Start index in State::joint_angles and State::joint_velocities.
    /// Layout-relative: index 0 is the first non-root joint in THIS layout,
    /// regardless of the joint's position in the full skeleton.
    uint32_t state_index;

    /// @brief Start index in the JOINT portion of the error-state vector
    /// (i.e. after the root's contribution, if any).
    /// Layout-relative: 0 for the first active joint in this layout.
    /// Absolute error-state position index = root_error_dof_count() + error_index.
    /// Absolute error-state velocity index = root_error_dof_count() + joint_active_dof_count()
    ///                                         + root_error_dof_count() + error_index.
    uint32_t error_index;

    bool is_floating_root;  ///< Always false in joints() list (root not included here)

    std::array<Eigen::Vector2d, 3> limits;  ///< Joint limits [min, max] per DOF
    uint32_t limit_count;                   ///< Number of active limit pairs (0-3)
    std::array<bool, 3> active_dof_mask;    ///< Which axes are free (SPHERICAL)
};

/// @brief Precomputed, immutable DOF layout for a (sub)set of skeleton joints.
///
/// Created once via factory functions, then shared between UKF, process model,
/// sigma point generator, and SkeletonState through shared_ptr<const SkeletonLayout>.
///
/// Pointer equality (layout_a.get() == layout_b.get()) can be used to cheaply
/// confirm two handles refer to the exact same layout object, but there is no
/// built-in check that two distinct layouts were derived from the same skeleton.
/// That responsibility lies with the caller (see build_index_map_from()).
class SkeletonLayout {
   public:
    // -------------------------------------------------------------------------
    // Factory functions — pure queries, do NOT mutate the Skeleton
    // -------------------------------------------------------------------------

    /// @brief Build layout for ALL joints in the skeleton.
    /// The kinematic root (joint with no parent) is treated as a floating body:
    /// has_floating_root() returns true.
    static std::shared_ptr<const SkeletonLayout> from_full_skeleton(Skeleton const& skeleton);

    /// @brief Build layout for joints whose group field matches one of group_names.
    ///
    /// If the kinematic root's group is listed, has_floating_root() returns true.
    /// Otherwise has_floating_root() returns false (child filter mode).
    ///
    /// @param skeleton  Source skeleton (not mutated)
    /// @param group_names  Group names to include (e.g. {"main"} or {"HandR"})
    /// @throws std::invalid_argument if group_names is empty or no joints match
    static std::shared_ptr<const SkeletonLayout>
    from_groups(Skeleton const& skeleton, std::vector<std::string> const& group_names);

    // -------------------------------------------------------------------------
    // O(1) accessors (all values precomputed at construction)
    // -------------------------------------------------------------------------

    /// @brief Non-root joints in state-vector order, filtered to this layout.
    /// Root joint is NOT included; use has_floating_root() to handle it.
    std::vector<JointDesc> const& joints() const { return joints_; }

    /// @brief Look up joint by name. O(1) via internal unordered_map.
    /// @return Pointer to descriptor, or nullptr if not in this layout.
    JointDesc const* get_joint(std::string const& name) const;

    /// @brief Size of State::joint_angles / State::joint_velocities for this layout.
    /// Sum of storage_dof_count across all joints (1 per REVOLUTE, 3 per SPHERICAL).
    uint32_t total_storage_dof_count() const { return total_storage_dof_count_; }

    /// @brief Number of free (active) joint DOFs, excluding root's 6.
    /// Used to compute error_state_dim().
    uint32_t joint_active_dof_count() const { return joint_active_dof_count_; }

    /// @brief Contribution of the root to the error-state vector.
    /// Returns 6 if has_floating_root(), else 0.
    uint32_t root_error_dof_count() const { return has_floating_root_ ? 6u : 0u; }

    /// @brief Total error-state dimension used by UKF / sigma points.
    /// = 2 * (root_error_dof_count() + joint_active_dof_count())
    int error_state_dim() const {
        return 2 * static_cast<int>(root_error_dof_count() + joint_active_dof_count_);
    }

    /// @brief True if the kinematic root is a free-floating body in this layout.
    /// False for child filters (e.g. hand filter) where root pose is set externally.
    bool has_floating_root() const { return has_floating_root_; }

    // -------------------------------------------------------------------------
    // Index mapping (for extract / merge between subset and full layouts)
    // -------------------------------------------------------------------------

    /// @brief Build a DOF index map from a subset layout into this (full) layout.
    ///
    /// The returned vector has one entry per storage DOF in @p subset.
    /// Entry i is the state_index in THIS layout that corresponds to subset DOF i.
    ///
    /// Intended use:
    /// @code
    ///   // Called ONCE at SubsetUKF construction:
    ///   auto merge_map = full_layout->build_index_map_from(*hand_layout);
    ///
    ///   // At runtime (O(N) array copy, no name lookups):
    ///   for (size_t i = 0; i < merge_map.size(); ++i)
    ///       full_angles[merge_map[i]] = subset_angles[i];
    /// @endcode
    ///
    /// @note It is the caller's responsibility to ensure that both layouts were
    ///       derived from the same Skeleton. This function matches joints by name
    ///       only; it cannot detect that two layouts came from structurally
    ///       similar but distinct skeletons.
    /// @throws std::invalid_argument if any joint in subset is not present in this layout.
    std::vector<uint32_t> build_index_map_from(SkeletonLayout const& subset) const;

   private:
    SkeletonLayout() = default;  // Only factory functions construct

    /// @brief Core construction logic shared by both factory functions.
    /// @param skeleton          Source skeleton
    /// @param include_all       If true, include all non-fixed joints (from_full_skeleton)
    /// @param group_names       Groups to include (used when include_all is false)
    static std::shared_ptr<const SkeletonLayout> build(Skeleton const& skeleton, bool include_all,
                                                       std::vector<std::string> const& group_names);

    std::vector<JointDesc> joints_;                          ///< Non-root joints in order
    std::unordered_map<std::string, uint32_t> name_to_idx_;  ///< name → index in joints_
    uint32_t total_storage_dof_count_ = 0;
    uint32_t joint_active_dof_count_ = 0;
    bool has_floating_root_ = false;
};

}  // namespace posetrak

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
#include "posetrak/core/state.hpp"
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
    int limit_count;                        ///< Number of active limit pairs (0-3)
    std::array<bool, 3> active_dof_mask;    ///< Which axes are free (SPHERICAL)

    // Scale-group fields (PRISMATIC joints only)
    /// Original bone length in metres (|original_offset|). State stores a proportional
    /// scale factor s; the Pinocchio q value is q = s * nominal_length.
    double nominal_length = 0.0;

    /// Scale group name for PRISMATIC joints (same value for leader and all followers in a group).
    std::string scale_group;

    /// True for PRISMATIC joints that are the 2nd-or-later member of a scale group.
    /// Followers share state_index with the group leader and contribute no independent
    /// DOF (storage_dof_count = 0, active_dof_count = 0). FK still reads their slot
    /// via state_index (which equals the leader's) and multiplies by their own nominal_length.
    bool is_scale_follower = false;
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
    static std::shared_ptr<const SkeletonLayout>
    from_full_skeleton(std::shared_ptr<const Skeleton> skeleton);

    /// @brief Build layout for joints whose group field matches one of group_names.
    ///
    /// If the kinematic root's group is listed, has_floating_root() returns true.
    /// Otherwise has_floating_root() returns false (child filter mode).
    ///
    /// @param skeleton  Source skeleton (not mutated)
    /// @param group_names  Group names to include (e.g. {"main"} or {"HandR"})
    /// @throws std::invalid_argument if group_names is empty or no joints match
    static std::shared_ptr<const SkeletonLayout>
    from_groups(std::shared_ptr<const Skeleton> skeleton,
                std::vector<std::string> const& group_names);

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
    int total_storage_dof_count() const { return total_storage_dof_count_; }

    /// @brief Number of free (active) joint DOFs, excluding root's 6.
    /// Used to compute error_state_dim().
    int joint_active_dof_count() const { return joint_active_dof_count_; }

    /// @brief Contribution of the root to the error-state vector.
    /// Returns 6 if has_floating_root(), else 0.
    int root_error_dof_count() const { return has_floating_root_ ? 6 : 0; }

    /// @brief Total error-state dimension used by UKF / sigma points.
    /// = 2 * (root_error_dof_count() + joint_active_dof_count())
    int error_state_dim() const {
        return 2 * static_cast<int>(root_error_dof_count() + joint_active_dof_count_);
    }

    /// @brief True if the kinematic root is a free-floating body in this layout.
    /// False for child filters (e.g. hand filter) where root pose is set externally.
    bool has_floating_root() const { return has_floating_root_; }

    /// @brief The skeleton this layout was derived from.
    /// Guaranteed non-null after construction via any factory function.
    std::shared_ptr<const Skeleton> const& skeleton() const { return skeleton_; }

    /// @brief Find the parent marker ID for a given marker.
    /// Traverses the joint parent chain to find the nearest ancestor joint with a marker.
    /// @return Parent marker index, or -1 for root markers or markers with no ancestor marker.
    int parent_marker_id(int marker_id) const;

    /// @brief Skeleton-tree distance between two markers in joint hops.
    /// Distance is the number of joint edges on the shortest path between the joints
    /// that the two markers are attached to. Returns INT_MAX for invalid indices.
    int hierarchy_distance(int marker_a, int marker_b) const;

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
    std::vector<int> build_index_map_from(SkeletonLayout const& subset) const;

    /// @brief Build an ERROR-STATE DOF index map from a subset layout into this
    /// (full) layout -- the error_index/active_dof_count analogue of
    /// build_index_map_from() (which is state_index/storage_dof_count based).
    ///
    /// Storage indexing and error-state indexing diverge whenever a joint has a
    /// locked axis (a SPHERICAL joint with an equal-limits axis, e.g. this
    /// codebase's own ball-jointed finger phalanges): storage always reserves 3
    /// slots per SPHERICAL joint, but a locked axis contributes no slot to the
    /// error state / UKF covariance. Do NOT reuse build_index_map_from()'s map
    /// for anything indexed into a covariance matrix or its diagonal -- it will
    /// silently misalign the moment a locked axis is involved. Use
    /// error_blob_index() to turn an entry of this map into an absolute offset
    /// into an error-state vector or covariance diagonal.
    ///
    /// The returned vector has one entry per active DOF in @p subset, in the
    /// same order subset's own error_index assignment used. Entry i is the
    /// layout-relative error_index (joint-block-only -- see error_blob_index()
    /// for the absolute offset) in THIS layout that corresponds to subset's
    /// i-th active joint DOF.
    ///
    /// @throws std::invalid_argument if any joint in subset is not present in
    ///         this layout, or if a shared joint's active_dof_count differs
    ///         between the two layouts (active_dof_count is a property of the
    ///         joint's own limits, not the layout, so a mismatch means the two
    ///         layouts were not built from the same skeleton).
    std::vector<int> build_error_index_map_from(SkeletonLayout const& subset) const;

    /// @brief Absolute offset of a layout-relative error_index (this layout's
    /// own joint block) within this layout's error-state vector / covariance
    /// diagonal:
    /// [root_pos(3), root_ori(3), joint_pos(joint_active_dof_count()),
    ///  root_vel(3), root_angvel(3), joint_vel(joint_active_dof_count())].
    /// This is the dimension the UKF's covariance matrix (and therefore
    /// tracking_results.cov_diag) actually uses -- see error_state_dim().
    /// @param error_index A layout-relative error_index for a joint in THIS
    ///        layout (e.g. one entry of build_error_index_map_from()'s result).
    /// @param is_velocity False for the position sub-block, true for velocity.
    int error_blob_index(int error_index, bool is_velocity) const {
        int const r = root_error_dof_count();
        return is_velocity ? 2 * r + joint_active_dof_count_ + error_index : r + error_index;
    }

    /// @brief Slice a full-skeleton State to match this (subset) layout's dimensions.
    ///
    /// If the input state already matches this layout's size, returns a copy.
    /// Otherwise, extracts DOFs from @p full_state according to the joint names
    /// in this layout, using an internal index map.
    ///
    /// Intended use: when loading IK results or CSV states (full-skeleton sized)
    /// into a layout-scoped UKF.
    ///
    /// @param full_state  State vector sized for the full skeleton
    /// @return State vector sized for this layout (total_storage_dof_count())
    /// @throws std::invalid_argument if skeleton pointers do not match
    State slice_state(State const& full_state) const;

   private:
    SkeletonLayout() = default;  // Only factory functions construct

    /// @brief Core construction logic shared by both factory functions.
    /// @param skeleton          Source skeleton (shared ownership stored in layout)
    /// @param include_all       If true, include all non-fixed joints (from_full_skeleton)
    /// @param group_names       Groups to include (used when include_all is false)
    static std::shared_ptr<const SkeletonLayout> build(std::shared_ptr<const Skeleton> skeleton,
                                                       bool include_all,
                                                       std::vector<std::string> const& group_names);

    std::vector<JointDesc> joints_;                     ///< Non-root joints in order
    std::unordered_map<std::string, int> name_to_idx_;  ///< name → index in joints_
    std::shared_ptr<const Skeleton> skeleton_;          ///< Source skeleton (immutable)
    int total_storage_dof_count_ = 0;
    int joint_active_dof_count_ = 0;
    bool has_floating_root_ = false;
    std::unordered_map<int, int>
        marker_to_parent_marker_;  ///< marker_id → parent_marker_id (-1 for root)
    std::vector<std::vector<int>>
        marker_dist_matrix_;  ///< [n_markers][n_markers] shortest joint-hop distance
};

}  // namespace posetrak

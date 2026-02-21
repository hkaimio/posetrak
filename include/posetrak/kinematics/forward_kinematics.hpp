/**
 * @file forward_kinematics.hpp
 * @brief Forward kinematics computation using Pinocchio
 *
 * Adapted from cpp-tracker-test (proven zero-error implementation)
 * Critical: Must call both forwardKinematics() and updateFramePlacements()
 */

#pragma once

#include <Eigen/Dense>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/core/state.hpp"
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace posetrak {

/**
 * @brief Forward kinematics computer using Pinocchio
 *
 * This class provides efficient FK computation by:
 * 1. Converting State to Pinocchio configuration vectors
 * 2. Computing FK with Pinocchio's optimized algorithms
 * 3. Extracting marker positions from computed frame transforms
 *
 * CRITICAL: Pinocchio quaternions use [x, y, z, w] order
 * CRITICAL: Must call updateFramePlacements() after forwardKinematics()
 *
 * The layout fully describes which joints are active and owns the Skeleton pointer.
 * Both the full-skeleton path and the child-subtree path use the same constructor;
 * the branching (full vs compact) is implicit in the layout's joint list.
 */
class ForwardKinematics {
   public:
    /**
     * @brief Construct FK computer.
     *
     * Works for both the full-skeleton path (layout = from_full_skeleton) and the
     * child-subtree path (layout = from_groups without root).  The Skeleton is
     * accessed via layout->skeleton() — no separate Skeleton argument needed.
     *
     * @param model   Pinocchio model (built from skeleton or build_subtree_model)
     * @param data    Pinocchio data structure
     * @param marker_frame_map  Map from marker name to frame index
     * @param layout  Layout whose joint list and skeleton() drive state_to_config
     */
    ForwardKinematics(pinocchio::Model const& model, pinocchio::Data& data,
                      std::map<std::string, pinocchio::FrameIndex> const& marker_frame_map,
                      std::shared_ptr<const SkeletonLayout> layout);

    /**
     * @brief Compute forward kinematics from State.
     *
     * Uses the full-skeleton path if constructed without a layout, or the layout-aware
     * path if constructed with one.
     *
     * @param state State with root pose and joint angles
     * @return Map of marker name → 3D position in world frame
     */
    std::unordered_map<std::string, Eigen::Vector3d> compute(State const& state);

    /**
     * @brief Compute forward kinematics from configuration vector directly
     * @param q Configuration vector (nq-dimensional)
     * @return Map of marker name → 3D position in world frame
     */
    std::unordered_map<std::string, Eigen::Vector3d> compute(Eigen::VectorXd const& q) const;

    /**
     * @brief Convert State to Pinocchio configuration vector (full-skeleton path).
     *
     * For a skeleton with root + joints:
     * - Root: 7 DOF (3 position + 4 quaternion [x,y,z,w])
     * - Spherical joints: 4 DOF each (quaternion [x,y,z,w])
     * - Revolute joints: 1 DOF each (angle)
     *
     * @param state State with root pose and joint angles
     * @param skeleton Skeleton structure (for joint ordering and types)
     * @return Configuration vector q (nq-dimensional)
     */
    /**
     * @brief Convert State to Pinocchio configuration vector using the layout.
     *
     * Works for both the full-skeleton and child-subtree pinocchio models.  Both models
     * start with a 7-DOF freeflyer (pos+quat), followed by the layout's joints in
     * insertion order.  The root pos/quat always comes from state.root_position() /
     * state.root_orientation() (for child filters the coordinator injects these).
     *
     * @param state  State whose root pose and joint_angles (indexed by desc.state_index)
     *               are mapped to the q vector.
     * @param layout Layout whose joints() drive the joint section of q.  Skeleton is
     *               obtained from layout.skeleton() for type lookups.
     * @return Configuration vector q (nq-dimensional).
     */
    static Eigen::VectorXd state_to_config(State const& state, SkeletonLayout const& layout);

    /**
     * @brief Convert State to Pinocchio configuration vector (legacy, skeleton-only path).
     *
     * Iterates skeleton.get_joints_ordered() directly.  Kept for tests that construct
     * FK from a skeleton without a layout.  Prefer the layout overload for new code.
     */
    static Eigen::VectorXd state_to_config(State const& state, Skeleton const& skeleton);

    /**
     * @brief Get world-frame pose of a named skeleton joint after compute().
     *
     * Reads from data_.oMi[] which is populated by forwardKinematics() inside compute().
     * Must be called after at least one compute() call for a meaningful result.
     *
     * Only works for joints that are actual Pinocchio joints (non-FIXED skeleton joints,
     * plus any FreeFlyer added by build_subtree_model).  FIXED joints in the full-skeleton
     * model are folded into their parent; querying them throws.
     *
     * @param joint_name  Pinocchio joint name (matches skeleton joint name for non-FIXED joints).
     * @return {position, orientation} of the joint frame in world coordinates.
     * @throws std::out_of_range if joint_name is not a known Pinocchio joint.
     */
    std::pair<Eigen::Vector3d, Eigen::Quaterniond>
    world_transform(std::string const& joint_name) const;

   private:
    pinocchio::Model const& model_;
    pinocchio::Data& data_;
    std::map<std::string, pinocchio::FrameIndex> marker_frame_map_;  ///< owned copy, not a ref
    std::shared_ptr<const SkeletonLayout> layout_;                   ///< Always non-null
    std::unordered_map<std::string, pinocchio::JointIndex> joint_id_map_;  ///< name → oMi index
};

}  // namespace posetrak

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
#include "posetrak/core/state.hpp"
#include <map>
#include <string>
#include <unordered_map>

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
 */
class ForwardKinematics {
   public:
    /**
     * @brief Construct FK computer with Pinocchio model and marker frame mapping
     * @param model Pinocchio model (built from skeleton)
     * @param data Pinocchio data structure
     * @param marker_frame_map Map from marker name to frame index
     * @param skeleton Skeleton structure (needed for State→config conversion)
     */
    ForwardKinematics(pinocchio::Model const& model, pinocchio::Data& data,
                      std::map<std::string, pinocchio::FrameIndex> const& marker_frame_map,
                      Skeleton const& skeleton);

    /**
     * @brief Compute forward kinematics from State
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
     * @brief Convert State to Pinocchio configuration vector
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
    static Eigen::VectorXd state_to_config(State const& state, Skeleton const& skeleton);

   private:
    pinocchio::Model const& model_;
    pinocchio::Data& data_;
    std::map<std::string, pinocchio::FrameIndex> const& marker_frame_map_;
    Skeleton const& skeleton_;
};

}  // namespace posetrak

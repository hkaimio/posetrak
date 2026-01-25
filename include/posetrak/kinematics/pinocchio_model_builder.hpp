/**
 * @file pinocchio_model_builder.hpp
 * @brief Build Pinocchio models from posetrak skeleton structures
 *
 * Adapted from cpp-tracker-test (proven zero-error implementation)
 */

#pragma once

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "posetrak/core/skeleton.hpp"
#include <map>
#include <string>

namespace posetrak {

/**
 * @brief Builds Pinocchio models from skeleton definitions
 *
 * This class converts posetrak Skeleton representation into Pinocchio's
 * Model structure, handling joint type mapping, frame attachment,
 * and model validation.
 *
 * Joint type mapping:
 * - Root (empty parent) → JointModelFreeFlyer (6 DOF: 3 position + 3 orientation)
 * - SPHERICAL (3 DOF) → JointModelSpherical (quaternion rotation)
 * - REVOLUTE (1 DOF) → JointModelRX/RY/RZ (single axis rotation)
 * - FIXED (0 DOF) → Skipped (markers attached directly to parent joint frame)
 *
 * Markers are attached as operational frames at specified offsets.
 */
class PinocchioModelBuilder {
   public:
    /**
     * @brief Build a Pinocchio model from a skeleton
     *
     * @param skeleton The skeleton structure to convert
     * @param[out] model The model to build (will be cleared and rebuilt)
     * @throws std::runtime_error if skeleton is invalid or conversion fails
     */
    static void build_model(Skeleton const& skeleton, pinocchio::Model& model);

    /**
     * @brief Build model and create corresponding Data structure
     *
     * @param skeleton The skeleton structure
     * @param[out] model The built model
     * @param[out] data The corresponding data structure
     * @throws std::runtime_error if skeleton is invalid or conversion fails
     */
    static void build_model_and_data(Skeleton const& skeleton, pinocchio::Model& model,
                                     pinocchio::Data& data);

    /**
     * @brief Get frame ID for a marker by name
     *
     * @param model The Pinocchio model
     * @param marker_name Name of the marker
     * @return Frame ID, or -1 if not found
     */
    static int get_marker_frame_id(pinocchio::Model const& model, std::string const& marker_name);

    /**
     * @brief Map from marker name to frame ID
     *
     * @param model The Pinocchio model
     * @param skeleton The skeleton (for marker names)
     * @return Map of marker_name → frame_id
     */
    static std::map<std::string, pinocchio::FrameIndex>
    build_marker_frame_map(pinocchio::Model const& model, Skeleton const& skeleton);

    /**
     * @brief Print model structure for debugging
     *
     * @param model The Pinocchio model
     * @param skeleton The original skeleton (for comparison)
     */
    static void print_model_info(pinocchio::Model const& model, Skeleton const& skeleton);

   private:
    /**
     * @brief Add a joint to the model recursively
     *
     * @param model The model being built
     * @param skeleton The skeleton structure
     * @param joint The joint to add
     * @param parent_id Pinocchio joint ID of parent (0 for root = universe)
     * @param joint_to_id Map to track joint name → pinocchio joint ID
     */
    static void add_joint_recursive(pinocchio::Model& model, Skeleton const& skeleton,
                                    Joint const& joint, pinocchio::JointIndex parent_id,
                                    std::map<std::string, pinocchio::JointIndex>& joint_to_id);

    /**
     * @brief Add marker frames to model
     *
     * @param model The model being built
     * @param skeleton The skeleton structure (for markers)
     * @param joint_to_id Map from joint name to Pinocchio joint ID
     */
    static void add_marker_frames(pinocchio::Model& model, Skeleton const& skeleton,
                                  std::map<std::string, pinocchio::JointIndex> const& joint_to_id);

    /**
     * @brief Determine rotation axis for revolute joint
     *
     * @param joint The revolute joint
     * @return 'X', 'Y', or 'Z' indicating rotation axis
     */
    static char get_revolute_axis(Joint const& joint);
};

}  // namespace posetrak

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
#include "posetrak/core/skeleton_layout.hpp"
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

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
     * @brief Build a Pinocchio model for a subtree of the skeleton.
     *
     * The resulting model has freeflyer_joint_name as its free-flyer root (placed at
     * SE3::Identity). The child joints — those in group_names and descendants of
     * freeflyer_joint_name — are added in skeleton insertion order. FIXED joints are
     * mapped to their parent pinocchio id (markers still attach correctly) but do not
     * contribute any pinocchio joint or DOF.
     *
     * @param skeleton            Full skeleton (read-only).
     * @param freeflyer_joint_name  Skeleton joint that becomes the Pinocchio free-flyer.
     *                              Its world transform is set externally each frame by
     *                              the coordinator; it contributes no DOFs to child state.
     * @param group_names         Groups whose joints form the child subtree. Every
     *                            non-fixed joint in these groups must be a descendant of
     *                            freeflyer_joint_name.
     * @param[out] model          Cleared and rebuilt by this call.
     * @throws std::invalid_argument if freeflyer_joint_name is not in the skeleton, or
     *         if any in-group non-fixed joint is not a descendant of freeflyer_joint_name.
     */
    static void build_subtree_model(Skeleton const& skeleton,
                                    std::string const& freeflyer_joint_name,
                                    std::vector<std::string> const& group_names,
                                    pinocchio::Model& model);

    /**
     * @brief Build marker frame map for a subtree model.
     *
     * Returns a frame map for every marker that is present in the pinocchio model
     * AND whose parent joint is reachable in @p layout (as computed by
     * SkeletonLayout::markers()). This is the single source of truth for which
     * markers are included — no secondary group-membership filtering.
     *
     * @param model   Subtree model built by build_subtree_model().
     * @param layout  Layout corresponding to the subtree (must be derived from the
     *                same skeleton that was passed to build_subtree_model()).
     * @return Map of marker_name → frame_id (only for markers in the layout).
     */
    static std::map<std::string, pinocchio::FrameIndex>
    build_subtree_marker_frame_map(pinocchio::Model const& model, SkeletonLayout const& layout);

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
     * @brief Recursively add subtree joints to model for build_subtree_model().
     *
     * Walks skeleton children of parent_skel_idx. For each child whose group is in
     * group_set (and is non-FIXED), a Pinocchio joint is added. FIXED joints in the
     * subtree are mapped to their parent's Pinocchio id. Non-group, non-FIXED joints
     * that are children of the freeflyer anchor stop the traversal (they are outside
     * the requested subtree).
     */
    static void
    add_subtree_joints_recursive(pinocchio::Model& model, Skeleton const& skeleton,
                                 uint32_t parent_skel_idx, pinocchio::JointIndex parent_pin_id,
                                 std::unordered_set<std::string> const& group_set,
                                 std::map<std::string, pinocchio::JointIndex>& joint_to_id);

    /**
     * @brief Determine rotation axis for revolute joint
     *
     * @param joint The revolute joint
     * @return 'X', 'Y', or 'Z' indicating rotation axis
     */
    static char get_revolute_axis(Joint const& joint);
};

}  // namespace posetrak

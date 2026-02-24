/**
 * @file pinocchio_model_builder.cpp
 * @brief Implementation of Pinocchio model builder
 *
 * Adapted from cpp-tracker-test (proven zero-error implementation)
 * Key preservation: Quaternion ordering [x,y,z,w], root offset handling, frame updates
 */

#include "posetrak/kinematics/pinocchio_model_builder.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/multibody/joint/joint-free-flyer.hpp>
#include <pinocchio/multibody/joint/joint-revolute.hpp>
#include <pinocchio/multibody/joint/joint-spherical.hpp>

#include <fmt/core.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

namespace posetrak {

void PinocchioModelBuilder::build_model(Skeleton const& skeleton, pinocchio::Model& model) {
    // Validate skeleton first
    auto validation_error = skeleton.validate();
    if (validation_error.has_value()) {
        throw std::runtime_error("Cannot build model from invalid skeleton: " +
                                 validation_error.value());
    }

    // Set model name
    model.name = "posetrak_skeleton";

    // Map to track joint name → Pinocchio joint index
    std::map<std::string, pinocchio::JointIndex> joint_to_id;

    // Find the root joint (no parent)
    Joint const* root_joint = nullptr;
    for (auto const& joint : skeleton.joints()) {
        if (!joint.parent_index.has_value()) {
            root_joint = &joint;
            break;
        }
    }

    if (!root_joint) {
        throw std::runtime_error("Skeleton has no root joint (joint with no parent)");
    }
    // Add root joint recursively (starts from universe = 0)
    add_joint_recursive(model, skeleton, *root_joint, 0, joint_to_id);

    // Add marker frames
    add_marker_frames(model, skeleton, joint_to_id);
}

void PinocchioModelBuilder::build_model_and_data(Skeleton const& skeleton, pinocchio::Model& model,
                                                 pinocchio::Data& data) {
    build_model(skeleton, model);
    data = pinocchio::Data(model);
}

void PinocchioModelBuilder::add_joint_recursive(
    pinocchio::Model& model, Skeleton const& skeleton, Joint const& joint,
    pinocchio::JointIndex parent_id, std::map<std::string, pinocchio::JointIndex>& joint_to_id) {
    // Create placement (SE3) for this joint relative to parent
    pinocchio::SE3 placement = pinocchio::SE3::Identity();

    // CRITICAL: Root joint should be at origin, ignore its offset
    // Only non-root joints use offset in SE3 placement
    bool is_root = !joint.parent_index.has_value();
    if (!is_root) {
        placement.translation() = joint.offset;

        // Apply rest orientation if specified (ZYX Euler angles)
        // Check if rest_orientation has non-zero components
        if (joint.rest_orientation.norm() > 0) {
            // Convert ZYX intrinsic Euler angles to rotation matrix
            // ZYX intrinsic means: rotate about Z, then about the new Y, then about the new X
            // This is equivalent to: R = Rx * Ry * Rz in extrinsic (fixed-frame) order
            double z = joint.rest_orientation[0];
            double y = joint.rest_orientation[1];
            double x = joint.rest_orientation[2];

            Eigen::Matrix3d R = Eigen::AngleAxisd(x, Eigen::Vector3d::UnitX()).toRotationMatrix() *
                                Eigen::AngleAxisd(y, Eigen::Vector3d::UnitY()).toRotationMatrix() *
                                Eigen::AngleAxisd(z, Eigen::Vector3d::UnitZ()).toRotationMatrix();

            placement.rotation() = R;

            // Debug output for arm joints
            if (joint.name.find("forearm") != std::string::npos ||
                joint.name.find("upper_arm") != std::string::npos) {
                std::cout << "Joint " << joint.name << " rest orientation (ZYX): [" << z << ", "
                          << y << ", " << x << "]\n";
                std::cout << "Rotation matrix:\n" << R << std::endl;
            }
        }
    }

    pinocchio::JointIndex joint_id;

    // Add joint based on type
    if (is_root) {
        // Root joint: Free-flyer (3 translation + 4 quaternion = 7 config DOF, 6 velocity DOF)
        joint_id =
            model.addJoint(parent_id, pinocchio::JointModelFreeFlyer(), placement, joint.name);
    } else if (joint.type == JointType::SPHERICAL) {
        // Spherical joint: 4 quaternion (4 config DOF, 3 velocity DOF)
        joint_id =
            model.addJoint(parent_id, pinocchio::JointModelSpherical(), placement, joint.name);
    } else if (joint.type == JointType::REVOLUTE) {
        // Single-axis revolute joint
        char axis = get_revolute_axis(joint);

        if (axis == 'X') {
            joint_id = model.addJoint(parent_id, pinocchio::JointModelRX(), placement, joint.name);
        } else if (axis == 'Y') {
            joint_id = model.addJoint(parent_id, pinocchio::JointModelRY(), placement, joint.name);
        } else {  // Z
            joint_id = model.addJoint(parent_id, pinocchio::JointModelRZ(), placement, joint.name);
        }
    } else if (joint.type == JointType::FIXED) {
        // Fixed joint: Skip adding to Pinocchio (markers will attach to parent)
        // Store mapping but don't recurse
        joint_to_id[joint.name] = parent_id;  // Map to parent's ID
        return;
    } else {
        throw std::runtime_error("Unknown joint type for joint: " + joint.name);
    }

    // Add body inertia (minimal for kinematics-only)
    pinocchio::Inertia inertia = pinocchio::Inertia::Identity();
    model.appendBodyToJoint(joint_id, inertia);

    // Store mapping
    joint_to_id[joint.name] = joint_id;

    // Recursively add child joints
    // Find the index of the current joint
    uint32_t current_joint_idx = UINT32_MAX;
    for (uint32_t i = 0; i < skeleton.joints().size(); ++i) {
        if (&skeleton.joints()[i] == &joint) {
            current_joint_idx = i;
            break;
        }
    }

    if (current_joint_idx != UINT32_MAX) {
        for (auto const& child_joint : skeleton.joints()) {
            if (child_joint.parent_index.has_value() &&
                child_joint.parent_index.value() == current_joint_idx) {
                add_joint_recursive(model, skeleton, child_joint, joint_id, joint_to_id);
            }
        }
    }
}

void PinocchioModelBuilder::add_marker_frames(
    pinocchio::Model& model, Skeleton const& skeleton,
    std::map<std::string, pinocchio::JointIndex> const& joint_to_id) {
    for (auto const& marker : skeleton.markers()) {
        // Find parent joint ID using joint index
        std::string const& joint_name = skeleton.joints()[marker.joint_index].name;
        auto it = joint_to_id.find(joint_name);
        if (it == joint_to_id.end()) {
            std::cerr << "Warning: Marker '" << marker.name << "' references unknown joint '"
                      << joint_name << "', skipping" << std::endl;
            continue;
        }

        pinocchio::JointIndex parent_joint_id = it->second;

        // Create frame placement (offset from joint)
        pinocchio::SE3 frame_placement = pinocchio::SE3::Identity();
        frame_placement.translation() = marker.local_pos;

        // Get the parent frame (body frame of the joint)
        pinocchio::FrameIndex parent_frame_id = model.getBodyId(model.names[parent_joint_id]);

        // Add frame
        model.addFrame(pinocchio::Frame(marker.name, parent_joint_id, parent_frame_id,
                                        frame_placement,
                                        pinocchio::OP_FRAME  // Operational frame type
                                        ));
    }
}

char PinocchioModelBuilder::get_revolute_axis(Joint const& joint) {
    if (joint.type != JointType::REVOLUTE) {
        throw std::runtime_error("get_revolute_axis called on non-revolute joint");
    }

    // Determine axis from which limit is active
    // num_limits tells us how many limits are specified (0-3)
    // For revolute joints, we expect num_limits >= 1
    // The first active limit determines the rotation axis

    if (joint.num_limits == 0) {
        // Default to X axis if no limits specified
        return 'X';
    }

    // Check which limit is defined (has non-zero range or is explicitly set)
    // Limits are in order [X, Y, Z]
    for (size_t i = 0; i < joint.num_limits && i < 3; ++i) {
        // If this limit is the first one, use the corresponding axis
        if (i == 0)
            return 'X';
        if (i == 1)
            return 'Y';
        if (i == 2)
            return 'Z';
    }

    // Default to X
    return 'X';
}

int PinocchioModelBuilder::get_marker_frame_id(pinocchio::Model const& model,
                                               std::string const& marker_name) {
    if (model.existFrame(marker_name)) {
        return model.getFrameId(marker_name);
    }
    return -1;
}

std::map<std::string, pinocchio::FrameIndex>
PinocchioModelBuilder::build_marker_frame_map(pinocchio::Model const& model,
                                              Skeleton const& skeleton) {
    std::map<std::string, pinocchio::FrameIndex> marker_map;

    for (auto const& marker : skeleton.markers()) {
        if (model.existFrame(marker.name)) {
            marker_map[marker.name] = model.getFrameId(marker.name);
        } else {
            std::cerr << "Warning: Marker '" << marker.name << "' not found in model" << std::endl;
        }
    }

    return marker_map;
}

// ---------------------------------------------------------------------------
// Subtree model building
// ---------------------------------------------------------------------------

namespace {

/// Return true if ancestor_idx appears anywhere in the ancestor chain of joint_idx.
bool is_descendant_of(Skeleton const& skeleton, uint32_t joint_idx, uint32_t ancestor_idx) {
    auto const& joints = skeleton.joints();
    std::optional<uint32_t> cur = joints[joint_idx].parent_index;
    while (cur.has_value()) {
        if (cur.value() == ancestor_idx)
            return true;
        cur = joints[cur.value()].parent_index;
    }
    return false;
}

}  // anonymous namespace

void PinocchioModelBuilder::add_subtree_joints_recursive(
    pinocchio::Model& model, Skeleton const& skeleton, uint32_t parent_skel_idx,
    pinocchio::JointIndex parent_pin_id, std::unordered_set<std::string> const& group_set,
    std::map<std::string, pinocchio::JointIndex>& joint_to_id) {
    auto const& joints = skeleton.joints();

    for (uint32_t i = 0; i < static_cast<uint32_t>(joints.size()); ++i) {
        auto const& child = joints[i];
        if (!child.parent_index.has_value() || child.parent_index.value() != parent_skel_idx) {
            continue;
        }

        bool const in_group = skeleton.joint_in_groups(child.name, group_set);

        if (child.type == JointType::FIXED) {
            // Fixed joints contribute no pinocchio joint. Map them to parent so
            // that any markers attached here resolve to the correct pinocchio joint.
            joint_to_id[child.name] = parent_pin_id;
            // Still recurse — descendants might be in-group.
            add_subtree_joints_recursive(model, skeleton, i, parent_pin_id, group_set, joint_to_id);
            continue;
        }

        if (!in_group) {
            // Non-group, non-fixed joint — outside requested subtree, stop recursion here.
            continue;
        }

        // Build placement (offset + rest orientation), same as add_joint_recursive().
        pinocchio::SE3 placement = pinocchio::SE3::Identity();
        placement.translation() = child.offset;
        if (child.rest_orientation.norm() > 0) {
            double z = child.rest_orientation[0];
            double y = child.rest_orientation[1];
            double x = child.rest_orientation[2];
            placement.rotation() =
                Eigen::AngleAxisd(x, Eigen::Vector3d::UnitX()).toRotationMatrix() *
                Eigen::AngleAxisd(y, Eigen::Vector3d::UnitY()).toRotationMatrix() *
                Eigen::AngleAxisd(z, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        }

        pinocchio::JointIndex pin_id;
        if (child.type == JointType::SPHERICAL) {
            pin_id = model.addJoint(parent_pin_id, pinocchio::JointModelSpherical(), placement,
                                    child.name);
        } else {  // REVOLUTE
            char axis = get_revolute_axis(child);
            if (axis == 'X')
                pin_id =
                    model.addJoint(parent_pin_id, pinocchio::JointModelRX(), placement, child.name);
            else if (axis == 'Y')
                pin_id =
                    model.addJoint(parent_pin_id, pinocchio::JointModelRY(), placement, child.name);
            else
                pin_id =
                    model.addJoint(parent_pin_id, pinocchio::JointModelRZ(), placement, child.name);
        }

        model.appendBodyToJoint(pin_id, pinocchio::Inertia::Identity());
        joint_to_id[child.name] = pin_id;

        add_subtree_joints_recursive(model, skeleton, i, pin_id, group_set, joint_to_id);
    }
}

void PinocchioModelBuilder::build_subtree_model(Skeleton const& skeleton,
                                                std::string const& freeflyer_joint_name,
                                                std::vector<std::string> const& group_names,
                                                pinocchio::Model& model) {
    // --- 1. Find freeflyer joint index ---
    uint32_t ff_idx = UINT32_MAX;
    auto const& joints = skeleton.joints();
    for (uint32_t i = 0; i < static_cast<uint32_t>(joints.size()); ++i) {
        if (joints[i].name == freeflyer_joint_name) {
            ff_idx = i;
            break;
        }
    }
    if (ff_idx == UINT32_MAX) {
        throw std::invalid_argument(
            fmt::format("build_subtree_model: freeflyer joint '{}' not found in skeleton",
                        freeflyer_joint_name));
    }

    // --- 2. Build group set ---
    std::unordered_set<std::string> group_set(group_names.begin(), group_names.end());

    // --- 3. Connectivity assertion ---
    // Every non-fixed in-group joint must be a descendant of freeflyer_joint_name.
    for (uint32_t i = 0; i < static_cast<uint32_t>(joints.size()); ++i) {
        auto const& j = joints[i];
        if (!skeleton.joint_in_groups(j.name, group_set))
            continue;
        if (j.type == JointType::FIXED)
            continue;
        if (!is_descendant_of(skeleton, i, ff_idx)) {
            throw std::invalid_argument(
                fmt::format("build_subtree_model: joint '{}' is not a descendant of '{}'", j.name,
                            freeflyer_joint_name));
        }
    }

    // --- 4. Build pinocchio model ---
    model = pinocchio::Model{};
    model.name = fmt::format("subtree_{}", freeflyer_joint_name);

    std::map<std::string, pinocchio::JointIndex> joint_to_id;

    // Add freeflyer at universe (placement = Identity; world transform injected each frame)
    pinocchio::JointIndex ff_pin_id = model.addJoint(
        0, pinocchio::JointModelFreeFlyer(), pinocchio::SE3::Identity(), freeflyer_joint_name);
    model.appendBodyToJoint(ff_pin_id, pinocchio::Inertia::Identity());
    joint_to_id[freeflyer_joint_name] = ff_pin_id;

    // Recursively add in-group joints rooted at the freeflyer.
    add_subtree_joints_recursive(model, skeleton, ff_idx, ff_pin_id, group_set, joint_to_id);

    // --- 5. Add marker frames for the subtree ---
    // Reuse add_marker_frames — it attaches every skeleton marker whose parent joint
    // is present in joint_to_id.  Markers outside the subtree are warned and skipped.
    // Override: only attach markers on joints in group_set or the freeflyer itself.
    for (auto const& marker : skeleton.markers()) {
        std::string const& parent_joint_name = joints[marker.joint_index].name;
        auto it = joint_to_id.find(parent_joint_name);
        if (it == joint_to_id.end())
            continue;  // marker is outside the subtree — skip silently

        pinocchio::JointIndex parent_joint_id = it->second;
        pinocchio::SE3 frame_placement = pinocchio::SE3::Identity();
        frame_placement.translation() = marker.local_pos;
        pinocchio::FrameIndex parent_frame_id = model.getBodyId(model.names[parent_joint_id]);
        model.addFrame(pinocchio::Frame(marker.name, parent_joint_id, parent_frame_id,
                                        frame_placement, pinocchio::OP_FRAME));
    }
}

std::map<std::string, pinocchio::FrameIndex>
PinocchioModelBuilder::build_subtree_marker_frame_map(pinocchio::Model const& model,
                                                      SkeletonLayout const& layout) {
    // The layout already knows exactly which markers belong to this subtree
    // (populated by SkeletonLayout::build() via the reachable-joint set logic that
    // mirrors add_subtree_joints_recursive).  Just look each one up in the model.
    std::map<std::string, pinocchio::FrameIndex> marker_map;
    for (MarkerDesc const& m : layout.markers()) {
        if (model.existFrame(m.name)) {
            marker_map[m.name] = model.getFrameId(m.name);
        }
    }
    return marker_map;
}

// ---------------------------------------------------------------------------

void PinocchioModelBuilder::print_model_info(pinocchio::Model const& model,
                                             Skeleton const& skeleton) {
    std::cout << "\n=== Pinocchio Model Information ===" << std::endl;
    std::cout << "Model name: " << model.name << std::endl;
    std::cout << "\nDimensionality:" << std::endl;
    std::cout << "  nq (config DOF):  " << model.nq << std::endl;
    std::cout << "  nv (velocity DOF): " << model.nv << std::endl;
    std::cout << "  Skeleton total DOF: " << skeleton.total_dof() << std::endl;

    std::cout << "\nJoints:" << std::endl;
    std::cout << "  Model joints: " << model.njoints << " (including universe)" << std::endl;
    std::cout << "  Skeleton joints: " << skeleton.joints().size() << std::endl;

    std::cout << "\nFrames:" << std::endl;
    std::cout << "  Model frames: " << model.nframes << std::endl;
    std::cout << "  Skeleton markers: " << skeleton.markers().size() << std::endl;

    std::cout << "\nJoint Details:" << std::endl;
    for (pinocchio::JointIndex i = 1; i < static_cast<pinocchio::JointIndex>(model.njoints); ++i) {
        std::cout << "  [" << i << "] " << model.names[i] << " (parent: " << model.parents[i] << ")"
                  << " nq=" << model.nqs[i] << " nv=" << model.nvs[i] << std::endl;
    }

    std::cout << "\nOperational Frames (markers):" << std::endl;
    int op_frame_count = 0;
    for (pinocchio::FrameIndex i = 0; i < static_cast<pinocchio::FrameIndex>(model.nframes); ++i) {
        auto const& frame = model.frames[i];
        if (frame.type == pinocchio::OP_FRAME) {
            std::cout << "  [" << i << "] " << frame.name
                      << " (joint: " << model.names[frame.parentJoint] << ")" << std::endl;
            op_frame_count++;
        }
    }
    std::cout << "Total operational frames: " << op_frame_count << std::endl;
}

}  // namespace posetrak

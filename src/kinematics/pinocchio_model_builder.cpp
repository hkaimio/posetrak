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

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

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

    // Find the root joint (empty parent string)
    Joint const* root_joint = nullptr;
    for (auto const& [name, joint] : skeleton.joints()) {
        if (joint.parent.empty()) {
            root_joint = &joint;
            break;
        }
    }

    if (!root_joint) {
        throw std::runtime_error("Skeleton has no root joint (joint with empty parent)");
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
    bool is_root = joint.parent.empty();
    if (!is_root) {
        placement.translation() = joint.offset;

        // Apply rest orientation if specified (ZYX Euler angles)
        if (joint.has_rest_orientation) {
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
    for (auto const& [child_name, child_joint] : skeleton.joints()) {
        if (child_joint.parent == joint.name) {
            add_joint_recursive(model, skeleton, child_joint, joint_id, joint_to_id);
        }
    }
}

void PinocchioModelBuilder::add_marker_frames(
    pinocchio::Model& model, Skeleton const& skeleton,
    std::map<std::string, pinocchio::JointIndex> const& joint_to_id) {
    for (auto const& [marker_name, marker] : skeleton.markers()) {
        // Find parent joint ID
        auto it = joint_to_id.find(marker.joint);
        if (it == joint_to_id.end()) {
            std::cerr << "Warning: Marker '" << marker.name << "' references unknown joint '"
                      << marker.joint << "', skipping" << std::endl;
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

    for (auto const& [marker_name, marker] : skeleton.markers()) {
        if (model.existFrame(marker.name)) {
            marker_map[marker.name] = model.getFrameId(marker.name);
        } else {
            std::cerr << "Warning: Marker '" << marker.name << "' not found in model" << std::endl;
        }
    }

    return marker_map;
}

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

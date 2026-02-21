/**
 * @file forward_kinematics.cpp
 * @brief Forward kinematics computation using Pinocchio
 *
 * Adapted from cpp-tracker-test (proven zero-error implementation)
 * Key preservation: Quaternion [x,y,z,w] order, angle-axis conversion for spherical joints
 */

#include "posetrak/kinematics/forward_kinematics.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <iostream>
#include <stdexcept>

namespace posetrak {

ForwardKinematics::ForwardKinematics(
    pinocchio::Model const& model, pinocchio::Data& data,
    std::map<std::string, pinocchio::FrameIndex> const& marker_frame_map,
    std::shared_ptr<const SkeletonLayout> layout)
    : model_(model), data_(data), marker_frame_map_(marker_frame_map), layout_(std::move(layout)) {
    // Populate joint_id_map_ from pinocchio joint names.
    // Index 0 is universe (skip); 1..njoints-1 are real joints.
    for (pinocchio::JointIndex i = 1; i < static_cast<pinocchio::JointIndex>(model_.njoints); ++i) {
        joint_id_map_[model_.names[i]] = i;
    }
}

std::pair<Eigen::Vector3d, Eigen::Quaterniond>
ForwardKinematics::world_transform(std::string const& joint_name) const {
    auto it = joint_id_map_.find(joint_name);
    if (it == joint_id_map_.end()) {
        throw std::out_of_range("world_transform: unknown joint '" + joint_name + "'");
    }
    pinocchio::SE3 const& T = data_.oMi[it->second];
    return {T.translation(), Eigen::Quaterniond(T.rotation())};
}

std::unordered_map<std::string, Eigen::Vector3d> ForwardKinematics::compute(State const& state) {
    Eigen::VectorXd q = state_to_config(state, *layout_);
    return compute(q);
}

std::unordered_map<std::string, Eigen::Vector3d>
ForwardKinematics::compute(Eigen::VectorXd const& q) const {
    // Ensure configuration has correct dimensions
    if (q.size() != model_.nq) {
        throw std::runtime_error("Configuration vector size mismatch: expected " +
                                 std::to_string(model_.nq) + ", got " + std::to_string(q.size()));
    }

    // CRITICAL: Compute forward kinematics
    pinocchio::forwardKinematics(model_, data_, q);

    // CRITICAL: Update all frame placements (including marker frames)
    // Forgetting this step results in incorrect marker positions!
    pinocchio::updateFramePlacements(model_, data_);

    // Extract marker positions
    std::unordered_map<std::string, Eigen::Vector3d> marker_positions;

    for (auto const& [marker_name, frame_id] : marker_frame_map_) {
        // Get frame transform in world frame
        auto const& frame_transform = data_.oMf[frame_id];
        marker_positions[marker_name] = frame_transform.translation();
    }

    return marker_positions;
}

Eigen::VectorXd ForwardKinematics::state_to_config(State const& state,
                                                   SkeletonLayout const& layout) {
    // Both full-skeleton and child-subtree pinocchio models open with a 7-DOF freeflyer
    // (pos(3) + quat_xyzw(4)), followed by layout joints in skeleton insertion order.
    int nq = 7;
    for (auto const& desc : layout.joints()) {
        nq += (desc.type == JointType::SPHERICAL) ? 4 : 1;
    }

    Eigen::VectorXd q(nq);

    // Root freeflyer: pos + quaternion.
    // For the full-skeleton path these are the root body's pose.
    // For the child path the coordinator injects the freeflyer world transform here.
    q.segment<3>(0) = state.root_position();
    Eigen::Quaterniond const& ori = state.root_orientation();
    q[3] = ori.x();
    q[4] = ori.y();
    q[5] = ori.z();
    q[6] = ori.w();

    int idx = 7;
    for (auto const& desc : layout.joints()) {
        if (desc.type == JointType::SPHERICAL) {
            Eigen::Vector3d angles =
                state.joint_angles().segment<3>(static_cast<int>(desc.state_index));
            double const angle = angles.norm();
            Eigen::Quaterniond quat;
            if (angle == 0.0) {
                quat = Eigen::Quaterniond::Identity();
            } else {
                quat = Eigen::Quaterniond(Eigen::AngleAxisd(angle, angles / angle));
            }
            q[idx++] = quat.x();
            q[idx++] = quat.y();
            q[idx++] = quat.z();
            q[idx++] = quat.w();
        } else {  // REVOLUTE
            q[idx++] = state.joint_angles()[static_cast<int>(desc.state_index)];
        }
    }
    return q;
}

Eigen::VectorXd ForwardKinematics::state_to_config(State const& state, Skeleton const& skeleton) {
    // Get ordered joints for consistent indexing
    auto joints_ordered = skeleton.get_joints_ordered();

    // Calculate required configuration size
    int nq = 0;

    // Root joint (if exists): 7 DOF (3 position + 4 quaternion)
    bool has_root = false;
    for (auto const& joint : joints_ordered) {
        if (!joint.parent_index.has_value()) {
            has_root = true;
            nq += 7;  // position (3) + quaternion (4)
            break;
        }
    }

    // Count DOF for other joints
    for (auto const& joint : joints_ordered) {
        if (joint.parent_index.has_value()) {  // Skip root (already counted)
            if (joint.type == JointType::SPHERICAL) {
                nq += 4;  // quaternion
            } else if (joint.type == JointType::REVOLUTE) {
                nq += 1;  // single angle
            }
            // FIXED joints: 0 DOF, skip
        }
    }

    Eigen::VectorXd q(nq);
    int idx = 0;

    // Process root joint first if it exists
    if (has_root) {
        // Root position (3)
        q.segment<3>(idx) = state.root_position();
        idx += 3;

        // Root orientation quaternion (4) - CRITICAL: Pinocchio uses [x, y, z, w] order
        auto const& quat = state.root_orientation();
        q[idx++] = quat.x();
        q[idx++] = quat.y();
        q[idx++] = quat.z();
        q[idx++] = quat.w();
    }

    // Process other joints in order
    int joint_angle_idx = 0;
    for (auto const& joint : joints_ordered) {
        if (!joint.parent_index.has_value()) {
            // Skip root (already processed)
            continue;
        }

        if (joint.type == JointType::SPHERICAL) {
            // Spherical joint: 3 rotation angles in state → 4 quaternion in config
            // Get the 3 angles from state
            Eigen::Vector3d angles = state.joint_angles().segment<3>(joint_angle_idx);
            joint_angle_idx += 3;

            // Convert to quaternion via angle-axis
            // CRITICAL: This is the proven conversion from cpp-tracker-test
            double const angle = angles.norm();
            Eigen::Quaterniond quat;

            if (angle == 0.0) {
                // No rotation
                quat = Eigen::Quaterniond::Identity();
            } else {
                Eigen::Vector3d const axis = angles / angle;  // normalized axis
                quat = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
            }

            // CRITICAL: Pinocchio uses [x, y, z, w] order
            q[idx++] = quat.x();
            q[idx++] = quat.y();
            q[idx++] = quat.z();
            q[idx++] = quat.w();

        } else if (joint.type == JointType::REVOLUTE) {
            // Revolute joint: single angle
            q[idx++] = state.joint_angles()[joint_angle_idx++];
        }
        // FIXED joints: no config DOF, skip
    }

    if (idx != nq) {
        throw std::runtime_error(
            "Configuration vector size mismatch during construction: expected " +
            std::to_string(nq) + " elements, but filled " + std::to_string(idx));
    }

    if (joint_angle_idx != state.joint_angles().size()) {
        throw std::runtime_error("Joint angle count mismatch: state has " +
                                 std::to_string(state.joint_angles().size()) +
                                 " angles, but processed " + std::to_string(joint_angle_idx));
    }

    return q;
}

}  // namespace posetrak

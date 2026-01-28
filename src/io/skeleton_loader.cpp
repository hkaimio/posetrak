#include <posetrak/io/skeleton_loader.hpp>

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace posetrak {

namespace {

/// @brief Parse joint type from string
JointType parse_joint_type(std::string const& type_str) {
    if (type_str == "root" || type_str == "fixed")
        return JointType::FIXED;
    if (type_str == "revolute")
        return JointType::REVOLUTE;
    if (type_str == "ball" || type_str == "spherical")
        return JointType::SPHERICAL;
    throw std::runtime_error("Unknown joint type: " + type_str);
}

/// @brief Parse Vec3 from YAML node
Eigen::Vector3d parse_vec3(YAML::Node const& node) {
    if (!node.IsSequence() || node.size() != 3) {
        throw std::runtime_error("Expected 3-element array for Vec3");
    }
    return Eigen::Vector3d(node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
}

}  // anonymous namespace

Skeleton load_skeleton_from_yaml(std::string const& filepath) {
    // Load YAML file
    YAML::Node root;
    try {
        root = YAML::LoadFile(filepath);
    } catch (YAML::Exception const& e) {
        throw std::runtime_error("Failed to load YAML file '" + filepath + "': " + e.what());
    }

    // Parse skeleton name
    std::string name = root["name"].as<std::string>("unnamed_skeleton");

    // Create skeleton
    Skeleton skeleton;

    // Parse joints
    if (!root["joints"]) {
        throw std::runtime_error("YAML file missing 'joints' section");
    }

    std::unordered_map<std::string, uint32_t> joint_name_to_idx;

    for (auto const& joint_node : root["joints"]) {
        std::string joint_name = joint_node["name"].as<std::string>();
        std::string type_str = joint_node["type"].as<std::string>();
        JointType type = parse_joint_type(type_str);

        // Parse parent
        std::optional<uint32_t> parent_index;
        if (joint_node["parent"] && !joint_node["parent"].IsNull()) {
            std::string parent_name = joint_node["parent"].as<std::string>();
            auto it = joint_name_to_idx.find(parent_name);
            if (it == joint_name_to_idx.end()) {
                throw std::runtime_error("Parent joint '" + parent_name +
                                         "' not found for joint '" + joint_name + "'");
            }
            parent_index = it->second;
        }

        // Parse offset
        Eigen::Vector3d offset = Eigen::Vector3d::Zero();
        if (joint_node["offset"]) {
            offset = parse_vec3(joint_node["offset"]);
        }

        // Parse group if present
        std::string group = "";
        if (joint_node["group"]) {
            group = joint_node["group"].as<std::string>();
        }

        // Parse rest orientation (ZYX Euler angles in radians)
        Eigen::Vector3d rest_orientation = Eigen::Vector3d::Zero();
        if (joint_node["orientation"]) {
            rest_orientation = parse_vec3(joint_node["orientation"]);
        }

        // Add joint to skeleton
        uint32_t joint_idx =
            skeleton.add_joint(joint_name, parent_index, type, offset, group, rest_orientation);
        joint_name_to_idx[joint_name] = joint_idx;

        // Parse and set limits if present
        if (joint_node["limits"]) {
            std::array<Eigen::Vector2d, 3> limits;
            size_t num_limits = 0;

            if (type == JointType::REVOLUTE) {
                // Revolute: [min, max] array
                auto const& limits_node = joint_node["limits"];
                if (limits_node.IsSequence() && limits_node.size() == 2) {
                    limits[0] =
                        Eigen::Vector2d(limits_node[0].as<double>(), limits_node[1].as<double>());
                    num_limits = 1;
                }
            } else if (type == JointType::SPHERICAL) {
                // Spherical: {x: [min, max], y: [min, max], z: [min, max]}
                auto const& limits_node = joint_node["limits"];
                if (limits_node.IsMap()) {
                    size_t axis_idx = 0;
                    for (auto const& axis : {"x", "y", "z"}) {
                        if (limits_node[axis]) {
                            auto const& axis_limits = limits_node[axis];
                            if (axis_limits.IsSequence() && axis_limits.size() == 2) {
                                limits[axis_idx] = Eigen::Vector2d(axis_limits[0].as<double>(),
                                                                   axis_limits[1].as<double>());
                                ++axis_idx;
                            }
                        }
                    }
                    num_limits = axis_idx;
                    // Ensure we have exactly 3 limits for spherical joints
                    if (num_limits != 3) {
                        throw std::runtime_error("Spherical joint '" + joint_name +
                                                 "' must have limits for all 3 axes (x, y, z)");
                    }
                }
            }

            if (num_limits > 0) {
                skeleton.set_joint_limits(joint_idx, limits, num_limits);
            }
        }
    }
    // Parse markers
    if (root["markers"]) {
        for (auto const& marker_node : root["markers"]) {
            std::string marker_name = marker_node["name"].as<std::string>();
            std::string parent_name = marker_node["parent"].as<std::string>();

            // Find joint index
            auto it = joint_name_to_idx.find(parent_name);
            if (it == joint_name_to_idx.end()) {
                throw std::runtime_error("Marker '" + marker_name + "' references unknown joint '" +
                                         parent_name + "'");
            }

            Eigen::Vector3d offset = Eigen::Vector3d::Zero();
            if (marker_node["offset"]) {
                offset = parse_vec3(marker_node["offset"]);
            }

            std::optional<int> coco_id;
            if (marker_node["openpose_keypoint"]) {
                coco_id = static_cast<int>(marker_node["openpose_keypoint"].as<size_t>());
            }

            skeleton.add_marker(marker_name, it->second, offset, coco_id);
        }
    }

    // Parse groups and apply active joints
    if (root["tracking"] && root["tracking"]["active_groups"]) {
        std::unordered_set<std::string> active_joint_names;

        for (auto const& group_name_node : root["tracking"]["active_groups"]) {
            std::string group_name = group_name_node.as<std::string>();

            // Find the group definition
            if (!root["groups"]) {
                throw std::runtime_error(
                    "tracking.active_groups references groups but no 'groups' section exists");
            }

            bool found = false;
            for (auto const& group_node : root["groups"]) {
                if (group_node["name"].as<std::string>() == group_name) {
                    found = true;
                    if (group_node["joints"]) {
                        for (auto const& joint_name_node : group_node["joints"]) {
                            active_joint_names.insert(joint_name_node.as<std::string>());
                        }
                    }
                    break;
                }
            }

            if (!found) {
                throw std::runtime_error("Active group '" + group_name +
                                         "' not found in groups section");
            }
        }

        // Convert names to joint names vector
        std::vector<std::string> active_joint_names_vec(active_joint_names.begin(),
                                                        active_joint_names.end());

        if (!active_joint_names_vec.empty()) {
            skeleton.set_active_joints(active_joint_names_vec);
        }
    }

    return skeleton;
}

}  // namespace posetrak

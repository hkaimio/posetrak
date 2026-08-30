// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include <posetrak/io/skeleton_loader.hpp>

#include <fmt/core.h>
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
    if (type_str == "prismatic")
        return JointType::PRISMATIC;
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

/// @brief Parse a skeleton from an already-loaded YAML::Node (shared implementation)
static Skeleton parse_skeleton_node(YAML::Node const& root);

Skeleton load_skeleton_from_yaml(std::string const& filepath) {
    // Load YAML file
    YAML::Node root;
    try {
        root = YAML::LoadFile(filepath);
    } catch (YAML::Exception const& e) {
        throw std::runtime_error("Failed to load YAML file '" + filepath + "': " + e.what());
    }
    return parse_skeleton_node(root);
}

Skeleton load_skeleton_from_yaml_string(std::string const& yaml_content) {
    YAML::Node root;
    try {
        root = YAML::Load(yaml_content);
    } catch (YAML::Exception const& e) {
        throw std::runtime_error(std::string("Failed to parse skeleton YAML: ") + e.what());
    }
    return parse_skeleton_node(root);
}

static Skeleton parse_skeleton_node(YAML::Node const& root) {
    // Parse skeleton name
    std::string name = root["name"].as<std::string>("unnamed_skeleton");

    // Create skeleton
    Skeleton skeleton;

    // Parse input_tracks: (design §5.1) -- a skeleton with none behaves
    // exactly as today (openpose_keypoint/coco_id-only markers).
    if (root["input_tracks"]) {
        for (auto const& track_node : root["input_tracks"]) {
            std::string track_id = track_node["id"].as<std::string>();
            std::string track_type = track_node["type"].as<std::string>("");
            skeleton.add_input_track(track_id, track_type);
        }
    }

    // Parse groups section first to build joint-to-group and marker-to-group mappings
    std::unordered_map<std::string, std::string> joint_to_group_map;
    std::unordered_map<std::string, std::string> marker_to_group_map;
    std::unordered_map<std::string, std::vector<std::string>>
        group_dependencies;  // group -> depends_on list
    std::unordered_set<std::string> optional_groups;

    if (root["groups"]) {
        for (auto const& group_node : root["groups"]) {
            std::string group_name = group_node["name"].as<std::string>();

            // Parse optional attribute (defaults to true if not specified)
            bool is_optional = true;
            if (group_node["optional"]) {
                is_optional = group_node["optional"].as<bool>();
            }
            if (is_optional) {
                optional_groups.insert(group_name);
            }

            // Parse depends_on attribute — accepts both a scalar string and a sequence
            if (group_node["depends_on"]) {
                auto const& dep_node = group_node["depends_on"];
                std::vector<std::string> deps;
                if (dep_node.IsSequence()) {
                    for (auto const& item : dep_node) {
                        deps.push_back(item.as<std::string>());
                    }
                } else {
                    deps.push_back(dep_node.as<std::string>());
                }
                group_dependencies[group_name] = std::move(deps);
            }

            // Map all joints in this group to the group name (Joint::group --
            // last-declared-wins "primary" group per joint) and keep the raw
            // per-group list (SkeletonGroup::joints -- supports a joint
            // belonging to more than one group; see Skeleton::is_joint_in_groups()).
            std::vector<std::string> group_joint_names;
            if (group_node["joints"]) {
                for (auto const& joint_name_node : group_node["joints"]) {
                    std::string joint_name = joint_name_node.as<std::string>();
                    joint_to_group_map[joint_name] = group_name;
                    group_joint_names.push_back(joint_name);
                }
            }

            // Same idea for markers.
            std::vector<std::string> group_marker_names;
            if (group_node["markers"]) {
                for (auto const& marker_name_node : group_node["markers"]) {
                    std::string marker_name = marker_name_node.as<std::string>();
                    marker_to_group_map[marker_name] = group_name;
                    group_marker_names.push_back(marker_name);
                }
            }

            // Hierarchical-solver child-stage metadata (both optional; empty for
            // groups like "main" that are not a child stage). See
            // docs/skeleton-format.md's "groups:" section and SkeletonGroup's
            // doc comment for what these mean.
            std::string freeflyer_joint = group_node["freeflyer_joint"].as<std::string>("");
            std::string ref_marker = group_node["ref_marker"].as<std::string>("");
            skeleton.add_group(group_name, group_joint_names, group_marker_names, freeflyer_joint,
                               ref_marker);
        }
    }

    // Parse joints
    if (!root["joints"]) {
        throw std::runtime_error("YAML file missing 'joints' section");
    }

    // Pre-parse scale_groups to know which joints need a prismatic joint inserted before them
    // Maps joint_name -> scale_group_name so the prismatic joint knows its group.
    std::unordered_map<std::string, std::string> scale_group_joints;  // joint_name -> group_name
    std::unordered_map<std::string, Eigen::Vector2d>
        scale_group_limits;                                    // group_name -> [min, max]
    std::unordered_set<std::string> scale_group_leaders_seen;  // groups that already have a leader
    if (root["scale_groups"]) {
        for (auto const& sg_node : root["scale_groups"]) {
            if (sg_node["joints"]) {
                std::string const sg_name = sg_node["name"].as<std::string>();
                for (auto const& jname_node : sg_node["joints"]) {
                    scale_group_joints[jname_node.as<std::string>()] = sg_name;
                }
                // Optional per-group scale factor limits [min, max].
                // Defaults to [0.3, 3.0] to prevent negative/runaway scale factors.
                if (sg_node["limits"] && sg_node["limits"].IsSequence() &&
                    sg_node["limits"].size() == 2) {
                    scale_group_limits[sg_name] = Eigen::Vector2d(
                        sg_node["limits"][0].as<double>(), sg_node["limits"][1].as<double>());
                } else {
                    scale_group_limits[sg_name] = Eigen::Vector2d(0.3, 3.0);
                }
            }
        }
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

        // Parse group if present (from individual joint or from groups section)
        std::string group = "";
        if (joint_node["group"]) {
            group = joint_node["group"].as<std::string>();
        } else if (joint_to_group_map.count(joint_name) > 0) {
            group = joint_to_group_map[joint_name];
        }

        // Parse rest orientation (ZYX Euler angles in radians)
        Eigen::Vector3d rest_orientation = Eigen::Vector3d::Zero();
        if (joint_node["orientation"]) {
            rest_orientation = parse_vec3(joint_node["orientation"]);
        }

        // Add joint to skeleton
        // If this joint is in a scale group, insert a prismatic joint first so that the
        // bone-length DOF can be calibrated.  The prismatic joint slides along
        // normalize(original_offset) and the child joint's offset becomes zero.
        if (scale_group_joints.count(joint_name) && parent_index.has_value()) {
            double const offset_norm = offset.norm();
            if (offset_norm < 1e-6) {
                throw std::runtime_error(
                    "Joint '" + joint_name +
                    "' in scale_groups has a near-zero offset; cannot determine prismatic axis");
            }
            Eigen::Vector3d const axis = offset / offset_norm;
            std::string const pris_name = "prismatic_" + joint_name;
            std::string const sg_name = scale_group_joints.at(joint_name);
            uint32_t pris_idx = skeleton.add_joint(pris_name, parent_index, JointType::PRISMATIC,
                                                   Eigen::Vector3d::Zero(), group);
            skeleton.set_joint_axis(pris_idx, axis);
            skeleton.set_joint_nominal_length(pris_idx, offset_norm);
            skeleton.set_joint_scale_group(pris_idx, sg_name);
            // Mark as follower if another joint in the same group has already been seen.
            // Followers share the leader's state slot and don't get their own DOF in the state.
            bool const is_follower = scale_group_leaders_seen.count(sg_name) > 0;
            if (!is_follower) {
                scale_group_leaders_seen.insert(sg_name);
            }
            skeleton.set_joint_scale_follower(pris_idx, is_follower);
            // Apply per-group scale factor limits to prevent geometry inversion (s < 0).
            // Limits are parsed from YAML `limits:` field or defaulted to [0.3, 3.0].
            {
                Eigen::Vector2d const lim = scale_group_limits.count(sg_name)
                                                ? scale_group_limits.at(sg_name)
                                                : Eigen::Vector2d(0.3, 3.0);
                std::array<Eigen::Vector2d, 3> limits_arr{lim, Eigen::Vector2d::Zero(),
                                                          Eigen::Vector2d::Zero()};
                skeleton.set_joint_limits(pris_idx, limits_arr, 1);
            }
            joint_name_to_idx[pris_name] = pris_idx;

            // Redirect child: its parent is now the prismatic joint; offset absorbed into q
            parent_index = pris_idx;
            offset = Eigen::Vector3d::Zero();
        }

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

            // track/landmark (design §5.1): a marker bound to a dynamic
            // observation source instead of (or in addition to) coco_id.
            std::string track = marker_node["track"].as<std::string>("");
            std::string landmark = marker_node["landmark"].as<std::string>("");
            if (!track.empty() && skeleton.get_input_track(track) == nullptr) {
                throw std::runtime_error("Marker '" + marker_name +
                                         "' references undeclared track '" + track +
                                         "' (add it to input_tracks:)");
            }

            uint32_t marker_idx =
                skeleton.add_marker(marker_name, it->second, offset, coco_id, track, landmark);

            // Assign group from groups section if defined
            if (marker_to_group_map.count(marker_name) > 0) {
                skeleton.markers()[marker_idx].group = marker_to_group_map[marker_name];
            }
        }
    }

    // Warn (don't fail the load) about groups: entries naming a joint or marker
    // that doesn't actually exist in this skeleton. Historically this was a
    // silent no-op -- the joint_to_group_map/marker_to_group_map lookups above
    // just never match -- which let stale entries (e.g. a renamed joint) survive
    // undetected. Now that group membership is load-bearing for the
    // hierarchical solver (SkeletonLayout::from_groups() building a real
    // child-filter subset), a stale reference silently produces a wrong or
    // empty subtree instead of a wrong-but-harmless lookup miss.
    for (auto const& [joint_name, group_name] : joint_to_group_map) {
        if (skeleton.get_joint(joint_name) == nullptr) {
            fmt::print(stderr,
                       "WARNING: skeleton '{}': groups: entry '{}' references joint '{}', "
                       "which does not exist in this skeleton\n",
                       name, group_name, joint_name);
        }
    }
    for (auto const& [marker_name, group_name] : marker_to_group_map) {
        if (skeleton.get_marker(marker_name) == nullptr) {
            fmt::print(stderr,
                       "WARNING: skeleton '{}': groups: entry '{}' references marker '{}', "
                       "which does not exist in this skeleton\n",
                       name, group_name, marker_name);
        }
    }

    return skeleton;
}

}  // namespace posetrak

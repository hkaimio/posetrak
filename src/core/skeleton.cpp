#include "posetrak/core/skeleton.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace posetrak {

// Skeleton implementation

uint32_t Skeleton::add_joint(std::string const& name, std::optional<uint32_t> parent_index,
                             JointType type, Eigen::Vector3d const& offset,
                             std::string const& group, Eigen::Vector3d const& rest_orientation) {
    // Check for duplicates
    for (auto const& existing : joints_) {
        if (existing.name == name) {
            throw std::invalid_argument(fmt::format("Joint '{}' already exists", name));
        }
    }

    // Validate parent index
    if (parent_index.has_value() && parent_index.value() >= joints_.size()) {
        throw std::invalid_argument(
            fmt::format("Invalid parent index {} for joint '{}'", parent_index.value(), name));
    }

    // Determine DOF and limits
    int dof = (type == JointType::FIXED ? 0 : (type == JointType::SPHERICAL ? 3 : 1));
    std::array<Eigen::Vector2d, 3> limits;
    size_t num_limits = 0;

    if (type == JointType::REVOLUTE) {
        limits[0] = Eigen::Vector2d(-M_PI, M_PI);
        num_limits = 1;
    } else if (type == JointType::SPHERICAL) {
        limits[0] = Eigen::Vector2d(-M_PI, M_PI);
        limits[1] = Eigen::Vector2d(-M_PI, M_PI);
        limits[2] = Eigen::Vector2d(-M_PI, M_PI);
        num_limits = 3;
    }

    uint32_t index = static_cast<uint32_t>(joints_.size());
    joints_.push_back(
        Joint{name, parent_index, type, dof, limits, num_limits, group, offset, rest_orientation});

    return index;
}

uint32_t Skeleton::add_marker(std::string const& name, uint32_t joint_index,
                              Eigen::Vector3d const& local_pos, std::optional<int> coco_id) {
    // Check for duplicates
    for (auto const& existing : markers_) {
        if (existing.name == name) {
            throw std::invalid_argument(fmt::format("Marker '{}' already exists", name));
        }
    }

    // Validate joint index
    if (joint_index >= joints_.size()) {
        throw std::invalid_argument(
            fmt::format("Invalid joint index {} for marker '{}'", joint_index, name));
    }

    uint32_t index = static_cast<uint32_t>(markers_.size());
    markers_.push_back(Marker{name, joint_index, local_pos, coco_id});

    return index;
}

void Skeleton::set_joint_limits(uint32_t joint_index, std::array<Eigen::Vector2d, 3> const& limits,
                                size_t num_limits) {
    if (joint_index >= joints_.size()) {
        throw std::invalid_argument(
            fmt::format("Invalid joint index {} for set_joint_limits", joint_index));
    }

    joints_[joint_index].limits = limits;
    joints_[joint_index].num_limits = num_limits;
}

std::optional<std::string> Skeleton::validate() const {
    if (joints_.empty()) {
        return "Skeleton has no joints";
    }

    // Check all parent indices are valid
    for (size_t i = 0; i < joints_.size(); ++i) {
        auto const& joint = joints_[i];
        if (joint.parent_index.has_value()) {
            if (joint.parent_index.value() >= joints_.size()) {
                return fmt::format("Joint '{}' has invalid parent index {}", joint.name,
                                   joint.parent_index.value());
            }
        }
    }

    // Check for cycles
    std::unordered_set<uint32_t> visited;
    for (uint32_t i = 0; i < joints_.size(); ++i) {
        visited.clear();
        if (detect_cycle(i, visited)) {
            return fmt::format("Cycle detected in hierarchy at joint '{}'", joints_[i].name);
        }
    }

    // Find root
    auto root_idx = find_root();
    if (!root_idx.has_value()) {
        return "No unique root joint found (joint with no parent)";
    }

    // Check all marker joint indices are valid
    for (auto const& marker : markers_) {
        if (marker.joint_index >= joints_.size()) {
            return fmt::format("Marker '{}' has invalid joint index {}", marker.name,
                               marker.joint_index);
        }
    }

    return std::nullopt;
}

int Skeleton::total_dof() const {
    int total = 0;
    for (auto const& joint : joints_) {
        total += joint.dof;
    }
    return total;
}

int Skeleton::total_dof_count() const {
    // For state storage: always 3 DOFs for SPHERICAL joints regardless of locked DOFs
    int total = 0;
    for (auto const& joint : joints_) {
        if (joint.type == JointType::REVOLUTE) {
            total += 1;
        } else if (joint.type == JointType::SPHERICAL) {
            total += 3;  // Always 3, regardless of locked DOFs
        }
        // FIXED has 0 DOF
    }
    return total;
}

int Skeleton::active_dof() const {
    if (!filter_active_) {
        // Count active DOFs considering locked axes
        // Python convention: active_dof INCLUDES root's 6 DOFs (3 pos + 3 rot)
        // With 99 body joint DOFs + 6 root = 105 total, error_dim = 2*105 = 210
        int total = 6;  // Root always contributes 6 DOFs (3 pos + 3 rot)
        for (auto const& joint : joints_) {
            if (!joint.parent_index.has_value()) {
                continue;  // Skip root in iteration (already counted above)
            }
            total += joint.active_dof();
        }
        return total;
    }

    int total = 6;  // Include root
    for (auto const& joint : joints_) {
        if (!joint.parent_index.has_value()) {
            continue;  // Skip root (already counted)
        }
        if (active_joints_.contains(joint.name) && active_joints_.at(joint.name)) {
            total += joint.active_dof();
        }
    }
    return total;
}

void Skeleton::set_active_groups(std::vector<std::string> const& groups) {
    active_joints_.clear();
    std::unordered_set<std::string> group_set(groups.begin(), groups.end());

    for (auto const& joint : joints_) {
        active_joints_[joint.name] = (group_set.find(joint.group) != group_set.end());
    }
    filter_active_ = true;
}

void Skeleton::set_active_joints(std::vector<std::string> const& joint_names) {
    active_joints_.clear();
    std::unordered_set<std::string> joint_set(joint_names.begin(), joint_names.end());

    for (auto const& joint : joints_) {
        active_joints_[joint.name] = (joint_set.find(joint.name) != joint_set.end());
    }
    filter_active_ = true;
}

void Skeleton::clear_active_filter() {
    active_joints_.clear();
    filter_active_ = false;
}

Joint const* Skeleton::get_joint(std::string const& name) const {
    for (auto const& joint : joints_) {
        if (joint.name == name) {
            return &joint;
        }
    }
    return nullptr;
}

Marker const* Skeleton::get_marker(std::string const& name) const {
    for (auto const& marker : markers_) {
        if (marker.name == name) {
            return &marker;
        }
    }
    return nullptr;
}

std::vector<Joint> Skeleton::get_joints_ordered() const {
    // Now just return a copy since vector already preserves insertion order
    return joints_;
}

bool Skeleton::is_joint_active(std::string const& name) const {
    if (!filter_active_) {
        return true;
    }
    auto it = active_joints_.find(name);
    return it != active_joints_.end() && it->second;
}

nlohmann::json Skeleton::to_json() const {
    nlohmann::json j;

    // Serialize joints with parent names
    nlohmann::json joints_arr = nlohmann::json::array();
    for (auto const& joint : joints_) {
        nlohmann::json joint_json;
        joint_json["name"] = joint.name;

        // Convert parent index to parent name
        if (joint.parent_index.has_value()) {
            joint_json["parent"] = joints_[joint.parent_index.value()].name;
        } else {
            joint_json["parent"] = "";
        }

        joint_json["type"] = (joint.type == JointType::REVOLUTE    ? "revolute"
                              : joint.type == JointType::SPHERICAL ? "spherical"
                                                                   : "fixed");
        joint_json["dof"] = joint.dof;

        // Serialize limits
        nlohmann::json limits_json = nlohmann::json::array();
        for (size_t i = 0; i < joint.num_limits; ++i) {
            limits_json.push_back({joint.limits[i][0], joint.limits[i][1]});
        }
        joint_json["limits"] = limits_json;

        joint_json["group"] = joint.group;
        joint_json["offset"] = {joint.offset[0], joint.offset[1], joint.offset[2]};

        // Always include rest_orientation (even if zero)
        joint_json["rest_orientation"] = {joint.rest_orientation[0], joint.rest_orientation[1],
                                          joint.rest_orientation[2]};

        joints_arr.push_back(joint_json);
    }
    j["joints"] = joints_arr;

    // Serialize markers with joint names
    nlohmann::json markers_arr = nlohmann::json::array();
    for (auto const& marker : markers_) {
        nlohmann::json marker_json;
        marker_json["name"] = marker.name;
        marker_json["joint"] = joints_[marker.joint_index].name;
        marker_json["local_pos"] = {marker.local_pos[0], marker.local_pos[1], marker.local_pos[2]};
        if (marker.coco_id) {
            marker_json["coco_id"] = *marker.coco_id;
        }
        markers_arr.push_back(marker_json);
    }
    j["markers"] = markers_arr;

    return j;
}

Skeleton Skeleton::from_json(nlohmann::json const& j) {
    Skeleton skel;
    std::unordered_map<std::string, uint32_t> name_to_index;

    // Deserialize joints
    for (auto const& joint_json : j.at("joints")) {
        std::string name = joint_json.at("name").get<std::string>();
        std::string parent_name = joint_json.value("parent", std::string{});

        std::optional<uint32_t> parent_index;
        if (!parent_name.empty()) {
            auto it = name_to_index.find(parent_name);
            if (it == name_to_index.end()) {
                throw std::runtime_error(
                    fmt::format("Parent '{}' not found for joint '{}'", parent_name, name));
            }
            parent_index = it->second;
        }

        // Parse type
        std::string type_str = joint_json.at("type").get<std::string>();
        JointType type;
        if (type_str == "revolute") {
            type = JointType::REVOLUTE;
        } else if (type_str == "spherical") {
            type = JointType::SPHERICAL;
        } else if (type_str == "fixed") {
            type = JointType::FIXED;
        } else {
            throw std::invalid_argument(fmt::format("Unknown joint type: {}", type_str));
        }

        // Parse offset
        Eigen::Vector3d offset;
        auto const& offset_arr = joint_json.at("offset");
        offset << offset_arr[0].get<double>(), offset_arr[1].get<double>(),
            offset_arr[2].get<double>();

        std::string group = joint_json.value("group", std::string{});

        // Parse rest orientation
        Eigen::Vector3d rest_orientation = Eigen::Vector3d::Zero();
        if (joint_json.contains("rest_orientation")) {
            auto const& orient_arr = joint_json.at("rest_orientation");
            rest_orientation << orient_arr[0].get<double>(), orient_arr[1].get<double>(),
                orient_arr[2].get<double>();
        }

        uint32_t idx = skel.add_joint(name, parent_index, type, offset, group, rest_orientation);
        name_to_index[name] = idx;

        // Parse and set limits if present
        if (joint_json.contains("limits")) {
            auto const& limits_arr = joint_json.at("limits");
            size_t num_limits = std::min(limits_arr.size(), size_t(3));
            std::array<Eigen::Vector2d, 3> limits;
            for (size_t i = 0; i < num_limits; ++i) {
                limits[i] =
                    Eigen::Vector2d(limits_arr[i][0].get<double>(), limits_arr[i][1].get<double>());
            }
            skel.set_joint_limits(idx, limits, num_limits);
        }
    }

    // Deserialize markers
    if (j.contains("markers")) {
        for (auto const& marker_json : j.at("markers")) {
            std::string name = marker_json.at("name").get<std::string>();
            std::string joint_name = marker_json.at("joint").get<std::string>();

            auto it = name_to_index.find(joint_name);
            if (it == name_to_index.end()) {
                throw std::runtime_error(
                    fmt::format("Joint '{}' not found for marker '{}'", joint_name, name));
            }

            Eigen::Vector3d local_pos;
            auto const& pos_arr = marker_json.at("local_pos");
            local_pos << pos_arr[0].get<double>(), pos_arr[1].get<double>(),
                pos_arr[2].get<double>();

            std::optional<int> coco_id;
            if (marker_json.contains("coco_id")) {
                coco_id = marker_json.at("coco_id").get<int>();
            }

            skel.add_marker(name, it->second, local_pos, coco_id);
        }
    }

    return skel;
}

std::optional<uint32_t> Skeleton::find_root() const {
    std::optional<uint32_t> root;
    for (uint32_t i = 0; i < joints_.size(); ++i) {
        if (!joints_[i].parent_index.has_value()) {
            if (root.has_value()) {
                return std::nullopt;  // Multiple roots
            }
            root = i;
        }
    }
    return root;
}

bool Skeleton::detect_cycle(uint32_t joint_index, std::unordered_set<uint32_t>& visited) const {
    // Already visited in current path = cycle
    if (visited.count(joint_index)) {
        return true;
    }

    visited.insert(joint_index);

    // Check parent
    auto const& joint = joints_[joint_index];
    if (joint.parent_index.has_value()) {
        if (detect_cycle(joint.parent_index.value(), visited)) {
            return true;
        }
    }

    visited.erase(joint_index);
    return false;
}

}  // namespace posetrak

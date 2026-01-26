#include "posetrak/core/skeleton.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace posetrak {

// Joint implementation

Joint::Joint(std::string const& name_, std::string const& parent_, JointType type_,
             Eigen::Vector3d const& offset_, std::string const& group_)
    : name(name_),
      parent(parent_),
      type(type_),
      dof(type_ == JointType::FIXED ? 0 : (type_ == JointType::SPHERICAL ? 3 : 1)),
      num_limits(0),
      group(group_),
      offset(offset_),
      rest_orientation(Eigen::Vector3d::Zero()),
      has_rest_orientation(false) {
    // Initialize default limits based on joint type
    if (type_ == JointType::REVOLUTE) {
        limits[0] = Eigen::Vector2d(-M_PI, M_PI);
        num_limits = 1;
    } else if (type_ == JointType::SPHERICAL) {
        limits[0] = Eigen::Vector2d(-M_PI, M_PI);
        limits[1] = Eigen::Vector2d(-M_PI, M_PI);
        limits[2] = Eigen::Vector2d(-M_PI, M_PI);
        num_limits = 3;
    }
    // FIXED joints have no limits (num_limits = 0)
}

nlohmann::json Joint::to_json() const {
    nlohmann::json j;
    j["name"] = name;
    j["parent"] = parent;
    j["type"] = (type == JointType::REVOLUTE    ? "revolute"
                 : type == JointType::SPHERICAL ? "spherical"
                                                : "fixed");
    j["dof"] = dof;

    // Serialize limits
    nlohmann::json limits_json = nlohmann::json::array();
    for (size_t i = 0; i < num_limits; ++i) {
        limits_json.push_back({limits[i][0], limits[i][1]});
    }
    j["limits"] = limits_json;

    j["group"] = group;
    j["offset"] = {offset[0], offset[1], offset[2]};
    if (has_rest_orientation) {
        j["rest_orientation"] = {rest_orientation[0], rest_orientation[1], rest_orientation[2]};
    }
    return j;
}

Joint Joint::from_json(nlohmann::json const& j) {
    std::string const type_str = j.at("type").get<std::string>();
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

    Eigen::Vector3d offset;
    auto const& offset_arr = j.at("offset");
    offset << offset_arr[0].get<double>(), offset_arr[1].get<double>(), offset_arr[2].get<double>();

    Joint joint(j.at("name").get<std::string>(), j.at("parent").get<std::string>(), type, offset,
                j.value("group", std::string{}));

    // Parse limits array
    if (j.contains("limits")) {
        auto const& limits_arr = j.at("limits");
        joint.num_limits = std::min(limits_arr.size(), size_t(3));
        for (size_t i = 0; i < joint.num_limits; ++i) {
            joint.limits[i] =
                Eigen::Vector2d(limits_arr[i][0].get<double>(), limits_arr[i][1].get<double>());
        }
    }

    // Parse rest orientation
    if (j.contains("rest_orientation")) {
        auto const& orient_arr = j.at("rest_orientation");
        joint.rest_orientation << orient_arr[0].get<double>(), orient_arr[1].get<double>(),
            orient_arr[2].get<double>();
        joint.has_rest_orientation = true;
    }

    return joint;
}

// Marker implementation

Marker::Marker(std::string const& name_, std::string const& joint_,
               Eigen::Vector3d const& local_pos_, std::optional<int> coco_id_)
    : name(name_), joint(joint_), local_pos(local_pos_), coco_id(coco_id_) {}

nlohmann::json Marker::to_json() const {
    nlohmann::json j;
    j["name"] = name;
    j["joint"] = joint;
    j["local_pos"] = {local_pos[0], local_pos[1], local_pos[2]};
    if (coco_id) {
        j["coco_id"] = *coco_id;
    }
    return j;
}

Marker Marker::from_json(nlohmann::json const& j) {
    Eigen::Vector3d local_pos;
    auto const& pos_arr = j.at("local_pos");
    local_pos << pos_arr[0].get<double>(), pos_arr[1].get<double>(), pos_arr[2].get<double>();

    std::optional<int> coco_id = std::nullopt;
    if (j.contains("coco_id")) {
        coco_id = j.at("coco_id").get<int>();
    }

    return Marker(j.at("name").get<std::string>(), j.at("joint").get<std::string>(), local_pos,
                  coco_id);
}

// Skeleton implementation

void Skeleton::add_joint(Joint const& joint) {
    if (joints_.find(joint.name) != joints_.end()) {
        throw std::invalid_argument(fmt::format("Joint '{}' already exists", joint.name));
    }
    joints_.emplace(joint.name, joint);
}

void Skeleton::add_marker(Marker const& marker) {
    if (markers_.find(marker.name) != markers_.end()) {
        throw std::invalid_argument(fmt::format("Marker '{}' already exists", marker.name));
    }
    if (joints_.find(marker.joint) == joints_.end()) {
        throw std::invalid_argument(
            fmt::format("Joint '{}' not found for marker '{}'", marker.joint, marker.name));
    }
    markers_.emplace(marker.name, marker);
}

std::optional<std::string> Skeleton::validate() const {
    if (joints_.empty()) {
        return "Skeleton has no joints";
    }

    // Check all parents exist first
    for (auto const& [name, joint] : joints_) {
        if (!joint.parent.empty() && joints_.find(joint.parent) == joints_.end()) {
            return fmt::format("Parent '{}' of joint '{}' not found", joint.parent, name);
        }
    }

    // Check for cycles (before checking for root, since a cycle means no root)
    std::unordered_map<std::string, bool> visited;
    for (auto const& [name, _] : joints_) {
        visited[name] = false;
    }

    for (auto const& [name, _] : joints_) {
        if (detect_cycle(name, visited)) {
            return fmt::format("Cycle detected in hierarchy at joint '{}'", name);
        }
    }

    // Find root
    std::string const root = find_root();
    if (root.empty()) {
        return "No unique root joint found (joint with empty parent)";
    }

    // Check all marker joints exist
    for (auto const& [name, marker] : markers_) {
        if (joints_.find(marker.joint) == joints_.end()) {
            return fmt::format("Joint '{}' not found for marker '{}'", marker.joint, name);
        }
    }

    return std::nullopt;
}

int Skeleton::total_dof() const {
    int total = 0;
    for (auto const& [_, joint] : joints_) {
        total += joint.dof;
    }
    return total;
}

int Skeleton::active_dof() const {
    if (!filter_active_) {
        return total_dof();
    }

    int total = 0;
    for (auto const& [name, joint] : joints_) {
        if (active_joints_.contains(name) && active_joints_.at(name)) {
            total += joint.dof;
        }
    }
    return total;
}

void Skeleton::set_active_groups(std::vector<std::string> const& groups) {
    active_joints_.clear();
    std::unordered_set<std::string> group_set(groups.begin(), groups.end());

    for (auto const& [name, joint] : joints_) {
        active_joints_[name] = (group_set.find(joint.group) != group_set.end());
    }
    filter_active_ = true;
}

void Skeleton::set_active_joints(std::vector<std::string> const& joint_names) {
    active_joints_.clear();
    std::unordered_set<std::string> joint_set(joint_names.begin(), joint_names.end());

    for (auto const& [name, _] : joints_) {
        active_joints_[name] = (joint_set.find(name) != joint_set.end());
    }
    filter_active_ = true;
}

void Skeleton::clear_active_filter() {
    active_joints_.clear();
    filter_active_ = false;
}

Joint const* Skeleton::get_joint(std::string const& name) const {
    auto it = joints_.find(name);
    return it != joints_.end() ? &it->second : nullptr;
}

Marker const* Skeleton::get_marker(std::string const& name) const {
    auto it = markers_.find(name);
    return it != markers_.end() ? &it->second : nullptr;
}

std::vector<Joint> Skeleton::get_joints_ordered() const {
    std::vector<Joint> ordered;
    ordered.reserve(joints_.size());

    // Find root and do depth-first traversal
    std::string const root = find_root();
    if (root.empty()) {
        return ordered;  // Invalid skeleton
    }

    std::vector<std::string> stack = {root};
    std::unordered_set<std::string> visited;

    while (!stack.empty()) {
        std::string const current = stack.back();
        stack.pop_back();

        if (visited.find(current) != visited.end()) {
            continue;
        }
        visited.insert(current);

        auto it = joints_.find(current);
        if (it != joints_.end()) {
            ordered.push_back(it->second);

            // Add children to stack
            for (auto const& [name, joint] : joints_) {
                if (joint.parent == current) {
                    stack.push_back(name);
                }
            }
        }
    }

    return ordered;
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

    // Serialize joints in depth-first order
    std::vector<Joint> const ordered = get_joints_ordered();
    nlohmann::json joints_arr = nlohmann::json::array();
    for (auto const& joint : ordered) {
        joints_arr.push_back(joint.to_json());
    }
    j["joints"] = joints_arr;

    // Serialize markers
    nlohmann::json markers_arr = nlohmann::json::array();
    for (auto const& [_, marker] : markers_) {
        markers_arr.push_back(marker.to_json());
    }
    j["markers"] = markers_arr;

    return j;
}

Skeleton Skeleton::from_json(nlohmann::json const& j) {
    Skeleton skel;

    // Deserialize joints
    for (auto const& joint_json : j.at("joints")) {
        skel.add_joint(Joint::from_json(joint_json));
    }

    // Deserialize markers
    if (j.contains("markers")) {
        for (auto const& marker_json : j.at("markers")) {
            skel.add_marker(Marker::from_json(marker_json));
        }
    }

    return skel;
}

std::string Skeleton::find_root() const {
    std::string root;
    for (auto const& [name, joint] : joints_) {
        if (joint.parent.empty()) {
            if (!root.empty()) {
                return "";  // Multiple roots
            }
            root = name;
        }
    }
    return root;
}

bool Skeleton::detect_cycle(std::string const& joint_name,
                            std::unordered_map<std::string, bool>& visited) const {
    // Already visited in current path = cycle
    if (visited[joint_name]) {
        return true;
    }

    visited[joint_name] = true;

    // Check parent
    auto it = joints_.find(joint_name);
    if (it != joints_.end() && !it->second.parent.empty()) {
        if (detect_cycle(it->second.parent, visited)) {
            return true;
        }
    }

    visited[joint_name] = false;
    return false;
}

}  // namespace posetrak

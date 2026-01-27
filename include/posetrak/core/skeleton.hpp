#pragma once

#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace posetrak {

/// @brief Type of joint in skeleton hierarchy
enum class JointType {
    REVOLUTE,   ///< Single-axis rotation
    SPHERICAL,  ///< 3-DOF ball joint (represented as 3 consecutive revolute joints)
    FIXED       ///< No DOF (virtual joint for marker attachment)
};

/// @brief Joint definition in skeleton hierarchy
struct Joint {
    std::string name;    ///< Unique joint name
    std::string parent;  ///< Parent joint name (empty for root)
    int parent_index;    ///< Parent joint index in skeleton (-1 for root)
    int skeleton_index;  ///< Index in skeleton's joints vector (-1 if not in skeleton)
    JointType type;      ///< Joint type
    int dof;             ///< Degrees of freedom (1 for revolute, 3 for spherical, 0 for fixed)
    std::array<Eigen::Vector2d, 3> limits;  ///< Joint limits [min, max] per DOF (max 3)
    size_t num_limits;                      ///< Number of active limit pairs (0-3)
    std::string group;                      ///< Joint group for filtering (e.g., "legs", "arms")
    Eigen::Vector3d offset;                 ///< Translation from parent in parent's frame
    Eigen::Vector3d rest_orientation;       ///< Rest orientation as ZYX Euler angles (radians)
    bool has_rest_orientation;              ///< Whether rest orientation is specified

    /// @brief Construct joint with defaults
    Joint(std::string const& name_, std::string const& parent_,
          JointType type_ = JointType::REVOLUTE,
          Eigen::Vector3d const& offset_ = Eigen::Vector3d::Zero(), std::string const& group_ = "");

    /// @brief Serialize to JSON
    nlohmann::json to_json() const;

    /// @brief Deserialize from JSON
    static Joint from_json(nlohmann::json const& j);
};

/// @brief Marker attached to skeleton for observations
struct Marker {
    std::string name;            ///< Unique marker name
    std::string joint;           ///< Attached joint name
    int joint_index;             ///< Attached joint index in skeleton (-1 if not set)
    Eigen::Vector3d local_pos;   ///< Position in joint's local frame
    std::optional<int> coco_id;  ///< Optional COCO keypoint ID for compatibility

    /// @brief Construct marker
    Marker(std::string const& name_, std::string const& joint_, Eigen::Vector3d const& local_pos_,
           std::optional<int> coco_id_ = std::nullopt);

    /// @brief Serialize to JSON
    nlohmann::json to_json() const;

    /// @brief Deserialize from JSON
    static Marker from_json(nlohmann::json const& j);
};

/// @brief Skeleton hierarchy with joints and markers
///
/// Represents a kinematic tree with arbitrary joint structure.
/// Supports active joint filtering by group or explicit list.
class Skeleton {
   public:
    /// @brief Construct empty skeleton
    Skeleton() = default;

    /// @brief Add joint to skeleton
    /// @param joint Joint to add
    /// @throws std::invalid_argument if joint name already exists
    void add_joint(Joint&& joint);

    /// @brief Add marker to skeleton
    /// @param marker Marker to add
    /// @throws std::invalid_argument if marker name already exists or joint not found
    void add_marker(Marker&& marker);

    /// @brief Validate skeleton structure
    ///
    /// Checks:
    /// - Root joint exists (joint with no parent)
    /// - No cycles in hierarchy
    /// - All parent joints exist
    /// - All marker joints exist
    ///
    /// @return Error message if invalid, std::nullopt if valid
    std::optional<std::string> validate() const;

    /// @brief Get total DOF count (all joints)
    /// @return Total degrees of freedom
    int total_dof() const;

    /// @brief Get active DOF count based on filter
    /// @return Active degrees of freedom
    int active_dof() const;

    /// @brief Set active joints by group names
    /// @param groups List of group names to activate
    void set_active_groups(std::vector<std::string> const& groups);

    /// @brief Set active joints by explicit joint names
    /// @param joint_names List of joint names to activate
    void set_active_joints(std::vector<std::string> const& joint_names);

    /// @brief Clear active filter (all joints active)
    void clear_active_filter();

    /// @brief Get joint by name
    /// @param name Joint name
    /// @return Pointer to joint or nullptr if not found
    Joint const* get_joint(std::string const& name) const;

    /// @brief Get marker by name
    /// @param name Marker name
    /// @return Pointer to marker or nullptr if not found
    Marker const* get_marker(std::string const& name) const;

    /// @brief Get all joints in depth-first order
    /// @return Ordered list of joints
    std::vector<Joint> get_joints_ordered() const;

    /// @brief Check if joint is active
    /// @param name Joint name
    /// @return True if joint is active
    bool is_joint_active(std::string const& name) const;

    /// @brief Get all joints in state vector order
    /// @return Vector of joints
    std::vector<Joint> const& joints() const { return joints_; }

    /// @brief Get all markers in state vector order
    /// @return Vector of markers
    std::vector<Marker> const& markers() const { return markers_; }

    /// @brief Serialize to JSON
    nlohmann::json to_json() const;

    /// @brief Deserialize from JSON
    static Skeleton from_json(nlohmann::json const& j);

   private:
    /// @brief Find root joint (joint with no parent)
    /// @return Root joint name or empty string if not found/multiple roots
    std::string find_root() const;

    /// @brief Detect cycles in hierarchy starting from joint
    /// @param joint_name Starting joint
    /// @param visited Set of visited joints
    /// @return True if cycle detected
    bool detect_cycle(std::string const& joint_name,
                      std::unordered_map<std::string, bool>& visited) const;

    std::vector<Joint> joints_;    ///< Joint definitions (in state vector order)
    std::vector<Marker> markers_;  ///< Marker definitions (in state vector order)
    std::unordered_map<std::string, bool> active_joints_;  ///< Active joint filter
    bool filter_active_ = false;                           ///< Whether active filter is enabled
};

}  // namespace posetrak

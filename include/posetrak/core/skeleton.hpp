#pragma once

#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    std::string name;                      ///< Unique joint name
    std::optional<uint32_t> parent_index;  ///< Parent joint index (nullopt for root)
    JointType type;                        ///< Joint type
    int dof;  ///< Degrees of freedom (1 for revolute, 3 for spherical, 0 for fixed)
    std::array<Eigen::Vector2d, 3> limits;  ///< Joint limits [min, max] per DOF (max 3)
    size_t num_limits;                      ///< Number of active limit pairs (0-3)
    Eigen::Vector3d offset;                 ///< Translation from parent in parent's frame
    Eigen::Vector3d rest_orientation;       ///< Rest orientation as ZYX Euler angles (radians)

    /// @brief Get mask of active (non-locked) DOFs
    /// @return Array indicating which DOFs are active (true) or locked (false)
    /// A DOF is considered locked if its min and max limits are equal (within tolerance)
    std::array<bool, 3> get_active_dof_mask() const {
        std::array<bool, 3> mask = {false, false, false};

        if (type == JointType::REVOLUTE) {
            mask[0] = true;
        } else if (type == JointType::SPHERICAL) {
            for (size_t i = 0; i < num_limits && i < 3; ++i) {
                double const min_limit = limits[i].x();
                double const max_limit = limits[i].y();
                // DOF is active if limits differ by more than tolerance
                // Using 1e-4 tolerance to match Python's locked DOF detection
                mask[i] = std::abs(max_limit - min_limit) > 1e-4;
            }
            // If no limits set, all DOFs are active
            if (num_limits == 0) {
                mask[0] = mask[1] = mask[2] = true;
            }
        }
        // FIXED joints have all false (no active DOFs)
        return mask;
    }

    /// @brief Get number of active (non-locked) DOFs
    /// @return Count of active DOFs
    int active_dof() const {
        auto mask = get_active_dof_mask();
        return static_cast<int>(mask[0]) + static_cast<int>(mask[1]) + static_cast<int>(mask[2]);
    }
};

/// @brief Marker attached to skeleton for observations
struct Marker {
    std::string name;            ///< Unique marker name
    uint32_t joint_index;        ///< Attached joint index in skeleton
    Eigen::Vector3d local_pos;   ///< Position in joint's local frame
    std::optional<int> coco_id;  ///< Optional COCO keypoint ID for compatibility
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
    /// @param name Unique joint name
    /// @param parent_index Parent joint index (nullopt for root)
    /// @param type Joint type
    /// @param offset Translation from parent in parent's frame
    /// @param rest_orientation Rest orientation as ZYX Euler angles
    /// @return Index of the added joint
    /// @throws std::invalid_argument if joint name already exists or parent index invalid
    uint32_t add_joint(std::string const& name, std::optional<uint32_t> parent_index,
                       JointType type, Eigen::Vector3d const& offset = Eigen::Vector3d::Zero(),
                       Eigen::Vector3d const& rest_orientation = Eigen::Vector3d::Zero());

    /// @brief Register a named group with its joint and marker members.
    ///
    /// A joint or marker may belong to multiple groups (m:n relationship).
    /// Calling register_group multiple times for the same group_name is allowed
    /// and cumulative (adds to the existing member sets).
    void register_group(std::string const& group_name, std::vector<std::string> const& joint_names,
                        std::vector<std::string> const& marker_names);

    /// @brief Return true if joint_name belongs to any of the given groups.
    bool joint_in_groups(std::string const& joint_name,
                         std::unordered_set<std::string> const& groups) const;

    /// @brief Return true if marker_name belongs to any of the given groups.
    bool marker_in_groups(std::string const& marker_name,
                          std::unordered_set<std::string> const& groups) const;

    /// @brief Add marker to skeleton
    /// @param name Unique marker name
    /// @param joint_index Attached joint index
    /// @param local_pos Position in joint's local frame
    /// @param coco_id Optional COCO keypoint ID
    /// @return Index of the added marker
    /// @throws std::invalid_argument if marker name already exists or joint index invalid
    uint32_t add_marker(std::string const& name, uint32_t joint_index,
                        Eigen::Vector3d const& local_pos,
                        std::optional<int> coco_id = std::nullopt);

    /// @brief Set joint limits
    /// @param joint_index Index of the joint
    /// @param limits Array of limit pairs [min, max] for each DOF
    /// @param num_limits Number of limit pairs (1 for REVOLUTE, 3 for SPHERICAL)
    /// @throws std::invalid_argument if joint index invalid
    void set_joint_limits(uint32_t joint_index, std::array<Eigen::Vector2d, 3> const& limits,
                          size_t num_limits);

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

    /// @brief Get total DOF count for state storage (always 3 for SPHERICAL joints)
    /// @return Total storage DOFs needed for state vector
    /// @note This returns the size needed for State::joint_angles vector.
    ///       For SPHERICAL joints, always counts 3 DOFs even if some are locked.
    int total_dof_count() const;

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

    /// @brief Get all joints in state vector order
    /// @return Vector of joints
    std::vector<Joint> const& joints() const { return joints_; }

    /// @brief Get all markers in state vector order (const)
    /// @return Vector of markers
    std::vector<Marker> const& markers() const { return markers_; }

    /// @brief Get all markers in state vector order (mutable)
    /// @return Vector of markers
    std::vector<Marker>& markers() { return markers_; }

    /// @brief Serialize to JSON
    nlohmann::json to_json() const;

    /// @brief Deserialize from JSON
    static Skeleton from_json(nlohmann::json const& j);

   private:
    /// @brief Find root joint index (joint with no parent)
    /// @return Root joint index or nullopt if not found/multiple roots
    std::optional<uint32_t> find_root() const;

    /// @brief Detect cycles in hierarchy starting from joint
    /// @param joint_index Starting joint index
    /// @param visited Set of visited joint indices
    /// @return True if cycle detected
    bool detect_cycle(uint32_t joint_index, std::unordered_set<uint32_t>& visited) const;

    std::vector<Joint> joints_;    ///< Joint definitions (in state vector order)
    std::vector<Marker> markers_;  ///< Marker definitions (in state vector order)

    /// group_name → set of joint names belonging to that group
    std::unordered_map<std::string, std::unordered_set<std::string>> group_joints_;
    /// group_name → set of marker names belonging to that group
    std::unordered_map<std::string, std::unordered_set<std::string>> group_markers_;
};

}  // namespace posetrak

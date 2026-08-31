// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

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
    FIXED,      ///< No DOF (virtual joint for marker attachment)
    PRISMATIC   ///< 1-DOF sliding joint; used for bone-length calibration
};

/// @brief Joint definition in skeleton hierarchy
struct Joint {
    std::string name;                      ///< Unique joint name
    std::optional<uint32_t> parent_index;  ///< Parent joint index (nullopt for root)
    JointType type;                        ///< Joint type
    int dof;  ///< Degrees of freedom (1 for revolute, 3 for spherical, 0 for fixed)
    std::array<Eigen::Vector2d, 3> limits;  ///< Joint limits [min, max] per DOF (max 3)
    size_t num_limits;                      ///< Number of active limit pairs (0-3)
    std::string group;                      ///< Joint group for filtering (e.g., "legs", "arms")
    Eigen::Vector3d offset;                 ///< Translation from parent in parent's frame
    Eigen::Vector3d rest_orientation;       ///< Rest orientation as ZYX Euler angles (radians)
    Eigen::Vector3d prismatic_axis{
        0, 1, 0};             ///< Sliding axis for PRISMATIC joints (unit vector, parent frame)
    std::string scale_group;  ///< Scale group name for PRISMATIC joints ("" if ungrouped)
    double nominal_length =
        0.0;  ///< |original_offset| in metres for PRISMATIC joints (q = scale × nominal_length)
    bool is_scale_follower = false;  ///< True for non-first PRISMATIC joints in a scale group; they
                                     ///< share the leader's state slot and do not occupy their own

    /// @brief Get mask of active (non-locked) DOFs
    /// @return Array indicating which DOFs are active (true) or locked (false)
    /// A DOF is considered locked if its min and max limits are equal (within tolerance)
    std::array<bool, 3> get_active_dof_mask() const {
        std::array<bool, 3> mask = {false, false, false};

        if (type == JointType::REVOLUTE || type == JointType::PRISMATIC) {
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
    std::string group;           ///< Marker group for filtering (e.g., "main", "HandL")

    /// @brief Named observation source this marker is bound to (skeleton YAML's
    /// input_tracks:, marker-based-mocap design doc §5.1) -- e.g. "prop_markers"
    /// for a generated prop skeleton. Empty for a marker with no track: field
    /// (the "exactly as today" case: coco_id-only, or a marker not bound to
    /// any dynamic observation source at all).
    std::string track;

    /// @brief Landmark name within `track`, resolved against the bound
    /// sequence's keypoint manifest at load time (design §5.1/§4.3) -- e.g.
    /// "hilt:c0" for a coded-marker corner, or a dot's own name. Meaningless
    /// when `track` is empty.
    std::string landmark;
};

/// @brief One skeleton input_tracks: entry (design doc §5.1) -- a named
/// observation source a marker can bind to via Marker::track, with a layout
/// implied by `type` ("coco133" et al. use the legacy openpose_keypoint
/// mapping; "labeled_points" resolves landmark names via the bound
/// sequence's own pose_sequence_keypoints manifest at load time).
struct InputTrack {
    std::string id;
    std::string type;
};

/// @brief Metadata for a named joint/marker group beyond membership (which
/// lives on each Joint::group / Marker::group).
///
/// freeflyer_joint and ref_marker together describe a group that can run as
/// its own hierarchical-solver child stage (see
/// docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md
/// and docs/skeleton-format.md's "groups:" section): freeflyer_joint is the
/// joint PinocchioModelBuilder::build_subtree_model() treats as the
/// subtree's externally-supplied, fixed root, and ref_marker is the marker
/// build_ref_marker_pair_observations() measures every other marker in the
/// group relative to via PAIR_DIFF. Both are empty for a group that is not
/// (or not yet) wired up as a child stage, e.g. "main".
struct SkeletonGroup {
    std::string name;
    std::string freeflyer_joint;  ///< Empty if this group is not a child-stage group
    std::string ref_marker;       ///< Empty if this group is not a child-stage group

    /// @brief Joint/marker names declared under this group's groups: YAML
    /// entry, in declaration order. A name may appear in more than one
    /// group's list -- e.g. a wrist joint solved by both a parent and a
    /// child group (see docs/skeleton-format.md and the hierarchical
    /// solver design doc's "wrist ownership" section) -- unlike
    /// Joint::group/Marker::group, which hold only one ("primary") group
    /// name per joint/marker. Skeleton::is_joint_in_groups() is what
    /// actually resolves multi-group membership from these lists.
    std::vector<std::string> joints;
    std::vector<std::string> markers;
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
    /// @param group Joint group for filtering
    /// @param rest_orientation Rest orientation as ZYX Euler angles
    /// @return Index of the added joint
    /// @throws std::invalid_argument if joint name already exists or parent index invalid
    uint32_t add_joint(std::string const& name, std::optional<uint32_t> parent_index,
                       JointType type, Eigen::Vector3d const& offset = Eigen::Vector3d::Zero(),
                       std::string const& group = "",
                       Eigen::Vector3d const& rest_orientation = Eigen::Vector3d::Zero());

    /// @brief Add marker to skeleton
    /// @param name Unique marker name
    /// @param joint_index Attached joint index
    /// @param local_pos Position in joint's local frame
    /// @param coco_id Optional COCO keypoint ID
    /// @param track Optional input_tracks: id this marker is bound to (design §5.1);
    ///        empty means "not bound to a dynamic observation source"
    /// @param landmark Optional landmark name within `track`; meaningless if track is empty
    /// @return Index of the added marker
    /// @throws std::invalid_argument if marker name already exists or joint index invalid
    uint32_t add_marker(std::string const& name, uint32_t joint_index,
                        Eigen::Vector3d const& local_pos, std::optional<int> coco_id = std::nullopt,
                        std::string const& track = "", std::string const& landmark = "");

    /// @brief Register an input_tracks: entry (design §5.1).
    /// @param id Track id, referenced by Marker::track
    /// @param type Layout type ("coco133", "labeled_points", ...)
    void add_input_track(std::string const& id, std::string const& type);

    /// @brief All declared input_tracks:, in YAML order. Empty for a skeleton
    /// with no input_tracks: section -- "behaves exactly as today" (design §5.1).
    std::vector<InputTrack> const& input_tracks() const { return input_tracks_; }

    /// @brief Look up one input_tracks: entry by id.
    /// @return Pointer to the entry, or nullptr if no such track is declared.
    InputTrack const* get_input_track(std::string const& id) const;

    /// @brief True if this skeleton has no active DOF below the root -- a
    /// free-flyer root plus markers only, generated from a marker body
    /// definition (design §5.3) rather than an articulated body. Single
    /// source of truth for the "use closed-form rigid-body init instead of
    /// triangulation+IK" decision (Tracker::initialize()) and for callers
    /// that need to know a rest-pose fallback would be meaningless (a
    /// free-floating prop has no anchor to fall back to, unlike a person).
    bool is_rigid_body() const {
        for (auto const& j : joints_) {
            if (j.parent_index.has_value() && j.type != JointType::FIXED && j.active_dof() > 0) {
                return false;
            }
        }
        return true;
    }

    /// @brief Set joint limits
    /// @param joint_index Index of the joint
    /// @param limits Array of limit pairs [min, max] for each DOF
    /// @param num_limits Number of limit pairs (1 for REVOLUTE, 3 for SPHERICAL)
    /// @throws std::invalid_argument if joint index invalid
    void set_joint_limits(uint32_t joint_index, std::array<Eigen::Vector2d, 3> const& limits,
                          size_t num_limits);

    /// @brief Set prismatic axis for a PRISMATIC joint
    /// @param joint_index Index of the joint (must be PRISMATIC type)
    /// @param axis Unit vector in parent frame defining the sliding direction
    /// @throws std::invalid_argument if joint index invalid or joint is not PRISMATIC
    void set_joint_axis(uint32_t joint_index, Eigen::Vector3d const& axis);

    /// @brief Set the nominal bone length for a PRISMATIC joint (metres).
    /// @param joint_index Index of the joint (must be PRISMATIC type)
    /// @param length |original_offset|; state stores scale factor s where q = s * length
    void set_joint_nominal_length(uint32_t joint_index, double length);

    /// @brief Set the scale group name for a PRISMATIC joint.
    /// @param joint_index Index of the joint (must be PRISMATIC type)
    /// @param group_name Name of the scale group (from scale_groups in YAML)
    void set_joint_scale_group(uint32_t joint_index, std::string const& group_name);

    /// @brief Mark a PRISMATIC joint as a scale-group follower.
    /// Followers share the leader's state slot and do not occupy their own.
    /// @param joint_index Index of the joint (must be PRISMATIC type)
    /// @param value True if this joint is a follower (non-first in its group)
    void set_joint_scale_follower(uint32_t joint_index, bool value);

    /// @brief Register (or update) a named group's metadata from a YAML
    /// groups: entry.
    /// @param name Group name
    /// @param joints Raw joints: list for this group (declaration order; may overlap
    ///        with another group's list -- see SkeletonGroup::joints)
    /// @param markers Raw markers: list for this group (see SkeletonGroup::markers)
    /// @param freeflyer_joint See SkeletonGroup::freeflyer_joint. Empty if absent.
    /// @param ref_marker See SkeletonGroup::ref_marker. Empty if absent.
    void add_group(std::string const& name, std::vector<std::string> const& joints = {},
                   std::vector<std::string> const& markers = {},
                   std::string const& freeflyer_joint = "", std::string const& ref_marker = "");

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

    /// @brief Get group metadata by name.
    /// @return Pointer to the group's metadata, or nullptr if no groups:
    ///         entry with this name was declared (e.g. a skeleton with no
    ///         groups: section at all, or a name that only ever appears as
    ///         a Joint::group/Marker::group value with no matching entry).
    SkeletonGroup const* get_group(std::string const& name) const;

    /// @brief All declared groups, in groups: YAML order.
    std::vector<SkeletonGroup> const& groups() const { return groups_; }

    /// @brief True if joint_name belongs to any of group_names.
    ///
    /// For a name with a registered SkeletonGroup (a groups: YAML entry),
    /// membership is resolved against that group's declared SkeletonGroup::joints
    /// list -- this is what lets a joint belong to more than one group at once
    /// (e.g. a wrist joint solved by both a parent and a child group). For a
    /// name with no registered SkeletonGroup (a skeleton built without a
    /// groups: section, e.g. directly via add_joint(group=...) in a test),
    /// falls back to joint_own_group == name, matching the group-filtering
    /// behavior every layer of this codebase had before SkeletonGroup existed.
    ///
    /// @param joint_name Joint to test
    /// @param joint_own_group The joint's own Joint::group value (passed by the
    ///        caller, who already has the Joint, to avoid a redundant name lookup)
    /// @param group_names Groups to test membership against
    bool is_joint_in_groups(std::string const& joint_name, std::string const& joint_own_group,
                            std::vector<std::string> const& group_names) const;

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

    std::vector<Joint> joints_;             ///< Joint definitions (in state vector order)
    std::vector<Marker> markers_;           ///< Marker definitions (in state vector order)
    std::vector<SkeletonGroup> groups_;     ///< Group metadata, in groups: YAML order
    std::vector<InputTrack> input_tracks_;  ///< input_tracks: entries, in YAML order
};

}  // namespace posetrak

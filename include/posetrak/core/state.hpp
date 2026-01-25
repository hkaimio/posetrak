#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>

namespace posetrak {

/// Motion capture state in error-state formulation
/// 
/// Represents the full state of a skeleton with root pose and joint angles.
/// Uses quaternions for orientation and error-state formulation for filtering.
class State {
public:
    /// Construct state with specified DOF count
    explicit State(int num_dof);

    /// Construct from components
    State(Eigen::Vector3d const& root_position,
          Eigen::Quaterniond const& root_orientation,
          Eigen::VectorXd const& joint_angles,
          Eigen::VectorXd const& root_velocity,
          Eigen::VectorXd const& joint_velocities);

    // Accessors
    Eigen::Vector3d const& root_position() const { return root_position_; }
    Eigen::Quaterniond const& root_orientation() const { return root_orientation_; }
    Eigen::VectorXd const& joint_angles() const { return joint_angles_; }
    Eigen::VectorXd const& root_velocity() const { return root_velocity_; }
    Eigen::VectorXd const& joint_velocities() const { return joint_velocities_; }

    void set_root_position(Eigen::Vector3d const& pos) { root_position_ = pos; }
    void set_root_orientation(Eigen::Quaterniond const& quat) { root_orientation_ = quat.normalized(); }
    void set_joint_angles(Eigen::VectorXd const& angles) { joint_angles_ = angles; }
    void set_root_velocity(Eigen::VectorXd const& vel) { root_velocity_ = vel; }
    void set_joint_velocities(Eigen::VectorXd const& vel) { joint_velocities_ = vel; }

    /// Get total state dimension (for error-state: 3 + 3 + n_dof for position, orientation, joints)
    int error_state_dim() const { return 3 + 3 + joint_angles_.size(); }
    
    /// Get number of DOF (joint angles only)
    int num_dof() const { return static_cast<int>(joint_angles_.size()); }

    /// Convert to error-state vector (position, axis-angle orientation, joint angles)
    Eigen::VectorXd to_error_vector() const;

    /// Apply error-state update (with manifold operations for quaternion)
    void apply_error_update(Eigen::VectorXd const& error_delta);

    /// Convert quaternion to axis-angle representation
    static Eigen::Vector3d quaternion_to_axis_angle(Eigen::Quaterniond const& q);

    /// Convert axis-angle to quaternion
    static Eigen::Quaterniond axis_angle_to_quaternion(Eigen::Vector3d const& axis_angle);

    /// Serialize to JSON
    nlohmann::json to_json() const;

    /// Deserialize from JSON
    static State from_json(nlohmann::json const& j);

private:
    Eigen::Vector3d root_position_;        ///< Root position in world frame
    Eigen::Quaterniond root_orientation_;  ///< Root orientation (wxyz convention)
    Eigen::VectorXd joint_angles_;         ///< Joint angles (radians)
    Eigen::VectorXd root_velocity_;        ///< Root linear velocity
    Eigen::VectorXd joint_velocities_;     ///< Joint angular velocities
};

}  // namespace posetrak

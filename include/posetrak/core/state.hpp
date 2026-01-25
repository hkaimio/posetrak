#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <nlohmann/json.hpp>

namespace posetrak {

/// @brief Motion capture state in error-state formulation
///
/// Represents the full state of a skeleton with root pose and joint angles.
/// Uses quaternions for orientation and error-state formulation for filtering.
/// The error-state formulation uses axis-angle representation for orientation
/// updates to handle the quaternion manifold properly.
class State {
   public:
    /// @brief Construct state with specified DOF count
    /// @param num_dof Number of degrees of freedom (joint angles)
    /// @throws std::invalid_argument if num_dof is negative
    explicit State(int num_dof);

    /// @brief Construct from components
    /// @param root_position Root position in world frame (meters)
    /// @param root_orientation Root orientation as unit quaternion (wxyz)
    /// @param joint_angles Joint angles in radians
    /// @param root_velocity Root linear velocity (m/s)
    /// @param joint_velocities Joint angular velocities (rad/s)
    /// @throws std::invalid_argument if sizes are inconsistent
    State(Eigen::Vector3d const& root_position, Eigen::Quaterniond const& root_orientation,
          Eigen::VectorXd const& joint_angles, Eigen::VectorXd const& root_velocity,
          Eigen::VectorXd const& joint_velocities);

    // Accessors

    /// @brief Get root position in world frame
    /// @return 3D position vector
    Eigen::Vector3d const& root_position() const { return root_position_; }

    /// @brief Get root orientation as unit quaternion
    /// @return Quaternion in (w, x, y, z) convention
    Eigen::Quaterniond const& root_orientation() const { return root_orientation_; }

    /// @brief Get joint angles
    /// @return Vector of joint angles in radians
    Eigen::VectorXd const& joint_angles() const { return joint_angles_; }

    /// @brief Get root linear velocity
    /// @return 3D velocity vector (m/s)
    Eigen::VectorXd const& root_velocity() const { return root_velocity_; }

    /// @brief Get joint angular velocities
    /// @return Vector of joint velocities (rad/s)
    Eigen::VectorXd const& joint_velocities() const { return joint_velocities_; }

    /// @brief Set root position
    /// @param pos New position in world frame
    void set_root_position(Eigen::Vector3d const& pos) { root_position_ = pos; }

    /// @brief Set root orientation
    /// @param quat New orientation (will be normalized)
    void set_root_orientation(Eigen::Quaterniond const& quat) {
        root_orientation_ = quat.normalized();
    }

    /// @brief Set joint angles
    /// @param angles New joint angles in radians
    void set_joint_angles(Eigen::VectorXd const& angles) { joint_angles_ = angles; }

    /// @brief Set root velocity
    /// @param vel New root velocity
    void set_root_velocity(Eigen::VectorXd const& vel) { root_velocity_ = vel; }

    /// @brief Set joint velocities
    /// @param vel New joint velocities
    void set_joint_velocities(Eigen::VectorXd const& vel) { joint_velocities_ = vel; }

    /// @brief Get total error-state dimension
    /// @return Dimension of error-state vector (3 position + 3 orientation + n_dof joints)
    int error_state_dim() const { return 3 + 3 + joint_angles_.size(); }

    /// @brief Get number of degrees of freedom
    /// @return Number of joint angles
    int num_dof() const { return static_cast<int>(joint_angles_.size()); }

    /// @brief Convert to error-state vector
    ///
    /// Converts state to error-state representation suitable for filtering.
    /// Uses axis-angle representation for orientation (tangent space).
    ///
    /// @return Error-state vector [position(3), axis_angle(3), joint_angles(n)]
    Eigen::VectorXd to_error_vector() const;

    /// @brief Apply error-state update with manifold operations
    ///
    /// Updates state using error-state delta. Position and joint angles
    /// are updated additively, while orientation uses multiplicative update
    /// on the quaternion manifold.
    ///
    /// @param error_delta Error-state update vector
    /// @throws std::invalid_argument if error_delta size doesn't match error_state_dim()
    void apply_error_update(Eigen::VectorXd const& error_delta);

    /// @brief Convert quaternion to axis-angle representation
    ///
    /// Extracts rotation axis and angle from unit quaternion.
    /// For small rotations (near identity), returns zero vector.
    ///
    /// @param q Input quaternion (will be normalized)
    /// @return Axis-angle vector where magnitude is angle and direction is axis
    static Eigen::Vector3d quaternion_to_axis_angle(Eigen::Quaterniond const& q);

    /// @brief Convert axis-angle to quaternion
    ///
    /// Constructs unit quaternion from axis-angle representation.
    /// For small angles, returns identity quaternion.
    ///
    /// @param axis_angle Rotation vector (magnitude is angle, direction is axis)
    /// @return Unit quaternion representing the rotation
    static Eigen::Quaterniond axis_angle_to_quaternion(Eigen::Vector3d const& axis_angle);

    /// @brief Serialize to JSON
    /// @return JSON object with all state components
    nlohmann::json to_json() const;

    /// @brief Deserialize from JSON
    /// @param j JSON object with state data
    /// @return Reconstructed State object
    /// @throws nlohmann::json::exception if required fields are missing
    static State from_json(nlohmann::json const& j);

   private:
    Eigen::Vector3d root_position_;        ///< Root position in world frame
    Eigen::Quaterniond root_orientation_;  ///< Root orientation (wxyz convention)
    Eigen::VectorXd joint_angles_;         ///< Joint angles (radians)
    Eigen::VectorXd root_velocity_;        ///< Root linear velocity
    Eigen::VectorXd joint_velocities_;     ///< Joint angular velocities
};

}  // namespace posetrak

#include <posetrak/core/state.hpp>

#include <cmath>
#include <stdexcept>

namespace posetrak {

/// @brief Construct state with zero-initialized components
State::State(int num_dof)
    : root_position_(Eigen::Vector3d::Zero()), root_orientation_(Eigen::Quaterniond::Identity()) {
    if (num_dof < 0) {
        throw std::invalid_argument("Number of DOF must be non-negative");
    }
    joint_angles_ = Eigen::VectorXd::Zero(num_dof);
    root_velocity_ = Eigen::VectorXd::Zero(3);
    joint_velocities_ = Eigen::VectorXd::Zero(num_dof);
}

/// @brief Construct state from individual components with validation
State::State(Eigen::Vector3d const& root_position, Eigen::Quaterniond const& root_orientation,
             Eigen::VectorXd const& joint_angles, Eigen::VectorXd const& root_velocity,
             Eigen::VectorXd const& joint_velocities)
    : root_position_(root_position),
      root_orientation_(root_orientation.normalized()),
      joint_angles_(joint_angles),
      root_velocity_(root_velocity),
      joint_velocities_(joint_velocities) {
    if (joint_angles.size() != joint_velocities.size()) {
        throw std::invalid_argument("Joint angles and velocities must have same size");
    }
    if (root_velocity.size() != 3) {
        throw std::invalid_argument("Root velocity must be 3D");
    }
}

/// @brief Convert state to error-state representation for filtering
Eigen::VectorXd State::to_error_vector() const {
    int const n_dof = num_dof();
    Eigen::VectorXd error(3 + 3 + n_dof);

    // Position (direct)
    error.segment<3>(0) = root_position_;

    // Orientation (axis-angle)
    error.segment<3>(3) = quaternion_to_axis_angle(root_orientation_);

    // Joint angles (direct)
    error.segment(6, n_dof) = joint_angles_;

    return error;
}

/// @brief Apply additive and manifold updates to state components
void State::apply_error_update(Eigen::VectorXd const& error_delta) {
    int const n_dof = num_dof();
    if (error_delta.size() != 3 + 3 + n_dof) {
        throw std::invalid_argument("Error delta size mismatch");
    }

    // Update position (additive)
    root_position_ += error_delta.segment<3>(0);

    // Update orientation (multiplicative on manifold)
    Eigen::Vector3d const delta_axis_angle = error_delta.segment<3>(3);
    Eigen::Quaterniond const delta_q = axis_angle_to_quaternion(delta_axis_angle);
    root_orientation_ = (delta_q * root_orientation_).normalized();

    // Update joint angles (additive)
    joint_angles_ += error_delta.segment(6, n_dof);
}

/// @brief Extract axis and angle from quaternion using logarithmic map
Eigen::Vector3d State::quaternion_to_axis_angle(Eigen::Quaterniond const& q) {
    // Ensure quaternion is normalized
    Eigen::Quaterniond qn = q.normalized();

    // Extract angle
    double const angle = 2.0 * std::acos(std::clamp(qn.w(), -1.0, 1.0));

    // Handle small angles (near identity)
    double constexpr epsilon = 1e-8;
    if (angle < epsilon) {
        return Eigen::Vector3d::Zero();
    }

    // Extract axis and scale by angle
    double const sin_half_angle = std::sin(angle / 2.0);
    Eigen::Vector3d axis(qn.x(), qn.y(), qn.z());
    axis /= sin_half_angle;

    return angle * axis;
}

/// @brief Construct quaternion from axis-angle using exponential map
Eigen::Quaterniond State::axis_angle_to_quaternion(Eigen::Vector3d const& axis_angle) {
    double const angle = axis_angle.norm();

    // Handle small angles (near identity)
    double constexpr epsilon = 1e-8;
    if (angle < epsilon) {
        return Eigen::Quaterniond::Identity();
    }

    // Normalize axis and construct quaternion
    Eigen::Vector3d const axis = axis_angle / angle;
    double const half_angle = angle / 2.0;
    double const sin_half = std::sin(half_angle);
    double const cos_half = std::cos(half_angle);

    return Eigen::Quaterniond(cos_half, sin_half * axis.x(), sin_half * axis.y(),
                              sin_half * axis.z());
}

/// @brief Convert all state components to JSON format
nlohmann::json State::to_json() const {
    nlohmann::json j;

    // Root position
    j["root_position"] = {root_position_.x(), root_position_.y(), root_position_.z()};

    // Root orientation (wxyz)
    j["root_orientation"] = {root_orientation_.w(), root_orientation_.x(), root_orientation_.y(),
                             root_orientation_.z()};

    // Joint angles
    j["joint_angles"] =
        std::vector<double>(joint_angles_.data(), joint_angles_.data() + joint_angles_.size());

    // Velocities
    j["root_velocity"] = {root_velocity_(0), root_velocity_(1), root_velocity_(2)};
    j["joint_velocities"] = std::vector<double>(
        joint_velocities_.data(), joint_velocities_.data() + joint_velocities_.size());

    return j;
}

/// @brief Parse JSON and reconstruct State object
State State::from_json(nlohmann::json const& j) {
    // Parse root position
    auto const& pos_arr = j.at("root_position");
    Eigen::Vector3d root_position(pos_arr[0], pos_arr[1], pos_arr[2]);

    // Parse root orientation
    auto const& quat_arr = j.at("root_orientation");
    Eigen::Quaterniond root_orientation(quat_arr[0], quat_arr[1], quat_arr[2], quat_arr[3]);

    // Parse joint angles
    auto const& angles_arr = j.at("joint_angles");
    Eigen::VectorXd joint_angles(angles_arr.size());
    for (size_t i = 0; i < angles_arr.size(); ++i) {
        joint_angles(i) = angles_arr[i];
    }

    // Parse velocities
    auto const& vel_arr = j.at("root_velocity");
    Eigen::VectorXd root_velocity(3);
    root_velocity << vel_arr[0], vel_arr[1], vel_arr[2];

    auto const& jvel_arr = j.at("joint_velocities");
    Eigen::VectorXd joint_velocities(jvel_arr.size());
    for (size_t i = 0; i < jvel_arr.size(); ++i) {
        joint_velocities(i) = jvel_arr[i];
    }

    return State(root_position, root_orientation, joint_angles, root_velocity, joint_velocities);
}

}  // namespace posetrak

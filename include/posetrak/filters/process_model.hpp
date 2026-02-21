/**
 * @file process_model.hpp
 * @brief Process model for UKF prediction step
 *
 * Defines how the state evolves over time. The ConstantVelocityModel assumes
 * positions and joint angles change linearly with their respective velocities,
 * and velocities remain constant (with process noise added by UKF).
 */

#pragma once

#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/core/state.hpp"

namespace posetrak {

/**
 * @brief Abstract base class for process models
 *
 * Process models define how the state evolves over time during prediction.
 * They implement the function: x_{t+1} = f(x_t, dt)
 */
class ProcessModel {
   public:
    virtual ~ProcessModel() = default;

    /**
     * @brief Propagate state forward in time
     *
     * @param state Current state
     * @param dt Time step (seconds)
     * @return Predicted next state
     */
    virtual State propagate(State const& state, double dt) const = 0;

    /**
     * @brief Get process noise covariance matrix for given time step
     *
     * Process noise represents uncertainty in the motion model. Typically
     * scales with dt (or dt²) depending on the quantity.
     *
     * @param dt Time step (seconds)
     * @param state_dim Dimension of error state (for sizing matrix)
     * @return Process noise covariance Q (state_dim × state_dim)
     */
    virtual Eigen::MatrixXd get_process_noise(double dt, int state_dim) const = 0;
};

/**
 * @brief Constant velocity process model with joint limits
 *
 * Dynamics:
 * - Root position: p' = p + v * dt
 * - Root orientation: q' = q ⊗ exp(ω * dt / 2)
 * - Joint angles: θ' = θ + ω * dt
 * - All velocities: v' = v (constant, noise added by UKF)
 *
 * Joint limits are enforced after propagation for revolute joints.
 */
class ConstantVelocityModel : public ProcessModel {
   public:
    /**
     * @brief Construct constant velocity model
     *
     * @param skeleton Skeleton with joint hierarchy and limits
     * @param process_noise_std Standard deviation for process noise
     *        - Position: meters
     *        - Orientation: radians
     *        - Joint angles: radians
     *        - Velocities: meters/s or radians/s
     */
    explicit ConstantVelocityModel(std::shared_ptr<const SkeletonLayout> layout,
                                   double process_noise_std = 0.1);

    State propagate(State const& state, double dt) const override;

    Eigen::MatrixXd get_process_noise(double dt, int state_dim) const override;

    /**
     * @brief Set process noise standard deviation
     * @param std_dev Standard deviation (consistent units)
     */
    void set_process_noise_std(double std_dev);

    /**
     * @brief Get current process noise standard deviation
     */
    double get_process_noise_std() const { return process_noise_std_; }

   private:
    /**
     * @brief Enforce joint limits on propagated state
     *
     * For revolute joints, clamps angles to [min, max].
     * For spherical joints, limits are not enforced (quaternions normalized instead).
     *
     * @param state State to modify (in-place)
     */
    void enforce_joint_limits(State& state) const;

    std::shared_ptr<const SkeletonLayout> layout_;  ///< Precomputed DOF index table
    double process_noise_std_;
};

}  // namespace posetrak

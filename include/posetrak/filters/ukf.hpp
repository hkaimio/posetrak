/**
 * @file ukf.hpp
 * @brief Unscented Kalman Filter for pose tracking in joint space
 */

#pragma once

#include <Eigen/Core>

#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/state.hpp"
#include "posetrak/filters/process_model.hpp"
#include "posetrak/filters/sigma_points.hpp"
#include <memory>
#include <vector>

namespace posetrak {

/**
 * @brief Unscented Kalman Filter for joint space tracking
 *
 * Implements UKF with:
 * - Error-state formulation for manifold operations
 * - Constant velocity process model
 * - Manifold-aware mean and covariance computation
 */
class UnscentedKalmanFilter {
   public:
    /**
     * @brief Construct UKF with default parameters
     * @param skeleton Skeleton structure
     * @param process_noise_std Process noise standard deviation (m/s^2 for positions, rad/s^2
     * for rotations)
     * @param alpha UKF spread parameter (default: 0.001)
     * @param beta UKF distribution parameter (default: 2.0 for Gaussian)
     * @param kappa UKF secondary scaling (default: 0.0)
     */
    UnscentedKalmanFilter(Skeleton const& skeleton, double process_noise_std = 0.1,
                          double alpha = 0.001, double beta = 2.0, double kappa = 0.0);

    /**
     * @brief Prediction step: propagate state and covariance forward in time
     * @param dt Time step in seconds
     *
     * Uses constant velocity process model:
     * - Position: p(t+dt) = p(t) + v*dt
     * - Quaternion: q(t+dt) = q(t) ⊗ exp(ω*dt/2)
     * - Velocities: v(t+dt) = v(t), ω(t+dt) = ω(t)
     */
    void predict(double dt);

    /**
     * @brief Get current state estimate
     * @return Current state
     */
    State const& state() const { return state_; }

    /**
     * @brief Get current covariance estimate (in error space)
     * @return Covariance matrix
     */
    Eigen::MatrixXd const& covariance() const { return covariance_; }

    /**
     * @brief Set current state
     * @param state New state
     */
    void set_state(State const& state) { state_ = state; }

    /**
     * @brief Set current covariance
     * @param covariance New covariance (in error space)
     * @throws std::invalid_argument if size doesn't match error dimension
     */
    void set_covariance(Eigen::MatrixXd const& covariance);

    /**
     * @brief Get error state dimension
     * @return Dimension of error state (2 * (6 + active_dof))
     */
    int error_dim() const { return sigma_gen_.error_dim(); }

   private:
    /**
     * @brief Compute weighted mean of states (manifold-aware)
     * @param states Sigma points
     * @param weights Mean weights
     * @return Mean state
     */
    State compute_state_mean(std::vector<State> const& states,
                             Eigen::VectorXd const& weights) const;

    /**
     * @brief Compute state covariance in error space
     * @param states Sigma points
     * @param mean_state Mean state
     * @param weights Covariance weights
     * @return Covariance matrix in error space
     */
    Eigen::MatrixXd compute_state_covariance(std::vector<State> const& states,
                                             State const& mean_state,
                                             Eigen::VectorXd const& weights) const;

    /**
     * @brief Compute error between two states in tangent space
     * @param state State to compute error for
     * @param reference Reference state
     * @return Error vector in tangent space
     */
    Eigen::VectorXd compute_state_error(State const& state, State const& reference) const;

    Skeleton const& skeleton_;             ///< Skeleton structure
    State state_;                          ///< Current state estimate
    Eigen::MatrixXd covariance_;           ///< Covariance in error space
    Eigen::MatrixXd process_noise_;        ///< Process noise covariance
    SigmaPointGenerator sigma_gen_;        ///< Sigma point generator
    ConstantVelocityModel process_model_;  ///< Process model
};

}  // namespace posetrak

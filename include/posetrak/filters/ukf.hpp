/**
 * @file ukf.hpp
 * @brief Unscented Kalman Filter for pose tracking in joint space
 */

#pragma once

#include <Eigen/Core>

#include "posetrak/core/camera.hpp"
#include "posetrak/core/observation.hpp"
#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/state.hpp"
#include "posetrak/filters/process_model.hpp"
#include "posetrak/filters/sigma_points.hpp"
#include "posetrak/filters/update_result.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include <memory>
#include <optional>
#include <unordered_map>
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
     * @brief Update step: correct state with observations
     * @param observations Marker observations from cameras
     * @param cameras Map of camera_id -> Camera
     * @param fk Forward kinematics computer
     * @param measurement_noise_std Measurement noise standard deviation (pixels)
     * @param outlier_threshold_mahalanobis Mahalanobis distance threshold for outlier rejection
     *        (0.0 = disabled). Recommended: 5.991 (95% confidence, 2-DOF chi-squared)
     *
     * Updates state and covariance using unscented transform:
     * 1. For each sigma point: FK → camera projection
     * 2. Compute predicted measurements and innovation covariance
     * 3. Perform outlier rejection (if threshold > 0)
     * 4. Compute Kalman gain
     * 5. Update state and covariance
     *
     * @return UpdateResult with diagnostics (inliers, outliers, Mahalanobis distances)
     */
    UpdateResult update(std::vector<Observation> const& observations,
                        std::unordered_map<int, Camera> const& cameras, ForwardKinematics& fk,
                        double measurement_noise_std = 5.0,
                        double outlier_threshold_mahalanobis = 0.0);

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

    // Debug instrumentation
    /**
     * @brief Enable debug mode to export UKF internals
     * @param enable True to enable debug exports
     * @param debug_dir Directory to write debug files
     */
    void enable_debug(bool enable, std::string const& debug_dir = "cpp_results/debug");

    /**
     * @brief Set current frame number for debug exports
     * @param frame_num Frame number
     */
    void set_frame_number(int frame_num) { frame_number_ = frame_num; }

    // Testing-only accessors (for unit test verification)
    /**
     * @brief Generate sigma points from current state (for testing)
     * @return Vector of sigma point states
     */
    std::vector<State> generate_sigma_points_for_testing() const {
        return sigma_gen_.generate_sigma_points(state_, covariance_);
    }

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

    /**
     * @brief Predict measurements for a state using FK and camera projection
     * @param state State to predict measurements for
     * @param observations Observations to predict (defines which markers/cameras)
     * @param cameras Map of camera_id -> Camera
     * @param fk Forward kinematics computer
     * @return Vector of predicted pixel measurements (x1,y1,x2,y2,...)
     */
    Eigen::VectorXd predict_measurements(State const& state,
                                         std::vector<Observation> const& observations,
                                         std::unordered_map<int, Camera> const& cameras,
                                         ForwardKinematics& fk) const;

    /**
     * @brief Compute Mahalanobis distance for a 2D innovation
     * @param innovation Innovation vector [u_err, v_err]
     * @param covariance 2x2 covariance matrix
     * @return Mahalanobis distance
     */
    double compute_mahalanobis_distance(Eigen::Vector2d const& innovation,
                                        Eigen::Matrix2d const& covariance) const;

    /**
     * @brief Perform outlier rejection on observations
     * @param observations All observations
     * @param predicted_measurements Predicted measurements from sigma points (dim x n_sigma)
     * @param measurement_mean Mean predicted measurement
     * @param innovation_cov Innovation covariance matrix
     * @param threshold Mahalanobis distance threshold
     * @return Tuple of (inlier observations, all observation results)
     */
    std::pair<std::vector<Observation>, std::vector<ObservationResult>>
    reject_outliers(std::vector<Observation> const& observations,
                    Eigen::MatrixXd const& predicted_measurements,
                    Eigen::VectorXd const& measurement_mean, Eigen::MatrixXd const& innovation_cov,
                    double threshold) const;

    /**
     * @brief Compute observation diagnostics without rejection
     * @param observations All observations
     * @param measurement_mean Mean predicted measurement
     * @param innovation_cov Innovation covariance matrix
     * @return Vector of observation results with Mahalanobis distances
     */
    std::vector<ObservationResult>
    compute_observation_diagnostics(std::vector<Observation> const& observations,
                                    Eigen::VectorXd const& measurement_mean,
                                    Eigen::MatrixXd const& innovation_cov) const;

    /**
     * @brief Convert observations to measurement vector
     * @param observations Observations
     * @return Vector of pixel measurements (x1,y1,x2,y2,...)
     */
    Eigen::VectorXd observations_to_vector(std::vector<Observation> const& observations) const;

    /**
     * @brief Enforce joint limits on current state
     *
     * Clamps joint angles to their valid ranges and zeros out velocities
     * for joints that hit limits. Modifies state_ in place.
     */
    void enforce_joint_limits();

    /**
     * @brief Damp velocity covariance for joints that were modified by limit enforcement
     * @param prev_state State before limit enforcement
     * @param current_state State after limit enforcement
     * @param damping_factor Factor to multiply velocity covariance by (default 0.01)
     *
     * Compares the two states and damps velocity covariance for any velocities
     * that were changed by limit enforcement, preventing oscillation.
     */
    void damp_velocity_covariance_at_limits(State const& prev_state, State const& current_state,
                                            double damping_factor = 0.01);

    Skeleton const& skeleton_;             ///< Skeleton structure
    State state_;                          ///< Current state estimate
    Eigen::MatrixXd covariance_;           ///< Covariance in error space
    Eigen::MatrixXd process_noise_;        ///< Process noise covariance
    SigmaPointGenerator sigma_gen_;        ///< Sigma point generator
    ConstantVelocityModel process_model_;  ///< Process model

    // Debug state
    bool debug_enabled_ = false;  ///< Debug mode flag
    std::string debug_dir_;       ///< Debug output directory
    int frame_number_ = 0;        ///< Current frame number
};

}  // namespace posetrak

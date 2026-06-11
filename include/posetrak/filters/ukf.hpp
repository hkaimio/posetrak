/**
 * @file ukf.hpp
 * @brief Unscented Kalman Filter for pose tracking in joint space
 */

#pragma once

#include <Eigen/Core>

#include "posetrak/core/camera.hpp"
#include "posetrak/core/observation.hpp"
#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/core/state.hpp"
#include "posetrak/filters/process_model.hpp"
#include "posetrak/filters/sigma_points.hpp"
#include "posetrak/filters/update_result.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include <future>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace posetrak {

/**
 * @brief Result returned by UnscentedKalmanFilter::predict().
 *
 * Contains the sigma-point cross-covariance D between the posterior
 * x_{k|k} and the prior x_{k+1|k}.  Required by the RTS smoother.
 */
struct PredictResult {
    /// Async computation of D = sum_i W_c^i * e_pre_i * e_prop_i^T (error/tangent space).
    /// Shape: error_dim x error_dim.  Launched as std::async in predict(); resolve with get()
    /// after update() has run so the 16ms computation overlaps with the 56ms update step.
    std::future<Eigen::MatrixXd> cross_cov_future;

    // Per-operation wall times (milliseconds)
    double sigma_gen_ms = 0.0;  ///< Cholesky + sigma point generation
    double propagate_ms = 0.0;  ///< Process model propagation (n_sigma calls)
    double mean_cov_ms = 0.0;   ///< compute_state_mean + compute_state_covariance
    double rts_ms = 0.0;        ///< time to launch the async cross-cov task
};

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
    UnscentedKalmanFilter(std::shared_ptr<const SkeletonLayout> layout,
                          double process_noise_std = 0.1, double alpha = 0.001, double beta = 2.0,
                          double kappa = 0.0);

    /**
     * @brief Prediction step: propagate state and covariance forward in time
     * @param dt Time step in seconds
     *
     * Uses constant velocity process model:
     * - Position: p(t+dt) = p(t) + v*dt
     * - Quaternion: q(t+dt) = q(t) ⊗ exp(ω*dt/2)
     * - Velocities: v(t+dt) = v(t), ω(t+dt) = ω(t)
     *
     * @return PredictResult with sigma-point cross-covariance for RTS smoother.
     *         May be discarded by callers that do not use smoothing.
     */
    PredictResult predict(double dt);

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
     * @brief Inject the externally-known root transform for child-filter mode.
     *
     * Must be called once per frame by the coordinator before predict()/update().
     * Has no effect if layout_->has_floating_root() == true (safety no-op for
     * parent filters — calling it on a parent filter is harmless but meaningless).
     *
     * Updates state_ root immediately so predict()'s sigma generation starts from
     * the correct nominal root, and also stores the transform for overwriting
     * process-model root drift in every propagated sigma point.
     *
     * @param position     World-frame position of the freeflyer joint (e.g. wrist.R)
     * @param orientation  World-frame orientation of the freeflyer joint
     */
    void set_root_transform(Eigen::Vector3d const& position, Eigen::Quaterniond const& orientation);

    /**
     * @brief Get error state dimension
     * @return Dimension of error state (2 * (6 + active_dof))
     */
    int error_dim() const { return sigma_gen_.error_dim(); }

    /// Return the layout used to build this UKF (joints, DOF indices, etc.).
    std::shared_ptr<const SkeletonLayout> layout() const { return layout_; }

    // Debug instrumentation
    /**
     * @brief Enable debug mode to export UKF internals
     * @param enable True to enable debug exports
     * @param debug_dir Directory to write debug files
     */
    void enable_debug(bool enable, std::string const& debug_dir = "cpp_results/debug");

    /**
     * @brief Enable calibration mode: prismatic (bone-length) DOFs receive small
     * process noise so the filter can update bone lengths from marker residuals.
     * @param prismatic_noise_std Sigma per sqrt(s) for each prismatic DOF (default 0.1 mm/sqrt(s))
     */
    void enable_calibration_mode(double prismatic_noise_std = 0.0001);

    /**
     * @brief Set a separate process noise std for velocity DOFs.
     *
     * When set, the position/angle block uses the original process_noise_std and
     * the velocity block uses this value.  Call before the first predict() step.
     *
     * @param vel_noise_std Sigma for velocity DOFs (rad/s or m/s per sqrt(s)).
     *        Values larger than process_noise_std allow velocities to adapt quickly
     *        while keeping angle estimates tightly constrained.
     */
    void set_vel_noise_std(double vel_noise_std);

    /**
     * @brief Set the velocity half-life for exponential velocity damping.
     *
     * Applies a per-frame decay α = pow(0.5, dt / half_life_s) to all velocity
     * components of each propagated sigma point.  This caps steady-state velocity
     * variance at σ_vel² · half_life_s / (2·ln2) instead of growing unboundedly,
     * preventing covariance ill-conditioning over long runs.
     *
     * @param half_life_s  Time (seconds) for velocity to decay to half its value.
     *        Practical range: 0.25–2.0 s.  0.0 = no damping (default behaviour).
     */
    void set_vel_half_life(double half_life_s);

    /**
     * @brief Set current frame number for debug exports
     * @param frame_num Frame number
     */
    void set_frame_number(int frame_num) { frame_number_ = frame_num; }

    /**
     * @brief Check if debug mode is enabled
     * @return True if debug is enabled
     */
    bool is_debug_enabled() const { return debug_enabled_; }

    /**
     * @brief Get debug output directory
     * @return Debug directory path
     */
    std::string const& get_debug_dir() const { return debug_dir_; }

    /**
     * @brief Get current frame number
     * @return Frame number
     */
    int get_frame_number() const { return frame_number_; }

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
     * @brief Apply error to state (retraction from tangent space to manifold)
     * @param nominal_state Nominal state (on manifold)
     * @param error Error vector in tangent space (active DOFs only)
     * @return New state with error applied
     */
    State apply_error_to_state(State const& nominal_state, Eigen::VectorXd const& error) const;

    /**
     * @brief Predict measurements for a state using FK and camera projection.
     *
     * For VELOCITY-mode observations, the prediction is project(state) - prev_proj,
     * where prev_proj is the pre-computed projection of the previous posterior state.
     * prev_projections may be empty when no velocity observations are present.
     */
    Eigen::VectorXd
    predict_measurements(State const& state, std::vector<Observation> const& observations,
                         std::unordered_map<int, Camera> const& cameras, ForwardKinematics& fk,
                         std::unordered_map<int, std::unordered_map<int, Eigen::Vector2d>> const&
                             prev_projections) const;

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

    /**
     * @brief Write sigma points to CSV file for debugging
     * @param sigma_points Vector of sigma point states
     *
     * Writes sigma points in the same format as Python implementation for comparison.
     * CSV format: sigma_idx, root_pos (x,y,z), root_quat (w,x,y,z), root_vel (x,y,z),
     * root_angvel (x,y,z), joint angles (in skeleton order), joint velocities (in skeleton order)
     */
    void write_sigma_points_csv(std::vector<State> const& sigma_points) const;

    /**
     * @brief Write matrix to CSV file for debugging
     * @param matrix Matrix to write
     * @param filename Filename (without path, e.g., "prior_covariance.csv")
     *
     * Writes matrix to debug_dir/frame_XXXX/filename
     */
    void write_matrix_csv(Eigen::MatrixXd const& matrix, std::string const& filename) const;

    std::shared_ptr<const SkeletonLayout> layout_;  ///< Precomputed DOF index table
    State state_;                                   ///< Current state estimate
    Eigen::MatrixXd covariance_;                    ///< Covariance in error space
    Eigen::MatrixXd process_noise_;                 ///< Process noise covariance
    SigmaPointGenerator sigma_gen_;                 ///< Sigma point generator
    ConstantVelocityModel process_model_;           ///< Process model

    // Calibration mode
    double base_noise_std_;                ///< Process noise std for position/angle block
    double vel_noise_std_;                 ///< Process noise std for velocity block
    double vel_half_life_s_ = 0.0;         ///< Velocity decay half-life in seconds (0 = no decay)
    bool calibration_mode_ = false;        ///< Whether prismatic DOFs have active process noise
    double prismatic_noise_std_ = 0.0001;  ///< Sigma for prismatic DOFs in calibration mode
    void rebuild_process_noise();          ///< Rebuild process_noise_ matrix with per-DOF values

    // Posterior state saved at the start of predict() for velocity-mode measurement prediction.
    // Initialized to the same zero state as state_; overwritten on first predict() call.
    State prev_posterior_state_{0};

    // Child-filter fixed root (only meaningful when !layout_->has_floating_root())
    Eigen::Vector3d fixed_root_pos_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond fixed_root_ori_ = Eigen::Quaterniond::Identity();

    // Per-thread pinocchio Data pool for parallel FK evaluation
    mutable std::vector<pinocchio::Data> data_pool_;
    void ensure_data_pool(ForwardKinematics const& fk) const;

    // PSD fix statistics
    mutable int psd_fix_count_ = 0;  ///< Frames where LLT failed and eigensolver was needed
   public:
    int psd_fix_count() const { return psd_fix_count_; }

   private:
    // Debug state
    bool debug_enabled_ = false;  ///< Debug mode flag
    std::string debug_dir_;       ///< Debug output directory
    int frame_number_ = 0;        ///< Current frame number
};

}  // namespace posetrak

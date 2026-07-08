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
#include <string>
#include <unordered_map>
#include <unordered_set>
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
     * @param pose_noise_std Pose estimation error (pixels in model input image; scaled by
     *        obs.crop_scale to original video pixels)
     * @param calib_noise_std Calibration error (pixels in original video)
     * @param outlier_threshold_mahalanobis Mahalanobis distance threshold for outlier rejection
     *        (0.0 = disabled). Recommended: 5.991 (95% confidence, 2-DOF chi-squared)
     *
     * Per-observation noise: (pose_noise_std * crop_scale + calib_noise_std) / max(conf, 0.1)
     *
     * @return UpdateResult with diagnostics (inliers, outliers, Mahalanobis distances)
     */
    UpdateResult update(std::vector<Observation> const& observations,
                        std::unordered_map<int, Camera> const& cameras, ForwardKinematics& fk,
                        double pose_noise_std = 0.0, double calib_noise_std = 5.0,
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
     * @brief Configure velocity-driven per-DOF process noise (adaptive process
     * noise, Phase 1 / "Mechanism A" — see
     * docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md).
     *
     * Each active DOF's own process noise variance is scaled by
     * `(1 + gain * |velocity_dof| / vel_ref)^2`, clamped to
     * `kMaxVelocityNoiseMultiplier`, using that DOF's velocity from the posterior
     * state at the start of predict() (before propagation). Root position and
     * orientation DOFs share one gain/reference; joint DOFs share another, since
     * root moves in world units (metres/rad) and joints in radians and a shared
     * gain would conflate the two scales. Prismatic (bone-length) DOFs are never
     * scaled — they stay under the existing frozen/calibration-mode noise.
     *
     * A gain of 0.0 disables scaling for that DOF class and reproduces the exact
     * pre-Phase-1 static process noise.
     *
     * @param gain_joint Velocity gain for joint DOFs (0.0 = disabled).
     * @param vel_ref_joint Reference velocity for joint DOFs (rad/s). Must be > 0
     *        if gain_joint > 0.
     * @param gain_root Velocity gain for root DOFs (0.0 = disabled).
     * @param vel_ref_root Reference velocity for root DOFs (m/s for position,
     *        rad/s for orientation — both share this one reference). Must be > 0
     *        if gain_root > 0.
     * @param joint_names Literal skeleton joint names (Joint::name, e.g. "spine1",
     *        "thigh.L") the joint gain applies to. Empty (default) = all joints.
     *        Joints not in this list keep the plain static process noise --
     *        added after finding that a single body-wide gain over-loosens fast,
     *        normally-fast-moving limbs (arms) while barely engaging for the
     *        slower torso/hip motion it was tuned for, degrading limb tracking.
     *        Deliberately name-based rather than skeleton-group-based: group
     *        definitions in existing skeleton YAMLs aren't fine-grained enough
     *        (e.g. one "main" group spanning the whole body) and adding a finer
     *        split would mean editing every person's skeleton file. Does not
     *        affect gain_root/vel_ref_root.
     */
    void set_velocity_noise_gain(double gain_joint, double vel_ref_joint, double gain_root,
                                 double vel_ref_root,
                                 std::vector<std::string> const& joint_names = {});

    /**
     * @brief Configure a second, independent velocity-driven gain for a disjoint
     * joint scope (e.g. arms) — same Singer-model formula as
     * set_velocity_noise_gain(), but with its own gain/reference velocity so it
     * doesn't have to share set_velocity_noise_gain()'s joint_names tuning.
     *
     * Added after set_velocity_noise_gain()'s joint_names scoping excluded arms
     * entirely (see its doc comment) to fix the original body-wide-gain
     * regression, which left arms with only the reactive NIS-feedback safety net
     * (set_nis_feedback_scopes()) and no proactive velocity-driven headroom.
     * Once pose regularization (set_pose_regularization()) separately fixed the
     * spine issue that originally forced a high core gain, a modest, independent
     * arm gain became worth trying without reintroducing that regression.
     *
     * A joint listed in both this scope and set_velocity_noise_gain()'s
     * joint_names is scaled by the primary scope only (this one is skipped for
     * it) -- scopes are expected to be disjoint (core vs. arms) by construction.
     *
     * @param gain Velocity gain for this scope's joint DOFs (0.0 = disabled).
     * @param vel_ref Reference velocity (rad/s). Must be > 0 if gain > 0.
     * @param joint_names Literal skeleton joint names this scope covers. Empty =
     *        disabled (not "all joints" -- unlike the primary scope, there's no
     *        sensible default here).
     */
    void set_velocity_noise_gain_arms(double gain, double vel_ref,
                                      std::vector<std::string> const& joint_names);

    /**
     * @brief Configure pose regularization for a kinematically redundant joint
     * chain (e.g. spine1/spine2) — see
     * docs/roadmap/features/pose-regularization/pose-regularization-design.md.
     *
     * Fuses two soft pseudo-measurements into a second, small Kalman update
     * pass run at the end of update(), after the real camera-observation
     * correction: for every pair of joints in joint_names, per shared active
     * axis, a pseudo-residual `angle_i - angle_j -> 0` (equal-split); and for
     * every joint in joint_names, per active axis, a pseudo-residual
     * `angle_i -> 0` (rest-pose pull — the joint's own zero configuration,
     * since Skeleton::Joint::rest_orientation is already baked into the
     * kinematic model's fixed frame, not a target in State::joint_angles()
     * space). Both use the same sigma-point/Kalman-gain machinery as the main
     * update, so they degrade gracefully in the presence of strong,
     * disambiguating real observations and only meaningfully act when the
     * real data leaves the redundant direction underdetermined.
     *
     * Not anatomically accurate by design — a heuristic to avoid one joint in
     * a redundant chain absorbing all available rotation (and hitting its own
     * limit) while the others stay near neutral, not a biomechanical model.
     *
     * @param joint_names Joints forming the redundant chain (e.g.
     *        {"spine1", "spine2"}). Fewer than 2 disables the equal-split
     *        residual (rest-pose can still apply to a single joint). Empty
     *        disables the whole mechanism.
     * @param equal_split_noise_std Std of the equal-split pseudo-measurement
     *        (radians) — the "stiffness" of that spring. 0.0 = disabled.
     * @param rest_pose_noise_std Std of the rest-pose pseudo-measurement
     *        (radians). 0.0 = disabled.
     */
    void set_pose_regularization(std::vector<std::string> const& joint_names,
                                 double equal_split_noise_std, double rest_pose_noise_std);

    /**
     * @brief Configure NIS-feedback regional fading scopes (Mechanism B) — see
     * docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md.
     *
     * A reactive safety net alongside Mechanism A (set_velocity_noise_gain()):
     * each scope is a named group of joints (e.g. "arms"). Every predict() call,
     * that scope's currently-set multiplier (see set_scope_noise_multiplier())
     * is applied on top of Mechanism A's velocity-driven scaling to every active
     * DOF of every joint in the scope — same per-DOF loop, composed
     * multiplicatively in variance domain. The multiplier itself is computed
     * upstream (Tracker, from a windowed average of per-observation Mahalanobis
     * distances attributed to the scope's joints via ObservationResult::marker_name)
     * since that bookkeeping isn't intrinsic to the UKF's own state.
     *
     * Root DOFs are not covered by any scope (scopes are joint-name lists;
     * Mechanism A already has its own separate root gain).
     *
     * @param scopes (name, joint_names) pairs. A joint may appear in more than
     *        one scope; their multipliers compose multiplicatively. Empty =
     *        disabled. Calling this resets any previously-set multipliers to 1.0.
     */
    void set_nis_feedback_scopes(
        std::vector<std::pair<std::string, std::vector<std::string>>> const& scopes);

    /**
     * @brief Set the current per-scope process-noise multiplier for Mechanism B.
     * Called once per step (typically by Tracker, before the next predict())
     * with the scope's latest windowed NIS/DOF-derived value. Unrecognized
     * scope_name (not passed to set_nis_feedback_scopes()) is a silent no-op.
     * @param scope_name Must match a name passed to set_nis_feedback_scopes().
     * @param multiplier Variance-domain multiplier, >= 1.0 expected (< 1.0 is
     *        accepted but unusual — Mechanism B only ever widens noise).
     */
    void set_scope_noise_multiplier(std::string const& scope_name, double multiplier);

    /**
     * @brief Per-scope process-noise multiplier most recently set via
     * set_scope_noise_multiplier(). Debug/tuning use only.
     */
    std::unordered_map<std::string, double> const& last_scope_noise_multipliers() const {
        return nis_feedback_scope_multiplier_;
    }

    /**
     * @brief Per-DOF velocity-noise multiplier computed by the most recent predict()
     * call, keyed by absolute error-state position index (same indexing as
     * process_noise_'s position block). Empty if velocity-driven scaling is
     * disabled (both gains 0.0). Debug/tuning use only.
     */
    std::unordered_map<int, double> const& last_velocity_noise_scale() const {
        return vel_noise_scale_debug_;
    }

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

    /**
     * @brief Run the pose-regularization update pass directly (for testing),
     * without needing a full camera-observation update() call.
     * @see set_pose_regularization()
     */
    void apply_pose_regularization_for_testing() { apply_pose_regularization(); }

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
     * @brief Append one row to debug_dir_/process_noise_velocity_scale.csv with the
     * per-DOF velocity-noise std multiplier from the most recent apply_velocity_scaling()
     * call (see last_velocity_noise_scale()).
     *
     * Unlike the frame-0/frame-1-only debug exports above, this runs every predict()
     * call so the tuning process can inspect how the gain reacts frame-to-frame over
     * a whole run without rebuilding. Writes the header (root + per-joint-DOF column
     * names, from layout_) on frame 0 only. No-op if velocity scaling is disabled
     * (both gains 0.0) -- nothing changes frame to frame in that case anyway.
     */
    void write_velocity_noise_scale_csv() const;

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

    // Velocity-driven per-DOF process noise (Phase 1 adaptive process noise).
    // 0.0 gain = disabled, reproduces the exact static process_noise_ built by
    // rebuild_process_noise() above.
    double vel_noise_gain_joint_ = 0.0;
    double vel_noise_ref_joint_ = 1.0;
    double vel_noise_gain_root_ = 0.0;
    double vel_noise_ref_root_ = 1.0;
    // Set directly from set_velocity_noise_gain()'s joint_names argument: joints the
    // joint gain applies to. Empty means "all joints" -- checked via
    // vel_noise_joint_scope_all_ rather than by leaving this empty and treating
    // empty-set as "no joints", which would silently disable the feature by default.
    std::unordered_set<std::string> vel_noise_joint_names_;
    bool vel_noise_joint_scope_all_ = true;
    // Second, independent gain scope -- see set_velocity_noise_gain_arms(). 0.0 gain =
    // disabled. Unlike vel_noise_joint_names_, empty joint_names means "no joints"
    // (there's no sensible "all joints" default for a secondary scope).
    double vel_noise_gain_arms_ = 0.0;
    double vel_noise_ref_arms_ = 1.0;
    std::unordered_set<std::string> vel_noise_joint_names_arms_;
    static constexpr double kMaxVelocityNoiseMultiplier = 10.0;
    /// Returns a copy of the static process_noise_ baseline (as built by
    /// rebuild_process_noise()) with each active DOF's diagonal entries scaled by
    /// its own velocity-driven multiplier; returns process_noise_ unchanged if both
    /// gains are 0.0. Called fresh every predict() -- never mutates process_noise_
    /// itself, so per-step scaling never compounds across frames.
    Eigen::MatrixXd apply_velocity_scaling(State const& velocity_state) const;
    /// Per-DOF multiplier from the most recent apply_velocity_scaling() call, for
    /// last_velocity_noise_scale()'s debug/tuning accessor above.
    mutable std::unordered_map<int, double> vel_noise_scale_debug_;

    // Pose regularization (kinematic redundancy) -- see set_pose_regularization().
    // Precomputed once there from the configured joint_names, as (state_index_a,
    // state_index_b) pairs (one per shared active axis) for the equal-split
    // residual, and state_index list (one per active axis) for the rest-pose
    // residual. Empty = disabled.
    std::vector<std::pair<int, int>> pose_reg_pairs_;
    std::vector<int> pose_reg_rest_indices_;
    double pose_reg_equal_split_var_ = 0.0;
    double pose_reg_rest_pose_var_ = 0.0;
    /// Second, small Kalman update pass fusing the pose-regularization
    /// pseudo-measurements, run at the end of update() after the real
    /// camera-observation correction. No-op if set_pose_regularization() was
    /// never called with a non-empty joint_names.
    void apply_pose_regularization();

    // NIS-feedback regional fading safety net (Mechanism B) -- see
    // set_nis_feedback_scopes() / set_scope_noise_multiplier(). joint_to_scope_names_
    // is precomputed once in set_nis_feedback_scopes() (joint name -> scope names it
    // belongs to, usually one, multiple allowed); nis_feedback_scope_multiplier_ is
    // updated every step by set_scope_noise_multiplier(), defaulting to 1.0 (no-op)
    // for any scope never explicitly set. Empty joint_to_scope_names_ = disabled.
    std::unordered_map<std::string, std::vector<std::string>> joint_to_scope_names_;
    std::unordered_map<std::string, double> nis_feedback_scope_multiplier_;
    /// Returns a copy of scaled_in (Mechanism A's output) with each active DOF's
    /// diagonal entries additionally scaled by its joint's current scope
    /// multiplier/multipliers (multiplicative if more than one); returns scaled_in
    /// unchanged if no scopes are configured. Root DOFs are never scoped (see
    /// set_nis_feedback_scopes() doc comment).
    Eigen::MatrixXd apply_nis_feedback_scaling(Eigen::MatrixXd const& scaled_in) const;

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

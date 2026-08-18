// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace posetrak {

/**
 * @brief One named scope for the NIS-feedback regional fading safety net
 * (Mechanism B) -- see
 * docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md.
 */
struct NisFeedbackScope {
    std::string name;  ///< Identifier used in debug export / logging.
    std::vector<std::string>
        joint_names;  ///< Joints this scope covers (e.g. {"upper_arm.L", ...}).
};

/**
 * @brief One additional, independent adaptive process noise gain scope beyond
 * the primary one -- see UnscentedKalmanFilter::set_velocity_noise_gain_scopes().
 */
struct VelocityNoiseScope {
    std::string name;                      ///< Identifier used in debug/logging only.
    std::vector<std::string> joint_names;  ///< Joints this scope covers.
    double gain = 0.0;                     ///< 0.0 = this scope disabled.
    double vel_ref = 1.0;                  ///< Reference velocity (rad/s).
};

/**
 * @brief Configuration parameters for Tracker
 */
struct TrackerConfig {
    // UKF parameters
    double process_noise_std = 0.1;  ///< Process noise std for angle/position DOFs
    std::optional<double>
        process_noise_vel_std;  ///< Process noise std for velocity DOFs (nullopt = same as pos)
    std::optional<double>
        velocity_half_life_s;          ///< Velocity decay half-life in seconds (nullopt = no decay)
    double pose_noise_std = 0.0;       ///< Pose estimation error (pixels in model input image)
    double calib_noise_std = 5.0;      ///< Calibration error (pixels in original video)
    double outlier_threshold = 5.991;  ///< Chi-squared threshold (95% for 2-DOF)

    // === Adaptive process noise (Phase 1 — velocity-driven per-DOF scaling) ===
    // See docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md.
    // 0.0 gain = disabled (exact pre-Phase-1 static process noise).
    double process_noise_vel_gain_joint = 0.0;  ///< Velocity gain for joint DOFs
    double process_noise_vel_ref_joint = 1.0;   ///< Reference velocity for joint DOFs (rad/s)
    double process_noise_vel_gain_root = 0.0;   ///< Velocity gain for root DOFs
    double process_noise_vel_ref_root = 1.0;    ///< Reference velocity for root DOFs (m/s, rad/s)
    /// Literal joint names (e.g. "spine1", "thigh.L") the joint gain applies to.
    /// Empty (default) = all joints. Added after finding a body-wide gain
    /// over-loosens fast-but-normal limb motion (arms) while barely engaging for
    /// the slower torso/hip motion it targets. Name-based rather than
    /// skeleton-group-based since existing skeleton YAMLs don't define groups
    /// fine-grained enough (one "main" group spans the whole body) and adding a
    /// finer split would mean editing every person's skeleton file.
    std::vector<std::string> process_noise_vel_joint_names;
    /// Additional, independent velocity gain scopes beyond the primary one -- see
    /// UnscentedKalmanFilter::set_velocity_noise_gain_scopes(). Empty = none.
    std::vector<VelocityNoiseScope> process_noise_vel_scopes;

    // === Pose regularization (kinematic redundancy) — Phase 1 ===
    // See docs/roadmap/features/pose-regularization/pose-regularization-design.md.
    // Empty pose_reg_joint_names = disabled.
    std::vector<std::string> pose_reg_joint_names;  ///< Redundant chain, e.g. {"spine1","spine2"}
    double pose_reg_equal_split_noise_std = 0.0;    ///< 0.0 = disabled
    double pose_reg_rest_pose_noise_std = 0.0;      ///< 0.0 = disabled

    // === Soft joint-limit repulsion — Phase 1 ===
    // See docs/roadmap/features/soft-joint-limits/soft-joint-limits-design.md.
    // Empty soft_limit_joint_names = disabled.
    std::vector<std::string> soft_limit_joint_names;  ///< e.g. {"upper_arm.L","upper_arm.R"}
    double soft_limit_margin_rad = 0.0;               ///< Width of the soft zone (radians)
    double soft_limit_noise_std = 0.0;                ///< 0.0 = disabled

    // === Near-limit process-noise damping ===
    // See docs/roadmap/features/tracking-crisis-debugging-log.md, "Proposals".
    // Empty near_limit_damping_joint_names = disabled.
    std::vector<std::string> near_limit_damping_joint_names;
    double near_limit_margin_rad = 0.0;
    double near_limit_spread_sigma = 3.0;
    double near_limit_damping_factor = 1.0;  ///< 1.0 = disabled

    // === NIS-feedback regional fading safety net (Mechanism B) ===
    // See docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md.
    // Empty nis_feedback_scopes = disabled.
    std::vector<NisFeedbackScope> nis_feedback_scopes;
    int nis_feedback_window = 8;                ///< Moving window size, in tracker steps
    double nis_feedback_threshold = 1.5;        ///< Windowed NIS/DOF above this triggers fading
    double nis_feedback_max_multiplier = 10.0;  ///< Cap on the variance-domain multiplier

    // === Trusted keypoint edits (Phase 0) ===
    // See docs/roadmap/features/hand-detection-refinement/hand-detection-refinement-design.md.
    // 0.0 = disabled (edited keypoints use the normal computed noise and outlier gate,
    // same as any other observation). When > 0: every keypoint slot touched by a
    // pose_observation_edits row (human-placed, is_outlier=false) gets this value as
    // Observation::noise_std_override instead of the usual pose/calibration-error
    // formula, and is exempted from both outlier-rejection paths in
    // UnscentedKalmanFilter::reject_outliers() (Observation::force_inlier). The right
    // value is an open empirical question -- there is no principled default, a human
    // placing a keypoint is not zero-error. Interim convention: set to the run's own
    // calib_noise_std (puts edits on par with a normal detection, not privileged).
    double edited_kp_noise_std = 0.0;

    // UKF sigma point parameters
    double ukf_alpha = 0.5;  ///< Sigma point spread (0.001 for Python compatibility)
    double ukf_beta = 2.0;   ///< Gaussian distribution parameter
    double ukf_kappa = 0.0;  ///< Secondary scaling parameter

    // Initialization parameters
    // With n≈218 error DOFs and alpha=0.5, sigma point spread = sqrt(n+λ) ≈ 7.4 × init_std.
    // Values must be small enough that sigma points stay in the linear regime of camera
    // projection (sigma_spread_orient = 7.4 × 0.05 ≈ 0.37 rad ≈ 21°).
    // Larger init_std (e.g. 0.5 rad from old defaults) causes sigma points at ±211°,
    // corrupting the cross-covariance and producing a catastrophic first-frame update.
    // These defaults reflect post-IK accuracy (~3 cm / ~3° root, ~3° joints).
    double init_position_std = 0.05;     ///< Initial position uncertainty (meters)
    double init_orientation_std = 0.05;  ///< Initial orientation uncertainty (radians)
    double init_joint_std = 0.05;        ///< Initial joint angle uncertainty (radians)
    double init_velocity_std = 0.01;     ///< Initial velocity uncertainty (m/s or rad/s)

    int ik_max_iterations = 1000;  ///< Max IK iterations for initialization
    double ik_tolerance = 0.01;    ///< IK convergence tolerance (meters)
    int min_cameras_for_init = 2;  ///< Minimum cameras required for triangulation

    // Layout selection
    std::vector<std::string> active_joint_groups;  ///< Joint groups to track (empty = all)

    /// Explicit freeflyer/anchor joint for the subtree model built from
    /// active_joint_groups. Empty (default) = use the skeleton's own root
    /// joint, exactly today's behaviour -- a genuinely floating root,
    /// estimated by the UKF. Non-empty = build the subtree rooted at this
    /// named joint instead; if active_joint_groups excludes it and every
    /// joint between it and the skeleton's true root (the child-filter
    /// case -- e.g. "forearm.L" with active_joint_groups={"HandL"}),
    /// SkeletonLayout::from_groups() naturally reports has_floating_root()
    /// == false (it only sets that flag when it reaches the skeleton's
    /// actual root while filtering by group, per skeleton_layout.cpp), and
    /// the resulting Tracker expects the caller to drive its root pose
    /// externally every frame via set_external_root_transform() before
    /// track_frame() -- see
    /// docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md.
    std::string fixed_root_joint_name;

    // === Velocity-mode cameras ===
    /// Camera IDs that use frame-to-frame pixel velocity instead of absolute position.
    /// Useful for cameras with large systematic extrinsic or lens-distortion errors.
    std::vector<int> velocity_mode_camera_ids;
    /// Measurement noise std for velocity-mode cameras (pixels/frame).
    /// Typically smaller than calib_noise_std because the systematic bias cancels in the diff.
    /// nullopt = use calib_noise_std (conservative fallback).
    std::optional<double> velocity_measurement_noise_std;

    // === Relative observations (Phase 3 — hierarchical pairs) ===
    /// When true, a RELATIVE observation is emitted for each (child, parent) marker pair
    /// visible in the same frame/camera with sufficient confidence. Calibration error cancels
    /// in the pixel difference, leaving only pose estimation noise (pose_noise_std * sqrt(2)).
    bool use_relative_observations = false;
    /// Minimum keypoint confidence for both child and parent to form a RELATIVE pair.
    double relative_min_confidence = 0.5;

    // === Relative observations (Phase 4 — spatial cross-pairs) ===
    /// When > 0, emit RELATIVE observations for marker pairs visible in the same frame/camera
    /// whose image-space distance (px) is below this threshold AND whose skeleton-tree distance
    /// is > 2 joint hops. Targets interactions like hands touching. 0 = disabled.
    double cross_pair_max_px = 0.0;
    /// Maximum number of spatial cross-pairs to emit per frame per camera (sorted by proximity).
    int cross_pair_max_n = 10;

    // === Cross-person relative observations (Phase 5 — MultiPersonTracker) ===
    /// When > 0, marker pairs from two different people within this 3D world distance (mm)
    /// become candidates for cross-person PAIR_DIFF anchoring. 0 = feature disabled.
    double cross_person_max_world_mm = 0.0;
    /// Minimum keypoint confidence for both people's detections to form a cross-person anchor.
    double cross_person_min_confidence = 0.5;
    /// Maximum number of cross-person anchor observations per person pair per camera per frame
    /// (sorted by proximity), mirroring cross_pair_max_n.
    int cross_person_max_n = 10;

    // === Calibration ===
    bool calibration_mode = false;  ///< Enable bone-length calibration DOFs
    double prismatic_process_noise_std =
        0.0001;  ///< σ for prismatic DOFs in calibration mode (m/√s)

    // === Debug ===
    /// Print per-marker 3D errors (prior and posterior vs triangulated) for the first N frames.
    int debug_init_frames = 0;
};

/**
 * @brief Application configuration for tracker command-line tool
 *
 * All parameters needed to run the tracker CLI, loaded from TOML file.
 * Contains both tracking parameters and file paths/output options.
 */
struct TrackerAppConfig {
    // === Data paths ===
    std::filesystem::path skeleton_path;
    std::filesystem::path cameras_path;
    std::optional<std::filesystem::path> sync_path;
    std::filesystem::path observations_dir;
    int person_id = 0;
    std::vector<std::string> active_joint_groups;  ///< Joint groups to track (empty = all)

    // === Tracking parameters ===
    double process_noise_std = 0.5;
    std::optional<double>
        process_noise_vel_std;  ///< Velocity DOF noise std (nullopt = same as pos)
    std::optional<double>
        velocity_half_life_s;      ///< Velocity decay half-life in seconds (nullopt = no decay)
    double pose_noise_std = 0.0;   ///< Pose estimation error (pixels in model input image)
    double calib_noise_std = 2.0;  ///< Calibration error (pixels in original video)
    double outlier_threshold = 4.0;

    // === Adaptive process noise (Phase 1 — velocity-driven per-DOF scaling) ===
    // 0.0 gain = disabled (exact pre-Phase-1 static process noise).
    double process_noise_vel_gain_joint = 0.0;
    double process_noise_vel_ref_joint = 1.0;
    double process_noise_vel_gain_root = 0.0;
    double process_noise_vel_ref_root = 1.0;
    std::vector<std::string> process_noise_vel_joint_names;
    std::vector<VelocityNoiseScope> process_noise_vel_scopes;

    // === Pose regularization (kinematic redundancy) — Phase 1 ===
    std::vector<std::string> pose_reg_joint_names;
    double pose_reg_equal_split_noise_std = 0.0;
    double pose_reg_rest_pose_noise_std = 0.0;

    // === Soft joint-limit repulsion — Phase 1 ===
    std::vector<std::string> soft_limit_joint_names;
    double soft_limit_margin_rad = 0.0;
    double soft_limit_noise_std = 0.0;

    // === Near-limit process-noise damping ===
    std::vector<std::string> near_limit_damping_joint_names;
    double near_limit_margin_rad = 0.0;
    double near_limit_spread_sigma = 3.0;
    double near_limit_damping_factor = 1.0;

    // === NIS-feedback regional fading safety net (Mechanism B) ===
    std::vector<NisFeedbackScope> nis_feedback_scopes;
    int nis_feedback_window = 8;
    double nis_feedback_threshold = 1.5;
    double nis_feedback_max_multiplier = 10.0;

    // === Trusted keypoint edits (Phase 0) ===
    double edited_kp_noise_std = 0.0;

    // === Initialization ===
    std::optional<std::filesystem::path> python_state_path;  // Optional: use Python state for init
    int ik_max_iterations = 1000;
    double ik_tolerance = 0.02;
    double init_position_std = 0.1;
    double init_orientation_std = 0.1;
    double init_joint_std = 0.1;
    double init_velocity_std = 0.1;
    int min_cameras_for_init = 2;

    // === UKF parameters ===
    double ukf_alpha = 0.5;
    double ukf_beta = 2.0;
    double ukf_kappa = 0.0;

    // === Output ===
    std::filesystem::path output_dir = "tracking_output";
    bool export_tracking_results = true;
    bool export_statistics = true;
    bool export_debug = false;

    // === Processing ===
    double start_time = 0.0;     // Start time in seconds
    double end_time = -1.0;      // End time in seconds (-1 = use all data)
    double tracker_fps = 100.0;  // Tracker sample rate (Hz)

    // === Velocity-mode cameras ===
    std::vector<int> velocity_mode_camera_ids;
    std::optional<double> velocity_measurement_noise_std;

    // === Relative observations (Phase 3 — hierarchical pairs) ===
    bool use_relative_observations = false;
    double relative_min_confidence = 0.5;

    // === Relative observations (Phase 4 — spatial cross-pairs) ===
    double cross_pair_max_px = 0.0;
    int cross_pair_max_n = 10;

    // === Calibration ===
    bool calibration_mode = false;  ///< Enable bone-length calibration DOFs
    double prismatic_process_noise_std =
        0.0001;  ///< σ for prismatic DOFs in calibration mode (m/√s)

    // === Debug ===
    int debug_init_frames = 0;  ///< Print per-marker 3D errors for the first N tracked frames.

    /**
     * @brief Load configuration from TOML file
     *
     * @param config_path Path to TOML configuration file
     * @return Loaded configuration
     * @throws std::runtime_error if file cannot be loaded or parsed
     */
    static TrackerAppConfig load(std::filesystem::path const& config_path);

    /**
     * @brief Validate configuration
     *
     * Checks that:
     * - Required files exist
     * - Parameters are in valid ranges
     * - Output directory can be created
     *
     * @throws std::runtime_error if validation fails
     */
    void validate() const;

    /**
     * @brief Convert to TrackerConfig for the Tracker class
     *
     * Extracts just the tracking parameters needed by Tracker constructor.
     */
    TrackerConfig to_tracker_config() const;
};

// Inline implementation
inline TrackerConfig TrackerAppConfig::to_tracker_config() const {
    TrackerConfig tc;
    tc.process_noise_std = process_noise_std;
    tc.process_noise_vel_std = process_noise_vel_std;
    tc.velocity_half_life_s = velocity_half_life_s;
    tc.pose_noise_std = pose_noise_std;
    tc.calib_noise_std = calib_noise_std;
    tc.outlier_threshold = outlier_threshold;
    tc.process_noise_vel_gain_joint = process_noise_vel_gain_joint;
    tc.process_noise_vel_ref_joint = process_noise_vel_ref_joint;
    tc.process_noise_vel_gain_root = process_noise_vel_gain_root;
    tc.process_noise_vel_ref_root = process_noise_vel_ref_root;
    tc.process_noise_vel_joint_names = process_noise_vel_joint_names;
    tc.process_noise_vel_scopes = process_noise_vel_scopes;
    tc.pose_reg_joint_names = pose_reg_joint_names;
    tc.pose_reg_equal_split_noise_std = pose_reg_equal_split_noise_std;
    tc.pose_reg_rest_pose_noise_std = pose_reg_rest_pose_noise_std;
    tc.soft_limit_joint_names = soft_limit_joint_names;
    tc.soft_limit_margin_rad = soft_limit_margin_rad;
    tc.soft_limit_noise_std = soft_limit_noise_std;
    tc.near_limit_damping_joint_names = near_limit_damping_joint_names;
    tc.near_limit_margin_rad = near_limit_margin_rad;
    tc.near_limit_spread_sigma = near_limit_spread_sigma;
    tc.near_limit_damping_factor = near_limit_damping_factor;
    tc.nis_feedback_scopes = nis_feedback_scopes;
    tc.nis_feedback_window = nis_feedback_window;
    tc.nis_feedback_threshold = nis_feedback_threshold;
    tc.nis_feedback_max_multiplier = nis_feedback_max_multiplier;
    tc.edited_kp_noise_std = edited_kp_noise_std;
    tc.ukf_alpha = ukf_alpha;
    tc.ukf_beta = ukf_beta;
    tc.ukf_kappa = ukf_kappa;
    tc.init_position_std = init_position_std;
    tc.init_orientation_std = init_orientation_std;
    tc.init_joint_std = init_joint_std;
    tc.init_velocity_std = init_velocity_std;
    tc.ik_max_iterations = ik_max_iterations;
    tc.ik_tolerance = ik_tolerance;
    tc.min_cameras_for_init = min_cameras_for_init;
    tc.active_joint_groups = active_joint_groups;
    tc.velocity_mode_camera_ids = velocity_mode_camera_ids;
    tc.velocity_measurement_noise_std = velocity_measurement_noise_std;
    tc.use_relative_observations = use_relative_observations;
    tc.relative_min_confidence = relative_min_confidence;
    tc.cross_pair_max_px = cross_pair_max_px;
    tc.cross_pair_max_n = cross_pair_max_n;
    tc.calibration_mode = calibration_mode;
    tc.prismatic_process_noise_std = prismatic_process_noise_std;
    tc.debug_init_frames = debug_init_frames;
    return tc;
}

}  // namespace posetrak

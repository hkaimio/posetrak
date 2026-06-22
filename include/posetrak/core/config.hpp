#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace posetrak {

/**
 * @brief Configuration for a single child UKF filter in a hierarchical setup.
 *
 * A child filter tracks a named subset of joints (e.g. one hand) using only
 * the markers whose group is listed in observation_groups. It runs after the
 * parent filter has produced a world-space root pose for the child's root.
 */
struct ChildFilterConfig {
    /// Identifier used in log output and diagnostics.
    std::string name;

    /// Skeleton joint groups this filter tracks (e.g. {"HandR"}).
    std::vector<std::string> joint_groups;

    /// Marker observation groups this filter uses (e.g. {"HandR"}).
    /// Typically matches joint_groups but may differ when marker naming diverges.
    std::vector<std::string> observation_groups;

    double process_noise_std = 0.3;  ///< Process noise std for child joints.
    double pose_noise_std = 0.0;     ///< Pose estimation error (pixels in model input).
    double calib_noise_std = 2.0;    ///< Calibration error (pixels in original video).
    double outlier_threshold = 4.0;  ///< Chi-squared outlier rejection threshold.

    /// Reject update if fewer than this fraction of expected markers are inliers.
    double min_inliers_ratio = 0.3;

    /// Reject individual observation if innovation norm exceeds this (pixels).
    double max_innovation_norm = 200.0;
};

/**
 * @brief Configuration for hierarchical (multi-filter) tracking.
 *
 * When enabled, a parent filter tracks the body skeleton using only
 * parent_joint_groups, then each child filter refines its own subset
 * starting from the parent's pose estimate.
 *
 * When disabled (the default), the single monolithic filter in TrackerAppConfig
 * is used and all child entries are ignored.
 */
struct HierarchicalConfig {
    /// Enable hierarchical multi-filter tracking. False = monolithic tracker.
    bool enabled = false;

    /// After each parent step, synchronise child root poses from parent output.
    bool enable_sync = true;

    /// After child update, write back the child's covariance into the parent's
    /// covariance matrix for the child's DOFs.
    bool sync_covariance = false;

    /// Joint groups assigned to the parent filter.
    std::vector<std::string> parent_joint_groups;

    /// Marker observation groups consumed by the parent filter.
    std::vector<std::string> parent_observation_groups;

    double parent_process_noise_std = 0.5;
    double parent_pose_noise_std = 0.0;
    double parent_calib_noise_std = 2.0;
    double parent_outlier_threshold = 4.0;

    /// One entry per child filter, processed in order after the parent step.
    std::vector<ChildFilterConfig> children;
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

    // === Velocity-mode cameras ===
    /// Camera IDs that use frame-to-frame pixel velocity instead of absolute position.
    /// Useful for cameras with large systematic extrinsic or lens-distortion errors.
    std::vector<int> velocity_mode_camera_ids;
    /// Measurement noise std for velocity-mode cameras (pixels/frame).
    /// Typically smaller than calib_noise_std because the systematic bias cancels in the diff.
    /// nullopt = use calib_noise_std (conservative fallback).
    std::optional<double> velocity_measurement_noise_std;

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

    // === Hierarchical tracking ===
    HierarchicalConfig hierarchical;

    // === Velocity-mode cameras ===
    std::vector<int> velocity_mode_camera_ids;
    std::optional<double> velocity_measurement_noise_std;

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
    tc.calibration_mode = calibration_mode;
    tc.prismatic_process_noise_std = prismatic_process_noise_std;
    tc.debug_init_frames = debug_init_frames;
    return tc;
}

}  // namespace posetrak

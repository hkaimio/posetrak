#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace posetrak {

/**
 * @brief Configuration parameters for Tracker
 */
struct TrackerConfig {
    // UKF parameters
    double process_noise_std = 0.1;      ///< Process noise std deviation
    double measurement_noise_std = 5.0;  ///< Measurement noise std (pixels)
    double outlier_threshold = 5.991;    ///< Chi-squared threshold (95% for 2-DOF)

    // UKF sigma point parameters
    double ukf_alpha = 0.5;  ///< Sigma point spread (0.001 for Python compatibility)
    double ukf_beta = 2.0;   ///< Gaussian distribution parameter
    double ukf_kappa = 0.0;  ///< Secondary scaling parameter

    // Initialization parameters
    double init_position_std = 0.5;     ///< Initial position uncertainty (meters)
    double init_orientation_std = 0.5;  ///< Initial orientation uncertainty (radians)
    double init_joint_std = 0.3;        ///< Initial joint angle uncertainty (radians)
    double init_velocity_std = 0.1;     ///< Initial velocity uncertainty (m/s or rad/s)

    int ik_max_iterations = 50;    ///< Max IK iterations for initialization
    double ik_tolerance = 0.01;    ///< IK convergence tolerance (meters)
    int min_cameras_for_init = 2;  ///< Minimum cameras required for triangulation
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
    double measurement_noise_std = 2.0;
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
    tc.measurement_noise_std = measurement_noise_std;
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
    return tc;
}

}  // namespace posetrak

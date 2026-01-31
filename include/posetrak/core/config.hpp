#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace posetrak {

// Forward declaration - full definition in tracker.hpp
struct TrackerConfig;

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

    // === Tracking parameters ===
    double process_noise_std = 0.5;
    double measurement_noise_std = 2.0;
    double outlier_threshold = 4.0;

    // === Initialization ===
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
    int start_frame = 0;
    int max_frames = -1;  // -1 = all frames

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
     * Requires inclusion of tracker.hpp to use.
     */
    TrackerConfig to_tracker_config() const;
};

// Inline implementation (requires tracker.hpp to be included first)
#ifdef POSETRAK_TRACKER_HPP_INCLUDED
inline TrackerConfig TrackerAppConfig::to_tracker_config() const {
    TrackerConfig tc;
    tc.process_noise_std = process_noise_std;
    tc.measurement_noise_std = measurement_noise_std;
    tc.outlier_threshold = outlier_threshold;
    tc.init_position_std = init_position_std;
    tc.init_orientation_std = init_orientation_std;
    tc.init_joint_std = init_joint_std;
    tc.init_velocity_std = init_velocity_std;
    tc.ik_max_iterations = ik_max_iterations;
    tc.ik_tolerance = ik_tolerance;
    tc.min_cameras_for_init = min_cameras_for_init;
    return tc;
}
#endif

}  // namespace posetrak

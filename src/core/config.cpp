#include "posetrak/core/config.hpp"

#include <fmt/core.h>

#include <toml++/toml.h>

#include <fstream>
#include <stdexcept>

namespace posetrak {

TrackerAppConfig TrackerAppConfig::load(std::filesystem::path const& config_path) {
    // Parse TOML file
    toml::table config;
    try {
        config = toml::parse_file(config_path.string());
    } catch (toml::parse_error const& err) {
        throw std::runtime_error(
            fmt::format("Failed to parse config file {}: {}", config_path.string(), err.what()));
    }

    TrackerAppConfig result;

    // === Data paths ===
    auto data = config["data"];
    if (!data) {
        throw std::runtime_error("Missing [data] section in config file");
    }

    result.skeleton_path = data["skeleton"].value_or("");
    if (result.skeleton_path.empty()) {
        throw std::runtime_error("Missing data.skeleton in config file");
    }

    result.cameras_path = data["cameras"].value_or("");
    if (result.cameras_path.empty()) {
        throw std::runtime_error("Missing data.cameras in config file");
    }

    if (auto sync = data["sync"].value<std::string>()) {
        result.sync_path = *sync;
    }

    result.observations_dir = data["observations_dir"].value_or("");
    if (result.observations_dir.empty()) {
        throw std::runtime_error("Missing data.observations_dir in config file");
    }

    result.person_id = data["person_id"].value_or(0);

    // === Tracking parameters ===
    if (auto tracking = config["tracking"]) {
        result.process_noise_std = tracking["process_noise_std"].value_or(0.5);
        result.measurement_noise_std = tracking["measurement_noise_std"].value_or(2.0);
        result.outlier_threshold = tracking["outlier_threshold"].value_or(4.0);

        // Initialization sub-section
        if (auto init = tracking["initialization"]) {
            result.ik_max_iterations = init["ik_max_iterations"].value_or(1000);
            result.ik_tolerance = init["ik_tolerance"].value_or(0.02);
            result.init_position_std = init["init_position_std"].value_or(0.1);
            result.init_orientation_std = init["init_orientation_std"].value_or(0.1);
            result.init_joint_std = init["init_joint_std"].value_or(0.1);
            result.init_velocity_std = init["init_velocity_std"].value_or(0.1);
            result.min_cameras_for_init = init["min_cameras_for_init"].value_or(2);
        }

        // UKF sub-section
        if (auto ukf = tracking["ukf"]) {
            result.ukf_alpha = ukf["alpha"].value_or(0.5);
            result.ukf_beta = ukf["beta"].value_or(2.0);
            result.ukf_kappa = ukf["kappa"].value_or(0.0);
        }
    }

    // === Output ===
    if (auto output = config["output"]) {
        result.output_dir = output["directory"].value_or("tracking_output");
        result.export_tracking_results = output["export_tracking_results"].value_or(true);
        result.export_statistics = output["export_statistics"].value_or(true);
        result.export_debug = output["export_debug"].value_or(false);
    }

    // === Processing ===
    if (auto processing = config["processing"]) {
        result.start_frame = processing["start_frame"].value_or(0);
        result.max_frames = processing["max_frames"].value_or(-1);
    }

    return result;
}

void TrackerAppConfig::validate() const {
    // Check required files exist
    if (!std::filesystem::exists(skeleton_path)) {
        throw std::runtime_error(
            fmt::format("Skeleton file does not exist: {}\n\n"
                        "Please check:\n"
                        "  - Is the path correct in your config file?\n"
                        "  - Are you running from the correct directory?",
                        skeleton_path.string()));
    }

    if (!std::filesystem::exists(cameras_path)) {
        throw std::runtime_error(
            fmt::format("Camera file does not exist: {}\n\n"
                        "Please check:\n"
                        "  - Is the path correct in your config file?\n"
                        "  - Are you running from the correct directory?",
                        cameras_path.string()));
    }

    if (sync_path && !std::filesystem::exists(*sync_path)) {
        throw std::runtime_error(
            fmt::format("Sync file does not exist: {}\n\n"
                        "Please check:\n"
                        "  - Is the path correct in your config file?\n"
                        "  - Are you running from the correct directory?\n"
                        "  - Or remove the sync field if not needed",
                        sync_path->string()));
    }

    if (!std::filesystem::exists(observations_dir)) {
        throw std::runtime_error(
            fmt::format("Observations directory does not exist: {}\n\n"
                        "Please check:\n"
                        "  - Is the path correct in your config file?\n"
                        "  - Did you run OpenPose to generate detections?",
                        observations_dir.string()));
    }

    // Validate parameter ranges
    if (process_noise_std <= 0.0) {
        throw std::runtime_error(
            fmt::format("Invalid process_noise_std: {} (must be > 0)", process_noise_std));
    }

    if (measurement_noise_std <= 0.0) {
        throw std::runtime_error(
            fmt::format("Invalid measurement_noise_std: {} (must be > 0)", measurement_noise_std));
    }

    if (outlier_threshold <= 0.0) {
        throw std::runtime_error(
            fmt::format("Invalid outlier_threshold: {} (must be > 0)", outlier_threshold));
    }

    if (ik_max_iterations <= 0) {
        throw std::runtime_error(
            fmt::format("Invalid ik_max_iterations: {} (must be > 0)", ik_max_iterations));
    }

    if (ik_tolerance <= 0.0) {
        throw std::runtime_error(
            fmt::format("Invalid ik_tolerance: {} (must be > 0)", ik_tolerance));
    }

    if (ukf_alpha <= 0.0 || ukf_alpha > 1.0) {
        throw std::runtime_error(
            fmt::format("Invalid ukf_alpha: {} (must be in (0, 1])", ukf_alpha));
    }

    if (min_cameras_for_init < 2) {
        throw std::runtime_error(
            fmt::format("Invalid min_cameras_for_init: {} (must be >= 2 for triangulation)",
                        min_cameras_for_init));
    }

    if (start_frame < 0) {
        throw std::runtime_error(
            fmt::format("Invalid start_frame: {} (must be >= 0)", start_frame));
    }

    // Check output directory can be created
    try {
        std::filesystem::create_directories(output_dir);
    } catch (std::exception const& e) {
        throw std::runtime_error(
            fmt::format("Cannot create output directory {}: {}", output_dir.string(), e.what()));
    }
}

}  // namespace posetrak

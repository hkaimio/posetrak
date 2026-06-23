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

    // Load active joint groups (optional)
    if (auto groups_array = data["active_joint_groups"].as_array()) {
        for (auto&& elem : *groups_array) {
            if (auto str = elem.value<std::string>()) {
                result.active_joint_groups.push_back(*str);
            }
        }
    }

    // === Tracking parameters ===
    if (auto tracking = config["tracking"]) {
        result.process_noise_std = tracking["process_noise_std"].value_or(0.5);
        if (auto v = tracking["process_noise_vel_std"].value<double>())
            result.process_noise_vel_std = *v;
        if (auto v = tracking["velocity_half_life_s"].value<double>())
            result.velocity_half_life_s = *v;
        // New keys take precedence; fall back to legacy measurement_noise_std → calib_noise_std.
        double legacy_noise = tracking["measurement_noise_std"].value_or(2.0);
        result.calib_noise_std = tracking["calib_noise_std"].value_or(legacy_noise);
        result.pose_noise_std = tracking["pose_noise_std"].value_or(0.0);
        result.outlier_threshold = tracking["outlier_threshold"].value_or(4.0);
        if (auto vel_cams = tracking["velocity_mode_camera_ids"].as_array()) {
            for (auto&& elem : *vel_cams) {
                if (auto v = elem.value<int64_t>())
                    result.velocity_mode_camera_ids.push_back(static_cast<int>(*v));
            }
        }
        if (auto v = tracking["velocity_measurement_noise_std"].value<double>())
            result.velocity_measurement_noise_std = *v;
        result.use_relative_observations = tracking["use_relative_observations"].value_or(false);
        result.relative_min_confidence = tracking["relative_min_confidence"].value_or(0.5);

        // Initialization sub-section
        if (auto init = tracking["initialization"]) {
            if (auto state_path = init["python_state_path"].value<std::string>()) {
                result.python_state_path = *state_path;
            }
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
        result.start_time = processing["start_time"].value_or(0.0);
        result.end_time = processing["end_time"].value_or(-1.0);
        result.tracker_fps = processing["tracker_fps"].value_or(100.0);
    }

    // === Hierarchical tracking ===
    if (auto hier = config["hierarchical"]) {
        result.hierarchical.enabled = hier["enabled"].value_or(false);
        result.hierarchical.enable_sync = hier["enable_sync"].value_or(true);
        result.hierarchical.sync_covariance = hier["sync_covariance"].value_or(false);

        auto load_string_array = [](auto node, std::vector<std::string>& out) {
            if (auto arr = node.as_array()) {
                for (auto const& elem : *arr) {
                    if (auto s = elem.template value<std::string>())
                        out.push_back(*s);
                }
            }
        };

        load_string_array(hier["parent_joint_groups"], result.hierarchical.parent_joint_groups);
        load_string_array(hier["parent_observation_groups"],
                          result.hierarchical.parent_observation_groups);

        result.hierarchical.parent_process_noise_std =
            hier["parent_process_noise_std"].value_or(0.5);
        double parent_legacy = hier["parent_measurement_noise_std"].value_or(2.0);
        result.hierarchical.parent_calib_noise_std =
            hier["parent_calib_noise_std"].value_or(parent_legacy);
        result.hierarchical.parent_pose_noise_std = hier["parent_pose_noise_std"].value_or(0.0);
        result.hierarchical.parent_outlier_threshold =
            hier["parent_outlier_threshold"].value_or(4.0);

        if (auto children_arr = hier["children"].as_array()) {
            for (auto&& elem : *children_arr) {
                auto* tbl = elem.as_table();
                if (!tbl)
                    continue;
                toml::node_view<const toml::node> child_view(*tbl);

                ChildFilterConfig child;
                child.name = child_view["name"].value_or(std::string{});
                load_string_array(child_view["joint_groups"], child.joint_groups);
                load_string_array(child_view["observation_groups"], child.observation_groups);
                child.process_noise_std = child_view["process_noise_std"].value_or(0.3);
                double child_legacy = child_view["measurement_noise_std"].value_or(2.0);
                child.calib_noise_std = child_view["calib_noise_std"].value_or(child_legacy);
                child.pose_noise_std = child_view["pose_noise_std"].value_or(0.0);
                child.outlier_threshold = child_view["outlier_threshold"].value_or(4.0);
                child.min_inliers_ratio = child_view["min_inliers_ratio"].value_or(0.3);
                child.max_innovation_norm = child_view["max_innovation_norm"].value_or(200.0);

                result.hierarchical.children.push_back(std::move(child));
            }
        }
    }

    // === Calibration ===
    if (auto calib = config["calibration"]) {
        result.calibration_mode = calib["enabled"].value_or(false);
        result.prismatic_process_noise_std = calib["prismatic_process_noise_std"].value_or(0.0001);
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

    if (pose_noise_std < 0.0 || calib_noise_std < 0.0 ||
        (pose_noise_std == 0.0 && calib_noise_std <= 0.0)) {
        throw std::runtime_error(
            fmt::format("Invalid noise parameters: pose_noise_std={}, calib_noise_std={} "
                        "(both must be >= 0, at least one > 0)",
                        pose_noise_std, calib_noise_std));
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

    if (start_time < 0.0) {
        throw std::runtime_error(fmt::format("Invalid start_time: {} (must be >= 0)", start_time));
    }

    if (tracker_fps <= 0.0) {
        throw std::runtime_error(fmt::format("Invalid tracker_fps: {} (must be > 0)", tracker_fps));
    }

    if (end_time >= 0.0 && end_time <= start_time) {
        throw std::runtime_error(
            fmt::format("Invalid end_time: {} (must be > start_time or -1)", end_time));
    }

    // Check output directory can be created
    try {
        std::filesystem::create_directories(output_dir);
    } catch (std::exception const& e) {
        throw std::runtime_error(
            fmt::format("Cannot create output directory {}: {}", output_dir.string(), e.what()));
    }

    // === Hierarchical config ===
    if (hierarchical.enabled) {
        if (hierarchical.parent_joint_groups.empty()) {
            throw std::runtime_error(
                "hierarchical.parent_joint_groups must not be empty when hierarchical "
                "tracking is enabled");
        }
        if (hierarchical.parent_observation_groups.empty()) {
            throw std::runtime_error(
                "hierarchical.parent_observation_groups must not be empty when hierarchical "
                "tracking is enabled");
        }
        if (hierarchical.parent_process_noise_std <= 0.0) {
            throw std::runtime_error(
                fmt::format("Invalid hierarchical.parent_process_noise_std: {} (must be > 0)",
                            hierarchical.parent_process_noise_std));
        }
        for (auto const& child : hierarchical.children) {
            if (child.name.empty()) {
                throw std::runtime_error(
                    "Each hierarchical child filter must have a non-empty name");
            }
            if (child.joint_groups.empty()) {
                throw std::runtime_error(
                    fmt::format("Child filter '{}' has empty joint_groups", child.name));
            }
            if (child.observation_groups.empty()) {
                throw std::runtime_error(
                    fmt::format("Child filter '{}' has empty observation_groups", child.name));
            }
            if (child.process_noise_std <= 0.0) {
                throw std::runtime_error(
                    fmt::format("Child filter '{}': process_noise_std must be > 0", child.name));
            }
            if (child.min_inliers_ratio < 0.0 || child.min_inliers_ratio > 1.0) {
                throw std::runtime_error(fmt::format(
                    "Child filter '{}': min_inliers_ratio must be in [0, 1]", child.name));
            }
        }
    }
}

}  // namespace posetrak

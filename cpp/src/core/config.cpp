// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

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
        result.cross_pair_max_px = tracking["cross_pair_max_px"].value_or(0.0);
        result.cross_pair_max_n = tracking["cross_pair_max_n"].value_or(10);
        result.process_noise_vel_gain_joint =
            tracking["process_noise_vel_gain_joint"].value_or(0.0);
        result.process_noise_vel_ref_joint = tracking["process_noise_vel_ref_joint"].value_or(1.0);
        result.process_noise_vel_gain_root = tracking["process_noise_vel_gain_root"].value_or(0.0);
        result.process_noise_vel_ref_root = tracking["process_noise_vel_ref_root"].value_or(1.0);
        if (auto names = tracking["process_noise_vel_joint_names"].as_array()) {
            for (auto&& elem : *names) {
                if (auto str = elem.value<std::string>())
                    result.process_noise_vel_joint_names.push_back(*str);
            }
        }
        if (auto scopes = tracking["process_noise_vel_scopes"].as_array()) {
            for (auto&& elem : *scopes) {
                if (auto scope_table = elem.as_table()) {
                    VelocityNoiseScope scope;
                    scope.name = (*scope_table)["name"].value_or(std::string{});
                    scope.gain = (*scope_table)["gain"].value_or(0.0);
                    scope.vel_ref = (*scope_table)["vel_ref"].value_or(1.0);
                    if (auto names = (*scope_table)["joint_names"].as_array()) {
                        for (auto&& name_elem : *names) {
                            if (auto str = name_elem.value<std::string>())
                                scope.joint_names.push_back(*str);
                        }
                    }
                    result.process_noise_vel_scopes.push_back(std::move(scope));
                }
            }
        }
        if (auto names = tracking["pose_reg_joint_names"].as_array()) {
            for (auto&& elem : *names) {
                if (auto str = elem.value<std::string>())
                    result.pose_reg_joint_names.push_back(*str);
            }
        }
        result.pose_reg_equal_split_noise_std =
            tracking["pose_reg_equal_split_noise_std"].value_or(0.0);
        result.pose_reg_rest_pose_noise_std =
            tracking["pose_reg_rest_pose_noise_std"].value_or(0.0);

        if (auto names = tracking["soft_limit_joint_names"].as_array()) {
            for (auto&& elem : *names) {
                if (auto str = elem.value<std::string>())
                    result.soft_limit_joint_names.push_back(*str);
            }
        }
        result.soft_limit_margin_rad = tracking["soft_limit_margin_rad"].value_or(0.0);
        result.soft_limit_noise_std = tracking["soft_limit_noise_std"].value_or(0.0);

        if (auto names = tracking["near_limit_damping_joint_names"].as_array()) {
            for (auto&& elem : *names) {
                if (auto str = elem.value<std::string>())
                    result.near_limit_damping_joint_names.push_back(*str);
            }
        }
        result.near_limit_margin_rad = tracking["near_limit_margin_rad"].value_or(0.0);
        result.near_limit_spread_sigma = tracking["near_limit_spread_sigma"].value_or(3.0);
        result.near_limit_damping_factor = tracking["near_limit_damping_factor"].value_or(1.0);

        if (auto scopes = tracking["nis_feedback_scopes"].as_array()) {
            for (auto&& elem : *scopes) {
                if (auto scope_table = elem.as_table()) {
                    NisFeedbackScope scope;
                    scope.name = (*scope_table)["name"].value_or(std::string{});
                    if (auto names = (*scope_table)["joint_names"].as_array()) {
                        for (auto&& name_elem : *names) {
                            if (auto str = name_elem.value<std::string>())
                                scope.joint_names.push_back(*str);
                        }
                    }
                    result.nis_feedback_scopes.push_back(std::move(scope));
                }
            }
        }
        result.nis_feedback_window = tracking["nis_feedback_window"].value_or(8);
        result.nis_feedback_threshold = tracking["nis_feedback_threshold"].value_or(1.5);
        result.nis_feedback_max_multiplier = tracking["nis_feedback_max_multiplier"].value_or(10.0);

        result.edited_kp_noise_std = tracking["edited_kp_noise_std"].value_or(0.0);

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
            result.rigid_init_max_residual_m = init["rigid_init_max_residual_m"].value_or(0.02);
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

    if ((process_noise_vel_gain_joint > 0.0 && process_noise_vel_ref_joint <= 0.0) ||
        (process_noise_vel_gain_root > 0.0 && process_noise_vel_ref_root <= 0.0)) {
        throw std::runtime_error(
            "process_noise_vel_ref_joint/root must be > 0 when the corresponding "
            "process_noise_vel_gain_joint/root is > 0 (used as a divisor)");
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

    if (rigid_init_max_residual_m <= 0.0) {
        throw std::runtime_error(fmt::format("Invalid rigid_init_max_residual_m: {} (must be > 0)",
                                             rigid_init_max_residual_m));
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
}

}  // namespace posetrak

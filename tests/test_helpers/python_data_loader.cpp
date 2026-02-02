#include "python_data_loader.hpp"

#include <fmt/core.h>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace posetrak::test_helpers {

State load_state_from_json(std::string const& json_path, Skeleton const& skeleton) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open: " + json_path);
    }

    nlohmann::json j;
    file >> j;

    // Parse root position
    Eigen::Vector3d root_position(j["root_position"][0].get<double>(),
                                  j["root_position"][1].get<double>(),
                                  j["root_position"][2].get<double>());

    // Parse root quaternion (w, x, y, z format)
    Eigen::Quaterniond root_orientation(
        j["root_quaternion"][0].get<double>(), j["root_quaternion"][1].get<double>(),
        j["root_quaternion"][2].get<double>(), j["root_quaternion"][3].get<double>());

    // Parse root velocity
    Eigen::Vector3d root_velocity(j["root_velocity"][0].get<double>(),
                                  j["root_velocity"][1].get<double>(),
                                  j["root_velocity"][2].get<double>());

    // Parse root angular velocity
    Eigen::Vector3d root_angular_velocity(j["root_angular_velocity"][0].get<double>(),
                                          j["root_angular_velocity"][1].get<double>(),
                                          j["root_angular_velocity"][2].get<double>());

    // Parse joint angles (JSON has joint names as keys with arrays of DOF values)
    int num_dof = skeleton.total_dof_count();
    Eigen::VectorXd joint_angles = Eigen::VectorXd::Zero(num_dof);
    auto const& joint_angles_json = j["joint_angles"];

    int idx = 0;
    for (auto const& joint : skeleton.joints()) {
        if (joint_angles_json.contains(joint.name)) {
            auto const& joint_vals = joint_angles_json[joint.name];
            // Handle both array (multi-DOF) and scalar (single-DOF) values
            if (joint_vals.is_array()) {
                for (int i = 0; i < joint.dof && i < static_cast<int>(joint_vals.size()); ++i) {
                    joint_angles(idx + i) = joint_vals[i];
                }
            } else {
                joint_angles(idx) = joint_vals;
            }
        }
        idx += joint.dof;
    }

    // Parse joint velocities (same mapping)
    Eigen::VectorXd joint_velocities = Eigen::VectorXd::Zero(num_dof);
    auto const& joint_velocities_json = j["joint_velocities"];

    idx = 0;
    for (auto const& joint : skeleton.joints()) {
        if (joint_velocities_json.contains(joint.name)) {
            auto const& joint_vals = joint_velocities_json[joint.name];
            if (joint_vals.is_array()) {
                for (int i = 0; i < joint.dof && i < static_cast<int>(joint_vals.size()); ++i) {
                    joint_velocities(idx + i) = joint_vals[i];
                }
            } else {
                joint_velocities(idx) = joint_vals;
            }
        }
        idx += joint.dof;
    }

    return State(root_position, root_orientation, joint_angles, root_velocity,
                 root_angular_velocity, joint_velocities);
}

Eigen::MatrixXd load_matrix_from_csv(std::string const& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open: " + csv_path);
    }

    std::vector<std::vector<double>> data;
    std::string line;

    // Skip header line (if it starts with non-numeric character)
    std::getline(file, line);
    if (line.empty() || (!std::isdigit(line[0]) && line[0] != '-' && line[0] != '+')) {
        // Header line, skip it
    } else {
        // No header, process this line
        file.clear();
        file.seekg(0);
    }

    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        // Skip first column if it's an index
        bool first = true;
        while (std::getline(ss, cell, ',')) {
            if (first && cell.empty()) {
                first = false;
                continue;
            }
            first = false;

            try {
                row.push_back(std::stod(cell));
            } catch (...) {
                // Skip non-numeric cells (like empty strings or headers)
                continue;
            }
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    if (data.empty()) {
        throw std::runtime_error("No data found in: " + csv_path);
    }

    // Convert to Eigen matrix
    size_t rows = data.size();
    size_t cols = data[0].size();
    Eigen::MatrixXd matrix(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix(i, j) = data[i][j];
        }
    }

    return matrix;
}

std::pair<std::vector<Observation>, std::vector<bool>>
load_observations_from_csv(std::string const& csv_path, Skeleton const& skeleton,
                           std::unordered_map<std::string, int> const& camera_name_to_id) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open: " + csv_path);
    }

    // Build marker name to ID map
    std::unordered_map<std::string, int> marker_name_to_id;
    for (size_t i = 0; i < skeleton.markers().size(); ++i) {
        marker_name_to_id[skeleton.markers()[i].name] = static_cast<int>(i);
    }

    std::vector<Observation> observations;
    std::vector<bool> outlier_flags;

    std::string line;
    std::getline(file, line);  // Skip header

    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> fields;

        while (std::getline(ss, cell, ',')) {
            fields.push_back(cell);
        }

        // Expected columns (from Python debug CSV):
        // 0: index, 1: tracker_frame_idx, 2: camera_source_frame_idx, 3: timestamp,
        // 4: person_id, 5: camera_name, 6: marker_name, 7: observed_u, 8: observed_v,
        // 9: predicted_u, 10: predicted_v, 11: residual_u, 12: residual_v,
        // 13: residual_norm, 14: mahalanobis_distance, 15: confidence,
        // 16: is_outlier, 17: outlier_reason, 18: was_used_in_update

        if (fields.size() < 19) {
            continue;  // Skip incomplete rows
        }

        Observation obs;

        // Parse camera name to ID
        std::string camera_name = fields[5];
        auto cam_it = camera_name_to_id.find(camera_name);
        if (cam_it == camera_name_to_id.end()) {
            throw std::runtime_error("Unknown camera: " + camera_name);
        }
        obs.camera_id = cam_it->second;

        // Parse marker name to ID
        std::string marker_name = fields[6];
        auto marker_it = marker_name_to_id.find(marker_name);
        if (marker_it == marker_name_to_id.end()) {
            throw std::runtime_error("Unknown marker: " + marker_name);
        }
        obs.marker_id = marker_it->second;

        // Parse observed position
        obs.position.x() = std::stod(fields[7]);
        obs.position.y() = std::stod(fields[8]);

        // Parse confidence
        obs.confidence = std::stod(fields[15]);

        // Parse timestamp and frame
        obs.timestamp = std::stod(fields[3]);
        obs.frame_idx = std::stoi(fields[1]);

        // Position distorted = position for undistorted case
        obs.position_distorted = obs.position;

        observations.push_back(obs);

        // Parse outlier flag
        std::string is_outlier_str = fields[16];
        bool is_outlier =
            (is_outlier_str == "True" || is_outlier_str == "true" || is_outlier_str == "1");
        outlier_flags.push_back(is_outlier);
    }

    return {observations, outlier_flags};
}

PythonFrame0Data load_python_frame0_data(std::string const& debug_dir, Skeleton const& skeleton) {
    PythonFrame0Data data{};

    // Load states
    data.prior_state =
        std::make_optional(load_state_from_json(debug_dir + "/prior_state.json", skeleton));
    data.posterior_state =
        std::make_optional(load_state_from_json(debug_dir + "/posterior_state.json", skeleton));

    // Load covariances
    data.prior_covariance = load_matrix_from_csv(debug_dir + "/prior_covariance.csv");
    data.posterior_covariance = load_matrix_from_csv(debug_dir + "/posterior_covariance.csv");

    // Load intermediate computation results
    data.sigma_points = load_matrix_from_csv(debug_dir + "/sigma_points.csv");
    data.predicted_obs = load_matrix_from_csv(debug_dir + "/predicted_observations.csv");
    data.innovation_cov = load_matrix_from_csv(debug_dir + "/innovation_covariance.csv");
    data.kalman_gain = load_matrix_from_csv(debug_dir + "/kalman_gain.csv");

    // Load observations - need camera name to ID map
    // For now, use hardcoded mapping (can be made configurable)
    std::unordered_map<std::string, int> camera_name_to_id = {
        {"cam0", 0}, {"cam1", 1}, {"cam2", 2}, {"cam3", 3}, {"cam4", 4}, {"cam5", 5}};

    auto [obs, outliers] = load_observations_from_csv(debug_dir + "/all_observations.csv", skeleton,
                                                      camera_name_to_id);
    data.observations = obs;
    data.outlier_flags = outliers;

    return data;
}

}  // namespace posetrak::test_helpers

#pragma once

#include <posetrak/core/observation.hpp>
#include <posetrak/core/skeleton.hpp>
#include <posetrak/core/state.hpp>

#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace posetrak::test_helpers {

/// @brief All Python debug data for frame 0 comparison
struct PythonFrame0Data {
    std::optional<State> prior_state;
    Eigen::MatrixXd prior_covariance;
    std::vector<Observation> observations;
    Eigen::MatrixXd sigma_points;    // For verification (n_dims × n_sigma)
    Eigen::MatrixXd predicted_obs;   // For verification (2*n_obs × n_sigma)
    Eigen::MatrixXd innovation_cov;  // For verification
    Eigen::MatrixXd kalman_gain;     // For verification
    std::optional<State> posterior_state;
    Eigen::MatrixXd posterior_covariance;
    std::vector<bool> outlier_flags;  // True if observation was outlier
};

/// @brief Load Python frame 0 debug data from directory
/// @param debug_dir Path to frame_0000 debug directory
/// @param skeleton Skeleton (needed to construct State objects)
/// @return All loaded data
PythonFrame0Data load_python_frame0_data(std::string const& debug_dir, Skeleton const& skeleton);

/// @brief Load state from Python JSON format
/// @param json_path Path to prior_state.json or posterior_state.json
/// @param skeleton Skeleton (needed to construct State)
/// @return Loaded state
State load_state_from_json(std::string const& json_path, Skeleton const& skeleton);

/// @brief Load matrix from CSV file
/// @param csv_path Path to CSV file
/// @return Loaded matrix
Eigen::MatrixXd load_matrix_from_csv(std::string const& csv_path);

/// @brief Load observations from Python CSV format
/// @param csv_path Path to all_observations.csv
/// @param skeleton Skeleton (to map marker names to IDs)
/// @param camera_name_to_id Map from camera name to camera ID
/// @return Pair of (observations, outlier_flags)
std::pair<std::vector<Observation>, std::vector<bool>>
load_observations_from_csv(std::string const& csv_path, Skeleton const& skeleton,
                           std::unordered_map<std::string, int> const& camera_name_to_id);

}  // namespace posetrak::test_helpers

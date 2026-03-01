#pragma once

#include <string>
#include <vector>

namespace posetrak {

/// Per-scale-group result from a calibration post-processing run.
struct ScaleGroupResult {
    std::string name;    ///< Scale group name (e.g. "femur")
    double final_scale;  ///< Mean of s over the convergence window
    double scale_std;    ///< Std dev of s over the window
    bool converged;      ///< True when scale_std < converge_std threshold
};

/// Tuning parameters for the convergence check.
struct ScaleCalibrationOptions {
    /// Number of frames at the end of the run used to judge convergence.
    int window_frames = 100;
    /// Std dev threshold below which a group is declared converged.
    double converge_std = 0.005;
};

/// Parse a state_vectors.csv produced by calibration-mode tracking and compute
/// per-scale-group convergence statistics over the final `opts.window_frames`
/// frames.
///
/// Columns matching the pattern `scale_group_<name>` (non-velocity) are
/// detected automatically from the CSV header.
///
/// @param state_vectors_csv_path  Path to the state_vectors.csv output file.
/// @param opts                    Convergence tuning.
/// @return One ScaleGroupResult per detected scale group, sorted by name.
/// @throws std::runtime_error if the file cannot be opened or has no scale columns.
std::vector<ScaleGroupResult> check_scale_convergence(std::string const& state_vectors_csv_path,
                                                      ScaleCalibrationOptions const& opts = {});

/// Produce a calibrated skeleton YAML by absorbing the final scale factors into
/// each affected joint's `offset:` field and removing the `scale_groups:` section.
///
/// @param input_yaml_path   Path to the reference skeleton YAML (read-only).
/// @param output_yaml_path  Where to write the calibrated skeleton YAML.
/// @param results           Output of check_scale_convergence().
/// @throws std::runtime_error on file I/O or YAML parse errors.
void write_calibrated_yaml(std::string const& input_yaml_path, std::string const& output_yaml_path,
                           std::vector<ScaleGroupResult> const& results);

}  // namespace posetrak

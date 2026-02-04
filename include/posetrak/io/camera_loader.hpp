#pragma once

#include <posetrak/core/camera.hpp>

#include <map>
#include <string>

namespace posetrak {

/// @brief Load cameras from Pose2Sim TOML calibration file
/// @param filepath Path to TOML calibration file
/// @return Ordered map of camera section name to Camera object
/// @note Camera IDs are assigned based on TOML file order (first camera = ID 0, etc.)
///       The returned map preserves this order during iteration.
/// @throws std::runtime_error if file cannot be read or parsed
std::map<std::string, Camera> load_cameras_from_toml(std::string const& filepath);

}  // namespace posetrak

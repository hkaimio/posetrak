#pragma once

#include <posetrak/core/camera.hpp>

#include <string>
#include <unordered_map>

namespace posetrak {

/// @brief Load cameras from Pose2Sim TOML calibration file
/// @param filepath Path to TOML calibration file
/// @return Map of camera name to Camera object
/// @throws std::runtime_error if file cannot be read or parsed
std::unordered_map<std::string, Camera> load_cameras_from_toml(std::string const& filepath);

}  // namespace posetrak

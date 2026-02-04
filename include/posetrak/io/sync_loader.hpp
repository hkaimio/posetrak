#pragma once

#include <posetrak/core/camera.hpp>

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace posetrak {

// SyncPoint is defined in camera.hpp

/// @brief Load synchronization metadata from JSON file
/// @param filepath Path to JSON sync metadata file
/// @return Map of camera name to sync points array
/// @throws std::runtime_error if file cannot be read or parsed
std::unordered_map<std::string, std::vector<SyncPoint>>
load_sync_metadata(std::string const& filepath);

/// @brief Apply synchronization metadata to cameras
/// @param cameras Ordered map of camera name to Camera object (modified in-place)
/// @param sync_data Synchronization metadata (camera name -> sync points)
/// @param strict If true, throw on camera name mismatch. If false, skip missing cameras.
/// @throws std::runtime_error if strict=true and camera name not found
void apply_sync_metadata(std::map<std::string, Camera>& cameras,
                         std::unordered_map<std::string, std::vector<SyncPoint>> const& sync_data,
                         bool strict = false);

}  // namespace posetrak

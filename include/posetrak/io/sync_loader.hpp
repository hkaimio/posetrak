#pragma once

#include <posetrak/core/camera.hpp>

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace posetrak {

// SyncPoint is defined in camera.hpp

/// @brief Camera synchronization data including FPS and sync points
struct CameraSyncData {
    double fps = 30.0;                   ///< Frame rate (default 30 fps)
    std::vector<SyncPoint> sync_points;  ///< Synchronization points
};

/// @brief Load synchronization metadata from JSON file
/// @param filepath Path to JSON sync metadata file
/// @return Map of camera name to sync data (fps + sync points)
/// @throws std::runtime_error if file cannot be read or parsed
///
/// Supports two formats:
/// 1. Legacy array format: {"cam1": [{"frame": 100, "timestamp": 0.0}, ...]}
/// 2. Object format: {"cam1": {"fps": 120, "sync_points": [{"frame": 100, "timestamp": 0.0}, ...]}}
std::unordered_map<std::string, CameraSyncData> load_sync_metadata(std::string const& filepath);

/// @brief Apply synchronization metadata to cameras
/// @param cameras Ordered map of camera name to Camera object (modified in-place)
/// @param sync_data Synchronization metadata (camera name -> sync data with fps and sync points)
/// @param strict If true, throw on camera name mismatch. If false, skip missing cameras.
/// @throws std::runtime_error if strict=true and camera name not found
void apply_sync_metadata(std::map<std::string, Camera>& cameras,
                         std::unordered_map<std::string, CameraSyncData> const& sync_data,
                         bool strict = false);

}  // namespace posetrak

// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include <posetrak/io/sync_loader.hpp>

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace posetrak {

namespace {

// Validate sync point ordering and values
void validate_sync_points(std::vector<SyncPoint> const& points, std::string const& cam_name) {
    if (points.empty()) {
        return;  // Empty is valid (means no sync points)
    }

    // Check monotonic ordering
    for (size_t i = 1; i < points.size(); ++i) {
        if (points[i].frame_idx <= points[i - 1].frame_idx) {
            throw std::runtime_error("Camera '" + cam_name +
                                     "': sync point frame indices must be strictly increasing");
        }
        if (points[i].timestamp_sec <= points[i - 1].timestamp_sec) {
            throw std::runtime_error("Camera '" + cam_name +
                                     "': sync point timestamps must be strictly increasing");
        }
    }
}

}  // namespace

std::unordered_map<std::string, CameraSyncData> load_sync_metadata(std::string const& filepath) {
    // Read JSON file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open sync metadata file: " + filepath);
    }

    nlohmann::json root;
    try {
        file >> root;
    } catch (nlohmann::json::parse_error const& e) {
        throw std::runtime_error("Failed to parse sync metadata JSON: " + std::string(e.what()));
    }

    if (!root.is_object()) {
        throw std::runtime_error("Sync metadata root must be a JSON object");
    }

    std::unordered_map<std::string, CameraSyncData> sync_data;

    // Parse each camera's sync data
    for (auto const& [cam_name, cam_data] : root.items()) {
        CameraSyncData data;

        // Handle null (equivalent to empty)
        if (cam_data.is_null()) {
            sync_data[cam_name] = data;
            continue;
        }

        // Support two formats:
        // 1. Legacy array format: [{"frame": 100, "timestamp": 0.0}, ...]
        // 2. Object format: {"fps": 120, "sync_points": [{"frame": 100, "timestamp": 0.0}, ...]}
        if (cam_data.is_array()) {
            // Legacy format - array of sync points (default fps = 30.0)
            data.fps = 30.0;
            for (auto const& pt : cam_data) {
                if (!pt.is_object()) {
                    throw std::runtime_error("Camera '" + cam_name +
                                             "': sync point must be an object");
                }
                if (!pt.contains("frame") || !pt.contains("timestamp")) {
                    throw std::runtime_error("Camera '" + cam_name +
                                             "': sync point missing 'frame' or 'timestamp'");
                }

                SyncPoint sync_pt;
                sync_pt.frame_idx = pt["frame"].get<uint32_t>();
                sync_pt.timestamp_sec = pt["timestamp"].get<double>();
                data.sync_points.push_back(sync_pt);
            }
        } else if (cam_data.is_object()) {
            // New format - object with optional fps and sync_points array
            // Validate that if it's an object, it's not a misplaced sync point
            if (cam_data.contains("frame") || cam_data.contains("timestamp")) {
                throw std::runtime_error("Camera '" + cam_name +
                                         "': sync point objects must be inside 'sync_points' "
                                         "array, not at camera level");
            }

            data.fps = cam_data.value("fps", 30.0);  // Default to 30 fps if not specified

            // sync_points can be missing (empty), or an array.
            // Accept both "sync_points" and "syncpoints" key names.
            std::string const sync_key = cam_data.contains("sync_points")  ? "sync_points"
                                         : cam_data.contains("syncpoints") ? "syncpoints"
                                                                           : "";
            if (!sync_key.empty()) {
                auto const& points_arr = cam_data[sync_key];
                if (!points_arr.is_array()) {
                    throw std::runtime_error("Camera '" + cam_name +
                                             "': 'sync_points' must be an array");
                }

                for (auto const& pt : points_arr) {
                    if (!pt.is_object()) {
                        throw std::runtime_error("Camera '" + cam_name +
                                                 "': sync point must be an object");
                    }
                    if (!pt.contains("frame") || !pt.contains("timestamp")) {
                        throw std::runtime_error("Camera '" + cam_name +
                                                 "': sync point missing 'frame' or 'timestamp'");
                    }

                    SyncPoint sync_pt;
                    sync_pt.frame_idx = pt["frame"].get<uint32_t>();
                    sync_pt.timestamp_sec = pt["timestamp"].get<double>();
                    data.sync_points.push_back(sync_pt);
                }
            }
        } else {
            throw std::runtime_error("Camera '" + cam_name +
                                     "' sync data must be an array or object");
        }

        // Validate FPS
        if (data.fps <= 0.0) {
            throw std::runtime_error("Camera '" + cam_name + "': fps must be positive");
        }

        // Validate sync points
        validate_sync_points(data.sync_points, cam_name);
        sync_data[cam_name] = std::move(data);
    }

    if (sync_data.empty()) {
        throw std::runtime_error("Sync metadata contains no cameras");
    }

    return sync_data;
}

void apply_sync_metadata(std::map<std::string, Camera>& cameras,
                         std::unordered_map<std::string, CameraSyncData> const& sync_data,
                         bool strict) {
    for (auto const& [cam_name, data] : sync_data) {
        auto it = cameras.find(cam_name);
        if (it == cameras.end()) {
            if (strict) {
                throw std::runtime_error("Sync metadata references unknown camera: " + cam_name);
            }
            continue;  // Skip if not strict
        }

        Camera& camera = it->second;

        // Apply FPS from sync metadata
        camera.set_fps(data.fps);

        // Apply sync points if non-empty
        if (!data.sync_points.empty()) {
            camera.set_sync_points(data.sync_points);
        }
    }
}

}  // namespace posetrak

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

    // Check for negative values
    for (auto const& pt : points) {
        if (pt.timestamp_sec < 0.0) {
            throw std::runtime_error("Camera '" + cam_name +
                                     "': sync point has negative timestamp");
        }
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

std::unordered_map<std::string, std::vector<SyncPoint>>
load_sync_metadata(std::string const& filepath) {
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

    std::unordered_map<std::string, std::vector<SyncPoint>> sync_data;

    // Parse each camera's sync points array
    for (auto const& [cam_name, cam_arr] : root.items()) {
        std::vector<SyncPoint> points;

        // Handle null (equivalent to empty array)
        if (cam_arr.is_null()) {
            sync_data[cam_name] = points;
            continue;
        }

        if (!cam_arr.is_array()) {
            throw std::runtime_error("Camera '" + cam_name + "' sync data must be an array");
        }

        for (auto const& pt : cam_arr) {
            if (!pt.is_object()) {
                throw std::runtime_error("Camera '" + cam_name + "': sync point must be an object");
            }
            if (!pt.contains("frame") || !pt.contains("timestamp")) {
                throw std::runtime_error("Camera '" + cam_name +
                                         "': sync point missing 'frame' or 'timestamp'");
            }

            SyncPoint sync_pt;
            sync_pt.frame_idx = pt["frame"].get<uint32_t>();
            sync_pt.timestamp_sec = pt["timestamp"].get<double>();
            points.push_back(sync_pt);
        }

        // Validate sync points
        validate_sync_points(points, cam_name);
        sync_data[cam_name] = std::move(points);
    }

    if (sync_data.empty()) {
        throw std::runtime_error("Sync metadata contains no cameras");
    }

    return sync_data;
}

void apply_sync_metadata(std::map<std::string, Camera>& cameras,
                         std::unordered_map<std::string, std::vector<SyncPoint>> const& sync_data,
                         bool strict) {
    for (auto const& [cam_name, points] : sync_data) {
        auto it = cameras.find(cam_name);
        if (it == cameras.end()) {
            if (strict) {
                throw std::runtime_error("Sync metadata references unknown camera: " + cam_name);
            }
            continue;  // Skip if not strict
        }

        Camera& camera = it->second;

        // Apply sync points if non-empty
        if (!points.empty()) {
            camera.set_sync_points(points);
        }
    }
}

}  // namespace posetrak

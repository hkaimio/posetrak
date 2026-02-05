#include <posetrak/io/observation_loader.hpp>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace posetrak {

namespace {

// Parse OpenPose "pose_keypoints_2d" array for one person
// Returns vector of Observation structs (one per keypoint)
// coco_to_marker_idx maps OpenPose/COCO keypoint IDs to skeleton marker indices
std::vector<Observation>
parse_openpose_person(nlohmann::json const& person, Camera const& camera, uint32_t frame_idx,
                      double min_confidence,
                      std::unordered_map<int, int> const& coco_to_marker_idx) {
    if (!person.contains("pose_keypoints_2d")) {
        return {};
    }

    auto const& keypoints_flat = person["pose_keypoints_2d"];
    if (!keypoints_flat.is_array() || keypoints_flat.size() % 3 != 0) {
        throw std::runtime_error("Invalid pose_keypoints_2d format");
    }

    size_t num_keypoints = keypoints_flat.size() / 3;
    std::vector<Observation> observations;
    observations.reserve(num_keypoints);

    double timestamp = camera.get_timestamp(frame_idx);

    for (size_t i = 0; i < num_keypoints; ++i) {
        double x = keypoints_flat[i * 3 + 0].get<double>();
        double y = keypoints_flat[i * 3 + 1].get<double>();
        double conf = keypoints_flat[i * 3 + 2].get<double>();

        if (conf >= min_confidence) {
            // Map COCO ID to marker index
            int coco_id = static_cast<int>(i);
            auto it = coco_to_marker_idx.find(coco_id);
            if (it == coco_to_marker_idx.end()) {
                // This COCO keypoint doesn't have a corresponding marker in skeleton - skip
                continue;
            }

            Observation obs;
            obs.camera_id = camera.id();  // Use camera's assigned ID
            obs.marker_id = it->second;   // Use skeleton marker index, not COCO ID
            obs.frame_idx = static_cast<int>(frame_idx);
            obs.timestamp = timestamp;
            obs.position_distorted = Eigen::Vector2d(x, y);
            obs.position = camera.undistort(obs.position_distorted);
            obs.confidence = conf;

            observations.push_back(obs);
        }
    }

    return observations;
}

}  // namespace

ObservationSequence load_openpose_frame(std::string const& filepath, Camera const& camera,
                                        std::string const& camera_name, Skeleton const& skeleton,
                                        uint32_t frame_idx, double min_confidence, int person_id) {
    // Build COCO ID -> marker index map
    std::unordered_map<int, int> coco_to_marker_idx;
    auto const& markers = skeleton.markers();
    for (size_t i = 0; i < markers.size(); ++i) {
        if (markers[i].coco_id.has_value()) {
            coco_to_marker_idx[markers[i].coco_id.value()] = static_cast<int>(i);
        }
    }

    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open OpenPose file: " + filepath);
    }

    nlohmann::json root;
    try {
        file >> root;
    } catch (nlohmann::json::parse_error const& e) {
        throw std::runtime_error("Failed to parse OpenPose JSON: " + std::string(e.what()));
    }

    if (!root.contains("people") || !root["people"].is_array()) {
        throw std::runtime_error("OpenPose JSON missing 'people' array");
    }

    auto const& people = root["people"];
    if (people.empty()) {
        // No people detected - return empty sequence
        ObservationSequence seq;
        seq.camera_id = camera.id();
        seq.camera_name = camera_name;
        return seq;
    }

    // Check person_id is in range
    if (person_id < 0 || person_id >= static_cast<int>(people.size())) {
        // Person ID out of range - return empty
        ObservationSequence seq;
        seq.camera_id = camera.id();
        seq.camera_name = camera_name;
        return seq;
    }

    auto observations = parse_openpose_person(people[person_id], camera, frame_idx, min_confidence,
                                              coco_to_marker_idx);

    ObservationSequence seq;
    seq.camera_id = camera.id();
    seq.camera_name = camera_name;
    seq.observations = std::move(observations);

    return seq;
}

ObservationSet load_openpose_sequence(std::string const& base_dir,
                                      std::map<std::string, Camera> const& cameras,
                                      Skeleton const& skeleton, double start_time, double end_time,
                                      double min_confidence, int person_id) {
    namespace fs = std::filesystem;

    if (!fs::exists(base_dir)) {
        throw std::runtime_error("Base directory does not exist: " + base_dir);
    }

    ObservationSet obs_set(person_id);

    // Iterate cameras in deterministic order (std::map is ordered)
    for (auto const& [cam_name, camera] : cameras) {
        fs::path cam_dir = fs::path(base_dir) / cam_name;
        if (!fs::exists(cam_dir) || !fs::is_directory(cam_dir)) {
            throw std::runtime_error("Camera directory does not exist: " + cam_dir.string());
        }

        // Find all JSON files matching pattern: camera_name_NNNNNN.json
        std::regex frame_pattern(cam_name + R"(_(\d{6})\.json)");
        std::vector<std::pair<uint32_t, fs::path>> frame_files;

        for (auto const& entry : fs::directory_iterator(cam_dir)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".json") {
                continue;
            }

            std::smatch match;
            std::string filename = entry.path().filename().string();
            if (std::regex_match(filename, match, frame_pattern)) {
                uint32_t frame_num = std::stoul(match[1].str());
                frame_files.emplace_back(frame_num, entry.path());
            }
        }

        // Sort by frame number
        std::sort(frame_files.begin(), frame_files.end(),
                  [](auto const& a, auto const& b) { return a.first < b.first; });

        // Load observations for each frame and build a sequence
        ObservationSequence full_seq;
        full_seq.camera_id = camera.id();
        full_seq.camera_name = cam_name;

        for (auto const& [frame_num, filepath] : frame_files) {
            auto frame_seq = load_openpose_frame(filepath.string(), camera, cam_name, skeleton,
                                                 frame_num, min_confidence, person_id);

            // Filter observations by timestamp
            for (auto const& obs : frame_seq.observations) {
                // Skip observations before start_time
                if (obs.timestamp < start_time) {
                    continue;
                }
                // Skip observations at or after end_time (if specified)
                if (end_time >= 0.0 && obs.timestamp >= end_time) {
                    continue;
                }
                full_seq.observations.push_back(obs);
            }
        }

        // Add the sequence to the observation set
        obs_set.add_sequence(full_seq);
    }

    return obs_set;
}

}  // namespace posetrak

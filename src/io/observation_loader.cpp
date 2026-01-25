#include <posetrak/io/observation_loader.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace posetrak {

namespace {

// Parse OpenPose "pose_keypoints_2d" array for one person
// Returns vector of Observation structs (one per keypoint)
std::vector<Observation> parse_openpose_person(nlohmann::json const& person, Camera const& camera,
                                               uint32_t frame_idx, double min_confidence) {
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
            Observation obs;
            obs.camera_id = 0;  // Will be set by caller if needed
            obs.marker_id = static_cast<int>(i);
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
                                        std::string const& camera_name, uint32_t frame_idx,
                                        double min_confidence, int person_id) {
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
        seq.camera_id = 0;
        seq.camera_name = camera_name;
        return seq;
    }

    // Check person_id is in range
    if (person_id < 0 || person_id >= static_cast<int>(people.size())) {
        // Person ID out of range - return empty
        ObservationSequence seq;
        seq.camera_id = 0;
        seq.camera_name = camera_name;
        return seq;
    }

    auto observations = parse_openpose_person(people[person_id], camera, frame_idx, min_confidence);

    ObservationSequence seq;
    seq.camera_id = 0;
    seq.camera_name = camera_name;
    seq.observations = std::move(observations);

    return seq;
}

ObservationSet load_openpose_sequence(std::string const& base_dir,
                                      std::unordered_map<std::string, Camera> const& cameras,
                                      std::pair<uint32_t, uint32_t> frame_range,
                                      double min_confidence, int person_id) {
    namespace fs = std::filesystem;

    if (!fs::exists(base_dir)) {
        throw std::runtime_error("Base directory does not exist: " + base_dir);
    }

    ObservationSet obs_set(person_id);

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
                if (frame_num >= frame_range.first && frame_num <= frame_range.second) {
                    frame_files.emplace_back(frame_num, entry.path());
                }
            }
        }

        // Sort by frame number
        std::sort(frame_files.begin(), frame_files.end(),
                  [](auto const& a, auto const& b) { return a.first < b.first; });

        // Load observations for each frame and build a sequence
        ObservationSequence full_seq;
        full_seq.camera_id = 0;
        full_seq.camera_name = cam_name;

        for (auto const& [frame_num, filepath] : frame_files) {
            auto frame_seq = load_openpose_frame(filepath.string(), camera, cam_name, frame_num,
                                                 min_confidence, person_id);

            // Append all observations from this frame to the full sequence
            full_seq.observations.insert(full_seq.observations.end(),
                                         frame_seq.observations.begin(),
                                         frame_seq.observations.end());
        }

        // Add the sequence to the observation set
        obs_set.add_sequence(full_seq);
    }

    return obs_set;
}

}  // namespace posetrak

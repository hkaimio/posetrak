#include <posetrak/core/observation.hpp>
#include <posetrak/core/skeleton.hpp>
#include <posetrak/io/camera_loader.hpp>
#include <posetrak/io/observation_loader.hpp>
#include <posetrak/io/skeleton_loader.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <filesystem>
#include <fstream>
#include <vector>

using namespace posetrak;

// Helper to create a minimal test skeleton with COCO markers
static Skeleton create_test_skeleton() {
    Skeleton skeleton;

    // Add a root joint
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());

    // Add markers with COCO IDs (0-24 for COCO body25 format)
    for (int i = 0; i < 25; ++i) {
        skeleton.add_marker("marker_" + std::to_string(i), 0, Eigen::Vector3d::Zero(), i);
    }

    return skeleton;
}

TEST_CASE("Load single OpenPose frame", "[observation_loader]") {
    // Load camera for undistortion
    auto cameras = load_cameras_from_toml("tests/data/pose2sim_camera_calib.toml");
    REQUIRE(cameras.count("cam1") > 0);
    auto const& camera = cameras.at("cam1");

    auto skeleton = create_test_skeleton();

    auto seq = load_openpose_frame("tests/data/openpose/cam1/cam1_000001.json", camera, "cam1",
                                   skeleton, 1);

    SECTION("Sequence has observations") {
        REQUIRE(!seq.observations.empty());
        REQUIRE(seq.camera_name == "cam1");
    }

    SECTION("Observations have correct structure") {
        // COCO-133 has 133 keypoints, but only valid ones (conf >= 0.1) are included
        REQUIRE(seq.observations.size() > 0);

        // Check first observation
        auto const& obs = seq.observations[0];
        REQUIRE(obs.marker_id >= 0);
        REQUIRE(obs.frame_idx == 1);
        REQUIRE(obs.confidence >= 0.1);
        REQUIRE(obs.timestamp >= 0.0);
    }

    SECTION("Positions are undistorted") {
        auto const& obs = seq.observations[0];
        // Undistorted position should differ from distorted
        // (unless distortion is negligible, but we just check they're set)
        bool position_set = (obs.position.x() != 0.0 || obs.position.y() != 0.0);
        bool distorted_set =
            (obs.position_distorted.x() != 0.0 || obs.position_distorted.y() != 0.0);
        REQUIRE(position_set);
        REQUIRE(distorted_set);
    }

    SECTION("Low confidence keypoints are filtered") {
        auto seq_low =
            load_openpose_frame("tests/data/openpose/cam1/cam1_000001.json", camera, "cam1",
                                create_test_skeleton(), 1, 5.0);  // High threshold

        // Should have fewer observations with higher threshold
        REQUIRE(seq_low.observations.size() < seq.observations.size());
    }

    SECTION("Can extract specific person") {
        auto seq_p0 = load_openpose_frame("tests/data/openpose/cam1/cam1_000001.json", camera,
                                          "cam1", create_test_skeleton(), 1, 0.1, 0);
        auto seq_p1 = load_openpose_frame("tests/data/openpose/cam1/cam1_000001.json", camera,
                                          "cam1", create_test_skeleton(), 1, 0.1, 1);

        REQUIRE(!seq_p0.observations.empty());
        REQUIRE(!seq_p1.observations.empty());
        // Different people may have different number of valid keypoints
    }
}

TEST_CASE("Load OpenPose sequence", "[observation_loader]") {
    auto cameras = load_cameras_from_toml("tests/data/pose2sim_camera_calib.toml");

    // Extract just cam1 and cam2 (the ones that have OpenPose data)
    std::map<std::string, Camera> openpose_cameras;
    openpose_cameras.emplace("cam1", cameras.at("cam1"));
    openpose_cameras.emplace("cam2", cameras.at("cam2"));

    auto obs_set = load_openpose_sequence("tests/data/openpose", openpose_cameras,
                                          create_test_skeleton(), {1, 10}, 0.1, 0);

    SECTION("Loads sequences for all cameras") {
        auto const* seq1 = obs_set.get_sequence("cam1");
        auto const* seq2 = obs_set.get_sequence("cam2");

        REQUIRE(seq1 != nullptr);
        REQUIRE(seq2 != nullptr);
    }

    SECTION("Sequences contain observations from multiple frames") {
        auto const* seq = obs_set.get_sequence("cam1");
        REQUIRE(seq != nullptr);
        REQUIRE(!seq->observations.empty());

        // With 10 frames and ~133 keypoints per frame, should have many observations
        REQUIRE(seq->observations.size() > 100);
    }

    SECTION("Can load subset of frames") {
        auto obs_set_subset = load_openpose_sequence("tests/data/openpose", openpose_cameras,
                                                     create_test_skeleton(), {1, 5}, 0.1, 0);

        auto const* seq_full = obs_set.get_sequence("cam1");
        auto const* seq_subset = obs_set_subset.get_sequence("cam1");

        REQUIRE(seq_subset->observations.size() < seq_full->observations.size());
    }

    SECTION("Observations have correct timestamps") {
        auto const* seq = obs_set.get_sequence("cam1");
        REQUIRE(seq != nullptr);
        REQUIRE(!seq->observations.empty());

        // All observations should have timestamps
        for (auto const& obs : seq->observations) {
            REQUIRE(obs.timestamp >= 0.0);
        }
    }
}

TEST_CASE("Observation loader error handling", "[observation_loader][errors]") {
    auto cameras = load_cameras_from_toml("tests/data/pose2sim_camera_calib.toml");
    auto const& camera = cameras.at("cam1");

    SECTION("Non-existent file throws") {
        REQUIRE_THROWS_AS(load_openpose_frame("tests/data/openpose/nonexistent.json", camera,
                                              "cam1", create_test_skeleton(), 0),
                          std::runtime_error);
    }

    SECTION("Invalid JSON throws") {
        auto get_temp_dir = []() {
            static auto temp = std::filesystem::temp_directory_path() / "posetrak_tests";
            std::filesystem::create_directories(temp);
            return temp;
        };

        auto test_file = get_temp_dir() / "invalid.json";
        std::ofstream f(test_file);
        f << R"({"broken json)";
        f.close();

        REQUIRE_THROWS_AS(
            load_openpose_frame(test_file.string(), camera, "cam1", create_test_skeleton(), 0),
            std::runtime_error);
    }

    SECTION("Missing 'people' array throws") {
        auto get_temp_dir = []() {
            static auto temp = std::filesystem::temp_directory_path() / "posetrak_tests";
            std::filesystem::create_directories(temp);
            return temp;
        };

        auto test_file = get_temp_dir() / "no_people.json";
        std::ofstream f(test_file);
        f << R"({"version": 1.3})";
        f.close();

        REQUIRE_THROWS_AS(
            load_openpose_frame(test_file.string(), camera, "cam1", create_test_skeleton(), 0),
            std::runtime_error);
    }

    SECTION("Empty people array returns empty sequence") {
        auto get_temp_dir = []() {
            static auto temp = std::filesystem::temp_directory_path() / "posetrak_tests";
            std::filesystem::create_directories(temp);
            return temp;
        };

        auto test_file = get_temp_dir() / "empty_people.json";
        std::ofstream f(test_file);
        f << R"({"version": 1.3, "people": []})";
        f.close();

        auto result =
            load_openpose_frame(test_file.string(), camera, "cam1", create_test_skeleton(), 0);
        REQUIRE(result.observations.empty());
    }

    SECTION("Non-existent base directory throws") {
        std::map<std::string, Camera> test_cameras;
        test_cameras.emplace("cam1", cameras.at("cam1"));

        REQUIRE_THROWS_AS(load_openpose_sequence("tests/data/nonexistent_dir", test_cameras,
                                                 create_test_skeleton()),
                          std::runtime_error);
    }

    SECTION("Missing camera directory throws") {
        std::map<std::string, Camera> test_cameras;
        test_cameras.emplace("nonexistent_cam", cameras.at("cam1"));

        REQUIRE_THROWS_AS(
            load_openpose_sequence("tests/data/openpose", test_cameras, create_test_skeleton()),
            std::runtime_error);
    }
}

TEST_CASE("Observation loader handles edge cases", "[observation_loader]") {
    auto cameras = load_cameras_from_toml("tests/data/pose2sim_camera_calib.toml");
    auto const& camera = cameras.at("cam1");

    SECTION("Handles file with person missing pose_keypoints_2d") {
        auto get_temp_dir = []() {
            static auto temp = std::filesystem::temp_directory_path() / "posetrak_tests";
            std::filesystem::create_directories(temp);
            return temp;
        };

        auto test_file = get_temp_dir() / "missing_keypoints.json";
        std::ofstream f(test_file);
        f << R"({"version": 1.3, "people": [{"person_id": [0]}]})";
        f.close();

        auto result =
            load_openpose_frame(test_file.string(), camera, "cam1", create_test_skeleton(), 0);
        // Should return empty sequence (person has no keypoints)
        REQUIRE(result.observations.empty());
    }

    SECTION("Person ID out of range returns empty sequence") {
        auto result =
            load_openpose_frame("tests/data/openpose/cam1/cam1_000001.json", camera, "cam1",
                                create_test_skeleton(), 1, 0.1, 999);  // Only 2 people exist
        REQUIRE(result.observations.empty());
    }

    SECTION("Zero confidence threshold includes more keypoints") {
        auto seq_zero = load_openpose_frame("tests/data/openpose/cam1/cam1_000001.json", camera,
                                            "cam1", create_test_skeleton(), 1, 0.0);
        auto seq_default = load_openpose_frame("tests/data/openpose/cam1/cam1_000001.json", camera,
                                               "cam1", create_test_skeleton(), 1, 0.1);

        // With 0.0 threshold, should have at least as many observations
        REQUIRE(seq_zero.observations.size() >= seq_default.observations.size());
    }
}

#include <posetrak/io/camera_loader.hpp>
#include <posetrak/io/sync_loader.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <filesystem>
#include <fstream>

using namespace posetrak;

TEST_CASE("Load synchronization metadata", "[sync_loader]") {
    auto sync_data = load_sync_metadata("tests/data/sync_metadata.json");

    SECTION("Correct number of cameras") {
        REQUIRE(sync_data.size() == 3);
    }

    SECTION("Sync points loaded for cam1") {
        REQUIRE(sync_data.count("cam1") == 1);
        auto const& cam1_sync = sync_data.at("cam1");
        REQUIRE(cam1_sync.size() == 3);

        REQUIRE(cam1_sync[0].frame_idx == 0);
        REQUIRE_THAT(cam1_sync[0].timestamp_sec, Catch::Matchers::WithinRel(0.0, 1e-6));

        REQUIRE(cam1_sync[1].frame_idx == 150);
        REQUIRE_THAT(cam1_sync[1].timestamp_sec, Catch::Matchers::WithinRel(5.0, 1e-6));

        REQUIRE(cam1_sync[2].frame_idx == 300);
        REQUIRE_THAT(cam1_sync[2].timestamp_sec, Catch::Matchers::WithinRel(10.0, 1e-6));
    }

    SECTION("Camera with offset (cam2)") {
        auto const& cam2_sync = sync_data.at("cam2");
        REQUIRE(cam2_sync.size() == 3);

        // 83ms offset from cam1
        REQUIRE_THAT(cam2_sync[0].timestamp_sec, Catch::Matchers::WithinRel(0.083, 1e-6));
        REQUIRE_THAT(cam2_sync[1].timestamp_sec, Catch::Matchers::WithinRel(5.083, 1e-6));
    }

    SECTION("Camera without sync points (cam3)") {
        auto const& cam3_sync = sync_data.at("cam3");
        REQUIRE(cam3_sync.empty());
    }
}

TEST_CASE("Apply sync metadata to cameras", "[sync_loader]") {
    // Load cameras from calibration
    auto cameras = load_cameras_from_toml("tests/data/pose2sim_camera_calib.toml");

    // Create minimal sync data for 2 cameras
    auto get_temp_dir = []() {
        static auto temp = std::filesystem::temp_directory_path() / "posetrak_tests";
        std::filesystem::create_directories(temp);
        return temp;
    };

    auto sync_file = get_temp_dir() / "test_sync.json";
    {
        std::ofstream f(sync_file);
        f << R"({
  "cam1": [
    {"frame": 0, "timestamp": 0.0},
    {"frame": 100, "timestamp": 3.333}
  ],
  "cam2": [
    {"frame": 0, "timestamp": 0.05},
    {"frame": 100, "timestamp": 3.383}
  ]
})";
        f.close();
    }

    auto sync_data = load_sync_metadata(sync_file.string());

    SECTION("Apply sync points to cameras (non-strict)") {
        apply_sync_metadata(cameras, sync_data, false);

        // Check that sync points were applied
        auto const& cam1 = cameras.at("cam1");
        auto t1 = cam1.get_timestamp(50);
        // Should interpolate: 50 is halfway between 0 and 100, so t ≈ 1.666s
        REQUIRE_THAT(t1, Catch::Matchers::WithinRel(1.6665, 1e-3));
    }

    SECTION("Strict mode throws on missing camera") {
        // Add sync data for non-existent camera
        std::vector<SyncPoint> dummy_points{{0, 0.0}};
        sync_data["nonexistent_cam"] = dummy_points;

        // Should not throw (nonexistent_cam in sync but not in cameras)
        REQUIRE_NOTHROW(apply_sync_metadata(cameras, sync_data, false));

        // But strict mode should throw
        std::map<std::string, Camera> empty_cameras;
        REQUIRE_THROWS_AS(apply_sync_metadata(empty_cameras, sync_data, true), std::runtime_error);
    }
}

TEST_CASE("Sync loader error handling", "[sync_loader][errors]") {
    auto get_temp_dir = []() {
        static auto temp = std::filesystem::temp_directory_path() / "posetrak_tests";
        std::filesystem::create_directories(temp);
        return temp;
    };

    SECTION("Non-existent file throws") {
        REQUIRE_THROWS_AS(load_sync_metadata("tests/data/nonexistent_sync.json"),
                          std::runtime_error);
    }

    SECTION("Invalid JSON syntax throws") {
        auto test_file = get_temp_dir() / "invalid_sync.json";
        std::ofstream f(test_file);
        f << R"({"cam1": [broken json)";
        f.close();
        REQUIRE_THROWS_AS(load_sync_metadata(test_file.string()), std::runtime_error);
    }

    SECTION("Non-monotonic frame indices throw") {
        auto test_file = get_temp_dir() / "bad_frame_order.json";
        std::ofstream f(test_file);
        f << R"({
  "cam1": [
    {"frame": 100, "timestamp": 3.0},
    {"frame": 50, "timestamp": 5.0}
  ]
})";
        f.close();
        REQUIRE_THROWS_AS(load_sync_metadata(test_file.string()), std::runtime_error);
    }

    SECTION("Non-monotonic timestamps throw") {
        auto test_file = get_temp_dir() / "bad_timestamp_order.json";
        std::ofstream f(test_file);
        f << R"({
  "cam1": [
    {"frame": 0, "timestamp": 5.0},
    {"frame": 100, "timestamp": 3.0}
  ]
})";
        f.close();
        REQUIRE_THROWS_AS(load_sync_metadata(test_file.string()), std::runtime_error);
    }

    SECTION("Negative timestamp throws") {
        auto test_file = get_temp_dir() / "negative_timestamp.json";
        std::ofstream f(test_file);
        f << R"({
  "cam1": [
    {"frame": 0, "timestamp": -1.0}
  ]
})";
        f.close();
        REQUIRE_THROWS_AS(load_sync_metadata(test_file.string()), std::runtime_error);
    }

    SECTION("Empty cameras throws") {
        auto test_file = get_temp_dir() / "empty.json";
        std::ofstream f(test_file);
        f << R"({})";
        f.close();
        REQUIRE_THROWS_AS(load_sync_metadata(test_file.string()), std::runtime_error);
    }

    SECTION("Sync point missing fields throws") {
        auto test_file = get_temp_dir() / "missing_sync_field.json";
        std::ofstream f(test_file);
        f << R"({
  "cam1": [
    {"frame": 0}
  ]
})";
        f.close();
        REQUIRE_THROWS_AS(load_sync_metadata(test_file.string()), std::runtime_error);
    }

    SECTION("Root is not object throws") {
        auto test_file = get_temp_dir() / "not_object.json";
        std::ofstream f(test_file);
        f << R"([1, 2, 3])";
        f.close();
        REQUIRE_THROWS_AS(load_sync_metadata(test_file.string()), std::runtime_error);
    }

    SECTION("Camera value is not array throws") {
        auto test_file = get_temp_dir() / "not_array.json";
        std::ofstream f(test_file);
        f << R"({"cam1": {"frame": 0, "timestamp": 0.0}})";
        f.close();
        REQUIRE_THROWS_AS(load_sync_metadata(test_file.string()), std::runtime_error);
    }
}

TEST_CASE("Sync loader handles minimal files", "[sync_loader]") {
    auto get_temp_dir = []() {
        static auto temp = std::filesystem::temp_directory_path() / "posetrak_tests";
        std::filesystem::create_directories(temp);
        return temp;
    };

    SECTION("Empty array is valid") {
        auto test_file = get_temp_dir() / "empty_array.json";
        std::ofstream f(test_file);
        f << R"({
  "cam1": [],
  "cam2": []
})";
        f.close();

        auto sync_data = load_sync_metadata(test_file.string());
        REQUIRE(sync_data.size() == 2);
        REQUIRE(sync_data.at("cam1").empty());
        REQUIRE(sync_data.at("cam2").empty());
    }

    SECTION("Null sync points is valid") {
        auto test_file = get_temp_dir() / "null_sync.json";
        std::ofstream f(test_file);
        f << R"({
  "cam1": null
})";
        f.close();

        auto sync_data = load_sync_metadata(test_file.string());
        REQUIRE(sync_data.at("cam1").empty());
    }
}

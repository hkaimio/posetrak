#include <posetrak/io/camera_loader.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>

using namespace posetrak;

TEST_CASE("Load Pose2Sim camera calibration", "[camera_loader]") {
    auto cameras = load_cameras_from_toml("tests/data/pose2sim_camera_calib.toml");

    SECTION("Correct number of cameras loaded") {
        REQUIRE(cameras.size() == 6);
    }

    SECTION("Camera names are correct") {
        REQUIRE(cameras.count("cam1") == 1);
        REQUIRE(cameras.count("cam2") == 1);
        REQUIRE(cameras.count("cam6") == 1);

        REQUIRE(cameras.at("cam1").name() == "int_cam1_img");
        REQUIRE(cameras.at("cam2").name() == "int_cam2_img");
    }

    SECTION("Camera intrinsics are correct") {
        auto const& cam1 = cameras.at("cam1");
        auto const& intrinsics = cam1.intrinsics();

        REQUIRE_THAT(intrinsics.fx, Catch::Matchers::WithinRel(1658.600215949592, 1e-6));
        REQUIRE_THAT(intrinsics.fy, Catch::Matchers::WithinRel(1793.7965703657846, 1e-6));
        REQUIRE_THAT(intrinsics.cx, Catch::Matchers::WithinRel(1882.5305217044147, 1e-6));
        REQUIRE_THAT(intrinsics.cy, Catch::Matchers::WithinRel(1119.7168267684224, 1e-6));

        REQUIRE(intrinsics.width == 3840);
        REQUIRE(intrinsics.height == 2160);

        REQUIRE(intrinsics.model == Intrinsics::DistortionModel::BrownConrady);
        REQUIRE(intrinsics.distortion_coeffs.size() == 4);
    }

    SECTION("Camera with non-zero distortion") {
        auto const& cam4 = cameras.at("cam4");
        auto const& dist = cam4.intrinsics().distortion_coeffs;

        REQUIRE(dist.size() == 4);
        REQUIRE_THAT(dist[0], Catch::Matchers::WithinRel(-0.04662083577019837, 1e-6));
        REQUIRE_THAT(dist[1], Catch::Matchers::WithinRel(0.008958661433299635, 1e-6));
        REQUIRE_THAT(dist[2], Catch::Matchers::WithinRel(-5.684305186268812e-05, 1e-6));
        REQUIRE_THAT(dist[3], Catch::Matchers::WithinRel(0.004028643053932034, 1e-6));
    }

    SECTION("Camera extrinsics are loaded") {
        auto const& cam1 = cameras.at("cam1");
        auto const& extrinsics = cam1.extrinsics();

        // The TOML format stores OpenCV-convention extrinsics: point_cam = R * point_world + t.
        // The loader converts to our convention where position is the camera centre in world
        // coordinates: position = -R^T * t.
        // For cam1: t = [-4.371, -0.706, 8.673], R from rvec [1.465, 1.304, -0.917]
        // → position = -R^T * t ≈ [9.080, 2.905, 1.982]
        REQUIRE_THAT(extrinsics.position[0], Catch::Matchers::WithinRel(9.08011254621924, 1e-5));
        REQUIRE_THAT(extrinsics.position[1], Catch::Matchers::WithinRel(2.905263546282427, 1e-5));
        REQUIRE_THAT(extrinsics.position[2], Catch::Matchers::WithinRel(1.9817287797111272, 1e-5));

        // Check rotation (quaternion should be normalized)
        REQUIRE_THAT(extrinsics.orientation.norm(), Catch::Matchers::WithinRel(1.0, 1e-6));
    }

    SECTION("Rodrigues rotation is converted correctly") {
        // For a small rotation, Rodrigues vector ≈ rotation axis * angle
        // We can verify the quaternion represents a valid rotation (norm=1)
        auto const& cam1 = cameras.at("cam1");
        auto const& q = cam1.extrinsics().orientation;

        // Quaternion should be normalized
        REQUIRE_THAT(q.w() * q.w() + q.x() * q.x() + q.y() * q.y() + q.z() * q.z(),
                     Catch::Matchers::WithinRel(1.0, 1e-6));
    }

    SECTION("Default FPS and start frame") {
        auto const& cam1 = cameras.at("cam1");
        REQUIRE_THAT(cam1.fps(), Catch::Matchers::WithinRel(30.0, 1e-6));
        REQUIRE(cam1.start_frame() == 0);
    }
}

TEST_CASE("Camera loader error handling", "[camera_loader][errors]") {
    auto get_temp_dir = []() {
        static auto temp = std::filesystem::temp_directory_path() / "posetrak_tests";
        std::filesystem::create_directories(temp);
        return temp;
    };

    SECTION("Non-existent file throws") {
        REQUIRE_THROWS_AS(load_cameras_from_toml("tests/data/nonexistent.toml"),
                          std::runtime_error);
    }

    SECTION("Invalid TOML syntax throws") {
        auto test_file = get_temp_dir() / "invalid_camera.toml";
        std::ofstream f(test_file);
        f << "[cam1]\nname = \"test\"\nsize = [broken toml\n";
        f.close();
        REQUIRE_THROWS_AS(load_cameras_from_toml(test_file.string()), std::runtime_error);
    }

    SECTION("Missing required fields throws") {
        auto test_file = get_temp_dir() / "missing_fields.toml";
        std::ofstream f(test_file);
        f << "[cam1]\n"
          << "name = \"test\"\n"
          << "size = [640, 480]\n";
        // Missing matrix, distortions, rotation, translation, fisheye
        f.close();
        REQUIRE_THROWS_AS(load_cameras_from_toml(test_file.string()), std::runtime_error);
    }

    SECTION("Invalid matrix size throws") {
        auto test_file = get_temp_dir() / "bad_matrix.toml";
        std::ofstream f(test_file);
        f << "[cam1]\n"
          << "name = \"test\"\n"
          << "size = [640, 480]\n"
          << "matrix = [[100, 0], [0, 100]]\n"  // Wrong size (2x2 instead of 3x3)
          << "distortions = [0, 0, 0, 0]\n"
          << "rotation = [0, 0, 0]\n"
          << "translation = [0, 0, 0]\n"
          << "fisheye = false\n";
        f.close();
        REQUIRE_THROWS_AS(load_cameras_from_toml(test_file.string()), std::runtime_error);
    }

    SECTION("Empty file (no cameras) throws") {
        auto test_file = get_temp_dir() / "empty.toml";
        std::ofstream f(test_file);
        f << "[metadata]\nerror = 0.0\n";
        f.close();
        REQUIRE_THROWS_AS(load_cameras_from_toml(test_file.string()), std::runtime_error);
    }
}

TEST_CASE("Camera loader handles different image sizes", "[camera_loader]") {
    auto cameras = load_cameras_from_toml("tests/data/pose2sim_camera_calib.toml");

    SECTION("Cameras have different resolutions") {
        // cam1: 3840x2160 (4K)
        REQUIRE(cameras.at("cam1").intrinsics().width == 3840);
        REQUIRE(cameras.at("cam1").intrinsics().height == 2160);

        // cam4: 3584x2016
        REQUIRE(cameras.at("cam4").intrinsics().width == 3584);
        REQUIRE(cameras.at("cam4").intrinsics().height == 2016);

        // cam5: 1920x1080 (Full HD)
        REQUIRE(cameras.at("cam5").intrinsics().width == 1920);
        REQUIRE(cameras.at("cam5").intrinsics().height == 1080);
    }
}

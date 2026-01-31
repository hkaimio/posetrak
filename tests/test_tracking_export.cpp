#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/camera.hpp"
#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/state.hpp"
#include "posetrak/filters/update_result.hpp"
#include "posetrak/io/tracking_export.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace posetrak;

TEST_CASE("TrackingExporter basic functionality", "[io][export]") {
    // Create a simple skeleton with 1 marker
    Skeleton skeleton;
    auto root_idx =
        skeleton.add_joint("pelvis", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero());
    auto spine_idx =
        skeleton.add_joint("spine", root_idx, JointType::SPHERICAL, Eigen::Vector3d(0, 0.15, 0));

    skeleton.add_marker("pelvis_center", root_idx, Eigen::Vector3d::Zero());
    skeleton.add_marker("spine_base", spine_idx, Eigen::Vector3d::Zero());

    // Create a simple camera
    std::unordered_map<int, Camera> cameras;
    Intrinsics intrinsics;
    intrinsics.fx = 1000.0;
    intrinsics.fy = 1000.0;
    intrinsics.cx = 640.0;
    intrinsics.cy = 360.0;
    intrinsics.width = 1280;
    intrinsics.height = 720;
    intrinsics.model = Intrinsics::DistortionModel::BrownConrady;
    intrinsics.distortion_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};

    Extrinsics extrinsics;
    extrinsics.position = Eigen::Vector3d(0, 0, -3);
    extrinsics.orientation = Eigen::Quaterniond::Identity();

    cameras.emplace(0, Camera(0, "cam1", intrinsics, extrinsics, 30.0, 0));

    // Create temporary output directory
    std::filesystem::path temp_dir =
        std::filesystem::temp_directory_path() / "posetrak_export_test";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    // Create exporter
    TrackingExporter exporter(temp_dir, skeleton, cameras);
    exporter.open();

    SECTION("Write single frame") {
        // Create a simple state
        int num_dof = skeleton.total_dof_count();
        State state(num_dof);

        // Create marker positions
        std::map<std::string, Eigen::Vector3d> marker_positions;
        marker_positions["pelvis_center"] = Eigen::Vector3d(0, 0, 0);
        marker_positions["spine_base"] = Eigen::Vector3d(0, 0.15, 0);

        // Create observations
        std::vector<Observation> observations;
        Observation obs1;
        obs1.camera_id = 0;
        obs1.marker_id = 0;
        obs1.frame_idx = 0;
        obs1.timestamp = 0.0;
        obs1.position = Eigen::Vector2d(640, 360);
        obs1.position_distorted = Eigen::Vector2d(640, 360);
        obs1.confidence = 0.95;
        observations.push_back(obs1);

        // Create update result
        UpdateResult update_result;
        update_result.num_observations = 1;
        update_result.num_inliers = 1;
        update_result.num_outliers = 0;

        ObservationResult obs_result;
        obs_result.marker_name = "pelvis_center";
        obs_result.camera_id = 0;
        obs_result.is_outlier = false;
        obs_result.mahalanobis_distance = 1.5;
        obs_result.innovation = Eigen::Vector2d(0.5, 0.3);
        obs_result.predicted = Eigen::Vector2d(639.5, 359.7);
        obs_result.actual = Eigen::Vector2d(640, 360);
        update_result.observations.push_back(obs_result);

        // Write frame
        exporter.write_frame(0, 0.0, state, marker_positions, observations, update_result);
        exporter.close();

        // Verify files were created
        REQUIRE(std::filesystem::exists(temp_dir / "tracking_results.csv"));
        REQUIRE(std::filesystem::exists(temp_dir / "joint_angles.csv"));
        REQUIRE(std::filesystem::exists(temp_dir / "root_pose.csv"));
        REQUIRE(std::filesystem::exists(temp_dir / "marker_projections.csv"));
        REQUIRE(std::filesystem::exists(temp_dir / "observations.csv"));

        // Verify tracking_results.csv has correct header and data
        std::ifstream tracking_file(temp_dir / "tracking_results.csv");
        std::string line;
        std::getline(tracking_file, line);
        REQUIRE(line == "frame,timestamp,marker_id,marker_name,x_3d,y_3d,z_3d,is_visible");

        // Should have 2 markers
        std::getline(tracking_file, line);
        REQUIRE(line.find("pelvis_center") != std::string::npos);
        std::getline(tracking_file, line);
        REQUIRE(line.find("spine_base") != std::string::npos);

        // Verify observations.csv
        std::ifstream obs_file(temp_dir / "observations.csv");
        std::getline(obs_file, line);
        REQUIRE(line ==
                "frame,timestamp,marker_id,marker_name,camera_id,pixel_x,pixel_y,confidence,used_"
                "in_tracking");
        std::getline(obs_file, line);
        REQUIRE(line.find("pelvis_center") != std::string::npos);
        REQUIRE(line.find("640") != std::string::npos);
        REQUIRE(line.find("360") != std::string::npos);
        REQUIRE(line.find("true") != std::string::npos);  // used_in_tracking
    }

    // Cleanup
    std::filesystem::remove_all(temp_dir);
}

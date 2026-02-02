#include <posetrak/core/camera.hpp>
#include <posetrak/core/skeleton.hpp>
#include <posetrak/filters/ukf.hpp>
#include <posetrak/io/camera_loader.hpp>
#include <posetrak/io/skeleton_loader.hpp>
#include <posetrak/kinematics/forward_kinematics.hpp>
#include <posetrak/kinematics/pinocchio_model_builder.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/io/camera_loader.hpp"
#include "posetrak/io/skeleton_loader.hpp"
#include "test_helpers/matrix_comparison.hpp"
#include "test_helpers/python_data_loader.hpp"
#include <filesystem>

using namespace posetrak;
using namespace posetrak::test_helpers;

namespace {

// Fixture to load common test data
struct Frame0TestFixture {
    Frame0TestFixture() {
        // Paths to test data
        skeleton_path =
            "../../tracking_tests/cpp-python-comparison/python_results/skeleton_structure/"
            "skeleton.yaml";
        cameras_path =
            "../../tracking_tests/cpp-python-comparison/python_results/skeleton_structure/"
            "cameras.toml";
        debug_dir = "../../tracking_tests/cpp-python-comparison/python_results/debug/frame_0000";

        // Check if files exist
        if (!std::filesystem::exists(skeleton_path)) {
            WARN("Skeleton file not found: " << skeleton_path);
            WARN("Test may be running from unexpected directory. Trying absolute path...");
            skeleton_path =
                "/mnt/d/mocap/2026-01-11-kotegaesh-joint-space-test/"
                "Harri_skeleton-shouldery-rot.yaml";
            cameras_path = "/mnt/d/mocap/2026-01-11-kotegaesh-joint-space-test/Calib_scene.toml";
            debug_dir = std::filesystem::current_path().string() +
                        "/tracking_tests/cpp-python-comparison/python_results/debug/frame_0000";
        }

        // Load skeleton and cameras
        skeleton = load_skeleton_from_yaml(skeleton_path);
        cameras_by_name = load_cameras_from_toml(cameras_path);

        // Convert to ID-keyed map for UKF
        for (auto const& [name, camera] : cameras_by_name) {
            cameras.insert({camera.id(), camera});
            camera_name_to_id[name] = camera.id();
        }

        // Build Pinocchio model for FK
        model = std::make_unique<pinocchio::Model>();
        data = std::make_unique<pinocchio::Data>();
        PinocchioModelBuilder::build_model_and_data(skeleton, *model, *data);
        marker_frame_map = PinocchioModelBuilder::build_marker_frame_map(*model, skeleton);

        // Create FK computer
        fk = std::make_unique<ForwardKinematics>(*model, *data, marker_frame_map, skeleton);
    }

    std::string skeleton_path;
    std::string cameras_path;
    std::string debug_dir;

    Skeleton skeleton;
    std::unordered_map<std::string, Camera> cameras_by_name;  // Original from loader
    std::unordered_map<int, Camera> cameras;                  // Keyed by camera ID for UKF
    std::unordered_map<std::string, int> camera_name_to_id;

    std::unique_ptr<pinocchio::Model> model;
    std::unique_ptr<pinocchio::Data> data;
    std::map<std::string, pinocchio::FrameIndex> marker_frame_map;
    std::unique_ptr<ForwardKinematics> fk;
};

}  // namespace

TEST_CASE("Frame 0: Load Python debug data", "[ukf][frame0][data]") {
    Frame0TestFixture fixture;

    SECTION("Python debug files exist") {
        REQUIRE(std::filesystem::exists(fixture.debug_dir));
        REQUIRE(std::filesystem::exists(fixture.debug_dir + "/prior_state.json"));
        REQUIRE(std::filesystem::exists(fixture.debug_dir + "/prior_covariance.csv"));
        REQUIRE(std::filesystem::exists(fixture.debug_dir + "/all_observations.csv"));
        REQUIRE(std::filesystem::exists(fixture.debug_dir + "/sigma_points.csv"));
    }

    SECTION("Load Python frame 0 data") {
        auto python_data = load_python_frame0_data(fixture.debug_dir, fixture.skeleton);

        // Verify data was loaded
        INFO("Loaded " << python_data.observations.size() << " observations");
        REQUIRE(python_data.observations.size() == 341);
        REQUIRE(python_data.prior_state.has_value());
        REQUIRE(python_data.posterior_state.has_value());

        INFO("Prior state root position: " << python_data.prior_state->root_position().transpose());
        REQUIRE(python_data.prior_state->root_position().norm() > 0.0);

        INFO("Sigma points shape: " << python_data.sigma_points.rows() << " × "
                                    << python_data.sigma_points.cols());
        REQUIRE(python_data.sigma_points.cols() == 145);  // 2*n+1 where n=72

        INFO("Prior covariance shape: " << python_data.prior_covariance.rows() << " × "
                                        << python_data.prior_covariance.cols());
        REQUIRE(python_data.prior_covariance.rows() == python_data.prior_covariance.cols());

        // Verify specific values match what we expect from Python
        Eigen::Vector3d expected_root_pos(9.08100621021731, 1.7802621944156272, 1.9826858437081682);
        REQUIRE_THAT(python_data.prior_state->root_position()(0),
                     Catch::Matchers::WithinAbs(expected_root_pos(0), 1e-6));
        REQUIRE_THAT(python_data.prior_state->root_position()(1),
                     Catch::Matchers::WithinAbs(expected_root_pos(1), 1e-6));
        REQUIRE_THAT(python_data.prior_state->root_position()(2),
                     Catch::Matchers::WithinAbs(expected_root_pos(2), 1e-6));
    }

    SECTION("Verify observation outlier flags") {
        auto python_data = load_python_frame0_data(fixture.debug_dir, fixture.skeleton);

        // Count inliers and outliers
        size_t num_inliers = 0;
        size_t num_outliers = 0;
        for (bool is_outlier : python_data.outlier_flags) {
            if (is_outlier) {
                num_outliers++;
            } else {
                num_inliers++;
            }
        }

        INFO("Python had " << num_inliers << " inliers, " << num_outliers << " outliers");
        // Python should have 115 inliers based on earlier debug output
        REQUIRE(num_inliers > 100);  // Rough sanity check
        REQUIRE(num_outliers > 200);
    }
}

TEST_CASE("Frame 0: Initialize C++ UKF with Python prior state", "[ukf][frame0][init]") {
    Frame0TestFixture fixture;
    auto python_data = load_python_frame0_data(fixture.debug_dir, fixture.skeleton);

    SECTION("Create UKF with Python parameters") {
        // Python UKF parameters (must match exactly)
        double alpha = 0.5;  // Spread parameter
        double beta = 2.0;   // Gaussian distribution parameter
        double kappa = 0.0;  // Secondary scaling
        double process_noise_std = 0.01;

        UnscentedKalmanFilter ukf(fixture.skeleton, process_noise_std, alpha, beta, kappa);

        // Set state and covariance from Python
        ukf.set_state(*python_data.prior_state);
        ukf.set_covariance(python_data.prior_covariance);

        // Verify state was set correctly
        State const& cpp_state = ukf.state();
        State const& python_state = *python_data.prior_state;

        INFO("C++ root position: " << cpp_state.root_position().transpose());
        INFO("Python root position: " << python_state.root_position().transpose());

        REQUIRE_THAT(cpp_state.root_position()(0),
                     Catch::Matchers::WithinAbs(python_state.root_position()(0), 1e-12));
        REQUIRE_THAT(cpp_state.root_position()(1),
                     Catch::Matchers::WithinAbs(python_state.root_position()(1), 1e-12));
        REQUIRE_THAT(cpp_state.root_position()(2),
                     Catch::Matchers::WithinAbs(python_state.root_position()(2), 1e-12));

        // Verify covariance was set correctly
        Eigen::MatrixXd const& cpp_cov = ukf.covariance();
        Eigen::MatrixXd const& python_cov = python_data.prior_covariance;

        REQUIRE(cpp_cov.rows() == python_cov.rows());
        REQUIRE(cpp_cov.cols() == python_cov.cols());
        REQUIRE(matrices_equal(cpp_cov, python_cov, 1e-12));
    }
}

TEST_CASE("Frame 0: Compare sigma point generation", "[ukf][frame0][sigma_points]") {
    Frame0TestFixture fixture;
    auto python_data = load_python_frame0_data(fixture.debug_dir, fixture.skeleton);

    SECTION("Generate sigma points and compare with Python") {
        // Create UKF with Python's prior state
        double alpha = 0.5;
        double beta = 2.0;
        double kappa = 0.0;
        double process_noise_std = 0.01;

        UnscentedKalmanFilter ukf(fixture.skeleton, process_noise_std, alpha, beta, kappa);
        ukf.set_state(*python_data.prior_state);
        ukf.set_covariance(python_data.prior_covariance);

        // Generate sigma points
        auto cpp_sigma_points = ukf.generate_sigma_points_for_testing();

        INFO("Generated " << cpp_sigma_points.size() << " sigma points");
        REQUIRE(cpp_sigma_points.size() == 145);  // 2*72 + 1

        // Compare each sigma point
        size_t num_differences = 0;
        double max_diff = 0.0;
        int first_diff_idx = -1;

        for (size_t i = 0; i < cpp_sigma_points.size(); ++i) {
            // Convert C++ state to vector manually
            State const& s = cpp_sigma_points[i];
            Eigen::VectorXd cpp_vec(s.error_state_dim());
            int idx = 0;
            cpp_vec.segment<3>(idx) = s.root_position();
            idx += 3;
            cpp_vec.segment<3>(idx) = s.root_orientation().vec();
            idx += 3;
            cpp_vec.segment(idx, s.joint_angles().size()) = s.joint_angles();
            idx += s.joint_angles().size();
            cpp_vec.segment<3>(idx) = s.root_velocity();
            idx += 3;
            cpp_vec.segment<3>(idx) = s.root_angular_velocity();
            idx += 3;
            cpp_vec.segment(idx, s.joint_velocities().size()) = s.joint_velocities();

            // Get Python sigma point
            Eigen::VectorXd python_vec = python_data.sigma_points.col(i);

            // Compare
            if (!matrices_equal(cpp_vec, python_vec, 1e-10)) {
                if (first_diff_idx < 0) {
                    first_diff_idx = static_cast<int>(i);
                }
                num_differences++;

                // Find max difference
                for (int j = 0; j < cpp_vec.size(); ++j) {
                    double diff = std::abs(cpp_vec(j) - python_vec(j));
                    max_diff = std::max(max_diff, diff);
                }
            }
        }

        if (num_differences > 0) {
            WARN("Found " << num_differences << " differing sigma points");
            WARN("First difference at sigma point " << first_diff_idx);
            WARN("Max difference: " << max_diff);

            // Print detailed info about first differing sigma point
            if (first_diff_idx >= 0) {
                // Convert C++ state to vector manually
                State const& s = cpp_sigma_points[first_diff_idx];
                Eigen::VectorXd cpp_vec(s.error_state_dim());
                int idx = 0;
                cpp_vec.segment<3>(idx) = s.root_position();
                idx += 3;
                cpp_vec.segment<3>(idx) = s.root_orientation().vec();
                idx += 3;
                cpp_vec.segment(idx, s.joint_angles().size()) = s.joint_angles();
                idx += s.joint_angles().size();
                cpp_vec.segment<3>(idx) = s.root_velocity();
                idx += 3;
                cpp_vec.segment<3>(idx) = s.root_angular_velocity();
                idx += 3;
                cpp_vec.segment(idx, s.joint_velocities().size()) = s.joint_velocities();

                Eigen::VectorXd python_vec = python_data.sigma_points.col(first_diff_idx);

                INFO("Sigma point " << first_diff_idx << " comparison:");
                INFO(matrix_diff_string(cpp_vec, python_vec, 1e-10));
            }
        }

        REQUIRE(num_differences == 0);
    }
}

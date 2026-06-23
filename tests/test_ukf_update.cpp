/**
 * @file test_ukf_update.cpp
 * @brief Tests for UKF update step (measurement correction)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/filters/ukf.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include "posetrak/kinematics/pinocchio_model_builder.hpp"
#include <iostream>

using namespace posetrak;
using Catch::Matchers::WithinAbs;

TEST_CASE("UKF update with single observation", "[ukf][update]") {
    // Create simple skeleton: root + single marker
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    uint32_t marker_idx = skeleton.add_marker("test_marker", 0, Eigen::Vector3d(0, 0, 0));

    // Build Pinocchio model for FK
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
    auto marker_frame_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
    auto fk_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));

    ForwardKinematics fk(model, data, marker_frame_map, fk_layout);

    // Create simple camera looking at origin
    Intrinsics intrinsics;
    intrinsics.fx = 500.0;
    intrinsics.fy = 500.0;
    intrinsics.cx = 320.0;
    intrinsics.cy = 240.0;
    intrinsics.width = 640;
    intrinsics.height = 480;
    intrinsics.model = Intrinsics::DistortionModel::BrownConrady;
    intrinsics.distortion_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};

    Extrinsics extrinsics;
    extrinsics.position = Eigen::Vector3d(0, 0, -2.0);  // Camera 2m back
    extrinsics.orientation = Eigen::Quaterniond::Identity();

    Camera camera(0, "cam0", intrinsics, extrinsics);
    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, camera);

    // Create UKF
    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.01);

    // Set initial state at origin
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles = Eigen::VectorXd::Zero(0);  // No joints
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(0);

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    // Set initial covariance (large uncertainty)
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(12, 12) * 0.1;
    ukf.set_covariance(cov);

    // Create observation of marker at image center (confirming position)
    Observation obs;
    obs.camera_id = 0;
    obs.marker_id = marker_idx;
    obs.frame_idx = 0;
    obs.timestamp = 0.0;
    obs.position = Eigen::Vector2d(320.0, 240.0);  // Center of image
    obs.confidence = 1.0;

    std::vector<Observation> observations = {obs};

    // Perform update
    ukf.update(observations, cameras, fk);

    // Check that state didn't change much (observation confirms initial state)
    REQUIRE_THAT(ukf.state().root_position().x(), WithinAbs(0.0, 0.1));
    REQUIRE_THAT(ukf.state().root_position().y(), WithinAbs(0.0, 0.1));
    REQUIRE_THAT(ukf.state().root_position().z(), WithinAbs(0.0, 0.1));

    // Check that covariance decreased (uncertainty reduced)
    Eigen::MatrixXd updated_cov = ukf.covariance();
    REQUIRE(updated_cov.trace() < cov.trace());
}

TEST_CASE("UKF update corrects position error", "[ukf][update]") {
    // Create simple skeleton: root + single marker
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    uint32_t marker_idx = skeleton.add_marker("test_marker", 0, Eigen::Vector3d(0, 0, 0));

    // Build Pinocchio model
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
    auto marker_frame_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
    auto fk_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));

    ForwardKinematics fk(model, data, marker_frame_map, fk_layout);

    // Create camera
    Intrinsics intrinsics;
    intrinsics.fx = 500.0;
    intrinsics.fy = 500.0;
    intrinsics.cx = 320.0;
    intrinsics.cy = 240.0;
    intrinsics.width = 640;
    intrinsics.height = 480;
    intrinsics.model = Intrinsics::DistortionModel::BrownConrady;
    intrinsics.distortion_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};

    Extrinsics extrinsics;
    extrinsics.position = Eigen::Vector3d(0, 0, -2.0);
    extrinsics.orientation = Eigen::Quaterniond::Identity();

    Camera camera(0, "cam0", intrinsics, extrinsics);
    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, camera);

    // Create UKF
    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.01);

    // Set WRONG initial state (shifted right by 0.2m)
    Eigen::Vector3d pos(0.2, 0.0, 0.0);
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles = Eigen::VectorXd::Zero(0);
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(0);

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    // Set moderate covariance
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(12, 12) * 0.01;
    ukf.set_covariance(cov);

    // Create observation of marker at image center (true position at origin)
    Observation obs;
    obs.camera_id = 0;
    obs.marker_id = marker_idx;
    obs.frame_idx = 0;
    obs.timestamp = 0.0;
    obs.position = Eigen::Vector2d(320.0, 240.0);  // Center (marker at origin)
    obs.confidence = 1.0;

    std::vector<Observation> observations = {obs};

    // Perform update
    ukf.update(observations, cameras, fk);

    // Check that position was corrected toward origin
    REQUIRE(std::abs(ukf.state().root_position().x()) < 0.2);  // Should move toward 0
}

TEST_CASE("UKF update with multiple observations", "[ukf][update]") {
    // Create skeleton with 2 markers
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    uint32_t marker1 = skeleton.add_marker("marker1", 0, Eigen::Vector3d(-0.1, 0, 0));
    uint32_t marker2 = skeleton.add_marker("marker2", 0, Eigen::Vector3d(0.1, 0, 0));

    // Build Pinocchio model
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
    auto marker_frame_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
    auto fk_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));

    ForwardKinematics fk(model, data, marker_frame_map, fk_layout);

    // Create camera
    Intrinsics intrinsics;
    intrinsics.fx = 500.0;
    intrinsics.fy = 500.0;
    intrinsics.cx = 320.0;
    intrinsics.cy = 240.0;
    intrinsics.width = 640;
    intrinsics.height = 480;
    intrinsics.model = Intrinsics::DistortionModel::BrownConrady;
    intrinsics.distortion_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};

    Extrinsics extrinsics;
    extrinsics.position = Eigen::Vector3d(0, 0, -2.0);
    extrinsics.orientation = Eigen::Quaterniond::Identity();

    Camera camera(0, "cam0", intrinsics, extrinsics);
    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, camera);

    // Create UKF
    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.01);

    // Set initial state
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles = Eigen::VectorXd::Zero(0);
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(0);

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(12, 12) * 0.1;
    ukf.set_covariance(cov);

    // Create observations for both markers
    Observation obs1;
    obs1.camera_id = 0;
    obs1.marker_id = marker1;
    obs1.position = *camera.project_undistorted(Eigen::Vector3d(-0.1, 0, 0));
    obs1.confidence = 1.0;

    Observation obs2;
    obs2.camera_id = 0;
    obs2.marker_id = marker2;
    obs2.position = *camera.project_undistorted(Eigen::Vector3d(0.1, 0, 0));
    obs2.confidence = 1.0;

    std::vector<Observation> observations = {obs1, obs2};

    // Perform update
    ukf.update(observations, cameras, fk);

    // Check covariance decreased (more observations = more information)
    REQUIRE(ukf.covariance().trace() < cov.trace());
}

TEST_CASE("UKF update with missing observations", "[ukf][update]") {
    // Create skeleton with marker
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_marker("test_marker", 0, Eigen::Vector3d(0, 0, 0));

    // Build Pinocchio model
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
    auto marker_frame_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
    auto fk_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));

    ForwardKinematics fk(model, data, marker_frame_map, fk_layout);

    std::unordered_map<int, Camera> cameras;

    // Create UKF
    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.01);

    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles = Eigen::VectorXd::Zero(0);
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(0);

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(12, 12) * 0.1;
    ukf.set_covariance(cov);

    // Empty observations (no markers visible)
    std::vector<Observation> empty_observations;

    // Perform update (should do nothing)
    ukf.update(empty_observations, cameras, fk);

    // State should be unchanged
    REQUIRE_THAT(ukf.state().root_position().x(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(ukf.state().root_position().y(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(ukf.state().root_position().z(), WithinAbs(0.0, 1e-9));

    // Covariance should be unchanged
    REQUIRE_THAT(ukf.covariance().norm(), WithinAbs(cov.norm(), 1e-9));
}

TEST_CASE("UKF update maintains positive definite covariance", "[ukf][update]") {
    // Create skeleton
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    uint32_t marker_idx = skeleton.add_marker("test_marker", 0, Eigen::Vector3d(0, 0, 0));

    // Build Pinocchio model
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
    auto marker_frame_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
    auto fk_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));

    ForwardKinematics fk(model, data, marker_frame_map, fk_layout);

    // Create camera
    Intrinsics intrinsics;
    intrinsics.fx = 500.0;
    intrinsics.fy = 500.0;
    intrinsics.cx = 320.0;
    intrinsics.cy = 240.0;
    intrinsics.width = 640;
    intrinsics.height = 480;
    intrinsics.model = Intrinsics::DistortionModel::BrownConrady;
    intrinsics.distortion_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};

    Extrinsics extrinsics;
    extrinsics.position = Eigen::Vector3d(0, 0, -2.0);
    extrinsics.orientation = Eigen::Quaterniond::Identity();

    Camera camera(0, "cam0", intrinsics, extrinsics);
    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, camera);

    // Create UKF
    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.01);

    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles = Eigen::VectorXd::Zero(0);
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(0);

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(12, 12) * 0.1;
    ukf.set_covariance(cov);

    // Create observation
    Observation obs;
    obs.camera_id = 0;
    obs.marker_id = marker_idx;
    obs.position = Eigen::Vector2d(320.0, 240.0);
    obs.confidence = 1.0;

    std::vector<Observation> observations = {obs};

    // Perform multiple updates
    for (int i = 0; i < 10; ++i) {
        ukf.update(observations, cameras, fk);
    }

    // Check that covariance is still positive definite
    Eigen::MatrixXd final_cov = ukf.covariance();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(final_cov);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();

    // All eigenvalues should be positive
    for (int i = 0; i < eigenvalues.size(); ++i) {
        REQUIRE(eigenvalues(i) > 0.0);
    }
}

TEST_CASE("UKF update handles markers behind camera gracefully", "[ukf][update][robustness]") {
    // Create simple skeleton: root + single marker
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    uint32_t marker_idx = skeleton.add_marker("test_marker", 0, Eigen::Vector3d(0, 0, 0));

    // Build Pinocchio model for FK
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
    auto marker_frame_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
    auto fk_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));

    ForwardKinematics fk(model, data, marker_frame_map, fk_layout);

    // Create camera
    Intrinsics intrinsics;
    intrinsics.fx = 500.0;
    intrinsics.fy = 500.0;
    intrinsics.cx = 320.0;
    intrinsics.cy = 240.0;
    intrinsics.width = 640;
    intrinsics.height = 480;
    intrinsics.model = Intrinsics::DistortionModel::BrownConrady;
    intrinsics.distortion_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};

    Extrinsics extrinsics;
    extrinsics.position = Eigen::Vector3d(0, 0, -2.0);  // Camera at -2m
    extrinsics.orientation = Eigen::Quaterniond::Identity();

    Camera camera(0, "cam0", intrinsics, extrinsics);
    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, camera);

    // Create UKF
    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.01);

    // Place marker BEHIND camera (at z = -3, while camera is at z = -2)
    // This will cause projection to fail (produce NaN/inf)
    Eigen::Vector3d pos = Eigen::Vector3d(0, 0, -3.0);
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles = Eigen::VectorXd::Zero(0);
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(0);

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(12, 12) * 0.1;
    ukf.set_covariance(cov);

    // Create observation (marker projected to center, confidence low)
    Observation obs;
    obs.camera_id = 0;
    obs.marker_id = marker_idx;
    obs.position = Eigen::Vector2d(320.0, 240.0);
    obs.confidence = 0.5;

    std::vector<Observation> observations = {obs};

    // This should NOT crash - the update should handle failed projections gracefully
    REQUIRE_NOTHROW(ukf.update(observations, cameras, fk));

    // State should still be finite (not NaN)
    State const& final_state = ukf.state();
    REQUIRE(std::isfinite(final_state.root_position().x()));
    REQUIRE(std::isfinite(final_state.root_position().y()));
    REQUIRE(std::isfinite(final_state.root_position().z()));

    // Covariance should still be positive definite
    Eigen::MatrixXd final_cov = ukf.covariance();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(final_cov);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();

    for (int i = 0; i < eigenvalues.size(); ++i) {
        REQUIRE(eigenvalues(i) > 0.0);
    }
}

TEST_CASE("UKF update with outlier rejection", "[ukf][update][outlier]") {
    // Create simple skeleton: root + 2 markers
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    uint32_t marker1_idx = skeleton.add_marker("marker1", 0, Eigen::Vector3d(-0.5, 0, 0));
    uint32_t marker2_idx = skeleton.add_marker("marker2", 0, Eigen::Vector3d(0.5, 0, 0));

    // Build Pinocchio model for FK
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
    auto marker_frame_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
    auto fk_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));

    ForwardKinematics fk(model, data, marker_frame_map, fk_layout);

    // Setup camera
    Intrinsics intrinsics;
    intrinsics.fx = 500.0;
    intrinsics.fy = 500.0;
    intrinsics.cx = 320.0;
    intrinsics.cy = 240.0;
    intrinsics.width = 640;
    intrinsics.height = 480;
    intrinsics.model = Intrinsics::DistortionModel::BrownConrady;
    intrinsics.distortion_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};

    Extrinsics extrinsics;
    extrinsics.position = Eigen::Vector3d(0, 0, -2.0);
    extrinsics.orientation = Eigen::Quaterniond::Identity();

    Camera camera(0, "cam0", intrinsics, extrinsics);
    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, camera);

    // Create UKF with state at origin
    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.01);

    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles = Eigen::VectorXd::Zero(0);
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(0);

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(12, 12) * 0.1;
    ukf.set_covariance(cov);

    // Create observations:
    // marker1 at correct position (inlier)
    // marker2 at very wrong position (outlier)
    Observation obs1;
    obs1.camera_id = 0;
    obs1.marker_id = marker1_idx;
    obs1.position =
        Eigen::Vector2d(195.0, 240.0);  // Close to expected (-0.5, 0, 0) projects to ~195
    obs1.confidence = 0.9;

    Observation obs2;
    obs2.camera_id = 0;
    obs2.marker_id = marker2_idx;
    obs2.position =
        Eigen::Vector2d(100.0, 100.0);  // Very far from expected (0.5, 0, 0) projects to ~445, 240
    obs2.confidence = 0.9;

    std::vector<Observation> observations = {obs1, obs2};

    // Update with outlier rejection (chi-squared threshold for 2-DOF at 95% confidence is 5.991)
    double threshold = 4.0;  // Lower threshold to reject the outlier
    // Args: pose_noise_std=0.0, calib_noise_std=5.0, outlier_threshold=threshold
    UpdateResult result = ukf.update(observations, cameras, fk, 0.0, 5.0, threshold);

    // Check that outlier was detected
    REQUIRE(result.num_observations == 2);
    REQUIRE(result.num_outliers == 1);
    REQUIRE(result.num_inliers == 1);
    REQUIRE(result.observations.size() == 2);

    // Marker1 should be inlier, marker2 should be outlier
    bool marker1_is_inlier = false;
    bool marker2_is_outlier = false;

    for (auto const& obs_result : result.observations) {
        if (obs_result.marker_name == "marker1" && !obs_result.is_outlier) {
            marker1_is_inlier = true;
            REQUIRE(obs_result.mahalanobis_distance < threshold);
        }
        if (obs_result.marker_name == "marker2" && obs_result.is_outlier) {
            marker2_is_outlier = true;
            REQUIRE(obs_result.mahalanobis_distance > threshold);
        }
    }

    REQUIRE(marker1_is_inlier);
    REQUIRE(marker2_is_outlier);

    // State should still be finite
    State const& final_state = ukf.state();
    REQUIRE(std::isfinite(final_state.root_position().x()));
    REQUIRE(std::isfinite(final_state.root_position().y()));
    REQUIRE(std::isfinite(final_state.root_position().z()));
}

TEST_CASE("UKF velocity damping at joint limits", "[ukf][update][damping]") {
    // Create skeleton with revolute joint that has limits
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    uint32_t elbow_idx =
        skeleton.add_joint("elbow", 0, JointType::REVOLUTE, Eigen::Vector3d(0.3, 0, 0));
    uint32_t marker_idx = skeleton.add_marker("elbow_marker", elbow_idx, Eigen::Vector3d::Zero());

    // Set joint limits for elbow (0 to PI radians)
    auto& joints = const_cast<std::vector<Joint>&>(skeleton.joints());
    joints[elbow_idx].num_limits = 1;
    joints[elbow_idx].limits[0] = Eigen::Vector2d(0.0, M_PI);

    // Build Pinocchio model for FK
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
    auto marker_frame_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
    auto fk_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));

    ForwardKinematics fk(model, data, marker_frame_map, fk_layout);

    // Setup camera
    Intrinsics intrinsics;
    intrinsics.fx = 500.0;
    intrinsics.fy = 500.0;
    intrinsics.cx = 320.0;
    intrinsics.cy = 240.0;
    intrinsics.width = 640;
    intrinsics.height = 480;
    intrinsics.model = Intrinsics::DistortionModel::BrownConrady;
    intrinsics.distortion_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};

    Extrinsics extrinsics;
    extrinsics.position = Eigen::Vector3d(0, 0, -2.0);
    extrinsics.orientation = Eigen::Quaterniond::Identity();

    Camera camera(0, "cam0", intrinsics, extrinsics);
    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, camera);

    // Create UKF with joint angle near upper limit and positive velocity
    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.01);

    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles(1);
    angles(0) = M_PI - 0.01;  // Very close to upper limit (PI)
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels(1);
    joint_vels(0) = 0.2;  // Positive velocity would push beyond limit

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    // First, do a predict step to push the joint beyond its limit
    ukf.predict(0.1);  // This will enforce limits and zero velocity

    // Set covariance with significant velocity uncertainty
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(14, 14) * 0.01;
    cov(13, 13) = 1.0;  // Large uncertainty in joint velocity (error_dim=14, vel starts at 7, joint
                        // vel at 13)
    ukf.set_covariance(cov);

    // Store velocity covariance before update
    double vel_cov_before = ukf.covariance()(13, 13);

    // Manually set joint to be AT the limit with some velocity
    // This simulates an update that would try to push through the limit
    State test_state = ukf.state();
    Eigen::VectorXd test_angles(1);
    test_angles(0) = M_PI;  // Exactly at limit
    test_state.set_joint_angles(test_angles);
    Eigen::VectorXd test_vels(1);
    test_vels(0) = 0.5;  // Non-zero velocity
    test_state.set_joint_velocities(test_vels);
    ukf.set_state(test_state);

    // Create observation
    Observation obs;
    obs.camera_id = 0;
    obs.marker_id = marker_idx;
    obs.position = Eigen::Vector2d(320.0, 240.0);
    obs.confidence = 0.9;

    std::vector<Observation> observations = {obs};

    // Update (this will enforce limits if the correction pushes beyond)
    ukf.update(observations, cameras, fk);

    // Check the joint state after update
    State const& final_state = ukf.state();

    // Debug output
    // The joint should still be at or clamped to the limit
    REQUIRE(final_state.joint_angles()(0) <= M_PI);

    // If the joint was clamped (stayed at limit), velocity should be zeroed
    bool at_limit = std::abs(final_state.joint_angles()(0) - M_PI) < 1e-6;

    // Check that velocity covariance was damped if velocity was zeroed
    double vel_cov_after = ukf.covariance()(13, 13);

    if (at_limit && std::abs(final_state.joint_velocities()(0)) < 1e-6) {
        // Velocity was zeroed, covariance should be damped
        REQUIRE(vel_cov_after < vel_cov_before);
        REQUIRE(vel_cov_after < 0.1);
    }

    // State should still be finite
    REQUIRE(std::isfinite(final_state.joint_angles()(0)));
    REQUIRE(std::isfinite(final_state.joint_velocities()(0)));
}

// ===========================================================================
// Phase 3e: Child-filter fixed-root injection
// ===========================================================================

/// Build a minimal skeleton for child-filter tests:
///   pelvis (SPHERICAL, root, group="main")
///     └── wrist.R (FIXED, group="main")    ← freeflyer anchor for HandR
///           └── palm.R  (SPHERICAL, group="HandR")
///                └── finger1.R (REVOLUTE, group="HandR")
static Skeleton make_child_filter_skeleton() {
    Skeleton skel;
    uint32_t pelvis = skel.add_joint("pelvis", std::nullopt, JointType::SPHERICAL,
                                     Eigen::Vector3d::Zero(), "main");
    uint32_t wrist =
        skel.add_joint("wrist.R", pelvis, JointType::FIXED, Eigen::Vector3d(0, 0, 1.0), "main");
    uint32_t palm =
        skel.add_joint("palm.R", wrist, JointType::SPHERICAL, Eigen::Vector3d(0.05, 0, 0), "HandR");
    skel.add_joint("finger1.R", palm, JointType::REVOLUTE, Eigen::Vector3d(0.04, 0, 0), "HandR");
    skel.add_marker("MRK-palm", palm, Eigen::Vector3d(0, 0, 0.01));
    return skel;
}

TEST_CASE("Child UKF: set_root_transform updates state immediately", "[ukf][child_filter]") {
    Skeleton skeleton = make_child_filter_skeleton();
    auto skel_ptr = std::make_shared<const Skeleton>(skeleton);
    auto layout = SkeletonLayout::from_groups(skel_ptr, {"HandR"});
    REQUIRE_FALSE(layout->has_floating_root());

    UnscentedKalmanFilter ukf(layout, 0.1);

    Eigen::Vector3d injected_pos(1.5, 2.5, 3.5);
    Eigen::Quaterniond injected_ori =
        Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()));

    ukf.set_root_transform(injected_pos, injected_ori);

    // Root in state() must match the injected transform immediately
    REQUIRE_THAT(ukf.state().root_position().x(), WithinAbs(injected_pos.x(), 1e-10));
    REQUIRE_THAT(ukf.state().root_position().y(), WithinAbs(injected_pos.y(), 1e-10));
    REQUIRE_THAT(ukf.state().root_position().z(), WithinAbs(injected_pos.z(), 1e-10));
    REQUIRE_THAT(ukf.state().root_orientation().w(),
                 WithinAbs(injected_ori.normalized().w(), 1e-10));
    REQUIRE_THAT(ukf.state().root_orientation().x(),
                 WithinAbs(injected_ori.normalized().x(), 1e-10));
}

TEST_CASE("Child UKF: predict keeps root fixed despite large root velocity",
          "[ukf][child_filter]") {
    Skeleton skeleton = make_child_filter_skeleton();
    auto skel_ptr = std::make_shared<const Skeleton>(skeleton);
    auto layout = SkeletonLayout::from_groups(skel_ptr, {"HandR"});
    REQUIRE_FALSE(layout->has_floating_root());

    UnscentedKalmanFilter ukf(layout, 0.1);

    Eigen::Vector3d injected_pos(1.0, 2.0, 3.0);
    Eigen::Quaterniond injected_ori =
        Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()));
    ukf.set_root_transform(injected_pos, injected_ori);

    // Inject a large root velocity into state — process model would drift root significantly.
    State s_with_vel(injected_pos, injected_ori.normalized(), ukf.state().joint_angles(),
                     Eigen::Vector3d(100.0, 100.0, 100.0),  // huge root translational velocity
                     Eigen::Vector3d(50.0, 50.0, 50.0),     // huge root angular velocity
                     ukf.state().joint_velocities());
    ukf.set_state(s_with_vel);

    // After predict, root must still be the injected transform (not drifted).
    double dt = 0.033;
    ukf.predict(dt);

    REQUIRE_THAT(ukf.state().root_position().x(), WithinAbs(injected_pos.x(), 1e-6));
    REQUIRE_THAT(ukf.state().root_position().y(), WithinAbs(injected_pos.y(), 1e-6));
    REQUIRE_THAT(ukf.state().root_position().z(), WithinAbs(injected_pos.z(), 1e-6));
    REQUIRE_THAT(ukf.state().root_orientation().w(),
                 WithinAbs(injected_ori.normalized().w(), 1e-6));
}

TEST_CASE("UKF RELATIVE mode: predict_measurements returns child minus parent projection",
          "[ukf][update][relative]") {
    // Skeleton: root (SPHERICAL) → arm (REVOLUTE)
    // marker0 on root at local_pos (-0.3, 0, 0) → world (-0.3, 0, 0) at identity state
    // marker1 on arm at local_pos (0.3, 0, 0) → world (0.3, 0, 0) at identity state
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "main");
    uint32_t arm_idx =
        skeleton.add_joint("arm", 0, JointType::REVOLUTE, Eigen::Vector3d::Zero(), "main");
    skeleton.add_marker("parent_mrk", 0, Eigen::Vector3d(-0.3, 0, 0));      // marker 0
    skeleton.add_marker("child_mrk", arm_idx, Eigen::Vector3d(0.3, 0, 0));  // marker 1

    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
    auto marker_frame_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
    auto fk_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    ForwardKinematics fk(model, data, marker_frame_map, fk_layout);

    Intrinsics intr;
    intr.fx = 500.0;
    intr.fy = 500.0;
    intr.cx = 320.0;
    intr.cy = 240.0;
    intr.width = 640;
    intr.height = 480;
    intr.model = Intrinsics::DistortionModel::BrownConrady;
    intr.distortion_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};
    Extrinsics extr;
    extr.position = Eigen::Vector3d(0, 0, -2.0);
    extr.orientation = Eigen::Quaterniond::Identity();
    Camera camera(0, "cam0", intr, extr);
    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, camera);

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.01);

    // State: root at origin, arm angle=0 → both markers at their local_pos world positions
    Eigen::VectorXd angles = Eigen::VectorXd::Zero(1);  // one REVOLUTE DOF
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), angles,
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::VectorXd::Zero(1));
    ukf.set_state(state);
    ukf.set_covariance(Eigen::MatrixXd::Identity(14, 14) * 0.1);

    // Camera at (0,0,-2) looking +Z. Pin-hole projection:
    //   u = cx + fx * Xc / Zc  where Xc = world_x, Zc = 2 (distance along camera z)
    //   parent_mrk (-0.3, 0, 0): u = 320 + 500*(-0.3)/2 = 245, v = 240
    //   child_mrk  ( 0.3, 0, 0): u = 320 + 500*(0.3)/2  = 395, v = 240
    // RELATIVE observation: child - parent = (395-245, 0) = (150, 0)
    Observation rel_obs;
    rel_obs.camera_id = 0;
    rel_obs.marker_id = 1;      // child_mrk index
    rel_obs.ref_marker_id = 0;  // parent_mrk index
    rel_obs.position = Eigen::Vector2d(150.0, 0.0);
    rel_obs.confidence = 0.9;
    rel_obs.mode = MeasurementMode::RELATIVE;
    rel_obs.noise_std_override = 5.0 * std::sqrt(2.0);

    std::vector<Observation> observations = {rel_obs};
    // No outlier rejection (threshold=0): just check the update runs and produces finite state.
    UpdateResult result = ukf.update(observations, cameras, fk, 0.0, 5.0, 0.0);

    // Innovation is near zero (predicted ≈ observed), so state barely changes.
    REQUIRE(result.num_observations == 1);
    REQUIRE(result.num_inliers == 1);
    REQUIRE(result.num_outliers == 0);
    State const& post = ukf.state();
    REQUIRE(std::isfinite(post.root_position().x()));
    REQUIRE(std::isfinite(post.root_position().y()));
    REQUIRE(std::isfinite(post.root_position().z()));
    REQUIRE_THAT(post.root_position().x(), WithinAbs(0.0, 0.05));
    REQUIRE_THAT(post.root_position().y(), WithinAbs(0.0, 0.05));
}

TEST_CASE("Child UKF: set_root_transform is no-op on parent filter", "[ukf][child_filter]") {
    // Parent layout: full skeleton, has_floating_root == true
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_marker("m", 0, Eigen::Vector3d(0, 0, 0));
    auto skel_ptr = std::make_shared<const Skeleton>(skeleton);
    auto layout = SkeletonLayout::from_full_skeleton(skel_ptr);
    REQUIRE(layout->has_floating_root());

    UnscentedKalmanFilter ukf(layout, 0.1);
    Eigen::Vector3d orig_pos = ukf.state().root_position();

    // Calling set_root_transform on a parent filter must not change state
    ukf.set_root_transform(Eigen::Vector3d(99.0, 99.0, 99.0), Eigen::Quaterniond::Identity());
    REQUIRE_THAT(ukf.state().root_position().x(), WithinAbs(orig_pos.x(), 1e-10));
}

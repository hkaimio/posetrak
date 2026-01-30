/**
 * @file test_ukf_update.cpp
 * @brief Tests for UKF update step (measurement correction)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
    ForwardKinematics fk(model, data, marker_frame_map, skeleton);

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
    UnscentedKalmanFilter ukf(skeleton, 0.01);

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
    ForwardKinematics fk(model, data, marker_frame_map, skeleton);

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
    UnscentedKalmanFilter ukf(skeleton, 0.01);

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
    ForwardKinematics fk(model, data, marker_frame_map, skeleton);

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
    UnscentedKalmanFilter ukf(skeleton, 0.01);

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
    obs1.position = camera.project_undistorted(Eigen::Vector3d(-0.1, 0, 0));
    obs1.confidence = 1.0;

    Observation obs2;
    obs2.camera_id = 0;
    obs2.marker_id = marker2;
    obs2.position = camera.project_undistorted(Eigen::Vector3d(0.1, 0, 0));
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
    ForwardKinematics fk(model, data, marker_frame_map, skeleton);

    std::unordered_map<int, Camera> cameras;

    // Create UKF
    UnscentedKalmanFilter ukf(skeleton, 0.01);

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
    ForwardKinematics fk(model, data, marker_frame_map, skeleton);

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
    UnscentedKalmanFilter ukf(skeleton, 0.01);

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
    ForwardKinematics fk(model, data, marker_frame_map, skeleton);

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
    UnscentedKalmanFilter ukf(skeleton, 0.01);

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
    ForwardKinematics fk(model, data, marker_frame_map, skeleton);

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
    UnscentedKalmanFilter ukf(skeleton, 0.01);

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
    UpdateResult result = ukf.update(observations, cameras, fk, 5.0, threshold);

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
    ForwardKinematics fk(model, data, marker_frame_map, skeleton);

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

    // Create UKF with joint angle near upper limit
    UnscentedKalmanFilter ukf(skeleton, 0.01);

    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles(1);
    angles(0) = M_PI - 0.05;  // Very close to upper limit
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels(1);
    joint_vels(0) = 0.0;

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    // Set covariance with significant velocity uncertainty
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(14, 14) * 0.01;
    cov(10, 10) = 1.0;  // Large uncertainty in joint velocity
    ukf.set_covariance(cov);

    // Store velocity covariance before update
    double vel_cov_before = ukf.covariance()(10, 10);

    // Create observation
    Observation obs;
    obs.camera_id = 0;
    obs.marker_id = marker_idx;
    obs.position = Eigen::Vector2d(320.0, 240.0);
    obs.confidence = 0.9;

    std::vector<Observation> observations = {obs};

    // Update (velocity damping should be applied automatically)
    ukf.update(observations, cameras, fk);

    // Check that velocity covariance was damped
    double vel_cov_after = ukf.covariance()(10, 10);
    REQUIRE(vel_cov_after < vel_cov_before);
    REQUIRE(vel_cov_after < 0.1);  // Should be significantly reduced

    // State should still be finite
    State const& final_state = ukf.state();
    REQUIRE(std::isfinite(final_state.joint_angles()(0)));
    REQUIRE(std::isfinite(final_state.joint_velocities()(0)));
}

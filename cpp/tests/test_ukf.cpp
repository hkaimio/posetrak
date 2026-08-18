// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_ukf.cpp
 * @brief Tests for Unscented Kalman Filter
 */

#include <Eigen/Eigenvalues>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/state.hpp"
#include "posetrak/filters/ukf.hpp"

using namespace posetrak;
using Catch::Matchers::WithinAbs;

TEST_CASE("UKF construction and initialization", "[ukf]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("joint1", 0, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 1));

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout);

    // Check initial state
    REQUIRE(ukf.state().joint_angles().size() == 1);
    REQUIRE(ukf.state().joint_velocities().size() == 1);

    // Check covariance dimension: 2*(6 + 1) = 14
    REQUIRE(ukf.covariance().rows() == 14);
    REQUIRE(ukf.covariance().cols() == 14);
    REQUIRE(ukf.error_dim() == 14);
}

TEST_CASE("UKF set and get state", "[ukf]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("joint1", 0, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 1));

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout);

    // Create a test state
    Eigen::Vector3d pos(1.0, 2.0, 3.0);
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles(1);
    angles << 0.5;
    Eigen::Vector3d vel(0.1, 0.2, 0.3);
    Eigen::Vector3d angvel(0.01, 0.02, 0.03);
    Eigen::VectorXd joint_vels(1);
    joint_vels << 0.1;

    State test_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(test_state);

    // Verify state was set
    REQUIRE(ukf.state().root_position().isApprox(pos));
    REQUIRE(ukf.state().joint_angles().isApprox(angles));
    REQUIRE(ukf.state().root_velocity().isApprox(vel));
}

TEST_CASE("UKF set covariance with validation", "[ukf]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("joint1", 0, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 1));

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout);

    // Valid covariance
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(14, 14) * 0.5;
    REQUIRE_NOTHROW(ukf.set_covariance(cov));
    REQUIRE(ukf.covariance().isApprox(cov));

    // Invalid size should throw
    Eigen::MatrixXd bad_cov = Eigen::MatrixXd::Identity(10, 10);
    REQUIRE_THROWS_AS(ukf.set_covariance(bad_cov), std::invalid_argument);
}

TEST_CASE("UKF prediction maintains state structure", "[ukf]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("joint1", 0, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 1));

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout);

    // Set initial state with non-zero values
    Eigen::Vector3d pos(1.0, 2.0, 3.0);
    Eigen::Quaterniond quat(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()));
    Eigen::VectorXd angles(1);
    angles << 0.5;
    Eigen::Vector3d vel(0.1, 0.2, 0.3);
    Eigen::Vector3d angvel(0.01, 0.02, 0.03);
    Eigen::VectorXd joint_vels(1);
    joint_vels << 0.1;

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    // Predict
    double dt = 0.01;
    ukf.predict(dt);

    // Check state dimensions unchanged
    REQUIRE(ukf.state().joint_angles().size() == 1);
    REQUIRE(ukf.state().joint_velocities().size() == 1);

    // Check quaternion is normalized
    REQUIRE_THAT(ukf.state().root_orientation().norm(), WithinAbs(1.0, 1e-9));
}

TEST_CASE("UKF prediction with zero velocity", "[ukf]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("joint1", 0, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 1));

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.01);  // Small process noise

    // Set initial state with zero velocities
    Eigen::Vector3d pos(1.0, 2.0, 3.0);
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles(1);
    angles << 0.5;
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels(1);
    joint_vels << 0.0;

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    // Predict with small timestep
    double dt = 0.01;
    ukf.predict(dt);

    // With zero velocity and small process noise, state should stay close
    REQUIRE((ukf.state().root_position() - pos).norm() < 0.1);
    REQUIRE((ukf.state().joint_angles() - angles).norm() < 0.1);
}

TEST_CASE("UKF prediction with constant velocity", "[ukf]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("joint1", 0, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 1));

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.001);  // Very small process noise

    // Set initial state with constant velocity
    Eigen::Vector3d pos(1.0, 2.0, 3.0);
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles(1);
    angles << 0.5;
    Eigen::Vector3d vel(0.1, 0.0, 0.0);  // Moving in X direction
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels(1);
    joint_vels << 0.1;  // Joint rotating

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    // Set tight covariance
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(14, 14) * 0.0001;
    ukf.set_covariance(cov);

    // Predict
    double dt = 1.0;
    ukf.predict(dt);

    // Check motion approximately follows constant velocity
    // Expected position: pos + vel * dt = [1.1, 2.0, 3.0]
    Eigen::Vector3d expected_pos = pos + vel * dt;
    REQUIRE((ukf.state().root_position() - expected_pos).norm() < 0.05);

    // Expected joint angle: 0.5 + 0.1 * 1.0 = 0.6
    double expected_angle = 0.5 + 0.1 * dt;
    REQUIRE(std::abs(ukf.state().joint_angles()(0) - expected_angle) < 0.05);
}

TEST_CASE("UKF prediction with rotation", "[ukf]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("joint1", 0, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 1));

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.001);

    // Set initial state with angular velocity
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles(1);
    angles << 0.0;
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel(0.0, 0.0, 0.1);  // Rotating around Z
    Eigen::VectorXd joint_vels(1);
    joint_vels << 0.0;

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    // Set tight covariance
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(14, 14) * 0.0001;
    ukf.set_covariance(cov);

    // Predict
    double dt = 0.1;
    ukf.predict(dt);

    // Check rotation approximately correct
    // Expected rotation: exp(angvel * dt/2) = rotation of 0.01 radians around Z
    Eigen::Quaterniond expected_quat(Eigen::AngleAxisd(angvel.z() * dt, Eigen::Vector3d::UnitZ()));
    double angle_diff = expected_quat.angularDistance(ukf.state().root_orientation());
    REQUIRE(angle_diff < 0.01);  // Within 0.01 radians

    // Quaternion should remain normalized
    REQUIRE_THAT(ukf.state().root_orientation().norm(), WithinAbs(1.0, 1e-9));
}

TEST_CASE("UKF prediction increases covariance", "[ukf]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("joint1", 0, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 1));

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.1);  // Process noise

    // Set tight initial covariance
    Eigen::MatrixXd initial_cov = Eigen::MatrixXd::Identity(14, 14) * 0.01;
    ukf.set_covariance(initial_cov);

    double initial_trace = initial_cov.trace();

    // Predict
    double dt = 0.1;
    ukf.predict(dt);

    // Covariance should increase (trace increases)
    double final_trace = ukf.covariance().trace();
    REQUIRE(final_trace > initial_trace);

    // Covariance should remain symmetric
    REQUIRE(ukf.covariance().isApprox(ukf.covariance().transpose(), 1e-9));

    // Covariance should remain positive semi-definite (all eigenvalues >= 0)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(ukf.covariance());
    REQUIRE(eigensolver.eigenvalues().minCoeff() >= -1e-9);
}

TEST_CASE("UKF prediction with spherical joint", "[ukf]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("shoulder", 0, JointType::SPHERICAL, Eigen::Vector3d(0, 0, 0.1));

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.01);

    // Set initial state
    Eigen::Vector3d pos(1.0, 2.0, 3.0);
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles(3);  // 3 DOF for spherical
    angles << 0.1, 0.2, 0.0;
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels(3);
    joint_vels << 0.0, 0.0, 0.1;

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    // Predict
    double dt = 0.1;
    ukf.predict(dt);

    // Check dimensions maintained
    REQUIRE(ukf.state().joint_angles().size() == 3);
    REQUIRE(ukf.state().joint_velocities().size() == 3);

    // Error dimension: 2*(6 + 3) = 18
    REQUIRE(ukf.error_dim() == 18);
    REQUIRE(ukf.covariance().rows() == 18);
}

TEST_CASE("UKF prediction with locked DOFs", "[ukf]") {
    // Test with locked DOFs - state always stores 3 DOFs for spherical joints
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    uint32_t shoulder =
        skeleton.add_joint("shoulder", 0, JointType::SPHERICAL, Eigen::Vector3d(0, 0, 0.1));

    // Lock X and Y, only Z active
    std::array<Eigen::Vector2d, 3> limits;
    limits[0] = Eigen::Vector2d(0.0, 0.0);     // X locked
    limits[1] = Eigen::Vector2d(0.0, 0.0);     // Y locked
    limits[2] = Eigen::Vector2d(-M_PI, M_PI);  // Z active
    skeleton.set_joint_limits(shoulder, limits, 3);

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.01);

    // Set initial state - now always 3 DOFs for spherical joint
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles(3);  // Always 3 DOFs for spherical
    angles << 0.0, 0.0, 0.5;    // X, Y locked at 0, Z active at 0.5
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels(3);
    joint_vels << 0.0, 0.0, 0.1;  // Only Z has velocity

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    // Error dimension: only 1 active DOF (Z), so 2*(6 + 1) = 14.
    // Storage always holds 3 DOFs for spherical joints, but the UKF error
    // state is compacted to active DOFs only to keep sigma points correct.
    REQUIRE(ukf.error_dim() == 14);

    // Predict
    double dt = 0.1;
    ukf.predict(dt);

    // Check dimensions maintained
    REQUIRE(ukf.state().joint_angles().size() == 3);
    REQUIRE(ukf.state().joint_velocities().size() == 3);

    // Check locked DOFs remain at 0 (UKF enforces limits after prediction)
    REQUIRE_THAT(ukf.state().joint_angles()(0), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(ukf.state().joint_angles()(1), WithinAbs(0.0, 1e-6));
}

TEST_CASE("UKF multiple prediction steps", "[ukf]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("joint1", 0, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 1));

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.01);

    // Set initial state with velocity
    Eigen::Vector3d pos(0.0, 0.0, 0.0);
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles(1);
    angles << 0.0;
    Eigen::Vector3d vel(1.0, 0.0, 0.0);
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels(1);
    joint_vels << 0.0;

    State initial_state(pos, quat, angles, vel, angvel, joint_vels);
    ukf.set_state(initial_state);

    // Set tight covariance
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(14, 14) * 0.0001;
    ukf.set_covariance(cov);

    // Predict 10 steps
    double dt = 0.1;
    for (int i = 0; i < 10; ++i) {
        ukf.predict(dt);
    }

    // Check total motion: roughly 1.0 m/s * 1.0 s = 1.0 m in X
    REQUIRE(std::abs(ukf.state().root_position().x() - 1.0) < 0.1);

    // Quaternion should remain normalized
    REQUIRE_THAT(ukf.state().root_orientation().norm(), WithinAbs(1.0, 1e-9));
}

// -----------------------------------------------------------------------------
// Adaptive process noise (Phase 1 — velocity-driven per-DOF scaling)
// docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md
// -----------------------------------------------------------------------------

namespace {
/// Two-revolute-joint skeleton (plus floating root) for adaptive-noise tests:
/// joint1 and joint2 can be independently set to different velocities.
Skeleton make_two_joint_skeleton() {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("joint1", 0, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 1));
    skeleton.add_joint("joint2", 0, JointType::REVOLUTE, Eigen::Vector3d(1, 0, 0));
    return skeleton;
}

Eigen::VectorXd zero_angles_2() {
    Eigen::VectorXd angles(2);
    angles << 0.0, 0.0;
    return angles;
}
}  // namespace

TEST_CASE("Adaptive process noise disabled by default leaves scale map empty",
          "[ukf][adaptive-noise]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_two_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);

    Eigen::VectorXd joint_vels(2);
    joint_vels << 5.0, 0.0;  // joint1 moving fast, joint2 still
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), zero_angles_2(),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), joint_vels);
    ukf.set_state(state);

    ukf.predict(0.01);

    // No set_velocity_noise_gain() call -- both gains default to 0.0 (disabled).
    REQUIRE(ukf.last_velocity_noise_scale().empty());
}

TEST_CASE("Adaptive process noise disabled reproduces the exact static baseline",
          "[ukf][adaptive-noise]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_two_joint_skeleton()));
    UnscentedKalmanFilter ukf_still(layout, 0.1);
    UnscentedKalmanFilter ukf_moving(layout, 0.1);

    Eigen::MatrixXd tiny_cov =
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 1e-12;
    ukf_still.set_covariance(tiny_cov);
    ukf_moving.set_covariance(tiny_cov);

    State still_state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), zero_angles_2(),
                      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), zero_angles_2());
    Eigen::VectorXd fast_vels(2);
    fast_vels << 50.0, 50.0;
    State moving_state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), zero_angles_2(),
                       Eigen::Vector3d(10.0, 10.0, 10.0), Eigen::Vector3d(10.0, 10.0, 10.0),
                       fast_vels);
    ukf_still.set_state(still_state);
    ukf_moving.set_state(moving_state);

    ukf_still.predict(0.05);
    ukf_moving.predict(0.05);

    // With both gains at their 0.0 default, process noise must not depend on velocity at
    // all -- the resulting covariance (dominated entirely by process noise, since the
    // pre-process-noise covariance starts near machine-zero) must match regardless of
    // how fast the state was moving.
    REQUIRE(ukf_still.covariance().isApprox(ukf_moving.covariance(), 1e-9));
}

TEST_CASE("Adaptive process noise scales a moving joint DOF more than a stationary one",
          "[ukf][adaptive-noise]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_two_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_velocity_noise_gain(/*gain_joint=*/1.0, /*vel_ref_joint=*/1.0, /*gain_root=*/0.0,
                                /*vel_ref_root=*/1.0);

    Eigen::VectorXd joint_vels(2);
    // joint1 moving at 1x the reference velocity, joint2 still. Kept below the
    // sqrt(kMaxVelocityNoiseMultiplier) =~ 3.162 clamp so this test isolates the
    // unclamped scaling formula (see the separate clamping test below for that case).
    joint_vels << 1.0, 0.0;
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), zero_angles_2(),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), joint_vels);
    ukf.set_state(state);

    ukf.predict(0.01);

    auto const& scale = ukf.last_velocity_noise_scale();
    int const root_n = layout->root_error_dof_count();
    REQUIRE(root_n == 6);
    // joint1 is the first joint after the root block (error_index 0), joint2 next (error_index 1).
    int const joint1_idx = root_n + 0;
    int const joint2_idx = root_n + 1;

    REQUIRE(scale.count(joint1_idx) == 1);
    REQUIRE(scale.count(joint2_idx) == 1);
    // joint1: std_mult = 1 + 1.0 * 1.0 / 1.0 = 2.0
    REQUIRE_THAT(scale.at(joint1_idx), WithinAbs(2.0, 1e-6));
    // joint2: zero velocity -> no scaling
    REQUIRE_THAT(scale.at(joint2_idx), WithinAbs(1.0, 1e-6));
    REQUIRE(scale.at(joint1_idx) > scale.at(joint2_idx));
}

TEST_CASE("Adaptive process noise extra scope applies its own independent gain",
          "[ukf][adaptive-noise]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_two_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    // Primary scope covers joint1 with gain 1.0; an extra scope covers joint2 with a
    // different, independent gain 1.5.
    ukf.set_velocity_noise_gain(/*gain_joint=*/1.0, /*vel_ref_joint=*/1.0, /*gain_root=*/0.0,
                                /*vel_ref_root=*/1.0, {"joint1"});
    // Kept below the sqrt(kMaxVelocityNoiseMultiplier) =~ 3.162 clamp so this test
    // isolates the unclamped scaling formula (see the separate clamping test above).
    ukf.set_velocity_noise_gain_scopes({{"scope2", {"joint2"}, /*gain=*/1.5, /*vel_ref=*/1.0}});

    Eigen::VectorXd joint_vels(2);
    joint_vels << 1.0, 1.0;  // both joints moving at the same speed
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), zero_angles_2(),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), joint_vels);
    ukf.set_state(state);

    ukf.predict(0.01);

    auto const& scale = ukf.last_velocity_noise_scale();
    int const root_n = layout->root_error_dof_count();
    int const joint1_idx = root_n + 0;
    int const joint2_idx = root_n + 1;

    // joint1 (primary scope): std_mult = 1 + 1.0 * 1.0 / 1.0 = 2.0
    REQUIRE_THAT(scale.at(joint1_idx), WithinAbs(2.0, 1e-6));
    // joint2 (extra scope, independent gain): std_mult = 1 + 1.5 * 1.0 / 1.0 = 2.5
    REQUIRE_THAT(scale.at(joint2_idx), WithinAbs(2.5, 1e-6));
}

TEST_CASE(
    "Adaptive process noise extra scopes disabled by default leave scale map "
    "unaffected for non-primary joints",
    "[ukf][adaptive-noise]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_two_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_velocity_noise_gain(/*gain_joint=*/1.0, /*vel_ref_joint=*/1.0, /*gain_root=*/0.0,
                                /*vel_ref_root=*/1.0, {"joint1"});
    // No set_velocity_noise_gain_scopes() call -- disabled by default.

    Eigen::VectorXd joint_vels(2);
    joint_vels << 1.0, 1.0;
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), zero_angles_2(),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), joint_vels);
    ukf.set_state(state);

    ukf.predict(0.01);

    auto const& scale = ukf.last_velocity_noise_scale();
    int const root_n = layout->root_error_dof_count();
    int const joint2_idx = root_n + 1;
    // joint2 outside the primary scope and no extra scopes configured -- unscaled.
    REQUIRE(scale.count(joint2_idx) == 0);
}

TEST_CASE("Adaptive process noise supports more than one extra scope, first match wins",
          "[ukf][adaptive-noise]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("joint1", 0, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 1));
    skeleton.add_joint("joint2", 0, JointType::REVOLUTE, Eigen::Vector3d(1, 0, 0));
    skeleton.add_joint("joint3", 0, JointType::REVOLUTE, Eigen::Vector3d(0, 1, 0));
    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    UnscentedKalmanFilter ukf(layout, 0.1);

    // Primary scope covers joint1. Two extra scopes with different gains; joint3 is
    // deliberately listed in both -- the first one in the list ("proximal") should win.
    ukf.set_velocity_noise_gain(/*gain_joint=*/1.0, /*vel_ref_joint=*/1.0, /*gain_root=*/0.0,
                                /*vel_ref_root=*/1.0, {"joint1"});
    ukf.set_velocity_noise_gain_scopes({
        {"proximal", {"joint2", "joint3"}, /*gain=*/1.5, /*vel_ref=*/1.0},
        {"distal", {"joint3"}, /*gain=*/2.0, /*vel_ref=*/1.0},
    });

    Eigen::VectorXd joint_vels(3);
    joint_vels << 1.0, 1.0, 1.0;
    Eigen::VectorXd angles(3);
    angles << 0.0, 0.0, 0.0;
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), angles,
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), joint_vels);
    ukf.set_state(state);

    ukf.predict(0.01);

    auto const& scale = ukf.last_velocity_noise_scale();
    int const root_n = layout->root_error_dof_count();
    int const joint1_idx = root_n + 0;
    int const joint2_idx = root_n + 1;
    int const joint3_idx = root_n + 2;

    REQUIRE_THAT(scale.at(joint1_idx), WithinAbs(2.0, 1e-6));  // primary: 1 + 1.0
    REQUIRE_THAT(scale.at(joint2_idx), WithinAbs(2.5, 1e-6));  // "proximal": 1 + 1.5
    // joint3 is listed in both extra scopes -- "proximal" (listed first) wins, not
    // "distal"'s 1 + 2.0 = 3.0.
    REQUIRE_THAT(scale.at(joint3_idx), WithinAbs(2.5, 1e-6));
}

TEST_CASE("Adaptive process noise scales root DOFs from root velocity, independently per axis",
          "[ukf][adaptive-noise]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_two_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_velocity_noise_gain(/*gain_joint=*/0.0, /*vel_ref_joint=*/1.0, /*gain_root=*/1.0,
                                /*vel_ref_root=*/2.0);

    // Root moving only along X; Y and Z (and all angular) stationary.
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), zero_angles_2(),
                Eigen::Vector3d(2.0, 0.0, 0.0), Eigen::Vector3d::Zero(), zero_angles_2());
    ukf.set_state(state);

    ukf.predict(0.01);

    auto const& scale = ukf.last_velocity_noise_scale();
    // Root error-state layout: 0-2 = position (x,y,z), 3-5 = orientation.
    REQUIRE(scale.count(0) == 1);
    REQUIRE(scale.count(1) == 1);
    // root_pos_x: std_mult = 1 + 1.0 * 2.0 / 2.0 = 2.0
    REQUIRE_THAT(scale.at(0), WithinAbs(2.0, 1e-6));
    // root_pos_y: zero velocity -> no scaling
    REQUIRE_THAT(scale.at(1), WithinAbs(1.0, 1e-6));
}

TEST_CASE("Adaptive process noise multiplier is clamped for an extreme velocity",
          "[ukf][adaptive-noise]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_two_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_velocity_noise_gain(/*gain_joint=*/1.0, /*vel_ref_joint=*/1.0, /*gain_root=*/0.0,
                                /*vel_ref_root=*/1.0);

    Eigen::VectorXd joint_vels(2);
    joint_vels << 1.0e6, 0.0;  // absurdly large -- must not blow up unbounded
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), zero_angles_2(),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), joint_vels);
    ukf.set_state(state);

    ukf.predict(0.01);

    int const root_n = layout->root_error_dof_count();
    double const joint1_scale = ukf.last_velocity_noise_scale().at(root_n + 0);
    // Variance-domain clamp is documented as 10x -> std-domain clamp is sqrt(10) =~ 3.1623.
    REQUIRE(joint1_scale < 4.0);
    REQUIRE(joint1_scale > 3.0);  // confirms it actually hit the clamp, not some tiny value
}

// -----------------------------------------------------------------------------
// Pose regularization (kinematic redundancy)
// docs/roadmap/features/pose-regularization/pose-regularization-design.md
// -----------------------------------------------------------------------------

namespace {
/// Two-spherical-joint skeleton (spine1 -> spine2 chain, plus floating root),
/// matching spine1/spine2's shape: both fully free (all 3 axes active, no
/// limits configured).
Skeleton make_two_spherical_joint_skeleton() {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("spine1", 0, JointType::SPHERICAL, Eigen::Vector3d(0, 0.1, 0));
    skeleton.add_joint("spine2", 1, JointType::SPHERICAL, Eigen::Vector3d(0, 0.1, 0));
    return skeleton;
}

/// angles = [spine1_x, spine1_y, spine1_z, spine2_x, spine2_y, spine2_z].
State make_pose_reg_test_state(std::shared_ptr<const SkeletonLayout> const& layout,
                               double spine1_x) {
    Eigen::VectorXd angles(6);
    angles << spine1_x, 0.0, 0.0, 0.0, 0.0, 0.0;
    Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(6);
    return State(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), angles,
                 Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), joint_vels);
}
}  // namespace

TEST_CASE("Pose regularization disabled by default is a no-op", "[ukf][pose-regularization]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_two_spherical_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);

    ukf.set_state(make_pose_reg_test_state(layout, 0.3));
    ukf.set_covariance(
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 0.01);

    // No set_pose_regularization() call -- disabled by default.
    ukf.apply_pose_regularization_for_testing();

    REQUIRE_THAT(ukf.state().joint_angles()(0), WithinAbs(0.3, 1e-12));
    REQUIRE_THAT(ukf.state().joint_angles()(3), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Pose regularization equal-split pulls two joints' angles toward each other",
          "[ukf][pose-regularization]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_two_spherical_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_pose_regularization({"spine1", "spine2"}, /*equal_split_noise_std=*/0.05,
                                /*rest_pose_noise_std=*/0.0);

    ukf.set_state(make_pose_reg_test_state(layout, 0.3));  // spine1_x=0.3, spine2_x=0.0
    ukf.set_covariance(
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 0.01);

    ukf.apply_pose_regularization_for_testing();

    double const spine1_x = ukf.state().joint_angles()(0);
    double const spine2_x = ukf.state().joint_angles()(3);
    // Both should move toward each other (spine1 down, spine2 up), narrowing the gap.
    REQUIRE(std::abs(spine1_x - spine2_x) < 0.3);
    REQUIRE(spine1_x < 0.3);
    REQUIRE(spine2_x > 0.0);
    // The y/z axes started equal (0) on both joints -- should stay near 0, no spurious
    // cross-axis coupling from a mechanism that's supposed to be strictly per-axis.
    REQUIRE_THAT(ukf.state().joint_angles()(1), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(ukf.state().joint_angles()(2), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(ukf.state().joint_angles()(4), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(ukf.state().joint_angles()(5), WithinAbs(0.0, 1e-6));
}

TEST_CASE("Pose regularization rest-pose pulls a joint's angle toward zero",
          "[ukf][pose-regularization]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_two_spherical_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_pose_regularization({"spine1"}, /*equal_split_noise_std=*/0.0,
                                /*rest_pose_noise_std=*/0.05);

    ukf.set_state(make_pose_reg_test_state(layout, 0.3));
    ukf.set_covariance(
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 0.01);

    ukf.apply_pose_regularization_for_testing();

    double const spine1_x = ukf.state().joint_angles()(0);
    double const spine2_x = ukf.state().joint_angles()(3);
    REQUIRE(spine1_x < 0.3);                       // pulled toward 0
    REQUIRE(spine1_x > 0.0);                       // but not past it (gentle pull, not a snap)
    REQUIRE_THAT(spine2_x, WithinAbs(0.0, 1e-6));  // spine2 not in the chain -- untouched
}

TEST_CASE("Pose regularization with unresolvable joint names is a no-op",
          "[ukf][pose-regularization]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_two_spherical_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_pose_regularization({"does_not_exist_1", "does_not_exist_2"},
                                /*equal_split_noise_std=*/0.05, /*rest_pose_noise_std=*/0.05);

    ukf.set_state(make_pose_reg_test_state(layout, 0.3));
    ukf.set_covariance(
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 0.01);

    REQUIRE_NOTHROW(ukf.apply_pose_regularization_for_testing());
    REQUIRE_THAT(ukf.state().joint_angles()(0), WithinAbs(0.3, 1e-12));
}

// -----------------------------------------------------------------------------
// Soft joint-limit repulsion
// docs/roadmap/features/soft-joint-limits/soft-joint-limits-design.md
// -----------------------------------------------------------------------------

namespace {
/// Single spherical joint ("arm") with limits [-0.5, 0.5] rad on all three axes,
/// plus floating root -- small, easy-to-reason-about range for testing the soft
/// limit's margin/pull behavior.
Skeleton make_limited_joint_skeleton() {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    uint32_t arm_idx =
        skeleton.add_joint("arm", 0, JointType::SPHERICAL, Eigen::Vector3d(0, 0.1, 0));
    std::array<Eigen::Vector2d, 3> limits;
    limits[0] = Eigen::Vector2d(-0.5, 0.5);
    limits[1] = Eigen::Vector2d(-0.5, 0.5);
    limits[2] = Eigen::Vector2d(-0.5, 0.5);
    skeleton.set_joint_limits(arm_idx, limits, 3);
    return skeleton;
}

/// angles = [arm_x, arm_y, arm_z].
State make_soft_limit_test_state(double arm_x) {
    Eigen::VectorXd angles(3);
    angles << arm_x, 0.0, 0.0;
    Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(3);
    return State(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), angles,
                 Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), joint_vels);
}
}  // namespace

TEST_CASE("Soft joint limits disabled by default is a no-op", "[ukf][soft-joint-limits]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_limited_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);

    ukf.set_state(make_soft_limit_test_state(0.48));  // deep in the margin band, if it were on
    ukf.set_covariance(
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 0.01);

    // No set_soft_joint_limits() call -- disabled by default.
    ukf.apply_soft_joint_limits_for_testing();

    REQUIRE_THAT(ukf.state().joint_angles()(0), WithinAbs(0.48, 1e-12));
}

TEST_CASE("Soft joint limits leave the interior untouched", "[ukf][soft-joint-limits]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_limited_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_soft_joint_limits({"arm"}, /*margin_rad=*/0.1, /*noise_std=*/0.02);

    // Limits are [-0.5, 0.5], margin 0.1 -> interior [-0.4, 0.4]. 0.0 is deep inside.
    ukf.set_state(make_soft_limit_test_state(0.0));
    ukf.set_covariance(
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 0.01);

    ukf.apply_soft_joint_limits_for_testing();

    REQUIRE_THAT(ukf.state().joint_angles()(0), WithinAbs(0.0, 1e-6));
}

TEST_CASE("Soft joint limits pull an angle in the margin band back toward the interior",
          "[ukf][soft-joint-limits]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_limited_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_soft_joint_limits({"arm"}, /*margin_rad=*/0.1, /*noise_std=*/0.02);

    // Limits are [-0.5, 0.5], margin 0.1 -> interior_hi = 0.4. 0.45 is inside the
    // margin band (past interior_hi, short of the hard limit).
    ukf.set_state(make_soft_limit_test_state(0.45));
    ukf.set_covariance(
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 0.01);

    ukf.apply_soft_joint_limits_for_testing();

    double const arm_x = ukf.state().joint_angles()(0);
    REQUIRE(arm_x < 0.45);  // pulled back toward the interior
    REQUIRE(arm_x > 0.35);  // gentle pull, not a snap past interior_hi
    // y/z started at 0 and have no configured pull direction there -- unaffected.
    REQUIRE_THAT(ukf.state().joint_angles()(1), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(ukf.state().joint_angles()(2), WithinAbs(0.0, 1e-6));
}

TEST_CASE("Soft joint limits pull strengthens for an angle further past the hard limit",
          "[ukf][soft-joint-limits]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_limited_joint_skeleton()));

    auto correction_for = [&](double start_angle) {
        UnscentedKalmanFilter ukf(layout, 0.1);
        ukf.set_soft_joint_limits({"arm"}, /*margin_rad=*/0.1, /*noise_std=*/0.02);
        ukf.set_state(make_soft_limit_test_state(start_angle));
        ukf.set_covariance(
            Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 0.01);
        ukf.apply_soft_joint_limits_for_testing();
        return start_angle - ukf.state().joint_angles()(0);  // how far it got pulled back
    };

    // Hard limit is 0.5. 0.55 is just past it; 0.9 is well past it. Both should get
    // pulled back toward the interior, and the more-overshot one should get pulled
    // back further -- the residual (and thus the pull) doesn't saturate at the wall.
    double const correction_near = correction_for(0.55);
    double const correction_far = correction_for(0.9);
    REQUIRE(correction_near > 0.0);
    REQUIRE(correction_far > correction_near);
}

TEST_CASE("Soft joint limits with unresolvable joint names is a no-op",
          "[ukf][soft-joint-limits]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_limited_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_soft_joint_limits({"does_not_exist"}, /*margin_rad=*/0.1, /*noise_std=*/0.02);

    ukf.set_state(make_soft_limit_test_state(0.6));
    ukf.set_covariance(
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 0.01);

    REQUIRE_NOTHROW(ukf.apply_soft_joint_limits_for_testing());
    REQUIRE_THAT(ukf.state().joint_angles()(0), WithinAbs(0.6, 1e-12));
}

// -----------------------------------------------------------------------------
// Near-limit process-noise damping
// docs/roadmap/features/tracking-crisis-debugging-log.md, "Proposals"
// -----------------------------------------------------------------------------

TEST_CASE("Near-limit damping disabled by default is a no-op", "[ukf][near-limit-damping]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_limited_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_state(make_soft_limit_test_state(0.48));  // near the limit, if damping were on
    ukf.set_covariance(
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 0.01);

    // No set_near_limit_damping() call -- disabled by default.
    Eigen::MatrixXd const input =
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim());
    Eigen::MatrixXd const result = ukf.apply_near_limit_damping_for_testing(input);

    REQUIRE(result.isApprox(input));
}

TEST_CASE("Near-limit damping leaves a deep-interior joint with tight covariance untouched",
          "[ukf][near-limit-damping]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_limited_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_near_limit_damping({"arm"}, /*margin_rad=*/0.1, /*spread_sigma=*/3.0,
                               /*damping_factor=*/0.3);

    // Limits are [-0.5, 0.5]. 0.0 is deep interior; tiny covariance means the
    // 3-sigma spread doesn't come close to either bound.
    ukf.set_state(make_soft_limit_test_state(0.0));
    ukf.set_covariance(
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 1e-6);

    Eigen::MatrixXd const input =
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim());
    Eigen::MatrixXd const result = ukf.apply_near_limit_damping_for_testing(input);

    REQUIRE(result.isApprox(input));
}

TEST_CASE("Near-limit damping shrinks process noise when the mean is close to a hard limit",
          "[ukf][near-limit-damping]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_limited_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_near_limit_damping({"arm"}, /*margin_rad=*/0.1, /*spread_sigma=*/3.0,
                               /*damping_factor=*/0.3);

    // Limits are [-0.5, 0.5], margin 0.1 -> detection zone starts at 0.4. 0.45 is
    // inside it even with negligible covariance-implied spread.
    ukf.set_state(make_soft_limit_test_state(0.45));
    ukf.set_covariance(
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 1e-6);

    Eigen::MatrixXd const input =
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim());
    Eigen::MatrixXd const result = ukf.apply_near_limit_damping_for_testing(input);

    REQUIRE(result.trace() < input.trace());
}

TEST_CASE(
    "Near-limit damping shrinks process noise for a deep-interior mean when covariance spread "
    "reaches the limit",
    "[ukf][near-limit-damping]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_limited_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_near_limit_damping({"arm"}, /*margin_rad=*/0.1, /*spread_sigma=*/3.0,
                               /*damping_factor=*/0.3);

    // Mean at 0.0 is deep interior (limit is 0.5, margin-adjusted detection zone
    // starts at 0.4) -- but a large covariance means the 3-sigma spread reaches
    // well past it. This is the key case a mean-only check would miss.
    ukf.set_state(make_soft_limit_test_state(0.0));
    ukf.set_covariance(
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 1.0);

    Eigen::MatrixXd const input =
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim());
    Eigen::MatrixXd const result = ukf.apply_near_limit_damping_for_testing(input);

    REQUIRE(result.trace() < input.trace());
}

TEST_CASE("Near-limit damping with unresolvable joint names is a no-op",
          "[ukf][near-limit-damping]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_limited_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_near_limit_damping({"does_not_exist"}, /*margin_rad=*/0.1, /*spread_sigma=*/3.0,
                               /*damping_factor=*/0.3);
    ukf.set_state(make_soft_limit_test_state(0.49));
    ukf.set_covariance(
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 1.0);

    Eigen::MatrixXd const input =
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim());
    Eigen::MatrixXd const result = ukf.apply_near_limit_damping_for_testing(input);

    REQUIRE(result.isApprox(input));
}

// -----------------------------------------------------------------------------
// NIS-feedback regional fading safety net (Mechanism B)
// docs/roadmap/features/adaptive-process-noise/adaptive-process-noise-design.md
// -----------------------------------------------------------------------------

TEST_CASE("NIS feedback disabled by default leaves multiplier map empty", "[ukf][nis-feedback]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_two_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);

    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), zero_angles_2(),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), zero_angles_2());
    ukf.set_state(state);
    ukf.predict(0.01);

    // No set_nis_feedback_scopes() call -- disabled by default.
    REQUIRE(ukf.last_scope_noise_multipliers().empty());
}

TEST_CASE("NIS feedback disabled reproduces the exact static baseline", "[ukf][nis-feedback]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_two_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);

    Eigen::MatrixXd tiny_cov =
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 1e-12;
    ukf.set_covariance(tiny_cov);
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), zero_angles_2(),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), zero_angles_2());
    ukf.set_state(state);

    ukf.predict(0.05);
    Eigen::MatrixXd const baseline_cov = ukf.covariance();

    // Calling set_scope_noise_multiplier() for a scope name that was never registered
    // via set_nis_feedback_scopes() must be a silent no-op.
    ukf.set_scope_noise_multiplier("nonexistent_scope", 5.0);
    ukf.set_state(state);
    ukf.set_covariance(tiny_cov);
    ukf.predict(0.05);

    REQUIRE(ukf.covariance().isApprox(baseline_cov, 1e-9));
}

TEST_CASE("NIS feedback scope multiplier scales only the joints in that scope",
          "[ukf][nis-feedback]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_two_joint_skeleton()));
    UnscentedKalmanFilter ukf(layout, 0.1);
    ukf.set_nis_feedback_scopes({{"scope1", {"joint1"}}});
    ukf.set_scope_noise_multiplier("scope1", 4.0);

    Eigen::MatrixXd tiny_cov =
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 1e-12;
    ukf.set_covariance(tiny_cov);
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), zero_angles_2(),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), zero_angles_2());
    ukf.set_state(state);

    ukf.predict(0.05);

    int const root_n = layout->root_error_dof_count();
    int const joint1_idx = root_n + 0;
    int const joint2_idx = root_n + 1;

    auto const& mult = ukf.last_scope_noise_multipliers();
    REQUIRE(mult.count("scope1") == 1);
    REQUIRE_THAT(mult.at("scope1"), WithinAbs(4.0, 1e-9));

    // Since covariance started at ~0, the covariance after predict() is dominated
    // entirely by process noise -- joint1's diagonal entry should be ~4x joint2's
    // (joint2 is outside any configured scope, so unaffected).
    double const ratio =
        ukf.covariance()(joint1_idx, joint1_idx) / ukf.covariance()(joint2_idx, joint2_idx);
    REQUIRE_THAT(ratio, WithinAbs(4.0, 1e-6));
}

TEST_CASE("NIS feedback and adaptive process noise compose multiplicatively",
          "[ukf][nis-feedback]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(make_two_joint_skeleton()));
    UnscentedKalmanFilter ukf_a(layout, 0.1);   // Mechanism A only
    UnscentedKalmanFilter ukf_ab(layout, 0.1);  // Mechanism A + B

    ukf_a.set_velocity_noise_gain(/*gain_joint=*/1.0, /*vel_ref_joint=*/1.0, /*gain_root=*/0.0,
                                  /*vel_ref_root=*/1.0);
    ukf_ab.set_velocity_noise_gain(/*gain_joint=*/1.0, /*vel_ref_joint=*/1.0, /*gain_root=*/0.0,
                                   /*vel_ref_root=*/1.0);
    ukf_ab.set_nis_feedback_scopes({{"scope1", {"joint1"}}});
    ukf_ab.set_scope_noise_multiplier("scope1", 3.0);

    Eigen::MatrixXd tiny_cov =
        Eigen::MatrixXd::Identity(layout->error_state_dim(), layout->error_state_dim()) * 1e-12;
    ukf_a.set_covariance(tiny_cov);
    ukf_ab.set_covariance(tiny_cov);

    Eigen::VectorXd joint_vels(2);
    joint_vels << 1.0, 0.0;  // joint1 moving (engages Mechanism A), joint2 still
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), zero_angles_2(),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), joint_vels);
    ukf_a.set_state(state);
    ukf_ab.set_state(state);

    ukf_a.predict(0.05);
    ukf_ab.predict(0.05);

    int const root_n = layout->root_error_dof_count();
    int const joint1_idx = root_n + 0;
    int const joint2_idx = root_n + 1;

    // joint1: Mechanism B's variance-domain 3.0x multiplies on top of whatever
    // Mechanism A already computed from velocity.
    double const ratio_joint1 =
        ukf_ab.covariance()(joint1_idx, joint1_idx) / ukf_a.covariance()(joint1_idx, joint1_idx);
    REQUIRE_THAT(ratio_joint1, WithinAbs(3.0, 1e-6));

    // joint2: outside the scope, and zero velocity means Mechanism A doesn't engage
    // either -- must be identical between the two filters.
    REQUIRE_THAT(ukf_ab.covariance()(joint2_idx, joint2_idx),
                 WithinAbs(ukf_a.covariance()(joint2_idx, joint2_idx), 1e-12));
}

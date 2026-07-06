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

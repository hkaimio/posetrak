/**
 * @file test_sigma_points.cpp
 * @brief Tests for sigma point generation
 */

#include <posetrak/filters/sigma_points.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace posetrak;
using Catch::Matchers::WithinAbs;

TEST_CASE("SigmaPointGenerator construction", "[sigma_points]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("joint1", 0, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 1));

    SigmaPointGenerator gen(skeleton);

    // Error dim should be 2 * (root 6 DOF + 1 joint DOF) = 2 * 7 = 14
    REQUIRE(gen.error_dim() == 14);

    // Check weights sum correctly
    double wm_sum = gen.get_mean_weights().sum();
    double wc_sum = gen.get_covariance_weights().sum();

    REQUIRE_THAT(wm_sum, WithinAbs(1.0, 1e-9));
    // Covariance weights sum to 1 + (1 - alpha^2 + beta)
    // With default alpha=0.001, beta=2.0: wc_sum ≈ 4.0
    double expected_wc_sum = wm_sum + (1.0 - 0.001 * 0.001 + 2.0);
    REQUIRE_THAT(wc_sum, WithinAbs(expected_wc_sum, 1e-6));

    // Should have 2n+1 weights
    REQUIRE(gen.get_mean_weights().size() == 2 * 14 + 1);
    REQUIRE(gen.get_covariance_weights().size() == 2 * 14 + 1);
}

TEST_CASE("Sigma point generation creates correct number of points", "[sigma_points]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("joint1", 0, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 1));

    SigmaPointGenerator gen(skeleton);

    // Create nominal state
    State nominal(1);  // 1 DOF joint

    // Create covariance (14x14 for error state)
    int const n = 14;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(n, n) * 0.01;

    auto sigma_points = gen.generate_sigma_points(nominal, cov);

    // Should have 2n+1 sigma points
    REQUIRE(sigma_points.size() == 2 * n + 1);

    // First sigma point should be close to nominal (zero error)
    REQUIRE(sigma_points[0].root_position().isApprox(nominal.root_position(), 1e-9));
    REQUIRE(sigma_points[0].root_orientation().isApprox(nominal.root_orientation(), 1e-9));
}

TEST_CASE("Sigma points spread around nominal state", "[sigma_points]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("joint1", 0, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 1));

    SigmaPointGenerator gen(skeleton);

    // Create nominal state with non-zero values
    Eigen::Vector3d pos(1.0, 2.0, 3.0);
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles(1);
    angles << 0.5;
    Eigen::Vector3d vel(0.1, 0.2, 0.3);
    Eigen::Vector3d angvel(0.01, 0.02, 0.03);
    Eigen::VectorXd joint_vels(1);
    joint_vels << 0.1;

    State nominal(pos, quat, angles, vel, angvel, joint_vels);

    // Create covariance with different variances
    int const n = 14;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(n, n);
    cov.diagonal().setConstant(0.1);  // Variance of 0.1

    auto sigma_points = gen.generate_sigma_points(nominal, cov);

    // Check that sigma points spread in both directions
    // Positive sigma points (1 to n)
    bool has_positive_spread = false;
    for (size_t i = 1; i <= n; ++i) {
        // Check if any component increased
        if ((sigma_points[i].root_position() - nominal.root_position()).norm() > 1e-6) {
            has_positive_spread = true;
            break;
        }
    }
    REQUIRE(has_positive_spread);

    // Negative sigma points (n+1 to 2n)
    bool has_negative_spread = false;
    for (size_t i = n + 1; i <= 2 * n; ++i) {
        // Check if any component is different from nominal
        if ((sigma_points[i].root_position() - nominal.root_position()).norm() > 1e-6) {
            has_negative_spread = true;
            break;
        }
    }
    REQUIRE(has_negative_spread);
}

TEST_CASE("Sigma points preserve quaternion normalization", "[sigma_points]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());

    SigmaPointGenerator gen(skeleton);

    State nominal(0);  // No joints, just root

    int const n = 12;  // 2 * (3 pos + 3 rot) = 12
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(n, n) * 0.01;

    auto sigma_points = gen.generate_sigma_points(nominal, cov);

    // All sigma points should have normalized quaternions
    for (auto const& sp : sigma_points) {
        REQUIRE_THAT(sp.root_orientation().norm(), WithinAbs(1.0, 1e-9));
    }
}

TEST_CASE("Sigma points handle spherical joints", "[sigma_points]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_joint("shoulder", 0, JointType::SPHERICAL, Eigen::Vector3d(0, 0, 0.1));

    SigmaPointGenerator gen(skeleton);

    // Error dim: 2 * (6 root + 3 shoulder) = 18
    REQUIRE(gen.error_dim() == 18);

    // Create state with spherical joint
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles(3);
    angles << 0.1, 0.2, 0.3;  // Axis-angle for spherical
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels(3);
    joint_vels << 0.01, 0.02, 0.03;

    State nominal(pos, quat, angles, vel, angvel, joint_vels);

    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(18, 18) * 0.01;
    auto sigma_points = gen.generate_sigma_points(nominal, cov);

    REQUIRE(sigma_points.size() == 2 * 18 + 1);

    // First sigma point should match nominal
    REQUIRE(sigma_points[0].joint_angles().isApprox(nominal.joint_angles(), 1e-9));
}

TEST_CASE("Sigma points handle locked DOFs in spherical joints", "[sigma_points]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    uint32_t shoulder_idx =
        skeleton.add_joint("shoulder", 0, JointType::SPHERICAL, Eigen::Vector3d(0, 0, 0.1));

    // Lock X and Y, only Z active
    std::array<Eigen::Vector2d, 3> limits;
    limits[0] = Eigen::Vector2d(0.0, 0.0);     // X locked
    limits[1] = Eigen::Vector2d(0.0, 0.0);     // Y locked
    limits[2] = Eigen::Vector2d(-M_PI, M_PI);  // Z active
    skeleton.set_joint_limits(shoulder_idx, limits, 3);

    SigmaPointGenerator gen(skeleton);

    // Error dim: 2 * (6 root + 1 active DOF) = 14
    // Only Z axis is active (not locked), so 1 active DOF not 3 storage DOFs
    REQUIRE(gen.error_dim() == 14);

    // Create state
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd angles(3);
    angles << 0.0, 0.0, 0.3;  // Only Z should change
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels(3);
    joint_vels << 0.0, 0.0, 0.1;

    State nominal(pos, quat, angles, vel, angvel, joint_vels);

    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(14, 14) * 0.01;
    auto sigma_points = gen.generate_sigma_points(nominal, cov);

    REQUIRE(sigma_points.size() == 2 * 14 + 1);

    // Note: Only 1 active (non-locked) DOF, so error_dim=14 not 18
    // Locked DOFs are not explored in sigma point generation
    //  (no check for locked DOFs here - that's correct behavior)
}

TEST_CASE("Covariance decomposition fallback to eigenvalue", "[sigma_points]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());

    SigmaPointGenerator gen(skeleton);

    State nominal(0);

    // Create nearly singular covariance
    int const n = 12;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(n, n) * 1e-10;

    // Should not throw - will use eigenvalue decomposition
    auto sigma_points = gen.generate_sigma_points(nominal, cov);

    REQUIRE(sigma_points.size() == 2 * n + 1);
}

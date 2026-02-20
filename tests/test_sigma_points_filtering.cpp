/**
 * @file test_sigma_points_filtering.cpp
 * @brief Tests for sigma point generation with active joint filtering
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/state.hpp"
#include "posetrak/filters/sigma_points.hpp"

using namespace posetrak;

TEST_CASE("Sigma points with filtered joints", "[sigma_points][filtering]") {
    // Create skeleton with some joints in group "main" and some in group "extra"
    Skeleton skeleton;

    // Root (always active)
    uint32_t root = skeleton.add_joint("pelvis", std::nullopt, JointType::SPHERICAL,
                                       Eigen::Vector3d::Zero(), "main");

    // Main group joints
    skeleton.add_joint("spine", root, JointType::REVOLUTE, Eigen::Vector3d(0, 0.1, 0), "main");
    skeleton.add_joint("left_hip", root, JointType::SPHERICAL, Eigen::Vector3d(-0.1, 0, 0), "main");

    // Extra group joints (will be filtered out)
    skeleton.add_joint("extra1", root, JointType::REVOLUTE, Eigen::Vector3d(0.1, 0, 0), "extra");
    skeleton.add_joint("extra2", root, JointType::SPHERICAL, Eigen::Vector3d(0, -0.1, 0), "extra");

    SECTION("Without filtering - all joints active") {
        // No filter: root(3) + spine(1) + left_hip(3) + extra1(1) + extra2(3) = 11 DOF
        REQUIRE(skeleton.active_dof() == 11);

        State state(skeleton.total_dof_count());
        state.set_root_position(Eigen::Vector3d(1.0, 2.0, 3.0));
        state.set_root_orientation(Eigen::Quaterniond::Identity());
        state.set_root_velocity(Eigen::Vector3d::Zero());
        state.set_root_angular_velocity(Eigen::Vector3d::Zero());
        state.set_joint_angles(Eigen::VectorXd::Zero(skeleton.total_dof_count()));
        state.set_joint_velocities(Eigen::VectorXd::Zero(skeleton.total_dof_count()));

        auto layout = SkeletonLayout::from_full_skeleton(skeleton);
        SigmaPointGenerator gen(skeleton, layout, 0.1, 2.0, 0.0);
        Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(22, 22) * 0.01;  // 2 * 11 = 22

        // Should succeed with all joints
        auto sigma_states = gen.generate_sigma_points(state, cov);
        REQUIRE(sigma_states.size() == 2 * 22 + 1);  // 2*n + 1
    }

    SECTION("With group filtering - only main group active") {
        skeleton.set_active_groups({"main"});
        // Filtered: root(3) + spine(1) + left_hip(3) = 7 DOF
        REQUIRE(skeleton.active_dof() == 7);

        State state(skeleton.total_dof_count());
        state.set_root_position(Eigen::Vector3d(1.0, 2.0, 3.0));
        state.set_root_orientation(Eigen::Quaterniond::Identity());
        state.set_root_velocity(Eigen::Vector3d::Zero());
        state.set_root_angular_velocity(Eigen::Vector3d::Zero());
        state.set_joint_angles(Eigen::VectorXd::Zero(skeleton.total_dof_count()));
        state.set_joint_velocities(Eigen::VectorXd::Zero(skeleton.total_dof_count()));

        auto layout = SkeletonLayout::from_full_skeleton(skeleton);
        SigmaPointGenerator gen(skeleton, layout, 0.1, 2.0, 0.0);
        Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(14, 14) * 0.01;  // 2 * 7 = 14

        // Should succeed with filtered joints
        auto sigma_states = gen.generate_sigma_points(state, cov);
        REQUIRE(sigma_states.size() == 2 * 14 + 1);  // 2*n + 1

        // Verify that sigma states are valid
        for (auto const& sigma_state : sigma_states) {
            // Root position should be near nominal
            REQUIRE((sigma_state.root_position() - state.root_position()).norm() < 1.0);

            // Joint angles should be valid (same size as skeleton storage)
            REQUIRE(sigma_state.joint_angles().size() == skeleton.total_dof_count());
        }
    }

    SECTION("Error state dimensions match filtered DOFs") {
        skeleton.set_active_groups({"main"});
        REQUIRE(skeleton.active_dof() == 7);

        State state(skeleton.total_dof_count());
        state.set_root_position(Eigen::Vector3d::Zero());
        state.set_root_orientation(Eigen::Quaterniond::Identity());
        state.set_root_velocity(Eigen::Vector3d::Zero());
        state.set_root_angular_velocity(Eigen::Vector3d::Zero());
        state.set_joint_angles(Eigen::VectorXd::Zero(skeleton.total_dof_count()));
        state.set_joint_velocities(Eigen::VectorXd::Zero(skeleton.total_dof_count()));

        auto layout = SkeletonLayout::from_full_skeleton(skeleton);
        SigmaPointGenerator gen(skeleton, layout, 0.1, 2.0, 0.0);

        // Error dimension should be 2 * active_dof = 14
        Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(14, 14) * 0.01;

        // Should not throw
        REQUIRE_NOTHROW(gen.generate_sigma_points(state, cov));
    }

    SECTION("Wrong covariance dimension throws") {
        skeleton.set_active_groups({"main"});
        REQUIRE(skeleton.active_dof() == 7);

        State state(skeleton.total_dof_count());
        state.set_root_position(Eigen::Vector3d::Zero());
        state.set_root_orientation(Eigen::Quaterniond::Identity());
        state.set_root_velocity(Eigen::Vector3d::Zero());
        state.set_root_angular_velocity(Eigen::Vector3d::Zero());
        state.set_joint_angles(Eigen::VectorXd::Zero(skeleton.total_dof_count()));
        state.set_joint_velocities(Eigen::VectorXd::Zero(skeleton.total_dof_count()));

        auto layout = SkeletonLayout::from_full_skeleton(skeleton);
        SigmaPointGenerator gen(skeleton, layout, 0.1, 2.0, 0.0);

        // Wrong dimension - should be 14, but use 22 (all joints)
        Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(22, 22) * 0.01;

        // Should throw due to dimension mismatch
        REQUIRE_THROWS_AS(gen.generate_sigma_points(state, cov), std::invalid_argument);
    }

    SECTION("Filtered joints maintain zero values") {
        skeleton.set_active_groups({"main"});

        State state(skeleton.total_dof_count());
        state.set_root_position(Eigen::Vector3d::Zero());
        state.set_root_orientation(Eigen::Quaterniond::Identity());
        state.set_root_velocity(Eigen::Vector3d::Zero());
        state.set_root_angular_velocity(Eigen::Vector3d::Zero());

        // Set all joint angles to specific non-zero values
        Eigen::VectorXd angles = Eigen::VectorXd::Constant(skeleton.total_dof_count(), 0.5);
        state.set_joint_angles(angles);
        state.set_joint_velocities(Eigen::VectorXd::Zero(skeleton.total_dof_count()));

        auto layout = SkeletonLayout::from_full_skeleton(skeleton);
        SigmaPointGenerator gen(skeleton, layout, 0.1, 2.0, 0.0);
        Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(14, 14) * 0.01;

        auto sigma_states = gen.generate_sigma_points(state, cov);

        // Check that filtered joints (extra1, extra2) maintain their original angles
        // in all sigma states
        auto joints_ordered = skeleton.get_joints_ordered();
        int joint_idx = 0;

        for (auto const& joint : joints_ordered) {
            if (!joint.parent_index.has_value()) {
                continue;  // Skip root
            }

            if (!skeleton.is_joint_active(joint.name)) {
                // Inactive joints should maintain their nominal values in all sigma states
                for (auto const& sigma_state : sigma_states) {
                    if (joint.type == JointType::REVOLUTE) {
                        REQUIRE(sigma_state.joint_angles()(joint_idx) ==
                                Catch::Matchers::WithinAbs(0.5, 1e-10));
                        joint_idx += 1;
                    } else if (joint.type == JointType::SPHERICAL) {
                        REQUIRE(sigma_state.joint_angles()(joint_idx) ==
                                Catch::Matchers::WithinAbs(0.5, 1e-10));
                        REQUIRE(sigma_state.joint_angles()(joint_idx + 1) ==
                                Catch::Matchers::WithinAbs(0.5, 1e-10));
                        REQUIRE(sigma_state.joint_angles()(joint_idx + 2) ==
                                Catch::Matchers::WithinAbs(0.5, 1e-10));
                        joint_idx += 3;
                    }
                }
                // Reset joint_idx for next sigma state iteration
                joint_idx = 0;
                for (auto const& j2 : joints_ordered) {
                    if (!j2.parent_index.has_value())
                        continue;
                    if (j2.name == joint.name)
                        break;
                    if (j2.type == JointType::REVOLUTE)
                        joint_idx += 1;
                    else if (j2.type == JointType::SPHERICAL)
                        joint_idx += 3;
                }
            }
        }
    }
}

TEST_CASE("Sigma point error vector bounds checking", "[sigma_points][bounds]") {
    Skeleton skeleton;

    uint32_t root = skeleton.add_joint("pelvis", std::nullopt, JointType::SPHERICAL,
                                       Eigen::Vector3d::Zero(), "main");
    skeleton.add_joint("joint1", root, JointType::REVOLUTE, Eigen::Vector3d(0, 0.1, 0), "main");

    skeleton.set_active_groups({"main"});

    State state(skeleton.total_dof_count());
    state.set_root_position(Eigen::Vector3d::Zero());
    state.set_root_orientation(Eigen::Quaterniond::Identity());
    state.set_root_velocity(Eigen::Vector3d::Zero());
    state.set_root_angular_velocity(Eigen::Vector3d::Zero());
    state.set_joint_angles(Eigen::VectorXd::Zero(skeleton.total_dof_count()));
    state.set_joint_velocities(Eigen::VectorXd::Zero(skeleton.total_dof_count()));

    // active_dof = 3 (root) + 1 (joint1) = 4
    // error_dim = 2 * 4 = 8
    REQUIRE(skeleton.active_dof() == 4);

    auto layout = SkeletonLayout::from_full_skeleton(skeleton);
    SigmaPointGenerator gen(skeleton, layout, 0.1, 2.0, 0.0);
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(8, 8) * 0.01;

    // Should complete without bounds violations
    REQUIRE_NOTHROW(gen.generate_sigma_points(state, cov));
}

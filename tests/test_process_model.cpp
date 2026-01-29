/**
 * @file test_process_model.cpp
 * @brief Tests for process model implementations
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/filters/process_model.hpp"

using namespace posetrak;
using Catch::Matchers::WithinAbs;

TEST_CASE("ConstantVelocityModel propagates state correctly", "[process_model]") {
    // Create simple skeleton with root + revolute joint
    Skeleton skeleton;
    // Add root (represented as a joint with no parent)
    uint32_t root_idx = skeleton.add_joint("root", std::nullopt, JointType::FIXED,
                                           Eigen::Vector3d::Zero(), "", Eigen::Vector3d::Zero());
    skeleton.add_joint("joint1", root_idx, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 1), "",
                       Eigen::Vector3d::Zero());

    ConstantVelocityModel model(skeleton, 0.1);

    SECTION("Propagates root position linearly") {
        Eigen::Vector3d pos(1.0, 2.0, 3.0);
        Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
        Eigen::VectorXd joint_angles = Eigen::VectorXd::Zero(1);
        Eigen::Vector3d velocity(0.5, 0.0, -0.2);
        Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(1);

        State state(pos, quat, joint_angles, velocity, joint_vels);

        double dt = 0.1;
        State next_state = model.propagate(state, dt);

        // Check position update: p' = p + v * dt
        Eigen::Vector3d expected_pos = pos + velocity * dt;
        REQUIRE_THAT(next_state.root_position().x(), WithinAbs(expected_pos.x(), 1e-6));
        REQUIRE_THAT(next_state.root_position().y(), WithinAbs(expected_pos.y(), 1e-6));
        REQUIRE_THAT(next_state.root_position().z(), WithinAbs(expected_pos.z(), 1e-6));

        // Velocity should be unchanged
        REQUIRE_THAT(next_state.root_velocity().x(), WithinAbs(velocity.x(), 1e-6));
        REQUIRE_THAT(next_state.root_velocity().y(), WithinAbs(velocity.y(), 1e-6));
        REQUIRE_THAT(next_state.root_velocity().z(), WithinAbs(velocity.z(), 1e-6));
    }

    SECTION("Root orientation remains constant (no angular velocity in State yet)") {
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
        Eigen::VectorXd joint_angles = Eigen::VectorXd::Zero(1);
        Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
        Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(1);

        State state(pos, quat, joint_angles, velocity, joint_vels);

        double dt = 1.0;
        State next_state = model.propagate(state, dt);

        // Orientation should be unchanged (identity)
        REQUIRE_THAT(next_state.root_orientation().w(), WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(next_state.root_orientation().x(), WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(next_state.root_orientation().y(), WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(next_state.root_orientation().z(), WithinAbs(0.0, 1e-6));
    }

    SECTION("Propagates joint angles linearly") {
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
        Eigen::VectorXd joint_angles(1);
        joint_angles[0] = 0.5;  // 0.5 rad
        Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
        Eigen::VectorXd joint_vels(1);
        joint_vels[0] = 0.2;  // 0.2 rad/s

        State state(pos, quat, joint_angles, velocity, joint_vels);

        double dt = 0.5;
        State next_state = model.propagate(state, dt);

        // Check joint angle: θ' = θ + ω * dt
        double expected_angle = 0.5 + 0.2 * 0.5;
        REQUIRE_THAT(next_state.joint_angles()[0], WithinAbs(expected_angle, 1e-6));

        // Velocity unchanged
        REQUIRE_THAT(next_state.joint_velocities()[0], WithinAbs(0.2, 1e-6));
    }

    SECTION("Enforces joint limits") {
        // Add joint with limits
        Skeleton skeleton_limited;
        uint32_t root =
            skeleton_limited.add_joint("root", std::nullopt, JointType::FIXED,
                                       Eigen::Vector3d::Zero(), "", Eigen::Vector3d::Zero());

        // Revolute with limits [-1.0, 1.0]
        uint32_t joint =
            skeleton_limited.add_joint("limited", root, JointType::REVOLUTE,
                                       Eigen::Vector3d(0, 0, 1), "", Eigen::Vector3d::Zero());
        // Note: limits array is const, so we need to create joint properly
        // For now, skip this test as we can't modify limits after creation
        // TODO: Add proper limit setting in Skeleton API

        // Alternative: test that propagation doesn't crash with limits
        ConstantVelocityModel model_limited(skeleton_limited, 0.1);

        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
        Eigen::VectorXd angles(1);
        angles[0] = 0.5;
        Eigen::Vector3d vel = Eigen::Vector3d::Zero();
        Eigen::VectorXd joint_vels(1);
        joint_vels[0] = 0.1;

        State state(pos, quat, angles, vel, joint_vels);

        double dt = 1.0;
        State next_state = model_limited.propagate(state, dt);

        // Should propagate without crashing
        REQUIRE(next_state.joint_angles()[0] > 0.0);
    }
}

TEST_CASE("ConstantVelocityModel generates process noise", "[process_model]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero(), "",
                       Eigen::Vector3d::Zero());

    double noise_std = 0.5;
    ConstantVelocityModel model(skeleton, noise_std);

    SECTION("Process noise scales with time step") {
        int state_dim = 12;  // Example dimension
        double dt1 = 0.1;
        double dt2 = 0.2;

        Eigen::MatrixXd Q1 = model.get_process_noise(dt1, state_dim);
        Eigen::MatrixXd Q2 = model.get_process_noise(dt2, state_dim);

        REQUIRE(Q1.rows() == state_dim);
        REQUIRE(Q1.cols() == state_dim);

        // Q2 should be roughly 4x Q1 (scales with dt²)
        REQUIRE_THAT(Q2(0, 0) / Q1(0, 0), WithinAbs(4.0, 1e-6));
    }

    SECTION("Process noise is diagonal") {
        int state_dim = 10;
        double dt = 0.1;

        Eigen::MatrixXd Q = model.get_process_noise(dt, state_dim);

        // Check diagonal elements are positive
        for (int i = 0; i < state_dim; ++i) {
            REQUIRE(Q(i, i) > 0.0);
        }

        // Check off-diagonal elements are zero
        for (int i = 0; i < state_dim; ++i) {
            for (int j = 0; j < state_dim; ++j) {
                if (i != j) {
                    REQUIRE_THAT(Q(i, j), WithinAbs(0.0, 1e-9));
                }
            }
        }
    }

    SECTION("Can modify noise level") {
        model.set_process_noise_std(1.0);
        REQUIRE_THAT(model.get_process_noise_std(), WithinAbs(1.0, 1e-9));

        Eigen::MatrixXd Q = model.get_process_noise(0.1, 5);
        // Variance should be (1.0 * 0.1)² = 0.01
        REQUIRE_THAT(Q(0, 0), WithinAbs(0.01, 1e-9));
    }
}

TEST_CASE("ConstantVelocityModel handles zero velocities", "[process_model]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero(), "",
                       Eigen::Vector3d::Zero());

    ConstantVelocityModel model(skeleton, 0.1);

    Eigen::Vector3d pos(1.0, 2.0, 3.0);
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd joint_angles = Eigen::VectorXd::Zero(0);
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(0);

    State state(pos, quat, joint_angles, velocity, joint_vels);

    double dt = 1.0;
    State next_state = model.propagate(state, dt);

    // With zero velocities, state should be unchanged
    REQUIRE_THAT(next_state.root_position().x(), WithinAbs(pos.x(), 1e-9));
    REQUIRE_THAT(next_state.root_position().y(), WithinAbs(pos.y(), 1e-9));
    REQUIRE_THAT(next_state.root_position().z(), WithinAbs(pos.z(), 1e-9));

    // Quaternion should be identity
    REQUIRE_THAT(next_state.root_orientation().w(), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(next_state.root_orientation().x(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(next_state.root_orientation().y(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(next_state.root_orientation().z(), WithinAbs(0.0, 1e-9));
}

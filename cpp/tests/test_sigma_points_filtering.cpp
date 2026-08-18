// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_sigma_points_filtering.cpp
 * @brief Tests for sigma point generation with layout-based joint filtering.
 *
 * Phase 3c: Filtering is now done via SkeletonLayout::from_groups().
 * Skeleton is immutable; no set_active_groups() call needed.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/core/state.hpp"
#include "posetrak/filters/sigma_points.hpp"

using namespace posetrak;

// Skeleton: root(SPHERICAL,"main") + spine(REVOLUTE,"main") +
//           left_hip(SPHERICAL,"main") + extra1(REVOLUTE,"extra") +
//           extra2(SPHERICAL,"extra")
// Non-root storage DOFs: spine=0, left_hip=1-3, extra1=4, extra2=5-7  (total 8)
static Skeleton make_skeleton() {
    Skeleton s;
    uint32_t root =
        s.add_joint("pelvis", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "main");
    s.add_joint("spine", root, JointType::REVOLUTE, Eigen::Vector3d(0, 0.1, 0), "main");
    s.add_joint("left_hip", root, JointType::SPHERICAL, Eigen::Vector3d(-0.1, 0, 0), "main");
    s.add_joint("extra1", root, JointType::REVOLUTE, Eigen::Vector3d(0.1, 0, 0), "extra");
    s.add_joint("extra2", root, JointType::SPHERICAL, Eigen::Vector3d(0, -0.1, 0), "extra");
    return s;
}

static State make_state(Skeleton const& s) {
    State st(s.total_dof_count());
    st.set_root_position(Eigen::Vector3d(1.0, 2.0, 3.0));
    st.set_root_orientation(Eigen::Quaterniond::Identity());
    st.set_root_velocity(Eigen::Vector3d::Zero());
    st.set_root_angular_velocity(Eigen::Vector3d::Zero());
    st.set_joint_angles(Eigen::VectorXd::Zero(s.total_dof_count()));
    st.set_joint_velocities(Eigen::VectorXd::Zero(s.total_dof_count()));
    return st;
}

TEST_CASE("Sigma points: full layout includes all joints", "[sigma_points][filtering]") {
    auto skel = make_skeleton();
    auto skel_ptr = std::make_shared<const Skeleton>(skel);
    auto layout = SkeletonLayout::from_full_skeleton(skel_ptr);

    // error_dim = 2*(6 + 1 + 3 + 1 + 3) = 28
    REQUIRE(layout->error_state_dim() == 28);

    auto state = make_state(skel);
    SigmaPointGenerator gen(layout, 0.1, 2.0, 0.0);
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(28, 28) * 0.01;

    auto sigma_states = gen.generate_sigma_points(state, cov);
    REQUIRE(sigma_states.size() == 2 * 28 + 1);
}

TEST_CASE("Sigma points: group layout reduces error dimension", "[sigma_points][filtering]") {
    auto skel = make_skeleton();
    auto skel_ptr = std::make_shared<const Skeleton>(skel);
    auto layout = SkeletonLayout::from_groups(skel_ptr, {"main"});

    // error_dim = 2*(6 + 1 + 3) = 20  (root + spine + left_hip, "extra" excluded)
    REQUIRE(layout->error_state_dim() == 20);

    auto state = make_state(skel);
    SigmaPointGenerator gen(layout, 0.1, 2.0, 0.0);
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(20, 20) * 0.01;

    auto sigma_states = gen.generate_sigma_points(state, cov);
    REQUIRE(sigma_states.size() == 2 * 20 + 1);

    // All sigma states should have valid root positions (within 1m of nominal)
    for (auto const& ss : sigma_states) {
        REQUIRE((ss.root_position() - state.root_position()).norm() < 1.0);
        REQUIRE(ss.joint_angles().size() == skel.total_dof_count());
    }
}

TEST_CASE("Sigma points: wrong covariance dimension throws", "[sigma_points][filtering]") {
    auto skel = make_skeleton();
    auto layout = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"main"});
    // error_dim = 20; pass 28 → should throw
    auto state = make_state(skel);
    SigmaPointGenerator gen(layout, 0.1, 2.0, 0.0);
    Eigen::MatrixXd wrong_cov = Eigen::MatrixXd::Identity(28, 28) * 0.01;
    REQUIRE_THROWS_AS(gen.generate_sigma_points(state, wrong_cov), std::invalid_argument);
}

TEST_CASE("Sigma points: joints outside layout are not perturbed", "[sigma_points][filtering]") {
    // Use from_groups({"main"}) — extra1 and extra2 are NOT in layout.
    // Their joint_angles values in every sigma state should equal the nominal.
    auto skel = make_skeleton();
    auto skel_ptr = std::make_shared<const Skeleton>(skel);
    auto layout = SkeletonLayout::from_groups(skel_ptr, {"main"});

    State state = make_state(skel);
    // Set all joint angles to 0.5 (including extra1 and extra2)
    Eigen::VectorXd angles = Eigen::VectorXd::Constant(skel.total_dof_count(), 0.5);
    state.set_joint_angles(angles);

    SigmaPointGenerator gen(layout, 0.1, 2.0, 0.0);
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(20, 20) * 0.01;
    auto sigma_states = gen.generate_sigma_points(state, cov);

    // State indices (non-root only): spine=0, left_hip=1,2,3, extra1=4, extra2=5,6,7
    for (auto const& ss : sigma_states) {
        REQUIRE(ss.joint_angles()(4) == Catch::Approx(0.5));  // extra1
        REQUIRE(ss.joint_angles()(5) == Catch::Approx(0.5));  // extra2[0]
        REQUIRE(ss.joint_angles()(6) == Catch::Approx(0.5));  // extra2[1]
        REQUIRE(ss.joint_angles()(7) == Catch::Approx(0.5));  // extra2[2]
    }
}

TEST_CASE("Sigma point error vector bounds checking", "[sigma_points][bounds]") {
    Skeleton skeleton;
    uint32_t root = skeleton.add_joint("pelvis", std::nullopt, JointType::SPHERICAL,
                                       Eigen::Vector3d::Zero(), "main");
    skeleton.add_joint("joint1", root, JointType::REVOLUTE, Eigen::Vector3d(0, 0.1, 0), "main");
    auto skel_ptr = std::make_shared<const Skeleton>(skeleton);

    // error_dim = 2*(6 + 1) = 14
    auto layout = SkeletonLayout::from_groups(skel_ptr, {"main"});
    REQUIRE(layout->error_state_dim() == 14);

    State state(skeleton.total_dof_count());
    state.set_root_position(Eigen::Vector3d::Zero());
    state.set_root_orientation(Eigen::Quaterniond::Identity());
    state.set_root_velocity(Eigen::Vector3d::Zero());
    state.set_root_angular_velocity(Eigen::Vector3d::Zero());
    state.set_joint_angles(Eigen::VectorXd::Zero(skeleton.total_dof_count()));
    state.set_joint_velocities(Eigen::VectorXd::Zero(skeleton.total_dof_count()));

    SigmaPointGenerator gen(layout, 0.1, 2.0, 0.0);
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(14, 14) * 0.01;

    REQUIRE_NOTHROW(gen.generate_sigma_points(state, cov));
}

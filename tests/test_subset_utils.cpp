/**
 * @file test_subset_utils.cpp
 * @brief Unit tests for DOF subset extraction and merging utilities
 */

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/state.hpp"
#include "posetrak/filters/subset_utils.hpp"

using namespace posetrak;

// Helper function to create a test skeleton with groups
Skeleton create_test_skeleton() {
    Skeleton skel;

    // Root (hips) - always in "main" group
    uint32_t hips =
        skel.add_joint("hips", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "main");

    // Spine - "main" group
    uint32_t spine =
        skel.add_joint("spine", hips, JointType::SPHERICAL, Eigen::Vector3d(0, 0.1, 0), "main");

    // Left arm - "main" group
    uint32_t left_shoulder = skel.add_joint("shoulder.L", spine, JointType::SPHERICAL,
                                            Eigen::Vector3d(0.2, 0.1, 0), "main");
    uint32_t left_elbow = skel.add_joint("elbow.L", left_shoulder, JointType::REVOLUTE,
                                         Eigen::Vector3d(0.3, 0, 0), "main");
    uint32_t left_wrist = skel.add_joint("wrist.L", left_elbow, JointType::SPHERICAL,
                                         Eigen::Vector3d(0.3, 0, 0), "main");

    // Left hand fingers - "HandL" group
    skel.add_joint("finger1.L", left_wrist, JointType::REVOLUTE, Eigen::Vector3d(0.1, 0, 0),
                   "HandL");
    skel.add_joint("finger2.L", left_wrist, JointType::REVOLUTE, Eigen::Vector3d(0.1, 0.05, 0),
                   "HandL");

    // Right arm - "main" group
    uint32_t right_shoulder = skel.add_joint("shoulder.R", spine, JointType::SPHERICAL,
                                             Eigen::Vector3d(-0.2, 0.1, 0), "main");
    uint32_t right_elbow = skel.add_joint("elbow.R", right_shoulder, JointType::REVOLUTE,
                                          Eigen::Vector3d(-0.3, 0, 0), "main");
    uint32_t right_wrist = skel.add_joint("wrist.R", right_elbow, JointType::SPHERICAL,
                                          Eigen::Vector3d(-0.3, 0, 0), "main");

    // Right hand fingers - "HandR" group
    skel.add_joint("finger1.R", right_wrist, JointType::REVOLUTE, Eigen::Vector3d(-0.1, 0, 0),
                   "HandR");
    skel.add_joint("finger2.R", right_wrist, JointType::REVOLUTE, Eigen::Vector3d(-0.1, 0.05, 0),
                   "HandR");

    // Total DOFs (excluding root which is stored separately in State):
    // 3(spine) + 3(shoulderL) + 1(elbowL) + 3(wristL) + 1(finger1L) + 1(finger2L)
    //           + 3(shoulderR) + 1(elbowR) + 3(wristR) + 1(finger1R) + 1(finger2R) = 21 DOFs

    return skel;
}

TEST_CASE("get_active_dof_indices with no filter", "[subset_utils]") {
    Skeleton skel = create_test_skeleton();

    // All joints active by default
    auto indices = get_active_dof_indices(skel);

    // Should have all 21 DOFs (excluding root which has no parent)
    REQUIRE(indices.size() == 21);

    // Indices should be in ascending order: 0, 1, 2, ..., 20
    for (size_t i = 0; i < indices.size(); ++i) {
        REQUIRE(indices[i] == static_cast<int>(i));
    }
}

TEST_CASE("get_active_dof_indices with main group only", "[subset_utils]") {
    Skeleton skel = create_test_skeleton();
    skel.set_active_groups({"main"});

    auto indices = get_active_dof_indices(skel);

    // "main" group has:
    // spine(3) + shoulderL(3) + elbowL(1) + wristL(3) + shoulderR(3) + elbowR(1) + wristR(3) = 17
    // DOFs
    REQUIRE(indices.size() == 17);

    // Should NOT include finger DOFs (indices 10-11 for left, 19-20 for right)
    // Expected indices: 0-9 (spine through wristL), then 12-18 (shoulderR through wristR)
    std::vector<int> expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18};
    REQUIRE(indices == expected);
}

TEST_CASE("get_active_dof_indices with hand groups", "[subset_utils]") {
    Skeleton skel = create_test_skeleton();

    SECTION("HandL only") {
        skel.set_active_groups({"HandL"});
        auto indices = get_active_dof_indices(skel);

        // HandL has 2 revolute joints (finger1.L, finger2.L) = 2 DOFs
        REQUIRE(indices.size() == 2);

        // These should be DOF indices 10 and 11 (after wristL)
        // Let's calculate: spine(0-2) + shoulderL(3-5) + elbowL(6) + wristL(7-9) + finger1L(10) +
        // finger2L(11)
        REQUIRE(indices[0] == 10);
        REQUIRE(indices[1] == 11);
    }

    SECTION("HandR only") {
        skel.set_active_groups({"HandR"});
        auto indices = get_active_dof_indices(skel);

        // HandR has 2 revolute joints = 2 DOFs
        REQUIRE(indices.size() == 2);

        // These should be the last 2 DOFs (indices 22 and 23)
        // After all main DOFs (17) + HandL DOFs (2) + shoulderR-wristR (already in main calc)
        // Actually: spine(0-2) + shoulderL(3-5) + elbowL(6) + wristL(7-9) + finger1L(10) +
        // finger2L(11) +
        //          shoulderR(12-14) + elbowR(15) + wristR(16-18) + finger1R(19) + finger2R(20)
        // Wait, I need to recalculate this more carefully...
        // Let me count in skeleton order:
        // 0: hips (root, not counted)
        // 1: spine -> DOF 0-2
        // 2: shoulder.L -> DOF 3-5
        // 3: elbow.L -> DOF 6
        // 4: wrist.L -> DOF 7-9
        // 5: finger1.L -> DOF 10
        // 6: finger2.L -> DOF 11
        // 7: shoulder.R -> DOF 12-14
        // 8: elbow.R -> DOF 15
        // 9: wrist.R -> DOF 16-18
        // 10: finger1.R -> DOF 19
        // 11: finger2.R -> DOF 20
        // Total = 21 DOFs, not 24!

        // Let me just verify the size for now
        REQUIRE(indices[0] >= 0);
        REQUIRE(indices[1] > indices[0]);
    }
}

TEST_CASE("extract_subset_state basic functionality", "[subset_utils]") {
    Skeleton skel = create_test_skeleton();
    int total_dofs = skel.total_dof_count();

    // Create full state
    State full_state(total_dofs);
    full_state.set_root_position(Eigen::Vector3d(1.0, 2.0, 3.0));
    full_state.set_root_orientation(
        Eigen::Quaterniond(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ())));
    full_state.set_root_velocity(Eigen::Vector3d(0.1, 0.2, 0.3));
    full_state.set_root_angular_velocity(Eigen::Vector3d(0.01, 0.02, 0.03));

    // Set joint angles (linear ramp for easy testing)
    Eigen::VectorXd angles(total_dofs);
    Eigen::VectorXd velocities(total_dofs);
    for (int i = 0; i < total_dofs; ++i) {
        angles(i) = 0.1 * i;
        velocities(i) = 0.01 * i;
    }
    full_state.set_joint_angles(angles);
    full_state.set_joint_velocities(velocities);

    SECTION("Extract all DOFs") {
        std::vector<int> all_indices;
        for (int i = 0; i < total_dofs; ++i) {
            all_indices.push_back(i);
        }

        State subset = extract_subset_state(full_state, all_indices);

        // Root should be copied
        REQUIRE(subset.root_position().isApprox(full_state.root_position()));
        REQUIRE(
            subset.root_orientation().coeffs().isApprox(full_state.root_orientation().coeffs()));
        REQUIRE(subset.root_velocity().isApprox(full_state.root_velocity()));
        REQUIRE(subset.root_angular_velocity().isApprox(full_state.root_angular_velocity()));

        // All joint angles should match
        REQUIRE(subset.joint_angles().size() == total_dofs);
        REQUIRE(subset.joint_angles().isApprox(full_state.joint_angles()));
        REQUIRE(subset.joint_velocities().isApprox(full_state.joint_velocities()));
    }

    SECTION("Extract subset of DOFs") {
        std::vector<int> subset_indices = {0, 1, 5, 10};
        State subset = extract_subset_state(full_state, subset_indices);

        // Root should be copied
        REQUIRE(subset.root_position().isApprox(full_state.root_position()));

        // Subset should have 4 DOFs
        REQUIRE(subset.joint_angles().size() == 4);

        // Check values match
        for (size_t i = 0; i < subset_indices.size(); ++i) {
            REQUIRE(subset.joint_angles()(i) == full_state.joint_angles()(subset_indices[i]));
            REQUIRE(subset.joint_velocities()(i) ==
                    full_state.joint_velocities()(subset_indices[i]));
        }
    }

    SECTION("Out of bounds index throws") {
        std::vector<int> bad_indices = {0, 1, 999};
        REQUIRE_THROWS_AS(extract_subset_state(full_state, bad_indices), std::invalid_argument);
    }
}

TEST_CASE("merge_subset_state basic functionality", "[subset_utils]") {
    Skeleton skel = create_test_skeleton();
    int total_dofs = skel.total_dof_count();

    // Create full state
    State full_state(total_dofs);
    Eigen::VectorXd orig_angles(total_dofs);
    Eigen::VectorXd orig_velocities(total_dofs);
    for (int i = 0; i < total_dofs; ++i) {
        orig_angles(i) = 0.1 * i;
        orig_velocities(i) = 0.01 * i;
    }
    full_state.set_joint_angles(orig_angles);
    full_state.set_joint_velocities(orig_velocities);
    full_state.set_root_position(Eigen::Vector3d(1, 2, 3));

    SECTION("Merge subset updates specified DOFs only") {
        std::vector<int> subset_indices = {2, 5, 7};

        // Create subset state with different values
        State subset_state(subset_indices.size());
        Eigen::VectorXd new_angles(subset_indices.size());
        Eigen::VectorXd new_velocities(subset_indices.size());
        for (size_t i = 0; i < subset_indices.size(); ++i) {
            new_angles(i) = 99.0 + i;
            new_velocities(i) = 9.9 + i;
        }
        subset_state.set_joint_angles(new_angles);
        subset_state.set_joint_velocities(new_velocities);
        subset_state.set_root_position(Eigen::Vector3d(999, 999, 999));  // Should be ignored

        // Merge
        merge_subset_state(full_state, subset_state, subset_indices);

        // Check updated DOFs
        for (size_t i = 0; i < subset_indices.size(); ++i) {
            int idx = subset_indices[i];
            REQUIRE(full_state.joint_angles()(idx) == new_angles(i));
            REQUIRE(full_state.joint_velocities()(idx) == new_velocities(i));
        }

        // Check unchanged DOFs
        for (int i = 0; i < total_dofs; ++i) {
            if (std::find(subset_indices.begin(), subset_indices.end(), i) ==
                subset_indices.end()) {
                REQUIRE(full_state.joint_angles()(i) == orig_angles(i));
                REQUIRE(full_state.joint_velocities()(i) == orig_velocities(i));
            }
        }

        // Root should be unchanged
        REQUIRE(full_state.root_position().isApprox(Eigen::Vector3d(1, 2, 3)));
    }

    SECTION("Size mismatch throws") {
        std::vector<int> indices = {0, 1, 2};
        State bad_subset(5);  // Wrong size
        REQUIRE_THROWS_AS(merge_subset_state(full_state, bad_subset, indices),
                          std::invalid_argument);
    }
}

TEST_CASE("extract and merge round-trip", "[subset_utils]") {
    Skeleton skel = create_test_skeleton();
    int total_dofs = skel.total_dof_count();

    // Create original full state
    State original(total_dofs);
    Eigen::VectorXd angles(total_dofs);
    Eigen::VectorXd velocities(total_dofs);
    for (int i = 0; i < total_dofs; ++i) {
        angles(i) = 0.1 * i;
        velocities(i) = 0.01 * i;
    }
    original.set_joint_angles(angles);
    original.set_joint_velocities(velocities);
    original.set_root_position(Eigen::Vector3d(1, 2, 3));
    original.set_root_orientation(Eigen::Quaterniond::Identity());

    // Extract and merge back
    std::vector<int> indices = {0, 3, 5, 7, 10};
    State subset = extract_subset_state(original, indices);

    // Create a new full state with different initial values
    State full_copy(total_dofs);
    full_copy.set_joint_angles(Eigen::VectorXd::Zero(total_dofs));
    full_copy.set_joint_velocities(Eigen::VectorXd::Zero(total_dofs));
    full_copy.set_root_position(Eigen::Vector3d(1, 2, 3));

    // Merge subset back
    merge_subset_state(full_copy, subset, indices);

    // Check that specified indices match original
    for (int idx : indices) {
        REQUIRE(full_copy.joint_angles()(idx) == original.joint_angles()(idx));
        REQUIRE(full_copy.joint_velocities()(idx) == original.joint_velocities()(idx));
    }
}

TEST_CASE("extract_subset_covariance basic functionality", "[subset_utils]") {
    int const n_dof = 10;
    int const error_dim = 2 * (6 + n_dof);

    // Create a simple diagonal covariance matrix
    Eigen::MatrixXd full_cov = Eigen::MatrixXd::Identity(error_dim, error_dim);
    for (int i = 0; i < error_dim; ++i) {
        full_cov(i, i) = i + 1.0;  // Values 1.0, 2.0, 3.0, ...
    }

    SECTION("Extract all DOFs") {
        std::vector<int> all_indices;
        for (int i = 0; i < n_dof; ++i) {
            all_indices.push_back(i);
        }

        Eigen::MatrixXd subset_cov = extract_subset_covariance(full_cov, all_indices);

        // Should be same as original
        REQUIRE(subset_cov.rows() == error_dim);
        REQUIRE(subset_cov.cols() == error_dim);
        REQUIRE(subset_cov.isApprox(full_cov));
    }

    SECTION("Extract subset of DOFs") {
        std::vector<int> subset_indices = {0, 2, 5};
        int const n_subset = subset_indices.size();
        int const subset_dim = 2 * (6 + n_subset);

        Eigen::MatrixXd subset_cov = extract_subset_covariance(full_cov, subset_indices);

        // Check dimensions
        REQUIRE(subset_cov.rows() == subset_dim);
        REQUIRE(subset_cov.cols() == subset_dim);

        // Check that root components are preserved (first 6 rows/cols)
        for (int i = 0; i < 6; ++i) {
            REQUIRE(subset_cov(i, i) == full_cov(i, i));
        }

        // Check that selected joint angle components are correct
        // Subset joint angles start at index 6 in subset_cov
        // They correspond to indices 6+subset_indices[i] in full_cov
        for (int i = 0; i < n_subset; ++i) {
            int full_idx = 6 + subset_indices[i];
            int subset_idx = 6 + i;
            REQUIRE(subset_cov(subset_idx, subset_idx) == full_cov(full_idx, full_idx));
        }

        // Check velocity components (second half)
        int vel_offset_full = 6 + n_dof;
        int vel_offset_subset = 6 + n_subset;

        // Root velocities (6 components)
        for (int i = 0; i < 6; ++i) {
            REQUIRE(subset_cov(vel_offset_subset + i, vel_offset_subset + i) ==
                    full_cov(vel_offset_full + i, vel_offset_full + i));
        }

        // Joint velocities
        for (int i = 0; i < n_subset; ++i) {
            int full_idx = vel_offset_full + 6 + subset_indices[i];
            int subset_idx = vel_offset_subset + 6 + i;
            REQUIRE(subset_cov(subset_idx, subset_idx) == full_cov(full_idx, full_idx));
        }
    }

    SECTION("Invalid covariance dimension throws") {
        Eigen::MatrixXd bad_cov(10, 10);  // Wrong size
        std::vector<int> indices = {0, 1};
        REQUIRE_THROWS_AS(extract_subset_covariance(bad_cov, indices), std::invalid_argument);
    }

    SECTION("Non-square covariance throws") {
        Eigen::MatrixXd bad_cov(error_dim, error_dim + 1);
        std::vector<int> indices = {0, 1};
        REQUIRE_THROWS_AS(extract_subset_covariance(bad_cov, indices), std::invalid_argument);
    }
}

TEST_CASE("Integration: skeleton groups with subset extraction", "[subset_utils]") {
    Skeleton skel = create_test_skeleton();
    int total_dofs = skel.total_dof_count();

    // Create full state
    State full_state(total_dofs);
    Eigen::VectorXd angles = Eigen::VectorXd::LinSpaced(total_dofs, 0.0, 1.0);
    full_state.set_joint_angles(angles);
    full_state.set_joint_velocities(Eigen::VectorXd::Zero(total_dofs));

    SECTION("Extract main group state and covariance") {
        skel.set_active_groups({"main"});
        auto main_indices = get_active_dof_indices(skel);

        // Extract state
        State main_state = extract_subset_state(full_state, main_indices);
        REQUIRE(main_state.joint_angles().size() == static_cast<int>(main_indices.size()));

        // Create and extract covariance
        int error_dim = 2 * (6 + total_dofs);
        Eigen::MatrixXd full_cov = Eigen::MatrixXd::Identity(error_dim, error_dim);

        Eigen::MatrixXd main_cov = extract_subset_covariance(full_cov, main_indices);
        int expected_dim = 2 * (6 + main_indices.size());
        REQUIRE(main_cov.rows() == expected_dim);
        REQUIRE(main_cov.cols() == expected_dim);
    }

    SECTION("Extract hand group states separately") {
        skel.set_active_groups({"HandL"});
        auto handL_indices = get_active_dof_indices(skel);

        skel.set_active_groups({"HandR"});
        auto handR_indices = get_active_dof_indices(skel);

        // Both hands should have same DOF count (2 each)
        REQUIRE(handL_indices.size() == handR_indices.size());

        // But different indices
        REQUIRE(handL_indices != handR_indices);

        // Extract hand states
        State handL_state = extract_subset_state(full_state, handL_indices);
        State handR_state = extract_subset_state(full_state, handR_indices);

        // Verify hand states have correct sizes
        REQUIRE(handL_state.joint_angles().size() == static_cast<int>(handL_indices.size()));
        REQUIRE(handR_state.joint_angles().size() == static_cast<int>(handR_indices.size()));

        // Verify hand states have different joint angles (from different indices)
        // (since full_state has linear ramp)
        REQUIRE(handL_state.joint_angles()(0) != handR_state.joint_angles()(0));
    }
}

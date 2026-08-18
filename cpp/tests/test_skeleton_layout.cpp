// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_skeleton_layout.cpp
 * @brief Unit tests for SkeletonLayout — precomputed DOF index table
 */

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/skeleton_layout.hpp"
#include <limits>

using namespace posetrak;

// ---------------------------------------------------------------------------
// Test skeleton
// ---------------------------------------------------------------------------
//
// Joint state-vector layout (all storage DOFs, root excluded):
//   spine       :  0, 1, 2   (SPHERICAL, 3 DOFs)
//   shoulder.L  :  3, 4, 5   (SPHERICAL, 3 DOFs)
//   elbow.L     :  6         (REVOLUTE,  1 DOF)
//   wrist.L     :  7, 8, 9   (SPHERICAL, 3 DOFs)
//   finger1.L   : 10         (REVOLUTE,  1 DOF)  group=HandL
//   finger2.L   : 11         (REVOLUTE,  1 DOF)  group=HandL
//   shoulder.R  : 12,13,14   (SPHERICAL, 3 DOFs)
//   elbow.R     : 15         (REVOLUTE,  1 DOF)
//   wrist.R     : 16,17,18   (SPHERICAL, 3 DOFs)
//   finger1.R   : 19         (REVOLUTE,  1 DOF)  group=HandR
//   finger2.R   : 20         (REVOLUTE,  1 DOF)  group=HandR
//   Total: 21 storage DOFs
//
// Groups:
//   "main"  — hips(root), spine, shoulder.L, elbow.L, wrist.L,
//                         shoulder.R, elbow.R, wrist.R
//   "HandL" — finger1.L, finger2.L
//   "HandR" — finger1.R, finger2.R

static Skeleton create_test_skeleton() {
    Skeleton skel;

    uint32_t hips =
        skel.add_joint("hips", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "main");
    uint32_t spine =
        skel.add_joint("spine", hips, JointType::SPHERICAL, Eigen::Vector3d(0, 0.1, 0), "main");
    uint32_t left_shoulder = skel.add_joint("shoulder.L", spine, JointType::SPHERICAL,
                                            Eigen::Vector3d(0.2, 0.1, 0), "main");
    uint32_t left_elbow = skel.add_joint("elbow.L", left_shoulder, JointType::REVOLUTE,
                                         Eigen::Vector3d(0.3, 0, 0), "main");
    uint32_t left_wrist = skel.add_joint("wrist.L", left_elbow, JointType::SPHERICAL,
                                         Eigen::Vector3d(0.3, 0, 0), "main");
    skel.add_joint("finger1.L", left_wrist, JointType::REVOLUTE, Eigen::Vector3d(0.1, 0, 0),
                   "HandL");
    skel.add_joint("finger2.L", left_wrist, JointType::REVOLUTE, Eigen::Vector3d(0.1, 0.05, 0),
                   "HandL");

    uint32_t right_shoulder = skel.add_joint("shoulder.R", spine, JointType::SPHERICAL,
                                             Eigen::Vector3d(-0.2, 0.1, 0), "main");
    uint32_t right_elbow = skel.add_joint("elbow.R", right_shoulder, JointType::REVOLUTE,
                                          Eigen::Vector3d(-0.3, 0, 0), "main");
    uint32_t right_wrist = skel.add_joint("wrist.R", right_elbow, JointType::SPHERICAL,
                                          Eigen::Vector3d(-0.3, 0, 0), "main");
    skel.add_joint("finger1.R", right_wrist, JointType::REVOLUTE, Eigen::Vector3d(-0.1, 0, 0),
                   "HandR");
    skel.add_joint("finger2.R", right_wrist, JointType::REVOLUTE, Eigen::Vector3d(-0.1, 0.05, 0),
                   "HandR");

    return skel;
}

// ---------------------------------------------------------------------------
// Tests: from_full_skeleton
// ---------------------------------------------------------------------------

TEST_CASE("from_full_skeleton: covers all 21 storage DOFs", "[skeleton_layout]") {
    Skeleton skel = create_test_skeleton();
    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));

    REQUIRE(layout->total_storage_dof_count() == 21u);
    REQUIRE(layout->joint_active_dof_count() == 21u);
    REQUIRE(layout->joints().size() == 11u);  // 11 non-root non-fixed joints
}

TEST_CASE("from_full_skeleton: has_floating_root is true", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(create_test_skeleton()));
    REQUIRE(layout->has_floating_root() == true);
    REQUIRE(layout->root_error_dof_count() == 6u);
}

TEST_CASE("from_full_skeleton: error_state_dim includes root", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(create_test_skeleton()));
    // error_state_dim = 2 * (6 + 21) = 54
    REQUIRE(layout->error_state_dim() == 54);
}

TEST_CASE("from_full_skeleton: state_index is layout-relative and starts at 0",
          "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(create_test_skeleton()));
    auto const& joints = layout->joints();

    // spine is first non-root joint, so state_index == 0
    REQUIRE(joints[0].name == "spine");
    REQUIRE(joints[0].state_index == 0u);
    REQUIRE(joints[0].storage_dof_count == 3u);
}

TEST_CASE("from_full_skeleton: state_index correct for all joints", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(create_test_skeleton()));

    struct Expected {
        std::string name;
        int state_index;
        int storage;
    };
    std::vector<Expected> expected = {
        {"spine", 0, 3},      {"shoulder.L", 3, 3}, {"elbow.L", 6, 1},     {"wrist.L", 7, 3},
        {"finger1.L", 10, 1}, {"finger2.L", 11, 1}, {"shoulder.R", 12, 3}, {"elbow.R", 15, 1},
        {"wrist.R", 16, 3},   {"finger1.R", 19, 1}, {"finger2.R", 20, 1},
    };

    for (auto const& e : expected) {
        auto const* desc = layout->get_joint(e.name);
        REQUIRE(desc != nullptr);
        REQUIRE(desc->state_index == e.state_index);
        REQUIRE(desc->storage_dof_count == e.storage);
    }
}

TEST_CASE("from_full_skeleton: error_index is cumulative active_dof_count", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(create_test_skeleton()));

    // All joints here are either SPHERICAL (active=3) or REVOLUTE (active=1)
    // and no limits are configured, so active_dof_mask should be all-true.
    // error_index follows the same pattern as state_index (since all active=storage).
    auto const* spine = layout->get_joint("spine");
    REQUIRE(spine != nullptr);
    REQUIRE(spine->error_index == 0u);

    auto const* finger1L = layout->get_joint("finger1.L");
    REQUIRE(finger1L != nullptr);
    REQUIRE(finger1L->error_index == 10u);

    auto const* finger1R = layout->get_joint("finger1.R");
    REQUIRE(finger1R != nullptr);
    REQUIRE(finger1R->error_index == 19u);
}

TEST_CASE("from_full_skeleton: get_joint returns nullptr for unknown name", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(create_test_skeleton()));
    REQUIRE(layout->get_joint("nonexistent") == nullptr);
    REQUIRE(layout->get_joint("hips") == nullptr);  // root is NOT in joints_ list
}

// ---------------------------------------------------------------------------
// Tests: from_groups — main group
// ---------------------------------------------------------------------------

TEST_CASE("from_groups main: 17 storage DOFs (fingers excluded)", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_groups(
        std::make_shared<const Skeleton>(create_test_skeleton()), {"main"});

    REQUIRE(layout->total_storage_dof_count() == 17u);
    REQUIRE(layout->joint_active_dof_count() == 17u);
    REQUIRE(layout->joints().size() == 7u);  // 7 main non-root joints
}

TEST_CASE("from_groups main: has_floating_root because hips is in main", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_groups(
        std::make_shared<const Skeleton>(create_test_skeleton()), {"main"});
    REQUIRE(layout->has_floating_root() == true);
    REQUIRE(layout->root_error_dof_count() == 6u);
}

TEST_CASE("from_groups main: error_state_dim = 2*(6+17) = 46", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_groups(
        std::make_shared<const Skeleton>(create_test_skeleton()), {"main"});
    REQUIRE(layout->error_state_dim() == 46);
}

TEST_CASE("from_groups main: state_index is layout-relative from 0", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_groups(
        std::make_shared<const Skeleton>(create_test_skeleton()), {"main"});

    // Layout-relative indices: fingers excluded, indices are 0-based within this layout
    struct Expected {
        std::string name;
        int state_index;
    };
    std::vector<Expected> expected = {
        {"spine", 0},       {"shoulder.L", 3}, {"elbow.L", 6},  {"wrist.L", 7},
        {"shoulder.R", 10}, {"elbow.R", 13},   {"wrist.R", 14},
    };

    for (auto const& e : expected) {
        auto const* desc = layout->get_joint(e.name);
        REQUIRE(desc != nullptr);
        REQUIRE(desc->state_index == e.state_index);
    }
}

TEST_CASE("from_groups main: finger joints not present in main layout", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_groups(
        std::make_shared<const Skeleton>(create_test_skeleton()), {"main"});
    REQUIRE(layout->get_joint("finger1.L") == nullptr);
    REQUIRE(layout->get_joint("finger2.L") == nullptr);
    REQUIRE(layout->get_joint("finger1.R") == nullptr);
    REQUIRE(layout->get_joint("finger2.R") == nullptr);
}

// ---------------------------------------------------------------------------
// Tests: from_groups — child hand groups (no floating root)
// ---------------------------------------------------------------------------

TEST_CASE("from_groups HandL: 2 DOFs, no floating root", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_groups(
        std::make_shared<const Skeleton>(create_test_skeleton()), {"HandL"});

    REQUIRE(layout->total_storage_dof_count() == 2u);
    REQUIRE(layout->joint_active_dof_count() == 2u);
    REQUIRE(layout->has_floating_root() == false);
    REQUIRE(layout->root_error_dof_count() == 0u);
}

TEST_CASE("from_groups HandL: error_state_dim = 2*(0+2) = 4", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_groups(
        std::make_shared<const Skeleton>(create_test_skeleton()), {"HandL"});
    REQUIRE(layout->error_state_dim() == 4);
}

TEST_CASE("from_groups HandL: finger state_index starts at 0", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_groups(
        std::make_shared<const Skeleton>(create_test_skeleton()), {"HandL"});

    auto const* f1 = layout->get_joint("finger1.L");
    auto const* f2 = layout->get_joint("finger2.L");
    REQUIRE(f1 != nullptr);
    REQUIRE(f2 != nullptr);
    REQUIRE(f1->state_index == 0u);
    REQUIRE(f2->state_index == 1u);
    REQUIRE(f1->storage_dof_count == 1u);
    REQUIRE(f2->storage_dof_count == 1u);
}

TEST_CASE("from_groups HandR: symmetric with HandL", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_groups(
        std::make_shared<const Skeleton>(create_test_skeleton()), {"HandR"});

    REQUIRE(layout->total_storage_dof_count() == 2u);
    REQUIRE(layout->has_floating_root() == false);
    REQUIRE(layout->error_state_dim() == 4);

    auto const* f1 = layout->get_joint("finger1.R");
    auto const* f2 = layout->get_joint("finger2.R");
    REQUIRE(f1 != nullptr);
    REQUIRE(f2 != nullptr);
    REQUIRE(f1->state_index == 0u);
    REQUIRE(f2->state_index == 1u);
}

// ---------------------------------------------------------------------------
// Tests: build_index_map_from
// ---------------------------------------------------------------------------

TEST_CASE("build_index_map_from: HandL subset into full layout", "[skeleton_layout]") {
    Skeleton skel = create_test_skeleton();
    auto full = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    auto hand_l = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"HandL"});

    auto map = full->build_index_map_from(*hand_l);

    // finger1.L is at full-layout state_index 10
    // finger2.L is at full-layout state_index 11
    REQUIRE(map.size() == 2u);
    REQUIRE(map[0] == 10u);  // finger1.L
    REQUIRE(map[1] == 11u);  // finger2.L
}

TEST_CASE("build_index_map_from: HandR subset into full layout", "[skeleton_layout]") {
    Skeleton skel = create_test_skeleton();
    auto full = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    auto hand_r = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"HandR"});

    auto map = full->build_index_map_from(*hand_r);

    // finger1.R is at full-layout state_index 19
    // finger2.R is at full-layout state_index 20
    REQUIRE(map.size() == 2u);
    REQUIRE(map[0] == 19u);  // finger1.R
    REQUIRE(map[1] == 20u);  // finger2.R
}

TEST_CASE("build_index_map_from: main subset into full layout", "[skeleton_layout]") {
    Skeleton skel = create_test_skeleton();
    auto full = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    auto main = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"main"});

    auto map = full->build_index_map_from(*main);

    // main layout DOFs in order:
    //   spine(0,1,2) -> full 0,1,2
    //   shoulder.L(3,4,5) -> full 3,4,5
    //   elbow.L(6) -> full 6
    //   wrist.L(7,8,9) -> full 7,8,9
    //   shoulder.R(10,11,12) -> full 12,13,14
    //   elbow.R(13) -> full 15
    //   wrist.R(14,15,16) -> full 16,17,18
    std::vector<int> expected = {
        0,  1,  2,   // spine
        3,  4,  5,   // shoulder.L
        6,           // elbow.L
        7,  8,  9,   // wrist.L
        12, 13, 14,  // shoulder.R
        15,          // elbow.R
        16, 17, 18   // wrist.R
    };
    REQUIRE(map == expected);
}

TEST_CASE("build_index_map_from: reflexive — full into full yields identity", "[skeleton_layout]") {
    Skeleton skel = create_test_skeleton();
    auto full = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));

    auto map = full->build_index_map_from(*full);

    REQUIRE(map.size() == 21u);
    for (int i = 0; i < 21; ++i) {
        REQUIRE(map[i] == i);
    }
}

TEST_CASE("build_index_map_from: throws when subset joint not in this layout",
          "[skeleton_layout]") {
    Skeleton skel = create_test_skeleton();
    auto main = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"main"});
    auto hand_l = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"HandL"});

    // HandL's fingers are not in "main" layout — should throw
    REQUIRE_THROWS_AS(main->build_index_map_from(*hand_l), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Tests: build_error_index_map_from
// ---------------------------------------------------------------------------
//
// Locks wrist.L's middle axis (SPHERICAL, in "main") so its active_dof_count
// (2) differs from its storage_dof_count (3) -- exactly the "ball joint with
// one equal-limits axis" pattern already live in this codebase's own finger
// joints (e.g. f_index.01.L's y-axis in tests/data/Harri_skeleton-regress-
// test.yaml). This makes state_index and error_index diverge for every joint
// that follows wrist.L in traversal order (finger1.L, finger2.L, shoulder.R,
// elbow.R, wrist.R, finger1.R, finger2.R), so a test built on this fixture
// actually exercises the divergence build_index_map_from() cannot handle.

static Skeleton create_test_skeleton_with_locked_wrist_axis() {
    Skeleton skel;

    uint32_t hips =
        skel.add_joint("hips", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "main");
    uint32_t spine =
        skel.add_joint("spine", hips, JointType::SPHERICAL, Eigen::Vector3d(0, 0.1, 0), "main");
    uint32_t left_shoulder = skel.add_joint("shoulder.L", spine, JointType::SPHERICAL,
                                            Eigen::Vector3d(0.2, 0.1, 0), "main");
    uint32_t left_elbow = skel.add_joint("elbow.L", left_shoulder, JointType::REVOLUTE,
                                         Eigen::Vector3d(0.3, 0, 0), "main");
    uint32_t left_wrist = skel.add_joint("wrist.L", left_elbow, JointType::SPHERICAL,
                                         Eigen::Vector3d(0.3, 0, 0), "main");
    // Lock the middle (y) axis: min == max.
    std::array<Eigen::Vector2d, 3> wrist_limits;
    wrist_limits[0] = Eigen::Vector2d(-1.0, 1.0);
    wrist_limits[1] = Eigen::Vector2d(0.0, 0.0);  // locked
    wrist_limits[2] = Eigen::Vector2d(-1.0, 1.0);
    skel.set_joint_limits(left_wrist, wrist_limits, 3);
    skel.add_joint("finger1.L", left_wrist, JointType::REVOLUTE, Eigen::Vector3d(0.1, 0, 0),
                   "HandL");
    skel.add_joint("finger2.L", left_wrist, JointType::REVOLUTE, Eigen::Vector3d(0.1, 0.05, 0),
                   "HandL");

    uint32_t right_shoulder = skel.add_joint("shoulder.R", spine, JointType::SPHERICAL,
                                             Eigen::Vector3d(-0.2, 0.1, 0), "main");
    uint32_t right_elbow = skel.add_joint("elbow.R", right_shoulder, JointType::REVOLUTE,
                                          Eigen::Vector3d(-0.3, 0, 0), "main");
    uint32_t right_wrist = skel.add_joint("wrist.R", right_elbow, JointType::SPHERICAL,
                                          Eigen::Vector3d(-0.3, 0, 0), "main");
    skel.add_joint("finger1.R", right_wrist, JointType::REVOLUTE, Eigen::Vector3d(-0.1, 0, 0),
                   "HandR");
    skel.add_joint("finger2.R", right_wrist, JointType::REVOLUTE, Eigen::Vector3d(-0.1, 0.05, 0),
                   "HandR");

    return skel;
}

TEST_CASE("locked wrist fixture: wrist.L active_dof_count is 2, storage_dof_count is 3",
          "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(create_test_skeleton_with_locked_wrist_axis()));
    auto const* wrist_l = layout->get_joint("wrist.L");
    REQUIRE(wrist_l != nullptr);
    REQUIRE(wrist_l->storage_dof_count == 3u);
    REQUIRE(wrist_l->active_dof_count == 2u);
}

TEST_CASE("build_error_index_map_from: state_index and error_index diverge after a locked axis",
          "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(create_test_skeleton_with_locked_wrist_axis()));

    // wrist.L itself: state_index 7 (3 slots: 7,8,9), error_index 7 (2 slots: 7,8 --
    // the locked middle axis contributes no error-state slot).
    auto const* wrist_l = layout->get_joint("wrist.L");
    REQUIRE(wrist_l->state_index == 7u);
    REQUIRE(wrist_l->error_index == 7u);

    // shoulder.R comes after wrist.L and its 2 excluded (finger) siblings in
    // traversal order. state_index accounts for all 3 wrist.L storage slots
    // (10,11,12 already consumed by fingers) -> shoulder.R state_index == 12.
    // error_index only advances by wrist.L's 2 active slots -> shoulder.R
    // error_index == 11, one less than state_index. This is exactly the
    // divergence build_index_map_from() cannot represent.
    auto const* shoulder_r = layout->get_joint("shoulder.R");
    REQUIRE(shoulder_r != nullptr);
    REQUIRE(shoulder_r->state_index == 12u);
    REQUIRE(shoulder_r->error_index == 11u);
}

TEST_CASE("build_error_index_map_from: main subset into full layout, locked-axis fixture",
          "[skeleton_layout]") {
    Skeleton skel = create_test_skeleton_with_locked_wrist_axis();
    auto full = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    auto main = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"main"});

    // Sanity: state_index-based map still walks storage slots 1:1, unaware of
    // the locked axis -- this is build_index_map_from()'s known limitation,
    // not a bug in it; it's simply the wrong tool for a covariance/error-state
    // merge, which is exactly why build_error_index_map_from() exists.
    auto state_map = full->build_index_map_from(*main);
    REQUIRE(state_map.size() == 17u);

    auto error_map = full->build_error_index_map_from(*main);
    // main's active DOF count is 16 (17 storage slots minus wrist.L's 1 locked axis).
    REQUIRE(error_map.size() == 16u);

    // main layout error-index order:
    //   spine(0,1,2) -> full 0,1,2
    //   shoulder.L(3,4,5) -> full 3,4,5
    //   elbow.L(6) -> full 6
    //   wrist.L(7,8) -> full 7,8            (only 2 active axes, not 3)
    //   shoulder.R(9,10,11) -> full 11,12,13
    //   elbow.R(12) -> full 14
    //   wrist.R(13,14,15) -> full 15,16,17
    std::vector<int> expected = {
        0,  1,  2,   // spine
        3,  4,  5,   // shoulder.L
        6,           // elbow.L
        7,  8,       // wrist.L (locked middle axis excluded)
        11, 12, 13,  // shoulder.R
        14,          // elbow.R
        15, 16, 17   // wrist.R
    };
    REQUIRE(error_map == expected);
}

TEST_CASE("build_error_index_map_from: HandL subset into full layout, locked-axis fixture",
          "[skeleton_layout]") {
    Skeleton skel = create_test_skeleton_with_locked_wrist_axis();
    auto full = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    auto hand_l = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"HandL"});

    auto error_map = full->build_error_index_map_from(*hand_l);

    // finger1.L/finger2.L are unaffected by wrist.L's locked axis themselves
    // (both REVOLUTE, single active DOF each), but their full-layout
    // error_index (9, 10) is now one less than their full-layout state_index
    // (10, 11) because wrist.L only contributed 2 error-state slots ahead of
    // them instead of 3.
    REQUIRE(error_map.size() == 2u);
    REQUIRE(error_map[0] == 9);   // finger1.L
    REQUIRE(error_map[1] == 10);  // finger2.L
}

TEST_CASE("build_error_index_map_from: reflexive -- full into full yields identity",
          "[skeleton_layout]") {
    auto full = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(create_test_skeleton_with_locked_wrist_axis()));

    auto map = full->build_error_index_map_from(*full);

    REQUIRE(map.size() == static_cast<size_t>(full->joint_active_dof_count()));
    for (size_t i = 0; i < map.size(); ++i) {
        REQUIRE(map[i] == static_cast<int>(i));
    }
}

TEST_CASE("build_error_index_map_from: throws when subset joint not in this layout",
          "[skeleton_layout]") {
    Skeleton skel = create_test_skeleton();
    auto main = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"main"});
    auto hand_l = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"HandL"});

    REQUIRE_THROWS_AS(main->build_error_index_map_from(*hand_l), std::invalid_argument);
}

TEST_CASE("build_error_index_map_from: throws when active_dof_count mismatches between layouts",
          "[skeleton_layout]") {
    // Build "main" from a skeleton where wrist.L has a locked axis, and a full
    // layout from a DIFFERENT skeleton object where it doesn't -- same joint
    // name, different active_dof_count, must be rejected rather than silently
    // producing a map that misaligns by one slot per divergent joint.
    auto full_unlocked = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(create_test_skeleton()));
    auto main_locked = SkeletonLayout::from_groups(
        std::make_shared<const Skeleton>(create_test_skeleton_with_locked_wrist_axis()), {"main"});

    REQUIRE_THROWS_AS(full_unlocked->build_error_index_map_from(*main_locked),
                      std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Tests: error_blob_index
// ---------------------------------------------------------------------------

TEST_CASE("error_blob_index: floating-root layout offsets position and velocity blocks",
          "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(create_test_skeleton()));
    // root_error_dof_count() == 6, joint_active_dof_count() == 21.
    REQUIRE(layout->error_blob_index(0, /*is_velocity=*/false) == 6);
    REQUIRE(layout->error_blob_index(20, /*is_velocity=*/false) == 26);
    REQUIRE(layout->error_blob_index(0, /*is_velocity=*/true) == 33);   // 2*6 + 21 + 0
    REQUIRE(layout->error_blob_index(20, /*is_velocity=*/true) == 53);  // 2*6 + 21 + 20
    // error_state_dim() is 54 (0..53), so the last velocity index is in range.
    REQUIRE(layout->error_state_dim() == 54);
}

TEST_CASE("error_blob_index: fixed-root (child) layout has no root offset", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_groups(
        std::make_shared<const Skeleton>(create_test_skeleton()), {"HandL"});
    // root_error_dof_count() == 0, joint_active_dof_count() == 2.
    REQUIRE(layout->error_blob_index(0, /*is_velocity=*/false) == 0);
    REQUIRE(layout->error_blob_index(0, /*is_velocity=*/true) == 2);  // 2*0 + 2 + 0
    REQUIRE(layout->error_state_dim() == 4);
}

// ---------------------------------------------------------------------------
// Tests: error handling
// ---------------------------------------------------------------------------

TEST_CASE("from_groups: throws on empty group list", "[skeleton_layout]") {
    REQUIRE_THROWS_AS(
        SkeletonLayout::from_groups(std::make_shared<const Skeleton>(create_test_skeleton()), {}),
        std::invalid_argument);
}

TEST_CASE("from_groups: throws when no joints match group", "[skeleton_layout]") {
    REQUIRE_THROWS_AS(
        SkeletonLayout::from_groups(std::make_shared<const Skeleton>(create_test_skeleton()),
                                    {"nonexistent_group"}),
        std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Tests: JointDesc field correctness
// ---------------------------------------------------------------------------

TEST_CASE("JointDesc type field is set correctly", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(create_test_skeleton()));

    auto const* spine = layout->get_joint("spine");
    REQUIRE(spine != nullptr);
    REQUIRE(spine->type == JointType::SPHERICAL);

    auto const* elbow = layout->get_joint("elbow.L");
    REQUIRE(elbow != nullptr);
    REQUIRE(elbow->type == JointType::REVOLUTE);
}

TEST_CASE("JointDesc active_dof_mask is all-true for unconstrained joints", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(create_test_skeleton()));

    // spine is SPHERICAL with no limits set → all 3 axes active
    auto const* spine = layout->get_joint("spine");
    REQUIRE(spine != nullptr);
    REQUIRE(spine->active_dof_mask[0] == true);
    REQUIRE(spine->active_dof_mask[1] == true);
    REQUIRE(spine->active_dof_mask[2] == true);
    REQUIRE(spine->active_dof_count == 3u);

    // elbow.L is REVOLUTE → only axis 0 active
    auto const* elbow = layout->get_joint("elbow.L");
    REQUIRE(elbow != nullptr);
    REQUIRE(elbow->active_dof_mask[0] == true);
    REQUIRE(elbow->active_dof_mask[1] == false);
    REQUIRE(elbow->active_dof_mask[2] == false);
    REQUIRE(elbow->active_dof_count == 1u);
}

TEST_CASE("JointDesc is_floating_root is always false in joints() list", "[skeleton_layout]") {
    auto layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(create_test_skeleton()));
    for (auto const& desc : layout->joints()) {
        REQUIRE(desc.is_floating_root == false);
    }
}

// ---------------------------------------------------------------------------
// Tests: parent_marker_id()
// ---------------------------------------------------------------------------

TEST_CASE("parent_marker_id: root marker has no parent (-1)", "[skeleton_layout]") {
    Skeleton skel;
    uint32_t root =
        skel.add_joint("root", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "main");
    skel.add_marker("hip_mrk", root, Eigen::Vector3d::Zero());

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    REQUIRE(layout->parent_marker_id(0) == -1);
}

TEST_CASE("parent_marker_id: direct child marker points to parent joint's marker",
          "[skeleton_layout]") {
    // root → child, both have markers
    Skeleton skel;
    uint32_t root =
        skel.add_joint("root", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "main");
    uint32_t child =
        skel.add_joint("child", root, JointType::REVOLUTE, Eigen::Vector3d(0.3, 0, 0), "main");
    // marker 0 on root, marker 1 on child
    skel.add_marker("root_mrk", root, Eigen::Vector3d::Zero());
    skel.add_marker("child_mrk", child, Eigen::Vector3d::Zero());

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    REQUIRE(layout->parent_marker_id(0) == -1);  // root marker has no parent
    REQUIRE(layout->parent_marker_id(1) == 0);   // child's parent marker is root_mrk
}

TEST_CASE("parent_marker_id: skips intermediate joint with no marker", "[skeleton_layout]") {
    // root (marker) → mid (no marker) → leaf (marker)
    // leaf's parent marker should be root's marker, skipping mid
    Skeleton skel;
    uint32_t root =
        skel.add_joint("root", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "main");
    uint32_t mid =
        skel.add_joint("mid", root, JointType::REVOLUTE, Eigen::Vector3d(0.2, 0, 0), "main");
    uint32_t leaf =
        skel.add_joint("leaf", mid, JointType::REVOLUTE, Eigen::Vector3d(0.2, 0, 0), "main");
    skel.add_marker("root_mrk", root, Eigen::Vector3d::Zero());  // marker 0
    skel.add_marker("leaf_mrk", leaf, Eigen::Vector3d::Zero());  // marker 1, mid has no marker

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    REQUIRE(layout->parent_marker_id(0) == -1);  // root marker: no parent
    REQUIRE(layout->parent_marker_id(1) == 0);   // leaf skips mid, reaches root_mrk
}

// ---------------------------------------------------------------------------
// Tests: hierarchy_distance()
// ---------------------------------------------------------------------------

TEST_CASE("hierarchy_distance: same marker is distance 0", "[skeleton_layout]") {
    Skeleton skel;
    uint32_t root =
        skel.add_joint("root", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "main");
    uint32_t child =
        skel.add_joint("child", root, JointType::REVOLUTE, Eigen::Vector3d(0.3, 0, 0), "main");
    skel.add_marker("root_mrk", root, Eigen::Vector3d::Zero());    // 0
    skel.add_marker("child_mrk", child, Eigen::Vector3d::Zero());  // 1

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    REQUIRE(layout->hierarchy_distance(0, 0) == 0);
    REQUIRE(layout->hierarchy_distance(1, 1) == 0);
}

TEST_CASE("hierarchy_distance: adjacent markers (parent-child) are distance 1",
          "[skeleton_layout]") {
    // root → child; direct joint edge → distance 1
    Skeleton skel;
    uint32_t root =
        skel.add_joint("root", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "main");
    uint32_t child =
        skel.add_joint("child", root, JointType::REVOLUTE, Eigen::Vector3d(0.3, 0, 0), "main");
    skel.add_marker("root_mrk", root, Eigen::Vector3d::Zero());    // 0
    skel.add_marker("child_mrk", child, Eigen::Vector3d::Zero());  // 1

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    REQUIRE(layout->hierarchy_distance(0, 1) == 1);
    REQUIRE(layout->hierarchy_distance(1, 0) == 1);  // symmetric
}

TEST_CASE("hierarchy_distance: linear chain root→mid→leaf gives correct distances",
          "[skeleton_layout]") {
    // root → mid → leaf; markers on all three joints
    Skeleton skel;
    uint32_t root =
        skel.add_joint("root", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "main");
    uint32_t mid =
        skel.add_joint("mid", root, JointType::REVOLUTE, Eigen::Vector3d(0.2, 0, 0), "main");
    uint32_t leaf =
        skel.add_joint("leaf", mid, JointType::REVOLUTE, Eigen::Vector3d(0.2, 0, 0), "main");
    skel.add_marker("root_mrk", root, Eigen::Vector3d::Zero());  // 0
    skel.add_marker("mid_mrk", mid, Eigen::Vector3d::Zero());    // 1
    skel.add_marker("leaf_mrk", leaf, Eigen::Vector3d::Zero());  // 2

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    REQUIRE(layout->hierarchy_distance(0, 1) == 1);  // root ↔ mid
    REQUIRE(layout->hierarchy_distance(1, 2) == 1);  // mid ↔ leaf
    REQUIRE(layout->hierarchy_distance(0, 2) == 2);  // root ↔ leaf (through mid)
    REQUIRE(layout->hierarchy_distance(2, 0) == 2);  // symmetric
}

TEST_CASE("hierarchy_distance: siblings share parent, distance 2", "[skeleton_layout]") {
    // root → left, root → right; both siblings → distance 2 (left→root→right)
    Skeleton skel;
    uint32_t root =
        skel.add_joint("root", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "main");
    uint32_t left =
        skel.add_joint("left", root, JointType::REVOLUTE, Eigen::Vector3d(-0.2, 0, 0), "main");
    uint32_t right =
        skel.add_joint("right", root, JointType::REVOLUTE, Eigen::Vector3d(0.2, 0, 0), "main");
    skel.add_marker("left_mrk", left, Eigen::Vector3d::Zero());    // 0
    skel.add_marker("right_mrk", right, Eigen::Vector3d::Zero());  // 1

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    REQUIRE(layout->hierarchy_distance(0, 1) == 2);
    REQUIRE(layout->hierarchy_distance(1, 0) == 2);
}

TEST_CASE("hierarchy_distance: out-of-range marker returns INT_MAX", "[skeleton_layout]") {
    Skeleton skel;
    uint32_t root =
        skel.add_joint("root", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "main");
    skel.add_marker("root_mrk", root, Eigen::Vector3d::Zero());

    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    REQUIRE(layout->hierarchy_distance(-1, 0) == std::numeric_limits<int>::max());
    REQUIRE(layout->hierarchy_distance(0, 99) == std::numeric_limits<int>::max());
}

// ---------------------------------------------------------------------------
// from_groups: dual group membership (a joint declared in more than one
// group's SkeletonGroup::joints list -- e.g. the hierarchical solver's
// wrist joint, solved by both a parent and a child group; see
// docs/skeleton-format.md and Skeleton::is_joint_in_groups()).
// ---------------------------------------------------------------------------

namespace {

/// root -> forearm -> hand -> finger. "main" declares [root, forearm, hand];
/// "HandL" declares [hand, finger] -- "hand" is in BOTH.
Skeleton create_dual_membership_skeleton() {
    Skeleton skel;
    uint32_t root =
        skel.add_joint("root", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "main");
    uint32_t forearm =
        skel.add_joint("forearm", root, JointType::REVOLUTE, Eigen::Vector3d(0.1, 0, 0), "main");
    uint32_t hand =
        skel.add_joint("hand", forearm, JointType::SPHERICAL, Eigen::Vector3d(0.1, 0, 0), "HandL");
    skel.add_joint("finger", hand, JointType::REVOLUTE, Eigen::Vector3d(0.05, 0, 0), "HandL");

    skel.add_group("main", {"root", "forearm", "hand"}, {});
    skel.add_group("HandL", {"hand", "finger"}, {}, /*freeflyer_joint=*/"forearm",
                   /*ref_marker=*/"");
    (void)root;
    return skel;
}

}  // namespace

TEST_CASE("from_groups: a joint declared in two groups' lists is included in both layouts",
          "[skeleton_layout][groups]") {
    auto skel_ptr = std::make_shared<const Skeleton>(create_dual_membership_skeleton());

    auto main_layout = SkeletonLayout::from_groups(skel_ptr, {"main"});
    auto hand_layout = SkeletonLayout::from_groups(skel_ptr, {"HandL"});

    CHECK(main_layout->get_joint("hand") != nullptr);
    CHECK(hand_layout->get_joint("hand") != nullptr);

    // finger is HandL-only: excluded from main, included in HandL.
    CHECK(main_layout->get_joint("finger") == nullptr);
    CHECK(hand_layout->get_joint("finger") != nullptr);

    // forearm is main-only: included in main, excluded from HandL (it's the
    // freeflyer boundary, not one of HandL's own estimated joints).
    CHECK(main_layout->get_joint("forearm") != nullptr);
    CHECK(hand_layout->get_joint("forearm") == nullptr);
}

TEST_CASE("from_groups: falls back to Joint::group when no SkeletonGroup is registered",
          "[skeleton_layout][groups]") {
    // No add_group() calls at all -- matches every pre-existing from_groups()
    // test/caller that builds a Skeleton directly via add_joint(group=...).
    Skeleton skel;
    skel.add_joint("root", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "main");
    skel.add_joint("arm", 0u, JointType::REVOLUTE, Eigen::Vector3d(0.1, 0, 0), "main");
    skel.add_joint("finger", 1u, JointType::REVOLUTE, Eigen::Vector3d(0.05, 0, 0), "HandL");

    auto skel_ptr = std::make_shared<const Skeleton>(skel);
    auto main_layout = SkeletonLayout::from_groups(skel_ptr, {"main"});
    auto hand_layout = SkeletonLayout::from_groups(skel_ptr, {"HandL"});

    CHECK(main_layout->get_joint("arm") != nullptr);
    CHECK(main_layout->get_joint("finger") == nullptr);
    CHECK(hand_layout->get_joint("finger") != nullptr);
    CHECK(hand_layout->get_joint("arm") == nullptr);
}

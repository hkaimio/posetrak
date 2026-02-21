/**
 * @file test_skeleton_layout.cpp
 * @brief Unit tests for SkeletonLayout — precomputed DOF index table
 */

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/skeleton_layout.hpp"

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

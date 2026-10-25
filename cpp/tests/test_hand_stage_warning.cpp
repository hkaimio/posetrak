// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include "posetrak/core/skeleton.hpp"
#include "posetrak/tracking/multi_person_tracker.hpp"

using namespace posetrak;

namespace {

Skeleton make_arm_skeleton() {
    Skeleton skel;
    uint32_t root = skel.add_joint("root", std::nullopt, JointType::REVOLUTE,
                                   Eigen::Vector3d::Zero(), "", Eigen::Vector3d::Zero());
    skel.add_joint("forearm", root, JointType::REVOLUTE, Eigen::Vector3d(0.5, 0.0, 0.0), "",
                   Eigen::Vector3d::Zero());
    return skel;
}

}  // namespace

TEST_CASE("hand_stage_cost_warning flags hand groups outside the restricted hierarchical form",
          "[multi_person_tracker]") {
    Skeleton plain = make_arm_skeleton();
    Skeleton with_hand = make_arm_skeleton();
    with_hand.add_group("main", {"root", "forearm"}, {});
    with_hand.add_group("HandL", {"forearm"}, {}, /*freeflyer_joint=*/"forearm",
                        /*ref_marker=*/"wrist");

    TrackerConfig unrestricted;
    TrackerConfig restricted;
    restricted.active_joint_groups = {"main"};

    CHECK(hand_stage_cost_warning(plain, unrestricted, false).empty());
    CHECK_FALSE(hand_stage_cost_warning(with_hand, unrestricted, false).empty());
    CHECK_FALSE(hand_stage_cost_warning(with_hand, unrestricted, true).empty());
    CHECK(hand_stage_cost_warning(with_hand, restricted, true).empty());
}

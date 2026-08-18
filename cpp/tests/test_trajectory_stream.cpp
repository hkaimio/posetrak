// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include "posetrak/kinematics/pinocchio_model_builder.hpp"
#include "posetrak/tracking/trajectory_stream.hpp"
#include <memory>

using namespace posetrak;

namespace {

// root (free-flyer, no parent) -> forearm (REVOLUTE, offset (0.5, 0, 0) along X)
Skeleton make_simple_arm_skeleton() {
    Skeleton skel;
    uint32_t root_idx = skel.add_joint("root", std::nullopt, JointType::REVOLUTE,
                                       Eigen::Vector3d::Zero(), "", Eigen::Vector3d::Zero());
    skel.add_joint("forearm", root_idx, JointType::REVOLUTE, Eigen::Vector3d(0.5, 0.0, 0.0), "",
                   Eigen::Vector3d::Zero());
    return skel;
}

SmoothedFrame make_frame(double timestamp, Eigen::Vector3d const& root_pos, int n_dof) {
    State state(root_pos, Eigen::Quaterniond::Identity(), Eigen::VectorXd::Zero(n_dof),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::VectorXd::Zero(n_dof));
    return SmoothedFrame{timestamp, state, Eigen::MatrixXd()};
}

}  // namespace

TEST_CASE("BatchTrajectoryStream yields forearm world transform per smoothed frame",
          "[trajectory_stream]") {
    Skeleton skel = make_simple_arm_skeleton();
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skel, model, data);
    auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skel);
    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    ForwardKinematics fk(model, data, marker_map, layout);

    int n = static_cast<int>(layout->total_storage_dof_count());
    std::vector<SmoothedFrame> frames;
    frames.push_back(make_frame(0.0, Eigen::Vector3d(0.0, 0.0, 0.0), n));
    frames.push_back(make_frame(1.0 / 30.0, Eigen::Vector3d(1.0, 0.0, 0.0), n));
    frames.push_back(make_frame(2.0 / 30.0, Eigen::Vector3d(2.0, 0.0, 0.0), n));

    BatchTrajectoryStream stream(frames, fk, "forearm");
    REQUIRE(stream.size() == 3);

    SECTION("frames come back in order with the expected world position") {
        auto f0 = stream.next();
        REQUIRE(f0.has_value());
        CHECK_THAT(f0->timestamp, Catch::Matchers::WithinAbs(0.0, 1e-9));
        CHECK_THAT(f0->position[0], Catch::Matchers::WithinAbs(0.5, 1e-6));

        auto f1 = stream.next();
        REQUIRE(f1.has_value());
        CHECK_THAT(f1->timestamp, Catch::Matchers::WithinAbs(1.0 / 30.0, 1e-9));
        CHECK_THAT(f1->position[0], Catch::Matchers::WithinAbs(1.5, 1e-6));

        auto f2 = stream.next();
        REQUIRE(f2.has_value());
        CHECK_THAT(f2->timestamp, Catch::Matchers::WithinAbs(2.0 / 30.0, 1e-9));
        CHECK_THAT(f2->position[0], Catch::Matchers::WithinAbs(2.5, 1e-6));
    }

    SECTION("stream is exhausted after the last frame") {
        stream.next();
        stream.next();
        stream.next();
        auto past_end = stream.next();
        CHECK_FALSE(past_end.has_value());
    }

    SECTION("orientation is identity throughout (zero-angle fixture)") {
        auto f0 = stream.next();
        REQUIRE(f0.has_value());
        CHECK_THAT(f0->orientation.angularDistance(Eigen::Quaterniond::Identity()),
                   Catch::Matchers::WithinAbs(0.0, 1e-9));
    }
}

TEST_CASE("BatchTrajectoryStream propagates unknown joint name as std::out_of_range",
          "[trajectory_stream]") {
    Skeleton skel = make_simple_arm_skeleton();
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skel, model, data);
    auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skel);
    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    ForwardKinematics fk(model, data, marker_map, layout);

    int n = static_cast<int>(layout->total_storage_dof_count());
    std::vector<SmoothedFrame> frames{make_frame(0.0, Eigen::Vector3d::Zero(), n)};

    BatchTrajectoryStream stream(frames, fk, "nonexistent");
    REQUIRE_THROWS_AS(stream.next(), std::out_of_range);
}

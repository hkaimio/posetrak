// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include <posetrak/tracking/hierarchical_solver.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace posetrak;
using Catch::Matchers::WithinAbs;

namespace {

SkeletonGroup make_group(std::string const& name, std::string const& freeflyer_joint,
                         std::string const& ref_marker) {
    SkeletonGroup g;
    g.name = name;
    g.freeflyer_joint = freeflyer_joint;
    g.ref_marker = ref_marker;
    return g;
}

}  // namespace

TEST_CASE("build_stage_tracker_config: inherits every field the stage doesn't override",
          "[hierarchical_solver]") {
    TrackerConfig parent;
    parent.process_noise_std = 0.5;
    parent.pose_noise_std = 2.0;
    parent.calib_noise_std = 3.0;
    parent.outlier_threshold = 5.991;
    parent.init_joint_std = 0.05;
    parent.init_velocity_std = 0.01;
    parent.ukf_alpha = 0.5;  // no tracker_config_stages column for this -- must still inherit
    parent.active_joint_groups = {"main"};

    StageConfigOverrides overrides;  // every field nullopt
    overrides.group_name = "HandL";
    auto group = make_group("HandL", "forearm.L", "MRK-wrist.L");

    TrackerConfig child = build_stage_tracker_config(parent, overrides, group);

    CHECK_THAT(child.process_noise_std, WithinAbs(0.5, 1e-12));
    CHECK_THAT(child.pose_noise_std, WithinAbs(2.0, 1e-12));
    CHECK_THAT(child.calib_noise_std, WithinAbs(3.0, 1e-12));
    CHECK_THAT(child.outlier_threshold, WithinAbs(5.991, 1e-12));
    CHECK_THAT(child.init_joint_std, WithinAbs(0.05, 1e-12));
    CHECK_THAT(child.init_velocity_std, WithinAbs(0.01, 1e-12));
    CHECK_THAT(child.ukf_alpha, WithinAbs(0.5, 1e-12));

    CHECK(child.active_joint_groups == std::vector<std::string>{"HandL"});
    CHECK(child.fixed_root_joint_name == "forearm.L");
}

TEST_CASE("build_stage_tracker_config: applies every non-nullopt override",
          "[hierarchical_solver]") {
    TrackerConfig parent;
    parent.process_noise_std = 0.5;
    parent.pose_noise_std = 2.0;
    parent.calib_noise_std = 3.0;
    parent.outlier_threshold = 5.991;
    parent.init_joint_std = 0.05;
    parent.init_velocity_std = 0.01;

    StageConfigOverrides overrides;
    overrides.group_name = "HandR";
    overrides.process_noise_std = 0.1;
    overrides.process_noise_vel_std = 0.2;
    overrides.velocity_half_life_s = 0.3;
    overrides.pose_noise_std = 1.5;
    overrides.calib_noise_std = 1.0;
    overrides.outlier_threshold = 4.0;
    overrides.init_joint_std = 0.3;
    overrides.init_velocity_std = 0.02;

    auto group = make_group("HandR", "forearm.R", "MRK-wrist.R");
    TrackerConfig child = build_stage_tracker_config(parent, overrides, group);

    CHECK_THAT(child.process_noise_std, WithinAbs(0.1, 1e-12));
    REQUIRE(child.process_noise_vel_std.has_value());
    CHECK_THAT(*child.process_noise_vel_std, WithinAbs(0.2, 1e-12));
    REQUIRE(child.velocity_half_life_s.has_value());
    CHECK_THAT(*child.velocity_half_life_s, WithinAbs(0.3, 1e-12));
    CHECK_THAT(child.pose_noise_std, WithinAbs(1.5, 1e-12));
    CHECK_THAT(child.calib_noise_std, WithinAbs(1.0, 1e-12));
    CHECK_THAT(child.outlier_threshold, WithinAbs(4.0, 1e-12));
    CHECK_THAT(child.init_joint_std, WithinAbs(0.3, 1e-12));
    CHECK_THAT(child.init_velocity_std, WithinAbs(0.02, 1e-12));

    CHECK(child.active_joint_groups == std::vector<std::string>{"HandR"});
    CHECK(child.fixed_root_joint_name == "forearm.R");
}

TEST_CASE("build_stage_tracker_config: partial override leaves the rest inherited",
          "[hierarchical_solver]") {
    TrackerConfig parent;
    parent.process_noise_std = 0.5;
    parent.outlier_threshold = 5.991;

    StageConfigOverrides overrides;
    overrides.group_name = "HandL";
    overrides.process_noise_std = 0.1;  // only this one overridden

    auto group = make_group("HandL", "forearm.L", "MRK-wrist.L");
    TrackerConfig child = build_stage_tracker_config(parent, overrides, group);

    CHECK_THAT(child.process_noise_std, WithinAbs(0.1, 1e-12));    // overridden
    CHECK_THAT(child.outlier_threshold, WithinAbs(5.991, 1e-12));  // inherited
}

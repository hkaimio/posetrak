// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_config.cpp
 * @brief Tests for TrackerAppConfig loading, including hierarchical tracking config.
 *
 * Tests use fixture TOML files under tests/data/ (read relative to the
 * project source root, which is the working directory when the test binary runs).
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "posetrak/core/config.hpp"

using namespace posetrak;
using Catch::Approx;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a TrackerAppConfig with real paths so that validate() can reach
/// past its file-existence checks first.
static TrackerAppConfig make_valid_base_config() {
    TrackerAppConfig cfg;
    cfg.skeleton_path = "cpp/tests/data/simple_humanoid.yaml";
    cfg.cameras_path = "cpp/tests/data/pose2sim_camera_calib.toml";
    cfg.observations_dir = "cpp/tests/data/openpose";
    cfg.python_state_path = std::nullopt;
    cfg.process_noise_std = 0.5;
    cfg.calib_noise_std = 2.0;
    cfg.outlier_threshold = 4.0;
    cfg.ukf_alpha = 0.5;
    cfg.ukf_beta = 2.0;
    cfg.ukf_kappa = 0.0;
    cfg.ik_max_iterations = 100;
    cfg.ik_tolerance = 0.01;
    cfg.init_position_std = 0.1;
    cfg.init_orientation_std = 0.1;
    cfg.init_joint_std = 0.1;
    cfg.init_velocity_std = 0.1;
    cfg.min_cameras_for_init = 2;
    cfg.start_time = 0.0;
    cfg.end_time = -1.0;
    cfg.tracker_fps = 100.0;
    cfg.output_dir = "/tmp/posetrak_test_output";
    return cfg;
}

// ---------------------------------------------------------------------------
// Tests: minimal config loading
// ---------------------------------------------------------------------------

TEST_CASE("TrackerAppConfig load minimal config applies defaults", "[config]") {
    auto cfg = TrackerAppConfig::load("cpp/tests/data/minimal_config_test.toml");

    REQUIRE(cfg.skeleton_path == "cpp/tests/data/simple_humanoid.yaml");
    REQUIRE(cfg.cameras_path == "cpp/tests/data/pose2sim_camera_calib.toml");
    REQUIRE(cfg.observations_dir == "cpp/tests/data/openpose");
    REQUIRE(cfg.person_id == 0);
    REQUIRE(cfg.active_joint_groups.empty());

    // Tracking defaults
    REQUIRE(cfg.process_noise_std == Approx(0.5));
    REQUIRE(cfg.calib_noise_std == Approx(2.0));
    REQUIRE(cfg.outlier_threshold == Approx(4.0));
    // Marker-based-mocap dot assignment (dot-assignment-architecture-design.md
    // sec 8, sub-phase C2.9) -- TOML-only, same as rigid_init_max_residual_m's
    // own real shape, no tracker_configs DB column.
    REQUIRE(cfg.dot_assignment_gate_mahalanobis == Approx(9.21));

    // UKF defaults
    REQUIRE(cfg.ukf_alpha == Approx(0.5));
    REQUIRE(cfg.ukf_beta == Approx(2.0));
    REQUIRE(cfg.ukf_kappa == Approx(0.0));
}

// ---------------------------------------------------------------------------
// Tests: dot_assignment_gate_mahalanobis (design doc sub-phase C2.9)
// ---------------------------------------------------------------------------

TEST_CASE("TrackerAppConfig to_tracker_config threads dot_assignment_gate_mahalanobis through",
          "[config]") {
    auto cfg = make_valid_base_config();
    cfg.dot_assignment_gate_mahalanobis = 12.5;
    TrackerConfig tc = cfg.to_tracker_config();
    REQUIRE(tc.dot_assignment_gate_mahalanobis == Approx(12.5));
}

TEST_CASE("TrackerAppConfig validate rejects non-positive dot_assignment_gate_mahalanobis",
          "[config]") {
    auto cfg = make_valid_base_config();
    cfg.dot_assignment_gate_mahalanobis = 0.0;
    REQUIRE_THROWS_AS(cfg.validate(), std::runtime_error);

    cfg.dot_assignment_gate_mahalanobis = -1.0;
    REQUIRE_THROWS_AS(cfg.validate(), std::runtime_error);
}

TEST_CASE("TrackerAppConfig validate accepts a positive dot_assignment_gate_mahalanobis",
          "[config]") {
    auto cfg = make_valid_base_config();
    cfg.dot_assignment_gate_mahalanobis = 9.21;
    REQUIRE_NOTHROW(cfg.validate());
}

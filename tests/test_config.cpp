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

/// Build a TrackerAppConfig with real paths so that validate() can reach the
/// hierarchical validation checks without tripping over file-not-found first.
static TrackerAppConfig make_valid_base_config() {
    TrackerAppConfig cfg;
    cfg.skeleton_path = "tests/data/simple_humanoid.yaml";
    cfg.cameras_path = "tests/data/pose2sim_camera_calib.toml";
    cfg.observations_dir = "tests/data/openpose";
    cfg.python_state_path = std::nullopt;
    cfg.process_noise_std = 0.5;
    cfg.measurement_noise_std = 2.0;
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
    auto cfg = TrackerAppConfig::load("tests/data/minimal_config_test.toml");

    REQUIRE(cfg.skeleton_path == "tests/data/simple_humanoid.yaml");
    REQUIRE(cfg.cameras_path == "tests/data/pose2sim_camera_calib.toml");
    REQUIRE(cfg.observations_dir == "tests/data/openpose");
    REQUIRE(cfg.person_id == 0);
    REQUIRE(cfg.active_joint_groups.empty());

    // Tracking defaults
    REQUIRE(cfg.process_noise_std == Approx(0.5));
    REQUIRE(cfg.measurement_noise_std == Approx(2.0));
    REQUIRE(cfg.outlier_threshold == Approx(4.0));

    // UKF defaults
    REQUIRE(cfg.ukf_alpha == Approx(0.5));
    REQUIRE(cfg.ukf_beta == Approx(2.0));
    REQUIRE(cfg.ukf_kappa == Approx(0.0));

    // Hierarchical defaults: disabled, no children
    REQUIRE_FALSE(cfg.hierarchical.enabled);
    REQUIRE(cfg.hierarchical.children.empty());
}

// ---------------------------------------------------------------------------
// Tests: hierarchical config loading
// ---------------------------------------------------------------------------

TEST_CASE("TrackerAppConfig loads hierarchical section", "[config]") {
    auto cfg = TrackerAppConfig::load("tests/data/hierarchical_config_test.toml");

    SECTION("Base parameters loaded correctly") {
        REQUIRE(cfg.person_id == 1);
        REQUIRE(cfg.active_joint_groups == std::vector<std::string>{"main", "HandL", "HandR"});
        REQUIRE(cfg.process_noise_std == Approx(0.4));
        REQUIRE(cfg.measurement_noise_std == Approx(3.0));
        REQUIRE(cfg.outlier_threshold == Approx(5.0));
    }

    SECTION("UKF params loaded correctly") {
        REQUIRE(cfg.ukf_alpha == Approx(0.3));
        REQUIRE(cfg.tracker_fps == Approx(50.0));
        REQUIRE(cfg.start_time == Approx(1.0));
        REQUIRE(cfg.end_time == Approx(10.0));
    }

    SECTION("Hierarchical enabled flag") {
        REQUIRE(cfg.hierarchical.enabled);
        REQUIRE(cfg.hierarchical.enable_sync);
        REQUIRE_FALSE(cfg.hierarchical.sync_covariance);
    }

    SECTION("Parent filter parameters") {
        REQUIRE(cfg.hierarchical.parent_joint_groups == std::vector<std::string>{"main"});
        REQUIRE(cfg.hierarchical.parent_observation_groups == std::vector<std::string>{"main"});
        REQUIRE(cfg.hierarchical.parent_process_noise_std == Approx(0.5));
        REQUIRE(cfg.hierarchical.parent_measurement_noise_std == Approx(2.0));
        REQUIRE(cfg.hierarchical.parent_outlier_threshold == Approx(4.0));
    }

    SECTION("Two child filters loaded in order") {
        REQUIRE(cfg.hierarchical.children.size() == 2);

        auto const& handR = cfg.hierarchical.children[0];
        REQUIRE(handR.name == "HandR");
        REQUIRE(handR.joint_groups == std::vector<std::string>{"HandR"});
        REQUIRE(handR.observation_groups == std::vector<std::string>{"HandR"});
        REQUIRE(handR.process_noise_std == Approx(0.1));
        REQUIRE(handR.measurement_noise_std == Approx(1.5));
        REQUIRE(handR.outlier_threshold == Approx(3.5));
        REQUIRE(handR.min_inliers_ratio == Approx(0.4));
        REQUIRE(handR.max_innovation_norm == Approx(150.0));

        auto const& handL = cfg.hierarchical.children[1];
        REQUIRE(handL.name == "HandL");
        REQUIRE(handL.process_noise_std == Approx(0.15));
        REQUIRE(handL.min_inliers_ratio == Approx(0.35));
        REQUIRE(handL.max_innovation_norm == Approx(180.0));
    }
}

TEST_CASE("TrackerAppConfig hierarchical child defaults", "[config]") {
    // When optional child fields are absent, defaults should apply.
    // We do this by loading the minimal config (no hierarchical section)
    // and constructing a child manually.
    ChildFilterConfig child;

    // Verify struct defaults without touching the parser
    REQUIRE(child.process_noise_std == Approx(0.3));
    REQUIRE(child.measurement_noise_std == Approx(2.0));
    REQUIRE(child.outlier_threshold == Approx(4.0));
    REQUIRE(child.min_inliers_ratio == Approx(0.3));
    REQUIRE(child.max_innovation_norm == Approx(200.0));
}

// ---------------------------------------------------------------------------
// Tests: HierarchicalConfig defaults
// ---------------------------------------------------------------------------

TEST_CASE("HierarchicalConfig struct defaults", "[config]") {
    HierarchicalConfig h;

    REQUIRE_FALSE(h.enabled);
    REQUIRE(h.enable_sync);
    REQUIRE_FALSE(h.sync_covariance);
    REQUIRE(h.parent_joint_groups.empty());
    REQUIRE(h.parent_observation_groups.empty());
    REQUIRE(h.parent_process_noise_std == Approx(0.5));
    REQUIRE(h.parent_measurement_noise_std == Approx(2.0));
    REQUIRE(h.parent_outlier_threshold == Approx(4.0));
    REQUIRE(h.children.empty());
}

// ---------------------------------------------------------------------------
// Tests: validate() catches invalid hierarchical configs
// ---------------------------------------------------------------------------

TEST_CASE("validate() passes when hierarchical is disabled", "[config]") {
    auto cfg = make_valid_base_config();
    cfg.hierarchical.enabled = false;
    REQUIRE_NOTHROW(cfg.validate());
}

TEST_CASE("validate() rejects enabled hierarchical with empty parent groups", "[config]") {
    auto cfg = make_valid_base_config();
    cfg.hierarchical.enabled = true;
    // parent_joint_groups is empty by default

    REQUIRE_THROWS_AS(cfg.validate(), std::runtime_error);
}

TEST_CASE("validate() rejects enabled hierarchical with empty parent observation groups",
          "[config]") {
    auto cfg = make_valid_base_config();
    cfg.hierarchical.enabled = true;
    cfg.hierarchical.parent_joint_groups = {"main"};
    // parent_observation_groups is still empty

    REQUIRE_THROWS_AS(cfg.validate(), std::runtime_error);
}

TEST_CASE("validate() rejects child with empty name", "[config]") {
    auto cfg = make_valid_base_config();
    cfg.hierarchical.enabled = true;
    cfg.hierarchical.parent_joint_groups = {"main"};
    cfg.hierarchical.parent_observation_groups = {"main"};
    cfg.hierarchical.parent_process_noise_std = 0.5;

    ChildFilterConfig child;
    // child.name is empty by default
    child.joint_groups = {"HandR"};
    child.observation_groups = {"HandR"};
    cfg.hierarchical.children.push_back(child);

    REQUIRE_THROWS_AS(cfg.validate(), std::runtime_error);
}

TEST_CASE("validate() rejects child with empty joint_groups", "[config]") {
    auto cfg = make_valid_base_config();
    cfg.hierarchical.enabled = true;
    cfg.hierarchical.parent_joint_groups = {"main"};
    cfg.hierarchical.parent_observation_groups = {"main"};

    ChildFilterConfig child;
    child.name = "HandR";
    // joint_groups is empty
    child.observation_groups = {"HandR"};
    cfg.hierarchical.children.push_back(child);

    REQUIRE_THROWS_AS(cfg.validate(), std::runtime_error);
}

TEST_CASE("validate() rejects child with invalid min_inliers_ratio", "[config]") {
    auto cfg = make_valid_base_config();
    cfg.hierarchical.enabled = true;
    cfg.hierarchical.parent_joint_groups = {"main"};
    cfg.hierarchical.parent_observation_groups = {"main"};

    ChildFilterConfig child;
    child.name = "HandR";
    child.joint_groups = {"HandR"};
    child.observation_groups = {"HandR"};
    child.min_inliers_ratio = 1.5;  // > 1.0: invalid
    cfg.hierarchical.children.push_back(child);

    REQUIRE_THROWS_AS(cfg.validate(), std::runtime_error);
}

TEST_CASE("validate() accepts well-formed hierarchical config", "[config]") {
    auto cfg = make_valid_base_config();
    cfg.hierarchical.enabled = true;
    cfg.hierarchical.parent_joint_groups = {"main"};
    cfg.hierarchical.parent_observation_groups = {"main"};
    cfg.hierarchical.parent_process_noise_std = 0.5;

    ChildFilterConfig child;
    child.name = "HandR";
    child.joint_groups = {"HandR"};
    child.observation_groups = {"HandR"};
    child.process_noise_std = 0.1;
    child.min_inliers_ratio = 0.3;
    cfg.hierarchical.children.push_back(child);

    REQUIRE_NOTHROW(cfg.validate());
}

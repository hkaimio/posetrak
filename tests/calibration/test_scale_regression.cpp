//
// Regression guard for the scale calibration pipeline.
//
// These tests pin the numerical behaviour of check_scale_convergence() and
// write_calibrated_yaml() against known-good inputs and expected outputs.
// Any change to the computation logic that shifts the calibrated offsets by
// more than 1e-9 m will cause these tests to fail.
//
// The tests are deliberately data-driven (not just "does it compile") so that
// refactors in the YAML writer, CSV parser, or window logic are caught here.
//
#include <posetrak/calibration/scale_calibration.hpp>

#include <yaml-cpp/yaml.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <filesystem>
#include <fstream>

using namespace posetrak;
using Catch::Matchers::WithinAbs;

namespace {

std::filesystem::path temp_dir() {
    static std::filesystem::path d =
        std::filesystem::temp_directory_path() / "posetrak_scale_regression";
    std::filesystem::create_directories(d);
    return d;
}

// Write a synthetic state_vectors.csv with two scale groups whose values step
// from an initial level to a stable final level, mimicking a converging UKF.
//
//   Group "arm":     starts at 1.0, settles to 1.2 after 100 frames, held for 150 frames.
//   Group "forearm": starts at 1.0, settles to 0.85 after 100 frames, held for 150 frames.
//
// Total: 250 frames.  With window_frames=150 the expected final_scale values
// are exactly 1.2 and 0.85.
std::filesystem::path write_regression_csv() {
    auto path = temp_dir() / "regression_input.csv";
    std::ofstream f(path);

    f << "tracker_frame_idx,timestamp,root_position_x,"
         "scale_group_arm,scale_group_arm_velocity,"
         "scale_group_forearm,scale_group_forearm_velocity\n";

    for (int i = 0; i < 250; ++i) {
        double arm = (i < 100) ? (1.0 + 0.2 * i / 100.0) : 1.2;
        double forearm = (i < 100) ? (1.0 - 0.15 * i / 100.0) : 0.85;
        f << i << "," << i * 0.008 << ",0," << arm << ",0," << forearm << ",0\n";
    }
    return path;
}

}  // namespace

// ---------------------------------------------------------------------------
// Pin the convergence results from the synthetic calibration sequence
// ---------------------------------------------------------------------------

TEST_CASE("Regression: check_scale_convergence produces stable results", "[scale_regression]") {
    auto csv = write_regression_csv();

    ScaleCalibrationOptions opts;
    opts.window_frames = 150;
    opts.converge_std = 0.005;

    auto results = check_scale_convergence(csv.string(), opts);

    REQUIRE(results.size() == 2);

    // Results are alphabetically sorted: arm, forearm
    REQUIRE(results[0].name == "arm");
    REQUIRE(results[1].name == "forearm");

    SECTION("arm group: final scale and convergence") {
        CHECK_THAT(results[0].final_scale, WithinAbs(1.2, 1e-9));
        CHECK_THAT(results[0].scale_std, WithinAbs(0.0, 1e-9));
        CHECK(results[0].converged);
    }

    SECTION("forearm group: final scale and convergence") {
        CHECK_THAT(results[1].final_scale, WithinAbs(0.85, 1e-9));
        CHECK_THAT(results[1].scale_std, WithinAbs(0.0, 1e-9));
        CHECK(results[1].converged);
    }
}

// ---------------------------------------------------------------------------
// Pin the calibrated YAML offsets that result from the above scale factors
// applied to the fixture skeleton
// ---------------------------------------------------------------------------

TEST_CASE("Regression: write_calibrated_yaml produces stable offsets", "[scale_regression]") {
    // Use the stable final values from the regression CSV above
    std::vector<ScaleGroupResult> results = {
        {"arm", 1.2, 0.0, true},
        {"forearm", 0.85, 0.0, true},
    };

    auto out = temp_dir() / "regression_output.yaml";
    write_calibrated_yaml("tests/data/scale_group_test.yaml", out.string(), results);

    REQUIRE(std::filesystem::exists(out));
    auto doc = YAML::LoadFile(out.string());

    // Fixture offsets and bone_tip_offsets:
    //   root:      offset [0,0,0] (root joint); bone_tip_offset [0.3,0,0] * 1.2 → [0.36, 0, 0]
    //   upper_arm: offset [0.3, 0.0, 0.0]  * 1.2  → [0.36, 0.0, 0.0]
    //              bone_tip_offset [0.25, 0, 0] * 0.85 → [0.2125, 0, 0]
    //   lower_arm: offset [0.25, 0.0, 0.0] * 0.85 → [0.2125, 0.0, 0.0]
    //   spine:     offset [0.0, 0.4, 0.0]  — unscaled
    //              bone_tip_offset [0.0, 0.4, 0.0] — spine has no scaled children → unchanged

    bool found_root = false;
    bool found_upper_arm = false;
    bool found_lower_arm = false;
    bool found_spine = false;

    for (auto const& j : doc["joints"]) {
        std::string name = j["name"].as<std::string>();
        if (name == "root") {
            found_root = true;
            REQUIRE(j["bone_tip_offset"]);
            CHECK_THAT(j["bone_tip_offset"][0].as<double>(), WithinAbs(0.3 * 1.2, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][1].as<double>(), WithinAbs(0.0, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][2].as<double>(), WithinAbs(0.0, 1e-9));
        } else if (name == "upper_arm") {
            found_upper_arm = true;
            CHECK_THAT(j["offset"][0].as<double>(), WithinAbs(0.3 * 1.2, 1e-9));
            CHECK_THAT(j["offset"][1].as<double>(), WithinAbs(0.0, 1e-9));
            CHECK_THAT(j["offset"][2].as<double>(), WithinAbs(0.0, 1e-9));
            REQUIRE(j["bone_tip_offset"]);
            CHECK_THAT(j["bone_tip_offset"][0].as<double>(), WithinAbs(0.25 * 0.85, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][1].as<double>(), WithinAbs(0.0, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][2].as<double>(), WithinAbs(0.0, 1e-9));
        } else if (name == "lower_arm") {
            found_lower_arm = true;
            CHECK_THAT(j["offset"][0].as<double>(), WithinAbs(0.25 * 0.85, 1e-9));
            CHECK_THAT(j["offset"][1].as<double>(), WithinAbs(0.0, 1e-9));
            CHECK_THAT(j["offset"][2].as<double>(), WithinAbs(0.0, 1e-9));
        } else if (name == "spine") {
            found_spine = true;
            CHECK_THAT(j["offset"][0].as<double>(), WithinAbs(0.0, 1e-9));
            CHECK_THAT(j["offset"][1].as<double>(), WithinAbs(0.4, 1e-9));
            CHECK_THAT(j["offset"][2].as<double>(), WithinAbs(0.0, 1e-9));
            REQUIRE(j["bone_tip_offset"]);
            CHECK_THAT(j["bone_tip_offset"][0].as<double>(), WithinAbs(0.0, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][1].as<double>(), WithinAbs(0.4, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][2].as<double>(), WithinAbs(0.0, 1e-9));
        }
    }

    CHECK(found_root);
    CHECK(found_upper_arm);
    CHECK(found_lower_arm);
    CHECK(found_spine);

    // scale_groups must be absent in calibrated output
    CHECK_FALSE(doc["scale_groups"]);

    // Structural counts must be preserved
    CHECK(doc["joints"].size() == 4);
    CHECK(doc["markers"].size() == 2);
}

// ---------------------------------------------------------------------------
// Pipeline round-trip: convergence check followed by YAML write
// ---------------------------------------------------------------------------

TEST_CASE("Regression: full pipeline round-trip is deterministic", "[scale_regression]") {
    auto csv = write_regression_csv();

    ScaleCalibrationOptions opts;
    opts.window_frames = 150;

    // Run twice to confirm pure-function behaviour
    auto r1 = check_scale_convergence(csv.string(), opts);
    auto r2 = check_scale_convergence(csv.string(), opts);

    REQUIRE(r1.size() == r2.size());
    for (size_t i = 0; i < r1.size(); ++i) {
        CHECK(r1[i].name == r2[i].name);
        CHECK(r1[i].final_scale == r2[i].final_scale);
        CHECK(r1[i].scale_std == r2[i].scale_std);
        CHECK(r1[i].converged == r2[i].converged);
    }

    auto out1 = temp_dir() / "roundtrip1.yaml";
    auto out2 = temp_dir() / "roundtrip2.yaml";
    write_calibrated_yaml("tests/data/scale_group_test.yaml", out1.string(), r1);
    write_calibrated_yaml("tests/data/scale_group_test.yaml", out2.string(), r2);

    auto doc1 = YAML::LoadFile(out1.string());
    auto doc2 = YAML::LoadFile(out2.string());

    // Both outputs must agree on every scaled joint offset
    for (std::size_t ji = 0; ji < doc1["joints"].size(); ++ji) {
        auto const& j1 = doc1["joints"][ji];
        auto const& j2 = doc2["joints"][ji];
        REQUIRE(j1["name"].as<std::string>() == j2["name"].as<std::string>());
        for (std::size_t k = 0; k < 3; ++k) {
            CHECK(j1["offset"][k].as<double>() == j2["offset"][k].as<double>());
        }
    }
}

// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

//
// Unit tests for the scale calibration library.
//
// Coverage:
//   - check_scale_convergence(): CSV parsing, windowing, converged/not-converged flags
//   - write_calibrated_yaml():   offset scaling, scale_groups removal, field preservation
//
#include <posetrak/calibration/scale_calibration.hpp>

#include <yaml-cpp/yaml.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>

using namespace posetrak;
using Catch::Matchers::WithinAbs;

namespace {

std::filesystem::path temp_dir() {
    static std::filesystem::path d =
        std::filesystem::temp_directory_path() / "posetrak_scale_tests";
    std::filesystem::create_directories(d);
    return d;
}

// Write a minimal state_vectors.csv with the given scale columns and
// rows of data.  Every non-scale column is filled with a placeholder value.
//
// scale_values: map from group_name -> vector of per-frame values.
// All vectors must have the same length.
//
// Returns the path to the written file.
std::filesystem::path
write_synthetic_csv(std::string const& filename,
                    std::vector<std::pair<std::string, std::vector<double>>> const& scale_columns,
                    int extra_rows = 0  // extra rows before the scale values, all zeros
) {
    auto path = temp_dir() / filename;
    std::ofstream f(path);
    REQUIRE(f.is_open());

    // Header
    f << "tracker_frame_idx,timestamp,root_position_x,root_position_y,root_position_z";
    for (auto const& [name, _] : scale_columns) {
        f << ",scale_group_" << name;
        f << ",scale_group_" << name << "_velocity";
    }
    f << "\n";

    int n_frames = scale_columns.empty() ? 0 : (int)scale_columns[0].second.size();

    // Optional dummy rows before the actual data
    for (int r = 0; r < extra_rows; ++r) {
        f << r << ",0,0,0,0";
        for (size_t c = 0; c < scale_columns.size(); ++c)
            f << ",1.0,0.0";
        f << "\n";
    }

    // Actual scale data rows
    for (int r = 0; r < n_frames; ++r) {
        f << (extra_rows + r) << "," << (extra_rows + r) * 0.008 << ",0,0,0";
        for (auto const& [name, vals] : scale_columns) {
            f << "," << vals[r] << ",0.0";
        }
        f << "\n";
    }
    return path;
}

}  // namespace

// ---------------------------------------------------------------------------
// check_scale_convergence
// ---------------------------------------------------------------------------

TEST_CASE("check_scale_convergence detects columns from header", "[scale_calibration]") {
    // Two scale groups: "arm" (converged) and "leg" (not converged).
    // arm:  constant 0.9 → std = 0, converged.
    // leg:  linearly varying from 0.8 to 1.2 → large std, not converged.

    int const N = 120;
    std::vector<double> arm_vals(N, 0.9);
    std::vector<double> leg_vals(N);
    for (int i = 0; i < N; ++i)
        leg_vals[i] = 0.8 + 0.4 * i / (N - 1);

    auto path = write_synthetic_csv("col_detect.csv", {{"arm", arm_vals}, {"leg", leg_vals}});

    ScaleCalibrationOptions opts;
    opts.window_frames = N;
    opts.converge_std = 0.005;

    auto results = check_scale_convergence(path.string(), opts);

    REQUIRE(results.size() == 2);

    // Results are sorted alphabetically
    REQUIRE(results[0].name == "arm");
    REQUIRE(results[1].name == "leg");

    SECTION("arm group converges") {
        CHECK_THAT(results[0].final_scale, WithinAbs(0.9, 1e-9));
        CHECK_THAT(results[0].scale_std, WithinAbs(0.0, 1e-9));
        CHECK(results[0].converged);
    }

    SECTION("leg group does not converge") {
        // Mean of linearly-spaced [0.8..1.2] = 1.0
        CHECK_THAT(results[1].final_scale, WithinAbs(1.0, 1e-6));
        CHECK(results[1].scale_std > 0.005);
        CHECK_FALSE(results[1].converged);
    }
}

TEST_CASE("check_scale_convergence respects window_frames", "[scale_calibration]") {
    // 200 rows: first 100 rows vary wildly; last 100 rows are constant 0.85.
    // With window_frames=100 the function should see only the last 100 rows → converged.
    int const N = 200;
    std::vector<double> vals(N);
    for (int i = 0; i < 100; ++i)
        vals[i] = (i % 2 == 0) ? 0.5 : 1.5;  // noisy
    for (int i = 100; i < N; ++i)
        vals[i] = 0.85;  // stable

    auto path = write_synthetic_csv("window.csv", {{"bone", vals}});

    ScaleCalibrationOptions opts;
    opts.window_frames = 100;
    opts.converge_std = 0.005;

    auto results = check_scale_convergence(path.string(), opts);
    REQUIRE(results.size() == 1);

    CHECK_THAT(results[0].final_scale, WithinAbs(0.85, 1e-9));
    CHECK_THAT(results[0].scale_std, WithinAbs(0.0, 1e-9));
    CHECK(results[0].converged);
}

TEST_CASE("check_scale_convergence ignores velocity columns", "[scale_calibration]") {
    // velocity columns (scale_group_X_velocity) must not appear as separate groups
    auto path = write_synthetic_csv("velocity_cols.csv", {{"arm", std::vector<double>(50, 1.0)}});

    auto results = check_scale_convergence(path.string());

    // Should only have one result "arm", not "arm_velocity"
    REQUIRE(results.size() == 1);
    CHECK(results[0].name == "arm");
}

TEST_CASE("check_scale_convergence throws on missing file", "[scale_calibration]") {
    CHECK_THROWS(check_scale_convergence("/tmp/definitely_does_not_exist_xyz.csv"));
}

TEST_CASE("check_scale_convergence throws when no scale columns present", "[scale_calibration]") {
    auto path = temp_dir() / "no_scale.csv";
    std::ofstream f(path);
    f << "tracker_frame_idx,timestamp,root_position_x\n";
    f << "0,0.0,1.0\n";

    CHECK_THROWS(check_scale_convergence(path.string()));
}

TEST_CASE("check_scale_convergence handles window larger than available frames",
          "[scale_calibration]") {
    // Only 10 frames but window_frames=100 — should use all available frames
    std::vector<double> vals(10, 1.2);
    auto path = write_synthetic_csv("short.csv", {{"bone", vals}});

    ScaleCalibrationOptions opts;
    opts.window_frames = 100;
    auto results = check_scale_convergence(path.string(), opts);

    REQUIRE(results.size() == 1);
    CHECK_THAT(results[0].final_scale, WithinAbs(1.2, 1e-9));
    CHECK(results[0].converged);
}

// ---------------------------------------------------------------------------
// write_calibrated_yaml
// ---------------------------------------------------------------------------

namespace {

// Load the scale_group_test.yaml fixture and apply the given scale results,
// writing to a temp file.  Returns the path to the output.
std::filesystem::path run_write_calibrated_yaml(std::vector<ScaleGroupResult> const& results,
                                                std::string const& out_name) {
    auto out = temp_dir() / out_name;
    write_calibrated_yaml("tests/data/scale_group_test.yaml", out.string(), results);
    return out;
}

}  // namespace

TEST_CASE("write_calibrated_yaml applies scale to joint offsets", "[scale_calibration]") {
    // upper_arm has offset [0.3, 0.0, 0.0] — scale_group "arm" with scale 1.5
    // → expected calibrated offset magnitude = 0.3 * 1.5 = 0.45
    // lower_arm has offset [0.25, 0.0, 0.0] — scale_group "forearm" with scale 0.8
    // → expected = 0.25 * 0.8 = 0.20

    std::vector<ScaleGroupResult> results = {
        {"arm", 1.5, 0.001, true},
        {"forearm", 0.8, 0.001, true},
    };
    auto out = run_write_calibrated_yaml(results, "scaled.yaml");

    REQUIRE(std::filesystem::exists(out));
    auto doc = YAML::LoadFile(out.string());
    auto joints = doc["joints"];
    REQUIRE(joints.IsSequence());

    // Find upper_arm — should be scaled by 1.5
    bool found_upper_arm = false;
    bool found_lower_arm = false;
    bool found_spine = false;
    for (auto const& j : joints) {
        std::string name = j["name"].as<std::string>();
        if (name == "upper_arm") {
            found_upper_arm = true;
            auto off = j["offset"];
            CHECK_THAT(off[0].as<double>(), WithinAbs(0.3 * 1.5, 1e-9));
            CHECK_THAT(off[1].as<double>(), WithinAbs(0.0, 1e-9));
            CHECK_THAT(off[2].as<double>(), WithinAbs(0.0, 1e-9));
        } else if (name == "lower_arm") {
            found_lower_arm = true;
            auto off = j["offset"];
            CHECK_THAT(off[0].as<double>(), WithinAbs(0.25 * 0.8, 1e-9));
        } else if (name == "spine") {
            found_spine = true;
            // spine is not in any scale group — offset must be unchanged
            auto off = j["offset"];
            CHECK_THAT(off[1].as<double>(), WithinAbs(0.4, 1e-9));
        }
    }
    REQUIRE(found_upper_arm);
    REQUIRE(found_lower_arm);
    REQUIRE(found_spine);
}

TEST_CASE("write_calibrated_yaml removes scale_groups section", "[scale_calibration]") {
    std::vector<ScaleGroupResult> results = {
        {"arm", 1.0, 0.0, true},
        {"forearm", 1.0, 0.0, true},
    };
    auto out = run_write_calibrated_yaml(results, "no_scale_groups.yaml");

    auto doc = YAML::LoadFile(out.string());
    CHECK_FALSE(doc["scale_groups"]);
}

TEST_CASE("write_calibrated_yaml preserves non-calibrated fields verbatim", "[scale_calibration]") {
    std::vector<ScaleGroupResult> results = {
        {"arm", 1.0, 0.0, true},
        {"forearm", 1.0, 0.0, true},
    };
    auto out = run_write_calibrated_yaml(results, "preserved.yaml");

    auto doc = YAML::LoadFile(out.string());

    // Top-level keys present
    CHECK(doc["joints"]);
    CHECK(doc["markers"]);
    CHECK(doc["groups"]);

    // Marker count unchanged
    CHECK(doc["markers"].size() == 2);

    // Joint count unchanged (prismatic joints were never written into the reference YAML)
    CHECK(doc["joints"].size() == 4);
}

TEST_CASE("write_calibrated_yaml identity scale leaves offsets unchanged", "[scale_calibration]") {
    // Scale = 1.0 for all groups → offsets should be bitwise identical to original
    std::vector<ScaleGroupResult> results = {
        {"arm", 1.0, 0.0, true},
        {"forearm", 1.0, 0.0, true},
    };
    auto out = run_write_calibrated_yaml(results, "identity.yaml");
    auto doc = YAML::LoadFile(out.string());

    for (auto const& j : doc["joints"]) {
        std::string name = j["name"].as<std::string>();
        if (name == "upper_arm") {
            CHECK_THAT(j["offset"][0].as<double>(), WithinAbs(0.3, 1e-9));
        } else if (name == "lower_arm") {
            CHECK_THAT(j["offset"][0].as<double>(), WithinAbs(0.25, 1e-9));
        } else if (name == "spine") {
            CHECK_THAT(j["offset"][1].as<double>(), WithinAbs(0.4, 1e-9));
        }
    }
}

TEST_CASE("write_calibrated_yaml handles missing scale group gracefully", "[scale_calibration]") {
    // Provide results for "arm" only — "forearm" in the YAML gets no entry.
    // Joints in "forearm" should retain their original offsets.
    std::vector<ScaleGroupResult> results = {
        {"arm", 2.0, 0.0, true},
    };
    auto out = run_write_calibrated_yaml(results, "partial.yaml");
    auto doc = YAML::LoadFile(out.string());

    for (auto const& j : doc["joints"]) {
        std::string name = j["name"].as<std::string>();
        if (name == "upper_arm") {
            CHECK_THAT(j["offset"][0].as<double>(), WithinAbs(0.3 * 2.0, 1e-9));
        } else if (name == "lower_arm") {
            // forearm group not in results — offset unchanged
            CHECK_THAT(j["offset"][0].as<double>(), WithinAbs(0.25, 1e-9));
        }
    }
}

TEST_CASE("write_calibrated_yaml throws on unwritable output path", "[scale_calibration]") {
    std::vector<ScaleGroupResult> results = {{"arm", 1.0, 0.0, true}};
    CHECK_THROWS(write_calibrated_yaml("tests/data/scale_group_test.yaml",
                                       "/no_such_dir/xyz/out.yaml", results));
}

// ---------------------------------------------------------------------------
// bone_tip_offset scaling
// ---------------------------------------------------------------------------

TEST_CASE("write_calibrated_yaml scales bone_tip_offset of parent joint", "[scale_calibration]") {
    // root's bone_tip_offset [0.3, 0, 0] matches upper_arm's offset → scale by arm factor (1.2).
    // upper_arm's bone_tip_offset [0.25, 0, 0] matches lower_arm's offset → scale by forearm
    // (0.85). spine's bone_tip_offset [0, 0.4, 0] — spine itself has no scaled children →
    // unchanged.
    std::vector<ScaleGroupResult> results = {
        {"arm", 1.2, 0.0, true},
        {"forearm", 0.85, 0.0, true},
    };
    auto out = temp_dir() / "bone_tip_test.yaml";
    write_calibrated_yaml("tests/data/scale_group_test.yaml", out.string(), results);

    auto doc = YAML::LoadFile(out.string());

    bool found_root = false, found_upper_arm = false, found_spine = false;
    for (auto const& j : doc["joints"]) {
        std::string const name = j["name"].as<std::string>();
        if (name == "root") {
            found_root = true;
            // bone_tip_offset should be [0.3 * 1.2, 0, 0] = [0.36, 0, 0]
            REQUIRE(j["bone_tip_offset"]);
            CHECK_THAT(j["bone_tip_offset"][0].as<double>(), WithinAbs(0.3 * 1.2, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][1].as<double>(), WithinAbs(0.0, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][2].as<double>(), WithinAbs(0.0, 1e-9));
        } else if (name == "upper_arm") {
            found_upper_arm = true;
            // bone_tip_offset should be [0.25 * 0.85, 0, 0] = [0.2125, 0, 0]
            REQUIRE(j["bone_tip_offset"]);
            CHECK_THAT(j["bone_tip_offset"][0].as<double>(), WithinAbs(0.25 * 0.85, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][1].as<double>(), WithinAbs(0.0, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][2].as<double>(), WithinAbs(0.0, 1e-9));
        } else if (name == "spine") {
            found_spine = true;
            // spine has no scaled children → bone_tip_offset unchanged
            REQUIRE(j["bone_tip_offset"]);
            CHECK_THAT(j["bone_tip_offset"][0].as<double>(), WithinAbs(0.0, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][1].as<double>(), WithinAbs(0.4, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][2].as<double>(), WithinAbs(0.0, 1e-9));
        }
    }
    CHECK(found_root);
    CHECK(found_upper_arm);
    CHECK(found_spine);
}

TEST_CASE("write_calibrated_yaml bone_tip_offset: no close match leaves unchanged",
          "[scale_calibration]") {
    // Inline YAML: one parent with bone_tip pointing in a direction that doesn't
    // closely match any scaled child's offset. bone_tip_offset must not be modified.
    std::string const fixture = R"yaml(
name: "no_match_test"
joints:
  - name: root
    type: root
    parent: null
    offset: [0.0, 0.0, 0.0]
    bone_tip_offset: [0.0, 0.5, 0.0]   # points +Y, 0.5 m

  - name: arm
    type: revolute
    parent: root
    offset: [0.3, 0.0, 0.0]   # +X, 0.3 m — far from bone_tip
    axis: [1.0, 0.0, 0.0]
    limits: [-1.5, 1.5]
scale_groups:
  - name: arm_grp
    joints: [arm]
)yaml";

    auto yaml_path = temp_dir() / "no_match_input.yaml";
    {
        std::ofstream f(yaml_path);
        f << fixture;
    }

    std::vector<ScaleGroupResult> results = {{"arm_grp", 1.5, 0.0, true}};
    auto out = temp_dir() / "no_match_output.yaml";
    write_calibrated_yaml(yaml_path.string(), out.string(), results);

    auto doc = YAML::LoadFile(out.string());
    for (auto const& j : doc["joints"]) {
        if (j["name"].as<std::string>() == "root") {
            REQUIRE(j["bone_tip_offset"]);
            // Unchanged: root's bone_tip [0, 0.5, 0] is 0.583 m from arm's offset [0.3, 0, 0]
            CHECK_THAT(j["bone_tip_offset"][0].as<double>(), WithinAbs(0.0, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][1].as<double>(), WithinAbs(0.5, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][2].as<double>(), WithinAbs(0.0, 1e-9));
        }
    }
}

TEST_CASE("write_calibrated_yaml bone_tip_offset: multi-child picks only matching child",
          "[scale_calibration]") {
    // hub has two scaled children in different directions.
    // bone_tip_offset matches arm.L; arm.R scale must NOT be applied.
    std::string const fixture = R"yaml(
name: "multi_child_test"
joints:
  - name: hub
    type: root
    parent: null
    offset: [0.0, 0.0, 0.0]
    bone_tip_offset: [0.2, 0.0, 0.0]   # matches arm.L (offset +X 0.2)

  - name: arm.L
    type: revolute
    parent: hub
    offset: [0.2, 0.0, 0.0]
    axis: [1.0, 0.0, 0.0]
    limits: [-1.5, 1.5]

  - name: arm.R
    type: revolute
    parent: hub
    offset: [-0.2, 0.0, 0.0]
    axis: [1.0, 0.0, 0.0]
    limits: [-1.5, 1.5]
scale_groups:
  - name: left
    joints: [arm.L]
  - name: right
    joints: [arm.R]
)yaml";

    auto yaml_path = temp_dir() / "multi_child_input.yaml";
    {
        std::ofstream f(yaml_path);
        f << fixture;
    }

    // arm.L scale=1.3, arm.R scale=0.8 — very different, easy to distinguish
    std::vector<ScaleGroupResult> results = {
        {"left", 1.3, 0.0, true},
        {"right", 0.8, 0.0, true},
    };
    auto out = temp_dir() / "multi_child_output.yaml";
    write_calibrated_yaml(yaml_path.string(), out.string(), results);

    auto doc = YAML::LoadFile(out.string());
    bool found_hub = false;
    for (auto const& j : doc["joints"]) {
        if (j["name"].as<std::string>() == "hub") {
            found_hub = true;
            REQUIRE(j["bone_tip_offset"]);
            // bone_tip should be scaled by left (1.3), NOT right (0.8)
            CHECK_THAT(j["bone_tip_offset"][0].as<double>(), WithinAbs(0.2 * 1.3, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][1].as<double>(), WithinAbs(0.0, 1e-9));
            CHECK_THAT(j["bone_tip_offset"][2].as<double>(), WithinAbs(0.0, 1e-9));
        }
        if (j["name"].as<std::string>() == "arm.L") {
            CHECK_THAT(j["offset"][0].as<double>(), WithinAbs(0.2 * 1.3, 1e-9));
        }
        if (j["name"].as<std::string>() == "arm.R") {
            CHECK_THAT(j["offset"][0].as<double>(), WithinAbs(-0.2 * 0.8, 1e-9));
        }
    }
    CHECK(found_hub);
}

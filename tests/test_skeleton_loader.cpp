#include <posetrak/io/skeleton_loader.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>

using namespace posetrak;

namespace {
// Get temporary directory for test files
std::filesystem::path get_temp_test_dir() {
    static std::filesystem::path temp_dir =
        std::filesystem::temp_directory_path() / "posetrak_tests";
    std::filesystem::create_directories(temp_dir);
    return temp_dir;
}
}  // namespace

TEST_CASE("Load simple humanoid skeleton", "[skeleton_loader]") {
    Skeleton skeleton = load_skeleton_from_yaml("tests/data/simple_humanoid.yaml");

    SECTION("Basic skeleton properties") {
        auto const& joints = skeleton.joints();
        auto const& markers = skeleton.markers();

        // Simple humanoid has: pelvis, 3 spine, 2 shoulders, 2 elbows, 2 wrists = 10 joints
        REQUIRE(joints.size() == 10);
        REQUIRE(markers.size() == 8);
    }

    SECTION("Joint types and hierarchy") {
        auto const& joints = skeleton.joints();

        // Find pelvis (root)
        auto pelvis_it = std::find_if(joints.begin(), joints.end(), [](auto const& pair) {
            return pair.second.name == "pelvis";
        });
        REQUIRE(pelvis_it != joints.end());
        REQUIRE(pelvis_it->second.parent.empty());  // Root has no parent
        REQUIRE(pelvis_it->second.type == JointType::FIXED);

        // Find spine joints (ball/spherical)
        auto spine_lower_it = std::find_if(joints.begin(), joints.end(), [](auto const& pair) {
            return pair.second.name == "spine_lower";
        });
        REQUIRE(spine_lower_it != joints.end());
        REQUIRE(spine_lower_it->second.parent == "pelvis");
        REQUIRE(spine_lower_it->second.type == JointType::SPHERICAL);
        REQUIRE(spine_lower_it->second.dof == 3);

        // Find elbow (revolute)
        auto r_elbow_it = std::find_if(joints.begin(), joints.end(), [](auto const& pair) {
            return pair.second.name == "r_elbow";
        });
        REQUIRE(r_elbow_it != joints.end());
        REQUIRE(r_elbow_it->second.parent == "r_shoulder");
        REQUIRE(r_elbow_it->second.type == JointType::REVOLUTE);
        REQUIRE(r_elbow_it->second.dof == 1);
    }

    SECTION("Joint limits") {
        auto const& joints = skeleton.joints();

        // Check revolute joint limits
        auto r_elbow_it = std::find_if(joints.begin(), joints.end(), [](auto const& pair) {
            return pair.second.name == "r_elbow";
        });
        REQUIRE(r_elbow_it != joints.end());
        REQUIRE(r_elbow_it->second.num_limits == 1);
        REQUIRE_THAT(r_elbow_it->second.limits[0][0], Catch::Matchers::WithinRel(0.0, 1e-6));
        REQUIRE_THAT(r_elbow_it->second.limits[0][1], Catch::Matchers::WithinRel(2.8, 1e-6));

        // Check spherical joint limits (3 axes)
        auto r_shoulder_it = std::find_if(joints.begin(), joints.end(), [](auto const& pair) {
            return pair.second.name == "r_shoulder";
        });
        REQUIRE(r_shoulder_it != joints.end());
        REQUIRE(r_shoulder_it->second.num_limits == 3);
        REQUIRE_THAT(r_shoulder_it->second.limits[0][0], Catch::Matchers::WithinRel(-3.0, 1e-6));
        REQUIRE_THAT(r_shoulder_it->second.limits[0][1], Catch::Matchers::WithinRel(3.0, 1e-6));
        REQUIRE_THAT(r_shoulder_it->second.limits[1][0], Catch::Matchers::WithinRel(-1.5, 1e-6));
        REQUIRE_THAT(r_shoulder_it->second.limits[1][1], Catch::Matchers::WithinRel(1.5, 1e-6));
        REQUIRE_THAT(r_shoulder_it->second.limits[2][0], Catch::Matchers::WithinRel(-1.0, 1e-6));
        REQUIRE_THAT(r_shoulder_it->second.limits[2][1], Catch::Matchers::WithinRel(1.0, 1e-6));
    }

    SECTION("Joint offsets") {
        auto const& joints = skeleton.joints();

        auto r_shoulder_it = std::find_if(joints.begin(), joints.end(), [](auto const& pair) {
            return pair.second.name == "r_shoulder";
        });
        REQUIRE(r_shoulder_it != joints.end());

        // Right shoulder offset from spine_upper
        REQUIRE_THAT(r_shoulder_it->second.offset[0], Catch::Matchers::WithinRel(0.15, 1e-6));
        REQUIRE_THAT(r_shoulder_it->second.offset[1], Catch::Matchers::WithinRel(0.05, 1e-6));
        REQUIRE_THAT(r_shoulder_it->second.offset[2], Catch::Matchers::WithinRel(0.0, 1e-6));
    }

    SECTION("Markers") {
        auto const& markers = skeleton.markers();

        // Find pelvis marker
        auto pelvis_marker_it = std::find_if(markers.begin(), markers.end(), [](auto const& pair) {
            return pair.second.name == "pelvis_center";
        });
        REQUIRE(pelvis_marker_it != markers.end());
        REQUIRE(pelvis_marker_it->second.joint == "pelvis");
        REQUIRE(pelvis_marker_it->second.coco_id.has_value());
        REQUIRE(*pelvis_marker_it->second.coco_id == 8);

        // Find shoulder marker
        auto r_shoulder_marker_it =
            std::find_if(markers.begin(), markers.end(),
                         [](auto const& pair) { return pair.second.name == "r_shoulder_marker"; });
        REQUIRE(r_shoulder_marker_it != markers.end());
        REQUIRE(r_shoulder_marker_it->second.joint == "r_shoulder");
        REQUIRE(r_shoulder_marker_it->second.coco_id.has_value());
        REQUIRE(*r_shoulder_marker_it->second.coco_id == 2);
    }

    SECTION("Active joints from groups") {
        // Check that active joints were set from tracking.active_groups
        // Simple humanoid has groups: core, right_arm, left_arm - all 10 joints should be active
        auto const& joints = skeleton.joints();
        int active_count = 0;
        for (auto const& [name, joint] : joints) {
            if (skeleton.is_joint_active(name)) {
                active_count++;
            }
        }
        REQUIRE(active_count == 10);
    }
}

TEST_CASE("Load production Harri skeleton", "[skeleton_loader]") {
    Skeleton skeleton = load_skeleton_from_yaml("tests/data/Harri_skeleton-shouldery-rot.yaml");

    auto const& joints = skeleton.joints();
    auto const& markers = skeleton.markers();

    SECTION("Skeleton has expected counts") {
        // Production skeleton should have multiple joints and markers
        REQUIRE(joints.size() > 10);
        REQUIRE(markers.size() > 5);
    }

    SECTION("Hierarchy is valid") {
        // Find the root joint(s) - should have exactly one
        int root_count = 0;
        for (auto const& [name, joint] : joints) {
            if (joint.parent.empty()) {
                root_count++;
            }
        }
        REQUIRE(root_count == 1);

        // All non-root joints should have valid parents
        for (auto const& [name, joint] : joints) {
            if (!joint.parent.empty()) {
                REQUIRE(joints.find(joint.parent) != joints.end());
            }
        }
    }

    SECTION("Markers reference valid joints") {
        for (auto const& [name, marker] : markers) {
            REQUIRE(joints.find(marker.joint) != joints.end());
        }
    }

    SECTION("Spherical joints have 3D limits") {
        for (auto const& [name, joint] : joints) {
            if (joint.type == JointType::SPHERICAL) {
                REQUIRE(joint.num_limits == 3);
                REQUIRE(joint.dof == 3);
                // All limits should have min <= max (allow fixed axes with [0, 0])
                for (size_t i = 0; i < 3; ++i) {
                    REQUIRE(joint.limits[i][0] <= joint.limits[i][1]);
                }
            }
        }
    }

    SECTION("Revolute joints have 1D limits") {
        for (auto const& [name, joint] : joints) {
            if (joint.type == JointType::REVOLUTE) {
                REQUIRE(joint.num_limits == 1);
                REQUIRE(joint.dof == 1);
                REQUIRE(joint.limits[0][0] < joint.limits[0][1]);
            }
        }
    }

    SECTION("Skeleton is valid") {
        // Should not throw
        REQUIRE_NOTHROW(skeleton.validate());
    }
}

TEST_CASE("Skeleton loader error handling", "[skeleton_loader][errors]") {
    SECTION("Non-existent file throws") {
        REQUIRE_THROWS_AS(load_skeleton_from_yaml("tests/data/nonexistent.yaml"),
                          std::runtime_error);
    }

    SECTION("Invalid YAML syntax throws") {
        auto test_file = get_temp_test_dir() / "invalid_syntax.yaml";
        std::ofstream f(test_file);
        f << "joints:\n  - name: test\n    invalid: [broken yaml";
        f.close();
        REQUIRE_THROWS_AS(load_skeleton_from_yaml(test_file.string()), std::runtime_error);
    }

    SECTION("Missing joints section throws") {
        auto test_file = get_temp_test_dir() / "missing_joints.yaml";
        std::ofstream f(test_file);
        f << "markers:\n  - name: test\n";
        f.close();
        REQUIRE_THROWS_AS(load_skeleton_from_yaml(test_file.string()), std::runtime_error);
    }

    SECTION("Unknown joint type throws") {
        auto test_file = get_temp_test_dir() / "unknown_joint_type.yaml";
        std::ofstream f(test_file);
        f << "joints:\n"
          << "  - name: root\n"
          << "    type: unknown_type\n"
          << "    offset: [0, 0, 0]\n";
        f.close();
        REQUIRE_THROWS_AS(load_skeleton_from_yaml(test_file.string()), std::runtime_error);
    }

    SECTION("Marker referencing unknown joint throws") {
        auto test_file = get_temp_test_dir() / "invalid_marker.yaml";
        std::ofstream f(test_file);
        f << "joints:\n"
          << "  - name: root\n"
          << "    type: root\n"
          << "    offset: [0, 0, 0]\n"
          << "markers:\n"
          << "  - name: test_marker\n"
          << "    parent: nonexistent_joint\n"
          << "    offset: [0, 0, 0]\n";
        f.close();
        REQUIRE_THROWS_AS(load_skeleton_from_yaml(test_file.string()), std::runtime_error);
    }

    SECTION("Spherical joint with incomplete limits throws") {
        auto test_file = get_temp_test_dir() / "incomplete_spherical_limits.yaml";
        std::ofstream f(test_file);
        f << "joints:\n"
          << "  - name: root\n"
          << "    type: root\n"
          << "    offset: [0, 0, 0]\n"
          << "  - name: shoulder\n"
          << "    type: ball\n"
          << "    parent: root\n"
          << "    offset: [0.15, 0, 0]\n"
          << "    limits:\n"
          << "      x: [-3, 3]\n"
          << "      y: [-1.5, 1.5]\n";  // Missing z
        f.close();
        REQUIRE_THROWS_AS(load_skeleton_from_yaml(test_file.string()), std::runtime_error);
    }
}

TEST_CASE("Skeleton loader handles optional fields", "[skeleton_loader]") {
    SECTION("Joint without limits gets defaults") {
        auto test_file = get_temp_test_dir() / "no_limits.yaml";
        std::ofstream f(test_file);
        f << "joints:\n"
          << "  - name: root\n"
          << "    type: revolute\n"
          << "    offset: [0, 0, 0]\n";
        f.close();

        Skeleton skeleton = load_skeleton_from_yaml(test_file.string());
        auto const& joint = skeleton.joints().at("root");
        REQUIRE(joint.num_limits == 1);
        REQUIRE_THAT(joint.limits[0][0], Catch::Matchers::WithinRel(-M_PI, 1e-6));
        REQUIRE_THAT(joint.limits[0][1], Catch::Matchers::WithinRel(M_PI, 1e-6));
    }

    SECTION("Joint without group field has empty string") {
        auto test_file = get_temp_test_dir() / "no_group.yaml";
        std::ofstream f(test_file);
        f << "joints:\n"
          << "  - name: root\n"
          << "    type: root\n"
          << "    offset: [0, 0, 0]\n";
        f.close();

        Skeleton skeleton = load_skeleton_from_yaml(test_file.string());
        REQUIRE(skeleton.joints().at("root").group.empty());
    }

    SECTION("Marker without COCO ID has nullopt") {
        auto test_file = get_temp_test_dir() / "no_coco_id.yaml";
        std::ofstream f(test_file);
        f << "joints:\n"
          << "  - name: root\n"
          << "    type: root\n"
          << "    offset: [0, 0, 0]\n"
          << "markers:\n"
          << "  - name: test_marker\n"
          << "    parent: root\n"
          << "    offset: [0, 0, 0]\n";
        f.close();

        Skeleton skeleton = load_skeleton_from_yaml(test_file.string());
        REQUIRE(!skeleton.markers().at("test_marker").coco_id.has_value());
    }

    SECTION("Skeleton without tracking section loads") {
        auto test_file = get_temp_test_dir() / "no_tracking.yaml";
        std::ofstream f(test_file);
        f << "joints:\n"
          << "  - name: root\n"
          << "    type: root\n"
          << "    offset: [0, 0, 0]\n";
        f.close();

        REQUIRE_NOTHROW(load_skeleton_from_yaml(test_file.string()));
    }
}

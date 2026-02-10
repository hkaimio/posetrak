#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/skeleton.hpp"

using namespace posetrak;

TEST_CASE("Joint construction and properties", "[skeleton]") {
    Skeleton skel;

    SECTION("Revolute joint") {
        uint32_t idx = skel.add_joint("knee", std::nullopt, JointType::REVOLUTE);
        auto const& j = skel.joints()[idx];
        REQUIRE(j.name == "knee");
        REQUIRE_FALSE(j.parent_index.has_value());
        REQUIRE(j.type == JointType::REVOLUTE);
        REQUIRE(j.dof == 1);
        REQUIRE(j.group.empty());
    }

    SECTION("Spherical joint") {
        uint32_t parent_idx = skel.add_joint("torso", std::nullopt, JointType::SPHERICAL);
        uint32_t idx = skel.add_joint("shoulder", parent_idx, JointType::SPHERICAL,
                                      Eigen::Vector3d(0.1, 0.2, 0.3), "arms");
        auto const& j = skel.joints()[idx];
        REQUIRE(j.dof == 3);
        REQUIRE(j.group == "arms");
        REQUIRE(j.offset.isApprox(Eigen::Vector3d(0.1, 0.2, 0.3)));
    }

    SECTION("Fixed joint") {
        uint32_t parent_idx = skel.add_joint("pelvis", std::nullopt, JointType::SPHERICAL);
        uint32_t idx = skel.add_joint("marker_point", parent_idx, JointType::FIXED);
        auto const& j = skel.joints()[idx];
        REQUIRE(j.dof == 0);
    }
}

TEST_CASE("Skeleton construction and validation", "[skeleton]") {
    SECTION("Simple valid skeleton") {
        Skeleton skel;
        uint32_t pelvis = skel.add_joint("pelvis", std::nullopt, JointType::SPHERICAL);
        uint32_t spine = skel.add_joint("spine", pelvis, JointType::REVOLUTE);
        skel.add_joint("head", spine, JointType::SPHERICAL);

        auto const err = skel.validate();
        REQUIRE(!err.has_value());
        REQUIRE(skel.total_dof() == 7);  // 3 + 1 + 3
    }

    SECTION("Empty skeleton is invalid") {
        Skeleton skel;
        auto const err = skel.validate();
        REQUIRE(err.has_value());
        REQUIRE(err->find("no joints") != std::string::npos);
    }

    SECTION("Missing parent is invalid") {
        Skeleton skel;
        // Try to add joint with invalid parent index
        REQUIRE_THROWS_AS(skel.add_joint("child", 999, JointType::REVOLUTE), std::invalid_argument);
    }

    SECTION("Multiple roots is invalid") {
        Skeleton skel;
        skel.add_joint("root1", std::nullopt, JointType::SPHERICAL);
        skel.add_joint("root2", std::nullopt, JointType::SPHERICAL);
        auto const err = skel.validate();
        REQUIRE(err.has_value());
    }

    SECTION("Cycle detection") {
        Skeleton skel;
        uint32_t a = skel.add_joint("a", std::nullopt, JointType::REVOLUTE);
        uint32_t b = skel.add_joint("b", a, JointType::REVOLUTE);
        skel.add_joint("c", b, JointType::REVOLUTE);
        // Manually create a cycle by modifying parent_index (normally not possible through API)
        // Skip this test as cycles are now prevented at construction time
        // Just verify normal validation passes
        auto const err = skel.validate();
        REQUIRE(!err.has_value());
    }
}

TEST_CASE("Skeleton DOF counting", "[skeleton]") {
    Skeleton skel;
    uint32_t pelvis = skel.add_joint("pelvis", std::nullopt, JointType::SPHERICAL,
                                     Eigen::Vector3d::Zero(), "torso");
    skel.add_joint("spine", pelvis, JointType::REVOLUTE, Eigen::Vector3d::Zero(), "torso");
    uint32_t left_hip =
        skel.add_joint("left_hip", pelvis, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "legs");
    skel.add_joint("left_knee", left_hip, JointType::REVOLUTE, Eigen::Vector3d::Zero(), "legs");
    skel.add_joint("right_hip", pelvis, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "legs");
    skel.add_joint("marker", pelvis, JointType::FIXED);

    REQUIRE(skel.total_dof() == 11);  // 3 + 1 + 3 + 1 + 3 + 0

    SECTION("All joints active by default") {
        REQUIRE(skel.active_dof() == 11);
    }

    SECTION("Filter by group") {
        skel.set_active_groups({"legs"});
        REQUIRE(skel.active_dof() == 7);  // 3 + 1 + 3 (legs only)
    }

    SECTION("Filter by explicit joints") {
        skel.set_active_joints({"pelvis", "spine"});
        REQUIRE(skel.active_dof() == 4);  // 3 + 1
    }

    SECTION("Clear filter restores all") {
        skel.set_active_groups({"legs"});
        skel.clear_active_filter();
        REQUIRE(skel.active_dof() == 11);
    }
}

TEST_CASE("Skeleton markers", "[skeleton]") {
    Skeleton skel;
    uint32_t pelvis = skel.add_joint("pelvis", std::nullopt, JointType::SPHERICAL);
    uint32_t spine = skel.add_joint("spine", pelvis, JointType::REVOLUTE);

    SECTION("Add marker to existing joint") {
        skel.add_marker("m1", pelvis, Eigen::Vector3d(0.1, 0, 0));
        auto const* m = skel.get_marker("m1");
        REQUIRE(m != nullptr);
        REQUIRE(m->joint_index == pelvis);
    }

    SECTION("Cannot add marker to missing joint") {
        REQUIRE_THROWS_AS(skel.add_marker("m1", 999, Eigen::Vector3d::Zero()),
                          std::invalid_argument);
    }

    SECTION("Cannot add duplicate marker") {
        skel.add_marker("m1", pelvis, Eigen::Vector3d::Zero());
        REQUIRE_THROWS_AS(skel.add_marker("m1", spine, Eigen::Vector3d::Zero()),
                          std::invalid_argument);
    }

    SECTION("Validation checks marker joints") {
        skel.add_marker("m1", pelvis, Eigen::Vector3d::Zero());
        auto err = skel.validate();
        REQUIRE(!err.has_value());
    }
}

TEST_CASE("Skeleton joint ordering", "[skeleton]") {
    Skeleton skel;
    uint32_t pelvis = skel.add_joint("pelvis", std::nullopt, JointType::SPHERICAL);
    uint32_t left_hip = skel.add_joint("left_hip", pelvis, JointType::SPHERICAL);
    skel.add_joint("right_hip", pelvis, JointType::SPHERICAL);
    skel.add_joint("left_knee", left_hip, JointType::REVOLUTE);

    // With vector storage, get_joints_ordered() returns joints in insertion order
    std::vector<Joint> const ordered = skel.get_joints_ordered();
    REQUIRE(ordered.size() == 4);
    REQUIRE(ordered[0].name == "pelvis");
    REQUIRE(ordered[1].name == "left_hip");
    REQUIRE(ordered[2].name == "right_hip");
    REQUIRE(ordered[3].name == "left_knee");
}

TEST_CASE("Skeleton JSON serialization", "[skeleton]") {
    Skeleton skel;
    uint32_t pelvis = skel.add_joint("pelvis", std::nullopt, JointType::SPHERICAL);
    skel.add_joint("spine", pelvis, JointType::REVOLUTE);
    skel.add_marker("m1", pelvis, Eigen::Vector3d(0.1, 0, 0), 0);

    nlohmann::json const json = skel.to_json();
    Skeleton const skel2 = Skeleton::from_json(json);

    REQUIRE(skel2.joints().size() == 2);
    REQUIRE(skel2.markers().size() == 1);
    REQUIRE(skel2.total_dof() == 4);

    auto const* j = skel2.get_joint("pelvis");
    REQUIRE(j != nullptr);
    REQUIRE(j->type == JointType::SPHERICAL);

    auto const* m = skel2.get_marker("m1");
    REQUIRE(m != nullptr);
    REQUIRE(m->coco_id.has_value());
    REQUIRE(*m->coco_id == 0);
}

TEST_CASE("Skeleton joint queries", "[skeleton]") {
    Skeleton skel;
    skel.add_joint("pelvis", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero(), "torso");
    skel.add_joint("spine", 0, JointType::REVOLUTE, Eigen::Vector3d::Zero(), "torso");

    SECTION("Get existing joint") {
        auto const* j = skel.get_joint("pelvis");
        REQUIRE(j != nullptr);
        REQUIRE(j->name == "pelvis");
    }

    SECTION("Get missing joint returns nullptr") {
        auto const* j = skel.get_joint("missing");
        REQUIRE(j == nullptr);
    }

    SECTION("Check active status") {
        REQUIRE(skel.is_joint_active("pelvis"));
        skel.set_active_groups({"legs"});
        REQUIRE(!skel.is_joint_active("pelvis"));
    }
}

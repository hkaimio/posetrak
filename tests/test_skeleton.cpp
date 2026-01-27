#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/skeleton.hpp"

using namespace posetrak;

TEST_CASE("Joint construction and properties", "[skeleton]") {
    SECTION("Revolute joint") {
        Joint j("knee", "thigh", JointType::REVOLUTE);
        REQUIRE(j.name == "knee");
        REQUIRE(j.parent == "thigh");
        REQUIRE(j.type == JointType::REVOLUTE);
        REQUIRE(j.dof == 1);
        REQUIRE(j.group.empty());
    }

    SECTION("Spherical joint") {
        Joint j("shoulder", "torso", JointType::SPHERICAL, Eigen::Vector3d(0.1, 0.2, 0.3), "arms");
        REQUIRE(j.dof == 3);
        REQUIRE(j.group == "arms");
        REQUIRE(j.offset.isApprox(Eigen::Vector3d(0.1, 0.2, 0.3)));
    }

    SECTION("Fixed joint") {
        Joint j("marker_point", "pelvis", JointType::FIXED);
        REQUIRE(j.dof == 0);
    }
}

TEST_CASE("Joint JSON serialization", "[skeleton]") {
    Joint j("hip", "pelvis", JointType::REVOLUTE, Eigen::Vector3d(0.1, 0, 0), "legs");
    j.limits = {Eigen::Vector2d(-1.5, 1.5)};

    nlohmann::json const json = j.to_json();
    Joint const j2 = Joint::from_json(json);

    REQUIRE(j2.name == j.name);
    REQUIRE(j2.parent == j.parent);
    REQUIRE(j2.type == j.type);
    REQUIRE(j2.dof == j.dof);
    REQUIRE(j2.group == j.group);
    REQUIRE(j2.offset.isApprox(j.offset));
    REQUIRE(j2.num_limits == j.num_limits);
    for (size_t i = 0; i < j.num_limits; ++i) {
        REQUIRE(j2.limits[i].isApprox(j.limits[i]));
    }
}

TEST_CASE("Spherical joint with 3D limits", "[skeleton]") {
    Joint j("shoulder", "spine", JointType::SPHERICAL, Eigen::Vector3d(0.15, 0.05, 0), "arms");
    j.limits[0] = Eigen::Vector2d(-3.0, 3.0);  // X-axis rotation
    j.limits[1] = Eigen::Vector2d(-1.5, 1.5);  // Y-axis rotation
    j.limits[2] = Eigen::Vector2d(-1.0, 1.0);  // Z-axis rotation
    j.num_limits = 3;

    REQUIRE(j.dof == 3);
    REQUIRE(j.num_limits == 3);
    REQUIRE(j.limits[0].isApprox(Eigen::Vector2d(-3.0, 3.0)));
    REQUIRE(j.limits[1].isApprox(Eigen::Vector2d(-1.5, 1.5)));
    REQUIRE(j.limits[2].isApprox(Eigen::Vector2d(-1.0, 1.0)));

    // Test JSON round-trip
    nlohmann::json const json = j.to_json();
    Joint const j2 = Joint::from_json(json);

    REQUIRE(j2.name == j.name);
    REQUIRE(j2.dof == 3);
    REQUIRE(j2.limits.size() == 3);
    for (size_t i = 0; i < 3; ++i) {
        REQUIRE(j2.limits[i].isApprox(j.limits[i]));
    }
}

TEST_CASE("Marker construction and JSON", "[skeleton]") {
    Marker m("left_knee", "knee_joint", Eigen::Vector3d(0.05, 0, 0), 13);

    REQUIRE(m.name == "left_knee");
    REQUIRE(m.joint == "knee_joint");
    REQUIRE(m.local_pos.isApprox(Eigen::Vector3d(0.05, 0, 0)));
    REQUIRE(m.coco_id.has_value());
    REQUIRE(*m.coco_id == 13);

    nlohmann::json const json = m.to_json();
    Marker const m2 = Marker::from_json(json);

    REQUIRE(m2.name == m.name);
    REQUIRE(m2.joint == m.joint);
    REQUIRE(m2.local_pos.isApprox(m.local_pos));
    REQUIRE(m2.coco_id == m.coco_id);
}

TEST_CASE("Skeleton construction and validation", "[skeleton]") {
    SECTION("Simple valid skeleton") {
        Skeleton skel;
        skel.add_joint(Joint("pelvis", "", JointType::SPHERICAL));
        skel.add_joint(Joint("spine", "pelvis", JointType::REVOLUTE));
        skel.add_joint(Joint("head", "spine", JointType::SPHERICAL));

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
        skel.add_joint(Joint("child", "missing_parent", JointType::REVOLUTE));
        auto const err = skel.validate();
        REQUIRE(err.has_value());
        REQUIRE(err->find("not found") != std::string::npos);
    }

    SECTION("Multiple roots is invalid") {
        Skeleton skel;
        skel.add_joint(Joint("root1", "", JointType::SPHERICAL));
        skel.add_joint(Joint("root2", "", JointType::SPHERICAL));
        auto const err = skel.validate();
        REQUIRE(err.has_value());
    }

    SECTION("Cycle detection") {
        Skeleton skel;
        skel.add_joint(Joint("a", "b", JointType::REVOLUTE));
        skel.add_joint(Joint("b", "c", JointType::REVOLUTE));
        skel.add_joint(Joint("c", "a", JointType::REVOLUTE));
        auto const err = skel.validate();
        REQUIRE(err.has_value());
        REQUIRE(err->find("Cycle") != std::string::npos);
    }
}

TEST_CASE("Skeleton DOF counting", "[skeleton]") {
    Skeleton skel;
    skel.add_joint(Joint("pelvis", "", JointType::SPHERICAL, Eigen::Vector3d::Zero(), "torso"));
    skel.add_joint(Joint("spine", "pelvis", JointType::REVOLUTE, Eigen::Vector3d::Zero(), "torso"));
    skel.add_joint(
        Joint("left_hip", "pelvis", JointType::SPHERICAL, Eigen::Vector3d::Zero(), "legs"));
    skel.add_joint(
        Joint("left_knee", "left_hip", JointType::REVOLUTE, Eigen::Vector3d::Zero(), "legs"));
    skel.add_joint(
        Joint("right_hip", "pelvis", JointType::SPHERICAL, Eigen::Vector3d::Zero(), "legs"));
    skel.add_joint(Joint("marker", "pelvis", JointType::FIXED));

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
    skel.add_joint(Joint("pelvis", "", JointType::SPHERICAL));
    skel.add_joint(Joint("spine", "pelvis", JointType::REVOLUTE));

    SECTION("Add marker to existing joint") {
        skel.add_marker(Marker("m1", "pelvis", Eigen::Vector3d(0.1, 0, 0)));
        auto const* m = skel.get_marker("m1");
        REQUIRE(m != nullptr);
        REQUIRE(m->joint == "pelvis");
    }

    SECTION("Cannot add marker to missing joint") {
        REQUIRE_THROWS_AS(skel.add_marker(Marker("m1", "missing", Eigen::Vector3d::Zero())),
                          std::invalid_argument);
    }

    SECTION("Cannot add duplicate marker") {
        skel.add_marker(Marker("m1", "pelvis", Eigen::Vector3d::Zero()));
        REQUIRE_THROWS_AS(skel.add_marker(Marker("m1", "spine", Eigen::Vector3d::Zero())),
                          std::invalid_argument);
    }

    SECTION("Validation checks marker joints") {
        skel.add_marker(Marker("m1", "pelvis", Eigen::Vector3d::Zero()));
        auto err = skel.validate();
        REQUIRE(!err.has_value());
    }
}

TEST_CASE("Skeleton joint ordering", "[skeleton]") {
    Skeleton skel;
    skel.add_joint(Joint("pelvis", "", JointType::SPHERICAL));
    skel.add_joint(Joint("left_hip", "pelvis", JointType::SPHERICAL));
    skel.add_joint(Joint("right_hip", "pelvis", JointType::SPHERICAL));
    skel.add_joint(Joint("left_knee", "left_hip", JointType::REVOLUTE));

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
    skel.add_joint(Joint("pelvis", "", JointType::SPHERICAL));
    skel.add_joint(Joint("spine", "pelvis", JointType::REVOLUTE));
    skel.add_marker(Marker("m1", "pelvis", Eigen::Vector3d(0.1, 0, 0), 0));

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
    skel.add_joint(Joint("pelvis", "", JointType::SPHERICAL, Eigen::Vector3d::Zero(), "torso"));
    skel.add_joint(Joint("spine", "pelvis", JointType::REVOLUTE, Eigen::Vector3d::Zero(), "torso"));

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

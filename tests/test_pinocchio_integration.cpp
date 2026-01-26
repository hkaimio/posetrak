#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/kinematics/forward_kinematics.hpp"
#include "posetrak/kinematics/pinocchio_model_builder.hpp"

using namespace posetrak;

TEST_CASE("PinocchioModelBuilder builds model from skeleton", "[pinocchio]") {
    // Create a simple skeleton: root + 1 revolute joint + 1 marker
    Skeleton skeleton;

    // Root joint (pelvis)
    Joint root("pelvis", "", JointType::REVOLUTE, Eigen::Vector3d::Zero());
    skeleton.add_joint(root);

    // Child joint (spine)
    Joint spine("spine", "pelvis", JointType::REVOLUTE, Eigen::Vector3d(0, 0, 0.1));
    skeleton.add_joint(spine);

    // Marker on spine
    Marker marker("head_marker", "spine", Eigen::Vector3d(0, 0, 0.2));
    skeleton.add_marker(marker);

    SECTION("Model builds successfully") {
        pinocchio::Model model;
        REQUIRE_NOTHROW(PinocchioModelBuilder::build_model(skeleton, model));

        // Check model dimensions
        // Root: 7 DOF (3 pos + 4 quat), Spine: 1 DOF (revolute)
        REQUIRE(model.nq == 8);  // 7 + 1
        REQUIRE(model.nv == 7);  // 6 (root velocity) + 1

        // Should have 2 joints + universe
        REQUIRE(model.njoints == 3);
    }

    SECTION("Marker frames are created") {
        pinocchio::Model model;
        PinocchioModelBuilder::build_model(skeleton, model);

        // Check marker frame exists
        REQUIRE(model.existFrame("head_marker"));
        auto frame_id = model.getFrameId("head_marker");
        REQUIRE(frame_id >= 0);

        // Check frame type
        REQUIRE(model.frames[frame_id].type == pinocchio::OP_FRAME);
    }

    SECTION("Marker frame map builds correctly") {
        pinocchio::Model model;
        PinocchioModelBuilder::build_model(skeleton, model);

        auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
        REQUIRE(marker_map.size() == 1);
        REQUIRE(marker_map.count("head_marker") > 0);
    }
}

TEST_CASE("ForwardKinematics computes marker positions", "[forward_kinematics]") {
    // Create a simple skeleton
    Skeleton skeleton;
    Joint root("pelvis", "", JointType::REVOLUTE, Eigen::Vector3d::Zero());
    skeleton.add_joint(root);

    Joint spine("spine", "pelvis", JointType::REVOLUTE, Eigen::Vector3d(0, 0, 0.1));
    skeleton.add_joint(spine);

    Marker marker("head_marker", "spine", Eigen::Vector3d(0, 0, 0.2));
    skeleton.add_marker(marker);

    // Build Pinocchio model
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);

    auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);

    // Create FK computer
    ForwardKinematics fk(model, data, marker_map);

    SECTION("FK with zero configuration") {
        // Configuration: root at origin, no rotation, spine at 0
        Eigen::VectorXd q = Eigen::VectorXd::Zero(model.nq);
        q[3] = 0.0;  // quat x
        q[4] = 0.0;  // quat y
        q[5] = 0.0;  // quat z
        q[6] = 1.0;  // quat w (identity)
        // q[7] = 0.0;  // spine angle (already zero)

        auto positions = fk.compute(q);

        REQUIRE(positions.size() == 1);
        REQUIRE(positions.count("head_marker") > 0);

        // Marker should be at [0, 0, 0.3] (0.1 from root to spine + 0.2 from spine to marker)
        auto const& pos = positions["head_marker"];
        REQUIRE_THAT(pos.x(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(pos.y(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(pos.z(), Catch::Matchers::WithinAbs(0.3, 1e-6));
    }

    SECTION("state_to_config converts State correctly") {
        // Create a state
        Eigen::Vector3d root_pos(1.0, 2.0, 3.0);
        Eigen::Quaterniond root_quat = Eigen::Quaterniond::Identity();
        Eigen::VectorXd joint_angles(1);
        joint_angles << 0.5;                                  // spine angle
        Eigen::VectorXd root_vel = Eigen::VectorXd::Zero(3);  // 3D linear velocity
        Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(1);

        State state(root_pos, root_quat, joint_angles, root_vel, joint_vels);

        auto q = ForwardKinematics::state_to_config(state, skeleton);

        REQUIRE(q.size() == model.nq);

        // Check root position
        REQUIRE_THAT(q[0], Catch::Matchers::WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(q[1], Catch::Matchers::WithinAbs(2.0, 1e-6));
        REQUIRE_THAT(q[2], Catch::Matchers::WithinAbs(3.0, 1e-6));

        // Check root quaternion (identity: x=0, y=0, z=0, w=1)
        REQUIRE_THAT(q[3], Catch::Matchers::WithinAbs(0.0, 1e-6));  // x
        REQUIRE_THAT(q[4], Catch::Matchers::WithinAbs(0.0, 1e-6));  // y
        REQUIRE_THAT(q[5], Catch::Matchers::WithinAbs(0.0, 1e-6));  // z
        REQUIRE_THAT(q[6], Catch::Matchers::WithinAbs(1.0, 1e-6));  // w

        // Check spine angle
        REQUIRE_THAT(q[7], Catch::Matchers::WithinAbs(0.5, 1e-6));
    }
}

TEST_CASE("ForwardKinematics handles spherical joints", "[forward_kinematics]") {
    // Create skeleton with spherical joint
    Skeleton skeleton;
    Joint root("pelvis", "", JointType::REVOLUTE, Eigen::Vector3d::Zero());
    skeleton.add_joint(root);

    Joint shoulder("shoulder", "pelvis", JointType::SPHERICAL, Eigen::Vector3d(0.2, 0, 0));
    skeleton.add_joint(shoulder);

    Marker marker("hand", "shoulder", Eigen::Vector3d(0.3, 0, 0));
    skeleton.add_marker(marker);

    // Build model
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);

    auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
    ForwardKinematics fk(model, data, marker_map);

    SECTION("state_to_config converts spherical joint correctly") {
        // State with spherical joint angles
        Eigen::Vector3d root_pos = Eigen::Vector3d::Zero();
        Eigen::Quaterniond root_quat = Eigen::Quaterniond::Identity();
        Eigen::VectorXd joint_angles(3);  // 3 DOF for spherical
        joint_angles << 0.1, 0.2, 0.3;    // rotation angles

        Eigen::VectorXd root_vel = Eigen::VectorXd::Zero(3);  // 3D linear velocity
        Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(3);

        State state(root_pos, root_quat, joint_angles, root_vel, joint_vels);

        auto q = ForwardKinematics::state_to_config(state, skeleton);

        // Root: 7 DOF, Shoulder (spherical): 4 DOF (quaternion)
        REQUIRE(q.size() == 11);

        // Verify quaternion conversion (angle-axis)
        Eigen::Vector3d angles(0.1, 0.2, 0.3);
        double angle = angles.norm();
        Eigen::Vector3d axis = angles.normalized();
        Eigen::Quaterniond expected_quat(Eigen::AngleAxisd(angle, axis));

        // Check shoulder quaternion [x,y,z,w] at indices 7-10
        REQUIRE_THAT(q[7], Catch::Matchers::WithinAbs(expected_quat.x(), 1e-6));
        REQUIRE_THAT(q[8], Catch::Matchers::WithinAbs(expected_quat.y(), 1e-6));
        REQUIRE_THAT(q[9], Catch::Matchers::WithinAbs(expected_quat.z(), 1e-6));
        REQUIRE_THAT(q[10], Catch::Matchers::WithinAbs(expected_quat.w(), 1e-6));
    }
}

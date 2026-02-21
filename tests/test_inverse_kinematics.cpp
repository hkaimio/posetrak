#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/io/skeleton_loader.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include "posetrak/kinematics/inverse_kinematics.hpp"
#include "posetrak/kinematics/pinocchio_model_builder.hpp"

using namespace posetrak;

TEST_CASE("InverseKinematics solves simple 2-joint problem", "[ik][inverse_kinematics]") {
    // Create simple skeleton: root + revolute joint + marker
    Skeleton skeleton;

    uint32_t pelvis_idx = skeleton.add_joint("pelvis", std::nullopt, JointType::REVOLUTE,
                                             Eigen::Vector3d::Zero(), "", Eigen::Vector3d::Zero());

    // Spine joint at 0.1m above pelvis
    uint32_t spine_idx =
        skeleton.add_joint("spine", pelvis_idx, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 0.1), "",
                           Eigen::Vector3d::Zero());

    // Marker on spine at 0.2m above joint
    skeleton.add_marker("head", spine_idx, Eigen::Vector3d(0, 0, 0.2), std::nullopt);

    // Build Pinocchio model
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
    auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);

    // Create FK and IK
    auto fk_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));

    ForwardKinematics fk(model, data, marker_map, fk_layout);
    InverseKinematics ik(model, data, fk, marker_map);

    SECTION("IK finds zero configuration") {
        // Target: marker at [0, 0, 0.3] (zero configuration)
        std::map<std::string, Eigen::Vector3d> targets;
        targets["head"] = Eigen::Vector3d(0, 0, 0.3);

        // Initial guess: slightly off
        Eigen::Vector3d root_pos(0, 0, 0);
        Eigen::Quaterniond root_quat = Eigen::Quaterniond::Identity();
        Eigen::VectorXd joint_angles(1);
        joint_angles[0] = 0.1;  // 0.1 rad off
        Eigen::Vector3d root_vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d root_angvel = Eigen::Vector3d::Zero();
        Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(1);
        State initial(root_pos, root_quat, joint_angles, root_vel, root_angvel, joint_vels);

        auto result = ik.solve(targets, skeleton, initial, 50, 0.001);

        INFO("Converged: " << result.converged);
        INFO("Iterations: " << result.iterations);
        INFO("Residual: " << result.residual);

        REQUIRE(result.converged);
        REQUIRE(result.residual < 0.001);  // < 1mm error
    }

    SECTION("IK handles non-zero configurations") {
        // Create target by FK with known configuration
        Eigen::VectorXd q = Eigen::VectorXd::Zero(model.nq);
        q[6] = 1.0;  // Root quaternion w
        q[7] = 0.5;  // Spine angle = 0.5 rad

        auto target_markers = fk.compute(q);
        REQUIRE(target_markers.count("head") > 0);

        Eigen::Vector3d target_pos = target_markers["head"];

        // Initial guess: zero
        Eigen::Vector3d root_pos(0, 0, 0);
        Eigen::Quaterniond root_quat = Eigen::Quaterniond::Identity();
        Eigen::VectorXd joint_angles = Eigen::VectorXd::Zero(1);
        Eigen::Vector3d root_vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d root_angvel = Eigen::Vector3d::Zero();
        Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(1);
        State initial(root_pos, root_quat, joint_angles, root_vel, root_angvel, joint_vels);

        std::map<std::string, Eigen::Vector3d> targets;
        targets["head"] = target_pos;

        auto result = ik.solve(targets, skeleton, initial, 50, 0.01);

        INFO("Target: " << target_pos.transpose());
        INFO("Converged: " << result.converged);
        INFO("Iterations: " << result.iterations);
        INFO("Residual: " << result.residual);

        // Should converge to reasonable error
        REQUIRE(result.residual < 0.02);  // < 2cm (loose for now)
    }
}

TEST_CASE("InverseKinematics handles multiple markers", "[ik][inverse_kinematics]") {
    // Create skeleton with 2 markers
    Skeleton skeleton;

    uint32_t root_idx = skeleton.add_joint("pelvis", std::nullopt, JointType::REVOLUTE,
                                           Eigen::Vector3d::Zero(), "", Eigen::Vector3d::Zero());

    uint32_t spine_idx =
        skeleton.add_joint("spine", root_idx, JointType::REVOLUTE, Eigen::Vector3d(0, 0, 0.1), "",
                           Eigen::Vector3d::Zero());

    // Two markers
    skeleton.add_marker("marker1", root_idx, Eigen::Vector3d(0.1, 0, 0), std::nullopt);
    skeleton.add_marker("marker2", spine_idx, Eigen::Vector3d(0, 0, 0.2), std::nullopt);

    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
    auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);

    auto fk_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));

    ForwardKinematics fk(model, data, marker_map, fk_layout);
    InverseKinematics ik(model, data, fk, marker_map);

    // Target positions (zero config)
    std::map<std::string, Eigen::Vector3d> targets;
    targets["marker1"] = Eigen::Vector3d(0.1, 0, 0);
    targets["marker2"] = Eigen::Vector3d(0, 0, 0.3);

    Eigen::Vector3d root_pos(0, 0, 0);
    Eigen::Quaterniond root_quat = Eigen::Quaterniond::Identity();
    Eigen::VectorXd joint_angles = Eigen::VectorXd::Constant(1, 0.2);  // Start with offset
    Eigen::Vector3d root_vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d root_angvel = Eigen::Vector3d::Zero();
    Eigen::VectorXd joint_vels = Eigen::VectorXd::Zero(1);
    State initial(root_pos, root_quat, joint_angles, root_vel, root_angvel, joint_vels);

    auto result = ik.solve(targets, skeleton, initial, 50, 0.01);

    INFO("Converged: " << result.converged);
    INFO("Iterations: " << result.iterations);
    INFO("Residual: " << result.residual);

    // With multiple markers, should still converge
    REQUIRE(result.residual < 0.02);
}

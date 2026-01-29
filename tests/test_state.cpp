#include <posetrak/core/state.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

using namespace posetrak;
using Catch::Matchers::WithinAbs;

TEST_CASE("State construction", "[state]") {
    SECTION("Default construction with DOF") {
        State state(5);

        REQUIRE(state.num_dof() == 5);
        // 2 * (3 pos + 3 ori + 5 dof) = 22
        REQUIRE(state.error_state_dim() == 2 * (3 + 3 + 5));
        REQUIRE(state.root_position().isZero());
        REQUIRE(state.root_orientation().isApprox(Eigen::Quaterniond::Identity()));
        REQUIRE(state.joint_angles().isZero());
        REQUIRE(state.root_velocity().isZero());
        REQUIRE(state.joint_velocities().isZero());
    }

    SECTION("Construction from components") {
        Eigen::Vector3d pos(1.0, 2.0, 3.0);
        Eigen::Quaterniond quat(1.0, 0.0, 0.0, 0.0);  // Identity (w, x, y, z)
        Eigen::VectorXd angles(3);
        angles << 0.1, 0.2, 0.3;
        Eigen::Vector3d root_vel(0.5, 0.6, 0.7);
        Eigen::Vector3d root_angvel(0.01, 0.02, 0.03);
        Eigen::VectorXd joint_vel(3);
        joint_vel << 0.1, 0.2, 0.3;

        State state(pos, quat, angles, root_vel, root_angvel, joint_vel);

        REQUIRE(state.root_position().isApprox(pos));
        REQUIRE(state.root_orientation().isApprox(quat));
        REQUIRE(state.joint_angles().isApprox(angles));
        REQUIRE(state.root_velocity().isApprox(root_vel));
        REQUIRE(state.root_angular_velocity().isApprox(root_angvel));
        REQUIRE(state.joint_velocities().isApprox(joint_vel));
    }

    SECTION("Invalid DOF throws") {
        REQUIRE_THROWS_AS(State(-1), std::invalid_argument);
    }

    SECTION("Mismatched velocity sizes throw") {
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
        Eigen::VectorXd angles(3);
        angles.setZero();
        Eigen::Vector3d root_vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d root_angvel = Eigen::Vector3d::Zero();
        Eigen::VectorXd joint_vel(2);  // Wrong size!
        joint_vel.setZero();

        REQUIRE_THROWS_AS(State(pos, quat, angles, root_vel, root_angvel, joint_vel),
                          std::invalid_argument);
    }
}

TEST_CASE("Quaternion to axis-angle conversion", "[state]") {
    SECTION("Identity quaternion") {
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
        Eigen::Vector3d aa = State::quaternion_to_axis_angle(q);

        REQUIRE(aa.norm() < 1e-6);
    }

    SECTION("90 degree rotation around Z axis") {
        double const angle = M_PI / 2.0;
        Eigen::Quaterniond q(std::cos(angle / 2), 0.0, 0.0, std::sin(angle / 2));
        Eigen::Vector3d aa = State::quaternion_to_axis_angle(q);

        REQUIRE_THAT(aa.norm(), WithinAbs(M_PI / 2.0, 1e-6));
        REQUIRE_THAT(aa.z(), WithinAbs(M_PI / 2.0, 1e-6));
        REQUIRE_THAT(aa.x(), WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(aa.y(), WithinAbs(0.0, 1e-6));
    }

    SECTION("Arbitrary rotation") {
        Eigen::Vector3d axis(1.0, 2.0, 3.0);
        axis.normalize();
        double const angle = 0.7;

        Eigen::Quaterniond q(Eigen::AngleAxisd(angle, axis));
        Eigen::Vector3d aa = State::quaternion_to_axis_angle(q);

        REQUIRE_THAT(aa.norm(), WithinAbs(angle, 1e-6));
        Eigen::Vector3d recovered_axis = aa.normalized();
        REQUIRE(recovered_axis.isApprox(axis, 1e-6));
    }
}

TEST_CASE("Axis-angle to quaternion conversion", "[state]") {
    SECTION("Zero rotation") {
        Eigen::Vector3d aa = Eigen::Vector3d::Zero();
        Eigen::Quaterniond q = State::axis_angle_to_quaternion(aa);

        REQUIRE(q.isApprox(Eigen::Quaterniond::Identity()));
    }

    SECTION("90 degree rotation around X axis") {
        Eigen::Vector3d aa(M_PI / 2.0, 0.0, 0.0);
        Eigen::Quaterniond q = State::axis_angle_to_quaternion(aa);

        Eigen::Quaterniond expected(std::cos(M_PI / 4), std::sin(M_PI / 4), 0.0, 0.0);
        REQUIRE(q.isApprox(expected, 1e-6));
    }

    SECTION("Round-trip conversion") {
        Eigen::Vector3d axis(1.0, -2.0, 0.5);
        axis.normalize();
        double const angle = 1.2;
        Eigen::Quaterniond q_orig(Eigen::AngleAxisd(angle, axis));

        Eigen::Vector3d aa = State::quaternion_to_axis_angle(q_orig);
        Eigen::Quaterniond q_recovered = State::axis_angle_to_quaternion(aa);

        REQUIRE(q_recovered.isApprox(q_orig, 1e-6));
    }
}

TEST_CASE("Error-state conversion", "[state]") {
    SECTION("Identity state") {
        State state(3);
        Eigen::VectorXd error = state.to_error_vector();

        // 2 * (3 pos + 3 orient + 3 joints) = 18
        REQUIRE(error.size() == 18);
        REQUIRE(error.isZero());
    }

    SECTION("Non-trivial state") {
        Eigen::Vector3d pos(1.0, 2.0, 3.0);
        Eigen::Quaterniond quat(Eigen::AngleAxisd(0.5, Eigen::Vector3d(0, 0, 1)));
        Eigen::VectorXd angles(2);
        angles << 0.1, 0.2;
        Eigen::Vector3d root_vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d root_angvel = Eigen::Vector3d::Zero();
        Eigen::VectorXd joint_vel = Eigen::VectorXd::Zero(2);

        State state(pos, quat, angles, root_vel, root_angvel, joint_vel);
        Eigen::VectorXd error = state.to_error_vector();

        // 2 * (3 + 3 + 2) = 16
        REQUIRE(error.size() == 16);
        REQUIRE(error.segment<3>(0).isApprox(pos));
        REQUIRE_THAT(error.segment<3>(3).norm(), WithinAbs(0.5, 1e-6));
        REQUIRE(error.segment<2>(6).isApprox(angles));
        // Velocities should be zero
        REQUIRE(error.segment<3>(8).isZero());   // root_vel
        REQUIRE(error.segment<3>(11).isZero());  // root_angvel
        REQUIRE(error.segment<2>(14).isZero());  // joint_vel
    }
}

TEST_CASE("Error-state update", "[state]") {
    SECTION("Position update") {
        State state(2);
        // 2 * (3 + 3 + 2) = 16
        Eigen::VectorXd delta = Eigen::VectorXd::Zero(16);
        delta.segment<3>(0) << 1.0, 2.0, 3.0;  // Position delta

        state.apply_error_update(delta);

        REQUIRE(state.root_position().isApprox(Eigen::Vector3d(1.0, 2.0, 3.0)));
        REQUIRE(state.root_orientation().isApprox(Eigen::Quaterniond::Identity()));
    }

    SECTION("Orientation update") {
        State state(2);
        Eigen::VectorXd delta = Eigen::VectorXd::Zero(16);
        delta.segment<3>(3) << 0.0, 0.0, M_PI / 2.0;  // 90 deg rotation around Z

        state.apply_error_update(delta);

        Eigen::Quaterniond expected(std::cos(M_PI / 4), 0.0, 0.0, std::sin(M_PI / 4));
        REQUIRE(state.root_orientation().isApprox(expected, 1e-6));
    }

    SECTION("Joint angle update") {
        State state(3);
        // 2 * (3 + 3 + 3) = 18
        Eigen::VectorXd delta = Eigen::VectorXd::Zero(18);
        delta.segment<3>(6) << 0.1, 0.2, 0.3;

        state.apply_error_update(delta);

        Eigen::VectorXd expected(3);
        expected << 0.1, 0.2, 0.3;
        REQUIRE(state.joint_angles().isApprox(expected));
    }

    SECTION("Round-trip update") {
        Eigen::Vector3d pos(1.0, 2.0, 3.0);
        Eigen::Quaterniond quat(Eigen::AngleAxisd(0.3, Eigen::Vector3d(1, 1, 0).normalized()));
        Eigen::VectorXd angles(2);
        angles << 0.5, -0.3;
        Eigen::Vector3d root_vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d root_angvel = Eigen::Vector3d::Zero();
        Eigen::VectorXd joint_vel = Eigen::VectorXd::Zero(2);

        State state(pos, quat, angles, root_vel, root_angvel, joint_vel);

        // Small perturbation: 2 * (3 + 3 + 2) = 16
        Eigen::VectorXd delta = Eigen::VectorXd::Zero(16);
        delta << 0.01, 0.02, 0.03, 0.001, 0.002, 0.003, 0.05, -0.05,  // pos, ori, joints
            0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.05, 0.06;              // velocities

        state.apply_error_update(delta);

        // Check position update
        REQUIRE(state.root_position().isApprox(pos + delta.segment<3>(0), 1e-6));

        // Check orientation update (more involved)
        Eigen::Quaterniond delta_q = State::axis_angle_to_quaternion(delta.segment<3>(3));
        Eigen::Quaterniond expected_quat = delta_q * quat;
        REQUIRE(state.root_orientation().isApprox(expected_quat, 1e-6));

        // Check joint angles
        REQUIRE(state.joint_angles().isApprox(angles + delta.segment<2>(6), 1e-6));
    }
}

TEST_CASE("JSON serialization", "[state]") {
    SECTION("Round-trip serialization") {
        Eigen::Vector3d pos(1.5, -2.3, 4.7);
        Eigen::Quaterniond quat(Eigen::AngleAxisd(0.8, Eigen::Vector3d(1, 0, 1).normalized()));
        Eigen::VectorXd angles(4);
        angles << 0.1, -0.5, 1.2, -0.3;
        Eigen::Vector3d root_vel(0.5, 0.6, 0.7);
        Eigen::Vector3d root_angvel(0.01, 0.02, 0.03);
        Eigen::VectorXd joint_vel(4);
        joint_vel << 0.1, 0.2, -0.1, 0.3;

        State original(pos, quat, angles, root_vel, root_angvel, joint_vel);

        nlohmann::json j = original.to_json();
        State recovered = State::from_json(j);

        REQUIRE(recovered.root_position().isApprox(original.root_position(), 1e-10));
        REQUIRE(recovered.root_orientation().isApprox(original.root_orientation(), 1e-10));
        REQUIRE(recovered.joint_angles().isApprox(original.joint_angles(), 1e-10));
        REQUIRE(recovered.root_velocity().isApprox(original.root_velocity(), 1e-10));
        REQUIRE(recovered.joint_velocities().isApprox(original.joint_velocities(), 1e-10));
    }

    SECTION("Zero DOF state") {
        State state(0);
        nlohmann::json j = state.to_json();
        State recovered = State::from_json(j);

        REQUIRE(recovered.num_dof() == 0);
        REQUIRE(recovered.root_position().isZero());
    }
}

TEST_CASE("Different DOF counts", "[state]") {
    SECTION("Zero DOF") {
        State state(0);
        REQUIRE(state.num_dof() == 0);
        // 2 * (3 pos + 3 ori + 0 dof) = 12
        REQUIRE(state.error_state_dim() == 2 * (3 + 3 + 0));
    }

    SECTION("Large DOF") {
        State state(120);  // Typical for full body
        REQUIRE(state.num_dof() == 120);
        // 2 * (3 pos + 3 ori + 120 dof) = 252
        REQUIRE(state.error_state_dim() == 2 * (3 + 3 + 120));
        REQUIRE(state.joint_angles().size() == 120);
    }
}

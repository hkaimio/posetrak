/**
 * @file test_subset_ukf.cpp
 * @brief Unit tests for SubsetUKF — UKF tracking a named subset of skeleton joints.
 *
 * Test skeleton (same as test_skeleton_layout.cpp / test_skeleton_state.cpp):
 *
 *   Joint storage-DOF layout (root hips excluded from non-root count):
 *     spine      :  0–2   (SPHERICAL, group=main)
 *     shoulder.L :  3–5   (SPHERICAL, group=main)
 *     elbow.L    :  6     (REVOLUTE,  group=main)
 *     wrist.L    :  7–9   (SPHERICAL, group=main)
 *     finger1.L  : 10     (REVOLUTE,  group=HandL)
 *     finger2.L  : 11     (REVOLUTE,  group=HandL)
 *     shoulder.R : 12–14  (SPHERICAL, group=main)
 *     elbow.R    : 15     (REVOLUTE,  group=main)
 *     wrist.R    : 16–18  (SPHERICAL, group=main)
 *     finger1.R  : 19     (REVOLUTE,  group=HandR)
 *     finger2.R  : 20     (REVOLUTE,  group=HandR)
 *     Total: 21 storage DOFs
 *
 * SubsetUKF("HandR") tracks finger1.R + finger2.R (state indices 19, 20).
 * error_dim = 2 (one per REVOLUTE DOF), no floating root.
 */

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;

#include "posetrak/core/observation.hpp"
#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/state.hpp"
#include "posetrak/filters/subset_ukf.hpp"

using namespace posetrak;

// ---------------------------------------------------------------------------
// Test skeleton (identical to test_skeleton_layout.cpp)
// ---------------------------------------------------------------------------

static Skeleton create_test_skeleton() {
    Skeleton skel;

    uint32_t hips =
        skel.add_joint("hips", std::nullopt, JointType::SPHERICAL, Eigen::Vector3d::Zero());
    uint32_t spine =
        skel.add_joint("spine", hips, JointType::SPHERICAL, Eigen::Vector3d(0, 0.1, 0));
    uint32_t left_shoulder =
        skel.add_joint("shoulder.L", spine, JointType::SPHERICAL, Eigen::Vector3d(0.2, 0.1, 0));
    uint32_t left_elbow =
        skel.add_joint("elbow.L", left_shoulder, JointType::REVOLUTE, Eigen::Vector3d(0.3, 0, 0));
    uint32_t left_wrist =
        skel.add_joint("wrist.L", left_elbow, JointType::SPHERICAL, Eigen::Vector3d(0.3, 0, 0));
    skel.add_joint("finger1.L", left_wrist, JointType::REVOLUTE, Eigen::Vector3d(0.1, 0, 0));
    skel.add_joint("finger2.L", left_wrist, JointType::REVOLUTE, Eigen::Vector3d(0.1, 0.05, 0));

    uint32_t right_shoulder =
        skel.add_joint("shoulder.R", spine, JointType::SPHERICAL, Eigen::Vector3d(-0.2, 0.1, 0));
    uint32_t right_elbow =
        skel.add_joint("elbow.R", right_shoulder, JointType::REVOLUTE, Eigen::Vector3d(-0.3, 0, 0));
    uint32_t right_wrist =
        skel.add_joint("wrist.R", right_elbow, JointType::SPHERICAL, Eigen::Vector3d(-0.3, 0, 0));
    skel.add_joint("finger1.R", right_wrist, JointType::REVOLUTE, Eigen::Vector3d(-0.1, 0, 0));
    skel.add_joint("finger2.R", right_wrist, JointType::REVOLUTE, Eigen::Vector3d(-0.1, 0.05, 0));

    skel.register_group(
        "main",
        {"hips", "spine", "shoulder.L", "elbow.L", "wrist.L", "shoulder.R", "elbow.R", "wrist.R"},
        {});
    skel.register_group("HandL", {"finger1.L", "finger2.L"}, {});
    skel.register_group("HandR", {"finger1.R", "finger2.R"}, {});

    return skel;
}

/// Build a zero-value full-skeleton State (skeleton.total_dof_count() = 24 DOFs).
static State make_full_state(double finger1_r = 0.0, double finger2_r = 0.0) {
    // total_dof_count() counts all joints including root hips SPHERICAL (3 DOF),
    // hence 24 = 3 (root hips) + 21 (non-root joints). State vector size must match.
    State s(24);
    Eigen::VectorXd angles = Eigen::VectorXd::Zero(24);
    angles(19) = finger1_r;
    angles(20) = finger2_r;
    s.set_joint_angles(angles);
    s.set_joint_velocities(Eigen::VectorXd::Zero(24));
    s.set_root_position(Eigen::Vector3d::Zero());
    s.set_root_orientation(Eigen::Quaterniond::Identity());
    s.set_root_velocity(Eigen::Vector3d::Zero());
    s.set_root_angular_velocity(Eigen::Vector3d::Zero());
    return s;
}

// ---------------------------------------------------------------------------
// Tests: construction
// ---------------------------------------------------------------------------

TEST_CASE("SubsetUKF construction — HandR subset", "[subset_ukf]") {
    auto skel = create_test_skeleton();
    SubsetUKF filter(skel, {"HandR"}, {"HandR"}, 0.3);

    // HandR has 2 REVOLUTE joints (1 active DOF each) → error_dim = 2*(0+2) = 4.
    // error_state_dim = 2*(root_error_dof_count + joint_active_dof_count)
    //                 = 2*(0 + 2) = 4  (angles + velocities, no floating root)
    REQUIRE(filter.error_dim() == 4);

    // Compact layout covers only finger1.R + finger2.R (2 storage DOFs).
    REQUIRE(filter.compact_layout()->total_storage_dof_count() == 2);

    // merge_map_: compact DOF 0 → full state index 19, compact DOF 1 → full state index 20.
    REQUIRE(filter.merge_map().size() == 2);
    REQUIRE(filter.merge_map()[0] == 19);
    REQUIRE(filter.merge_map()[1] == 20);
}

TEST_CASE("SubsetUKF construction — main subset", "[subset_ukf]") {
    auto skel = create_test_skeleton();
    SubsetUKF filter(skel, {"main"}, {"main"}, 0.3);

    // main active joints: spine(3) + shoulder.L(3) + elbow.L(1) + wrist.L(3)
    //                    + shoulder.R(3) + elbow.R(1) + wrist.R(3) = 17 joint DOFs
    // Plus root hips in group main → has_floating_root → root_error_dof_count=6
    // error_state_dim = 2*(6 + 17) = 46
    REQUIRE(filter.compact_layout()->total_storage_dof_count() == 17);
    REQUIRE(filter.error_dim() == 46);
    REQUIRE(static_cast<int>(filter.merge_map().size()) == 17);
}

// ---------------------------------------------------------------------------
// Tests: initialize + state access
// ---------------------------------------------------------------------------

TEST_CASE("SubsetUKF initialize sets state and covariance", "[subset_ukf]") {
    auto skel = create_test_skeleton();
    SubsetUKF filter(skel, {"HandR"}, {"HandR"}, 0.3);

    State init = make_full_state(0.5, 0.7);
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(4, 4) * 0.01;
    filter.initialize(init, cov);

    // full_state() should reflect the initialised values.
    REQUIRE(filter.full_state().joint_angles()(19) == Approx(0.5));
    REQUIRE(filter.full_state().joint_angles()(20) == Approx(0.7));

    // Covariance should match.
    REQUIRE(filter.covariance()(0, 0) == Approx(0.01));
    REQUIRE(filter.covariance()(1, 1) == Approx(0.01));
    REQUIRE(filter.covariance()(2, 2) == Approx(0.01));
    REQUIRE(filter.covariance()(3, 3) == Approx(0.01));
}

// ---------------------------------------------------------------------------
// Tests: skeleton_state extraction
// ---------------------------------------------------------------------------

TEST_CASE("SubsetUKF skeleton_state extracts child DOFs", "[subset_ukf]") {
    auto skel = create_test_skeleton();
    SubsetUKF filter(skel, {"HandR"}, {"HandR"}, 0.3);

    State init = make_full_state(0.42, 0.84);
    filter.initialize(init, Eigen::MatrixXd::Identity(4, 4));

    SkeletonState ss = filter.skeleton_state();

    // Compact state has 2 DOFs corresponding to finger1.R and finger2.R.
    REQUIRE(ss.state().joint_angles().size() == 2);
    REQUIRE(ss.state().joint_angles()(0) == Approx(0.42));
    REQUIRE(ss.state().joint_angles()(1) == Approx(0.84));
}

// ---------------------------------------------------------------------------
// Tests: set_parent_state
// ---------------------------------------------------------------------------

TEST_CASE("SubsetUKF set_parent_state updates background", "[subset_ukf]") {
    auto skel = create_test_skeleton();
    SubsetUKF filter(skel, {"HandR"}, {"HandR"}, 0.3);

    // Initialise child DOFs to known values.
    State init = make_full_state(0.1, 0.2);
    filter.initialize(init, Eigen::MatrixXd::Identity(4, 4));

    // Provide a different parent state where elbow.R (index 15) is non-zero.
    State parent = make_full_state(0.1, 0.2);
    Eigen::VectorXd angles = parent.joint_angles();
    angles(15) = 1.23;  // elbow.R — parent DOF, not tracked by child
    parent.set_joint_angles(angles);

    filter.set_parent_state(parent);

    // After sync (triggered by predict with dt=0, which still merges background),
    // the full_state should have elbow.R from background AND fingers from child.
    filter.predict(0.0);

    // Finger values should still come from child (unchanged by predict at dt=0)
    REQUIRE(filter.full_state().joint_angles()(19) == Approx(0.1).margin(0.01));
    REQUIRE(filter.full_state().joint_angles()(20) == Approx(0.2).margin(0.01));

    // elbow.R should come from background
    REQUIRE(filter.full_state().joint_angles()(15) == Approx(1.23));
}

// ---------------------------------------------------------------------------
// Tests: predict smoke test
// ---------------------------------------------------------------------------

TEST_CASE("SubsetUKF predict does not crash", "[subset_ukf]") {
    auto skel = create_test_skeleton();
    SubsetUKF filter(skel, {"HandR"}, {"HandR"}, 0.3);
    filter.initialize(make_full_state(), Eigen::MatrixXd::Identity(4, 4));

    // Should not throw.
    REQUIRE_NOTHROW(filter.predict(1.0 / 60.0));
    REQUIRE_NOTHROW(filter.predict(1.0 / 60.0));
}

// ---------------------------------------------------------------------------
// Tests: filter_observations
// ---------------------------------------------------------------------------

TEST_CASE("SubsetUKF filter_observations returns HandR markers only", "[subset_ukf]") {
    auto skel = create_test_skeleton();

    // Add markers: two for HandR joints, one for a main joint.
    uint32_t m_finger1 =
        skel.add_marker("finger1_r_marker", 10 /*finger1.R joint index*/, Eigen::Vector3d::Zero());
    uint32_t m_finger2 =
        skel.add_marker("finger2_r_marker", 11 /*finger2.R joint index*/, Eigen::Vector3d::Zero());
    uint32_t m_elbow =
        skel.add_marker("elbow_r_marker", 8 /*elbow.R joint index*/, Eigen::Vector3d::Zero());

    // Set marker groups directly (add_marker has no group parameter).
    skel.markers()[m_finger1].groups = {"HandR"};
    skel.markers()[m_finger2].groups = {"HandR"};
    skel.markers()[m_elbow].groups = {"main"};

    SubsetUKF filter(skel, {"HandR"}, {"HandR"}, 0.3);

    // Build three dummy observations — one per marker.
    auto make_obs = [](int marker_id) -> Observation {
        Observation obs;
        obs.marker_id = marker_id;
        obs.camera_id = 0;
        obs.frame_idx = 0;
        obs.timestamp = 0.0;
        obs.confidence = 1.0;
        obs.position = Eigen::Vector2d::Zero();
        obs.position_distorted = Eigen::Vector2d::Zero();
        return obs;
    };

    std::vector<Observation> all_obs = {
        make_obs(static_cast<int>(m_finger1)),
        make_obs(static_cast<int>(m_finger2)),
        make_obs(static_cast<int>(m_elbow)),
    };

    auto filtered = filter.filter_observations(all_obs);

    REQUIRE(filtered.size() == 2);
    REQUIRE(filtered[0].marker_id == static_cast<int>(m_finger1));
    REQUIRE(filtered[1].marker_id == static_cast<int>(m_finger2));
}

TEST_CASE("SubsetUKF filter_observations skips out-of-range marker_id", "[subset_ukf]") {
    auto skel = create_test_skeleton();
    SubsetUKF filter(skel, {"HandR"}, {"HandR"}, 0.3);

    Observation obs;
    obs.marker_id = 9999;  // no such marker
    obs.camera_id = 0;
    obs.frame_idx = 0;
    obs.timestamp = 0.0;
    obs.confidence = 1.0;
    obs.position = Eigen::Vector2d::Zero();
    obs.position_distorted = Eigen::Vector2d::Zero();

    auto filtered = filter.filter_observations({obs});
    REQUIRE(filtered.empty());
}

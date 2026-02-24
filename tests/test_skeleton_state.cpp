/**
 * @file test_skeleton_state.cpp
 * @brief Unit tests for SkeletonState — compact State paired with SkeletonLayout.
 *
 * These tests replace the old test_subset_utils.cpp.  The same test skeleton
 * is used (identical to test_skeleton_layout.cpp) so index expectations are
 * directly comparable.
 *
 * Joint storage-DOF layout (root hips excluded):
 *   spine      :  0–2   (SPHERICAL, group=main)
 *   shoulder.L :  3–5   (SPHERICAL, group=main)
 *   elbow.L    :  6     (REVOLUTE,  group=main)
 *   wrist.L    :  7–9   (SPHERICAL, group=main)
 *   finger1.L  : 10     (REVOLUTE,  group=HandL)
 *   finger2.L  : 11     (REVOLUTE,  group=HandL)
 *   shoulder.R : 12–14  (SPHERICAL, group=main)
 *   elbow.R    : 15     (REVOLUTE,  group=main)
 *   wrist.R    : 16–18  (SPHERICAL, group=main)
 *   finger1.R  : 19     (REVOLUTE,  group=HandR)
 *   finger2.R  : 20     (REVOLUTE,  group=HandR)
 *   Total: 21 storage DOFs / 21 active DOFs
 *
 * Main group (17 DOFs): spine + shoulder.L + elbow.L + wrist.L
 *                     + shoulder.R + elbow.R + wrist.R
 * HandL group (2 DOFs): finger1.L + finger2.L
 * HandR group (2 DOFs): finger1.R + finger2.R
 */

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

using Catch::Approx;

#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/core/skeleton_state.hpp"
#include "posetrak/core/state.hpp"

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a compact State for @p layout with joint_angles = 0.1 * i and
/// joint_velocities = 0.01 * i (where i counts compact DOFs).
static State make_compact_state(SkeletonLayout const& layout,
                                Eigen::Vector3d const& root_pos = Eigen::Vector3d(1, 2, 3)) {
    int const n = layout.total_storage_dof_count();
    State s(n);
    s.set_root_position(root_pos);
    s.set_root_orientation(Eigen::Quaterniond::Identity());
    s.set_root_velocity(Eigen::Vector3d::Zero());
    s.set_root_angular_velocity(Eigen::Vector3d::Zero());
    Eigen::VectorXd angles(n), vels(n);
    for (int i = 0; i < n; ++i) {
        angles(i) = 0.1 * i;
        vels(i) = 0.01 * i;
    }
    s.set_joint_angles(angles);
    s.set_joint_velocities(vels);
    return s;
}

// ---------------------------------------------------------------------------
// Tests: SkeletonState::create()
// ---------------------------------------------------------------------------

TEST_CASE("SkeletonState create() preserves layout and state", "[skeleton_state]") {
    Skeleton skel = create_test_skeleton();
    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    State s = make_compact_state(*layout);

    SkeletonState ss = SkeletonState::create(layout, s);

    REQUIRE(ss.layout().get() == layout.get());
    REQUIRE(ss.state().joint_angles().isApprox(s.joint_angles()));
    REQUIRE(ss.state().root_position().isApprox(s.root_position()));
}

TEST_CASE("SkeletonState create() throws on size mismatch", "[skeleton_state]") {
    Skeleton skel = create_test_skeleton();
    auto layout =
        SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"main"});  // 17 DOFs

    State wrong_size(5);  // Wrong

    REQUIRE_THROWS_AS(SkeletonState::create(layout, wrong_size), std::invalid_argument);
}

TEST_CASE("SkeletonState create() with null layout throws", "[skeleton_state]") {
    State s(3);
    REQUIRE_THROWS_AS(SkeletonState::create(nullptr, s), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Tests: SkeletonState::merge_into()
// ---------------------------------------------------------------------------

TEST_CASE("merge_into() scatters compact DOFs into target", "[skeleton_state]") {
    Skeleton skel = create_test_skeleton();
    auto full_layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(skel));  // 21 storage DOFs
    auto main_layout = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel),
                                                   {"main"});  // 17 storage DOFs

    // Build merge_map: full_layout's state_index positions for each main DOF
    // (same as build_index_map_from)
    auto merge_map = full_layout->build_index_map_from(*main_layout);
    REQUIRE(static_cast<int>(merge_map.size()) == main_layout->total_storage_dof_count());

    // Create target full state with all-zero joint angles
    State full_state(full_layout->total_storage_dof_count());
    full_state.set_joint_angles(Eigen::VectorXd::Zero(full_layout->total_storage_dof_count()));
    full_state.set_joint_velocities(Eigen::VectorXd::Zero(full_layout->total_storage_dof_count()));
    full_state.set_root_position(Eigen::Vector3d(5, 6, 7));

    SkeletonState target = SkeletonState::create(full_layout, full_state);

    // Create subset (main) state with known values
    State main_state = make_compact_state(*main_layout);
    SkeletonState subset = SkeletonState::create(main_layout, main_state);

    // Merge subset into target
    subset.merge_into(target, merge_map);

    SECTION("Merged DOFs have subset values at merge_map positions") {
        for (int i = 0; i < main_layout->total_storage_dof_count(); ++i) {
            int idx = merge_map[i];
            REQUIRE(target.state().joint_angles()(idx) == Approx(main_state.joint_angles()(i)));
            REQUIRE(target.state().joint_velocities()(idx) ==
                    Approx(main_state.joint_velocities()(i)));
        }
    }

    SECTION("Finger DOFs (not in merge_map) remain zero") {
        // finger1.L is full_layout state_index 10 = not in main, stays 0
        REQUIRE(target.state().joint_angles()(10) == Approx(0.0));
        REQUIRE(target.state().joint_angles()(11) == Approx(0.0));
        REQUIRE(target.state().joint_angles()(19) == Approx(0.0));
        REQUIRE(target.state().joint_angles()(20) == Approx(0.0));
    }

    SECTION("Root pose NOT transferred") {
        // Target root should still be (5,6,7) — merge_into does not copy root
        REQUIRE(target.state().root_position().isApprox(Eigen::Vector3d(5, 6, 7)));
    }
}

TEST_CASE("merge_into() throws on wrong merge_map size", "[skeleton_state]") {
    Skeleton skel = create_test_skeleton();
    auto full_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    auto main_layout =
        SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"main"});

    SkeletonState target = SkeletonState::create(full_layout, make_compact_state(*full_layout));
    SkeletonState subset = SkeletonState::create(main_layout, make_compact_state(*main_layout));

    std::vector<int> bad_map(5, 0);  // Wrong size
    REQUIRE_THROWS_AS(subset.merge_into(target, bad_map), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Tests: SkeletonState round-trip (extract compact state + merge back)
// ---------------------------------------------------------------------------

TEST_CASE("Round-trip: build_index_map_from + merge_into recovers original values",
          "[skeleton_state]") {
    Skeleton skel = create_test_skeleton();
    auto full_layout =
        SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));  // 21 DOFs
    auto main_layout =
        SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"main"});  // 17 DOFs

    // Full state with a linear ramp, easy to verify
    State original_full(full_layout->total_storage_dof_count());
    Eigen::VectorXd orig_angles =
        Eigen::VectorXd::LinSpaced(full_layout->total_storage_dof_count(), 1.0, 21.0);
    original_full.set_joint_angles(orig_angles);
    original_full.set_joint_velocities(-orig_angles);  // negatives for easy distinction
    SkeletonState full_ss = SkeletonState::create(full_layout, original_full);

    // Build merge_map
    auto merge_map = full_layout->build_index_map_from(*main_layout);

    // Build compact main state from full (manually extract DOFs via merge_map)
    State main_state(main_layout->total_storage_dof_count());
    {
        Eigen::VectorXd a(main_layout->total_storage_dof_count());
        Eigen::VectorXd v(main_layout->total_storage_dof_count());
        for (int i = 0; i < static_cast<int>(merge_map.size()); ++i) {
            a(i) = orig_angles(merge_map[i]);
            v(i) = -orig_angles(merge_map[i]);
        }
        main_state.set_joint_angles(a);
        main_state.set_joint_velocities(v);
    }
    SkeletonState main_ss = SkeletonState::create(main_layout, main_state);

    // Merge into a fresh full state (all-zero angles)
    State recovery_state(full_layout->total_storage_dof_count());
    recovery_state.set_joint_angles(Eigen::VectorXd::Zero(full_layout->total_storage_dof_count()));
    recovery_state.set_joint_velocities(
        Eigen::VectorXd::Zero(full_layout->total_storage_dof_count()));
    SkeletonState recovery_ss = SkeletonState::create(full_layout, recovery_state);

    main_ss.merge_into(recovery_ss, merge_map);

    // The main-group DOFs should now match the original
    for (int i = 0; i < static_cast<int>(merge_map.size()); ++i) {
        int idx = merge_map[i];
        REQUIRE(recovery_ss.state().joint_angles()(idx) ==
                Approx(original_full.joint_angles()(idx)));
        REQUIRE(recovery_ss.state().joint_velocities()(idx) ==
                Approx(original_full.joint_velocities()(idx)));
    }
}

// ---------------------------------------------------------------------------
// Tests: SkeletonState::extract_covariance()
// ---------------------------------------------------------------------------

TEST_CASE("extract_covariance() dimensions match layout error_state_dim", "[skeleton_state]") {
    Skeleton skel = create_test_skeleton();
    auto full_layout = SkeletonLayout::from_full_skeleton(
        std::make_shared<const Skeleton>(skel));  // error_dim = 2*(6+21)=54
    auto main_layout = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel),
                                                   {"main"});  // error_dim = 2*(6+17)=46

    int const full_dim = full_layout->error_state_dim();
    REQUIRE(full_dim == 54);
    REQUIRE(main_layout->error_state_dim() == 46);

    Eigen::MatrixXd full_cov = Eigen::MatrixXd::Identity(full_dim, full_dim);

    SkeletonState main_ss = SkeletonState::create(main_layout, make_compact_state(*main_layout));
    Eigen::MatrixXd main_cov = main_ss.extract_covariance(full_cov, *full_layout);

    REQUIRE(main_cov.rows() == 46);
    REQUIRE(main_cov.cols() == 46);
}

TEST_CASE("extract_covariance() result is symmetric", "[skeleton_state]") {
    Skeleton skel = create_test_skeleton();
    auto full_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    auto main_layout =
        SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"main"});

    int const full_dim = full_layout->error_state_dim();
    // Build a random symmetric pos-def covariance
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(full_dim, full_dim);
    Eigen::MatrixXd full_cov = A * A.transpose() + Eigen::MatrixXd::Identity(full_dim, full_dim);

    SkeletonState main_ss = SkeletonState::create(main_layout, make_compact_state(*main_layout));
    Eigen::MatrixXd main_cov = main_ss.extract_covariance(full_cov, *full_layout);

    REQUIRE(main_cov.isApprox(main_cov.transpose()));
}

TEST_CASE("extract_covariance() root diagonal preserved", "[skeleton_state]") {
    Skeleton skel = create_test_skeleton();
    auto full_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    auto main_layout =
        SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"main"});

    int const full_dim = full_layout->error_state_dim();

    // Diagonal covariance with full_cov(i,i) = i + 1
    Eigen::MatrixXd full_cov = Eigen::MatrixXd::Zero(full_dim, full_dim);
    for (int i = 0; i < full_dim; ++i) {
        full_cov(i, i) = static_cast<double>(i + 1);
    }

    SkeletonState main_ss = SkeletonState::create(main_layout, make_compact_state(*main_layout));
    Eigen::MatrixXd main_cov = main_ss.extract_covariance(full_cov, *full_layout);

    // Root position+orientation entries (first 6 rows/cols of both) must match.
    for (int i = 0; i < 6; ++i) {
        REQUIRE(main_cov(i, i) == Approx(full_cov(i, i)));
    }
}

TEST_CASE("extract_covariance() joint angle entries match full layout positions",
          "[skeleton_state]") {
    Skeleton skel = create_test_skeleton();
    auto full_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    auto main_layout =
        SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"main"});

    int const full_dim = full_layout->error_state_dim();

    // Diagonal covariance: full_cov(i,i) = i + 1 so we can trace where each value came from
    Eigen::MatrixXd full_cov = Eigen::MatrixXd::Zero(full_dim, full_dim);
    for (int i = 0; i < full_dim; ++i) {
        full_cov(i, i) = static_cast<double>(i + 1);
    }

    SkeletonState main_ss = SkeletonState::create(main_layout, make_compact_state(*main_layout));
    Eigen::MatrixXd main_cov = main_ss.extract_covariance(full_cov, *full_layout);

    // For each joint in main_layout, verify that the extracted diagonal value at the
    // main error-state position matches the full_cov diagonal at the full error-state position.
    int const full_root = full_layout->root_error_dof_count();  // 6
    int const sub_root = main_layout->root_error_dof_count();   // 6

    int sub_err_cursor = sub_root;  // current position in main error-state (joint block)

    for (auto const& jdesc : main_layout->joints()) {
        JointDesc const* full_jdesc = full_layout->get_joint(jdesc.name);
        REQUIRE(full_jdesc != nullptr);

        int const full_err_start = full_root + static_cast<int>(full_jdesc->error_index);

        for (int d = 0; d < static_cast<int>(jdesc.active_dof_count); ++d) {
            // main_cov diagonal at this sub position must equal full_cov diagonal at full position
            REQUIRE(main_cov(sub_err_cursor + d, sub_err_cursor + d) ==
                    Approx(full_cov(full_err_start + d, full_err_start + d)));
        }
        sub_err_cursor += static_cast<int>(jdesc.active_dof_count);
    }
}

TEST_CASE("extract_covariance() joint velocity entries match full layout positions",
          "[skeleton_state]") {
    Skeleton skel = create_test_skeleton();
    auto full_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    auto main_layout =
        SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"main"});

    int const full_dim = full_layout->error_state_dim();

    Eigen::MatrixXd full_cov = Eigen::MatrixXd::Zero(full_dim, full_dim);
    for (int i = 0; i < full_dim; ++i) {
        full_cov(i, i) = static_cast<double>(i + 1);
    }

    SkeletonState main_ss = SkeletonState::create(main_layout, make_compact_state(*main_layout));
    Eigen::MatrixXd main_cov = main_ss.extract_covariance(full_cov, *full_layout);

    int const full_root = full_layout->root_error_dof_count();
    int const full_jac = full_layout->joint_active_dof_count();
    int const sub_root = main_layout->root_error_dof_count();
    int const sub_jac = main_layout->joint_active_dof_count();

    int const full_half = full_root + full_jac;
    int const sub_half = sub_root + sub_jac;

    // Root velocity block (positions sub_half..sub_half+5 <-> full_half..full_half+5)
    for (int i = 0; i < sub_root; ++i) {
        REQUIRE(main_cov(sub_half + i, sub_half + i) ==
                Approx(full_cov(full_half + i, full_half + i)));
    }

    // Joint velocity block
    int sub_vel_cursor = sub_half + sub_root;

    for (auto const& jdesc : main_layout->joints()) {
        JointDesc const* full_jdesc = full_layout->get_joint(jdesc.name);
        REQUIRE(full_jdesc != nullptr);

        int const full_vel_start =
            full_half + full_root + static_cast<int>(full_jdesc->error_index);

        for (int d = 0; d < static_cast<int>(jdesc.active_dof_count); ++d) {
            REQUIRE(main_cov(sub_vel_cursor + d, sub_vel_cursor + d) ==
                    Approx(full_cov(full_vel_start + d, full_vel_start + d)));
        }
        sub_vel_cursor += static_cast<int>(jdesc.active_dof_count);
    }
}

TEST_CASE("extract_covariance() throws on mismatched full_cov dimensions", "[skeleton_state]") {
    Skeleton skel = create_test_skeleton();
    auto full_layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));
    auto main_layout =
        SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel), {"main"});

    SkeletonState main_ss = SkeletonState::create(main_layout, make_compact_state(*main_layout));

    SECTION("Wrong covariance size") {
        Eigen::MatrixXd bad_cov = Eigen::MatrixXd::Identity(10, 10);
        REQUIRE_THROWS_AS(main_ss.extract_covariance(bad_cov, *full_layout), std::invalid_argument);
    }

    SECTION("Non-square covariance") {
        int const full_dim = full_layout->error_state_dim();
        Eigen::MatrixXd bad_cov(full_dim, full_dim + 1);
        REQUIRE_THROWS_AS(main_ss.extract_covariance(bad_cov, *full_layout), std::invalid_argument);
    }
}

TEST_CASE("extract_covariance() on child filter (no floating root)", "[skeleton_state]") {
    Skeleton skel = create_test_skeleton();
    auto full_layout =
        SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skel));  // has root
    auto handl_layout = SkeletonLayout::from_groups(std::make_shared<const Skeleton>(skel),
                                                    {"HandL"});  // no root, 2 DOFs

    REQUIRE(!handl_layout->has_floating_root());
    REQUIRE(handl_layout->error_state_dim() == 4);  // 2*(0+2)

    int const full_dim = full_layout->error_state_dim();  // 54
    Eigen::MatrixXd full_cov = Eigen::MatrixXd::Zero(full_dim, full_dim);
    for (int i = 0; i < full_dim; ++i) {
        full_cov(i, i) = static_cast<double>(i + 1);
    }

    SkeletonState handl_ss = SkeletonState::create(handl_layout, make_compact_state(*handl_layout));
    Eigen::MatrixXd handl_cov = handl_ss.extract_covariance(full_cov, *full_layout);

    REQUIRE(handl_cov.rows() == 4);
    REQUIRE(handl_cov.cols() == 4);

    // finger1.L: full error_index=10, so full position idx = 6+10=16 → full_cov(16,16)=17
    // finger2.L: full error_index=11, so full position idx = 6+11=17 → full_cov(17,17)=18
    // handl_cov diagonal: [finger1.L_angle, finger2.L_angle, finger1.L_vel, finger2.L_vel]
    int const full_root = full_layout->root_error_dof_count();
    JointDesc const* f1l = full_layout->get_joint("finger1.L");
    JointDesc const* f2l = full_layout->get_joint("finger2.L");
    REQUIRE(f1l != nullptr);
    REQUIRE(f2l != nullptr);

    // Position portion
    REQUIRE(handl_cov(0, 0) ==
            Approx(full_cov(full_root + f1l->error_index, full_root + f1l->error_index)));
    REQUIRE(handl_cov(1, 1) ==
            Approx(full_cov(full_root + f2l->error_index, full_root + f2l->error_index)));

    // Velocity portion
    int const full_half =
        full_layout->root_error_dof_count() + full_layout->joint_active_dof_count();
    REQUIRE(handl_cov(2, 2) == Approx(full_cov(full_half + full_root + f1l->error_index,
                                               full_half + full_root + f1l->error_index)));
    REQUIRE(handl_cov(3, 3) == Approx(full_cov(full_half + full_root + f2l->error_index,
                                               full_half + full_root + f2l->error_index)));
}

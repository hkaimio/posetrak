/**
 * @file test_tracker_child_filter.cpp
 * @brief Phase 3h tests: child filter construction, per-frame sequencing, state merge.
 *
 * Skeleton fixture:
 *   pelvis   (SPHERICAL, root, "main")  -- FreeFlyer in Pinocchio; floating body root
 *   forearm.R (REVOLUTE, "main")        -- anchor joint for hand child filter
 *   palm.R   (SPHERICAL, "HandR")       -- first child-controlled joint
 *   finger1.R (REVOLUTE, "HandR")       -- second child-controlled joint
 *
 * Rest-pose world positions (root at origin):
 *   forearm.R : (0.30, 0, 0)
 *   palm.R    : (0.55, 0, 0)   [offset 0.25 from forearm.R]
 *   finger1.R : (0.59, 0, 0)
 *
 * Markers:
 *   MRK-body  on pelvis,   offset (0, 0.1, 0) → world (0, 0.1, 0) at rest
 *   MRK-palm  on palm.R,   offset (0, 0, 0.02) → world (0.55, 0, 0.02) at rest
 *
 * Full-skeleton storage DOF layout (state_index in joint_angles):
 *   forearm.R [0]     (REVOLUTE, 1 DOF)
 *   palm.R    [1,2,3] (SPHERICAL, 3 DOFs)
 *   finger1.R [4]     (REVOLUTE, 1 DOF)
 *   total = 5
 *
 * Child layout (HandR only):
 *   palm.R    [0,1,2] (3 DOFs)
 *   finger1.R [3]     (1 DOF)
 *   merge_map = [1, 2, 3, 4]  (maps child DOF i → full state index)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/config.hpp"
#include "posetrak/core/observation.hpp"
#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/core/state.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include "posetrak/kinematics/pinocchio_model_builder.hpp"
#include "posetrak/tracking/tracker.hpp"
#include <cmath>

using namespace posetrak;
using Catch::Matchers::WithinAbs;

namespace {

// ---------------------------------------------------------------------------
// Skeleton fixture
// ---------------------------------------------------------------------------
struct HierarchicalFixture {
    std::shared_ptr<const Skeleton> skeleton;

    // Marker IDs (indices into skeleton->markers(), same as add_marker return values)
    uint32_t body_marker_id{};
    uint32_t palm_marker_id{};

    // Full-skeleton DOF count: 1 (forearm) + 3 (palm) + 1 (finger) = 5
    static constexpr int kTotalDof = 5;

    // Indices into joint_angles for palm.R DOFs (first one is most interesting)
    static constexpr int kPalmDof0 = 1;  // palm.R axis-angle[0] in full layout
    static constexpr int kPalmDof1 = 2;
    static constexpr int kPalmDof2 = 3;
    static constexpr int kFinger1Dof = 4;

    HierarchicalFixture() {
        Skeleton skel;
        uint32_t pelvis = skel.add_joint("pelvis", std::nullopt, JointType::SPHERICAL,
                                         Eigen::Vector3d::Zero());  // root → FreeFlyer
        uint32_t forearm = skel.add_joint("forearm.R", pelvis, JointType::REVOLUTE,
                                          Eigen::Vector3d(0.30, 0, 0));  // anchor joint
        uint32_t palm =
            skel.add_joint("palm.R", forearm, JointType::SPHERICAL, Eigen::Vector3d(0.25, 0, 0));
        uint32_t finger =
            skel.add_joint("finger1.R", palm, JointType::REVOLUTE, Eigen::Vector3d(0.04, 0, 0));
        (void)finger;
        skel.register_group("main", {"pelvis", "forearm.R"}, {});
        skel.register_group("HandR", {"palm.R", "finger1.R"}, {});

        body_marker_id = skel.add_marker("MRK-body", pelvis, Eigen::Vector3d(0, 0.1, 0));
        palm_marker_id = skel.add_marker("MRK-palm", palm, Eigen::Vector3d(0, 0, 0.02));

        skeleton = std::make_shared<const Skeleton>(std::move(skel));
    }

    /// Build a simple camera map: one camera at (0,0,-3) looking along +z.
    static std::unordered_map<int, Camera> make_cameras() {
        Intrinsics intr;
        intr.fx = 500.0;
        intr.fy = 500.0;
        intr.cx = 320.0;
        intr.cy = 240.0;
        intr.width = 640;
        intr.height = 480;
        intr.model = Intrinsics::DistortionModel::BrownConrady;
        intr.distortion_coeffs = {0, 0, 0, 0, 0};

        Extrinsics extr;
        extr.position = Eigen::Vector3d(0, 0, -3.0);
        extr.orientation = Eigen::Quaterniond::Identity();

        std::unordered_map<int, Camera> cameras;
        cameras.emplace(0, Camera(0, "cam0", intr, extr));
        return cameras;
    }

    /// Build a TrackerConfig with one child filter (HandR) anchored at forearm.R.
    static TrackerConfig make_config_with_child() {
        TrackerConfig config;
        // active_joint_groups empty → parent = full skeleton (all 5 DOFs)

        ChildFilterConfig ccfg;
        ccfg.name = "hand_r";
        ccfg.joint_groups = {"HandR"};
        ccfg.anchor_joint_name = "forearm.R";
        ccfg.process_noise_std = 0.05;
        ccfg.measurement_noise_std = 2.0;
        ccfg.outlier_threshold = 10.0;  // lenient for unit tests
        config.child_filters = {ccfg};
        config.measurement_noise_std = 2.0;
        config.outlier_threshold = 10.0;
        return config;
    }

    /// Generate observations from a full-skeleton state using a separate FK.
    std::vector<Observation> make_observations(State const& state) const {
        // Build stand-alone FK for observation generation
        pinocchio::Model model;
        pinocchio::Data data;
        PinocchioModelBuilder::build_model_and_data(*skeleton, model, data);
        auto marker_frame_map = PinocchioModelBuilder::build_marker_frame_map(model, *skeleton);
        auto layout = SkeletonLayout::from_full_skeleton(skeleton);
        ForwardKinematics fk(model, data, marker_frame_map, layout);

        auto marker_positions = fk.compute(state);

        auto cameras = make_cameras();
        Camera const& cam = cameras.at(0);

        std::vector<Observation> obs;
        // Markers are identified by their index in skeleton->markers()
        auto const& markers = skeleton->markers();
        for (int idx = 0; idx < static_cast<int>(markers.size()); ++idx) {
            auto const& marker = markers[static_cast<std::size_t>(idx)];
            auto it = marker_positions.find(marker.name);
            if (it == marker_positions.end())
                continue;

            auto proj = cam.project_undistorted(it->second);
            if (!proj.has_value() || !cam.is_in_bounds(*proj))
                continue;

            Observation o;
            o.camera_id = cam.id();
            o.marker_id = idx;
            o.frame_idx = 0;
            o.timestamp = 0.0;
            o.position = *proj;
            o.position_distorted = *proj;
            o.confidence = 1.0;
            obs.push_back(o);
        }
        return obs;
    }

    /// Build a zero-velocity rest-pose State with the given joint angles.
    static State make_state(Eigen::VectorXd const& joint_angles) {
        return State(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), joint_angles,
                     Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                     Eigen::VectorXd::Zero(joint_angles.size()));
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// TEST 1: Child filter construction
// ---------------------------------------------------------------------------
TEST_CASE("Child filter construction: num_children == 1 after init", "[tracker][3h]") {
    HierarchicalFixture fix;
    auto cameras = HierarchicalFixture::make_cameras();
    auto config = HierarchicalFixture::make_config_with_child();

    Tracker tracker(fix.skeleton, cameras, config);
    REQUIRE(tracker.num_children() == 0);  // not yet initialized

    tracker.initialize_from_rest_pose(0.0);
    REQUIRE(tracker.num_children() == 1);

    // Re-initializing must not double children
    tracker.initialize_from_rest_pose(0.0);
    REQUIRE(tracker.num_children() == 1);
}

// ---------------------------------------------------------------------------
// TEST 2: Per-frame sequencing — track_frame runs without crash, state valid
// ---------------------------------------------------------------------------
TEST_CASE("Child filter sequencing: track_frame produces valid state", "[tracker][3h]") {
    HierarchicalFixture fix;
    auto cameras = HierarchicalFixture::make_cameras();
    auto config = HierarchicalFixture::make_config_with_child();

    Tracker tracker(fix.skeleton, cameras, config);
    tracker.initialize_from_rest_pose(0.0);
    REQUIRE(tracker.num_children() == 1);

    // Generate observations from rest pose (all-zero joint angles)
    Eigen::VectorXd angles = Eigen::VectorXd::Zero(HierarchicalFixture::kTotalDof);
    auto obs = fix.make_observations(HierarchicalFixture::make_state(angles));
    REQUIRE_FALSE(obs.empty());

    TrackingResult result = tracker.track_frame(obs, 0.033);

    SECTION("not tracking_lost") {
        REQUIRE_FALSE(result.tracking_lost);
    }

    SECTION("state joint_angles has full-skeleton DOF count") {
        REQUIRE(tracker.state().joint_angles().size() ==
                static_cast<Eigen::Index>(HierarchicalFixture::kTotalDof));
    }

    SECTION("all joint angles are finite") {
        for (int i = 0; i < tracker.state().joint_angles().size(); ++i) {
            REQUIRE(std::isfinite(tracker.state().joint_angles()[i]));
        }
    }

    SECTION("root position is finite") {
        REQUIRE(std::isfinite(tracker.state().root_position().x()));
        REQUIRE(std::isfinite(tracker.state().root_position().y()));
        REQUIRE(std::isfinite(tracker.state().root_position().z()));
    }
}

// ---------------------------------------------------------------------------
// TEST 3: State merge — child estimates flow back into parent state
// ---------------------------------------------------------------------------
TEST_CASE("Child filter state merge: child angles appear in parent state after update",
          "[tracker][3h]") {
    HierarchicalFixture fix;
    auto cameras = HierarchicalFixture::make_cameras();
    auto config = HierarchicalFixture::make_config_with_child();

    // Initial state: palm.R[0] = 0.15 rad (small spherical rotation about x)
    Eigen::VectorXd angles = Eigen::VectorXd::Zero(HierarchicalFixture::kTotalDof);
    angles[HierarchicalFixture::kPalmDof0] = 0.15;
    auto initial_state = HierarchicalFixture::make_state(angles);

    // Generate observations from that exact pose
    auto obs = fix.make_observations(initial_state);
    REQUIRE_FALSE(obs.empty());

    Tracker tracker(fix.skeleton, cameras, config);
    tracker.initialize_from_state(initial_state, 0.0);
    REQUIRE(tracker.num_children() == 1);

    SECTION("child init: palm DOF in parent state reflects initial angles") {
        // Before any track_frame: parent UKF was set_state(initial_state),
        // so palm.R[0] should already be 0.15 in the parent state.
        REQUIRE_THAT(tracker.state().joint_angles()[HierarchicalFixture::kPalmDof0],
                     WithinAbs(0.15, 1e-9));
    }

    SECTION("after track_frame: palm DOF remains close to initial angle") {
        // Observations were generated at palm.R[0]=0.15, so the child should
        // confirm (or converge to) that angle. After merge, the parent state's
        // palm DOF should still be near 0.15.
        tracker.track_frame(obs, 0.033);
        double palm0_after = tracker.state().joint_angles()[HierarchicalFixture::kPalmDof0];

        // Loose tolerance: child UKF with a single update step won't be perfect
        REQUIRE(std::isfinite(palm0_after));
        REQUIRE_THAT(palm0_after, WithinAbs(0.15, 0.15));  // within 0.15 rad of truth
    }

    SECTION("merge wrote to all child DOF slots") {
        // All child-DOF indices (palm [1,2,3] + finger [4]) must be finite
        tracker.track_frame(obs, 0.033);
        REQUIRE(std::isfinite(tracker.state().joint_angles()[HierarchicalFixture::kPalmDof0]));
        REQUIRE(std::isfinite(tracker.state().joint_angles()[HierarchicalFixture::kPalmDof1]));
        REQUIRE(std::isfinite(tracker.state().joint_angles()[HierarchicalFixture::kPalmDof2]));
        REQUIRE(std::isfinite(tracker.state().joint_angles()[HierarchicalFixture::kFinger1Dof]));
    }
}

// ---------------------------------------------------------------------------
// TEST 4: Monolithic mode unaffected — no child_filters → num_children == 0
// ---------------------------------------------------------------------------
TEST_CASE("Monolithic tracker: no child filters with default config", "[tracker][3h]") {
    HierarchicalFixture fix;
    auto cameras = HierarchicalFixture::make_cameras();
    TrackerConfig config;  // default: no child_filters

    Tracker tracker(fix.skeleton, cameras, config);
    tracker.initialize_from_rest_pose(0.0);
    REQUIRE(tracker.num_children() == 0);

    Eigen::VectorXd angles = Eigen::VectorXd::Zero(HierarchicalFixture::kTotalDof);
    auto obs = fix.make_observations(HierarchicalFixture::make_state(angles));
    auto result = tracker.track_frame(obs, 0.033);
    REQUIRE_FALSE(result.tracking_lost);
}

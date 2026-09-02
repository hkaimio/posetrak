// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the Tracker::predict_step()/update_step() split and
 * Tracker::predict_dot_slot_predictions() -- see
 * docs/roadmap/features/marker-based-mocap/dot-assignment-architecture-design.md
 * for why the shared dot-assignment phase needs every tracked subject's
 * predict() to run before any of them commits an update.
 *
 * The core regression this file exists to catch: calling predict_step()
 * then update_step() must be observably identical to calling
 * track_frame() -- an external orchestrator inserting a dot-assignment
 * resolution step between them must not change what a subject with nothing
 * to resolve experiences.
 */
#include <posetrak/core/skeleton.hpp>
#include <posetrak/core/skeleton_layout.hpp>
#include <posetrak/io/skeleton_loader.hpp>
#include <posetrak/kinematics/forward_kinematics.hpp>
#include <posetrak/kinematics/pinocchio_model_builder.hpp>
#include <posetrak/tracking/tracker.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>

using namespace posetrak;

namespace {

std::vector<Camera> make_semicircle_cameras(int num_cameras = 3, double radius = 4.0,
                                            double height = 1.5) {
    std::vector<Camera> cameras;
    for (int i = 0; i < num_cameras; ++i) {
        double angle = M_PI * static_cast<double>(i) / static_cast<double>(num_cameras - 1);
        Eigen::Vector3d pos(radius * std::cos(angle), radius * std::sin(angle), height);

        Eigen::Vector3d target(0, 0, height);
        Eigen::Vector3d look_dir = (target - pos).normalized();
        Eigen::Vector3d up(0, 0, 1);
        Eigen::Vector3d right = look_dir.cross(up).normalized();
        up = right.cross(look_dir).normalized();

        Eigen::Matrix3d R_cam_to_world;
        R_cam_to_world.col(0) = right;
        R_cam_to_world.col(1) = -up;
        R_cam_to_world.col(2) = look_dir;
        Eigen::Matrix3d const R = R_cam_to_world.transpose();

        Intrinsics intr;
        intr.fx = 600.0;
        intr.fy = 600.0;
        intr.cx = 640.0;
        intr.cy = 360.0;
        intr.width = 1280;
        intr.height = 720;
        intr.model = Intrinsics::DistortionModel::BrownConrady;
        intr.distortion_coeffs = {0, 0, 0, 0, 0};

        Extrinsics extr;
        extr.position = pos;
        extr.orientation = Eigen::Quaterniond(R);

        cameras.emplace_back(i, "camera_" + std::to_string(i), intr, extr);
    }
    return cameras;
}

/// Builds the same articulated-skeleton + semicircle-camera fixture as
/// test_marker_projection_std.cpp, with a deterministic ground-truth
/// trajectory and synthetic noisy observations for num_frames frames
/// (frame 0 is the init frame; observations[0] is what initialize() sees).
struct ArticulatedFixture {
    Skeleton skeleton;
    std::unordered_map<int, Camera> camera_map;
    std::vector<std::vector<Observation>> observations;
    double dt;
};

ArticulatedFixture make_articulated_fixture(int num_frames = 6) {
    ArticulatedFixture fx;
    fx.skeleton = load_skeleton_from_yaml("cpp/tests/data/simple_humanoid.yaml");
    fx.dt = 1.0 / 30.0;

    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(fx.skeleton, model, data);
    auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, fx.skeleton);
    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(fx.skeleton));
    ForwardKinematics fk(model, data, marker_map, layout);

    auto cameras = make_semicircle_cameras();
    for (auto const& cam : cameras)
        fx.camera_map.emplace(cam.id(), cam);

    int const num_dof = fx.skeleton.total_dof_count();
    std::mt19937 rng(11);
    std::normal_distribution<double> noise_dist(0.0, 2.0);

    std::vector<State> ground_truth;
    for (int frame = 0; frame < num_frames; ++frame) {
        double t = frame * fx.dt;
        Eigen::VectorXd angles = Eigen::VectorXd::Zero(num_dof);
        for (int i = 0; i < num_dof; ++i) {
            double freq = 0.5 + 0.1 * (i % 5);
            angles(i) = 0.03 * std::sin(2.0 * M_PI * freq * t + i * 0.3);
        }
        ground_truth.emplace_back(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), angles,
                                  Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                  Eigen::VectorXd::Zero(num_dof));
    }

    std::vector<std::string> marker_names;
    for (auto const& m : fx.skeleton.markers())
        marker_names.push_back(m.name);

    fx.observations.resize(static_cast<size_t>(num_frames));
    for (int frame = 0; frame < num_frames; ++frame) {
        auto marker_positions = fk.compute(ground_truth[static_cast<size_t>(frame)]);
        for (size_t mi = 0; mi < marker_names.size(); ++mi) {
            auto it = marker_positions.find(marker_names[mi]);
            if (it == marker_positions.end())
                continue;
            for (auto const& cam : cameras) {
                auto proj = cam.project_undistorted(it->second);
                if (!proj.has_value())
                    continue;
                Eigen::Vector2d pos = *proj;
                pos.x() += noise_dist(rng);
                pos.y() += noise_dist(rng);
                if (!cam.is_in_bounds(pos))
                    continue;

                Observation obs;
                obs.camera_id = cam.id();
                obs.marker_id = static_cast<int>(mi);
                obs.frame_idx = frame;
                obs.timestamp = frame * fx.dt;
                obs.position = pos;
                obs.position_distorted = pos;
                obs.confidence = 0.9;
                fx.observations[static_cast<size_t>(frame)].push_back(obs);
            }
        }
    }
    return fx;
}

TrackerConfig make_fixture_config() {
    TrackerConfig config;
    config.process_noise_std = 0.1;
    config.calib_noise_std = 2.0;
    config.outlier_threshold = 4.0;
    config.init_position_std = 0.05;
    config.init_orientation_std = 0.05;
    config.init_joint_std = 0.05;
    config.init_velocity_std = 0.05;
    config.min_cameras_for_init = 2;
    config.ik_max_iterations = 1000;
    config.ik_tolerance = 0.02;
    return config;
}

void require_states_match(State const& a, State const& b, double tol) {
    REQUIRE(a.root_position().isApprox(b.root_position(), tol));
    REQUIRE(a.root_orientation().coeffs().isApprox(b.root_orientation().coeffs(), tol));
    REQUIRE(a.joint_angles().isApprox(b.joint_angles(), tol));
    REQUIRE(a.root_velocity().isApprox(b.root_velocity(), tol));
    REQUIRE(a.root_angular_velocity().isApprox(b.root_angular_velocity(), tol));
    REQUIRE(a.joint_velocities().isApprox(b.joint_velocities(), tol));
}

}  // namespace

TEST_CASE("predict_step()+update_step() matches track_frame() frame-for-frame",
          "[tracker][predict_update_split]") {
    auto fx = make_articulated_fixture(6);
    auto config = make_fixture_config();
    double const tol = 1e-9;

    auto skel_ptr = std::make_shared<const Skeleton>(fx.skeleton);

    Tracker via_track_frame(skel_ptr, fx.camera_map, config);
    Tracker via_split(skel_ptr, fx.camera_map, config);

    REQUIRE(via_track_frame.initialize(fx.observations[0], 0.0));
    REQUIRE(via_split.initialize(fx.observations[0], 0.0));

    // Both trackers must start from the identical initial state -- initialize()
    // involves IK, which is deterministic given identical input, but confirm it
    // before attributing any later mismatch to the split rather than init itself.
    require_states_match(via_track_frame.state(), via_split.state(), tol);

    for (int frame = 1; frame < static_cast<int>(fx.observations.size()); ++frame) {
        double const timestamp = frame * fx.dt;

        TrackingResult const expected =
            via_track_frame.track_frame(fx.observations[static_cast<size_t>(frame)], timestamp);

        via_split.predict_step(fx.dt);
        TrackingResult const actual =
            via_split.update_step(fx.observations[static_cast<size_t>(frame)], timestamp);

        INFO("frame=" << frame);
        REQUIRE(actual.timestamp == expected.timestamp);
        REQUIRE(actual.tracking_lost == expected.tracking_lost);
        REQUIRE(actual.failure_reason == expected.failure_reason);
        REQUIRE(actual.num_observations_used == expected.num_observations_used);
        require_states_match(actual.state, expected.state, tol);
        REQUIRE(actual.covariance.isApprox(expected.covariance, tol));

        REQUIRE(actual.update_info.num_observations == expected.update_info.num_observations);
        REQUIRE(actual.update_info.num_inliers == expected.update_info.num_inliers);
        REQUIRE(actual.update_info.num_outliers == expected.update_info.num_outliers);
        REQUIRE(actual.update_info.nis == Catch::Approx(expected.update_info.nis).margin(tol));
        REQUIRE(actual.update_info.nis_dof == expected.update_info.nis_dof);
        REQUIRE(actual.update_info.observations.size() == expected.update_info.observations.size());
        for (size_t i = 0; i < actual.update_info.observations.size(); ++i) {
            auto const& a = actual.update_info.observations[i];
            auto const& e = expected.update_info.observations[i];
            REQUIRE(a.marker_name == e.marker_name);
            REQUIRE(a.camera_id == e.camera_id);
            REQUIRE(a.is_outlier == e.is_outlier);
            REQUIRE(a.predicted.isApprox(e.predicted, tol));
            REQUIRE(a.mahalanobis_distance == Catch::Approx(e.mahalanobis_distance).margin(tol));
        }
    }

    // The two trackers must also have ended up in the identical live state, not
    // just returned identical TrackingResults -- guards against the split leaking
    // stale UKF/FK state that a subsequent frame's predict_step() would consume.
    require_states_match(via_track_frame.state(), via_split.state(), tol);
    REQUIRE(via_track_frame.covariance().isApprox(via_split.covariance(), tol));
}

TEST_CASE("update_step() throws when called without a preceding predict_step()",
          "[tracker][predict_update_split]") {
    auto fx = make_articulated_fixture(2);
    auto config = make_fixture_config();
    Tracker tracker(std::make_shared<const Skeleton>(fx.skeleton), fx.camera_map, config);
    REQUIRE(tracker.initialize(fx.observations[0], 0.0));

    REQUIRE_THROWS_AS(tracker.update_step(fx.observations[1], fx.dt), std::runtime_error);
}

TEST_CASE("predict_dot_slot_predictions() throws without a preceding predict_step()",
          "[tracker][predict_update_split]") {
    // Root-only (rigid-body) skeleton so the is_rigid_body() gate doesn't fire first --
    // this test is specifically about the predict_pending_ guard.
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_input_track("dots", "unlabeled_points");
    skeleton.add_marker("dot0", 0, Eigen::Vector3d(0.05, 0.0, 0.0), std::nullopt, "dots", "dot0");

    Intrinsics intr;
    intr.fx = 1000.0;
    intr.fy = 1000.0;
    intr.cx = 640.0;
    intr.cy = 360.0;
    intr.width = 1280;
    intr.height = 720;
    intr.model = Intrinsics::DistortionModel::BrownConrady;
    intr.distortion_coeffs = {0, 0, 0, 0, 0};
    Extrinsics extr;
    extr.position = Eigen::Vector3d(0.0, 0.0, -2.0);
    extr.orientation = Eigen::Quaterniond::Identity();
    Camera camera(0, "cam0", intr, extr);
    std::unordered_map<int, Camera> camera_map;
    camera_map.emplace(0, camera);

    TrackerConfig config;
    Tracker tracker(std::make_shared<const Skeleton>(skeleton), camera_map, config);
    State initial_state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), Eigen::VectorXd(0),
                        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::VectorXd(0));
    tracker.initialize_from_state(initial_state, 0.0);

    REQUIRE_THROWS_AS(tracker.predict_dot_slot_predictions(0), std::runtime_error);
}

TEST_CASE("predict_dot_slot_predictions() predicts an unlabeled_points marker on a rigid body",
          "[tracker][predict_dot_slot_predictions]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_input_track("dots", "unlabeled_points");
    // Local offset off the root -- a true dot slot, not the degenerate
    // zero-offset case test_marker_projection_std.cpp uses.
    Eigen::Vector3d const local_pos(0.05, 0.02, 0.0);
    skeleton.add_marker("dot0", 0, local_pos, std::nullopt, "dots", "dot0");
    // A regular (non-dot) marker must be left out of the result.
    skeleton.add_marker("labeled0", 0, Eigen::Vector3d(0.1, 0.0, 0.0));

    Intrinsics intr;
    intr.fx = 1000.0;
    intr.fy = 1000.0;
    intr.cx = 640.0;
    intr.cy = 360.0;
    intr.width = 1280;
    intr.height = 720;
    intr.model = Intrinsics::DistortionModel::BrownConrady;
    intr.distortion_coeffs = {0, 0, 0, 0, 0};
    Extrinsics extr;
    extr.position = Eigen::Vector3d(0.0, 0.0, -2.0);
    extr.orientation = Eigen::Quaterniond::Identity();
    Camera camera(0, "cam0", intr, extr);
    std::unordered_map<int, Camera> camera_map;
    camera_map.emplace(0, camera);

    TrackerConfig config;
    config.init_position_std = 0.05;
    config.init_orientation_std = 0.1;
    config.init_joint_std = 0.1;
    config.init_velocity_std = 0.1;
    Tracker tracker(std::make_shared<const Skeleton>(skeleton), camera_map, config);
    State initial_state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), Eigen::VectorXd(0),
                        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::VectorXd(0));
    tracker.initialize_from_state(initial_state, 0.0);

    tracker.predict_step(1.0 / 30.0);
    auto predictions = tracker.predict_dot_slot_predictions(0);

    REQUIRE(predictions.size() == 1);
    int dot_marker_id = -1;
    for (size_t i = 0; i < skeleton.markers().size(); ++i) {
        if (skeleton.markers()[i].name == "dot0")
            dot_marker_id = static_cast<int>(i);
    }
    REQUIRE(dot_marker_id >= 0);
    REQUIRE(predictions.count(dot_marker_id) == 1);

    // Root at world origin, identity orientation -> world position == local_pos.
    // Camera at (0,0,-2), identity orientation, looking down +Z.
    auto expected_proj = camera.project_undistorted(local_pos, /*clip_to_bounds=*/false);
    REQUIRE(expected_proj.has_value());
    MarkerPrediction const& pred = predictions.at(dot_marker_id);
    REQUIRE(pred.position.isApprox(*expected_proj, 1e-9));
    REQUIRE(pred.covariance(0, 0) > 0.0);
    REQUIRE(pred.covariance(1, 1) > 0.0);

    // A tracker still mid-frame can be asked again for a different camera without
    // needing another predict_step() -- confirms the guard is "predict happened
    // this frame", not "consumed exactly once".
    REQUIRE_NOTHROW(tracker.predict_dot_slot_predictions(0));
}

TEST_CASE("predict_dot_slot_predictions() throws for a non-rigid-body skeleton",
          "[tracker][predict_dot_slot_predictions]") {
    auto fx = make_articulated_fixture(2);
    auto config = make_fixture_config();
    Tracker tracker(std::make_shared<const Skeleton>(fx.skeleton), fx.camera_map, config);
    REQUIRE(tracker.initialize(fx.observations[0], 0.0));

    tracker.predict_step(fx.dt);
    REQUIRE_THROWS_AS(tracker.predict_dot_slot_predictions(0), std::runtime_error);
}

TEST_CASE("predict_dot_slot_predictions() throws for an unknown camera id",
          "[tracker][predict_dot_slot_predictions]") {
    Skeleton skeleton;
    skeleton.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton.add_input_track("dots", "unlabeled_points");
    skeleton.add_marker("dot0", 0, Eigen::Vector3d(0.05, 0.0, 0.0), std::nullopt, "dots", "dot0");

    Intrinsics intr;
    intr.fx = 1000.0;
    intr.fy = 1000.0;
    intr.cx = 640.0;
    intr.cy = 360.0;
    intr.width = 1280;
    intr.height = 720;
    intr.model = Intrinsics::DistortionModel::BrownConrady;
    intr.distortion_coeffs = {0, 0, 0, 0, 0};
    Extrinsics extr;
    extr.position = Eigen::Vector3d(0.0, 0.0, -2.0);
    extr.orientation = Eigen::Quaterniond::Identity();
    Camera camera(0, "cam0", intr, extr);
    std::unordered_map<int, Camera> camera_map;
    camera_map.emplace(0, camera);

    TrackerConfig config;
    Tracker tracker(std::make_shared<const Skeleton>(skeleton), camera_map, config);
    State initial_state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), Eigen::VectorXd(0),
                        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::VectorXd(0));
    tracker.initialize_from_state(initial_state, 0.0);
    tracker.predict_step(1.0 / 30.0);

    REQUIRE_THROWS_AS(tracker.predict_dot_slot_predictions(99), std::runtime_error);
}

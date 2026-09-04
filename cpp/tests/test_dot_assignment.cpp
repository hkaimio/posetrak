// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the shared dot-assignment phase (see
 * docs/roadmap/features/marker-based-mocap/dot-assignment-architecture-design.md
 * §5.2/§7.1) -- resolve_dot_assignment() (the pure core, tested here against
 * fabricated predictions/candidates, no Tracker involved) and
 * resolve_shared_dot_assignment() (the thin Tracker-calling wrapper, tested
 * against a real rigid-body Tracker fixture).
 *
 * The double-claim scenario is the whole reason this phase exists as a
 * shared, joint resolution rather than one independent solve per subject
 * (design doc §5.3) -- the two ambiguous-candidate test cases below are what
 * actually exercise that, not the straightforward single-subject cases.
 */
#include <posetrak/core/skeleton.hpp>
#include <posetrak/tracking/dot_assignment.hpp>
#include <posetrak/tracking/tracker.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>

using namespace posetrak;

namespace {

UnlabeledCandidate make_candidate(int camera_id, double px, double py) {
    UnlabeledCandidate c;
    c.camera_id = camera_id;
    c.frame_idx = 0;
    c.timestamp = 0.0;
    c.position = Eigen::Vector2d(px, py);
    c.position_distorted = c.position;
    c.confidence = 1.0;
    c.area = 10.0;
    c.compactness = 0.9;
    c.major_axis = 3.6;  // a round dot (major == minor) -- no noise-std-override kicks in
    c.minor_axis = 3.6;
    return c;
}

/// Isotropic covariance MarkerPrediction -- diag(std^2, std^2) -- enough for
/// every test here, which only cares about which candidate a prediction is
/// closest to, not a real projected-uncertainty shape.
MarkerPrediction make_prediction(double px, double py, double std = 2.0) {
    MarkerPrediction p;
    p.position = Eigen::Vector2d(px, py);
    p.covariance = Eigen::Matrix2d::Identity() * (std * std);
    return p;
}

constexpr double kGate = 9.21;  // chi-squared 99% for 2-DOF, matches the real config default

}  // namespace

TEST_CASE("resolve_dot_assignment: single subject, single candidate, clean match",
          "[dot_assignment]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][7] = make_prediction(100.0, 200.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_candidate(0, 100.5, 200.5)};

    auto result = resolve_dot_assignment({subject}, candidates, kGate, /*frame_idx=*/3, 1.5);

    REQUIRE(result.count(0) == 1);
    REQUIRE(result.at(0).resolved.size() == 1);
    Observation const& obs = result.at(0).resolved[0];
    REQUIRE(obs.camera_id == 0);
    REQUIRE(obs.marker_id == 7);
    REQUIRE(obs.frame_idx == 3);
    REQUIRE(obs.timestamp == Catch::Approx(1.5));
    REQUIRE(obs.position.isApprox(Eigen::Vector2d(100.5, 200.5)));
    REQUIRE(obs.crop_scale == Catch::Approx(0.0));
}

TEST_CASE("resolve_dot_assignment: candidate beyond the gate resolves to nothing",
          "[dot_assignment]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][7] = make_prediction(100.0, 200.0, /*std=*/1.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_candidate(0, 500.0, 500.0)};  // far outside 1px std

    auto result = resolve_dot_assignment({subject}, candidates, kGate, 0, 0.0);

    REQUIRE(result.count(0) == 0);
}

TEST_CASE("resolve_dot_assignment: a clearly-closer subject wins, the other gets nothing",
          "[dot_assignment]") {
    // Two subjects' dot slots predict to distinct positions; one candidate
    // sits right on top of subject A's prediction and far from B's -- not
    // actually ambiguous, just two subjects in the same solve.
    SubjectDotPredictions a;
    a.subject_id = 0;
    a.predictions_by_camera[0][1] = make_prediction(100.0, 100.0);

    SubjectDotPredictions b;
    b.subject_id = 1;
    b.predictions_by_camera[0][1] = make_prediction(300.0, 300.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_candidate(0, 100.2, 100.1)};

    auto result = resolve_dot_assignment({a, b}, candidates, kGate, 0, 0.0);

    REQUIRE(result.count(0) == 1);
    REQUIRE(result.at(0).resolved.size() == 1);
    REQUIRE(result.count(1) == 0);
}

TEST_CASE(
    "resolve_dot_assignment: the actual double-claim scenario -- an ambiguous "
    "candidate goes to exactly one subject, never both",
    "[dot_assignment]") {
    // Two subjects' predictions are both close to the one available
    // candidate (within gate of both) -- a naive per-subject solve would let
    // both claim it independently. The joint solve must produce exactly one
    // resolved Observation total, not two.
    SubjectDotPredictions a;
    a.subject_id = 0;
    a.predictions_by_camera[0][1] = make_prediction(100.0, 100.0, /*std=*/5.0);

    SubjectDotPredictions b;
    b.subject_id = 1;
    b.predictions_by_camera[0][1] = make_prediction(103.0, 100.0, /*std=*/5.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_candidate(0, 101.5, 100.0)};  // roughly equidistant from both

    auto result = resolve_dot_assignment({a, b}, candidates, kGate, 0, 0.0);

    int total_resolved = 0;
    for (auto const& [subject_id, assignment] : result) {
        total_resolved += static_cast<int>(assignment.resolved.size());
    }
    REQUIRE(total_resolved == 1);  // never both -- the whole point of this phase existing
}

TEST_CASE(
    "resolve_dot_assignment: genuinely equidistant candidate -- the loser gets "
    "no Observation, not a forced pairing",
    "[dot_assignment]") {
    // Exactly equidistant (same std, mirrored positions): whichever subject
    // the solver picks, the other must end up with nothing for this camera --
    // never a second, worse-fit Observation manufactured to avoid "wasting"
    // the candidate.
    SubjectDotPredictions a;
    a.subject_id = 0;
    a.predictions_by_camera[0][1] = make_prediction(99.0, 100.0);

    SubjectDotPredictions b;
    b.subject_id = 1;
    b.predictions_by_camera[0][1] = make_prediction(101.0, 100.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_candidate(0, 100.0, 100.0)};  // exactly midway

    auto result = resolve_dot_assignment({a, b}, candidates, kGate, 0, 0.0);

    bool a_won = result.count(0) == 1 && !result.at(0).resolved.empty();
    bool b_won = result.count(1) == 1 && !result.at(1).resolved.empty();
    REQUIRE(a_won != b_won);  // exactly one, via logical XOR
}

TEST_CASE("resolve_dot_assignment: two markers, one candidate each, no cross-assignment",
          "[dot_assignment]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][1] = make_prediction(50.0, 50.0);
    subject.predictions_by_camera[0][2] = make_prediction(400.0, 400.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_candidate(0, 50.5, 49.5), make_candidate(0, 400.2, 399.8)};

    auto result = resolve_dot_assignment({subject}, candidates, kGate, 0, 0.0);

    REQUIRE(result.at(0).resolved.size() == 2);
    std::vector<int> marker_ids;
    for (auto const& obs : result.at(0).resolved)
        marker_ids.push_back(obs.marker_id);
    std::sort(marker_ids.begin(), marker_ids.end());
    REQUIRE(marker_ids == std::vector<int>{1, 2});
}

TEST_CASE("resolve_dot_assignment: cameras resolve independently", "[dot_assignment]") {
    // Subject's slot-1 prediction differs per camera (as it would in
    // reality -- different projection); each camera's own candidate must
    // only ever compete against that camera's own predictions.
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][1] = make_prediction(10.0, 10.0);
    subject.predictions_by_camera[1][1] = make_prediction(900.0, 900.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_candidate(0, 10.1, 9.9)};
    candidates[1] = {make_candidate(1, 900.2, 899.9)};

    auto result = resolve_dot_assignment({subject}, candidates, kGate, 0, 0.0);

    REQUIRE(result.at(0).resolved.size() == 2);
    for (auto const& obs : result.at(0).resolved) {
        if (obs.camera_id == 0)
            REQUIRE(obs.position.isApprox(Eigen::Vector2d(10.1, 9.9)));
        else
            REQUIRE(obs.position.isApprox(Eigen::Vector2d(900.2, 899.9)));
    }
}

TEST_CASE("resolve_dot_assignment: no candidates for a camera resolves to nothing, no crash",
          "[dot_assignment]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][1] = make_prediction(10.0, 10.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {};  // present but empty -- e.g. a processed frame that saw nothing

    auto result = resolve_dot_assignment({subject}, candidates, kGate, 0, 0.0);
    REQUIRE(result.empty());
}

TEST_CASE("resolve_dot_assignment: no subjects resolves to an empty map", "[dot_assignment]") {
    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_candidate(0, 10.0, 10.0)};

    auto result = resolve_dot_assignment({}, candidates, kGate, 0, 0.0);
    REQUIRE(result.empty());
}

// ---------------------------------------------------------------------------
// resolve_shared_dot_assignment(): the Tracker-calling wrapper, against a
// real rigid-body Tracker fixture (same shape as
// test_tracker_predict_update_split.cpp's dot-slot-prediction tests).
// ---------------------------------------------------------------------------

namespace {

std::shared_ptr<Skeleton> make_rigid_dot_skeleton() {
    auto skeleton = std::make_shared<Skeleton>();
    skeleton->add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    skeleton->add_input_track("dots", "unlabeled_points");
    skeleton->add_marker("dot0", 0, Eigen::Vector3d(0.05, 0.0, 0.0), std::nullopt, "dots", "dot0");
    return skeleton;
}

Camera make_test_camera(int id, double cx_offset) {
    Intrinsics intr;
    intr.fx = 1000.0;
    intr.fy = 1000.0;
    intr.cx = 640.0 + cx_offset;
    intr.cy = 360.0;
    intr.width = 1280;
    intr.height = 720;
    intr.model = Intrinsics::DistortionModel::BrownConrady;
    intr.distortion_coeffs = {0, 0, 0, 0, 0};
    Extrinsics extr;
    extr.position = Eigen::Vector3d(0.0, 0.0, -2.0);
    extr.orientation = Eigen::Quaterniond::Identity();
    return Camera(id, "cam" + std::to_string(id), intr, extr);
}

}  // namespace

TEST_CASE("resolve_shared_dot_assignment: wires Tracker predictions into the pure core",
          "[dot_assignment]") {
    auto skeleton_a = make_rigid_dot_skeleton();
    auto skeleton_b = make_rigid_dot_skeleton();

    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, make_test_camera(0, 0.0));

    TrackerConfig config;
    config.dot_assignment_gate_mahalanobis = kGate;

    Tracker tracker_a(skeleton_a, cameras, config);
    Tracker tracker_b(skeleton_b, cameras, config);

    State state_a(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), Eigen::VectorXd(0),
                  Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::VectorXd(0));
    // Subject B's root is offset far enough that its dot slot projects well
    // away from subject A's -- not an ambiguous case, just confirming the
    // wrapper's own plumbing (predict_step() -> predict_dot_slot_predictions()
    // -> resolve_dot_assignment()) end to end.
    State state_b(Eigen::Vector3d(0.5, 0.0, 0.0), Eigen::Quaterniond::Identity(),
                  Eigen::VectorXd(0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                  Eigen::VectorXd(0));

    tracker_a.initialize_from_state(state_a, 0.0);
    tracker_b.initialize_from_state(state_b, 0.0);

    tracker_a.predict_step(1.0 / 30.0);
    tracker_b.predict_step(1.0 / 30.0);

    auto pred_a = tracker_a.predict_dot_slot_predictions(0);
    auto pred_b = tracker_b.predict_dot_slot_predictions(0);
    REQUIRE(pred_a.size() == 1);
    REQUIRE(pred_b.size() == 1);
    REQUIRE(!pred_a.begin()->second.position.isApprox(pred_b.begin()->second.position));

    std::vector<DotAssignmentSubject> subjects = {
        DotAssignmentSubject{0, &tracker_a},
        DotAssignmentSubject{1, &tracker_b},
    };
    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    // One candidate at each subject's own predicted position -- both should
    // resolve, to their own subject, via the real predict_dot_slot_predictions()
    // path rather than fabricated data this time.
    candidates[0] = {
        make_candidate(0, pred_a.begin()->second.position.x(), pred_a.begin()->second.position.y()),
        make_candidate(0, pred_b.begin()->second.position.x(), pred_b.begin()->second.position.y()),
    };

    auto result = resolve_shared_dot_assignment(subjects, candidates, config, 0, 0.0);

    REQUIRE(result.count(0) == 1);
    REQUIRE(result.count(1) == 1);
    REQUIRE(result.at(0).resolved.size() == 1);
    REQUIRE(result.at(1).resolved.size() == 1);
}

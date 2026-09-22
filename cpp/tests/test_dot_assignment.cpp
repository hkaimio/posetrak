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

/// A streak candidate: major_axis - minor_axis = elongation, with a
/// (possibly not yet sign-resolved) axis direction.
UnlabeledCandidate make_streak_candidate(int camera_id, double px, double py, double elongation,
                                         double dir_x, double dir_y) {
    UnlabeledCandidate c = make_candidate(camera_id, px, py);
    c.minor_axis = 3.6;
    c.major_axis = c.minor_axis + elongation;
    c.dir_x = dir_x;
    c.dir_y = dir_y;
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

DotAssignmentContext make_context(int frame_idx = 0, double timestamp = 0.0) {
    DotAssignmentContext context;
    context.gate_mahalanobis = kGate;
    context.frame_idx = frame_idx;
    context.timestamp = timestamp;
    return context;
}

}  // namespace

TEST_CASE("resolve_dot_assignment: single subject, single candidate, clean match",
          "[dot_assignment]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][7] = make_prediction(100.0, 200.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_candidate(0, 100.5, 200.5)};

    auto result = resolve_dot_assignment({subject}, candidates, make_context(/*frame_idx=*/3, 1.5));

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

    auto result = resolve_dot_assignment({subject}, candidates, make_context());

    REQUIRE(result.count(0) == 0);
}

// ---------------------------------------------------------------------------
// Tracklet gate relaxation -- see dot_assignment.hpp's own doc comment on
// resolve_dot_assignment()'s dot_tracklet_gate_multiplier/prev_tracklet_ids
// parameters. Real data motivated it: only ~4-6% of raw candidates survive
// this gate during a fast swing, unmodified.
// ---------------------------------------------------------------------------

TEST_CASE(
    "resolve_dot_assignment: a same-tracklet candidate beyond the plain gate is still accepted",
    "[dot_assignment]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][7] = make_prediction(100.0, 200.0, /*std=*/1.0);

    // diff=(0,3.5), mahal_sq = 3.5^2 / 1.0^2 = 12.25 -- beyond kGate=9.21 unmodified.
    UnlabeledCandidate cand = make_candidate(0, 100.0, 203.5);
    cand.tracklet_id = 42;
    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {cand};

    PrevDotTrackletIds prev_tracklet_ids;
    prev_tracklet_ids[0][0][7] = 42;  // same tracklet resolved into this slot last frame

    auto context = make_context();
    context.dot_tracklet_gate_multiplier = 2.0;
    context.prev_tracklet_ids = prev_tracklet_ids;
    auto result = resolve_dot_assignment({subject}, candidates, context);

    REQUIRE(result.count(0) == 1);
    REQUIRE(result.at(0).resolved.size() == 1);
    REQUIRE(result.at(0).resolved[0].tracklet_id == 42);
}

TEST_CASE(
    "resolve_dot_assignment: a different-tracklet candidate beyond the plain gate is still "
    "rejected",
    "[dot_assignment]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][7] = make_prediction(100.0, 200.0, /*std=*/1.0);

    UnlabeledCandidate cand = make_candidate(0, 100.0, 203.5);  // same mahal_sq=12.25 as above
    cand.tracklet_id = 99;  // does NOT match what resolved into this slot last frame
    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {cand};

    PrevDotTrackletIds prev_tracklet_ids;
    prev_tracklet_ids[0][0][7] = 42;

    auto context = make_context();
    context.dot_tracklet_gate_multiplier = 2.0;
    context.prev_tracklet_ids = prev_tracklet_ids;
    auto result = resolve_dot_assignment({subject}, candidates, context);

    REQUIRE(result.count(0) == 0);
}

TEST_CASE(
    "resolve_dot_assignment: tracklet relaxation is a no-op when the multiplier is left at 1.0",
    "[dot_assignment]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][7] = make_prediction(100.0, 200.0, /*std=*/1.0);

    UnlabeledCandidate cand = make_candidate(0, 100.0, 203.5);
    cand.tracklet_id = 42;
    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {cand};

    PrevDotTrackletIds prev_tracklet_ids;
    prev_tracklet_ids[0][0][7] = 42;  // a real match, but the multiplier default (1.0) should
                                      // still leave this pairing beyond the gate

    auto result = resolve_dot_assignment({subject}, candidates, make_context());

    REQUIRE(result.count(0) == 0);
}

TEST_CASE("resolve_dot_assignment: a resolved Observation carries its candidate's tracklet_id",
          "[dot_assignment]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][7] = make_prediction(100.0, 200.0);

    UnlabeledCandidate cand = make_candidate(0, 100.5, 200.5);
    cand.tracklet_id = 7;
    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {cand};

    auto result = resolve_dot_assignment({subject}, candidates, make_context());

    REQUIRE(result.at(0).resolved[0].tracklet_id == 7);
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

    auto result = resolve_dot_assignment({a, b}, candidates, make_context());

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

    auto result = resolve_dot_assignment({a, b}, candidates, make_context());

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

    auto result = resolve_dot_assignment({a, b}, candidates, make_context());

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

    auto result = resolve_dot_assignment({subject}, candidates, make_context());

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

    auto result = resolve_dot_assignment({subject}, candidates, make_context());

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

    auto result = resolve_dot_assignment({subject}, candidates, make_context());
    REQUIRE(result.empty());
}

TEST_CASE("resolve_dot_assignment: no subjects resolves to an empty map", "[dot_assignment]") {
    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_candidate(0, 10.0, 10.0)};

    auto result = resolve_dot_assignment({}, candidates, make_context());
    REQUIRE(result.empty());
}

// ---------------------------------------------------------------------------
// Streak-derived velocity extension (streak-velocity-design.md §3/§4) --
// resolve_dot_assignment()'s optional extra VELOCITY Observation for a
// streaked, already-continuous candidate.
// ---------------------------------------------------------------------------

TEST_CASE(
    "resolve_dot_assignment: streak velocity disabled by default -- one "
    "Observation, not two",
    "[dot_assignment][streak_velocity]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][7] = make_prediction(110.0, 100.0, /*std=*/20.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_streak_candidate(0, 110.0, 100.0, /*elongation=*/6.0, 1.0, 0.0)};

    // Default StreakVelocityConfig{} (enabled=false) and empty prev_positions --
    // exactly what an existing caller/test not passing these arguments gets.
    auto result = resolve_dot_assignment({subject}, candidates, make_context());

    REQUIRE(result.at(0).resolved.size() == 1);
    REQUIRE(result.at(0).resolved[0].mode == MeasurementMode::POSITION);
}

TEST_CASE(
    "resolve_dot_assignment: streak velocity needs a real previous position -- "
    "none available, still just one Observation",
    "[dot_assignment][streak_velocity]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][7] = make_prediction(110.0, 100.0, /*std=*/20.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_streak_candidate(0, 110.0, 100.0, 6.0, 1.0, 0.0)};

    StreakVelocityConfig streak_cfg;
    streak_cfg.enabled = true;
    streak_cfg.k_min_samples = 1;
    PrevDotPositions prev_positions;  // empty -- this exact slot was never resolved before

    auto context = make_context();
    context.streak_config = streak_cfg;
    context.prev_positions = prev_positions;
    auto result = resolve_dot_assignment({subject}, candidates, context);

    REQUIRE(result.at(0).resolved.size() == 1);
}

TEST_CASE(
    "resolve_dot_assignment: streak velocity needs real elongation -- a round "
    "dot never emits one even with real movement",
    "[dot_assignment][streak_velocity]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][7] = make_prediction(110.0, 100.0, /*std=*/20.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_candidate(0, 110.0, 100.0)};  // major==minor, no streak axis

    StreakVelocityConfig streak_cfg;
    streak_cfg.enabled = true;
    streak_cfg.k_min_samples = 1;
    PrevDotPositions prev_positions;
    prev_positions[0][0][7] = Eigen::Vector2d(100.0, 100.0);  // real, well-above-gate movement

    auto context = make_context();
    context.streak_config = streak_cfg;
    context.prev_positions = prev_positions;
    auto result = resolve_dot_assignment({subject}, candidates, context);

    REQUIRE(result.at(0).resolved.size() == 1);
}

TEST_CASE("resolve_dot_assignment: streak velocity respects the movement gate",
          "[dot_assignment][streak_velocity]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][7] = make_prediction(100.2, 100.0, /*std=*/20.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_streak_candidate(0, 100.2, 100.0, 6.0, 1.0, 0.0)};

    StreakVelocityConfig streak_cfg;
    streak_cfg.enabled = true;
    streak_cfg.k_min_samples = 1;
    streak_cfg.min_displacement_px = 3.0;
    PrevDotPositions prev_positions;
    prev_positions[0][0][7] = Eigen::Vector2d(100.0, 100.0);  // 0.2px -- below the 3.0px gate

    std::unordered_map<int, StreakKAccumulator> streak_k;
    auto context = make_context();
    context.streak_config = streak_cfg;
    context.prev_positions = prev_positions;
    context.streak_k_state = &streak_k;
    auto result = resolve_dot_assignment({subject}, candidates, context);

    REQUIRE(result.at(0).resolved.size() == 1);  // no VELOCITY observation
    REQUIRE(streak_k[0].sample_count() == 0);    // and the sample was never admitted either
}

TEST_CASE(
    "resolve_dot_assignment: streak velocity withholds the Observation until k "
    "has enough samples, but still accumulates them",
    "[dot_assignment][streak_velocity]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][7] = make_prediction(110.0, 100.0, /*std=*/20.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_streak_candidate(0, 110.0, 100.0, 6.0, 1.0, 0.0)};

    StreakVelocityConfig streak_cfg;
    streak_cfg.enabled = true;
    streak_cfg.k_min_samples = 5;  // this call's own sample is only the 1st
    PrevDotPositions prev_positions;
    prev_positions[0][0][7] = Eigen::Vector2d(100.0, 100.0);

    std::unordered_map<int, StreakKAccumulator> streak_k;
    auto context = make_context();
    context.streak_config = streak_cfg;
    context.prev_positions = prev_positions;
    context.streak_k_state = &streak_k;
    auto result = resolve_dot_assignment({subject}, candidates, context);

    REQUIRE(result.at(0).resolved.size() == 1);  // not trusted yet -- no VELOCITY observation
    REQUIRE(streak_k[0].sample_count() == 1);    // but the sample was admitted
}

TEST_CASE(
    "resolve_dot_assignment: a trusted k emits a correctly-scaled VELOCITY "
    "observation alongside the POSITION one",
    "[dot_assignment][streak_velocity]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][7] = make_prediction(110.0, 100.0, /*std=*/20.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    double const elongation = 6.0;
    candidates[0] = {make_streak_candidate(0, 110.0, 100.0, elongation, 1.0, 0.0)};

    StreakVelocityConfig streak_cfg;
    streak_cfg.enabled = true;
    streak_cfg.k_min_samples = 2;  // trusted once THIS frame's sample makes it the 2nd
    PrevDotPositions prev_positions;
    prev_positions[0][0][7] = Eigen::Vector2d(100.0, 100.0);  // disp = (10, 0), disp_px = 10

    std::unordered_map<int, StreakKAccumulator> streak_k;
    streak_k[0].add(/*streak_px=*/3.0, /*disp_px=*/10.0, /*window=*/200);  // pre-existing sample

    auto context = make_context();
    context.streak_config = streak_cfg;
    context.prev_positions = prev_positions;
    context.streak_k_state = &streak_k;
    auto result = resolve_dot_assignment({subject}, candidates, context);

    REQUIRE(result.at(0).resolved.size() == 2);
    Observation const* vel = nullptr;
    for (auto const& obs : result.at(0).resolved) {
        if (obs.mode == MeasurementMode::VELOCITY)
            vel = &obs;
    }
    REQUIRE(vel != nullptr);
    REQUIRE(vel->camera_id == 0);
    REQUIRE(vel->marker_id == 7);
    REQUIRE(vel->noise_std_override == Catch::Approx(streak_cfg.velocity_noise_std));

    // k = (3.0 + elongation) / (10.0 + 10.0) -- this frame's own sample is admitted
    // before k is read, same as resolve_dot_assignment()'s own doc comment says.
    double const expected_k = (3.0 + elongation) / (10.0 + 10.0);
    Eigen::Vector2d const expected_disp(elongation / expected_k,
                                        0.0);  // along +x, matching real disp
    Eigen::Vector2d const measured = vel->position - vel->prev_position;
    REQUIRE(measured.x() == Catch::Approx(expected_disp.x()));
    REQUIRE(measured.y() == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE(
    "resolve_dot_assignment: the streak axis's sign ambiguity is resolved "
    "against the real measured displacement, not left as detected",
    "[dot_assignment][streak_velocity]") {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][7] = make_prediction(110.0, 100.0, /*std=*/20.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    // Streak axis stored pointing -x (dot_blob_detector.py's canonicalization has no
    // way to know the true forward direction), but the real displacement is +x.
    candidates[0] = {make_streak_candidate(0, 110.0, 100.0, 6.0, -1.0, 0.0)};

    StreakVelocityConfig streak_cfg;
    streak_cfg.enabled = true;
    streak_cfg.k_min_samples = 1;
    PrevDotPositions prev_positions;
    prev_positions[0][0][7] = Eigen::Vector2d(100.0, 100.0);  // disp = (+10, 0)

    std::unordered_map<int, StreakKAccumulator> streak_k;
    auto context = make_context();
    context.streak_config = streak_cfg;
    context.prev_positions = prev_positions;
    context.streak_k_state = &streak_k;
    auto result = resolve_dot_assignment({subject}, candidates, context);

    Observation const* vel = nullptr;
    for (auto const& obs : result.at(0).resolved) {
        if (obs.mode == MeasurementMode::VELOCITY)
            vel = &obs;
    }
    REQUIRE(vel != nullptr);
    Eigen::Vector2d const measured = vel->position - vel->prev_position;
    REQUIRE(measured.x() > 0.0);  // resolved to +x (real motion), not -x (raw streak encoding)
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

TEST_CASE(
    "resolve_shared_dot_assignment: gathers a real Tracker's own "
    "prev_observations() into the streak-velocity extension",
    "[dot_assignment][streak_velocity]") {
    auto skeleton = make_rigid_dot_skeleton();
    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, make_test_camera(0, 0.0));

    TrackerConfig config;
    // Wide open -- this test is about the prev_observations()/streak_k_accumulators()
    // plumbing, not the assignment gate itself (already covered elsewhere).
    config.dot_assignment_gate_mahalanobis = 1e6;
    config.dot_streak_velocity_enabled = true;
    config.dot_streak_k_min_samples = 1;
    config.dot_streak_min_displacement_px = 1.0;

    Tracker tracker(skeleton, cameras, config);
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), Eigen::VectorXd(0),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::VectorXd(0));
    tracker.initialize_from_state(state, 0.0);

    // Frame 0: a real, successful track_frame() call is what populates
    // prev_observations() -- resolve_dot_assignment()'s own POSITION-mode
    // Observations from the previous frame aren't special-cased for this, they go
    // through the exact same bookkeeping every other Observation does.
    tracker.predict_step(1.0 / 30.0);
    auto pred0 = tracker.predict_dot_slot_predictions(0);
    REQUIRE(pred0.size() == 1);
    int const marker_id = pred0.begin()->first;
    Eigen::Vector2d const pos0 = pred0.begin()->second.position;

    Observation obs0;
    obs0.camera_id = 0;
    obs0.marker_id = marker_id;
    obs0.frame_idx = 0;
    obs0.timestamp = 1.0 / 30.0;
    obs0.position = pos0;
    obs0.position_distorted = pos0;
    obs0.confidence = 1.0;
    obs0.crop_scale = 0.0;
    auto frame0_result = tracker.update_step({obs0}, 1.0 / 30.0);
    REQUIRE_FALSE(frame0_result.tracking_lost);
    REQUIRE(tracker.prev_observations().at(0).count(marker_id) == 1);

    // Frame 1: a streaked candidate offset from frame 0's own resolved position --
    // real "movement" for the streak-velocity extension to work with.
    tracker.predict_step(1.0 / 30.0);
    std::vector<DotAssignmentSubject> subjects = {DotAssignmentSubject{0, &tracker}};
    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {make_streak_candidate(0, pos0.x() + 10.0, pos0.y(), 6.0, 1.0, 0.0)};

    auto result = resolve_shared_dot_assignment(subjects, candidates, config, 1, 2.0 / 30.0);

    REQUIRE(result.count(0) == 1);
    REQUIRE(result.at(0).resolved.size() == 2);  // POSITION + the new VELOCITY observation
    REQUIRE(tracker.streak_k_accumulators()[0].sample_count() == 1);
}

// ---------------------------------------------------------------------------
// A streaked dot's POSITION and VELOCITY Observations share one (camera,
// marker, frame) triple -- the exact case that broke UnscentedKalmanFilter::
// update()'s internal bookkeeping (see observation-results-semantics.md)
// because it matched observations by that triple
// alone, which no longer uniquely identifies one.
// ---------------------------------------------------------------------------

TEST_CASE(
    "Tracker::update_step: a marker's POSITION and VELOCITY Observations "
    "keep independent, uncorrupted diagnostics",
    "[dot_assignment][streak_velocity][observation_results]") {
    auto skeleton = make_rigid_dot_skeleton();
    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, make_test_camera(0, 0.0));

    TrackerConfig config;
    Tracker tracker(skeleton, cameras, config);
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), Eigen::VectorXd(0),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::VectorXd(0));
    tracker.initialize_from_state(state, 0.0);

    tracker.predict_step(1.0 / 30.0);
    auto pred = tracker.predict_dot_slot_predictions(0);
    REQUIRE(pred.size() == 1);
    int const marker_id = pred.begin()->first;
    Eigen::Vector2d const predicted_pos = pred.begin()->second.position;

    // Deliberately large noise overrides on both -- this test is about whether the
    // two observations' own diagnostics stay correctly paired with their own
    // observation, not about outlier gating, so both are made trivially inliers
    // regardless of the tracker's exact internal prior/velocity-reference value.
    Observation obs_position;
    obs_position.camera_id = 0;
    obs_position.marker_id = marker_id;
    obs_position.frame_idx = 0;
    obs_position.timestamp = 1.0 / 30.0;
    obs_position.position = predicted_pos + Eigen::Vector2d(15.0, -8.0);
    obs_position.position_distorted = obs_position.position;
    obs_position.confidence = 1.0;
    obs_position.crop_scale = 0.0;
    obs_position.noise_std_override = 1000.0;

    Observation obs_velocity = obs_position;
    obs_velocity.mode = MeasurementMode::VELOCITY;
    obs_velocity.position = predicted_pos;
    obs_velocity.prev_position = predicted_pos - Eigen::Vector2d(40.0, 25.0);
    obs_velocity.noise_std_override = 1000.0;

    auto result = tracker.update_step({obs_position, obs_velocity}, 1.0 / 30.0);

    ObservationResult const* pos_result = nullptr;
    ObservationResult const* vel_result = nullptr;
    for (auto const& r : result.update_info.observations) {
        if (r.mode == MeasurementMode::POSITION)
            pos_result = &r;
        if (r.mode == MeasurementMode::VELOCITY)
            vel_result = &r;
    }
    REQUIRE(pos_result != nullptr);
    REQUIRE(vel_result != nullptr);
    REQUIRE_FALSE(pos_result->is_outlier);
    REQUIRE_FALSE(vel_result->is_outlier);

    // Before the fix, the post-outlier-rejection innovation-recompute pass matched
    // "this observation" by (marker_id, camera_id, frame_idx) alone -- identical for
    // both here -- so one's recomputed predicted/innovation could land on the
    // *other*'s ObservationResult. Each must reflect its own measured value.
    Eigen::Vector2d const expected_velocity_measurement =
        obs_velocity.position - obs_velocity.prev_position;
    REQUIRE(pos_result->actual.isApprox(obs_position.position, 1e-6));
    REQUIRE(vel_result->actual.isApprox(expected_velocity_measurement, 1e-6));
    REQUIRE((pos_result->actual - pos_result->predicted).isApprox(pos_result->innovation, 1e-6));
    REQUIRE((vel_result->actual - vel_result->predicted).isApprox(vel_result->innovation, 1e-6));
}

// ---------------------------------------------------------------------------
// Shared pool across subjects: duplicate candidates, per-subject camera
// eligibility and agreement of the shared dot-assignment settings.
// ---------------------------------------------------------------------------

constexpr std::uint64_t kSubject0 = 1U << 0;
constexpr std::uint64_t kSubject1 = 1U << 1;

TEST_CASE("append_unique_candidates: a copy of another subject's candidate is merged, not added",
          "[dot_assignment][shared_pool]") {
    UnlabeledCandidate a = make_candidate(0, 100.0, 200.0);
    a.tracklet_id = 7;
    UnlabeledCandidate b = make_candidate(0, 300.0, 400.0);
    UnlabeledCandidate c = make_candidate(0, 500.0, 600.0);

    std::vector<UnlabeledCandidate> pool;
    append_unique_candidates(pool, {a, b}, kSubject0);
    REQUIRE(pool.size() == 2);
    REQUIRE(pool[0].subject_mask == kSubject0);

    // A second subject holds a byte-identical copy of `a` and one new candidate.
    append_unique_candidates(pool, {a, c}, kSubject1);
    REQUIRE(pool.size() == 3);
    REQUIRE(pool[0].subject_mask == (kSubject0 | kSubject1));  // `a` is now owned by both
    REQUIRE(pool[1].subject_mask == kSubject0);                // `b` only by the first
    REQUIRE(pool[2].subject_mask == kSubject1);                // `c` only by the second
}

TEST_CASE("append_unique_candidates: candidates that differ in any identifying field are kept",
          "[dot_assignment][shared_pool]") {
    UnlabeledCandidate a = make_candidate(0, 100.0, 200.0);
    a.tracklet_id = 7;
    std::vector<UnlabeledCandidate> pool = {a};

    UnlabeledCandidate other_tracklet = a;
    other_tracklet.tracklet_id = 8;
    UnlabeledCandidate other_frame = a;
    other_frame.frame_idx = 1;
    UnlabeledCandidate other_position = a;
    other_position.position_distorted.x() += 0.5;
    append_unique_candidates(pool, {other_tracklet, other_frame, other_position}, kSubject1);
    REQUIRE(pool.size() == 4);
}

TEST_CASE("append_unique_candidates: one subject's own list is never merged with itself",
          "[dot_assignment][shared_pool]") {
    UnlabeledCandidate a = make_candidate(0, 100.0, 200.0);
    std::vector<UnlabeledCandidate> pool;
    append_unique_candidates(pool, {a, a}, kSubject0);
    REQUIRE(pool.size() == 2);
}

TEST_CASE("find_dot_config_disagreement: names the first setting that differs",
          "[dot_assignment][shared_pool]") {
    TrackerConfig a;
    TrackerConfig b;
    REQUIRE(find_dot_config_disagreement({&a, &b}).empty());
    REQUIRE(find_dot_config_disagreement({&a}).empty());

    b.dot_assignment_gate_mahalanobis = a.dot_assignment_gate_mahalanobis + 1.0;
    REQUIRE(find_dot_config_disagreement({&a, &b}) == "dot_assignment_gate_mahalanobis");

    b = a;
    b.dot_streak_velocity_enabled = !a.dot_streak_velocity_enabled;
    REQUIRE(find_dot_config_disagreement({&a, &b}) == "dot_streak_velocity_enabled");

    // A setting the shared solve does not read is not a disagreement.
    b = a;
    b.outlier_threshold = a.outlier_threshold + 1.0;
    REQUIRE(find_dot_config_disagreement({&a, &b}).empty());
}

TEST_CASE("resolve_shared_dot_assignment: a subject cannot claim a candidate it does not own",
          "[dot_assignment][shared_pool]") {
    auto skeleton = make_rigid_dot_skeleton();
    std::unordered_map<int, Camera> cameras;
    cameras.emplace(0, make_test_camera(0, 0.0));

    TrackerConfig config;
    config.dot_assignment_gate_mahalanobis = kGate;
    Tracker tracker(skeleton, cameras, config);
    State state(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), Eigen::VectorXd(0),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::VectorXd(0));
    tracker.initialize_from_state(state, 0.0);
    tracker.predict_step(1.0 / 30.0);

    // A candidate exactly at the subject's prediction, so only ownership can
    // keep it from being claimed.
    auto pred = tracker.predict_dot_slot_predictions(0);
    UnlabeledCandidate candidate =
        make_candidate(0, pred.begin()->second.position.x(), pred.begin()->second.position.y());
    std::vector<DotAssignmentSubject> subjects = {DotAssignmentSubject{0, &tracker}};

    auto const claimed = [&](std::uint64_t owners) {
        candidate.subject_mask = owners;
        std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
        candidates[0] = {candidate};
        auto result = resolve_shared_dot_assignment(subjects, candidates, config, 0, 0.0);
        auto it = result.find(0);  // a subject that claims nothing has no entry
        return it == result.end() ? size_t{0} : it->second.resolved.size();
    };

    REQUIRE(claimed(kSubject0) == 1);              // owned by subject 0
    REQUIRE(claimed(kSubject0 | kSubject1) == 1);  // shared: either may claim it
    REQUIRE(claimed(kSubject1) == 0);              // owned by another subject only
}

// ---------------------------------------------------------------------------
// The cost modifiers on their own, and the chain default_cost_modifiers() builds.
// ---------------------------------------------------------------------------

TEST_CASE("TrackletContinuityModifier: supports a candidate continuing the slot's tracklet",
          "[dot_assignment]") {
    MarkerPrediction const prediction = make_prediction(100.0, 200.0);
    DotSlotRef const slot{/*subject_id=*/0, /*camera_id=*/0, /*marker_id=*/7, &prediction};
    DotAssignmentContext context = make_context();
    context.dot_tracklet_gate_multiplier = 2.5;
    context.prev_tracklet_ids[0][0][7] = 42;

    UnlabeledCandidate candidate = make_candidate(0, 100.0, 200.0);
    candidate.tracklet_id = 42;
    TrackletContinuityModifier const modifier;

    CostSupport const same = modifier.apply(slot, candidate, context);
    REQUIRE(same.factor == 2.5);
    REQUIRE_FALSE(same.exclude);

    candidate.tracklet_id = 43;  // another tracklet
    REQUIRE(modifier.apply(slot, candidate, context).factor == 1.0);

    candidate.tracklet_id = -1;  // no tracklet at all
    REQUIRE(modifier.apply(slot, candidate, context).factor == 1.0);

    candidate.tracklet_id = 42;
    DotSlotRef const other_marker{0, 0, 8, &prediction};  // the slot has no previous tracklet
    REQUIRE(modifier.apply(other_marker, candidate, context).factor == 1.0);
    DotSlotRef const other_camera{0, 1, 7, &prediction};
    REQUIRE(modifier.apply(other_camera, candidate, context).factor == 1.0);
    DotSlotRef const other_subject{1, 0, 7, &prediction};
    REQUIRE(modifier.apply(other_subject, candidate, context).factor == 1.0);
}

TEST_CASE("CandidateOwnershipModifier: excludes a candidate the subject does not own",
          "[dot_assignment]") {
    MarkerPrediction const prediction = make_prediction(100.0, 200.0);
    DotAssignmentContext const context = make_context();
    UnlabeledCandidate candidate = make_candidate(0, 100.0, 200.0);
    candidate.subject_mask = 0b10;  // owned by subject 1 only
    CandidateOwnershipModifier const modifier;

    REQUIRE(modifier.apply({0, 0, 7, &prediction}, candidate, context).exclude);
    REQUIRE_FALSE(modifier.apply({1, 0, 7, &prediction}, candidate, context).exclude);
    // A subject id beyond the mask's 64 bits cannot be denied a candidate.
    REQUIRE_FALSE(modifier.apply({64, 0, 7, &prediction}, candidate, context).exclude);
}

TEST_CASE("default_cost_modifiers: ownership always, tracklet continuity only when configured",
          "[dot_assignment]") {
    DotAssignmentContext context = make_context();
    auto modifiers = default_cost_modifiers(context);
    REQUIRE(modifiers.size() == 1);
    REQUIRE(dynamic_cast<CandidateOwnershipModifier*>(modifiers[0].get()) != nullptr);

    context.dot_tracklet_gate_multiplier = 2.0;  // no previous tracklet ids yet
    REQUIRE(default_cost_modifiers(context).size() == 1);

    context.prev_tracklet_ids[0][0][7] = 1;
    modifiers = default_cost_modifiers(context);
    REQUIRE(modifiers.size() == 2);
    REQUIRE(dynamic_cast<TrackletContinuityModifier*>(modifiers[0].get()) != nullptr);
    REQUIRE(dynamic_cast<CandidateOwnershipModifier*>(modifiers[1].get()) != nullptr);

    context.dot_reacquire_gap_frames = 5;
    modifiers = default_cost_modifiers(context);
    REQUIRE(modifiers.size() == 3);
    REQUIRE(dynamic_cast<ReacquisitionGateModifier*>(modifiers[2].get()) != nullptr);
}

// ---------------------------------------------------------------------------
// ReacquisitionGateModifier and the near-prediction-cameras pre-pass it reads
// through resolve_dot_assignment() (§2.2/§2.3 of the WS2 assignment
// robustness plan).
// ---------------------------------------------------------------------------

TEST_CASE(
    "ReacquisitionGateModifier: no-op for a slot that has never resolved or is not yet gapped",
    "[dot_assignment]") {
    MarkerPrediction const prediction = make_prediction(100.0, 200.0);
    DotSlotRef const slot{0, 0, 7, &prediction};
    UnlabeledCandidate const candidate =
        make_candidate(0, 500.0, 500.0);  // far from the prediction
    ReacquisitionGateModifier const modifier;

    DotAssignmentContext context = make_context(/*frame_idx=*/10);
    context.dot_reacquire_gap_frames = 3;
    REQUIRE_FALSE(modifier.apply(slot, candidate, context).exclude);  // never resolved

    context.prev_resolved_frame[0][0][7] = 8;  // gap of 2, below the threshold of 3
    REQUIRE_FALSE(modifier.apply(slot, candidate, context).exclude);
}

TEST_CASE("ReacquisitionGateModifier: excludes a gapped slot with no independent evidence",
          "[dot_assignment]") {
    MarkerPrediction const prediction = make_prediction(100.0, 200.0);
    DotSlotRef const slot{0, 0, 7, &prediction};
    UnlabeledCandidate const candidate = make_candidate(0, 500.0, 500.0);
    ReacquisitionGateModifier const modifier;

    DotAssignmentContext context = make_context(/*frame_idx=*/10);
    context.dot_reacquire_gap_frames = 3;
    context.prev_resolved_frame[0][0][7] = 5;  // gap of 5, at/above the threshold

    REQUIRE(modifier.apply(slot, candidate, context).exclude);
}

TEST_CASE("ReacquisitionGateModifier: tracklet continuity admits a gapped slot",
          "[dot_assignment]") {
    MarkerPrediction const prediction = make_prediction(100.0, 200.0);
    DotSlotRef const slot{0, 0, 7, &prediction};
    UnlabeledCandidate candidate = make_candidate(0, 500.0, 500.0);
    candidate.tracklet_id = 42;
    ReacquisitionGateModifier const modifier;

    DotAssignmentContext context = make_context(/*frame_idx=*/10);
    context.dot_reacquire_gap_frames = 3;
    context.prev_resolved_frame[0][0][7] = 5;
    context.prev_tracklet_ids[0][0][7] = 42;

    REQUIRE_FALSE(modifier.apply(slot, candidate, context).exclude);
}

TEST_CASE("ReacquisitionGateModifier: two corroborating cameras admit a gapped slot, one does not",
          "[dot_assignment]") {
    MarkerPrediction const prediction = make_prediction(100.0, 200.0);
    DotSlotRef const slot{0, 0, 7, &prediction};
    UnlabeledCandidate const candidate = make_candidate(0, 500.0, 500.0);
    ReacquisitionGateModifier const modifier;

    DotAssignmentContext context = make_context(/*frame_idx=*/10);
    context.dot_reacquire_gap_frames = 3;
    context.prev_resolved_frame[0][0][7] = 5;

    context.near_prediction_cameras[0][7] = {0};  // only this camera -- not enough on its own
    REQUIRE(modifier.apply(slot, candidate, context).exclude);

    context.near_prediction_cameras[0][7] = {0, 1};  // corroborated by a second camera
    REQUIRE_FALSE(modifier.apply(slot, candidate, context).exclude);
}

TEST_CASE(
    "resolve_dot_assignment: a gapped slot skips an unsupported candidate and resolves a "
    "corroborated one",
    "[dot_assignment]") {
    // Two subjects share no data here -- one slot, three cameras, the middle one gapped.
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][7] = make_prediction(100.0, 200.0);
    subject.predictions_by_camera[1][7] = make_prediction(300.0, 400.0);
    subject.predictions_by_camera[2][7] = make_prediction(500.0, 600.0);

    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    // Camera 0: near its own prediction -- corroborating evidence for the gate.
    candidates[0] = {make_candidate(0, 100.5, 200.5)};
    // Camera 1: the gapped slot's own candidate, also near its prediction -- with camera 0's
    // corroboration this is 2 cameras, so the gate should admit it.
    candidates[1] = {make_candidate(1, 300.5, 400.5)};
    // Camera 2: not gapped, so the reacquisition gate never applies to it -- its
    // candidate is far from the prediction and is rejected by the ordinary
    // assignment gate instead, same as any ungated slot.
    candidates[2] = {make_candidate(2, 999.0, 999.0)};

    DotAssignmentContext context = make_context(/*frame_idx=*/10);
    context.dot_reacquire_gap_frames = 3;
    context.dot_reacquire_max_px = 5.0;
    context.prev_resolved_frame[0][1][7] = 5;  // camera 1's copy of the slot is gapped

    auto result = resolve_dot_assignment({subject}, candidates, context);

    REQUIRE(result.count(0) == 1);
    std::vector<int> resolved_cameras;
    for (auto const& obs : result.at(0).resolved)
        resolved_cameras.push_back(obs.camera_id);
    std::sort(resolved_cameras.begin(), resolved_cameras.end());
    REQUIRE(resolved_cameras == std::vector<int>{0, 1});  // camera 2's candidate is too far to gate
}

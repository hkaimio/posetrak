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

    auto result = resolve_dot_assignment({subject}, candidates, kGate, 0, 0.0, 5.0, {}, {}, nullptr,
                                         /*dot_tracklet_gate_multiplier=*/2.0, prev_tracklet_ids);

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

    auto result = resolve_dot_assignment({subject}, candidates, kGate, 0, 0.0, 5.0, {}, {}, nullptr,
                                         /*dot_tracklet_gate_multiplier=*/2.0, prev_tracklet_ids);

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

    auto result = resolve_dot_assignment({subject}, candidates, kGate, 0, 0.0);

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

    auto result = resolve_dot_assignment({subject}, candidates, kGate, 0, 0.0);

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
    auto result = resolve_dot_assignment({subject}, candidates, kGate, 0, 0.0);

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

    auto result = resolve_dot_assignment({subject}, candidates, kGate, 0, 0.0, 5.0, streak_cfg,
                                         prev_positions);

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

    auto result = resolve_dot_assignment({subject}, candidates, kGate, 0, 0.0, 5.0, streak_cfg,
                                         prev_positions);

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
    auto result = resolve_dot_assignment({subject}, candidates, kGate, 0, 0.0, 5.0, streak_cfg,
                                         prev_positions, &streak_k);

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
    auto result = resolve_dot_assignment({subject}, candidates, kGate, 0, 0.0, 5.0, streak_cfg,
                                         prev_positions, &streak_k);

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

    auto result = resolve_dot_assignment({subject}, candidates, kGate, 0, 0.0, 5.0, streak_cfg,
                                         prev_positions, &streak_k);

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
    auto result = resolve_dot_assignment({subject}, candidates, kGate, 0, 0.0, 5.0, streak_cfg,
                                         prev_positions, &streak_k);

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

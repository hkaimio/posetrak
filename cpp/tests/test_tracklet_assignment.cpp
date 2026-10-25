// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for decisions about whole dot tracklets (tracklet_assignment.hpp) and for their use in
 * the shared dot assignment (TrackletAssignmentModifier).
 */
#include <posetrak/tracking/dot_assignment.hpp>
#include <posetrak/tracking/tracklet_assignment.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>

using namespace posetrak;

namespace {

UnlabeledCandidate candidate(int camera_id, int tracklet_id, int frame_idx, double px, double py) {
    UnlabeledCandidate c;
    c.camera_id = camera_id;
    c.tracklet_id = tracklet_id;
    c.frame_idx = frame_idx;
    c.timestamp = 0.0;
    c.position = Eigen::Vector2d(px, py);
    c.position_distorted = c.position;
    c.confidence = 1.0;
    c.area = 10.0;
    c.compactness = 0.9;
    c.major_axis = 3.6;
    c.minor_axis = 3.6;
    return c;
}

MarkerPrediction prediction(double px, double py) {
    MarkerPrediction p;
    p.position = Eigen::Vector2d(px, py);
    p.covariance = Eigen::Matrix2d::Identity() * 4.0;
    return p;
}

DotAssignmentContext context_with(TrackletAssignments const& rules, int frame_idx = 0) {
    DotAssignmentContext context;
    context.frame_idx = frame_idx;
    context.tracklet_assignments = &rules;
    return context;
}

/// Marker 1 is predicted at (100, 100), marker 2 at (160, 100); one dot at (101, 100)
/// sits on marker 1's prediction, and a far dot (200, 100) matches nothing.
SubjectDotPredictions two_markers() {
    SubjectDotPredictions subject;
    subject.subject_id = 0;
    subject.predictions_by_camera[0][1] = prediction(100.0, 100.0);
    subject.predictions_by_camera[0][2] = prediction(160.0, 100.0);
    return subject;
}

int marker_taken_by(std::unordered_map<int, SubjectDotAssignment> const& result, double x) {
    auto it = result.find(0);
    if (it == result.end())
        return -1;
    for (auto const& obs : it->second.resolved) {
        if (std::abs(obs.position.x() - x) < 1e-6)
            return obs.marker_id;
    }
    return -1;
}

}  // namespace

TEST_CASE("TrackletAssignments: the latest assign or reject covering a frame wins",
          "[tracklet_assignment]") {
    TrackletAssignments rules;
    rules.add(0, 7, {TrackletRuleKind::SUGGEST, 3, INT_MIN, INT_MAX});
    rules.add(0, 7, {TrackletRuleKind::FORBID, 3, INT_MIN, INT_MAX});
    rules.add(0, 7, {TrackletRuleKind::ASSIGN, 5, 10, 19});

    auto inside = rules.verdict(0, 7, 12);
    CHECK(inside.kind == TrackletVerdict::Kind::ASSIGNED);
    CHECK(inside.assigned_marker == 5);

    auto outside = rules.verdict(0, 7, 25);  // the assign covers 10..19 only
    CHECK(outside.kind == TrackletVerdict::Kind::NONE);
    REQUIRE(outside.forbidden_markers.size() == 1);
    CHECK(outside.forbidden_markers[0] == 3);

    rules.add(0, 7, {TrackletRuleKind::REJECT, -1, 15, 30});
    CHECK(rules.verdict(0, 12, 12).kind == TrackletVerdict::Kind::NONE);  // another tracklet
    CHECK(rules.verdict(0, 7, 12).kind == TrackletVerdict::Kind::ASSIGNED);
    CHECK(rules.verdict(0, 7, 16).kind ==
          TrackletVerdict::Kind::EXCLUDED);  // the later reject wins
}

TEST_CASE("TrackletAssignments: a later forbid overrides an earlier assign of the same marker",
          "[tracklet_assignment]") {
    TrackletAssignments rules;
    rules.add(0, 7, {TrackletRuleKind::ASSIGN, 5, INT_MIN, INT_MAX});  // e.g. an automatic proposal
    rules.add(0, 7,
              {TrackletRuleKind::FORBID, 5, INT_MIN, INT_MAX});  // then a person says it is not
    auto verdict = rules.verdict(0, 7, 0);
    CHECK(verdict.kind == TrackletVerdict::Kind::NONE);
    CHECK(verdict.forbidden_markers.size() == 1);

    rules.add(0, 7, {TrackletRuleKind::ASSIGN, 6, INT_MIN, INT_MAX});  // and then what it is
    CHECK(rules.verdict(0, 7, 0).assigned_marker == 6);
}

TEST_CASE("TrackletAssignmentModifier: an assign takes a dot the gate would reject",
          "[tracklet_assignment]") {
    // The dot at (100 + 40, 100) is 20 sigma from marker 1's prediction: gated out without a rule.
    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {candidate(0, 7, 0, 140.0, 100.0)};
    TrackletAssignments none;
    CHECK(marker_taken_by(resolve_dot_assignment({two_markers()}, candidates, context_with(none)),
                          140.0) == -1);

    TrackletAssignments rules;
    rules.add(0, 7, {TrackletRuleKind::ASSIGN, 1, INT_MIN, INT_MAX});
    CHECK(marker_taken_by(resolve_dot_assignment({two_markers()}, candidates, context_with(rules)),
                          140.0) == 1);
}

TEST_CASE("TrackletAssignmentModifier: an assign keeps the dot off other markers",
          "[tracklet_assignment]") {
    // The dot is on marker 2's prediction, but its tracklet is marker 1.
    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {candidate(0, 7, 0, 160.5, 100.0)};
    TrackletAssignments rules;
    rules.add(0, 7, {TrackletRuleKind::ASSIGN, 1, INT_MIN, INT_MAX});
    auto result = resolve_dot_assignment({two_markers()}, candidates, context_with(rules));
    CHECK(marker_taken_by(result, 160.5) != 2);
}

TEST_CASE(
    "TrackletAssignmentModifier: reject, other and forbid exclude, a rule outside its frames does "
    "not",
    "[tracklet_assignment]") {
    std::unordered_map<int, std::vector<UnlabeledCandidate>> candidates;
    candidates[0] = {candidate(0, 7, 4, 101.0, 100.0)};

    TrackletAssignments reject;
    reject.add(0, 7, {TrackletRuleKind::REJECT, -1, INT_MIN, INT_MAX});
    CHECK(marker_taken_by(
              resolve_dot_assignment({two_markers()}, candidates, context_with(reject, 4)),
              101.0) == -1);

    TrackletAssignments other;
    other.add(0, 7, {TrackletRuleKind::OTHER, -1, INT_MIN, INT_MAX});
    CHECK(
        marker_taken_by(resolve_dot_assignment({two_markers()}, candidates, context_with(other, 4)),
                        101.0) == -1);

    TrackletAssignments forbid;
    forbid.add(0, 7, {TrackletRuleKind::FORBID, 1, INT_MIN, INT_MAX});
    CHECK(marker_taken_by(
              resolve_dot_assignment({two_markers()}, candidates, context_with(forbid, 4)),
              101.0) == -1);

    TrackletAssignments later;  // the candidate is in frame 4; the reject covers 10 and up
    later.add(0, 7, {TrackletRuleKind::REJECT, -1, 10, INT_MAX});
    CHECK(
        marker_taken_by(resolve_dot_assignment({two_markers()}, candidates, context_with(later, 4)),
                        101.0) == 1);
}

TEST_CASE("load_tracklet_assignments: reads rules and skips unknown cameras and markers",
          "[tracklet_assignment]") {
    auto const path =
        (std::filesystem::temp_directory_path() / "tracklet_assignments_test.csv").string();
    {
        std::ofstream out(path);
        out << "camera_label,tracklet_id,first_frame,last_frame,kind,marker_name\n"
            << "camA,7,,,assign,wrist\n"
            << "camA,8,100,199,forbid,wrist\n"
            << "camA,9,,,reject,\n"
            << "camB,7,,,assign,wrist\n"     // unknown camera
            << "camA,7,,,assign,no_such\n";  // unknown marker
    }
    auto const file = load_tracklet_assignments(path, {{"camA", 3}}, {{"wrist", 12}});
    std::remove(path.c_str());
    CHECK(file.rows_read == 5);
    CHECK(file.rows_skipped == 2);
    CHECK(file.assignments.size() == 3);
    CHECK(file.assignments.verdict(3, 7, 50).assigned_marker == 12);
    CHECK(file.assignments.verdict(3, 8, 50).forbidden_markers.empty());  // outside 100..199
    REQUIRE(file.assignments.verdict(3, 8, 150).forbidden_markers.size() == 1);
    CHECK(file.assignments.verdict(3, 9, 0).kind == TrackletVerdict::Kind::EXCLUDED);
}

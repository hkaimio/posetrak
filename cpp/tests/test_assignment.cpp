// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/tracking/assignment.hpp"
#include <limits>

using namespace posetrak;
using Catch::Matchers::WithinAbs;

namespace {

bool has_pair(std::vector<AssignmentPair> const& result, int row, int col) {
    for (auto const& p : result) {
        if (p.row == row && p.col == col)
            return true;
    }
    return false;
}

constexpr double kInf = std::numeric_limits<double>::infinity();

}  // namespace

TEST_CASE("solve_assignment: square matrix, obvious optimal pairing", "[assignment]") {
    // 3x3: row i's cheapest column is column i, by a wide margin.
    std::vector<double> cost = {
        1.0,  10.0, 10.0,  //
        10.0, 1.0,  10.0,  //
        10.0, 10.0, 1.0,   //
    };
    auto result = solve_assignment(cost, 3, 3, kInf);
    REQUIRE(result.size() == 3);
    REQUIRE(has_pair(result, 0, 0));
    REQUIRE(has_pair(result, 1, 1));
    REQUIRE(has_pair(result, 2, 2));
}

TEST_CASE("solve_assignment: prefers globally optimal over greedy-first-match", "[assignment]") {
    // Row 0's best match (col 0, cost 1) is also row 1's only good match
    // (col 0, cost 2; col 1, cost 100). A greedy-first solver processing
    // row 0 first would grab col 0 for row 0 and strand row 1 with cost
    // 100; the true optimum gives col 0 to row 1 (saving 99) and col 1 to
    // row 0 (costing only 9 more) for a total of 2+9=11 vs. greedy's 1+100=101.
    std::vector<double> cost = {
        1.0,
        10.0,  //
        2.0,
        100.0,  //
    };
    auto result = solve_assignment(cost, 2, 2, kInf);
    REQUIRE(result.size() == 2);
    REQUIRE(has_pair(result, 1, 0));
    REQUIRE(has_pair(result, 0, 1));
}

TEST_CASE("solve_assignment: more candidates than slots (non-square, extra rows)", "[assignment]") {
    // 3 candidates (rows), 2 slots (cols). Candidate 1 is a clear outlier
    // (expensive against both slots) and should be left unmatched.
    std::vector<double> cost = {
        1.0,  10.0,  // candidate 0 -> slot 0
        50.0, 60.0,  // candidate 1 -> neither slot fits well
        10.0, 1.0,   // candidate 2 -> slot 1
    };
    auto result = solve_assignment(cost, 3, 2, kInf);
    REQUIRE(result.size() == 2);
    REQUIRE(has_pair(result, 0, 0));
    REQUIRE(has_pair(result, 2, 1));
}

TEST_CASE("solve_assignment: more slots than candidates (non-square, extra columns)",
          "[assignment]") {
    // 2 candidates (rows), 3 slots (cols) -- one slot must go unmatched.
    std::vector<double> cost = {
        1.0,  10.0, 10.0,  // candidate 0 -> slot 0
        10.0, 10.0, 1.0,   // candidate 1 -> slot 2
    };
    auto result = solve_assignment(cost, 2, 3, kInf);
    REQUIRE(result.size() == 2);
    REQUIRE(has_pair(result, 0, 0));
    REQUIRE(has_pair(result, 1, 2));
}

TEST_CASE("solve_assignment: gating drops a pairing above threshold instead of forcing it",
          "[assignment]") {
    // Single candidate, single slot, but the only possible pairing costs
    // more than the gate -- "drop, don't guess" (marker-detection-analysis.md):
    // must return empty, not a forced above-gate pairing.
    std::vector<double> cost = {100.0};
    auto result = solve_assignment(cost, 1, 1, /*gate=*/5.0);
    REQUIRE(result.empty());
}

TEST_CASE("solve_assignment: gating accepts a pairing at or below threshold", "[assignment]") {
    std::vector<double> cost = {5.0};
    auto result = solve_assignment(cost, 1, 1, /*gate=*/5.0);
    REQUIRE(result.size() == 1);
    REQUIRE(has_pair(result, 0, 0));
}

TEST_CASE("solve_assignment: gating is per-pair, not all-or-nothing", "[assignment]") {
    // 2x2: (0,0) is cheap and within gate; (1,1) is expensive and outside
    // gate. Only the good pairing should survive, even though a complete
    // assignment over the padded matrix technically produces a mapping
    // for every row.
    std::vector<double> cost = {
        1.0,
        1000.0,  //
        1000.0,
        50.0,  //
    };
    auto result = solve_assignment(cost, 2, 2, /*gate=*/10.0);
    REQUIRE(result.size() == 1);
    REQUIRE(has_pair(result, 0, 0));
}

TEST_CASE("solve_assignment: empty input returns empty result", "[assignment]") {
    std::vector<double> cost;
    REQUIRE(solve_assignment(cost, 0, 0, kInf).empty());
}

TEST_CASE("solve_assignment: returned cost matches the input cost matrix", "[assignment]") {
    std::vector<double> cost = {
        3.5,
        10.0,  //
        10.0,
        2.25,  //
    };
    auto result = solve_assignment(cost, 2, 2, kInf);
    REQUIRE(result.size() == 2);
    for (auto const& p : result) {
        double expected = cost[static_cast<size_t>(p.row) * 2 + static_cast<size_t>(p.col)];
        REQUIRE_THAT(p.cost, WithinAbs(expected, 1e-9));
    }
}

TEST_CASE("solve_assignment: several-tens scale is exact (real dot-assignment magnitude)",
          "[assignment]") {
    // Sanity check at the scale the design doc's own scaling analysis
    // (dot-assignment-architecture-design.md sec 7) assumes -- a diagonal-
    // dominant 40x40 problem should still recover the diagonal exactly.
    constexpr int n = 40;
    std::vector<double> cost(static_cast<size_t>(n) * static_cast<size_t>(n), 100.0);
    for (int i = 0; i < n; ++i) {
        cost[static_cast<size_t>(i) * n + static_cast<size_t>(i)] = 1.0;
    }
    auto result = solve_assignment(cost, n, n, kInf);
    REQUIRE(result.size() == static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        REQUIRE(has_pair(result, i, i));
    }
}

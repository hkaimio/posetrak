// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file assignment.hpp
 * @brief Gated rectangular assignment (Hungarian algorithm) for anonymous
 * reflective-dot candidate resolution -- marker-based-mocap design doc's
 * dot-assignment-architecture-design.md, sub-phase C2.1.
 *
 * Header-only, matching posetrak/db/blob_codec.hpp's own convention for a
 * small, self-contained, independently-testable piece of math with no
 * tracker/skeleton/camera dependency (see that design doc's §7/§10).
 */
#pragma once

#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

namespace posetrak {

/// @brief One resolved (row, column) pairing from solve_assignment(), plus
/// its cost -- callers needing per-pair cost (e.g. downstream diagnostics)
/// don't have to re-index into the original cost matrix.
struct AssignmentPair {
    int row;
    int col;
    double cost;
};

/// @brief Solve a rectangular minimum-cost bipartite assignment with gating,
/// via the Hungarian (Kuhn-Munkres) algorithm.
///
/// *cost* is `n_rows` x `n_cols` (row-major: `cost[r * n_cols + c]`), any
/// non-negative real cost -- for this project's use (marker-mocap design
/// doc's dot-assignment-architecture-design.md §7), rows are a camera's
/// candidate detections and columns are the union of every participating
/// subject's dot slots that frame, and cost is squared Mahalanobis
/// distance, but nothing here assumes that.
///
/// Handles non-square matrices by internally padding to square with
/// dummy rows/columns at `gate + 1` (i.e. always above the gate, so a
/// dummy pairing never survives the gate filter below) -- the standard
/// technique for a rectangular Hungarian problem, rather than requiring
/// the caller to pad. A **result pair is only returned when
/// `cost <= gate`** ("ambiguity policy -- drop, don't guess",
/// marker-detection-analysis.md): a genuinely unmatched real row or
/// column (every candidate cost for it exceeds the gate) is simply
/// absent from the result, never forced into a pairing just because the
/// solver internally produces a complete one over the padded matrix.
///
/// @param cost     Row-major `n_rows * n_cols` cost matrix.
/// @param n_rows   Number of real rows (candidates).
/// @param n_cols   Number of real columns (slots).
/// @param gate     Maximum acceptable cost for a pairing to be returned.
///                 Use `std::numeric_limits<double>::infinity()` to accept
///                 every pairing the underlying complete assignment finds
///                 (only meaningful for a genuinely square, ungated case).
/// @return Every accepted pairing (`cost <= gate`), unordered. A row or
///         column absent from every returned pair was left unmatched by
///         the gate, not physically excluded from the problem.
inline std::vector<AssignmentPair> solve_assignment(std::vector<double> const& cost, int n_rows,
                                                    int n_cols, double gate) {
    std::vector<AssignmentPair> result;
    if (n_rows <= 0 || n_cols <= 0) {
        return result;
    }

    int const n = std::max(n_rows, n_cols);
    // Padded, square, 1-indexed cost matrix (Jonker-Volgenant-free classic
    // Hungarian formulation below is most naturally written 1-indexed;
    // keeping that convention here rather than fighting it with off-by-one
    // translations throughout).
    constexpr double kInf = std::numeric_limits<double>::infinity();
    double const dummy_cost = std::isfinite(gate) ? gate + 1.0 : 1.0;
    std::vector<std::vector<double>> a(static_cast<size_t>(n) + 1,
                                       std::vector<double>(static_cast<size_t>(n) + 1, 0.0));
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            double v = dummy_cost;
            if (r < n_rows && c < n_cols) {
                v = cost[static_cast<size_t>(r) * static_cast<size_t>(n_cols) +
                         static_cast<size_t>(c)];
            }
            a[static_cast<size_t>(r) + 1][static_cast<size_t>(c) + 1] = v;
        }
    }

    // Classic O(n^3) Hungarian algorithm (Jonker-Volgenant potentials +
    // shortest augmenting path per row), 1-indexed as is conventional for
    // this exact formulation.
    std::vector<double> u(static_cast<size_t>(n) + 1, 0.0), v(static_cast<size_t>(n) + 1, 0.0);
    std::vector<int> p(static_cast<size_t>(n) + 1, 0), way(static_cast<size_t>(n) + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(static_cast<size_t>(n) + 1, kInf);
        std::vector<bool> used(static_cast<size_t>(n) + 1, false);
        do {
            used[static_cast<size_t>(j0)] = true;
            int i0 = p[static_cast<size_t>(j0)];
            double delta = kInf;
            int j1 = -1;
            for (int j = 1; j <= n; ++j) {
                if (used[static_cast<size_t>(j)])
                    continue;
                double cur = a[static_cast<size_t>(i0)][static_cast<size_t>(j)] -
                             u[static_cast<size_t>(i0)] - v[static_cast<size_t>(j)];
                if (cur < minv[static_cast<size_t>(j)]) {
                    minv[static_cast<size_t>(j)] = cur;
                    way[static_cast<size_t>(j)] = j0;
                }
                if (minv[static_cast<size_t>(j)] < delta) {
                    delta = minv[static_cast<size_t>(j)];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[static_cast<size_t>(j)]) {
                    u[static_cast<size_t>(p[static_cast<size_t>(j)])] += delta;
                    v[static_cast<size_t>(j)] -= delta;
                } else {
                    minv[static_cast<size_t>(j)] -= delta;
                }
            }
            j0 = j1;
        } while (p[static_cast<size_t>(j0)] != 0);
        do {
            int j1 = way[static_cast<size_t>(j0)];
            p[static_cast<size_t>(j0)] = p[static_cast<size_t>(j1)];
            j0 = j1;
        } while (j0 != 0);
    }

    // p[j] = row assigned to column j (1-indexed, over the padded matrix).
    // Undo the padding and the gate filter together: only a pairing where
    // both indices are real (< n_rows / < n_cols) AND its real cost passes
    // the gate is reported.
    for (int j = 1; j <= n; ++j) {
        int i = p[static_cast<size_t>(j)] - 1;
        int c = j - 1;
        if (i < n_rows && c < n_cols) {
            double real_cost =
                cost[static_cast<size_t>(i) * static_cast<size_t>(n_cols) + static_cast<size_t>(c)];
            if (real_cost <= gate) {
                result.push_back(AssignmentPair{i, c, real_cost});
            }
        }
    }
    return result;
}

}  // namespace posetrak

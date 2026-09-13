// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/tracking/dot_predict_profile.hpp"

#include <fmt/core.h>

#include <cstdlib>

namespace posetrak::dot_predict_profile {

namespace {
struct Accumulators {
    double sigma_gen_ms = 0.0;
    double predict_loop_ms = 0.0;
    double aggregate_ms = 0.0;
    long long n_calls = 0;
    long long n_sigma_evals = 0;
    long long n_marker_evals = 0;
};

Accumulators& accumulators() {
    static Accumulators acc;
    return acc;
}
}  // namespace

bool enabled() {
    static bool const value = std::getenv("POSETRAK_PROFILE_DOT_PREDICT") != nullptr;
    return value;
}

void add_sigma_gen_ms(double ms) {
    if (enabled())
        accumulators().sigma_gen_ms += ms;
}

void add_predict_loop_ms(double ms) {
    if (enabled())
        accumulators().predict_loop_ms += ms;
}

void add_aggregate_ms(double ms) {
    if (enabled())
        accumulators().aggregate_ms += ms;
}

void add_call(int n_sigma, int n_markers) {
    if (!enabled())
        return;
    auto& acc = accumulators();
    acc.n_calls += 1;
    acc.n_sigma_evals += n_sigma;
    acc.n_marker_evals += n_markers;
}

Snapshot snapshot() {
    auto const& acc = accumulators();
    return Snapshot{acc.sigma_gen_ms, acc.predict_loop_ms, acc.aggregate_ms,
                    acc.n_calls,      acc.n_sigma_evals,   acc.n_marker_evals};
}

void print_summary() {
    if (!enabled())
        return;
    auto const s = snapshot();
    if (s.n_calls == 0) {
        fmt::print(
            "[dot_predict_profile] enabled but predict_marker_slots() was never called "
            "this run\n");
        return;
    }
    double const total_ms = s.sigma_gen_ms + s.predict_loop_ms + s.aggregate_ms;
    fmt::print("\n=== DOT-SLOT PREDICTION PROFILE (predict_marker_slots) ===\n");
    fmt::print("calls: {}  (avg {:.1f} sigma points, {:.1f} markers per call)\n", s.n_calls,
               static_cast<double>(s.n_sigma_evals) / static_cast<double>(s.n_calls),
               static_cast<double>(s.n_marker_evals) / static_cast<double>(s.n_calls));
    fmt::print("  sigma-point generation:  {:9.1f} ms total  ({:6.3f} ms/call, {:5.1f}%)\n",
               s.sigma_gen_ms, s.sigma_gen_ms / static_cast<double>(s.n_calls),
               100.0 * s.sigma_gen_ms / total_ms);
    fmt::print(
        "  predict_measurements loop (FK + projection, {} sigma evals): {:9.1f} ms total "
        "({:6.3f} ms/call, {:5.1f}%)\n",
        s.n_sigma_evals, s.predict_loop_ms, s.predict_loop_ms / static_cast<double>(s.n_calls),
        100.0 * s.predict_loop_ms / total_ms);
    fmt::print("  mean/covariance aggregation: {:9.1f} ms total ({:6.3f} ms/call, {:5.1f}%)\n",
               s.aggregate_ms, s.aggregate_ms / static_cast<double>(s.n_calls),
               100.0 * s.aggregate_ms / total_ms);
    fmt::print("  TOTAL: {:.1f} ms across {} calls ({:.3f} ms/call average)\n", total_ms, s.n_calls,
               total_ms / static_cast<double>(s.n_calls));
    fmt::print("============================================================\n\n");
}

}  // namespace posetrak::dot_predict_profile

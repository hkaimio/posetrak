// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace posetrak::dot_predict_profile {

/// @file dot_predict_profile.hpp
/// Opt-in, lightweight wall-time accounting for
/// UnscentedKalmanFilter::predict_marker_slots() -- added to answer where
/// the time goes when dots are tracked: with a 16-marker leg module,
/// tracked-fps dropped from ~10 (markerless) to ~4-5 even with per-camera
/// batching. Splits
/// predict_marker_slots()'s own cost into sigma-point generation, the
/// per-sigma-point predict_measurements() loop (forward kinematics +
/// camera projection combined -- not split further here, since that
/// would mean instrumenting predict_measurements() itself, which is also
/// on the main per-frame update() hot path and would conflate the two
/// callers' costs), and the final per-marker mean/covariance aggregation.
///
/// Disabled by default (one bool check, effectively free) -- enable by
/// setting the POSETRAK_PROFILE_DOT_PREDICT environment variable before
/// running `posetrak-tracker track`. Not thread-safe: fine for this
/// project's current single-threaded-per-Tracker usage, not safe to
/// enable if that ever changes.

/// True if POSETRAK_PROFILE_DOT_PREDICT is set in the environment
/// (checked once, cached).
bool enabled();

void add_sigma_gen_ms(double ms);
void add_predict_loop_ms(double ms);
void add_aggregate_ms(double ms);
void add_call(int n_sigma, int n_markers);

struct Snapshot {
    double sigma_gen_ms = 0.0;
    double predict_loop_ms = 0.0;
    double aggregate_ms = 0.0;
    long long n_calls = 0;
    long long n_sigma_evals = 0;   ///< sum of n_sigma across calls
    long long n_marker_evals = 0;  ///< sum of n_markers across calls
};

Snapshot snapshot();

/// Prints a one-block human-readable summary (fmt::print to stdout) if
/// enabled() and at least one call was recorded; no-op otherwise.
void print_summary();

}  // namespace posetrak::dot_predict_profile

// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace posetrak::frame_step_profile {

/// @file frame_step_profile.hpp
/// Opt-in wall-time accounting for the per-frame pieces of a dot-augmented
/// tracking step that sit *outside* Tracker::predict_step()/update_step()
/// (already covered by TrackingResult's own predict_ms/update_ms) and
/// outside UnscentedKalmanFilter::predict_marker_slots[_all_cameras]()
/// (covered by dot_predict_profile) -- added because reconciling the two
/// against real observed frame time left ~64ms/frame (about a third of the
/// whole per-frame budget) unaccounted for by either.
///
/// Disabled by default (one bool check) -- enable by setting the
/// POSETRAK_PROFILE_FRAME_STEP environment variable. Not thread-safe: fine
/// for this project's current single-threaded-per-step orchestration.

bool enabled();

void add_bucket_candidates_ms(double ms);
/// Whole resolve_shared_dot_assignment() call time, *including* the nested
/// predict_dot_slot_predictions_all_cameras() cost dot_predict_profile
/// already measures -- callers wanting the assignment-only remainder should
/// subtract a dot_predict_profile::snapshot() delta taken around the same
/// call (see track.cpp's call site).
void add_dot_assignment_total_ms(double ms);
void add_get_observations_ms(double ms);
void add_export_predicted_obs_ms(double ms);
void add_export_state_vector_ms(double ms);
void add_fk_compute_posterior_ms(double ms);
void add_exporter_write_frame_ms(double ms);
void add_result_writer_write_ms(double ms);
void add_step();

struct Snapshot {
    double bucket_candidates_ms = 0.0;
    double dot_assignment_total_ms = 0.0;
    double get_observations_ms = 0.0;
    double export_predicted_obs_ms = 0.0;
    double export_state_vector_ms = 0.0;
    double fk_compute_posterior_ms = 0.0;
    double exporter_write_frame_ms = 0.0;
    double result_writer_write_ms = 0.0;
    long long n_steps = 0;
};

Snapshot snapshot();

/// Prints a one-block human-readable summary if enabled() and at least one
/// step was recorded; no-op otherwise. `dot_predict_profile_total_ms` is
/// the *raw accumulated total* (not per-frame) from
/// dot_predict_profile::snapshot(), used to report the assignment-only
/// remainder (whole resolve_shared_dot_assignment() call minus the nested
/// marker-slot-prediction cost it already includes) instead of the
/// double-counted whole-call figure.
void print_summary(double dot_predict_profile_total_ms);

}  // namespace posetrak::frame_step_profile

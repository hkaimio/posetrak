// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/tracking/frame_step_profile.hpp"

#include <fmt/core.h>

#include <cstdlib>

namespace posetrak::frame_step_profile {

namespace {
struct Accumulators {
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

Accumulators& accumulators() {
    static Accumulators acc;
    return acc;
}
}  // namespace

bool enabled() {
    static bool const value = std::getenv("POSETRAK_PROFILE_FRAME_STEP") != nullptr;
    return value;
}

void add_bucket_candidates_ms(double ms) {
    if (enabled())
        accumulators().bucket_candidates_ms += ms;
}
void add_dot_assignment_total_ms(double ms) {
    if (enabled())
        accumulators().dot_assignment_total_ms += ms;
}
void add_get_observations_ms(double ms) {
    if (enabled())
        accumulators().get_observations_ms += ms;
}
void add_export_predicted_obs_ms(double ms) {
    if (enabled())
        accumulators().export_predicted_obs_ms += ms;
}
void add_export_state_vector_ms(double ms) {
    if (enabled())
        accumulators().export_state_vector_ms += ms;
}
void add_fk_compute_posterior_ms(double ms) {
    if (enabled())
        accumulators().fk_compute_posterior_ms += ms;
}
void add_exporter_write_frame_ms(double ms) {
    if (enabled())
        accumulators().exporter_write_frame_ms += ms;
}
void add_result_writer_write_ms(double ms) {
    if (enabled())
        accumulators().result_writer_write_ms += ms;
}
void add_step() {
    if (enabled())
        accumulators().n_steps += 1;
}

Snapshot snapshot() {
    auto const& acc = accumulators();
    return Snapshot{
        acc.bucket_candidates_ms,    acc.dot_assignment_total_ms, acc.get_observations_ms,
        acc.export_predicted_obs_ms, acc.export_state_vector_ms,  acc.fk_compute_posterior_ms,
        acc.exporter_write_frame_ms, acc.result_writer_write_ms,  acc.n_steps};
}

void print_summary(double dot_predict_profile_total_ms) {
    if (!enabled())
        return;
    auto const s = snapshot();
    if (s.n_steps == 0) {
        fmt::print("[frame_step_profile] enabled but no steps were recorded this run\n");
        return;
    }
    double const n = static_cast<double>(s.n_steps);
    double const assignment_only_ms = s.dot_assignment_total_ms - dot_predict_profile_total_ms;
    double const total_ms = s.bucket_candidates_ms + assignment_only_ms + s.get_observations_ms +
                            s.export_predicted_obs_ms + s.export_state_vector_ms +
                            s.fk_compute_posterior_ms + s.exporter_write_frame_ms +
                            s.result_writer_write_ms;
    fmt::print("\n=== FRAME-STEP PROFILE (outside predict()/update()/predict_marker_slots) ===\n");
    fmt::print("steps: {}\n", s.n_steps);
    fmt::print("  bucket_candidates_by_camera():        {:9.1f} ms total ({:6.3f} ms/frame)\n",
               s.bucket_candidates_ms, s.bucket_candidates_ms / n);
    fmt::print(
        "  resolve_shared_dot_assignment()"
        " (assignment only, marker-slot prediction subtracted):"
        " {:9.1f} ms total ({:6.3f} ms/frame)\n",
        assignment_only_ms, assignment_only_ms / n);
    fmt::print("  observations.get_all_in_range():      {:9.1f} ms total ({:6.3f} ms/frame)\n",
               s.get_observations_ms, s.get_observations_ms / n);
    fmt::print("  export_predicted_observations():      {:9.1f} ms total ({:6.3f} ms/frame)\n",
               s.export_predicted_obs_ms, s.export_predicted_obs_ms / n);
    fmt::print("  export_state_vector():                {:9.1f} ms total ({:6.3f} ms/frame)\n",
               s.export_state_vector_ms, s.export_state_vector_ms / n);
    fmt::print("  fk->compute() on posterior state:     {:9.1f} ms total ({:6.3f} ms/frame)\n",
               s.fk_compute_posterior_ms, s.fk_compute_posterior_ms / n);
    fmt::print("  exporter->write_frame():              {:9.1f} ms total ({:6.3f} ms/frame)\n",
               s.exporter_write_frame_ms, s.exporter_write_frame_ms / n);
    fmt::print("  result_writer->write_frame/obs():     {:9.1f} ms total ({:6.3f} ms/frame)\n",
               s.result_writer_write_ms, s.result_writer_write_ms / n);
    fmt::print("  TOTAL (these pieces only): {:.1f} ms across {} steps ({:.3f} ms/frame)\n",
               total_ms, s.n_steps, total_ms / n);
    fmt::print("================================================================\n\n");
}

}  // namespace posetrak::frame_step_profile

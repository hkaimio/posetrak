// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file streak_k_accumulator.hpp
 * @brief Running per-camera k = exposure_time/frame_time estimate for the
 * streak-derived dot velocity mechanism -- see
 * docs/roadmap/features/marker-based-mocap/streak-velocity-design.md §3/§4.
 *
 * A motion-blur streak's length is a marker's image-plane displacement
 * during the camera's exposure window, not the full inter-frame interval;
 * k converts one into the other. k is provably independent of the object's
 * own speed (the velocity term cancels out of streak_length/frame_displacement),
 * so a real per-camera constant estimated from any resolved motion is
 * exactly what a stable estimate should look like -- see the design doc's
 * real validation numbers (2026-09-05).
 *
 * Deliberately plain data with no Tracker/skeleton/camera access of its own
 * (same "pure core, directly testable" split this codebase already uses for
 * dot_assignment.hpp's resolve_dot_assignment()) -- Tracker owns one
 * instance per camera and resolve_dot_assignment() reads/updates it via
 * plain references, so this class never needs to know where its samples
 * came from.
 */
#pragma once

#include <deque>
#include <optional>
#include <utility>

namespace posetrak {

class StreakKAccumulator {
   public:
    /// @brief Add one (streak_length_px, frame_displacement_px) sample, evicting the
    /// oldest sample once the window would exceed *window*.
    ///
    /// Accumulates as a running sum rather than re-summing the deque every call --
    /// k() is a ratio of sums (streak-velocity-design.md §3: not a mean of per-sample
    /// ratios, which is unstable as an individual sample's displacement approaches
    /// zero), so the sums, not the individual samples, are what k() actually needs;
    /// the deque exists only to know what to subtract off on eviction.
    void add(double streak_px, double disp_px, int window);

    /// @brief Current k estimate, or std::nullopt if fewer than *min_samples* samples
    /// are in the window yet (not trustworthy) or the accumulated displacement is
    /// zero (would divide by zero).
    std::optional<double> k(int min_samples) const;

    /// @brief Samples currently in the window (for diagnostics/tests).
    int sample_count() const { return static_cast<int>(samples_.size()); }

   private:
    std::deque<std::pair<double, double>> samples_;  ///< (streak_px, disp_px), oldest first
    double sum_streak_px_ = 0.0;
    double sum_disp_px_ = 0.0;
};

}  // namespace posetrak

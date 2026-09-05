// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/tracking/streak_k_accumulator.hpp"

namespace posetrak {

void StreakKAccumulator::add(double streak_px, double disp_px, int window) {
    samples_.emplace_back(streak_px, disp_px);
    sum_streak_px_ += streak_px;
    sum_disp_px_ += disp_px;
    while (static_cast<int>(samples_.size()) > window && window >= 0) {
        sum_streak_px_ -= samples_.front().first;
        sum_disp_px_ -= samples_.front().second;
        samples_.pop_front();
    }
}

std::optional<double> StreakKAccumulator::k(int min_samples) const {
    if (static_cast<int>(samples_.size()) < min_samples || sum_disp_px_ <= 0.0)
        return std::nullopt;
    return sum_streak_px_ / sum_disp_px_;
}

}  // namespace posetrak

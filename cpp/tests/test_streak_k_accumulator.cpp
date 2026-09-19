// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "posetrak/tracking/streak_k_accumulator.hpp"

using posetrak::StreakKAccumulator;

TEST_CASE("StreakKAccumulator: k() is nullopt with no samples", "[streak_k]") {
    StreakKAccumulator acc;
    REQUIRE_FALSE(acc.k(1).has_value());
}

TEST_CASE("StreakKAccumulator: k() is nullopt below min_samples", "[streak_k]") {
    StreakKAccumulator acc;
    acc.add(3.0, 10.0, 200);
    acc.add(3.0, 10.0, 200);
    REQUIRE_FALSE(acc.k(3).has_value());
    REQUIRE(acc.k(2).has_value());
}

TEST_CASE("StreakKAccumulator: k() is a ratio of sums, not a mean of per-sample ratios",
          "[streak_k]") {
    // The whole point of summing before dividing (streak-velocity-design.md §3): a
    // single sample with a tiny, noise-dominated displacement must not be able to
    // dominate the estimate the way it would under a mean of per-sample ratios.
    StreakKAccumulator acc;
    acc.add(4.0, 20.0, 200);  // per-sample ratio 0.20
    acc.add(0.5, 0.1, 200);   // per-sample ratio 5.0 -- noisy outlier

    double const mean_of_ratios = (4.0 / 20.0 + 0.5 / 0.1) / 2.0;
    auto k = acc.k(2);
    REQUIRE(k.has_value());
    REQUIRE(*k == Catch::Approx((4.0 + 0.5) / (20.0 + 0.1)));
    REQUIRE(*k < mean_of_ratios / 2.0);
}

TEST_CASE("StreakKAccumulator: recovers a constant true k regardless of sample speed",
          "[streak_k]") {
    // k = exposure_time/frame_time should come out the same whether the underlying
    // motion was slow or fast (design doc §1: the velocity term cancels out of the
    // ratio) -- simulated here as several samples sharing one true k at different
    // speeds.
    double const true_k = 0.3;
    StreakKAccumulator acc;
    for (double disp : {5.0, 12.0, 40.0, 3.0}) {
        acc.add(disp * true_k, disp, 200);
    }
    auto k = acc.k(1);
    REQUIRE(k.has_value());
    REQUIRE(*k == Catch::Approx(true_k));
}

TEST_CASE("StreakKAccumulator: evicts the oldest sample once the window is exceeded",
          "[streak_k]") {
    StreakKAccumulator acc;
    acc.add(100.0, 10.0, 2);  // ratio 10 -- will be evicted
    acc.add(1.0, 10.0, 2);
    acc.add(1.0, 10.0, 2);
    REQUIRE(acc.sample_count() == 2);
    auto k = acc.k(2);
    REQUIRE(k.has_value());
    REQUIRE(*k == Catch::Approx((1.0 + 1.0) / (10.0 + 10.0)));
}

TEST_CASE("StreakKAccumulator: k() is nullopt when accumulated displacement is zero",
          "[streak_k]") {
    StreakKAccumulator acc;
    acc.add(5.0, 0.0, 200);
    REQUIRE_FALSE(acc.k(1).has_value());
}

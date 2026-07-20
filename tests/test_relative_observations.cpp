#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/tracking/relative_observations.hpp"

using namespace posetrak;
using Catch::Matchers::WithinAbs;

namespace {

Observation make_obs(int marker_id, int camera_id, Eigen::Vector2d const& pos,
                     double confidence = 0.9, double crop_scale = 1.0) {
    Observation obs;
    obs.marker_id = marker_id;
    obs.camera_id = camera_id;
    obs.frame_idx = 7;
    obs.timestamp = 1.23;
    obs.position = pos;
    obs.position_distorted = pos;
    obs.confidence = confidence;
    obs.crop_scale = crop_scale;
    return obs;
}

constexpr int kWrist = 0;
constexpr int kIndex = 1;
constexpr int kPinky = 2;

}  // namespace

TEST_CASE("build_ref_marker_pair_observations: basic pair construction",
          "[relative_observations]") {
    std::vector<Observation> frame_obs = {
        make_obs(kWrist, /*camera=*/0, {100.0, 200.0}),
        make_obs(kIndex, /*camera=*/0, {110.0, 215.0}),
    };

    auto result = build_ref_marker_pair_observations(frame_obs, kWrist, /*pose_noise_std=*/3.0);

    REQUIRE(result.size() == 1);
    Observation const& rel = result[0];
    CHECK(rel.marker_id == kIndex);
    CHECK(rel.ref_marker_id == kWrist);
    CHECK(rel.camera_id == 0);
    CHECK(rel.mode == MeasurementMode::PAIR_DIFF);
    CHECK_THAT(rel.position.x(), WithinAbs(10.0, 1e-9));
    CHECK_THAT(rel.position.y(), WithinAbs(15.0, 1e-9));
    CHECK_THAT(rel.noise_std_override, WithinAbs(3.0 * std::sqrt(2.0), 1e-9));
}

TEST_CASE("build_ref_marker_pair_observations: multiple markers and cameras",
          "[relative_observations]") {
    std::vector<Observation> frame_obs = {
        make_obs(kWrist, 0, {0.0, 0.0}),   make_obs(kWrist, 1, {5.0, 5.0}),
        make_obs(kIndex, 0, {3.0, 4.0}),   make_obs(kIndex, 1, {8.0, 2.0}),
        make_obs(kPinky, 0, {-1.0, -2.0}),
        // pinky not detected in camera 1 this frame
    };

    auto result = build_ref_marker_pair_observations(frame_obs, kWrist, 3.0);

    // index: cam0 + cam1, pinky: cam0 only -> 3 pairs total
    REQUIRE(result.size() == 3);
    for (Observation const& rel : result) {
        CHECK(rel.ref_marker_id == kWrist);
        CHECK(rel.marker_id != kWrist);
    }
}

TEST_CASE("build_ref_marker_pair_observations: no reference marker in camera -> no pair",
          "[relative_observations]") {
    std::vector<Observation> frame_obs = {
        make_obs(kIndex, 0, {3.0, 4.0}),  // wrist never detected in camera 0
    };

    auto result = build_ref_marker_pair_observations(frame_obs, kWrist, 3.0);
    CHECK(result.empty());
}

TEST_CASE("build_ref_marker_pair_observations: min_confidence excludes low-confidence detections",
          "[relative_observations]") {
    std::vector<Observation> frame_obs = {
        make_obs(kWrist, 0, {0.0, 0.0}, /*confidence=*/0.9),
        make_obs(kIndex, 0, {3.0, 4.0}, /*confidence=*/0.2),
    };

    auto result = build_ref_marker_pair_observations(frame_obs, kWrist, 3.0,
                                                     /*min_confidence=*/0.5);
    CHECK(result.empty());

    auto result_low_threshold =
        build_ref_marker_pair_observations(frame_obs, kWrist, 3.0, /*min_confidence=*/0.1);
    REQUIRE(result_low_threshold.size() == 1);
    // confidence = min(marker, reference)
    CHECK_THAT(result_low_threshold[0].confidence, WithinAbs(0.2, 1e-9));
}

TEST_CASE("build_ref_marker_pair_observations: reference marker never emitted as its own pair",
          "[relative_observations]") {
    std::vector<Observation> frame_obs = {
        make_obs(kWrist, 0, {0.0, 0.0}),
    };

    auto result = build_ref_marker_pair_observations(frame_obs, kWrist, 3.0);
    CHECK(result.empty());
}

TEST_CASE("build_ref_marker_pair_observations: crop_scale flows through from the paired marker",
          "[relative_observations]") {
    std::vector<Observation> frame_obs = {
        make_obs(kWrist, 0, {0.0, 0.0}, 0.9, /*crop_scale=*/1.0),
        make_obs(kIndex, 0, {3.0, 4.0}, 0.9, /*crop_scale=*/2.5),
    };

    auto result = build_ref_marker_pair_observations(frame_obs, kWrist, /*pose_noise_std=*/2.0);
    REQUIRE(result.size() == 1);
    CHECK_THAT(result[0].noise_std_override, WithinAbs(2.0 * std::sqrt(2.0) * 2.5, 1e-9));
}

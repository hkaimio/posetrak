#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/tracking/relative_observations.hpp"
#include <stdexcept>
#include <string>

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

// ---------------------------------------------------------------------------
// reconstruct_pair_diff_absolute
// ---------------------------------------------------------------------------

namespace {

ObservationResult make_result(std::string const& marker_name, int camera_id,
                              Eigen::Vector2d const& actual, Eigen::Vector2d const& predicted,
                              double mahal = 1.5, bool is_outlier = false) {
    ObservationResult r;
    r.marker_name = marker_name;
    r.camera_id = camera_id;
    r.camera_frame_idx = 0;
    r.is_outlier = is_outlier;
    r.mahalanobis_distance = mahal;
    r.innovation = actual - predicted;
    r.predicted = predicted;
    r.actual = actual;
    return r;
}

}  // namespace

TEST_CASE("reconstruct_pair_diff_absolute: shifts non-reference entries back to absolute pixels",
          "[relative_observations][reconstruct_pair_diff_absolute]") {
    // Reference marker's own (absolute) result for camera 0.
    auto ref = make_result("MRK-wrist", 0, {100.0, 200.0}, {101.0, 199.0});
    // A PAIR_DIFF-derived entry: actual/predicted are DIFFERENCES (index - wrist).
    auto diff = make_result("MRK-index_1", 0, {10.0, 15.0}, {9.0, 14.0}, /*mahal=*/2.2,
                            /*is_outlier=*/true);

    auto [out, reconstructed] = reconstruct_pair_diff_absolute({ref, diff}, "MRK-wrist");

    REQUIRE(out.size() == 2);
    REQUIRE(reconstructed.size() == 2);

    // Reference entry passes through unchanged, flagged as not reconstructed.
    CHECK(reconstructed[0] == 0);
    CHECK_THAT(out[0].actual.x(), WithinAbs(100.0, 1e-9));
    CHECK_THAT(out[0].actual.y(), WithinAbs(200.0, 1e-9));
    CHECK_THAT(out[0].predicted.x(), WithinAbs(101.0, 1e-9));
    CHECK_THAT(out[0].predicted.y(), WithinAbs(199.0, 1e-9));

    // The diff entry is shifted back to absolute pixels: actual = diff + ref.actual.
    CHECK(reconstructed[1] == 1);
    CHECK_THAT(out[1].actual.x(), WithinAbs(110.0, 1e-9));     // 10 + 100
    CHECK_THAT(out[1].actual.y(), WithinAbs(215.0, 1e-9));     // 15 + 200
    CHECK_THAT(out[1].predicted.x(), WithinAbs(110.0, 1e-9));  // 9 + 101
    CHECK_THAT(out[1].predicted.y(), WithinAbs(213.0, 1e-9));  // 14 + 199

    // mahalanobis_distance/is_outlier/marker_name/camera_id carried through untouched.
    CHECK_THAT(out[1].mahalanobis_distance, WithinAbs(2.2, 1e-9));
    CHECK(out[1].is_outlier);
    CHECK(out[1].marker_name == "MRK-index_1");
    CHECK(out[1].camera_id == 0);
}

TEST_CASE("reconstruct_pair_diff_absolute: multiple cameras use their own reference entry",
          "[relative_observations][reconstruct_pair_diff_absolute]") {
    auto ref0 = make_result("MRK-wrist", 0, {100.0, 200.0}, {100.0, 200.0});
    auto ref1 = make_result("MRK-wrist", 1, {50.0, 60.0}, {50.0, 60.0});
    auto diff0 = make_result("MRK-pinky_1", 0, {1.0, 1.0}, {1.0, 1.0});
    auto diff1 = make_result("MRK-pinky_1", 1, {2.0, 2.0}, {2.0, 2.0});

    auto [out, reconstructed] =
        reconstruct_pair_diff_absolute({ref0, ref1, diff0, diff1}, "MRK-wrist");

    REQUIRE(out.size() == 4);
    CHECK_THAT(out[2].actual.x(), WithinAbs(101.0, 1e-9));  // cam0: 1 + 100
    CHECK_THAT(out[3].actual.x(), WithinAbs(52.0, 1e-9));   // cam1: 2 + 50
}

TEST_CASE("reconstruct_pair_diff_absolute: throws when the reference has no entry for a camera",
          "[relative_observations][reconstruct_pair_diff_absolute]") {
    auto diff = make_result("MRK-index_1", 3, {1.0, 1.0}, {1.0, 1.0});
    CHECK_THROWS_AS(reconstruct_pair_diff_absolute({diff}, "MRK-wrist"), std::runtime_error);
}

TEST_CASE("reconstruct_pair_diff_absolute: empty input returns empty output",
          "[relative_observations][reconstruct_pair_diff_absolute]") {
    auto [out, reconstructed] = reconstruct_pair_diff_absolute({}, "MRK-wrist");
    CHECK(out.empty());
    CHECK(reconstructed.empty());
}

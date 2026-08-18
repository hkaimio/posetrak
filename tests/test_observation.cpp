// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "posetrak/core/observation.hpp"

using namespace posetrak;

TEST_CASE("Observation construction and accessors", "[observation]") {
    Observation obs;
    obs.camera_id = 0;
    obs.marker_id = 5;
    obs.frame_idx = 42;
    obs.timestamp = 1.4;
    obs.position = Eigen::Vector2d(320.5, 240.3);
    obs.position_distorted = Eigen::Vector2d(322.1, 241.5);
    obs.confidence = 0.85;

    REQUIRE(obs.camera_id == 0);
    REQUIRE(obs.marker_id == 5);
    REQUIRE(obs.frame_idx == 42);
    REQUIRE_THAT(obs.timestamp, Catch::Matchers::WithinAbs(1.4, 1e-10));
    REQUIRE(obs.position.isApprox(Eigen::Vector2d(320.5, 240.3)));
    REQUIRE(obs.confidence == 0.85);
}

TEST_CASE("Observation measurement noise", "[observation]") {
    Observation obs;

    SECTION("High confidence reduces noise") {
        obs.confidence = 0.9;
        double noise = obs.measurement_noise_std(5.0);
        REQUIRE_THAT(noise, Catch::Matchers::WithinAbs(5.0 / 0.9, 1e-6));
    }

    SECTION("Low confidence increases noise") {
        obs.confidence = 0.2;
        double noise = obs.measurement_noise_std(5.0);
        REQUIRE_THAT(noise, Catch::Matchers::WithinAbs(5.0 / 0.2, 1e-6));
    }

    SECTION("Very low confidence clamped") {
        obs.confidence = 0.05;
        double noise = obs.measurement_noise_std(5.0);
        REQUIRE_THAT(noise, Catch::Matchers::WithinAbs(5.0 / 0.1, 1e-6));
    }
}

TEST_CASE("Observation JSON serialization", "[observation]") {
    Observation obs;
    obs.camera_id = 1;
    obs.marker_id = 10;
    obs.frame_idx = 100;
    obs.timestamp = 3.333;
    obs.position = Eigen::Vector2d(400.0, 300.0);
    obs.position_distorted = Eigen::Vector2d(402.0, 301.0);
    obs.confidence = 0.75;

    nlohmann::json const json = obs.to_json();
    Observation const obs2 = Observation::from_json(json);

    REQUIRE(obs2.camera_id == obs.camera_id);
    REQUIRE(obs2.marker_id == obs.marker_id);
    REQUIRE(obs2.frame_idx == obs.frame_idx);
    REQUIRE_THAT(obs2.timestamp, Catch::Matchers::WithinAbs(obs.timestamp, 1e-10));
    REQUIRE(obs2.position.isApprox(obs.position));
    REQUIRE(obs2.position_distorted.isApprox(obs.position_distorted));
    REQUIRE(obs2.confidence == obs.confidence);
}

TEST_CASE("ObservationSequence basic operations", "[observation]") {
    ObservationSequence seq;
    seq.camera_id = 0;
    seq.camera_name = "camera_1";

    REQUIRE(seq.empty());
    REQUIRE(seq.size() == 0);

    // Add observations
    for (int i = 0; i < 5; ++i) {
        Observation obs;
        obs.camera_id = 0;
        obs.marker_id = i;
        obs.frame_idx = i * 10;
        obs.timestamp = i * 0.1;
        obs.position = Eigen::Vector2d(100.0 + i, 200.0 + i);
        obs.position_distorted = obs.position;
        obs.confidence = 0.8;
        seq.observations.push_back(obs);
    }

    REQUIRE(!seq.empty());
    REQUIRE(seq.size() == 5);
}

TEST_CASE("ObservationSequence time queries", "[observation]") {
    ObservationSequence seq;
    seq.camera_id = 0;
    seq.camera_name = "camera_1";

    // Add observations at t = 0.0, 0.1, 0.2, 0.3, 0.4
    for (int i = 0; i < 5; ++i) {
        Observation obs;
        obs.camera_id = 0;
        obs.marker_id = i;
        obs.timestamp = i * 0.1;
        obs.position = Eigen::Vector2d(i, i);
        obs.position_distorted = obs.position;
        obs.confidence = 0.8;
        seq.observations.push_back(obs);
    }

    // NOTE: get_at_time() removed - use get_in_range() instead
    // SECTION("Get at exact time") {
    //     auto obs = seq.get_at_time(0.2);
    //     REQUIRE(obs.size() == 1);
    //     REQUIRE(obs[0].marker_id == 2);
    // }

    SECTION("Get in range") {
        auto obs = seq.get_in_range(0.1, 0.3);
        REQUIRE(obs.size() == 2);
        REQUIRE(obs[0].marker_id == 1);
        REQUIRE(obs[1].marker_id == 2);
    }

    SECTION("Min and max time") {
        REQUIRE_THAT(seq.min_time(), Catch::Matchers::WithinAbs(0.0, 1e-10));
        REQUIRE_THAT(seq.max_time(), Catch::Matchers::WithinAbs(0.4, 1e-10));
    }
}

TEST_CASE("ObservationSequence JSON serialization", "[observation]") {
    ObservationSequence seq;
    seq.camera_id = 1;
    seq.camera_name = "camera_2";

    for (int i = 0; i < 3; ++i) {
        Observation obs;
        obs.camera_id = 1;
        obs.marker_id = i;
        obs.timestamp = i * 0.1;
        obs.position = Eigen::Vector2d(i, i);
        obs.position_distorted = obs.position;
        obs.confidence = 0.8;
        seq.observations.push_back(obs);
    }

    nlohmann::json const json = seq.to_json();
    ObservationSequence const seq2 = ObservationSequence::from_json(json);

    REQUIRE(seq2.camera_id == seq.camera_id);
    REQUIRE(seq2.camera_name == seq.camera_name);
    REQUIRE(seq2.size() == seq.size());
    REQUIRE(seq2.observations[0].marker_id == 0);
    REQUIRE(seq2.observations[2].marker_id == 2);
}

TEST_CASE("ObservationSet construction and cameras", "[observation]") {
    ObservationSet obs_set(123);

    REQUIRE(obs_set.person_id() == 123);
    REQUIRE(obs_set.empty());
    REQUIRE(obs_set.camera_count() == 0);
    REQUIRE(obs_set.total_observations() == 0);

    // Add sequences
    ObservationSequence seq1;
    seq1.camera_id = 0;
    seq1.camera_name = "camera_1";

    ObservationSequence seq2;
    seq2.camera_id = 1;
    seq2.camera_name = "camera_2";

    obs_set.add_sequence(seq1);
    obs_set.add_sequence(seq2);

    REQUIRE(!obs_set.empty());
    REQUIRE(obs_set.camera_count() == 2);

    auto names = obs_set.camera_names();
    REQUIRE(names.size() == 2);
}

TEST_CASE("ObservationSet multi-camera queries", "[observation]") {
    ObservationSet obs_set(0);

    // Camera 1: observations at t = 0.0, 0.1, 0.2
    ObservationSequence seq1;
    seq1.camera_id = 0;
    seq1.camera_name = "camera_1";
    for (int i = 0; i < 3; ++i) {
        Observation obs;
        obs.camera_id = 0;
        obs.marker_id = i;
        obs.timestamp = i * 0.1;
        obs.frame_idx = i;
        obs.position = Eigen::Vector2d(i, i);
        obs.position_distorted = obs.position;
        obs.confidence = 0.8;
        seq1.observations.push_back(obs);
    }

    // Camera 2: observations at t = 0.0, 0.1, 0.2 (different markers)
    ObservationSequence seq2;
    seq2.camera_id = 1;
    seq2.camera_name = "camera_2";
    for (int i = 0; i < 3; ++i) {
        Observation obs;
        obs.camera_id = 1;
        obs.marker_id = i + 10;
        obs.timestamp = i * 0.1;
        obs.frame_idx = i;
        obs.position = Eigen::Vector2d(i + 100, i + 100);
        obs.position_distorted = obs.position;
        obs.confidence = 0.9;
        seq2.observations.push_back(obs);
    }

    obs_set.add_sequence(seq1);
    obs_set.add_sequence(seq2);

    // NOTE: get_all_at_time() removed - use get_all_in_range() instead
    // SECTION("Get all at time") {
    //     auto obs = obs_set.get_all_at_time(0.1);
    //     REQUIRE(obs.size() == 2);  // One from each camera
    //     REQUIRE(obs[0].camera_id != obs[1].camera_id);
    // }

    SECTION("Get all in range") {
        auto obs = obs_set.get_all_in_range(0.05, 0.15);
        REQUIRE(obs.size() == 2);
    }

    // NOTE: get_all_at_frames() removed - frame-based queries are obsolete
    // SECTION("Get all at frames") {
    //     std::map<std::string, int> frames;
    //     frames["camera_1"] = 1;
    //     frames["camera_2"] = 2;
    //     auto obs = obs_set.get_all_at_frames(frames);
    //     REQUIRE(obs.size() == 2);
    //     REQUIRE(obs[0].frame_idx == 1);
    //     REQUIRE(obs[1].frame_idx == 2);
    // }

    SECTION("Min and max time") {
        REQUIRE_THAT(obs_set.min_time(), Catch::Matchers::WithinAbs(0.0, 1e-10));
        REQUIRE_THAT(obs_set.max_time(), Catch::Matchers::WithinAbs(0.2, 1e-10));
    }

    // NOTE: get_unique_timestamps() removed - not needed for time-range based tracking
    // SECTION("Unique timestamps") {
    //     auto times = obs_set.get_unique_timestamps();
    //     REQUIRE(times.size() == 3);
    //     REQUIRE_THAT(times[0], Catch::Matchers::WithinAbs(0.0, 1e-10));
    //     REQUIRE_THAT(times[1], Catch::Matchers::WithinAbs(0.1, 1e-10));
    //     REQUIRE_THAT(times[2], Catch::Matchers::WithinAbs(0.2, 1e-10));
    // }

    SECTION("Total observations") {
        REQUIRE(obs_set.total_observations() == 6);
    }
}

TEST_CASE("ObservationSet sequence lookup", "[observation]") {
    ObservationSet obs_set(0);

    ObservationSequence seq;
    seq.camera_id = 0;
    seq.camera_name = "camera_1";
    obs_set.add_sequence(seq);

    SECTION("Get existing sequence") {
        auto const* s = obs_set.get_sequence("camera_1");
        REQUIRE(s != nullptr);
        REQUIRE(s->camera_name == "camera_1");
    }

    SECTION("Get non-existing sequence") {
        auto const* s = obs_set.get_sequence("camera_99");
        REQUIRE(s == nullptr);
    }
}

TEST_CASE("ObservationSet JSON serialization", "[observation]") {
    ObservationSet obs_set(42);

    ObservationSequence seq1;
    seq1.camera_id = 0;
    seq1.camera_name = "camera_1";
    Observation obs1;
    obs1.camera_id = 0;
    obs1.marker_id = 5;
    obs1.timestamp = 1.0;
    obs1.position = Eigen::Vector2d(100, 200);
    obs1.position_distorted = obs1.position;
    obs1.confidence = 0.7;
    seq1.observations.push_back(obs1);

    ObservationSequence seq2;
    seq2.camera_id = 1;
    seq2.camera_name = "camera_2";
    Observation obs2;
    obs2.camera_id = 1;
    obs2.marker_id = 6;
    obs2.timestamp = 1.0;
    obs2.position = Eigen::Vector2d(300, 400);
    obs2.position_distorted = obs2.position;
    obs2.confidence = 0.8;
    seq2.observations.push_back(obs2);

    obs_set.add_sequence(seq1);
    obs_set.add_sequence(seq2);

    nlohmann::json const json = obs_set.to_json();
    ObservationSet const obs_set2 = ObservationSet::from_json(json);

    REQUIRE(obs_set2.person_id() == 42);
    REQUIRE(obs_set2.camera_count() == 2);
    REQUIRE(obs_set2.total_observations() == 2);

    auto const* s1 = obs_set2.get_sequence("camera_1");
    REQUIRE(s1 != nullptr);
    REQUIRE(s1->observations.size() == 1);
    REQUIRE(s1->observations[0].marker_id == 5);
}

TEST_CASE("ObservationSequence edge cases", "[observation]") {
    ObservationSequence seq;
    seq.camera_id = 0;
    seq.camera_name = "test_camera";

    SECTION("Empty sequence min/max time") {
        REQUIRE(std::isinf(seq.min_time()));
        REQUIRE(std::isinf(seq.max_time()));
        REQUIRE(seq.min_time() > 0);  // +inf
        REQUIRE(seq.max_time() < 0);  // -inf
    }

    SECTION("Empty sequence queries") {
        // NOTE: get_at_time() removed - use get_in_range() instead
        // auto obs1 = seq.get_at_time(0.0);
        // REQUIRE(obs1.empty());

        auto obs2 = seq.get_in_range(0.0, 1.0);
        REQUIRE(obs2.empty());
    }
}

TEST_CASE("ObservationSet edge cases", "[observation]") {
    ObservationSet obs_set(0);

    SECTION("Empty set queries") {
        // NOTE: get_all_at_time() removed
        // auto obs1 = obs_set.get_all_at_time(0.0);
        // REQUIRE(obs1.empty());

        auto obs2 = obs_set.get_all_in_range(0.0, 1.0);
        REQUIRE(obs2.empty());

        // NOTE: get_unique_timestamps() removed
        // auto times = obs_set.get_unique_timestamps();
        // REQUIRE(times.empty());
    }

    SECTION("Empty set min/max time") {
        REQUIRE(std::isinf(obs_set.min_time()));
        REQUIRE(std::isinf(obs_set.max_time()));
    }
}

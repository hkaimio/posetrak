// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/core/observation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace posetrak {

// Observation implementation

nlohmann::json Observation::to_json() const {
    nlohmann::json j;
    j["camera_id"] = camera_id;
    j["marker_id"] = marker_id;
    j["frame_idx"] = frame_idx;
    j["timestamp"] = timestamp;
    j["position"] = {position.x(), position.y()};
    j["position_distorted"] = {position_distorted.x(), position_distorted.y()};
    j["confidence"] = confidence;
    return j;
}

Observation Observation::from_json(nlohmann::json const& j) {
    Observation obs;
    obs.camera_id = j.at("camera_id").get<int>();
    obs.marker_id = j.at("marker_id").get<int>();
    obs.frame_idx = j.at("frame_idx").get<int>();
    obs.timestamp = j.at("timestamp").get<double>();

    auto const& pos = j.at("position");
    obs.position = Eigen::Vector2d(pos[0].get<double>(), pos[1].get<double>());

    auto const& pos_dist = j.at("position_distorted");
    obs.position_distorted = Eigen::Vector2d(pos_dist[0].get<double>(), pos_dist[1].get<double>());

    obs.confidence = j.at("confidence").get<double>();

    return obs;
}

// ObservationSequence implementation

std::vector<Observation> ObservationSequence::get_in_range(double t_start, double t_end) const {
    // Binary search, not a linear scan: a linear scan cost ~34 ms/frame
    // (18.5% of the whole per-frame budget, per frame_step_profile) just to
    // fetch one frame's real observations.
    // Relies on `observations` being sorted by timestamp (non-decreasing --
    // several observations from the same frame legitimately share one
    // timestamp), which every populating path already guarantees: the DB
    // loader's own query is `ORDER BY camera_instance_id, video_frame`
    // (session_reader.cpp), and the legacy JSON loader explicitly sorts by
    // frame number before appending (observation_loader.cpp). Nothing
    // mutates an already-built sequence's `observations` afterward.
    auto const lo =
        std::lower_bound(observations.begin(), observations.end(), t_start,
                         [](Observation const& obs, double t) { return obs.timestamp < t; });
    auto const hi =
        std::lower_bound(observations.begin(), observations.end(), t_end,
                         [](Observation const& obs, double t) { return obs.timestamp < t; });
    return std::vector<Observation>(lo, hi);
}

double ObservationSequence::min_time() const {
    if (observations.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    double min_t = observations[0].timestamp;
    for (auto const& obs : observations) {
        min_t = std::min(min_t, obs.timestamp);
    }
    return min_t;
}

double ObservationSequence::max_time() const {
    if (observations.empty()) {
        return -std::numeric_limits<double>::infinity();
    }

    double max_t = observations[0].timestamp;
    for (auto const& obs : observations) {
        max_t = std::max(max_t, obs.timestamp);
    }
    return max_t;
}

nlohmann::json ObservationSequence::to_json() const {
    nlohmann::json j;
    j["camera_id"] = camera_id;
    j["camera_name"] = camera_name;

    nlohmann::json obs_array = nlohmann::json::array();
    for (auto const& obs : observations) {
        obs_array.push_back(obs.to_json());
    }
    j["observations"] = obs_array;

    return j;
}

ObservationSequence ObservationSequence::from_json(nlohmann::json const& j) {
    ObservationSequence seq;
    seq.camera_id = j.at("camera_id").get<int>();
    seq.camera_name = j.at("camera_name").get<std::string>();

    for (auto const& obs_json : j.at("observations")) {
        seq.observations.push_back(Observation::from_json(obs_json));
    }

    return seq;
}

// ObservationSet implementation

ObservationSet::ObservationSet(int person_id) : person_id_(person_id) {}

void ObservationSet::add_sequence(ObservationSequence const& sequence) {
    sequences_[sequence.camera_name] = sequence;
}

ObservationSequence const* ObservationSet::get_sequence(std::string const& camera_name) const {
    auto it = sequences_.find(camera_name);
    return (it != sequences_.end()) ? &it->second : nullptr;
}

std::vector<Observation> ObservationSet::get_all_in_range(double t_start, double t_end) const {
    std::vector<Observation> result;

    for (auto const& [_, seq] : sequences_) {
        auto obs = seq.get_in_range(t_start, t_end);
        result.insert(result.end(), obs.begin(), obs.end());
    }

    return result;
}

double ObservationSet::min_time() const {
    double min_t = std::numeric_limits<double>::infinity();

    for (auto const& [_, seq] : sequences_) {
        min_t = std::min(min_t, seq.min_time());
    }

    return min_t;
}

double ObservationSet::max_time() const {
    double max_t = -std::numeric_limits<double>::infinity();

    for (auto const& [_, seq] : sequences_) {
        max_t = std::max(max_t, seq.max_time());
    }

    return max_t;
}

size_t ObservationSet::total_observations() const {
    size_t total = 0;
    for (auto const& [_, seq] : sequences_) {
        total += seq.size();
    }
    return total;
}

std::vector<std::string> ObservationSet::camera_names() const {
    std::vector<std::string> names;
    names.reserve(sequences_.size());

    for (auto const& [name, _] : sequences_) {
        names.push_back(name);
    }

    return names;
}

nlohmann::json ObservationSet::to_json() const {
    nlohmann::json j;
    j["person_id"] = person_id_;

    nlohmann::json sequences_obj = nlohmann::json::object();
    for (auto const& [name, seq] : sequences_) {
        sequences_obj[name] = seq.to_json();
    }
    j["sequences"] = sequences_obj;

    return j;
}

ObservationSet ObservationSet::from_json(nlohmann::json const& j) {
    ObservationSet obs_set(j.at("person_id").get<int>());

    for (auto const& [name, seq_json] : j.at("sequences").items()) {
        obs_set.add_sequence(ObservationSequence::from_json(seq_json));
    }

    return obs_set;
}

}  // namespace posetrak

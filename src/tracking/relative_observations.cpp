// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/tracking/relative_observations.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace posetrak {

std::vector<Observation>
build_ref_marker_pair_observations(std::vector<Observation> const& frame_obs, int ref_marker_id,
                                   double pose_noise_std, double min_confidence) {
    // Group by camera; within each camera, find the reference marker's own
    // detection (at most one expected per camera per frame).
    //
    // frame_obs may legitimately contain non-POSITION entries too -- e.g. when
    // the source ObservationSet was loaded with use_relative_observations=true,
    // every marker (including the reference marker) also carries a PAIR_DIFF
    // entry against its own skeleton-tree parent marker (session_reader.cpp's
    // general-purpose within-person relative pairs, unrelated to this
    // function's own ref_marker_id). Skipping non-POSITION entries here is
    // what stops that PAIR_DIFF entry from silently winning the
    // last-write-wins ref_obs_by_camera lookup below and poisoning every
    // resulting pair's rel.position with the parent marker's own pixel
    // offset baked in.
    std::unordered_map<int, Observation const*> ref_obs_by_camera;
    for (Observation const& obs : frame_obs) {
        if (obs.mode != MeasurementMode::POSITION) {
            continue;
        }
        if (obs.marker_id == ref_marker_id && obs.confidence >= min_confidence) {
            ref_obs_by_camera[obs.camera_id] = &obs;
        }
    }

    std::vector<Observation> result;
    result.reserve(frame_obs.size());

    for (Observation const& obs : frame_obs) {
        if (obs.mode != MeasurementMode::POSITION) {
            continue;  // see the ref_obs_by_camera loop above -- same reasoning
        }
        if (obs.marker_id == ref_marker_id) {
            continue;  // the reference marker is never paired with itself
        }
        if (obs.confidence < min_confidence) {
            continue;
        }
        auto ref_it = ref_obs_by_camera.find(obs.camera_id);
        if (ref_it == ref_obs_by_camera.end()) {
            continue;  // reference marker not detected in this camera this frame
        }
        Observation const& ref_obs = *ref_it->second;

        Observation rel;
        rel.camera_id = obs.camera_id;
        rel.marker_id = obs.marker_id;
        rel.ref_marker_id = ref_marker_id;
        rel.frame_idx = obs.frame_idx;
        rel.timestamp = obs.timestamp;
        // Observed measurement = marker_pixel - reference_pixel
        rel.position = obs.position - ref_obs.position;
        rel.position_distorted = obs.position_distorted - ref_obs.position_distorted;
        rel.confidence = std::min(obs.confidence, ref_obs.confidence);
        rel.mode = MeasurementMode::PAIR_DIFF;
        rel.crop_scale = obs.crop_scale;
        // Calibration error cancels in the pixel difference -- matches every
        // other PAIR_DIFF use in this codebase.
        rel.noise_std_override = pose_noise_std * std::sqrt(2.0) * obs.crop_scale;
        result.push_back(rel);
    }

    return result;
}

std::pair<std::vector<ObservationResult>, std::vector<uint8_t>>
reconstruct_pair_diff_absolute(std::vector<ObservationResult> const& results,
                               std::string const& ref_marker_name) {
    // Index the reference marker's own entries by camera (at most one expected
    // per camera per frame).
    std::unordered_map<int, ObservationResult const*> ref_by_camera;
    for (ObservationResult const& r : results) {
        if (r.marker_name == ref_marker_name) {
            ref_by_camera[r.camera_id] = &r;
        }
    }

    std::vector<ObservationResult> out;
    std::vector<uint8_t> reconstructed;
    out.reserve(results.size());
    reconstructed.reserve(results.size());

    for (ObservationResult const& r : results) {
        if (r.marker_name == ref_marker_name) {
            out.push_back(r);
            reconstructed.push_back(0);
            continue;
        }
        auto ref_it = ref_by_camera.find(r.camera_id);
        if (ref_it == ref_by_camera.end()) {
            throw std::runtime_error("reconstruct_pair_diff_absolute: no reference marker '" +
                                     ref_marker_name + "' result for camera " +
                                     std::to_string(r.camera_id) +
                                     " (required to reconstruct marker '" + r.marker_name + "')");
        }
        ObservationResult shifted = r;
        shifted.actual += ref_it->second->actual;
        shifted.predicted += ref_it->second->predicted;
        // innovation/mahalanobis_distance are representation-invariant under this
        // shift (actual and predicted move by the same additive amount), so they
        // are left as computed by the PAIR_DIFF update -- no correction needed.
        out.push_back(shifted);
        reconstructed.push_back(1);
    }

    return {out, reconstructed};
}

}  // namespace posetrak

#include "posetrak/tracking/relative_observations.hpp"

#include <cmath>
#include <unordered_map>

namespace posetrak {

std::vector<Observation>
build_ref_marker_pair_observations(std::vector<Observation> const& frame_obs, int ref_marker_id,
                                   double pose_noise_std, double min_confidence) {
    // Group by camera; within each camera, find the reference marker's own
    // detection (at most one expected per camera per frame).
    std::unordered_map<int, Observation const*> ref_obs_by_camera;
    for (Observation const& obs : frame_obs) {
        if (obs.marker_id == ref_marker_id && obs.confidence >= min_confidence) {
            ref_obs_by_camera[obs.camera_id] = &obs;
        }
    }

    std::vector<Observation> result;
    result.reserve(frame_obs.size());

    for (Observation const& obs : frame_obs) {
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

}  // namespace posetrak

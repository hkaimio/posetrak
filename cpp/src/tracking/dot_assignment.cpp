// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/tracking/dot_assignment.hpp"

#include "posetrak/tracking/assignment.hpp"
#include <cmath>

namespace posetrak {

std::unordered_map<int, SubjectDotAssignment> resolve_dot_assignment(
    std::vector<SubjectDotPredictions> const& subjects,
    std::unordered_map<int, std::vector<UnlabeledCandidate>> const& candidates_by_camera,
    double gate_mahalanobis, int frame_idx, double timestamp, double calib_noise_std,
    StreakVelocityConfig const& streak_config, PrevDotPositions const& prev_positions,
    std::unordered_map<int, StreakKAccumulator>* streak_k_state) {
    std::unordered_map<int, SubjectDotAssignment> result;

    // Only touched when streak_config.enabled -- see the streak_k parameter doc
    // comment (dot_assignment.hpp) for why a throwaway local is fine when the
    // caller doesn't want k to persist across calls.
    std::unordered_map<int, StreakKAccumulator> local_streak_k_state;
    std::unordered_map<int, StreakKAccumulator>& streak_k =
        streak_k_state ? *streak_k_state : local_streak_k_state;

    // One column per (subject, marker) slot with a prediction for this camera.
    struct Column {
        int subject_id;
        int marker_id;
        MarkerPrediction const* prediction;
    };

    for (auto const& [camera_id, candidates] : candidates_by_camera) {
        if (candidates.empty())
            continue;

        std::vector<Column> columns;
        for (auto const& subject : subjects) {
            auto cam_it = subject.predictions_by_camera.find(camera_id);
            if (cam_it == subject.predictions_by_camera.end())
                continue;
            for (auto const& [marker_id, prediction] : cam_it->second) {
                columns.push_back(Column{subject.subject_id, marker_id, &prediction});
            }
        }
        if (columns.empty())
            continue;

        int const n_rows = static_cast<int>(candidates.size());
        int const n_cols = static_cast<int>(columns.size());
        std::vector<double> cost(static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols));
        for (int r = 0; r < n_rows; ++r) {
            Eigen::Vector2d const cand_pos = candidates[static_cast<size_t>(r)].position;
            for (int c = 0; c < n_cols; ++c) {
                MarkerPrediction const& pred = *columns[static_cast<size_t>(c)].prediction;
                Eigen::Vector2d const diff = cand_pos - pred.position;
                double const mahal_sq = diff.transpose() * pred.covariance.inverse() * diff;
                cost[static_cast<size_t>(r) * static_cast<size_t>(n_cols) +
                     static_cast<size_t>(c)] = mahal_sq;
            }
        }

        auto pairs = solve_assignment(cost, n_rows, n_cols, gate_mahalanobis);
        for (auto const& pair : pairs) {
            Column const& col = columns[static_cast<size_t>(pair.col)];
            UnlabeledCandidate const& cand = candidates[static_cast<size_t>(pair.row)];

            Observation obs;
            obs.camera_id = camera_id;
            obs.marker_id = col.marker_id;
            obs.frame_idx = frame_idx;
            obs.timestamp = timestamp;
            obs.position = cand.position;
            obs.position_distorted = cand.position_distorted;
            obs.confidence = cand.confidence;
            // Same reasoning as the ArUco corner and dot-detector write paths'
            // own noise_scale=0.0 convention: a dot candidate's centroid comes
            // from thresholding the full-resolution frame directly, not a
            // fixed-input-resolution network, so there is no crop-scaled
            // detection-algorithm error to describe -- calibration error (ec)
            // alone should dominate Observation::measurement_noise_std().
            obs.crop_scale = 0.0;

            // Motion-blur streak (dot_blob_detector.py's elongated-blob
            // acceptance path): its centroid is genuinely less precise than
            // a round dot's, roughly in proportion to how far the dot moved
            // during the exposure. A first cut, not yet modeling *direction*
            // -- the real uncertainty is larger along the blur than across
            // it, but Observation::measurement_noise_std() is a single
            // scalar; logged as a design gap in status.md's 2026-09-04
            // entry alongside the future velocity-from-streak and
            // blinking-LED sub-frame-timing ideas it also records. Treats
            // the elongation beyond the round-dot footprint as spread
            // uniformly over the exposure window (stddev = width/sqrt(12)).
            // Left at 0.0 (the default -- normal formula applies) for a
            // round dot, where major_axis == minor_axis.
            double const elongation = cand.major_axis - cand.minor_axis;
            if (elongation > 1.0) {
                obs.noise_std_override = calib_noise_std + elongation / std::sqrt(12.0);
            }

            result[col.subject_id].resolved.push_back(obs);

            // Streak-derived velocity observation (streak-velocity-design.md §3/§4) --
            // an optional, orthogonal extra Observation alongside the POSITION one
            // above, not a replacement for it. See resolve_dot_assignment()'s own doc
            // comment (dot_assignment.hpp) for the documented "can't help a dot
            // reacquired after being lost entirely" scope limit this real-previous-
            // position requirement implies.
            if (streak_config.enabled && elongation >= streak_config.min_elongation_px &&
                (cand.dir_x != 0.0 || cand.dir_y != 0.0)) {
                auto subj_it = prev_positions.find(col.subject_id);
                if (subj_it != prev_positions.end()) {
                    auto cam_it2 = subj_it->second.find(camera_id);
                    if (cam_it2 != subj_it->second.end()) {
                        auto marker_it = cam_it2->second.find(col.marker_id);
                        if (marker_it != cam_it2->second.end()) {
                            Eigen::Vector2d const disp = cand.position - marker_it->second;
                            double const disp_px = disp.norm();
                            if (disp_px >= streak_config.min_displacement_px) {
                                StreakKAccumulator& acc = streak_k[camera_id];
                                acc.add(elongation, disp_px, streak_config.k_window);
                                if (auto k = acc.k(streak_config.k_min_samples);
                                    k.has_value() && *k > 0.0) {
                                    // The streak's axis is undirected (dot_blob_detector.py's
                                    // own canonicalization) -- resolve the sign against this
                                    // frame's real, measured displacement direction.
                                    Eigen::Vector2d streak_dir(cand.dir_x, cand.dir_y);
                                    if (streak_dir.dot(disp) < 0.0) {
                                        streak_dir = -streak_dir;
                                    }
                                    // Rescale the sub-exposure streak displacement up to a
                                    // full-frame-equivalent one (design doc §1: v =
                                    // streak_length / (k * frame_time), and this is
                                    // v * frame_time = streak_length / k, in the streak's
                                    // own resolved direction).
                                    Eigen::Vector2d const frame_equiv_disp =
                                        streak_dir * (elongation / *k);

                                    Observation vel_obs;
                                    vel_obs.camera_id = camera_id;
                                    vel_obs.marker_id = col.marker_id;
                                    vel_obs.frame_idx = frame_idx;
                                    vel_obs.timestamp = timestamp;
                                    vel_obs.confidence = cand.confidence;
                                    vel_obs.mode = MeasurementMode::VELOCITY;
                                    // position - prev_position must equal frame_equiv_disp --
                                    // see UnscentedKalmanFilter::observations_to_vector().
                                    vel_obs.position = cand.position;
                                    vel_obs.prev_position = cand.position - frame_equiv_disp;
                                    vel_obs.crop_scale = 0.0;
                                    vel_obs.noise_std_override = streak_config.velocity_noise_std;

                                    result[col.subject_id].resolved.push_back(vel_obs);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}

std::unordered_map<int, SubjectDotAssignment> resolve_shared_dot_assignment(
    std::vector<DotAssignmentSubject> const& subjects,
    std::unordered_map<int, std::vector<UnlabeledCandidate>> const& candidates_by_camera,
    TrackerConfig const& config, int frame_idx, double timestamp) {
    std::vector<SubjectDotPredictions> predictions;
    predictions.reserve(subjects.size());
    PrevDotPositions prev_positions;
    for (auto const& subject : subjects) {
        SubjectDotPredictions sp;
        sp.subject_id = subject.subject_id;
        for (auto const& [camera_id, candidates] : candidates_by_camera) {
            if (candidates.empty())
                continue;
            sp.predictions_by_camera[camera_id] =
                subject.tracker->predict_dot_slot_predictions(camera_id);
        }
        predictions.push_back(std::move(sp));
        prev_positions[subject.subject_id] = subject.tracker->prev_observations();
    }

    StreakVelocityConfig streak_config;
    streak_config.enabled = config.dot_streak_velocity_enabled;
    streak_config.k_window = config.dot_streak_k_window;
    streak_config.k_min_samples = config.dot_streak_k_min_samples;
    streak_config.min_displacement_px = config.dot_streak_min_displacement_px;
    streak_config.min_elongation_px = config.dot_streak_min_elongation_px;
    streak_config.velocity_noise_std = config.dot_streak_velocity_noise_std;

    // See this function's own doc comment (dot_assignment.hpp) for why the first
    // subject's accumulator map is used rather than a true per-camera-shared one.
    std::unordered_map<int, StreakKAccumulator>* streak_k_state =
        subjects.empty() ? nullptr : &subjects.front().tracker->streak_k_accumulators();

    return resolve_dot_assignment(
        predictions, candidates_by_camera, config.dot_assignment_gate_mahalanobis, frame_idx,
        timestamp, config.calib_noise_std, streak_config, prev_positions, streak_k_state);
}

}  // namespace posetrak
